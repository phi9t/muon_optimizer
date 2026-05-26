"""Low-level tensor operations used by the streamed Qwen execution path."""

from __future__ import annotations

from typing import Optional

import torch
import torch.nn.functional as F


def rms_norm(hidden: torch.Tensor, weight: torch.Tensor, eps: float) -> torch.Tensor:
    """Apply Qwen RMSNorm over the last dimension.

    Args:
        hidden: Tensor of shape ``[batch, seq, hidden_size]``.
        weight: RMSNorm gain, typically shape ``[hidden_size]``.
        eps: Numerical epsilon.
    """

    mean_sq = torch.mean(hidden * hidden, dim=-1, keepdim=True)
    return hidden * torch.rsqrt(mean_sq + eps) * weight


def apply_rope(
    q: torch.Tensor,
    k: torch.Tensor,
    positions: torch.Tensor,
    rope_theta: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Apply Qwen rotary position embeddings.

    Args:
        q: Query tensor shaped ``[batch, seq, num_attention_heads, head_dim]``.
        k: Key tensor shaped ``[batch, seq, num_key_value_heads, head_dim]``.
        positions: Explicit integer positions as shape ``[seq]`` or ``[batch, seq]``.
        rope_theta: Rotary theta value from config.

    Returns:
        A tuple ``(q_rot, k_rot)`` with matching input shapes.
    """

    if q.dim() != 4 or k.dim() != 4:
        raise ValueError("q and k must be rank-4 tensors [batch, seq, heads, head_dim].")

    if q.shape[-1] != k.shape[-1]:
        raise ValueError("q and k must have the same head dimension.")

    batch, seq_len, _, head_dim = q.shape

    if positions.dim() == 1:
        if positions.shape[0] != seq_len:
            raise ValueError("positions must match sequence length when 1-D.")
        positions = positions.expand(batch, seq_len)
    elif positions.dim() == 2:
        if positions.shape != (batch, seq_len):
            raise ValueError("positions must have shape [batch, seq].")
    else:
        raise ValueError("positions must be [seq] or [batch, seq].")

    if head_dim % 2 != 0:
        raise ValueError("head_dim must be even for RoPE.")

    half_dim = head_dim // 2
    half_idx = torch.arange(0, half_dim, device=q.device, dtype=q.dtype)
    inv_freq = 1.0 / (rope_theta ** (half_idx / half_dim))

    pos = positions.to(device=q.device, dtype=q.dtype)
    angles = pos[:, :, None, None] * inv_freq[None, None, None, :]
    cos = torch.cos(angles)
    sin = torch.sin(angles)
    cos = torch.cat([cos, cos], dim=-1)
    sin = torch.cat([sin, sin], dim=-1)

    def _rotate_half(x: torch.Tensor) -> torch.Tensor:
        x1 = x[..., :half_dim]
        x2 = x[..., half_dim:]
        return torch.cat((-x2, x1), dim=-1)

    # `cos`/`sin` are [batch, seq, 1, head_dim], so they broadcast over q/k heads.
    q_rot = q * cos + _rotate_half(q) * sin
    k_rot = k * cos + _rotate_half(k) * sin

    return q_rot, k_rot


def repeat_kv_for_gqa(
    kv: torch.Tensor,
    num_attention_heads: int,
    num_key_value_heads: int,
) -> torch.Tensor:
    """Repeat grouped key/value heads for multi-query attention.

    Args:
        kv: Tensor shaped ``[batch, seq, num_key_value_heads, head_dim]``.
        num_attention_heads: Number of attention heads in the model.
        num_key_value_heads: Number of KV heads in the checkpoint.
    """

    if num_attention_heads <= 0:
        raise ValueError("num_attention_heads must be positive.")
    if num_key_value_heads <= 0:
        raise ValueError("num_key_value_heads must be positive.")
    if num_key_value_heads != kv.shape[2]:
        raise ValueError("kv tensor head count does not match num_key_value_heads.")
    if num_attention_heads % num_key_value_heads != 0:
        raise ValueError("num_attention_heads must be divisible by num_key_value_heads.")

    repeat_count = num_attention_heads // num_key_value_heads
    return kv.repeat_interleave(repeat_count, dim=2)


def causal_attention(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    attention_mask: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """Compute causal scaled dot-product attention.

    Expected shapes:
        q: [batch, query_len, num_attention_heads, head_dim]
        k: [batch, key_len, num_key_value_heads, head_dim]
        v: [batch, key_len, num_key_value_heads, head_dim]

    The output shape is [batch, query_len, num_attention_heads, head_dim].
    If ``attention_mask`` is provided, it should be broadcastable to
    ``[batch, num_attention_heads, query_len, key_len]``.

    Boolean masks should encode keep/disallow as ``True``/``False``.
    Non-boolean masks are interpreted as additive logits masks.
    """

    if q.dim() != 4 or k.dim() != 4 or v.dim() != 4:
        raise ValueError("q, k, and v must be rank-4 tensors.")

    batch, query_len, num_heads, head_dim = q.shape
    kv_batch, key_len, num_kv_heads, kv_head_dim = k.shape
    v_batch, v_key_len, v_kv_heads, v_head_dim = v.shape

    if batch != kv_batch or batch != v_batch:
        raise ValueError("q, k, and v must share batch size.")
    if key_len != v_key_len:
        raise ValueError("k and v key lengths must match.")
    if num_kv_heads != v_kv_heads:
        raise ValueError("k and v head counts must match.")
    if kv_head_dim != head_dim:
        raise ValueError("q and k/v head dimensions must match.")

    k = repeat_kv_for_gqa(k, num_attention_heads=num_heads, num_key_value_heads=num_kv_heads)
    v = repeat_kv_for_gqa(v, num_attention_heads=num_heads, num_key_value_heads=num_kv_heads)

    q_t = q.transpose(1, 2)
    k_t = k.transpose(1, 2)
    v_t = v.transpose(1, 2)

    scale = 1.0 / (head_dim**0.5)
    scores = torch.matmul(q_t, k_t.transpose(-1, -2)) * scale

    query_idx = torch.arange(query_len, device=scores.device)
    key_idx = torch.arange(key_len, device=scores.device)

    # Support two common call sites:
    # - full prefill where key_len == query_len
    # - decode-style where the query chunk is a suffix of a longer key/value cache.
    if query_len == key_len:
        allowed = query_idx[:, None] >= key_idx[None, :]
    else:
        offset = key_len - query_len
        allowed = (query_idx[None, :] + offset) >= key_idx[:, None]
        allowed = allowed.T

    scores = scores.masked_fill(~allowed.view(1, 1, query_len, key_len), float("-inf"))

    if attention_mask is not None:
        if attention_mask.dtype == torch.bool:
            scores = scores.masked_fill(~attention_mask, float("-inf"))
        else:
            scores = scores + attention_mask

    probs = F.softmax(scores, dim=-1)
    output = torch.matmul(probs, v_t)
    return output.transpose(1, 2)


def gated_mlp(
    hidden: torch.Tensor,
    gate_proj_weight: torch.Tensor,
    up_proj_weight: torch.Tensor,
    down_proj_weight: torch.Tensor,
    *,
    gate_proj_bias: Optional[torch.Tensor] = None,
    up_proj_bias: Optional[torch.Tensor] = None,
    down_proj_bias: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """Apply Qwen gated MLP.

    Computes ``down_proj(silu(gate_proj(x)) * up_proj(x))`` with raw tensors.
    """

    gate = F.linear(hidden, gate_proj_weight, bias=gate_proj_bias)
    up = F.linear(hidden, up_proj_weight, bias=up_proj_bias)
    hidden = F.silu(gate) * up
    return F.linear(hidden, down_proj_weight, bias=down_proj_bias)


def apply_rotary_embedding(*args, **kwargs) -> tuple[torch.Tensor, torch.Tensor]:
    """Backward-compatible alias for ``apply_rope``."""

    return apply_rope(*args, **kwargs)
