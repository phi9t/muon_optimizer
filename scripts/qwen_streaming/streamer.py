"""Layer-by-layer streamed Qwen execution."""

from __future__ import annotations

import gc
from dataclasses import dataclass

import torch
import torch.nn.functional as F

from .kv_cache import KVSpillStore, LayerKV
from .memory import MemoryGuard
from .model_ops import apply_rope, causal_attention, gated_mlp, rms_norm
from .spec import QwenStreamedModelSpec
from .weights import SafetensorWeightLoader


_LAYER_TENSOR_SUFFIXES = (
    "input_layernorm.weight",
    "post_attention_layernorm.weight",
    "self_attn.q_proj.weight",
    "self_attn.q_norm.weight",
    "self_attn.k_proj.weight",
    "self_attn.k_norm.weight",
    "self_attn.v_proj.weight",
    "self_attn.o_proj.weight",
    "mlp.gate_proj.weight",
    "mlp.up_proj.weight",
    "mlp.down_proj.weight",
)


def _layer_name(layer_index: int, suffix: str) -> str:
    return f"model.layers.{layer_index}.{suffix}"


def _layer_positions(seq_len: int) -> torch.Tensor:
    return torch.arange(seq_len, dtype=torch.long)


def _check_memory(memory_guard: MemoryGuard | None, model_label: str | None, stage: str, layer_index: int) -> None:
    if memory_guard is None:
        return
    prefix = model_label if model_label is not None else "streamed-qwen"
    memory_guard.check(f"{prefix}/{stage}/layer-{layer_index:03d}")


def _check_input_ids(input_ids: torch.Tensor) -> tuple[int, int]:
    if input_ids.device.type != "cpu":
        raise ValueError("input_ids must be on CPU.")
    if input_ids.dtype != torch.long:
        raise TypeError("input_ids must be torch.long.")
    if input_ids.dim() != 2:
        raise ValueError("input_ids must have shape [batch, seq].")

    batch_size, seq_len = input_ids.shape
    if input_ids.min() < 0:
        raise ValueError("input_ids contains negative token ids.")
    if batch_size <= 0 or seq_len <= 0:
        raise ValueError("input_ids must contain at least one token.")
    return batch_size, seq_len


def _reshape_qkv(
    tensor: torch.Tensor,
    batch_size: int,
    seq_len: int,
    num_heads: int,
    head_dim: int,
) -> torch.Tensor:
    expected_width = num_heads * head_dim
    if tensor.shape[-1] != expected_width:
        raise ValueError(
            f"Unexpected projection width: expected {expected_width}, got {tensor.shape[-1]}."
        )
    return tensor.reshape(batch_size, seq_len, num_heads, head_dim)


@dataclass
class QwenLayerStreamer:
    """Stream one Qwen model one layer at a time from safetensor weights."""

    spec: QwenStreamedModelSpec
    loader: SafetensorWeightLoader
    memory_guard: MemoryGuard | None = None
    model_label: str | None = None

    def prefill(self, input_ids: torch.Tensor, cache: KVSpillStore) -> torch.Tensor:
        batch_size, seq_len = _check_input_ids(input_ids)
        if input_ids.max() >= self.spec.vocab_size:
            raise ValueError("input_ids contains token ids outside model vocabulary.")

        embedding = self.loader.load_tensor("model.embed_tokens.weight")
        hidden = embedding[input_ids]
        del embedding
        gc.collect()

        positions = _layer_positions(seq_len)

        for layer_index in range(self.spec.num_hidden_layers):
            tensor_names = tuple(_layer_name(layer_index, suffix) for suffix in _LAYER_TENSOR_SUFFIXES)
            layer_tensors = self.loader.load_tensors(tensor_names)

            layer_input = rms_norm(
                hidden,
                layer_tensors[_layer_name(layer_index, "input_layernorm.weight")],
                self.spec.rms_norm_eps,
            )

            q = F.linear(layer_input, layer_tensors[_layer_name(layer_index, "self_attn.q_proj.weight")])
            k = F.linear(layer_input, layer_tensors[_layer_name(layer_index, "self_attn.k_proj.weight")])
            v = F.linear(layer_input, layer_tensors[_layer_name(layer_index, "self_attn.v_proj.weight")])

            q = _reshape_qkv(q, batch_size, seq_len, self.spec.num_attention_heads, self.spec.head_dim)
            k = _reshape_qkv(k, batch_size, seq_len, self.spec.num_key_value_heads, self.spec.head_dim)
            v = _reshape_qkv(v, batch_size, seq_len, self.spec.num_key_value_heads, self.spec.head_dim)
            q = rms_norm(
                q,
                layer_tensors[_layer_name(layer_index, "self_attn.q_norm.weight")],
                self.spec.rms_norm_eps,
            )
            k = rms_norm(
                k,
                layer_tensors[_layer_name(layer_index, "self_attn.k_norm.weight")],
                self.spec.rms_norm_eps,
            )

            q, k = apply_rope(q, k, positions, self.spec.rope_theta)
            attention = causal_attention(q, k, v)
            attention = attention.reshape(batch_size, seq_len, self.spec.hidden_size)
            attention = F.linear(
                attention,
                layer_tensors[_layer_name(layer_index, "self_attn.o_proj.weight")],
            )

            hidden = hidden + attention

            mlp_in = rms_norm(
                hidden,
                layer_tensors[_layer_name(layer_index, "post_attention_layernorm.weight")],
                self.spec.rms_norm_eps,
            )
            layer_output = gated_mlp(
                mlp_in,
                gate_proj_weight=layer_tensors[_layer_name(layer_index, "mlp.gate_proj.weight")],
                up_proj_weight=layer_tensors[_layer_name(layer_index, "mlp.up_proj.weight")],
                down_proj_weight=layer_tensors[_layer_name(layer_index, "mlp.down_proj.weight")],
            )
            hidden = hidden + layer_output

            cache.write(
                layer_index,
                LayerKV(
                    key=k.to(torch.float32),
                    value=v.to(torch.float32),
                ),
            )

            del layer_tensors
            del layer_input
            del q
            del k
            del v
            del attention
            del mlp_in
            del layer_output
            gc.collect()
            _check_memory(self.memory_guard, self.model_label, "prefill", layer_index)

        return hidden

    def decode_one(self, token_id: int, position: int, cache: KVSpillStore) -> torch.Tensor:
        if not isinstance(token_id, int):
            raise TypeError("token_id must be an int.")
        if not isinstance(position, int) or position < 0:
            raise ValueError("position must be a non-negative int.")

        embedding = self.loader.load_tensor("model.embed_tokens.weight")
        if token_id < 0 or token_id >= int(embedding.shape[0]):
            raise ValueError("token_id is outside the model vocabulary.")
        hidden = embedding[[token_id]].reshape(1, 1, self.spec.hidden_size)
        del embedding
        gc.collect()

        pos = torch.tensor([position], dtype=torch.long)

        for layer_index in range(self.spec.num_hidden_layers):
            existing = cache.read(layer_index)
            if existing is None:
                raise ValueError(f"Missing KV cache for layer {layer_index}.")

            tensor_names = tuple(_layer_name(layer_index, suffix) for suffix in _LAYER_TENSOR_SUFFIXES)
            layer_tensors = self.loader.load_tensors(tensor_names)

            layer_input = rms_norm(
                hidden,
                layer_tensors[_layer_name(layer_index, "input_layernorm.weight")],
                self.spec.rms_norm_eps,
            )

            q = F.linear(layer_input, layer_tensors[_layer_name(layer_index, "self_attn.q_proj.weight")])
            k_new = F.linear(layer_input, layer_tensors[_layer_name(layer_index, "self_attn.k_proj.weight")])
            v_new = F.linear(layer_input, layer_tensors[_layer_name(layer_index, "self_attn.v_proj.weight")])

            q = _reshape_qkv(q, 1, 1, self.spec.num_attention_heads, self.spec.head_dim)
            k_new = _reshape_qkv(k_new, 1, 1, self.spec.num_key_value_heads, self.spec.head_dim)
            v_new = _reshape_qkv(v_new, 1, 1, self.spec.num_key_value_heads, self.spec.head_dim)
            q = rms_norm(
                q,
                layer_tensors[_layer_name(layer_index, "self_attn.q_norm.weight")],
                self.spec.rms_norm_eps,
            )
            k_new = rms_norm(
                k_new,
                layer_tensors[_layer_name(layer_index, "self_attn.k_norm.weight")],
                self.spec.rms_norm_eps,
            )

            q, k_new = apply_rope(q, k_new, pos, self.spec.rope_theta)
            k = torch.cat([existing.key, k_new.to(torch.float32)], dim=1)
            v = torch.cat([existing.value, v_new.to(torch.float32)], dim=1)
            attention = causal_attention(q, k, v)

            attention = attention.reshape(1, 1, self.spec.hidden_size)
            attention = F.linear(
                attention,
                layer_tensors[_layer_name(layer_index, "self_attn.o_proj.weight")],
            )
            hidden = hidden + attention

            cache.write(
                layer_index,
                LayerKV(
                    key=k,
                    value=v,
                ),
            )

            mlp_in = rms_norm(
                hidden,
                layer_tensors[_layer_name(layer_index, "post_attention_layernorm.weight")],
                self.spec.rms_norm_eps,
            )
            layer_output = gated_mlp(
                mlp_in,
                gate_proj_weight=layer_tensors[_layer_name(layer_index, "mlp.gate_proj.weight")],
                up_proj_weight=layer_tensors[_layer_name(layer_index, "mlp.up_proj.weight")],
                down_proj_weight=layer_tensors[_layer_name(layer_index, "mlp.down_proj.weight")],
            )
            hidden = hidden + layer_output

            del layer_tensors
            del existing
            del layer_input
            del q
            del k_new
            del v_new
            del k
            del v
            del attention
            del mlp_in
            del layer_output
            gc.collect()
            _check_memory(self.memory_guard, self.model_label, "decode", layer_index)

        return hidden

    def logits_from_hidden(self, hidden: torch.Tensor) -> torch.Tensor:
        if hidden.dim() != 3:
            raise ValueError("hidden must have shape [batch, seq, hidden_size].")
        if hidden.shape[0] != 1:
            raise ValueError("logits_from_hidden supports batch size 1.")
        if hidden.shape[-1] != self.spec.hidden_size:
            raise ValueError("hidden second-last dimension must match hidden_size.")

        normalized = rms_norm(hidden[:, -1, :], self.loader.load_tensor("model.norm.weight"), self.spec.rms_norm_eps)
        if self.spec.tie_word_embeddings:
            lm_head = self.loader.load_tensor("model.embed_tokens.weight")
        else:
            lm_head = self.loader.load_tensor("lm_head.weight")

        logits = F.linear(normalized, lm_head).squeeze(0)
        del normalized
        del lm_head
        gc.collect()
        return logits
