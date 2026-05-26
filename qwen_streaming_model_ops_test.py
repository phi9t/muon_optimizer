"""Tests for Qwen streamed tensor ops."""

from __future__ import annotations

import torch

from scripts.qwen_streaming.model_ops import (
    apply_rope,
    causal_attention,
    gated_mlp,
    repeat_kv_for_gqa,
    rms_norm,
)


def test_rms_norm_matches_reference() -> None:
    hidden = torch.tensor(
        [
            [[1.0, -2.0, 3.0], [4.0, -5.0, 6.0]],
            [[-7.0, 8.0, -9.0], [10.0, -11.0, 12.0]],
        ],
        dtype=torch.float32,
    )
    weight = torch.tensor([1.5, 0.5, 2.0], dtype=torch.float32)
    eps = 1e-6

    result = rms_norm(hidden, weight, eps)

    manual = hidden * torch.rsqrt(torch.mean(hidden * hidden, dim=-1, keepdim=True) + eps) * weight
    assert torch.allclose(result, manual)


def test_apply_rope_preserves_shape_and_pos_zero() -> None:
    q = torch.arange(1, 1 + 2 * 2 * 1 * 4, dtype=torch.float32).reshape(1, 2, 2, 4)
    k = torch.arange(101, 101 + 2 * 2 * 1 * 4, dtype=torch.float32).reshape(1, 2, 2, 4)
    positions = torch.tensor([0, 1], dtype=torch.int64)

    q_rot, k_rot = apply_rope(q, k, positions, rope_theta=10000.0)

    assert q_rot.shape == q.shape
    assert k_rot.shape == k.shape
    assert torch.allclose(q_rot[:, 0], q[:, 0])
    assert torch.allclose(k_rot[:, 0], k[:, 0])


def test_repeat_kv_for_gqa_repeats_values() -> None:
    kv = torch.tensor(
        [
            [
                [[1.0, 2.0], [3.0, 4.0]],
                [[5.0, 6.0], [7.0, 8.0]],
            ]
        ],
        dtype=torch.float32,
    )

    out = repeat_kv_for_gqa(kv, num_attention_heads=4, num_key_value_heads=2)

    assert out.shape == (1, 2, 4, 2)
    assert torch.equal(out[:, :, 0], kv[:, :, 0])
    assert torch.equal(out[:, :, 1], kv[:, :, 0])
    assert torch.equal(out[:, :, 2], kv[:, :, 1])
    assert torch.equal(out[:, :, 3], kv[:, :, 1])


def test_gated_mlp_matches_reference_formula() -> None:
    hidden = torch.tensor(
        [[
            [0.25, -0.5, 1.0],
            [1.5, -2.0, 0.75],
        ]],
        dtype=torch.float32,
    )
    gate_w = torch.tensor(
        [[0.1, 0.2, 0.3], [0.4, 0.5, 0.6], [0.7, 0.8, 0.9]],
        dtype=torch.float32,
    )
    up_w = torch.tensor(
        [[1.1, 0.0, -0.2], [0.4, -0.7, 0.3], [0.8, 0.9, 0.5]],
        dtype=torch.float32,
    )
    down_w = torch.tensor(
        [[1.2, -0.3, 0.7], [0.4, 0.9, -0.1], [0.8, 0.2, 0.5]],
        dtype=torch.float32,
    )

    out = gated_mlp(hidden, gate_w, up_w, down_w)

    gate = hidden @ gate_w.T
    up = hidden @ up_w.T
    expected = (torch.nn.functional.silu(gate) * up) @ down_w.T
    assert torch.allclose(out, expected)


def test_causal_attention_masks_future_tokens() -> None:
    q = torch.ones((1, 1, 1, 1), dtype=torch.float32)
    k = torch.tensor([1.0, 2.0], dtype=torch.float32).view(1, 2, 1, 1)
    v = torch.tensor([10.0, 20.0], dtype=torch.float32).view(1, 2, 1, 1)

    unmasked = causal_attention(q, k, v, attention_mask=None)
    explicit_mask = torch.tensor(
        [[[[True, False]]]],
        dtype=torch.bool,
    )
    masked = causal_attention(q, k, v, attention_mask=explicit_mask)

    assert not torch.allclose(unmasked, masked)
    assert torch.allclose(masked[0, 0, 0, 0], torch.tensor(10.0))
