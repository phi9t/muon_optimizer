"""Tests for streaming KV cache spill behavior."""

from __future__ import annotations

import torch

from scripts.qwen_streaming.kv_cache import KVSpillStore, LayerKV


def _tensor_cache_pair() -> LayerKV:
    return LayerKV(
        key=torch.tensor([[1.0, 2.0], [3.0, 4.0]], dtype=torch.float32),
        value=torch.tensor([[5.0, 6.0], [7.0, 8.0]], dtype=torch.float64),
    )


def test_kv_read_missing_layer_returns_none(tmp_path) -> None:
    store = KVSpillStore(root=tmp_path, model_label="model-A", prompt_index=0)
    assert store.read(0) is None


def test_kv_round_trip_preserves_shape_dtype_and_values(tmp_path) -> None:
    store = KVSpillStore(root=tmp_path, model_label="model-B", prompt_index=2)
    kv = _tensor_cache_pair()

    store.write(0, kv)
    restored = store.read(0)
    assert restored is not None
    assert restored.key.shape == kv.key.shape
    assert restored.value.shape == kv.value.shape
    assert restored.key.dtype == kv.key.dtype
    assert restored.value.dtype == kv.value.dtype
    assert torch.equal(restored.key, kv.key)
    assert torch.equal(restored.value, kv.value)


def test_kv_write_overwrite(tmp_path) -> None:
    store = KVSpillStore(root=tmp_path, model_label="model-C", prompt_index=1)
    initial = _tensor_cache_pair()
    store.write(0, initial)

    replacement = LayerKV(
        key=torch.tensor([[10.0, 20.0]], dtype=torch.float32),
        value=torch.tensor([[30.0]], dtype=torch.float64),
    )
    store.write(0, replacement)

    restored = store.read(0)
    assert restored is not None
    assert torch.equal(restored.key, replacement.key)
    assert torch.equal(restored.value, replacement.value)


def test_kv_clear_removes_prompt_directory(tmp_path) -> None:
    store = KVSpillStore(root=tmp_path, model_label="model-D", prompt_index=4)
    store.write(0, _tensor_cache_pair())
    assert store.prompt_root.exists()

    store.clear()

    assert not store.prompt_root.exists()
    assert not store.read(0)
