"""Tests for streamed Qwen model metadata planning and tensor loading."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
import torch
from safetensors.torch import save_file

from scripts.qwen_streaming.spec import QwenStreamedModelSpec
from scripts.qwen_streaming.weights import SafetensorWeightLoader


_QWEN2_CONFIG = {
    "architectures": ["Qwen3ForCausalLM"],
    "model_type": "qwen3",
    "hidden_size": 16,
    "num_hidden_layers": 2,
    "vocab_size": 42,
    "num_attention_heads": 4,
    "num_key_value_heads": 2,
    "head_dim": 4,
    "intermediate_size": 32,
    "rms_norm_eps": 1e-6,
    "rope_theta": 10000.0,
    "tie_word_embeddings": False,
}


def _required_names(config: dict[str, object]) -> list[str]:
    num_layers = int(config["num_hidden_layers"])
    names = [
        "model.embed_tokens.weight",
        "model.norm.weight",
        "lm_head.weight",
    ]
    layer_names = [
        "input_layernorm.weight",
        "post_attention_layernorm.weight",
        "self_attn.q_proj.weight",
        "self_attn.k_proj.weight",
        "self_attn.v_proj.weight",
        "self_attn.o_proj.weight",
        "mlp.gate_proj.weight",
        "mlp.up_proj.weight",
        "mlp.down_proj.weight",
    ]
    for layer_idx in range(num_layers):
        names.extend([f"model.layers.{layer_idx}.{suffix}" for suffix in layer_names])
    return names


def _write_json(path: Path, payload: dict[str, object]) -> None:
    path.write_text(json.dumps(payload), encoding="utf-8")


def _write_index_checkpoint(tmp_path: Path, config: dict[str, object], shard_tensors: dict[str, dict[str, torch.Tensor]]) -> Path:
    config_path = tmp_path / "config.json"
    _write_json(config_path, config)

    weight_map: dict[str, str] = {}
    shard_files = []
    for index, (shard_name, tensors) in enumerate(shard_tensors.items()):
        shard_path = tmp_path / shard_name
        save_file(tensors, shard_path)
        shard_files.append(shard_name)
        for tensor_name in tensors:
            weight_map[tensor_name] = shard_name
        assert index < 1000

    index_payload = {
        "metadata": {"total_size": 0},
        "weight_map": weight_map,
    }
    _write_json(tmp_path / "model.safetensors.index.json", index_payload)
    return tmp_path


def _write_single_file_checkpoint(tmp_path: Path, config: dict[str, object], tensors: dict[str, torch.Tensor]) -> Path:
    _write_json(tmp_path / "config.json", config)
    save_file(tensors, tmp_path / "model.safetensors")
    return tmp_path


def _make_linear_shards(config: dict[str, object], tie_word_embeddings: bool) -> dict[str, dict[str, torch.Tensor]]:
    shard1: dict[str, torch.Tensor] = {
        "model.embed_tokens.weight": torch.randn(config["vocab_size"], config["hidden_size"]),
        "model.norm.weight": torch.randn(config["hidden_size"]),
    }
    if not tie_word_embeddings:
        shard1["lm_head.weight"] = torch.randn(config["vocab_size"], config["hidden_size"])

    layer_names = [
        "input_layernorm.weight",
        "post_attention_layernorm.weight",
        "self_attn.q_proj.weight",
        "self_attn.k_proj.weight",
        "self_attn.v_proj.weight",
        "self_attn.o_proj.weight",
        "mlp.gate_proj.weight",
        "mlp.up_proj.weight",
        "mlp.down_proj.weight",
    ]

    layer0 = {}
    layer1 = {}
    layers = (layer0, layer1)
    for layer_index in range(config["num_hidden_layers"]):
        local = layers[layer_index % len(layers)]
        for suffix in layer_names:
            # Keep these tiny; only shape consistency matters for planning.
            shape = (config["hidden_size"], config["hidden_size"])
            if "layernorm" in suffix:
                shape = (config["hidden_size"],)
            local[f"model.layers.{layer_index}.{suffix}"] = torch.randn(*shape)

    return {
        "model-00001-of-00002.safetensors": shard1,
        "model-00002-of-00002.safetensors": layer0 | layer1,
    }


def _patch_snapshot(monkeypatch, local_dir: Path):
    monkeypatch.setattr(
        "scripts.qwen_streaming.spec.snapshot_download",
        lambda model_id, allow_patterns=(): str(local_dir),
    )


def test_required_tensors_from_indexed_checkpoint(monkeypatch, tmp_path: Path):
    config = dict(_QWEN2_CONFIG)
    shard_tensors = _make_linear_shards(config, tie_word_embeddings=config["tie_word_embeddings"])
    _write_index_checkpoint(tmp_path, config, shard_tensors)
    _patch_snapshot(monkeypatch, tmp_path)

    spec = QwenStreamedModelSpec.from_model_id("local/test")

    required = spec.required_tensor_names()
    assert spec.local_dir == tmp_path.resolve()
    assert spec.num_hidden_layers == int(config["num_hidden_layers"])
    assert "model.embed_tokens.weight" in spec.weight_map
    assert "lm_head.weight" in spec.weight_map
    assert spec.validate_required_tensors() is None
    for tensor_name in required:
        assert tensor_name in spec.weight_map


def test_single_file_checkpoint_planning(monkeypatch, tmp_path: Path):
    config = dict(_QWEN2_CONFIG)
    tensors = {}
    for name in _required_names(config):
        if name.endswith("weight") and "layernorm.weight" in name:
            tensors[name] = torch.randn(config["hidden_size"])
        elif name.endswith("weight") and name.endswith("embed_tokens.weight"):
            tensors[name] = torch.randn(config["vocab_size"], config["hidden_size"])
        elif "norm.weight" in name and name != "model.embed_tokens.weight":
            tensors[name] = torch.randn(config["hidden_size"])
        elif "norm.weight" in name:
            tensors[name] = torch.randn(config["hidden_size"])
        elif "bias" in name:
            continue
        else:
            tensors[name] = torch.randn(config["hidden_size"], config["hidden_size"])

    _write_single_file_checkpoint(tmp_path, config, tensors)
    _patch_snapshot(monkeypatch, tmp_path)

    spec = QwenStreamedModelSpec.from_model_id("local/test")

    assert "model.safetensors" in spec.weight_map.values().__iter__().__next__().as_posix()
    assert len(set(spec.weight_map.values())) == 1
    for name in _required_names(config):
        assert name in spec.weight_map


def test_validate_required_tensors_reports_missing_keys(monkeypatch, tmp_path: Path):
    config = dict(_QWEN2_CONFIG)
    tensors = _required_names(config)
    tensor_map = {name: torch.randn(config["hidden_size"]) for name in tensors[:-1]}
    tensor_map.pop("lm_head.weight")
    _write_single_file_checkpoint(tmp_path, config, tensor_map)
    _patch_snapshot(monkeypatch, tmp_path)

    spec = QwenStreamedModelSpec.from_model_id("local/test")
    with pytest.raises(ValueError, match="Missing required tensors"):
        spec.validate_required_tensors()


def test_safetensor_loader_returns_cpu_float32(monkeypatch, tmp_path: Path):
    tensor_a = torch.randn(2, 3, dtype=torch.float16)
    tensor_b = torch.randn(4, dtype=torch.float32)
    file_path = tmp_path / "model.safetensors"
    save_file({"tensor_a": tensor_a, "tensor_b": tensor_b}, file_path)

    weight_map = {"tensor_a": file_path, "tensor_b": file_path}
    loader = SafetensorWeightLoader(weight_map=weight_map, dtype=torch.float32)
    loaded = loader.load_tensors(["tensor_a", "tensor_b"])

    assert tuple(loaded["tensor_a"].shape) == tuple(tensor_a.shape)
    assert loaded["tensor_a"].device.type == "cpu"
    assert loaded["tensor_a"].dtype == torch.float32
    assert loaded["tensor_b"].dtype == torch.float32
