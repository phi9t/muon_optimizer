"""Tests for streamed Qwen layer execution."""

from __future__ import annotations

from pathlib import Path

import pytest
import torch
from safetensors.torch import save_file

from scripts.qwen_streaming.kv_cache import KVSpillStore
from scripts.qwen_streaming.spec import QwenStreamedModelSpec
from scripts.qwen_streaming.streamer import QwenLayerStreamer
from scripts.qwen_streaming.weights import SafetensorWeightLoader


_QWEN_TEST_CONFIG = {
    "model_type": "qwen3",
    "architectures": ["Qwen3ForCausalLM"],
    "hidden_size": 4,
    "num_hidden_layers": 2,
    "vocab_size": 13,
    "num_attention_heads": 2,
    "num_key_value_heads": 1,
    "head_dim": 2,
    "intermediate_size": 4,
    "rms_norm_eps": 1e-6,
    "rope_theta": 10000.0,
    "tie_word_embeddings": False,
}


def _build_checkpoint_tensors(config: dict[str, object], tie_word_embeddings: bool) -> dict[str, torch.Tensor]:
    torch.manual_seed(11)

    hidden_size = int(config["hidden_size"])
    vocab_size = int(config["vocab_size"])
    num_layers = int(config["num_hidden_layers"])
    intermediate_size = int(config["intermediate_size"])

    tensors: dict[str, torch.Tensor] = {
        "model.embed_tokens.weight": torch.randn(vocab_size, hidden_size, dtype=torch.float32),
        "model.norm.weight": torch.randn(hidden_size, dtype=torch.float32),
    }
    if not tie_word_embeddings:
        tensors["lm_head.weight"] = torch.randn(vocab_size, hidden_size, dtype=torch.float32)

    for layer_index in range(num_layers):
        prefix = f"model.layers.{layer_index}."
        tensors[prefix + "input_layernorm.weight"] = torch.randn(hidden_size, dtype=torch.float32)
        tensors[prefix + "post_attention_layernorm.weight"] = torch.randn(hidden_size, dtype=torch.float32)
        kv_width = int(config["num_key_value_heads"]) * int(config["head_dim"])
        tensors[prefix + "self_attn.q_proj.weight"] = torch.randn(hidden_size, hidden_size, dtype=torch.float32)
        tensors[prefix + "self_attn.k_proj.weight"] = torch.randn(kv_width, hidden_size, dtype=torch.float32)
        tensors[prefix + "self_attn.v_proj.weight"] = torch.randn(kv_width, hidden_size, dtype=torch.float32)
        tensors[prefix + "self_attn.o_proj.weight"] = torch.randn(hidden_size, hidden_size, dtype=torch.float32)
        tensors[prefix + "mlp.gate_proj.weight"] = torch.randn(intermediate_size, hidden_size, dtype=torch.float32)
        tensors[prefix + "mlp.up_proj.weight"] = torch.randn(intermediate_size, hidden_size, dtype=torch.float32)
        tensors[prefix + "mlp.down_proj.weight"] = torch.randn(hidden_size, intermediate_size, dtype=torch.float32)

    return tensors


def _write_checkpoint(tmp_path: Path, config: dict[str, object], tie_word_embeddings: bool) -> dict[str, Path]:
    tensors = _build_checkpoint_tensors(config, tie_word_embeddings)
    checkpoint = tmp_path / "model.safetensors"
    save_file(tensors, checkpoint)
    return {name: checkpoint for name in tensors}


def _make_spec(tmp_path: Path, config: dict[str, object], weight_map: dict[str, Path], tie_word_embeddings: bool) -> QwenStreamedModelSpec:
    return QwenStreamedModelSpec(
        model_id="qwen3/fake",
        local_dir=tmp_path,
        config=config,
        weight_map=weight_map,
        num_hidden_layers=int(config["num_hidden_layers"]),
        hidden_size=int(config["hidden_size"]),
        vocab_size=int(config["vocab_size"]),
        num_attention_heads=int(config["num_attention_heads"]),
        num_key_value_heads=int(config["num_key_value_heads"]),
        head_dim=int(config["head_dim"]),
        intermediate_size=int(config["intermediate_size"]),
        rope_theta=float(config["rope_theta"]),
        rms_norm_eps=float(config["rms_norm_eps"]),
        tie_word_embeddings=tie_word_embeddings,
    )


def _make_streamer(spec: QwenStreamedModelSpec) -> QwenLayerStreamer:
    return QwenLayerStreamer(
        spec=spec,
        loader=SafetensorWeightLoader(spec.weight_map, torch.float32),
        model_label="fake-qwen",
    )


def test_prefill_writes_kv_for_each_layer(tmp_path: Path) -> None:
    config = dict(_QWEN_TEST_CONFIG)
    tie_word_embeddings = False
    weight_map = _write_checkpoint(tmp_path, config, tie_word_embeddings)
    spec = _make_spec(tmp_path, config, weight_map, tie_word_embeddings)
    streamer = _make_streamer(spec)
    cache = KVSpillStore(root=tmp_path, model_label="qwen", prompt_index=0)

    input_ids = torch.tensor([[1, 2, 3]], dtype=torch.long)
    hidden = streamer.prefill(input_ids, cache)

    assert hidden.shape == (1, 3, int(config["hidden_size"]))
    for layer_index in range(int(config["num_hidden_layers"])):
        stored = cache.read(layer_index)
        assert stored is not None
        assert stored.key.shape == (1, 3, int(config["num_key_value_heads"]), int(config["head_dim"]))
        assert stored.value.shape == (1, 3, int(config["num_key_value_heads"]), int(config["head_dim"]))


def test_decode_one_appends_kv_and_updates_hidden_shape(tmp_path: Path) -> None:
    config = dict(_QWEN_TEST_CONFIG)
    tie_word_embeddings = False
    weight_map = _write_checkpoint(tmp_path, config, tie_word_embeddings)
    spec = _make_spec(tmp_path, config, weight_map, tie_word_embeddings)
    streamer = _make_streamer(spec)
    cache = KVSpillStore(root=tmp_path, model_label="qwen", prompt_index=1)

    prompt = torch.tensor([[4, 5, 6]], dtype=torch.long)
    prefill_hidden = streamer.prefill(prompt, cache)
    decoded_hidden = streamer.decode_one(token_id=7, position=prompt.size(1), cache=cache)

    assert prefill_hidden.shape == (1, 3, int(config["hidden_size"]))
    assert decoded_hidden.shape == (1, 1, int(config["hidden_size"]))
    for layer_index in range(int(config["num_hidden_layers"])):
        stored = cache.read(layer_index)
        assert stored is not None
        assert stored.key.shape == (1, 4, int(config["num_key_value_heads"]), int(config["head_dim"]))
        assert stored.value.shape == (1, 4, int(config["num_key_value_heads"]), int(config["head_dim"]))


def test_logits_from_hidden_returns_vocab_logits(tmp_path: Path) -> None:
    config = dict(_QWEN_TEST_CONFIG)
    tie_word_embeddings = False
    weight_map = _write_checkpoint(tmp_path, config, tie_word_embeddings)
    spec = _make_spec(tmp_path, config, weight_map, tie_word_embeddings)
    streamer = _make_streamer(spec)

    hidden = torch.randn(1, 3, int(config["hidden_size"]), dtype=torch.float32)
    logits = streamer.logits_from_hidden(hidden)

    assert logits.shape == (int(config["vocab_size"]),)
    assert logits.dtype == torch.float32
    assert logits.device == torch.device("cpu")


def test_decode_one_missing_cache_layer_raises_clear_error(tmp_path: Path) -> None:
    config = dict(_QWEN_TEST_CONFIG)
    tie_word_embeddings = False
    weight_map = _write_checkpoint(tmp_path, config, tie_word_embeddings)
    spec = _make_spec(
        tmp_path=tmp_path,
        config=config,
        weight_map=weight_map,
        tie_word_embeddings=tie_word_embeddings,
    )
    streamer = _make_streamer(spec)
    cache = KVSpillStore(root=tmp_path, model_label="qwen", prompt_index=2)

    with pytest.raises(ValueError, match="layer 0"):
        streamer.decode_one(token_id=1, position=3, cache=cache)


def test_logit_head_uses_tied_embeddings_when_configured(tmp_path: Path) -> None:
    config = dict(_QWEN_TEST_CONFIG)
    config["tie_word_embeddings"] = True
    weight_map = _write_checkpoint(tmp_path, config, tie_word_embeddings=True)
    spec = _make_spec(tmp_path, config, weight_map, tie_word_embeddings=True)
    streamer = _make_streamer(spec)

    hidden = torch.randn(1, 2, int(config["hidden_size"]), dtype=torch.float32)
    logits = streamer.logits_from_hidden(hidden)

    assert logits.shape == (int(config["vocab_size"]),)
