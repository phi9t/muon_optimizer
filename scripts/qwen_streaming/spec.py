"""Model metadata planning for streaming Qwen checkpoints."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from huggingface_hub import snapshot_download


_WEIGHT_ALLOW_PATTERNS = (
    "config.json",
    "*.json",
    "*.safetensors",
    "*.safetensors.index.json",
    "tokenizer.model",
    "tokenizer.json",
    "tokenizer_config.json",
    "special_tokens_map.json",
)

_QWEN3_LAYERS = (
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


def _ensure_positive_int(config: dict[str, Any], key: str) -> int:
    value = config.get(key)
    if not isinstance(value, int):
        raise ValueError(f"{key} must be an integer in config.json.")
    if value <= 0:
        raise ValueError(f"{key} must be positive.")
    return value


def _ensure_positive_float(config: dict[str, Any], key: str) -> float:
    value = config.get(key)
    if not isinstance(value, (int, float)):
        raise ValueError(f"{key} must be a number in config.json.")
    if float(value) <= 0.0:
        raise ValueError(f"{key} must be positive.")
    return float(value)


def _validate_qwen3_architecture(config: dict[str, Any]) -> None:
    model_type = str(config.get("model_type", "")).lower()
    architectures = config.get("architectures", [])
    if isinstance(architectures, str):
        architectures = [architectures]

    if model_type != "qwen3" and not (
        any(isinstance(arch, str) and "qwen3" in arch.lower() for arch in architectures)
    ):
        raise ValueError(
            "Only Qwen3 causal-LM checkpoints are supported. "
            f"Got model_type={model_type!r}, architectures={architectures!r}."
        )


def _read_model_config(local_dir: Path) -> dict[str, Any]:
    config_path = local_dir / "config.json"
    if not config_path.is_file():
        raise FileNotFoundError(
            f"Could not find config.json in snapshot directory: {local_dir}"
        )
    config = json.loads(config_path.read_text(encoding="utf-8"))
    if not isinstance(config, dict):
        raise ValueError("config.json must contain a JSON object.")
    return config


def _load_index_weight_map(local_dir: Path) -> dict[str, Path]:
    index_path = local_dir / "model.safetensors.index.json"
    with index_path.open("r", encoding="utf-8") as fp:
        payload = json.load(fp)

    weight_map_raw = payload.get("weight_map")
    if not isinstance(weight_map_raw, dict):
        raise ValueError("model.safetensors.index.json is missing a valid weight_map.")

    return {
        tensor_name: local_dir / shard_path
        for tensor_name, shard_path in weight_map_raw.items()
    }


def _load_single_file_weight_map(local_dir: Path) -> dict[str, Path]:
    from safetensors import safe_open

    single_file = local_dir / "model.safetensors"
    if not single_file.is_file():
        raise FileNotFoundError("Expected model.safetensors but found no single file.")

    with safe_open(single_file, framework="pt", device="cpu") as f:
        names = list(f.keys())

    return {name: single_file for name in names}


@dataclass(frozen=True)
class QwenStreamedModelSpec:
    """Validated metadata for streamed Qwen checkpoint execution."""

    model_id: str
    local_dir: Path
    config: dict[str, Any]
    weight_map: dict[str, Path]
    num_hidden_layers: int
    hidden_size: int
    vocab_size: int
    num_attention_heads: int
    num_key_value_heads: int
    head_dim: int
    intermediate_size: int
    rope_theta: float
    rms_norm_eps: float
    tie_word_embeddings: bool
    max_position_embeddings: int | None

    @classmethod
    def from_model_id(cls, model_id: str) -> "QwenStreamedModelSpec":
        local_dir = Path(
            snapshot_download(repo_id=model_id, allow_patterns=_WEIGHT_ALLOW_PATTERNS)
        ).resolve()

        config = _read_model_config(local_dir)
        _validate_qwen3_architecture(config)

        num_hidden_layers = _ensure_positive_int(config, "num_hidden_layers")
        hidden_size = _ensure_positive_int(config, "hidden_size")
        vocab_size = _ensure_positive_int(config, "vocab_size")
        num_attention_heads = _ensure_positive_int(config, "num_attention_heads")
        num_key_value_heads = _ensure_positive_int(config, "num_key_value_heads")
        head_dim = _ensure_positive_int(config, "head_dim")
        intermediate_size = _ensure_positive_int(config, "intermediate_size")
        rope_theta = _ensure_positive_float(config, "rope_theta")
        rms_norm_eps = _ensure_positive_float(config, "rms_norm_eps")
        tie_word_embeddings = bool(config.get("tie_word_embeddings", False))
        max_position_embeddings_raw = config.get("max_position_embeddings")
        max_position_embeddings = (
            _ensure_positive_int(config, "max_position_embeddings")
            if max_position_embeddings_raw is not None
            else None
        )

        if num_attention_heads % num_key_value_heads != 0:
            raise ValueError(
                "num_attention_heads must be divisible by num_key_value_heads for Qwen3 GQA."
            )

        if hidden_size % num_attention_heads != 0:
            raise ValueError("hidden_size must be divisible by num_attention_heads.")

        index_path = local_dir / "model.safetensors.index.json"
        if index_path.is_file():
            weight_map = _load_index_weight_map(local_dir)
        else:
            weight_map = _load_single_file_weight_map(local_dir)

        return cls(
            model_id=model_id,
            local_dir=local_dir,
            config=config,
            weight_map=weight_map,
            num_hidden_layers=num_hidden_layers,
            hidden_size=hidden_size,
            vocab_size=vocab_size,
            num_attention_heads=num_attention_heads,
            num_key_value_heads=num_key_value_heads,
            head_dim=head_dim,
            intermediate_size=intermediate_size,
            rope_theta=rope_theta,
            rms_norm_eps=rms_norm_eps,
            tie_word_embeddings=tie_word_embeddings,
            max_position_embeddings=max_position_embeddings,
        )

    def required_tensor_names(self) -> list[str]:
        names: list[str] = [
            "model.embed_tokens.weight",
            "model.norm.weight",
        ]

        for layer_index in range(self.num_hidden_layers):
            prefix = f"model.layers.{layer_index}."
            for suffix in _QWEN3_LAYERS:
                names.append(prefix + suffix)

        if not self.tie_word_embeddings:
            names.append("lm_head.weight")

        return names

    def validate_required_tensors(self) -> None:
        missing = [name for name in self.required_tensor_names() if name not in self.weight_map]
        if missing:
            formatted = ", ".join(sorted(missing))
            raise ValueError(f"Missing required tensors in weight map: {formatted}")
