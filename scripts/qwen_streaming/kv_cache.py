"""KV cache spill abstractions for streamed Qwen execution."""

from __future__ import annotations

import shutil
import tempfile
from dataclasses import dataclass
from pathlib import Path

import torch


@dataclass(frozen=True)
class LayerKV:
    """Container for one layer's attention key/value cache tensors."""

    key: torch.Tensor
    value: torch.Tensor


class KVSpillStore:
    """Cache directory writer/reader for streamed attention key-value tensors."""

    def __init__(self, root: Path, model_label: str, prompt_index: int) -> None:
        self.root = Path(root)
        self.model_label = model_label
        self.prompt_index = int(prompt_index)
        self.prompt_root = self.root / model_label / f"prompt-{prompt_index:03d}"
        self.prompt_root.mkdir(parents=True, exist_ok=True)

    @staticmethod
    def _layer_filename(layer_index: int) -> str:
        return f"layer-{layer_index:03d}.pt"

    def _layer_path(self, layer_index: int) -> Path:
        return self.prompt_root / self._layer_filename(layer_index=layer_index)

    def read(self, layer_index: int) -> LayerKV | None:
        path = self._layer_path(layer_index=layer_index)
        if not path.is_file():
            return None

        payload = torch.load(path, map_location="cpu", weights_only=True)
        return LayerKV(key=payload["key"], value=payload["value"])

    def write(self, layer_index: int, kv: LayerKV) -> None:
        path = self._layer_path(layer_index=layer_index)
        path.parent.mkdir(parents=True, exist_ok=True)

        with tempfile.NamedTemporaryFile(
            dir=path.parent, prefix=f".{path.name}.", suffix=".tmp", delete=False
        ) as temp_file:
            temp_path = Path(temp_file.name)

        try:
            torch.save({"key": kv.key, "value": kv.value}, temp_path)
            temp_path.replace(path)
        finally:
            if temp_path.exists():
                temp_path.unlink(missing_ok=True)

    def clear(self) -> None:
        shutil.rmtree(self.prompt_root, ignore_errors=True)
