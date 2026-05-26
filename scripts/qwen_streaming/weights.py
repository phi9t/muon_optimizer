"""Weight loading helpers for streaming Qwen checkpoints."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, Mapping

import torch
from safetensors import safe_open


@dataclass
class SafetensorWeightLoader:
    """Load tensors from safetensors shards without reading full checkpoints."""

    weight_map: Mapping[str, Path]
    dtype: torch.dtype

    def load_tensor(self, name: str) -> torch.Tensor:
        if name not in self.weight_map:
            raise KeyError(f"Tensor {name!r} is not present in weight map.")

        path = Path(self.weight_map[name])
        with safe_open(path, framework="pt", device="cpu") as f:
            tensor = f.get_tensor(name)
        return tensor.to(device="cpu", dtype=self.dtype)

    def load_tensors(self, names: Iterable[str]) -> Dict[str, torch.Tensor]:
        return {name: self.load_tensor(name) for name in names}
