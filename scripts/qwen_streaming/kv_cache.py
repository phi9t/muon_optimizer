"""KV cache spill abstractions for streamed Qwen execution (Task 1)."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class LayerKV:
    """Container for one layer's key/value cache tensors."""

    key: object
    value: object


class KVSpillStore:
    """Placeholder cache directory writer/reader."""

    def __init__(self, cache_root: str | Path) -> None:
        self.cache_root = Path(cache_root)

    def read(self, layer_index: int) -> LayerKV | None:
        raise NotImplementedError("streamed KV cache is not implemented yet")

    def write(self, layer_index: int, kv: LayerKV) -> None:
        raise NotImplementedError("streamed KV cache is not implemented yet")

    def clear(self) -> None:
        raise NotImplementedError("streamed KV cache is not implemented yet")

