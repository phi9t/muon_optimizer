"""Layer-by-layer Qwen execution scaffolding (Task 1)."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class QwenLayerStreamer:
    """Placeholder for future streamed Qwen layer execution."""

    def prefill(self, *args, **kwargs):
        raise NotImplementedError("streamed Qwen execution is not implemented yet")

    def decode_one(self, *args, **kwargs):
        raise NotImplementedError("streamed Qwen execution is not implemented yet")

