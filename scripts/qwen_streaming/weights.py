"""Weight loading helpers for the streamed Qwen path (Task 1 scaffold)."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Iterable, Mapping


@dataclass
class SafetensorWeightLoader:
    """Placeholder loader class for Task 1.

    The implementation is intentionally minimal and non-functional; later tasks
    will replace this with checkpoint-aware loading logic.
    """

    manifest: Mapping[str, Any] | None = None

    def load_tensor(self, name: str) -> object:
        raise NotImplementedError("streamed Qwen loading is not implemented yet")

    def load_tensors(self, names: Iterable[str]) -> Dict[str, object]:
        raise NotImplementedError("streamed Qwen loading is not implemented yet")

