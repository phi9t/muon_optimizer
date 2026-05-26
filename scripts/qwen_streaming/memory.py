"""Memory guard scaffolding for streamed Qwen execution (Task 1)."""

from __future__ import annotations


class MemoryGuard:
    """Temporary placeholder that documents the streaming memory contract."""

    def __init__(self, cap_gb: float) -> None:
        if cap_gb <= 0:
            raise ValueError("--memory-cap-gb must be a positive value.")
        self.cap_gb = float(cap_gb)

    def check(self, label: str) -> int:
        raise NotImplementedError("streamed memory guard is not implemented yet")

