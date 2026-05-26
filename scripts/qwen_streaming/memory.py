"""RSS memory guard for streamed Qwen execution."""

from __future__ import annotations

import psutil


class MemoryGuard:
    """Track resident set size and raise when the cap is exceeded."""

    _GIB_BYTES = 1024**3

    def __init__(self, cap_gb: float) -> None:
        if cap_gb <= 0:
            raise ValueError("--memory-cap-gb must be a positive value.")
        self.cap_gb = float(cap_gb)
        self._cap_bytes = int(self.cap_gb * self._GIB_BYTES)

    def check(self, label: str) -> int:
        process = psutil.Process()
        rss_bytes = int(process.memory_info().rss)

        if rss_bytes > self._cap_bytes:
            current_gib = rss_bytes / self._GIB_BYTES
            raise MemoryError(
                f"Memory cap exceeded at {label}: RSS={current_gib:.3f} GiB, "
                f"cap={self.cap_gb:.3f} GiB."
            )

        return rss_bytes
