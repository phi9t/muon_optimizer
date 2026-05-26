"""Tests for streamed Qwen memory guard behavior."""

from __future__ import annotations

import pytest

from scripts.qwen_streaming.memory import MemoryGuard


class _FakeMemoryInfo:
    def __init__(self, rss: int) -> None:
        self.rss = rss


class _FakeProcess:
    def __init__(self, rss: int) -> None:
        self._rss = rss

    def memory_info(self) -> _FakeMemoryInfo:
        return _FakeMemoryInfo(self._rss)


@pytest.mark.parametrize("rss_gb", [0.25])
def test_memory_guard_raises_when_cap_is_too_low(monkeypatch, rss_gb: float) -> None:
    monkeypatch.setattr("scripts.qwen_streaming.memory.psutil.Process", lambda: _FakeProcess(int(rss_gb * 1024**3)))

    guard = MemoryGuard(cap_gb=0.1)
    with pytest.raises(MemoryError) as exc_info:
        guard.check("prefill/layer-003")

    message = str(exc_info.value)
    assert "prefill/layer-003" in message
    assert "GiB" in message


def test_memory_guard_returns_rss_bytes(monkeypatch) -> None:
    fake_rss = int(0.25 * 1024**3)
    monkeypatch.setattr("scripts.qwen_streaming.memory.psutil.Process", lambda: _FakeProcess(fake_rss))

    guard = MemoryGuard(cap_gb=1.0)
    assert guard.check("decode/layer-001") == fake_rss
