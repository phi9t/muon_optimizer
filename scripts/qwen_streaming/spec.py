"""Model specifications for the streamed Qwen path (Task 1 scaffold)."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class QwenStreamedModelSpec:
    """Minimal metadata container for streamed model execution."""

    model_id: str

