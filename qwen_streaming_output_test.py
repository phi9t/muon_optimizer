"""Tests for streamed Qwen comparison output and orchestration."""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from scripts.qwen_streaming import comparison


class FakeTokenizer:
    def __init__(self, token_map: dict[int, str], prompts: dict[str, list[int]], eos_token_id: int | None):
        self.token_map = dict(token_map)
        self.prompts = prompts
        self.eos_token_id = eos_token_id
        self.name_or_path = "fake/tokenizer"

    def __len__(self) -> int:
        return len(self.token_map)

    def convert_ids_to_tokens(self, token_id: int) -> str:
        return self.token_map[int(token_id)]

    def decode(self, token_ids: list[int], skip_special_tokens: bool = True) -> str:
        return "".join(self.convert_ids_to_tokens(int(token_id)) for token_id in token_ids)

    def __call__(self, prompt: str, return_tensors: str = "pt", add_special_tokens: bool = True):
        ids = self.prompts[prompt]
        input_ids = torch.tensor([ids], dtype=torch.long)
        attention_mask = torch.ones_like(input_ids)
        return {"input_ids": input_ids, "attention_mask": attention_mask}


class FakeMemoryGuard:
    def __init__(self, cap_gb: float) -> None:
        self.cap_gb = cap_gb

    def check(self, label: str) -> int:
        return 0


class FakeLoader:
    def __init__(self, weight_map, dtype):
        self.weight_map = weight_map
        self.dtype = dtype


class FakeSpec:
    def __init__(self, model_id: str, vocab_size: int):
        self.model_id = model_id
        self.local_dir = Path(".")
        self.config = {}
        self.weight_map = {f"{model_id}-weights": Path("fake.safetensors")}
        self.vocab_size = vocab_size

    def validate_required_tensors(self) -> None:
        return None


class FakeStreamer:
    def __init__(self, spec: FakeSpec, model_label: str, logits_by_step: list[torch.Tensor]):
        self.spec = spec
        self.model_label = model_label
        self.logits_by_step = logits_by_step
        self.logit_call_index = 0
        self.decode_calls: list[tuple[int, int]] = []
        self.prefill_called = False

    def prefill(self, input_ids: torch.Tensor, cache) -> int:
        self.prefill_called = True
        return 0

    def decode_one(self, token_id: int, position: int, cache) -> int:
        self.decode_calls.append((token_id, position))
        return 0

    def logits_from_hidden(self, hidden: int) -> torch.Tensor:
        if self.logit_call_index >= len(self.logits_by_step):
            raise AssertionError("Unexpected extra logits request from comparison loop.")

        logits = self.logits_by_step[self.logit_call_index]
        self.logit_call_index += 1
        return logits


def _make_fake_tokenizer() -> FakeTokenizer:
    return FakeTokenizer(
        token_map={
            0: "<pad>",
            1: "</s>",
            2: "A",
            3: "B",
            4: "C",
            5: "D",
        },
        prompts={"prompt one": [2, 3]},
        eos_token_id=1,
    )


def _make_args(
    tmp_path: Path,
    *,
    top_k: int = 3,
    limit_prompts: int = 1,
    max_new_tokens: int = 2,
    memory_cap_gb: float = 1.0,
) -> SimpleNamespace:
    return SimpleNamespace(
        top_k=top_k,
        limit_prompts=limit_prompts,
        max_new_tokens=max_new_tokens,
        memory_cap_gb=memory_cap_gb,
        kv_cache_dir=tmp_path / "kv_cache",
        output=tmp_path / "qwen_logits.json",
        student_model="student",
        teacher_model="teacher",
        dry_plan=False,
    )


def test_streamed_comparison_shape_and_step_metrics(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    tokenizer = _make_fake_tokenizer()

    student_logits = [
        torch.tensor([0.1, 1.0, 2.4, 0.7, 0.3], dtype=torch.float32),
        torch.tensor([0.2, 0.8, 1.1, 0.4, 3.6], dtype=torch.float32),
    ]
    teacher_logits = [
        torch.tensor([1.1, 0.8, 2.0, 0.2, 0.4], dtype=torch.float32),
        torch.tensor([0.6, 1.7, 0.8, 0.2, 1.9], dtype=torch.float32),
    ]

    fake_student = FakeStreamer(
        spec=FakeSpec("student", vocab_size=len(tokenizer)),
        model_label="student",
        logits_by_step=student_logits,
    )
    fake_teacher = FakeStreamer(
        spec=FakeSpec("teacher", vocab_size=len(tokenizer)),
        model_label="teacher",
        logits_by_step=teacher_logits,
    )

    def fake_loader(*, weight_map, dtype):  # noqa: ARG001
        return FakeLoader(weight_map=weight_map, dtype=dtype)

    def fake_from_model_id(model_id: str) -> FakeSpec:
        return FakeSpec(model_id=model_id, vocab_size=len(tokenizer))

    def fake_streamer_ctor(*, spec, loader, memory_guard, model_label=None) -> FakeStreamer:
        if model_label == "student":
            return fake_student
        return fake_teacher

    def fake_transformers():
        class AutoTokenizer:
            @staticmethod
            def from_pretrained(model_id: str) -> FakeTokenizer:
                return tokenizer

        return (AutoTokenizer,)

    monkeypatch.setattr(comparison, "_load_transformers", fake_transformers)
    monkeypatch.setattr(comparison, "SafetensorWeightLoader", fake_loader)
    monkeypatch.setattr(comparison, "QwenStreamedModelSpec", SimpleNamespace(from_model_id=fake_from_model_id))
    monkeypatch.setattr(comparison, "QwenLayerStreamer", fake_streamer_ctor)
    monkeypatch.setattr(comparison, "MemoryGuard", FakeMemoryGuard)

    args = _make_args(tmp_path=tmp_path)
    comparison.run_streamed_comparison(args, prompts=["prompt one"])

    payload = json.loads((tmp_path / "qwen_logits.json").read_text(encoding="utf-8"))
    assert payload["metadata"]["mode"] == "streamed"
    assert len(payload["prompts"]) == 1
    prompt_payload = payload["prompts"][0]

    assert prompt_payload["prompt"] == "prompt one"
    assert prompt_payload["steps_count"] == 2
    assert prompt_payload["generated_token_ids"] == [2, 4]
    assert prompt_payload["generated_text"] == "AC"
    assert prompt_payload["steps"][0]["step_index"] == 0
    assert prompt_payload["steps"][0]["generated_token_id"] == 2
    assert prompt_payload["steps"][1]["step_index"] == 1
    assert prompt_payload["steps"][1]["generated_token_id"] == 4

    step_keys = {
        "step_index",
        "generated_token_id",
        "generated_token",
        "top_teacher_tokens",
        "top_student_tokens",
        "overlapping_top_k_tokens",
        "ranked_logit_deltas",
        "kl_divergence",
        "cosine_similarity",
        "mean_absolute_logit_delta",
        "max_absolute_logit_delta",
    }
    assert step_keys.issubset(prompt_payload["steps"][0].keys())


def test_streamed_comparison_uses_student_argmax_and_shared_positions(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    tokenizer = _make_fake_tokenizer()

    student_logits = [
        torch.tensor([0.1, 1.0, 2.4, 0.7, 0.3], dtype=torch.float32),
        torch.tensor([0.2, 0.8, 1.1, 0.4, 3.6], dtype=torch.float32),
    ]
    teacher_logits = [
        torch.tensor([1.1, 0.8, 2.0, 0.2, 0.4], dtype=torch.float32),
        torch.tensor([0.6, 1.7, 0.8, 0.2, 1.9], dtype=torch.float32),
    ]

    fake_student = FakeStreamer(
        spec=FakeSpec("student", vocab_size=len(tokenizer)),
        model_label="student",
        logits_by_step=student_logits,
    )
    fake_teacher = FakeStreamer(
        spec=FakeSpec("teacher", vocab_size=len(tokenizer)),
        model_label="teacher",
        logits_by_step=teacher_logits,
    )

    def fake_loader(*, weight_map, dtype):  # noqa: ARG001
        return FakeLoader(weight_map=weight_map, dtype=dtype)

    def fake_from_model_id(model_id: str) -> FakeSpec:
        return FakeSpec(model_id=model_id, vocab_size=len(tokenizer))

    def fake_streamer_ctor(*, spec, loader, memory_guard, model_label=None) -> FakeStreamer:
        if model_label == "student":
            return fake_student
        return fake_teacher

    def fake_transformers():
        class AutoTokenizer:
            @staticmethod
            def from_pretrained(model_id: str) -> FakeTokenizer:
                return tokenizer

        return (AutoTokenizer,)

    monkeypatch.setattr(comparison, "_load_transformers", fake_transformers)
    monkeypatch.setattr(comparison, "SafetensorWeightLoader", fake_loader)
    monkeypatch.setattr(comparison, "QwenStreamedModelSpec", SimpleNamespace(from_model_id=fake_from_model_id))
    monkeypatch.setattr(comparison, "QwenLayerStreamer", fake_streamer_ctor)
    monkeypatch.setattr(comparison, "MemoryGuard", FakeMemoryGuard)

    args = _make_args(tmp_path=tmp_path)
    comparison.run_streamed_comparison(args, prompts=["prompt one"])

    assert fake_student.decode_calls == [(2, 2)]
    assert fake_teacher.decode_calls == fake_student.decode_calls


@pytest.mark.parametrize(
    "bad_field,bad_value",
    [
        ("top_k", 0),
        ("max_new_tokens", 0),
        ("memory_cap_gb", 0.0),
        ("limit_prompts", 0),
    ],
)
def test_streamed_comparison_invalid_args(tmp_path: Path, bad_field: str, bad_value: float) -> None:
    args = _make_args(
        tmp_path=tmp_path,
        top_k=3,
        max_new_tokens=2,
        memory_cap_gb=1.0,
        limit_prompts=1,
    )
    setattr(args, bad_field, bad_value)
    with pytest.raises(ValueError):
        comparison.run_streamed_comparison(args, prompts=["prompt one"])
