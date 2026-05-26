"""Streamed comparison orchestration for Qwen streamed execution."""

from __future__ import annotations

import gc
import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Tuple

import torch

from .kv_cache import KVSpillStore
from .memory import MemoryGuard
from .spec import QwenStreamedModelSpec
from .streamer import QwenLayerStreamer
from .weights import SafetensorWeightLoader

DEFAULT_PROMPTS: List[str] = [
    "The world breaks everyone, and afterward, many are strong at the broken places.",
    "What is 7 + 5?",
    "Write a tiny poem about a cat.",
    "Explain why gradients matter in optimization.",
    "Generate a quick planning checklist for a short trip.",
    "Name three colors and one common use for each.",
]


def _load_transformers() -> Tuple[Any, Any]:
    try:
        from transformers import AutoTokenizer
    except ModuleNotFoundError as exc:
        raise RuntimeError(
            "transformers is required to run streamed Qwen comparison. "
            "Use `uv run --group qwen-logits python scripts/run_qwen_logits_comparison.py` "
            "or install transformers manually."
        ) from exc
    return (AutoTokenizer,)


def _resolve_hf_cache_dir(raw_cache_dir: Path | None) -> Path | None:
    if raw_cache_dir is None:
        return None
    return Path(raw_cache_dir).resolve()


def _validate_prompt_count(prompts: List[str], limit: int) -> List[str]:
    if limit <= 0:
        raise ValueError("--limit-prompts must be a positive integer.")
    selected = prompts[:limit]
    if not selected:
        raise ValueError("No prompts selected; check --limit-prompts.")
    return selected


def _validate_tokenizer_compatibility(teacher_tokenizer: Any, student_tokenizer: Any) -> None:
    teacher_vocab = int(len(teacher_tokenizer))
    student_vocab = int(len(student_tokenizer))

    if teacher_vocab != student_vocab:
        raise ValueError(
            "Incompatible tokenizers detected. Expected teacher and student tokenizers to have "
            f"the same vocabulary size. Got teacher={teacher_vocab}, student={student_vocab}."
        )

    for token_id in range(teacher_vocab):
        teacher_token = teacher_tokenizer.convert_ids_to_tokens(token_id)
        student_token = student_tokenizer.convert_ids_to_tokens(token_id)
        if teacher_token != student_token:
            raise ValueError(
                "Incompatible tokenizers detected. Token IDs do not map to the same tokens. "
                f"First mismatch at id={token_id}: teacher={teacher_token!r}, student={student_token!r}."
            )


def _safe_float(value: torch.Tensor) -> float:
    return float(value.detach().to(torch.float32).item())


def _token_label(tokenizer: Any, token_id: int) -> str:
    try:
        token = tokenizer.convert_ids_to_tokens(int(token_id))
    except (KeyError, IndexError, TypeError, ValueError):
        token = None

    if token is None:
        return f"[token_{int(token_id)}]"
    return str(token)


def _decode_token_ids(tokenizer: Any, token_ids: List[int]) -> str:
    if not token_ids:
        return ""
    try:
        decoded = tokenizer.decode(token_ids, skip_special_tokens=True)
    except (KeyError, IndexError, TypeError, ValueError):
        decoded = ""

    if decoded:
        return str(decoded)
    return "".join(_token_label(tokenizer, token_id) for token_id in token_ids)


def _topk_payload(logits: torch.Tensor, tokenizer: Any, top_k: int) -> List[Dict[str, Any]]:
    k = min(int(top_k), int(logits.numel()))
    values, indices = torch.topk(logits, k=k, dim=0)
    probs = torch.softmax(logits, dim=0)

    out: List[Dict[str, Any]] = []
    for rank, (value, index) in enumerate(zip(values.tolist(), indices.tolist()), start=1):
        token = _token_label(tokenizer, int(index))
        out.append(
            {
                "rank": rank,
                "token_id": int(index),
                "token": token,
                "logit": float(value),
                "probability": float(probs[int(index)].item()),
            }
        )
    return out


def _overlap_payload(
    teacher_topk: List[Dict[str, Any]],
    student_topk: List[Dict[str, Any]],
) -> List[Dict[str, Any]]:
    student_ranks = {entry["token_id"]: entry["rank"] for entry in student_topk}
    overlap: List[Dict[str, Any]] = []

    for entry in teacher_topk:
        token_id = entry["token_id"]
        if token_id in student_ranks:
            overlap.append(
                {
                    "token_id": token_id,
                    "token": entry["token"],
                    "teacher_rank": entry["rank"],
                    "student_rank": student_ranks[token_id],
                }
            )
    return overlap


def _rank_lookup(entries: List[Dict[str, Any]]) -> Dict[int, int]:
    return {int(entry["token_id"]): int(entry["rank"]) for entry in entries}


def _ranked_delta_payload(
    teacher_logits: torch.Tensor,
    student_logits: torch.Tensor,
    teacher_topk: List[Dict[str, Any]],
    student_topk: List[Dict[str, Any]],
    tokenizer: Any,
) -> List[Dict[str, Any]]:
    teacher_probs = torch.softmax(teacher_logits, dim=0)
    student_probs = torch.softmax(student_logits, dim=0)
    teacher_ranks = _rank_lookup(teacher_topk)
    student_ranks = _rank_lookup(student_topk)
    token_ids = sorted(set(teacher_ranks) | set(student_ranks))

    rows: List[Dict[str, Any]] = []
    for token_id in token_ids:
        teacher_logit = float(teacher_logits[token_id].item())
        student_logit = float(student_logits[token_id].item())
        delta = student_logit - teacher_logit
        rows.append(
            {
                "token_id": token_id,
                "token": _token_label(tokenizer, token_id),
                "teacher_rank": teacher_ranks.get(token_id),
                "student_rank": student_ranks.get(token_id),
                "teacher_logit": teacher_logit,
                "student_logit": student_logit,
                "teacher_probability": float(teacher_probs[token_id].item()),
                "student_probability": float(student_probs[token_id].item()),
                "logit_delta": delta,
                "absolute_logit_delta": abs(delta),
            }
        )

    rows.sort(key=lambda row: row["absolute_logit_delta"], reverse=True)
    return rows


def _compute_metrics(
    teacher_logits: torch.Tensor,
    student_logits: torch.Tensor,
    tokenizer: Any,
    top_k: int,
) -> Dict[str, Any]:
    if teacher_logits.shape != student_logits.shape:
        raise ValueError(
            "Mismatched logits shape between teacher and student for comparison: "
            f"{teacher_logits.shape} vs {student_logits.shape}"
        )

    teacher_topk = _topk_payload(teacher_logits, tokenizer, top_k)
    student_topk = _topk_payload(student_logits, tokenizer, top_k)
    overlap = _overlap_payload(teacher_topk, student_topk)
    ranked_deltas = _ranked_delta_payload(
        teacher_logits,
        student_logits,
        teacher_topk,
        student_topk,
        tokenizer,
    )

    teacher_log_probs = torch.log_softmax(teacher_logits, dim=0)
    student_log_probs = torch.log_softmax(student_logits, dim=0)
    teacher_probs = torch.exp(teacher_log_probs)

    kl_divergence = torch.sum(teacher_probs * (teacher_log_probs - student_log_probs))
    cosine_similarity = torch.nn.functional.cosine_similarity(
        teacher_logits,
        student_logits,
        dim=0,
    )
    logit_delta = student_logits - teacher_logits
    abs_delta = torch.abs(logit_delta)

    return {
        "top_teacher_tokens": teacher_topk,
        "top_student_tokens": student_topk,
        "overlapping_top_k_tokens": {
            "count": len(overlap),
            "tokens": overlap,
        },
        "ranked_logit_deltas": ranked_deltas,
        "kl_divergence": _safe_float(kl_divergence),
        "cosine_similarity": _safe_float(cosine_similarity),
        "mean_absolute_logit_delta": _safe_float(abs_delta.mean()),
        "max_absolute_logit_delta": _safe_float(abs_delta.max()),
    }


def _validate_vocab_sizes(
    shared_tokenizer: Any,
    teacher_spec: QwenStreamedModelSpec,
    student_spec: QwenStreamedModelSpec,
) -> None:
    tokenizer_vocab = int(len(shared_tokenizer))
    if tokenizer_vocab <= 0:
        raise ValueError("Tokenizer vocabulary size is not positive.")
    if teacher_spec.vocab_size != student_spec.vocab_size:
        raise ValueError(
            "Incompatible vocabularies detected. Teacher and student model vocab sizes differ. "
            f"Got teacher={teacher_spec.vocab_size}, student={student_spec.vocab_size}."
        )
    if tokenizer_vocab > teacher_spec.vocab_size:
        raise ValueError(
            "Incompatible vocabularies detected. Tokenizer exposes more IDs than the model output head. "
            f"Got tokenizer={tokenizer_vocab}, model={teacher_spec.vocab_size}."
        )


def _vocab_plan_payload(
    shared_tokenizer: Any,
    teacher_spec: QwenStreamedModelSpec,
    student_spec: QwenStreamedModelSpec,
) -> Dict[str, Any]:
    tokenizer_vocab = int(len(shared_tokenizer))
    issues: list[str] = []
    if teacher_spec.vocab_size != student_spec.vocab_size:
        issues.append(
            f"teacher spec vocab_size={teacher_spec.vocab_size} != student spec vocab_size={student_spec.vocab_size}"
        )
    if tokenizer_vocab > teacher_spec.vocab_size:
        issues.append(
            f"tokenizer vocab_size={tokenizer_vocab} exceeds model vocab_size={teacher_spec.vocab_size}"
        )

    return {
        "tokenizer_vocab_size": tokenizer_vocab,
        "teacher_vocab_size": teacher_spec.vocab_size,
        "student_vocab_size": student_spec.vocab_size,
        "reserved_model_token_count": max(teacher_spec.vocab_size - tokenizer_vocab, 0),
        "compatible": len(issues) == 0,
        "issues": issues,
    }


def _estimate_max_cache_bytes_for_model(
    spec: QwenStreamedModelSpec,
    prompt_lengths: List[int],
    max_new_tokens: int,
) -> int:
    if not prompt_lengths:
        return 0

    max_prompt_tokens = max(prompt_lengths)
    max_sequence_tokens = int(max_prompt_tokens) + int(max_new_tokens)
    bytes_per_token_per_layer = 2 * spec.num_key_value_heads * spec.head_dim * 4
    return int(spec.num_hidden_layers * bytes_per_token_per_layer * max_sequence_tokens)


def _kv_cache_plan_payload(
    student_spec: QwenStreamedModelSpec,
    teacher_spec: QwenStreamedModelSpec,
    prompt_lengths: List[int],
    max_new_tokens: int,
    kv_cache_dir: Path,
) -> Dict[str, Any]:
    prompt_count = len(prompt_lengths)
    student_cache_bytes = _estimate_max_cache_bytes_for_model(student_spec, prompt_lengths, max_new_tokens)
    teacher_cache_bytes = _estimate_max_cache_bytes_for_model(teacher_spec, prompt_lengths, max_new_tokens)
    total_per_prompt_cache_bytes = student_cache_bytes + teacher_cache_bytes
    total_run_cache_bytes = total_per_prompt_cache_bytes * max(int(prompt_count), 1)

    return {
        "kv_cache_dir": str(kv_cache_dir),
        "prompt_count": prompt_count,
        "max_prompt_tokens": max(prompt_lengths) if prompt_lengths else 0,
        "generated_tokens_per_prompt": int(max_new_tokens),
        "max_sequence_tokens_per_prompt": (max(prompt_lengths) + int(max_new_tokens))
        if prompt_lengths
        else int(max_new_tokens),
        "per_model_max_kv_bytes": {
            "student": student_cache_bytes,
            "teacher": teacher_cache_bytes,
        },
        "max_kv_cache_bytes": total_per_prompt_cache_bytes,
        "estimated_run_kv_bytes": total_run_cache_bytes,
    }


def _spec_plan_summary(spec: QwenStreamedModelSpec) -> Dict[str, Any]:
    return {
        "model_id": spec.model_id,
        "local_dir": str(spec.local_dir),
        "num_hidden_layers": spec.num_hidden_layers,
        "hidden_size": spec.hidden_size,
        "vocab_size": spec.vocab_size,
        "num_attention_heads": spec.num_attention_heads,
        "num_key_value_heads": spec.num_key_value_heads,
        "head_dim": spec.head_dim,
        "intermediate_size": spec.intermediate_size,
        "rope_theta": spec.rope_theta,
        "rms_norm_eps": spec.rms_norm_eps,
        "tie_word_embeddings": spec.tie_word_embeddings,
        "max_position_embeddings": spec.max_position_embeddings,
        "required_tensor_count": len(spec.required_tensor_names()),
        "required_tensors": spec.required_tensor_names(),
        "weight_files": sorted({str(path) for path in set(spec.weight_map.values())}),
        "shard_count": len(set(str(path) for path in spec.weight_map.values())),
    }


def _describe_prompt_lengths(tokenizer: Any, prompts: List[str]) -> List[int]:
    return [
        int(
            tokenizer(prompt, return_tensors="pt", add_special_tokens=True)["input_ids"].shape[1]
        )
        for prompt in prompts
    ]


def _validate_context_lengths(
    prompt_lengths: List[int],
    max_new_tokens: int,
    teacher_spec: QwenStreamedModelSpec,
    student_spec: QwenStreamedModelSpec,
) -> None:
    if not prompt_lengths:
        return

    required_length = max(prompt_lengths) + int(max_new_tokens)
    context_limits = [
        limit
        for limit in (teacher_spec.max_position_embeddings, student_spec.max_position_embeddings)
        if limit is not None
    ]
    if not context_limits:
        return

    context_limit = min(context_limits)
    if required_length > context_limit:
        raise ValueError(
            "Prompt plus generated tokens exceeds supported Qwen context length. "
            f"Required {required_length} tokens, smallest model limit is {context_limit}."
        )


def _run_dry_plan(args: Any, teacher_spec: QwenStreamedModelSpec, student_spec: QwenStreamedModelSpec, shared_tokenizer: Any, selected_prompts: List[str]) -> None:
    hf_cache_dir = _resolve_hf_cache_dir(getattr(args, "hf_cache_dir", None))
    prompt_lengths = _describe_prompt_lengths(shared_tokenizer, selected_prompts)
    _validate_context_lengths(
        prompt_lengths=prompt_lengths,
        max_new_tokens=int(args.max_new_tokens),
        teacher_spec=teacher_spec,
        student_spec=student_spec,
    )
    kv_plan = _kv_cache_plan_payload(
        student_spec=student_spec,
        teacher_spec=teacher_spec,
        prompt_lengths=prompt_lengths,
        max_new_tokens=int(args.max_new_tokens),
        kv_cache_dir=Path(args.kv_cache_dir).resolve(),
    )
    cache = {
        **kv_plan,
        "hf_cache_dir": str(hf_cache_dir) if hf_cache_dir is not None else None,
        "local_files_only": bool(getattr(args, "offline", False)),
        "snapshot_dirs": {
            "teacher": str(teacher_spec.local_dir),
            "student": str(student_spec.local_dir),
        },
        "memory_cap_gb": float(args.memory_cap_gb),
        "cache_per_model_per_prompt_kv_gib": {
            "student": kv_plan["per_model_max_kv_bytes"]["student"] / (1024**3),
            "teacher": kv_plan["per_model_max_kv_bytes"]["teacher"] / (1024**3),
        },
        "max_run_kv_gib": kv_plan["estimated_run_kv_bytes"] / (1024**3),
    }

    payload = {
        "mode": "streamed",
        "dry_plan": True,
        "student_model": _spec_plan_summary(student_spec),
        "teacher_model": _spec_plan_summary(teacher_spec),
        "vocab": _vocab_plan_payload(
            shared_tokenizer=shared_tokenizer,
            teacher_spec=teacher_spec,
            student_spec=student_spec,
        ),
        "tokenizer": {
            "name_or_path": getattr(shared_tokenizer, "name_or_path", "unknown"),
            "vocab_size": int(len(shared_tokenizer)),
            "eos_token_id": getattr(shared_tokenizer, "eos_token_id", None),
            "prompt_count": len(selected_prompts),
            "prompt_lengths": prompt_lengths,
            "max_prompt_tokens": kv_plan["max_prompt_tokens"],
            "generated_tokens_per_prompt": kv_plan["generated_tokens_per_prompt"],
        },
        "cache_and_memory": cache,
    }

    print(json.dumps(payload, indent=2))


def _encode_prompt(tokenizer: Any, prompt: str) -> Dict[str, torch.Tensor]:
    tokenized = tokenizer(prompt, return_tensors="pt", add_special_tokens=True)
    return {
        "input_ids": tokenized["input_ids"].to(dtype=torch.long, device="cpu"),
        "attention_mask": tokenized["attention_mask"].to(dtype=torch.long, device="cpu"),
    }


def _select_student_token(logits: torch.Tensor) -> int:
    return int(torch.argmax(logits).item())


def run_streamed_comparison(args: Any, prompts: List[str] | None = None) -> None:
    if int(getattr(args, "top_k", 0)) <= 0:
        raise ValueError("--top-k must be a positive integer.")
    if int(getattr(args, "limit_prompts", 0)) <= 0:
        raise ValueError("--limit-prompts must be a positive integer.")
    if int(getattr(args, "max_new_tokens", 0)) <= 0:
        raise ValueError("--max-new-tokens must be a positive integer.")
    if float(getattr(args, "memory_cap_gb", 0.0)) <= 0:
        raise ValueError("--memory-cap-gb must be a positive value.")

    top_k = int(args.top_k)
    max_new_tokens = int(args.max_new_tokens)

    selected_prompts = prompts if prompts is not None else _validate_prompt_count(
        DEFAULT_PROMPTS,
        int(args.limit_prompts),
    )
    selected_prompts = _validate_prompt_count(selected_prompts, int(args.limit_prompts))

    logging.info("Loading tokenizers.")
    (AutoTokenizer,) = _load_transformers()
    hf_cache_dir = _resolve_hf_cache_dir(getattr(args, "hf_cache_dir", None))
    local_files_only = bool(getattr(args, "offline", False))
    tokenizer_kwargs = {
        "cache_dir": hf_cache_dir,
        "local_files_only": local_files_only,
    }
    student_tokenizer = AutoTokenizer.from_pretrained(args.student_model, **tokenizer_kwargs)
    teacher_tokenizer = AutoTokenizer.from_pretrained(args.teacher_model, **tokenizer_kwargs)
    _validate_tokenizer_compatibility(teacher_tokenizer, student_tokenizer)
    shared_tokenizer = teacher_tokenizer

    logging.info("Building model specs from checkpoints.")
    teacher_spec = QwenStreamedModelSpec.from_model_id(
        args.teacher_model,
        cache_dir=hf_cache_dir,
        local_files_only=local_files_only,
    )
    student_spec = QwenStreamedModelSpec.from_model_id(
        args.student_model,
        cache_dir=hf_cache_dir,
        local_files_only=local_files_only,
    )
    teacher_spec.validate_required_tensors()
    student_spec.validate_required_tensors()
    prompt_lengths = _describe_prompt_lengths(shared_tokenizer, selected_prompts)

    if getattr(args, "dry_plan", False):
        _run_dry_plan(
            args=args,
            teacher_spec=teacher_spec,
            student_spec=student_spec,
            shared_tokenizer=shared_tokenizer,
            selected_prompts=selected_prompts,
        )
        return

    _validate_vocab_sizes(shared_tokenizer, teacher_spec, student_spec)
    _validate_context_lengths(
        prompt_lengths=prompt_lengths,
        max_new_tokens=max_new_tokens,
        teacher_spec=teacher_spec,
        student_spec=student_spec,
    )

    logging.info(
        "Loading safetensors weight readers and streamed model runners "
        "(using max memory cap %.3f GiB).",
        float(args.memory_cap_gb),
    )
    memory_guard = MemoryGuard(cap_gb=float(args.memory_cap_gb))
    student_loader = SafetensorWeightLoader(
        weight_map=student_spec.weight_map,
        dtype=torch.float32,
    )
    teacher_loader = SafetensorWeightLoader(
        weight_map=teacher_spec.weight_map,
        dtype=torch.float32,
    )

    student_streamer = QwenLayerStreamer(
        spec=student_spec,
        loader=student_loader,
        memory_guard=memory_guard,
        model_label="student",
    )
    teacher_streamer = QwenLayerStreamer(
        spec=teacher_spec,
        loader=teacher_loader,
        memory_guard=memory_guard,
        model_label="teacher",
    )

    prompt_outputs: List[Dict[str, Any]] = []
    global_aggregate: Dict[str, float] = {
        "kl_divergence": 0.0,
        "cosine_similarity": 0.0,
        "mean_absolute_logit_delta": 0.0,
        "max_absolute_logit_delta": 0.0,
        "overlap_count": 0.0,
    }
    generated_step_count = 0
    run_id = datetime.utcnow().strftime("run_%Y%m%dT%H%M%SZ")
    kv_root = Path(args.kv_cache_dir).resolve() / run_id

    for prompt_index, prompt in enumerate(selected_prompts):
        encoded = _encode_prompt(shared_tokenizer, prompt)
        input_ids = encoded["input_ids"]

        student_cache = KVSpillStore(
            root=kv_root,
            model_label="student",
            prompt_index=prompt_index,
        )
        teacher_cache = KVSpillStore(
            root=kv_root,
            model_label="teacher",
            prompt_index=prompt_index,
        )

        generated_token_ids: List[int] = []
        step_payloads: List[Dict[str, Any]] = []
        prompt_len = int(input_ids.shape[1])

        try:
            student_prompt_hidden = student_streamer.prefill(input_ids=input_ids, cache=student_cache)
            teacher_prompt_hidden = teacher_streamer.prefill(input_ids=input_ids, cache=teacher_cache)
            student_logits = student_streamer.logits_from_hidden(student_prompt_hidden)
            teacher_logits = teacher_streamer.logits_from_hidden(teacher_prompt_hidden)

            del student_prompt_hidden
            del teacher_prompt_hidden
            gc.collect()

            eos_token_id = shared_tokenizer.eos_token_id

            for step_index in range(max_new_tokens):
                step_metrics = _compute_metrics(
                    teacher_logits=teacher_logits,
                    student_logits=student_logits,
                    tokenizer=shared_tokenizer,
                    top_k=top_k,
                )
                generated_token_id = _select_student_token(student_logits)
                generated_token_ids.append(generated_token_id)
                step_payloads.append(
                    {
                        "step_index": step_index,
                        "generated_token_id": generated_token_id,
                        "generated_token": _token_label(shared_tokenizer, generated_token_id),
                        "top_teacher_tokens": step_metrics["top_teacher_tokens"],
                        "top_student_tokens": step_metrics["top_student_tokens"],
                        "overlapping_top_k_tokens": step_metrics["overlapping_top_k_tokens"],
                        "ranked_logit_deltas": step_metrics["ranked_logit_deltas"],
                        "kl_divergence": step_metrics["kl_divergence"],
                        "cosine_similarity": step_metrics["cosine_similarity"],
                        "mean_absolute_logit_delta": step_metrics["mean_absolute_logit_delta"],
                        "max_absolute_logit_delta": step_metrics["max_absolute_logit_delta"],
                    }
                )

                if eos_token_id is not None and generated_token_id == int(eos_token_id):
                    break

                if step_index < max_new_tokens - 1:
                    student_prompt_hidden = student_streamer.decode_one(
                        token_id=generated_token_id,
                        position=prompt_len + step_index,
                        cache=student_cache,
                    )
                    teacher_prompt_hidden = teacher_streamer.decode_one(
                        token_id=generated_token_id,
                        position=prompt_len + step_index,
                        cache=teacher_cache,
                    )
                    student_logits = student_streamer.logits_from_hidden(student_prompt_hidden)
                    teacher_logits = teacher_streamer.logits_from_hidden(teacher_prompt_hidden)

                    del student_prompt_hidden
                    del teacher_prompt_hidden
                    gc.collect()
        finally:
            student_cache.clear()
            teacher_cache.clear()

        prompt_len_f = float(len(step_payloads))
        if prompt_len_f > 0:
            overlap_count_sum = sum(
                int(step["overlapping_top_k_tokens"]["count"]) for step in step_payloads
            )
            mean_kl_divergence = sum(step["kl_divergence"] for step in step_payloads) / prompt_len_f
            mean_cosine_similarity = (
                sum(step["cosine_similarity"] for step in step_payloads) / prompt_len_f
            )
            mean_abs_delta = (
                sum(step["mean_absolute_logit_delta"] for step in step_payloads) / prompt_len_f
            )
            mean_max_abs_delta = (
                sum(step["max_absolute_logit_delta"] for step in step_payloads) / prompt_len_f
            )
            mean_overlap_count = overlap_count_sum / prompt_len_f
        else:
            mean_kl_divergence = 0.0
            mean_cosine_similarity = 0.0
            mean_abs_delta = 0.0
            mean_max_abs_delta = 0.0
            mean_overlap_count = 0.0

        prompt_outputs.append(
            {
                "prompt_index": prompt_index,
                "prompt": prompt,
                "generated_token_ids": generated_token_ids,
                "generated_text": _decode_token_ids(shared_tokenizer, generated_token_ids),
                "steps": step_payloads,
                "prompt_length": prompt_len,
                "steps_count": len(step_payloads),
                "kl_divergence": mean_kl_divergence,
                "cosine_similarity": mean_cosine_similarity,
                "mean_absolute_logit_delta": mean_abs_delta,
                "max_absolute_logit_delta": mean_max_abs_delta,
                "mean_overlapping_top_k_count": mean_overlap_count,
            }
        )

        if prompt_len_f > 0:
            global_aggregate["kl_divergence"] += mean_kl_divergence * prompt_len_f
            global_aggregate["cosine_similarity"] += mean_cosine_similarity * prompt_len_f
            global_aggregate["mean_absolute_logit_delta"] += mean_abs_delta * prompt_len_f
            global_aggregate["max_absolute_logit_delta"] += mean_max_abs_delta * prompt_len_f
            global_aggregate["overlap_count"] += mean_overlap_count * prompt_len_f
            generated_step_count += prompt_len_f
        gc.collect()

    generated_step_count_float = float(generated_step_count)
    aggregate = {
        "prompt_count": len(selected_prompts),
        "generated_step_count": int(generated_step_count),
        "mean_kl_divergence": global_aggregate["kl_divergence"] / generated_step_count_float
        if generated_step_count_float
        else 0.0,
        "mean_cosine_similarity": global_aggregate["cosine_similarity"] / generated_step_count_float
        if generated_step_count_float
        else 0.0,
        "mean_absolute_logit_delta": (
            global_aggregate["mean_absolute_logit_delta"] / generated_step_count_float
            if generated_step_count_float
            else 0.0
        ),
        "mean_max_absolute_logit_delta": (
            global_aggregate["max_absolute_logit_delta"] / generated_step_count_float
            if generated_step_count_float
            else 0.0
        ),
        "mean_overlapping_top_k_count": (
            global_aggregate["overlap_count"] / generated_step_count_float
            if generated_step_count_float
            else 0.0
        ),
    }

    payload = {
        "metadata": {
            "generated_at_utc": datetime.utcnow().isoformat() + "Z",
            "student_model": args.student_model,
            "teacher_model": args.teacher_model,
            "tokenizer_model": getattr(shared_tokenizer, "name_or_path", "unknown"),
            "device": "cpu",
            "dtype": "float32",
            "top_k": top_k,
            "prompt_count": len(selected_prompts),
            "mode": "streamed",
            "max_new_tokens": max_new_tokens,
            "memory_cap_gb": float(args.memory_cap_gb),
        },
        "prompts": prompt_outputs,
        "aggregate": aggregate,
    }

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    logging.info("Wrote streamed Qwen comparison data to %s", output)
