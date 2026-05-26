#!/usr/bin/env python3
"""Compare next-token logits between Qwen3 student and teacher checkpoints."""

from __future__ import annotations

import argparse
import gc
import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Tuple, TYPE_CHECKING

if TYPE_CHECKING:
    import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
OUTPUT_DEFAULT = REPO_ROOT / "explorer" / "public" / "data" / "qwen_logits.json"

DEFAULT_STUDENT_MODEL = "Qwen/Qwen3-0.6B"
DEFAULT_TEACHER_MODEL = "Qwen/Qwen3-1.7B"
DEFAULT_PROMPTS: List[str] = [
    "The world breaks everyone, and afterward, many are strong at the broken places.",
    "What is 7 + 5?",
    "Write a tiny poem about a cat.",
    "Explain why gradients matter in optimization.",
    "Generate a quick planning checklist for a short trip.",
    "Name three colors and one common use for each.",
]
DEFAULT_TOP_K = 10


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compare next-token logits of Qwen3 teacher and student checkpoints."
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=OUTPUT_DEFAULT,
        help="Output JSON path (default: explorer/public/data/qwen_logits.json).",
    )
    parser.add_argument(
        "--mode",
        choices=("full", "streamed"),
        default="full",
        help="Comparison mode to run (default: full).",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=DEFAULT_TOP_K,
        help="Top-K token count for report sections and overlap (default: 10).",
    )
    parser.add_argument(
        "--limit-prompts",
        type=int,
        default=len(DEFAULT_PROMPTS),
        help="Limit number of curated prompts to evaluate (default: 6).",
    )
    parser.add_argument(
        "--memory-cap-gb",
        type=float,
        default=6.0,
        help="Streaming memory cap in GiB (default: 6.0).",
    )
    parser.add_argument(
        "--max-new-tokens",
        type=int,
        default=8,
        help="Maximum new tokens to generate in streamed mode (default: 8).",
    )
    parser.add_argument(
        "--kv-cache-dir",
        type=Path,
        default=Path(".qwen_kv_cache"),
        help="Directory for streamed KV cache files (default: .qwen_kv_cache).",
    )
    parser.add_argument(
        "--hf-cache-dir",
        type=Path,
        default=None,
        help="Hugging Face cache directory for streamed snapshots/tokenizers (default: HF default).",
    )
    parser.add_argument(
        "--offline",
        action="store_true",
        help="Streamed mode: load from local cache only and fail if files are missing.",
    )
    parser.add_argument(
        "--dry-plan",
        action="store_true",
        help="Show streaming plan and exit without running generation (streamed mode only).",
    )
    parser.add_argument(
        "--student-model",
        default=DEFAULT_STUDENT_MODEL,
        help=f"Student model ID (default: {DEFAULT_STUDENT_MODEL}).",
    )
    parser.add_argument(
        "--teacher-model",
        default=DEFAULT_TEACHER_MODEL,
        help=f"Teacher model ID (default: {DEFAULT_TEACHER_MODEL}).",
    )
    parser.add_argument(
        "--log-level",
        default="INFO",
        choices=("DEBUG", "INFO", "WARNING", "ERROR"),
        help="Log level (default: INFO).",
    )
    return parser.parse_args()


def _run_streamed(args: argparse.Namespace) -> None:
    import sys
    from pathlib import Path

    scripts_dir = Path(__file__).resolve().parent
    if str(scripts_dir) not in sys.path:
        sys.path.insert(0, str(scripts_dir))

    from qwen_streaming.comparison import run_streamed_comparison

    prompts = _validate_prompt_count(DEFAULT_PROMPTS, args.limit_prompts)
    run_streamed_comparison(args, prompts=prompts)


def _load_transformers() -> Tuple[Any, Any]:
    try:
        from transformers import AutoModelForCausalLM, AutoTokenizer
    except ModuleNotFoundError as exc:
        raise RuntimeError(
            "transformers is required to run this script. Use "
            "`uv run --group qwen-logits python scripts/run_qwen_logits_comparison.py` "
            "or install transformers manually."
        ) from exc
    return AutoModelForCausalLM, AutoTokenizer


def _validate_prompt_count(prompts: List[str], limit: int) -> List[str]:
    if limit <= 0:
        raise ValueError("--limit-prompts must be a positive integer.")
    selected = prompts[:limit]
    if not selected:
        raise ValueError("No prompts selected; check --limit-prompts.")
    return selected


def _model_vocab_size(model: Any) -> int:
    # Prefer output embedding size, fallback to config vocab_size.
    out_embed = model.get_output_embeddings()
    if out_embed is not None:
        return int(out_embed.weight.shape[0])
    return int(getattr(model.config, "vocab_size", -1))


def _validate_vocab_compatibility(
    tokenizer: Any,
    teacher_vocab_size: int,
    student_vocab_size: int,
) -> Tuple[int, int]:
    tokenizer_vocab = int(len(tokenizer))
    teacher_vocab = int(teacher_vocab_size)
    student_vocab = int(student_vocab_size)

    if tokenizer_vocab <= 0:
        raise ValueError("Tokenizer vocab size is not positive.")
    if teacher_vocab <= 0:
        raise ValueError("Teacher model vocab size is not positive.")
    if student_vocab <= 0:
        raise ValueError("Student model vocab size is not positive.")

    if tokenizer_vocab != teacher_vocab or tokenizer_vocab != student_vocab:
        raise ValueError(
            "Incompatible vocabularies detected. Expected tokenizer, teacher, and student "
            "vocab sizes to match before token-id comparisons. "
            f"Got tokenizer={tokenizer_vocab}, teacher={teacher_vocab}, student={student_vocab}."
        )

    return tokenizer_vocab, teacher_vocab


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
                f"First mismatch at id={token_id}: teacher={teacher_token!r}, "
                f"student={student_token!r}."
            )


def _load_torch() -> Any:
    try:
        import torch
    except ModuleNotFoundError as exc:
        raise RuntimeError(
            "torch is required to run this script. Use "
            "`uv run --group qwen-logits python scripts/run_qwen_logits_comparison.py` "
            "or install torch manually."
        ) from exc
    return torch


def _encode_prompt(tokenizer: Any, prompt: str, device: "torch.device") -> Dict[str, "torch.Tensor"]:
    tokenized = tokenizer(prompt, return_tensors="pt", add_special_tokens=True)
    return {
        "input_ids": tokenized["input_ids"].to(device),
        "attention_mask": tokenized["attention_mask"].to(device),
    }


def _collect_next_logits(
    model_id: str,
    prompts: List[str],
    tokenizer: Any,
    model_cls: Any,
    device: "torch.device",
    dtype: "torch.dtype",
) -> Tuple[List["torch.Tensor"], int]:
    logits_by_prompt: List["torch.Tensor"] = []

    logging.info("Loading model: %s", model_id)
    model = model_cls.from_pretrained(
        model_id,
        torch_dtype=dtype,
        low_cpu_mem_usage=True,
    )
    model.to(device=device, dtype=dtype)
    model.eval()
    vocab_size = _model_vocab_size(model)

    with torch.inference_mode():
        for prompt in prompts:
            encoded = _encode_prompt(tokenizer, prompt, device)
            outputs = model(**encoded)
            logits = outputs.logits[:, -1, :].to(dtype=dtype).squeeze(0).cpu().contiguous()
            logits_by_prompt.append(logits)
            del outputs
            del encoded

    # Free model and any residual buffers before loading the next model.
    del model
    gc.collect()

    return logits_by_prompt, vocab_size


def _topk_payload(logits: "torch.Tensor", tokenizer: Any, top_k: int) -> List[Dict[str, Any]]:
    k = min(int(top_k), int(logits.numel()))
    values, indices = torch.topk(logits, k=k, dim=0)
    probs = torch.softmax(logits, dim=0)

    out: List[Dict[str, Any]] = []
    for rank, (value, index) in enumerate(zip(values.tolist(), indices.tolist()), start=1):
        token = tokenizer.convert_ids_to_tokens(int(index))
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


def _safe_float(value: torch.Tensor) -> float:
    return float(value.detach().to(torch.float32).item())


def _rank_lookup(entries: List[Dict[str, Any]]) -> Dict[int, int]:
    return {int(entry["token_id"]): int(entry["rank"]) for entry in entries}


def _ranked_delta_payload(
    teacher_logits: "torch.Tensor",
    student_logits: "torch.Tensor",
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
                "token": tokenizer.convert_ids_to_tokens(token_id),
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

    # KL(P_teacher || Q_student)
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


def _build_output(
    prompts: List[str],
    teacher_logits_by_prompt: List[torch.Tensor],
    student_logits_by_prompt: List[torch.Tensor],
    tokenizer: Any,
    top_k: int,
    student_model: str,
    teacher_model: str,
) -> Dict[str, Any]:
    results: List[Dict[str, Any]] = []
    aggregate: Dict[str, float] = {
        "kl_divergence": 0.0,
        "cosine_similarity": 0.0,
        "mean_absolute_logit_delta": 0.0,
        "max_absolute_logit_delta": 0.0,
    }
    overlap_counts: List[int] = []

    for idx, prompt in enumerate(prompts):
        metrics = _compute_metrics(
            teacher_logits_by_prompt[idx],
            student_logits_by_prompt[idx],
            tokenizer,
            top_k,
        )

        overlap_counts.append(metrics["overlapping_top_k_tokens"]["count"])
        aggregate["kl_divergence"] += metrics["kl_divergence"]
        aggregate["cosine_similarity"] += metrics["cosine_similarity"]
        aggregate["mean_absolute_logit_delta"] += metrics["mean_absolute_logit_delta"]
        aggregate["max_absolute_logit_delta"] += metrics["max_absolute_logit_delta"]

        results.append(
            {
                "prompt_index": idx,
                "prompt": prompt,
                "top_teacher_tokens": metrics["top_teacher_tokens"],
                "top_student_tokens": metrics["top_student_tokens"],
                "overlapping_top_k_tokens": metrics["overlapping_top_k_tokens"],
                "ranked_logit_deltas": metrics["ranked_logit_deltas"],
                "kl_divergence": metrics["kl_divergence"],
                "cosine_similarity": metrics["cosine_similarity"],
                "mean_absolute_logit_delta": metrics["mean_absolute_logit_delta"],
                "max_absolute_logit_delta": metrics["max_absolute_logit_delta"],
            }
        )

    num_prompts = float(len(prompts))
    aggregate = {
        "prompt_count": len(prompts),
        "mean_kl_divergence": aggregate["kl_divergence"] / num_prompts,
        "mean_cosine_similarity": aggregate["cosine_similarity"] / num_prompts,
        "mean_absolute_logit_delta": aggregate["mean_absolute_logit_delta"] / num_prompts,
        "mean_max_absolute_logit_delta": aggregate["max_absolute_logit_delta"] / num_prompts,
        "mean_overlapping_top_k_count": sum(overlap_counts) / num_prompts,
    }

    return {
        "metadata": {
            "generated_at_utc": datetime.utcnow().isoformat() + "Z",
            "student_model": student_model,
            "teacher_model": teacher_model,
            "tokenizer_model": getattr(tokenizer, "name_or_path", "unknown"),
            "device": "cpu",
            "dtype": "float32",
            "top_k": top_k,
            "prompt_count": len(prompts),
        },
        "prompts": results,
        "aggregate": aggregate,
    }


def _load_tokenizer(tokenizer_model: str) -> Any:
    _, AutoTokenizer = _load_transformers()
    return AutoTokenizer.from_pretrained(tokenizer_model)


def _run(args: argparse.Namespace) -> None:
    if args.mode == "streamed":
        _run_streamed(args)
        return

    global torch

    logging.basicConfig(level=getattr(logging, args.log_level))

    if args.top_k <= 0:
        raise ValueError("--top-k must be a positive integer.")
    if args.limit_prompts <= 0:
        raise ValueError("--limit-prompts must be a positive integer.")

    prompts = _validate_prompt_count(DEFAULT_PROMPTS, args.limit_prompts)

    tokenizer = _load_tokenizer(args.teacher_model)
    student_tokenizer = _load_tokenizer(args.student_model)
    _validate_tokenizer_compatibility(tokenizer, student_tokenizer)
    AutoModelForCausalLM, _ = _load_transformers()
    torch = _load_torch()

    teacher_logits, teacher_vocab_size = _collect_next_logits(
        model_id=args.teacher_model,
        prompts=prompts,
        tokenizer=tokenizer,
        model_cls=AutoModelForCausalLM,
        device=torch.device("cpu"),
        dtype=torch.float32,
    )

    student_logits, student_vocab_size = _collect_next_logits(
        model_id=args.student_model,
        prompts=prompts,
        tokenizer=tokenizer,
        model_cls=AutoModelForCausalLM,
        device=torch.device("cpu"),
        dtype=torch.float32,
    )

    # Validate tokenizer assumptions before token-id comparisons.
    _validate_vocab_compatibility(tokenizer, teacher_vocab_size, student_vocab_size)

    payload = _build_output(
        prompts=prompts,
        teacher_logits_by_prompt=teacher_logits,
        student_logits_by_prompt=student_logits,
        tokenizer=tokenizer,
        top_k=args.top_k,
        student_model=args.student_model,
        teacher_model=args.teacher_model,
    )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    logging.info("Wrote logits comparison data to %s", args.output)


def main() -> None:
    args = parse_args()
    _run(args)


if __name__ == "__main__":
    main()
