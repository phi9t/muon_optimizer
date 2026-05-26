# Qwen3 Layer-Streaming Logits And Generation Design

Date: 2026-05-26

## Goal

Add a constrained-memory Qwen3 runner for the existing logits explorer workflow. The runner should let `Qwen/Qwen3-0.6B` act as the student generator while `Qwen/Qwen3-1.7B` scores the same prefixes as the teacher. The priority is bounded peak memory on this 8 GiB MacBook Air. Runtime may be slow.

The default memory cap is 6 GiB RSS. The implementation may be Qwen3-family-specific and should initially target the smaller Qwen3 checkpoints used by this project.

## Architecture

The current full-model comparison path should remain available as the simple baseline. The new constrained path should execute each Qwen3 model one stage at a time:

1. Load tokenizer, config, checkpoint index, and metadata.
2. Resolve required tensors from Hugging Face safetensors shards.
3. Tokenize one prompt at a time.
4. Prefill student and teacher layer by layer, spilling each layer's KV cache to local disk.
5. Decode with the student as the token generator.
6. Decode the teacher on the same generated token positions.
7. Compare student and teacher logits for each generated step.
8. Emit explorer-compatible JSON with prompt-level and generation-step-level summaries.

Only one model stage's weights should be resident at a time. Embeddings, each transformer block, final norm, and LM head are loaded only when needed and dropped immediately after use.

## Components

### `QwenStreamedModelSpec`

Reads and validates model metadata: `config.json`, tokenizer, safetensors index, hidden size, layer count, vocab size, dtype, and expected Qwen3 tensor key names. It should fail before a long run if the checkpoint is not a supported Qwen3 causal LM shape.

Implementation file: `scripts/qwen_streaming/spec.py`.

Public surface:

```python
@dataclass(frozen=True)
class QwenStreamedModelSpec:
    model_id: str
    local_dir: Path
    config: dict[str, Any]
    weight_map: dict[str, Path]
    num_hidden_layers: int
    hidden_size: int
    vocab_size: int
    num_attention_heads: int
    num_key_value_heads: int
    head_dim: int
    intermediate_size: int
    rope_theta: float
    rms_norm_eps: float
    tie_word_embeddings: bool

    @classmethod
    def from_model_id(cls, model_id: str) -> "QwenStreamedModelSpec": ...

    def required_tensor_names(self) -> list[str]: ...
    def validate_required_tensors(self) -> None: ...
```

The spec should use `huggingface_hub.snapshot_download(..., allow_patterns=[...])` for metadata and safetensor shards. It should read `model.safetensors.index.json` when present. If a checkpoint has a single `model.safetensors`, synthesize a one-file weight map by inspecting the keys.

### `SafetensorWeightLoader`

Maps tensor names to shard files and loads only requested tensors with `safetensors.safe_open`. It should avoid materializing entire checkpoint shards as Python dictionaries. File handles and tensors are released after each stage.

Implementation file: `scripts/qwen_streaming/weights.py`.

Public surface:

```python
class SafetensorWeightLoader:
    def __init__(self, weight_map: Mapping[str, Path], dtype: torch.dtype) -> None: ...
    def load_tensor(self, name: str) -> torch.Tensor: ...
    def load_tensors(self, names: Iterable[str]) -> dict[str, torch.Tensor]: ...
```

All tensors should be loaded onto CPU. The default runtime dtype remains `torch.float32` because this machine is CPU-only and the existing qwen-logits dependency group pins CPU PyTorch.

### `QwenLayerStreamer`

Runs embeddings, transformer blocks, final norm, and LM head. The preferred implementation is to construct one Hugging Face Qwen3 block at a time from config, load just that block's state dict, run it, then delete it. If Hugging Face internals make that impractical under the memory cap, use a narrow manual Qwen3 implementation for RMSNorm, RoPE attention, and MLP.

Implementation files:

- `scripts/qwen_streaming/model_ops.py` for Qwen3 math helpers.
- `scripts/qwen_streaming/streamer.py` for prefill/decode orchestration.

The first implementation should use a manual block instead of instantiating the complete Hugging Face model. This avoids accidental allocation of all layers. The manual code only needs the Qwen3 architecture used by `Qwen/Qwen3-0.6B` and `Qwen/Qwen3-1.7B`:

- token embedding lookup,
- RMSNorm,
- grouped-query causal self-attention,
- rotary position embeddings,
- gated MLP with SiLU activation,
- final RMSNorm,
- untied LM head unless config says `tie_word_embeddings`.

The streamer should expose:

```python
@dataclass(frozen=True)
class StreamedLogits:
    logits: torch.Tensor
    token_id: int | None
    token_text: str | None

class QwenLayerStreamer:
    def prefill(self, input_ids: torch.Tensor, cache: "KVSpillStore") -> torch.Tensor: ...
    def decode_one(self, token_id: int, position: int, cache: "KVSpillStore") -> torch.Tensor: ...
    def logits_from_hidden(self, hidden: torch.Tensor) -> torch.Tensor: ...
```

### `KVSpillStore`

Stores per-layer attention key/value tensors on disk. The cache is required for token-by-token generation because the student and teacher both need to decode against prior context without keeping all layer caches in RAM.

Implementation file: `scripts/qwen_streaming/kv_cache.py`.

Public surface:

```python
@dataclass(frozen=True)
class LayerKV:
    key: torch.Tensor
    value: torch.Tensor

class KVSpillStore:
    def __init__(self, root: Path, model_label: str, prompt_index: int) -> None: ...
    def read(self, layer_index: int) -> LayerKV | None: ...
    def write(self, layer_index: int, kv: LayerKV) -> None: ...
    def clear(self) -> None: ...
```

KV files should live under a temp or user-provided cache root such as `.qwen_kv_cache/<run-id>/<model-label>/prompt-000/layer-000.pt`. Writes should be atomic: save to a temporary file in the same directory, then replace the target path.

### `MemoryGuard`

Samples process RSS at key boundaries and aborts if it exceeds `--memory-cap-gb`, default `6.0`. Logs should identify the model, prompt, stage, layer, and decode step.

Implementation file: `scripts/qwen_streaming/memory.py`.

Use `resource.getrusage` or `psutil` if available. Because `ru_maxrss` reports peak RSS rather than current RSS on macOS, prefer `psutil.Process().memory_info().rss` and add `psutil>=5.9` to the `qwen-logits` dependency group. The guard should expose a `check(label: str) -> int` method returning RSS bytes and raising `MemoryError` when over cap.

## File Structure

Create a small package under `scripts/qwen_streaming/` so the runner remains readable:

- `scripts/qwen_streaming/__init__.py`: package marker and public exports.
- `scripts/qwen_streaming/spec.py`: Hugging Face snapshot metadata, config parsing, weight-map validation.
- `scripts/qwen_streaming/weights.py`: safetensors tensor loading.
- `scripts/qwen_streaming/memory.py`: RSS guard.
- `scripts/qwen_streaming/kv_cache.py`: disk-backed KV cache.
- `scripts/qwen_streaming/model_ops.py`: Qwen3 tensor operations.
- `scripts/qwen_streaming/streamer.py`: layer-by-layer prefill/decode.
- `scripts/qwen_streaming/comparison.py`: generation loop and JSON assembly for generated steps.
- `scripts/run_qwen_logits_comparison.py`: keep the current full-model path and add a constrained mode flag, or delegate constrained mode to a new `scripts/run_qwen_streaming_comparison.py` CLI.

Add tests next to the current root-level tests:

- `qwen_streaming_spec_test.py`
- `qwen_streaming_kv_cache_test.py`
- `qwen_streaming_memory_test.py`
- `qwen_streaming_output_test.py`

These tests should use fake configs, fake tokenizers, and tiny generated safetensors. They must not require downloading Qwen weights.

## Execution Flow

For each prompt:

1. Tokenize the prompt.
2. Prefill the student layer by layer, writing student KV tensors to disk.
3. Prefill the teacher layer by layer, writing teacher KV tensors to disk.
4. For each generation step:
   1. Run a streamed student decode step using the student KV cache.
   2. Read student logits and choose the next token greedily by default.
   3. Append the chosen token to the shared sequence.
   4. Run a streamed teacher decode step for that same token using the teacher KV cache.
   5. Compare student and teacher logits at that generated position.
   6. Persist updated KV tensors for both models.
5. Stop at `--max-new-tokens`, EOS, or another deterministic stop condition.
6. Aggregate metrics across steps and prompts.

Defaults should be conservative: one prompt at a time, CPU `float32`, greedy decoding, `--max-new-tokens 8`, and a 6 GiB RSS cap.

## Output Shape

The output should remain suitable for the existing React explorer, while adding generation-step detail:

- `metadata`: student model, teacher model, device, dtype, memory cap, prompt count, max new tokens, timestamp.
- `prompts[]`: prompt text, generated text, prompt-level aggregate metrics, and `steps[]`.
- `steps[]`: step index, generated token id/text, teacher top tokens, student top tokens, overlapping top-k tokens, ranked logit deltas, KL divergence, cosine similarity, mean absolute logit delta, and max absolute logit delta.
- `aggregate`: averages across all generated steps and prompts.

The current fixed-prompt final-logit JSON can remain supported, but the constrained runner should produce step-aware data.

## Validation And Errors

The runner should fail explicitly when assumptions are not met:

- Teacher and student tokenizer ID mappings must match.
- Teacher and student model vocab sizes must match.
- Required Qwen3 tensor keys must exist before execution starts.
- Safetensors index and shard paths must be available through the Hugging Face cache or download path.
- RSS must stay under the configured cap.
- KV cache writes and reads must succeed.
- Prompt length must fit the configured context limit. Default behavior should fail rather than silently truncate.

## Testing

Tests that do not require full model downloads:

- safetensors index parsing with a tiny fake index,
- Qwen3 tensor key planning,
- tokenizer compatibility checks with fake tokenizers,
- memory guard cap behavior,
- KV spill store round trip with small tensors,
- JSON shape for prompt and generated-step summaries.

Runtime verification should be staged:

1. `--help`
2. dry-plan or metadata validation mode,
3. `--limit-prompts 1 --max-new-tokens 1`,
4. explorer build with `npm --prefix explorer run build`,
5. optional full constrained run.

The full constrained run may be slow and should not be required before every commit.

## Implementation Plan

The implementation should be done in small commits and should keep each commit runnable.

### Task 1: Package Skeleton And CLI Shape

Create `scripts/qwen_streaming/` and add an importable package. Add CLI flags without changing the existing default behavior:

- `--mode full|streamed`, default `full`;
- `--memory-cap-gb`, default `6.0`;
- `--max-new-tokens`, default `8`;
- `--kv-cache-dir`, default `.qwen_kv_cache`;
- `--dry-plan`, validates metadata and required tensor keys without running model math.

Verification:

```bash
python3 scripts/run_qwen_logits_comparison.py --help
python3 -m py_compile scripts/run_qwen_logits_comparison.py scripts/qwen_streaming/*.py
```

### Task 2: Metadata And Weight Planning

Implement `QwenStreamedModelSpec` and `SafetensorWeightLoader`. Add tests that build a tiny temporary safetensors checkpoint with Qwen3-shaped keys and verify:

- index parsing maps tensor names to shard paths;
- single-file safetensors checkpoints are supported;
- missing required keys raise a clear `ValueError`;
- all planned keys for embeddings, each layer, final norm, and LM head are present.

Verification:

```bash
uv run --group qwen-logits pytest qwen_streaming_spec_test.py -v
```

### Task 3: Memory Guard And KV Spill Store

Implement `MemoryGuard` and `KVSpillStore`. Tests should verify:

- a cap below current RSS raises `MemoryError`;
- a generous cap returns RSS bytes;
- KV writes and reads preserve tensor shape, dtype, and values;
- `clear()` removes prompt cache files.

Verification:

```bash
uv run --group qwen-logits pytest qwen_streaming_memory_test.py qwen_streaming_kv_cache_test.py -v
```

### Task 4: Qwen3 Manual Ops

Implement the Qwen3 math helpers in `model_ops.py`:

- `rms_norm(hidden, weight, eps)`;
- `apply_rope(q, k, positions, theta)`;
- `repeat_kv_for_gqa(k, v, num_attention_heads, num_key_value_heads)`;
- `causal_attention(q, k, v, attention_mask)`;
- `gated_mlp(hidden, gate_proj, up_proj, down_proj)`.

Tests should use tiny tensors with deterministic values. For `rms_norm` and `gated_mlp`, compare to direct PyTorch formulas. For attention, verify output shapes and that future positions are masked.

Verification:

```bash
uv run --group qwen-logits pytest qwen_streaming_model_ops_test.py -v
```

### Task 5: Layer Streamer

Implement `QwenLayerStreamer` using the spec, weight loader, memory guard, and KV store. Start with one prompt at a time and CPU `float32` only. The streamer should:

- load embeddings for prefill, then drop them;
- load one layer's weights, update hidden states, write that layer's KV, then drop weights;
- during decode, read that layer's existing KV, append new K/V, write it back, and return the new hidden state;
- load final norm and LM head only for logits.

Tests should use a tiny synthetic Qwen-like checkpoint with one or two layers. The goal is shape correctness and cache flow, not numerical parity with Qwen.

Verification:

```bash
uv run --group qwen-logits pytest qwen_streaming_streamer_test.py -v
```

### Task 6: Generation Comparison Output

Implement the streamed comparison loop in `comparison.py`. It should:

- prefill student and teacher on the prompt;
- generate `--max-new-tokens` greedily from student logits;
- decode teacher on the same generated token;
- reuse the existing metric code for top-k, overlap, KL, cosine, mean absolute delta, and max absolute delta;
- write `prompts[].steps[]` plus prompt and aggregate summaries.

Update the explorer types/UI only if needed to support step-aware output. Preserve the missing-data state.

Verification:

```bash
uv run --group qwen-logits pytest qwen_streaming_output_test.py -v
npm --prefix explorer run build
```

### Task 7: Dry Plan And Smoke Run

Implement `--dry-plan` so it downloads or resolves only metadata/index information, validates keys, prints model/layer/vocab/cache/memory-cap details, and exits before model math.

Then attempt:

```bash
uv run --group qwen-logits python scripts/run_qwen_logits_comparison.py \
  --mode streamed \
  --dry-plan \
  --student-model Qwen/Qwen3-0.6B \
  --teacher-model Qwen/Qwen3-1.7B
```

If dry-plan succeeds and resources allow, attempt one generated token:

```bash
uv run --group qwen-logits python scripts/run_qwen_logits_comparison.py \
  --mode streamed \
  --limit-prompts 1 \
  --max-new-tokens 1 \
  --memory-cap-gb 6
```

The full six-prompt, eight-token run is optional and may be left for a later manual run.
