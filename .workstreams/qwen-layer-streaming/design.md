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

### `SafetensorWeightLoader`

Maps tensor names to shard files and loads only requested tensors with `safetensors.safe_open`. It should avoid materializing entire checkpoint shards as Python dictionaries. File handles and tensors are released after each stage.

### `QwenLayerStreamer`

Runs embeddings, transformer blocks, final norm, and LM head. The preferred implementation is to construct one Hugging Face Qwen3 block at a time from config, load just that block's state dict, run it, then delete it. If Hugging Face internals make that impractical under the memory cap, use a narrow manual Qwen3 implementation for RMSNorm, RoPE attention, and MLP.

### `KVSpillStore`

Stores per-layer attention key/value tensors on disk. The cache is required for token-by-token generation because the student and teacher both need to decode against prior context without keeping all layer caches in RAM.

### `MemoryGuard`

Samples process RSS at key boundaries and aborts if it exceeds `--memory-cap-gb`, default `6.0`. Logs should identify the model, prompt, stage, layer, and decode step.

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
