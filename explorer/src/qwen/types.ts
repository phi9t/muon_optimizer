export interface QwenTopToken {
  rank: number
  token_id: number
  token: string
  logit: number
  probability: number
}

export interface QwenOverlappingToken {
  token_id: number
  token: string
  teacher_rank: number
  student_rank: number
}

export interface QwenStreamStep {
  step_index: number
  generated_token_id: number
  generated_token: string
  top_teacher_tokens: QwenTopToken[]
  top_student_tokens: QwenTopToken[]
  overlapping_top_k_tokens: {
    count: number
    tokens: QwenOverlappingToken[]
  }
  ranked_logit_deltas: QwenRankedLogitDelta[]
  kl_divergence: number
  cosine_similarity: number
  mean_absolute_logit_delta: number
  max_absolute_logit_delta: number
}

export interface QwenPromptResult {
  prompt_index: number
  prompt: string
  generated_token_ids?: number[]
  generated_text?: string
  steps?: QwenStreamStep[]
  prompt_length?: number
  steps_count?: number
  mean_overlapping_top_k_count?: number
  top_teacher_tokens?: QwenTopToken[]
  top_student_tokens?: QwenTopToken[]
  overlapping_top_k_tokens?: {
    count: number
    tokens: QwenOverlappingToken[]
  }
  ranked_logit_deltas?: QwenRankedLogitDelta[]
  kl_divergence: number
  cosine_similarity: number
  mean_absolute_logit_delta: number
  max_absolute_logit_delta: number
}

export interface QwenRankedLogitDelta {
  token_id: number
  token: string
  teacher_rank: number | null
  student_rank: number | null
  teacher_logit: number
  student_logit: number
  teacher_probability: number
  student_probability: number
  logit_delta: number
  absolute_logit_delta: number
}

export interface QwenAggregate {
  prompt_count: number
  mean_kl_divergence: number
  mean_cosine_similarity: number
  mean_absolute_logit_delta: number
  mean_max_absolute_logit_delta: number
  mean_overlapping_top_k_count: number
}

export interface QwenMetadata {
  generated_at_utc: string
  student_model: string
  teacher_model: string
  tokenizer_model: string
  device: string
  dtype: string
  top_k: number
  prompt_count: number
  mode?: 'full' | 'streamed'
  max_new_tokens?: number
  memory_cap_gb?: number
}

export interface QwenLogitsPayload {
  metadata: QwenMetadata
  prompts: QwenPromptResult[]
  aggregate: QwenAggregate
}
