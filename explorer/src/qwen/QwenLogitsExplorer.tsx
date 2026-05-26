import { useEffect, useMemo, useState } from 'react'
import {
  Bar,
  BarChart,
  CartesianGrid,
  Legend,
  ResponsiveContainer,
  Tooltip,
  XAxis,
  YAxis,
} from 'recharts'
import { errorMessage, fetchExplorerJson, formatMetric } from '../lib/fetch'
import type { QwenLogitsPayload, QwenPromptResult, QwenRankedLogitDelta, QwenTopToken } from './types'

interface TokenRankedRow {
  token_id: number
  token: string
  teacher_rank: number | null
  student_rank: number | null
  teacher_probability: number | null
  student_probability: number | null
  token_label: string
  delta_logit: number
  abs_delta_logit: number
}

interface RankedBarData extends TokenRankedRow {
  label_rank: number
}

function formatPercent(value: number | null): string {
  if (value == null || !Number.isFinite(value)) return '--'
  return `${(value * 100).toFixed(2)}%`
}

function normalizeToken(token: string): string {
  const safe = token?.replace(/\r?\n/g, '\\n').trim() || '[empty token]'
  return safe.length > 22 ? `${safe.slice(0, 19)}...` : safe
}

function buildProbabilityMap(tokens: QwenTopToken[] | undefined) {
  if (!tokens || tokens.length === 0) return new Map<number, { token: string; rank: number; prob: number; logit: number }>()

  const out = new Map<number, { token: string; rank: number; prob: number; logit: number }>()
  tokens.forEach((token) => {
    out.set(token.token_id, {
      token: token.token,
      rank: token.rank,
      logit: token.logit,
      prob: token.probability,
    })
  })

  return out
}

function deltaToRow(delta: QwenRankedLogitDelta, labelRank: number): RankedBarData {
  return {
    token_id: delta.token_id,
    token: delta.token,
    teacher_rank: delta.teacher_rank,
    student_rank: delta.student_rank,
    teacher_probability: delta.teacher_probability,
    student_probability: delta.student_probability,
    token_label: normalizeToken(delta.token),
    delta_logit: delta.logit_delta,
    abs_delta_logit: delta.absolute_logit_delta,
    label_rank: labelRank,
  }
}

function formatCell(value: number | null): string {
  if (value == null || !Number.isFinite(value)) return '--'
  return formatMetric(value)
}

function isMissingDataError(message: string): boolean {
  return message.includes('404') || message.toLowerCase().includes('not found')
}

export default function QwenLogitsExplorer() {
  const [payload, setPayload] = useState<QwenLogitsPayload | null>(null)
  const [selectedPromptIndex, setSelectedPromptIndex] = useState(0)
  const [loading, setLoading] = useState(true)
  const [error, setError] = useState<string | null>(null)
  const [missingData, setMissingData] = useState(false)

  useEffect(() => {
    async function load() {
      try {
        setLoading(true)
        setError(null)
        setMissingData(false)
        const data = await fetchExplorerJson<QwenLogitsPayload>('qwen_logits.json')
        setPayload(data)
      } catch (err: unknown) {
        const message = errorMessage(err)
        if (isMissingDataError(message)) {
          setMissingData(true)
          setPayload(null)
        } else {
          setError(message)
          setPayload(null)
        }
      } finally {
        setLoading(false)
      }
    }

    void load()
  }, [])

  useEffect(() => {
    if (!payload) return
    if (selectedPromptIndex >= payload.prompts.length) {
      setSelectedPromptIndex(0)
    }
  }, [payload, selectedPromptIndex])

  const selectedPrompt: QwenPromptResult | null = useMemo(() => {
    if (!payload) return null
    return payload.prompts[selectedPromptIndex] ?? payload.prompts[0] ?? null
  }, [payload, selectedPromptIndex])

  const promptRows = useMemo(() => {
    if (!payload?.prompts.length) return []
    return payload.prompts.map((prompt, idx) => ({
      promptIndex: prompt.prompt_index,
      index: idx,
      prompt: prompt.prompt,
    }))
  }, [payload?.prompts])

  const tokenRows = useMemo<RankedBarData[]>(() => {
    if (!selectedPrompt) return []

    if (selectedPrompt.ranked_logit_deltas?.length) {
      return selectedPrompt.ranked_logit_deltas
        .map((delta) =>
          deltaToRow(
            delta,
            Math.min(
              delta.teacher_rank ?? Number.POSITIVE_INFINITY,
              delta.student_rank ?? Number.POSITIVE_INFINITY,
            ),
          ),
        )
        .sort((a, b) => a.label_rank - b.label_rank)
    }

    const topK = payload?.metadata.top_k ?? 10
    const teacher = buildProbabilityMap(selectedPrompt.top_teacher_tokens)
    const student = buildProbabilityMap(selectedPrompt.top_student_tokens)

    const tokenIds = new Set([...teacher.keys(), ...student.keys()])
    const rows = [...tokenIds].map((tokenId) => {
      const teacherEntry = teacher.get(tokenId) ?? null
      const studentEntry = student.get(tokenId) ?? null
      const teacherLogit = teacherEntry?.logit ?? null
      const studentLogit = studentEntry?.logit ?? null

      const teacherProbability = teacherEntry?.prob ?? null
      const studentProbability = studentEntry?.prob ?? null

      const teacherToken = teacherEntry?.token
      const studentToken = studentEntry?.token
      const token = teacherToken ?? studentToken ?? `[token_${tokenId}]`

      const delta = (studentLogit ?? 0) - (teacherLogit ?? 0)

      return {
        token_id: tokenId,
        token,
        teacher_rank: teacherEntry?.rank ?? null,
        student_rank: studentEntry?.rank ?? null,
        teacher_probability: teacherProbability,
        student_probability: studentProbability,
        token_label: normalizeToken(token),
        delta_logit: delta,
        abs_delta_logit: Number.isFinite(delta) ? Math.abs(delta) : 0,
        label_rank: Math.min(
          teacherEntry?.rank ?? Number.POSITIVE_INFINITY,
          studentEntry?.rank ?? Number.POSITIVE_INFINITY,
          topK + 1,
        ),
      }
    })

    return rows.sort((a, b) => a.label_rank - b.label_rank)
  }, [selectedPrompt, payload?.metadata.top_k])

  const deltaRows = useMemo(
    () => {
      if (selectedPrompt?.ranked_logit_deltas?.length) {
        return selectedPrompt.ranked_logit_deltas.map((delta, index) => deltaToRow(delta, index + 1))
      }
      return tokenRows.slice().sort((a, b) => b.abs_delta_logit - a.abs_delta_logit)
    },
    [selectedPrompt?.ranked_logit_deltas, tokenRows],
  )

  const topK = payload?.metadata.top_k ?? 0
  const aggregate = payload?.aggregate

  if (loading) {
    return (
        <div className="loading-container" role="status">
          <div className="spinner" aria-hidden="true" />
          <p>Loading Qwen logits comparison data...</p>
      </div>
    )
  }

  if (missingData) {
    return (
      <div className="main-view-panel">
        <section className="card-view qwen-empty-state">
          <h2>Qwen Logits Comparison Data Missing</h2>
          <p className="text-muted">
            The file <code>explorer/public/data/qwen_logits.json</code> has not been generated yet.
          </p>
          <p className="text-muted">
            Run <code>uv run --group qwen-logits python scripts/run_qwen_logits_comparison.py</code>{' '}
            from the repository root to generate logits and enable this panel.
          </p>
        </section>
      </div>
    )
  }

  if (error || !payload || !selectedPrompt) {
    return (
      <div className="main-view-panel">
        <section className="card-view text-center">
          <p className="text-danger">Failed to load Qwen logits data: {error ?? 'unknown'}</p>
          <button
            type="button"
            className="pg-btn retry-btn"
            onClick={() => window.location.reload()}
          >
            Retry
          </button>
        </section>
      </div>
    )
  }

  return (
    <div className="main-view-panel">
      <section className="card-view qwen-section">
        <div className="qwen-topline">
          <div className="qwen-title-block">
            <h2>Qwen Logits Comparison</h2>
            <p className="text-muted qwen-subtitle">
              Teacher: {payload.metadata.teacher_model}
              {' | '}
              Student: {payload.metadata.student_model}
              {' | '}
              Top-k: {topK}
            </p>
          </div>
          <div className="qwen-prompt-control">
            <label className="qwen-prompt-label" htmlFor="qwen-prompt-select">
              Prompt
            </label>
            <select
              id="qwen-prompt-select"
              className="qwen-prompt-select"
              value={selectedPromptIndex}
              onChange={(event) => setSelectedPromptIndex(Number(event.target.value))}
            >
              {promptRows.map((entry) => (
                <option key={entry.index} value={entry.index}>
                  {`${entry.promptIndex + 1}: ${entry.prompt.slice(0, 80)}`}
                  {entry.prompt.length > 80 ? '...' : ''}
                </option>
              ))}
            </select>
          </div>
        </div>

        <p className="text-secondary">{selectedPrompt.prompt}</p>

        <div className="qwen-metric-grid">
          <article className="qwen-metric-card">
            <h3>Average KL</h3>
            <p className="qwen-metric-value">
              {aggregate ? formatCell(aggregate.mean_kl_divergence) : '--'}
            </p>
          </article>
          <article className="qwen-metric-card">
            <h3>Average cosine similarity</h3>
            <p className="qwen-metric-value">
              {aggregate ? formatCell(aggregate.mean_cosine_similarity) : '--'}
            </p>
          </article>
          <article className="qwen-metric-card">
            <h3>Mean absolute delta</h3>
            <p className="qwen-metric-value">
              {aggregate ? formatCell(aggregate.mean_absolute_logit_delta) : '--'}
            </p>
          </article>
          <article className="qwen-metric-card">
            <h3>Average top-k overlap</h3>
            <p className="qwen-metric-value">
              {aggregate && topK > 0
                ? `${formatMetric(aggregate.mean_overlapping_top_k_count)} / ${topK} (${formatMetric(
                    (aggregate.mean_overlapping_top_k_count / topK) * 100,
                  )}%)`
                : '--'}
            </p>
          </article>
        </div>
      </section>

      <div className="qwen-inspector-grid">
        <section className="card-view">
          <h3>Top-token probabilities</h3>
          <p className="text-muted">
            Teacher and student probabilities for the selected prompt top tokens.
          </p>
          <div className="qwen-chart-wrap">
            <ResponsiveContainer width="100%" height={Math.max(260, tokenRows.length * 24)}>
              <BarChart
                data={tokenRows}
                layout="vertical"
                margin={{ top: 8, right: 20, left: 36, bottom: 12 }}
              >
                <CartesianGrid stroke="rgba(255,255,255,0.06)" horizontal={false} />
                <XAxis
                  type="number"
                  domain={[0, 'dataMax']}
                  tickFormatter={(value) =>
                    typeof value === 'number' ? `${(value * 100).toFixed(2)}%` : `${value}`
                  }
                  stroke="#9ca3af"
                  fontSize={12}
                />
                <YAxis
                  type="category"
                  dataKey="token_label"
                  width={160}
                  tick={{ fill: '#9ca3af', fontSize: 11 }}
                  interval={0}
                  stroke="#9ca3af"
                />
                <Tooltip
                  contentStyle={{
                    background: '#111827',
                    border: '1px solid rgba(255,255,255,0.1)',
                    borderRadius: 8,
                  }}
                  formatter={(value) => [
                    typeof value === 'number' ? `${(value * 100).toFixed(2)}%` : `${value}`,
                    '',
                  ]}
                />
                <Legend />
                <Bar
                  dataKey="teacher_probability"
                  name="Teacher"
                  fill="#6366f1"
                  radius={[0, 6, 6, 0]}
                />
                <Bar
                  dataKey="student_probability"
                  name="Student"
                  fill="#10b981"
                  radius={[0, 6, 6, 0]}
                />
              </BarChart>
            </ResponsiveContainer>
          </div>
        </section>

        <section className="card-view">
          <h3>Ranked logit delta table</h3>
          <div className="table-wrapper">
            <table className="data-table">
              <thead>
                <tr>
                  <th>Ranked token</th>
                  <th>Teacher rank</th>
                  <th>Student rank</th>
                  <th>Teacher P</th>
                  <th>Student P</th>
                  <th>Logit delta</th>
                  <th>Abs delta</th>
                </tr>
              </thead>
              <tbody>
                {deltaRows.map((row, rowIndex) => (
                  <tr key={`${row.token_id}-${rowIndex}`}>
                    <td className="qwen-token">{row.token}</td>
                    <td>{row.teacher_rank ?? '--'}</td>
                    <td>{row.student_rank ?? '--'}</td>
                    <td>{row.teacher_probability == null ? '--' : formatPercent(row.teacher_probability)}</td>
                    <td>{row.student_probability == null ? '--' : formatPercent(row.student_probability)}</td>
                    <td>{formatCell(row.delta_logit)}</td>
                    <td>{formatMetric(row.abs_delta_logit)}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </section>
      </div>
    </div>
  )
}
