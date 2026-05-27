import { useEffect, useState, useMemo } from 'react'
import {
  LineChart,
  Line,
  BarChart,
  Bar,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  Legend,
  ResponsiveContainer,
} from 'recharts'
import { Activity, Layers, BarChart2, TrendingUp, Info } from 'lucide-react'
import { errorMessage, fetchExplorerJson } from '../lib/fetch'

interface MatrixFactorizationData {
  losses: number[]
  cond_numbers: number[]
  singular_values: {
    [step: string]: number[]
  }
}

type MatrixFactorizationPayload = Record<string, MatrixFactorizationData>

interface TransformerSpectralData {
  losses: number[]
  entropies: number[]
  singular_values: {
    [step: string]: number[]
  }
}

type TransformerSpectralPayload = Record<string, TransformerSpectralData>

const OPT_COLORS: Record<string, string> = {
  SGD: '#ef4444',
  Adam: '#3b82f6',
  AdamW: '#3b82f6',
  Muon: '#10b981',
  'Muon+Aux': '#a855f7',
}

function buildTrajectoryChartData<T extends Record<string, any>>(
  payload: T,
  field: string,
): Record<string, any>[] {
  const optimizers = Object.keys(payload)
  if (optimizers.length === 0) return []
  const maxLen = Math.max(...optimizers.map((opt) => payload[opt][field]?.length ?? 0), 0)
  const rows = []
  for (let i = 0; i < maxLen; i++) {
    const row: Record<string, any> = { step: i }
    for (const opt of optimizers) {
      const arr = payload[opt][field]
      if (arr && i < arr.length) {
        row[opt] = arr[i]
      }
    }
    rows.push(row)
  }
  return rows
}

function buildSpectrumChartData<T extends Record<string, any>>(
  payload: T,
  step: string,
): Record<string, any>[] {
  const optimizers = Object.keys(payload)
  if (optimizers.length === 0) return []
  const lengths = optimizers.map((opt) => payload[opt].singular_values?.[step]?.length ?? 0)
  const maxLen = Math.max(...lengths, 0)
  const rows = []
  for (let i = 0; i < maxLen; i++) {
    const row: Record<string, any> = { index: i + 1 }
    for (const opt of optimizers) {
      const vals = payload[opt].singular_values?.[step]
      if (vals && i < vals.length) {
        row[opt] = vals[i]
      }
    }
    rows.push(row)
  }
  return rows
}

const CustomTooltip = ({ active, payload, label, suffix = '', labelPrefix = 'Step ' }: any) => {
  if (active && payload && payload.length) {
    return (
      <div
        className="custom-tooltip"
        style={{
          background: 'rgba(17, 24, 39, 0.95)',
          border: '1px solid rgba(255, 255, 255, 0.1)',
          borderRadius: '12px',
          padding: '12px 16px',
          boxShadow: '0 8px 30px rgba(0, 0, 0, 0.5)',
          backdropFilter: 'blur(8px)',
        }}
      >
        <p style={{ margin: '0 0 8px', fontSize: '12px', color: '#9ca3af', fontWeight: 600 }}>
          {labelPrefix}
          {label}
        </p>
        <ul style={{ listStyle: 'none', padding: 0, margin: 0 }}>
          {payload.map((entry: any, index: number) => (
            <li
              key={index}
              style={{
                fontSize: '13px',
                color: '#f3f4f6',
                margin: '4px 0',
                display: 'flex',
                alignItems: 'center',
                gap: '8px',
              }}
            >
              <span
                style={{
                  display: 'inline-block',
                  width: '8px',
                  height: '8px',
                  borderRadius: '50%',
                  background: entry.color,
                }}
              />
              <span style={{ fontWeight: 500 }}>{entry.name}:</span>
              <span style={{ color: '#10b981', fontFamily: 'monospace', marginLeft: 'auto' }}>
                {entry.value?.toLocaleString(undefined, {
                  minimumFractionDigits: 4,
                  maximumFractionDigits: 4,
                })}
                {suffix}
              </span>
            </li>
          ))}
        </ul>
      </div>
    )
  }
  return null
}

export default function MatrixDynamicsExplorer() {
  const [mfData, setMfData] = useState<MatrixFactorizationPayload | null>(null)
  const [tsData, setTsData] = useState<TransformerSpectralPayload | null>(null)
  const [activeTab, setActiveTab] = useState<'factorization' | 'transformer'>('factorization')
  const [loading, setLoading] = useState(true)
  const [error, setError] = useState<string | null>(null)

  // Sub-controls
  const [mfStep, setMfStep] = useState<string>('100')
  const [mfSpectrumType, setMfSpectrumType] = useState<'line' | 'bar'>('line')
  const [mfCondScale, setMfCondScale] = useState<'linear' | 'log'>('log')

  const [tsStep, setTsStep] = useState<string>('30')
  const [tsSpectrumType, setTsSpectrumType] = useState<'line' | 'bar'>('line')

  useEffect(() => {
    async function load() {
      try {
        setLoading(true)
        setError(null)
        const [mf, ts] = await Promise.all([
          fetchExplorerJson<MatrixFactorizationPayload>('matrix_factorization.json'),
          fetchExplorerJson<TransformerSpectralPayload>('transformer_spectral.json'),
        ])
        setMfData(mf)
        setTsData(ts)
      } catch (err: unknown) {
        setError(errorMessage(err))
      } finally {
        setLoading(false)
      }
    }
    void load()
  }, [])

  // Deep Matrix Factorization Data processing
  const mfLossData = useMemo(() => {
    if (!mfData) return []
    return buildTrajectoryChartData(mfData, 'losses')
  }, [mfData])

  const mfCondData = useMemo(() => {
    if (!mfData) return []
    return buildTrajectoryChartData(mfData, 'cond_numbers')
  }, [mfData])

  const mfSpectrumData = useMemo(() => {
    if (!mfData) return []
    return buildSpectrumChartData(mfData, mfStep)
  }, [mfData, mfStep])

  const mfMetrics = useMemo(() => {
    if (!mfData) return null
    const metrics: Record<string, { finalLoss: number; finalCond: number; maxCond: number }> = {}
    for (const opt of Object.keys(mfData)) {
      const losses = mfData[opt].losses ?? []
      const conds = mfData[opt].cond_numbers ?? []
      metrics[opt] = {
        finalLoss: losses[losses.length - 1] ?? 0,
        finalCond: conds[conds.length - 1] ?? 0,
        maxCond: Math.max(...conds, 1),
      }
    }
    return metrics
  }, [mfData])

  // MNIST Vision Transformer Data processing
  const tsLossData = useMemo(() => {
    if (!tsData) return []
    return buildTrajectoryChartData(tsData, 'losses')
  }, [tsData])

  const tsEntropyData = useMemo(() => {
    if (!tsData) return []
    return buildTrajectoryChartData(tsData, 'entropies')
  }, [tsData])

  const tsSpectrumData = useMemo(() => {
    if (!tsData) return []
    return buildSpectrumChartData(tsData, tsStep)
  }, [tsData, tsStep])

  const tsMetrics = useMemo(() => {
    if (!tsData) return null
    const metrics: Record<string, { finalLoss: number; finalEntropy: number; finalCond: number }> = {}
    for (const opt of Object.keys(tsData)) {
      const losses = tsData[opt].losses ?? []
      const entropies = tsData[opt].entropies ?? []
      const sigmas = tsData[opt].singular_values?.['50'] ?? []
      const finalCond = sigmas.length > 0 ? sigmas[0] / sigmas[sigmas.length - 1] : 1
      metrics[opt] = {
        finalLoss: losses[losses.length - 1] ?? 0,
        finalEntropy: entropies[entropies.length - 1] ?? 0,
        finalCond,
      }
    }
    return metrics
  }, [tsData])

  if (loading) {
    return (
      <div className="loading-container" role="status">
        <div className="spinner" aria-hidden="true" />
        <p>Loading matrix dynamics and spectral data…</p>
      </div>
    )
  }

  if (error) {
    return (
      <div className="card-view text-center">
        <p className="text-danger">Failed to load matrix data: {error}</p>
        <button type="button" className="pg-btn retry-btn" onClick={() => window.location.reload()}>
          Retry
        </button>
      </div>
    )
  }

  return (
    <div className="mo-shell pb-shell">
      <style>{`
        .matrix-explorer {
          display: flex;
          flex-direction: column;
          gap: 24px;
        }
        .matrix-header {
          margin-bottom: 8px;
        }
        .matrix-grid {
          display: grid;
          grid-template-columns: 1fr;
          gap: 20px;
        }
        @media (min-width: 1024px) {
          .matrix-grid {
            grid-template-columns: 1fr 1fr;
          }
          .matrix-grid-full {
            grid-template-columns: 1fr;
          }
        }
        .matrix-card {
          background: var(--bg-card);
          border: 1px solid var(--border-color);
          border-radius: 16px;
          padding: 24px;
          backdrop-filter: blur(16px);
          box-shadow: 0 4px 30px rgba(0, 0, 0, 0.3);
          display: flex;
          flex-direction: column;
          gap: 16px;
        }
        .matrix-card-header {
          display: flex;
          justify-content: space-between;
          align-items: center;
          flex-wrap: wrap;
          gap: 12px;
          border-bottom: 1px solid rgba(255, 255, 255, 0.05);
          padding-bottom: 12px;
          margin-bottom: 4px;
        }
        .matrix-card-title {
          margin: 0;
          font-size: 15px;
          font-weight: 700;
          display: flex;
          align-items: center;
          gap: 8px;
          color: var(--text-primary);
        }
        .matrix-controls-row {
          display: flex;
          gap: 12px;
          align-items: center;
          flex-wrap: wrap;
        }
        .toggle-group {
          display: flex;
          background: rgba(255, 255, 255, 0.03);
          border: 1px solid var(--border-color);
          border-radius: 8px;
          padding: 2px;
        }
        .toggle-btn {
          background: transparent;
          border: none;
          color: var(--text-secondary);
          font-size: 11px;
          font-weight: 600;
          padding: 4px 10px;
          cursor: pointer;
          border-radius: 6px;
          transition: all 0.2s ease;
        }
        .toggle-btn.active {
          background: rgba(255, 255, 255, 0.08);
          color: var(--text-primary);
          box-shadow: 0 2px 4px rgba(0, 0, 0, 0.2);
        }
        .step-selector {
          display: flex;
          gap: 6px;
          flex-wrap: wrap;
        }
        .step-btn {
          background: rgba(255, 255, 255, 0.03);
          border: 1px solid var(--border-color);
          color: var(--text-secondary);
          font-size: 11px;
          font-weight: 600;
          padding: 5px 10px;
          cursor: pointer;
          border-radius: 6px;
          transition: all 0.2s ease;
        }
        .step-btn:hover {
          background: rgba(255, 255, 255, 0.05);
          color: var(--text-primary);
        }
        .step-btn.active {
          background: var(--color-primary);
          border-color: var(--color-primary);
          color: #ffffff;
        }
        .metrics-summary-grid {
          display: grid;
          grid-template-columns: repeat(auto-fit, minmax(220px, 1fr));
          gap: 12px;
          margin-bottom: 8px;
        }
        .metric-pill {
          background: rgba(255, 255, 255, 0.02);
          border: 1px solid var(--border-color);
          border-radius: 12px;
          padding: 12px 16px;
          display: flex;
          flex-direction: column;
          gap: 8px;
        }
        .metric-pill-label {
          font-size: 11px;
          color: var(--text-muted);
          text-transform: uppercase;
          letter-spacing: 0.05em;
          font-weight: 600;
        }
        .metric-pill-val-row {
          display: flex;
          justify-content: space-between;
          align-items: center;
          font-size: 13px;
        }
        .metric-pill-opt {
          font-weight: 500;
          color: var(--text-secondary);
          display: flex;
          align-items: center;
          gap: 6px;
        }
        .metric-pill-value {
          font-family: var(--font-mono);
          font-weight: 700;
        }
        .opt-swatch {
          width: 8px;
          height: 8px;
          border-radius: 50%;
          display: inline-block;
        }
        .educational-banner {
          background: rgba(99, 102, 241, 0.04);
          border: 1px dashed rgba(99, 102, 241, 0.2);
          border-radius: 12px;
          padding: 16px;
          font-size: 13px;
          color: var(--text-secondary);
          line-height: 1.6;
          display: flex;
          gap: 12px;
          align-items: flex-start;
        }
        .educational-banner svg {
          color: var(--color-primary);
          flex-shrink: 0;
          margin-top: 2px;
        }
      `}</style>

      <div className="matrix-explorer">
        <div className="matrix-header">
          <div className="tab-row" role="tablist" aria-label="Matrix Dynamics Models">
            <button
              type="button"
              role="tab"
              aria-selected={activeTab === 'factorization'}
              className={`tab-button ${activeTab === 'factorization' ? 'active' : ''}`}
              onClick={() => setActiveTab('factorization')}
            >
              <Layers size={16} aria-hidden="true" />
              <span>Deep Matrix Factorization</span>
            </button>
            <button
              type="button"
              role="tab"
              aria-selected={activeTab === 'transformer'}
              className={`tab-button ${activeTab === 'transformer' ? 'active' : ''}`}
              onClick={() => setActiveTab('transformer')}
            >
              <Activity size={16} aria-hidden="true" />
              <span>MNIST Vision Transformer</span>
            </button>
          </div>
        </div>

        {activeTab === 'factorization' && mfData && mfMetrics && (
          <>
            <div className="educational-banner">
              <Info size={16} aria-hidden="true" />
              <div>
                <strong>Deep Matrix Factorization Dynamics (6-layer linear network)</strong>
                <br />
                This experiment factorizes a matrix by training a chain of linear layers W = W_L * W_(L-1) * ... * W_1. 
                Without proper optimization, linear networks suffer from ill-conditioning, causing the condition number 
                κ(W₂) of intermediate layers to explode. Muon utilizes online orthogonalization (via Newton-Schulz iteration) 
                to keep weight matrices close to the manifold of orthogonal matrices, stabilizing singular values and accelerating convergence.
              </div>
            </div>

            <div className="metrics-summary-grid">
              <div className="metric-pill">
                <span className="metric-pill-label">Final MSE Loss (Step 499)</span>
                {Object.keys(mfMetrics).map((opt) => (
                  <div className="metric-pill-val-row" key={opt}>
                    <span className="metric-pill-opt">
                      <span className="opt-swatch" style={{ background: OPT_COLORS[opt] }} />
                      {opt}
                    </span>
                    <span className="metric-pill-value" style={{ color: OPT_COLORS[opt] }}>
                      {mfMetrics[opt].finalLoss.toFixed(6)}
                    </span>
                  </div>
                ))}
              </div>

              <div className="metric-pill">
                <span className="metric-pill-label">Final Condition Number κ(W₂)</span>
                {Object.keys(mfMetrics).map((opt) => (
                  <div className="metric-pill-val-row" key={opt}>
                    <span className="metric-pill-opt">
                      <span className="opt-swatch" style={{ background: OPT_COLORS[opt] }} />
                      {opt}
                    </span>
                    <span className="metric-pill-value" style={{ color: OPT_COLORS[opt] }}>
                      {mfMetrics[opt].finalCond.toFixed(2)}
                    </span>
                  </div>
                ))}
              </div>

              <div className="metric-pill">
                <span className="metric-pill-label">Peak Condition Number κ(W₂)</span>
                {Object.keys(mfMetrics).map((opt) => (
                  <div className="metric-pill-val-row" key={opt}>
                    <span className="metric-pill-opt">
                      <span className="opt-swatch" style={{ background: OPT_COLORS[opt] }} />
                      {opt}
                    </span>
                    <span className="metric-pill-value" style={{ color: OPT_COLORS[opt] }}>
                      {mfMetrics[opt].maxCond.toFixed(2)}
                    </span>
                  </div>
                ))}
              </div>
            </div>

            <div className="matrix-grid">
              <div className="matrix-card">
                <div className="matrix-card-header">
                  <h4 className="matrix-card-title">
                    <TrendingUp size={16} aria-hidden="true" />
                    MSE Loss Trajectory
                  </h4>
                </div>
                <ResponsiveContainer width="100%" height={280}>
                  <LineChart data={mfLossData} margin={{ top: 8, right: 16, left: -20, bottom: 8 }}>
                    <CartesianGrid stroke="rgba(255,255,255,0.06)" />
                    <XAxis dataKey="step" stroke="#9ca3af" fontSize={11} />
                    <YAxis stroke="#9ca3af" fontSize={11} />
                    <Tooltip content={<CustomTooltip labelPrefix="Step: " />} />
                    <Legend iconType="circle" />
                    {Object.keys(mfData).map((opt) => (
                      <Line
                        key={opt}
                        type="monotone"
                        dataKey={opt}
                        stroke={OPT_COLORS[opt] || '#9ca3af'}
                        dot={false}
                        strokeWidth={2}
                      />
                    ))}
                  </LineChart>
                </ResponsiveContainer>
              </div>

              <div className="matrix-card">
                <div className="matrix-card-header">
                  <h4 className="matrix-card-title">
                    <Activity size={16} aria-hidden="true" />
                    Condition Number κ(W₂)
                  </h4>
                  <div className="matrix-controls-row">
                    <span className="text-muted" style={{ fontSize: '11px' }}>Y-Scale:</span>
                    <div className="toggle-group">
                      <button
                        type="button"
                        className={`toggle-btn ${mfCondScale === 'linear' ? 'active' : ''}`}
                        onClick={() => setMfCondScale('linear')}
                      >
                        Linear
                      </button>
                      <button
                        type="button"
                        className={`toggle-btn ${mfCondScale === 'log' ? 'active' : ''}`}
                        onClick={() => setMfCondScale('log')}
                      >
                        Log
                      </button>
                    </div>
                  </div>
                </div>
                <ResponsiveContainer width="100%" height={280}>
                  <LineChart data={mfCondData} margin={{ top: 8, right: 16, left: -20, bottom: 8 }}>
                    <CartesianGrid stroke="rgba(255,255,255,0.06)" />
                    <XAxis dataKey="step" stroke="#9ca3af" fontSize={11} />
                    <YAxis
                      stroke="#9ca3af"
                      fontSize={11}
                      scale={mfCondScale}
                      domain={mfCondScale === 'log' ? [1, 'auto'] : ['auto', 'auto']}
                    />
                    <Tooltip content={<CustomTooltip labelPrefix="Step: " />} />
                    <Legend iconType="circle" />
                    {Object.keys(mfData).map((opt) => (
                      <Line
                        key={opt}
                        type="monotone"
                        dataKey={opt}
                        stroke={OPT_COLORS[opt] || '#9ca3af'}
                        dot={false}
                        strokeWidth={2}
                      />
                    ))}
                  </LineChart>
                </ResponsiveContainer>
              </div>
            </div>

            <div className="matrix-grid matrix-grid-full">
              <div className="matrix-card">
                <div className="matrix-card-header">
                  <h4 className="matrix-card-title">
                    <BarChart2 size={16} aria-hidden="true" />
                    Singular Value Spectrum Profile
                  </h4>
                  <div className="matrix-controls-row">
                    <div className="step-selector" role="group" aria-label="Select Training Step">
                      {['0', '50', '100', '250', '499'].map((step) => (
                        <button
                          key={step}
                          type="button"
                          className={`step-btn ${mfStep === step ? 'active' : ''}`}
                          onClick={() => setMfStep(step)}
                        >
                          Step {step}
                        </button>
                      ))}
                    </div>
                    <div className="toggle-group">
                      <button
                        type="button"
                        className={`toggle-btn ${mfSpectrumType === 'line' ? 'active' : ''}`}
                        onClick={() => setMfSpectrumType('line')}
                      >
                        Line
                      </button>
                      <button
                        type="button"
                        className={`toggle-btn ${mfSpectrumType === 'bar' ? 'active' : ''}`}
                        onClick={() => setMfSpectrumType('bar')}
                      >
                        Bar
                      </button>
                    </div>
                  </div>
                </div>

                <p className="text-secondary" style={{ fontSize: '13px', margin: '0 0 8px 0' }}>
                  Viewing spectrum profile of W₂ at **Step {mfStep}**. Flat spectra indicate orthogonal weights 
                  (σ_i ≈ 1), which prevents gradient vanishing or exploding across deep layers.
                </p>

                <ResponsiveContainer width="100%" height={320}>
                  {mfSpectrumType === 'line' ? (
                    <LineChart data={mfSpectrumData} margin={{ top: 8, right: 16, left: -20, bottom: 8 }}>
                      <CartesianGrid stroke="rgba(255,255,255,0.06)" />
                      <XAxis dataKey="index" stroke="#9ca3af" fontSize={11} label={{ value: 'Singular Value Index', position: 'insideBottom', offset: -5, fill: '#9ca3af', fontSize: 11 }} />
                      <YAxis stroke="#9ca3af" fontSize={11} />
                      <Tooltip content={<CustomTooltip labelPrefix="Singular Value #" />} />
                      <Legend iconType="circle" />
                      {Object.keys(mfData).map((opt) => (
                        <Line
                          key={opt}
                          type="monotone"
                          dataKey={opt}
                          stroke={OPT_COLORS[opt] || '#9ca3af'}
                          dot={false}
                          strokeWidth={2}
                        />
                      ))}
                    </LineChart>
                  ) : (
                    <BarChart data={mfSpectrumData} margin={{ top: 8, right: 16, left: -20, bottom: 8 }}>
                      <CartesianGrid stroke="rgba(255,255,255,0.06)" />
                      <XAxis dataKey="index" stroke="#9ca3af" fontSize={11} label={{ value: 'Singular Value Index', position: 'insideBottom', offset: -5, fill: '#9ca3af', fontSize: 11 }} />
                      <YAxis stroke="#9ca3af" fontSize={11} />
                      <Tooltip content={<CustomTooltip labelPrefix="Singular Value #" />} />
                      <Legend iconType="circle" />
                      {Object.keys(mfData).map((opt) => (
                        <Bar
                          key={opt}
                          dataKey={opt}
                          fill={OPT_COLORS[opt] || '#9ca3af'}
                          radius={[2, 2, 0, 0]}
                        />
                      ))}
                    </BarChart>
                  )}
                </ResponsiveContainer>
              </div>
            </div>
          </>
        )}

        {activeTab === 'transformer' && tsData && tsMetrics && (
          <>
            <div className="educational-banner">
              <Info size={16} aria-hidden="true" />
              <div>
                <strong>MNIST Vision Transformer Spectral Dynamics (QK Projection Layer)</strong>
                <br />
                This dashboard tracks the query projection weight matrix spectral properties inside a Vision Transformer (ViT) trained on MNIST. 
                In attention mechanisms, representation collapse is often correlated with attention entropy decay. 
                Using Muon (with auxiliary SGD/AdamW parameters for scalar biases/non-matrix weights) keeps the projection matrix 
                singular values bounded and prevents the entropy of attention maps from collapsing rapidly, preserving representation capacity.
              </div>
            </div>

            <div className="metrics-summary-grid">
              <div className="metric-pill">
                <span className="metric-pill-label">Final Training Loss</span>
                {Object.keys(tsMetrics).map((opt) => (
                  <div className="metric-pill-val-row" key={opt}>
                    <span className="metric-pill-opt">
                      <span className="opt-swatch" style={{ background: OPT_COLORS[opt] }} />
                      {opt}
                    </span>
                    <span className="metric-pill-value" style={{ color: OPT_COLORS[opt] }}>
                      {tsMetrics[opt].finalLoss.toFixed(5)}
                    </span>
                  </div>
                ))}
              </div>

              <div className="metric-pill">
                <span className="metric-pill-label">Final Attention Entropy</span>
                {Object.keys(tsMetrics).map((opt) => (
                  <div className="metric-pill-val-row" key={opt}>
                    <span className="metric-pill-opt">
                      <span className="opt-swatch" style={{ background: OPT_COLORS[opt] }} />
                      {opt}
                    </span>
                    <span className="metric-pill-value" style={{ color: OPT_COLORS[opt] }}>
                      {tsMetrics[opt].finalEntropy.toFixed(4)}
                    </span>
                  </div>
                ))}
              </div>

              <div className="metric-pill">
                <span className="metric-pill-label">Projection Condition Number (Step 50)</span>
                {Object.keys(tsMetrics).map((opt) => (
                  <div className="metric-pill-val-row" key={opt}>
                    <span className="metric-pill-opt">
                      <span className="opt-swatch" style={{ background: OPT_COLORS[opt] }} />
                      {opt}
                    </span>
                    <span className="metric-pill-value" style={{ color: OPT_COLORS[opt] }}>
                      {tsMetrics[opt].finalCond.toFixed(2)}
                    </span>
                  </div>
                ))}
              </div>
            </div>

            <div className="matrix-grid">
              <div className="matrix-card">
                <div className="matrix-card-header">
                  <h4 className="matrix-card-title">
                    <TrendingUp size={16} aria-hidden="true" />
                    Training Loss Trajectory
                  </h4>
                </div>
                <ResponsiveContainer width="100%" height={280}>
                  <LineChart data={tsLossData} margin={{ top: 8, right: 16, left: -20, bottom: 8 }}>
                    <CartesianGrid stroke="rgba(255,255,255,0.06)" />
                    <XAxis dataKey="step" stroke="#9ca3af" fontSize={11} />
                    <YAxis stroke="#9ca3af" fontSize={11} />
                    <Tooltip content={<CustomTooltip labelPrefix="Step: " />} />
                    <Legend iconType="circle" />
                    {Object.keys(tsData).map((opt) => (
                      <Line
                        key={opt}
                        type="monotone"
                        dataKey={opt}
                        stroke={OPT_COLORS[opt] || '#9ca3af'}
                        dot={false}
                        strokeWidth={2}
                      />
                    ))}
                  </LineChart>
                </ResponsiveContainer>
              </div>

              <div className="matrix-card">
                <div className="matrix-card-header">
                  <h4 className="matrix-card-title">
                    <Activity size={16} aria-hidden="true" />
                    Attention Map Entropy
                  </h4>
                </div>
                <ResponsiveContainer width="100%" height={280}>
                  <LineChart data={tsEntropyData} margin={{ top: 8, right: 16, left: -20, bottom: 8 }}>
                    <CartesianGrid stroke="rgba(255,255,255,0.06)" />
                    <XAxis dataKey="step" stroke="#9ca3af" fontSize={11} />
                    <YAxis stroke="#9ca3af" fontSize={11} domain={['auto', 'auto']} />
                    <Tooltip content={<CustomTooltip labelPrefix="Step: " />} />
                    <Legend iconType="circle" />
                    {Object.keys(tsData).map((opt) => (
                      <Line
                        key={opt}
                        type="monotone"
                        dataKey={opt}
                        stroke={OPT_COLORS[opt] || '#9ca3af'}
                        dot={false}
                        strokeWidth={2}
                      />
                    ))}
                  </LineChart>
                </ResponsiveContainer>
              </div>
            </div>

            <div className="matrix-grid matrix-grid-full">
              <div className="matrix-card">
                <div className="matrix-card-header">
                  <h4 className="matrix-card-title">
                    <BarChart2 size={16} aria-hidden="true" />
                    Query Projection Singular Value Spectrum
                  </h4>
                  <div className="matrix-controls-row">
                    <div className="step-selector" role="group" aria-label="Select Training Step">
                      {['0', '10', '30', '50'].map((step) => (
                        <button
                          key={step}
                          type="button"
                          className={`step-btn ${tsStep === step ? 'active' : ''}`}
                          onClick={() => setTsStep(step)}
                        >
                          Step {step}
                        </button>
                      ))}
                    </div>
                    <div className="toggle-group">
                      <button
                        type="button"
                        className={`toggle-btn ${tsSpectrumType === 'line' ? 'active' : ''}`}
                        onClick={() => setTsSpectrumType('line')}
                      >
                        Line
                      </button>
                      <button
                        type="button"
                        className={`toggle-btn ${tsSpectrumType === 'bar' ? 'active' : ''}`}
                        onClick={() => setTsSpectrumType('bar')}
                      >
                        Bar
                      </button>
                    </div>
                  </div>
                </div>

                <p className="text-secondary" style={{ fontSize: '13px', margin: '0 0 8px 0' }}>
                  Viewing spectrum profile of Query Projection weight matrix (W_Q) at **Step {tsStep}**. 
                  Muon+Aux restrains singular value decay, ensuring that the projection maintains high rank/dimension 
                  expressiveness compared to standard AdamW.
                </p>

                <ResponsiveContainer width="100%" height={320}>
                  {tsSpectrumType === 'line' ? (
                    <LineChart data={tsSpectrumData} margin={{ top: 8, right: 16, left: -20, bottom: 8 }}>
                      <CartesianGrid stroke="rgba(255,255,255,0.06)" />
                      <XAxis dataKey="index" stroke="#9ca3af" fontSize={11} label={{ value: 'Singular Value Index', position: 'insideBottom', offset: -5, fill: '#9ca3af', fontSize: 11 }} />
                      <YAxis stroke="#9ca3af" fontSize={11} />
                      <Tooltip content={<CustomTooltip labelPrefix="Singular Value #" />} />
                      <Legend iconType="circle" />
                      {Object.keys(tsData).map((opt) => (
                        <Line
                          key={opt}
                          type="monotone"
                          dataKey={opt}
                          stroke={OPT_COLORS[opt] || '#9ca3af'}
                          dot={false}
                          strokeWidth={2}
                        />
                      ))}
                    </LineChart>
                  ) : (
                    <BarChart data={tsSpectrumData} margin={{ top: 8, right: 16, left: -20, bottom: 8 }}>
                      <CartesianGrid stroke="rgba(255,255,255,0.06)" />
                      <XAxis dataKey="index" stroke="#9ca3af" fontSize={11} label={{ value: 'Singular Value Index', position: 'insideBottom', offset: -5, fill: '#9ca3af', fontSize: 11 }} />
                      <YAxis stroke="#9ca3af" fontSize={11} />
                      <Tooltip content={<CustomTooltip labelPrefix="Singular Value #" />} />
                      <Legend iconType="circle" />
                      {Object.keys(tsData).map((opt) => (
                        <Bar
                          key={opt}
                          dataKey={opt}
                          fill={OPT_COLORS[opt] || '#9ca3af'}
                          radius={[2, 2, 0, 0]}
                        />
                      ))}
                    </BarChart>
                  )}
                </ResponsiveContainer>
              </div>
            </div>
          </>
        )}
      </div>
    </div>
  )
}
