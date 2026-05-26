import { useEffect, useMemo, useState } from 'react'
import { errorMessage, fetchExplorerJson } from '../lib/fetch'
import ContourPlot from './ContourPlot'
import TrajectoryTable from './TrajectoryTable'
import type { LandscapeIndex, LandscapePayload } from './types'

const OPTIMIZERS = ['SGD', 'Adam', 'Muon'] as const

export default function LandscapesExplorer() {
  const [index, setIndex] = useState<LandscapeIndex | null>(null)
  const [selectedId, setSelectedId] = useState<string | null>(null)
  const [payload, setPayload] = useState<LandscapePayload | null>(null)
  const [tableOptimizer, setTableOptimizer] = useState<string>('Muon')
  const [loadingIndex, setLoadingIndex] = useState(true)
  const [loadingProblem, setLoadingProblem] = useState(false)
  const [error, setError] = useState<string | null>(null)

  useEffect(() => {
    async function loadIndex() {
      try {
        setLoadingIndex(true)
        setError(null)
        const data = await fetchExplorerJson<LandscapeIndex>('landscapes/index.json')
        setIndex(data)
        setSelectedId(data.problems[0]?.id ?? null)
      } catch (err: unknown) {
        setError(errorMessage(err))
      } finally {
        setLoadingIndex(false)
      }
    }
    void loadIndex()
  }, [])

  useEffect(() => {
    if (!selectedId) return
    let cancelled = false

    async function loadProblem() {
      try {
        setLoadingProblem(true)
        setError(null)
        const data = await fetchExplorerJson<LandscapePayload>(`landscapes/${selectedId}.json`)
        if (!cancelled) {
          setPayload(data)
        }
      } catch (err: unknown) {
        if (!cancelled) {
          setPayload(null)
          setError(errorMessage(err))
        }
      } finally {
        if (!cancelled) {
          setLoadingProblem(false)
        }
      }
    }

    void loadProblem()
    return () => {
      cancelled = true
    }
  }, [selectedId])

  const selectedEntry = useMemo(
    () => index?.problems.find((p) => p.id === selectedId),
    [index, selectedId],
  )

  if (loadingIndex) {
    return (
      <div className="loading-container" role="status">
        <div className="spinner" aria-hidden="true" />
        <p>Loading landscape index…</p>
      </div>
    )
  }

  if (error && !index) {
    return (
      <div className="card-view text-center">
        <p className="text-danger">{error}</p>
      </div>
    )
  }

  return (
    <div className="dashboard-grid">
      <aside className="sidebar-panel">
        <h2 className="sidebar-title">Problems</h2>
        {index?.profile === 'lite' && (
          <p className="text-muted" style={{ fontSize: 12, margin: 0 }}>
            Lite profile — 3 landscapes
          </p>
        )}
        <ul className="dataset-list">
          {index?.problems.map((problem) => (
            <li key={problem.id}>
              <button
                type="button"
                className={`dataset-item ${selectedId === problem.id ? 'active' : ''}`}
                onClick={() => setSelectedId(problem.id)}
              >
                <div className="dataset-item-left">
                  <span className="dataset-name-label">{problem.name}</span>
                  <span className="dataset-desc-label">{problem.description}</span>
                </div>
                <div className="dataset-item-right">
                  <span className="dataset-row-count">{problem.steps} steps</span>
                </div>
              </button>
            </li>
          ))}
        </ul>
      </aside>

      <div className="main-view-panel">
        {loadingProblem && (
          <div className="loading-container" role="status">
            <div className="spinner" aria-hidden="true" />
          </div>
        )}

        {!loadingProblem && payload && selectedEntry && (
          <>
            <section className="card-view">
              <div className="dataset-header-section">
                <div className="dataset-title-meta">
                  <h2>{payload.name}</h2>
                  <p>{payload.description}</p>
                </div>
                <div className="dataset-attributes">
                  <span className="attr-badge">
                    Start: ({payload.initial_point.x}, {payload.initial_point.y})
                  </span>
                  {payload.minimum.x != null && (
                    <span className="attr-badge">
                      Min: ({payload.minimum.x}, {payload.minimum.y})
                    </span>
                  )}
                </div>
              </div>

              <div className="mo-legend" aria-label="Trajectory legend">
                {OPTIMIZERS.map((name) => (
                  <span key={name} className="mo-legend-item">
                    <span className={`mo-legend-swatch mo-legend-${name.toLowerCase()}`} />
                    {name}
                  </span>
                ))}
                <span className="mo-legend-item">
                  <span className="mo-legend-swatch" style={{ background: '#f59e0b' }} />
                  Start
                </span>
                <span className="mo-legend-item">
                  <span className="mo-legend-swatch" style={{ background: '#06b6d4' }} />
                  Minimum
                </span>
              </div>

              <ContourPlot
                grid={payload.grid}
                trajectories={payload.trajectories}
                bounds={{
                  x: payload.bounds.x as [number, number],
                  y: payload.bounds.y as [number, number],
                }}
                minimum={payload.minimum}
                initialPoint={payload.initial_point}
              />
            </section>

            <section className="card-view">
              <div className="tab-row" role="tablist" aria-label="Trajectory table optimizer">
                {OPTIMIZERS.map((name) => (
                  <button
                    key={name}
                    type="button"
                    role="tab"
                    aria-selected={tableOptimizer === name}
                    className={`tab-button ${tableOptimizer === name ? 'active' : ''}`}
                    onClick={() => setTableOptimizer(name)}
                  >
                    {name}
                  </button>
                ))}
              </div>
              <TrajectoryTable
                trajectories={payload.trajectories}
                activeOptimizer={tableOptimizer}
              />
            </section>
          </>
        )}

        {error && index && (
          <p className="text-danger text-center">{error}</p>
        )}
      </div>
    </div>
  )
}
