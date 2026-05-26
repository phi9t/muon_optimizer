import { useEffect, useMemo, useState } from 'react'
import { errorMessage, fetchExplorerJson } from '../lib/fetch'
import Charts from './Charts'
import Overview from './Overview'
import type { DefinitionsPayload, MnistPayload } from './types'

export default function BenchmarksExplorer() {
  const [definitions, setDefinitions] = useState<DefinitionsPayload | null>(null)
  const [mnist, setMnist] = useState<MnistPayload | null>(null)
  const [selectedOptimizers, setSelectedOptimizers] = useState<string[]>([])
  const [loading, setLoading] = useState(true)
  const [error, setError] = useState<string | null>(null)

  useEffect(() => {
    async function load() {
      try {
        setLoading(true)
        setError(null)
        const [defs, data] = await Promise.all([
          fetchExplorerJson<DefinitionsPayload>('definitions.json'),
          fetchExplorerJson<MnistPayload>('mnist.json'),
        ])
        setDefinitions(defs)
        setMnist(data)
        setSelectedOptimizers(data.optimizers.map((o) => o.name))
      } catch (err: unknown) {
        setError(errorMessage(err))
      } finally {
        setLoading(false)
      }
    }
    void load()
  }, [])

  const visibleOptimizers = useMemo(() => {
    if (!mnist) return []
    const selected = new Set(selectedOptimizers)
    return mnist.optimizers.filter((o) => selected.has(o.name))
  }, [mnist, selectedOptimizers])

  function toggleOptimizer(name: string) {
    setSelectedOptimizers((prev) =>
      prev.includes(name) ? prev.filter((n) => n !== name) : [...prev, name],
    )
  }

  if (loading) {
    return (
      <div className="loading-container" role="status">
        <div className="spinner" aria-hidden="true" />
        <p>Loading MNIST benchmark data…</p>
      </div>
    )
  }

  if (error || !definitions || !mnist) {
    return (
      <div className="card-view text-center">
        <p className="text-danger">Failed to load benchmark data: {error ?? 'unknown'}</p>
        <button type="button" className="pg-btn retry-btn" onClick={() => window.location.reload()}>
          Retry
        </button>
      </div>
    )
  }

  return (
    <div className="mo-shell pb-shell">
      {mnist.profile === 'lite' && (
        <p className="mo-caveat">
          Lite profile ({mnist.epochs} epochs). Regenerate with{' '}
          <code>uv run python scripts/export_explorer_data.py --profile full</code> for the full
          15-epoch MNIST run.
        </p>
      )}

      <Overview
        optimizers={mnist.optimizers}
        definitions={definitions}
        selectedOptimizers={selectedOptimizers}
        onToggleOptimizer={toggleOptimizer}
      />

      <Charts optimizers={visibleOptimizers} />
    </div>
  )
}
