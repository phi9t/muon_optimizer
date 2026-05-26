import { useMemo } from 'react'
import type { LandscapeGrid, TrajectoryPoint } from './types'

const TRAJ_COLORS: Record<string, string> = {
  SGD: '#ef4444',
  Adam: '#3b82f6',
  Muon: '#10b981',
}

function lossToColor(t: number): string {
  const r = Math.round(30 + t * 80)
  const g = Math.round(40 + t * 60)
  const b = Math.round(100 + (1 - t) * 120)
  return `rgb(${r},${g},${b})`
}

export default function ContourPlot({
  grid,
  trajectories,
  bounds,
  minimum,
  initialPoint,
  width = 640,
  height = 400,
}: {
  grid: LandscapeGrid
  trajectories: Record<string, TrajectoryPoint[]>
  bounds: { x: [number, number]; y: [number, number] }
  minimum: { x: number | null; y: number | null }
  initialPoint: { x: number; y: number }
  width?: number
  height?: number
}) {
  const { cells, pathData } = useMemo(() => {
    const [x0, x1] = bounds.x
    const [y0, y1] = bounds.y
    const nx = grid.x.length
    const ny = grid.y.length
    const cellH = height / ny
    const zRange = grid.z_max - grid.z_min || 1

    const cellEls: { key: string; x: number; y: number; fill: string }[] = []
    for (let j = 0; j < ny; j += 1) {
      for (let i = 0; i < nx; i += 1) {
        const z = grid.values[j][i]
        const t = Math.min(1, Math.max(0, (z - grid.z_min) / zRange))
        const px = ((grid.x[i] - x0) / (x1 - x0)) * width
        const py = height - ((grid.y[j] - y0) / (y1 - y0)) * height
        cellEls.push({
          key: `${i}-${j}`,
          x: px,
          y: py - cellH,
          fill: lossToColor(t),
        })
      }
    }

    const toSvg = (x: number, y: number) => {
      const px = ((x - x0) / (x1 - x0)) * width
      const py = height - ((y - y0) / (y1 - y0)) * height
      return `${px},${py}`
    }

    const paths: { name: string; d: string; color: string }[] = []
    for (const [name, points] of Object.entries(trajectories)) {
      if (points.length === 0) continue
      const d = points.map((p, idx) => `${idx === 0 ? 'M' : 'L'} ${toSvg(p.x, p.y)}`).join(' ')
      paths.push({ name, d, color: TRAJ_COLORS[name] ?? '#6366f1' })
    }

    return { cells: cellEls, pathData: paths }
  }, [grid, trajectories, bounds, width, height])

  const minPt =
    minimum.x != null && minimum.y != null
      ? (() => {
          const [x0, x1] = bounds.x
          const [y0, y1] = bounds.y
          const px = ((minimum.x - x0) / (x1 - x0)) * width
          const py = height - ((minimum.y - y0) / (y1 - y0)) * height
          return { px, py }
        })()
      : null

  const [x0, x1] = bounds.x
  const [y0, y1] = bounds.y
  const initPx = ((initialPoint.x - x0) / (x1 - x0)) * width
  const initPy = height - ((initialPoint.y - y0) / (y1 - y0)) * height

  return (
    <div className="mo-contour-wrap">
      <svg
        className="mo-contour-svg"
        viewBox={`0 0 ${width} ${height}`}
        role="img"
        aria-label="Loss contour with optimizer trajectories"
      >
        {cells.map((c) => (
          <rect
            key={c.key}
            x={c.x}
            y={c.y}
            width={width / grid.x.length + 1}
            height={height / grid.y.length + 1}
            fill={c.fill}
          />
        ))}
        {pathData.map((p) => (
          <path
            key={p.name}
            d={p.d}
            fill="none"
            stroke={p.color}
            strokeWidth={2}
            strokeLinecap="round"
            strokeLinejoin="round"
            opacity={0.9}
          />
        ))}
        <circle cx={initPx} cy={initPy} r={5} fill="#f59e0b" stroke="#fff" strokeWidth={1.5} />
        {minPt && (
          <circle cx={minPt.px} cy={minPt.py} r={5} fill="#06b6d4" stroke="#fff" strokeWidth={1.5} />
        )}
      </svg>
    </div>
  )
}
