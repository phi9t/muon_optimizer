import type { TrajectoryPoint } from './types'

export default function TrajectoryTable({
  trajectories,
  activeOptimizer,
}: {
  trajectories: Record<string, TrajectoryPoint[]>
  activeOptimizer: string
}) {
  const points = trajectories[activeOptimizer] ?? []

  return (
    <div className="table-wrapper">
      <table className="data-table">
        <thead>
          <tr>
            <th>Step</th>
            <th>X</th>
            <th>Y</th>
            <th>Loss</th>
          </tr>
        </thead>
        <tbody>
          {points.map((p) => (
            <tr key={p.step}>
              <td>{p.step}</td>
              <td className="key-cell">{p.x.toFixed(4)}</td>
              <td className="key-cell">{p.y.toFixed(4)}</td>
              <td>{p.loss.toFixed(4)}</td>
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  )
}
