import {
  CartesianGrid,
  Legend,
  Line,
  LineChart,
  ResponsiveContainer,
  Tooltip,
  XAxis,
  YAxis,
} from 'recharts'
import type { OptimizerSeries } from './types'

const COLORS: Record<string, string> = {
  SGD: '#ef4444',
  Adam: '#3b82f6',
  Muon: '#10b981',
}

function buildChartRows(optimizers: OptimizerSeries[], field: 'test_accuracies' | 'train_losses') {
  const maxLen = Math.max(...optimizers.map((o) => o[field].length), 0)
  const rows: Record<string, number | string>[] = []
  for (let i = 0; i < maxLen; i += 1) {
    const row: Record<string, number | string> = { epoch: i + 1 }
    for (const opt of optimizers) {
      if (i < opt[field].length) {
        row[opt.name] = opt[field][i]
      }
    }
    rows.push(row)
  }
  return rows
}

export default function Charts({ optimizers }: { optimizers: OptimizerSeries[] }) {
  if (optimizers.length === 0) return null

  const accData = buildChartRows(optimizers, 'test_accuracies')
  const lossData = buildChartRows(optimizers, 'train_losses')

  return (
    <div className="chart-grid">
      <div className="chart-card">
        <h4>Test accuracy by epoch</h4>
        <ResponsiveContainer width="100%" height={280}>
          <LineChart data={accData} margin={{ top: 8, right: 16, left: 0, bottom: 8 }}>
            <CartesianGrid stroke="rgba(255,255,255,0.06)" />
            <XAxis dataKey="epoch" stroke="#9ca3af" fontSize={12} />
            <YAxis stroke="#9ca3af" fontSize={12} domain={['auto', 'auto']} />
            <Tooltip
              contentStyle={{
                background: '#111827',
                border: '1px solid rgba(255,255,255,0.1)',
                borderRadius: 8,
              }}
            />
            <Legend />
            {optimizers.map((opt) => (
              <Line
                key={opt.name}
                type="monotone"
                dataKey={opt.name}
                stroke={COLORS[opt.name] ?? '#6366f1'}
                dot={false}
                strokeWidth={2}
              />
            ))}
          </LineChart>
        </ResponsiveContainer>
      </div>

      <div className="chart-card">
        <h4>Training loss by epoch</h4>
        <ResponsiveContainer width="100%" height={280}>
          <LineChart data={lossData} margin={{ top: 8, right: 16, left: 0, bottom: 8 }}>
            <CartesianGrid stroke="rgba(255,255,255,0.06)" />
            <XAxis dataKey="epoch" stroke="#9ca3af" fontSize={12} />
            <YAxis stroke="#9ca3af" fontSize={12} />
            <Tooltip
              contentStyle={{
                background: '#111827',
                border: '1px solid rgba(255,255,255,0.1)',
                borderRadius: 8,
              }}
            />
            <Legend />
            {optimizers.map((opt) => (
              <Line
                key={opt.name}
                type="monotone"
                dataKey={opt.name}
                stroke={COLORS[opt.name] ?? '#6366f1'}
                dot={false}
                strokeWidth={2}
              />
            ))}
          </LineChart>
        </ResponsiveContainer>
      </div>
    </div>
  )
}
