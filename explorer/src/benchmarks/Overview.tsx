import type { DefinitionsPayload, OptimizerSeries } from './types'
import { formatMetric } from '../lib/fetch'
import MetricHelp from './MetricHelp'

const METRICS = [
  'best_test_acc',
  'final_test_acc',
  'convergence_epoch',
  'avg_epoch_time',
  'final_train_loss',
] as const

type MetricKey = (typeof METRICS)[number]

function metricLabel(metric: string): string {
  return metric.replaceAll('_', ' ')
}

function metricValue(series: OptimizerSeries, metric: MetricKey): number {
  return series.metrics[metric]
}

export default function Overview({
  optimizers,
  definitions,
  selectedOptimizers,
  onToggleOptimizer,
}: {
  optimizers: OptimizerSeries[]
  definitions: DefinitionsPayload
  selectedOptimizers: string[]
  onToggleOptimizer: (name: string) => void
}) {
  const selected = new Set(selectedOptimizers)
  const visible = optimizers.filter((o) => selected.has(o.name))

  return (
    <section className="pb-overview" aria-label="MNIST benchmark overview">
      <div className="pb-agent-row" role="group" aria-label="Optimizer filters">
        {optimizers.map((opt) => (
          <label key={opt.name} className="pb-agent-toggle">
            <input
              type="checkbox"
              checked={selected.has(opt.name)}
              onChange={() => onToggleOptimizer(opt.name)}
            />
            <span>{opt.name}</span>
          </label>
        ))}
      </div>

      <div className="pb-metric-grid">
        {visible.map((opt) => (
          <article key={opt.name} className="pb-agent-card">
            <div className="pb-agent-card-header">
              <h3>{opt.name}</h3>
              <p className="pb-gate-pass">
                <span>
                  Best test acc: {opt.metrics.best_test_acc.toFixed(2)}%
                </span>
              </p>
            </div>
            <dl className="pb-metrics">
              {METRICS.map((metric) => (
                <div key={metric}>
                  <dt>
                    <span>{metricLabel(metric)}</span>
                    <MetricHelp definitions={definitions} metric={metric} />
                  </dt>
                  <dd>
                    {metric === 'best_test_acc' || metric === 'final_test_acc'
                      ? `${formatMetric(metricValue(opt, metric))}%`
                      : formatMetric(metricValue(opt, metric))}
                  </dd>
                </div>
              ))}
            </dl>
          </article>
        ))}
      </div>

      {visible.length === 0 && (
        <p className="text-muted text-center">Select at least one optimizer to compare metrics.</p>
      )}
    </section>
  )
}
