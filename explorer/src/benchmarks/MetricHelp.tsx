import type { DefinitionsPayload } from './types'

export default function MetricHelp({
  definitions,
  metric,
}: {
  definitions: DefinitionsPayload
  metric: string
}) {
  const text = definitions.metrics[metric]?.plain
  if (!text) return null

  return (
    <details className="pb-help">
      <summary aria-label={`Help for ${metric}`}>?</summary>
      <span className="pb-help-tooltip" role="tooltip">
        {text}
      </span>
    </details>
  )
}
