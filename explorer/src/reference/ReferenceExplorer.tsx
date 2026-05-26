import { ExternalLink } from 'lucide-react'
import { PAPER_URL, REPO_HOME } from '../lib/assets'

const PIPELINE_STEPS = [
  {
    name: 'Momentum',
    desc: 'Exponential moving average of gradients (optional Nesterov lookahead).',
  },
  {
    name: 'Orthogonalize',
    desc: 'Newton-Schulz or Polar Express iteration maps each 2D update to a near-orthogonal matrix.',
  },
  {
    name: 'Scale & apply',
    desc: 'Update scaled by matrix dimension factor √(max(rows, cols)) and learning rate.',
  },
]

const OPTIMIZER_CLASSES = [
  {
    class: 'SingleDeviceMuon',
    use: 'Single GPU / CPU training on 2D+ parameters.',
  },
  {
    class: 'Muon',
    use: 'Distributed multi-GPU; parameters sorted and gathered across ranks.',
  },
  {
    class: 'SingleDeviceMuonWithAuxAdam',
    use: 'Hybrid: Muon for matrices, AdamW for 1D / embeddings / heads.',
  },
  {
    class: 'MuonWithAuxAdam',
    use: 'Distributed hybrid optimizer with automatic param-group routing.',
  },
  {
    class: 'create_muon_param_groups()',
    use: 'Builds Muon + AdamW groups from model parameter names and shapes.',
  },
]

const SNIPPET_BASIC = `import torch.nn as nn
from muon_optimizer import SingleDeviceMuon

model = nn.Sequential(nn.Linear(784, 512), nn.ReLU(), nn.Linear(512, 10))
optimizer = SingleDeviceMuon(model.parameters(), lr=0.02, momentum=0.95)

for data, target in dataloader:
    optimizer.zero_grad()
    loss = criterion(model(data), target)
    loss.backward()
    optimizer.step()`

const SNIPPET_HYBRID = `from muon_optimizer import SingleDeviceMuonWithAuxAdam, create_muon_param_groups

param_groups = create_muon_param_groups(
    model,
    muon_lr=0.02,
    adam_lr=3e-4,
    muon_momentum=0.95,
    weight_decay=0.01,
)
optimizer = SingleDeviceMuonWithAuxAdam(param_groups)`

export default function ReferenceExplorer() {
  return (
    <div className="main-view-panel">
      <section className="card-view">
        <h2>Muon update pipeline</h2>
        <p className="text-secondary" style={{ marginTop: 0 }}>
          MomentUm Orthogonalized by Newton-schulz replaces raw 2D gradient updates with
          orthogonalized momentum steps for improved stability on matrix weights.
        </p>
        <div className="pipeline-strip">
          {PIPELINE_STEPS.map((step, i) => (
            <div key={step.name} className="pipeline-step">
              <span className="pipeline-step-num">{i + 1}</span>
              <div>
                <div className="pipeline-step-name">{step.name}</div>
                <p className="pipeline-step-desc">{step.desc}</p>
              </div>
            </div>
          ))}
        </div>
        <p style={{ marginTop: 16 }}>
          <a href={PAPER_URL} target="_blank" rel="noopener noreferrer" className="text-info">
            <ExternalLink size={14} style={{ verticalAlign: 'middle', marginRight: 6 }} />
            Keller Jordan — Muon post
          </a>
        </p>
      </section>

      <section className="card-view">
        <h2>Optimizer API</h2>
        <table className="mo-class-table">
          <thead>
            <tr>
              <th>Symbol</th>
              <th>When to use</th>
            </tr>
          </thead>
          <tbody>
            {OPTIMIZER_CLASSES.map((row) => (
              <tr key={row.class}>
                <td>
                  <code>{row.class}</code>
                </td>
                <td>{row.use}</td>
              </tr>
            ))}
          </tbody>
        </table>
        <p className="text-muted" style={{ fontSize: 13, marginTop: 12 }}>
          Muon applies to parameters with <code>ndim &gt;= 2</code> (linear layers, conv weights).
          Biases, embeddings, layer norms, and output heads typically use AdamW in hybrid setups.
        </p>
      </section>

      <section className="card-view">
        <h2>Usage snippets</h2>
        <h4 style={{ margin: '0 0 8px' }}>Basic SingleDeviceMuon</h4>
        <pre className="mo-snippet-block">{SNIPPET_BASIC}</pre>
        <h4 style={{ margin: '20px 0 8px' }}>Hybrid Muon + AdamW</h4>
        <pre className="mo-snippet-block">{SNIPPET_HYBRID}</pre>
        <p style={{ marginTop: 16 }}>
          <a href={REPO_HOME} target="_blank" rel="noopener noreferrer" className="text-primary">
            Full documentation on GitHub
          </a>
        </p>
      </section>
    </div>
  )
}
