import { useState } from 'react'
import { ArrowLeft } from 'lucide-react'
import { REPO_HOME, logoMarkUrl } from './lib/assets'
import BenchmarksExplorer from './benchmarks/BenchmarksExplorer'
import ReferenceExplorer from './reference/ReferenceExplorer'
import QwenLogitsExplorer from './qwen/QwenLogitsExplorer'
import MatrixDynamicsExplorer from './matrix/MatrixDynamicsExplorer'

export type ExplorerFamily = 'benchmarks' | 'reference' | 'qwen' | 'matrix'

const FAMILY_SUBTITLES: Record<ExplorerFamily, string> = {
  benchmarks: 'MNIST CNN training — Muon vs SGD vs Adam',
  reference: 'Algorithm overview, API classes, and usage snippets',
  qwen: 'Qwen teacher-student logits comparison',
  matrix: 'Weight conditioning and singular value spectra',
}

export default function App() {
  const [family, setFamily] = useState<ExplorerFamily>('benchmarks')

  return (
    <div className="relative min-h-screen">
      <div className="observatory-bg" aria-hidden="true" />

      <div className="explorer-container">
        <header className="explorer-header family-header">
          <div className="header-title-section">
            <a href="#main-content" className="skip-link">
              Skip to main content
            </a>
            <a
              href={REPO_HOME}
              className="back-home-link"
              target="_blank"
              rel="noopener noreferrer"
            >
              <ArrowLeft size={16} aria-hidden="true" />
              <span>Back to muon_optimizer repo</span>
            </a>
            <div className="header-title-row">
              <img
                src={logoMarkUrl()}
                alt="Muon Optimizer"
                className="header-logo"
                width={32}
                height={32}
              />
              <h1>Muon Optimizer Explorer</h1>
            </div>
            <p>{FAMILY_SUBTITLES[family]}</p>
          </div>

          <div className="family-switch" role="group" aria-label="Explorer section">
            <button
              type="button"
              className={`family-switch-btn ${family === 'benchmarks' ? 'active' : ''}`}
              aria-pressed={family === 'benchmarks'}
              onClick={() => setFamily('benchmarks')}
            >
              Benchmarks
            </button>

            <button
              type="button"
              className={`family-switch-btn ${family === 'reference' ? 'active' : ''}`}
              aria-pressed={family === 'reference'}
              onClick={() => setFamily('reference')}
            >
              Reference
            </button>
            <button
              type="button"
              className={`family-switch-btn ${family === 'qwen' ? 'active' : ''}`}
              aria-pressed={family === 'qwen'}
              onClick={() => setFamily('qwen')}
            >
              Qwen Logits
            </button>
            <button
              type="button"
              className={`family-switch-btn ${family === 'matrix' ? 'active' : ''}`}
              aria-pressed={family === 'matrix'}
              onClick={() => setFamily('matrix')}
            >
              Matrix Dynamics
            </button>
          </div>
        </header>

        <div id="main-content">
          {family === 'benchmarks' && <BenchmarksExplorer />}
          {family === 'reference' && <ReferenceExplorer />}
          {family === 'qwen' && <QwenLogitsExplorer />}
          {family === 'matrix' && <MatrixDynamicsExplorer />}
        </div>
      </div>
    </div>
  )
}
