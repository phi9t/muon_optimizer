import { useState } from 'react'
import { ArrowLeft } from 'lucide-react'
import { REPO_HOME, logoMarkUrl } from './lib/assets'
import BenchmarksExplorer from './benchmarks/BenchmarksExplorer'
import LandscapesExplorer from './landscapes/LandscapesExplorer'
import ReferenceExplorer from './reference/ReferenceExplorer'

export type ExplorerFamily = 'benchmarks' | 'landscapes' | 'reference'

const FAMILY_SUBTITLES: Record<ExplorerFamily, string> = {
  benchmarks: 'MNIST CNN training — Muon vs SGD vs Adam',
  landscapes: '2D loss surfaces and optimizer trajectories',
  reference: 'Algorithm overview, API classes, and usage snippets',
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
              className={`family-switch-btn ${family === 'landscapes' ? 'active' : ''}`}
              aria-pressed={family === 'landscapes'}
              onClick={() => setFamily('landscapes')}
            >
              2D Landscapes
            </button>
            <button
              type="button"
              className={`family-switch-btn ${family === 'reference' ? 'active' : ''}`}
              aria-pressed={family === 'reference'}
              onClick={() => setFamily('reference')}
            >
              Reference
            </button>
          </div>
        </header>

        <div id="main-content">
          {family === 'benchmarks' && <BenchmarksExplorer />}
          {family === 'landscapes' && <LandscapesExplorer />}
          {family === 'reference' && <ReferenceExplorer />}
        </div>
      </div>
    </div>
  )
}
