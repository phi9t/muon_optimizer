# Muon Optimizer Explorer

Vite + React dashboard showcasing MNIST optimizer benchmarks, 2D loss landscapes, and API reference.

**Live demo:** [https://phi9t.github.io/muon_optimizer/](https://phi9t.github.io/muon_optimizer/)

Enable Pages once: repo **Settings → Pages → Build and deployment → Source: GitHub Actions**.

## Local development

```bash
cd explorer
npm install && npm run dev
```

Open the URL printed by Vite (typically `http://localhost:5173`).

## Regenerate data bundles

Committed JSON under `public/data/` uses the **lite** profile by default (fast CI builds).

```bash
# From repo root
uv sync --group benchmarking
uv run python scripts/export_explorer_data.py --profile lite
```

Full benchmarks (15 MNIST epochs, all landscape problems):

```bash
uv run python scripts/export_explorer_data.py --profile full
```

## GitHub Pages build

```bash
npm run build:pages
```

Preview the production bundle:

```bash
npm run preview
```

## Structure

- `src/benchmarks/` — MNIST SGD vs Adam vs Muon charts
- `src/landscapes/` — 2D contour plots and trajectory tables
- `src/reference/` — Algorithm overview and code snippets
- `public/data/` — Static JSON exported by `scripts/export_explorer_data.py`
