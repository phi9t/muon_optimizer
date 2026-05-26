# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Commands

This project uses [uv](https://docs.astral.sh/uv/) for Python environments and dependency management. PyTorch is resolved from the CPU wheel index by default.

### Setup
- Install uv: `curl -LsSf https://astral.sh/uv/install.sh | sh`
- Create/sync environment: `uv sync`
- Sync with benchmarks: `uv sync --group benchmarking`
- Sync everything: `uv sync --all-groups`

### Testing
- Run all tests: `uv run pytest muon_optimizer_test.py -v`
- Run example tests: `uv run pytest example_usage_test.py -v`
- Run specific test class: `uv run pytest muon_optimizer_test.py::TestMuonOptimizer -v`
- Run single test method: `uv run pytest muon_optimizer_test.py::TestMuonOptimizer::test_initialization -v`

### Code Quality
Development tools are installed via the `dev` dependency group (`uv sync` includes it by default):
- Code formatting: `uv run black --line-length=120 muon_optimizer.py`
- Import sorting: `uv run isort --line-length=120 muon_optimizer.py`
- Linting: `uv run flake8 muon_optimizer.py --max-line-length=120 --extend-ignore=E203,W503,E501`
- Type checking: `uv run mypy muon_optimizer.py`

Note: The project uses a 120-character line length limit with E501 (line too long) errors ignored for flexibility.

### CI/CD Pipeline
The project uses GitHub Actions for continuous integration:
- **Main CI**: `.github/workflows/ci.yml` - Tests on Python 3.11-3.13, code quality, examples, explorer build, package build check
- **GitHub Pages**: `.github/workflows/pages.yml` - Deploy `explorer/` static UI
- **Dependencies**: `.github/workflows/dependencies.yml` - Weekly dependency updates and security scans

### Pre-commit Hooks
Install and use pre-commit hooks for local development:
- Install hooks: `uv run pre-commit install`
- Run manually: `uv run pre-commit run --all-files`
- Configuration: `.pre-commit-config.yaml`

### Installation
- Development install: `uv sync`
- Install with benchmarking tools: `uv sync --group benchmarking`
- Install all groups: `uv sync --all-groups`

PyTorch CPU wheels are configured in `pyproject.toml` via `[tool.uv.sources]` pointing at `https://download.pytorch.org/whl/cpu`.

### Running Examples
- Basic usage examples: `uv run python example_usage.py`
- MNIST benchmark: `uv run python mnist_optimizer_benchmark.py`
- Simple quadratic optimization: `uv run python minimalist_quadratic_optimization.py`

### Explorer UI
- Local dev: `cd explorer && npm install && npm run dev`
- Build for GitHub Pages: `cd explorer && npm run build:pages`
- Regenerate static JSON: `uv sync --group benchmarking && uv run python scripts/export_explorer_data.py --profile lite`
- Deploy workflow: `.github/workflows/pages.yml` (publishes `explorer/dist` on push to `main`)

## Architecture Overview

### Core Module Structure
The main implementation is in `muon_optimizer.py`, which contains:

**Core Functions:**
- `zeropower_via_newtonschulz5()`: Newton-Schulz orthogonalization algorithm using quintic iteration
- `muon_update()`: Core Muon update combining momentum and orthogonalization
- `adam_update()`: Standard Adam optimizer implementation

**Optimizer Classes:**
- `Muon`: Main distributed optimizer for multi-GPU training
- `SingleDeviceMuon`: Single-device variant for local training
- `MuonWithAuxAdam`: Hybrid optimizer combining Muon for 2D+ parameters and AdamW for others
- `SingleDeviceMuonWithAuxAdam`: Single-device hybrid optimizer

**Utility:**
- `create_muon_param_groups()`: Automatically separates model parameters for hybrid optimization

### Key Design Principles

**Parameter Type Handling:**
- **Muon optimization**: Applied to 2D+ matrix parameters (linear layers, conv weights)
- **AdamW fallback**: Used for 1D parameters (biases), embeddings, and output layers
- **Automatic reshaping**: 4D conv filters are reshaped to 2D for orthogonalization

**Orthogonalization Process:**
1. Update momentum buffer using exponential moving average
2. Apply Nesterov momentum if enabled
3. Orthogonalize update using Newton-Schulz iteration (converts to bfloat16)
4. Scale by matrix dimension factor
5. Reshape back to original parameter shape

**Distributed Training:**
- Parameters sorted by size for efficient distribution
- Uses `torch.distributed.all_gather()` for synchronization
- Automatic padding for even distribution across processes

### Important Implementation Details

**Numerical Stability:**
- All orthogonalization performed in bfloat16 for GPU efficiency
- Normalization by spectral norm with 1e-7 epsilon
- Quintic Newton-Schulz with optimized coefficients (3.4445, -4.7750, 2.0315)

**Memory Management:**
- Momentum buffers initialized as zeros_like(parameter)
- State dictionary tracks momentum_buffer for Muon, exp_avg/exp_avg_sq for Adam
- In-place operations used where possible

**Parameter Validation:**
- Learning rates must be non-negative
- Momentum in [0, 1) range
- Newton-Schulz steps must be positive integers
- All parameter groups require 'use_muon' boolean flag for hybrid optimizers

### Testing Strategy
The test suite (`muon_optimizer_test.py`) covers:
- Core function correctness (orthogonalization, momentum updates)
- All optimizer classes with various parameter configurations
- Error handling and edge cases (None gradients, invalid parameters)
- Integration testing with actual model training
- Distributed vs single-device behavior

### Example Usage Patterns
- Single device training: Use `SingleDeviceMuon` for simplicity
- Multi-GPU training: Use `Muon` with proper distributed initialization
- Mixed parameter types: Use hybrid optimizers with `create_muon_param_groups()`
- Learning rate scheduling: Standard PyTorch schedulers work with all optimizer classes

## Dependencies
Core: PyTorch >=2.7.1 (CPU index configured in pyproject.toml for uv)
Examples: rich (optional `[examples]` extra)
Benchmarking: dash, plotly, matplotlib, torchvision, seaborn
Development: pytest, black, isort, flake8, mypy

## CI testing
- Example tests in CI use `pytest example_usage_test.py -m "not slow"` for speed
- Full integration tests: `pytest example_usage_test.py -m slow -v`