# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Commands

### Testing
- Run all tests: `python -m pytest muon_optimizer_test.py -v`
- Run example tests: `python -m pytest example_usage_test.py -v`
- Run specific test class: `python -m pytest muon_optimizer_test.py::TestMuonOptimizer -v`
- Run single test method: `python -m pytest muon_optimizer_test.py::TestMuonOptimizer::test_initialization -v`

### Code Quality
The project includes development dependencies for code quality tools (in setup.py extras_require):
- Code formatting: `black --line-length=120 muon_optimizer.py`
- Import sorting: `isort --line-length=120 muon_optimizer.py`
- Linting: `flake8 muon_optimizer.py --max-line-length=120 --extend-ignore=E203,W503,E501`
- Type checking: `mypy muon_optimizer.py`

Note: The project uses a 120-character line length limit with E501 (line too long) errors ignored for flexibility.

### CI/CD Pipeline
The project uses GitHub Actions for continuous integration and deployment:
- **Main CI**: `.github/workflows/ci.yml` - Tests on Python 3.11-3.13, code quality, examples
- **Release**: `.github/workflows/release.yml` - Automated package publishing to PyPI
- **Dependencies**: `.github/workflows/dependencies.yml` - Weekly dependency updates and security scans

### Pre-commit Hooks
Install and use pre-commit hooks for local development:
- Install: `pip install pre-commit && pre-commit install`
- Run manually: `pre-commit run --all-files`
- Configuration: `.pre-commit-config.yaml`

### Installation
- Development install: `pip install -e .`
- Install with dev dependencies: `pip install -e .[dev]`
- Install dependencies from pyproject.toml: `pip install -r requirements.txt` (if generated) or directly from pyproject.toml

### Running Examples
- Basic usage examples: `python example_usage.py`
- MNIST benchmark: `python mnist_optimizer_benchmark.py`
- Simple quadratic optimization: `python minimalist_quadratic_optimization.py`

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
Core: PyTorch >=1.9.0 (>=2.7.1 in pyproject.toml)
Benchmarking: dash, plotly, matplotlib, torchvision, rich, seaborn
Development: pytest, black, isort, flake8, mypy