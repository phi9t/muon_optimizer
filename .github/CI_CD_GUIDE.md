# CI/CD Configuration Guide

This guide provides comprehensive documentation for the CI/CD pipeline configuration in the Muon Optimizer project.

## 🔄 Workflows

### 1. Main CI Pipeline (`.github/workflows/ci.yml`)
**Triggers**: Push/PR to `main` or `develop` branches, weekly schedule

**Jobs**:
- **Test Matrix**: Python 3.11, 3.12, 3.13 on Ubuntu
- **Code Quality**: Black, isort, flake8, mypy, bandit security checks
- **Examples**: Test quadratic optimization and MNIST benchmark examples
- **Build**: Package building and installation testing
- **Documentation**: README and CLAUDE.md validation

**Key Features**:
- Comprehensive test coverage with pytest
- Code coverage reporting via Codecov
- Security scanning with bandit
- Example validation with timeout protection
- Multi-Python version compatibility testing

### 2. Dependency Management (`.github/workflows/dependencies.yml`)
**Triggers**: Weekly schedule, manual dispatch

**Features**:
- Automated dependency updates using `uv`
- Security vulnerability scanning
- Test validation with updated dependencies
- Automatic PR creation for updates
- Multi-tool security analysis (safety, bandit, semgrep)

## 🔧 Configuration Files

### Pre-commit Hooks (`.pre-commit-config.yaml`)
**Local Development Quality Gates**:
- Code formatting (black)
- Import sorting (isort)
- Linting (flake8)
- Type checking (mypy)
- Security scanning (bandit)
- Basic security validation
- Test execution on push

**Installation**:
```bash
pip install pre-commit
pre-commit install
```

### Issue Templates
- **Bug Report**: Structured bug reporting with environment info
- **Feature Request**: Feature proposal template with impact analysis

### Contributing Guide (`CONTRIBUTING.md`)
Complete development workflow documentation including:
- Development setup instructions
- Testing procedures
- Code quality requirements
- Contribution guidelines

## 📊 Quality Gates

### Required Checks
All PRs must pass:
- ✅ Tests on all supported Python versions
- ✅ Code formatting (black)
- ✅ Import sorting (isort)
- ✅ Linting (flake8)
- ✅ Type checking (mypy)
- ✅ Security scanning (bandit)
- ✅ Example validation

### Coverage Requirements
- Core functionality: >90% coverage
- Test coverage reporting via Codecov
- Coverage reports in CI artifacts

### Security Standards
- Automated dependency vulnerability scanning
- Code security analysis with bandit and semgrep
- No hardcoded secrets or credentials
- Regular security audits via scheduled workflows

## 📦 Package build check

The CI **build** job runs `uv build` and `twine check` to verify the project still packages correctly. This repo is not published to PyPI; the check is only for installability validation.

## 📈 Monitoring and Observability

### Build Metrics
- Test execution time tracking
- Coverage trend monitoring
- Dependency update frequency
- Security scan results

### Alert Channels
- Failed builds → GitHub notifications
- Security vulnerabilities → Automated issues
- Dependency updates → PR notifications

## 🛠️ Local Development

### Quick Setup
```bash
# Clone and setup
git clone <repository>
cd muon_optimizer

# Install development dependencies
uv sync

# Install pre-commit hooks
uv run pre-commit install

# Run tests
uv run pytest muon_optimizer_test.py -v
uv run pytest example_usage_test.py -m "not slow" -v

# Check code quality
uv run pre-commit run --all-files
```

### IDE Integration
The CI configuration is compatible with:
- VS Code with Python extension
- PyCharm Professional
- Vim/Neovim with appropriate plugins
- Any IDE supporting pytest and pre-commit

## 🔍 Troubleshooting

### Common Issues
1. **Test Failures**: Check Python version compatibility
2. **Formatting Issues**: Run `black` and `isort` locally
3. **Import Errors**: Verify development install (`pip install -e .[dev]`)
4. **Security Alerts**: Review bandit reports in CI artifacts

### Debug Commands
```bash
# Run specific test
pytest muon_optimizer_test.py::TestName -v

# Check formatting
black --check --diff muon_optimizer.py

# Lint check
flake8 muon_optimizer.py --max-line-length=88

# Type check
mypy muon_optimizer.py --ignore-missing-imports
```

---

This CI/CD setup ensures high code quality, comprehensive testing, and secure automated deployments while maintaining developer productivity and project reliability.