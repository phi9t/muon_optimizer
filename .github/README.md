# CI/CD Configuration

This directory contains the complete CI/CD pipeline configuration for the Muon Optimizer project.

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

### 2. Release Automation (`.github/workflows/release.yml`)
**Triggers**: Git tags (v*) and GitHub releases

**Process**:
1. **Pre-release Testing**: Full test suite validation
2. **Version Consistency**: Verify tag matches module version
3. **Package Building**: Create wheel and source distributions
4. **Test PyPI**: Upload to test PyPI for validation
5. **Production Release**: Publish to PyPI on GitHub release
6. **Post-release**: Verification and documentation updates

**Security**: Uses trusted publishing (OIDC) for PyPI uploads

### 3. Dependency Management (`.github/workflows/dependencies.yml`)
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
- Release process documentation

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

## 🚀 Deployment Pipeline

### Staging (Test PyPI)
1. Triggered on version tags (`v*`)
2. Full test suite execution
3. Package building and validation
4. Upload to test.pypi.org
5. Installation testing from test PyPI

### Production (PyPI)
1. Triggered on GitHub releases
2. Inherits all staging validations
3. Upload to pypi.org using trusted publishing
4. Automatic GitHub release asset creation
5. Post-deployment verification

### Version Management
- Semantic versioning (MAJOR.MINOR.PATCH)
- Automatic version consistency validation
- Git tag-based release triggers
- Changelog automation

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
- Release status → Release notes

## 🛠️ Local Development

### Quick Setup
```bash
# Clone and setup
git clone <repository>
cd muon_optimizer

# Install development dependencies
pip install -e .[dev]

# Install pre-commit hooks
pre-commit install

# Run tests
pytest -v

# Check code quality
pre-commit run --all-files
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