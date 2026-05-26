# Contributing to Muon Optimizer

Thank you for your interest in contributing to the Muon Optimizer project! This guide will help you get started.

## 🚀 Quick Start

### Development Setup

1. **Fork and clone the repository**:
   ```bash
   git clone https://github.com/YOUR_USERNAME/muon_optimizer.git
   cd muon_optimizer
   ```

2. **Set up development environment**:
   ```bash
   # Install uv: https://docs.astral.sh/uv/getting-started/installation/
   curl -LsSf https://astral.sh/uv/install.sh | sh

   # Install dependencies (CPU PyTorch is configured in pyproject.toml)
   uv sync

   # Install pre-commit hooks
   uv run pre-commit install
   ```

3. **Verify your setup**:
   ```bash
   uv run pytest muon_optimizer_test.py -v
   uv run pytest example_usage_test.py -m "not slow" -v

   uv run black --check muon_optimizer.py
   uv run flake8 muon_optimizer.py
   uv run mypy muon_optimizer.py
   ```

## 🧪 Testing

### Running Tests

```bash
# All tests
pytest -v

# Specific test files
pytest muon_optimizer_test.py -v
pytest example_usage_test.py -v

# With coverage
pytest --cov=muon_optimizer --cov-report=term-missing

# Specific test class or method
pytest muon_optimizer_test.py::TestMuonOptimizer::test_initialization -v
```

### Writing Tests

- Add tests for new features in `muon_optimizer_test.py`
- Add example tests in `example_usage_test.py`
- Follow existing test patterns and naming conventions
- Ensure good test coverage for edge cases

## 🎨 Code Style

We use several tools to maintain code quality:

### Formatting and Linting
```bash
# Auto-format code
black muon_optimizer.py example_usage.py

# Sort imports
isort muon_optimizer.py example_usage.py

# Check linting
flake8 muon_optimizer.py example_usage.py

# Type checking
mypy muon_optimizer.py
```

### Pre-commit Hooks
Pre-commit hooks will automatically run these checks:
```bash
# Install hooks
pre-commit install

# Run manually
pre-commit run --all-files
```

## 📝 Contribution Guidelines

### Types of Contributions

1. **🐛 Bug Fixes**: Fix issues and improve reliability
2. **✨ New Features**: Add new optimizer variants or functionality  
3. **📚 Documentation**: Improve docs, examples, and tutorials
4. **🚀 Performance**: Optimize code for speed or memory usage
5. **🧪 Testing**: Add or improve test coverage

### Submission Process

1. **Create an Issue**: For significant changes, create an issue first to discuss the approach
2. **Create a Branch**: Use descriptive branch names like `feature/new-optimizer` or `fix/gradient-bug`
3. **Make Changes**: Follow the code style and add tests
4. **Run Tests**: Ensure all tests pass locally
5. **Submit PR**: Create a pull request with a clear description

### Pull Request Guidelines

**PR Title Format**:
- `feat: add new feature description`
- `fix: resolve specific bug`
- `docs: improve documentation`
- `test: add test coverage`
- `refactor: improve code structure`

**PR Description Should Include**:
- Clear description of changes
- Motivation and context
- Testing performed
- Breaking changes (if any)
- Related issue references

**Checklist for PRs**:
- [ ] Tests pass locally (`uv run pytest muon_optimizer_test.py -v` and `uv run pytest example_usage_test.py -m "not slow" -v`)
- [ ] Code is formatted (`uv run black`, `uv run isort`)
- [ ] Linting passes (`uv run flake8`, `uv run mypy`)
- [ ] Documentation updated if needed

## 🏗️ Architecture Guidelines

### Core Principles

1. **Type Safety**: Use type hints throughout
2. **Error Handling**: Provide clear error messages
3. **Documentation**: Document all public APIs
4. **Performance**: Consider GPU/CPU efficiency
5. **Compatibility**: Maintain backwards compatibility when possible

### Code Organization

- `muon_optimizer.py`: Core optimizer implementations
- `example_usage.py`: Usage examples and demonstrations
- `*_test.py`: Test files
- `*.md`: Documentation files

### Adding New Features

1. **Core Functions**: Add to `muon_optimizer.py`
2. **Examples**: Add usage examples to `example_usage.py`
3. **Tests**: Add comprehensive tests
4. **Documentation**: Update README and docstrings

## 🔒 Security

### Security Considerations
- Never commit secrets or API keys
- Be careful with `eval()`, `exec()`, and similar functions
- Validate all user inputs
- Use secure dependencies

### Reporting Security Issues
For security vulnerabilities, please email [maintainer] instead of creating public issues.

## ❓ Getting Help

### Resources
- 📖 [README](README.md): Basic usage and API reference
- 🤖 [CLAUDE.md](CLAUDE.md): AI assistant guidance
- 🐛 [Issues](https://github.com/phi9t/muon_optimizer/issues): Bug reports and feature requests
- 💬 [Discussions](https://github.com/phi9t/muon_optimizer/discussions): General questions

### Support Channels
- **GitHub Issues**: Bug reports and feature requests
- **GitHub Discussions**: Questions and community support
- **Code Reviews**: Feedback on pull requests

## 🙏 Recognition

Contributors will be recognized in:
- GitHub contributors list
- README acknowledgments

---

Thank you for contributing to Muon Optimizer! Your help makes this project better for everyone. 🚀