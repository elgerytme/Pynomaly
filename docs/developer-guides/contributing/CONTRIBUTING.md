# Contributing to Pynomaly

🍞 **Breadcrumb:** 🏠 [Home](../../index.md) > 👨‍💻 [Developer Guides](../README.md) > 🤝 [Contributing](README.md) > 📄 Contributing

---

Thank you for your interest in contributing to Pynomaly! This comprehensive guide will help you get started with contributing to our state-of-the-art anomaly detection library.

## 🌟 Welcome Contributors!

We welcome contributions of all kinds:
- 🐛 Bug reports and fixes
- 🚀 New features and algorithms
- 📚 Documentation improvements
- 🧪 Tests and benchmarks
- 💡 Performance optimizations
- 🎨 Examples and tutorials

## Code of Conduct

By participating in this project, you agree to abide by our Code of Conduct. Please read it before contributing.

## How to Contribute

### Reporting Issues

- Check existing issues before creating a new one
- Use issue templates when available
- Provide clear descriptions and steps to reproduce
- Include system information and error messages

### Suggesting Features

- Open a discussion before implementing major features
- Explain the use case and benefits
- Consider the impact on existing functionality
- Provide examples if possible

### Pull Requests

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Make your changes
4. Add tests for new functionality
5. Ensure all tests pass
6. Update documentation as needed
7. Commit with clear messages
8. Push to your fork
9. Open a pull request

## 🚀 Quick Start

### 1. Set Up Development Environment

```bash
# Clone the repository
git clone https://github.com/pynomaly/pynomaly.git
cd pynomaly

# Set up virtual environment (using our preferred structure)
mkdir -p environments
python -m venv environments/.venv
source environments/.venv/bin/activate  # On Windows: environments\.venv\Scripts\activate

# Install development dependencies
pip install -e ".[dev,test,docs]"

# Install pre-commit hooks
pre-commit install

# Verify installation
python -c "import pynomaly; print('✅ Pynomaly installed successfully!')"
```

### 2. Run Tests

```bash
# Run all tests
pytest

# Run with coverage
pytest --cov=src/pynomaly --cov-report=html

# Run specific test categories
pytest tests/unit/          # Unit tests
pytest tests/integration/   # Integration tests
pytest tests/performance/   # Performance tests
```

### 3. Code Quality Checks

```bash
# Format code
black src/ tests/
isort src/ tests/

# Type checking
mypy src/pynomaly

# Linting
flake8 src/ tests/
pylint src/pynomaly

# All-in-one quality check
pre-commit run --all-files
```

### Running Tests

```bash
# Run all tests
pytest

# Run with coverage
pytest --cov=pynomaly

# Run specific test file
pytest tests/domain/test_entities.py

# Run with verbose output
pytest -v
```

### Code Style

We use several tools to maintain code quality:

```bash
# Format code with black
black src tests

# Sort imports with isort
isort src tests

# Type checking with mypy
mypy src

# Linting with ruff
ruff src tests

# Security checks with bandit
bandit -r src

# All checks
make lint
```

### Documentation

```bash
# Build documentation
mkdocs build

# Serve documentation locally
mkdocs serve

# Build API documentation
make apidoc
```

## Architecture Guidelines

### Domain Layer
- Keep domain logic pure and framework-agnostic
- Use value objects for domain concepts
- Ensure entities are self-validating
- Avoid external dependencies

### Application Layer
- Implement use cases as single-purpose classes
- Use DTOs for data transfer
- Keep application services thin
- Handle orchestration and transactions

### Infrastructure Layer
- Implement adapters for external services
- Use protocols for interfaces
- Keep infrastructure details isolated
- Ensure adapters are replaceable

### Presentation Layer
- Keep controllers/endpoints thin
- Use appropriate serialization
- Handle HTTP concerns only
- Delegate business logic to use cases

## Testing Guidelines

### Test Structure
- Mirror the source code structure
- One test file per source file
- Group related tests in classes
- Use descriptive test names

### Test Types
- **Unit Tests**: Test individual components
- **Integration Tests**: Test component interactions
- **End-to-End Tests**: Test complete workflows
- **Performance Tests**: Test efficiency

### Test Best Practices
- Use fixtures for common setup
- Keep tests independent
- Test edge cases
- Mock external dependencies
- Aim for high coverage

## Commit Guidelines

### Commit Messages

Follow the conventional commits specification:

```
<type>(<scope>): <subject>

<body>

<footer>
```

Types:
- `feat`: New feature
- `fix`: Bug fix
- `docs`: Documentation changes
- `style`: Code style changes
- `refactor`: Code refactoring
- `test`: Test additions/changes
- `chore`: Build/tooling changes

### Examples

```
feat(domain): add confidence intervals to anomaly scores

- Add ConfidenceInterval value object
- Update AnomalyScore to support intervals
- Add validation for interval bounds

Closes #123
```

## Release Process

1. Update version in `pyproject.toml`
2. Update CHANGELOG.md
3. Create a release branch
4. Run full test suite
5. Build and test package
6. Create pull request
7. Merge after approval
8. Tag release
9. Deploy to PyPI

## Getting Help

- Check documentation first
- Search existing issues
- Ask in discussions
- Join our community chat
- Email maintainers

## Recognition

Contributors will be recognized in:
- CONTRIBUTORS.md file
- Release notes
- Project documentation
- GitHub contributors page

Thank you for contributing to Pynomaly!

---

## 🔗 **Related Documentation**

### **Development**
- **[Contributing Guidelines](../contributing/CONTRIBUTING.md)** - How to contribute
- **[Development Setup](../contributing/README.md)** - Local development environment
- **[Architecture Overview](../architecture/overview.md)** - System design
- **[Implementation Guide](../contributing/IMPLEMENTATION_GUIDE.md)** - Coding standards

### **API Integration**
- **[REST API](../api-integration/rest-api.md)** - HTTP API reference
- **[Python SDK](../api-integration/python-sdk.md)** - Python client library
- **[CLI Reference](../api-integration/cli.md)** - Command-line interface
- **[Authentication](../api-integration/authentication.md)** - Security and auth

### **User Documentation**
- **[User Guides](../../user-guides/README.md)** - Feature usage guides
- **[Getting Started](../../getting-started/README.md)** - Installation and setup
- **[Examples](../../examples/README.md)** - Real-world use cases

### **Deployment**
- **[Production Deployment](../../deployment/README.md)** - Production deployment
- **[Security Setup](../../deployment/SECURITY.md)** - Security configuration
- **[Monitoring](../../user-guides/basic-usage/monitoring.md)** - System observability

---

## 🆘 **Getting Help**

- **[Development Troubleshooting](../contributing/troubleshooting/)** - Development issues
- **[GitHub Issues](https://github.com/your-org/pynomaly/issues)** - Report bugs
- **[Contributing Guidelines](../contributing/CONTRIBUTING.md)** - Contribution process
