# Python Package Template

A comprehensive Python package template with modern tooling and best practices.

## Features

- **Modern Python**: Python 3.11+ with type hints
- **Build System**: Hatch for modern Python packaging
- **Code Quality**: Ruff, MyPy, pre-commit hooks
- **Testing**: pytest with coverage and fixtures
- **Documentation**: Sphinx with auto-generated docs
- **CI/CD**: GitHub Actions workflows
- **Security**: Bandit, safety, and vulnerability scanning

## Directory Structure

```
my-package/
├── build/                 # Build artifacts
├── deploy/                # Deployment configurations
├── docs/                  # Documentation
├── env/                   # Environment configurations
├── temp/                  # Temporary files
├── src/                   # Source code
│   └── my_package/
│       ├── __init__.py
│       ├── core/          # Core functionality
│       ├── utils/         # Utilities
│       └── cli/           # Command-line interface
├── pkg/                   # Package metadata
├── examples/              # Usage examples
├── tests/                 # Test suites
├── .github/               # GitHub workflows
├── pyproject.toml         # Project configuration
├── README.md              # Project documentation
├── TODO.md                # Task tracking
└── CHANGELOG.md           # Version history
```

## Quick Start

1. **Clone the template**:
   ```bash
   git clone <template-repo> my-package
   cd my-package
   ```

2. **Initialize the project**:
   ```bash
   ./scripts/init.sh
   ```

3. **Install dependencies**:
   ```bash
   pip install -e .
   ```

4. **Run tests**:
   ```bash
   pytest
   ```

5. **Build package**:
   ```bash
   hatch build
   ```

## Development

### Setup Development Environment

```bash
# Create virtual environment
python -m venv env
source env/bin/activate  # On Windows: env\Scripts\activate

# Install development dependencies
pip install -e ".[dev,test,docs]"

# Install pre-commit hooks
pre-commit install
```

### Code Quality

```bash
# Format code
ruff format

# Lint code
ruff check

# Type checking
mypy src/

# Run all quality checks
hatch run quality
```

### Testing

```bash
# Run tests
pytest

# Run tests with coverage
pytest --cov=src/my_package

# Run specific test file
pytest tests/test_core.py

# Run tests with specific markers
pytest -m "unit"
```

### Documentation

```bash
# Build documentation
cd docs
make html

# Serve documentation locally
python -m http.server 8000 -d _build/html/
```

### Publishing

```bash
# Build package
hatch build

# Publish to PyPI
hatch publish

# Publish to test PyPI
hatch publish -r test
```

## Configuration

The template uses modern Python packaging with `pyproject.toml`:

- **Build System**: Hatch
- **Dependencies**: Defined in `pyproject.toml`
- **Scripts**: Entry points for CLI commands
- **Development Tools**: Ruff, MyPy, pytest configuration

## Best Practices

1. **Code Organization**: Follow PEP 8 and use meaningful module names
2. **Type Hints**: Use type hints for all public APIs
3. **Documentation**: Document all public functions and classes
4. **Testing**: Aim for >80% test coverage
5. **Versioning**: Use semantic versioning (SemVer)
6. **Security**: Regularly scan for vulnerabilities

## License

MIT License - see LICENSE file for details