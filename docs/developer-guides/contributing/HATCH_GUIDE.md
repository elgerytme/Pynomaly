# Hatch Development Guide

🍞 **Breadcrumb:** 🏠 [Home](../../index.md) > 👨‍💻 [Developer Guides](../README.md) > 🤝 [Contributing](README.md) > 📦 Hatch Guide

---


This guide covers the complete migration of Pynomaly from Poetry to Hatch for modern Python packaging and development workflows.

## Overview

Pynomaly has migrated to **Hatch**, a modern Python project manager that provides:

- **PEP 621** compliant pyproject.toml configuration
- **Environment management** with matrix testing support
- **Git-based versioning** with semantic versioning
- **Optimized building** for wheel and source distributions
- **Cross-platform support** for development and deployment

## Quick Start

### Prerequisites

Ensure Hatch is installed:

```bash
# Install via pip (user installation recommended)
pip install --user hatch

# Or install via pipx (isolated installation)
pipx install hatch

# Verify installation
hatch --version
```

### Basic Commands

```bash
# Show project version (git-based)
hatch version

# List all environments
hatch env show

# Build the package
hatch build

# Install in development mode
hatch env run pip install -e .
```

## Environment Management

### Available Environments

| Environment | Purpose | Key Dependencies |
|-------------|---------|------------------|
| `default` | Basic development and testing | pytest, pytest-cov, pytest-asyncio |
| `test` | Comprehensive testing with matrix | All test deps + hypothesis, faker |
| `lint` | Code quality and type checking | ruff, black, isort, mypy |
| `docs` | Documentation building | mkdocs, mkdocs-material |
| `dev` | Development tools | pre-commit, tox, pip-tools |
| `prod` | Production deployment | All production dependencies |
| `cli` | CLI-specific testing | CLI dependencies + minimal ML |

### Environment Usage

```bash
# Create environments
hatch env create

# Run commands in specific environments
hatch env run test:run                    # Run tests
hatch env run lint:style                  # Check code style
hatch env run lint:fmt                    # Format code
hatch env run docs:serve                  # Serve documentation
hatch env run prod:serve-api              # Start production API

# Run in multiple environments (matrix testing)
hatch env run test:run                    # Runs in both py3.11 and py3.12

# Clean up environments
hatch env remove test
hatch env prune                           # Remove unused environments
```

## Development Workflows

### Testing

```bash
# Run tests in default environment
hatch env run test

# Run tests with coverage
hatch env run test-cov

# Run tests in parallel
hatch env run test:run-parallel

# Run specific test file
hatch env run test:run tests/test_specific.py

# Matrix testing (all Python versions)
hatch env run test:run
```

### Code Quality

```bash
# Check code style and typing
hatch env run lint:all

# Format code automatically
hatch env run lint:fmt

# Type checking only
hatch env run lint:typing

# Style checking only
hatch env run lint:style
```

### Documentation

```bash
# Build documentation
hatch env run docs:build

# Serve documentation locally
hatch env run docs:serve

# Access at http://localhost:8080
```

### Development Setup

```bash
# Set up development environment
hatch env run dev:setup

# Update pre-commit hooks
hatch env run dev:update

# Clean build artifacts
hatch env run dev:clean
```

## Building and Distribution

### Build Configuration

The build system is configured for optimal packaging:

```toml
[tool.hatch.build]
include = [
    "src/pynomaly/**/*.py",
    "src/pynomaly/**/*.pyi",
    "src/pynomaly/py.typed",
]
exclude = [
    "src/pynomaly/**/*_test.py",
    "src/pynomaly/**/test_*.py",
]
```

### Building

```bash
# Clean build (removes existing artifacts)
hatch build --clean

# Build specific targets
hatch build --target wheel
hatch build --target sdist

# Output: dist/pynomaly-{version}-py3-none-any.whl
#         dist/pynomaly-{version}.tar.gz
```

### Version Management

Hatch uses git-based versioning:

```bash
# Show current version
hatch version

# Version is automatically determined from git tags
# Format: {next_version}.dev{commits_since_tag}+g{git_hash}.d{date}
# Example: 0.1.dev335+g6121aa6.d20250625
```

## Production Deployment

### Production Environment

```bash
# Install production dependencies
hatch env run prod:pip install -e .[production]

# Start API server (development)
hatch env run prod:serve-api

# Start API server (production with workers)
hatch env run prod:serve-api-prod
```

### Deployment Configuration

The production environment includes:

- FastAPI + Uvicorn with worker processes
- Redis caching
- OpenTelemetry monitoring
- Prometheus metrics
- JWT authentication
- Circuit breakers and resilience

## Optional Dependencies

Hatch manages optional dependencies through extras:

### Core Extras

```bash
# Install specific functionality
pip install -e .[cli]                    # CLI interface
pip install -e .[api]                    # REST API
pip install -e .[minimal]                # Basic ML functionality
pip install -e .[server]                 # CLI + API + basic ML
```

### Infrastructure Extras

```bash
pip install -e .[auth]                   # Authentication
pip install -e .[caching]                # Redis caching  
pip install -e .[monitoring]             # OpenTelemetry + Prometheus
pip install -e .[infrastructure]         # Resilience patterns
```

### ML and Data Extras

```bash
pip install -e .[torch]                  # PyTorch backend
pip install -e .[tensorflow]             # TensorFlow backend
pip install -e .[jax]                    # JAX backend
pip install -e .[graph]                  # Graph neural networks
pip install -e .[automl]                 # AutoML capabilities
pip install -e .[explainability]         # SHAP/LIME integration
```

### Comprehensive Installations

```bash
pip install -e .[ml-all]                 # All ML backends
pip install -e .[data-all]               # All data processing
pip install -e .[production]             # Production deployment
pip install -e .[all]                    # Everything
```

## CI/CD Integration

### GitHub Actions

Update your workflows to use Hatch:

```yaml
- name: Set up Python
  uses: actions/setup-python@v4
  with:
    python-version: '3.11'

- name: Install Hatch
  run: pip install hatch

- name: Run tests
  run: hatch env run test:run

- name: Run linting
  run: hatch env run lint:all

- name: Build package
  run: hatch build
```

### Test Matrix

Hatch automatically handles Python version matrices:

```yaml
strategy:
  matrix:
    python-version: ['3.11', '3.12']

steps:
- name: Run tests for Python ${{ matrix.python-version }}
  run: hatch env run test:run
```

## Migration from Poetry

### Configuration Differences

| Poetry | Hatch | Notes |
|--------|-------|--------|
| `poetry install` | `hatch env create` | Environment creation |
| `poetry run pytest` | `hatch env run test` | Running commands |
| `poetry build` | `hatch build` | Building packages |
| `poetry shell` | `hatch shell` | Activating environment |
| `pyproject.toml [tool.poetry]` | `pyproject.toml [tool.hatch]` | Configuration section |

### Backup and Migration

The migration script automatically:

1. **Backs up** existing Poetry configuration to `backup_poetry_config/`
2. **Creates** new Hatch-based pyproject.toml
3. **Preserves** all existing tool configurations (black, mypy, etc.)
4. **Generates** utility scripts for version and environment management

### Utility Scripts

Additional helper scripts in `scripts/`:

```bash
# Version management
python scripts/hatch_version.py          # Show current version
python scripts/hatch_version.py 1.0.0    # Set version (for releases)

# Environment management  
python scripts/hatch_env.py              # List environments
python scripts/hatch_env.py create test  # Create environment
python scripts/hatch_env.py remove test  # Remove environment
```

## Best Practices

### Environment Isolation

```bash
# Use detached environments for tools that don't need project dependencies
hatch env run lint:style                 # Lint environment is detached

# Use project environments for testing
hatch env run test:run                   # Test environment includes project
```

### Development Workflow

1. **Start with environment setup:**
   ```bash
   hatch env run dev:setup
   ```

2. **Make changes and test:**
   ```bash
   hatch env run test:run
   hatch env run lint:fmt
   ```

3. **Build and verify:**
   ```bash
   hatch build --clean
   ```

4. **Run comprehensive checks:**
   ```bash
   hatch env run lint:all
   hatch env run test:run-cov
   ```

### Performance Tips

- **Use matrix environments** for cross-version testing
- **Leverage detached environments** for tools that don't need project deps
- **Cache environments** in CI/CD for faster builds
- **Use `--force-continue`** for running multiple environment commands

## Troubleshooting

### Common Issues

1. **Environment Creation Fails:**
   ```bash
   # Clean and recreate
   hatch env prune
   hatch env create
   ```

2. **Build Errors:**
   ```bash
   # Check pyproject.toml syntax
   hatch build --clean --verbose
   ```

3. **Version Issues:**
   ```bash
   # Ensure git tags are present
   git tag v0.1.0
   hatch version
   ```

4. **Dependency Conflicts:**
   ```bash
   # Use specific environment
   hatch env run --env test:run
   ```

### Debug Commands

```bash
# Verbose environment information
hatch env show --ascii

# Debug build process
hatch build --debug

# Check project configuration
hatch project metadata
```

## Advanced Configuration

### Custom Environments

Add custom environments to pyproject.toml:

```toml
[tool.hatch.envs.custom]
dependencies = ["custom-package>=1.0.0"]
scripts = {custom-cmd = "python custom_script.py"}
```

### Environment Variables

```bash
# Set environment variables
HATCH_ENV=test hatch env run run
export HATCH_ENV=prod
```

### Build Hooks

Customize build process:

```toml
[tool.hatch.build.hooks.custom]
# Custom build hook configuration
```

## Migration Benefits

The Hatch migration provides:

✅ **Modern packaging** with PEP 621 compliance  
✅ **Better dependency management** with optional extras  
✅ **Improved build performance** with optimized configurations  
✅ **Enhanced development experience** with environment isolation  
✅ **Git-based versioning** for automated version management  
✅ **Cross-platform support** with consistent behavior  
✅ **Production readiness** with deployment-focused environments  

---

For more information, see the [official Hatch documentation](https://hatch.pypa.io/).

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
