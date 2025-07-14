# Pynomaly Workspace Management

This document describes the workspace management tools and conventions for the Pynomaly monorepo.

## Overview

Pynomaly uses a **monorepo workspace architecture** with the following structure:

```
Pynomaly/
├── src/
│   ├── pynomaly/                 # Core library
│   └── packages/                 # Workspace packages
│       ├── algorithms/           # ML algorithm adapters
│       ├── infrastructure/       # External integrations
│       ├── services/            # Application services
│       ├── api/                 # FastAPI endpoints
│       ├── cli/                 # Command-line interface
│       ├── web/                 # Web UI
│       ├── data_transformation/ # Data processing
│       ├── data_profiling/      # Statistical analysis
│       ├── data_quality/        # Validation rules
│       ├── data_science/        # Advanced analytics
│       ├── mlops/               # MLOps platform
│       ├── sdk/                 # Client SDKs
│       ├── enterprise/          # Enterprise features
│       └── testing/             # Shared testing utilities
├── scripts/                     # Workspace management tools
├── tests/                       # Test suites
├── docs/                        # Documentation
└── workspace.json              # Workspace configuration
```

## Quick Start

### 1. Environment Setup

```bash
# Set up development environment
python scripts/dev-setup.py

# Activate virtual environment
source environments/.venv/bin/activate  # Linux/macOS
# or
environments\.venv\Scripts\activate      # Windows
```

### 2. Workspace Commands

```bash
# List all packages
python scripts/workspace.py list

# Install all packages
python scripts/workspace.py install all

# Run tests for all packages
python scripts/workspace.py test all

# Build all packages
python scripts/workspace.py build all

# Show dependency graph
python scripts/workspace.py deps --graph

# Show packages affected by changes
python scripts/workspace.py deps --affected package_name
```

### 3. Validation

```bash
# Validate workspace integrity
python scripts/workspace-validate.py

# Validate with strict warnings
python scripts/workspace-validate.py --fail-on-warnings
```

## Workspace Tools

### `scripts/workspace.py`

Main workspace management tool with commands:

- **`list`** - List workspace packages
- **`install`** - Install package dependencies
- **`test`** - Run package tests
- **`build`** - Build packages
- **`deps`** - Show dependency information
- **`config`** - Workspace configuration management

### `scripts/dev-setup.py`

Development environment setup tool:

- Sets up Python virtual environment
- Installs development dependencies
- Configures pre-commit hooks
- Sets up IDE configuration
- Validates environment

### `scripts/workspace-validate.py`

Workspace validation tool:

- Validates directory structure
- Checks package integrity
- Validates dependencies
- Checks configuration files
- Validates tools and setup

## Package Management

### Package Types

1. **Main Package** (`pynomaly`) - Core anomaly detection library
2. **Workspace Packages** - Specialized functionality packages
3. **Application Packages** - CLI, API, Web UI applications

### Dependency Rules

- **Domain packages** should not depend on infrastructure packages
- **Application packages** can depend on domain and infrastructure packages
- **Infrastructure packages** should only depend on domain packages
- **Circular dependencies** are not allowed

### Adding New Packages

1. Create package directory in `src/packages/`
2. Add `__init__.py` and `pyproject.toml`
3. Update `workspace.json` with package information
4. Add package to dependency graph
5. Run validation: `python scripts/workspace-validate.py`

## Development Workflow

### Daily Development

```bash
# Start development session
source environments/.venv/bin/activate

# Run affected tests when working on a package
python scripts/workspace.py deps --affected my_package
python scripts/workspace.py test affected_package

# Format and lint code
hatch run format
hatch run lint

# Run full test suite before commits
python scripts/workspace.py test all
```

### Package Development

```bash
# Work on specific package
cd src/packages/my_package

# Install package in development mode
python -m pip install -e .

# Run package-specific tests
python scripts/workspace.py test my_package

# Build package
python scripts/workspace.py build my_package
```

### Integration Testing

```bash
# Test packages that depend on your changes
python scripts/workspace.py deps --affected my_package
python scripts/workspace.py test all  # or specific affected packages

# Validate workspace integrity
python scripts/workspace-validate.py
```

## Configuration

### `workspace.json`

Central workspace configuration containing:

- Package definitions and metadata
- Dependency graph
- Build and test configurations
- Environment settings
- Tool configurations

### `pyproject.toml`

Main project configuration with:

- Project metadata
- Dependencies
- Build system configuration
- Tool settings (pytest, black, ruff, etc.)

### Package-Level Configuration

Each package can have its own `pyproject.toml` for:

- Package-specific dependencies
- Local build settings
- Package metadata

## Best Practices

### Package Design

1. **Single Responsibility** - Each package should have a clear, focused purpose
2. **Clean Dependencies** - Follow dependency inversion principle
3. **Interface Segregation** - Use protocols for abstractions
4. **Minimal Coupling** - Minimize dependencies between packages

### Development Practices

1. **Test-Driven Development** - Write tests for new functionality
2. **Continuous Integration** - Run tests on all affected packages
3. **Code Quality** - Use formatters, linters, and type checkers
4. **Documentation** - Document public APIs and architectural decisions

### Workspace Management

1. **Regular Validation** - Run workspace validation regularly
2. **Dependency Audits** - Review dependencies for circular references
3. **Build Optimization** - Use incremental builds and affected testing
4. **Environment Consistency** - Use standardized development setup

## Troubleshooting

### Common Issues

1. **Import Errors**
   ```bash
   # Reinstall packages in development mode
   python scripts/workspace.py install all
   ```

2. **Test Failures**
   ```bash
   # Run tests with verbose output
   python scripts/workspace.py test package_name --args "-v"
   ```

3. **Dependency Issues**
   ```bash
   # Check dependency graph
   python scripts/workspace.py deps --graph
   
   # Validate workspace
   python scripts/workspace-validate.py
   ```

4. **Environment Issues**
   ```bash
   # Reset development environment
   rm -rf environments/.venv
   python scripts/dev-setup.py
   ```

### Getting Help

- **Documentation**: `docs/developer-guides/`
- **Issues**: GitHub Issues
- **Discussions**: GitHub Discussions
- **Workspace Validation**: `python scripts/workspace-validate.py`

## Advanced Usage

### Custom Scripts

Add custom workspace scripts to `scripts/` directory following the pattern:

```python
#!/usr/bin/env python3
"""Custom workspace script."""

from pathlib import Path
import sys

# Add workspace management utilities
sys.path.insert(0, str(Path(__file__).parent))
from workspace import WorkspaceManager

def main():
    workspace = WorkspaceManager(Path.cwd())
    # Custom logic here
    
if __name__ == "__main__":
    main()
```

### CI/CD Integration

```yaml
# Example GitHub Actions workflow
name: Workspace CI
on: [push, pull_request]

jobs:
  validate:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v3
      - name: Setup Python
        uses: actions/setup-python@v4
        with:
          python-version: '3.11'
      - name: Setup Development Environment
        run: python scripts/dev-setup.py --minimal
      - name: Validate Workspace
        run: python scripts/workspace-validate.py
      - name: Run Tests
        run: python scripts/workspace.py test all
```

### Performance Optimization

For large workspaces:

1. Use **affected testing** instead of running all tests
2. Enable **parallel test execution** with pytest-xdist
3. Use **build caching** with Buck2 or similar tools
4. Implement **incremental builds** for packages

---

For more detailed information, see the [Developer Guides](docs/developer-guides/) directory.