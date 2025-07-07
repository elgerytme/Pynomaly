# Development Setup Guide

## Environment Overview

This project uses a **clean architecture** approach with strict dependency management. Development environments are organized in the `environments/` directory with dot-prefix naming.

## Prerequisites

### System Requirements
- **Python 3.11+** (currently using Python 3.12.3)
- **Node.js 16+** for web UI dependencies
- **Git** for version control

### Environment Constraints
- **Externally managed Python environment** (Ubuntu/WSL2)
- Package installation requires virtual environments or user-level installs
- Poetry configuration needs version specification fixes

## Quick Start

### 1. Environment Setup

```bash
# Create main development environment
mkdir -p environments/.venv
python3 -m venv environments/.venv

# Activate environment
source environments/.venv/bin/activate

# Upgrade pip and install build tools
pip install --upgrade pip setuptools wheel
```

### 2. Core Dependencies

```bash
# Install core dependencies from pyproject.toml
pip install pyod>=2.0.5 numpy>=1.26.0 pandas>=2.2.3
pip install pydantic>=2.10.4 structlog>=24.4.0
pip install dependency-injector>=4.42.0 networkx>=3.0

# Install web framework dependencies
pip install fastapi uvicorn httpx

# Install testing dependencies
pip install pytest pytest-asyncio hypothesis

# Install optional ML dependencies (if available)
pip install shap lime  # May fail in externally managed environments
```

### 3. Web UI Dependencies

```bash
# Install Node.js dependencies for PWA
npm install htmx.org d3 echarts tailwindcss
```

### 4. Development Tools

```bash
# Install development tools
pip install mypy black ruff pre-commit

# Install code quality tools
pip install bandit safety
```

## Development Environments

### Environment Structure
```
environments/
├── README.md                    # Environment documentation  
├── .venv/                      # Main development environment
├── .test-env/                  # Testing environment
├── .prod-env/                  # Production simulation
└── .benchmark-env/             # Performance testing
```

### Environment Types

#### 1. Development Environment (`.venv`)
- **Purpose**: Main development work
- **Python**: 3.11+
- **Dependencies**: All core + development tools
- **Usage**: `source environments/.venv/bin/activate`

#### 2. Testing Environment (`.test-env`)
- **Purpose**: Isolated testing
- **Python**: Same as development
- **Dependencies**: Core + testing tools only
- **Usage**: `source environments/.test-env/bin/activate`

#### 3. Production Environment (`.prod-env`)
- **Purpose**: Production simulation
- **Python**: Production version
- **Dependencies**: Core dependencies only
- **Usage**: `source environments/.prod-env/bin/activate`

## Development Workflow

### 1. Activate Environment
```bash
source environments/.venv/bin/activate
```

### 2. Install Dependencies
```bash
# For development
pip install -e .

# For testing
pip install -e ".[testing]"

# For web development
pip install -e ".[web]"
```

### 3. Run Development Server
```bash
# FastAPI server
PYTHONPATH="src" uvicorn pynomaly.presentation.api:app --reload

# Web UI server  
PYTHONPATH="src" python scripts/run/run_web_app.py

# CLI interface
PYTHONPATH="src" python -m pynomaly.presentation.cli
```

### 4. Run Tests
```bash
# Unit tests
PYTHONPATH="src" pytest tests/unit/

# Integration tests
PYTHONPATH="src" pytest tests/integration/

# All tests with coverage
PYTHONPATH="src" pytest tests/ --cov=pynomaly --cov-report=html
```

### 5. Code Quality
```bash
# Type checking
PYTHONPATH="src" mypy src/pynomaly/

# Linting
ruff check src/ tests/

# Formatting
black src/ tests/

# Security scanning
bandit -r src/
```

## Configuration

### Environment Variables
```bash
# Core configuration
export PYNOMALY_ENVIRONMENT="development"
export PYNOMALY_LOG_LEVEL="DEBUG"
export PYNOMALY_CACHE_ENABLED="true"

# Database configuration
export PYNOMALY_DATABASE_URL="sqlite:///storage/dev.db"

# Web configuration
export PYNOMALY_WEB_HOST="localhost"
export PYNOMALY_WEB_PORT="8000"
```

### pyproject.toml Configuration

The project uses Hatch as the build backend, but Poetry commands fail due to missing version specification. To fix:

```toml
[tool.poetry]
name = "pynomaly"
version = "0.1.0"  # Add this line
description = "State-of-the-art Python anomaly detection package"
```

## Architecture Compliance

### Clean Architecture Layers
1. **Domain**: Pure Python entities (no external dependencies)
2. **Application**: Use cases and DTOs
3. **Infrastructure**: External integrations and adapters
4. **Presentation**: FastAPI, CLI, SDK, PWA

### Development Rules
- ✅ Domain layer uses only Python standard library
- ✅ External dependencies in infrastructure layer only
- ✅ Dependency injection for all external services
- ✅ Test coverage >90% for domain/application layers

## Troubleshooting

### Common Issues

#### 1. Package Installation Fails
```bash
# Error: externally-managed-environment
# Solution: Use virtual environment
python3 -m venv environments/.venv
source environments/.venv/bin/activate
pip install package_name
```

#### 2. Import Errors
```bash
# Error: ModuleNotFoundError
# Solution: Set PYTHONPATH
export PYTHONPATH="$(pwd)/src"
python script.py
```

#### 3. Poetry Configuration Invalid
```bash
# Error: Either [project.version] or [tool.poetry.version] is required
# Solution: Add version to pyproject.toml
[tool.poetry]
version = "0.1.0"
```

#### 4. Missing Virtual Environment Package
```bash
# Error: ensurepip is not available
# Solution: Install python3-venv
sudo apt install python3.12-venv
```

### Validation Commands

```bash
# Validate environment setup
python scripts/validation/validate_environment_organization.py

# Validate file organization  
python scripts/validation/validate_file_organization.py

# Health check
python scripts/testing/test_health_check.py

# Run current environment tests
./scripts/testing/test-current.sh
```

## Next Steps

1. **Create additional environments** as needed
2. **Set up pre-commit hooks** for code quality
3. **Configure CI/CD pipeline** with GitHub Actions
4. **Implement dependency scanning** for security
5. **Add performance monitoring** for benchmarks

---

**Last Updated**: 2025-01-07  
**Environment**: WSL2 Ubuntu with Python 3.12.3  
**Architecture**: Clean Architecture + DDD + Hexagonal