# Monorepo Template

A comprehensive monorepo template based on the Pynomaly architecture, implementing Clean Architecture with Domain-Driven Design principles.

## Architecture Overview

This template follows a **Clean Architecture** pattern with **Domain-Driven Design (DDD)** principles:

- **Hexagonal Architecture**: Clear separation between domain logic and infrastructure
- **Layered Architecture**: Domain → Application → Infrastructure → Presentation
- **Dependency Inversion**: High-level modules don't depend on low-level modules
- **Rich Domain Models**: Entities with business logic and validation

## Directory Structure

```
project-name/
├── src/
│   ├── packages/           # Core packages (clean architecture)
│   │   ├── core/          # Domain logic & business rules
│   │   ├── infrastructure/# Technical infrastructure  
│   │   ├── services/      # Application services
│   │   ├── api/           # REST API server
│   │   ├── cli/           # Command-line interface
│   │   ├── web/           # Web UI & dashboard
│   │   ├── enterprise/    # Enterprise features
│   │   ├── algorithms/    # ML algorithm adapters (optional)
│   │   ├── sdks/          # Client SDKs
│   │   ├── testing/       # Testing utilities
│   │   └── tools/         # Development tools
│   ├── apps/              # Standalone applications
│   ├── infrastructure/    # Deployment & infrastructure
│   ├── documentation/     # All documentation
│   ├── development_scripts/# Build & utility scripts
│   └── integration_tests/ # Test suites
├── build/                 # Build artifacts
├── deploy/                # Deployment configurations
├── docs/                  # Documentation
├── env/                   # Environment configurations
├── temp/                  # Temporary files
├── examples/              # Usage examples
├── tests/                 # Test suites
├── .github/               # GitHub workflows
├── .project-rules/        # Project rules and validation
├── scripts/               # Automation scripts
├── pyproject.toml         # Project configuration
├── README.md              # Project documentation
├── TODO.md                # Task tracking
└── CHANGELOG.md           # Version history
```

## Quick Start

1. **Clone the template**:
   ```bash
   git clone <template-repo> my-project
   cd my-project
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

## Package Structure

### Core Package (`src/packages/core/`)
- **Domain Layer**: Entities, value objects, domain services
- **Application Layer**: Use cases, DTOs, application services
- **Shared**: Common utilities and protocols

### Infrastructure Package (`src/packages/infrastructure/`)
- **Repositories**: Data access implementations
- **Adapters**: External service integrations
- **Configuration**: Settings and dependency injection

### Services Package (`src/packages/services/`)
- **Application Services**: High-level orchestration
- **Domain Services**: Business logic coordination

### API Package (`src/packages/api/`)
- **FastAPI Application**: REST API implementation
- **Endpoints**: API route handlers
- **Middleware**: Request/response processing

### CLI Package (`src/packages/cli/`)
- **Typer Application**: Command-line interface
- **Commands**: CLI command implementations
- **Utilities**: CLI helper functions

## Technology Stack

- **Python 3.11+**: Modern Python with type hints
- **FastAPI**: REST API framework
- **Typer**: CLI framework
- **Pydantic**: Data validation
- **SQLAlchemy**: ORM
- **pytest**: Testing framework
- **Ruff**: Linting and formatting
- **MyPy**: Static type checking

## Development Workflow

1. **Feature Development**: Create new features in appropriate packages
2. **Testing**: Write comprehensive tests for all layers
3. **Code Quality**: Use pre-commit hooks for quality checks
4. **Documentation**: Update docs and ADRs for architectural decisions

## Configuration

The template uses environment-based configuration with Pydantic settings:

```python
# src/packages/infrastructure/config/settings.py
from pydantic_settings import BaseSettings

class Settings(BaseSettings):
    app_name: str = "My App"
    debug: bool = False
    database_url: str = "sqlite:///./app.db"
    
    class Config:
        env_file = ".env"
```

## Extensibility

The template is designed for easy extension:

- **New Packages**: Add new packages to `src/packages/`
- **New Services**: Implement new services following the existing patterns
- **New Adapters**: Add adapters for external services
- **New Commands**: Add CLI commands to the CLI package

## License

MIT License - see LICENSE file for details