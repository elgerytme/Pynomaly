# Data Domain Package

Core data package for data management functionality

## Overview

This package contains the domain logic for data functionality following Domain-Driven Design (DDD) principles.

## Structure

```
data/
├── core/
│   ├── domain/              # Domain layer
│   │   ├── entities/        # Domain entities
│   │   ├── services/        # Domain services
│   │   ├── value_objects/   # Value objects
│   │   ├── repositories/    # Repository interfaces
│   │   └── exceptions/      # Domain exceptions
│   ├── application/         # Application layer
│   │   ├── services/        # Application services
│   │   └── use_cases/       # Use cases
│   └── dto/                 # Data transfer objects
├── infrastructure/          # Infrastructure layer
│   ├── adapters/           # External adapters
│   ├── persistence/        # Data persistence
│   └── external/           # External services
├── interfaces/             # Interface layer
│   ├── api/               # REST API endpoints
│   ├── cli/               # Command-line interface
│   ├── web/               # Web interface
│   └── python_sdk/        # Python SDK
├── tests/                  # Test suite
│   ├── unit/              # Unit tests
│   ├── integration/       # Integration tests
│   └── e2e/               # End-to-end tests
└── docs/                   # Documentation
```

## Domain Boundaries

This package follows strict domain boundaries:

### Allowed Concepts
- Data-specific business logic
- Domain entities and value objects
- Domain services and repositories
- Use cases and application services

### Prohibited Concepts
- Generic software infrastructure (belongs in `software/` package)
- Other domain concepts (belongs in respective domain packages)
- Cross-domain references (use interfaces and dependency injection)

## Installation

```bash
pip install data
```

## Development

### Setup
```bash
# Clone repository
git clone <repository-url>
cd <repository-name>

# Install development dependencies
pip install -e ".[dev]"

# Install pre-commit hooks
python scripts/install_domain_hooks.py
```

### Testing
```bash
# Run tests
pytest

# Run tests with coverage
pytest --cov=src --cov-report=html

# Run domain boundary validation
python scripts/domain_boundary_validator.py
```

### Code Quality
```bash
# Format code
ruff format src/ tests/

# Lint code
ruff check src/ tests/

# Type checking
mypy src/
```

## Architecture

This package follows Clean Architecture principles:

1. **Domain Layer**: Core business logic
2. **Application Layer**: Use cases and application services
3. **Infrastructure Layer**: External concerns
4. **Interface Layer**: User interfaces

## Domain Compliance

This package maintains strict domain boundary compliance:

- **Validation**: Automated domain boundary validation
- **Enforcement**: Pre-commit hooks and CI/CD integration
- **Monitoring**: Continuous compliance monitoring
- **Documentation**: Clear domain boundary rules

## Contributing

1. Follow domain boundary rules
2. Add comprehensive tests
3. Update documentation
4. Validate domain compliance
5. Submit pull request

## License

MIT License - see LICENSE file for details.
