# Pynomaly Python SDK Package

A comprehensive Python SDK package for the Pynomaly anomaly detection platform, following clean architecture principles.

## Architecture Overview

This package implements a clean architecture pattern with clear separation of concerns:

```
python_sdk/
├── domain/                  # Core business logic
│   ├── entities/           # Business entities
│   ├── value_objects/      # Immutable value objects
│   ├── repositories/       # Repository interfaces
│   ├── services/          # Domain services
│   └── exceptions/        # Domain-specific exceptions
├── application/            # Application layer
│   ├── services/          # Application services
│   ├── use_cases/         # Use case implementations
│   └── dto/               # Data Transfer Objects
├── infrastructure/        # External concerns
│   ├── adapters/          # External service adapters
│   ├── persistence/       # Data persistence
│   ├── external/          # External integrations
│   └── logging/           # Logging infrastructure
├── presentation/          # Presentation layer
│   ├── cli/               # Command-line interface
│   ├── api/               # REST API interface
│   └── web/               # Web interface components
├── tests/                 # Test suite
│   ├── unit/              # Unit tests
│   ├── integration/       # Integration tests
│   ├── e2e/               # End-to-end tests
│   └── fixtures/          # Test fixtures and data
└── docs/                  # Documentation
    ├── api/               # API documentation
    ├── examples/          # Usage examples
    └── guides/            # User guides
```

## Layer Responsibilities

### Domain Layer
- **Entities**: Core business objects with identity and lifecycle
- **Value Objects**: Immutable objects representing descriptive aspects
- **Repositories**: Interfaces for data access abstraction
- **Services**: Domain logic that doesn't belong to entities
- **Exceptions**: Domain-specific error handling

### Application Layer
- **Services**: Application-specific business logic
- **Use Cases**: Orchestration of domain services and entities
- **DTOs**: Data transfer objects for external communication

### Infrastructure Layer
- **Adapters**: Implementations of domain repository interfaces
- **Persistence**: Database and storage implementations
- **External**: Third-party service integrations
- **Logging**: Centralized logging configuration

### Presentation Layer
- **CLI**: Command-line interface for the SDK
- **API**: REST API endpoints and handlers
- **Web**: Web-based interface components

## Design Principles

1. **Dependency Inversion**: High-level modules don't depend on low-level modules
2. **Clean Architecture**: Dependencies point inward toward the domain
3. **Single Responsibility**: Each class has one reason to change
4. **Interface Segregation**: Clients depend only on interfaces they use
5. **Open/Closed**: Open for extension, closed for modification

## Usage Examples

```python
# Initialize SDK
from python_sdk import PynomaliSDK

sdk = PynomaliSDK(
    api_key="your-api-key",
    base_url="https://api.pynomaly.com"
)

# Detect anomalies
result = sdk.detect_anomalies(
    data=[1.0, 2.0, 3.0, 100.0, 4.0, 5.0],
    algorithm="isolation_forest",
    parameters={"contamination": 0.1}
)

print(f"Anomalies detected: {result.anomalies}")
```

## Development

### Running Tests
```bash
# Unit tests
pytest tests/unit/

# Integration tests  
pytest tests/integration/

# End-to-end tests
pytest tests/e2e/

# All tests with coverage
pytest tests/ --cov=python_sdk --cov-report=html
```

### Building Documentation
```bash
# Generate API docs
sphinx-build -b html docs/api/ docs/_build/api/

# Generate user guides
sphinx-build -b html docs/guides/ docs/_build/guides/
```

## Contributing

1. Follow the established architecture patterns
2. Add tests for new functionality
3. Update documentation for API changes
4. Ensure all tests pass before submitting PRs

## License

This project is licensed under the MIT License - see the LICENSE file for details.