# Pynomaly Core

The core domain logic and business rules for the Pynomaly anomaly detection platform.

## Overview

This package contains the pure business logic that drives the Pynomaly platform. It follows clean architecture principles with no external dependencies on infrastructure concerns, making it highly testable and reusable across different interfaces.

## Key Components

### Domain Layer
- **Entities**: Core business objects like `Anomaly`, `Dataset`, `Detector`, `Model`
- **Value Objects**: Immutable objects like `AnomalyScore`, `ContaminationRate`, `PerformanceMetrics`
- **Abstractions**: Base classes and interfaces for repositories and services
- **Exceptions**: Domain-specific error types

### Application Layer
- **Use Cases**: Business operations like `detect_anomalies`, `train_detector`, `evaluate_model`
- **DTOs**: Data Transfer Objects for moving data between layers
- **Services**: Domain services that orchestrate business logic

### Shared Utilities
- **Types**: Common type definitions
- **Exceptions**: Shared exception classes
- **Error Handling**: Common error handling patterns

## Design Principles

1. **Dependency Inversion**: Core depends on abstractions, not concrete implementations
2. **Single Responsibility**: Each component has a single, well-defined purpose
3. **Open/Closed**: Open for extension, closed for modification
4. **Domain-Driven Design**: Models the real-world problem domain
5. **Testability**: All components are easily unit testable

## Installation

```bash
pip install pynomaly-core
```

For development:
```bash
pip install pynomaly-core[dev]
```

## Usage

```python
from pynomaly.core import detect_anomalies, Dataset, Detector

# Create a dataset
dataset = Dataset.from_array(data)

# Create a detector
detector = Detector.isolation_forest()

# Detect anomalies
result = detect_anomalies(dataset, detector)
```

## Testing

```bash
pytest tests/
```

With coverage:
```bash
pytest --cov=pynomaly_core tests/
```

## Dependencies

The core package has minimal dependencies to maintain clean separation:
- `pydantic`: For data validation and serialization
- `structlog`: For structured logging
- `dependency-injector`: For dependency injection

## License

MIT License. See LICENSE file for details.