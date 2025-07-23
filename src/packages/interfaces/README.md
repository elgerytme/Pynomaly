# Interfaces Package

Domain contracts and interfaces for cross-domain communication in the monorepo.

## Overview

The interfaces package defines the contracts that enable different domains (AI, Data, Enterprise, Tools) to communicate with each other while maintaining proper boundaries. It provides a stable API surface that prevents tight coupling between domains.

## Core Components

### Domain Contracts
- **AI Domain**: Machine learning, MLOps, and neural-symbolic interfaces
- **Data Domain**: Analytics, quality, profiling, and transformation interfaces  
- **Enterprise Domain**: Authentication, governance, and scalability interfaces
- **Tools Domain**: Utilities and cross-cutting concern interfaces

### Communication Patterns
- **DTOs**: Data Transfer Objects for cross-domain communication
- **Events**: Domain events for loose coupling via event-driven architecture
- **Commands**: Command pattern for cross-domain operations
- **Queries**: Query pattern for cross-domain data retrieval

### Integration Interfaces
- **Repository Contracts**: Abstract data access patterns
- **Service Contracts**: Abstract business logic interfaces
- **Publisher/Subscriber**: Event bus and messaging patterns
- **Anti-Corruption Layers**: Translation between domain models

## Architecture Principles

### Domain Isolation
- Interfaces define contracts without implementation details
- No dependencies on concrete domain implementations
- Enables independent domain evolution

### Dependency Direction
- Domain packages depend on interfaces
- Interfaces package is dependency-free (except shared)
- Follows Interface Segregation Principle

### Stability
- Interfaces are designed for long-term stability
- Breaking changes require versioning and migration strategies
- Backward compatibility is prioritized

## Usage Examples

### Cross-Domain Service Communication
```python
from interfaces.ai import MLModelInterface
from interfaces.data import DataQualityInterface

class ModelTrainingService:
    def __init__(
        self,
        ml_service: MLModelInterface,
        quality_service: DataQualityInterface
    ):
        self.ml_service = ml_service
        self.quality_service = quality_service
    
    async def train_with_quality_check(self, dataset_id: str):
        # Check data quality first
        quality_result = await self.quality_service.validate_dataset(dataset_id)
        if not quality_result.is_valid:
            raise ValueError(f"Data quality issues: {quality_result.issues}")
        
        # Train model
        return await self.ml_service.train_model(dataset_id)
```

### Event-Driven Communication
```python
from interfaces.events import DomainEventBus, ModelTrainingCompleted

# Publishing domain events
event_bus = DomainEventBus()
event = ModelTrainingCompleted(
    model_id="model-123",
    accuracy=0.95,
    training_duration=3600
)
await event_bus.publish(event)

# Subscribing to events
@event_bus.subscribe(ModelTrainingCompleted)
async def update_model_registry(event: ModelTrainingCompleted):
    # Update model registry when training completes
    await model_registry.register_model(event.model_id, event.accuracy)
```

### DTOs for Data Transfer
```python
from interfaces.dto import DetectionRequest, DetectionResult

# Request/Response DTOs
request = DetectionRequest(
    data=sensor_readings,
    algorithm="isolation_forest",
    sensitivity=0.1
)

result: DetectionResult = await anomaly_service.detect(request)
print(f"Found {result.anomaly_count} anomalies")
```

## Package Structure

```
interfaces/
├── src/interfaces/
│   ├── __init__.py
│   ├── ai/                     # AI domain interfaces
│   │   ├── __init__.py
│   │   ├── ml_model.py
│   │   ├── mlops.py
│   │   └── neuro_symbolic.py
│   ├── data/                   # Data domain interfaces
│   │   ├── __init__.py
│   │   ├── analytics.py
│   │   ├── quality.py
│   │   ├── profiling.py
│   │   └── transformation.py
│   ├── enterprise/             # Enterprise domain interfaces
│   │   ├── __init__.py
│   │   ├── auth.py
│   │   ├── governance.py
│   │   └── scalability.py
│   ├── tools/                  # Tools domain interfaces
│   │   ├── __init__.py
│   │   └── utilities.py
│   ├── dto/                    # Data Transfer Objects
│   │   ├── __init__.py
│   │   ├── common.py
│   │   ├── ai_dto.py
│   │   ├── data_dto.py
│   │   └── enterprise_dto.py
│   ├── events/                 # Domain events
│   │   ├── __init__.py
│   │   ├── base.py
│   │   ├── ai_events.py
│   │   ├── data_events.py
│   │   └── enterprise_events.py
│   └── patterns/               # Integration patterns
│       ├── __init__.py
│       ├── repository.py
│       ├── service.py
│       ├── event_bus.py
│       └── anti_corruption.py
├── tests/
└── docs/
```

## Contract Testing

The interfaces package includes contract testing to ensure compatibility:

```bash
# Run contract tests
pytest -m contract

# Validate interface compliance
interfaces-validate

# Generate interface documentation
mkdocs build
```

## Versioning Strategy

### Semantic Versioning
- **Major**: Breaking changes to interfaces
- **Minor**: New interfaces or optional parameters
- **Patch**: Bug fixes and clarifications

### Compatibility Guarantees
- Interfaces maintain backward compatibility within major versions
- Deprecation warnings provided before removal
- Migration guides for breaking changes

## Contributing

1. **Design First**: Create interface contracts before implementations
2. **Stability**: Interfaces should be stable and well-thought-out
3. **Documentation**: All interfaces must be fully documented
4. **Testing**: Contract tests required for all interfaces
5. **Versioning**: Follow semantic versioning for changes

## API Documentation

For detailed interface documentation:
- [AI Domain Interfaces](docs/api/ai.md)
- [Data Domain Interfaces](docs/api/data.md)
- [Enterprise Interfaces](docs/api/enterprise.md)
- [DTOs Reference](docs/api/dto.md)
- [Events Reference](docs/api/events.md)
- [Integration Patterns](docs/api/patterns.md)