# Core Package Examples

This directory contains examples demonstrating the use of core components and abstractions.

## Quick Start Examples

- [`basic_entity_usage.py`](basic_entity_usage.py) - Using base entities
- [`value_object_examples.py`](value_object_examples.py) - Working with value objects
- [`repository_patterns.py`](repository_patterns.py) - Repository pattern implementation
- [`service_patterns.py`](service_patterns.py) - Service pattern examples

## Core Abstractions

### Base Entity Examples
- [`entity_lifecycle.py`](abstractions/entity_lifecycle.py) - Entity lifecycle management
- [`entity_validation.py`](abstractions/entity_validation.py) - Entity validation patterns
- [`entity_serialization.py`](abstractions/entity_serialization.py) - Entity serialization
- [`entity_comparison.py`](abstractions/entity_comparison.py) - Entity equality and comparison

### Value Object Examples
- [`immutable_values.py`](abstractions/immutable_values.py) - Immutable value objects
- [`value_validation.py`](abstractions/value_validation.py) - Value object validation
- [`value_composition.py`](abstractions/value_composition.py) - Composing value objects

### Repository Pattern Examples
- [`in_memory_repository.py`](abstractions/in_memory_repository.py) - In-memory repository
- [`file_repository.py`](abstractions/file_repository.py) - File-based repository
- [`database_repository.py`](abstractions/database_repository.py) - Database repository
- [`cached_repository.py`](abstractions/cached_repository.py) - Cached repository pattern

### Service Pattern Examples
- [`domain_service.py`](abstractions/domain_service.py) - Domain service patterns
- [`application_service.py`](abstractions/application_service.py) - Application service patterns
- [`service_composition.py`](abstractions/service_composition.py) - Service composition

## Domain-Driven Design Examples

### Specification Pattern
- [`specification_pattern.py`](ddd/specification_pattern.py) - Specification pattern usage
- [`composite_specifications.py`](ddd/composite_specifications.py) - Combining specifications
- [`query_specifications.py`](ddd/query_specifications.py) - Query specifications

### Aggregate Examples
- [`aggregate_root.py`](ddd/aggregate_root.py) - Aggregate root patterns
- [`aggregate_lifecycle.py`](ddd/aggregate_lifecycle.py) - Aggregate lifecycle
- [`domain_events.py`](ddd/domain_events.py) - Domain events

## Value Object Examples

### Core Value Objects
- [`performance_metrics.py`](value_objects/performance_metrics.py) - Performance metrics usage
- [`model_metrics.py`](value_objects/model_metrics.py) - Model metrics examples
- [`semantic_version.py`](value_objects/semantic_version.py) - Semantic versioning
- [`confidence_interval.py`](value_objects/confidence_interval.py) - Confidence intervals

### Configuration Value Objects
- [`hyperparameters.py`](value_objects/hyperparameters.py) - Hyperparameter management
- [`threshold_config.py`](value_objects/threshold_config.py) - Threshold configuration
- [`contamination_rate.py`](value_objects/contamination_rate.py) - Contamination rate handling

### Storage Value Objects
- [`storage_credentials.py`](value_objects/storage_credentials.py) - Storage credentials
- [`model_storage_info.py`](value_objects/model_storage_info.py) - Model storage information

## Error Handling Examples

### Exception Handling
- [`unified_exceptions.py`](error_handling/unified_exceptions.py) - Unified exception handling
- [`error_recovery.py`](error_handling/error_recovery.py) - Error recovery strategies
- [`resilience_patterns.py`](error_handling/resilience_patterns.py) - Resilience patterns

### Monitoring and Observability
- [`error_monitoring.py`](error_handling/error_monitoring.py) - Error monitoring
- [`metrics_collection.py`](error_handling/metrics_collection.py) - Metrics collection
- [`health_checks.py`](error_handling/health_checks.py) - Health check patterns

## Protocol Examples

### Data Loading Protocols
- [`data_loader_protocol.py`](protocols/data_loader_protocol.py) - Data loader protocol
- [`custom_data_loader.py`](protocols/custom_data_loader.py) - Custom data loader implementation

### Detector Protocols
- [`detector_protocol.py`](protocols/detector_protocol.py) - Detector protocol usage
- [`custom_detector.py`](protocols/custom_detector.py) - Custom detector implementation

### Repository Protocols
- [`repository_protocol.py`](protocols/repository_protocol.py) - Repository protocol
- [`async_repository.py`](protocols/async_repository.py) - Async repository implementation

### Export/Import Protocols
- [`export_protocol.py`](protocols/export_protocol.py) - Export protocol usage
- [`import_protocol.py`](protocols/import_protocol.py) - Import protocol usage
- [`format_converters.py`](protocols/format_converters.py) - Format conversion examples

## Type System Examples

### Core Types
- [`config_types.py`](types/config_types.py) - Configuration types
- [`metrics_types.py`](types/metrics_types.py) - Metrics types
- [`parameter_types.py`](types/parameter_types.py) - Parameter types
- [`result_types.py`](types/result_types.py) - Result types

### Type Validation
- [`type_validation.py`](types/type_validation.py) - Type validation examples
- [`type_conversion.py`](types/type_conversion.py) - Type conversion utilities

## Utility Examples

### Configuration Management
- [`config_management.py`](utilities/config_management.py) - Configuration management
- [`environment_config.py`](utilities/environment_config.py) - Environment configuration
- [`config_validation.py`](utilities/config_validation.py) - Configuration validation

### Serialization Examples
- [`json_serialization.py`](utilities/json_serialization.py) - JSON serialization
- [`pickle_serialization.py`](utilities/pickle_serialization.py) - Pickle serialization
- [`custom_serialization.py`](utilities/custom_serialization.py) - Custom serialization

## Testing Examples

### Unit Testing
- [`entity_testing.py`](testing/entity_testing.py) - Entity testing patterns
- [`value_object_testing.py`](testing/value_object_testing.py) - Value object testing
- [`service_testing.py`](testing/service_testing.py) - Service testing patterns

### Test Utilities
- [`test_factories.py`](testing/test_factories.py) - Test factory patterns
- [`mock_repositories.py`](testing/mock_repositories.py) - Mock repository implementations
- [`test_builders.py`](testing/test_builders.py) - Test builder patterns

### Property-Based Testing
- [`property_testing.py`](testing/property_testing.py) - Property-based testing
- [`hypothesis_strategies.py`](testing/hypothesis_strategies.py) - Hypothesis strategies

## Integration Examples

### Framework Integration
- [`fastapi_integration.py`](integration/fastapi_integration.py) - FastAPI integration
- [`django_integration.py`](integration/django_integration.py) - Django integration
- [`flask_integration.py`](integration/flask_integration.py) - Flask integration

### Database Integration
- [`sqlalchemy_integration.py`](integration/sqlalchemy_integration.py) - SQLAlchemy integration
- [`mongodb_integration.py`](integration/mongodb_integration.py) - MongoDB integration
- [`redis_integration.py`](integration/redis_integration.py) - Redis integration

## Advanced Examples

### Dependency Injection
- [`dependency_injection.py`](advanced/dependency_injection.py) - Dependency injection patterns
- [`service_locator.py`](advanced/service_locator.py) - Service locator pattern
- [`container_configuration.py`](advanced/container_configuration.py) - Container configuration

### Event Sourcing
- [`event_sourcing.py`](advanced/event_sourcing.py) - Event sourcing patterns
- [`event_store.py`](advanced/event_store.py) - Event store implementation
- [`projection_handlers.py`](advanced/projection_handlers.py) - Projection handlers

### CQRS Examples
- [`command_query_separation.py`](advanced/command_query_separation.py) - CQRS patterns
- [`command_handlers.py`](advanced/command_handlers.py) - Command handlers
- [`query_handlers.py`](advanced/query_handlers.py) - Query handlers

## Performance Examples

### Optimization Patterns
- [`lazy_loading.py`](performance/lazy_loading.py) - Lazy loading patterns
- [`caching_strategies.py`](performance/caching_strategies.py) - Caching strategies
- [`memory_optimization.py`](performance/memory_optimization.py) - Memory optimization

### Async Patterns
- [`async_services.py`](performance/async_services.py) - Async service patterns
- [`async_repositories.py`](performance/async_repositories.py) - Async repository patterns
- [`concurrent_processing.py`](performance/concurrent_processing.py) - Concurrent processing

## Running the Examples

### Prerequisites

1. Install the package:
   ```bash
   pip install -e .
   ```

2. Install development dependencies:
   ```bash
   pip install -e .[dev]
   ```

### Basic Usage

```bash
# Run a simple example
python examples/basic_entity_usage.py

# Run value object examples
python examples/value_object_examples.py

# Run repository pattern examples
python examples/repository_patterns.py
```

### Advanced Usage

```bash
# Run with debugging
python examples/error_handling/unified_exceptions.py --debug

# Run with specific configuration
python examples/utilities/config_management.py --config config/example.yaml

# Run performance examples
python examples/performance/async_services.py --workers 4
```

## Example Categories

### 🚀 **Beginner Level**
- Basic entity and value object usage
- Simple repository patterns
- Basic error handling
- Core type system

### 📊 **Intermediate Level**
- Advanced abstractions
- Domain-driven design patterns
- Protocol implementations
- Integration examples

### 🔬 **Advanced Level**
- Event sourcing and CQRS
- Advanced dependency injection
- Performance optimization
- Custom framework integration

## Best Practices Demonstrated

- **Domain-Driven Design**: Proper entity and value object design
- **SOLID Principles**: Single responsibility, open/closed, etc.
- **Clean Architecture**: Separation of concerns
- **Type Safety**: Proper type annotations and validation
- **Error Handling**: Comprehensive error handling strategies
- **Testing**: Unit testing and property-based testing
- **Performance**: Optimization techniques and patterns

## Support

For questions about examples:
- Check the main [README](../README.md)
- Review the [documentation](../docs/)
- Open an issue on GitHub
- Join our community discussions

## Contributing

To add new examples:
1. Create a new Python file with a descriptive name
2. Add comprehensive docstrings and comments
3. Include proper type annotations
4. Add unit tests for complex examples
5. Update this README with your example
6. Submit a pull request

---

**Note**: Examples focus on demonstrating core patterns and abstractions. They are designed to be educational and may require adaptation for specific use cases.