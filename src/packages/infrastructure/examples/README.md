# Infrastructure Package Examples

This directory contains practical examples of using the infrastructure package components.

## Available Examples

### Basic Usage
- **[Configuration Setup](basic_configuration.py)**: Setting up configuration management
- **[Database Connection](database_example.py)**: Connecting to and querying databases  
- **[Logging Setup](logging_example.py)**: Configuring structured logging
- **[Caching](caching_example.py)**: Using Redis-based caching

### Advanced Patterns
- **[Event-Driven Architecture](event_driven_example.py)**: Using event buses for decoupled communication
- **[Repository Pattern](repository_example.py)**: Implementing data access with repositories
- **[Unit of Work](unit_of_work_example.py)**: Managing transactions across multiple repositories
- **[Security Integration](security_example.py)**: JWT authentication and authorization

### Production Scenarios
- **[Health Checks](health_check_example.py)**: Implementing comprehensive health monitoring
- **[Metrics Collection](metrics_example.py)**: Collecting and exposing application metrics
- **[Distributed Tracing](tracing_example.py)**: Setting up distributed tracing
- **[Error Handling](error_handling_example.py)**: Robust error handling patterns

### Integration Examples
- **[FastAPI Integration](fastapi_integration.py)**: Using infrastructure with FastAPI
- **[Celery Integration](celery_integration.py)**: Background task processing
- **[Docker Setup](docker_example/)**: Containerized deployment configuration
- **[Kubernetes](k8s_example/)**: Kubernetes deployment manifests

## Running Examples

Each example is self-contained and can be run independently:

```bash
# Install dependencies
pip install infrastructure[all]

# Set environment variables (see .env.example)
export DATABASE_URL="postgresql://user:pass@localhost/db"
export REDIS_URL="redis://localhost:6379/0"

# Run an example
python examples/basic_configuration.py
```

## Example Structure

Each example follows this structure:
- **Setup**: Configuration and dependency initialization
- **Implementation**: Core functionality demonstration
- **Testing**: Basic validation and testing
- **Cleanup**: Proper resource cleanup

## Development Setup

For developing and testing examples:

```bash
# Create virtual environment
python -m venv venv
source venv/bin/activate  # Linux/Mac
# or
venv\Scripts\activate     # Windows

# Install development dependencies
pip install infrastructure[dev]

# Run tests for examples
pytest examples/tests/
```

## Contributing Examples

When adding new examples:

1. Follow the existing structure and naming conventions
2. Include comprehensive comments explaining the code
3. Add error handling and proper cleanup
4. Include a test file to validate the example works
5. Update this README with the new example

## Common Patterns

### Configuration Management
```python
from infrastructure.configuration import get_config

config = get_config()
db_url = config.database.url
```

### Database Operations
```python
from infrastructure.persistence import get_database, BaseRepository

db = get_database()
repository = UserRepository(db)
users = await repository.find_all()
```

### Event Publishing
```python
from infrastructure.messaging import EventBus

event_bus = EventBus()
await event_bus.publish(UserCreatedEvent(user_id="123"))
```

### Caching
```python
from infrastructure.caching import CacheManager

cache = CacheManager()
await cache.set("key", data, ttl=3600)
result = await cache.get("key")
```

### Logging
```python
from infrastructure.monitoring import get_logger

logger = get_logger(__name__)
logger.info("Operation completed", user_id="123", duration=1.5)
```

## Best Practices

1. **Always use dependency injection** for infrastructure components
2. **Handle errors gracefully** with proper exception handling
3. **Use async/await** for I/O operations
4. **Configure observability** with logging, metrics, and tracing
5. **Clean up resources** properly using context managers
6. **Use type hints** for better code documentation and IDE support