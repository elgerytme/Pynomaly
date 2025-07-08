# Forward-Reference-Free Dependency Injection System

A lightweight wrapper around FastAPI's `Depends` that eliminates circular import issues by allowing dependency declaration without type hints in router files, then injecting fully typed callables during application startup.

## Quick Start

### 1. Declare Dependencies in Router Files

```python
from fastapi import APIRouter
from pynomaly.infrastructure.dependencies import auth_service, detection_service

router = APIRouter()

# Declare without type hints - avoids circular imports
get_auth_service = auth_service()
get_detection_service = detection_service()

@router.post("/detect")
async def detect_anomalies(
    data: str,
    # No type hints on dependencies
    auth_svc = get_auth_service(),
    detection_svc = get_detection_service(),
) -> dict:
    if not detection_svc:
        return {"error": "Detection service unavailable"}
    
    # Use services normally
    result = detection_svc.detect(data)
    return {"anomalies": result}
```

### 2. Dependencies are Automatically Registered at Startup

Dependencies are registered in the FastAPI application's `lifespan` function:

```python
from pynomaly.infrastructure.dependencies.setup import setup_dependencies

@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator:
    # ... other startup code ...
    
    # Setup dependencies
    dependency_setup = setup_dependencies(container)
    
    yield
    
    # Cleanup on shutdown
    from pynomaly.infrastructure.dependencies import clear_dependencies
    clear_dependencies()
```

### 3. Add Custom Dependencies

```python
from pynomaly.infrastructure.dependencies import register_dependency, DependencyWrapper

# Register a custom service
custom_service = MyCustomService()
register_dependency("custom_service", custom_service)

# Use in router
get_custom_service = DependencyWrapper("custom_service", optional=True)

@router.get("/custom")
async def custom_endpoint(
    custom_svc = get_custom_service(),
) -> dict:
    if not custom_svc:
        return {"error": "Custom service unavailable"}
    
    return {"result": custom_svc.do_something()}
```

## Key Features

- **No Circular Imports**: Dependencies declared without type hints
- **Late Binding**: Services injected after models are defined
- **Graceful Degradation**: Optional services can fail without breaking the app
- **Easy Testing**: Built-in mock dependencies and validation tools
- **Type Safety**: Full type checking in service implementations

## Architecture

### Core Components

1. **DependencyWrapper**: Wraps FastAPI's `Depends` with late binding
2. **DependencyRegistry**: Manages dependency lifecycle
3. **DependencySetup**: Handles registration during startup
4. **Common Wrappers**: Pre-configured wrappers for typical services

### Dependency Resolution Flow

```
Router Declaration → Registry Lookup → Service Instance → FastAPI Depends
```

1. **Router Declaration**: `get_service = service_wrapper()`
2. **Registry Lookup**: Wrapper looks up service in registry
3. **Service Instance**: Registry returns actual service instance
4. **FastAPI Depends**: Wrapper creates `Depends` object for FastAPI

## Common Usage Patterns

### Authentication Service

```python
from pynomaly.infrastructure.dependencies import auth_service

get_auth_service = auth_service()

@router.post("/protected")
async def protected_endpoint(
    auth_svc = get_auth_service(),
) -> dict:
    if not auth_svc:
        raise HTTPException(status_code=503, detail="Auth unavailable")
    
    # Use auth service...
```

### Detection Service

```python
from pynomaly.infrastructure.dependencies import detection_service

get_detection_service = detection_service()

@router.post("/detect")
async def detect(
    data: str,
    detection_svc = get_detection_service(),
) -> dict:
    if not detection_svc:
        return {"error": "Detection service unavailable"}
    
    result = detection_svc.detect(data)
    return {"anomalies": result}
```

### Multiple Services

```python
@router.post("/complex")
async def complex_endpoint(
    auth_svc = get_auth_service(),
    detection_svc = get_detection_service(),
    model_svc = get_model_service(),
) -> dict:
    services = [auth_svc, detection_svc, model_svc]
    unavailable = [s for s in services if s is None]
    
    if unavailable:
        return {"error": f"Services unavailable: {len(unavailable)}"}
    
    # Use all services...
```

## Available Service Wrappers

### Core Services

- `auth_service()` - Authentication service (optional)
- `user_service()` - User management service (optional)
- `detection_service()` - Anomaly detection service (required)
- `model_service()` - Model management service (required)
- `database_service()` - Database service (optional)

### Infrastructure Services

- `cache_service()` - Caching service (optional)
- `metrics_service()` - Metrics collection service (optional)

### Custom Services

```python
from pynomaly.infrastructure.dependencies import DependencyWrapper

# Create custom wrapper
get_custom_service = DependencyWrapper("custom_service", optional=True)
```

## Testing

### Mock Dependencies

```python
from pynomaly.infrastructure.dependencies import (
    test_dependency_context,
    setup_test_dependencies,
)

def test_my_endpoint():
    with test_dependency_context():
        # Setup mock dependencies
        setup_test_dependencies()
        
        # Test endpoint...
```

### Validation Tools

```python
from pynomaly.infrastructure.dependencies import (
    validate_standard_dependencies,
    run_dependency_health_check,
)

# Validate dependencies
results = validate_standard_dependencies()
print(f"Available: {results['total_available']}/{results['total_expected']}")

# Run health check
run_dependency_health_check()
```

## Migration Guide

### From Traditional Dependencies

**Before (Problematic):**
```python
from typing import Annotated
from fastapi import Depends
from pynomaly.application.services import DetectionService

@router.post("/detect")
async def detect(
    detection_service: Annotated[DetectionService, Depends(get_detection_service)],
) -> dict:
    # Implementation
```

**After (Safe):**
```python
from pynomaly.infrastructure.dependencies import detection_service

get_detection_service = detection_service()

@router.post("/detect")
async def detect(
    detection_svc = get_detection_service(),
) -> dict:
    if not detection_svc:
        return {"error": "Service unavailable"}
    
    # Implementation
```

## Error Handling

### Service Unavailable

```python
@router.post("/endpoint")
async def my_endpoint(
    service = get_my_service(),
) -> dict:
    if not service:
        return {"error": "Service unavailable", "success": False}
    
    try:
        result = service.do_something()
        return {"result": result, "success": True}
    except Exception as e:
        return {"error": str(e), "success": False}
```

### Required vs Optional Services

```python
# Optional service - returns None if unavailable
get_optional_service = DependencyWrapper("optional_service", optional=True)

# Required service - raises HTTP 503 if unavailable
get_required_service = DependencyWrapper("required_service", optional=False)
```

## Troubleshooting

### Common Issues

1. **Service Not Found**: Check if service is registered in `setup.py`
2. **Import Errors**: Ensure no circular imports in service definitions
3. **Startup Failures**: Check logs for dependency registration errors

### Debug Tools

```python
from pynomaly.infrastructure.dependencies import is_dependency_available

# Check service availability
if is_dependency_available("my_service"):
    print("Service is available")
else:
    print("Service is not available")
```

### Logging

```python
import logging
logging.getLogger("pynomaly.infrastructure.dependencies").setLevel(logging.DEBUG)
```

## Best Practices

1. **Always Check Service Availability**: Use `if not service:` checks
2. **Use Optional for Non-Critical Services**: Cache, metrics, etc.
3. **Handle Errors Gracefully**: Return error responses instead of raising exceptions
4. **Test with Mocks**: Use `test_dependency_context` for isolated testing
5. **Validate Dependencies**: Use validation tools to ensure proper setup

## Examples

See the following files for complete examples:

- `src/pynomaly/presentation/api/endpoints/example_with_dependencies.py` - Router example
- `examples/dependency_example.py` - Complete usage example
- `tests/test_dependency_system.py` - Testing examples

## API Reference

### DependencyWrapper

```python
class DependencyWrapper:
    def __init__(self, dependency_key: str, optional: bool = False)
    def __call__(self) -> Depends
```

### Registry Functions

```python
def register_dependency(key: str, instance: Any) -> None
def register_dependency_provider(key: str, provider: Callable) -> None
def get_dependency(key: str) -> Any
def is_dependency_available(key: str) -> bool
def clear_dependencies() -> None
```

### Test Utilities

```python
def test_dependency_context() -> ContextManager
def create_mock_dependencies() -> Dict[str, Any]
def setup_test_dependencies() -> None
def validate_standard_dependencies() -> Dict[str, Any]
def run_dependency_health_check() -> bool
```

## Contributing

When adding new dependencies:

1. Add service to `setup.py`
2. Create wrapper function in `wrapper.py`
3. Add to `__init__.py` exports
4. Update documentation
5. Add tests

For more information, see the main `architecture.md` file.
