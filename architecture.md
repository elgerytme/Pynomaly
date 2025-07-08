# Forward-Reference-Free Dependency System Architecture

## Overview

This document describes the forward-reference-free dependency injection system designed to eliminate circular import issues in FastAPI applications. The system provides a clean way to declare dependencies without type hints in router files, then inject fully typed callables during application startup.

## Problem Statement

### Circular Import Issues

Traditional FastAPI dependency injection can cause circular import problems when:

1. **Domain entities** import from application services
2. **Application services** import from infrastructure layer
3. **Infrastructure layer** imports from domain entities
4. **Router files** import from all layers with type hints

This creates import cycles that can cause:
- Non-deterministic startup failures
- Difficult debugging
- IDE navigation issues
- Flaky tests

### Example of Problematic Pattern

```python
# This can cause circular imports
from pynomaly.application.services import DetectionService
from pynomaly.domain.entities import Detector

@router.post("/detect")
async def detect(
    detection_service: Annotated[DetectionService, Depends(get_detection_service)],
    detector: Annotated[Detector, Depends(get_detector)],
) -> DetectionResponse:
    # Implementation
    pass
```

## Solution Architecture

### Core Components

1. **DependencyWrapper**: Lightweight wrapper around `FastAPI.Depends`
2. **DependencyRegistry**: Registry for managing dependency lifecycle
3. **DependencySetup**: Handles registration during application startup
4. **Common Service Wrappers**: Pre-configured wrappers for typical services

### Key Principles

1. **No Type Hints in Routers**: Dependencies are declared without type hints
2. **Late Binding**: Services are injected after all models are defined
3. **Graceful Degradation**: Optional services can fail without breaking the app
4. **Startup Registration**: All dependencies are registered during app startup

## Implementation Guide

### Step 1: Declare Dependencies in Router Files

```python
from fastapi import APIRouter
from pynomaly.infrastructure.dependencies import (
    auth_service,
    detection_service,
    model_service,
)

router = APIRouter()

# Declare dependencies without type hints
get_auth_service = auth_service()
get_detection_service = detection_service()
get_model_service = model_service()

@router.post("/detect")
async def detect_anomalies(
    dataset_id: str,
    # Dependencies without type hints - avoids circular imports
    auth_svc = get_auth_service(),
    detection_svc = get_detection_service(),
    model_svc = get_model_service(),
) -> DetectionResponse:
    # Use services normally
    if not detection_svc:
        return DetectionResponse(success=False, message="Service unavailable")
    
    # Implementation...
    pass
```

### Step 2: Register Dependencies at Startup

Dependencies are automatically registered during application startup in the `lifespan` function:

```python
@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator:
    # ... other startup code ...
    
    # Setup forward-reference-free dependencies
    from pynomaly.infrastructure.dependencies.setup import setup_dependencies
    dependency_setup = setup_dependencies(container)
    app.state.dependency_setup = dependency_setup
    
    yield
    
    # Cleanup on shutdown
    from pynomaly.infrastructure.dependencies import clear_dependencies
    clear_dependencies()
```

### Step 3: Custom Dependencies

For custom services, you can create explicit wrappers:

```python
from pynomaly.infrastructure.dependencies import DependencyWrapper

# Create custom dependency wrapper
get_custom_service = DependencyWrapper("custom_service", optional=True)

# Register the service during startup
from pynomaly.infrastructure.dependencies import register_dependency

def setup_custom_dependencies():
    custom_service = MyCustomService()
    register_dependency("custom_service", custom_service)
```

## Adding New Dependencies

### 1. Create the Service

```python
# src/pynomaly/application/services/my_new_service.py
class MyNewService:
    def __init__(self, dependency: SomeDependency):
        self.dependency = dependency
    
    def do_something(self):
        return "Hello from MyNewService"
```

### 2. Add to Container (Optional)

If using the dependency injection container:

```python
# src/pynomaly/infrastructure/config/container.py
def my_new_service(self) -> MyNewService:
    return MyNewService(dependency=self.some_dependency())
```

### 3. Add to Dependency Setup

```python
# src/pynomaly/infrastructure/dependencies/setup.py
def register_core_services(self) -> None:
    # ... existing services ...
    
    # Add new service
    try:
        new_service = self.container.my_new_service()
        register_dependency("my_new_service", new_service)
        self._registered_services["my_new_service"] = new_service
        logger.debug("Registered my_new_service")
    except Exception as e:
        logger.warning(f"Failed to register my_new_service: {e}")
        register_dependency("my_new_service", None)
```

### 4. Create Wrapper Function (Optional)

```python
# src/pynomaly/infrastructure/dependencies/wrapper.py
def my_new_service() -> DependencyWrapper:
    """Get the new service dependency."""
    return DependencyWrapper("my_new_service", optional=True)
```

### 5. Use in Router

```python
from pynomaly.infrastructure.dependencies import DependencyWrapper

get_my_service = DependencyWrapper("my_new_service", optional=True)
# Or use the wrapper function if created
# get_my_service = my_new_service()

@router.post("/endpoint")
async def my_endpoint(
    my_svc = get_my_service(),
) -> MyResponse:
    if not my_svc:
        return MyResponse(success=False, message="Service unavailable")
    
    result = my_svc.do_something()
    return MyResponse(success=True, data=result)
```

## Best Practices

### 1. Service Availability

Always check if services are available before using them:

```python
async def my_endpoint(
    service = get_my_service(),
) -> MyResponse:
    if not service:
        return MyResponse(success=False, message="Service unavailable")
    
    # Use service...
```

### 2. Optional vs Required Services

- **Optional services**: Use `optional=True` for services that can gracefully degrade
- **Required services**: Use `optional=False` (default) for critical services

```python
# Optional service - app continues if unavailable
get_cache_service = DependencyWrapper("cache_service", optional=True)

# Required service - app fails if unavailable
get_detection_service = DependencyWrapper("detection_service", optional=False)
```

### 3. Error Handling

Handle service failures gracefully:

```python
async def my_endpoint(
    service = get_my_service(),
) -> MyResponse:
    try:
        if not service:
            return MyResponse(success=False, message="Service unavailable")
        
        result = service.do_something()
        return MyResponse(success=True, data=result)
    
    except Exception as e:
        logger.error(f"Service error: {e}")
        return MyResponse(success=False, message="Internal error")
```

### 4. Testing

For testing, you can override dependencies:

```python
import pytest
from pynomaly.infrastructure.dependencies import register_dependency

@pytest.fixture
def mock_service():
    return MockService()

def test_endpoint(mock_service):
    register_dependency("my_service", mock_service)
    # Test endpoint...
```

## Common Patterns

### 1. Authentication Service

```python
from pynomaly.infrastructure.dependencies import auth_service

get_auth_service = auth_service()

@router.post("/protected")
async def protected_endpoint(
    auth_svc = get_auth_service(),
) -> Response:
    if not auth_svc:
        raise HTTPException(status_code=503, detail="Auth service unavailable")
    
    # Use auth service...
```

### 2. Detection Service

```python
from pynomaly.infrastructure.dependencies import detection_service

get_detection_service = detection_service()

@router.post("/detect")
async def detect(
    detection_svc = get_detection_service(),
) -> DetectionResponse:
    if not detection_svc:
        return DetectionResponse(success=False, message="Detection service unavailable")
    
    # Use detection service...
```

### 3. Database Service

```python
from pynomaly.infrastructure.dependencies import database_service

get_db_service = database_service()

@router.get("/data")
async def get_data(
    db_svc = get_db_service(),
) -> DataResponse:
    if not db_svc:
        return DataResponse(success=False, message="Database unavailable")
    
    # Use database service...
```

## Migration from Traditional Dependencies

### Before (Problematic)

```python
from typing import Annotated
from fastapi import Depends
from pynomaly.application.services import DetectionService

@router.post("/detect")
async def detect(
    detection_service: Annotated[DetectionService, Depends(get_detection_service)],
) -> DetectionResponse:
    # Implementation
    pass
```

### After (Safe)

```python
from pynomaly.infrastructure.dependencies import detection_service

get_detection_service = detection_service()

@router.post("/detect")
async def detect(
    detection_svc = get_detection_service(),
) -> DetectionResponse:
    if not detection_svc:
        return DetectionResponse(success=False, message="Service unavailable")
    
    # Implementation
    pass
```

## Advanced Usage

### 1. Conditional Dependencies

```python
from pynomaly.infrastructure.dependencies import DependencyWrapper

def get_conditional_service(condition: bool) -> DependencyWrapper:
    if condition:
        return DependencyWrapper("service_a")
    else:
        return DependencyWrapper("service_b")
```

### 2. Service Composition

```python
@router.post("/complex")
async def complex_endpoint(
    auth_svc = get_auth_service(),
    detection_svc = get_detection_service(),
    model_svc = get_model_service(),
    db_svc = get_database_service(),
) -> ComplexResponse:
    # Use multiple services together
    services = [auth_svc, detection_svc, model_svc, db_svc]
    unavailable = [s for s in services if s is None]
    
    if unavailable:
        return ComplexResponse(
            success=False,
            message=f"Services unavailable: {len(unavailable)}"
        )
    
    # Use all services...
```

### 3. Provider Functions

Instead of registering instances, you can register provider functions:

```python
from pynomaly.infrastructure.dependencies import register_dependency_provider

def create_expensive_service():
    return ExpensiveService()

# Register provider instead of instance
register_dependency_provider("expensive_service", create_expensive_service)
```

## Troubleshooting

### Common Issues

1. **Service Not Found**: Check if the service is registered in `setup.py`
2. **Import Errors**: Ensure no circular imports in service definitions
3. **Startup Failures**: Check logs for dependency registration errors
4. **Service Unavailable**: Verify the service is properly configured

### Debug Tools

```python
from pynomaly.infrastructure.dependencies import is_dependency_available

# Check if service is available
if is_dependency_available("my_service"):
    print("Service is available")
else:
    print("Service is not available")
```

### Logging

Enable debug logging to see dependency registration:

```python
import logging
logging.getLogger("pynomaly.infrastructure.dependencies").setLevel(logging.DEBUG)
```

## Conclusion

The forward-reference-free dependency system provides a clean, maintainable way to handle dependencies in FastAPI applications while avoiding circular import issues. By following the patterns described in this document, you can safely add new dependencies without worrying about import cycles.

The system is designed to be:
- **Safe**: No circular imports
- **Flexible**: Support for optional and required services
- **Maintainable**: Clear patterns for adding new dependencies
- **Robust**: Graceful degradation when services are unavailable

For more examples, see the `example_with_dependencies.py` file in the API endpoints directory.
