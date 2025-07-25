# Migration Guide: Adopting New Package Interaction Patterns

This guide provides step-by-step instructions for migrating existing code to use the new package interaction patterns implemented in this monorepo.

## Overview

The new interaction patterns include:
- **Event-driven communication** via `interfaces/events.py`
- **Dependency injection** via `shared/dependency_injection.py`
- **Stable contracts** via `interfaces/dto.py` and `interfaces/patterns.py`
- **Automated boundary validation** via tools and CI/CD

## Migration Steps

### Step 1: Update Imports

#### Before (Direct Cross-Domain Imports)
```python
# ❌ Direct domain-to-domain imports
from data.quality.services import DataQualityService
from ai.anomaly_detection.services import DetectionService

class WorkflowService:
    def __init__(self):
        self.quality_service = DataQualityService()
        self.detection_service = DetectionService()
```

#### After (Interface-Based Imports)
```python
# ✅ Import via interfaces and dependency injection
from interfaces.dto import DataQualityRequest, DetectionRequest
from interfaces.patterns import Service
from shared import get_container, inject

class WorkflowService:
    @inject(get_container())
    def __init__(self, 
                 quality_service: Service,
                 detection_service: Service):
        self.quality_service = quality_service
        self.detection_service = detection_service
```

### Step 2: Replace Direct Method Calls with Events

#### Before (Tight Coupling)
```python
# ❌ Direct method calls between domains
from data.quality.services import DataQualityService

class DetectionService:
    def __init__(self):
        self.quality_service = DataQualityService()
    
    async def detect(self, dataset_id: str):
        quality_result = await self.quality_service.check_quality(dataset_id)
        if quality_result.score > 0.8:
            # Proceed with detection
            pass
```

#### After (Event-Driven)
```python
# ✅ Event-driven communication
from interfaces.events import DataQualityCheckCompleted
from interfaces.dto import DetectionRequest
from shared import publish_event, event_handler

class DetectionService:
    def __init__(self):
        # Subscribe to quality check events
        self._setup_event_subscriptions()
    
    async def detect(self, dataset_id: str):
        # Request quality check via event
        await publish_event(QualityCheckRequested(dataset_id=dataset_id))
        # Detection will proceed when quality check completes
    
    @event_handler(DataQualityCheckCompleted)
    async def _on_quality_check_completed(self, event: DataQualityCheckCompleted):
        if event.overall_score > 0.8:
            # Proceed with detection
            pass
```

### Step 3: Update Container Configuration

#### Before (Manual Wiring)
```python
# ❌ Manual service instantiation
class Container:
    def __init__(self):
        self.services = {}
        self._setup()
    
    def _setup(self):
        self.services['quality'] = DataQualityService()
        self.services['detection'] = DetectionService(self.services['quality'])
```

#### After (Dependency Injection)
```python
# ✅ Modern DI container
from shared import configure_container, register_service

def setup_services():
    def configure(container):
        register_service(container, DataQualityService, DataQualityService)
        register_service(container, DetectionService, DetectionService)
    
    configure_container(configure)
```

### Step 4: Migrate Data Transfer Objects

#### Before (Internal DTOs)
```python
# ❌ Package-specific DTOs
class DetectionRequest:
    def __init__(self, dataset_id: str, algorithm: str):
        self.dataset_id = dataset_id
        self.algorithm = algorithm
```

#### After (Shared DTOs)
```python
# ✅ Use interfaces DTOs
from interfaces.dto import DetectionRequest, DetectionResult

# DTOs are now shared and stable across domains
def process_detection(request: DetectionRequest) -> DetectionResult:
    # Implementation using shared contracts
    pass
```

## Package-Specific Migration

### Domain Packages (ai/, data/)

1. **Remove cross-domain imports**
   ```bash
   # Run boundary validator to find violations
   python src/packages/tools/import_boundary_validator/boundary_validator.py
   ```

2. **Update service interfaces**
   ```python
   # Before
   from other_domain.services import SomeService
   
   # After
   from interfaces.patterns import Service
   from shared import get_container
   
   container = get_container()
   service = container.resolve(Service)
   ```

3. **Add event handlers**
   ```python
   from interfaces.events import SomeEvent
   from shared import event_handler
   
   @event_handler(SomeEvent)
   async def handle_event(self, event: SomeEvent):
       # Handle cross-domain events
       pass
   ```

### Application Services

1. **Migrate to dependency injection**
   ```python
   # Before
   class MyApplicationService:
       def __init__(self):
           self.repo = MyRepository()
           self.external_service = ExternalService()
   
   # After
   from shared import inject, get_container
   
   class MyApplicationService:
       @inject(get_container())
       def __init__(self, 
                    repo: MyRepository,
                    external_service: ExternalService):
           self.repo = repo
           self.external_service = external_service
   ```

2. **Use stable DTOs**
   ```python
   # Before
   def process_data(self, data_dict: dict):
       # Process raw dictionary
       pass
   
   # After
   from interfaces.dto import DataProcessingRequest
   
   def process_data(self, request: DataProcessingRequest):
       # Process structured DTO
       pass
   ```

### Configuration Packages

1. **Update to use new DI container**
   ```python
   # Before
   def create_services():
       return {
           'service_a': ServiceA(),
           'service_b': ServiceB(),
       }
   
   # After
   from shared import DIContainer, configure_container
   
   def create_services():
       def setup(container):
           container.register_singleton(ServiceA)
           container.register_singleton(ServiceB)
       
       configure_container(setup)
   ```

## Migration Checklist

### Phase 1: Infrastructure Setup
- [ ] Install new shared infrastructure
- [ ] Set up event bus
- [ ] Configure DI container
- [ ] Add boundary validation tools

### Phase 2: Update Imports
- [ ] Replace direct cross-domain imports with interfaces
- [ ] Update DTOs to use shared contracts
- [ ] Add event-driven communication patterns
- [ ] Configure dependency injection

### Phase 3: Update Services
- [ ] Migrate application services to use DI
- [ ] Add event handlers for cross-domain communication
- [ ] Update service interfaces to use patterns
- [ ] Configure service registration

### Phase 4: Update Configuration
- [ ] Migrate to new DI container
- [ ] Update service wiring
- [ ] Configure event subscriptions
- [ ] Set up deployment-specific configurations

### Phase 5: Validation
- [ ] Run boundary validator
- [ ] Execute integration tests
- [ ] Validate event flows
- [ ] Check performance impact

## Common Migration Patterns

### Pattern 1: Request-Response via Events

```python
# Before: Direct method call
result = await other_service.process(data)

# After: Event-driven request-response
await publish_event(ProcessingRequested(data=data, correlation_id="123"))
# Listen for ProcessingCompleted event with matching correlation_id
```

### Pattern 2: Service Discovery via DI

```python
# Before: Manual service location
service = ServiceRegistry.get_service("my_service")

# After: Dependency injection
@inject(get_container())
def __init__(self, my_service: MyService):
    self.my_service = my_service
```

### Pattern 3: Configuration Composition

```python
# Before: Hardcoded dependencies
class AppConfig:
    def setup(self):
        self.db = Database("connection_string")
        self.cache = Redis("redis_url")
        self.service = MyService(self.db, self.cache)

# After: Dependency injection configuration
def configure_app(container):
    container.register_singleton(Database, factory=lambda: Database("connection_string"))
    container.register_singleton(Redis, factory=lambda: Redis("redis_url"))
    container.register_singleton(MyService)  # Auto-injected dependencies
```

## Automated Migration Tools

### 1. Boundary Validator
```bash
# Check current violations
python src/packages/tools/import_boundary_validator/boundary_validator.py

# Generate migration report
python src/packages/tools/import_boundary_validator/boundary_validator.py --format json --output migration-report.json
```

### 2. Import Refactoring Script
```bash
# Auto-fix simple import violations
python scripts/migration/auto_fix_imports.py --package data/quality --dry-run
python scripts/migration/auto_fix_imports.py --package data/quality --apply
```

### 3. DI Migration Helper
```bash
# Generate DI configuration for existing services
python scripts/migration/generate_di_config.py --package ai/anomaly_detection
```

## Testing During Migration

### 1. Unit Tests
```python
# Test with mocked dependencies
from shared import DIContainer
from unittest.mock import Mock

def test_service_with_mocked_dependencies():
    container = DIContainer()
    container.register_singleton(ExternalService, Mock())
    
    service = container.resolve(MyService)
    # Test service behavior
```

### 2. Integration Tests
```python
# Test event flows
from shared import get_event_bus
from interfaces.events import MyEvent

async def test_event_flow():
    event_bus = get_event_bus()
    await event_bus.start()
    
    # Publish event and verify handling
    await event_bus.publish(MyEvent(data="test"))
    
    await event_bus.stop()
```

### 3. Boundary Tests
```python
# Verify no forbidden imports
def test_no_cross_domain_imports():
    validator = ImportBoundaryValidator(Path("."))
    violations = validator.validate_all_packages()
    assert len(violations) == 0, f"Found boundary violations: {violations}"
```

## Performance Considerations

### Event Bus Overhead
- Event publishing adds ~1-2ms latency
- Use priority queues for critical events
- Configure appropriate buffer sizes

### Dependency Injection Overhead
- DI resolution adds ~0.1ms per service
- Use singletons for expensive services
- Cache resolved instances when appropriate

### Memory Usage
- Event bus maintains in-memory queues
- DI container holds singleton instances
- Monitor memory usage in production

## Rollback Strategy

If migration issues occur:

1. **Immediate Rollback**
   ```bash
   git revert <migration-commit>
   ```

2. **Gradual Rollback**
   - Disable event bus temporarily
   - Fall back to direct imports
   - Maintain old and new patterns side-by-side

3. **Monitoring**
   - Track error rates during migration
   - Monitor performance metrics
   - Set up alerting for boundary violations

## Support and Resources

- **Import Guidelines**: [IMPORT_GUIDELINES.md](IMPORT_GUIDELINES.md)
- **Architecture Documentation**: [ARCHITECTURE.md](ARCHITECTURE.md)
- **Examples**: [configurations/examples/](configurations/examples/)
- **Tools**: [tools/import_boundary_validator/](tools/import_boundary_validator/)

## FAQ

**Q: Can I migrate packages incrementally?**
A: Yes, the new patterns are designed for gradual adoption. Start with one package and migrate others over time.

**Q: What happens to existing tests?**
A: Most tests will continue to work. Update test setup to use the new DI container for dependency injection.

**Q: How do I handle circular dependencies?**
A: Use events to break circular dependencies, or restructure code to eliminate cycles.

**Q: What about performance impact?**
A: The overhead is minimal (<1ms per operation). Benefits of loose coupling outweigh small performance costs.

**Q: How do I debug event flows?**
A: Use the event bus metrics and logging. Enable debug mode to trace event propagation.

For additional support, consult the team leads or create an issue in the repository.