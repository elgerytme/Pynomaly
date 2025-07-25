# Developer Onboarding Guide: Package Interaction Framework

Welcome to the package interaction framework! This guide will help you understand and quickly adopt the new architectural patterns for cross-package communication in our monorepo.

## 🚀 Quick Start

### 1. Understanding the Architecture

Our monorepo uses a **domain-driven architecture** with **event-driven communication** between packages:

```
src/packages/
├── ai/              # AI and ML services
├── data/            # Data processing and quality
├── enterprise/      # Business logic and workflows  
├── integrations/    # External system integrations
├── configurations/ # System configuration
├── interfaces/     # Stable contracts (DTOs, Events, Patterns)
└── shared/         # Infrastructure (DI, Event Bus, Utilities)
```

### 2. Core Principles

- **🔒 No Direct Dependencies**: Packages don't import each other directly
- **📡 Event-Driven Communication**: Use events for cross-domain interactions
- **💉 Dependency Injection**: Services are resolved through DI container
- **🎯 Stable Interfaces**: Use DTOs and events from `interfaces/` package
- **📊 Observability**: Built-in monitoring and health checks

### 3. Your First Integration

Here's how to integrate a new service:

```python
# 1. Import stable contracts
from interfaces.dto import DetectionRequest, DetectionResult
from interfaces.events import AnomalyDetected, EventPriority
from interfaces.patterns import Service

# 2. Import shared infrastructure
from shared import get_container, publish_event, event_handler

# 3. Implement your service
class MyDetectionService(Service):
    async def execute(self, request: DetectionRequest) -> DetectionResult:
        # Your business logic here
        result = DetectionResult(...)
        
        # Publish events for other domains
        if result.anomalies_count > 0:
            event = AnomalyDetected(
                dataset_id=request.dataset_id,
                anomaly_count=result.anomalies_count,
                severity="high",
                priority=EventPriority.HIGH
            )
            await publish_event(event)
        
        return result

# 4. Register with DI container
def configure_services(container):
    container.register_singleton(MyDetectionService)

# 5. Subscribe to events from other domains
class MyEventHandler:
    @event_handler(DataQualityCheckCompleted)
    async def handle_quality_check(self, event):
        # React to events from data quality domain
        pass
```

## 📚 Core Concepts

### Event-Driven Communication

Events are the primary way packages communicate:

```python
# Publishing events
await publish_event(AnomalyDetected(
    dataset_id="customer_data",
    anomaly_count=5,
    severity="medium"
))

# Subscribing to events
@event_handler(AnomalyDetected)
async def handle_anomaly(event: AnomalyDetected):
    logger.info(f"Anomaly detected in {event.dataset_id}")
```

### Dependency Injection

Services are managed through DI:

```python
# Registration
container.register_singleton(DataQualityService)
container.register_transient(DetectionService)
container.register_scoped(AnalyticsService)

# Resolution
quality_service = container.resolve(DataQualityService)
```

### Stable DTOs

Use DTOs for data exchange:

```python
from interfaces.dto import DetectionRequest, DataQualityRequest

# Create requests
detection_request = DetectionRequest(
    dataset_id="sales_data",
    algorithm="isolation_forest",
    parameters={"contamination": 0.1}
)

# Services process DTOs
result = await detection_service.execute(detection_request)
```

## 🛠️ Development Workflow

### Step 1: Understand Your Domain

1. Identify which domain your feature belongs to:
   - **AI**: Machine learning, model training/serving
   - **Data**: Quality checks, profiling, transformations
   - **Enterprise**: Business workflows, reporting
   - **Integrations**: External APIs, third-party services

### Step 2: Design Your Service

```python
from interfaces.patterns import Service
from typing import Dict, Any

class MyService(Service):
    async def execute(self, request: MyRequest) -> MyResult:
        """Main service logic"""
        pass
    
    async def validate_request(self, request: MyRequest) -> bool:
        """Validate input"""
        return True
    
    def get_service_info(self) -> Dict[str, Any]:
        """Service metadata"""
        return {"name": "MyService", "version": "1.0"}
```

### Step 3: Define Events (if needed)

```python
from interfaces.events import DomainEvent, EventPriority
from dataclasses import dataclass
from datetime import datetime

@dataclass
class MyDomainEvent(DomainEvent):
    """Event specific to your domain"""
    entity_id: str
    status: str
    metadata: Dict[str, Any]
    
    def __post_init__(self):
        if not self.event_id:
            self.event_id = f"my_event_{self.entity_id}"
        if not self.event_type:
            self.event_type = "MyDomainEvent"
```

### Step 4: Configure Dependencies

```python
def configure_my_domain(container):
    """Configure services for your domain"""
    # Core services
    container.register_singleton(MyService)
    container.register_transient(MyRepository)
    
    # External dependencies
    container.register_singleton(
        ExternalApiClient,
        factory=lambda: ExternalApiClient(api_key=os.getenv("API_KEY"))
    )
```

### Step 5: Set Up Event Handling

```python
from shared import get_event_bus

class MyEventHandler:
    def __init__(self, my_service: MyService):
        self.my_service = my_service
        
        # Subscribe to relevant events
        event_bus = get_event_bus()
        event_bus.subscribe(DataQualityCheckCompleted, self.handle_quality_completed)
    
    @event_handler(DataQualityCheckCompleted)
    async def handle_quality_completed(self, event: DataQualityCheckCompleted):
        if event.status == "passed":
            # Trigger your domain logic
            await self.my_service.process_quality_passed(event.dataset_id)
```

## 🏗️ Architecture Patterns

### 1. Request-Response Pattern

For synchronous operations:

```python
# Direct service call through DI
quality_service = container.resolve(DataQualityService)
result = await quality_service.execute(quality_request)
```

### 2. Event-Driven Pattern

For asynchronous, decoupled operations:

```python
# Publish event
await publish_event(DataProcessingStarted(dataset_id="abc123"))

# Subscribe to events
@event_handler(DataProcessingCompleted)
async def handle_processing_complete(event):
    # Start next step in workflow
    pass
```

### 3. Workflow Pattern

For complex multi-step processes:

```python
class DataPipelineWorkflow:
    async def execute(self, dataset_id: str):
        # Step 1: Quality check
        quality_result = await self.quality_service.execute(
            DataQualityRequest(dataset_id=dataset_id)
        )
        
        # Step 2: Publish quality event
        await publish_event(DataQualityCheckCompleted(
            dataset_id=dataset_id,
            status=quality_result.status,
            overall_score=quality_result.overall_score
        ))
        
        # Step 3: Wait for downstream processing
        # (handled by event subscribers)
```

## 🔧 Testing Your Integration

### Unit Tests

```python
import pytest
from unittest.mock import Mock, AsyncMock

class TestMyService:
    @pytest.mark.asyncio
    async def test_service_execution(self):
        # Setup
        service = MyService()
        request = MyRequest(data="test")
        
        # Execute
        result = await service.execute(request)
        
        # Assert
        assert result.status == "completed"
    
    @pytest.mark.asyncio
    async def test_event_handling(self):
        # Test event publishing and handling
        event_bus = InMemoryEventBus()
        events_received = []
        
        event_bus.subscribe(MyEvent, lambda e: events_received.append(e))
        
        await event_bus.publish(MyEvent(entity_id="test"))
        
        assert len(events_received) == 1
```

### Integration Tests

```python
class TestWorkflowIntegration:
    @pytest.mark.asyncio
    async def test_end_to_end_workflow(self):
        # Setup DI container
        container = DIContainer()
        container.register_singleton(MyService)
        
        # Setup event bus
        event_bus = InMemoryEventBus()
        
        # Execute workflow
        service = container.resolve(MyService)
        result = await service.execute(request)
        
        # Verify workflow completion
        assert result.status == "completed"
```

## 📊 Monitoring and Observability

### Built-in Metrics

The framework provides automatic monitoring:

```python
from shared.observability import get_dashboard

# Get health status
dashboard = get_dashboard()
status_report = await dashboard.generate_status_report()
print(status_report)

# Get detailed metrics
metrics = await dashboard.generate_json_metrics()
```

### Custom Metrics

Add your own metrics:

```python
from shared.observability import get_metrics_collector

metrics = get_metrics_collector()

# Counter metrics
metrics.increment_counter("my_service.requests_processed")

# Gauge metrics
metrics.set_gauge("my_service.active_connections", 42)

# Histogram metrics
metrics.record_histogram("my_service.processing_time_ms", 150.5)
```

### Health Checks

Implement health checks for your services:

```python
from interfaces.patterns import HealthCheck

class MyServiceHealthCheck(HealthCheck):
    def __init__(self, my_service: MyService):
        self.my_service = my_service
    
    async def check_health(self) -> Dict[str, Any]:
        try:
            # Check service dependencies
            await self.my_service.ping()
            return {"status": "healthy", "service": "MyService"}
        except Exception as e:
            return {"status": "unhealthy", "error": str(e)}
    
    def get_component_name(self) -> str:
        return "my_service"
```

## 🚫 Common Pitfalls and How to Avoid Them

### ❌ Don't: Direct Package Imports

```python
# DON'T DO THIS
from data.anomaly_detection.services import DetectionService
from ai.mlops.services import ModelService
```

### ✅ Do: Use Dependency Injection

```python
# DO THIS
from interfaces.patterns import Service

class MyService(Service):
    def __init__(self, detection_service: DetectionService):
        self.detection_service = detection_service
```

### ❌ Don't: Tight Coupling Through Shared State

```python
# DON'T DO THIS
global_shared_state = {}

class ServiceA:
    def process(self):
        global_shared_state['result'] = "data"

class ServiceB:
    def process(self):
        return global_shared_state['result']
```

### ✅ Do: Event-Driven Communication

```python
# DO THIS
class ServiceA:
    async def process(self):
        await publish_event(DataProcessed(result="data"))

class ServiceB:
    @event_handler(DataProcessed)
    async def handle_data_processed(self, event):
        return event.result
```

### ❌ Don't: Blocking Operations in Event Handlers

```python
# DON'T DO THIS
@event_handler(MyEvent)
async def slow_handler(event):
    time.sleep(10)  # Blocks event processing
    heavy_computation()  # Synchronous and slow
```

### ✅ Do: Async Non-Blocking Operations

```python
# DO THIS
@event_handler(MyEvent)
async def fast_handler(event):
    await asyncio.sleep(0.01)  # Non-blocking
    await async_computation()  # Async operation
```

## 🎯 Best Practices

### 1. Design Events for Subscribers

```python
# Good: Rich events with context
@dataclass
class CustomerOrderCompleted(DomainEvent):
    order_id: str
    customer_id: str
    total_amount: float
    items: List[Dict[str, Any]]
    payment_method: str
    
    # Include all data subscribers might need
    def get_order_summary(self) -> str:
        return f"Order {self.order_id}: ${self.total_amount}"
```

### 2. Use Appropriate Event Priorities

```python
# Critical: System failures, security issues
SystemFailure(priority=EventPriority.CRITICAL)

# High: Business-critical operations
OrderCompleted(priority=EventPriority.HIGH)

# Normal: Regular business events
UserRegistered(priority=EventPriority.NORMAL)

# Low: Analytics, logging
PageViewed(priority=EventPriority.LOW)
```

### 3. Implement Graceful Error Handling

```python
class RobustService(Service):
    async def execute(self, request):
        try:
            return await self.process_request(request)
        except ValidationError as e:
            logger.warning(f"Validation failed: {e}")
            raise
        except ExternalServiceError as e:
            logger.error(f"External service unavailable: {e}")
            # Return degraded response or retry
            return self.create_fallback_response(request)
        except Exception as e:
            logger.error(f"Unexpected error: {e}")
            # Publish error event for monitoring
            await publish_event(ServiceError(
                service_name=self.__class__.__name__,
                error_message=str(e),
                request_id=request.id
            ))
            raise
```

### 4. Use Dependency Injection Scopes Appropriately

```python
# Singleton: Expensive to create, stateless
container.register_singleton(DatabaseConnection)
container.register_singleton(EventBus)

# Transient: Lightweight, stateful
container.register_transient(RequestProcessor)
container.register_transient(ValidationService)

# Scoped: Per-request/per-operation state
container.register_scoped(UserContext)
container.register_scoped(TransactionManager)
```

## 🔍 Debugging and Troubleshooting

### View Event Flow

```python
# Enable event tracing
import logging
logging.getLogger("shared.event_bus").setLevel(logging.DEBUG)

# Monitor event metrics
from shared.observability import get_dashboard
dashboard = get_dashboard()
report = await dashboard.generate_status_report()
```

### Check Service Health

```python
from shared.observability import get_health_check

health_check = get_health_check()
health_status = await health_check.check_health()
print(f"Framework status: {health_status['status']}")
```

### Validate Package Boundaries

```bash
# Run boundary validation tool
python src/packages/tools/import_boundary_validator/boundary_validator.py
```

## 📖 Additional Resources

- **Architecture Documentation**: [`ARCHITECTURE.md`](./ARCHITECTURE.md)
- **Import Guidelines**: [`IMPORT_GUIDELINES.md`](./IMPORT_GUIDELINES.md)
- **Real-World Examples**: [`configurations/examples/real_world_integration.py`](./configurations/examples/real_world_integration.py)
- **Testing Patterns**: [`shared/tests/test_interaction_patterns.py`](./shared/tests/test_interaction_patterns.py)
- **Validation Tools**: [`tools/import_boundary_validator/`](./tools/import_boundary_validator/)

## 🆘 Getting Help

### Quick Reference

```python
# Import essentials
from interfaces.dto import DetectionRequest, DataQualityRequest
from interfaces.events import AnomalyDetected, DataQualityCheckCompleted
from interfaces.patterns import Service, Repository, HealthCheck
from shared import get_container, get_event_bus, publish_event, event_handler

# Service implementation
class MyService(Service):
    async def execute(self, request): ...
    async def validate_request(self, request): ...
    def get_service_info(self): ...

# Event handling
@event_handler(SomeEvent)
async def handle_event(event): ...

# DI registration
container.register_singleton(MyService)
service = container.resolve(MyService)

# Event publishing
await publish_event(MyEvent(...))
```

### Common Commands

```bash
# Run tests
python -m pytest src/packages/shared/tests/

# Validate boundaries
python src/packages/tools/import_boundary_validator/boundary_validator.py

# Check health
python -c "from shared.observability import get_dashboard; import asyncio; print(asyncio.run(get_dashboard().generate_status_report()))"
```

### Troubleshooting Checklist

- [ ] Are you importing from `interfaces/` for DTOs and events?
- [ ] Is your service registered in the DI container?
- [ ] Are event handlers properly decorated with `@event_handler`?
- [ ] Is the event bus started before publishing events?
- [ ] Are you using appropriate event priorities?
- [ ] Are your services implementing the `Service` interface correctly?

---

**Welcome to the team!** 🎉 

This framework is designed to make cross-package communication clean, maintainable, and observable. Start with simple integrations and gradually adopt more advanced patterns as you become comfortable with the architecture.

For questions or issues, check the troubleshooting section above or review the example implementations in the `configurations/examples/` directory.