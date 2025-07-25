# Package Import Guidelines

This document provides comprehensive guidelines for cross-package imports in the monorepo, implementing the architectural recommendations for proper package interactions.

## Quick Reference

### ✅ Allowed Import Patterns

```python
# Via interfaces package (stable contracts)
from interfaces.dto import DetectionRequest, DataQualityResult
from interfaces.events import AnomalyDetected, DataQualityCheckCompleted
from interfaces.patterns import Repository, Service, EventBus

# Via shared package (common infrastructure)
from shared import get_event_bus, get_container, publish_event
from shared.dependency_injection import DIContainer, inject

# Application layer imports (within same domain)
from anomaly_detection.application.services.detection_service import DetectionService

# Configuration composition (only in configurations/)
from configurations.basic.mlops_basic import create_basic_mlops_config
```

### ❌ Forbidden Import Patterns

```python
# Direct domain-to-domain imports
from ai.machine_learning.domain.entities import Model  # From data package
from data.quality.domain.services import QualityService  # From ai package

# Domain importing enterprise/integrations directly
from enterprise.auth.domain import AuthService  # From domain package
from integrations.mlflow import MLflowAdapter  # From domain package

# Circular dependencies
from data.quality import DataQualityService  # From anomaly_detection package
from anomaly_detection import DetectionService  # From data.quality package
```

## Architecture Overview

```
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│  configurations │    │   enterprise    │    │  integrations   │
│                 │    │                 │    │                 │
│ • basic/        │    │ • auth/         │    │ • mlops/        │
│ • enterprise/   │    │ • governance/   │    │ • monitoring/   │
│ • custom/       │    │ • security/     │    │ • cloud/        │
└─────────────────┘    └─────────────────┘    └─────────────────┘
         │                       │                       │
         └───────────────────────┼───────────────────────┘
                                 │
         ┌─────────────────────────────────────────────────┐
         │                   Imports                       │
         └─────────────────────────────────────────────────┘
                                 │
    ┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
    │   interfaces    │    │     shared      │    │ domain packages │
    │                 │    │                 │    │                 │
    │ • dto.py        │    │ • event_bus.py  │    │ • ai/           │
    │ • events.py     │    │ • dependency_   │    │ • data/         │
    │ • patterns.py   │    │   injection.py  │    │                 │
    └─────────────────┘    └─────────────────┘    └─────────────────┘
```

## Detailed Guidelines by Package Type

### 1. Domain Packages (`ai/`, `data/`)

**Purpose:** Pure business logic for specific domains.

**Can Import:**
- `interfaces/` - For stable contracts and DTOs
- `shared/` - For common utilities only

**Cannot Import:**
- Other domain packages
- `enterprise/`
- `integrations/`
- `configurations/`

**Example:**
```python
# In data/quality/application/services/quality_service.py
from interfaces.dto import DataQualityRequest, DataQualityResult
from interfaces.events import DataQualityCheckCompleted
from shared import publish_event

class DataQualityService:
    async def execute(self, request: DataQualityRequest) -> DataQualityResult:
        # Business logic here
        result = DataQualityResult(...)
        
        # Publish event using shared infrastructure
        event = DataQualityCheckCompleted(...)
        await publish_event(event)
        
        return result
```

### 2. Interfaces Package

**Purpose:** Stable contracts for cross-domain communication.

**Can Import:**
- Standard library only
- Type annotations and dataclasses

**Cannot Import:**
- Any other packages in the monorepo

**Example:**
```python
# In interfaces/dto.py
from dataclasses import dataclass
from datetime import datetime
from typing import Dict, List, Optional

@dataclass
class DetectionRequest:
    dataset_id: str
    algorithm: str
    parameters: Dict[str, Any]
```

### 3. Shared Package

**Purpose:** Common infrastructure and utilities.

**Can Import:**
- `interfaces/` - For implementing abstract patterns
- External libraries for infrastructure

**Cannot Import:**
- Domain packages
- `enterprise/`
- `integrations/`
- `configurations/`

**Example:**
```python
# In shared/event_bus.py
from interfaces.events import DomainEvent, EventBus
from interfaces.patterns import EventHandler

class DistributedEventBus(EventBus):
    # Implementation using interfaces
    pass
```

### 4. Enterprise Package

**Purpose:** Cross-cutting enterprise services.

**Can Import:**
- `interfaces/` - For contracts
- `shared/` - For infrastructure
- External enterprise libraries

**Cannot Import:**
- Domain packages
- `integrations/`
- `configurations/`

**Example:**
```python
# In enterprise/auth/application/auth_service.py
from interfaces.patterns import Service
from shared import get_container

class EnterpriseAuthService(Service):
    # Enterprise authentication logic
    pass
```

### 5. Integrations Package

**Purpose:** External system connectors.

**Can Import:**
- `interfaces/` - For contracts
- `shared/` - For infrastructure
- External integration libraries

**Cannot Import:**
- Domain packages
- `enterprise/`
- `configurations/`

**Example:**
```python
# In integrations/mlops/mlflow/adapter.py
from interfaces.patterns import AntiCorruptionLayer
from interfaces.dto import ModelTrainingResult

class MLflowAdapter(AntiCorruptionLayer):
    async def translate_incoming(self, mlflow_data) -> ModelTrainingResult:
        # Translate MLflow format to internal DTO
        pass
```

### 6. Configurations Package

**Purpose:** Application composition and wiring.

**Can Import:**
- Everything (composition root)
- Domain packages
- `enterprise/`
- `integrations/`
- `shared/`
- `interfaces/`

**Example:**
```python
# In configurations/enterprise/mlops_enterprise.py
from data.quality.application.services import DataQualityService
from ai.machine_learning.application.services import ModelTrainingService
from enterprise.auth.application import EnterpriseAuthService
from integrations.mlflow import MLflowAdapter
from shared import DIContainer, configure_container

def create_enterprise_config():
    def setup(container: DIContainer):
        # Wire everything together
        container.register_singleton(DataQualityService)
        container.register_singleton(ModelTrainingService)
        container.register_singleton(EnterpriseAuthService)
        container.register_singleton(MLflowAdapter)
    
    configure_container(setup)
```

## Import Layers by Architecture

### Application Layer (Recommended for Cross-Package Imports)

The **application layer** is the best practice location for cross-package imports:

```python
# In anomaly_detection/application/services/integrated_detection_service.py
from interfaces.dto import DataQualityRequest, DataQualityResult
from interfaces.events import DataQualityCheckCompleted
from shared import get_event_bus, event_handler

class IntegratedDetectionService:
    def __init__(self):
        self.event_bus = get_event_bus()
        self.event_bus.subscribe(DataQualityCheckCompleted, self._on_quality_check)
    
    @event_handler(DataQualityCheckCompleted)
    async def _on_quality_check(self, event: DataQualityCheckCompleted):
        # React to data quality events
        if event.overall_score < 0.8:
            # Skip anomaly detection for poor quality data
            pass
```

### Domain Layer (Minimal Imports Only)

```python
# In anomaly_detection/domain/entities/detection.py
from interfaces.dto import BaseDTO  # OK - stable contract
from shared.base_classes import Entity  # OK - common infrastructure

class Detection(Entity):
    # Pure domain logic with minimal external dependencies
    pass
```

### Infrastructure Layer (External Dependencies)

```python
# In anomaly_detection/infrastructure/adapters/database_adapter.py
from interfaces.patterns import Repository
from shared import get_container
import sqlalchemy  # External dependency OK

class DatabaseDetectionRepository(Repository):
    # Infrastructure concerns
    pass
```

## Cross-Domain Communication Patterns

### 1. Event-Driven Communication (Recommended)

```python
# Publisher (data quality package)
from interfaces.events import DataQualityCheckCompleted
from shared import publish_event

async def complete_quality_check(result):
    event = DataQualityCheckCompleted(
        dataset_id=result.dataset_id,
        status=result.status,
        overall_score=result.overall_score,
        quality_result=result
    )
    await publish_event(event)

# Subscriber (anomaly detection package)
from interfaces.events import DataQualityCheckCompleted
from shared import subscribe_to_event

@event_handler(DataQualityCheckCompleted)
async def handle_quality_check(event: DataQualityCheckCompleted):
    if event.status == "failed":
        # Skip anomaly detection
        pass
    else:
        # Proceed with anomaly detection
        pass
```

### 2. Dependency Injection (For Application Services)

```python
# In application layer
from interfaces.patterns import Service
from shared import inject, get_container

class WorkflowService:
    @inject(get_container())
    def __init__(self, 
                 quality_service: DataQualityService,
                 detection_service: AnomalyDetectionService):
        self.quality_service = quality_service
        self.detection_service = detection_service
    
    async def run_workflow(self, dataset_id: str):
        # Use injected services
        quality_result = await self.quality_service.execute(...)
        detection_result = await self.detection_service.execute(...)
```

### 3. Anti-Corruption Layer (For External Integrations)

```python
# In integrations package
from interfaces.patterns import AntiCorruptionLayer
from interfaces.dto import ModelTrainingResult

class MLflowAntiCorruptionLayer(AntiCorruptionLayer):
    async def translate_incoming(self, mlflow_run) -> ModelTrainingResult:
        # Translate MLflow format to internal format
        return ModelTrainingResult(
            model_id=mlflow_run.info.run_id,
            status="completed" if mlflow_run.info.status == "FINISHED" else "failed",
            training_metrics=mlflow_run.data.metrics
        )
```

## Import Surface Minimization

### Use Interfaces for Stability

```python
# Instead of importing concrete classes
from data.quality.application.services.quality_service import ConcreteQualityService  # ❌

# Import via interfaces
from interfaces.dto import DataQualityRequest, DataQualityResult  # ✅
from interfaces.patterns import Service  # ✅
```

### Use Dependency Injection Instead of Direct Imports

```python
# Instead of direct imports
from other_package.service import SomeService  # ❌
service = SomeService()

# Use dependency injection
from shared import get_container  # ✅
container = get_container()
service = container.resolve(SomeService)
```

### Use Event Bus for Loose Coupling

```python
# Instead of direct method calls
from other_package.service import SomeService  # ❌
service = SomeService()
await service.do_something()

# Use events for loose coupling
from shared import publish_event  # ✅
from interfaces.events import SomethingRequested
await publish_event(SomethingRequested(...))
```

## Enforcement and Validation

### 1. Automated Import Checking

Use the provided domain boundary detector:

```bash
python src/packages/tools/domain_boundary_detector/cli.py --check-imports
```

### 2. Pre-commit Hooks

Add to `.pre-commit-config.yaml`:

```yaml
repos:
  - repo: local
    hooks:
      - id: check-package-boundaries
        name: Check Package Import Boundaries
        entry: python src/packages/tools/domain_boundary_detector/cli.py
        language: system
        pass_filenames: false
```

### 3. CI/CD Pipeline Integration

```yaml
# In GitHub Actions or similar
- name: Validate Package Boundaries
  run: |
    python src/packages/tools/domain_boundary_detector/cli.py --strict
    if [ $? -ne 0 ]; then
      echo "Package boundary violations detected!"
      exit 1
    fi
```

## Migration Guide

### From Direct Imports to Interface-Based

**Before:**
```python
from data.quality.services import DataQualityService
from ai.anomaly_detection.services import DetectionService

class WorkflowService:
    def __init__(self):
        self.quality_service = DataQualityService()
        self.detection_service = DetectionService()
```

**After:**
```python
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

### From Direct Method Calls to Events

**Before:**
```python
# In anomaly detection service
from data.quality.services import DataQualityService

class DetectionService:
    def __init__(self):
        self.quality_service = DataQualityService()
    
    async def detect(self, dataset_id: str):
        quality_result = await self.quality_service.check_quality(dataset_id)
        # Use quality result
```

**After:**
```python
# In detection service - publish event
from interfaces.events import QualityCheckRequested
from shared import publish_event

class DetectionService:
    async def detect(self, dataset_id: str):
        # Request quality check via event
        await publish_event(QualityCheckRequested(dataset_id=dataset_id))

# In quality service - handle event
from interfaces.events import QualityCheckRequested, QualityCheckCompleted
from shared import event_handler

class DataQualityService:
    @event_handler(QualityCheckRequested)
    async def handle_quality_check_request(self, event: QualityCheckRequested):
        result = await self.check_quality(event.dataset_id)
        await publish_event(QualityCheckCompleted(result=result))
```

## Best Practices Summary

1. **Use the Application Layer** for cross-package imports
2. **Import via Interfaces** for stable contracts
3. **Use Shared Infrastructure** for common utilities
4. **Use Events** for loose coupling between domains
5. **Use Dependency Injection** for testable code
6. **Limit Import Surface** to minimize coupling
7. **Configurations Handle Wiring** - not individual packages
8. **Never Import Across Domain Boundaries** directly

Following these guidelines ensures maintainable, testable, and loosely coupled code that can evolve independently while maintaining proper architectural boundaries.