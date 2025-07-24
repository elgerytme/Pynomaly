# 🚀 Developer Migration Guide

**Domain-Driven Architecture Migration Guide for Anomaly Detection Package**

---

## 📋 Overview

This guide helps developers understand and work with the new **Domain-Driven Design (DDD)** architecture implemented in the anomaly detection package. The migration maintains 100% backward compatibility while providing a foundation for better scalability and maintainability.

## 🎯 Quick Start

**TL;DR:** Your existing code continues to work without changes. Read the [Key Changes](#key-changes) section to understand the new architecture and consider adopting new patterns for future development.

---

## 🏗️ Architecture Changes

### Before: Layered Architecture
```
📦 anomaly_detection/
├── api/                    # REST API endpoints
├── cli/                    # Command-line interface
├── domain/                 # Domain logic and entities
├── infrastructure/         # Technical infrastructure
└── presentation/           # Web UI and external interfaces
```

### After: Domain-Driven Design
```
📦 Anomaly Detection Platform
├── 🤖 AI Domain
│   ├── machine_learning/     # Core ML algorithms and training
│   └── mlops/               # Model lifecycle and experiment tracking  
├── 📊 Data Domain
│   └── processing/          # Data entities and processing pipelines
├── 🔧 Shared Infrastructure
│   ├── infrastructure/      # Configuration, logging, security
│   └── observability/       # Monitoring, metrics, dashboards
└── 🎯 Application Layer      # Anomaly detection business logic
```

---

## 🔄 Key Changes

### 1. **100% Backward Compatibility** ✅

All existing imports and APIs continue to work:

```python
# ✅ These still work (no changes needed)
from anomaly_detection.domain.services import DetectionService
from anomaly_detection.infrastructure.repositories import ModelRepository
from anomaly_detection.domain.entities import DetectionResult

# Your existing code runs unchanged
service = DetectionService()
result = service.detect_anomalies(data, algorithm='iforest')
```

### 2. **New Domain-Aware Imports** (Optional)

For new features, consider using domain-aware imports:

```python
# 🆕 New domain-aware imports (optional)
from ai.machine_learning.algorithms import IsolationForestAdapter
from ai.mlops.services import ExperimentTrackingService
from data.processing.entities import Dataset
from shared.observability.metrics import MetricsCollector
```

### 3. **Fallback Import Patterns**

The codebase uses fallback patterns internally:

```python
# Internal pattern (you don't need to write this)
try:
    from ai.mlops.services import MLOpsService
except ImportError:
    from anomaly_detection.domain.services import MLOpsService
```

---

## 🛠️ Developer Workflow Changes

### For Existing Development

**No changes required!** Continue using your current workflow:

```python
# Continue using existing patterns
from anomaly_detection import AnomalyDetector

detector = AnomalyDetector()
results = detector.detect(your_data)
```

### For New Features

Consider domain boundaries when adding new features:

1. **AI/ML Features** → Place in `ai/machine_learning/` or `ai/mlops/`
2. **Data Features** → Place in `data/processing/`  
3. **Infrastructure** → Place in `shared/infrastructure/`
4. **Monitoring** → Place in `shared/observability/`
5. **Business Logic** → Place in application layer

---

## 📚 Development Patterns

### 1. Service Layer Development

```python
# ✅ Recommended: Use existing service layer
from anomaly_detection.domain.services import DetectionService

class CustomDetectionService:
    def __init__(self):
        self.base_service = DetectionService()
    
    def custom_detection(self, data):
        # Your custom logic here
        result = self.base_service.detect_anomalies(data, 'iforest')
        return self.post_process(result)
```

### 2. Algorithm Extension

```python
# ✅ Recommended: Extend existing adapters
from anomaly_detection.infrastructure.adapters.algorithms import SklearnAdapter

class CustomAlgorithmAdapter(SklearnAdapter):
    def __init__(self, **params):
        super().__init__('custom_algorithm', **params)
    
    def fit(self, data):
        # Your custom training logic
        pass
```

### 3. API Endpoint Development

```python
# ✅ Recommended: Follow existing patterns
from fastapi import APIRouter
from anomaly_detection.domain.services import DetectionService

router = APIRouter()
detection_service = DetectionService()

@router.post("/custom/detect")
async def custom_detect(request: CustomRequest):
    # Use existing services
    result = detection_service.detect_anomalies(
        request.data, 
        request.algorithm
    )
    return result.to_dict()
```

---

## 🔍 Domain Boundaries Guide

### When to Place Code in Each Domain

#### 🤖 AI/Machine Learning (`ai/machine_learning/`)
- **Use for:** Algorithm implementations, model training, ML utilities
- **Examples:** New detection algorithms, model evaluation, feature engineering
- **Pattern:** Pure ML functionality without business logic

```python
# Example: New algorithm implementation
from ai.machine_learning.base import BaseDetector

class DeepAnomalyDetector(BaseDetector):
    def fit(self, data):
        # ML training logic
        pass
```

#### 🔬 AI/MLOps (`ai/mlops/`)
- **Use for:** Model lifecycle, experiment tracking, deployment
- **Examples:** Model registry, experiment logging, A/B testing
- **Pattern:** ML operations and lifecycle management

```python
# Example: Experiment tracking
from ai.mlops.services import ExperimentTracker

tracker = ExperimentTracker()
tracker.log_experiment(model, metrics, parameters)
```

#### 📊 Data Processing (`data/processing/`)
- **Use for:** Data entities, validation, transformation
- **Examples:** Data models, preprocessing pipelines, validation rules
- **Pattern:** Data-centric functionality

```python
# Example: Data validation
from data.processing.validators import DataValidator

validator = DataValidator()
is_valid = validator.validate_dataset(data)
```

#### 🔧 Shared Infrastructure (`shared/infrastructure/`)
- **Use for:** Configuration, logging, security, utilities
- **Examples:** Settings management, authentication, caching
- **Pattern:** Cross-cutting technical concerns

```python
# Example: Configuration
from shared.infrastructure.config import Settings

settings = Settings()
api_key = settings.get_api_key()
```

#### 📈 Shared Observability (`shared/observability/`)
- **Use for:** Monitoring, metrics, health checks, dashboards
- **Examples:** Performance metrics, system health, alerting
- **Pattern:** System observability and monitoring

```python
# Example: Metrics collection
from shared.observability.metrics import MetricsCollector

metrics = MetricsCollector()
metrics.record_detection_time(duration)
```

#### 🎯 Application Layer (`anomaly_detection/application/`)
- **Use for:** Business logic, use cases, orchestration
- **Examples:** Detection workflows, business rules, API coordination
- **Pattern:** Business logic that coordinates across domains

```python
# Example: Business workflow
from anomaly_detection.application.services import DetectionWorkflow

workflow = DetectionWorkflow()
result = workflow.execute_detection_pipeline(data, config)
```

---

## 🧪 Testing Strategies

### 1. Unit Testing

```python
# ✅ Test individual components
import pytest
from anomaly_detection.domain.services import DetectionService

def test_detection_service():
    service = DetectionService()
    result = service.detect_anomalies(test_data, 'iforest')
    assert result.success
    assert len(result.anomalies) > 0
```

### 2. Integration Testing

```python
# ✅ Test domain interactions
def test_full_detection_pipeline():
    # Test that domains work together correctly
    detector = AnomalyDetector()
    result = detector.detect(data)
    assert result.anomaly_count >= 0
```

### 3. Domain Testing

```python
# ✅ Test domain boundaries
def test_ai_ml_domain():
    from ai.machine_learning.algorithms import IsolationForestAdapter
    adapter = IsolationForestAdapter()
    # Test ML domain independently
```

---

## 🚨 Migration Checklist

### For Existing Developers

- [ ] **Read this guide** - Understand new architecture
- [ ] **Run existing tests** - Ensure everything still works  
- [ ] **No immediate changes** - Continue current development
- [ ] **Plan future features** - Consider domain boundaries

### For New Features

- [ ] **Identify domain** - Determine which domain your feature belongs to
- [ ] **Use domain imports** - Consider new import patterns
- [ ] **Follow patterns** - Use established patterns in the domain
- [ ] **Test boundaries** - Ensure proper domain isolation

### For Architecture Improvements

- [ ] **Review current code** - Identify domain violations
- [ ] **Plan refactoring** - Move code to appropriate domains
- [ ] **Update imports** - Gradually adopt new import patterns
- [ ] **Document decisions** - Create ADRs for architectural choices

---

## 🔧 Development Environment Setup

### 1. Install Dependencies

```bash
# Install with all dependencies
pip install -e ".[dev,test,docs]"

# Or using poetry
poetry install --extras "dev test docs"
```

### 2. Configure IDE

**VS Code Settings:**
```json
{
    "python.defaultInterpreterPath": "./venv/bin/python",
    "python.testing.pytestEnabled": true,
    "python.linting.enabled": true,
    "python.formatting.provider": "black"
}
```

**PyCharm Settings:**
- Set source root to `src/anomaly_detection`
- Configure interpreter to use project virtual environment
- Enable pytest as test runner

### 3. Pre-commit Hooks

```bash
# Install pre-commit hooks
pre-commit install

# Run all hooks
pre-commit run --all-files
```

---

## 🐛 Troubleshooting

### Common Issues

#### Import Errors
```python
# ❌ If you see: ModuleNotFoundError: No module named 'ai.mlops'
# ✅ Solution: Use fallback imports or existing imports
from anomaly_detection.domain.services import MLOpsService
```

#### Test Configuration
```bash
# ❌ If pytest can't find modules
# ✅ Solution: Set PYTHONPATH
export PYTHONPATH=/path/to/src:$PYTHONPATH
pytest tests/
```

#### Missing Dependencies
```bash
# ❌ If domain modules are missing
# ✅ Solution: Install in development mode
pip install -e .
```

### Getting Help

1. **Check documentation** - [API Reference](api.md), [Architecture](architecture.md)
2. **Review examples** - See `examples/` directory
3. **Run validation** - Use `test_comprehensive_validation.py`
4. **Check logs** - Enable debug logging for more details

---

## 📈 Performance Considerations

### 1. Import Performance

The fallback import patterns add minimal overhead (~1ms per import). For performance-critical applications:

```python
# ✅ Use direct imports when possible
from anomaly_detection.domain.services import DetectionService

# ⚠️ Avoid nested imports in loops
for data_batch in batches:
    # Don't import inside loops
    result = service.detect(data_batch)
```

### 2. Memory Usage

Domain boundaries help with memory management:

```python
# ✅ Import only what you need
from anomaly_detection.domain.services import DetectionService
# Don't import: from anomaly_detection import *

# ✅ Use lazy loading for optional features
def get_mlops_service():
    try:
        from ai.mlops.services import MLOpsService
        return MLOpsService()
    except ImportError:
        return None
```

---

## 🚀 Best Practices

### 1. **Gradual Adoption**
- Continue using existing patterns
- Adopt new patterns for new features
- Refactor gradually over time

### 2. **Domain Respect**
- Don't create circular dependencies between domains
- Use dependency injection for cross-domain communication
- Keep domains loosely coupled

### 3. **Testing Strategy**
- Test each domain independently
- Test integration points between domains
- Maintain high test coverage

### 4. **Documentation**
- Document architectural decisions
- Update API documentation for new features
- Maintain migration notes

---

## 📚 Resources

### Architecture Documentation
- [API Reference](api.md) - Complete API documentation
- [Architecture Guide](architecture.md) - Detailed architecture overview
- [Migration Report](../MIGRATION_VALIDATION_REPORT.md) - Validation results

### Code Examples
- [`examples/`](../examples/) - Working code examples
- [`test_comprehensive_validation.py`](../test_comprehensive_validation.py) - Validation tests
- [`docs/templates/`](templates/) - Template code for common patterns

### Additional Guides
- [Getting Started](getting-started/) - Quick start tutorials
- [Algorithm Guide](algorithms.md) - Algorithm selection guide
- [Performance Guide](performance.md) - Performance optimization tips

---

## 💬 Questions & Support

**Common Questions:**

**Q: Do I need to change my existing code?**  
A: No! All existing code continues to work without changes.

**Q: Should I use new domain imports?**  
A: For existing code, no. For new features, consider domain-aware imports.

**Q: Will this affect performance?**  
A: No significant impact. Fallback imports add ~1ms overhead.

**Q: How do I know which domain to use?**  
A: Follow the [Domain Boundaries Guide](#domain-boundaries-guide) above.

**Q: What if I'm not sure about the architecture?**  
A: Start with existing patterns and gradually learn the new structure.

---

**Happy coding! 🎉**

The new domain architecture provides a solid foundation for future development while maintaining full compatibility with existing code.