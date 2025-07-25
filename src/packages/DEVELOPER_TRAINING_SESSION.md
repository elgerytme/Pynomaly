# Developer Training Session: Cross-Domain Integration Patterns

## 🎯 Training Overview

This comprehensive training session covers the enterprise-grade cross-domain integration patterns implemented in our hexagonal architecture monorepo.

**Duration:** 2 hours  
**Target Audience:** All development team members  
**Prerequisites:** Familiarity with Python, basic understanding of hexagonal architecture

## 📚 Learning Objectives

By the end of this session, you will:
1. Understand domain boundaries and their importance
2. Master cross-domain integration patterns
3. Know how to implement secure domain communication
4. Use the automated tools for boundary validation
5. Follow best practices for maintaining clean architecture

## 🏗️ Architecture Overview

### Domain Structure
Our monorepo is organized into clear domain boundaries:

```
src/packages/
├── shared/          # Universal utilities and interfaces
├── data/           # Data processing domains
│   ├── anomaly_detection/
│   ├── data_quality/
│   └── data_science/
├── ai/             # AI/ML domains
│   ├── machine_learning/
│   └── mlops/
├── enterprise/     # Enterprise features
│   └── security/
└── integrations/   # External integrations
    └── cloud/
```

### 🚨 Golden Rule: Domain Boundaries Must Not Be Violated

**✅ ALLOWED:**
- Import from `shared/`, `interfaces/`, `configurations/`
- Use dependency injection for cross-domain communication
- Communicate via events and message passing

**❌ FORBIDDEN:**
- Direct imports between domains (e.g., `from ai.machine_learning import ...`)
- Circular dependencies between domains
- Bypassing domain interfaces

## 🔧 Cross-Domain Integration Patterns

### 1. Domain Adapters Pattern

**Use Case:** When one domain needs to consume services from another domain.

```python
# ✅ CORRECT: Using domain adapter
from shared.integration.domain_adapters import DataQualityAdapter
from shared.integration.cross_domain_patterns import DomainServiceLocator

class MachineLearningService:
    def __init__(self, service_locator: DomainServiceLocator):
        self.data_quality = service_locator.get_adapter(DataQualityAdapter)
    
    async def validate_training_data(self, data):
        # Cross-domain call through adapter
        return await self.data_quality.validate_dataset(data)
```

**Why this works:**
- Loose coupling between domains
- Testable with mock adapters
- Clear interface contracts

### 2. Event Bus Pattern

**Use Case:** When domains need to react to events in other domains.

```python
# ✅ CORRECT: Event-driven communication
from shared.integration.cross_domain_patterns import EventBus, DomainEvent

class ModelTrainingCompleted(DomainEvent):
    def __init__(self, model_id: str, accuracy: float):
        super().__init__("ml.training.completed")
        self.model_id = model_id
        self.accuracy = accuracy

class MLOpsService:
    def __init__(self, event_bus: EventBus):
        self.event_bus = event_bus
        # Subscribe to events from ML domain
        self.event_bus.subscribe("ml.training.completed", self.handle_training_completion)
    
    async def handle_training_completion(self, event: ModelTrainingCompleted):
        if event.accuracy > 0.95:
            await self.deploy_model(event.model_id)
```

### 3. Saga Orchestration Pattern

**Use Case:** For complex workflows spanning multiple domains.

```python
# ✅ CORRECT: Saga for complex workflows
from shared.integration.cross_domain_patterns import SagaOrchestrator, SagaStep

class DataProcessingSaga(SagaOrchestrator):
    def __init__(self):
        super().__init__("data_processing_workflow")
        self.add_step(SagaStep("validate_data", self.validate_data, self.rollback_validation))
        self.add_step(SagaStep("process_data", self.process_data, self.rollback_processing))
        self.add_step(SagaStep("train_model", self.train_model, self.rollback_training))
        self.add_step(SagaStep("deploy_model", self.deploy_model, self.rollback_deployment))
    
    async def validate_data(self, context):
        # Use adapter to call data quality domain
        return await self.data_quality_adapter.validate(context.dataset)
```

## 🛡️ Security Integration

### Authentication & Authorization

All cross-domain calls must include proper authentication:

```python
from shared.infrastructure.security import SecurityContext, require_permission

class CrossDomainService:
    @require_permission("data:read")
    async def fetch_data(self, context: SecurityContext):
        # Security context is automatically validated
        pass
```

### Compliance Framework

Each domain can register compliance requirements:

```python
from shared.infrastructure.compliance import ComplianceFramework

# Register GDPR compliance for data operations
compliance = ComplianceFramework()
compliance.register_requirement("GDPR", "data_processing", gdpr_validator)
```

## 🔍 Development Tools

### 1. Boundary Violation Detection

Run before every commit:
```bash
python src/packages/deployment/scripts/boundary-violation-check.py src/packages --fail-on-violations
```

### 2. Integration Testing

Test cross-domain interactions:
```bash
python src/packages/deployment/validation/simplified_integration_test.py
```

### 3. Pre-commit Hooks

Automatically installed via:
```bash
python src/packages/deployment/scripts/pre-commit-checks.py --install
```

## 🎯 Hands-On Exercises

### Exercise 1: Implementing a Domain Adapter

**Scenario:** You need to add anomaly detection to the data quality pipeline.

**Task:** Implement an adapter that allows data quality domain to use anomaly detection services.

```python
# Your implementation here
class AnomalyDetectionAdapter:
    # TODO: Implement adapter interface
    pass
```

**Solution:**
```python
from shared.integration.domain_adapters import BaseDomainAdapter
from shared.integration.cross_domain_patterns import DomainServiceLocator

class AnomalyDetectionAdapter(BaseDomainAdapter):
    domain_name = "anomaly_detection"
    
    async def detect_anomalies(self, dataset: DataSet) -> AnomalyResult:
        # Use service locator to get anomaly detection service
        service = await self.service_locator.get_service("anomaly_detection", "detector")
        return await service.detect(dataset)
```

### Exercise 2: Event-Driven Integration

**Scenario:** When data quality validation fails, trigger data cleanup workflow.

**Task:** Implement event publishing and subscription.

### Exercise 3: Saga Implementation

**Scenario:** Implement end-to-end ML pipeline with proper rollback capabilities.

## ✅ Best Practices Checklist

### Before Writing Code:
- [ ] Identify which domain your feature belongs to
- [ ] Check if cross-domain communication is needed
- [ ] Choose appropriate integration pattern
- [ ] Design interface contracts first

### During Development:
- [ ] Use dependency injection for all external dependencies
- [ ] Write unit tests with mocked adapters
- [ ] Include proper error handling and timeouts
- [ ] Add logging for cross-domain calls

### Before Committing:
- [ ] Run boundary violation check
- [ ] Run integration tests
- [ ] Update interface documentation
- [ ] Verify security requirements are met

## 🚨 Common Pitfalls to Avoid

### 1. Direct Domain Imports
```python
# ❌ DON'T DO THIS
from ai.machine_learning.services import MLService

# ✅ DO THIS INSTEAD
from shared.integration.domain_adapters import MLAdapter
```

### 2. Tight Coupling
```python
# ❌ DON'T DO THIS
class DataProcessor:
    def __init__(self):
        self.ml_service = MLService()  # Direct dependency

# ✅ DO THIS INSTEAD
class DataProcessor:
    def __init__(self, ml_adapter: MLAdapter):
        self.ml_adapter = ml_adapter  # Injected dependency
```

### 3. Synchronous Cross-Domain Calls
```python
# ❌ DON'T DO THIS
result = other_domain.process_data(data)  # Blocking call

# ✅ DO THIS INSTEAD
result = await other_domain.process_data(data)  # Async call
```

## 🔧 Troubleshooting Guide

### Boundary Violation Errors
1. Check import statements
2. Verify adapter usage
3. Review domain mappings in config

### Integration Test Failures
1. Check service registration
2. Verify adapter implementations
3. Review event bus subscriptions

### Performance Issues
1. Check for blocking calls
2. Review connection pooling
3. Optimize batch sizes

## 📖 Additional Resources

- [Hexagonal Architecture Summary](./HEXAGONAL_ARCHITECTURE_SUMMARY.md)
- [Advanced Patterns Guide](./ADVANCED_PATTERNS_GUIDE.md)
- [Production Operations Guide](./PRODUCTION_OPERATIONS_GUIDE.md)
- [Security Framework Documentation](./enterprise/security/README.md)

## 🎓 Knowledge Check

### Quiz Questions:

1. What are the three main cross-domain integration patterns?
2. When should you use the Event Bus pattern vs Domain Adapters?
3. How do you ensure security compliance in cross-domain calls?
4. What tools help detect boundary violations?
5. What are the key benefits of the Saga pattern?

### Practical Assessment:

Implement a complete feature that:
1. Spans at least two domains
2. Uses proper integration patterns
3. Includes error handling and rollback
4. Passes all boundary violation checks
5. Has comprehensive test coverage

## 📅 Next Steps

After this training:
1. Apply learned patterns to your current work
2. Review existing code for boundary violations
3. Participate in code reviews focusing on architecture
4. Attend follow-up sessions on specific patterns
5. Contribute to pattern documentation

---

**Questions?** Reach out to the architecture team or create an issue in the repository.

**Remember:** Clean architecture is everyone's responsibility! 🏗️✨