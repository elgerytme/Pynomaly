# Hexagonal Architecture Guide - Machine Learning Package

This guide explains the hexagonal architecture implementation in the machine_learning package, following Domain-Driven Design (DDD) principles and the ports/adapters pattern.

## 📐 Architecture Overview

The machine_learning package follows **Hexagonal Architecture** (also known as Ports and Adapters) to achieve clean separation of concerns and technology independence.

```
┌─────────────────────────────────────────────────────────────┐
│                     Machine Learning Domain                 │
├─────────────────────────────────────────────────────────────┤
│  📊 Domain Entities           📋 Domain Services           │
│  ├── Model                    ├── AutoMLService            │
│  ├── Dataset                  ├── ExplainabilityService    │
│  ├── OptimizationResult       └── ModelSelectionService   │
│  └── TrainingJob                                           │
│                                                             │
│  🔌 Domain Interfaces (Ports)                              │
│  ├── AutoMLOptimizationPort      🔗 Dependency Injection   │
│  ├── ExplainabilityPort          ├── Container             │
│  ├── MonitoringPort              └── Configuration         │
│  └── DistributedTracingPort                                │
└─────────────────────────────────────────────────────────────┘
                              │
                    ┌─────────┼─────────┐
                    │                   │
          ┌─────────▼─────────┐ ┌───────▼────────┐
          │   Infrastructure  │ │   Application  │
          │     Adapters      │ │     Services   │
          ├───────────────────┤ ├────────────────┤
          │ 🤖 ML Libraries   │ │ 🌐 REST API    │
          │ ├── Scikit-learn  │ │ ├── FastAPI    │
          │ ├── Optuna        │ │ └── Endpoints  │
          │ └── SHAP/LIME     │ │                │
          │                   │ │ 🖥️  CLI        │
          │ 📊 Monitoring     │ │ ├── Commands   │
          │ ├── Prometheus    │ │ └── Scripts    │
          │ ├── Jaeger        │ │                │
          │ └── Zipkin        │ │ 📱 Web UI      │
          │                   │ │ └── Dashboard  │
          │ 🎭 Stubs          │ └────────────────┘
          │ └── Fallbacks     │
          └───────────────────┘
```

## 🏗️ Core Architectural Principles

### 1. **Domain Isolation**
- Domain logic is completely isolated from external concerns
- No direct dependencies on frameworks or libraries in domain layer
- Business rules remain pure and testable

### 2. **Dependency Inversion**
- Domain defines interfaces (ports) for what it needs
- Infrastructure provides implementations (adapters)
- Dependencies point inward toward the domain

### 3. **Technology Independence**
- Easy to swap external libraries (e.g., scikit-learn → TensorFlow)
- Adapters translate between domain concepts and library APIs
- Domain remains stable while technology evolves

### 4. **Testability**
- All external dependencies are mockable through interfaces
- Domain logic can be tested in isolation
- Integration tests focus on adapter implementations

## 📁 Package Structure

```
machine_learning/
├── domain/                                    # 🎯 Core Business Logic
│   ├── entities/                             # Domain Objects
│   │   ├── model.py                          # ML Model entity
│   │   ├── dataset.py                        # Dataset entity
│   │   └── optimization_result.py            # Optimization results
│   ├── interfaces/                           # 🔌 Ports (Contracts)
│   │   ├── automl_operations.py              # AutoML interfaces
│   │   ├── explainability_operations.py     # XAI interfaces
│   │   └── monitoring_operations.py          # Observability interfaces
│   ├── services/                             # Domain Services
│   │   ├── automl_service.py                 # AutoML orchestration
│   │   └── explainability_service.py         # XAI orchestration
│   └── value_objects/                        # Value Objects
│       ├── hyperparameters.py               # Algorithm parameters
│       └── performance_metrics.py           # Model metrics
├── infrastructure/                           # 🔧 Technical Implementation
│   ├── adapters/                            # 🔌 Adapters (Implementations)
│   │   ├── automl/                          # AutoML adapters
│   │   │   └── sklearn_automl_adapter.py    # Scikit-learn integration
│   │   ├── monitoring/                      # Monitoring adapters
│   │   │   └── distributed_tracing_adapter.py # Tracing integration
│   │   └── stubs/                           # Fallback implementations
│   │       ├── automl_stubs.py              # AutoML stubs
│   │       ├── explainability_stubs.py      # XAI stubs
│   │       └── monitoring_stubs.py          # Monitoring stubs
│   └── container/                           # 🏗️ Dependency Injection
│       └── container.py                     # IoC Container
├── application/                             # 🌐 Application Layer
│   ├── services/                            # Application Services
│   └── use_cases/                           # Use Case Implementations
└── presentation/                            # 📱 User Interface
    ├── api/                                 # REST API endpoints
    ├── cli/                                 # Command-line interface
    └── web/                                 # Web interface
```

## 🔌 Ports and Adapters

### Domain Interfaces (Ports)

#### AutoML Operations
```python
# machine_learning/domain/interfaces/automl_operations.py

class AutoMLOptimizationPort(ABC):
    """Port for automated machine learning optimization."""
    
    @abstractmethod
    async def optimize_model(
        self,
        dataset: Dataset,
        optimization_config: OptimizationConfig,
        ground_truth: Optional[Any] = None
    ) -> OptimizationResult:
        """Optimize ML model for given dataset."""
        pass
```

#### Explainability Operations
```python
# machine_learning/domain/interfaces/explainability_operations.py

class ExplainabilityPort(ABC):
    """Port for model explainability operations."""
    
    @abstractmethod
    async def explain_prediction(
        self, 
        request: ExplanationRequest
    ) -> ExplanationResult:
        """Generate explanation for specific prediction."""
        pass
```

#### Monitoring Operations  
```python
# machine_learning/domain/interfaces/monitoring_operations.py

class MonitoringPort(ABC):
    """Port for monitoring and observability."""
    
    @abstractmethod
    async def record_metric(self, metric: MetricValue) -> None:
        """Record a metric value."""
        pass
```

### Infrastructure Adapters

#### Scikit-learn AutoML Adapter
```python
# machine_learning/infrastructure/adapters/automl/sklearn_automl_adapter.py

class SklearnAutoMLAdapter(AutoMLOptimizationPort):
    """Adapter for scikit-learn based AutoML operations."""
    
    async def optimize_model(
        self,
        dataset: Dataset,
        optimization_config: OptimizationConfig,
        ground_truth: Optional[Any] = None
    ) -> OptimizationResult:
        # Implementation using scikit-learn + Optuna
        # Translates domain concepts to sklearn APIs
        pass
```

#### Distributed Tracing Adapter
```python
# machine_learning/infrastructure/adapters/monitoring/distributed_tracing_adapter.py

class DistributedTracingAdapter(DistributedTracingPort):
    """Adapter for distributed tracing systems."""
    
    def __init__(self, tracing_backend: str = "local"):
        # Supports Jaeger, Zipkin, or local tracing
        self._backend = tracing_backend
    
    async def start_trace(self, operation_name: str) -> TraceSpan:
        # Implementation specific to chosen backend
        pass
```

## 🏗️ Dependency Injection Container

The container manages all dependencies following the Service Locator pattern:

```python
# machine_learning/infrastructure/container/container.py

@dataclass
class ContainerConfig:
    """Configuration for dependency injection."""
    enable_sklearn_automl: bool = True
    enable_optuna_optimization: bool = True
    enable_distributed_tracing: bool = True
    tracing_backend: str = "local"

class Container:
    """Dependency injection container."""
    
    def __init__(self, config: Optional[ContainerConfig] = None):
        self._config = config or ContainerConfig()
        self._configure_adapters()
        self._configure_domain_services()
    
    def _configure_sklearn_automl_adapter(self):
        """Configure scikit-learn AutoML adapter."""
        adapter = SklearnAutoMLAdapter()
        self.register_singleton(AutoMLOptimizationPort, adapter)
    
    def get(self, interface: Type[T]) -> T:
        """Resolve service by interface."""
        return self._singletons[interface]
```

### Container Usage

```python
# Create container with configuration
config = ContainerConfig(
    enable_sklearn_automl=True,
    tracing_backend="jaeger"
)
container = Container(config)

# Get configured services
automl_service = container.get(AutoMLService)
monitoring_port = container.get(MonitoringPort)

# Services are automatically wired with their dependencies
result = await automl_service.optimize_prediction(dataset)
```

## 🎯 Domain Services

Domain services orchestrate business logic while delegating technical operations:

```python
# machine_learning/domain/services/refactored_automl_service.py

class AutoMLService:
    """AutoML service using hexagonal architecture."""
    
    def __init__(
        self,
        automl_port: AutoMLOptimizationPort,
        model_selection_port: ModelSelectionPort,
        monitoring_port: Optional[MonitoringPort] = None,
        tracing_port: Optional[DistributedTracingPort] = None
    ):
        # Dependencies injected through constructor
        self._automl_port = automl_port
        self._model_selection_port = model_selection_port
        self._monitoring_port = monitoring_port
        self._tracing_port = tracing_port
    
    async def optimize_prediction(
        self,
        dataset: Dataset,
        optimization_config: Optional[OptimizationConfig] = None,
        ground_truth: Optional[Any] = None,
    ) -> OptimizationResult:
        """Domain orchestration with clean separation."""
        
        # Start distributed tracing
        span = await self._tracing_port.start_trace("automl_optimization")
        
        try:
            # Record business metrics
            await self._monitoring_port.increment_counter("automl_optimizations_started")
            
            # Delegate to AutoML adapter
            result = await self._automl_port.optimize_model(
                dataset, optimization_config, ground_truth
            )
            
            # Apply domain business rules
            enhanced_result = self._apply_business_rules(result)
            
            # Record success metrics
            await self._monitoring_port.set_gauge("automl_best_score", result.best_score)
            
            return enhanced_result
            
        finally:
            await self._tracing_port.finish_trace(span)
    
    def _apply_business_rules(self, result: OptimizationResult) -> OptimizationResult:
        """Apply domain-specific business logic."""
        # Example: Add business recommendations
        if result.best_score < 0.7:
            result.recommendations["performance"] = [
                "Consider feature engineering to improve model performance"
            ]
        return result
```

## 🎭 Graceful Degradation with Stubs

When external libraries are unavailable, stubs provide fallback functionality:

```python
# machine_learning/infrastructure/adapters/stubs/automl_stubs.py

class AutoMLOptimizationStub(AutoMLOptimizationPort):
    """Stub implementation when AutoML libraries unavailable."""
    
    def __init__(self):
        self._logger.warning(
            "Using AutoML stub. Install scikit-learn and optuna for full functionality."
        )
    
    async def optimize_model(
        self,
        dataset: Dataset,
        optimization_config: OptimizationConfig,
        ground_truth: Optional[Any] = None
    ) -> OptimizationResult:
        """Provide basic fallback optimization."""
        
        # Return reasonable defaults with warnings
        return OptimizationResult(
            best_algorithm_type=AlgorithmType.ISOLATION_FOREST,
            best_config=AlgorithmConfig(
                algorithm_type=AlgorithmType.ISOLATION_FOREST,
                parameters={"contamination": 0.1}
            ),
            best_score=0.75,  # Mock score
            recommendations={
                "general": ["Stub optimization - install libraries for real results"]
            }
        )
```

## 🧪 Testing Strategy

### Unit Testing Domain Logic
```python
# tests/unit/domain/services/test_automl_service.py

@pytest.fixture
def mock_automl_port():
    return Mock(spec=AutoMLOptimizationPort)

@pytest.fixture
def mock_monitoring_port():
    return Mock(spec=MonitoringPort)

@pytest.fixture
def automl_service(mock_automl_port, mock_monitoring_port):
    return AutoMLService(
        automl_port=mock_automl_port,
        model_selection_port=Mock(),
        monitoring_port=mock_monitoring_port
    )

async def test_optimize_prediction_records_metrics(automl_service, mock_monitoring_port):
    """Test that optimization records proper metrics."""
    dataset = create_test_dataset()
    
    await automl_service.optimize_prediction(dataset)
    
    # Verify monitoring calls
    mock_monitoring_port.increment_counter.assert_called_with("automl_optimizations_started")
    mock_monitoring_port.set_gauge.assert_called()
```

### Integration Testing Adapters
```python
# tests/integration/infrastructure/test_sklearn_adapter.py

@pytest.mark.skipif(not sklearn_available, reason="scikit-learn not available")
async def test_sklearn_automl_adapter_real_optimization():
    """Test adapter with real scikit-learn integration."""
    adapter = SklearnAutoMLAdapter()
    dataset = load_test_dataset()
    
    config = OptimizationConfig(max_trials=5)
    result = await adapter.optimize_model(dataset, config)
    
    assert result.best_score > 0
    assert result.best_algorithm_type in AlgorithmType
    assert len(result.trial_history) == 5
```

## 🚀 Usage Examples

### Basic Setup
```python
from machine_learning.infrastructure.container import Container, ContainerConfig

# Configure container
config = ContainerConfig(
    enable_sklearn_automl=True,
    enable_distributed_tracing=True,
    tracing_backend="jaeger"
)

# Create container and get services
container = Container(config)
automl_service = container.get(AutoMLService)

# Use service
dataset = load_dataset("data.csv")
result = await automl_service.optimize_prediction(dataset)

print(f"Best algorithm: {result.best_algorithm_type.value}")
print(f"Best score: {result.best_score}")
```

### Advanced Configuration
```python
# Custom adapter configuration
container.configure_ml_integration(
    enable_sklearn=True,
    enable_optuna=True,
    sklearn_config={"n_jobs": 4},
    optuna_config={"n_trials": 100}
)

# Custom monitoring configuration  
container.configure_monitoring_integration(
    enable_tracing=True,
    tracing_backend="zipkin",
    monitoring_config={"prometheus_endpoint": "http://localhost:9090"}
)

# Get updated services
automl_service = container.get(AutoMLService)
```

### Explanation Generation
```python
# Get explainability service
explainer = container.get(ExplainabilityService)

# Generate prediction explanation
explanation_request = ExplanationRequest(
    model=trained_model,
    data=test_instance,
    method=ExplanationMethod.SHAP,
    scope=ExplanationScope.LOCAL
)

explanation = await explainer.explain_prediction(explanation_request)

print(f"Top contributing features:")
for contrib in explanation.feature_contributions[:5]:
    print(f"  {contrib.feature_name}: {contrib.contribution_value:.3f}")
```

## 🔧 Configuration Options

### Container Configuration
- **AutoML Settings**: Enable/disable scikit-learn, Optuna, algorithm selection
- **Explainability**: Configure SHAP, LIME, and other XAI libraries
- **Monitoring**: Set up Prometheus, Jaeger, Zipkin integration
- **Environment**: Development, staging, production presets

### Runtime Reconfiguration
```python
# Reconfigure at runtime
container.configure_ml_integration(enable_sklearn=False)  # Switch to stubs
container.configure_monitoring_integration(tracing_backend="local")  # Change backend
```

## 📊 Benefits Achieved

### 🎯 **Business Benefits**
- **Faster Development**: Clear separation enables parallel development
- **Reduced Risk**: Technology changes don't impact business logic
- **Better Quality**: Focused testing and validation possible
- **Future-Proof**: Architecture scales with business needs

### 🔧 **Technical Benefits**  
- **Maintainability**: Clear boundaries and responsibilities
- **Testability**: Mock all external dependencies easily
- **Flexibility**: Swap implementations without code changes
- **Reliability**: Graceful degradation when services unavailable

### 👥 **Team Benefits**
- **Clear Ownership**: Domain vs infrastructure responsibilities
- **Parallel Work**: Teams can work on different adapters simultaneously
- **Learning**: New team members understand boundaries easily
- **Documentation**: Architecture is self-documenting

This hexagonal architecture provides a solid foundation for the machine_learning package that will scale with business needs while maintaining clean, testable, and maintainable code.