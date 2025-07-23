# 🤖 Machine Learning Package

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/downloads/)
[![Code style: Black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)
[![Type checked: mypy](https://img.shields.io/badge/typed-mypy-blue.svg)](https://mypy-lang.org/)
[![MLflow](https://img.shields.io/badge/MLflow-1.28+-orange.svg)](https://mlflow.org/)
[![Optuna](https://img.shields.io/badge/Optuna-3.0+-blue.svg)](https://optuna.org/)

Comprehensive machine learning operations, model training, optimization, and lifecycle management for enterprise-grade ML workflows.

## 🎯 Overview

This package consolidates all machine learning operations into a unified, enterprise-ready platform. It provides comprehensive tools for the entire ML lifecycle - from experimentation and training to deployment and monitoring.

**Key Capabilities:**
- **🚀 Automated Training**: Multi-algorithm training pipelines with hyperparameter optimization
- **📊 Experiment Tracking**: Comprehensive experiment management with MLflow and Weights & Biases
- **🔄 Model Lifecycle**: Version control, registry, and automated deployment
- **📈 Performance Monitoring**: Real-time monitoring, drift detection, and alerting
- **🎛️ AutoML**: Automated machine learning with neural architecture search
- **🔒 ML Governance**: Compliance, audit trails, and model explainability

## 🚀 Quick Start

### Installation

```bash
# Install the machine learning package
cd src/packages/ai/machine_learning/
pip install -e .

# Install with all ML dependencies
pip install -e ".[ml_algorithms,tracking,distributed]"

# Development setup
pip install -e ".[dev,test]"
```

### Basic Usage

```python
# Train a model with automatic optimization
from machine_learning.training import AutoMLTrainer
from machine_learning.lifecycle import ModelRegistry

# Initialize trainer with automatic hyperparameter optimization
trainer = AutoMLTrainer(
    algorithms=["isolation_forest", "random_forest", "xgboost"],
    optimization_budget=100
)

# Train and optimize
model = trainer.fit(X_train, y_train)

# Register the best model
registry = ModelRegistry()
model_version = registry.register(
    model, 
    name="anomaly_detector_v1",
    metrics=trainer.best_metrics
)

print(f"Model registered: {model_version.name} v{model_version.version}")
```

### CLI Interface

```bash
# Train a model via CLI
machine_learning train --config config/training.yaml --data data/train.csv

# Start optimization experiment
machine_learning optimize --study anomaly_detection --trials 100

# Deploy a model
machine_learning deploy --model anomaly_detector_v1 --environment staging

# Monitor model performance
machine_learning monitor --model anomaly_detector_v1 --metrics accuracy,drift
```

## 🏗️ Architecture

The machine learning package follows clean architecture principles with specialized components:

```
machine_learning/
├── core/                           # Core domain logic
│   ├── domain/                     # ML domain entities and services
│   │   ├── entities/               # Model, Experiment, Pipeline entities
│   │   ├── services/               # Training, optimization services
│   │   ├── repositories/           # Model and experiment repositories
│   │   └── value_objects/          # Hyperparameters, metrics, configurations
│   ├── application/                # Application layer orchestration
│   │   ├── use_cases/              # Training, deployment, monitoring use cases
│   │   ├── services/               # Application services
│   │   └── dto/                    # Data transfer objects
│   └── dto/                        # Shared DTOs
├── infrastructure/                 # External integrations
│   ├── adapters/                   # MLflow, Optuna, cloud adapters
│   ├── persistence/                # Model storage and artifact management
│   ├── monitoring/                 # Performance and drift monitoring
│   └── deployment/                 # Model serving and deployment
├── interfaces/                     # User interfaces
│   ├── api/                        # REST API endpoints
│   ├── cli/                        # Command-line interface
│   └── web/                        # Web dashboard (if available)
├── training/                       # Training workflows and pipelines
├── optimization/                   # Hyperparameter optimization and AutoML
├── lifecycle/                      # Model versioning, registry, and deployment
├── monitoring/                     # Model performance monitoring and drift analysis
├── experiments/                    # Experiment tracking and management
├── governance/                     # ML governance and compliance
└── tests/                          # Comprehensive test suite
```

## 🎯 Key Features

### 🚀 Automated Training
- **Multi-Algorithm Support**: Supports 20+ algorithms including ensemble methods
- **Distributed Training**: Ray and Dask integration for large-scale training
- **Pipeline Automation**: End-to-end training pipelines with data validation
- **Cross-Validation**: Robust model validation with stratified sampling

### 🎛️ Hyperparameter Optimization
- **Optuna Integration**: Advanced optimization with pruning and multi-objective support
- **Neural Architecture Search**: Automated neural network design
- **Ensemble Optimization**: Automated ensemble method selection and weighting
- **Bayesian Optimization**: Efficient parameter space exploration

### 📊 Model Registry & Lifecycle
- **Version Control**: Git-like versioning for models and experiments
- **Artifact Management**: Centralized storage for models, data, and configurations
- **Automated Deployment**: CI/CD pipelines for model deployment
- **A/B Testing**: Framework for model comparison and gradual rollouts

### 📈 Monitoring & Observability
- **Real-time Monitoring**: Live performance tracking with Prometheus metrics
- **Drift Detection**: Statistical and ML-based drift detection methods
- **Alerting**: Configurable alerts for performance degradation
- **Explainability**: SHAP and LIME integration for model interpretability

## 📦 Installation Options

### Core Installation
```bash
pip install -e .
```

### With ML Algorithms
```bash
pip install -e ".[ml_algorithms]"
# Includes: torch, tensorflow, xgboost, lightgbm
```

### With Experiment Tracking
```bash
pip install -e ".[tracking]"
# Includes: wandb, tensorboard, mlflow
```

### With Distributed Computing
```bash
pip install -e ".[distributed]"
# Includes: ray, dask, distributed training support
```

### Full Installation
```bash
pip install -e ".[all]"
# Includes all optional dependencies
```

## 📚 Usage Examples

### Training Pipeline

```python
from machine_learning.training import TrainingPipeline
from machine_learning.core.domain.entities import TrainingConfig

# Configure training pipeline
config = TrainingConfig(
    algorithm="isolation_forest",
    hyperparameters={
        "contamination": 0.1,
        "n_estimators": 100
    },
    validation_strategy="time_series_split",
    early_stopping=True
)

# Initialize and run pipeline
pipeline = TrainingPipeline(config)
result = pipeline.execute(
    train_data=X_train,
    validation_data=X_val,
    test_data=X_test
)

print(f"Training completed. Best score: {result.best_score}")
print(f"Model artifact: {result.model_path}")
```

### Hyperparameter Optimization

```python
from machine_learning.optimization import OptunaOptimizer
from machine_learning.core.domain.value_objects import OptimizationConfig

# Configure optimization
opt_config = OptimizationConfig(
    study_name="anomaly_detection_optimization",
    objective="f1_score",
    direction="maximize",
    n_trials=100,
    timeout=3600  # 1 hour
)

# Run optimization
optimizer = OptunaOptimizer(opt_config)
study = optimizer.optimize(
    algorithm="isolation_forest",
    train_data=X_train,
    validation_data=X_val
)

print(f"Best parameters: {study.best_params}")
print(f"Best score: {study.best_value}")
```

### Model Registry Operations

```python
from machine_learning.lifecycle import ModelRegistry, ModelMetadata

# Initialize registry
registry = ModelRegistry()

# Register a model
metadata = ModelMetadata(
    name="anomaly_detector",
    version="1.0.0",
    algorithm="isolation_forest",
    metrics={"accuracy": 0.95, "precision": 0.92, "recall": 0.88},
    training_data_hash="abc123",
    hyperparameters={"n_estimators": 100, "contamination": 0.1}
)

model_version = registry.register_model(
    model=trained_model,
    metadata=metadata,
    tags=["production", "anomaly_detection"]
)

# Load a model
loaded_model = registry.load_model("anomaly_detector", version="1.0.0")

# List all model versions
versions = registry.list_versions("anomaly_detector")
for version in versions:
    print(f"Version {version.version}: {version.metrics}")
```

### Model Deployment

```python
from machine_learning.lifecycle import ModelDeployer
from machine_learning.core.domain.entities import DeploymentConfig

# Configure deployment
deploy_config = DeploymentConfig(
    environment="production",
    scaling_policy="auto",
    health_check_endpoint="/health",
    resource_limits={"cpu": "2", "memory": "4Gi"}
)

# Deploy model
deployer = ModelDeployer()
deployment = deployer.deploy(
    model_name="anomaly_detector",
    version="1.0.0",
    config=deploy_config
)

print(f"Model deployed to: {deployment.endpoint_url}")
print(f"Deployment status: {deployment.status}")
```

### Performance Monitoring

```python
from machine_learning.monitoring import ModelMonitor, DriftDetector

# Initialize monitoring
monitor = ModelMonitor(model_name="anomaly_detector")

# Track model performance
performance_metrics = monitor.log_prediction_batch(
    predictions=y_pred,
    actuals=y_true,
    features=X_test,
    timestamp=datetime.now()
)

# Check for data drift
drift_detector = DriftDetector()
drift_report = drift_detector.detect_drift(
    reference_data=X_train,
    current_data=X_new,
    threshold=0.05
)

if drift_report.drift_detected:
    print(f"Drift detected in features: {drift_report.drifted_features}")
    print(f"Recommended action: {drift_report.recommendation}")
```

### Experiment Tracking

```python
from machine_learning.experiments import ExperimentTracker

# Initialize experiment tracking
tracker = ExperimentTracker(
    experiment_name="anomaly_detection_optimization",
    tracking_backend="mlflow"  # or "wandb"
)

# Start experiment run
with tracker.start_run() as run:
    # Log parameters
    run.log_params({
        "algorithm": "isolation_forest",
        "n_estimators": 100,
        "contamination": 0.1
    })
    
    # Train model
    model = train_model(X_train, y_train)
    
    # Log metrics
    run.log_metrics({
        "accuracy": 0.95,
        "precision": 0.92,
        "recall": 0.88
    })
    
    # Log artifacts
    run.log_artifact(model, "model.pkl")
    run.log_artifact(config, "config.yaml")
```

## ⚙️ Configuration

### Training Configuration

```yaml
# config/training.yaml
training:
  algorithm: "isolation_forest"
  hyperparameters:
    n_estimators: 100
    contamination: 0.1
    random_state: 42
  
  validation:
    strategy: "time_series_split"
    n_splits: 5
    test_size: 0.2
  
  early_stopping:
    enabled: true
    patience: 10
    min_delta: 0.001

optimization:
  study_name: "anomaly_detection"
  n_trials: 100
  timeout: 3600
  pruner: "median"
  sampler: "tpe"
```

### Deployment Configuration

```yaml
# config/deployment.yaml
deployment:
  environment: "production"
  
  scaling:
    min_replicas: 2
    max_replicas: 10
    target_cpu_utilization: 70
  
  resources:
    requests:
      cpu: "500m"
      memory: "1Gi"
    limits:
      cpu: "2"
      memory: "4Gi"
  
  health_check:
    endpoint: "/health"
    interval: 30
    timeout: 5
    retries: 3
```

### Monitoring Configuration

```yaml
# config/monitoring.yaml
monitoring:
  metrics:
    - accuracy
    - precision
    - recall
    - latency
    - throughput
  
  drift_detection:
    method: "ks_test"
    threshold: 0.05
    window_size: 1000
  
  alerts:
    accuracy_threshold: 0.85
    drift_threshold: 0.05
    latency_threshold: 100  # ms
```

## 🧪 Testing

### Running Tests

```bash
# Run all tests
pytest tests/

# Run with coverage
pytest tests/ --cov=machine_learning --cov-report=html

# Run specific test categories
pytest tests/unit/                    # Unit tests
pytest tests/integration/             # Integration tests
pytest tests/performance/             # Performance tests

# Run tests with markers
pytest -m "not slow"                  # Skip slow tests
pytest -m "training"                  # Training-related tests only
```

### Test Structure

```
tests/
├── unit/                             # Unit tests
│   ├── training/                     # Training component tests
│   ├── optimization/                 # Optimization tests
│   ├── lifecycle/                    # Model lifecycle tests
│   └── monitoring/                   # Monitoring tests
├── integration/                      # Integration tests
│   ├── test_training_pipeline.py     # End-to-end training
│   ├── test_deployment_pipeline.py   # Deployment integration
│   └── test_monitoring_integration.py # Monitoring integration
├── performance/                      # Performance benchmarks
│   ├── test_training_performance.py  # Training benchmarks
│   └── test_inference_performance.py # Inference benchmarks
└── fixtures/                         # Test data and utilities
    ├── sample_data.py                # Sample datasets
    └── model_fixtures.py             # Pre-trained test models
```

## 🚀 Development

### Setting Up Development Environment

```bash
# Clone and navigate to package
cd src/packages/ai/machine_learning/

# Create virtual environment
python -m venv venv
source venv/bin/activate  # Linux/macOS
# venv\Scripts\activate   # Windows

# Install development dependencies
pip install -e ".[dev,test,all]"

# Install pre-commit hooks
pre-commit install

# Verify setup
python -c "import machine_learning; print('Setup successful')"
```

### Code Quality

```bash
# Run linting and formatting
ruff check . --fix
ruff format .

# Run type checking
mypy machine_learning/

# Run security checks
bandit -r machine_learning/

# Run tests with coverage
pytest tests/ --cov=machine_learning --cov-report=html
```

### Performance Benchmarking

```bash
# Run performance benchmarks
python -m pytest tests/performance/ -v

# Profile training performance
python -m cProfile -o profile_output.prof scripts/benchmark_training.py

# Memory profiling
python -m memory_profiler scripts/benchmark_memory.py
```

## 📊 Performance Considerations

### Training Performance
- **Distributed Training**: Use Ray or Dask for large datasets (>1GB)
- **GPU Acceleration**: Automatic GPU detection for deep learning frameworks
- **Memory Optimization**: Streaming data loading for memory-constrained environments
- **Caching**: Intelligent caching of preprocessed data and intermediate results

### Inference Performance
- **Model Optimization**: ONNX conversion for faster inference
- **Batch Processing**: Automatic batching for improved throughput
- **Caching**: Redis-based prediction caching for frequently accessed data
- **Load Balancing**: Automatic load balancing across multiple model instances

### Scalability Benchmarks
| Dataset Size | Training Time | Memory Usage | Throughput |
|-------------|---------------|--------------|-------------|
| 1K samples  | 2s           | 100MB        | 10K pred/s  |
| 100K samples| 5min         | 1GB          | 5K pred/s   |
| 1M samples  | 30min        | 8GB          | 2K pred/s   |
| 10M samples | 3hr          | 32GB         | 1K pred/s   |

## 🔧 API Reference

### Core Classes

#### ModelTrainer
Main interface for model training.

```python
class ModelTrainer:
    def __init__(self, config: TrainingConfig):
        """Initialize trainer with configuration."""
        
    def fit(self, X: np.ndarray, y: np.ndarray) -> TrainedModel:
        """Train a model on the provided data."""
        
    def evaluate(self, X: np.ndarray, y: np.ndarray) -> Dict[str, float]:
        """Evaluate model performance."""
```

#### HyperparameterOptimizer
Hyperparameter optimization interface.

```python
class HyperparameterOptimizer:
    def optimize(self, algorithm: str, X: np.ndarray, y: np.ndarray) -> Study:
        """Optimize hyperparameters for the given algorithm."""
        
    def get_best_params(self, study_name: str) -> Dict[str, Any]:
        """Get best parameters from completed study."""
```

#### ModelRegistry
Model lifecycle management.

```python
class ModelRegistry:
    def register_model(self, model: Any, metadata: ModelMetadata) -> ModelVersion:
        """Register a new model version."""
        
    def load_model(self, name: str, version: str) -> Any:
        """Load a specific model version."""
        
    def list_versions(self, name: str) -> List[ModelVersion]:
        """List all versions of a model."""
```

## 🔍 Troubleshooting

### Common Issues

#### Training Issues

**Problem**: `OutOfMemoryError` during training
```python
# Solution: Use batch training or reduce batch size
trainer = ModelTrainer(config=TrainingConfig(batch_size=32))
# Or enable streaming mode
trainer = ModelTrainer(config=TrainingConfig(streaming=True))
```

**Problem**: Training is too slow
```python
# Solution: Enable distributed training
trainer = ModelTrainer(config=TrainingConfig(
    distributed=True,
    n_workers=4
))
```

#### Deployment Issues

**Problem**: Model serving latency is high
```yaml
# Solution: Enable model optimization in deployment config
deployment:
  optimization:
    onnx_conversion: true
    quantization: true
    batch_size: 32
```

**Problem**: Memory usage is too high in production
```yaml
# Solution: Configure resource limits
deployment:
  resources:
    limits:
      memory: "2Gi"
  optimization:
    memory_efficient: true
```

#### Data Issues

**Problem**: Data drift detected
```python
# Solution: Retrain model with recent data
from machine_learning.training import RetrainingPipeline

retrainer = RetrainingPipeline()
retrainer.retrain_on_drift(
    model_name="anomaly_detector",
    new_data=recent_data,
    drift_threshold=0.05
)
```

### FAQ

**Q: How do I choose the right algorithm?**
A: Use the AutoML functionality to automatically select and optimize algorithms:
```python
trainer = AutoMLTrainer(algorithms="auto")  # Tests all available algorithms
```

**Q: Can I use custom algorithms?**
A: Yes, implement the `Algorithm` interface:
```python
class CustomAlgorithm(Algorithm):
    def fit(self, X, y):
        # Your implementation
        pass
    
    def predict(self, X):
        # Your implementation
        pass
```

**Q: How do I integrate with existing MLOps tools?**
A: The package supports multiple backends through adapters:
```python
# MLflow integration
tracker = ExperimentTracker(backend="mlflow")

# Kubeflow integration  
deployer = ModelDeployer(backend="kubeflow")
```

## 🤝 Contributing

We welcome contributions to the machine learning package! This package is critical for the monorepo's ML capabilities.

### How to Contribute

1. **Follow Clean Architecture**: Maintain the existing architectural patterns
2. **Add Comprehensive Tests**: Ensure high test coverage for new features
3. **Update Documentation**: Keep documentation current with changes
4. **Performance Testing**: Include performance benchmarks for new features

### Areas for Contribution

- **New Algorithms**: Integration of additional ML algorithms and frameworks
- **AutoML Enhancements**: Improved automated machine learning capabilities
- **Monitoring Tools**: Advanced model monitoring and observability features
- **Deployment Options**: Additional deployment targets and optimization strategies
- **Documentation**: Examples, tutorials, and best practices

For detailed guidelines, see the main repository [CONTRIBUTING.md](../../../docs/developer-guides/contributing/CONTRIBUTING.md).

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](../../../LICENSE) file for details.

## 🔗 Related Documentation

- [Clean Architecture Guidelines](../../../docs/architecture/clean-architecture.md)
- [ML Best Practices](../../../docs/ml/best-practices.md)
- [Model Deployment Guide](../../../docs/ml/deployment.md)
- [Monitoring and Observability](../../../docs/ml/monitoring.md)

---

**Note**: This package provides enterprise-grade machine learning capabilities with a focus on scalability, maintainability, and operational excellence. It integrates seamlessly with the monorepo's clean architecture principles.