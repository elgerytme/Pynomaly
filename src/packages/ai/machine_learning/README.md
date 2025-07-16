# Machine Learning Package

Comprehensive machine learning operations, model training, optimization, and lifecycle management for the Pynomaly platform.

## Overview

This package consolidates all machine learning operations that were previously scattered across different packages. It provides a unified interface for model training, optimization, deployment, and monitoring.

## Architecture

```
machine_learning/
├── training/           # Model training workflows and pipelines
├── optimization/       # Hyperparameter optimization and AutoML
├── lifecycle/          # Model versioning, registry, and deployment
├── monitoring/         # Model performance monitoring and drift detection
├── experiments/        # Experiment tracking and management
└── governance/         # ML governance and compliance
```

## Key Features

- **Model Training**: Automated training pipelines with multiple algorithms
- **Hyperparameter Optimization**: Optuna and Hyperopt integration
- **Model Registry**: Centralized model versioning and artifact management
- **Deployment**: Model serving and deployment automation
- **Monitoring**: Real-time model performance and drift monitoring
- **Experiment Tracking**: Comprehensive experiment management
- **Governance**: ML compliance and audit trails

## Installation

This package is part of the Pynomaly monorepo. Install ML dependencies:

```bash
# Core ML dependencies
pip install scikit-learn optuna mlflow

# Deep learning frameworks
pip install torch tensorflow

# Monitoring and tracking
pip install wandb tensorboard prometheus-client
```

## Usage

### Model Training
```python
from machine_learning.training import ModelTrainer

trainer = ModelTrainer()
model = trainer.train(algorithm="isolation_forest", data=X_train)
```

### Hyperparameter Optimization
```python
from machine_learning.optimization import HyperparameterOptimizer

optimizer = HyperparameterOptimizer()
best_params = optimizer.optimize(model_class, X_train, y_train)
```

### Model Registry
```python
from machine_learning.lifecycle import ModelRegistry

registry = ModelRegistry()
model_version = registry.register_model(model, metadata={"accuracy": 0.95})
```

### Performance Monitoring
```python
from machine_learning.monitoring import ModelMonitor

monitor = ModelMonitor()
metrics = monitor.track_performance(model, X_test, y_test)
```

## Dependencies

- **Core**: `scikit-learn`, `optuna`, `mlflow`
- **Deep Learning**: `torch`, `tensorflow`
- **Monitoring**: `wandb`, `tensorboard`, `prometheus-client`
- **Internal**: `anomaly_detection`, `data_platform`, `infrastructure`

## Components

### Training
- Automated training pipelines
- Multi-algorithm support
- Distributed training capabilities
- Training validation and testing

### Optimization
- Hyperparameter optimization with Optuna
- Neural architecture search
- AutoML pipeline optimization
- Multi-objective optimization

### Lifecycle
- Model versioning and registry
- Artifact management
- Deployment automation
- A/B testing frameworks

### Monitoring
- Real-time performance monitoring
- Data drift detection
- Model degradation alerts
- Performance dashboards

## Testing

```bash
pytest tests/machine_learning/
```

## Contributing

See the main repository CONTRIBUTING.md for guidelines.

## License

MIT License - see main repository LICENSE file.