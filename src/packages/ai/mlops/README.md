# MLOps

[![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/downloads/)
[![Code style: ruff](https://img.shields.io/badge/code%20style-ruff-000000.svg)](https://github.com/astral-sh/ruff)
[![Type checked: mypy](https://img.shields.io/badge/type%20checked-mypy-blue.svg)](http://mypy-lang.org/)

## Overview

Machine Learning Operations (MLOps) package and platform.

**Architecture Layer**: Application Layer  
**Package Type**: ML Operations  
**Status**: Production Ready

## Purpose

This package provides comprehensive MLOps capabilities for managing the lifecycle of machine learning models, from development and training to deployment and monitoring in production environments.

### Key Features

- **Model Lifecycle Management**: Version control, registration, and deployment
- **Experiment Tracking**: Track experiments, metrics, and parameters
- **Model Monitoring**: Performance monitoring and drift detection
- **Automated Pipelines**: CI/CD for ML models with automated testing
- **Model Registry**: Centralized model storage and metadata management
- **A/B Testing**: Model comparison and gradual rollout strategies
- **Feature Stores**: Feature engineering and serving infrastructure

### Use Cases

- Managing model versions and deployments
- Tracking experiment results and model performance
- Monitoring models in production for drift and degradation
- Implementing automated ML pipelines
- A/B testing different model versions
- Feature engineering and management
- Model governance and compliance

## Architecture

This package follows **Clean Architecture** principles with clear layer separation:

```
mlops/
├── mlops/                   # Main package source
│   ├── experiments/        # Experiment tracking and management
│   │   ├── tracking/      # Experiment tracking (MLflow integration)
│   │   ├── comparison/    # Model comparison utilities
│   │   └── reporting/     # Experiment reporting
│   ├── models/            # Model lifecycle management
│   │   ├── registry/      # Model registry and versioning
│   │   ├── deployment/    # Model deployment strategies
│   │   ├── serving/       # Model serving infrastructure
│   │   └── validation/    # Model validation and testing
│   ├── monitoring/        # Production model monitoring
│   │   ├── drift/         # Data and concept drift detection
│   │   ├── performance/   # Performance monitoring
│   │   ├── alerts/        # Alerting and notifications
│   │   └── dashboards/    # Monitoring dashboards
│   ├── pipelines/         # ML pipelines and workflows
│   │   ├── training/      # Training pipelines
│   │   ├── inference/     # Inference pipelines
│   │   ├── validation/    # Validation pipelines
│   │   └── orchestration/ # Pipeline orchestration
│   ├── features/          # Feature store and engineering
│   │   ├── store/         # Feature storage and serving
│   │   ├── engineering/   # Feature engineering pipelines
│   │   └── validation/    # Feature validation
│   └── governance/        # ML governance and compliance
│       ├── lineage/       # Data and model lineage
│       ├── audit/         # Audit logging
│       └── compliance/    # Compliance checks
├── tests/                 # Package-specific tests
│   ├── unit/             # Unit tests
│   ├── integration/      # Integration tests
│   └── e2e/             # End-to-end pipeline tests
├── docs/                 # Package documentation
└── examples/             # Usage examples and tutorials
```

### Dependencies

- **Internal Dependencies**: core, algorithms, infrastructure
- **External Dependencies**: MLflow, Kubeflow, Apache Airflow
- **Optional Dependencies**: DVC, Weights & Biases, Neptune

## Installation

### Prerequisites

- Python 3.11 or higher
- Docker (for containerized deployments)
- Kubernetes (for production deployments)
- MLflow server (for experiment tracking)

### Package Installation

```bash
# Install from source (development)
cd src/packages/ai/mlops
pip install -e .

# Install with all MLOps tools
pip install mlops[all]

# Install specific components
pip install mlops[mlflow,kubeflow,monitoring]
```

## Usage

### Quick Start

```python
from ai.mlops.experiments import ExperimentTracker
from ai.mlops.models import ModelRegistry
from ai.mlops.monitoring import ModelMonitor
from ai.core.domain.entities import Dataset, Detector

# Experiment tracking
tracker = ExperimentTracker(backend="mlflow")

with tracker.start_experiment("anomaly_detection_v1") as experiment:
    # Track parameters
    experiment.log_params({
        "algorithm": "isolation_forest",
        "contamination": 0.1,
        "n_estimators": 100
    })
    
    # Train model
    detector = Detector.isolation_forest(contamination=0.1)
    detector.fit(dataset)
    
    # Track metrics
    metrics = evaluate_model(detector, test_dataset)
    experiment.log_metrics(metrics)
    
    # Register model
    registry = ModelRegistry()
    model_version = registry.register_model(
        name="anomaly_detector_v1",
        model=detector,
        experiment_id=experiment.id,
        stage="staging"
    )

# Monitor model in production
monitor = ModelMonitor(model_version=model_version)
monitor.start_monitoring()
```

### Basic Examples

#### Example 1: Model Deployment Pipeline
```python
from ai.mlops.pipelines import TrainingPipeline, DeploymentPipeline
from ai.mlops.models.deployment import KubernetesDeployment

# Create training pipeline
training_pipeline = TrainingPipeline(
    name="anomaly_detection_training",
    schedule="0 2 * * *",  # Daily at 2 AM
    config={
        "data_source": "s3://<enter-name>/datasets/",
        "algorithms": ["isolation_forest", "lof", "one_class_svm"],
        "validation_split": 0.2,
        "cross_validation_folds": 5
    }
)

# Run training pipeline
training_result = await training_pipeline.run()

# Deploy best model
if training_result.best_model.score > 0.85:
    deployment = KubernetesDeployment(
        model_version=training_result.best_model,
        replicas=3,
        resources={"cpu": "500m", "memory": "1Gi"},
        autoscaling={"min_replicas": 2, "max_replicas": 10}
    )
    
    await deployment.deploy()
```

#### Example 2: A/B Testing Framework
```python
from ai.mlops.experiments import ABTestFramework
from ai.mlops.models.serving import ModelEnsemble

# Setup A/B test
ab_test = ABTestFramework(
    name="isolation_forest_vs_lof",
    traffic_split={"model_a": 0.5, "model_b": 0.5},
    success_metrics=["precision", "recall", "f1_score"],
    duration_days=14
)

# Configure model variants
model_a = ModelRegistry().get_model("isolation_forest_v1")
model_b = ModelRegistry().get_model("lof_v1")

# Create ensemble for A/B testing
ensemble = ModelEnsemble({
    "model_a": model_a,
    "model_b": model_b
})

# Start A/B test
test_results = await ab_test.run(
    ensemble=ensemble,
    evaluation_dataset=test_dataset
)

# Analyze results
if test_results.model_b.performance > test_results.model_a.performance:
    # Promote model B to production
    await ModelRegistry().promote_model(model_b.id, stage="production")
```

### Advanced Usage

Complete MLOps workflow with monitoring and governance:

```python
from ai.mlops.governance import MLGovernance
from ai.mlops.monitoring import DriftDetector, PerformanceMonitor
from ai.mlops.features import FeatureStore
import asyncio

async def production_ml_workflow():
    # Initialize governance
    governance = MLGovernance(
        compliance_rules=["data_privacy", "model_explainability"],
        audit_enabled=True,
        lineage_tracking=True
    )
    
    # Setup feature store
    feature_store = FeatureStore(
        backend="feast",
        offline_store="s3://monorepo-features/offline",
        online_store="redis://localhost:6379"
    )
    
    # Register features
    await feature_store.register_feature_view(
        name="data_features_v1",
        entities=["device_id", "timestamp"],
        features=["temperature", "pressure", "vibration"],
        ttl_hours=24
    )
    
    # Setup monitoring
    drift_detector = DriftDetector(
        reference_data=reference_dataset,
        detection_window="1h",
        alert_threshold=0.1
    )
    
    performance_monitor = PerformanceMonitor(
        model_version="model_v1",
        metrics=["precision", "recall", "latency"],
        alert_thresholds={"precision": 0.8, "latency": "100ms"}
    )
    
    # Start monitoring
    await asyncio.gather(
        drift_detector.start_monitoring(),
        performance_monitor.start_monitoring(),
        governance.start_audit_logging()
    )
    
    # Production inference with monitoring
    async for batch in streaming_data:
        # Get fresh features
        features = await feature_store.get_online_features(
            entities=batch.entity_ids,
            feature_view="anomaly_features_v1"
        )
        
        # Run inference
        with governance.track_inference() as tracking:
            predictions = await model.predict(features)
            
            # Log for monitoring
            await drift_detector.log_predictions(features, predictions)
            await performance_monitor.log_inference(predictions, batch.ground_truth)
            
            # Governance tracking
            tracking.log_prediction_metadata(predictions)

# Run workflow
asyncio.run(production_ml_workflow())
```

### Configuration

Configure MLOps components with comprehensive settings:

```python
from ai.mlops.config import MLOpsConfig
from ai.mlops.factory import create_mlops_stack

# MLOps configuration
config = MLOpsConfig(
    experiment_tracking={
        "backend": "mlflow",
        "tracking_uri": "http://mlflow-server:5000",
        "artifact_store": "s3://monorepo-artifacts"
    },
    model_registry={
        "backend": "mlflow",
        "stage_transitions": True,
        "approval_required": True
    },
    monitoring={
        "drift_detection": True,
        "performance_monitoring": True,
        "alert_channels": ["slack", "email"],
        "dashboard_enabled": True
    },
    deployment={
        "platform": "kubernetes",
        "auto_scaling": True,
        "canary_deployment": True,
        "rollback_enabled": True
    }
)

# Create MLOps stack
mlops_stack = create_mlops_stack(config)
await mlops_stack.initialize()
```

## API Reference

### Core Classes

#### Experiments
- **`ExperimentTracker`**: Experiment tracking and management
- **`ABTestFramework`**: A/B testing infrastructure
- **`ExperimentComparison`**: Compare experiment results
- **`HyperparameterTuning`**: Automated hyperparameter optimization

#### Models
- **`ModelRegistry`**: Model versioning and metadata management
- **`ModelDeployment`**: Model deployment strategies
- **`ModelServing`**: Production model serving
- **`ModelValidation`**: Model testing and validation

#### Monitoring
- **`DriftDetector`**: Data and concept drift detection
- **`PerformanceMonitor`**: Real-time performance monitoring
- **`AlertManager`**: Alert configuration and delivery
- **`MonitoringDashboard`**: Visualization and reporting

#### Pipelines
- **`TrainingPipeline`**: Automated training workflows
- **`InferencePipeline`**: Production inference pipelines
- **`ValidationPipeline`**: Model validation workflows
- **`PipelineOrchestrator`**: Pipeline scheduling and execution

### Key Functions

```python
# Experiment management
from ai.mlops.experiments import (
    start_experiment,
    log_metrics,
    compare_experiments,
    get_best_model
)

# Model lifecycle
from ai.mlops.models import (
    register_model,
    deploy_model,
    promote_model,
    rollback_deployment
)

# Monitoring operations
from ai.mlops.monitoring import (
    detect_drift,
    monitor_performance,
    create_alert,
    generate_report
)
```

### Exceptions

- **`MLOpsError`**: Base MLOps exception
- **`ExperimentError`**: Experiment tracking errors
- **`ModelRegistryError`**: Model registry operations errors
- **`DeploymentError`**: Model deployment failures
- **`MonitoringError`**: Monitoring system errors

## Performance

Optimized for production ML operations at scale:

- **Async Operations**: Non-blocking pipeline execution
- **Distributed Training**: Multi-node training support
- **Model Caching**: Efficient model loading and caching
- **Batch Processing**: Optimized batch inference
- **Resource Management**: Dynamic resource allocation

### Benchmarks

- **Model Registration**: 1K models/hour
- **Experiment Tracking**: 10K metrics/sec
- **Inference Serving**: 1K predictions/sec per replica
- **Monitoring Latency**: <10ms overhead per prediction

## Security

- **Model Security**: Model signing and verification
- **Access Control**: Role-based access to ML assets
- **Audit Logging**: Comprehensive audit trails
- **Data Privacy**: Privacy-preserving ML techniques
- **Compliance**: Regulatory compliance checks

## Troubleshooting

### Common Issues

**Issue**: Model deployment fails
**Solution**: Check Kubernetes cluster status and resource availability

**Issue**: Experiment tracking slow
**Solution**: Optimize MLflow backend configuration and storage

**Issue**: Drift detection false positives
**Solution**: Adjust detection thresholds and reference data

### Debug Mode

```python
from ai.mlops.config import enable_debug_mode

# Enable debug mode for MLOps
enable_debug_mode(
    experiment_tracking=True,
    model_serving=True,
    monitoring=True,
    pipeline_execution=True
)
```

## Compatibility

- **Python**: 3.11, 3.12, 3.13+
- **ML Frameworks**: PyTorch, TensorFlow, scikit-learn, PyOD
- **Orchestration**: Kubeflow, Apache Airflow, Prefect
- **Monitoring**: Prometheus, Grafana, MLflow
- **Deployment**: Kubernetes, Docker, AWS SageMaker, Azure ML

## Contributing

1. **Fork and Clone**: Fork the repository and clone your fork
2. **Branch**: Create a feature branch (`git checkout -b feature/mlops-enhancement`)
3. **Develop**: Implement new MLOps capabilities
4. **Test**: Add comprehensive tests including pipeline tests
5. **Document**: Update documentation and configuration examples
6. **Commit**: Use conventional commit messages
7. **Pull Request**: Submit a PR with clear description

### Adding New MLOps Components

Follow the MLOps pattern for consistency:

```python
from ai.mlops.base import BaseMLOpsComponent

class NewMLOpsComponent(BaseMLOpsComponent):
    def __init__(self, config: ComponentConfig):
        super().__init__(config)
        self.initialize_component()
    
    async def start(self) -> None:
        await self.setup_resources()
    
    async def stop(self) -> None:
        await self.cleanup_resources()
    
    async def health_check(self) -> bool:
        return await self.check_component_health()
```

## Support

- **Documentation**: [Package docs](docs/)
- **MLOps Guide**: [MLOps Best Practices Guide](docs/mlops_guide.md)
- **Pipeline Examples**: [Pipeline Cookbook](docs/pipeline_cookbook.md)
- **Issues**: [GitHub Issues](../../../issues)
- **Discussions**: [GitHub Discussions](../../../discussions)

## License

MIT License. See [LICENSE](../../../LICENSE) file for details.

---

**Part of the [monorepo](../../../) monorepo** - Advanced ML platform