# Getting Started with MLOps

This guide will help you get started with the MLOps package for managing machine learning operations in the anomaly detection platform.

## Prerequisites

Before you begin, ensure you have:
- Python 3.11 or higher
- Docker (for containerized deployments)
- Access to a Kubernetes cluster (for production deployments)
- MLflow server running (for experiment tracking)

## Installation

### Basic Installation

```bash
# Install the MLOps package
pip install monorepo-mlops

# Or install from source
cd src/packages/ai/mlops
pip install -e .
```

### Full Installation with All Dependencies

```bash
# Install with all MLOps tools
pip install monorepo-mlops[all]

# Or install specific components
pip install monorepo-mlops[mlflow,kubeflow,monitoring]
```

## Quick Start

### 1. Basic Experiment Tracking

Start with simple experiment tracking to understand the core concepts:

```python
from monorepo.mlops.experiments import ExperimentTracker
from monorepo.mlops.models import ModelRegistry
import numpy as np
from sklearn.ensemble import IsolationForest

# Initialize experiment tracker
tracker = ExperimentTracker(backend="mlflow")

# Start an experiment
with tracker.start_experiment("my_first_experiment") as experiment:
    # Log parameters
    experiment.log_params({
        "algorithm": "isolation_forest",
        "contamination": 0.1,
        "n_estimators": 100
    })
    
    # Create and train model
    model = IsolationForest(contamination=0.1, n_estimators=100)
    X = np.random.randn(1000, 10)
    model.fit(X)
    
    # Log metrics
    experiment.log_metrics({
        "training_samples": len(X),
        "features": X.shape[1]
    })
    
    # Register model
    registry = ModelRegistry()
    model_version = registry.register_model(
        name="isolation_forest_v1",
        model=model,
        experiment_id=experiment.id
    )
    
    print(f"Model registered with ID: {model_version.id}")
```

### 2. Model Deployment

Deploy your trained model to production:

```python
from monorepo.mlops.models.deployment import KubernetesDeployment

# Create deployment configuration
deployment = KubernetesDeployment(
    model_version=model_version,
    replicas=3,
    resources={"cpu": "500m", "memory": "1Gi"},
    autoscaling={"min_replicas": 2, "max_replicas": 10}
)

# Deploy model
result = await deployment.deploy()
print(f"Model deployed at: {result.endpoint_url}")
```

### 3. Model Monitoring

Set up monitoring for your deployed model:

```python
from monorepo.mlops.monitoring import ModelMonitor, DriftDetector

# Initialize monitoring
monitor = ModelMonitor(model_version=model_version)

# Configure drift detection
drift_detector = DriftDetector(
    reference_data=X,  # Training data as reference
    detection_window="1h",
    alert_threshold=0.1
)

# Start monitoring
await monitor.start_monitoring()
print("Monitoring started for model")
```

## Core Concepts

### Experiment Tracking

Experiment tracking helps you manage and compare different model training runs:

```python
# Track multiple experiments
experiments = []
for contamination in [0.05, 0.1, 0.15]:
    with tracker.start_experiment(f"contamination_{contamination}") as exp:
        exp.log_params({"contamination": contamination})
        
        model = IsolationForest(contamination=contamination)
        model.fit(X)
        
        # Evaluate model
        predictions = model.predict(X_test)
        accuracy = accuracy_score(y_test, predictions)
        
        exp.log_metrics({"accuracy": accuracy})
        experiments.append(exp)

# Find best experiment
best_exp = max(experiments, key=lambda e: e.get_metric("accuracy"))
print(f"Best contamination: {best_exp.get_param('contamination')}")
```

### Model Registry

The model registry manages model versions and lifecycle:

```python
# Register different model versions
registry = ModelRegistry()

# Register staging version
staging_version = registry.register_model(
    name="anomaly_detector",
    model=model,
    stage="staging"
)

# Test in staging
staging_results = evaluate_model(staging_version.model, test_data)

# Promote to production if good
if staging_results.accuracy > 0.9:
    registry.promote_model(staging_version.id, "production")
    print("Model promoted to production")
```

### Pipeline Automation

Automate your ML workflows with pipelines:

```python
from monorepo.mlops.pipelines import TrainingPipeline

# Create automated training pipeline
pipeline = TrainingPipeline(
    name="daily_anomaly_training",
    schedule="0 2 * * *",  # Daily at 2 AM
    config={
        "data_source": "s3://data/daily_batch/",
        "algorithms": ["isolation_forest", "lof"],
        "validation_split": 0.2,
        "auto_deploy": True,
        "approval_required": True
    }
)

# Run pipeline
result = await pipeline.run()
print(f"Pipeline completed. Best model: {result.best_model.name}")
```

## Advanced Features

### A/B Testing

Test different models in production:

```python
from monorepo.mlops.experiments import ABTestFramework
from monorepo.mlops.models.serving import ModelEnsemble

# Setup A/B test
ab_test = ABTestFramework(
    name="isolation_forest_vs_lof",
    traffic_split={"model_a": 0.5, "model_b": 0.5},
    success_metrics=["precision", "recall", "f1_score"],
    duration_days=14
)

# Create ensemble for A/B testing
model_a = registry.get_model("isolation_forest_v1")
model_b = registry.get_model("lof_v1")

ensemble = ModelEnsemble({
    "model_a": model_a,
    "model_b": model_b
})

# Run A/B test
test_results = await ab_test.run(
    ensemble=ensemble,
    evaluation_dataset=test_dataset
)

# Analyze results
print(f"Winner: {test_results.winner}")
print(f"Confidence: {test_results.confidence:.2f}")
```

### Feature Store Integration

Manage features across your ML pipeline:

```python
from monorepo.mlops.features import FeatureStore

# Initialize feature store
feature_store = FeatureStore(
    backend="feast",
    offline_store="s3://features/offline",
    online_store="redis://localhost:6379"
)

# Register feature view
await feature_store.register_feature_view(
    name="anomaly_features_v1",
    entities=["device_id", "timestamp"],
    features=["temperature", "pressure", "vibration"],
    ttl_hours=24
)

# Get features for inference
features = await feature_store.get_online_features(
    entities=["device_001", "device_002"],
    feature_view="anomaly_features_v1"
)
```

### Governance and Compliance

Implement ML governance for production systems:

```python
from monorepo.mlops.governance import MLGovernance

# Initialize governance
governance = MLGovernance(
    compliance_rules=["data_privacy", "model_explainability"],
    audit_enabled=True,
    lineage_tracking=True
)

# Track inference with governance
with governance.track_inference() as tracking:
    predictions = model.predict(features)
    
    # Log governance metadata
    tracking.log_prediction_metadata(predictions)
    tracking.log_feature_usage(features)
    
    # Generate explanation if required
    if governance.requires_explanation(predictions):
        explanations = explainer.explain(features, predictions)
        tracking.log_explanations(explanations)
```

## Configuration

### Environment Configuration

Set up your environment variables:

```bash
# MLflow configuration
export MLFLOW_TRACKING_URI=http://localhost:5000
export MLFLOW_S3_ENDPOINT_URL=http://localhost:9000

# Kubernetes configuration
export KUBECONFIG=/path/to/kubeconfig

# Monitoring configuration
export PROMETHEUS_URL=http://localhost:9090
export GRAFANA_URL=http://localhost:3000
```

### Configuration File

Create a comprehensive configuration:

```python
from monorepo.mlops.config import MLOpsConfig

config = MLOpsConfig(
    experiment_tracking={
        "backend": "mlflow",
        "tracking_uri": "http://mlflow-server:5000",
        "artifact_store": "s3://mlops-artifacts",
        "auto_log": True
    },
    model_registry={
        "backend": "mlflow",
        "stage_transitions": True,
        "approval_required": True,
        "webhook_url": "http://slack-webhook"
    },
    monitoring={
        "drift_detection": True,
        "performance_monitoring": True,
        "alert_channels": ["slack", "email"],
        "dashboard_enabled": True,
        "metrics_retention_days": 30
    },
    deployment={
        "platform": "kubernetes",
        "namespace": "mlops-production",
        "auto_scaling": True,
        "canary_deployment": True,
        "rollback_enabled": True,
        "health_check_interval": 30
    },
    feature_store={
        "backend": "feast",
        "offline_store": "s3://feature-store/offline",
        "online_store": "redis://redis-cluster:6379",
        "feature_ttl_hours": 24
    }
)
```

## Best Practices

### 1. Experiment Organization

```python
# Use hierarchical naming
experiment_names = [
    "anomaly_detection/isolation_forest/v1",
    "anomaly_detection/isolation_forest/v2",
    "anomaly_detection/lof/v1"
]

# Log comprehensive metadata
experiment.log_params({
    "algorithm": "isolation_forest",
    "contamination": 0.1,
    "n_estimators": 100,
    "random_state": 42,
    "data_version": "v2.1",
    "preprocessing": "standard_scaler"
})
```

### 2. Model Versioning

```python
# Use semantic versioning
model_versions = {
    "major": "1.0.0",  # Breaking changes
    "minor": "1.1.0",  # New features
    "patch": "1.1.1"   # Bug fixes
}

# Include model metadata
registry.register_model(
    name="anomaly_detector",
    model=model,
    version="1.1.0",
    metadata={
        "framework": "scikit-learn",
        "training_data": "production_2024_01",
        "performance": {"accuracy": 0.95, "f1": 0.92},
        "dependencies": ["scikit-learn==1.3.0"]
    }
)
```

### 3. Monitoring Strategy

```python
# Set up comprehensive monitoring
monitor_config = {
    "drift_detection": {
        "methods": ["ks_test", "psi", "js_divergence"],
        "threshold": 0.1,
        "window_size": "1h"
    },
    "performance_monitoring": {
        "metrics": ["precision", "recall", "f1_score"],
        "thresholds": {"precision": 0.8, "recall": 0.7},
        "alert_delay": "5m"
    },
    "system_monitoring": {
        "latency_threshold": "100ms",
        "error_rate_threshold": 0.01,
        "throughput_threshold": 1000
    }
}
```

## Common Use Cases

### 1. Daily Model Retraining

```python
from monorepo.mlops.pipelines import TrainingPipeline
from monorepo.mlops.data import DataLoader

# Automated daily retraining
async def daily_retraining():
    # Load fresh data
    data_loader = DataLoader("s3://data/daily/")
    new_data = await data_loader.load_latest()
    
    # Train new model
    pipeline = TrainingPipeline("daily_retrain")
    result = await pipeline.run(data=new_data)
    
    # Compare with current production model
    current_model = registry.get_model("anomaly_detector", "production")
    
    if result.best_model.score > current_model.score:
        # Deploy new model
        await deploy_model(result.best_model)
        print("New model deployed")
    else:
        print("Current model still best")
```

### 2. Multi-Model Ensemble

```python
# Create ensemble of different algorithms
models = {
    "isolation_forest": IsolationForest(contamination=0.1),
    "lof": LocalOutlierFactor(contamination=0.1),
    "one_class_svm": OneClassSVM(nu=0.1)
}

# Train all models
trained_models = {}
for name, model in models.items():
    model.fit(X_train)
    trained_models[name] = model

# Create ensemble
ensemble = ModelEnsemble(trained_models)
predictions = ensemble.predict(X_test, strategy="voting")
```

### 3. Model Performance Monitoring

```python
# Set up performance monitoring
performance_monitor = PerformanceMonitor(
    model_version="anomaly_detector_v1",
    metrics=["precision", "recall", "f1_score"],
    alert_thresholds={
        "precision": 0.8,
        "recall": 0.7,
        "f1_score": 0.75
    }
)

# Monitor in production
async def monitor_model_performance():
    async for batch in production_data_stream:
        predictions = model.predict(batch.features)
        ground_truth = batch.labels
        
        # Log performance metrics
        await performance_monitor.log_batch_performance(
            predictions=predictions,
            ground_truth=ground_truth,
            timestamp=batch.timestamp
        )
```

## Troubleshooting

### Common Issues

1. **MLflow connection errors**: Check MLflow server status and network connectivity
2. **Kubernetes deployment failures**: Verify cluster resources and permissions
3. **Model registry conflicts**: Use proper versioning and naming conventions
4. **Monitoring alerts**: Review threshold settings and data quality

### Debug Mode

```python
# Enable debug logging
from monorepo.mlops.config import enable_debug_mode

enable_debug_mode(
    experiment_tracking=True,
    model_serving=True,
    monitoring=True
)
```

## Next Steps

1. **Explore Examples**: Check out the [examples](../examples/) directory
2. **Read API Documentation**: Review the [API reference](API.md)
3. **Advanced Features**: Learn about [advanced usage patterns](advanced/README.md)
4. **Production Deployment**: Follow the [production guide](production.md)
5. **Integration**: See how to integrate with other packages

## Support

- **Documentation**: [Full documentation](../docs/)
- **Examples**: [Code examples](../examples/)
- **Issues**: [GitHub Issues](https://github.com/your-org/repo/issues)
- **Community**: [Discussions](https://github.com/your-org/repo/discussions)