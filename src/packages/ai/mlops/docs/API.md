# MLOps API Reference

## Overview

The MLOps package provides a comprehensive set of APIs for managing machine learning operations, from experiment tracking to model deployment and monitoring. This reference covers all public APIs available in the package.

## Core APIs

### ExperimentTracker

The main class for tracking machine learning experiments.

```python
from mlops.experiments import ExperimentTracker

tracker = ExperimentTracker(backend="mlflow", tracking_uri="http://localhost:5000")
```

#### Methods

##### `start_experiment(name: str, description: str = None) -> ExperimentContext`

Start a new experiment tracking session.

**Parameters:**
- `name`: Unique name for the experiment
- `description`: Optional description of the experiment

**Returns:** `ExperimentContext` - Context manager for the experiment

**Example:**
```python
with tracker.start_experiment("anomaly_detection_v1") as experiment:
    experiment.log_params({"algorithm": "isolation_forest"})
    experiment.log_metrics({"accuracy": 0.95})
```

##### `get_experiment(experiment_id: str) -> Experiment`

Retrieve an experiment by ID.

**Parameters:**
- `experiment_id`: Unique identifier for the experiment

**Returns:** `Experiment` - Experiment object with metadata

##### `list_experiments(limit: int = 100) -> List[Experiment]`

List all experiments.

**Parameters:**
- `limit`: Maximum number of experiments to return

**Returns:** `List[Experiment]` - List of experiments

### ModelRegistry

Central registry for managing model versions and metadata.

```python
from mlops.models import ModelRegistry

registry = ModelRegistry(backend="mlflow")
```

#### Methods

##### `register_model(name: str, model: Any, experiment_id: str = None, stage: str = "staging") -> ModelVersion`

Register a new model version.

**Parameters:**
- `name`: Model name
- `model`: Trained model object
- `experiment_id`: Associated experiment ID
- `stage`: Initial stage ("staging", "production", "archived")

**Returns:** `ModelVersion` - Registered model version

##### `get_model(name: str, version: str = "latest") -> ModelVersion`

Retrieve a model by name and version.

**Parameters:**
- `name`: Model name
- `version`: Model version or "latest"

**Returns:** `ModelVersion` - Model version object

##### `promote_model(model_id: str, stage: str) -> bool`

Promote a model to a different stage.

**Parameters:**
- `model_id`: Model version ID
- `stage`: Target stage

**Returns:** `bool` - Success status

##### `list_models(filter_dict: Dict = None) -> List[ModelVersion]`

List all models with optional filtering.

**Parameters:**
- `filter_dict`: Optional filter criteria

**Returns:** `List[ModelVersion]` - List of model versions

### ModelMonitor

Real-time monitoring for deployed models.

```python
from mlops.monitoring import ModelMonitor

monitor = ModelMonitor(model_version="anomaly_detector_v1")
```

#### Methods

##### `start_monitoring() -> None`

Start monitoring the model for drift and performance.

##### `stop_monitoring() -> None`

Stop monitoring the model.

##### `get_monitoring_status() -> MonitoringStatus`

Get current monitoring status.

**Returns:** `MonitoringStatus` - Current monitoring state

##### `set_alert_thresholds(thresholds: Dict[str, float]) -> None`

Configure alert thresholds.

**Parameters:**
- `thresholds`: Dictionary of metric names to threshold values

### DriftDetector

Detect data and concept drift in production models.

```python
from mlops.monitoring import DriftDetector

detector = DriftDetector(reference_data=reference_dataset)
```

#### Methods

##### `detect_drift(new_data: np.ndarray) -> DriftResult`

Detect drift in new data compared to reference.

**Parameters:**
- `new_data`: New data to check for drift

**Returns:** `DriftResult` - Drift detection results

##### `set_detection_window(window_size: str) -> None`

Set the detection window size.

**Parameters:**
- `window_size`: Window size (e.g., "1h", "1d")

## Pipeline APIs

### TrainingPipeline

Automated training pipeline for ML models.

```python
from mlops.pipelines import TrainingPipeline

pipeline = TrainingPipeline(
    name="anomaly_detection_training",
    schedule="0 2 * * *"
)
```

#### Methods

##### `run() -> TrainingResult`

Execute the training pipeline.

**Returns:** `TrainingResult` - Training results and best model

##### `set_config(config: Dict) -> None`

Set pipeline configuration.

**Parameters:**
- `config`: Configuration dictionary

### DeploymentPipeline

Automated deployment pipeline for ML models.

```python
from mlops.pipelines import DeploymentPipeline

pipeline = DeploymentPipeline(
    model_version="anomaly_detector_v1",
    target_environment="production"
)
```

#### Methods

##### `deploy() -> DeploymentResult`

Deploy the model to target environment.

**Returns:** `DeploymentResult` - Deployment status and details

##### `rollback() -> bool`

Rollback to previous deployment.

**Returns:** `bool` - Success status

## Model Serving APIs

### ModelEnsemble

Ensemble serving for multiple models.

```python
from mlops.models.serving import ModelEnsemble

ensemble = ModelEnsemble({
    "model_a": model_a,
    "model_b": model_b
})
```

#### Methods

##### `predict(data: np.ndarray, strategy: str = "average") -> np.ndarray`

Make predictions using ensemble strategy.

**Parameters:**
- `data`: Input data for prediction
- `strategy`: Ensemble strategy ("average", "voting", "weighted")

**Returns:** `np.ndarray` - Ensemble predictions

##### `add_model(name: str, model: Any, weight: float = 1.0) -> None`

Add a model to the ensemble.

**Parameters:**
- `name`: Model name
- `model`: Model object
- `weight`: Model weight in ensemble

##### `remove_model(name: str) -> None`

Remove a model from the ensemble.

**Parameters:**
- `name`: Model name to remove

## Feature Store APIs

### FeatureStore

Feature storage and serving infrastructure.

```python
from mlops.features import FeatureStore

store = FeatureStore(
    backend="feast",
    offline_store="s3://features/offline",
    online_store="redis://localhost:6379"
)
```

#### Methods

##### `register_feature_view(name: str, entities: List[str], features: List[str], ttl_hours: int = 24) -> None`

Register a feature view.

**Parameters:**
- `name`: Feature view name
- `entities`: List of entity names
- `features`: List of feature names
- `ttl_hours`: Time-to-live in hours

##### `get_online_features(entities: List[str], feature_view: str) -> Dict`

Get features from online store.

**Parameters:**
- `entities`: List of entity IDs
- `feature_view`: Feature view name

**Returns:** `Dict` - Feature values

##### `get_offline_features(entities: List[str], feature_view: str, start_time: datetime, end_time: datetime) -> pd.DataFrame`

Get features from offline store.

**Parameters:**
- `entities`: List of entity IDs
- `feature_view`: Feature view name
- `start_time`: Start time for features
- `end_time`: End time for features

**Returns:** `pd.DataFrame` - Feature dataset

## A/B Testing APIs

### ABTestFramework

Framework for A/B testing models.

```python
from mlops.experiments import ABTestFramework

ab_test = ABTestFramework(
    name="model_comparison",
    traffic_split={"model_a": 0.5, "model_b": 0.5}
)
```

#### Methods

##### `run(ensemble: ModelEnsemble, evaluation_dataset: Any) -> ABTestResult`

Run A/B test.

**Parameters:**
- `ensemble`: Model ensemble for testing
- `evaluation_dataset`: Dataset for evaluation

**Returns:** `ABTestResult` - Test results and statistics

##### `get_results() -> ABTestResult`

Get current A/B test results.

**Returns:** `ABTestResult` - Current test results

## Deployment APIs

### KubernetesDeployment

Kubernetes-based model deployment.

```python
from mlops.models.deployment import KubernetesDeployment

deployment = KubernetesDeployment(
    model_version="anomaly_detector_v1",
    replicas=3,
    resources={"cpu": "500m", "memory": "1Gi"}
)
```

#### Methods

##### `deploy() -> DeploymentResult`

Deploy model to Kubernetes.

**Returns:** `DeploymentResult` - Deployment status

##### `scale(replicas: int) -> bool`

Scale deployment.

**Parameters:**
- `replicas`: Target number of replicas

**Returns:** `bool` - Success status

##### `health_check() -> bool`

Check deployment health.

**Returns:** `bool` - Health status

## Governance APIs

### MLGovernance

ML governance and compliance framework.

```python
from mlops.governance import MLGovernance

governance = MLGovernance(
    compliance_rules=["data_privacy", "model_explainability"],
    audit_enabled=True
)
```

#### Methods

##### `track_inference() -> InferenceContext`

Track inference for governance.

**Returns:** `InferenceContext` - Context manager for inference tracking

##### `start_audit_logging() -> None`

Start audit logging.

##### `generate_compliance_report() -> ComplianceReport`

Generate compliance report.

**Returns:** `ComplianceReport` - Compliance status and recommendations

## Data Transfer Objects

### TrainingResult

Result object from training operations.

```python
class TrainingResult:
    best_model: ModelVersion
    metrics: Dict[str, float]
    training_time: float
    validation_score: float
    experiment_id: str
```

### DeploymentResult

Result object from deployment operations.

```python
class DeploymentResult:
    deployment_id: str
    status: str
    endpoint_url: str
    replicas: int
    resources: Dict[str, str]
    deployment_time: datetime
```

### DriftResult

Result object from drift detection.

```python
class DriftResult:
    drift_detected: bool
    drift_score: float
    drift_type: str  # "data" or "concept"
    affected_features: List[str]
    detection_time: datetime
```

### ABTestResult

Result object from A/B testing.

```python
class ABTestResult:
    test_id: str
    winner: str
    confidence: float
    metrics: Dict[str, Dict[str, float]]
    duration: timedelta
    traffic_split: Dict[str, float]
```

## Exception Handling

### MLOpsError

Base exception for all MLOps operations.

```python
class MLOpsError(Exception):
    """Base exception for MLOps operations."""
    pass
```

### ExperimentError

Exception for experiment tracking errors.

```python
class ExperimentError(MLOpsError):
    """Exception for experiment tracking operations."""
    pass
```

### ModelRegistryError

Exception for model registry operations.

```python
class ModelRegistryError(MLOpsError):
    """Exception for model registry operations."""
    pass
```

### DeploymentError

Exception for deployment operations.

```python
class DeploymentError(MLOpsError):
    """Exception for model deployment operations."""
    pass
```

### MonitoringError

Exception for monitoring operations.

```python
class MonitoringError(MLOpsError):
    """Exception for monitoring operations."""
    pass
```

## Configuration

### MLOpsConfig

Configuration class for MLOps components.

```python
from mlops.config import MLOpsConfig

config = MLOpsConfig(
    experiment_tracking={
        "backend": "mlflow",
        "tracking_uri": "http://localhost:5000"
    },
    model_registry={
        "backend": "mlflow",
        "stage_transitions": True
    },
    monitoring={
        "drift_detection": True,
        "alert_channels": ["slack", "email"]
    }
)
```

## Utility Functions

### create_mlops_stack

Factory function to create MLOps stack.

```python
from mlops.factory import create_mlops_stack

stack = create_mlops_stack(config)
await stack.initialize()
```

### enable_debug_mode

Enable debug mode for MLOps operations.

```python
from mlops.config import enable_debug_mode

enable_debug_mode(
    experiment_tracking=True,
    model_serving=True,
    monitoring=True
)
```

## Type Hints

The package provides comprehensive type hints for all public APIs. Import types from:

```python
from mlops.types import (
    ExperimentContext,
    ModelVersion,
    MonitoringStatus,
    TrainingResult,
    DeploymentResult,
    DriftResult,
    ABTestResult
)
```

## Usage Examples

Complete usage examples can be found in the [examples](../examples/) directory and the main [README](../README.md) file.

## Support

For API-specific questions and issues:
- Check the [troubleshooting guide](troubleshooting.md)
- Review [examples](../examples/)
- Open an issue on GitHub