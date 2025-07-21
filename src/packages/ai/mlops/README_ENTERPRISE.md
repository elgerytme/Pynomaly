# anomaly_detection MLOps

Comprehensive MLOps package for anomaly_detection with enterprise integrations.

## Features

### Core MLOps
- **Experiment Tracking**: Local file-based experiment management
- **Model Registry**: Model versioning and lifecycle management  
- **Pipeline Orchestration**: Basic ML pipeline execution
- **Performance Monitoring**: Built-in metrics collection

### Enterprise Integrations (Optional)
- **MLflow**: Experiment tracking and model registry
- **Kubeflow**: Pipeline orchestration and model serving
- **Datadog**: Application and model monitoring
- **New Relic**: Performance monitoring and alerting
- **Weights & Biases**: Experiment tracking and visualization
- **Neptune**: ML metadata management

## Installation

### Basic Installation
```bash
pip install anomaly_detection-mlops
```

### With Enterprise Features
```bash
# MLflow integration
pip install anomaly_detection-mlops[mlflow]

# Kubeflow integration  
pip install anomaly_detection-mlops[kubeflow]

# Monitoring integrations
pip install anomaly_detection-mlops[datadog,newrelic]

# All enterprise features
pip install anomaly_detection-mlops[enterprise]
```

## Quick Start

### Basic Usage
```python
from anomaly_detection_mlops import (
    ExperimentTrackingService, 
    ModelRegistryService,
    Model, ModelType
)

# Experiment tracking
tracker = ExperimentTrackingService(Path("./experiments"))
experiment_id = await tracker.create_experiment(
    name="anomaly_detection_v1",
    description="Testing isolation forest"
)

# Model registry
registry = ModelRegistryService(Path("./models"))
model = await registry.register_model(
    name="isolation_forest_v1",
    description="Isolation Forest for network anomalies",
    algorithm="IsolationForest",
    model_type=ModelType.UNSUPERVISED
)
```

### Enterprise Usage
```python
from anomaly_detection_mlops import (
    EnterpriseMlopsService,
    MLflowIntegration,
    DatadogIntegration
)

# Initialize integrations
mlflow = MLflowIntegration("http://mlflow-server:5000")
datadog = DatadogIntegration("api-key", "app-key")

# Create enterprise service
mlops = EnterpriseMlopsService(
    mlflow_integration=mlflow,
    datadog_integration=datadog
)

# Create experiment with external tracking
experiment = await mlops.create_experiment(
    name="production_experiment",
    project_name="anomaly_detection",
    tags={"environment": "prod", "model": "v2"}
)

# Log metrics to multiple platforms
await mlops.log_metrics(
    experiment=experiment,
    metrics={"accuracy": 0.95, "f1_score": 0.92}
)
```

## Architecture

### Core Components
- **Domain Entities**: Model, Experiment, Pipeline definitions
- **Application Services**: Business logic and orchestration
- **Infrastructure**: Storage and external integrations

### Enterprise Extensions
- **Optional Dependencies**: Enterprise features don't affect core functionality
- **Graceful Degradation**: Missing integrations are handled transparently
- **Multi-tenant Support**: Enterprise entities support tenant isolation

## Configuration

### Environment Variables
```bash
# MLflow
MLFLOW_TRACKING_URI=http://localhost:5000
MLFLOW_S3_ENDPOINT_URL=http://minio:9000

# Kubeflow
KUBEFLOW_HOST=http://kubeflow.local
KUBEFLOW_NAMESPACE=kubeflow-user

# Datadog
DATADOG_API_KEY=your-api-key
DATADOG_APP_KEY=your-app-key

# New Relic
NEW_RELIC_LICENSE_KEY=your-license-key
```

### Programmatic Configuration
```python
from anomaly_detection_mlops import EnterpriseMlopsService
from anomaly_detection_mlops.infrastructure.enterprise import (
    MLflowIntegration,
    KubeflowIntegration
)

# Configure integrations
mlflow = MLflowIntegration(
    tracking_uri="http://mlflow:5000",
    artifact_uri="s3://mlflow-artifacts",
    experiment_name="production"
)

kubeflow = KubeflowIntegration(
    host="http://kubeflow:8080",
    namespace="ml-pipelines"
)

# Create service with integrations
service = EnterpriseMlopsService(
    mlflow_integration=mlflow,
    kubeflow_integration=kubeflow
)
```

## Development

### Setup
```bash
git clone https://github.com/your-org/anomaly_detection
cd anomaly_detection/src/packages/ai/machine_learning/mlops

# Install with dev dependencies
pip install -e .[dev,enterprise]
```

### Testing
```bash
# Run tests
pytest

# Run with enterprise integration tests
pytest -m enterprise

# Run specific platform tests
pytest -m mlflow
pytest -m datadog
```

### Linting
```bash
# Format code
black .
ruff --fix .

# Type checking
mypy src/
```

## Contributing

See [CONTRIBUTING.md](CONTRIBUTING.md) for development guidelines.

## License

MIT License - see [LICENSE](LICENSE) for details.