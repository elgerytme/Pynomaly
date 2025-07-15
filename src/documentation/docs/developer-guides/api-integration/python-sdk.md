# Python SDK Reference

🍞 **Breadcrumb:** 🏠 [Home](../../index.md) > 👨‍💻 [Developer Guides](../README.md) > 🔌 [API Integration](README.md) > 🐍 Python SDK

---


The Pynomaly Python SDK provides a comprehensive programmatic interface for anomaly detection, built on clean architecture principles with full async support.

## Installation

```bash
# Basic installation
pip install pynomaly

# With all ML backends
pip install pynomaly[all]

# Specific backends
pip install pynomaly[tensorflow,pytorch,jax]

# With data processing libraries
pip install pynomaly[polars,spark]
```

## Quick Start

```python
import asyncio
from pynomaly import Pynomaly
from pynomaly.domain.entities import Dataset
import pandas as pd

async def main():
    # Initialize Pynomaly
    pynomaly = Pynomaly()

    # Load data
    data = pd.DataFrame({
        'feature_1': [1, 2, 3, 100],  # 100 is anomaly
        'feature_2': [1, 2, 3, 200]   # 200 is anomaly
    })

    # Create dataset
    dataset = await pynomaly.datasets.create_from_dataframe(
        data, name="sample_data"
    )

    # Create and train detector
    detector = await pynomaly.detectors.create(
        name="fraud_detector",
        algorithm="IsolationForest",
        contamination_rate=0.25
    )

    await detector.fit(dataset)

    # Detect anomalies
    result = await detector.predict(dataset)

    print(f"Found {len(result.anomalies)} anomalies")
    for anomaly in result.anomalies:
        print(f"  Index {anomaly.index}: score {anomaly.score.value:.3f}")

if __name__ == "__main__":
    asyncio.run(main())
```

## Core Classes

### Pynomaly Client

Main entry point for the SDK.

```python
from pynomaly import Pynomaly
from pynomaly.infrastructure.config import Settings

# Basic initialization
pynomaly = Pynomaly()

# With custom configuration
settings = Settings(
    storage_path="./custom_data",
    model_storage_path="./custom_models"
)
pynomaly = Pynomaly(settings=settings)

# With dependency injection container
from pynomaly.infrastructure.config.container import Container
container = Container()
pynomaly = Pynomaly(container=container)
```

### Dataset Management

#### Creating Datasets

```python
import pandas as pd
import numpy as np

# From pandas DataFrame
data = pd.DataFrame({
    'amount': [100, 200, 5000, 50],  # 5000 is anomalous
    'merchant_category': [1, 2, 1, 3]
})

dataset = await pynomaly.datasets.create_from_dataframe(
    data,
    name="transactions",
    description="Credit card transactions"
)

# From CSV file
dataset = await pynomaly.datasets.load_csv(
    "data.csv",
    name="csv_data",
    target_column="is_fraud"  # For labeled data
)

# From Parquet file
dataset = await pynomaly.datasets.load_parquet(
    "data.parquet",
    name="parquet_data"
)

# From numpy array
X = np.random.normal(0, 1, (1000, 5))
dataset = await pynomaly.datasets.create_from_numpy(
    X,
    name="numpy_data",
    feature_names=['f1', 'f2', 'f3', 'f4', 'f5']
)

# From JSON data
data = [
    {"amount": 100, "merchant": "grocery"},
    {"amount": 200, "merchant": "gas"},
    {"amount": 5000, "merchant": "online"}  # Anomalous
]

dataset = await pynomaly.datasets.create_from_json(
    data,
    name="json_data"
)
```

#### Dataset Operations

```python
# List datasets
datasets = await pynomaly.datasets.list()
for dataset in datasets:
    print(f"{dataset.name}: {dataset.n_samples} samples")

# Get dataset info
dataset = await pynomaly.datasets.get("dataset_id")
print(f"Features: {dataset.get_numeric_features()}")
print(f"Shape: {dataset.data.shape}")

# Dataset statistics
stats = await dataset.get_statistics()
print(f"Mean: {stats['mean']}")
print(f"Std: {stats['std']}")

# Data quality analysis
quality = await dataset.analyze_quality()
print(f"Missing values: {quality['missing_values']}")
print(f"Duplicates: {quality['duplicates']}")

# Sample dataset
sample = await dataset.sample(n=100, random_state=42)

# Split dataset
train_ds, test_ds = await dataset.split(
    train_size=0.8,
    random_state=42,
    stratify=True  # For labeled data
)
```

### Detector Management

#### Creating Detectors

```python
# PyOD algorithms
detector = await pynomaly.detectors.create(
    name="isolation_forest",
    algorithm="IsolationForest",
    adapter="pyod",
    contamination_rate=0.1,
    n_estimators=100,
    random_state=42
)

# Scikit-learn algorithms
detector = await pynomaly.detectors.create(
    name="sklearn_lof",
    algorithm="LocalOutlierFactor",
    adapter="sklearn",
    contamination_rate=0.1,
    n_neighbors=20
)

# TensorFlow neural networks
detector = await pynomaly.detectors.create(
    name="tensorflow_ae",
    algorithm="AutoEncoder",
    adapter="tensorflow",
    contamination_rate=0.1,
    encoding_dim=32,
    hidden_layers=[64, 32],
    epochs=100,
    learning_rate=0.001
)

# PyTorch models
detector = await pynomaly.detectors.create(
    name="pytorch_vae",
    algorithm="VAE",
    adapter="pytorch",
    contamination_rate=0.1,
    latent_dim=16,
    hidden_dims=[64, 32],
    beta=1.0
)

# JAX high-performance models
detector = await pynomaly.detectors.create(
    name="jax_autoencoder",
    algorithm="AutoEncoder",
    adapter="jax",
    contamination_rate=0.1,
    encoding_dim=32,
    hidden_dims=[64, 32],
    epochs=100,
    learning_rate=0.001
)
```

#### Algorithm Information

```python
# List available algorithms
algorithms = await pynomaly.detectors.list_algorithms()
for adapter, algos in algorithms.items():
    print(f"{adapter}: {', '.join(algos)}")

# Get algorithm details
info = await pynomaly.detectors.get_algorithm_info(
    "IsolationForest",
    adapter="pyod"
)
print(f"Description: {info['description']}")
print(f"Parameters: {info['parameters']}")

# Check algorithm compatibility
compatible = await pynomaly.detectors.check_compatibility(
    algorithm="AutoEncoder",
    adapter="tensorflow",
    dataset=dataset
)
```

#### Training Detectors

```python
# Basic training
await detector.fit(dataset)

# Training with validation
await detector.fit(
    dataset,
    validation_split=0.2,
    early_stopping=True,
    verbose=True
)

# Cross-validation training
cv_results = await detector.fit_with_cv(
    dataset,
    cv_folds=5,
    metrics=['precision', 'recall', 'f1_score']
)

# Hyperparameter tuning
best_params = await detector.tune_hyperparameters(
    dataset,
    param_grid={
        'contamination': [0.05, 0.1, 0.15],
        'n_estimators': [100, 200, 300]
    },
    cv_folds=3,
    scoring='f1_score'
)
```

### Anomaly Detection

#### Basic Detection

```python
# Detect on dataset
result = await detector.predict(dataset)

print(f"Detected {len(result.anomalies)} anomalies")
print(f"Anomaly rate: {result.anomaly_rate:.3f}")
print(f"Threshold: {result.threshold:.3f}")

# Access individual anomalies
for anomaly in result.anomalies:
    print(f"Index: {anomaly.index}")
    print(f"Score: {anomaly.score.value:.3f}")
    print(f"Severity: {anomaly.get_severity()}")

    if anomaly.timestamp:
        print(f"Timestamp: {anomaly.timestamp}")
```

#### Batch Detection

```python
# Process large datasets in batches
async for batch_result in detector.predict_batch(
    large_dataset,
    batch_size=1000
):
    print(f"Batch anomalies: {len(batch_result.anomalies)}")

    # Process anomalies immediately
    for anomaly in batch_result.anomalies:
        await send_alert(anomaly)
```

#### Real-time Detection

```python
# Single prediction
data_point = {"amount": 10000, "merchant_category": 5}
is_anomaly = await detector.predict_single(data_point)

if is_anomaly:
    print("Anomaly detected!")

# Streaming detection
async def handle_stream():
    async for data_point in data_stream:
        result = await detector.predict_single(data_point)
        if result.is_anomaly:
            await send_alert(result)
```

#### Ensemble Detection

```python
from pynomaly.application.services import EnsembleService

# Create ensemble of detectors
ensemble = EnsembleService([
    detector1,  # IsolationForest
    detector2,  # LOF
    detector3   # OCSVM
])

# Train ensemble
await ensemble.fit(dataset)

# Ensemble prediction with voting
result = await ensemble.predict(
    dataset,
    voting_strategy="soft",  # or "hard", "weighted"
    decision_threshold=0.6
)

# Get individual detector results
individual_results = await ensemble.predict_individual(dataset)
for detector_name, result in individual_results.items():
    print(f"{detector_name}: {len(result.anomalies)} anomalies")
```

### Model Persistence

#### Saving Models

```python
# Save detector
model_path = await detector.save("fraud_detector_v1.pkl")
print(f"Model saved to: {model_path}")

# Save with metadata
await detector.save(
    "fraud_detector_v1.pkl",
    metadata={
        "version": "1.0",
        "training_date": "2024-01-15",
        "dataset": "transactions_2024"
    }
)

# Export in different formats
await detector.export("model.joblib", format="joblib")
await detector.export("model.onnx", format="onnx")  # If supported
```

#### Loading Models

```python
# Load detector
detector = await pynomaly.detectors.load("fraud_detector_v1.pkl")

# Load with verification
detector = await pynomaly.detectors.load(
    "fraud_detector_v1.pkl",
    verify_checksum=True
)

# Get model metadata
metadata = await detector.get_metadata()
print(f"Model version: {metadata['version']}")
```

### Data Preprocessing

```python
from pynomaly.infrastructure.preprocessing import (
    DataCleaner, DataTransformer, PreprocessingPipeline
)

# Data cleaning
cleaner = DataCleaner()

# Clean dataset
clean_dataset = await cleaner.clean(
    dataset,
    handle_missing="interpolate",
    remove_duplicates=True,
    outlier_method="iqr",
    outlier_action="clip"
)

# Data transformation
transformer = DataTransformer()

# Transform features
transformed_dataset = await transformer.transform(
    dataset,
    scaling_method="standard",
    encoding_method="onehot",
    feature_selection="variance_threshold"
)

# Preprocessing pipeline
pipeline = PreprocessingPipeline([
    ("cleaner", DataCleaner()),
    ("transformer", DataTransformer())
])

# Fit and transform
processed_dataset = await pipeline.fit_transform(dataset)

# Save pipeline for reuse
await pipeline.save("preprocessing_pipeline.pkl")
```

### Experiment Management

```python
from pynomaly.application.services import ExperimentTrackingService

# Create experiment
experiment = await pynomaly.experiments.create(
    name="algorithm_comparison",
    description="Compare different algorithms on fraud data",
    dataset=dataset
)

# Add algorithms to experiment
await experiment.add_algorithm("IsolationForest", {"n_estimators": 100})
await experiment.add_algorithm("LOF", {"n_neighbors": 20})
await experiment.add_algorithm("OCSVM", {"gamma": "scale"})

# Run experiment
results = await experiment.run()

# Get best performing algorithm
best_algo = experiment.get_best_algorithm(metric="f1_score")
print(f"Best algorithm: {best_algo.name}")

# Export experiment results
await experiment.export_results("experiment_results.json")
```

### Performance Monitoring

```python
from pynomaly.infrastructure.monitoring import PerformanceService

# Initialize performance monitoring
perf_service = PerformanceService()

# Monitor detector performance
await perf_service.start_monitoring(detector)

# Get performance metrics
metrics = await perf_service.get_metrics(detector.id)
print(f"Average prediction time: {metrics['avg_prediction_time']}")
print(f"Memory usage: {metrics['memory_usage_mb']} MB")

# Performance alerts
await perf_service.set_alert_threshold(
    metric="prediction_time",
    threshold=1.0,  # seconds
    callback=lambda: print("Slow prediction detected!")
)
```

### Error Handling

```python
from pynomaly.domain.exceptions import (
    DetectorNotFittedError,
    InvalidAlgorithmError,
    FittingError,
    InvalidValueError
)

try:
    # Attempt detection without training
    result = await detector.predict(dataset)
except DetectorNotFittedError:
    print("Detector must be trained first")
    await detector.fit(dataset)
    result = await detector.predict(dataset)

try:
    # Invalid algorithm
    detector = await pynomaly.detectors.create(
        name="invalid",
        algorithm="NonExistentAlgorithm"
    )
except InvalidAlgorithmError as e:
    print(f"Algorithm error: {e}")
    print(f"Available algorithms: {e.available_algorithms}")

try:
    # Invalid contamination rate
    detector = await pynomaly.detectors.create(
        name="invalid_params",
        algorithm="IsolationForest",
        contamination_rate=1.5  # Invalid: > 0.5
    )
except InvalidValueError as e:
    print(f"Invalid parameter: {e}")
```

### Advanced Usage

#### Custom Algorithms

```python
from pynomaly.shared.protocols import DetectorProtocol
from pynomaly.domain.entities import Detector

class CustomDetector(Detector):
    """Custom anomaly detector implementation."""

    def __init__(self, **kwargs):
        super().__init__(
            name="CustomDetector",
            algorithm_name="Custom",
            **kwargs
        )
        self.threshold = 0.5

    async def fit(self, dataset):
        """Train the custom detector."""
        # Implementation here
        self._is_fitted = True

    async def predict(self, dataset):
        """Predict anomalies."""
        if not self._is_fitted:
            raise DetectorNotFittedError("Must fit before predict")

        # Custom prediction logic
        anomalies = []
        # ... detection implementation

        return DetectionResult(
            detector_id=self.id,
            dataset_id=dataset.id,
            anomalies=anomalies,
            # ... other fields
        )

# Register custom detector
pynomaly.detectors.register_custom(CustomDetector)

# Use custom detector
detector = await pynomaly.detectors.create(
    name="my_custom",
    algorithm="Custom"
)
```

#### Streaming Data Processing

```python
import asyncio
from pynomaly.infrastructure.streaming import StreamProcessor

async def process_kafka_stream():
    # Initialize stream processor
    processor = StreamProcessor(
        input_topic="raw_data",
        output_topic="anomalies",
        detector=detector
    )

    # Define processing logic
    async def process_message(message):
        data = json.loads(message.value)
        result = await detector.predict_single(data)

        if result.is_anomaly:
            await send_to_kafka("anomalies", result.to_json())

    # Start processing
    await processor.start(process_message)

# Run stream processing
asyncio.create_task(process_kafka_stream())
```

#### Multi-GPU Training

```python
# TensorFlow multi-GPU training
detector = await pynomaly.detectors.create(
    name="multi_gpu_ae",
    algorithm="AutoEncoder",
    adapter="tensorflow",
    strategy="MirroredStrategy",  # Multi-GPU strategy
    batch_size=512,
    epochs=100
)

# JAX multi-device training
detector = await pynomaly.detectors.create(
    name="jax_parallel",
    algorithm="VAE",
    adapter="jax",
    devices=["gpu:0", "gpu:1"],  # Specify devices
    parallel_training=True
)
```

#### Model Explainability

```python
from pynomaly.infrastructure.explainability import SHAPExplainer

# Create explainer
explainer = SHAPExplainer(detector)

# Explain predictions
explanations = await explainer.explain(dataset)

for i, explanation in enumerate(explanations):
    print(f"Sample {i}:")
    for feature, importance in explanation.feature_importance.items():
        print(f"  {feature}: {importance:.3f}")
```

## Configuration

### Settings Configuration

```python
from pynomaly.infrastructure.config import Settings

settings = Settings(
    # Storage paths
    storage_path="./data",
    model_storage_path="./models",
    experiment_storage_path="./experiments",

    # Database configuration
    database_url="postgresql://user:pass@localhost/pynomaly",

    # API configuration
    api_host="localhost",
    api_port=8000,
    api_cors_enabled=True,

    # Caching
    redis_url="redis://localhost:6379",
    cache_ttl=300,

    # Logging
    log_level="INFO",
    log_file="./logs/pynomaly.log",

    # Performance
    connection_pool_size=10,
    query_cache_size=1000,

    # Default algorithm parameters
    default_contamination_rate=0.1,
    default_random_state=42
)

pynomaly = Pynomaly(settings=settings)
```

### Environment Variables

```python
import os

# Set environment variables
os.environ["PYNOMALY_DATABASE_URL"] = "postgresql://..."
os.environ["PYNOMALY_REDIS_URL"] = "redis://..."
os.environ["PYNOMALY_LOG_LEVEL"] = "DEBUG"

# Settings will automatically load from environment
pynomaly = Pynomaly()
```

## Best Practices

### Memory Management

```python
# Use context managers for large datasets
async with pynomaly.datasets.load_large("huge_dataset.parquet") as dataset:
    # Process in chunks
    async for chunk in dataset.iter_chunks(chunk_size=10000):
        result = await detector.predict_batch(chunk)
        await process_results(result)

# Explicit cleanup
await dataset.cleanup()
await detector.cleanup()
```

### Async Patterns

```python
# Concurrent training of multiple detectors
detectors = [
    pynomaly.detectors.create(f"detector_{i}", "IsolationForest")
    for i in range(5)
]

# Train all detectors concurrently
training_tasks = [detector.fit(dataset) for detector in detectors]
await asyncio.gather(*training_tasks)

# Concurrent predictions
prediction_tasks = [detector.predict(dataset) for detector in detectors]
results = await asyncio.gather(*prediction_tasks)
```

### Error Recovery

```python
from tenacity import retry, stop_after_attempt, wait_exponential

@retry(
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=1, min=4, max=10)
)
async def robust_prediction(detector, dataset):
    try:
        return await detector.predict(dataset)
    except Exception as e:
        print(f"Prediction failed: {e}")
        # Optional: Reset detector state
        await detector.reset()
        raise

# Use robust prediction
result = await robust_prediction(detector, dataset)
```

This comprehensive Python SDK reference provides complete documentation for programmatic access to all Pynomaly functionality, with examples covering basic usage through advanced production scenarios.

---

## 🔗 **Related Documentation**

### **Development**
- **[Contributing Guidelines](../contributing/CONTRIBUTING.md)** - How to contribute
- **[Development Setup](../contributing/README.md)** - Local development environment
- **[Architecture Overview](../architecture/overview.md)** - System design
- **[Implementation Guide](../contributing/IMPLEMENTATION_GUIDE.md)** - Coding standards

### **API Integration**
- **[REST API](../api-integration/rest-api.md)** - HTTP API reference
- **[Python SDK](../api-integration/python-sdk.md)** - Python client library
- **[CLI Reference](../api-integration/cli.md)** - Command-line interface
- **[Authentication](../api-integration/authentication.md)** - Security and auth

### **User Documentation**
- **[User Guides](../../user-guides/README.md)** - Feature usage guides
- **[Getting Started](../../getting-started/README.md)** - Installation and setup
- **[Examples](../../examples/README.md)** - Real-world use cases

### **Deployment**
- **[Production Deployment](../../deployment/README.md)** - Production deployment
- **[Security Setup](../../deployment/SECURITY.md)** - Security configuration
- **[Monitoring](../../user-guides/basic-usage/monitoring.md)** - System observability

---

## 🆘 **Getting Help**

- **[Development Troubleshooting](../contributing/troubleshooting/)** - Development issues
- **[GitHub Issues](https://github.com/your-org/pynomaly/issues)** - Report bugs
- **[Contributing Guidelines](../contributing/CONTRIBUTING.md)** - Contribution process
