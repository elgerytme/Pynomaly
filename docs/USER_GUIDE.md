# Pynomaly User Guide

Welcome to Pynomaly - a comprehensive Python package for anomaly detection that integrates PyOD, PyGOD, scikit-learn, PyTorch, TensorFlow, and JAX through clean architecture patterns.

## Table of Contents

1. [Quick Start](#quick-start)
2. [Installation](#installation)
3. [Basic Usage](#basic-usage)
4. [Advanced Features](#advanced-features)
5. [API Reference](#api-reference)
6. [Performance Optimization](#performance-optimization)
7. [Troubleshooting](#troubleshooting)

## Quick Start

Get started with Pynomaly in just a few lines of code:

```python
import numpy as np
from pynomaly import IsolationForestDetector

# Generate sample data
data = np.random.normal(0, 1, (1000, 5))
anomalies = np.random.normal(5, 1, (50, 5))
X = np.vstack([data, anomalies])

# Create and train detector
detector = IsolationForestDetector(contamination=0.05)
detector.fit(X)

# Detect anomalies
scores = detector.decision_function(X)
predictions = detector.predict(X)

print(f"Found {sum(predictions == -1)} anomalies")
```

## Installation

### Prerequisites

- Python 3.11+
- pip or poetry

### Install via pip

```bash
pip install pynomaly
```

### Install via poetry

```bash
poetry add pynomaly
```

### Development Installation

```bash
git clone https://github.com/your-org/pynomaly.git
cd pynomaly
poetry install
```

### Optional Dependencies

For enhanced performance and additional features:

```bash
# Deep learning support
pip install pynomaly[deep]

# Graph anomaly detection
pip install pynomaly[graph] 

# All features
pip install pynomaly[all]
```

## Basic Usage

### 1. Data Loading and Preprocessing

```python
import pandas as pd
from pynomaly.data import DataLoader, DataPreprocessor

# Load data from various sources
loader = DataLoader()

# From CSV
data = loader.from_csv("data.csv")

# From database
data = loader.from_database("sqlite:///data.db", "SELECT * FROM metrics")

# From streaming source
stream = loader.from_stream("kafka://localhost:9092/topic")

# Preprocessing
preprocessor = DataPreprocessor()
cleaned_data = preprocessor.clean(data)
normalized_data = preprocessor.normalize(cleaned_data)
```

### 2. Anomaly Detection Algorithms

#### Traditional Machine Learning

```python
from pynomaly.detectors import (
    IsolationForestDetector,
    LocalOutlierFactorDetector,
    OneClassSVMDetector,
    EllipticEnvelopeDetector
)

# Isolation Forest
iso_detector = IsolationForestDetector(
    n_estimators=100,
    contamination=0.1,
    random_state=42
)

# Local Outlier Factor
lof_detector = LocalOutlierFactorDetector(
    n_neighbors=20,
    contamination=0.1
)

# One-Class SVM
svm_detector = OneClassSVMDetector(
    kernel='rbf',
    gamma='scale',
    nu=0.1
)
```

#### Deep Learning

```python
from pynomaly.detectors.deep import (
    AutoEncoderDetector,
    VAEDetector,
    GANDetector
)

# Autoencoder
ae_detector = AutoEncoderDetector(
    hidden_layers=[64, 32, 16, 32, 64],
    epochs=100,
    batch_size=32
)

# Variational Autoencoder
vae_detector = VAEDetector(
    latent_dim=10,
    encoder_layers=[64, 32],
    decoder_layers=[32, 64]
)
```

#### Time Series

```python
from pynomaly.detectors.timeseries import (
    LSTMDetector,
    IForestTimeSeriesDetector,
    STLDecompositionDetector
)

# LSTM-based detector
lstm_detector = LSTMDetector(
    sequence_length=50,
    lstm_units=64,
    dropout=0.2
)

# Time series specific Isolation Forest
ts_detector = IForestTimeSeriesDetector(
    window_size=24,
    contamination=0.05
)
```

### 3. Model Training and Evaluation

```python
from pynomaly.evaluation import AnomalyEvaluator
from sklearn.model_selection import train_test_split

# Split data
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Train detector
detector = IsolationForestDetector()
detector.fit(X_train)

# Make predictions
y_pred = detector.predict(X_test)
scores = detector.decision_function(X_test)

# Evaluate performance
evaluator = AnomalyEvaluator()
metrics = evaluator.evaluate(y_test, y_pred, scores)

print(f"ROC AUC: {metrics['roc_auc']:.3f}")
print(f"Precision: {metrics['precision']:.3f}")
print(f"Recall: {metrics['recall']:.3f}")
print(f"F1 Score: {metrics['f1']:.3f}")
```

### 4. Ensemble Methods

```python
from pynomaly.ensemble import (
    VotingEnsemble,
    StackingEnsemble,
    BaggingEnsemble
)

# Create ensemble with multiple detectors
detectors = [
    IsolationForestDetector(random_state=42),
    LocalOutlierFactorDetector(),
    OneClassSVMDetector()
]

# Voting ensemble
voting_ensemble = VotingEnsemble(
    detectors=detectors,
    voting='soft'  # or 'hard'
)

# Stacking ensemble
stacking_ensemble = StackingEnsemble(
    detectors=detectors,
    meta_classifier='logistic_regression'
)

# Train and predict
voting_ensemble.fit(X_train)
predictions = voting_ensemble.predict(X_test)
```

## Advanced Features

### 1. Streaming Anomaly Detection

```python
from pynomaly.streaming import StreamingDetector
import asyncio

# Create streaming detector
streaming_detector = StreamingDetector(
    base_detector=IsolationForestDetector(),
    window_size=1000,
    update_interval=100
)

# Async streaming processing
async def process_stream():
    async for batch in data_stream:
        anomalies = await streaming_detector.detect_async(batch)
        if anomalies:
            print(f"Found {len(anomalies)} anomalies")
            # Handle anomalies
            await handle_anomalies(anomalies)

# Run streaming detection
asyncio.run(process_stream())
```

### 2. Explainable AI

```python
from pynomaly.explainability import AnomalyExplainer

# Create explainer
explainer = AnomalyExplainer(detector)

# Get feature importance for anomalies
explanations = explainer.explain_instances(
    X_test[predictions == -1]  # Only anomalous instances
)

# Visualize explanations
for i, explanation in enumerate(explanations):
    print(f"Anomaly {i} explanation:")
    explainer.visualize_explanation(explanation)
```

### 3. AutoML for Anomaly Detection

```python
from pynomaly.automl import AutoAnomalyDetector

# Automatic algorithm selection and hyperparameter tuning
auto_detector = AutoAnomalyDetector(
    time_budget=300,  # 5 minutes
    metric='roc_auc',
    cross_validation=5
)

# Fit and get best model
auto_detector.fit(X_train, y_train)
best_model = auto_detector.best_estimator_

print(f"Best algorithm: {auto_detector.best_algorithm_}")
print(f"Best score: {auto_detector.best_score_:.3f}")
print(f"Best parameters: {auto_detector.best_params_}")
```

### 4. Distributed Processing

```python
from pynomaly.distributed import DistributedDetector
from dask.distributed import Client

# Setup Dask cluster
client = Client('localhost:8786')

# Create distributed detector
distributed_detector = DistributedDetector(
    base_detector=IsolationForestDetector(),
    client=client,
    n_partitions=8
)

# Process large dataset
large_dataset = load_large_dataset()  # Dask DataFrame
results = distributed_detector.fit_predict(large_dataset)
```

### 5. Model Deployment

```python
from pynomaly.deployment import ModelServer
from fastapi import FastAPI

# Create FastAPI app with anomaly detection
app = FastAPI()
server = ModelServer(detector, app)

# Add endpoints
server.add_detection_endpoint("/detect")
server.add_batch_endpoint("/detect_batch")
server.add_health_endpoint("/health")

# Run server
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
```

### 6. Data Drift Detection

```python
from pynomaly.drift import DriftDetector

# Setup drift detection
drift_detector = DriftDetector(
    reference_data=X_train,
    detector_type='ks_test',  # or 'chi2', 'psi'
    threshold=0.05
)

# Monitor new data for drift
new_data = get_new_data()
drift_result = drift_detector.detect_drift(new_data)

if drift_result.drift_detected:
    print(f"Drift detected! p-value: {drift_result.p_value}")
    print(f"Drifted features: {drift_result.drifted_features}")
    
    # Retrain model
    detector.fit(new_data)
```

## API Reference

### Core Classes

#### `BaseDetector`
Abstract base class for all anomaly detectors.

**Methods:**
- `fit(X, y=None)`: Train the detector
- `predict(X)`: Predict anomalies (-1 for anomaly, 1 for normal)
- `decision_function(X)`: Get anomaly scores
- `score_samples(X)`: Get per-sample anomaly scores

#### `IsolationForestDetector`
Isolation Forest implementation with optimizations.

**Parameters:**
- `n_estimators` (int): Number of trees
- `contamination` (float): Expected proportion of anomalies
- `max_samples` (int or float): Samples per tree
- `random_state` (int): Random seed

#### `EnsembleDetector`
Base class for ensemble methods.

**Parameters:**
- `detectors` (list): List of base detectors
- `voting` (str): 'hard' or 'soft' voting
- `weights` (list): Detector weights for voting

### Utilities

#### `DataPreprocessor`
Data preprocessing utilities.

**Methods:**
- `clean(data)`: Remove outliers and handle missing values
- `normalize(data)`: Normalize features
- `encode_categorical(data)`: Encode categorical variables
- `select_features(data, target)`: Feature selection

#### `AnomalyEvaluator`
Evaluation metrics for anomaly detection.

**Methods:**
- `evaluate(y_true, y_pred, scores)`: Comprehensive evaluation
- `roc_auc_score(y_true, scores)`: ROC AUC score
- `precision_recall_curve(y_true, scores)`: PR curve
- `plot_roc_curve(y_true, scores)`: ROC curve visualization

## Performance Optimization

Pynomaly includes built-in performance optimizations:

### 1. Automatic Memory Optimization

```python
from pynomaly.optimization import get_optimization_service

# Get optimization service
optimizer = get_optimization_service()

# Optimize data preprocessing
optimized_data = optimizer.optimize_data_preprocessing(X)

# Optimize detector training
optimized_detector, predictions = optimizer.optimize_anomaly_detection(
    detector, X_train, X_test
)

# Get performance metrics
metrics = optimizer.get_performance_metrics()
print(f"Memory saved: {metrics['memory_saved_mb']:.1f}MB")
print(f"Time saved: {metrics['time_saved_ms']:.1f}ms")
```

### 2. Caching

```python
# Enable caching for expensive operations
from pynomaly.cache import enable_caching

enable_caching(max_size=1000, ttl=3600)  # 1000 items, 1 hour TTL

# Subsequent identical operations will be cached
detector.fit(X)  # Computed
detector.fit(X)  # Cached (much faster)
```

### 3. Parallel Processing

```python
# Enable parallel processing
detector = IsolationForestDetector(
    n_jobs=-1,  # Use all CPU cores
    n_estimators=100
)

# Batch processing for large datasets
from pynomaly.utils import batch_predict

predictions = batch_predict(
    detector, 
    large_dataset, 
    batch_size=10000
)
```

## Troubleshooting

### Common Issues

#### 1. Memory Errors with Large Datasets

**Problem:** `MemoryError` when processing large datasets.

**Solutions:**
```python
# Use batch processing
from pynomaly.utils import batch_fit, batch_predict

# Train in batches
batch_fit(detector, large_X, batch_size=10000)

# Predict in batches
predictions = batch_predict(detector, large_X, batch_size=10000)

# Use streaming processing
from pynomaly.streaming import StreamingDetector
streaming_detector = StreamingDetector(detector)
```

#### 2. Slow Training/Prediction

**Problem:** Training or prediction is too slow.

**Solutions:**
```python
# Enable optimization
from pynomaly.optimization import get_optimization_service
optimizer = get_optimization_service()
optimized_detector, _ = optimizer.optimize_anomaly_detection(detector, X)

# Use parallel processing
detector = IsolationForestDetector(n_jobs=-1)

# Reduce data size
from pynomaly.sampling import sample_data
sampled_X = sample_data(X, method='stratified', size=10000)
```

#### 3. Poor Detection Performance

**Problem:** Low precision/recall or ROC AUC.

**Solutions:**
```python
# Try ensemble methods
from pynomaly.ensemble import VotingEnsemble
ensemble = VotingEnsemble([
    IsolationForestDetector(),
    LocalOutlierFactorDetector(),
    OneClassSVMDetector()
])

# Hyperparameter tuning
from pynomaly.automl import AutoAnomalyDetector
auto_detector = AutoAnomalyDetector()
auto_detector.fit(X_train, y_train)

# Feature engineering
from pynomaly.features import FeatureEngineer
engineer = FeatureEngineer()
enhanced_X = engineer.create_features(X)
```

#### 4. Data Format Issues

**Problem:** Incompatible data formats or types.

**Solutions:**
```python
# Automatic data conversion
from pynomaly.data import DataConverter
converter = DataConverter()

# Handle different formats
data = converter.to_numpy(pandas_df)  # DataFrame to numpy
data = converter.handle_categorical(data)  # Encode categorical
data = converter.handle_missing(data)  # Handle NaN values
```

### Getting Help

1. **Documentation**: Check the full API documentation at [docs.pynomaly.org](https://docs.pynomaly.org)
2. **Examples**: Browse example notebooks in `/examples/`
3. **Issues**: Report bugs at [github.com/your-org/pynomaly/issues](https://github.com/your-org/pynomaly/issues)
4. **Discussions**: Join community discussions at [github.com/your-org/pynomaly/discussions](https://github.com/your-org/pynomaly/discussions)

### Performance Benchmarks

Expected performance on standard datasets:

| Algorithm | Dataset Size | Training Time | Prediction Time | Memory Usage |
|-----------|--------------|---------------|-----------------|--------------|
| IsolationForest | 100K samples | 2-5 seconds | 0.5-1 second | 200-500 MB |
| LOF | 10K samples | 5-10 seconds | 2-5 seconds | 100-300 MB |
| OneClassSVM | 50K samples | 10-30 seconds | 1-3 seconds | 500-1000 MB |
| Autoencoder | 100K samples | 2-5 minutes | 1-2 seconds | 1-2 GB |

*Performance varies based on data dimensionality, hardware, and configuration.*

## Next Steps

- Explore the [Tutorial Notebooks](/examples/notebooks/)
- Read the [Advanced Configuration Guide](/docs/ADVANCED_CONFIG.md)
- Check out [Industry Use Cases](/examples/industry_use_cases/)
- Join the [Community Forum](https://community.pynomaly.org)

Happy anomaly hunting! 🔍
