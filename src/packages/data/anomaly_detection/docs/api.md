# Anomaly Detection API Reference

This document provides comprehensive API documentation for the Anomaly Detection package.

## Table of Contents

1. [Core Services](#core-services)
2. [Domain Entities](#domain-entities)
3. [Algorithm Adapters](#algorithm-adapters)
4. [REST API Endpoints](#rest-api-endpoints)
5. [Python SDK](#python-sdk)
6. [CLI Commands](#cli-commands)

## Core Services

### DetectionService

The main service for anomaly detection operations.

```python
from anomaly_detection.domain.services import DetectionService

service = DetectionService()
```

#### Methods

##### `detect_anomalies(data, algorithm, **kwargs)`

Detect anomalies in the provided data.

**Parameters:**
- `data` (np.ndarray | pd.DataFrame): Input data matrix (n_samples, n_features)
- `algorithm` (str): Algorithm to use ('iforest', 'lof', 'ocsvm', etc.)
- `contamination` (float, optional): Expected proportion of anomalies (default: 0.1)
- `**kwargs`: Algorithm-specific parameters

**Returns:**
- `DetectionResult`: Object containing predictions, scores, and anomalies

**Example:**
```python
import numpy as np

data = np.random.randn(1000, 5)
result = service.detect_anomalies(
    data=data,
    algorithm='iforest',
    contamination=0.05,
    n_estimators=100,
    random_state=42
)

print(f"Found {result.anomaly_count} anomalies")
print(f"Anomaly indices: {result.anomaly_indices}")
```

##### `fit(data, algorithm, **kwargs)`

Train a model on the provided data.

**Parameters:**
- `data` (np.ndarray | pd.DataFrame): Training data (typically normal data only)
- `algorithm` (str): Algorithm to use
- `**kwargs`: Algorithm-specific parameters

**Returns:**
- `None`

##### `predict(data, algorithm)`

Make predictions using a previously fitted model.

**Parameters:**
- `data` (np.ndarray | pd.DataFrame): Data to predict on
- `algorithm` (str): Algorithm name (must be previously fitted)

**Returns:**
- `DetectionResult`: Predictions for the input data

**Example:**
```python
# Train on normal data
normal_data = generate_normal_data()
service.fit(normal_data, algorithm='iforest')

# Predict on new data
new_data = get_new_data()
result = service.predict(new_data, algorithm='iforest')
```

##### `list_available_algorithms()`

Get list of available algorithms.

**Returns:**
- `List[str]`: Available algorithm names

##### `get_algorithm_info(algorithm)`

Get detailed information about an algorithm.

**Parameters:**
- `algorithm` (str): Algorithm name

**Returns:**
- `Dict[str, Any]`: Algorithm metadata including parameters, requirements, etc.

### EnsembleService

Service for combining multiple algorithms.

```python
from anomaly_detection.domain.services import EnsembleService

ensemble = EnsembleService()
```

#### Methods

##### `detect_with_ensemble(data, algorithms, method='majority', **kwargs)`

Detect anomalies using multiple algorithms.

**Parameters:**
- `data` (np.ndarray | pd.DataFrame): Input data
- `algorithms` (List[str]): List of algorithms to use
- `method` (str): Combination method ('majority', 'average', 'maximum', 'stacking')
- `weights` (List[float], optional): Algorithm weights for weighted voting
- `**kwargs`: Parameters passed to all algorithms

**Returns:**
- `DetectionResult`: Combined detection results

**Example:**
```python
result = ensemble.detect_with_ensemble(
    data=data,
    algorithms=['iforest', 'lof', 'ocsvm'],
    method='majority',
    contamination=0.1
)
```

##### `create_stacking_ensemble(data, base_algorithms, meta_algorithm='logistic')`

Create a stacking ensemble with meta-learning.

**Parameters:**
- `data` (np.ndarray): Training data with labels
- `base_algorithms` (List[str]): Base detector algorithms
- `meta_algorithm` (str): Meta-learner algorithm

**Returns:**
- `StackingEnsemble`: Trained stacking ensemble

### StreamingService

Service for real-time anomaly detection.

```python
from anomaly_detection.domain.services import StreamingService

streaming = StreamingService(window_size=100, update_frequency=10)
```

#### Methods

##### `process_sample(sample)`

Process a single data point.

**Parameters:**
- `sample` (np.ndarray): Single data point (1D array)

**Returns:**
- `StreamResult`: Detection result with drift indicators

##### `process_window(window)`

Process a window of samples.

**Parameters:**
- `window` (np.ndarray): Window of samples

**Returns:**
- `DetectionResult`: Results for the window

##### `detect_concept_drift()`

Check for concept drift in the stream.

**Returns:**
- `Dict[str, Any]`: Drift detection results

**Example:**
```python
streaming = StreamingService(window_size=500)

for sample in data_stream:
    result = streaming.process_sample(sample)
    if result.is_anomaly:
        print(f"Anomaly detected: {sample}")
    
    if streaming.samples_processed % 100 == 0:
        drift = streaming.detect_concept_drift()
        if drift['drift_detected']:
            print("Concept drift detected!")
```

## Domain Entities

### DetectionResult

Container for anomaly detection results.

**Attributes:**
- `predictions` (np.ndarray): Binary predictions (-1 for anomaly, 1 for normal)
- `anomaly_scores` (np.ndarray): Anomaly scores (higher = more anomalous)
- `anomaly_indices` (List[int]): Indices of detected anomalies
- `anomaly_count` (int): Number of anomalies detected
- `anomaly_rate` (float): Proportion of anomalies
- `algorithm` (str): Algorithm used
- `parameters` (Dict): Parameters used
- `metadata` (Dict): Additional metadata

**Methods:**
- `to_dataframe()`: Convert results to pandas DataFrame
- `to_dict()`: Convert to dictionary
- `save(filepath)`: Save results to file
- `plot()`: Visualize results (if matplotlib available)

### Anomaly

Represents a single anomaly.

**Attributes:**
- `index` (int): Sample index
- `score` (float): Anomaly score
- `confidence` (float): Detection confidence
- `features` (Dict[str, float]): Feature values
- `explanation` (Dict, optional): Explanation if available

### Dataset

Wrapper for input data with validation.

**Attributes:**
- `data` (np.ndarray): The data matrix
- `shape` (Tuple[int, int]): Data dimensions
- `feature_names` (List[str], optional): Feature names
- `metadata` (Dict): Dataset metadata

**Methods:**
- `validate()`: Validate data integrity
- `normalize()`: Normalize features
- `split(ratio)`: Split into train/test sets

## Algorithm Adapters

### SklearnAdapter

Adapter for scikit-learn algorithms.

**Supported Algorithms:**
- `iforest`: Isolation Forest
- `lof`: Local Outlier Factor
- `ocsvm`: One-Class SVM
- `elliptic`: Elliptic Envelope

**Example:**
```python
from anomaly_detection.infrastructure.adapters.algorithms import SklearnAdapter

adapter = SklearnAdapter('iforest', n_estimators=200)
adapter.fit(X_train)
predictions = adapter.predict(X_test)
```

### PyODAdapter

Adapter for PyOD algorithms.

**Supported Algorithms:**
- Over 40 algorithms including: ABOD, CBLOF, COF, COPOD, ECOD, HBOS, IForest, KNN, LODA, LOF, LOCI, LSCP, MAD, MCD, OCSVM, PCA, ROD, SOD, SOS, SUOD, VAE, and more

**Example:**
```python
from anomaly_detection.infrastructure.adapters.algorithms import PyODAdapter

adapter = PyODAdapter('knn', n_neighbors=5)
adapter.fit(X_train)
scores = adapter.decision_scores_
```

### DeepLearningAdapter

Adapter for neural network models.

**Supported Models:**
- `autoencoder`: Vanilla autoencoder
- `vae`: Variational autoencoder
- `deep_svdd`: Deep Support Vector Data Description

**Example:**
```python
from anomaly_detection.infrastructure.adapters.algorithms import DeepLearningAdapter

adapter = DeepLearningAdapter(
    'autoencoder',
    encoding_dim=32,
    epochs=50,
    batch_size=32
)
adapter.fit(X_train)
reconstruction_errors = adapter.predict(X_test)
```

## REST API Endpoints

### Detection Endpoints

#### `POST /api/v1/detect`

Detect anomalies in provided data.

**Request Body:**
```json
{
    "data": [[1.0, 2.0], [3.0, 4.0], ...],
    "algorithm": "isolation_forest",
    "contamination": 0.1,
    "parameters": {
        "n_estimators": 100,
        "random_state": 42
    }
}
```

**Response:**
```json
{
    "success": true,
    "anomalies": [
        {"index": 5, "score": 0.85, "confidence": 0.92},
        {"index": 12, "score": 0.79, "confidence": 0.88}
    ],
    "algorithm": "isolation_forest",
    "total_samples": 100,
    "anomalies_detected": 2,
    "anomaly_rate": 0.02,
    "processing_time_ms": 45,
    "timestamp": "2024-01-23T10:30:00Z"
}
```

#### `POST /api/v1/ensemble`

Detect using ensemble of algorithms.

**Request Body:**
```json
{
    "data": [[1.0, 2.0], [3.0, 4.0], ...],
    "algorithms": ["isolation_forest", "local_outlier_factor"],
    "method": "majority",
    "contamination": 0.1
}
```

### Model Management

#### `GET /api/v1/models`

List saved models.

**Response:**
```json
{
    "models": [
        {
            "id": "model_123",
            "name": "fraud_detector_v1",
            "algorithm": "isolation_forest",
            "created_at": "2024-01-20T10:00:00Z",
            "metrics": {
                "accuracy": 0.95,
                "precision": 0.89
            }
        }
    ],
    "total_count": 5
}
```

#### `POST /api/v1/models/{model_id}/predict`

Make predictions using saved model.

### Health & Monitoring

#### `GET /health`

Basic health check.

#### `GET /api/v1/health/detailed`

Detailed health status with component checks.

#### `GET /api/v1/metrics`

Get performance metrics and statistics.

## Python SDK

### Installation

```bash
pip install anomaly-detection
```

### Quick Start

```python
from anomaly_detection import AnomalyDetector

# Initialize detector
detector = AnomalyDetector(algorithm='iforest')

# Detect anomalies
anomalies = detector.detect(data)

# Get detailed results
results = detector.detect_detailed(data)
print(f"Anomaly rate: {results.anomaly_rate:.2%}")
print(f"Top anomalies: {results.get_top_anomalies(5)}")
```

### Advanced Usage

```python
from anomaly_detection import (
    AnomalyDetector, 
    EnsembleDetector,
    StreamingDetector
)

# Ensemble detection
ensemble = EnsembleDetector(
    algorithms=['iforest', 'lof', 'ocsvm'],
    voting='soft',
    weights=[0.4, 0.3, 0.3]
)
results = ensemble.detect(data)

# Streaming detection
stream_detector = StreamingDetector(
    algorithm='iforest',
    window_size=1000,
    update_interval=100
)

for batch in data_stream:
    anomalies = stream_detector.detect_batch(batch)
    if stream_detector.drift_detected:
        stream_detector.retrain()
```

## CLI Commands

### Basic Detection

```bash
# Detect anomalies in CSV file
anomaly-detection detect --input data.csv --algorithm iforest --output results.json

# With custom parameters
anomaly-detection detect \
    --input data.csv \
    --algorithm lof \
    --contamination 0.05 \
    --param n_neighbors=20 \
    --param metric=euclidean
```

### Ensemble Detection

```bash
# Use multiple algorithms
anomaly-detection ensemble \
    --input data.csv \
    --algorithms iforest lof ocsvm \
    --method majority \
    --output ensemble_results.json
```

### Streaming Detection

```bash
# Monitor real-time data stream
anomaly-detection stream \
    --source kafka://localhost:9092/sensor-data \
    --algorithm iforest \
    --window-size 1000 \
    --alert-webhook https://alerts.example.com/webhook
```

### Model Management

```bash
# Train and save model
anomaly-detection train \
    --input training_data.csv \
    --algorithm iforest \
    --save-model models/detector_v1.pkl

# Use saved model
anomaly-detection predict \
    --input new_data.csv \
    --model models/detector_v1.pkl \
    --output predictions.json
```

### Utilities

```bash
# List available algorithms
anomaly-detection algorithms list

# Get algorithm info
anomaly-detection algorithms info --name iforest

# Validate data
anomaly-detection validate --input data.csv

# Benchmark algorithms
anomaly-detection benchmark \
    --input labeled_data.csv \
    --algorithms all \
    --metrics precision recall f1
```

## Error Handling

All services and endpoints follow consistent error handling:

### Error Types

- `InputValidationError`: Invalid input data or parameters
- `AlgorithmError`: Algorithm-specific errors
- `ConfigurationError`: Configuration issues
- `ResourceError`: Resource availability issues

### Error Response Format

```json
{
    "error": {
        "type": "InputValidationError",
        "message": "Contamination must be between 0 and 0.5",
        "details": {
            "parameter": "contamination",
            "provided_value": 0.7,
            "valid_range": [0, 0.5]
        },
        "timestamp": "2024-01-23T10:30:00Z",
        "request_id": "req_123456"
    }
}
```

## Best Practices

1. **Data Preprocessing**: Always validate and preprocess data before detection
2. **Algorithm Selection**: Choose algorithms based on data characteristics
3. **Parameter Tuning**: Use cross-validation for parameter optimization
4. **Ensemble Methods**: Combine multiple algorithms for robustness
5. **Monitoring**: Track performance metrics in production
6. **Retraining**: Periodically retrain models to handle drift

## Rate Limits

API endpoints have the following rate limits:

- Detection endpoints: 100 requests/minute
- Model management: 50 requests/minute
- Health checks: 1000 requests/minute

## Versioning

The API follows semantic versioning. Current version: v1

Breaking changes will be introduced in new major versions (v2, v3, etc.)