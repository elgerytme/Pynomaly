# Pynomaly Detection User Guide

## Table of Contents
1. [Getting Started](#getting-started)
2. [Core Concepts](#core-concepts)
3. [Phase 2 Services](#phase-2-services)
4. [Advanced Features](#advanced-features)
5. [Use Cases](#use-cases)
6. [Best Practices](#best-practices)
7. [Troubleshooting](#troubleshooting)

## Getting Started

### Installation
```bash
pip install pynomaly-detection
```

### Basic Usage
```python
from pynomaly_detection import AnomalyDetector
import numpy as np

# Create sample data
data = np.random.randn(1000, 5)

# Initialize detector
detector = AnomalyDetector()

# Detect anomalies
predictions = detector.detect(data, contamination=0.1)

print(f"Detected {np.sum(predictions)} anomalies")
```

### Check Phase 2 Availability
```python
from pynomaly_detection import check_phase2_availability

availability = check_phase2_availability()
print(f"Phase 2 available: {availability}")
```

## Core Concepts

### Anomaly Detection
Anomaly detection identifies unusual patterns in data that don't conform to expected behavior.

### Contamination
The expected proportion of anomalies in your dataset (default: 0.1 = 10%).

### Algorithms
- **IsolationForest**: Fast, effective for most use cases
- **LOF**: Good for local anomalies
- **OneClassSVM**: Robust to outliers
- **AutoML**: Automatically selects best algorithm

## Phase 2 Services

### CoreDetectionService
The main service for anomaly detection:

```python
from pynomaly_detection import CoreDetectionService

service = CoreDetectionService()
result = service.detect_anomalies(
    data,
    algorithm="iforest",
    contamination=0.1
)

print(f"Algorithm: {result.algorithm}")
print(f"Anomalies: {result.n_anomalies}")
print(f"Samples: {result.n_samples}")
```

### AutoMLService
Intelligent algorithm selection:

```python
from pynomaly_detection import AutoMLService

automl = AutoMLService()

# Automatic detection
result = automl.auto_detect(data)

# Get recommendation
recommendation = automl.recommend_algorithm(data)
print(f"Recommended algorithm: {recommendation['algorithm']}")
```

### EnsembleService
Combine multiple algorithms:

```python
from pynomaly_detection import EnsembleService

ensemble = EnsembleService()

# Smart ensemble
result = ensemble.smart_ensemble(data, n_algorithms=3)

# Custom ensemble
result = ensemble.ensemble_detect(
    data,
    algorithms=["iforest", "lof", "svm"],
    voting="majority"
)
```

## Advanced Features

### Model Persistence
Save and load trained models:

```python
from pynomaly_detection import ModelPersistence

persistence = ModelPersistence()

# Save model
model_id = persistence.save_model(
    model_data={"algorithm": "iforest"},
    training_data=data,
    algorithm="iforest",
    performance_metrics={"accuracy": 0.95}
)

# Load model
loaded_model = persistence.load_model(model_id)
```

### Explainability
Understand why predictions were made:

```python
from pynomaly_detection import AdvancedExplainability

explainer = AdvancedExplainability(
    feature_names=["feature_1", "feature_2", "feature_3"]
)

explanation = explainer.explain_prediction(
    sample=data[0],
    sample_index=0,
    detection_result=result,
    training_data=data
)

print(f"Explanation: {explanation.explanation_text}")
```

### Monitoring
Track performance in production:

```python
from pynomaly_detection import MonitoringAlertingSystem

monitoring = MonitoringAlertingSystem()

# Record detection results
monitoring.record_detection_result(result, processing_time=0.5, source="production")

# Get metrics
metrics = monitoring.get_current_metrics()
print(f"Total anomalies detected: {metrics.total_anomalies_detected}")
```

## Use Cases

### Fraud Detection
```python
from pynomaly_detection import AutoMLService

# Load transaction data
transactions = load_transaction_data()

# Detect fraudulent transactions
automl = AutoMLService()
result = automl.auto_detect(transactions)

# Get fraud predictions
fraud_indices = np.where(result.predictions == 1)[0]
print(f"Detected {len(fraud_indices)} potentially fraudulent transactions")
```

### Network Security
```python
from pynomaly_detection import StreamingDetector

# Setup real-time network monitoring
detector = StreamingDetector(
    algorithm="lof",
    window_size=1000,
    drift_detection=True
)

# Process network traffic
for packet_batch in network_stream:
    result = detector.process_batch(packet_batch)
    
    if result.anomalies_detected:
        print(f"⚠️  Potential security threat detected!")
```

### Manufacturing Quality Control
```python
from pynomaly_detection import TimeSeriesDetector

# Monitor sensor data
ts_detector = TimeSeriesDetector()

result = ts_detector.detect_anomalies(
    sensor_data,
    method="statistical",
    window_size=100
)

# Alert on quality issues
if result.n_anomalies > 5:
    print("🚨 Quality control alert: Multiple anomalies detected")
```

## Best Practices

### Data Preparation
1. **Clean your data**: Remove missing values and outliers
2. **Scale features**: Use StandardScaler or MinMaxScaler
3. **Feature selection**: Remove irrelevant features

```python
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import SelectKBest

# Scale features
scaler = StandardScaler()
scaled_data = scaler.fit_transform(data)

# Select best features
selector = SelectKBest(k=10)
selected_data = selector.fit_transform(scaled_data, y)
```

### Algorithm Selection
- **Start with AutoML** for automatic selection
- **Use IsolationForest** for general purpose
- **Use LOF** for local anomalies
- **Use ensemble methods** for robust detection

### Performance Optimization
```python
from pynomaly_detection import BatchProcessor

# For large datasets
processor = BatchProcessor()
results = processor.process_large_dataset(
    large_data,
    algorithm="iforest",
    batch_size=10000,
    n_workers=4
)
```

### Production Deployment
1. **Monitor performance** with MonitoringAlertingSystem
2. **Set up alerts** for anomaly rate changes
3. **Version your models** with ModelPersistence
4. **Use streaming** for real-time applications

## Troubleshooting

### Common Issues

#### Import Errors
```python
# Check what's available
from pynomaly_detection import check_phase2_availability
print(check_phase2_availability())
```

#### Performance Issues
```python
# Use batch processing
from pynomaly_detection import BatchProcessor
processor = BatchProcessor()

# Optimize memory usage
from pynomaly_detection import MemoryOptimizer
optimizer = MemoryOptimizer()
optimized_data = optimizer.optimize_array_dtype(data)
```

#### Poor Detection Results
1. **Check data quality**: Look for missing values, outliers
2. **Adjust contamination**: Try different contamination rates
3. **Try different algorithms**: Use AutoML or ensemble methods
4. **Feature engineering**: Create better features

### Debug Mode
```python
import logging
logging.basicConfig(level=logging.DEBUG)

# Enable debug mode
detector = AnomalyDetector(config={"debug": True})
```

### Getting Help
- Check the [FAQ](FAQ.md)
- Post in [GitHub Discussions](https://github.com/pynomaly/pynomaly-detection/discussions)
- Open an [issue](https://github.com/pynomaly/pynomaly-detection/issues)

## Examples

### Complete Workflow
```python
from pynomaly_detection import (
    AutoMLService,
    ModelPersistence,
    AdvancedExplainability,
    MonitoringAlertingSystem
)

# 1. Load and prepare data
data = load_your_data()

# 2. Auto-detect anomalies
automl = AutoMLService()
result = automl.auto_detect(data)

# 3. Save the model
persistence = ModelPersistence()
model_id = persistence.save_model(
    model_data=automl.get_best_model(),
    training_data=data,
    algorithm=result.algorithm
)

# 4. Explain results
explainer = AdvancedExplainability()
explanation = explainer.explain_prediction(
    sample=data[0],
    sample_index=0,
    detection_result=result,
    training_data=data
)

# 5. Monitor in production
monitoring = MonitoringAlertingSystem()
monitoring.record_detection_result(result, 1.0, "production")
```

This user guide provides a comprehensive introduction to using Pynomaly Detection effectively. For more detailed information, see the [API Reference](API_REFERENCE.md) and [Advanced Tutorials](TUTORIALS.md).