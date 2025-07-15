# Basic Usage Guide

**Complete guide to using Pynomaly for anomaly detection tasks**

## Table of Contents

1. [Core Concepts](#core-concepts)
2. [Basic Detection Workflow](#basic-detection-workflow)
3. [Supported Algorithms](#supported-algorithms)
4. [Data Input Formats](#data-input-formats)
5. [Configuration Options](#configuration-options)
6. [Results Interpretation](#results-interpretation)
7. [Common Use Cases](#common-use-cases)

## Core Concepts

### Anomaly Detection
Anomaly detection identifies data points that deviate significantly from normal patterns. Pynomaly supports multiple approaches:

- **Statistical Methods**: Based on statistical distributions
- **Machine Learning**: Isolation Forest, One-Class SVM, etc.
- **Deep Learning**: Autoencoders, LSTM networks
- **Ensemble Methods**: Combining multiple algorithms

### Key Components

```python
from pynomaly import AnomalyDetector, Dataset, DetectionResult

# Core workflow components
detector = AnomalyDetector(algorithm="IsolationForest")
dataset = Dataset.from_array(data)
result = detector.fit_detect(dataset)
```

## Basic Detection Workflow

### 1. Import and Setup

```python
import numpy as np
import pandas as pd
from pynomaly import AnomalyDetector

# Generate sample data
np.random.seed(42)
normal_data = np.random.randn(1000, 3)
# Add some anomalies
anomalies = np.array([[5, 5, 5], [-5, -5, -5], [10, -10, 0]])
data = np.vstack([normal_data, anomalies])
```

### 2. Create and Configure Detector

```python
# Basic detector
detector = AnomalyDetector(
    algorithm="IsolationForest",
    threshold=0.1,  # Anomaly threshold (lower = more sensitive)
    random_state=42  # For reproducible results
)

# Alternative: Load from configuration
detector = AnomalyDetector.from_config({
    "algorithm": "IsolationForest",
    "threshold": 0.1,
    "n_estimators": 100,
    "contamination": 0.05
})
```

### 3. Train the Model

```python
# Fit the detector on your data
detector.fit(data)

# Check if model is fitted
print(f"Model fitted: {detector.is_fitted}")
```

### 4. Detect Anomalies

```python
# Detect anomalies in the same data
results = detector.detect(data)

# Or detect on new data
new_data = np.random.randn(100, 3)
new_results = detector.detect(new_data)

print(f"Found {results.anomaly_count} anomalies")
print(f"Anomaly indices: {results.anomaly_indices}")
```

### 5. Analyze Results

```python
# Get detailed results
print(f"Anomaly scores: {results.scores[:10]}...")  # First 10 scores
print(f"Threshold used: {results.threshold}")
print(f"Detection time: {results.detection_time}")

# Convert to DataFrame for easier analysis
df = results.to_dataframe()
print(df.head())
```

## Supported Algorithms

### Statistical Methods

```python
# Z-Score based detection
detector = AnomalyDetector(algorithm="ZScore", threshold=3.0)

# Modified Z-Score
detector = AnomalyDetector(algorithm="ModifiedZScore", threshold=3.5)

# Grubbs test
detector = AnomalyDetector(algorithm="Grubbs", alpha=0.05)
```

### Machine Learning Methods

```python
# Isolation Forest (recommended for general use)
detector = AnomalyDetector(
    algorithm="IsolationForest",
    n_estimators=100,
    contamination=0.1
)

# One-Class SVM
detector = AnomalyDetector(
    algorithm="OneClassSVM",
    kernel="rbf",
    gamma="scale"
)

# Local Outlier Factor
detector = AnomalyDetector(
    algorithm="LocalOutlierFactor",
    n_neighbors=20,
    contamination=0.1
)

# Elliptic Envelope
detector = AnomalyDetector(
    algorithm="EllipticEnvelope",
    contamination=0.1
)
```

### Deep Learning Methods

```python
# Autoencoder (requires additional dependencies)
detector = AnomalyDetector(
    algorithm="Autoencoder",
    hidden_layers=[64, 32, 16, 32, 64],
    epochs=100,
    threshold=0.95  # Reconstruction threshold
)

# LSTM for time series
detector = AnomalyDetector(
    algorithm="LSTM",
    sequence_length=50,
    hidden_size=128
)
```

### Ensemble Methods

```python
# Combine multiple algorithms
detector = AnomalyDetector(
    algorithm="EnsembleDetector",
    base_detectors=[
        "IsolationForest",
        "LocalOutlierFactor", 
        "OneClassSVM"
    ],
    voting="soft"  # or "hard"
)
```

## Data Input Formats

### NumPy Arrays

```python
import numpy as np

# 2D array: samples x features
data = np.random.randn(1000, 5)
detector.fit(data)
results = detector.detect(data)
```

### Pandas DataFrames

```python
import pandas as pd

# From DataFrame
df = pd.read_csv("data.csv")
detector.fit(df.values)
results = detector.detect(df.values)

# With column selection
features = ["feature1", "feature2", "feature3"]
detector.fit(df[features].values)
```

### CSV Files

```python
# Direct from CSV
detector = AnomalyDetector(algorithm="IsolationForest")
results = detector.fit_detect_csv("data.csv")

# With specific columns
results = detector.fit_detect_csv(
    "data.csv", 
    columns=["col1", "col2", "col3"]
)
```

### Time Series Data

```python
from pynomaly import TimeSeriesDetector

# Time series specific detector
ts_detector = TimeSeriesDetector(
    algorithm="SARIMA",
    seasonal_periods=24
)

# From pandas Series
ts_data = pd.read_csv("timeseries.csv", parse_dates=["timestamp"])
results = ts_detector.fit_detect(ts_data["value"])
```

## Configuration Options

### Algorithm-Specific Parameters

```python
# Isolation Forest parameters
detector = AnomalyDetector(
    algorithm="IsolationForest",
    n_estimators=100,      # Number of trees
    max_samples="auto",    # Samples per tree
    contamination=0.1,     # Expected anomaly ratio
    max_features=1.0,      # Features per tree
    bootstrap=False,       # Bootstrap sampling
    random_state=42
)

# SVM parameters
detector = AnomalyDetector(
    algorithm="OneClassSVM",
    kernel="rbf",          # Kernel type
    gamma="scale",         # Kernel coefficient
    nu=0.05,              # Upper bound on anomalies
    degree=3              # Polynomial degree
)
```

### Global Configuration

```python
# Configuration file (config.yaml)
app:
  name: "Pynomaly"
  version: "1.0.0"

detection:
  default_algorithm: "IsolationForest"
  default_threshold: 0.1
  cache_models: true

performance:
  n_jobs: -1  # Use all CPU cores
  batch_size: 1000

# Load configuration
detector = AnomalyDetector.from_config_file("config.yaml")
```

### Environment Variables

```bash
# Set environment variables
export PYNOMALY_ALGORITHM="IsolationForest"
export PYNOMALY_THRESHOLD=0.1
export PYNOMALY_N_JOBS=-1

# Use in Python
import os
detector = AnomalyDetector(
    algorithm=os.getenv("PYNOMALY_ALGORITHM", "IsolationForest"),
    threshold=float(os.getenv("PYNOMALY_THRESHOLD", 0.1))
)
```

## Results Interpretation

### Detection Results Structure

```python
# DetectionResult object
results = detector.detect(data)

# Basic properties
print(f"Total samples: {results.n_samples}")
print(f"Anomalies found: {results.anomaly_count}")
print(f"Anomaly rate: {results.anomaly_rate:.2%}")

# Detailed information
print(f"Anomaly indices: {results.anomaly_indices}")
print(f"Anomaly scores: {results.scores}")
print(f"Threshold: {results.threshold}")
print(f"Algorithm used: {results.algorithm}")
```

### Anomaly Scores

```python
# Understanding scores
scores = results.scores

# Higher scores = more anomalous (algorithm dependent)
print(f"Score range: {scores.min():.3f} to {scores.max():.3f}")
print(f"Mean score: {scores.mean():.3f}")
print(f"Std score: {scores.std():.3f}")

# Get top anomalies
top_anomalies = results.get_top_anomalies(n=10)
print(f"Top 10 anomaly indices: {top_anomalies}")
```

### Confidence and Uncertainty

```python
# Some algorithms provide confidence scores
if hasattr(results, 'confidence'):
    print(f"Detection confidence: {results.confidence:.2%}")

# Uncertainty quantification
if hasattr(results, 'uncertainty'):
    print(f"Prediction uncertainty: {results.uncertainty}")
```

## Common Use Cases

### 1. Financial Fraud Detection

```python
# Load transaction data
transactions = pd.read_csv("transactions.csv")
features = ["amount", "merchant_risk", "location_change", "time_since_last"]

# Configure for fraud detection
detector = AnomalyDetector(
    algorithm="IsolationForest",
    contamination=0.001,  # Expect 0.1% fraud
    random_state=42
)

# Detect fraudulent transactions
detector.fit(transactions[features])
results = detector.detect(transactions[features])

# Flag suspicious transactions
transactions["is_fraud"] = results.predictions
transactions["fraud_score"] = results.scores
suspicious = transactions[transactions["is_fraud"] == 1]
```

### 2. Network Intrusion Detection

```python
# Network traffic features
network_data = pd.read_csv("network_traffic.csv")
features = ["packet_size", "connection_duration", "bytes_sent", "bytes_received"]

# Real-time detection setup
detector = AnomalyDetector(
    algorithm="LocalOutlierFactor",
    n_neighbors=50,
    contamination=0.01
)

# Train on normal traffic
normal_traffic = network_data[network_data["label"] == "normal"]
detector.fit(normal_traffic[features])

# Detect intrusions in new traffic
new_traffic = pd.read_csv("new_traffic.csv")
results = detector.detect(new_traffic[features])
intrusions = new_traffic[results.predictions == 1]
```

### 3. Quality Control in Manufacturing

```python
# Sensor measurements from production line
sensor_data = pd.read_csv("sensor_readings.csv")
features = ["temperature", "pressure", "vibration", "speed"]

# Detect defective products
detector = AnomalyDetector(
    algorithm="EllipticEnvelope",
    contamination=0.05  # Expect 5% defects
)

detector.fit(sensor_data[features])
results = detector.detect(sensor_data[features])

# Products requiring inspection
defective_products = sensor_data[results.predictions == 1]
print(f"Found {len(defective_products)} potentially defective products")
```

### 4. Health Monitoring

```python
# Patient vital signs
vitals = pd.read_csv("patient_vitals.csv")
features = ["heart_rate", "blood_pressure_sys", "blood_pressure_dia", "temperature"]

# Detect abnormal health conditions
detector = AnomalyDetector(
    algorithm="OneClassSVM",
    gamma="scale",
    nu=0.1
)

detector.fit(vitals[features])
results = detector.detect(vitals[features])

# Alert for abnormal readings
abnormal_readings = vitals[results.predictions == 1]
for idx, reading in abnormal_readings.iterrows():
    print(f"Alert: Patient {reading['patient_id']} - Abnormal vitals detected")
```

## Performance Considerations

### Large Datasets

```python
# For large datasets, use sampling
from sklearn.utils import resample

# Sample for training if dataset is large
if len(data) > 100000:
    sample_data = resample(data, n_samples=50000, random_state=42)
    detector.fit(sample_data)
else:
    detector.fit(data)

# Use batch processing for detection
batch_size = 10000
results_list = []
for i in range(0, len(data), batch_size):
    batch = data[i:i+batch_size]
    batch_results = detector.detect(batch)
    results_list.append(batch_results)
```

### Memory Optimization

```python
# Use algorithms with lower memory footprint
detector = AnomalyDetector(
    algorithm="LocalOutlierFactor",
    memory_limit="2GB"  # Limit memory usage
)

# Or use streaming detection
from pynomaly import StreamingDetector
streaming_detector = StreamingDetector(
    algorithm="HalfSpaceTrees",
    window_size=1000
)
```

## Next Steps

- **[Advanced Features](./advanced-features.md)** - Explore advanced detection methods
- **[CLI Reference](./cli-reference.md)** - Command-line interface usage
- **[Web Interface](./web-interface.md)** - Using the web dashboard
- **[API Reference](../api-reference/core-api.md)** - Complete API documentation

## Getting Help

- 📖 [FAQ](../troubleshooting/faq.md)
- 🐛 [Report Issues](https://github.com/elgerytme/pynomaly/issues)
- 💬 [Community Discussions](https://github.com/elgerytme/pynomaly/discussions)

---

*Basic usage guide last updated: 2025-01-14*