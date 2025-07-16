# Quick Start Tutorial

**Get started with Pynomaly in 5 minutes**

## 1. Installation

```bash
pip install pynomaly
```

## 2. Your First Detection

### Python API

```python
from pynomaly import AnomalyDetector
import numpy as np

# Create sample data with anomalies
np.random.seed(42)
normal_data = np.random.randn(950, 3)
anomalies = np.array([[5, 5, 5], [-5, -5, -5]] * 25)  # 50 anomalies
data = np.vstack([normal_data, anomalies])

# Create and train detector
detector = AnomalyDetector(algorithm="IsolationForest")
detector.fit(data)

# Detect anomalies
results = detector.detect(data)

print(f"Found {results.anomaly_count} anomalies out of {len(data)} samples")
print(f"Anomaly indices: {results.anomaly_indices[:10]}...")  # First 10
```

### CLI Usage

```bash
# Create sample data file
echo "1,2,3
4,5,6
100,200,300
7,8,9" > sample_data.csv

# Run detection
pynomaly detect --data sample_data.csv --algorithm IsolationForest

# Output:
# Detected 1 anomaly in sample_data.csv
# Anomaly found at row 3: [100, 200, 300]
```

## 3. Web Interface

```bash
# Start web server
pynomaly web start

# Open browser to http://localhost:8000
# Upload your data and explore results visually
```

## 4. Key Concepts

### Algorithms

Pynomaly supports multiple anomaly detection algorithms:

```python
# Statistical methods
detector = AnomalyDetector(algorithm="IsolationForest")
detector = AnomalyDetector(algorithm="LocalOutlierFactor")

# Ensemble methods
detector = AnomalyDetector(algorithm="EnsembleDetector")

# Deep learning (requires additional dependencies)
detector = AnomalyDetector(algorithm="AutoEncoder")
```

### Detection Results

```python
results = detector.detect(data)

# Access results
print(f"Anomaly count: {results.anomaly_count}")
print(f"Anomaly scores: {results.scores}")
print(f"Anomaly indices: {results.anomaly_indices}")
print(f"Threshold used: {results.threshold}")

# Convert to pandas DataFrame
df = results.to_dataframe()
print(df.head())
```

### Configuration

```python
# Custom configuration
detector = AnomalyDetector(
    algorithm="IsolationForest",
    threshold=0.1,  # Anomaly threshold
    random_state=42,  # For reproducibility
    n_estimators=100,  # Algorithm-specific parameter
)

# Or load from config file
detector = AnomalyDetector.from_config("config.yaml")
```

## 5. Common Use Cases

### Time Series Anomaly Detection

```python
import pandas as pd
from pynomaly import TimeSeriesDetector

# Load time series data
data = pd.read_csv("timeseries.csv", parse_dates=["timestamp"])

# Create time series detector
detector = TimeSeriesDetector(
    algorithm="SARIMA",
    seasonality=24,  # Hourly data with daily seasonality
)

# Fit and detect
detector.fit(data["value"])
anomalies = detector.detect(data["value"])

print(f"Found {len(anomalies)} anomalous time points")
```

### Streaming Detection

```python
from pynomaly import StreamingDetector

# Create streaming detector
detector = StreamingDetector(
    algorithm="HalfSpaceTrees",
    window_size=1000,
)

# Process data stream
for batch in data_stream:
    results = detector.detect_batch(batch)
    if results.anomaly_count > 0:
        print(f"Anomalies detected: {results.anomaly_indices}")
```

### Multivariate Detection

```python
# Multi-dimensional data
data = np.random.randn(1000, 10)  # 1000 samples, 10 features

detector = AnomalyDetector(
    algorithm="OCSVM",  # One-Class SVM
    kernel="rbf",
    gamma="scale",
)

detector.fit(data)
results = detector.detect(data)
```

## 6. Visualization

```python
import matplotlib.pyplot as plt
from pynomaly.visualization import plot_anomalies

# Plot 2D data with anomalies highlighted
plot_anomalies(
    data=data[:, :2],  # First 2 dimensions
    results=results,
    title="Anomaly Detection Results"
)
plt.show()

# Interactive visualization
from pynomaly.visualization import interactive_plot
interactive_plot(data, results)
```

## 7. Model Persistence

```python
# Save trained model
detector.save("my_detector.pkl")

# Load model later
detector = AnomalyDetector.load("my_detector.pkl")

# Use for new predictions
new_results = detector.detect(new_data)
```

## 8. Integration Examples

### With Pandas

```python
import pandas as pd

df = pd.read_csv("data.csv")
detector = AnomalyDetector()
detector.fit(df.values)

# Add anomaly scores to DataFrame
df["anomaly_score"] = detector.predict_scores(df.values)
df["is_anomaly"] = detector.predict(df.values)

# Filter anomalies
anomalies_df = df[df["is_anomaly"]]
print(f"Found {len(anomalies_df)} anomalous records")
```

### With Scikit-learn Pipeline

```python
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from pynomaly.sklearn import PynomalyTransformer

# Create pipeline
pipeline = Pipeline([
    ("scaler", StandardScaler()),
    ("detector", PynomalyTransformer(algorithm="IsolationForest"))
])

# Fit and transform
pipeline.fit(data)
anomaly_labels = pipeline.transform(data)
```

## 9. Next Steps

Now that you've learned the basics:

1. **[Advanced Features](../user-guides/advanced-features.md)** - Explore ensemble methods, custom algorithms
2. **[CLI Reference](../user-guides/cli-reference.md)** - Complete command-line interface guide
3. **[API Documentation](../api-reference/core-api.md)** - Full API reference
4. **[Examples Gallery](../examples/basic-examples.md)** - Real-world use cases and examples

## 10. Performance Tips

```python
# For large datasets
detector = AnomalyDetector(
    algorithm="IsolationForest",
    n_jobs=-1,  # Use all CPU cores
    max_samples="auto",  # Automatic sample size
)

# For memory efficiency
detector = AnomalyDetector(
    algorithm="StreamingHalfSpaceTrees",
    memory_limit="1GB",
)

# For real-time detection
detector = AnomalyDetector(
    algorithm="OnlineKMeans",
    batch_size=100,
)
```

## Getting Help

- 📖 [Full Documentation](../README.md)
- 🎯 [Examples](../examples/)
- 🐛 [Report Issues](https://github.com/elgerytme/pynomaly/issues)
- 💬 [Community](https://github.com/elgerytme/pynomaly/discussions)

---

*Quick start guide last updated: 2025-01-14*