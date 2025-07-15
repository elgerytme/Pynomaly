# Your First Anomaly Detection

This guide walks you through creating your first anomaly detection workflow with Pynomaly.

## Prerequisites

Before you begin, make sure you have:
- [Installed Pynomaly](installation.md)
- Python 3.9+ environment
- Basic familiarity with Python and data analysis

## Quick Example

Let's detect anomalies in a simple dataset:

```python
import pandas as pd
import numpy as np
from pynomaly import create_detector, load_dataset, detect_anomalies

# 1. Create sample data
data = pd.DataFrame({
    'feature1': np.random.normal(0, 1, 100),
    'feature2': np.random.normal(0, 1, 100)
})

# Add some obvious outliers
data.loc[95:99, :] = np.random.normal(5, 1, (5, 2))

# 2. Load the dataset
dataset = load_dataset('my_first_dataset', data)
print(f"Dataset loaded: {dataset.name} with {len(dataset.data)} samples")

# 3. Create an anomaly detector
detector = create_detector('IsolationForest', contamination_rate=0.1)
print(f"Created detector: {detector.name}")

# 4. Run anomaly detection
result = detect_anomalies(dataset, detector)

# 5. Examine results
print(f"Detected {result.n_anomalies} anomalies")
print(f"Anomaly rate: {result.anomaly_rate:.1%}")

# Get the actual anomalous samples
anomalous_samples = result.get_anomalous_samples()
print(f"Anomalous sample indices: {anomalous_samples}")
```

## Understanding the Results

The `detect_anomalies` function returns a `DetectionResult` object with:

- **n_anomalies**: Number of anomalies detected
- **anomaly_rate**: Percentage of samples classified as anomalies
- **scores**: Anomaly scores for each sample
- **predictions**: Binary predictions (1 = anomaly, 0 = normal)

### Analyzing Anomaly Scores

```python
# Get anomaly scores
scores = result.scores
predictions = result.predictions

# Examine the most anomalous samples
import matplotlib.pyplot as plt

plt.figure(figsize=(12, 4))

# Plot 1: Data with anomalies highlighted
plt.subplot(1, 3, 1)
plt.scatter(data['feature1'], data['feature2'], c=predictions, cmap='coolwarm', alpha=0.7)
plt.title('Data with Anomalies Highlighted')
plt.xlabel('Feature 1')
plt.ylabel('Feature 2')

# Plot 2: Anomaly scores distribution
plt.subplot(1, 3, 2)
plt.hist(scores, bins=20, alpha=0.7)
plt.title('Anomaly Scores Distribution')
plt.xlabel('Anomaly Score')
plt.ylabel('Frequency')

# Plot 3: Scores vs predictions
plt.subplot(1, 3, 3)
plt.scatter(range(len(scores)), scores, c=predictions, cmap='coolwarm', alpha=0.7)
plt.title('Anomaly Scores by Sample')
plt.xlabel('Sample Index')
plt.ylabel('Anomaly Score')

plt.tight_layout()
plt.show()
```

## Different Detector Types

Pynomaly supports many anomaly detection algorithms:

```python
# Isolation Forest (good for tabular data)
detector_if = create_detector('IsolationForest', contamination_rate=0.05)

# Local Outlier Factor (density-based)
detector_lof = create_detector('LocalOutlierFactor', n_neighbors=20)

# One-Class SVM (boundary-based)
detector_svm = create_detector('OneClassSVM', nu=0.05)

# Elliptic Envelope (assumes gaussian distribution)
detector_ee = create_detector('EllipticEnvelope', contamination=0.05)
```

### Comparing Multiple Detectors

```python
detectors = {
    'IsolationForest': create_detector('IsolationForest', contamination_rate=0.05),
    'LocalOutlierFactor': create_detector('LocalOutlierFactor', n_neighbors=20),
    'OneClassSVM': create_detector('OneClassSVM', nu=0.05)
}

results = {}
for name, detector in detectors.items():
    result = detect_anomalies(dataset, detector)
    results[name] = result
    print(f"{name}: {result.n_anomalies} anomalies ({result.anomaly_rate:.1%})")
```

## Working with Real Data

### Loading from CSV
```python
# Load data from CSV file
import pandas as pd
data = pd.read_csv('your_data.csv')

# Basic data preprocessing
data = data.dropna()  # Remove missing values
data = data.select_dtypes(include=[np.number])  # Keep only numeric columns

dataset = load_dataset('csv_dataset', data)
```

### Handling Different Data Types

```python
# For time series data
from pynomaly.preprocessing import prepare_time_series
time_series_data = prepare_time_series(data, timestamp_col='timestamp')

# For categorical data
from pynomaly.preprocessing import encode_categorical
encoded_data = encode_categorical(data, categorical_columns=['category1', 'category2'])
```

## Tuning Detection Parameters

### Contamination Rate
The contamination rate determines what percentage of data is expected to be anomalous:

```python
# Conservative (expect fewer anomalies)
detector_conservative = create_detector('IsolationForest', contamination_rate=0.01)

# Liberal (expect more anomalies)
detector_liberal = create_detector('IsolationForest', contamination_rate=0.20)
```

### Algorithm-Specific Parameters

```python
# Isolation Forest parameters
detector_if = create_detector('IsolationForest', 
                            contamination_rate=0.05,
                            n_estimators=200,
                            max_samples='auto')

# LOF parameters
detector_lof = create_detector('LocalOutlierFactor',
                             n_neighbors=20,
                             leaf_size=30,
                             metric='minkowski')
```

## Next Steps

Now that you've run your first detection, explore these topics:

### Basic Usage
- [Data Preprocessing](../user-guides/data-preprocessing.md) - Prepare your data for detection
- [Algorithm Selection](../user-guides/algorithm-selection.md) - Choose the right detector
- [Result Analysis](../user-guides/result-analysis.md) - Interpret and visualize results

### Advanced Features
- [AutoML](../user-guides/automl.md) - Automatic algorithm selection and tuning
- [Explainability](../user-guides/explainability.md) - Understand why samples are anomalous
- [Production Deployment](../deployment/production-guide.md) - Deploy to production

### API & Integration
- [REST API](../api-reference/API_DOCUMENTATION.md) - Use the HTTP API
- [CLI Usage](../user-guides/cli-reference.md) - Command-line interface
- [Python SDK](../api-reference/python-sdk.md) - Full Python library reference

## Common Issues

### No Anomalies Detected
If your detector isn't finding anomalies:
1. Check your contamination rate - it might be too low
2. Try a different algorithm
3. Ensure your data is properly preprocessed
4. Verify your data actually contains anomalies

### Too Many Anomalies
If everything looks anomalous:
1. Lower the contamination rate
2. Check for data quality issues
3. Consider feature scaling/normalization
4. Try a different algorithm

### Poor Performance
For better detection performance:
1. Remove irrelevant features
2. Handle missing values appropriately
3. Scale/normalize features
4. Use domain knowledge for feature engineering

## Getting Help

- **Documentation**: Browse the [User Guides](../user-guides/) for detailed information
- **Examples**: Check out more [examples and tutorials](../examples/)
- **Community**: Join our [GitHub Discussions](https://github.com/elgerytme/Pynomaly/discussions)
- **Issues**: Report bugs on [GitHub Issues](https://github.com/elgerytme/Pynomaly/issues)

Congratulations! You've completed your first anomaly detection with Pynomaly. 🎉