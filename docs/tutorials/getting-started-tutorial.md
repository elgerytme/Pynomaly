# Getting Started Tutorial

## Welcome to Pynomaly

This tutorial will guide you through your first steps with Pynomaly, from installation to running your first anomaly detection analysis.

## Prerequisites

- Python 3.11 or higher
- Basic understanding of data science concepts
- Familiarity with pandas and numpy (helpful but not required)

## Step 1: Installation

### Quick Installation

For most users, the standard installation is recommended:

```bash
pip install pynomaly[standard]
```

### Complete Installation

For access to all features including advanced ML algorithms:

```bash
pip install pynomaly[all]
```

### Verify Installation

```python
import pynomaly
print(f"Pynomaly version: {pynomaly.__version__}")

# Test basic functionality
from pynomaly import AnomalyDetector
detector = AnomalyDetector()
print("Installation successful!")
```

## Step 2: Your First Anomaly Detection

### Load Sample Data

```python
import pandas as pd
import numpy as np
from pynomaly import AnomalyDetector

# Create sample data with anomalies
np.random.seed(42)
normal_data = np.random.normal(0, 1, (1000, 2))
anomalous_data = np.random.normal(5, 1, (50, 2))
data = np.vstack([normal_data, anomalous_data])

# Convert to DataFrame
df = pd.DataFrame(data, columns=['feature_1', 'feature_2'])
print(f"Dataset shape: {df.shape}")
```

### Basic Detection

```python
# Create and configure detector
detector = AnomalyDetector(
    algorithm='IsolationForest',
    contamination=0.05,  # Expected proportion of outliers
    random_state=42
)

# Fit and detect anomalies
detector.fit(df)
results = detector.detect(df)

# View results
print(f"Anomalies detected: {results.n_anomalies}")
print(f"Anomaly indices: {results.anomaly_indices[:10]}...")  # First 10
```

### Visualize Results

```python
import matplotlib.pyplot as plt

# Create visualization
plt.figure(figsize=(10, 6))

# Plot normal points
normal_mask = results.labels == 1
plt.scatter(df[normal_mask]['feature_1'], df[normal_mask]['feature_2'], 
           c='blue', alpha=0.6, label='Normal')

# Plot anomalies
anomaly_mask = results.labels == -1
plt.scatter(df[anomaly_mask]['feature_1'], df[anomaly_mask]['feature_2'], 
           c='red', alpha=0.8, label='Anomaly')

plt.xlabel('Feature 1')
plt.ylabel('Feature 2')
plt.title('Anomaly Detection Results')
plt.legend()
plt.grid(True, alpha=0.3)
plt.show()
```

## Step 3: Working with Real Data

### Load Real Dataset

```python
# Example with CSV data
try:
    # Load your own data
    real_data = pd.read_csv('your_data.csv')
    print(f"Real data shape: {real_data.shape}")
except FileNotFoundError:
    # Use sample credit card fraud dataset
    from sklearn.datasets import make_classification
    X, y = make_classification(n_samples=10000, n_features=20, 
                              n_redundant=0, n_clusters_per_class=1,
                              random_state=42)
    real_data = pd.DataFrame(X, columns=[f'feature_{i}' for i in range(20)])
    print("Using synthetic dataset for demo")
```

### Data Preprocessing

```python
from pynomaly.packages.data_transformation import DataTransformationEngine

# Create transformation engine
transformer = DataTransformationEngine()

# Basic preprocessing
preprocessing_config = {
    'handle_missing_values': True,
    'normalize_features': True,
    'remove_outliers': False  # We want to detect these!
}

# Transform data
transformed_data = transformer.preprocess(real_data, preprocessing_config)
print(f"Preprocessed data shape: {transformed_data.shape}")
```

### Advanced Detection

```python
# Create advanced detector
advanced_detector = AnomalyDetector(
    algorithm='AutoEnsemble',  # Automatically combines multiple algorithms
    algorithms=['IsolationForest', 'LocalOutlierFactor', 'OneClassSVM'],
    contamination='auto',  # Automatically estimate contamination
    ensemble_method='voting'
)

# Fit and detect with confidence scores
advanced_detector.fit(transformed_data)
advanced_results = advanced_detector.detect(transformed_data)

# Analyze results
print(f"Total anomalies detected: {advanced_results.n_anomalies}")
print(f"Average anomaly score: {advanced_results.scores.mean():.3f}")
print(f"Detection confidence: {advanced_results.confidence:.3f}")
```

## Step 4: Understanding Results

### Examining Anomaly Scores

```python
# Get detailed results
detailed_results = advanced_detector.get_detailed_results()

# Sort by anomaly score (most anomalous first)
sorted_anomalies = detailed_results.sort_values('anomaly_score', ascending=False)
print("Top 10 most anomalous records:")
print(sorted_anomalies.head(10)[['anomaly_score', 'confidence']])
```

### Feature Importance for Anomalies

```python
# Analyze which features contribute most to anomalies
feature_importance = advanced_detector.analyze_feature_importance()

print("Features most important for anomaly detection:")
for feature, importance in feature_importance.items():
    print(f"{feature}: {importance:.3f}")
```

## Step 5: Explaining Anomalies

```python
from pynomaly.packages.core.use_cases import ExplainAnomalyUseCase

# Create explainer
explainer = ExplainAnomalyUseCase()

# Explain top anomaly
top_anomaly_index = advanced_results.anomaly_indices[0]
explanation = explainer.explain_anomaly(
    model=advanced_detector,
    data=transformed_data,
    anomaly_index=top_anomaly_index
)

print(f"Explanation for anomaly at index {top_anomaly_index}:")
print(f"Anomaly score: {explanation.anomaly_score:.3f}")
print("Contributing features:")
for feature, contribution in explanation.feature_contributions.items():
    print(f"  {feature}: {contribution:.3f}")
```

## Step 6: Model Evaluation

### Using Labeled Data (if available)

```python
# If you have ground truth labels
has_labels = False  # Set to True if you have labels

if has_labels:
    # Assuming you have true_labels
    from pynomaly.evaluation import evaluate_detector
    
    evaluation_results = evaluate_detector(
        y_true=true_labels,
        y_pred=advanced_results.labels,
        y_scores=advanced_results.scores
    )
    
    print(f"Precision: {evaluation_results.precision:.3f}")
    print(f"Recall: {evaluation_results.recall:.3f}")
    print(f"F1-Score: {evaluation_results.f1_score:.3f}")
    print(f"AUC-ROC: {evaluation_results.auc_roc:.3f}")
```

### Unsupervised Evaluation

```python
# Evaluate without labels using internal metrics
internal_metrics = advanced_detector.evaluate_internal_metrics()

print("Internal evaluation metrics:")
print(f"Silhouette score: {internal_metrics.silhouette_score:.3f}")
print(f"Isolation score: {internal_metrics.isolation_score:.3f}")
print(f"Stability score: {internal_metrics.stability_score:.3f}")
```

## Step 7: Saving and Loading Models

### Save Trained Model

```python
# Save the detector for future use
advanced_detector.save_model('my_anomaly_detector.pkl')
print("Model saved successfully!")

# Save with metadata
model_metadata = {
    'training_date': pd.Timestamp.now(),
    'data_shape': transformed_data.shape,
    'contamination': 'auto',
    'algorithms_used': ['IsolationForest', 'LocalOutlierFactor', 'OneClassSVM']
}

advanced_detector.save_model('my_detector_with_metadata.pkl', metadata=model_metadata)
```

### Load and Use Saved Model

```python
# Load saved model
loaded_detector = AnomalyDetector.load_model('my_anomaly_detector.pkl')

# Use loaded model for new predictions
new_data = transformed_data.sample(100)  # Sample some data for testing
new_results = loaded_detector.detect(new_data)

print(f"New data anomalies: {new_results.n_anomalies}")
```

## Step 8: Setting Up Monitoring

### Real-time Anomaly Detection

```python
from pynomaly.monitoring import AnomalyMonitor

# Create monitoring system
monitor = AnomalyMonitor(
    detector=loaded_detector,
    alert_threshold=0.8,
    monitoring_interval='5min'
)

# Configure alerts
monitor.configure_alerts(
    email_alerts=True,
    email_recipients=['admin@company.com'],
    slack_webhook='your_slack_webhook_url'
)

# Start monitoring (in production, this would run continuously)
print("Anomaly monitoring configured!")
print("Use monitor.start_monitoring(data_stream) in production")
```

## Next Steps

### Explore Advanced Features

1. **Data Quality Integration**
   ```python
   from pynomaly.packages.data_quality import DataQualityEngine
   quality_engine = DataQualityEngine()
   quality_report = quality_engine.assess_quality(real_data)
   ```

2. **AutoML for Automatic Algorithm Selection**
   ```python
   from pynomaly.automl import AutoAnomalyDetector
   auto_detector = AutoAnomalyDetector()
   auto_results = auto_detector.fit_detect(transformed_data)
   ```

3. **Streaming Data Processing**
   ```python
   from pynomaly.streaming import StreamingAnomalyDetector
   streaming_detector = StreamingAnomalyDetector(detector=loaded_detector)
   ```

### Best Practices

1. **Data Preparation is Key**
   - Always profile your data first
   - Handle missing values appropriately
   - Consider feature scaling and normalization

2. **Choose the Right Algorithm**
   - Use ensemble methods for robustness
   - Consider your data characteristics (high-dimensional, temporal, etc.)
   - Experiment with different contamination rates

3. **Validate Your Results**
   - Use domain knowledge to validate detected anomalies
   - Implement feedback loops to improve detection
   - Monitor model performance over time

4. **Production Considerations**
   - Set up proper monitoring and alerting
   - Version your models and data
   - Implement model retraining strategies

## Troubleshooting Common Issues

### Installation Issues

```bash
# If you encounter dependency conflicts
pip install pynomaly[minimal]  # Install minimal version first
pip install --upgrade pynomaly[standard]  # Then upgrade
```

### Memory Issues

```python
# For large datasets
detector = AnomalyDetector(
    algorithm='IsolationForest',
    memory_efficient=True,
    n_jobs=1  # Reduce parallel processing
)

# Use data sampling for very large datasets
sample_data = real_data.sample(n=10000, random_state=42)
```

### Performance Issues

```python
# Enable parallel processing
detector = AnomalyDetector(
    algorithm='IsolationForest',
    n_jobs=-1  # Use all available cores
)

# Use approximate algorithms for speed
detector = AnomalyDetector(
    algorithm='FastIsolationForest',
    approximate=True
)
```

## Summary

You've now learned:

- ✅ How to install and verify Pynomaly
- ✅ Basic anomaly detection workflow
- ✅ Data preprocessing and transformation
- ✅ Advanced detection with ensemble methods
- ✅ Result interpretation and explanation
- ✅ Model evaluation and validation
- ✅ Saving and loading models
- ✅ Setting up monitoring

## What's Next?

1. Explore the [User Guides](../user-guides/) for detailed package documentation
2. Check out [Best Practices](./best-practices.md) for production deployment
3. Review [API Reference](../api-reference/) for comprehensive documentation
4. Try the [Advanced Tutorials](./advanced-anomaly-detection.md) for complex use cases

Happy anomaly detecting! 🔍