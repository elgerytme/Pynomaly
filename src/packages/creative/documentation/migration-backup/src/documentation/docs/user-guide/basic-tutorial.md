# Basic Tutorial: Your First Anomaly Detection Model

This tutorial will guide you through creating your first anomaly detection model with Pynomaly. You'll learn the fundamental concepts and build a working detector in under 30 minutes.

## 🎯 What You'll Learn

- Core Pynomaly concepts and workflow
- Loading and preparing data for anomaly detection
- Training your first anomaly detection model
- Evaluating model performance
- Making predictions on new data
- Visualizing results

## 📊 The Dataset

We'll use a synthetic dataset simulating network traffic data with normal patterns and anomalous events.

## 🚀 Step 1: Environment Setup

### Import Required Libraries
```python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime, timedelta

# Import Pynomaly
from pynomaly import AnomalyDetector
from pynomaly.data import DataLoader
from pynomaly.metrics import AnomalyMetrics
from pynomaly.visualization import AnomalyVisualizer
```

### Create Sample Data
```python
# Generate synthetic network traffic data
np.random.seed(42)
n_samples = 1000
timestamps = pd.date_range(start='2024-01-01', periods=n_samples, freq='1min')

# Normal network traffic patterns
normal_traffic = {
    'timestamp': timestamps,
    'bytes_per_second': np.random.normal(1000, 200, n_samples),
    'packets_per_second': np.random.normal(50, 10, n_samples),
    'connections_count': np.random.normal(20, 5, n_samples),
    'response_time_ms': np.random.normal(100, 20, n_samples)
}

# Add some anomalies (DDoS attacks, system failures, etc.)
anomaly_indices = np.random.choice(n_samples, size=50, replace=False)
for idx in anomaly_indices:
    normal_traffic['bytes_per_second'][idx] *= np.random.uniform(5, 10)  # Traffic spike
    normal_traffic['packets_per_second'][idx] *= np.random.uniform(8, 15)  # Packet flood
    normal_traffic['response_time_ms'][idx] *= np.random.uniform(10, 20)  # System stress

# Create DataFrame
data = pd.DataFrame(normal_traffic)
print(f"Generated {len(data)} samples with {len(anomaly_indices)} known anomalies")
print(data.head())
```

## 📈 Step 2: Data Exploration

### Basic Statistics
```python
# Display basic statistics
print("Dataset Overview:")
print(f"Shape: {data.shape}")
print(f"Time range: {data['timestamp'].min()} to {data['timestamp'].max()}")
print("\nFeature Statistics:")
print(data.describe())
```

### Visualize Data Patterns
```python
# Create visualization of the data
fig, axes = plt.subplots(2, 2, figsize=(15, 10))
fig.suptitle('Network Traffic Data Overview')

# Plot each feature
features = ['bytes_per_second', 'packets_per_second', 'connections_count', 'response_time_ms']
for i, feature in enumerate(features):
    ax = axes[i//2, i%2]
    ax.plot(data['timestamp'], data[feature], alpha=0.7)
    ax.set_title(f'{feature.replace("_", " ").title()}')
    ax.set_xlabel('Time')
    ax.tick_params(axis='x', rotation=45)

plt.tight_layout()
plt.show()
```

## 🔧 Step 3: Data Preparation

### Prepare Features
```python
# Extract feature columns (exclude timestamp)
feature_columns = ['bytes_per_second', 'packets_per_second', 'connections_count', 'response_time_ms']
X = data[feature_columns].values

print(f"Feature matrix shape: {X.shape}")
print(f"Features: {feature_columns}")
```

### Split Data
```python
# Split into training and testing sets (80/20 split)
split_idx = int(0.8 * len(X))
X_train = X[:split_idx]
X_test = X[split_idx:]

print(f"Training samples: {len(X_train)}")
print(f"Testing samples: {len(X_test)}")
```

## 🤖 Step 4: Train Anomaly Detection Model

### Initialize Detector
```python
# Create anomaly detector with Isolation Forest algorithm
detector = AnomalyDetector(
    algorithm='isolation_forest',
    contamination=0.05,  # Expect ~5% anomalies
    random_state=42
)

print(f"Initialized {detector.algorithm} detector")
print(f"Expected contamination rate: {detector.contamination}")
```

### Train the Model
```python
# Fit the model on training data
print("Training anomaly detection model...")
start_time = datetime.now()

detector.fit(X_train)

training_time = datetime.now() - start_time
print(f"✅ Training completed in {training_time.total_seconds():.2f} seconds")
```

### Model Information
```python
# Display model details
print("\nModel Details:")
print(f"Algorithm: {detector.algorithm}")
print(f"Number of estimators: {detector.model.n_estimators}")
print(f"Max samples: {detector.model.max_samples}")
print(f"Training samples: {len(X_train)}")
```

## 🔍 Step 5: Make Predictions

### Predict on Test Data
```python
# Get anomaly predictions
print("Making predictions on test data...")
predictions = detector.predict(X_test)
anomaly_scores = detector.decision_function(X_test)

# Count anomalies
n_anomalies = np.sum(predictions == -1)
anomaly_rate = n_anomalies / len(predictions) * 100

print(f"Detected {n_anomalies} anomalies out of {len(predictions)} samples")
print(f"Anomaly rate: {anomaly_rate:.2f}%")
```

### Analyze Anomaly Scores
```python
# Analyze score distribution
print(f"\nAnomaly Score Statistics:")
print(f"Mean score: {np.mean(anomaly_scores):.3f}")
print(f"Std score: {np.std(anomaly_scores):.3f}")
print(f"Min score: {np.min(anomaly_scores):.3f}")
print(f"Max score: {np.max(anomaly_scores):.3f}")

# Show most anomalous samples
most_anomalous_idx = np.argsort(anomaly_scores)[:5]
print(f"\nTop 5 Most Anomalous Samples:")
for i, idx in enumerate(most_anomalous_idx):
    score = anomaly_scores[idx]
    print(f"{i+1}. Index {idx}: Score {score:.3f}")
```

## 📊 Step 6: Evaluate Results

### Visualization
```python
# Create results visualization
fig, axes = plt.subplots(2, 2, figsize=(15, 10))
fig.suptitle('Anomaly Detection Results')

# Plot predictions over time
test_timestamps = data['timestamp'].iloc[split_idx:]
normal_mask = predictions == 1
anomaly_mask = predictions == -1

# Plot each feature with anomalies highlighted
for i, feature in enumerate(feature_columns):
    ax = axes[i//2, i%2]
    feature_values = X_test[:, i]
    
    # Plot normal points
    ax.scatter(test_timestamps[normal_mask], feature_values[normal_mask], 
              c='blue', alpha=0.6, s=20, label='Normal')
    
    # Plot anomalies
    ax.scatter(test_timestamps[anomaly_mask], feature_values[anomaly_mask], 
              c='red', alpha=0.8, s=50, label='Anomaly')
    
    ax.set_title(f'{feature.replace("_", " ").title()}')
    ax.set_xlabel('Time')
    ax.legend()
    ax.tick_params(axis='x', rotation=45)

plt.tight_layout()
plt.show()
```

### Score Distribution
```python
# Plot anomaly score distribution
plt.figure(figsize=(12, 4))

# Histogram of scores
plt.subplot(1, 2, 1)
plt.hist(anomaly_scores, bins=50, alpha=0.7, edgecolor='black')
plt.axvline(x=0, color='red', linestyle='--', label='Threshold')
plt.xlabel('Anomaly Score')
plt.ylabel('Frequency')
plt.title('Anomaly Score Distribution')
plt.legend()

# Box plot
plt.subplot(1, 2, 2)
normal_scores = anomaly_scores[predictions == 1]
anomaly_scores_detected = anomaly_scores[predictions == -1]
plt.boxplot([normal_scores, anomaly_scores_detected], 
           labels=['Normal', 'Anomaly'])
plt.ylabel('Anomaly Score')
plt.title('Score Distribution by Class')

plt.tight_layout()
plt.show()
```

## 🎯 Step 7: Advanced Analysis

### Feature Importance
```python
# Analyze which features contribute most to anomalies
anomaly_indices_test = np.where(predictions == -1)[0]
normal_indices_test = np.where(predictions == 1)[0]

print("Feature Analysis for Detected Anomalies:")
for i, feature in enumerate(feature_columns):
    anomaly_values = X_test[anomaly_indices_test, i]
    normal_values = X_test[normal_indices_test, i]
    
    anomaly_mean = np.mean(anomaly_values)
    normal_mean = np.mean(normal_values)
    difference = ((anomaly_mean - normal_mean) / normal_mean) * 100
    
    print(f"{feature:20s}: Normal={normal_mean:8.1f}, Anomaly={anomaly_mean:8.1f}, "
          f"Diff={difference:+6.1f}%")
```

### Real-time Prediction Example
```python
# Simulate real-time prediction on new data
print("\n" + "="*50)
print("REAL-TIME ANOMALY DETECTION SIMULATION")
print("="*50)

# Generate new sample
new_sample = np.array([[
    np.random.normal(1000, 200),      # bytes_per_second
    np.random.normal(50, 10),         # packets_per_second  
    np.random.normal(20, 5),          # connections_count
    np.random.normal(100, 20)         # response_time_ms
]]).reshape(1, -1)

# Make prediction
prediction = detector.predict(new_sample)[0]
confidence = detector.decision_function(new_sample)[0]

status = "🔴 ANOMALY DETECTED" if prediction == -1 else "✅ Normal"
print(f"\nPrediction: {status}")
print(f"Confidence Score: {confidence:.3f}")
print(f"Sample values: {new_sample[0]}")
```

## 🔧 Step 8: Model Persistence

### Save Model
```python
# Save the trained model
model_path = "network_anomaly_detector.pkl"
detector.save(model_path)
print(f"✅ Model saved to {model_path}")
```

### Load and Use Saved Model
```python
# Load the saved model
loaded_detector = AnomalyDetector.load(model_path)
print(f"✅ Model loaded from {model_path}")

# Verify it works
test_prediction = loaded_detector.predict(X_test[:5])
print(f"Test prediction with loaded model: {test_prediction}")
```

## 🎓 Key Concepts Learned

### 1. **Anomaly Detection Workflow**
- Data preparation and feature engineering
- Model training with unsupervised learning
- Prediction and scoring
- Result interpretation and visualization

### 2. **Isolation Forest Algorithm**
- Works by isolating anomalies in feature space
- Anomalies are easier to isolate (fewer splits needed)
- Returns anomaly scores (lower = more anomalous)
- Threshold-based classification (typically 0)

### 3. **Contamination Parameter**
- Estimates the proportion of anomalies in the dataset
- Affects the decision threshold
- Important for model calibration
- Should be tuned based on domain knowledge

### 4. **Evaluation Strategies**
- Visual inspection of results
- Score distribution analysis
- Feature-wise anomaly patterns
- Real-time prediction testing

## 🚀 Next Steps

Congratulations! You've successfully built your first anomaly detection model. Here's what to explore next:

### Immediate Next Steps
1. **Try Different Algorithms**: Experiment with `one_class_svm`, `local_outlier_factor`, or `autoencoder`
2. **Feature Engineering**: Add time-based features, rolling statistics, or domain-specific metrics
3. **Hyperparameter Tuning**: Optimize contamination rate and algorithm-specific parameters

### Advanced Topics
1. **[Real-time Detection](real-time-detection.md)**: Set up streaming anomaly detection
2. **[Model Training](model-training.md)**: Advanced training techniques and ensemble methods
3. **[Production Deployment](production-deployment.md)**: Deploy your model to production
4. **[ML Governance](ml-governance.md)**: Implement model lifecycle management

### Domain-Specific Tutorials
1. **[Financial Fraud Detection](tutorials/fraud-detection.md)**: Apply to financial transactions
2. **[IoT Sensor Monitoring](tutorials/iot-monitoring.md)**: Monitor industrial sensors
3. **[Time Series Anomalies](tutorials/time-series.md)**: Handle temporal dependencies

## 🤝 Getting Help

- **Documentation**: Browse our comprehensive guides
- **Examples**: Check out more tutorials and use cases
- **Community**: Join our forum for questions and discussions
- **GitHub**: Report issues and contribute to the project

Ready for more advanced features? Continue with [Data Management](data-management.md) to learn about handling larger datasets and complex data sources!