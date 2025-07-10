# Complete Beginner's Guide to Pynomaly

Welcome to **Pynomaly** - the most comprehensive anomaly detection platform! This guide will take you from zero to detecting anomalies like a pro in just 30 minutes.

## 🚀 What is Pynomaly?

Pynomaly is an enterprise-grade anomaly detection platform that makes machine learning accessible to everyone. Whether you're a data scientist, business analyst, or developer, Pynomaly provides:

- **🎯 One-Click Anomaly Detection**: Upload data, get results instantly
- **🧠 Multiple AI Algorithms**: 50+ state-of-the-art detection methods
- **🔒 Enterprise Security**: Bank-grade security with WAF protection
- **📊 Beautiful Dashboards**: Interactive visualizations and monitoring
- **⚡ Production Ready**: Scalable, reliable, and fast

## 🛠️ Quick Start (5 Minutes)

### Step 1: Access Pynomaly

Open your web browser and navigate to your Pynomaly instance:

```
http://localhost:8000
```

### Step 2: Create Your First Dataset

1. Click **"Upload Dataset"** on the main dashboard
2. Drag & drop your CSV file or click **"Browse Files"**
3. Preview your data and click **"Upload"**

> **💡 Tip**: Don't have data? Use our sample datasets in the "Examples" section!

### Step 3: Detect Anomalies

1. Click **"Create Detector"**
2. Choose your dataset from the dropdown
3. Select **"Auto-Detect"** for algorithm selection
4. Click **"Train & Detect"**

### Step 4: View Results

🎉 **Congratulations!** Your anomalies are detected and visualized instantly.

## 📚 Understanding Your Data

### Supported Data Formats

| Format | Description | Example Use Case |
|--------|-------------|------------------|
| **CSV** | Comma-separated values | Sales data, sensor readings |
| **JSON** | JavaScript Object Notation | API logs, web analytics |
| **Parquet** | Columnar storage format | Big data, data lakes |
| **Excel** | Microsoft Excel files | Business reports, financial data |

### Data Requirements

- **Minimum**: 100 rows recommended
- **Columns**: Numeric data works best
- **Size**: Up to 100MB per file
- **Missing Values**: Automatically handled

### Data Types Pynomaly Handles

- **📈 Time Series**: Stock prices, sensor data, metrics
- **🏢 Tabular**: Customer data, transactions, logs  
- **🌐 Network**: Graph data, connections, relationships
- **📱 Streaming**: Real-time data feeds

## 🎯 Choosing the Right Algorithm

Pynomaly offers 50+ algorithms, but here's how to pick the best one:

### For Beginners: Auto-Detection 🤖

```
✅ Best for: First-time users
✅ Pros: Automatically selects optimal algorithm
✅ Use when: You're not sure which algorithm to use
```

### Popular Algorithms by Use Case

#### 📊 **Statistical Anomalies**

- **Isolation Forest** - Great for general-purpose detection
- **Local Outlier Factor (LOF)** - Finds local anomalies
- **One-Class SVM** - Robust to noise

#### 📈 **Time Series Data**

- **LSTM Autoencoder** - Deep learning for complex patterns
- **Seasonal Decomposition** - Handles seasonal data
- **ARIMA-based Detection** - Classic statistical approach

#### 🏢 **Business Data**

- **Gaussian Mixture Model** - Multiple normal behaviors
- **K-Means Clustering** - Group-based anomalies
- **PCA** - Dimensionality reduction approach

## 🔧 Advanced Features

### AutoML - Automatic Machine Learning

Let Pynomaly automatically:

- Select the best algorithm
- Tune hyperparameters
- Validate results
- Generate explanations

**How to use:**

1. Upload your dataset
2. Click **"AutoML Detection"**
3. Wait for optimization (2-10 minutes)
4. Get the best possible results!

### Ensemble Methods

Combine multiple algorithms for better accuracy:

```python
# Example: Combining 3 algorithms
ensemble = [
    "IsolationForest",
    "LocalOutlierFactor", 
    "OneClassSVM"
]
```

### Real-Time Detection

Monitor live data streams:

1. Go to **"Streaming"** tab
2. Configure your data source
3. Set detection thresholds
4. Receive instant alerts

## 📊 Understanding Results

### Anomaly Scores

- **Score Range**: 0.0 to 1.0
- **0.0**: Completely normal
- **1.0**: Definitely an anomaly
- **0.7+**: Usually worth investigating

### Visualizations Available

- **📈 Scatter Plots**: See anomalies in data space
- **📉 Time Series**: Anomalies over time
- **🔥 Heatmaps**: Pattern intensity
- **📊 Histograms**: Score distributions

### Export Options

- **CSV**: Raw results with scores
- **PDF**: Professional reports
- **JSON**: API-friendly format
- **PNG/SVG**: Charts and visualizations

3. [Understanding Types of Anomalies](#understanding-types-of-anomalies)
4. [Your First 10 Minutes with Pynomaly](#your-first-10-minutes)
5. [Understanding Your Results](#understanding-results)
6. [Choosing the Right Algorithm](#choosing-algorithm)
7. [Common Use Cases](#common-use-cases)
8. [Next Steps](#next-steps)

## What is Anomaly Detection?

Anomaly detection is the process of identifying data points, events, or patterns that deviate significantly from the expected behavior in your dataset. Think of it as finding the "odd ones out" in your data.

### Real-World Examples

- **Banking**: Detecting fraudulent credit card transactions
- **Manufacturing**: Identifying defective products on an assembly line
- **Healthcare**: Spotting unusual patient vital signs
- **IT Systems**: Finding network intrusions or system failures
- **Quality Control**: Detecting faulty products before they reach customers

### Why is it Important?

Anomalies often represent:

- **Critical issues** that need immediate attention
- **Fraud or security breaches**
- **Quality problems** in manufacturing
- **System failures** or performance issues
- **Opportunities** for improvement or investigation

## Why Use Pynomaly?

Pynomaly is designed to make anomaly detection accessible to everyone:

### ✅ **Beginner-Friendly**

- No need to be a machine learning expert
- Simple APIs and clear documentation
- Built-in best practices and sensible defaults

### ✅ **Powerful and Flexible**

- 15+ state-of-the-art algorithms
- Handles different data types (tabular, time series, text)
- Scales from small datasets to enterprise workloads

### ✅ **Production-Ready**

- Enterprise-grade security and monitoring
- REST API and multiple client libraries
- Comprehensive testing and validation

### ✅ **Explainable Results**

- Understand why something was flagged as anomalous
- Visual explanations and confidence scores
- Export results in multiple formats

## Understanding Types of Anomalies

Before diving into detection, it's helpful to understand what kinds of anomalies exist:

### 1. Point Anomalies

**Individual data points that are unusual**

```
Example: A $10,000 credit card transaction when typical transactions are $20-$200
```

**When to use**: Most common type, good for fraud detection, outlier identification

### 2. Contextual Anomalies

**Data points that are normal in some contexts but unusual in others**

```
Example: Wearing a winter coat is normal in December but unusual in July
```

**When to use**: Time-dependent data, seasonal patterns, user behavior analysis

### 3. Collective Anomalies

**Groups of data points that together form an unusual pattern**

```
Example: A series of small transactions followed by a large withdrawal
```

**When to use**: Sequential data, complex fraud patterns, system behavior analysis

## Your First 10 Minutes

Let's get you up and running with your first anomaly detection in just 10 minutes!

### Step 1: Installation (2 minutes)

```bash
# Basic installation
pip install pynomaly

# Or with all features
pip install pynomaly[all]
```

### Step 2: Your First Detection (5 minutes)

Create a file called `my_first_detection.py`:

```python
import pandas as pd
from pynomaly import PynomalyClient

# Create some sample data
data = pd.DataFrame({
    'amount': [10, 12, 11, 9, 13, 8, 10, 500, 11, 12],  # Notice the 500!
    'time_of_day': [9, 10, 11, 12, 13, 14, 15, 16, 17, 18],
    'day_of_week': [1, 1, 1, 1, 1, 1, 1, 1, 1, 1]
})

# Initialize Pynomaly
client = PynomalyClient()

# Detect anomalies
results = client.detect_anomalies(data)

# Print results
print("Anomalies found:")
for i, is_anomaly in enumerate(results['predictions']):
    if is_anomaly:
        print(f"Row {i}: {data.iloc[i].to_dict()} - Score: {results['scores'][i]:.3f}")
```

### Step 3: Run Your Detection (1 minute)

```bash
python my_first_detection.py
```

You should see output like:

```
Anomalies found:
Row 7: {'amount': 500, 'time_of_day': 16, 'day_of_week': 1} - Score: 0.892
```

### Step 4: Understanding What Happened (2 minutes)

🎉 **Congratulations!** You just:

1. **Created sample data** with a clear outlier (the $500 transaction)
2. **Used Pynomaly's default algorithm** (Isolation Forest) to find anomalies
3. **Got results** showing which data points are unusual and how unusual they are

The score (0.892) indicates high confidence that this is an anomaly (scores closer to 1.0 are more anomalous).

## Understanding Your Results

When Pynomaly analyzes your data, it returns several important pieces of information:

### Prediction Results

```python
results = {
    'predictions': [False, False, False, True, False],  # Boolean: Is it an anomaly?
    'scores': [0.1, 0.2, 0.15, 0.85, 0.12],           # Float: How anomalous (0-1)?
    'algorithm': 'isolation_forest',                    # Which algorithm was used
    'metadata': {...}                                   # Additional information
}
```

### Interpreting Scores

| Score Range | Interpretation | Action |
|-------------|----------------|--------|
| 0.0 - 0.3   | Normal behavior | No action needed |
| 0.3 - 0.6   | Slightly unusual | Monitor |
| 0.6 - 0.8   | Suspicious | Investigate |
| 0.8 - 1.0   | Highly anomalous | Immediate attention |

### Getting Explanations

```python
# Get detailed explanations
explanations = client.explain_anomalies(data, results)

for explanation in explanations:
    print(f"Feature contributions:")
    for feature, contribution in explanation['feature_importance'].items():
        print(f"  {feature}: {contribution:.3f}")
```

This shows which features (columns) contributed most to the anomaly score.

## Choosing the Right Algorithm

Pynomaly offers many algorithms. Here's a simple guide to help you choose:

### For Beginners: Start Here

| Algorithm | Best For | Why Choose It |
|-----------|----------|---------------|
| **Isolation Forest** | General purpose, mixed data types | Fast, works well out-of-the-box, handles different data types |
| **Local Outlier Factor** | Dense, well-clustered data | Good for finding local anomalies in clustered data |
| **One-Class SVM** | High-dimensional data | Robust to noise, works well with many features |

### Quick Selection Guide

```python
# General purpose - start here
client = PynomalyClient(algorithm='isolation_forest')

# For time series data
client = PynomalyClient(algorithm='lstm_autoencoder')

# For very large datasets
client = PynomalyClient(algorithm='online_isolation_forest')

# For high-dimensional data
client = PynomalyClient(algorithm='one_class_svm')
```

### Let Pynomaly Choose for You

```python
# AutoML - Pynomaly picks the best algorithm for your data
client = PynomalyClient(algorithm='auto')
results = client.detect_anomalies(data)
print(f"Best algorithm chosen: {results['algorithm']}")
```

## Common Use Cases

### 1. Fraud Detection

```python
# Credit card transaction data
transactions = pd.DataFrame({
    'amount': [25.50, 67.89, 12.34, 15000.00, 45.67],  # Large amount suspicious
    'merchant_category': ['grocery', 'restaurant', 'gas', 'jewelry', 'grocery'],
    'time_since_last': [2, 4, 1, 0.1, 6],  # Very quick succession suspicious
    'location_risk': ['low', 'low', 'low', 'high', 'low']
})

# Detect fraudulent transactions
fraud_results = client.detect_anomalies(
    transactions, 
    algorithm='isolation_forest',
    contamination=0.1  # Expect ~10% fraud
)
```

### 2. Quality Control

```python
# Manufacturing sensor data
sensor_data = pd.DataFrame({
    'temperature': [98.5, 99.1, 98.8, 105.2, 98.9],  # High temp suspicious
    'pressure': [14.7, 14.8, 14.6, 14.9, 14.7],
    'vibration': [0.1, 0.12, 0.09, 0.85, 0.11],      # High vibration suspicious
    'production_rate': [100, 98, 102, 75, 99]        # Low rate suspicious
})

# Detect quality issues
quality_results = client.detect_anomalies(sensor_data)
```

### 3. System Monitoring

```python
# Server performance metrics
server_metrics = pd.DataFrame({
    'cpu_usage': [45, 52, 48, 95, 47],     # High CPU suspicious
    'memory_usage': [60, 65, 58, 62, 89],  # High memory suspicious
    'disk_io': [100, 120, 95, 110, 500],   # High I/O suspicious
    'network_traffic': [50, 60, 45, 55, 200]  # High traffic suspicious
})

# Detect system anomalies
system_results = client.detect_anomalies(server_metrics)
```

## Working with Real Data

### Data Preparation Tips

1. **Clean your data**:

   ```python
   # Remove missing values
   data = data.dropna()
   
   # Or fill missing values
   data = data.fillna(data.mean())
   ```

2. **Handle categorical data**:

   ```python
   # Pynomaly can handle categories automatically, but you can also encode them
   data['category_encoded'] = pd.Categorical(data['category']).codes
   ```

3. **Scale your features if needed**:

   ```python
   # For algorithms sensitive to scale
   from sklearn.preprocessing import StandardScaler
   
   scaler = StandardScaler()
   data_scaled = pd.DataFrame(
       scaler.fit_transform(data), 
       columns=data.columns
   )
   ```

### Handling Different Data Sizes

```python
# Small datasets (< 10,000 rows)
results = client.detect_anomalies(data)

# Medium datasets (10,000 - 1M rows)
results = client.detect_anomalies(data, algorithm='isolation_forest')

# Large datasets (> 1M rows)
results = client.detect_anomalies(data, algorithm='online_isolation_forest')

# Streaming data
stream_client = client.create_stream_processor()
for batch in data_batches:
    batch_results = stream_client.process_batch(batch)
```

## Improving Your Results

### 1. Tune Algorithm Parameters

```python
# Adjust contamination rate (expected % of anomalies)
results = client.detect_anomalies(
    data, 
    contamination=0.05  # Expect 5% anomalies
)

# Adjust sensitivity
results = client.detect_anomalies(
    data,
    algorithm='isolation_forest',
    n_estimators=200,  # More trees = more stable results
    max_samples=0.8    # Use 80% of data for each tree
)
```

### 2. Use Ensemble Methods

```python
# Combine multiple algorithms for better results
results = client.detect_anomalies(
    data,
    algorithm='ensemble',
    ensemble_methods=['isolation_forest', 'local_outlier_factor', 'one_class_svm'],
    voting='soft'  # Use probability averaging
)
```

### 3. Domain-Specific Features

```python
# Add domain knowledge as features
transactions['is_weekend'] = transactions['day_of_week'].isin([5, 6])
transactions['is_night'] = transactions['hour'].between(22, 6)
transactions['amount_vs_avg'] = transactions['amount'] / transactions.groupby('user_id')['amount'].transform('mean')
```

## Common Pitfalls and How to Avoid Them

### 1. **Wrong Contamination Rate**

❌ **Problem**: Setting contamination too high or too low
✅ **Solution**: Start with your best estimate, then validate with known anomalies

```python
# Try different contamination rates
for contamination in [0.01, 0.05, 0.1, 0.15]:
    results = client.detect_anomalies(data, contamination=contamination)
    print(f"Contamination {contamination}: {sum(results['predictions'])} anomalies found")
```

### 2. **Not Understanding Your Data**

❌ **Problem**: Applying algorithms without understanding data characteristics
✅ **Solution**: Always explore your data first

```python
# Understand your data before detection
print(data.describe())
print(data.info())
print(data.hist())
```

### 3. **Ignoring Data Quality**

❌ **Problem**: Running detection on dirty data
✅ **Solution**: Clean and validate your data

```python
# Check data quality
print(f"Missing values: {data.isnull().sum()}")
print(f"Duplicate rows: {data.duplicated().sum()}")
print(f"Data types: {data.dtypes}")
```

### 4. **Not Validating Results**

❌ **Problem**: Trusting algorithm output without validation
✅ **Solution**: Always validate with domain experts

```python
# Get validation metrics if you have labeled data
from sklearn.metrics import classification_report

if 'true_labels' in data.columns:
    print(classification_report(data['true_labels'], results['predictions']))
```

## Next Steps

Congratulations! You now understand the basics of anomaly detection with Pynomaly. Here's what to explore next:

### Immediate Next Steps

1. **Try the [Interactive Tutorial](interactive-tutorial.md)** - Hands-on Jupyter notebook
2. **Explore [Common Workflows](workflows-guide.md)** - Step-by-step guides for specific industries
3. **Read [Algorithm Selection Guide](../reference/algorithm-comparison.md)** - Deep dive into choosing algorithms

### As You Advance

1. **[Production Deployment Guide](../deployment/production-deployment.md)** - Deploy Pynomaly in production
2. **[API Reference](../api/)** - Complete API documentation
3. **[Custom Algorithms](advanced-customization.md)** - Build your own detection methods

### Get Help and Support

- **Community Forum**: [GitHub Discussions](https://github.com/pynomaly/pynomaly/discussions)
- **Bug Reports**: [GitHub Issues](https://github.com/pynomaly/pynomaly/issues)
- **Documentation**: [docs.pynomaly.ai](https://docs.pynomaly.ai)

### Practice Datasets

Try Pynomaly with these practice datasets:

- **[Credit Card Fraud](../examples/fraud-detection-tutorial.md)** - Classic fraud detection
- **[Network Intrusion](../examples/network-security-tutorial.md)** - Cybersecurity monitoring
- **[Quality Control](../examples/manufacturing-quality-tutorial.md)** - Manufacturing anomalies

---

## Quick Reference

### Basic Detection

```python
from pynomaly import PynomalyClient
client = PynomalyClient()
results = client.detect_anomalies(data)
```

### With Algorithm Choice

```python
client = PynomalyClient(algorithm='isolation_forest')
results = client.detect_anomalies(data, contamination=0.1)
```

### Get Explanations

```python
explanations = client.explain_anomalies(data, results)
```

### Export Results

```python
client.export_results(results, format='csv', filename='anomalies.csv')
```

---

**Ready to detect some anomalies?** Start with the [Interactive Tutorial](interactive-tutorial.md) or jump into a specific [workflow guide](workflows-guide.md) for your use case!
