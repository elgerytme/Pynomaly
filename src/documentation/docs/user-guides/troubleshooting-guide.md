# Troubleshooting Guide

## Overview

This guide helps you solve common problems when using Pynomaly for anomaly detection. Issues are organized by category with step-by-step solutions.

## Quick Diagnosis

**Having issues?** Start here:

1. **Installation problems** → [Installation Issues](#installation-issues)
2. **No anomalies detected** → [Detection Issues](#detection-issues)
3. **Too many false positives** → [False Positives](#false-positives)
4. **Poor performance** → [Performance Issues](#performance-issues)
5. **API/Integration errors** → [API Issues](#api-issues)
6. **Data preparation problems** → [Data Issues](#data-issues)

## Table of Contents

1. [Installation Issues](#installation-issues)
2. [Detection Issues](#detection-issues)
3. [False Positives](#false-positives)
4. [Performance Issues](#performance-issues)
5. [API Issues](#api-issues)
6. [Data Issues](#data-issues)
7. [Algorithm-Specific Issues](#algorithm-specific-issues)
8. [Production Issues](#production-issues)
9. [Getting Additional Help](#getting-help)

---

## Installation Issues {#installation-issues}

### Problem: `pip install pynomaly` fails

**Symptoms:**
```bash
ERROR: Could not find a version that satisfies the requirement pynomaly
ERROR: No matching distribution found for pynomaly
```

**Solutions:**

#### Solution 1: Update pip and try again
```bash
python -m pip install --upgrade pip
pip install pynomaly
```

#### Solution 2: Use specific Python version
```bash
python3 -m pip install pynomaly
```

#### Solution 3: Install from source
```bash
pip install git+https://github.com/pynomaly/pynomaly.git
```

#### Solution 4: Check Python version compatibility
```bash
python --version  # Should be 3.8 or higher
```

### Problem: Import errors after installation

**Symptoms:**
```python
ImportError: No module named 'pynomaly'
ModuleNotFoundError: No module named 'pynomaly'
```

**Solutions:**

#### Solution 1: Verify installation
```bash
pip list | grep pynomaly
```

#### Solution 2: Check Python environment
```python
import sys
print(sys.path)  # Verify pynomaly is in path
```

#### Solution 3: Reinstall in correct environment
```bash
# If using virtual environment
source venv/bin/activate  # or venv\Scripts\activate on Windows
pip install pynomaly

# If using conda
conda activate your_environment
pip install pynomaly
```

### Problem: Dependency conflicts

**Symptoms:**
```bash
ERROR: package-name has requirement other-package>=1.0, but you have other-package 0.9
```

**Solutions:**

#### Solution 1: Create clean environment
```bash
python -m venv pynomaly_env
source pynomaly_env/bin/activate
pip install pynomaly
```

#### Solution 2: Install with specific dependencies
```bash
pip install pynomaly[minimal]  # Minimal dependencies
pip install pynomaly[all]      # All optional dependencies
```

#### Solution 3: Resolve conflicts manually
```bash
pip install --upgrade conflicting-package
pip install pynomaly
```

---

## Detection Issues {#detection-issues}

### Problem: No anomalies detected

**Symptoms:**
```python
results = client.detect_anomalies(data)
print(sum(results['predictions']))  # Output: 0
```

**Diagnosis checklist:**
1. **Check data quality** - Is your data clean and properly formatted?
2. **Verify contamination rate** - Is it appropriate for your dataset?
3. **Examine data distribution** - Are there actually anomalies in your data?
4. **Algorithm selection** - Is the algorithm suitable for your data type?

**Solutions:**

#### Solution 1: Adjust contamination rate
```python
# Try different contamination rates
contamination_rates = [0.001, 0.01, 0.05, 0.1, 0.2]

for rate in contamination_rates:
    results = client.detect_anomalies(
        data=data, 
        contamination=rate
    )
    print(f"Contamination {rate}: {sum(results['predictions'])} anomalies")
```

#### Solution 2: Check data preprocessing
```python
import pandas as pd
import numpy as np

# Check for data issues
print("Data shape:", data.shape)
print("Missing values:", data.isnull().sum())
print("Data types:", data.dtypes)
print("Data range:", data.describe())

# Look for obvious outliers
for col in data.select_dtypes(include=[np.number]).columns:
    Q1 = data[col].quantile(0.25)
    Q3 = data[col].quantile(0.75)
    IQR = Q3 - Q1
    outliers = data[(data[col] < Q1 - 1.5*IQR) | (data[col] > Q3 + 1.5*IQR)]
    print(f"{col}: {len(outliers)} potential outliers")
```

#### Solution 3: Try different algorithms
```python
algorithms = ['isolation_forest', 'local_outlier_factor', 'one_class_svm']

for algorithm in algorithms:
    results = client.detect_anomalies(
        data=data,
        algorithm=algorithm,
        contamination=0.05
    )
    print(f"{algorithm}: {sum(results['predictions'])} anomalies")
```

#### Solution 4: Create synthetic anomalies for testing
```python
# Add obvious anomalies to test detection
test_data = data.copy()

# Add extreme values
extreme_row = test_data.iloc[0].copy()
for col in test_data.select_dtypes(include=[np.number]).columns:
    extreme_row[col] = test_data[col].max() * 3  # 3x maximum value

test_data = pd.concat([test_data, pd.DataFrame([extreme_row])], ignore_index=True)

# Test detection
results = client.detect_anomalies(test_data, contamination=0.05)
print(f"With synthetic anomaly: {sum(results['predictions'])} anomalies detected")
```

### Problem: Inconsistent results across runs

**Symptoms:**
```python
# Run 1
results1 = client.detect_anomalies(data)
# Run 2  
results2 = client.detect_anomalies(data)
# Different results!
```

**Solutions:**

#### Solution 1: Set random seed
```python
results = client.detect_anomalies(
    data=data,
    algorithm='isolation_forest',
    random_state=42  # Fixed seed for reproducibility
)
```

#### Solution 2: Use more stable algorithms
```python
# Local Outlier Factor is more deterministic
results = client.detect_anomalies(
    data=data,
    algorithm='local_outlier_factor'
)
```

#### Solution 3: Increase algorithm stability
```python
results = client.detect_anomalies(
    data=data,
    algorithm='isolation_forest',
    n_estimators=200,  # More trees = more stable
    max_samples=1.0,   # Use all data
    random_state=42
)
```

---

## False Positives {#false-positives}

### Problem: Too many false positives

**Symptoms:**
- Normal data points flagged as anomalies
- Business users questioning results
- High false positive rate in validation

**Diagnosis:**

#### Check false positive rate
```python
# If you have labeled data
from sklearn.metrics import classification_report, confusion_matrix

if 'true_labels' in data.columns:
    y_true = data['true_labels']
    y_pred = results['predictions']
    
    print("Classification Report:")
    print(classification_report(y_true, y_pred))
    
    print("\nConfusion Matrix:")
    print(confusion_matrix(y_true, y_pred))
```

**Solutions:**

#### Solution 1: Adjust contamination rate
```python
# Lower contamination = fewer anomalies detected
results = client.detect_anomalies(
    data=data,
    contamination=0.01  # Expect only 1% anomalies
)
```

#### Solution 2: Increase detection threshold
```python
# Use score threshold instead of contamination
results = client.detect_anomalies(data=data)

# Apply custom threshold
threshold = 0.7  # Only flag high-confidence anomalies
custom_predictions = [score > threshold for score in results['scores']]

print(f"Original anomalies: {sum(results['predictions'])}")
print(f"High-confidence anomalies: {sum(custom_predictions)}")
```

#### Solution 3: Feature engineering
```python
# Remove noisy features that cause false positives
print("Feature importance analysis:")

# Check correlation with anomaly scores
correlations = {}
for col in data.select_dtypes(include=[np.number]).columns:
    corr = np.corrcoef(data[col], results['scores'])[0,1]
    correlations[col] = abs(corr)

# Remove features with low correlation
important_features = [k for k, v in correlations.items() if v > 0.1]
print(f"Important features: {important_features}")

# Re-run with selected features
filtered_data = data[important_features]
improved_results = client.detect_anomalies(filtered_data, contamination=0.05)
```

#### Solution 4: Use ensemble methods
```python
# Combine multiple algorithms - only flag if majority agree
algorithms = ['isolation_forest', 'local_outlier_factor', 'one_class_svm']
all_results = []

for alg in algorithms:
    result = client.detect_anomalies(data=data, algorithm=alg, contamination=0.05)
    all_results.append(result['predictions'])

# Majority voting
ensemble_predictions = []
for i in range(len(data)):
    votes = sum([results[i] for results in all_results])
    ensemble_predictions.append(votes >= 2)  # At least 2/3 algorithms agree

print(f"Ensemble anomalies: {sum(ensemble_predictions)}")
```

### Problem: Domain experts disagree with results

**Solutions:**

#### Solution 1: Add domain knowledge as features
```python
# Example for fraud detection
def add_domain_features(data):
    enhanced_data = data.copy()
    
    # Business rules as features
    enhanced_data['is_high_amount'] = (data['amount'] > 1000).astype(int)
    enhanced_data['is_night_transaction'] = data['hour'].isin([22,23,0,1,2,3]).astype(int)
    enhanced_data['is_weekend'] = data['day_of_week'].isin([5,6]).astype(int)
    enhanced_data['velocity_risk'] = (data['time_since_last'] < 1).astype(int)
    
    return enhanced_data

enhanced_data = add_domain_features(data)
results = client.detect_anomalies(enhanced_data)
```

#### Solution 2: Create custom scoring
```python
def custom_anomaly_score(row, ml_score, domain_rules):
    """Combine ML score with domain knowledge"""
    
    domain_score = 0
    
    # Apply business rules
    if row['amount'] > 5000:
        domain_score += 0.3
    if row['hour'] in [0,1,2,3,4]:
        domain_score += 0.2
    if row.get('merchant_risk_level') == 'high':
        domain_score += 0.4
    
    # Weighted combination
    combined_score = 0.6 * ml_score + 0.4 * domain_score
    return min(combined_score, 1.0)

# Apply custom scoring
custom_scores = []
for idx, row in data.iterrows():
    ml_score = results['scores'][idx]
    custom_score = custom_anomaly_score(row, ml_score, {})
    custom_scores.append(custom_score)

# Create new predictions based on custom scores
threshold = 0.5
custom_predictions = [score > threshold for score in custom_scores]
```

---

## Performance Issues {#performance-issues}

### Problem: Slow detection on large datasets

**Symptoms:**
- Long processing times (> 5 minutes for 100K rows)
- Memory usage issues
- Timeouts in production

**Solutions:**

#### Solution 1: Use faster algorithms
```python
# Fast algorithms for large datasets
fast_algorithms = [
    'isolation_forest',      # Generally fastest
    'online_isolation_forest', # For streaming data
    'statistical_outliers'   # Very fast for simple cases
]

import time

for algorithm in fast_algorithms:
    start_time = time.time()
    results = client.detect_anomalies(
        data=data,
        algorithm=algorithm,
        contamination=0.05
    )
    elapsed = time.time() - start_time
    print(f"{algorithm}: {elapsed:.2f} seconds")
```

#### Solution 2: Batch processing
```python
def batch_anomaly_detection(data, batch_size=10000):
    """Process large datasets in batches"""
    
    all_predictions = []
    all_scores = []
    
    for i in range(0, len(data), batch_size):
        batch = data.iloc[i:i+batch_size]
        
        batch_results = client.detect_anomalies(
            data=batch,
            algorithm='isolation_forest',
            contamination=0.05
        )
        
        all_predictions.extend(batch_results['predictions'])
        all_scores.extend(batch_results['scores'])
        
        print(f"Processed batch {i//batch_size + 1}/{(len(data)-1)//batch_size + 1}")
    
    return {
        'predictions': all_predictions,
        'scores': all_scores
    }

# Use for large datasets
if len(data) > 50000:
    results = batch_anomaly_detection(data)
else:
    results = client.detect_anomalies(data)
```

#### Solution 3: Feature reduction
```python
from sklearn.decomposition import PCA
from sklearn.feature_selection import SelectKBest, f_classif

# Reduce dimensionality
if data.shape[1] > 20:  # If more than 20 features
    
    # Option 1: PCA
    pca = PCA(n_components=10)
    data_reduced = pd.DataFrame(
        pca.fit_transform(data),
        columns=[f'PC{i}' for i in range(10)]
    )
    
    # Option 2: Feature selection (if you have labels)
    # selector = SelectKBest(f_classif, k=10)
    # data_reduced = selector.fit_transform(data, labels)
    
    print(f"Original features: {data.shape[1]}")
    print(f"Reduced features: {data_reduced.shape[1]}")
    
    results = client.detect_anomalies(data_reduced)
```

#### Solution 4: Sampling for very large datasets
```python
# For datasets > 1M rows, use sampling
if len(data) > 1000000:
    # Sample for training
    sample_size = 100000
    sample_data = data.sample(n=sample_size, random_state=42)
    
    # Train on sample
    train_results = client.detect_anomalies(
        data=sample_data,
        algorithm='isolation_forest'
    )
    
    # Apply to full dataset (simplified approach)
    # In practice, you'd save the model and apply it
    print(f"Trained on {len(sample_data)} samples")
    print(f"Detected {sum(train_results['predictions'])} anomalies in sample")
```

### Problem: High memory usage

**Solutions:**

#### Solution 1: Process in chunks
```python
def memory_efficient_detection(data, chunk_size=5000):
    """Memory-efficient processing for large datasets"""
    
    results = {
        'predictions': [],
        'scores': []
    }
    
    for chunk_start in range(0, len(data), chunk_size):
        chunk_end = min(chunk_start + chunk_size, len(data))
        chunk = data.iloc[chunk_start:chunk_end]
        
        # Force garbage collection
        import gc
        gc.collect()
        
        chunk_results = client.detect_anomalies(
            data=chunk,
            algorithm='isolation_forest'
        )
        
        results['predictions'].extend(chunk_results['predictions'])
        results['scores'].extend(chunk_results['scores'])
        
        print(f"Processed rows {chunk_start}-{chunk_end}")
    
    return results
```

#### Solution 2: Use data types optimization
```python
# Optimize data types to reduce memory
def optimize_dtypes(df):
    """Optimize pandas DataFrame memory usage"""
    
    optimized_df = df.copy()
    
    # Optimize integers
    for col in optimized_df.select_dtypes(include=['int64']).columns:
        col_min = optimized_df[col].min()
        col_max = optimized_df[col].max()
        
        if col_min > np.iinfo(np.int8).min and col_max < np.iinfo(np.int8).max:
            optimized_df[col] = optimized_df[col].astype(np.int8)
        elif col_min > np.iinfo(np.int16).min and col_max < np.iinfo(np.int16).max:
            optimized_df[col] = optimized_df[col].astype(np.int16)
        elif col_min > np.iinfo(np.int32).min and col_max < np.iinfo(np.int32).max:
            optimized_df[col] = optimized_df[col].astype(np.int32)
    
    # Optimize floats
    for col in optimized_df.select_dtypes(include=['float64']).columns:
        optimized_df[col] = optimized_df[col].astype(np.float32)
    
    return optimized_df

# Apply optimization
print(f"Original memory usage: {data.memory_usage(deep=True).sum()} bytes")
optimized_data = optimize_dtypes(data)
print(f"Optimized memory usage: {optimized_data.memory_usage(deep=True).sum()} bytes")
```

---

## API Issues {#api-issues}

### Problem: API authentication failures

**Symptoms:**
```python
# Error messages
401 Unauthorized
403 Forbidden
Invalid API key
```

**Solutions:**

#### Solution 1: Verify API key setup
```python
import os

# Check environment variable
api_key = os.getenv('PYNOMALY_API_KEY')
if not api_key:
    print("API key not found in environment variables")
    print("Set it with: export PYNOMALY_API_KEY=your_key_here")
else:
    print(f"API key found: {api_key[:10]}...")

# Initialize client with explicit key
from pynomaly import PynomalyClient
client = PynomalyClient(api_key=api_key)
```

#### Solution 2: Test API connectivity
```python
# Test basic connectivity
try:
    # Test with small dataset
    test_data = pd.DataFrame({
        'feature1': [1, 2, 3, 4, 100],  # 100 is obvious outlier
        'feature2': [10, 20, 30, 40, 500]
    })
    
    results = client.detect_anomalies(test_data)
    print("API connection successful!")
    print(f"Test anomalies detected: {sum(results['predictions'])}")
    
except Exception as e:
    print(f"API connection failed: {e}")
    print("Check your API key and network connection")
```

### Problem: API rate limiting

**Symptoms:**
```
429 Too Many Requests
Rate limit exceeded
```

**Solutions:**

#### Solution 1: Implement retry logic
```python
import time
import random

def robust_anomaly_detection(data, max_retries=3):
    """Anomaly detection with retry logic for rate limiting"""
    
    for attempt in range(max_retries):
        try:
            results = client.detect_anomalies(data)
            return results
            
        except Exception as e:
            if "rate limit" in str(e).lower() or "429" in str(e):
                if attempt < max_retries - 1:
                    # Exponential backoff with jitter
                    wait_time = (2 ** attempt) + random.uniform(0, 1)
                    print(f"Rate limited. Waiting {wait_time:.1f} seconds...")
                    time.sleep(wait_time)
                else:
                    raise Exception("Rate limit exceeded after all retries")
            else:
                raise e
    
    return None

# Use robust detection
results = robust_anomaly_detection(data)
```

#### Solution 2: Batch requests efficiently
```python
def rate_limit_friendly_detection(data, batch_size=1000, delay=1.0):
    """Process large datasets with rate limiting consideration"""
    
    all_results = {'predictions': [], 'scores': []}
    
    for i in range(0, len(data), batch_size):
        batch = data.iloc[i:i+batch_size]
        
        # Add delay between requests
        if i > 0:
            time.sleep(delay)
        
        try:
            batch_results = client.detect_anomalies(batch)
            all_results['predictions'].extend(batch_results['predictions'])
            all_results['scores'].extend(batch_results['scores'])
            
            print(f"Processed batch {i//batch_size + 1}")
            
        except Exception as e:
            print(f"Error processing batch {i//batch_size + 1}: {e}")
            # Continue with next batch
            continue
    
    return all_results
```

### Problem: Timeout errors

**Solutions:**

#### Solution 1: Increase timeout
```python
# Configure longer timeout
client = PynomalyClient(
    api_key=your_api_key,
    timeout=300  # 5 minutes timeout
)
```

#### Solution 2: Process smaller chunks
```python
# Break large requests into smaller ones
def timeout_safe_detection(data, max_chunk_size=5000):
    if len(data) > max_chunk_size:
        print(f"Large dataset detected ({len(data)} rows). Processing in chunks...")
        return batch_anomaly_detection(data, batch_size=max_chunk_size)
    else:
        return client.detect_anomalies(data)
```

---

## Data Issues {#data-issues}

### Problem: Mixed data types

**Symptoms:**
```python
TypeError: Input contains NaN, infinity or a value too large
ValueError: could not convert string to float
```

**Solutions:**

#### Solution 1: Proper data preprocessing
```python
def preprocess_mixed_data(df):
    """Handle mixed data types for anomaly detection"""
    
    processed_df = df.copy()
    
    # Handle missing values
    for col in processed_df.columns:
        if processed_df[col].dtype == 'object':
            # Categorical data - fill with mode
            processed_df[col] = processed_df[col].fillna(processed_df[col].mode()[0])
        else:
            # Numerical data - fill with median
            processed_df[col] = processed_df[col].fillna(processed_df[col].median())
    
    # Encode categorical variables
    from sklearn.preprocessing import LabelEncoder
    
    categorical_columns = processed_df.select_dtypes(include=['object']).columns
    label_encoders = {}
    
    for col in categorical_columns:
        le = LabelEncoder()
        processed_df[col] = le.fit_transform(processed_df[col].astype(str))
        label_encoders[col] = le
    
    # Handle infinite values
    processed_df = processed_df.replace([np.inf, -np.inf], np.nan)
    processed_df = processed_df.fillna(processed_df.median())
    
    return processed_df, label_encoders

# Apply preprocessing
clean_data, encoders = preprocess_mixed_data(data)
results = client.detect_anomalies(clean_data)
```

#### Solution 2: Feature selection for mixed types
```python
def select_numeric_features(df):
    """Select only numeric features for anomaly detection"""
    
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    numeric_data = df[numeric_cols].copy()
    
    # Remove columns with too many missing values
    missing_threshold = 0.5  # Remove if >50% missing
    good_cols = []
    
    for col in numeric_cols:
        missing_ratio = numeric_data[col].isnull().sum() / len(numeric_data)
        if missing_ratio < missing_threshold:
            good_cols.append(col)
    
    return numeric_data[good_cols].fillna(numeric_data[good_cols].median())

# Use only clean numeric data
numeric_data = select_numeric_features(data)
results = client.detect_anomalies(numeric_data)
```

### Problem: Highly correlated features

**Symptoms:**
- Poor detection performance
- Unstable results
- Algorithm warnings about multicollinearity

**Solutions:**

#### Solution 1: Remove highly correlated features
```python
def remove_correlated_features(df, threshold=0.95):
    """Remove highly correlated features"""
    
    # Calculate correlation matrix
    corr_matrix = df.corr().abs()
    
    # Find highly correlated pairs
    upper_tri = corr_matrix.where(
        np.triu(np.ones(corr_matrix.shape), k=1).astype(bool)
    )
    
    # Find features to drop
    to_drop = [column for column in upper_tri.columns if any(upper_tri[column] > threshold)]
    
    print(f"Dropping {len(to_drop)} highly correlated features: {to_drop}")
    
    return df.drop(columns=to_drop)

# Remove correlated features
uncorrelated_data = remove_correlated_features(data)
results = client.detect_anomalies(uncorrelated_data)
```

#### Solution 2: Use PCA for dimensionality reduction
```python
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

def pca_transformation(df, n_components=0.95):
    """Apply PCA to reduce correlation and dimensionality"""
    
    # Standardize features
    scaler = StandardScaler()
    scaled_data = scaler.fit_transform(df)
    
    # Apply PCA
    pca = PCA(n_components=n_components)  # Keep 95% of variance
    pca_data = pca.fit_transform(scaled_data)
    
    print(f"Original features: {df.shape[1]}")
    print(f"PCA components: {pca_data.shape[1]}")
    print(f"Explained variance: {sum(pca.explained_variance_ratio_):.3f}")
    
    return pd.DataFrame(
        pca_data, 
        columns=[f'PC{i}' for i in range(pca_data.shape[1])]
    )

# Apply PCA transformation
pca_data = pca_transformation(data)
results = client.detect_anomalies(pca_data)
```

---

## Algorithm-Specific Issues {#algorithm-specific-issues}

### Isolation Forest Issues

#### Problem: Poor performance on high-dimensional data

**Solution:**
```python
# Optimize Isolation Forest for high dimensions
results = client.detect_anomalies(
    data=data,
    algorithm='isolation_forest',
    max_features=min(10, data.shape[1]),  # Limit feature sampling
    n_estimators=200,  # More trees for stability
    contamination=0.05
)
```

#### Problem: Sensitivity to data scaling

**Solution:**
```python
from sklearn.preprocessing import StandardScaler

# Standardize features for Isolation Forest
scaler = StandardScaler()
scaled_data = pd.DataFrame(
    scaler.fit_transform(data),
    columns=data.columns
)

results = client.detect_anomalies(scaled_data, algorithm='isolation_forest')
```

### Local Outlier Factor Issues

#### Problem: Poor performance on sparse data

**Solution:**
```python
# Use smaller neighborhood for sparse data
results = client.detect_anomalies(
    data=data,
    algorithm='local_outlier_factor',
    n_neighbors=min(20, len(data)//10),  # Smaller neighborhood
    contamination=0.05
)
```

### One-Class SVM Issues

#### Problem: Very slow on large datasets

**Solution:**
```python
# Sample data for One-Class SVM
if len(data) > 10000:
    sample_data = data.sample(n=10000, random_state=42)
    results = client.detect_anomalies(
        data=sample_data,
        algorithm='one_class_svm',
        nu=0.05  # Equivalent to contamination
    )
else:
    results = client.detect_anomalies(data, algorithm='one_class_svm')
```

---

## Production Issues {#production-issues}

### Problem: Model drift in production

**Symptoms:**
- Increasing false positive rates over time
- Decreasing detection accuracy
- Business users reporting missed anomalies

**Solutions:**

#### Solution 1: Monitor data distribution
```python
def check_data_drift(reference_data, current_data):
    """Check for data drift between reference and current data"""
    
    drift_scores = {}
    
    for col in reference_data.columns:
        if col in current_data.columns:
            # Calculate distribution difference (simplified)
            ref_mean = reference_data[col].mean()
            ref_std = reference_data[col].std()
            
            curr_mean = current_data[col].mean()
            curr_std = current_data[col].std()
            
            # Drift score based on mean and std changes
            mean_drift = abs(curr_mean - ref_mean) / ref_std if ref_std > 0 else 0
            std_drift = abs(curr_std - ref_std) / ref_std if ref_std > 0 else 0
            
            drift_scores[col] = max(mean_drift, std_drift)
    
    return drift_scores

# Example usage
training_data = pd.read_csv('training_data.csv')
current_data = pd.read_csv('current_data.csv')

drift_scores = check_data_drift(training_data, current_data)
print("Data drift scores (>0.5 indicates significant drift):")
for col, score in drift_scores.items():
    if score > 0.5:
        print(f"⚠️  {col}: {score:.3f}")
    else:
        print(f"✅ {col}: {score:.3f}")
```

#### Solution 2: Implement model retraining
```python
def should_retrain_model(drift_scores, performance_metrics):
    """Decide if model needs retraining"""
    
    # Retraining criteria
    high_drift_features = sum(1 for score in drift_scores.values() if score > 0.5)
    avg_drift = np.mean(list(drift_scores.values()))
    
    # Performance degradation
    current_precision = performance_metrics.get('precision', 1.0)
    
    retrain_needed = (
        high_drift_features > len(drift_scores) * 0.3 or  # >30% features drifted
        avg_drift > 0.3 or  # Average drift too high
        current_precision < 0.7  # Performance degraded
    )
    
    return retrain_needed, {
        'high_drift_features': high_drift_features,
        'avg_drift': avg_drift,
        'precision': current_precision
    }

# Check if retraining needed
needs_retrain, metrics = should_retrain_model(drift_scores, {'precision': 0.65})
if needs_retrain:
    print("🔄 Model retraining recommended")
    print(f"Metrics: {metrics}")
```

### Problem: Inconsistent results across environments

**Solutions:**

#### Solution 1: Environment validation
```python
def validate_environment():
    """Validate that environment is consistent for anomaly detection"""
    
    import sys
    import numpy as np
    import pandas as pd
    
    print("Environment Validation:")
    print(f"Python version: {sys.version}")
    print(f"NumPy version: {np.__version__}")
    print(f"Pandas version: {pd.__version__}")
    
    # Test reproducibility
    test_data = pd.DataFrame({
        'x': np.random.RandomState(42).normal(0, 1, 100),
        'y': np.random.RandomState(42).normal(0, 1, 100)
    })
    
    results1 = client.detect_anomalies(test_data, random_state=42)
    results2 = client.detect_anomalies(test_data, random_state=42)
    
    if results1['predictions'] == results2['predictions']:
        print("✅ Results are reproducible")
    else:
        print("❌ Results are not reproducible - check for randomness issues")

validate_environment()
```

#### Solution 2: Model serialization and loading
```python
import pickle
import hashlib

def save_model_state(model_results, data_hash, filename):
    """Save model state for consistent deployment"""
    
    model_state = {
        'predictions': model_results['predictions'],
        'scores': model_results['scores'],
        'algorithm': model_results.get('algorithm', 'unknown'),
        'parameters': model_results.get('parameters', {}),
        'data_hash': data_hash,
        'timestamp': pd.Timestamp.now().isoformat()
    }
    
    with open(filename, 'wb') as f:
        pickle.dump(model_state, f)
    
    print(f"Model state saved to {filename}")

def load_model_state(filename):
    """Load saved model state"""
    
    with open(filename, 'rb') as f:
        model_state = pickle.load(f)
    
    print(f"Model state loaded from {filename}")
    print(f"Algorithm: {model_state['algorithm']}")
    print(f"Timestamp: {model_state['timestamp']}")
    
    return model_state

# Create data hash for consistency checking
data_hash = hashlib.md5(str(data.values).encode()).hexdigest()
print(f"Data hash: {data_hash}")

# Save model state
save_model_state(results, data_hash, 'model_state.pkl')
```

---

## Getting Additional Help {#getting-help}

### Community Support

1. **GitHub Discussions** - [https://github.com/pynomaly/pynomaly/discussions](https://github.com/pynomaly/pynomaly/discussions)
   - General questions and discussions
   - Community troubleshooting
   - Best practices sharing

2. **GitHub Issues** - [https://github.com/pynomaly/pynomaly/issues](https://github.com/pynomaly/pynomaly/issues)
   - Bug reports
   - Feature requests
   - Technical issues

### Documentation Resources

1. **[Complete Beginner's Guide](complete-beginners-guide.md)** - Start here if you're new
2. **[Interactive Tutorial](interactive-tutorial.md)** - Hands-on practice
3. **[Workflow Guides](workflows-guide.md)** - Role-specific instructions
4. **[API Documentation](../api/)** - Complete API reference

### When to Contact Support

**Contact community support for:**
- General usage questions
- Best practices advice
- Feature discussions
- Open source contributions

**Contact enterprise support for:**
- Production deployment issues
- Custom algorithm development
- Performance optimization
- SLA requirements

### Preparing a Good Support Request

When asking for help, include:

1. **Environment information:**
```python
import sys
import pynomaly
import numpy as np
import pandas as pd

print("Environment Info:")
print(f"Python: {sys.version}")
print(f"Pynomaly: {pynomaly.__version__}")
print(f"NumPy: {np.__version__}")
print(f"Pandas: {pd.__version__}")
```

2. **Minimal reproducible example:**
```python
import pandas as pd
from pynomaly import PynomalyClient

# Minimal data that reproduces the issue
data = pd.DataFrame({
    'feature1': [1, 2, 3, 4, 5],
    'feature2': [10, 20, 30, 40, 50]
})

# Code that reproduces the problem
client = PynomalyClient()
try:
    results = client.detect_anomalies(data)
    print("Success!")
except Exception as e:
    print(f"Error: {e}")
```

3. **Clear description of:**
   - What you expected to happen
   - What actually happened
   - Steps to reproduce the issue
   - Any error messages (full traceback)

### Performance Benchmarking

If reporting performance issues, include benchmarks:

```python
import time
import psutil
import os

def benchmark_detection(data, algorithm='isolation_forest'):
    """Benchmark anomaly detection performance"""
    
    # Memory before
    process = psutil.Process(os.getpid())
    memory_before = process.memory_info().rss / 1024 / 1024  # MB
    
    # Timing
    start_time = time.time()
    
    results = client.detect_anomalies(
        data=data,
        algorithm=algorithm,
        contamination=0.05
    )
    
    end_time = time.time()
    
    # Memory after
    memory_after = process.memory_info().rss / 1024 / 1024  # MB
    
    print(f"Performance Benchmark:")
    print(f"Dataset size: {data.shape}")
    print(f"Algorithm: {algorithm}")
    print(f"Processing time: {end_time - start_time:.2f} seconds")
    print(f"Memory usage: {memory_after - memory_before:.1f} MB increase")
    print(f"Anomalies detected: {sum(results['predictions'])}")
    
    return {
        'processing_time': end_time - start_time,
        'memory_usage': memory_after - memory_before,
        'dataset_size': data.shape,
        'anomaly_count': sum(results['predictions'])
    }

# Run benchmark
benchmark_results = benchmark_detection(data)
```

---

## Quick Reference

### Common Issues Checklist

Before asking for help, check:

- ✅ **Installation**: Is Pynomaly properly installed?
- ✅ **Data quality**: Clean data, no NaN/inf values?
- ✅ **Contamination rate**: Appropriate for your dataset?
- ✅ **Algorithm choice**: Suitable for your data type and size?
- ✅ **Environment**: Consistent Python/package versions?
- ✅ **Memory**: Sufficient RAM for your dataset size?
- ✅ **API**: Valid API key and network connectivity?

### Emergency Debugging

Quick debug script to run when things go wrong:

```python
def emergency_debug(data):
    """Quick debugging for anomaly detection issues"""
    
    print("=== EMERGENCY DEBUG ===")
    
    # Data info
    print(f"Data shape: {data.shape}")
    print(f"Data types: {data.dtypes.value_counts()}")
    print(f"Missing values: {data.isnull().sum().sum()}")
    print(f"Infinite values: {np.isinf(data.select_dtypes(include=[np.number])).sum().sum()}")
    
    # Try simple detection
    try:
        simple_data = data.select_dtypes(include=[np.number]).fillna(0)
        if len(simple_data.columns) == 0:
            print("❌ No numeric columns found!")
            return
            
        results = client.detect_anomalies(
            data=simple_data.head(100),  # Small sample
            algorithm='isolation_forest',
            contamination=0.1
        )
        print(f"✅ Basic detection works: {sum(results['predictions'])} anomalies")
        
    except Exception as e:
        print(f"❌ Basic detection failed: {e}")
    
    print("=== END DEBUG ===")

# Run emergency debug
emergency_debug(data)
```

---

**Need more help?** Don't hesitate to reach out to the community or check the latest documentation for updates and new troubleshooting solutions!