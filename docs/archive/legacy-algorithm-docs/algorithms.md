# Algorithm Guide

This guide provides comprehensive information about the anomaly detection algorithms available in Pynomaly, when to use them, and how to configure them for optimal performance.

## Overview

Pynomaly integrates multiple state-of-the-art anomaly detection algorithms from various libraries including PyOD, PyGOD, and scikit-learn. Each algorithm has specific strengths and is suitable for different types of data and anomaly patterns.

## Algorithm Categories

### 1. Distance-Based Algorithms
Detect anomalies based on distance or density measures.

### 2. Tree-Based Algorithms
Use tree structures to isolate anomalies.

### 3. Clustering-Based Algorithms
Identify anomalies as points that don't fit well into clusters.

### 4. Statistical Algorithms
Use statistical models to identify unusual patterns.

### 5. Neural Network Algorithms
Deep learning approaches for complex pattern recognition.

## Supported Algorithms

### Isolation Forest

**Type:** Tree-based  
**Library:** scikit-learn  
**Best for:** Large datasets with numerical features

#### Description
Isolation Forest isolates anomalies by randomly selecting features and split values. Anomalies are easier to isolate and require fewer splits.

#### When to Use
- Large datasets (>1000 samples)
- Numerical features
- Need fast training and prediction
- Unknown contamination rate
- Minimal feature engineering required

#### Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `n_estimators` | int | 100 | Number of base estimators |
| `max_samples` | int/float | "auto" | Samples to draw for each estimator |
| `contamination` | float | 0.1 | Expected anomaly proportion |
| `max_features` | int/float | 1.0 | Features to draw for each estimator |
| `bootstrap` | bool | False | Bootstrap samples |
| `random_state` | int | None | Random seed |

#### Example Configuration
```python
detector = await detection_service.create_detector(
    name="Isolation Forest Detector",
    algorithm="IsolationForest",
    parameters={
        "contamination": 0.05,
        "n_estimators": 200,
        "max_samples": 256,
        "random_state": 42
    }
)
```

#### Performance Characteristics
- **Training Time:** O(n log n)
- **Prediction Time:** O(log n)
- **Memory Usage:** Low
- **Scalability:** Excellent

#### Pros
- Fast and scalable
- No assumptions about data distribution
- Handles high-dimensional data well
- Built-in contamination estimation

#### Cons
- May struggle with very sparse data
- Less effective with categorical features
- Can miss local anomalies in dense regions

---

### Local Outlier Factor (LOF)

**Type:** Density-based  
**Library:** scikit-learn  
**Best for:** Local anomalies in varying density regions

#### Description
LOF computes the local density deviation of a data point with respect to its neighbors. Points with substantially lower density than their neighbors are considered anomalies.

#### When to Use
- Data with varying density regions
- Need to detect local outliers
- Relatively small to medium datasets
- Mixed normal and anomalous clusters

#### Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `n_neighbors` | int | 20 | Number of neighbors for density estimation |
| `algorithm` | str | "auto" | Neighbor search algorithm |
| `leaf_size` | int | 30 | Leaf size for tree algorithms |
| `metric` | str | "minkowski" | Distance metric |
| `p` | int | 2 | Parameter for Minkowski metric |
| `contamination` | float | 0.1 | Expected anomaly proportion |

#### Example Configuration
```python
detector = await detection_service.create_detector(
    name="LOF Detector",
    algorithm="LOF",
    parameters={
        "n_neighbors": 30,
        "contamination": 0.08,
        "metric": "euclidean"
    }
)
```

#### Performance Characteristics
- **Training Time:** O(n²)
- **Prediction Time:** O(n)
- **Memory Usage:** High
- **Scalability:** Poor for large datasets

#### Pros
- Excellent for local anomalies
- Works well with varying densities
- Intuitive density-based approach
- Good for exploratory analysis

#### Cons
- Computationally expensive
- Sensitive to parameter choice
- Struggles with high-dimensional data
- Not suitable for streaming data

---

### One-Class SVM (OCSVM)

**Type:** Boundary-based  
**Library:** scikit-learn  
**Best for:** Non-linear decision boundaries

#### Description
One-Class SVM learns a decision function for novelty detection, mapping input data to a high-dimensional feature space and finding a hyperplane that separates normal data from the origin.

#### When to Use
- Complex non-linear decision boundaries
- Need robust boundary estimation
- High-dimensional data
- Limited training data

#### Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `kernel` | str | "rbf" | Kernel type |
| `degree` | int | 3 | Degree for polynomial kernel |
| `gamma` | str/float | "scale" | Kernel coefficient |
| `coef0` | float | 0.0 | Independent term in kernel |
| `tol` | float | 1e-3 | Tolerance for stopping criterion |
| `nu` | float | 0.5 | Upper bound on training errors |
| `shrinking` | bool | True | Use shrinking heuristic |

#### Example Configuration
```python
detector = await detection_service.create_detector(
    name="One-Class SVM Detector",
    algorithm="OCSVM",
    parameters={
        "kernel": "rbf",
        "gamma": "auto",
        "nu": 0.1,
        "tol": 1e-4
    }
)
```

#### Performance Characteristics
- **Training Time:** O(n²) to O(n³)
- **Prediction Time:** O(n)
- **Memory Usage:** High
- **Scalability:** Poor for large datasets

#### Pros
- Handles non-linear boundaries well
- Robust to outliers in training data
- Works with high-dimensional data
- Solid theoretical foundation

#### Cons
- Slow training on large datasets
- Sensitive to parameter tuning
- Difficult to interpret
- Memory intensive

---

### COPOD (Copula-Based Outlier Detection)

**Type:** Statistical  
**Library:** PyOD  
**Best for:** High-dimensional tabular data

#### Description
COPOD uses copula functions to model the joint distribution of features and identifies anomalies based on their probability under this model.

#### When to Use
- High-dimensional tabular data
- Mixed data types (numerical/categorical)
- Need probabilistic anomaly scores
- Complex feature dependencies

#### Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `contamination` | float | 0.1 | Expected anomaly proportion |
| `n_jobs` | int | 1 | Number of parallel jobs |

#### Example Configuration
```python
detector = await detection_service.create_detector(
    name="COPOD Detector",
    algorithm="COPOD",
    parameters={
        "contamination": 0.05
    }
)
```

#### Performance Characteristics
- **Training Time:** O(n log n)
- **Prediction Time:** O(log n)
- **Memory Usage:** Medium
- **Scalability:** Good

#### Pros
- Fast and scalable
- Handles mixed data types
- No hyperparameter tuning required
- Provides probability scores

#### Cons
- Assumes feature independence
- May miss complex interactions
- Less effective on low-dimensional data
- Newer algorithm with less validation

---

### ECOD (Empirical Cumulative Distribution Outlier Detection)

**Type:** Statistical  
**Library:** PyOD  
**Best for:** Large-scale datasets with mixed distributions

#### Description
ECOD uses empirical cumulative distribution functions to model each feature independently and combines them to detect anomalies.

#### Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `contamination` | float | 0.1 | Expected anomaly proportion |
| `n_jobs` | int | 1 | Number of parallel jobs |

#### Example Configuration
```python
detector = await detection_service.create_detector(
    name="ECOD Detector",
    algorithm="ECOD",
    parameters={
        "contamination": 0.08
    }
)
```

---

### k-Nearest Neighbors (KNN)

**Type:** Distance-based  
**Library:** PyOD  
**Best for:** Simple baseline and comparison

#### Description
KNN detector uses the distance to the k-th nearest neighbor as the anomaly score.

#### Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `n_neighbors` | int | 5 | Number of neighbors |
| `method` | str | "largest" | Method for combining distances |
| `radius` | float | 1.0 | Range of parameter space |
| `algorithm` | str | "auto" | Algorithm for nearest neighbors |
| `contamination` | float | 0.1 | Expected anomaly proportion |

#### Example Configuration
```python
detector = await detection_service.create_detector(
    name="KNN Detector",
    algorithm="KNN",
    parameters={
        "n_neighbors": 10,
        "method": "mean",
        "contamination": 0.1
    }
)
```

---

### Angle-Based Outlier Detector (ABOD)

**Type:** Angle-based  
**Library:** PyOD  
**Best for:** High-dimensional data with angle-based anomalies

#### Description
ABOD considers the variance in angles between each data point and all other points.

#### Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `contamination` | float | 0.1 | Expected anomaly proportion |
| `n_neighbors` | int | 10 | Number of neighbors for angle computation |

---

## Algorithm Selection Guide

### By Data Characteristics

#### Small Datasets (<1,000 samples)
1. **LOF** - Good for local anomalies
2. **OCSVM** - Non-linear boundaries
3. **KNN** - Simple baseline

#### Medium Datasets (1,000-100,000 samples)
1. **Isolation Forest** - Fast and robust
2. **COPOD** - High-dimensional tabular data
3. **ECOD** - Mixed distributions

#### Large Datasets (>100,000 samples)
1. **Isolation Forest** - Best scalability
2. **ECOD** - Fast statistical approach
3. **COPOD** - If memory allows

### By Data Type

#### Numerical Features Only
1. **Isolation Forest** - First choice
2. **LOF** - For local anomalies
3. **OCSVM** - Non-linear cases

#### Mixed Data Types
1. **COPOD** - Handles mixed types well
2. **Custom preprocessing** + any algorithm
3. **Ensemble** of specialized detectors

#### High-Dimensional Data
1. **COPOD** - Designed for high dimensions
2. **Isolation Forest** - Handles dimensions well
3. **ABOD** - Angle-based approach

#### Categorical Features
1. **Preprocessing** required for most algorithms
2. **Custom encoding** + Isolation Forest
3. **Domain-specific** algorithms

### By Anomaly Type

#### Global Anomalies
- **Isolation Forest**
- **OCSVM**
- **ECOD**

#### Local Anomalies
- **LOF**
- **KNN**
- **ABOD**

#### Collective Anomalies
- **Time series** specific algorithms
- **Sequential** pattern detectors
- **Custom ensemble** methods

## Performance Tuning

### Contamination Rate
The most important parameter for most algorithms.

```python
# Conservative estimate (fewer false positives)
contamination = 0.05  # 5%

# Liberal estimate (catch more anomalies)
contamination = 0.15  # 15%

# Data-driven approach
contamination = estimate_contamination(training_data)
```

### Memory Optimization

#### For Large Datasets
```python
# Use sampling for memory-intensive algorithms
detector_params = {
    "contamination": 0.1,
    "max_samples": min(1000, len(data)),  # Limit samples
    "n_jobs": -1  # Use all cores
}
```

#### For High-Dimensional Data
```python
# Reduce dimensionality first
from sklearn.decomposition import PCA

pca = PCA(n_components=50)
reduced_data = pca.fit_transform(data)
```

### Speed Optimization

#### Parallel Processing
```python
# Most algorithms support parallel processing
detector_params = {
    "n_jobs": -1,  # Use all available cores
    "random_state": 42  # For reproducibility
}
```

#### Batch Processing
```python
# Process data in batches for memory efficiency
async def detect_large_dataset(detector_id, data, batch_size=1000):
    results = []
    for i in range(0, len(data), batch_size):
        batch = data[i:i+batch_size]
        batch_results = await detection_service.detect_batch(detector_id, batch)
        results.extend(batch_results)
    return results
```

## Ensemble Methods

### Voting Strategies

#### Majority Voting
```python
ensemble_config = {
    "algorithms": ["IsolationForest", "LOF", "COPOD"],
    "voting_strategy": "majority",
    "weights": [1.0, 1.0, 1.0]  # Equal weights
}
```

#### Weighted Voting
```python
ensemble_config = {
    "algorithms": ["IsolationForest", "LOF", "COPOD"],
    "voting_strategy": "weighted",
    "weights": [0.5, 0.3, 0.2]  # Based on performance
}
```

#### Unanimous Voting
```python
ensemble_config = {
    "algorithms": ["IsolationForest", "LOF", "OCSVM"],
    "voting_strategy": "unanimous",  # All must agree
    "threshold": 0.8  # Minimum score threshold
}
```

## Evaluation Metrics

### Performance Metrics

#### Precision
Proportion of detected anomalies that are actually anomalous.
```python
precision = true_positives / (true_positives + false_positives)
```

#### Recall (Sensitivity)
Proportion of actual anomalies that were detected.
```python
recall = true_positives / (true_positives + false_negatives)
```

#### F1 Score
Harmonic mean of precision and recall.
```python
f1_score = 2 * (precision * recall) / (precision + recall)
```

#### ROC AUC
Area under the Receiver Operating Characteristic curve.

### Algorithm-Specific Evaluation

```python
async def evaluate_algorithm(algorithm, data, true_labels):
    detector = await create_detector(algorithm=algorithm)
    await train_detector(detector, data)
    predictions = await detect_anomalies(detector, data)
    
    metrics = {
        "precision": calculate_precision(predictions, true_labels),
        "recall": calculate_recall(predictions, true_labels),
        "f1_score": calculate_f1(predictions, true_labels),
        "roc_auc": calculate_roc_auc(predictions, true_labels)
    }
    
    return metrics
```

## Best Practices

### 1. Start Simple
Begin with Isolation Forest for initial prototyping:
```python
detector = await detection_service.create_detector(
    name="Baseline Detector",
    algorithm="IsolationForest",
    parameters={"contamination": 0.1}
)
```

### 2. Validate Parameters
Always validate contamination rate:
```python
if contamination <= 0 or contamination >= 0.5:
    raise ValueError("Contamination must be between 0 and 0.5")
```

### 3. Use Cross-Validation
Evaluate performance with multiple splits:
```python
from sklearn.model_selection import KFold

async def cross_validate_detector(algorithm, data, labels, k=5):
    kf = KFold(n_splits=k, shuffle=True, random_state=42)
    scores = []
    
    for train_idx, test_idx in kf.split(data):
        train_data = data[train_idx]
        test_data = data[test_idx]
        test_labels = labels[test_idx]
        
        detector = await create_detector(algorithm=algorithm)
        await train_detector(detector, train_data)
        predictions = await detect_anomalies(detector, test_data)
        
        score = calculate_f1(predictions, test_labels)
        scores.append(score)
    
    return np.mean(scores), np.std(scores)
```

### 4. Monitor Performance
Track algorithm performance over time:
```python
@dataclass
class PerformanceMetrics:
    timestamp: datetime
    algorithm: str
    precision: float
    recall: float
    f1_score: float
    execution_time: float
    
async def log_performance(detector_id, metrics):
    await metrics_repository.save(PerformanceMetrics(
        timestamp=datetime.now(),
        algorithm=detector.algorithm,
        **metrics
    ))
```

### 5. Handle Concept Drift
Retrain models when performance degrades:
```python
async def check_and_retrain(detector_id, current_performance, threshold=0.1):
    historical_performance = await get_historical_performance(detector_id)
    
    if current_performance < historical_performance - threshold:
        logger.warning(f"Performance drift detected for {detector_id}")
        await retrain_detector(detector_id)
```

## Troubleshooting

### Common Issues

#### Poor Performance
1. **Check contamination rate** - Too high/low affects all algorithms
2. **Validate data quality** - Missing values, outliers in training data
3. **Feature engineering** - Scale/normalize features appropriately
4. **Algorithm choice** - May not be suitable for your data type

#### Memory Issues
1. **Reduce data size** - Use sampling for training
2. **Feature selection** - Remove irrelevant features
3. **Streaming approach** - Process data in batches
4. **Algorithm choice** - Use memory-efficient algorithms

#### Slow Training
1. **Parallel processing** - Set `n_jobs=-1`
2. **Reduce parameters** - Lower `n_estimators`, `n_neighbors`
3. **Sampling** - Train on subset of data
4. **Algorithm choice** - Use faster algorithms for large data

### Error Messages and Solutions

#### "Contamination rate too high"
```python
# Solution: Reduce contamination rate
parameters["contamination"] = min(0.4, parameters["contamination"])
```

#### "Insufficient memory"
```python
# Solution: Process in batches
batch_size = min(1000, len(data) // 4)
results = await process_in_batches(data, batch_size)
```

#### "Algorithm not found"
```python
# Solution: Check algorithm registry
available_algorithms = get_available_algorithms()
if algorithm not in available_algorithms:
    raise ValueError(f"Algorithm {algorithm} not available")
```