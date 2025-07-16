# Core Algorithms

🍞 **Breadcrumb:** 🏠 [Home](../../index.md) > 📖 [Reference](../README.md) > 🧮 [Algorithms](README.md) > 🔵 Core Algorithms

---


## Overview

This guide covers the essential anomaly detection algorithms that form the foundation of Pynomaly. These 20+ algorithms handle most common use cases and provide excellent starting points for anomaly detection projects.

## Quick Selection Guide

### By Data Size
- **Small (< 1K)**: LOF, EllipticEnvelope, ABOD
- **Medium (1K-100K)**: IsolationForest, OneClassSVM, CBLOF  
- **Large (> 100K)**: IsolationForest, HBOS, ECOD

### By Performance Priority
- **Speed**: HBOS, ECOD, EllipticEnvelope
- **Accuracy**: IsolationForest, AutoEncoder, Ensemble
- **Interpretability**: Statistical methods, LOF, Z-Score

---

## Statistical Methods

### 1. Isolation Forest

**Type**: Tree-based ensemble  
**Library**: scikit-learn  
**Complexity**: O(n log n)  
**Best for**: General-purpose, high-dimensional data, large datasets

#### Description
Isolation Forest isolates anomalies by randomly selecting features and split values. Anomalies are easier to isolate and require fewer splits, making this one of the most effective and scalable algorithms.

#### Algorithm Details
- Creates isolation trees using random feature splits
- Anomalies have shorter path lengths (easier isolation)
- No assumptions about data distribution
- Efficient for high-dimensional data

#### Parameters

| Parameter | Type | Default | Range | Description |
|-----------|------|---------|-------|-------------|
| `n_estimators` | int | 100 | 50-500 | Number of isolation trees |
| `max_samples` | int/float | "auto" | int or 0.0-1.0 | Samples to draw per tree |
| `contamination` | float | 0.1 | 0.0-0.5 | Expected anomaly proportion |
| `max_features` | int/float | 1.0 | int or 0.0-1.0 | Features to draw per tree |
| `bootstrap` | bool | False | - | Bootstrap sampling |
| `random_state` | int | None | - | Random seed for reproducibility |

#### Usage Example
```python
from pynomaly.application.services import DetectionService

# Create detector
detector = await detection_service.create_detector(
    name="Production Isolation Forest",
    algorithm="IsolationForest",
    parameters={
        "contamination": 0.05,
        "n_estimators": 200,
        "max_samples": 256,
        "random_state": 42
    }
)

# Train and detect
await detection_service.train_detector(detector.id, dataset.id)
results = await detection_service.detect_anomalies(detector.id, dataset.id)
```

#### Performance Characteristics
- **Training Time**: O(n log n) - Very fast
- **Prediction Time**: O(log n) - Real-time capable
- **Memory Usage**: Low to moderate
- **Scalability**: Excellent (handles millions of samples)

#### Strengths
- ✅ Fast training and prediction
- ✅ No assumptions about data distribution  
- ✅ Handles high-dimensional data well
- ✅ Built-in contamination estimation
- ✅ Memory efficient

#### Limitations
- ❌ May struggle with very sparse data
- ❌ Less effective with categorical features
- ❌ Can miss local anomalies in dense regions
- ❌ Performance degrades with too many irrelevant features

#### When to Use
- General-purpose anomaly detection
- Large datasets (>1000 samples)
- Real-time detection systems
- Baseline comparisons
- High-dimensional numerical data

---

### 2. Local Outlier Factor (LOF)

**Type**: Density-based  
**Library**: scikit-learn  
**Complexity**: O(n²)  
**Best for**: Local anomalies, varying density regions

#### Description
LOF computes the local density deviation of each data point with respect to its neighbors. Points with substantially lower density than their neighbors are considered anomalies.

#### Algorithm Details
- Calculates local density for each point based on k-nearest neighbors
- Compares point density to neighbor densities
- High LOF score indicates local anomaly
- Adapts to varying data densities

#### Parameters

| Parameter | Type | Default | Range | Description |
|-----------|------|---------|-------|-------------|
| `n_neighbors` | int | 20 | 5-50 | Number of neighbors for density estimation |
| `algorithm` | str | "auto" | "auto", "ball_tree", "kd_tree", "brute" | Neighbor search algorithm |
| `leaf_size` | int | 30 | 10-100 | Leaf size for tree algorithms |
| `metric` | str | "minkowski" | "euclidean", "manhattan", etc. | Distance metric |
| `p` | int | 2 | 1-5 | Parameter for Minkowski metric |
| `contamination` | float | 0.1 | 0.0-0.5 | Expected anomaly proportion |

#### Usage Example
```python
detector = await detection_service.create_detector(
    name="Local Density Detector",
    algorithm="LOF",
    parameters={
        "n_neighbors": 30,
        "contamination": 0.08,
        "metric": "euclidean",
        "algorithm": "ball_tree"
    }
)
```

#### Performance Characteristics
- **Training Time**: O(n²) - Expensive for large datasets
- **Prediction Time**: O(k×n) - Moderate
- **Memory Usage**: High (stores all training data)
- **Scalability**: Poor for large datasets (>10K samples)

#### Strengths
- ✅ Excellent for local anomalies
- ✅ Adapts to varying data densities
- ✅ Intuitive density-based approach
- ✅ No assumptions about global data distribution

#### Limitations
- ❌ Computationally expensive O(n²)
- ❌ Sensitive to parameter choice (k)
- ❌ Curse of dimensionality (>20 features)
- ❌ Memory intensive for large datasets

#### When to Use
- Data with clusters of different densities
- Need to detect local outliers
- Small to medium datasets (<10K samples)
- Exploratory data analysis

---

### 3. One-Class SVM (OCSVM)

**Type**: Support vector machine  
**Library**: scikit-learn  
**Complexity**: O(n²) to O(n³)  
**Best for**: Non-linear decision boundaries, robust detection

#### Description
One-Class SVM learns a decision function for novelty detection by mapping input data to a high-dimensional feature space and finding a hyperplane that separates normal data from the origin.

#### Algorithm Details
- Maps data to high-dimensional space using kernels
- Finds hyperplane separating normal data from origin
- Uses support vectors to define decision boundary
- Robust to outliers during training

#### Parameters

| Parameter | Type | Default | Range | Description |
|-----------|------|---------|-------|-------------|
| `kernel` | str | "rbf" | "linear", "poly", "rbf", "sigmoid" | Kernel type |
| `degree` | int | 3 | 2-5 | Degree for polynomial kernel |
| `gamma` | str/float | "scale" | "scale", "auto", 0.001-1.0 | Kernel coefficient |
| `coef0` | float | 0.0 | 0.0-1.0 | Independent term for poly/sigmoid |
| `tol` | float | 1e-3 | 1e-5 to 1e-1 | Tolerance for stopping criterion |
| `nu` | float | 0.5 | 0.01-0.99 | Upper bound on training errors |

#### Usage Example
```python
detector = await detection_service.create_detector(
    name="Robust SVM Detector",
    algorithm="OneClassSVM",
    parameters={
        "kernel": "rbf",
        "gamma": "auto",
        "nu": 0.1,
        "tol": 1e-4
    }
)
```

#### Performance Characteristics
- **Training Time**: O(n²) to O(n³) - Very expensive
- **Prediction Time**: O(sv×d) - Depends on support vectors
- **Memory Usage**: High
- **Scalability**: Poor for large datasets (>5K samples)

#### Strengths
- ✅ Handles complex non-linear boundaries
- ✅ Robust to outliers in training data
- ✅ Works well with high-dimensional data
- ✅ Strong theoretical foundation

#### Limitations
- ❌ Very slow training on large datasets
- ❌ Sensitive to parameter tuning (gamma, nu)
- ❌ Difficult to interpret results
- ❌ Memory intensive

#### When to Use
- Complex non-linear decision boundaries needed
- Robust anomaly detection required
- Small to medium datasets (<5K samples)
- High-dimensional data with complex patterns

---

### 4. Elliptic Envelope

**Type**: Statistical/Gaussian  
**Library**: scikit-learn  
**Complexity**: O(n×d²)  
**Best for**: Gaussian data, fast detection, interpretable results

#### Description
Assumes data follows a multivariate Gaussian distribution and detects outliers using robust covariance estimation and Mahalanobis distance.

#### Parameters

| Parameter | Type | Default | Range | Description |
|-----------|------|---------|-------|-------------|
| `contamination` | float | 0.1 | 0.0-0.5 | Expected anomaly proportion |
| `support_fraction` | float | None | 0.5-1.0 | Proportion of points for covariance estimation |
| `store_precision` | bool | True | - | Store precision matrix |
| `assume_centered` | bool | False | - | Assume data is centered |

#### Usage Example
```python
detector = await detection_service.create_detector(
    name="Gaussian Detector",
    algorithm="EllipticEnvelope",
    parameters={
        "contamination": 0.05,
        "support_fraction": 0.8,
        "store_precision": True
    }
)
```

#### Strengths
- ✅ Very fast computation O(n×d²)
- ✅ Strong statistical foundation
- ✅ Highly interpretable results
- ✅ Good baseline for Gaussian data

#### Limitations
- ❌ Assumes multivariate Gaussian distribution
- ❌ Poor performance on non-Gaussian data
- ❌ Limited to elliptical decision boundaries
- ❌ Sensitive to high dimensions

---

## PyOD Advanced Algorithms

### 5. COPOD (Copula-Based Outlier Detection)

**Type**: Statistical  
**Library**: PyOD  
**Complexity**: O(n log n)  
**Best for**: High-dimensional tabular data, mixed data types

#### Description
COPOD uses copula functions to model the joint distribution of features and identifies anomalies based on their probability under this model.

#### Parameters

| Parameter | Type | Default | Range | Description |
|-----------|------|---------|-------|-------------|
| `contamination` | float | 0.1 | 0.0-0.5 | Expected anomaly proportion |
| `n_jobs` | int | 1 | 1 to -1 | Number of parallel jobs |

#### Usage Example
```python
detector = await detection_service.create_detector(
    name="COPOD Detector",
    algorithm="COPOD",
    library="pyod",
    parameters={
        "contamination": 0.05,
        "n_jobs": -1
    }
)
```

#### Performance Characteristics
- **Training Time**: O(n log n) - Fast
- **Prediction Time**: O(log n) - Very fast
- **Memory Usage**: Medium
- **Scalability**: Good (handles 100K+ samples)

#### Strengths
- ✅ Fast and scalable
- ✅ Handles mixed data types well
- ✅ No hyperparameter tuning required
- ✅ Provides probabilistic anomaly scores

#### Limitations
- ❌ Assumes feature independence
- ❌ May miss complex feature interactions
- ❌ Less effective on low-dimensional data

---

### 6. ECOD (Empirical Cumulative Distribution)

**Type**: Statistical  
**Library**: PyOD  
**Complexity**: O(n log n)  
**Best for**: Large-scale datasets, mixed distributions

#### Description
ECOD uses empirical cumulative distribution functions to model each feature independently and combines them to detect anomalies.

#### Parameters

| Parameter | Type | Default | Range | Description |
|-----------|------|---------|-------|-------------|
| `contamination` | float | 0.1 | 0.0-0.5 | Expected anomaly proportion |
| `n_jobs` | int | 1 | 1 to -1 | Number of parallel jobs |

#### Usage Example
```python
detector = await detection_service.create_detector(
    name="ECOD Fast Detector",
    algorithm="ECOD",
    library="pyod",
    parameters={
        "contamination": 0.08,
        "n_jobs": -1
    }
)
```

#### Strengths
- ✅ Extremely fast O(n log n)
- ✅ Handles large datasets efficiently
- ✅ No distribution assumptions
- ✅ Memory efficient

#### Limitations
- ❌ Assumes feature independence
- ❌ May miss multivariate patterns
- ❌ Simple univariate approach

---

### 7. k-Nearest Neighbors (KNN)

**Type**: Distance-based  
**Library**: PyOD  
**Complexity**: O(n²)  
**Best for**: Local anomalies, baseline comparisons

#### Description
KNN detector uses the distance to the k-th nearest neighbor as the anomaly score.

#### Parameters

| Parameter | Type | Default | Range | Description |
|-----------|------|---------|-------|-------------|
| `n_neighbors` | int | 5 | 3-20 | Number of neighbors |
| `method` | str | "largest" | "largest", "mean", "median" | Distance aggregation method |
| `radius` | float | 1.0 | 0.1-10.0 | Range parameter |
| `algorithm` | str | "auto" | "auto", "ball_tree", "kd_tree", "brute" | Nearest neighbor algorithm |
| `contamination` | float | 0.1 | 0.0-0.5 | Expected anomaly proportion |

#### Usage Example
```python
detector = await detection_service.create_detector(
    name="KNN Detector",
    algorithm="KNN",
    library="pyod",
    parameters={
        "n_neighbors": 10,
        "method": "mean",
        "contamination": 0.1
    }
)
```

#### Strengths
- ✅ Simple and intuitive
- ✅ Non-parametric approach
- ✅ Good for local anomalies
- ✅ No training phase required

#### Limitations
- ❌ Computationally expensive O(n²)
- ❌ Memory intensive
- ❌ Sensitive to dimensionality
- ❌ Sensitive to irrelevant features

---

### 8. HBOS (Histogram-Based Outlier Score)

**Type**: Statistical/histogram-based  
**Library**: PyOD  
**Complexity**: O(n)  
**Best for**: Fast detection, categorical features, large datasets

#### Description
HBOS builds histograms for each feature and calculates anomaly scores based on the inverse of histogram densities.

#### Parameters

| Parameter | Type | Default | Range | Description |
|-----------|------|---------|-------|-------------|
| `n_bins` | int | 10 | 5-100 | Number of histogram bins |
| `alpha` | float | 0.1 | 0.01-0.5 | Regularization parameter |
| `tol` | float | 0.5 | 0.1-1.0 | Tolerance for sparse bins |
| `contamination` | float | 0.1 | 0.0-0.5 | Expected anomaly proportion |

#### Usage Example
```python
detector = await detection_service.create_detector(
    name="Fast HBOS Detector",
    algorithm="HBOS",
    library="pyod",
    parameters={
        "n_bins": 15,
        "alpha": 0.1,
        "contamination": 0.05
    }
)
```

#### Strengths
- ✅ Extremely fast O(n)
- ✅ Handles categorical features well
- ✅ Good for large datasets
- ✅ Simple and interpretable

#### Limitations
- ❌ Assumes feature independence
- ❌ Poor for continuous features
- ❌ May miss multivariate patterns
- ❌ Sensitive to bin size selection

---

## Machine Learning Methods

### 9. AutoEncoder (Neural Network)

**Type**: Deep learning/reconstruction  
**Library**: PyTorch/TensorFlow  
**Complexity**: O(epochs×n)  
**Best for**: High-dimensional data, feature learning, complex patterns

#### Description
Neural network that learns to compress and reconstruct data. Anomaly score is based on reconstruction error - anomalies are harder to reconstruct accurately.

#### Parameters

| Parameter | Type | Default | Range | Description |
|-----------|------|---------|-------|-------------|
| `hidden_neurons` | list | [64, 32, 64] | Various architectures | Hidden layer sizes |
| `epochs` | int | 100 | 50-500 | Training epochs |
| `batch_size` | int | 32 | 16-128 | Training batch size |
| `learning_rate` | float | 0.001 | 0.0001-0.01 | Learning rate |
| `dropout_rate` | float | 0.2 | 0.0-0.5 | Dropout regularization |
| `contamination` | float | 0.1 | 0.0-0.5 | Expected anomaly proportion |

#### Usage Example
```python
detector = await detection_service.create_detector(
    name="Neural AutoEncoder",
    algorithm="AutoEncoder",
    library="pytorch",
    parameters={
        "hidden_neurons": [128, 64, 32, 64, 128],
        "epochs": 200,
        "batch_size": 64,
        "learning_rate": 0.001,
        "dropout_rate": 0.3
    }
)
```

#### Performance Characteristics
- **Training Time**: O(epochs×n) - Moderate to slow
- **Prediction Time**: O(1) - Very fast
- **Memory Usage**: High (GPU recommended)
- **Scalability**: Good with proper batching

#### Strengths
- ✅ Learns complex non-linear patterns
- ✅ Excellent for high-dimensional data
- ✅ Automatic feature learning
- ✅ Flexible architecture

#### Limitations
- ❌ Requires larger datasets (>1K samples)
- ❌ Many hyperparameters to tune
- ❌ Black box (poor interpretability)
- ❌ GPU recommended for large datasets

---

## Ensemble Methods

### 10. Simple Ensemble (Voting)

**Type**: Meta-algorithm  
**Complexity**: Sum of base algorithms  
**Best for**: Maximum accuracy, robust detection

#### Description
Combines predictions from multiple algorithms using voting strategies (majority, weighted, or unanimous).

#### Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `algorithms` | list | ["IsolationForest", "LOF"] | Base algorithms to ensemble |
| `voting_strategy` | str | "majority" | "majority", "weighted", "unanimous" |
| `weights` | list | None | Algorithm weights (for weighted voting) |
| `contamination` | float | 0.1 | Expected anomaly proportion |

#### Usage Example
```python
# Create ensemble detector
ensemble_config = {
    "algorithms": ["IsolationForest", "LOF", "COPOD"],
    "voting_strategy": "weighted",
    "weights": [0.5, 0.3, 0.2],
    "contamination": 0.1
}

detector = await detection_service.create_ensemble_detector(
    name="Robust Ensemble",
    config=ensemble_config
)
```

#### Strengths
- ✅ Higher accuracy than individual algorithms
- ✅ More robust to different data types
- ✅ Reduces false positives
- ✅ Combines different detection strategies

#### Limitations
- ❌ Increased computational cost
- ❌ More complex parameter tuning
- ❌ Harder to interpret
- ❌ May mask individual algorithm insights

---

## Performance Comparison Matrix

### Computational Complexity

| Algorithm | Training | Prediction | Memory | Scalability | Speed Rating |
|-----------|----------|------------|---------|-------------|--------------|
| HBOS | O(n) | O(1) | Low | Excellent | ⭐⭐⭐⭐⭐ |
| ECOD | O(n log n) | O(log n) | Low | Excellent | ⭐⭐⭐⭐⭐ |
| COPOD | O(n log n) | O(log n) | Medium | Good | ⭐⭐⭐⭐ |
| IsolationForest | O(n log n) | O(log n) | Medium | Excellent | ⭐⭐⭐⭐ |
| EllipticEnvelope | O(n×d²) | O(d²) | Low | Good | ⭐⭐⭐⭐ |
| KNN | O(1) | O(n) | High | Poor | ⭐⭐ |
| LOF | O(n²) | O(k×n) | High | Poor | ⭐⭐ |
| OneClassSVM | O(n³) | O(sv×d) | High | Poor | ⭐ |
| AutoEncoder | O(epochs×n) | O(1) | High | Good | ⭐⭐⭐ |

### Accuracy by Data Type

| Algorithm | Tabular | High-Dim | Small Data | Large Data | Overall |
|-----------|---------|----------|------------|------------|---------|
| IsolationForest | ⭐⭐⭐⭐ | ⭐⭐⭐⭐⭐ | ⭐⭐⭐ | ⭐⭐⭐⭐⭐ | ⭐⭐⭐⭐ |
| LOF | ⭐⭐⭐⭐⭐ | ⭐⭐ | ⭐⭐⭐⭐⭐ | ⭐ | ⭐⭐⭐ |
| AutoEncoder | ⭐⭐⭐⭐ | ⭐⭐⭐⭐⭐ | ⭐⭐ | ⭐⭐⭐⭐ | ⭐⭐⭐⭐ |
| COPOD | ⭐⭐⭐⭐ | ⭐⭐⭐⭐ | ⭐⭐⭐ | ⭐⭐⭐⭐ | ⭐⭐⭐⭐ |
| OneClassSVM | ⭐⭐⭐⭐ | ⭐⭐⭐ | ⭐⭐⭐⭐ | ⭐ | ⭐⭐⭐ |
| HBOS | ⭐⭐⭐ | ⭐⭐ | ⭐⭐ | ⭐⭐⭐⭐ | ⭐⭐ |

## Best Practices

### 1. Start Simple
Begin with IsolationForest for initial prototyping:
```python
# Quick baseline
detector = await detection_service.create_detector(
    name="Baseline",
    algorithm="IsolationForest",
    parameters={"contamination": 0.1}
)
```

### 2. Parameter Validation
Always validate contamination rate:
```python
if contamination <= 0 or contamination >= 0.5:
    raise ValueError("Contamination must be between 0 and 0.5")
```

### 3. Cross-Validation
Evaluate performance with multiple splits:
```python
# Use temporal splits for time series data
from sklearn.model_selection import TimeSeriesSplit, KFold

# Regular data
kf = KFold(n_splits=5, shuffle=True, random_state=42)

# Time series data  
tscv = TimeSeriesSplit(n_splits=5)
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
1. **Check contamination rate** - Most critical parameter
2. **Validate data quality** - Remove/impute missing values
3. **Feature engineering** - Scale/normalize features
4. **Algorithm choice** - May not suit your data type

#### Memory Issues
1. **Use sampling** - Train on subset for memory-intensive algorithms
2. **Feature selection** - Remove irrelevant features
3. **Batch processing** - Process large datasets in chunks
4. **Algorithm choice** - Switch to memory-efficient algorithms

#### Slow Training
1. **Parallel processing** - Set `n_jobs=-1`
2. **Reduce parameters** - Lower `n_estimators`, `n_neighbors`
3. **Sampling** - Train on subset of data
4. **Algorithm choice** - Use faster algorithms for large data

### Algorithm-Specific Tips

#### IsolationForest
- Increase `n_estimators` for stability (100-500)
- Use `max_samples` < 1.0 for very large datasets
- Consider `max_features` < 1.0 for high-dimensional data

#### LOF
- Start with `n_neighbors` = 20
- Increase for smoother boundaries, decrease for local patterns
- Use `algorithm="ball_tree"` for high dimensions

#### AutoEncoder
- Normalize input data (mean=0, std=1)
- Use learning rate scheduling
- Apply early stopping to prevent overfitting
- Start with simple architectures

---

## Related Documentation

- **[Specialized Algorithms](specialized-algorithms.md)** - Time series, graph, text algorithms
- **[Experimental Algorithms](experimental-algorithms.md)** - Advanced deep learning methods
- **[Algorithm Comparison](algorithm-comparison.md)** - Detailed performance analysis
- **[Autonomous Mode Guide](../../comprehensive/09-autonomous-classifier-selection-guide.md)** - Automatic selection
