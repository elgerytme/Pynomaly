# Algorithm Comparison

🍞 **Breadcrumb:** 🏠 [Home](../../index.md) > 📖 [Reference](../README.md) > 🧮 [Algorithms](README.md) > ⚖️ Algorithm Comparison

---


## Overview

This comprehensive comparison guide helps you select the optimal anomaly detection algorithm based on your specific requirements. It provides detailed performance metrics, use case recommendations, and decision frameworks.

## Quick Selection Guide

### By Primary Concern

| **Priority** | **Recommended Algorithms** | **Rationale** |
|--------------|---------------------------|---------------|
| **Speed** | HBOS, ECOD, EllipticEnvelope | O(n) or O(n log n) complexity |
| **Accuracy** | Ensemble, AutoEncoder, IsolationForest | State-of-the-art performance |
| **Scalability** | IsolationForest, HBOS, ECOD | Handle millions of samples |
| **Interpretability** | LOF, Statistical methods, Causal AD | Clear explanation of anomalies |
| **Memory Efficiency** | Statistical methods, HBOS | Low memory footprint |

### By Data Characteristics

| **Data Type** | **Size** | **First Choice** | **Alternatives** | **Avoid** |
|---------------|----------|------------------|------------------|-----------|
| **Tabular** | Small | LOF | EllipticEnvelope, ABOD | Deep Learning |
| **Tabular** | Large | IsolationForest | HBOS, ECOD | LOF, OneClassSVM |
| **Time Series** | Any | LSTM AutoEncoder | Matrix Profile, Prophet | Statistical methods |
| **High-Dimensional** | Any | AutoEncoder | COPOD, IsolationForest | LOF, KNN |
| **Text** | Any | TF-IDF+Clustering | Word Embeddings | Traditional ML |
| **Images** | Any | CNN AutoEncoder | VAE | Statistical methods |
| **Graphs** | Any | DOMINANT | GCNAE | Traditional methods |

---

## Comprehensive Performance Matrix

### Computational Complexity

| Algorithm | Training Time | Prediction Time | Memory Usage | Scalability Rating |
|-----------|---------------|-----------------|--------------|-------------------|
| **Statistical Methods** |
| Z-Score | O(n) | O(1) | Very Low | ⭐⭐⭐⭐⭐ |
| IQR | O(n) | O(1) | Very Low | ⭐⭐⭐⭐⭐ |
| EllipticEnvelope | O(n×d²) | O(d²) | Low | ⭐⭐⭐⭐ |
| **PyOD Algorithms** |
| HBOS | O(n) | O(1) | Low | ⭐⭐⭐⭐⭐ |
| ECOD | O(n log n) | O(log n) | Low | ⭐⭐⭐⭐⭐ |
| COPOD | O(n log n) | O(log n) | Medium | ⭐⭐⭐⭐ |
| **Scikit-learn** |
| IsolationForest | O(n log n) | O(log n) | Medium | ⭐⭐⭐⭐⭐ |
| LOF | O(n²) | O(k×n) | High | ⭐⭐ |
| OneClassSVM | O(n²-n³) | O(sv×d) | High | ⭐ |
| **Distance-based** |
| KNN | O(1) | O(n) | High | ⭐⭐ |
| ABOD | O(n³) | O(n²) | Very High | ⭐ |
| **Deep Learning** |
| AutoEncoder | O(epochs×n) | O(1) | High | ⭐⭐⭐ |
| VAE | O(epochs×n) | O(1) | High | ⭐⭐⭐ |
| LSTM | O(epochs×seq×n) | O(seq) | Very High | ⭐⭐ |
| **Ensemble** |
| Voting | Sum(base algorithms) | Sum(base algorithms) | High | ⭐⭐⭐ |

### Accuracy by Dataset Type

#### Tabular Data Performance

| Algorithm | Small Data (<1K) | Medium Data (1K-100K) | Large Data (>100K) | High-Dim (>100 features) |
|-----------|------------------|----------------------|-------------------|-------------------------|
| IsolationForest | ⭐⭐⭐ | ⭐⭐⭐⭐⭐ | ⭐⭐⭐⭐⭐ | ⭐⭐⭐⭐ |
| LOF | ⭐⭐⭐⭐⭐ | ⭐⭐⭐ | ⭐ | ⭐⭐ |
| OneClassSVM | ⭐⭐⭐⭐ | ⭐⭐⭐ | ⭐ | ⭐⭐⭐ |
| AutoEncoder | ⭐⭐ | ⭐⭐⭐⭐ | ⭐⭐⭐⭐⭐ | ⭐⭐⭐⭐⭐ |
| COPOD | ⭐⭐⭐ | ⭐⭐⭐⭐ | ⭐⭐⭐⭐ | ⭐⭐⭐⭐⭐ |
| HBOS | ⭐⭐ | ⭐⭐⭐ | ⭐⭐⭐⭐ | ⭐⭐ |
| Ensemble | ⭐⭐⭐⭐ | ⭐⭐⭐⭐⭐ | ⭐⭐⭐⭐⭐ | ⭐⭐⭐⭐⭐ |

#### Time Series Performance

| Algorithm | Trend Anomalies | Seasonal Anomalies | Point Anomalies | Pattern Anomalies |
|-----------|-----------------|-------------------|-----------------|-------------------|
| LSTM AutoEncoder | ⭐⭐⭐⭐⭐ | ⭐⭐⭐⭐⭐ | ⭐⭐⭐ | ⭐⭐⭐⭐⭐ |
| Matrix Profile | ⭐⭐ | ⭐⭐ | ⭐⭐⭐ | ⭐⭐⭐⭐⭐ |
| Prophet | ⭐⭐⭐⭐⭐ | ⭐⭐⭐⭐⭐ | ⭐⭐⭐ | ⭐⭐ |
| ARIMA | ⭐⭐⭐⭐ | ⭐⭐⭐⭐ | ⭐⭐⭐⭐ | ⭐⭐ |
| IsolationForest | ⭐⭐ | ⭐ | ⭐⭐⭐⭐ | ⭐⭐ |

#### Graph Data Performance

| Algorithm | Node Anomalies | Edge Anomalies | Community Anomalies | Attribute Anomalies |
|-----------|----------------|----------------|-------------------|-------------------|
| DOMINANT | ⭐⭐⭐⭐⭐ | ⭐⭐⭐ | ⭐⭐⭐⭐ | ⭐⭐⭐⭐⭐ |
| GCNAE | ⭐⭐⭐⭐ | ⭐⭐ | ⭐⭐⭐⭐⭐ | ⭐⭐⭐ |
| Graph Isolation | ⭐⭐⭐ | ⭐⭐⭐⭐ | ⭐⭐ | ⭐⭐ |

---

## Resource Requirements Comparison

### Memory Usage Analysis

| **Memory Tier** | **Algorithms** | **Typical Usage** | **Max Dataset Size** |
|-----------------|----------------|-------------------|---------------------|
| **Very Low** | Z-Score, IQR, HBOS | <100 MB | Unlimited |
| **Low** | EllipticEnvelope, ECOD | 100-500 MB | 1M+ samples |
| **Medium** | IsolationForest, COPOD | 500 MB - 2 GB | 500K samples |
| **High** | LOF, KNN, AutoEncoder | 2-8 GB | 100K samples |
| **Very High** | OneClassSVM, ABOD, LSTM | 8+ GB | 10K samples |

### GPU Requirements

| **GPU Tier** | **Algorithms** | **Minimum VRAM** | **Recommended VRAM** |
|---------------|----------------|-------------------|---------------------|
| **Not Required** | All statistical, LOF, KNN | - | - |
| **Optional** | AutoEncoder (small) | - | 4 GB |
| **Recommended** | AutoEncoder (large), VAE | 4 GB | 8 GB |
| **Required** | Transformer, CNN, Large LSTM | 8 GB | 16+ GB |

### Processing Time Benchmarks

*Based on 100K samples, 50 features, Intel i7-10700K, 32GB RAM*

| Algorithm | Training Time | Prediction Time (1K samples) | Total Time |
|-----------|---------------|------------------------------|------------|
| HBOS | 0.1s | 0.001s | 0.101s |
| ECOD | 0.3s | 0.002s | 0.302s |
| IsolationForest | 2.1s | 0.005s | 2.105s |
| COPOD | 1.8s | 0.003s | 1.803s |
| LOF | 45.2s | 12.3s | 57.5s |
| OneClassSVM | 128.4s | 8.7s | 137.1s |
| AutoEncoder | 120.0s | 0.1s | 120.1s |

---

## Feature Support Matrix

### Data Type Support

| Algorithm | Numerical | Categorical | Mixed | Text | Images | Graphs | Time Series |
|-----------|-----------|-------------|-------|------|--------|--------|-------------|
| IsolationForest | ✅ | ⚠️ | ⚠️ | ❌ | ❌ | ❌ | ⚠️ |
| LOF | ✅ | ❌ | ❌ | ❌ | ❌ | ❌ | ⚠️ |
| HBOS | ✅ | ✅ | ✅ | ❌ | ❌ | ❌ | ⚠️ |
| AutoEncoder | ✅ | ⚠️ | ⚠️ | ⚠️ | ✅ | ❌ | ✅ |
| CNN AutoEncoder | ❌ | ❌ | ❌ | ❌ | ✅ | ❌ | ⚠️ |
| LSTM | ❌ | ❌ | ❌ | ⚠️ | ❌ | ❌ | ✅ |
| DOMINANT | ❌ | ❌ | ❌ | ❌ | ❌ | ✅ | ❌ |
| TF-IDF | ❌ | ❌ | ❌ | ✅ | ❌ | ❌ | ❌ |

**Legend**: ✅ Native support, ⚠️ Requires preprocessing, ❌ Not supported

### Advanced Features

| Algorithm | Streaming | Online Learning | Incremental | GPU Acceleration | Distributed |
|-----------|-----------|-----------------|-------------|------------------|-------------|
| IsolationForest | ⚠️ | ❌ | ⚠️ | ❌ | ⚠️ |
| LOF | ❌ | ❌ | ❌ | ❌ | ❌ |
| AutoEncoder | ⚠️ | ✅ | ✅ | ✅ | ✅ |
| HBOS | ✅ | ✅ | ✅ | ❌ | ⚠️ |
| LSTM | ⚠️ | ✅ | ✅ | ✅ | ✅ |
| Ensemble | Depends on base | Depends on base | Depends on base | Depends on base | ✅ |

---

## Use Case Recommendations

### By Industry/Domain

#### **Financial Services**
| Use Case | Primary Algorithm | Backup Algorithm | Special Considerations |
|----------|------------------|------------------|----------------------|
| Credit Card Fraud | IsolationForest | Ensemble | Real-time requirements |
| Market Manipulation | LSTM AutoEncoder | Time Series Ensemble | Sequential patterns |
| Anti-Money Laundering | Graph Neural Networks | Network Analysis | Graph structure critical |
| Risk Assessment | Ensemble | AutoEncoder | High accuracy needed |

#### **Cybersecurity**
| Use Case | Primary Algorithm | Backup Algorithm | Special Considerations |
|----------|------------------|------------------|----------------------|
| Network Intrusion | LSTM | IsolationForest | Sequential patterns |
| Log Analysis | TF-IDF + Clustering | Ensemble | Text processing |
| User Behavior | LOF | OneClassSVM | Local anomalies |
| Malware Detection | CNN AutoEncoder | Deep Learning | Binary/executable analysis |

#### **Manufacturing**
| Use Case | Primary Algorithm | Backup Algorithm | Special Considerations |
|----------|------------------|------------------|----------------------|
| Quality Control | Statistical + ML Ensemble | IsolationForest | Fast decisions needed |
| Predictive Maintenance | Time Series LSTM | Prophet | Temporal dependencies |
| Supply Chain | Graph Analysis | Network Anomalies | Relationship modeling |
| Process Monitoring | Real-time Statistical | Streaming Algorithms | Continuous operation |

#### **Healthcare**
| Use Case | Primary Algorithm | Backup Algorithm | Special Considerations |
|----------|------------------|------------------|----------------------|
| Medical Imaging | CNN AutoEncoder | Vision Transformer | Image data |
| Patient Monitoring | Time Series Analysis | LSTM | Continuous signals |
| Drug Discovery | Graph Neural Networks | Molecular Analysis | Chemical structures |
| Electronic Health Records | Mixed Data Ensemble | Multiple Data Types | Privacy requirements |

#### **E-commerce**
| Use Case | Primary Algorithm | Backup Algorithm | Special Considerations |
|----------|------------------|------------------|----------------------|
| Recommendation Systems | Collaborative Filtering + AD | Matrix Factorization | User-item interactions |
| Price Monitoring | Time Series | Prophet + LSTM | Market dynamics |
| Review Analysis | Text Analytics | NLP + Anomaly Detection | Sentiment analysis |
| Inventory Management | Forecasting + AD | Multi-variate Time Series | Demand patterns |

---

## Decision Framework

### Step-by-Step Selection Process

#### 1. **Data Assessment**
```python
def assess_data_characteristics(data):
    characteristics = {
        "size": len(data),
        "features": data.shape[1] if len(data.shape) > 1 else 1,
        "data_types": analyze_data_types(data),
        "missing_values": data.isnull().sum().sum(),
        "temporal": is_temporal_data(data),
        "sparsity": calculate_sparsity(data)
    }
    return characteristics
```

#### 2. **Requirement Analysis**
```python
def analyze_requirements():
    requirements = {
        "performance": {
            "speed_priority": "high/medium/low",
            "accuracy_priority": "high/medium/low", 
            "memory_constraints": "strict/moderate/flexible"
        },
        "operational": {
            "real_time": True/False,
            "batch_processing": True/False,
            "interpretability": "required/preferred/optional"
        },
        "infrastructure": {
            "gpu_available": True/False,
            "distributed_compute": True/False,
            "cloud_deployment": True/False
        }
    }
    return requirements
```

#### 3. **Algorithm Filtering**
```python
def filter_algorithms(data_characteristics, requirements):
    candidate_algorithms = []
    
    # Filter by data size
    if data_characteristics["size"] > 100000:
        candidate_algorithms = ["IsolationForest", "HBOS", "ECOD"]
    elif data_characteristics["size"] < 1000:
        candidate_algorithms = ["LOF", "EllipticEnvelope", "ABOD"]
    else:
        candidate_algorithms = ["IsolationForest", "LOF", "OneClassSVM"]
    
    # Filter by speed requirements
    if requirements["performance"]["speed_priority"] == "high":
        candidate_algorithms = [alg for alg in candidate_algorithms 
                              if alg in ["HBOS", "ECOD", "EllipticEnvelope"]]
    
    # Filter by interpretability requirements
    if requirements["operational"]["interpretability"] == "required":
        candidate_algorithms = [alg for alg in candidate_algorithms 
                              if alg in ["LOF", "EllipticEnvelope", "Statistical"]]
    
    return candidate_algorithms
```

#### 4. **Final Selection**
```python
def select_final_algorithm(candidates, data_characteristics, requirements):
    scores = {}
    
    for algorithm in candidates:
        score = 0
        
        # Score based on data fit
        if is_good_fit(algorithm, data_characteristics):
            score += 40
        
        # Score based on requirements match
        if meets_requirements(algorithm, requirements):
            score += 40
        
        # Score based on proven performance
        score += get_benchmark_score(algorithm, data_characteristics) * 20
        
        scores[algorithm] = score
    
    return max(scores, key=scores.get)
```

### Decision Tree

```
Data Size?
├── Small (<1K)
│   ├── Need Interpretability? → LOF, Statistical Methods
│   ├── High Accuracy? → OneClassSVM, Ensemble
│   └── Fast? → EllipticEnvelope, Z-Score
├── Medium (1K-100K) 
│   ├── High Dimensional? → AutoEncoder, COPOD
│   ├── Mixed Data Types? → HBOS, Ensemble
│   └── General Purpose? → IsolationForest
└── Large (>100K)
    ├── Real-time? → HBOS, ECOD
    ├── Complex Patterns? → AutoEncoder, Deep Learning
    └── Memory Constrained? → Streaming Algorithms
```

---

## Performance Tuning Guidelines

### General Hyperparameter Optimization

#### Grid Search Strategy
```python
# Conservative grid search for production
conservative_grids = {
    "IsolationForest": {
        "n_estimators": [100, 200],
        "contamination": [0.05, 0.1, 0.15],
        "max_features": [0.8, 1.0]
    },
    "LOF": {
        "n_neighbors": [10, 20, 30],
        "contamination": [0.05, 0.1, 0.15]
    },
    "AutoEncoder": {
        "hidden_neurons": [[64, 32, 64], [128, 64, 128]],
        "epochs": [100, 200],
        "learning_rate": [0.001, 0.0005]
    }
}

# Aggressive grid search for research
aggressive_grids = {
    "IsolationForest": {
        "n_estimators": [50, 100, 200, 500],
        "contamination": [0.01, 0.05, 0.1, 0.15, 0.2],
        "max_features": [0.5, 0.7, 0.8, 1.0],
        "max_samples": ["auto", 0.5, 0.8]
    }
}
```

#### Bayesian Optimization
```python
from skopt import gp_minimize
from skopt.space import Real, Integer, Categorical

def bayesian_optimization_example():
    space = [
        Integer(50, 500, name='n_estimators'),
        Real(0.01, 0.3, name='contamination'),
        Real(0.1, 1.0, name='max_features'),
        Categorical(['auto', 0.5, 0.8], name='max_samples')
    ]
    
    def objective(params):
        model = IsolationForest(
            n_estimators=params[0],
            contamination=params[1], 
            max_features=params[2],
            max_samples=params[3]
        )
        return -cross_val_score(model, X, y, cv=5).mean()
    
    result = gp_minimize(objective, space, n_calls=50)
    return result.x
```

### Algorithm-Specific Tuning

#### IsolationForest Optimization
```python
def tune_isolation_forest(X, y):
    # Start with contamination rate
    contamination_scores = []
    for contamination in [0.05, 0.1, 0.15, 0.2]:
        model = IsolationForest(contamination=contamination)
        score = cross_val_score(model, X, y, cv=5).mean()
        contamination_scores.append((contamination, score))
    
    best_contamination = max(contamination_scores, key=lambda x: x[1])[0]
    
    # Then tune n_estimators
    n_estimators_scores = []
    for n_est in [50, 100, 200, 500]:
        model = IsolationForest(
            contamination=best_contamination,
            n_estimators=n_est
        )
        score = cross_val_score(model, X, y, cv=5).mean()
        n_estimators_scores.append((n_est, score))
    
    best_n_estimators = max(n_estimators_scores, key=lambda x: x[1])[0]
    
    return {
        "contamination": best_contamination,
        "n_estimators": best_n_estimators
    }
```

#### Deep Learning Tuning
```python
def tune_autoencoder(X, y):
    import optuna
    
    def objective(trial):
        # Architecture
        n_layers = trial.suggest_int('n_layers', 2, 5)
        layer_sizes = []
        current_size = X.shape[1]
        
        for i in range(n_layers):
            layer_size = trial.suggest_int(f'layer_{i}', 8, 256)
            layer_sizes.append(layer_size)
            current_size = layer_size
        
        # Symmetric decoder
        decoder_sizes = layer_sizes[:-1][::-1] + [X.shape[1]]
        
        # Training parameters
        learning_rate = trial.suggest_float('lr', 1e-5, 1e-2, log=True)
        batch_size = trial.suggest_categorical('batch_size', [16, 32, 64, 128])
        dropout = trial.suggest_float('dropout', 0.0, 0.5)
        
        model = AutoEncoder(
            encoder_layers=layer_sizes,
            decoder_layers=decoder_sizes,
            learning_rate=learning_rate,
            batch_size=batch_size,
            dropout=dropout
        )
        
        score = evaluate_autoencoder(model, X, y)
        return score
    
    study = optuna.create_study(direction='maximize')
    study.optimize(objective, n_trials=100)
    
    return study.best_params
```

---

## Ensemble Strategies

### Smart Ensemble Construction

#### Diversity-Based Selection
```python
def build_diverse_ensemble(algorithms, X, y, max_algorithms=5):
    """Build ensemble focusing on algorithm diversity"""
    
    # Calculate prediction diversity matrix
    predictions = {}
    for alg in algorithms:
        model = create_model(alg)
        model.fit(X)
        predictions[alg] = model.predict(X)
    
    # Calculate pairwise disagreement
    disagreement_matrix = {}
    for alg1 in algorithms:
        for alg2 in algorithms:
            if alg1 != alg2:
                disagreement = calculate_disagreement(
                    predictions[alg1], 
                    predictions[alg2]
                )
                disagreement_matrix[(alg1, alg2)] = disagreement
    
    # Select diverse subset
    selected = [algorithms[0]]  # Start with first algorithm
    
    while len(selected) < max_algorithms and len(selected) < len(algorithms):
        best_candidate = None
        best_avg_disagreement = 0
        
        for candidate in algorithms:
            if candidate not in selected:
                avg_disagreement = np.mean([
                    disagreement_matrix.get((candidate, selected_alg), 0)
                    for selected_alg in selected
                ])
                
                if avg_disagreement > best_avg_disagreement:
                    best_avg_disagreement = avg_disagreement
                    best_candidate = candidate
        
        if best_candidate:
            selected.append(best_candidate)
        else:
            break
    
    return selected
```

#### Performance-Based Weighting
```python
def calculate_performance_weights(algorithms, X, y, cv=5):
    """Calculate weights based on cross-validation performance"""
    
    performances = {}
    
    for alg in algorithms:
        model = create_model(alg)
        scores = cross_val_score(model, X, y, cv=cv, scoring='f1')
        performances[alg] = {
            'mean': scores.mean(),
            'std': scores.std(),
            'scores': scores
        }
    
    # Calculate weights using softmax of mean scores
    mean_scores = np.array([performances[alg]['mean'] for alg in algorithms])
    weights = softmax(mean_scores)
    
    return dict(zip(algorithms, weights))

def softmax(x, temperature=1.0):
    """Compute softmax values with temperature"""
    exp_x = np.exp((x - np.max(x)) / temperature)
    return exp_x / exp_x.sum()
```

---

## Validation and Evaluation

### Comprehensive Evaluation Framework

#### Multi-Metric Evaluation
```python
def comprehensive_evaluation(model, X_test, y_true):
    """Comprehensive anomaly detection evaluation"""
    
    # Get predictions and scores
    y_pred = model.predict(X_test)
    y_scores = model.decision_function(X_test)
    
    metrics = {}
    
    # Classification metrics
    metrics['precision'] = precision_score(y_true, y_pred)
    metrics['recall'] = recall_score(y_true, y_pred)
    metrics['f1'] = f1_score(y_true, y_pred)
    metrics['accuracy'] = accuracy_score(y_true, y_pred)
    
    # Ranking metrics
    metrics['roc_auc'] = roc_auc_score(y_true, y_scores)
    metrics['pr_auc'] = average_precision_score(y_true, y_scores)
    
    # At-k metrics
    for k in [10, 50, 100]:
        metrics[f'precision_at_{k}'] = precision_at_k(y_true, y_scores, k)
        metrics[f'recall_at_{k}'] = recall_at_k(y_true, y_scores, k)
    
    return metrics

def precision_at_k(y_true, y_scores, k):
    """Precision at k highest scores"""
    top_k_indices = np.argsort(y_scores)[-k:]
    top_k_true = y_true[top_k_indices]
    return top_k_true.sum() / k

def recall_at_k(y_true, y_scores, k):
    """Recall at k highest scores"""
    top_k_indices = np.argsort(y_scores)[-k:]
    top_k_true = y_true[top_k_indices]
    total_anomalies = y_true.sum()
    return top_k_true.sum() / total_anomalies if total_anomalies > 0 else 0
```

#### Cross-Validation Strategies
```python
def time_series_cv_evaluation(model, X, y, n_splits=5):
    """Time series specific cross-validation"""
    from sklearn.model_selection import TimeSeriesSplit
    
    tscv = TimeSeriesSplit(n_splits=n_splits)
    scores = []
    
    for train_idx, test_idx in tscv.split(X):
        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]
        
        model.fit(X_train)
        score = comprehensive_evaluation(model, X_test, y_test)
        scores.append(score)
    
    # Aggregate scores
    aggregated = {}
    for metric in scores[0].keys():
        values = [score[metric] for score in scores]
        aggregated[metric] = {
            'mean': np.mean(values),
            'std': np.std(values),
            'min': np.min(values),
            'max': np.max(values)
        }
    
    return aggregated

def stratified_cv_evaluation(model, X, y, contamination_ratio=0.1, n_splits=5):
    """Stratified cross-validation maintaining contamination ratio"""
    from sklearn.model_selection import StratifiedKFold
    
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
    scores = []
    
    for train_idx, test_idx in skf.split(X, y):
        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]
        
        # Ensure contamination ratio in training set
        train_contamination = y_train.sum() / len(y_train)
        if abs(train_contamination - contamination_ratio) > 0.02:
            # Adjust training set to match expected contamination
            X_train, y_train = adjust_contamination_ratio(
                X_train, y_train, contamination_ratio
            )
        
        model.fit(X_train)
        score = comprehensive_evaluation(model, X_test, y_test)
        scores.append(score)
    
    return aggregate_cv_scores(scores)
```

---

## Production Deployment Considerations

### Model Selection for Production

#### Latency Requirements
```python
def select_for_latency(algorithms, max_latency_ms=100):
    """Select algorithms meeting latency requirements"""
    
    latency_benchmarks = {
        "HBOS": 1,           # Very fast
        "ECOD": 2,           # Very fast  
        "EllipticEnvelope": 3, # Fast
        "IsolationForest": 15, # Medium
        "COPOD": 20,         # Medium
        "KNN": 50,           # Slow
        "LOF": 200,          # Very slow
        "OneClassSVM": 500,  # Very slow
        "AutoEncoder": 25,   # Medium (post-training)
    }
    
    suitable_algorithms = [
        alg for alg in algorithms 
        if latency_benchmarks.get(alg, float('inf')) <= max_latency_ms
    ]
    
    return suitable_algorithms
```

#### Throughput Requirements
```python
def select_for_throughput(algorithms, min_throughput_rps=1000):
    """Select algorithms meeting throughput requirements"""
    
    throughput_benchmarks = {
        "HBOS": 10000,        # Very high
        "ECOD": 8000,         # Very high
        "EllipticEnvelope": 5000, # High
        "IsolationForest": 2000,  # Medium-high
        "COPOD": 1500,        # Medium
        "AutoEncoder": 1000,  # Medium
        "LOF": 100,           # Low
        "OneClassSVM": 50,    # Very low
    }
    
    suitable_algorithms = [
        alg for alg in algorithms
        if throughput_benchmarks.get(alg, 0) >= min_throughput_rps
    ]
    
    return suitable_algorithms
```

#### Resource Constraints
```python
def select_for_resources(algorithms, max_memory_mb=1000, gpu_available=False):
    """Select algorithms fitting resource constraints"""
    
    memory_requirements = {
        "Z-Score": 10,
        "HBOS": 50,
        "ECOD": 100,
        "EllipticEnvelope": 200,
        "IsolationForest": 500,
        "COPOD": 800,
        "LOF": 2000,
        "OneClassSVM": 1500,
        "AutoEncoder": 1000,  # Without GPU
        "LSTM": 3000,         # Without GPU
    }
    
    gpu_algorithms = ["AutoEncoder", "VAE", "LSTM", "CNN", "Transformer"]
    
    suitable_algorithms = []
    
    for alg in algorithms:
        # Check memory constraint
        if memory_requirements.get(alg, float('inf')) <= max_memory_mb:
            # Check GPU requirement
            if alg in gpu_algorithms and not gpu_available:
                continue
            suitable_algorithms.append(alg)
    
    return suitable_algorithms
```

---

## Related Documentation

- **[Core Algorithms](core-algorithms.md)** - Essential algorithms for most use cases
- **[Specialized Algorithms](specialized-algorithms.md)** - Domain-specific algorithms  
- **[Experimental Algorithms](experimental-algorithms.md)** - Advanced research methods
- **[Autonomous Mode Guide](../../user-guides/basic-usage/autonomous-mode.md)** - Automated selection
- **[Performance Analysis](../../user-guides/advanced-features/performance.md)** - Performance monitoring and optimization
- **[AutoML Guide](../../user-guides/advanced-features/automl-and-intelligence.md)** - Automated hyperparameter tuning
