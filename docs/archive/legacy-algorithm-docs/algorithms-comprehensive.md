# Comprehensive Algorithm Reference

🍞 **Breadcrumb:** 🏠 [Home](../index.md) > 📁 Archive

---


## Overview

Pynomaly provides access to 100+ anomaly detection algorithms across multiple frameworks and libraries, offering the most comprehensive anomaly detection toolkit available. This guide covers all supported algorithms, their characteristics, use cases, and optimal configurations.

## 🧩 Algorithm Categories

### Statistical Methods
**Best for**: Well-understood data distributions, baseline detection, interpretable results

### Machine Learning Methods  
**Best for**: Complex patterns, non-linear relationships, high-dimensional data

### Deep Learning Methods
**Best for**: Complex data types, large datasets, representation learning

### Graph-based Methods
**Best for**: Network data, relationship analysis, social networks

### Time Series Methods
**Best for**: Temporal data, seasonal patterns, trend analysis

---

## 📊 Framework Coverage

| Framework | Algorithms | Strengths | Use Cases |
|-----------|------------|-----------|-----------|
| **scikit-learn** | 8 algorithms | Fast, reliable, interpretable | General purpose, baselines |
| **PyOD** | 40+ algorithms | Comprehensive, standardized | Research, comparison studies |
| **TODS** | 20+ algorithms | Time series focus, automated | Temporal anomaly detection |
| **PyGOD** | 15+ algorithms | Graph neural networks | Network analysis, social data |
| **PyTorch** | 10+ algorithms | Deep learning, GPU support | Large datasets, custom models |
| **TensorFlow** | 10+ algorithms | Production deployment, serving | Enterprise applications |
| **JAX** | 8+ algorithms | High performance, research | Scientific computing, research |

---

## 🔬 Scikit-learn Algorithms

### IsolationForest
**Type**: Tree-based ensemble  
**Complexity**: O(n log n)  
**Memory**: Low  

```python
from pynomaly import create_detector

detector = create_detector(
    algorithm="IsolationForest",
    parameters={
        "n_estimators": 100,
        "contamination": 0.1,
        "max_samples": "auto",
        "random_state": 42
    }
)
```

**Best for**:
- High-dimensional data
- Mixed data types
- Large datasets
- Baseline comparisons

**Limitations**:
- Assumes uniform contamination
- Less effective with very high dimensions
- Performance degrades with categorical features

### LocalOutlierFactor (LOF)
**Type**: Density-based  
**Complexity**: O(n²)  
**Memory**: High  

```python
detector = create_detector(
    algorithm="LocalOutlierFactor",
    parameters={
        "n_neighbors": 20,
        "contamination": 0.1,
        "metric": "minkowski",
        "novelty": True
    }
)
```

**Best for**:
- Local anomalies
- Varying densities
- Small to medium datasets
- Cluster-based anomalies

**Limitations**:
- Computationally expensive
- Sensitive to parameter selection
- Poor scalability

### OneClassSVM
**Type**: Support vector machine  
**Complexity**: O(n²) to O(n³)  
**Memory**: Medium  

```python
detector = create_detector(
    algorithm="OneClassSVM",
    parameters={
        "kernel": "rbf",
        "gamma": "scale",
        "nu": 0.1
    }
)
```

**Best for**:
- Non-linear boundaries
- Robust outlier detection
- Medium datasets
- Complex decision boundaries

**Limitations**:
- Expensive training
- Kernel selection critical
- Poor interpretability

### EllipticEnvelope
**Type**: Statistical/covariance-based  
**Complexity**: O(n)  
**Memory**: Low  

```python
detector = create_detector(
    algorithm="EllipticEnvelope",
    parameters={
        "contamination": 0.1,
        "support_fraction": None,
        "store_precision": True
    }
)
```

**Best for**:
- Gaussian distributions
- Low-dimensional data
- Fast detection
- Interpretable results

**Limitations**:
- Assumes Gaussian distribution
- Poor with non-linear patterns
- Limited to continuous features

---

## 🎯 PyOD Algorithms

### Angle-Based Outlier Detection (ABOD)
**Type**: Geometric  
**Complexity**: O(n³)  
**Memory**: High  

```python
detector = create_detector(
    algorithm="ABOD",
    library="pyod",
    parameters={
        "contamination": 0.1,
        "n_neighbors": 10
    }
)
```

**Best for**:
- High-dimensional data
- Geometric outliers
- Research applications
- Comprehensive analysis

### Cluster-Based Local Outlier Factor (CBLOF)
**Type**: Clustering-based  
**Complexity**: O(n²)  
**Memory**: Medium  

```python
detector = create_detector(
    algorithm="CBLOF",
    library="pyod",
    parameters={
        "n_clusters": 8,
        "contamination": 0.1,
        "clustering_estimator": None,
        "alpha": 0.9,
        "beta": 5
    }
)
```

**Best for**:
- Cluster-based anomalies
- Structured data
- Mixed anomaly types
- Medium datasets

### Histogram-Based Outlier Score (HBOS)
**Type**: Statistical/histogram-based  
**Complexity**: O(n)  
**Memory**: Low  

```python
detector = create_detector(
    algorithm="HBOS",
    library="pyod",
    parameters={
        "n_bins": 10,
        "alpha": 0.1,
        "tol": 0.5
    }
)
```

**Best for**:
- Fast detection
- Categorical features
- Large datasets
- Real-time applications

### k-Nearest Neighbors (KNN)
**Type**: Distance-based  
**Complexity**: O(n²)  
**Memory**: High  

```python
detector = create_detector(
    algorithm="KNN",
    library="pyod",
    parameters={
        "n_neighbors": 5,
        "method": "largest",
        "radius": 1.0,
        "metric": "minkowski"
    }
)
```

**Best for**:
- Local anomalies
- Variable densities
- Small to medium datasets
- Interpretable results

### Adversarially Learned Anomaly Detection (ALAD)
**Type**: Deep learning/GAN  
**Complexity**: O(n)  
**Memory**: High  

```python
detector = create_detector(
    algorithm="ALAD",
    library="pyod",
    parameters={
        "epochs": 500,
        "batch_size": 32,
        "lr_d": 0.0001,
        "lr_g": 0.0001
    }
)
```

**Best for**:
- High-dimensional data
- Complex patterns
- Large datasets
- Research applications

---

## 📈 Time Series Algorithms (TODS)

### Matrix Profile
**Type**: Pattern matching  
**Complexity**: O(n²)  
**Memory**: Medium  

```python
detector = create_detector(
    algorithm="MatrixProfile",
    library="tods",
    parameters={
        "window_size": 50,
        "normalize": True
    }
)
```

**Best for**:
- Motif discovery
- Pattern anomalies
- Univariate time series
- Sequence matching

### LSTM Autoencoder
**Type**: Deep learning/RNN  
**Complexity**: O(n)  
**Memory**: High  

```python
detector = create_detector(
    algorithm="LSTMAutoencoder",
    library="tods",
    parameters={
        "epochs": 100,
        "batch_size": 32,
        "sequence_length": 50,
        "hidden_dim": 64
    }
)
```

**Best for**:
- Sequential patterns
- Long-term dependencies
- Multivariate time series
- Complex temporal relationships

### Telemanom
**Type**: Deep learning/specialized  
**Complexity**: O(n)  
**Memory**: High  

```python
detector = create_detector(
    algorithm="Telemanom",
    library="tods",
    parameters={
        "l_s": 250,
        "n_predictions": 10,
        "batch_size": 70
    }
)
```

**Best for**:
- Spacecraft telemetry
- Multi-sensor data
- NASA-validated approach
- High-reliability systems

---

## 🕸️ Graph Neural Network Algorithms (PyGOD)

### DOMINANT
**Type**: Graph autoencoder  
**Complexity**: O(|E|)  
**Memory**: Medium  

```python
detector = create_detector(
    algorithm="DOMINANT",
    library="pygod",
    parameters={
        "hid_dim": 64,
        "num_layers": 4,
        "dropout": 0.3,
        "weight_decay": 0.0
    }
)
```

**Best for**:
- Attribute anomalies
- Structure anomalies
- Social networks
- Citation networks

### GraphConvolutional Autoencoder (GCNAE)
**Type**: Convolutional autoencoder  
**Complexity**: O(|E|)  
**Memory**: Medium  

```python
detector = create_detector(
    algorithm="GCNAE",
    library="pygod",
    parameters={
        "hid_dim": 64,
        "num_layers": 2,
        "dropout": 0.2,
        "act": "relu"
    }
)
```

**Best for**:
- Node anomalies
- Graph reconstruction
- Network analysis
- Community detection

---

## 🧠 Deep Learning Algorithms

### PyTorch Algorithms

#### Autoencoder
**Type**: Neural network/reconstruction  
**Complexity**: O(n)  
**Memory**: High  

```python
detector = create_detector(
    algorithm="AutoEncoder",
    library="pytorch",
    parameters={
        "hidden_neurons": [64, 32, 64],
        "epochs": 100,
        "batch_size": 32,
        "lr": 0.001,
        "weight_decay": 1e-5
    }
)
```

**Best for**:
- High-dimensional data
- Non-linear patterns
- Feature learning
- Reconstruction errors

#### Variational Autoencoder (VAE)
**Type**: Probabilistic/generative  
**Complexity**: O(n)  
**Memory**: High  

```python
detector = create_detector(
    algorithm="VAE",
    library="pytorch",
    parameters={
        "encoder_neurons": [64, 32],
        "decoder_neurons": [32, 64],
        "latent_dim": 16,
        "beta": 1.0
    }
)
```

**Best for**:
- Probabilistic modeling
- Generative anomalies
- Uncertainty quantification
- Complex distributions

#### Deep SVDD
**Type**: Deep one-class classification  
**Complexity**: O(n)  
**Memory**: High  

```python
detector = create_detector(
    algorithm="DeepSVDD",
    library="pytorch",
    parameters={
        "c": None,
        "nu": 0.1,
        "rep_dim": 32,
        "dropout": 0.2
    }
)
```

**Best for**:
- One-class classification
- Deep feature learning
- Complex decision boundaries
- High-dimensional data

### TensorFlow Algorithms

#### Deep Autoencoding Gaussian Mixture Model (DAGMM)
**Type**: Mixture model/deep learning  
**Complexity**: O(n)  
**Memory**: High  

```python
detector = create_detector(
    algorithm="DAGMM",
    library="tensorflow",
    parameters={
        "comp_hiddens": [60, 30, 10, 1],
        "comp_activation": "tanh",
        "est_hiddens": [10, 4],
        "est_activation": "tanh",
        "est_dropout_ratio": 0.5
    }
)
```

**Best for**:
- Gaussian mixture modeling
- Density estimation
- Complex distributions
- Production deployment

---

## 🚀 JAX Algorithms

### JAX Autoencoder
**Type**: Functional programming/autodiff  
**Complexity**: O(n)  
**Memory**: High  

```python
detector = create_detector(
    algorithm="JAXAutoencoder",
    library="jax",
    parameters={
        "hidden_dims": [64, 32, 16, 32, 64],
        "learning_rate": 0.001,
        "num_epochs": 100,
        "batch_size": 32
    }
)
```

**Best for**:
- High-performance computing
- Research applications
- Scientific computing
- JIT compilation benefits

---

## 🎛️ Algorithm Selection Guide

### By Dataset Size

| Dataset Size | Recommended Algorithms | Reasoning |
|--------------|----------------------|-----------|
| Small (< 1K) | LOF, EllipticEnvelope, ABOD | Thorough analysis possible |
| Medium (1K-100K) | IsolationForest, OneClassSVM, CBLOF | Balanced performance |
| Large (100K-1M) | HBOS, IsolationForest, KNN | Scalable algorithms |
| Very Large (> 1M) | HBOS, MinHash, Neural networks | High-performance methods |

### By Data Type

| Data Type | Recommended Algorithms | Considerations |
|-----------|----------------------|----------------|
| **Tabular** | IsolationForest, LOF, HBOS | General purpose |
| **Time Series** | Matrix Profile, LSTM, Telemanom | Temporal patterns |
| **Text** | Autoencoder, VAE, TF-IDF + LOF | NLP preprocessing |
| **Images** | CNN Autoencoder, VAE | Spatial features |
| **Graphs** | DOMINANT, GCNAE, SCAN | Network structure |
| **Mixed** | Autoencoder, DAGMM | Flexible representation |

### By Anomaly Type

| Anomaly Type | Recommended Algorithms | Examples |
|--------------|----------------------|----------|
| **Point** | IsolationForest, LOF, KNN | Individual outliers |
| **Contextual** | LSTM, Matrix Profile | Time-dependent anomalies |
| **Collective** | Matrix Profile, DOMINANT | Group anomalies |
| **Density** | LOF, HBOS, CBLOF | Sparse regions |
| **Pattern** | Matrix Profile, Autoencoder | Sequence anomalies |

### By Performance Requirements

| Requirement | Recommended Algorithms | Trade-offs |
|-------------|----------------------|------------|
| **Fast Training** | HBOS, EllipticEnvelope | Lower accuracy |
| **Fast Inference** | HBOS, KNN, Linear models | Memory vs speed |
| **High Accuracy** | Deep learning, Ensemble | Computational cost |
| **Interpretable** | Statistical methods, Trees | Complexity limits |
| **Scalable** | MinHash, Streaming algorithms | Approximation errors |

---

## 🔧 Parameter Tuning Guidelines

### General Principles

1. **Contamination Rate**: Start with domain knowledge or use AutoML
2. **Algorithm-Specific**: Follow algorithm documentation
3. **Cross-Validation**: Use temporal splits for time series
4. **Ensemble**: Combine multiple algorithms for robustness

### Common Parameters

#### contamination
```python
# Conservative approach
contamination = 0.05  # 5% anomalies

# Balanced approach  
contamination = 0.1   # 10% anomalies

# Aggressive approach
contamination = 0.2   # 20% anomalies
```

#### n_neighbors (for distance-based algorithms)
```python
# Small datasets
n_neighbors = min(10, len(data) // 10)

# Large datasets
n_neighbors = min(50, int(np.sqrt(len(data))))
```

#### Neural Network Architecture
```python
# Start simple
hidden_neurons = [input_dim // 2, input_dim // 4, input_dim // 2]

# Deep networks for complex data
hidden_neurons = [256, 128, 64, 32, 64, 128, 256]
```

---

## 🎯 Performance Comparison

### Computational Complexity

| Algorithm | Training | Inference | Memory | Scalability |
|-----------|----------|-----------|---------|-------------|
| HBOS | O(n) | O(1) | Low | Excellent |
| IsolationForest | O(n log n) | O(log n) | Medium | Good |
| LOF | O(n²) | O(k) | High | Poor |
| OneClassSVM | O(n³) | O(sv) | Medium | Poor |
| Autoencoder | O(epochs×n) | O(1) | High | Good |
| LSTM | O(epochs×n×T) | O(T) | High | Medium |

### Accuracy Benchmarks

*Note: Results vary significantly by dataset and parameters*

| Algorithm | Tabular Data | Time Series | High-Dim | Overall |
|-----------|--------------|-------------|----------|---------|
| IsolationForest | ⭐⭐⭐⭐ | ⭐⭐⭐ | ⭐⭐⭐⭐⭐ | ⭐⭐⭐⭐ |
| LOF | ⭐⭐⭐⭐⭐ | ⭐⭐ | ⭐⭐ | ⭐⭐⭐ |
| Autoencoder | ⭐⭐⭐⭐ | ⭐⭐⭐⭐ | ⭐⭐⭐⭐⭐ | ⭐⭐⭐⭐ |
| LSTM | ⭐⭐ | ⭐⭐⭐⭐⭐ | ⭐⭐⭐ | ⭐⭐⭐ |
| HBOS | ⭐⭐⭐ | ⭐⭐ | ⭐⭐ | ⭐⭐ |

---

## 🔄 Algorithm Migration Guide

### Upgrading from Simple to Advanced

```python
# Basic detection
detector = create_detector("IsolationForest")

# Advanced detection with ensemble
detector = create_ensemble([
    create_detector("IsolationForest"),
    create_detector("LOF"),
    create_detector("AutoEncoder", library="pytorch")
])

# AutoML-driven selection
detector = create_automl_detector(
    algorithms=["IsolationForest", "LOF", "AutoEncoder"],
    optimization_metric="f1_score",
    cross_validation_folds=5
)
```

### Framework Migration

```python
# From scikit-learn to PyOD
sklearn_detector = create_detector("IsolationForest")
pyod_detector = create_detector("IForest", library="pyod")

# From traditional to deep learning
traditional = create_detector("LOF")
deep_learning = create_detector("AutoEncoder", library="pytorch")

# Ensemble approach
ensemble = create_ensemble([traditional, deep_learning])
```

---

## 📚 Further Reading

- [PyOD Documentation](https://pyod.readthedocs.io/)
- [TODS Documentation](https://tods-doc.github.io/)
- [PyGOD Documentation](https://pygod.readthedocs.io/)
- [Algorithm Comparison Studies](../examples/algorithm-comparison.md)
- [Performance Benchmarks](../examples/performance-benchmarking.md)

---

## 🤝 Contributing New Algorithms

See our [Plugin Development Guide](../development/plugin-development.md) for adding custom algorithms to the Pynomaly ecosystem.
