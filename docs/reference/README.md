# Pynomaly Reference Documentation

🍞 **Breadcrumb:** 🏠 [Home](../index.md) > 📚 Reference

---

## 🎯 Reference Overview

This section provides comprehensive technical reference information for Pynomaly users and developers.

## 📋 Quick Navigation

### 🧠 **Algorithm Reference**
- **[Algorithm Comparison](algorithm-comparison.md)** - Detailed comparison of all algorithms
- **[Classifier Selection Guide](CLASSIFIER_SELECTION_GUIDE.md)** - How to choose the right algorithm
- **[Algorithm Details](algorithms/)** - Deep dive into specific algorithms

### 🔌 **API Reference**
- **[PWA API Reference](api/pwa-api-reference.md)** - Progressive Web App API documentation
- **[REST API Reference](api/rest-api-reference.md)** - Complete REST API documentation
- **[Python SDK Reference](api/python-sdk-reference.md)** - Python SDK documentation

### ⚙️ **Configuration Reference**
- **[Configuration Guide](configuration/README.md)** - Complete configuration options
- **[Environment Variables](configuration/environment-variables.md)** - Environment setup
- **[Performance Tuning](configuration/performance-tuning.md)** - Optimization settings

### 📊 **Data Reference**
- **[Dataset Formats](data/dataset-formats.md)** - Supported data formats
- **[Feature Engineering](data/feature-engineering.md)** - Data preprocessing
- **[Export Formats](data/export-formats.md)** - Output format options

## 🔍 Algorithm Categories

### **Linear Models**
- **Principal Component Analysis (PCA)** - Dimensionality reduction based detection
- **Minimum Covariance Determinant (MCD)** - Robust covariance estimation
- **One-Class SVM** - Support vector based boundary detection

### **Proximity-Based**
- **Local Outlier Factor (LOF)** - Local density based detection
- **k-Nearest Neighbors (k-NN)** - Distance based anomaly scoring
- **Connectivity-Based Outlier Factor (COF)** - Connectivity analysis

### **Probabilistic**
- **Gaussian Mixture Models (GMM)** - Probabilistic clustering
- **Copula-Based Outlier Detection (COPOD)** - Dependency modeling
- **Histogram-Based Outlier Score (HBOS)** - Distribution analysis

### **Ensemble Methods**
- **Isolation Forest** - Tree-based ensemble detection
- **Feature Bagging** - Bootstrap aggregating for outliers
- **Locally Selective Combination (LSCP)** - Dynamic ensemble selection

### **Deep Learning**
- **AutoEncoder** - Neural network reconstruction error
- **Variational AutoEncoder (VAE)** - Probabilistic neural networks
- **Deep Support Vector Data Description (Deep SVDD)** - Deep learning boundaries

### **Graph-Based**
- **Graph Neural Networks (GNN)** - Network structure analysis
- **Random Walk** - Graph traversal based detection
- **Community Detection** - Structural anomaly identification

## 🚀 Quick Start References

### **Algorithm Selection**
```python
# For tabular data
from pynomaly.algorithms import IsolationForest, LOF, OneClassSVM

# For time series
from pynomaly.algorithms import TimeSeriesDetector

# For graph data
from pynomaly.algorithms import GraphDetector
```

### **Configuration Examples**
```yaml
# Basic configuration
detector:
  algorithm: "IsolationForest"
  contamination: 0.1
  n_estimators: 100

# Advanced configuration
detector:
  algorithm: "EnsembleDetector"
  base_detectors:
    - algorithm: "IsolationForest"
      weight: 0.4
    - algorithm: "LOF"
      weight: 0.6
```

## 📈 Performance Guidelines

### **Algorithm Performance Comparison**

| Algorithm | Time Complexity | Space Complexity | Best Use Case |
|-----------|----------------|------------------|---------------|
| Isolation Forest | O(n log n) | O(n) | General purpose, large datasets |
| LOF | O(n²) | O(n) | Local density variations |
| One-Class SVM | O(n²) | O(n) | Small datasets, non-linear boundaries |
| PCA | O(n×p²) | O(p²) | High-dimensional data |
| COPOD | O(n×p) | O(n×p) | Multivariate dependencies |

### **Scaling Recommendations**

- **Small datasets (< 1K samples)**: One-Class SVM, LOF
- **Medium datasets (1K - 100K samples)**: Isolation Forest, PCA
- **Large datasets (> 100K samples)**: COPOD, HBOS, Ensemble methods
- **Streaming data**: Online variants of LOF, adaptive algorithms

## 🔗 Related Documentation

### **User Guides**
- **[Basic Usage](../user-guides/basic-usage/)** - How to use algorithms
- **[Advanced Features](../user-guides/advanced-features/)** - Complex scenarios
- **[Performance Tuning](../user-guides/advanced-features/performance-tuning.md)** - Optimization

### **Developer Guides**
- **[Architecture](../developer-guides/architecture/)** - System design
- **[API Integration](../developer-guides/api-integration/)** - Integration patterns
- **[Contributing](../developer-guides/contributing/)** - Development guidelines

### **Examples**
- **[Algorithm Examples](../examples/)** - Real-world usage
- **[Industry Applications](../examples/banking/)** - Domain-specific guides
- **[Tutorials](../examples/tutorials/)** - Step-by-step guides

---

**Last Updated**: 2025-01-09  
**Next Review**: 2025-02-09