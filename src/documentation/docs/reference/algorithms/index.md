# Algorithm Reference

🍞 **Breadcrumb:** 🏠 [Home](../../index.md) > 📚 [Reference](../index.md) > 🧠 Algorithms

---

## 🎯 Algorithm Reference Overview

This section provides comprehensive technical reference for all anomaly detection algorithms available in Pynomaly. Whether you're selecting algorithms for your use case or understanding algorithm internals, this reference will help you make informed decisions.

---

## 📋 Quick Navigation

### 🎯 **Algorithm Categories**
- **[Core Algorithms →](core-algorithms.md)** - Essential algorithms for most use cases
- **[Specialized Algorithms →](specialized-algorithms.md)** - Domain-specific algorithms
- **[Experimental Algorithms →](experimental-algorithms.md)** - Cutting-edge research algorithms

### 📊 **Algorithm Comparison**
- **[Algorithm Comparison →](algorithm-comparison.md)** - Detailed comparison of all algorithms
- **[Algorithm README →](README.md)** - Algorithm overview and selection guide

---

## 🧠 Algorithm Categories

### **🎯 Core Algorithms**
Essential algorithms for most use cases:
- **[Core Algorithms Guide](core-algorithms.md)** - 20+ core algorithms
- **Linear Models** - PCA, MCD, One-Class SVM
- **Proximity-Based** - LOF, k-NN, COF
- **Probabilistic** - GMM, COPOD, HBOS
- **Ensemble Methods** - Isolation Forest, Feature Bagging

### **🔬 Specialized Algorithms**
Domain-specific implementations:
- **[Specialized Algorithms Guide](specialized-algorithms.md)** - Specialized methods
- **Time Series** - Time series anomaly detection
- **Graph-Based** - Network and graph anomalies
- **Text & NLP** - Text anomaly detection
- **Image Processing** - Visual anomaly detection

### **🚀 Experimental Algorithms**
Cutting-edge research algorithms:
- **[Experimental Algorithms Guide](experimental-algorithms.md)** - Research methods
- **Deep Learning** - AutoEncoders, VAEs, GANs
- **Advanced Ensemble** - Dynamic selection methods
- **Streaming** - Online learning algorithms
- **Federated** - Distributed anomaly detection

---

## 📊 Algorithm Selection Guide

### **🎯 By Use Case**
Choose algorithms based on your scenario:

#### **📈 Tabular Data**
- **Small datasets (< 1K)**: One-Class SVM, LOF
- **Medium datasets (1K-100K)**: Isolation Forest, PCA
- **Large datasets (> 100K)**: COPOD, HBOS

#### **⏱️ Time Series**
- **Univariate**: ARIMA-based, Prophet
- **Multivariate**: VAR, DeepAR
- **Streaming**: Online algorithms

#### **🕸️ Graph Data**
- **Social Networks**: Community detection
- **Knowledge Graphs**: Graph neural networks
- **Infrastructure**: Network anomaly detection

### **⚡ By Performance Requirements**
Algorithm selection by performance needs:

#### **🚀 Real-Time (< 100ms)**
- **COPOD** - O(n×p) complexity
- **HBOS** - Histogram-based
- **Simple Ensemble** - Lightweight combinations

#### **🔄 Batch Processing (< 1 hour)**
- **Isolation Forest** - Scalable tree ensemble
- **PCA** - Linear dimensionality reduction
- **LOF** - Local outlier factor

#### **🧠 High Accuracy (No time limit)**
- **Deep Learning** - AutoEncoders, VAEs
- **Advanced Ensemble** - Multiple algorithm fusion
- **Specialized Methods** - Domain-specific algorithms

---

## 🔍 Algorithm Details

### **📊 Performance Comparison**
Detailed algorithm performance metrics:

| Algorithm | Time Complexity | Space Complexity | Best Use Case | Accuracy |
|-----------|-----------------|------------------|---------------|----------|
| **Isolation Forest** | O(n log n) | O(n) | General purpose | High |
| **LOF** | O(n²) | O(n) | Local anomalies | High |
| **One-Class SVM** | O(n²) | O(n) | Small datasets | Medium |
| **PCA** | O(n×p²) | O(p²) | High-dimensional | Medium |
| **COPOD** | O(n×p) | O(n×p) | Large datasets | High |
| **HBOS** | O(n×p) | O(p) | Fast detection | Medium |

### **🎯 Algorithm Suitability**
When to use each algorithm:

#### **🌟 General Purpose**
- **Isolation Forest** - Best overall performance
- **LOF** - Local density variations
- **COPOD** - Correlation-based detection

#### **🔬 Specialized Scenarios**
- **One-Class SVM** - Non-linear boundaries
- **PCA** - High-dimensional data
- **Deep Learning** - Complex patterns

#### **⚡ Performance-Critical**
- **HBOS** - Fastest detection
- **COPOD** - Scalable to large datasets
- **Ensemble** - Balanced accuracy/speed

---

## 🛠️ Algorithm Implementation

### **🔧 Configuration Examples**
Common algorithm configurations:

#### **Isolation Forest**
```python
from pynomaly.algorithms import IsolationForest

detector = IsolationForest(
    contamination=0.1,
    n_estimators=100,
    max_samples='auto',
    random_state=42
)
```

#### **Local Outlier Factor**
```python
from pynomaly.algorithms import LOF

detector = LOF(
    n_neighbors=20,
    contamination=0.1,
    algorithm='auto'
)
```

#### **Deep Learning**
```python
from pynomaly.algorithms import AutoEncoder

detector = AutoEncoder(
    hidden_dims=[128, 64, 32],
    latent_dim=16,
    epochs=100,
    batch_size=256
)
```

### **📈 Parameter Tuning**
Algorithm parameter optimization:
- **[Performance Tuning Guide](../../user-guides/advanced-features/performance-tuning.md)** - Optimization techniques
- **[AutoML Guide](../../user-guides/advanced-features/automl-and-intelligence.md)** - Automated tuning
- **[Algorithm Selection](../../examples/tutorials/05-algorithm-rationale-selection-guide.md)** - Selection strategies

---

## 🔗 Related Documentation

### **User Guides**
- **[Basic Usage](../../user-guides/basic-usage/)** - Using algorithms
- **[Advanced Features](../../user-guides/advanced-features/)** - Advanced usage
- **[Performance Tuning](../../user-guides/advanced-features/performance-tuning.md)** - Optimization

### **Examples**
- **[Banking Examples](../../examples/banking/)** - Financial use cases
- **[Data Quality](../../examples/Data_Quality_Anomaly_Detection_Guide.md)** - Data validation
- **[Algorithm Selection](../../examples/tutorials/05-algorithm-rationale-selection-guide.md)** - Selection guide

### **Technical**
- **[API Reference](../api/)** - Algorithm APIs
- **[Configuration](../configuration/)** - Algorithm configuration
- **[Developer Guides](../../developer-guides/)** - Implementation details

---

## 🆘 Getting Help

### **Algorithm Support**
- **[GitHub Issues](https://github.com/your-org/pynomaly/issues)** - Report algorithm issues
- **[GitHub Discussions](https://github.com/your-org/pynomaly/discussions)** - Ask questions
- **[Algorithm Comparison](algorithm-comparison.md)** - Detailed comparisons

### **Selection Help**
- **[Algorithm Selection](../../examples/tutorials/05-algorithm-rationale-selection-guide.md)** - Selection strategies
- **[Performance Guide](../../user-guides/advanced-features/performance-tuning.md)** - Performance optimization
- **[Autonomous Mode](../../user-guides/basic-usage/autonomous-mode.md)** - Automated selection

---

## 🚀 Quick Start

Ready to use algorithms? Choose your path:

### **🎯 For General Use**
Start with: **[Core Algorithms](core-algorithms.md)**

### **🔬 For Specialized Use**
Start with: **[Specialized Algorithms](specialized-algorithms.md)**

### **🚀 For Research**
Start with: **[Experimental Algorithms](experimental-algorithms.md)**

### **📊 For Comparison**
Start with: **[Algorithm Comparison](algorithm-comparison.md)**

---

**Last Updated**: 2025-01-09  
**Next Review**: 2025-02-09