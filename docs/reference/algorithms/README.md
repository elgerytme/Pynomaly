# Algorithm Reference

🍞 **Breadcrumb:** 🏠 [Home](../../index.md) > 📖 [Reference](../README.md) > 🧮 [Algorithms](README.md)

---


## Overview

This directory provides comprehensive documentation for all anomaly detection algorithms available in Pynomaly. The content is organized into specialized guides to help you find the most appropriate algorithm for your use case.

## Documentation Structure

### 📖 **Core Algorithm Guides**

- **[Core Algorithms](core-algorithms.md)** - Essential algorithms for most use cases (20-25 algorithms)
- **[Specialized Algorithms](specialized-algorithms.md)** - Domain-specific algorithms for time series, graphs, text, etc.
- **[Experimental Algorithms](experimental-algorithms.md)** - Advanced/research algorithms and cutting-edge methods
- **[Algorithm Comparison](algorithm-comparison.md)** - Performance comparisons and selection guidance

### 🎯 **Quick Navigation by Use Case**

#### **Data Type**
- **Tabular Data**: Core Algorithms → IsolationForest, LOF, OneClassSVM
- **Time Series**: Specialized Algorithms → LSTM, Matrix Profile, Prophet
- **Graph Data**: Specialized Algorithms → DOMINANT, GCNAE
- **Text Data**: Specialized Algorithms → TF-IDF+Clustering, Word Embeddings
- **Images**: Specialized Algorithms → CNN AutoEncoder, VAE

#### **Dataset Size**
- **Small (< 1K)**: Core Algorithms → LOF, EllipticEnvelope
- **Medium (1K-100K)**: Core Algorithms → IsolationForest, OneClassSVM
- **Large (> 100K)**: Core Algorithms → IsolationForest, ECOD, HBOS

#### **Performance Requirements**
- **High Speed**: Core Algorithms → HBOS, ECOD, Z-Score
- **High Accuracy**: Experimental Algorithms → Deep Learning, Ensembles
- **Interpretability**: Core Algorithms → Statistical methods, LOF

### 📊 **Algorithm Categories**

| Category | Count | Documentation | Best For |
|----------|-------|---------------|----------|
| **Statistical** | 8 | Core Algorithms | Baseline detection, interpretable results |
| **Machine Learning** | 12 | Core Algorithms | General-purpose, production systems |
| **Deep Learning** | 10 | Experimental Algorithms | Complex patterns, feature learning |
| **Specialized** | 15 | Specialized Algorithms | Domain-specific applications |
| **Ensemble** | 5 | Experimental Algorithms | Maximum accuracy, robust detection |

### 🚀 **Getting Started**

1. **New to Anomaly Detection?** Start with [Core Algorithms](core-algorithms.md)
2. **Specific Domain?** Check [Specialized Algorithms](specialized-algorithms.md)
3. **Need Best Performance?** Explore [Experimental Algorithms](experimental-algorithms.md)
4. **Comparing Options?** See [Algorithm Comparison](algorithm-comparison.md)

### 🔧 **Implementation Examples**

Each algorithm guide includes:
- ✅ **Description** and algorithm details
- ✅ **Parameters** with recommended ranges
- ✅ **Code examples** with Pynomaly API
- ✅ **Use cases** and when to apply
- ✅ **Performance characteristics** (speed, memory, scalability)
- ✅ **Strengths and limitations**

### 📈 **Performance Guidance**

For detailed performance comparisons and selection guidance, see:
- **[Algorithm Comparison](algorithm-comparison.md)** - Comprehensive comparison matrix
- **[Performance Benchmarks](../../examples/performance-benchmarking.md)** - Real-world performance data
- **[Autonomous Mode Guide](../../comprehensive/09-autonomous-classifier-selection-guide.md)** - Automated selection

### 🤝 **Contributing**

To add new algorithms or improve documentation:
- See [Plugin Development Guide](../../development/plugin-development.md)
- Follow [Documentation Standards](../../project/standards/documentation-standards.md)

---

**💡 Quick Tip**: Use the [Autonomous Mode](../../comprehensive/09-autonomous-classifier-selection-guide.md) for automatic algorithm selection based on your data characteristics.