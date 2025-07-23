# Anomaly Detection Documentation

Welcome to the comprehensive documentation for the Anomaly Detection package - a production-ready, enterprise-grade solution for detecting anomalies in various types of data.

## 🎯 **Quick Start**

New to anomaly detection? Start here:

1. **[Installation](installation.md)** - Get up and running in minutes
2. **[API Overview](api.md)** - Core concepts and basic usage
3. **[CLI Guide](cli.md)** - Command-line interface for power users
4. **[Algorithm Guide](algorithms.md)** - Choose the right algorithm

## 📚 **Complete Documentation Guide**

### **Core Functionality**
| Guide | Description | Difficulty | Est. Time |
|-------|-------------|------------|-----------|
| [🔧 **CLI User Guide**](cli.md) | Complete command-line interface with examples | Beginner | 15 min |
| [🤖 **Model Management**](model_management.md) | Training, persistence, versioning, and registry | Intermediate | 30 min |
| [🎯 **Ensemble Methods**](ensemble.md) | Advanced combination strategies and architectures | Advanced | 45 min |
| [🧠 **Explainability**](explainability.md) | SHAP integration and model interpretability | Intermediate | 30 min |

### **Production & Deployment** 
| Guide | Description | Difficulty | Est. Time |
|-------|-------------|------------|-----------|
| [⚡ **Performance Optimization**](performance.md) | Profiling, scaling, and optimization techniques | Advanced | 60 min |
| [🚀 **Production Deployment**](deployment.md) | Kubernetes, Docker, and auto-scaling | Advanced | 90 min |
| [🔄 **Streaming Detection**](streaming.md) | Real-time processing with Kafka/Redis | Advanced | 60 min |
| [🔌 **Integration Examples**](integration.md) | APIs, databases, and workflow integration | Intermediate | 45 min |

### **Security & Operations**
| Guide | Description | Difficulty | Est. Time |
|-------|-------------|------------|-----------|
| [🔒 **Security & Privacy**](security.md) | Authentication, encryption, and data protection | Advanced | 60 min |
| [🛠️ **Troubleshooting & FAQ**](troubleshooting.md) | Common issues, solutions, and diagnostic tools | All Levels | 30 min |

### **Reference Documentation**
| Guide | Description | Type |
|-------|-------------|------|
| [🏗️ **Architecture Overview**](architecture.md) | System design and DDD structure | Reference |
| [📖 **API Reference**](api.md) | Complete API documentation | Reference |
| [⚙️ **Configuration**](configuration.md) | All configuration options | Reference |
| [🔬 **Algorithm Guide**](algorithms.md) | Detailed algorithm documentation | Reference |

## 🎮 **Learning Paths**

### **🐣 Beginner Path** (2-3 hours)
Perfect if you're new to anomaly detection:

1. [Installation Guide](installation.md) *(10 min)*
2. [API Overview](api.md) *(20 min)*
3. [CLI User Guide](cli.md) *(30 min)*
4. [Basic Model Training](model_management.md#basic-training) *(45 min)*
5. [Common Issues](troubleshooting.md#common-error-messages) *(15 min)*

**🎯 Goal**: Detect your first anomaly and understand core concepts

### **💼 Data Scientist Path** (4-5 hours)
For data scientists and ML practitioners:

1. [Algorithm Selection](algorithms.md) *(30 min)*
2. [Model Management Deep Dive](model_management.md) *(60 min)*
3. [Ensemble Methods](ensemble.md) *(90 min)*
4. [Model Explainability](explainability.md) *(60 min)*
5. [Performance Optimization](performance.md) *(60 min)*

**🎯 Goal**: Build production-ready anomaly detection models

### **🔧 DevOps Engineer Path** (5-6 hours)
For deployment and infrastructure management:

1. [Production Deployment](deployment.md) *(120 min)*
2. [Streaming Detection](streaming.md) *(90 min)*
3. [Security Implementation](security.md) *(90 min)*
4. [Integration Patterns](integration.md) *(60 min)*
5. [Monitoring & Troubleshooting](troubleshooting.md) *(60 min)*

**🎯 Goal**: Deploy and maintain anomaly detection systems at scale

## 📋 **Quick Navigation**

### **By Task**
- **🚀 Getting Started**: [Installation](installation.md) → [API Overview](api.md) → [CLI Guide](cli.md)
- **🔧 Development**: [Model Management](model_management.md) → [Ensemble Methods](ensemble.md) → [Explainability](explainability.md)
- **🏭 Production**: [Performance](performance.md) → [Deployment](deployment.md) → [Security](security.md)
- **🔍 Troubleshooting**: [Common Issues](troubleshooting.md) → [Performance Problems](performance.md#performance-problems) → [API Issues](integration.md#api-integration)

### **By Audience**
- **👩‍💻 Developers**: Start with [API Overview](api.md) and [Integration Examples](integration.md)
- **👨‍🔬 Data Scientists**: Focus on [Model Management](model_management.md) and [Ensemble Methods](ensemble.md)
- **👩‍💼 DevOps Engineers**: Begin with [Deployment Guide](deployment.md) and [Security Guide](security.md)
- **👨‍💼 Business Users**: Check [Use Cases](examples/) and [Architecture Overview](architecture.md)

## 🔗 **Quick Links**

- **💻 Source Code**: Located in `src/anomaly_detection/`
- **✅ Tests**: Located in `tests/`
- **📝 Examples**: Located in `examples/`
- **🌐 API Documentation**: Available at `/docs` when running the server
- **📊 Live Demo**: Try the interactive examples

## Architecture

This package follows Domain-Driven Design (DDD) principles:

```
src/anomaly_detection/
├── application/     # Business logic and use cases
├── domain/          # Core domain entities and services
├── infrastructure/  # External dependencies and adapters
└── presentation/    # API endpoints, CLI, and web interfaces
```

## Core Components

### Domain Services

- **DetectionService**: Core anomaly detection logic
- **EnsembleService**: Ensemble method coordination
- **StreamingService**: Real-time stream processing

### Infrastructure Adapters

- **SklearnAdapter**: Scikit-learn algorithm integration
- **PyodAdapter**: PyOD library integration  
- **DeepLearningAdapter**: Neural network models

### Application Services

- **ExplanationAnalyzers**: Feature importance and explainability
- **PerformanceOptimizer**: Performance monitoring and optimization

## Supported Algorithms

### Single Algorithms
- **Isolation Forest**: Tree-based anomaly detection
- **One-Class SVM**: Support vector machine for outliers
- **Local Outlier Factor**: Density-based detection
- **Autoencoder**: Neural network reconstruction error
- **Gaussian Mixture**: Statistical modeling approach

### Ensemble Methods
- **Voting**: Majority vote combination
- **Averaging**: Score averaging across algorithms
- **Stacking**: Meta-learning combination

## Usage Patterns

### CLI Usage
```bash
# Basic detection
anomaly-detection detect run --input data.csv --algorithm isolation_forest

# Ensemble detection
anomaly-detection ensemble combine --input data.csv --algorithms isolation_forest one_class_svm

# Stream monitoring
anomaly-detection stream monitor --source kafka://topic --window-size 100
```

### API Usage
```python
# Start server
anomaly-detection-server

# Use REST API at http://localhost:8001/docs
```

### Python SDK
```python
from anomaly_detection import DetectionService, EnsembleService

# Single algorithm
service = DetectionService()
results = service.detect(data, algorithm="isolation_forest")

# Ensemble
ensemble = EnsembleService()
results = ensemble.detect(data, algorithms=["isolation_forest", "lof"])
```

## Getting Help

- Check the [examples/](../examples/) directory for usage examples
- Review the API documentation at `/docs` when running the server
- See the [algorithm guide](algorithms.md) for parameter tuning
- Check [streaming guide](streaming.md) for real-time processing

## Support

For issues and questions:

1. Check existing documentation
2. Review examples and test cases  
3. Open an issue in the repository