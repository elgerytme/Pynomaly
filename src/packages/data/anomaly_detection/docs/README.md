# Anomaly Detection Package Documentation

## Table of Contents

- [Architecture Overview](architecture.md)
- [API Reference](api.md) 
- [Algorithm Guide](algorithms.md)
- [Ensemble Methods](ensemble.md)
- [Streaming Detection](streaming.md)
- [Explainability](explainability.md)
- [Development Guide](development.md)
- [Examples](../examples/)

## Quick Links

- **API Documentation**: Available at `/docs` when running the server
- **Source Code**: Located in `src/anomaly_detection/`
- **Tests**: Located in `tests/`
- **Examples**: Located in `examples/`

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