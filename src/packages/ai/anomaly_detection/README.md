# Anomaly Detection Package

A clean, domain-focused anomaly detection package that integrates with the broader anomaly_detection ML infrastructure.

## Overview

This package provides core anomaly detection functionality while leveraging the ML/MLOps capabilities from `@src/packages/ai/machine_learning`. It follows clean architecture principles and focuses specifically on anomaly detection domain logic.

## Key Features

- **Core Detection Services**: Unified interfaces for anomaly detection, ensemble methods, and streaming detection
- **Algorithm Adapters**: Support for scikit-learn, PyOD, and deep learning frameworks
- **ML Integration**: Seamless integration with AutoML and MLOps services
- **Production Ready**: Clean architecture, type safety, and comprehensive testing
- **Streaming Support**: Real-time anomaly detection with concept drift monitoring

## Quick Start

### Basic Usage

```python
from anomaly_detection import AnomalyDetector
import numpy as np

# Generate sample data
X = np.random.rand(1000, 10)

# Create and use detector
detector = AnomalyDetector(algorithm="iforest")
anomalies = detector.detect(X)

print(f"Found {np.sum(anomalies)} anomalies")
```

### Advanced Usage

```python
from anomaly_detection import DetectionService, EnsembleService
from anomaly_detection.algorithms.adapters import SklearnAdapter, PyODAdapter

# Use detection service directly
service = DetectionService()
result = service.detect_anomalies(X, algorithm="lof", contamination=0.1)

# Create ensemble detector
ensemble = EnsembleService()
ensemble_result = ensemble.detect_with_ensemble(
    X, 
    algorithms=["iforest", "lof"], 
    combination_method="majority"
)

# Use specific adapters
sklearn_adapter = SklearnAdapter("iforest", n_estimators=200)
sklearn_adapter.fit(X)
predictions = sklearn_adapter.predict(X)
```

### Streaming Detection

```python
from anomaly_detection import StreamingService
import numpy as np

# Create streaming detector
streaming = StreamingService(window_size=500, update_frequency=50)

# Process data stream
for i in range(1000):
    sample = np.random.rand(10)  # New data point
    result = streaming.process_sample(sample)
    
    if result.predictions[0] == 1:
        print(f"Anomaly detected at sample {i}")

# Check for concept drift
drift_info = streaming.detect_concept_drift()
print("Drift detected:", drift_info["drift_detected"])
```

## Architecture

### Package Structure

```
anomaly_detection/
├── __init__.py              # Main public API
├── core/                    # Core domain logic
│   └── services/            # Detection, ensemble, streaming services
├── algorithms/              # Algorithm implementations
│   └── adapters/            # Framework adapters (sklearn, PyOD, deep learning)
├── data/                    # Data processing utilities
├── monitoring/              # Performance monitoring
├── integrations/            # External integrations
└── utils/                   # Shared utilities
```

### Core Services

1. **DetectionService**: Main service for anomaly detection operations
2. **EnsembleService**: Combines multiple algorithms for robust detection  
3. **StreamingService**: Real-time detection with incremental learning

### Algorithm Adapters

1. **SklearnAdapter**: Scikit-learn algorithms (Isolation Forest, LOF, One-Class SVM, PCA)
2. **PyODAdapter**: 40+ algorithms from PyOD library
3. **DeepLearningAdapter**: Autoencoder-based detection with TensorFlow/PyTorch

## Integration with ML Infrastructure

This package integrates with the broader ML infrastructure:

```python
# AutoML integration
from anomaly_detection import get_automl_service
automl = get_automl_service()

# MLOps integration  
from anomaly_detection import ModelManagementService
model_mgmt = ModelManagementService()
```

## Installation

### Core Package
```bash
pip install -e .
```

### With PyOD Support
```bash
pip install -e ".[pyod]"
```

### With Deep Learning Support
```bash
pip install -e ".[deeplearning]"
```

### Full Installation
```bash
pip install -e ".[full]"
```

## Available Algorithms

### Built-in Algorithms
- `iforest`: Isolation Forest
- `lof`: Local Outlier Factor

### With SklearnAdapter
- `iforest`: Isolation Forest
- `lof`: Local Outlier Factor  
- `ocsvm`: One-Class SVM
- `pca`: PCA-based detection

### With PyODAdapter (40+ algorithms)
- `iforest`, `lof`, `ocsvm`, `pca`, `knn`, `hbos`, `abod`, `feature_bagging`
- And many more specialized algorithms

### With DeepLearningAdapter
- `autoencoder`: Autoencoder-based detection (TensorFlow/PyTorch)

## Configuration

The package supports flexible configuration:

```python
# Algorithm-specific parameters
detector = AnomalyDetector(
    algorithm="iforest",
    n_estimators=200,
    contamination=0.05
)

# Service configuration
service = DetectionService()
service.register_adapter("custom_algo", CustomAdapter())
```

## Testing

Run tests with:
```bash
pytest tests/
```

With coverage:
```bash
pytest --cov=anomaly_detection tests/
```

## Development

### Code Quality
```bash
# Linting
ruff check anomaly_detection/

# Type checking  
mypy anomaly_detection/

# Formatting
black anomaly_detection/
```

## Migration from Legacy Package

This consolidated package replaces the previous complex structure with:

- **Reduced complexity**: From 118+ services to 3 core services
- **Clear separation**: Domain logic vs ML infrastructure
- **Better integration**: Proper ML/MLOps delegation
- **Maintainable structure**: Standard Python package organization

### Key Changes
1. AutoML functionality moved to `@src/packages/ai/machine_learning`
2. MLOps features moved to `@src/packages/ai/machine_learning/mlops`
3. Algorithm adapters consolidated and simplified
4. Core detection logic streamlined

## Contributing

1. Follow the existing code style and architecture
2. Add tests for new functionality
3. Update documentation for changes
4. Ensure integration with ML infrastructure works properly

## License

MIT License - see LICENSE file for details.