# Pynomaly Detection

Production-ready Python anomaly detection library with clean architecture, AutoML, and 40+ algorithms.

## Features

- **40+ Algorithms**: Support for PyOD, scikit-learn, PyTorch, TensorFlow, and JAX-based algorithms
- **Clean Architecture**: Domain-driven design with clear separation of concerns
- **AutoML**: Automatic algorithm selection and hyperparameter optimization
- **Real-time & Batch**: Support for both streaming and batch processing
- **Explainable AI**: Built-in explainability features using SHAP and LIME
- **Production Ready**: Enterprise features including monitoring, multi-tenancy, and scalability
- **Type Safe**: Full type annotations and strict mypy compliance

## Installation

### Basic Installation
```bash
pip install pynomaly-detection
```

### With ML Dependencies
```bash
pip install pynomaly-detection[ml]
```

### With AutoML
```bash
pip install pynomaly-detection[automl]
```

### With Deep Learning
```bash
pip install pynomaly-detection[torch]  # PyTorch
pip install pynomaly-detection[tensorflow]  # TensorFlow
pip install pynomaly-detection[jax]  # JAX
```

### Full Installation
```bash
pip install pynomaly-detection[all]
```

## Quick Start

### Basic Usage
```python
from pynomaly_detection import AnomalyDetector
import numpy as np

# Generate sample data
X = np.random.randn(1000, 10)
X[0:10] += 5  # Add some outliers

# Create detector
detector = AnomalyDetector()

# Fit and predict
detector.fit(X)
anomalies = detector.predict(X)

print(f"Found {anomalies.sum()} anomalies")
```

### With Specific Algorithm
```python
from pynomaly_detection import AnomalyDetector
from pynomaly_detection.algorithms import IsolationForestAdapter

# Use specific algorithm
detector = AnomalyDetector(algorithm=IsolationForestAdapter())
detector.fit(X)
results = detector.predict(X)
```

### AutoML Mode
```python
from pynomaly_detection.services import AutoMLService

# Automatic algorithm selection
automl = AutoMLService()
best_detector = automl.find_best_algorithm(X_train, y_train)
predictions = best_detector.predict(X_test)
```

### Explainable AI
```python
from pynomaly_detection.services import ExplainabilityService

explainer = ExplainabilityService()
explanations = explainer.explain_predictions(detector, X, predictions)
```

## Supported Algorithms

### Statistical Methods
- Isolation Forest
- Local Outlier Factor (LOF)
- One-Class SVM
- Elliptic Envelope
- Z-Score
- Modified Z-Score
- Interquartile Range (IQR)

### Machine Learning
- Auto-Encoder
- Variational Auto-Encoder (VAE)
- LSTM-based detectors
- k-NN based methods
- Principal Component Analysis (PCA)
- Independent Component Analysis (ICA)

### Deep Learning
- Deep Auto-Encoders (PyTorch/TensorFlow)
- Deep SVDD
- Adversarial Auto-Encoders
- Transformer-based detectors

### Ensemble Methods
- Feature Bagging
- Isolation Forest Ensemble
- LSCP (Locally Selective Combination)
- Average/Max/Min Ensemble

### Specialized
- Time Series Anomaly Detection
- Graph Anomaly Detection
- Text Anomaly Detection
- Multimodal Fusion

## Architecture

The library follows clean architecture principles:

```
pynomaly_detection/
├── core/                   # Domain logic
│   ├── domain/            # Entities, value objects, services
│   ├── use_cases/         # Application use cases
│   └── dto/               # Data transfer objects
├── algorithms/            # Algorithm implementations
│   └── adapters/          # Algorithm adapters
├── services/              # Application services
└── infrastructure/        # External concerns
```

## Configuration

### Environment Variables
```bash
export PYNOMALY_LOG_LEVEL=INFO
export PYNOMALY_CACHE_SIZE=1000
export PYNOMALY_MAX_WORKERS=4
```

### Configuration File
```python
from pynomaly_detection.core.dto import ConfigurationDTO

config = ConfigurationDTO(
    contamination=0.1,
    n_estimators=100,
    max_samples=256,
    bootstrap=True,
    n_jobs=-1
)

detector = AnomalyDetector(config=config)
```

## Performance

Benchmarks on common datasets:

| Algorithm | Dataset | Training Time | Prediction Time | F1-Score |
|-----------|---------|---------------|-----------------|----------|
| IsolationForest | KDD Cup 99 | 2.3s | 0.1s | 0.87 |
| AutoEncoder | Credit Card | 45s | 0.05s | 0.92 |
| LOF | Breast Cancer | 0.8s | 0.2s | 0.85 |

## Development

### Setup
```bash
git clone https://github.com/elgerytme/Pynomaly.git
cd Pynomaly/pynomaly-detection
pip install -e .[dev,test]
```

### Testing
```bash
pytest tests/
```

### Linting
```bash
ruff check src/
black src/
mypy src/
```

## Contributing

1. Fork the repository
2. Create a feature branch
3. Add tests for new functionality
4. Ensure all tests pass
5. Submit a pull request

## License

MIT License - see LICENSE file for details.

## Citation

```bibtex
@software{pynomaly_detection,
  title={Pynomaly Detection: Production-ready Python Anomaly Detection},
  author={Pynomaly Team},
  url={https://github.com/elgerytme/Pynomaly},
  year={2024}
}
```

## Support

- Documentation: [GitHub Pages](https://github.com/elgerytme/Pynomaly/blob/main/docs/)
- Issues: [GitHub Issues](https://github.com/elgerytme/Pynomaly/issues)
- Discussions: [GitHub Discussions](https://github.com/elgerytme/Pynomaly/discussions)

## Roadmap

- [ ] Distributed computing support
- [ ] ONNX model export
- [ ] REST API server
- [ ] Kubernetes operators
- [ ] MLflow integration
- [ ] Apache Kafka streaming