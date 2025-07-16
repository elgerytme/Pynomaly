# Anomaly Detection Package

Core anomaly detection algorithms and models for the Pynomaly platform.

## Overview

This package consolidates all anomaly and outlier detection functionality into a single, cohesive domain. It provides a clean architecture implementation with 40+ algorithms, AutoML capabilities, and enterprise-grade features.

## Architecture

The package follows clean architecture principles with clear separation of concerns:

```
anomaly_detection/
├── core/                   # Core domain logic from pynomaly_detection
├── algorithms/             # Algorithm implementations and adapters  
├── services/              # Detection and analysis services
├── adapters/              # External algorithm integrations
└── models/                # Data models and entities
```

## Key Features

- **40+ Algorithms**: Support for PyOD, scikit-learn, PyTorch, TensorFlow, and JAX-based algorithms
- **Clean Architecture**: Domain-driven design with clear separation of concerns
- **AutoML**: Automatic algorithm selection and hyperparameter optimization
- **Real-time & Batch**: Support for both streaming and batch processing
- **Explainable AI**: Built-in explainability features using SHAP and LIME
- **Production Ready**: Enterprise features including monitoring, multi-tenancy, and scalability

## Installation

This package is part of the Pynomaly monorepo. Install dependencies:

```bash
# Basic dependencies
pip install numpy pandas scikit-learn scipy pyod pydantic

# Optional ML dependencies
pip install torch tensorflow optuna shap lime
```

## Usage

```python
from anomaly_detection import AnomalyDetector
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

## Algorithm Categories

### Statistical Methods
- Isolation Forest, LOF, One-Class SVM, Elliptic Envelope
- Z-Score, Modified Z-Score, IQR

### Machine Learning  
- Auto-Encoder, VAE, LSTM-based detectors
- k-NN, PCA, ICA

### Deep Learning
- Deep Auto-Encoders, Deep SVDD
- Adversarial Auto-Encoders, Transformers

### Ensemble Methods
- Feature Bagging, Isolation Forest Ensemble
- LSCP, Average/Max/Min Ensemble

### Specialized
- Time Series, Graph, Text, Multimodal

## Dependencies

- **Core**: `numpy`, `pandas`, `scikit-learn`, `scipy`, `pyod`, `pydantic`
- **Optional**: `torch`, `tensorflow`, `optuna`, `shap`, `lime`
- **Internal**: `core`, `mathematics` packages

## Testing

```bash
pytest tests/
```

## Contributing

See the main repository CONTRIBUTING.md for guidelines.

## License

MIT License - see main repository LICENSE file.