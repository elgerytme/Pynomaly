# Pynomaly 🔍

[![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)
[![Imports: isort](https://img.shields.io/badge/%20imports-isort-%231674b1?style=flat&labelColor=ef8336)](https://pycqa.github.io/isort/)
[![Type checked: mypy](https://www.mypy-lang.org/static/mypy_badge.svg)](https://mypy-lang.org/)

State-of-the-art Python anomaly detection platform with clean architecture, 40+ algorithms, and production-ready features.

## Features

- 🏗️ **Clean Architecture**: Domain-driven design with hexagonal architecture
- 🔌 **40+ Algorithms**: PyOD, scikit-learn, and deep learning frameworks
- 🚀 **Production Ready**: Async support, monitoring, caching, error handling
- 🖥️ **Multiple Interfaces**: REST API, CLI, and Progressive Web App (PWA)
- 📊 **Visualizations**: Interactive charts with D3.js and Apache ECharts
- 🧪 **Experiment Tracking**: Compare models and track performance
- ⚡ **High Performance**: GPU support, batch processing, efficient memory usage
- 🛡️ **Type Safe**: Full type hints and runtime validation with Pydantic

## Installation

```bash
# Install with Poetry
poetry install

# Or install with extras
poetry install -E torch      # PyTorch support
poetry install -E tensorflow # TensorFlow support
poetry install -E jax        # JAX support
poetry install -E all        # All optional dependencies
```

## Quick Start

### CLI Usage

```bash
# List available algorithms
pynomaly detector algorithms

# Create a detector
pynomaly detector create --name "My Detector" --algorithm IsolationForest

# Load a dataset
pynomaly dataset load data.csv --name "My Data"

# Train and detect
pynomaly detect train <detector_id> <dataset_id>
pynomaly detect run <detector_id> <dataset_id>

# Start web UI
pynomaly server start
```

### Python API

```python
from pynomaly.infrastructure.config import create_container
from pynomaly.domain.entities import Detector, Dataset
import pandas as pd

# Initialize container
container = create_container()

# Create detector
detector = Detector(
    name="IForest Detector",
    algorithm="IsolationForest",
    parameters={"contamination": 0.1}
)
container.detector_repository().save(detector)

# Load dataset
data = pd.read_csv("data.csv")
dataset = Dataset(name="My Data", data=data)
container.dataset_repository().save(dataset)

# Train and detect
detection_service = container.detection_service()
await detection_service.train_and_detect(detector.id, dataset)
```

### Web Interface

Access the Progressive Web App at http://localhost:8000 after starting the server:

- Real-time anomaly detection dashboard
- Interactive visualizations with D3.js and ECharts
- Experiment tracking and model comparison
- Dataset quality analysis
- HTMX-powered dynamic updates

## Architecture

Pynomaly follows clean architecture principles with Domain-Driven Design:

```
src/pynomaly/
├── domain/          # Business logic and entities
│   ├── entities/    # Anomaly, Detector, Dataset, DetectionResult
│   ├── services/    # AnomalyScorer, ThresholdCalculator, etc.
│   └── exceptions/  # Domain-specific exceptions
├── application/     # Use cases and application services
│   ├── use_cases/   # DetectAnomalies, TrainDetector, etc.
│   ├── services/    # DetectionService, EnsembleService, etc.
│   └── dto/         # Data transfer objects
├── infrastructure/  # External adapters and implementations
│   ├── adapters/    # PyODAdapter, SklearnAdapter
│   ├── persistence/ # Repository implementations
│   └── config/      # DI container and settings
└── presentation/    # User interfaces
    ├── api/         # FastAPI REST endpoints
    ├── cli/         # Typer CLI commands
    └── web/         # HTMX + Tailwind PWA
```

## Supported Algorithms

### Statistical Methods
- Isolation Forest (IF)
- Local Outlier Factor (LOF)
- One-Class SVM (OCSVM)
- Minimum Covariance Determinant (MCD)
- Principal Component Analysis (PCA)

### Proximity-Based
- k-Nearest Neighbors (kNN)
- Connectivity-Based Outlier Factor (COF)
- Clustering-Based Local Outlier Factor (CBLOF)

### Probabilistic
- Gaussian Mixture Models (GMM)
- Copula-Based Outlier Detection (COPOD)
- Empirical Cumulative Distribution (ECOD)

### Linear Models
- Principal Component Analysis (PCA)
- Kernel PCA
- Robust Covariance

### Neural Networks
- AutoEncoder (AE)
- Variational AutoEncoder (VAE)
- Adversarially Learned Anomaly Detection (ALAD)

### Ensemble Methods
- Isolation Forest Ensemble
- Feature Bagging
- LSCP (Locally Selective Combination)
- XGBOD (Extreme Gradient Boosting)

And many more! Run `pynomaly detector algorithms` for the full list.

## Development

```bash
# Install dev dependencies
poetry install --with dev

# Run tests
pytest

# Linting and formatting
black src tests
isort src tests
mypy src

# Build documentation
mkdocs serve
```

## Contributing

We welcome contributions! Please see our [Contributing Guide](CONTRIBUTING.md) for details.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.