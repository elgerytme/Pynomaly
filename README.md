# Pynomaly

[![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)
[![Imports: isort](https://img.shields.io/badge/%20imports-isort-%231674b1?style=flat&labelColor=ef8336)](https://pycqa.github.io/isort/)
[![Type checked: mypy](https://www.mypy-lang.org/static/mypy_badge.svg)](https://mypy-lang.org/)

State-of-the-art Python anomaly detection package with clean architecture.

## Features

- 🏗️ **Clean Architecture**: Domain-driven design with clear separation of concerns
- 🔌 **Unified Interface**: Consistent API for PyOD, TODS, PyGOD, scikit-learn, and more
- 🚀 **Production Ready**: Built for scalability, monitoring, and reliability
- 🧠 **AutoML**: Automated algorithm selection and hyperparameter tuning
- 📊 **Multi-Modal**: Support for time-series, tabular, graph, and text data
- 🔍 **Explainability**: SHAP and LIME integration for interpretable results
- ⚡ **High Performance**: GPU acceleration and efficient memory usage
- 🛡️ **Type Safe**: 100% type coverage with strict mypy checking

## Installation

```bash
pip install pynomaly
```

For additional ML backends:

```bash
pip install pynomaly[torch]     # PyTorch support
pip install pynomaly[tensorflow] # TensorFlow support
pip install pynomaly[jax]       # JAX support
pip install pynomaly[all]       # All optional dependencies
```

## Quick Start

```python
from pynomaly import Detector, AnomalyScore
from pynomaly.infrastructure.adapters import PyODAdapter
import pandas as pd

# Load your data
data = pd.read_csv("data.csv")

# Create detector with PyOD's IsolationForest
detector = Detector(
    adapter=PyODAdapter("IsolationForest"),
    contamination_rate=0.1
)

# Fit and detect anomalies
detector.fit(data)
anomalies = detector.detect(data)

# Get anomaly scores with confidence intervals
scores: list[AnomalyScore] = detector.score(data)
```

## Architecture

Pynomaly follows clean architecture principles:

```
┌─────────────────────────────────────────────────┐
│                 Presentation Layer              │
│          (REST API, CLI, Python SDK)            │
├─────────────────────────────────────────────────┤
│                Application Layer                │
│    (Use Cases, Services, Orchestration)         │
├─────────────────────────────────────────────────┤
│                  Domain Layer                   │
│      (Entities, Value Objects, Services)        │
├─────────────────────────────────────────────────┤
│              Infrastructure Layer               │
│   (Adapters, Persistence, External Services)    │
└─────────────────────────────────────────────────┘
```

## Documentation

Full documentation is available at [https://pynomaly.readthedocs.io](https://pynomaly.readthedocs.io)

## Contributing

We welcome contributions! Please see our [Contributing Guide](CONTRIBUTING.md) for details.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.