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

### Quick Setup (Python + pip only)

If you want to run Pynomaly without Poetry, Docker, or Make:

```bash
# Run the setup script
python setup_simple.py

# Or manually:
python -m venv .venv
.venv\Scripts\activate  # Windows (or source .venv/bin/activate on Linux/Mac)
pip install -r requirements.txt
pip install -e .
```

Then run the app:
```bash
python cli.py --help
python cli.py server start
```

See [README_SIMPLE_SETUP.md](README_SIMPLE_SETUP.md) for detailed instructions.

### Full Setup (with Poetry)

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

## Business Intelligence Integrations 📊

Export anomaly detection results to major business intelligence and spreadsheet platforms:

### Supported Platforms

- **Excel**: Advanced formatting, charts, multiple worksheets
- **Power BI**: Real-time streaming, automated reports, Azure AD integration
- **Google Sheets**: Collaborative editing, real-time updates, sharing
- **Smartsheet**: Project tracking, workflow automation, team collaboration

### Installation

```bash
# Install all BI integrations
pip install pynomaly[bi-integrations]

# Or install specific platforms
pip install pynomaly[excel]      # Excel support
pip install pynomaly[powerbi]    # Power BI support  
pip install pynomaly[gsheets]    # Google Sheets support
pip install pynomaly[smartsheet] # Smartsheet support
```

### CLI Usage

```bash
# List available export formats
pynomaly export list-formats

# Export to Excel with charts and formatting
pynomaly export excel results.json report.xlsx --include-charts

# Export to Power BI workspace
pynomaly export powerbi results.json \
    --workspace-id "your-workspace-id" \
    --dataset-name "Anomaly Detection"

# Export to Google Sheets with sharing
pynomaly export gsheets results.json \
    --credentials-file creds.json \
    --share-emails user@company.com

# Export to multiple formats
pynomaly export multi results.json \
    --formats excel powerbi gsheets
```

### Python API

```python
from pynomaly.application.services.export_service import ExportService
from pynomaly.application.dto.export_options import ExportOptions

# Initialize export service
export_service = ExportService()

# Export to Excel with advanced features
excel_options = ExportOptions().for_excel()
excel_options.include_charts = True
excel_options.highlight_anomalies = True

result = export_service.export_results(
    detection_results,
    "anomaly_report.xlsx",
    excel_options
)

# Export to Power BI streaming dataset  
powerbi_options = ExportOptions().for_powerbi(
    workspace_id="workspace-123",
    dataset_name="Live Anomaly Feed"
)
powerbi_options.streaming_dataset = True

result = export_service.export_results(
    detection_results,
    "",  # No file path for cloud services
    powerbi_options
)

# Multi-platform export
results = export_service.export_multiple_formats(
    detection_results,
    base_path="anomaly_analysis",
    formats=[ExportFormat.EXCEL, ExportFormat.GSHEETS]
)
```

### Features

- **Real-time Collaboration**: Google Sheets and Smartsheet live updates
- **Advanced Visualizations**: Charts, conditional formatting, dashboards
- **Secure Authentication**: Azure AD, OAuth2, API tokens
- **Batch Processing**: Efficient handling of large datasets
- **Error Recovery**: Robust retry logic and validation
- **Template Support**: Pre-configured layouts and workflows

See [examples/bi_integrations_example.py](examples/bi_integrations_example.py) for detailed usage examples.

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