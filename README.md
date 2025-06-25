# Pynomaly 🔍

[![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)
[![Imports: isort](https://img.shields.io/badge/%20imports-isort-%231674b1?style=flat&labelColor=ef8336)](https://pycqa.github.io/isort/)
[![Type checked: mypy](https://www.mypy-lang.org/static/mypy_badge.svg)](https://mypy-lang.org/)

State-of-the-art Python anomaly detection package targeting Python 3.11+ with clean architecture principles, integrating multiple ML libraries (PyOD, TODS, PyGOD, scikit-learn, PyTorch, TensorFlow, JAX) through a unified, production-ready interface.

## Features

- 🏗️ **Clean Architecture**: Domain-driven design with hexagonal architecture (Ports & Adapters)
- 🔌 **Multi-Library Integration**: Unified interface for PyOD, TODS, PyGOD, scikit-learn, PyTorch, TensorFlow, JAX
- 🚀 **Production Ready**: Async/await, OpenTelemetry observability, Prometheus metrics, circuit breakers
- 🖥️ **Multiple Interfaces**: FastAPI REST API, Typer CLI, and Progressive Web App (PWA)
- 📊 **Modern Web UI**: HTMX + Tailwind CSS + D3.js + Apache ECharts with offline PWA capabilities
- 🧪 **Advanced Features**: AutoML, SHAP/LIME explainability, drift detection, active learning
- ⚡ **Multi-Modal Support**: Time-series, tabular, graph, and text anomaly detection
- 🛡️ **Type Safe**: 100% type coverage with mypy --strict, Pydantic validation
- 🔄 **Streaming & Batch**: Real-time processing with backpressure and large dataset support
- 🧰 **Extensible**: Plugin architecture with algorithm registry and custom adapters

## Installation

### Quick Setup (Python + pip only)

If you want to run Pynomaly without Poetry, Docker, or Make:

```bash
# Run the automated setup script
python scripts/setup_simple.py

# Or manually with different requirement levels:
python -m venv .venv
# Windows (PowerShell):
.venv\Scripts\Activate.ps1
# Windows (Command Prompt):
.venv\Scripts\activate.bat
# Linux/macOS:
source .venv/bin/activate

# Choose your installation level:
pip install -r requirements.txt          # Minimal (PyOD + NumPy + Pandas + Polars)
pip install -r requirements-minimal.txt  # + scikit-learn + scipy
pip install -r requirements-server.txt   # + API + CLI functionality  
pip install -r requirements-production.txt # Production-ready stack

pip install -e .
```

Then run the CLI:
```bash
# Primary method (after pip install -e .)
pynomaly --help
pynomaly server start

# Alternative methods
python scripts/cli.py --help
python -m pynomaly.presentation.cli.app --help
```

See [docs/getting-started/README_SIMPLE_SETUP.md](docs/getting-started/README_SIMPLE_SETUP.md) for detailed instructions.

### Full Setup (with Poetry)

```bash
# Minimal installation (PyOD, NumPy, Pandas, Polars + core architecture)
poetry install

# Or install with extras for specific functionality
poetry install -E minimal    # Add scikit-learn + scipy
poetry install -E api        # Web API functionality
poetry install -E cli        # Command-line interface
poetry install -E server     # API + CLI + basic features
poetry install -E production # Production-ready stack

# ML framework extras
poetry install -E torch      # PyTorch deep learning support
poetry install -E tensorflow # TensorFlow neural networks
poetry install -E jax        # JAX high-performance computing
poetry install -E graph      # PyGOD graph anomaly detection
poetry install -E automl     # AutoML with auto-sklearn2
poetry install -E explainability # SHAP/LIME model explanation

# Data processing extras
poetry install -E data-formats # Parquet, Excel, HDF5 support
poetry install -E database   # SQL database connectivity
poetry install -E spark      # Apache Spark integration

# Comprehensive installations
poetry install -E ml-all     # All ML frameworks and tools
poetry install -E all        # All optional dependencies
```

### Compatibility Note

**NumPy Version**: Pynomaly uses `numpy>=1.26.0,<2.2.0` to ensure compatibility with TensorFlow and other ML libraries. This constraint supports the latest NumPy features while maintaining compatibility with the broader ML ecosystem.

## Quick Start

### CLI Usage

```bash
# Show all available commands
pynomaly --help

# List available algorithms
pynomaly detector algorithms

# Create a detector
pynomaly detector create --name "My Detector" --algorithm IsolationForest

# Load a dataset
pynomaly dataset load data.csv --name "My Data"

# Train and detect
pynomaly detect train <detector_id> <dataset_id>
pynomaly detect run <detector_id> <dataset_id>

# Start web UI server
pynomaly server start
```

### Python API

```python
from pynomaly.infrastructure.config import create_container
from pynomaly.domain.entities import Detector, Dataset
from pynomaly.application.use_cases import DetectAnomalies, TrainDetector
import pandas as pd
import asyncio

async def main():
    # Initialize dependency injection container
    container = create_container()
    
    # Create detector with algorithm-specific parameters
    detector = Detector(
        name="Isolation Forest Detector",
        algorithm_name="IsolationForest",
        parameters={
            "contamination": 0.1,
            "n_estimators": 100,
            "random_state": 42
        }
    )
    
    # Load and prepare dataset
    data = pd.read_csv("data.csv")
    dataset = Dataset(
        name="Sensor Data",
        data=data,
        target_column="anomaly_label"  # Optional for supervised learning
    )
    
    # Use clean architecture use cases
    train_use_case = container.train_detector_use_case()
    detect_use_case = container.detect_anomalies_use_case()
    
    # Train detector
    training_result = await train_use_case.execute(detector, dataset)
    print(f"Training completed: {training_result.metrics}")
    
    # Detect anomalies
    detection_result = await detect_use_case.execute(detector.id, dataset)
    print(f"Found {len(detection_result.anomalies)} anomalies")
    
    # Get explanations (if supported by algorithm)
    explainer = container.explanation_service()
    explanations = await explainer.explain_anomalies(
        detector.id, detection_result.anomalies[:5]  # Top 5 anomalies
    )

# Run async code
asyncio.run(main())
```

### Web API & Interface

Access the API and Progressive Web App at http://localhost:8000 after starting the server.

**📚 Complete Setup Guide**: See [docs/WEB_API_SETUP_GUIDE.md](docs/WEB_API_SETUP_GUIDE.md) for detailed instructions across all environments.

**⚡ Quick Reference**: See [docs/API_QUICK_REFERENCE.md](docs/API_QUICK_REFERENCE.md) for commands and endpoints.

- **Real-time Dashboard**: Live anomaly detection with WebSocket updates
- **Interactive Visualizations**: D3.js custom charts and Apache ECharts statistical plots
- **Offline Capability**: Service worker enables offline operation and data caching
- **Installable PWA**: Install on desktop and mobile devices like a native app
- **HTMX Simplicity**: Server-side rendering with minimal JavaScript complexity
- **Modern UI**: Tailwind CSS for responsive, accessible design
- **Experiment Tracking**: Compare models, track performance metrics, A/B testing
- **Dataset Analysis**: Data quality reports, drift detection, feature importance

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

Pynomaly follows **Clean Architecture**, **Domain-Driven Design (DDD)**, and **Hexagonal Architecture (Ports & Adapters)**:

```
src/pynomaly/
├── domain/          # Pure business logic (no external dependencies)
│   ├── entities/    # Anomaly, Detector, Dataset, Score, DetectionResult
│   ├── value_objects/ # ContaminationRate, ConfidenceInterval, AnomalyScore
│   ├── services/    # Core detection logic, scoring algorithms
│   └── exceptions/  # Domain-specific exception hierarchy
├── application/     # Orchestrate use cases without implementation details
│   ├── use_cases/   # DetectAnomalies, TrainDetector, EvaluateModel, ExplainAnomaly
│   ├── services/    # DetectionService, EnsembleService, ModelPersistenceService
│   └── dto/         # Data transfer objects and request/response models
├── infrastructure/  # All external integrations and adapters
│   ├── adapters/    # PyODAdapter, TODSAdapter, PyGODAdapter, SklearnAdapter
│   ├── persistence/ # ModelRepository, ResultRepository, data sources
│   ├── config/      # Dependency injection container, settings
│   └── monitoring/  # OpenTelemetry, Prometheus metrics, observability
└── presentation/    # User interfaces and external APIs
    ├── api/         # FastAPI REST endpoints with async support
    ├── cli/         # Typer CLI with rich formatting
    └── web/         # Progressive Web App
        ├── static/  # CSS, JS, PWA assets (Tailwind, D3.js, ECharts)
        ├── templates/ # HTMX server-rendered templates
        └── assets/  # PWA manifest, service worker, icons
```

### Design Patterns

- **Repository Pattern**: Clean data access abstraction
- **Factory Pattern**: Algorithm instantiation and configuration
- **Strategy Pattern**: Pluggable detection algorithms
- **Observer Pattern**: Real-time detection notifications
- **Decorator Pattern**: Feature engineering pipeline
- **Chain of Responsibility**: Data preprocessing and validation

## Supported Algorithm Libraries

### PyOD (Python Outlier Detection)
- **Statistical**: Isolation Forest, Local Outlier Factor, One-Class SVM, MCD, PCA
- **Probabilistic**: GMM, COPOD, ECOD, Histogram-based, Sampling
- **Linear**: PCA, Kernel PCA, Robust Covariance, Feature Bagging
- **Proximity**: k-NN, Radius-based, Connectivity-based (COF), CBLOF
- **Neural Networks**: AutoEncoder, VAE, Deep SVDD, SO-GAAL, MO-GAAL

### TODS (Time-series Outlier Detection System)
- **Univariate**: Statistical tests, decomposition-based, prediction-based
- **Multivariate**: Matrix profile, tensor decomposition, deep learning
- **Streaming**: Online algorithms, change point detection, concept drift

### PyGOD (Python Graph Outlier Detection)
- **Node-level**: Anomalous node detection in graphs
- **Edge-level**: Anomalous edge and subgraph detection  
- **Graph-level**: Anomalous graph classification
- **Deep Learning**: Graph neural networks, graph autoencoders

### Scikit-learn Integration
- **Ensemble**: Isolation Forest, One-Class SVM
- **Neighbors**: Local Outlier Factor, Novelty detection
- **Clustering**: DBSCAN outliers, Gaussian Mixture
- **Covariance**: Elliptic Envelope, Robust Covariance

### Deep Learning Frameworks
- **PyTorch**: Custom neural architectures, GPU acceleration
- **TensorFlow**: Distributed training, TensorBoard integration
- **JAX**: High-performance computing, automatic differentiation

### Multi-Modal Detection
- **Tabular Data**: Traditional ML and statistical methods
- **Time Series**: Seasonal decomposition, LSTM, Transformers  
- **Graph Data**: GNN-based detection, network analysis
- **Text Data**: NLP-based anomaly detection, embedding methods

Run `pynomaly detector algorithms` to see all available algorithms with their parameters and performance characteristics.

## Development

### Environment Setup
```bash
# Install development dependencies
poetry install --with dev,test

# Activate virtual environment
poetry shell

# Install pre-commit hooks (optional)
pre-commit install
```

### Code Quality
```bash
# Run full test suite with coverage
poetry run pytest --cov=pynomaly --cov-report=html

# Type checking with strict mode
poetry run mypy --strict src/

# Code formatting
poetry run black src/ tests/
poetry run isort src/ tests/

# Linting
poetry run flake8 src/ tests/
poetry run bandit -r src/  # Security linting
```

### Testing
```bash
# Unit tests only
poetry run pytest tests/unit/

# Integration tests
poetry run pytest tests/integration/

# Property-based testing
poetry run pytest tests/property/

# Performance benchmarks
poetry run pytest benchmarks/
```

### Web API Server

#### Quick Start (Current Environment)
```bash
# Set Python path and start server
export PYTHONPATH=/path/to/Pynomaly/src
uvicorn pynomaly.presentation.api:app --host 0.0.0.0 --port 8000 --reload

# Or use the startup script
./scripts/start_api_bash.sh
```

#### PowerShell (Windows)
```powershell
# Set environment and start server
$env:PYTHONPATH = "C:\Users\your-user\Pynomaly\src"
uvicorn pynomaly.presentation.api:app --host 0.0.0.0 --port 8000 --reload

# Alternative: use Python module directly
python -m uvicorn pynomaly.presentation.api:app --host 0.0.0.0 --port 8000 --reload

# Or use PowerShell script (if available)
if (Test-Path "scripts\test_api_powershell.ps1") {
    pwsh -File scripts\test_api_powershell.ps1
}
```

#### Fresh Environment Setup
```bash
# Automated setup for new environments
./scripts/setup_fresh_environment.sh

# Manual setup
pip install --break-system-packages fastapi uvicorn pydantic structlog dependency-injector \
    numpy pandas scikit-learn pyod rich typer httpx aiofiles \
    pydantic-settings redis prometheus-client \
    opentelemetry-api opentelemetry-sdk opentelemetry-instrumentation-fastapi \
    jinja2 python-multipart passlib bcrypt prometheus-fastapi-instrumentator

export PYTHONPATH=/path/to/Pynomaly/src
uvicorn pynomaly.presentation.api:app --host 0.0.0.0 --port 8000
```

#### API Endpoints
Once running, access these endpoints:
- **Root API**: http://localhost:8000/
- **Interactive Documentation**: http://localhost:8000/api/docs
- **Health Check**: http://localhost:8000/api/health/
- **OpenAPI Schema**: http://localhost:8000/api/openapi.json

#### Testing Multiple Environments
```bash
# Comprehensive testing across environments
./scripts/test_all_environments.sh
```

### Development Commands
```bash
# Start development server with auto-reload (Unix/Linux/macOS)
export PYTHONPATH=/path/to/Pynomaly/src
uvicorn pynomaly.presentation.api:app --reload

# Windows PowerShell
$env:PYTHONPATH = "C:\path\to\Pynomaly\src"
uvicorn pynomaly.presentation.api:app --reload

# Build frontend assets
npm run build-css  # Tailwind CSS compilation
npm run watch-css  # Development with file watching

# Run CLI in development (after pip install -e .)
pynomaly --help

# Or alternative methods
python scripts/cli.py --help
python -m pynomaly.presentation.cli.app --help
```

## Contributing

We welcome contributions! Please see our [Contributing Guide](CONTRIBUTING.md) for details.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.