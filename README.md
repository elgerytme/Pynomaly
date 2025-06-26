# Pynomaly 🔍

[![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Build System: Hatch](https://img.shields.io/badge/build%20system-hatch-4051b5.svg)](https://hatch.pypa.io/)
[![Code style: ruff](https://img.shields.io/badge/code%20style-ruff-000000.svg)](https://github.com/astral-sh/ruff)
[![Type checked: mypy](https://www.mypy-lang.org/static/mypy_badge.svg)](https://mypy-lang.org/)
[![CI](https://github.com/yourusername/pynomaly/workflows/CI/badge.svg)](https://github.com/yourusername/pynomaly/actions)

State-of-the-art Python anomaly detection package targeting Python 3.11+ with clean architecture principles, integrating multiple ML libraries (PyOD, PyGOD, scikit-learn, PyTorch, TensorFlow, JAX) through a unified, production-ready interface.

**Built with modern Python tooling**: Hatch for build system and environment management, Ruff for lightning-fast linting and formatting, comprehensive CI/CD pipeline with automated testing and deployment.

## Features

- 🏗️ **Clean Architecture**: Domain-driven design with hexagonal architecture (Ports & Adapters)
- 🔌 **Multi-Library Integration**: Unified interface for PyOD, PyGOD, scikit-learn, PyTorch, TensorFlow, JAX
- 🚀 **Production Ready**: Async/await, Prometheus metrics, comprehensive monitoring, JWT authentication
- 🖥️ **Multiple Interfaces**: FastAPI REST API, Typer CLI, and Progressive Web App (PWA)
- 📊 **Modern Web UI**: HTMX + Tailwind CSS + D3.js + Apache ECharts with offline PWA capabilities
- 🧪 **Advanced Features**: AutoML, SHAP/LIME explainability, drift detection, active learning
- ⚡ **Multi-Modal Support**: Time-series, tabular, graph, and text anomaly detection
- 🛡️ **Type Safe**: 100% type coverage with mypy --strict, Pydantic validation
- 🔄 **Streaming & Batch**: Real-time processing with backpressure and large dataset support
- 🧰 **Extensible**: Plugin architecture with algorithm registry and custom adapters
- ✅ **Test-Driven Development**: Active TDD enforcement with 85% coverage threshold
- 🔧 **Modern Environment Management**: Centralized virtual environments with automated tooling

## Installation

### Quick Setup (Hatch - Recommended)

Pynomaly uses [Hatch](https://hatch.pypa.io/) for modern Python project management with automatic environment handling:

#### Prerequisites
```bash
# Install Hatch (one-time setup)
pip install hatch

# Verify installation
hatch --version
```

#### Automated Setup
```bash
# Clone and setup (handles everything automatically)
git clone https://github.com/yourusername/pynomaly.git
cd pynomaly

# Initialize project environments
make setup

# Install in development mode
make dev-install

# Run tests to verify installation
make test
```

#### Manual Hatch Setup
```bash
# Create and activate environments
hatch env create
hatch env show

# Install with specific feature sets
hatch env run dev:setup          # Development environment
hatch env run prod:setup         # Production environment  
hatch env run test:setup         # Testing environment

# Or install specific extras
hatch env run -e ml pip install -e ".[torch,tensorflow]"
hatch env run -e api pip install -e ".[api,cli]"
```

### Alternative Setup (Traditional pip/venv)

If you prefer traditional Python environment management, Pynomaly uses a centralized environment structure:

#### Quick Start
```bash
# Create virtual environment in organized directory structure
mkdir -p environments
python -m venv environments/.venv

# Activate environment
# Linux/macOS:
source environments/.venv/bin/activate
# Windows:
environments\.venv\Scripts\activate

# Install with desired features
pip install -e ".[server]"          # API + CLI + basic features
pip install -e ".[all]"             # All available features
```

**Environment Organization**: Pynomaly uses a centralized `environments/` directory with dot-prefix naming (`.venv`, `.test_env`) to keep the project root clean and organize all virtual environments in one location.

#### Feature-Specific Installation
```bash
# Core functionality only
pip install -e .

# ML frameworks
pip install -e ".[torch]"           # PyTorch deep learning
pip install -e ".[tensorflow]"      # TensorFlow neural networks
pip install -e ".[jax]"             # JAX high-performance computing
pip install -e ".[graph]"           # PyGOD graph anomaly detection
pip install -e ".[ml-all]"          # All ML frameworks

# Data processing
pip install -e ".[data-formats]"    # Parquet, Excel, HDF5 support
pip install -e ".[database]"        # SQL database connectivity
pip install -e ".[spark]"           # Apache Spark integration

# Advanced features
pip install -e ".[automl]"          # AutoML with Optuna and auto-sklearn2
pip install -e ".[explainability]" # SHAP/LIME model explanation
pip install -e ".[production]"      # Full production stack with monitoring

# Development
pip install -e ".[test]"            # Testing dependencies
pip install -e ".[ui-test]"         # UI testing with Playwright
pip install -e ".[lint]"            # Code quality tools
pip install -e ".[dev]"             # Development tools
```

### Cross-Platform Compatibility

Pynomaly is designed to work seamlessly across different operating systems and environments:

**Supported Platforms:**
- **Linux/Unix**: Full compatibility with bash shell environments
- **macOS**: Complete support for all features and commands
- **Windows**: Full compatibility with PowerShell and Command Prompt
- **WSL/WSL2**: Tested and verified on Windows Subsystem for Linux

**Shell Compatibility:**
- **Bash**: All commands and scripts tested and verified
- **PowerShell**: Cross-platform PowerShell support (Core 6.0+)
- **Command Prompt**: Basic functionality available
- **Zsh/Fish**: Compatible with alternative Unix shells

**Python Environment Support:**
- **Virtual Environments**: `venv`, `virtualenv`, `conda`, `pipenv`, `poetry`
- **Python Versions**: 3.11, 3.12, 3.13+
- **Package Managers**: pip, conda, poetry, pipenv

**Installation Methods:**
- **Package Installation**: `pip install -e .` (cross-platform)
- **Development Setup**: Poetry-based development environment
- **Container Deployment**: Docker support for all platforms
- **Cloud Deployment**: AWS, Azure, GCP compatible

**Path Handling:**
- Automatic cross-platform path normalization
- Windows backslash (`\`) and Unix forward slash (`/`) support
- Environment variable handling across all platforms

**NumPy Compatibility**: Uses `numpy>=1.26.0,<2.2.0` to ensure compatibility with TensorFlow and other ML libraries across all platforms.

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

## Data Export & Reporting 📊

Pynomaly provides comprehensive data export capabilities for analysis and reporting:

### Supported Export Formats

- **Excel**: Advanced formatting with charts and multiple worksheets
- **CSV**: Standard comma-separated values format
- **JSON**: Structured data with metadata and annotations
- **Parquet**: Efficient columnar storage for large datasets

### Export Features

- **Rich Formatting**: Conditional formatting, charts, and visual highlighting
- **Metadata Inclusion**: Algorithm parameters, detection settings, timestamps
- **Batch Processing**: Efficient handling of large result sets
- **Custom Templates**: Configurable output formats and layouts

### CLI Usage

```bash
# List available export formats
pynomaly export list-formats

# Export to Excel with formatting
pynomaly export results.json report.xlsx --format excel --include-charts

# Export to multiple formats
pynomaly export results.json output --formats csv json excel
```

### Python API

```python
from pynomaly.application.services.export_service import ExportService

# Initialize export service
export_service = ExportService()

# Export with custom formatting
result = export_service.export_results(
    detection_results,
    "anomaly_report.xlsx",
    include_charts=True,
    highlight_anomalies=True
)
```

*Note: Advanced business intelligence integrations (Power BI, Google Sheets, Smartsheet) are planned for future releases.*

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

### Time Series Algorithms (Custom Implementation)
- **Statistical**: Rolling statistics, percentile-based detection
- **Decomposition**: Seasonal decomposition with trend analysis
- **Change Point**: Statistical tests for abrupt changes

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

### Modern Development Workflow (Hatch)

Pynomaly uses Hatch for streamlined development with automatic environment management:

#### Quick Start
```bash
# Initial setup
make setup              # Install Hatch and create environments
make dev-install        # Install in development mode
make pre-commit         # Setup pre-commit hooks

# Daily development workflow
make format             # Auto-format code with Ruff
make test               # Run core tests
make lint               # Check code quality
make ci                 # Full CI pipeline locally
```

#### Code Quality & Testing
```bash
# Testing
make test               # Core tests (domain + application)
make test-all           # All tests including integration
make test-cov           # Tests with coverage report
make test-unit          # Unit tests only
make test-integration   # Integration tests only

# Code Quality
make lint               # Run all quality checks
make format             # Auto-format with Ruff
make style              # Check style only
make typing             # Type checking with mypy

# Build & Package
make build              # Build wheel and source distribution
make version            # Show current version
make clean              # Clean build artifacts
```

#### Environment Management
```bash
# Environment commands
make env-show           # List all environments
make env-clean          # Clean and recreate environments
make status             # Show project status

# Direct Hatch commands
hatch env run test:run                    # Run tests
hatch env run lint:style                  # Code style check
hatch env run lint:fmt                    # Auto-format code
hatch env run prod:serve-api              # Start API server
hatch env run cli:run --help              # CLI help
```

#### Test-Driven Development (TDD)

Pynomaly has **active TDD enforcement** with comprehensive tooling:

```bash
# TDD Status & Management
pynomaly tdd status --detailed           # Check TDD compliance
pynomaly tdd enable --coverage 0.9       # Enable with custom coverage
pynomaly tdd disable                     # Disable TDD enforcement

# Test Requirements Management
pynomaly tdd require "src/module.py" "function_name" \
  --desc "Test description" \
  --spec "Detailed specification" \
  --coverage 0.9 --tags "core,domain"

pynomaly tdd requirements --status pending    # List pending requirements
pynomaly tdd requirements --module "src/auth.py"  # Filter by module

# Validation & Compliance
pynomaly tdd validate --coverage --fix    # Validate with auto-fix
pynomaly tdd coverage --threshold 0.85    # Coverage analysis
pynomaly tdd report --format html         # Generate reports

# Git Integration
pynomaly tdd hooks --install --pre-commit # Install git hooks
pynomaly tdd hooks --status               # Check hook status
```

**Current Status**: TDD is **enabled** with 85% coverage threshold, enforcing test-first development for domain and application layers.

### Legacy Development (Poetry)

For those preferring Poetry:

```bash
# Setup
poetry install --with dev,test
poetry shell

# Quality & Testing
poetry run pytest --cov=pynomaly --cov-report=html
poetry run mypy --strict src/
poetry run ruff check src/ tests/
poetry run ruff format src/ tests/
```

### Web API & CLI

#### Modern Hatch Commands
```bash
# API Server
make prod-api           # Production API server
make prod-api-dev       # Development API server with reload
hatch env run prod:serve-api-prod    # Direct Hatch command

# CLI Interface
make cli-help           # Show CLI help
hatch env run cli:run --help         # Direct CLI access
hatch env run cli:test-cli           # Test CLI functionality
```

#### Traditional Methods
```bash
# Method 1: Direct uvicorn (after installation)
uvicorn pynomaly.presentation.api.app:app --host 0.0.0.0 --port 8000 --reload

# Method 2: Python module
python -m uvicorn pynomaly.presentation.api.app:app --reload

# Method 3: With environment setup
export PYTHONPATH=/path/to/Pynomaly/src
uvicorn pynomaly.presentation.api.app:app --reload
```

#### API Endpoints
Once running, access these endpoints:
- **Root API**: http://localhost:8000/
- **Interactive Documentation**: http://localhost:8000/docs
- **Health Check**: http://localhost:8000/api/health/
- **OpenAPI Schema**: http://localhost:8000/openapi.json
- **Progressive Web App**: http://localhost:8000/app

#### Frontend Development
```bash
# Install frontend dependencies
npm install -D tailwindcss @tailwindcss/forms @tailwindcss/typography
npm install htmx.org d3 echarts

# Build assets
npm run build-css       # Production build
npm run watch-css       # Development with watch mode

# PWA development
python -m http.server 8080 --directory src/pynomaly/presentation/web/static
```

#### CLI Usage
```bash
# After installation
pynomaly --help
pynomaly detector algorithms
pynomaly server start

# Alternative methods
python scripts/pynomaly_cli.py --help
python -m pynomaly.presentation.cli.app --help
```

## Contributing

We welcome contributions! Please see our [Contributing Guide](CONTRIBUTING.md) for details.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.