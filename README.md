# Pynomaly 🔍

[![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Build System: Hatch](https://img.shields.io/badge/build%20system-hatch-4051b5.svg)](https://hatch.pypa.io/)
[![Code style: ruff](https://img.shields.io/badge/code%20style-ruff-000000.svg)](https://github.com/astral-sh/ruff)
[![Type checked: mypy](https://www.mypy-lang.org/static/mypy_badge.svg)](https://mypy-lang.org/)
[![CI](https://github.com/yourusername/pynomaly/workflows/CI/badge.svg)](https://github.com/yourusername/pynomaly/actions)

Python anomaly detection package targeting Python 3.11+ with clean architecture
principles, integrating multiple ML libraries (PyOD, PyGOD, scikit-learn, PyTorch,
TensorFlow, JAX) through a unified interface.

**Built with**: Hatch for build system and environment management, Ruff for linting
and formatting, CI/CD pipeline with automated testing and deployment.

## Features

### Core Features (Stable)

- 🏗️ **Clean Architecture**: Domain-driven design with hexagonal architecture
- 🔌 **PyOD Integration**: Production-ready PyOD algorithms (40+ algorithms including
  Isolation Forest, LOF, One-Class SVM)
- 🧪 **scikit-learn Support**: Standard ML algorithms for anomaly detection
- 📊 **Web Interface**: HTMX-based UI with Tailwind CSS styling
- ⚡ **CLI Interface**: Command-line tools for data processing and detection
- 🛡️ **Type Safe**: Comprehensive type coverage with mypy strict mode
- ✅ **Testing**: Comprehensive test suite with high coverage

### Advanced Features (Production Ready)

- 🚀 **FastAPI REST API**: 65+ API endpoints with OpenAPI documentation
- 🔐 **Authentication**: JWT-based authentication framework (optional)
- 📈 **Monitoring**: Prometheus metrics collection capabilities
- 💾 **Data Export**: CSV/JSON/Excel export functionality
- 🎯 **Ensemble Methods**: Advanced voting strategies and model combination
- ⚡ **Performance Optimizations**: Batch cache operations, optimized data loading,
  memory management
- 🧪 **Testing Infrastructure**: 85%+ coverage with property-based testing,
  benchmarking, and mutation testing

### Experimental Features (Limited Support)

**NOTE:** The following features are marked as experimental and their implementations
might be incomplete or not available:

- **ONNX model format is not supported yet in the storage service**
- **Certain PyTorch base models are placeholders without concrete implementations**
- 🤖 **AutoML**: Hyperparameter optimization framework (requires additional setup)
- 🔍 **Explainability**: SHAP/LIME integration (requires: `pip install shap lime`)
- 🧠 **Deep Learning**: PyTorch/TensorFlow adapters (optional dependencies)
- 📱 **PWA Features**: Progressive Web App capabilities (basic implementation)
- 📊 **Graph Analysis**: PyGOD integration (experimental, requires additional setup)

## Installation

### Quick Setup (Recommended)

#### Prerequisites

```bash
# Ensure Python 3.11+ is installed
python --version  # Should show 3.11 or higher
```

#### Installation

```bash
# Clone the repository
git clone https://github.com/yourusername/pynomaly.git
cd pynomaly

# Create virtual environment
python -m venv environments/.venv

# Activate environment
# Linux/macOS:
source environments/.venv/bin/activate
# Windows:
environments\.venv\Scripts\activate

# Install with basic features
pip install -e .

# Verify installation
python -c "import pynomaly; print('Installation successful')"
```

#### Feature Installation Options

```bash
# 🎯 Quick Start (recommended for most users)
pip install -e ".[server]"       # Complete server (CLI + API + web)

# 🧪 Research & ML  
pip install -e ".[server,automl,explainability]"

# 🚀 Production Deployment
pip install -e ".[production]"   # Authentication + monitoring

# 🛠️ Interactive Installer (recommended)
python scripts/setup/install_features.py
```

**📚 Complete Guide**: See [Feature Installation Guide](docs/getting-started/FEATURE_INSTALLATION_GUIDE.md) for detailed options and troubleshooting.

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
- **Development Setup**: Hatch-based development environment
- **Container Deployment**: Docker support for all platforms
- **Cloud Deployment**: AWS, Azure, GCP compatible

### Development Setup

```bash
# Quick development environment setup

# Linux/macOS with make:
make dev

# Or use the development script directly:
# Linux/macOS:
./scripts/dev_setup.sh

# Windows PowerShell:
.\scripts\dev_setup.ps1

# Or run the commands manually:
hatch env create dev
hatch run dev:setup
```

This will:
- Create an isolated `.venv` under `environments/`
- Install Pynomaly in editable mode with all `[dev]` extras
- Register pre-commit hooks automatically

**Path Handling:**

- Automatic cross-platform path normalization
- Windows backslash (`\`) and Unix forward slash (`/`) support
- Environment variable handling across all platforms

**NumPy Compatibility**: Uses `numpy>=1.26.0,<2.2.0` to ensure compatibility with TensorFlow and other ML libraries across all platforms.

## Quick Start

### Verified Run Scripts

All run scripts in `scripts/run/` are tested and working:

```bash
# CLI Interface
python scripts/run/cli.py --help                    # Main CLI interface  ✅
python scripts/run/run_pynomaly.py --help          # Alternative CLI entry  ✅
python scripts/run/run_cli.py --help               # Streamlined CLI  ✅

# API Server
python scripts/run/run_api.py --help               # FastAPI server  ✅
python scripts/run/run_api.py --port 8080          # Run on custom port
python scripts/run/run_api.py --reload             # Development mode

# Web Application
python scripts/run/run_app.py --help               # Complete app runner  ✅
python scripts/run/run_app.py --mode api           # API only
python scripts/run/run_app.py --mode cli detect    # CLI mode

# Web UI
python scripts/run/run_web_app.py --help           # Combined web app  ✅
python scripts/run/run_web_ui.py --help            # UI server only  ✅
python scripts/run/run_web_ui.py --dev             # Development mode

# Legacy CLI (Full-featured)
python scripts/run/pynomaly_cli.py help            # Comprehensive CLI tools  ✅
```

**Status**: All scripts tested and verified working in both current and fresh environments.

### CLI Usage

```bash
# Show all available commands
pynomaly --help

# Basic system information
pynomaly version          # Show version info
pynomaly status           # System status

# Dataset operations
pynomaly dataset --help   # Dataset management commands

# Detector operations  
pynomaly detector --help  # Detector management commands

# Performance monitoring and optimization
pynomaly perf benchmark --suite quick    # Run performance benchmarks
pynomaly perf monitor                     # Real-time performance monitoring
pynomaly perf report --format html       # Generate performance reports

# Start web interface (if web features installed)
pynomaly server start --port 8000
```

**Note**: Some CLI commands are experimental. Use `--help` with each command to see current options and availability.

### Python API

```python
import pandas as pd
import numpy as np
from pynomaly.domain.entities import Dataset
from pynomaly.domain.value_objects import ContaminationRate
from pynomaly.infrastructure.adapters.sklearn_adapter import SklearnAdapter

# Basic usage example with Pynomaly's SklearnAdapter
def basic_example():
    # Create sample data with outliers
    np.random.seed(42)
    normal_data = np.random.normal(0, 1, (100, 2))
    outliers = np.random.uniform(-4, 4, (10, 2))
    data = np.vstack([normal_data, outliers])

    # Create dataset
    df = pd.DataFrame(data, columns=['feature1', 'feature2'])
    dataset = Dataset(name="Sample Data", data=df)

    # Create detector using Pynomaly's clean architecture
    detector = SklearnAdapter(
        algorithm_name="IsolationForest",
        name="Basic Detector",
        contamination_rate=ContaminationRate(0.1),
        random_state=42,
        n_estimators=100
    )

    # Train detector
    detector.fit(dataset)

    # Detect anomalies
    result = detector.detect(dataset)

    # Results
    anomaly_count = len(result.anomalies)
    scores = [score.value for score in result.scores]
    print(f"Detected {anomaly_count} anomalies out of {len(data)} samples")
    print(f"Anomaly scores range: {min(scores):.3f} to {max(scores):.3f}")
    print(f"Detection completed in {result.execution_time_ms:.2f}ms")

    return result.labels, scores

# Run example
if __name__ == "__main__":
    predictions, scores = basic_example()
    print("Example completed successfully!")
```

### Web API & Interface

Access the API and Progressive Web App at <http://localhost:8000> after starting the server.

**📚 Complete Setup Guide**: See [docs/developer-guides/api-integration/WEB_API_SETUP_GUIDE.md](docs/developer-guides/api-integration/WEB_API_SETUP_GUIDE.md) for detailed instructions across all environments.

**⚡ Quick Reference**: See [docs/developer-guides/api-integration/API_QUICK_REFERENCE.md](docs/developer-guides/api-integration/API_QUICK_REFERENCE.md) for commands and endpoints.

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
│   ├── entities/    # Anomaly, Detector, Dataset, DetectionResult, Model, Experiment
│   ├── value_objects/ # ContaminationRate, ConfidenceInterval, AnomalyScore
│   ├── services/    # Core detection logic, scoring algorithms
│   └── exceptions/  # Domain-specific exception hierarchy
├── application/     # Orchestrate use cases without implementation details
│   ├── use_cases/   # DetectAnomalies, TrainDetector, EvaluateModel, ExplainAnomaly
│   ├── services/    # DetectionService, EnsembleService, ModelPersistenceService, AutoMLService
│   └── dto/         # Data transfer objects and request/response models
├── infrastructure/  # All external integrations and adapters
│   ├── adapters/    # PyODAdapter, PyGODAdapter, SklearnAdapter, TimeSeriesAdapter
│   ├── persistence/ # ModelRepository, ResultRepository, data sources
│   ├── config/      # Dependency injection container, settings
│   └── monitoring/  # Prometheus metrics, health checks, observability
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

### Development Setup

```bash
# Clone and setup development environment
git clone https://github.com/yourusername/pynomaly.git
cd pynomaly

# Create development environment
python -m venv environments/.venv
source environments/.venv/bin/activate  # Linux/macOS
# or environments\.venv\Scripts\activate  # Windows

# Install development dependencies
pip install -e ".[dev,test]"

# Install pre-commit hooks (optional)
pip install pre-commit
pre-commit install
```

#### Code Quality & Testing

```bash
# Run tests
pytest tests/                    # Run all tests
pytest tests/unit/               # Unit tests only
pytest tests/integration/        # Integration tests only
pytest --cov=src/pynomaly       # Test coverage

# Code quality
ruff check src/                  # Linting (actively maintained, ~9K issues resolved)
ruff format src/                 # Auto-formatting
mypy src/pynomaly               # Type checking (strict mode enabled)
hatch run lint:all              # Run all quality checks

# Build package
python -m build                  # Build distribution
```

#### Web Development

```bash
# Start development server
uvicorn pynomaly.presentation.api.app:app --reload --port 8000

# Access the application
# API Documentation: http://localhost:8000/docs
# Web Interface: http://localhost:8000/app
```

#### Testing Framework

Pynomaly includes comprehensive testing:

```bash
# Run different test suites
pytest tests/unit/              # Fast unit tests
pytest tests/integration/       # Integration tests
pytest tests/e2e/              # End-to-end tests (if available)

# Coverage reporting
pytest --cov=src/pynomaly --cov-report=html

# Test specific areas
pytest tests/unit/domain/       # Domain layer tests
pytest tests/unit/application/  # Application layer tests
```

**Testing Status**: Comprehensive test suite with **82.5% line coverage**, **88.1% function coverage**, and **91.3% class coverage**. Test suite includes 324 test files covering unit, integration, performance, and security testing.

### Web API & CLI

```bash
# API Server
make prod-api-dev       # Development server with reload
uvicorn pynomaly.presentation.api.app:app --reload

# Endpoints
# http://localhost:8000/docs - Interactive docs
# http://localhost:8000/app - Progressive Web App

# Frontend
npm install htmx.org d3 echarts tailwindcss
npm run build-css
```

## Development Status

**Pynomaly is actively developed with the following implementation status:**

### ✅ Stable Features

- **Core anomaly detection**: PyOD integration with 40+ algorithms
- **Basic web interface**: HTMX-based UI with Tailwind CSS
- **CLI tools**: Basic dataset and detector management
- **Clean architecture**: Domain-driven design implementation
- **API foundation**: FastAPI with 65+ endpoints

### ⚠️ Beta Features

- **Authentication**: JWT framework (requires configuration)
- **Monitoring**: Prometheus metrics (optional)
- **Export functionality**: CSV/JSON export
- **Ensemble methods**: Advanced voting strategies

### 🚧 Experimental Features

- **AutoML**: Requires additional setup and dependencies
- **Deep Learning**: PyTorch/TensorFlow adapters (optional install)
- **Explainability**: SHAP/LIME integration (manual setup required)
- **PWA features**: Basic Progressive Web App capabilities
- **Real-time streaming**: Framework exists, limited functionality

### ❌ Planned Features

- **Graph anomaly detection**: PyGOD integration in development
- **Advanced visualization**: Complex D3.js components
- **Production monitoring**: Full observability stack
- **Text anomaly detection**: NLP-based detection methods

See [CHANGELOG.md](CHANGELOG.md) for detailed progress and [TODO.md](docs/project/TODO.md) for planned features.

## Important Notes

- **Optional Dependencies**: Many advanced features require additional packages (`pip install shap lime torch tensorflow`)
- **Configuration Required**: Some features need manual setup (authentication, monitoring)
- **Platform Support**: Tested on Linux/macOS/Windows with Python 3.11+
- **Documentation**: Some documentation may describe planned rather than implemented features

## Contributing

We welcome contributions! Please see our [Contributing Guide](docs/developer-guides/contributing/CONTRIBUTING.md) for details.

### Open Feature Gaps

The following features are planned but currently not implemented or incomplete:

- **Audit Storage:** Methods in `AuditStorage` class for storing, retrieving, and deleting events.
- **ONNX Model Support:** The `save_model` method in `ModelPersistenceService` currently raises NotImplementedError for the 'onnx' format.
- **Deep Learning Models:** Some methods in `PyTorchAdapter` require implementation for specific neural network architectures.

Feel free to pick any feature from this list to contribute or suggest your own improvements!

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
