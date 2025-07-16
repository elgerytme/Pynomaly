# Pynomaly 🔍

[![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Build System: Hatch](https://img.shields.io/badge/build%20system-hatch-4051b5.svg)](https://hatch.pypa.io/)
[![Code style: ruff](https://img.shields.io/badge/code%20style-ruff-000000.svg)](https://github.com/astral-sh/ruff)
[![Type checked: mypy](https://www.mypy-lang.org/static/mypy_badge.svg)](https://mypy-lang.org/)
[![CI](https://github.com/yourusername/pynomaly/workflows/CI/badge.svg)](https://github.com/yourusername/pynomaly/actions)
[![Security](https://github.com/yourusername/pynomaly/workflows/Security%20Scanning/badge.svg)](https://github.com/yourusername/pynomaly/actions)
[![Maintenance Status](https://github.com/yourusername/pynomaly/workflows/Scheduled%20Maintenance/badge.svg)](https://github.com/yourusername/pynomaly/actions)
[![Bandit](https://img.shields.io/badge/security-bandit-yellow)](https://bandit.readthedocs.io/)
[![Safety](https://img.shields.io/badge/safety-checked-green)](https://github.com/pyupio/safety)

Enterprise-ready anomaly detection platform with clean architecture, targeting Python 3.11+ and integrating multiple ML libraries (PyOD, PyGOD, scikit-learn, PyTorch, TensorFlow, JAX) through a unified interface.

**Built with**: Clean monorepo architecture, Hatch for build system and environment management, Ruff for linting and formatting, comprehensive CI/CD pipeline with automated testing and deployment.

## 🏗️ Monorepo Structure

This is a clean, enterprise-ready monorepo containing all Pynomaly packages, applications, and infrastructure organized for maximum modularity and maintainability.

```
pynomaly/
├── src/packages/           # 🎯 Core packages (domain-driven architecture)
│   ├── core/              # Domain logic & business rules  
│   ├── anomaly_detection/ # Consolidated anomaly & outlier detection
│   ├── machine_learning/  # ML operations, training & lifecycle
│   ├── people_ops/        # User management & authentication
│   ├── mathematics/       # Statistical analysis & computations
│   ├── data_platform/     # Data processing & quality pipeline
│   ├── infrastructure/    # Technical infrastructure adapters
│   ├── interfaces/        # User interfaces (CLI, API, Web)
│   ├── enterprise/        # Enterprise features & governance
│   ├── services/          # Application services
│   └── testing/           # Testing utilities
├── pkg/                   # 🔗 Third-party packages
│   ├── vendor_dependencies/ # Vendored dependencies
│   └── custom_forks/      # Custom package forks
├── scripts/              # 🛠️ Development & automation scripts
│   ├── governance/       # Repository organization enforcement
│   ├── analysis/         # Analysis and debugging tools
│   └── cleanup/          # Cleanup automation
├── templates/            # 📋 Standardized templates
│   └── package/         # Package structure templates
├── reports/             # 📊 Analysis reports
│   └── analysis/        # Repository analysis results
├── deployment/          # 🚀 Deployment configurations
├── configs/            # ⚙️ Configuration files
├── docs/               # 📚 Project documentation
└── tests/              # 🧪 Integration test suites
```

### 📦 Package Organization

All packages follow **Domain-Driven Design** and **Clean Architecture** principles:

#### 🏢 **Domain Packages** (Core Business Logic)
- **`core/`**: Fundamental domain logic, entities, value objects
- **`anomaly_detection/`**: Consolidated anomaly & outlier detection (40+ algorithms)
- **`mathematics/`**: Statistical analysis and mathematical computations

#### 🚀 **Application Packages** (Business Operations)
- **`machine_learning/`**: ML training, optimization, lifecycle management
- **`people_ops/`**: User management, authentication, authorization
- **`data_platform/`**: Data processing, quality, transformation pipelines
- **`enterprise/`**: Multi-tenancy, governance, compliance
- **`services/`**: Application services and use cases

#### 🔧 **Infrastructure Packages** (Technical Concerns)
- **`infrastructure/`**: Deployment, monitoring, persistence adapters
- **`interfaces/`**: CLI, API, Web UI (presentation layer)

Each package contains:
```
package_name/
├── package_name/     # Source code
├── tests/           # Package-specific tests
├── docs/            # Package documentation
├── README.md        # Package overview
├── pyproject.toml   # Package configuration
└── BUCK            # Build configuration
```

## 🚀 Quick Start

### Installation

```bash
# Install core functionality
pip install pynomaly-core

# Install with all features
pip install pynomaly[all]

# Install specific components
pip install pynomaly-api pynomaly-cli pynomaly-web
```

### Basic Usage

```python
from pynomaly.core import detect_anomalies, Dataset, Detector

# Load your data
dataset = Dataset.from_csv("data.csv")

# Create a detector
detector = Detector.isolation_forest()

# Detect anomalies
result = detect_anomalies(dataset, detector)
print(f"Found {len(result.anomalies)} anomalies")
```

## 🏢 Enterprise Features

- **🔐 Security**: RBAC, audit logging, SOC2 compliance
- **📊 Monitoring**: Prometheus metrics, distributed tracing
- **🌐 Multi-tenancy**: Complete data isolation and resource quotas
- **⚙️ MLOps**: Model lifecycle and lineage tracking
- **🔄 CI/CD**: Automated testing, deployment, and monitoring

## Automation and Maintenance

This project includes a robust scheduled maintenance workflow that automatically runs weekly checks to ensure code quality and security. To run maintenance tasks locally, use the following commands:

```bash
# Run structure validation
python scripts/validation/validate_structure.py

# Run linting with ruff
ruff check src/ tests/

# Run type checking with MyPy
mypy src/pynomaly/

# Run Bandit security scan
bandit -r src/

# Run Safety vulnerability check
safety check --full-report

# Run pip-audit for package vulnerabilities
pip-audit
```

These tools ensure that the code meets the quality standards and is free from vulnerabilities.

## Features

### Core Features (Stable)

- 🏗️ **Clean Architecture**: Domain-driven design with hexagonal architecture
- 🔌 **PyOD Integration**: Production-ready PyOD algorithms (40+ algorithms including Isolation Forest, LOF, One-Class SVM)
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
- ⚡ **Performance Optimizations**: Batch cache operations, optimized data loading, memory management
- 🧪 **Testing Infrastructure**: 85%+ coverage with property-based testing, benchmarking, and mutation testing

### Experimental Features (Limited Support)

**📋 [View Complete Feature Status →](docs/reference/FEATURE_IMPLEMENTATION_STATUS.md)**

**⚠️ IMPORTANT - READ BEFORE USE**: The following features have significant implementation limitations. Many are frameworks or placeholders rather than complete implementations:

- 🤖 **AutoML**: Framework exists but requires `optuna` installation and ML expertise to configure
- 🔍 **Explainability**: Service architecture exists but most methods return placeholder/mock data  
- 🧠 **Deep Learning**: Advanced PyTorch adapter available, TensorFlow/JAX support is minimal
- 📱 **PWA Features**: Basic Progressive Web App, offline functionality limited
- 📊 **Graph Analysis**: **NOT IMPLEMENTED** - PyGOD integration mentioned but not functional

**❌ NOT IMPLEMENTED (Despite Documentation Claims)**:

- **Real-time Streaming**: Framework classes exist but no actual streaming functionality
- **Advanced Business Intelligence**: Export formatting and BI integrations are placeholder  
- **Enterprise LDAP/SAML**: Only basic JWT authentication implemented

**✅ RECOMMENDATION**: Start with [Core Features](#core-features-stable) which are production-ready. Consult the [Feature Implementation Status Guide](docs/reference/FEATURE_IMPLEMENTATION_STATUS.md) for accurate implementation details before depending on any experimental feature.

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
- **Development Setup**: Poetry-based development environment
- **Container Deployment**: Docker support for all platforms
- **Cloud Deployment**: AWS, Azure, GCP compatible

**Path Handling:**

- Automatic cross-platform path normalization
- Windows backslash (`\`) and Unix forward slash (`/`) support
- Environment variable handling across all platforms

**NumPy Compatibility**: Uses `numpy>=1.26.0,<2.2.0` to ensure compatibility with TensorFlow and other ML libraries across all platforms.

## Quick Start

### Getting Started Commands

After installation, use these commands to get started:

```bash
# Core functionality
pynomaly --help                    # Main CLI interface
pynomaly version                   # Show version information
pynomaly status                    # System status check

# Dataset operations
pynomaly dataset list              # List available datasets
pynomaly dataset create            # Create new dataset
pynomaly dataset analyze           # Analyze dataset properties

# Detector operations
pynomaly detector list             # List available detectors
pynomaly detector create           # Create new detector
pynomaly detector detect           # Run anomaly detection

# Web interface (if installed with [server])
pynomaly server start --port 8000 # Start web server
# Then visit: http://localhost:8000

# API server (if installed with [server])
pynomaly api start --reload       # Start API with auto-reload
# API docs: http://localhost:8000/docs
```

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

# Install pre-commit hooks for code quality
pre-commit install

# Verify setup
python -c "import pynomaly; print('Setup successful')"
```

#### Repository Organization

The repository includes automated governance to maintain organization:

```bash
# Run structure validation
python3 scripts/governance/package_structure_enforcer.py

# Check for build artifacts
python3 scripts/governance/build_artifacts_checker.py

# Validate root directory organization  
python3 scripts/governance/root_directory_checker.py

# Auto-fix common issues
python3 scripts/governance/package_structure_enforcer.py --fix
python3 scripts/governance/root_directory_checker.py --fix
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

### 🚧 Experimental Features (Limited/Placeholder Implementations)

- **AutoML**: Framework exists, requires optuna + ML expertise
- **Deep Learning**: PyTorch adapter advanced, TensorFlow/JAX minimal
- **Explainability**: Service architecture exists, most methods return mock data
- **PWA features**: Basic Progressive Web App, offline functionality limited

### ❌ Not Implemented (Despite Some Documentation Claims)

- **Graph anomaly detection**: PyGOD integration not functional
- **Real-time streaming**: Framework classes exist, no actual streaming
- **Advanced BI export**: Basic CSV/JSON works, formatting/BI integrations placeholder
- **Enterprise security**: Basic JWT only, no LDAP/SAML integration
- **Advanced visualization**: Basic charts work, complex D3.js components limited

See [CHANGELOG.md](CHANGELOG.md) for detailed progress and [TODO.md](docs/project/TODO.md) for planned features.

## Important Notes

- **Feature Accuracy**: This README contains historical feature descriptions. For accurate implementation status, **always consult** the [Feature Implementation Status Guide](docs/reference/FEATURE_IMPLEMENTATION_STATUS.md)
- **Optional Dependencies**: Many advanced features require additional packages and may be placeholders (`pip install shap lime torch tensorflow`)
- **Configuration Required**: Most "advanced" features need manual setup, additional dependencies, or ML expertise
- **Platform Support**: Core features tested on Linux/macOS/Windows with Python 3.11+
- **Production Use**: Only [Core Features](#core-features-stable) and [Advanced Features (Production Ready)](#advanced-features-production-ready) are recommended for production

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
