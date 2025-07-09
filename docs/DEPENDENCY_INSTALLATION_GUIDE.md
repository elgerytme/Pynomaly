# Pynomaly Dependency Installation Guide

This guide provides comprehensive instructions for installing Pynomaly dependencies based on your specific use case and environment.

## Quick Start

### Basic Installation
```bash
pip install pynomaly
```

### Installation with Common Extras
```bash
# CLI tools with basic ML support
pip install pynomaly[cli,minimal]

# Full server with all features
pip install pynomaly[server]

# Development environment
pip install pynomaly[dev,test,lint]
```

## Installation Options by Use Case

### 1. Core Package (Minimal)
For basic anomaly detection with PyOD:
```bash
pip install pynomaly
```

**Includes:**
- PyOD (Python Outlier Detection)
- NumPy, Pandas, Polars
- Pydantic, StructLog
- NetworkX

### 2. CLI Usage
For command-line interface:
```bash
pip install pynomaly[cli]
```

**Additional dependencies:**
- Typer (CLI framework)
- Rich (formatted output)

### 3. Machine Learning Extended
For advanced ML algorithms:
```bash
pip install pynomaly[ml]
```

**Additional dependencies:**
- scikit-learn
- scipy

### 4. Web API/Server
For serving APIs:
```bash
pip install pynomaly[server]
```

**Additional dependencies:**
- FastAPI
- Uvicorn
- HTTP clients (httpx, requests)
- Jinja2 templates

### 5. Deep Learning Support

#### PyTorch
```bash
pip install pynomaly[torch]
```

#### TensorFlow
```bash
pip install pynomaly[tensorflow]
```

#### JAX
```bash
pip install pynomaly[jax]
```

#### All Deep Learning Frameworks
```bash
pip install pynomaly[ml-all]
```

### 6. Specialized Features

#### AutoML & Hyperparameter Optimization
```bash
pip install pynomaly[automl]
```

**Includes:**
- Optuna
- Hyperopt
- Auto-sklearn2

#### Explainable AI
```bash
pip install pynomaly[explainability]
```

**Includes:**
- SHAP
- LIME

#### Graph Anomaly Detection
```bash
pip install pynomaly[graph]
```

**Includes:**
- PyGOD
- PyTorch Geometric

#### Data Format Support
```bash
pip install pynomaly[data-formats]
```

**Includes:**
- PyArrow, FastParquet
- OpenPyXL, XlsxWriter
- HDF5 support

### 7. Production Deployment
```bash
pip install pynomaly[production]
```

**Includes:**
- FastAPI + Uvicorn
- Redis caching
- OpenTelemetry monitoring
- Prometheus metrics
- Authentication (JWT, passlib)

### 8. Complete Installation
```bash
pip install pynomaly[all]
```

**Includes all optional dependencies** (large installation ~2GB)

## Fixing Common Test Failures

### Missing PyTorch
```bash
# Error: "please install torch first"
pip install pynomaly[torch]
# or
pip install torch
```

### Missing SHAP/LIME
```bash
# Error: "SHAP not available. Install with: pip install shap"
pip install pynomaly[explainability]
# or
pip install shap lime
```

### Missing Optuna
```bash
# Error: "Optuna not available. Install with: pip install optuna"
pip install pynomaly[automl]
# or
pip install optuna
```

### Missing JAX
```bash
# Error: "JAX is required for JAXAdapter. Install with: pip install jax jaxlib"
pip install pynomaly[jax]
# or
pip install jax jaxlib
```

## Development Setup

### Complete Development Environment
```bash
# Clone the repository
git clone https://github.com/pynomaly/pynomaly.git
cd pynomaly

# Create virtual environment
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate

# Install in development mode with all dev tools
pip install -e .[dev,test,lint,docs]

# Setup pre-commit hooks
pre-commit install
```

### Testing Environment
```bash
# Install test dependencies
pip install pynomaly[test]

# Run tests
pytest

# Run tests with coverage
pytest --cov=pynomaly --cov-report=html
```

## Platform-Specific Instructions

### Linux/macOS
```bash
# Standard installation
pip install pynomaly[cli,ml]

# With CUDA support (if available)
pip install pynomaly[torch]
pip install torch --index-url https://download.pytorch.org/whl/cu118
```

### Windows
```bash
# Standard installation
pip install pynomaly[cli,ml]

# For Windows with CUDA
pip install pynomaly[torch]
pip install torch --index-url https://download.pytorch.org/whl/cu118
```

### Apple Silicon (M1/M2 Macs)
```bash
# Use conda for better compatibility
conda install -c conda-forge pynomaly
# or
pip install pynomaly[cli,ml]

# TensorFlow on Apple Silicon
pip install pynomaly[tensorflow]
```

## Docker Installation

### Basic Container
```dockerfile
FROM python:3.11-slim

RUN pip install pynomaly[cli,ml]

CMD ["pynomaly", "--help"]
```

### Production Container
```dockerfile
FROM python:3.11-slim

COPY requirements.txt .
RUN pip install pynomaly[production]

EXPOSE 8000
CMD ["pynomaly", "server", "start"]
```

## Conda Environment

### Create Environment
```bash
# Create conda environment
conda create -n pynomaly python=3.11
conda activate pynomaly

# Install from conda-forge (if available)
conda install -c conda-forge pynomaly

# Or install with pip in conda
pip install pynomaly[cli,ml]
```

### Environment File
```yaml
# environment.yml
name: pynomaly
channels:
  - conda-forge
  - pytorch
dependencies:
  - python=3.11
  - pip
  - pip:
    - pynomaly[cli,ml,explainability]
```

```bash
conda env create -f environment.yml
```

## Troubleshooting

### Common Issues

1. **CUDA/GPU Issues**
   ```bash
   # Check CUDA availability
   python -c "import torch; print(torch.cuda.is_available())"
   
   # Install CUDA-compatible PyTorch
   pip install torch --index-url https://download.pytorch.org/whl/cu118
   ```

2. **Memory Issues During Installation**
   ```bash
   # Install without cache
   pip install --no-cache-dir pynomaly[all]
   
   # Install one extra at a time
   pip install pynomaly[torch]
   pip install pynomaly[tensorflow]
   ```

3. **Permission Issues**
   ```bash
   # Install in user directory
   pip install --user pynomaly[cli,ml]
   
   # Or use virtual environment
   python -m venv pynomaly-env
   source pynomaly-env/bin/activate
   pip install pynomaly[cli,ml]
   ```

4. **Version Conflicts**
   ```bash
   # Create clean environment
   pip-tools compile requirements.in
   pip-sync requirements.txt
   
   # Or use fresh virtual environment
   python -m venv fresh-env
   source fresh-env/bin/activate
   pip install pynomaly[cli,ml]
   ```

### Dependency Resolution
```bash
# Check installed packages
pip list | grep -E "(pynomaly|torch|tensorflow|jax|shap|lime|optuna)"

# Check for conflicts
pip check

# Upgrade all packages
pip install --upgrade pynomaly[all]
```

## Recommended Combinations

### Data Scientist
```bash
pip install pynomaly[cli,ml,explainability,data-formats]
```

### ML Engineer
```bash
pip install pynomaly[server,ml-all,monitoring]
```

### DevOps/Production
```bash
pip install pynomaly[production,monitoring,infrastructure]
```

### Researcher
```bash
pip install pynomaly[all]
```

## Performance Optimization

### Faster Installation
```bash
# Use parallel downloads
pip install --upgrade pip
pip install pynomaly[cli,ml] --prefer-binary

# Pre-compiled wheels
pip install --only-binary=all pynomaly[ml]
```

### Reducing Installation Size
```bash
# Minimal functional installation
pip install pynomaly[minimal]

# Add features as needed
pip install pynomaly[cli]
pip install pynomaly[explainability]
```

## Environment Variables

Set these environment variables for optimal performance:

```bash
# PyTorch settings
export TORCH_HOME=/path/to/torch/cache
export CUDA_VISIBLE_DEVICES=0

# Pynomaly settings
export PYNOMALY_CACHE_DIR=/path/to/cache
export PYNOMALY_LOG_LEVEL=INFO
export PYNOMALY_STORAGE_PATH=/path/to/storage
```

## Verification

### Test Installation
```bash
# Basic functionality
python -c "import pynomaly; print('✓ Core package works')"

# CLI functionality
pynomaly --help
pynomaly version

# Feature testing
python -c "
try:
    import torch
    print('✓ PyTorch available')
except ImportError:
    print('✗ PyTorch not installed')

try:
    import shap
    print('✓ SHAP available')
except ImportError:
    print('✗ SHAP not installed')

try:
    import lime
    print('✓ LIME available')
except ImportError:
    print('✗ LIME not installed')
"
```

### Run Tests
```bash
# Quick smoke test
pynomaly detector create test_detector
pynomaly detector list

# Run test suite (if dev dependencies installed)
pytest tests/cli/test_cli_simple.py -v
```

This guide should help users install Pynomaly dependencies correctly based on their specific needs and resolve common installation issues.