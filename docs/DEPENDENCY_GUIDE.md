# Pynomaly Dependency Guide

This guide explains Pynomaly's modular dependency structure and how to install only what you need.

## Philosophy

Pynomaly follows a **minimal core + optional extras** approach:
- **Minimal core**: Only essential dependencies required for basic anomaly detection
- **Optional extras**: Additional functionality available through optional dependencies
- **Flexible installation**: Choose exactly what you need for your use case

## Core Required Dependencies

These dependencies are **always installed** with Pynomaly:

### Data & ML Core
- **`pyod`** - Primary anomaly detection library with 40+ algorithms
- **`numpy`** - Numerical computing foundation
- **`pandas`** - Data manipulation and analysis
- **`polars`** - High-performance DataFrame library

### Architecture Core  
- **`pydantic`** - Data validation and settings management
- **`structlog`** - Structured logging
- **`dependency-injector`** - Dependency injection framework

**Total core size**: ~50MB installed

## Optional Dependencies by Category

### ML & Data Processing

#### `minimal` - Basic ML Support
```bash
pip install pynomaly[minimal]
# or
poetry install -E minimal
```
**Adds**: scikit-learn, scipy
**Use for**: Basic ML algorithms, statistical functions
**Size**: +30MB

#### `ml` - Extended ML Support  
```bash
pip install pynomaly[ml]
```
**Adds**: scikit-learn, scipy
**Use for**: Standard ML workflows
**Size**: +30MB

#### Deep Learning Frameworks

##### `torch` - PyTorch Support
```bash
pip install pynomaly[torch]
```
**Adds**: torch
**Use for**: Neural network-based anomaly detection
**Size**: +500MB

##### `tensorflow` - TensorFlow Support
```bash
pip install pynomaly[tensorflow]
```
**Adds**: tensorflow, keras
**Use for**: TensorFlow-based deep learning models
**Size**: +400MB

##### `jax` - JAX Support
```bash
pip install pynomaly[jax]
```
**Adds**: jax, jaxlib, optax
**Use for**: High-performance numerical computing
**Size**: +200MB

#### Specialized ML

##### `graph` - Graph Anomaly Detection
```bash
pip install pynomaly[graph]
```
**Adds**: pygod, torch-geometric
**Use for**: Graph neural networks, network anomaly detection
**Size**: +600MB (includes PyTorch)

##### `automl` - Automated ML
```bash
pip install pynomaly[automl]  
```
**Adds**: optuna, hyperopt, auto-sklearn2, scikit-learn
**Use for**: Automated algorithm selection and hyperparameter tuning
**Size**: +100MB

##### `explainability` - Model Explanation
```bash
pip install pynomaly[explainability]
```
**Adds**: shap, lime
**Use for**: Model interpretation and explanation
**Size**: +50MB

### Web & API

#### `api` - Web API
```bash
pip install pynomaly[api]
```
**Adds**: fastapi, uvicorn, httpx, requests, python-multipart, jinja2, aiofiles, pydantic-settings
**Use for**: REST API, web services
**Size**: +20MB

#### `cli` - Command Line Interface
```bash
pip install pynomaly[cli]
```
**Adds**: typer, rich
**Use for**: Interactive command-line tools
**Size**: +5MB

### Infrastructure & Production

#### `auth` - Authentication
```bash
pip install pynomaly[auth]
```
**Adds**: pyjwt, passlib
**Use for**: JWT tokens, password hashing
**Size**: +5MB

#### `caching` - Redis Caching
```bash
pip install pynomaly[caching]
```
**Adds**: redis
**Use for**: Performance optimization, session storage
**Size**: +3MB

#### `monitoring` - Observability
```bash
pip install pynomaly[monitoring]
```
**Adds**: opentelemetry-api, opentelemetry-sdk, opentelemetry-instrumentation-fastapi, prometheus-client, psutil
**Use for**: Metrics, tracing, performance monitoring
**Size**: +15MB

#### `infrastructure` - Resilience
```bash
pip install pynomaly[infrastructure]
```
**Adds**: tenacity, circuitbreaker, pydantic-settings
**Use for**: Retry logic, circuit breakers, configuration
**Size**: +3MB

### Data Formats & Storage

#### `data-formats` - File Format Support
```bash
pip install pynomaly[data-formats]
```
**Adds**: pyarrow, fastparquet, openpyxl, xlsxwriter, h5py
**Use for**: Parquet, Excel, HDF5 file support
**Size**: +30MB

#### `database` - SQL Database Support
```bash
pip install pynomaly[database]
```
**Adds**: sqlalchemy, psycopg2-binary
**Use for**: PostgreSQL, SQL database connectivity
**Size**: +15MB

#### `spark` - Apache Spark
```bash
pip install pynomaly[spark]
```
**Adds**: pyspark
**Use for**: Big data processing with Spark
**Size**: +200MB

## Combined Installation Profiles

### `standard` - Basic Data Science
```bash
pip install pynomaly[standard]
```
**Includes**: minimal + pyarrow
**Use for**: Standard data science workflows
**Size**: Core + 60MB

### `server` - Web Server
```bash
pip install pynomaly[server]
```
**Includes**: api + cli + minimal + pyarrow
**Use for**: Web applications, API servers
**Size**: Core + 90MB

### `production` - Production Deployment
```bash
pip install pynomaly[production]
```
**Includes**: server + auth + caching + monitoring + infrastructure
**Use for**: Production deployments
**Size**: Core + 140MB

### `ml-all` - Complete ML Stack
```bash
pip install pynomaly[ml-all]
```
**Includes**: All ML frameworks and tools
**Use for**: ML research, comprehensive anomaly detection
**Size**: Core + 1.5GB

### `data-all` - Complete Data Stack
```bash
pip install pynomaly[data-all]
```
**Includes**: All data processing and storage options
**Use for**: Complex data pipelines, multiple data sources
**Size**: Core + 250MB

### `all` - Everything
```bash
pip install pynomaly[all]
```
**Includes**: All optional dependencies
**Use for**: Development, testing, maximum functionality
**Size**: Core + 2GB+

## Requirements Files

For non-Poetry installations, use these requirements files:

### Core Only
```bash
pip install -r requirements.txt
```
**Contents**: PyOD, NumPy, Pandas, Polars + core architecture
**Size**: ~50MB

### Minimal ML
```bash
pip install -r requirements-minimal.txt
```
**Contents**: Core + scikit-learn + scipy
**Size**: ~80MB

### Server Ready
```bash
pip install -r requirements-server.txt
```
**Contents**: Minimal + API + CLI functionality
**Size**: ~110MB

### Production Ready
```bash
pip install -r requirements-production.txt
```
**Contents**: Server + authentication + monitoring + caching
**Size**: ~160MB

## Installation Examples by Use Case

### Data Scientist (Local Analysis)
```bash
# Minimal setup for data exploration
pip install pynomaly[standard]

# With AutoML for automated analysis
pip install pynomaly[standard,automl]

# With explainability for model interpretation
pip install pynomaly[standard,automl,explainability]
```

### Web Developer (API Integration)
```bash
# Basic API development
pip install pynomaly[server]

# Production API with authentication
pip install pynomaly[production]

# Full-featured API with monitoring
pip install pynomaly[production,monitoring]
```

### ML Engineer (Model Development)
```bash
# PyTorch-based development
pip install pynomaly[torch,minimal]

# Multi-framework research
pip install pynomaly[ml-all]

# Complete development environment
pip install pynomaly[ml-all,server,explainability]
```

### DevOps Engineer (Deployment)
```bash
# Container deployment
pip install pynomaly[production]

# Kubernetes deployment with full monitoring
pip install pynomaly[production,monitoring,infrastructure]

# Multi-tenant SaaS deployment
pip install pynomaly[all]
```

### Academic Researcher (Comprehensive Analysis)
```bash
# Research environment with all ML tools
pip install pynomaly[ml-all,explainability,data-all]

# Full environment for reproducible research
pip install pynomaly[all]
```

## Dependency Management Best Practices

### For Libraries Building on Pynomaly
```python
# In your setup.py or pyproject.toml
dependencies = [
    "pynomaly[minimal]>=0.1.0",  # Specify minimum needed
]

# For optional features
extras_require = {
    "api": ["pynomaly[server]>=0.1.0"],
    "ml": ["pynomaly[torch]>=0.1.0"],
}
```

### For Applications Using Pynomaly
```bash
# Pin to specific extras in requirements.txt
pynomaly[production]==0.1.0

# Or use requirements files
-r pynomaly-requirements-production.txt
```

### For Docker Builds
```dockerfile
# Multi-stage build for smaller images
FROM python:3.11-slim as base

# Install only what's needed for production
RUN pip install pynomaly[production]

# Optional: Add ML capabilities in separate stage
FROM base as ml
RUN pip install pynomaly[torch]
```

## Dependency Size Comparison

| Installation Level | Size | Use Case |
|-------------------|------|----------|
| Core only | ~50MB | Library development |
| Minimal | ~80MB | Basic ML workflows |
| Standard | ~110MB | Data science |
| Server | ~140MB | Web applications |
| Production | ~180MB | Production deployment |
| ML-all | ~1.5GB | ML research |
| All | ~2GB+ | Development/testing |

## Troubleshooting Dependencies

### Import Errors
```python
# Check what's available
from pynomaly.infrastructure.adapters import get_available_adapters
print(get_available_adapters())
```

### Missing Optional Dependencies
```bash
# Install specific extras after initial installation
pip install pynomaly[torch,api]

# Or with Poetry
poetry install -E torch -E api
```

### Conflicting Dependencies
```bash
# Check dependency tree
pip show pynomaly
pipdeptree -p pynomaly

# Resolve conflicts
pip install --upgrade pynomaly[all]
```

### Docker Size Optimization
```dockerfile
# Use multi-stage builds
FROM python:3.11-slim as base
RUN pip install pynomaly[production]

FROM base as final
# Copy only necessary files
COPY --from=base /usr/local/lib/python3.11/site-packages /usr/local/lib/python3.11/site-packages
```

## Future Dependency Plans

### Planned Additions
- **`streaming`**: Real-time processing (Kafka, Redis Streams)
- **`cloud`**: Cloud provider integrations (AWS, GCP, Azure)
- **`ui`**: Enhanced web UI components
- **`mobile`**: Mobile app development tools

### Deprecation Timeline
- **Legacy extras**: Will be marked deprecated before removal
- **Version compatibility**: Maintained for at least 2 major versions
- **Migration guides**: Provided for all breaking changes

---

*Last updated: December 2024*  
*Pynomaly version: 0.1.0+*