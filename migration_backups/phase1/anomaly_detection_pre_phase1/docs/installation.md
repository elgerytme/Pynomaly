# Installation Guide

This guide covers various installation methods for the Anomaly Detection package across different environments.

## Table of Contents

1. [Requirements](#requirements)
2. [Quick Install](#quick-install)
3. [Installation Methods](#installation-methods)
4. [Optional Dependencies](#optional-dependencies)
5. [Development Installation](#development-installation)
6. [Docker Installation](#docker-installation)
7. [Cloud Deployments](#cloud-deployments)
8. [Troubleshooting](#troubleshooting)

## Requirements

### System Requirements

- **Python**: 3.11 or higher
- **Memory**: Minimum 2GB RAM (8GB+ recommended for production)
- **Disk Space**: 500MB for base installation
- **OS**: Linux, macOS, or Windows

### Core Dependencies

- NumPy >= 1.24.0
- Pandas >= 2.0.0
- Scikit-learn >= 1.3.0
- SciPy >= 1.10.0
- Pydantic >= 2.0.0
- FastAPI >= 0.104.0

## Quick Install

### Using pip

```bash
# Basic installation
pip install anomaly-detection

# With all optional dependencies
pip install anomaly-detection[full]

# From the monorepo
cd src/packages/data/anomaly_detection
pip install -e .
```

### Using conda

```bash
# Create conda environment
conda create -n anomaly-detection python=3.11
conda activate anomaly-detection

# Install package
conda install -c conda-forge anomaly-detection
```

### Using poetry

```bash
# Install poetry if not already installed
curl -sSL https://install.python-poetry.org | python3 -

# Install package
poetry add anomaly-detection
```

## Installation Methods

### 1. Production Installation

For production deployments with optimized dependencies:

```bash
# Install production dependencies only
pip install anomaly-detection[prod]

# With specific algorithm support
pip install anomaly-detection[prod,algorithms]

# Verify installation
python -c "import anomaly_detection; print(anomaly_detection.__version__)"
```

### 2. Minimal Installation

For lightweight deployments with core functionality only:

```bash
# Core package only
pip install anomaly-detection --no-deps
pip install numpy pandas scikit-learn

# Verify core functionality
anomaly-detection --version
```

### 3. Source Installation

Install from source for latest features:

```bash
# Clone repository
git clone https://github.com/monorepo/anomaly_detection.git
cd anomaly_detection

# Install in development mode
pip install -e .

# Run tests to verify
pytest tests/
```

### 4. Monorepo Installation

When working within the monorepo structure:

```bash
# Navigate to package directory
cd src/packages/data/anomaly_detection

# Install with all dependencies
pip install -e ".[all]"

# Install pre-commit hooks
pre-commit install
```

## Optional Dependencies

### Algorithm Libraries

```bash
# PyOD algorithms (40+ algorithms)
pip install anomaly-detection[pyod]

# Deep learning support
pip install anomaly-detection[deeplearning]

# All algorithms
pip install anomaly-detection[algorithms]
```

### Visualization

```bash
# Plotting support
pip install anomaly-detection[viz]
# Includes: matplotlib, seaborn, plotly
```

### Performance

```bash
# Performance monitoring
pip install anomaly-detection[performance]
# Includes: memory-profiler, py-spy, psutil
```

### Development

```bash
# Development tools
pip install anomaly-detection[dev]
# Includes: pytest, black, ruff, mypy, pre-commit
```

## Development Installation

### Setting Up Development Environment

1. **Clone the repository:**
```bash
git clone https://github.com/monorepo/anomaly_detection.git
cd anomaly_detection
```

2. **Create virtual environment:**
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. **Install in development mode:**
```bash
pip install -e ".[dev,test,docs]"
```

4. **Install pre-commit hooks:**
```bash
pre-commit install
pre-commit run --all-files  # Run on all files
```

5. **Verify development setup:**
```bash
# Run tests
pytest

# Run linting
ruff check .

# Run type checking
mypy src/anomaly_detection

# Build documentation
mkdocs serve
```

### IDE Setup

#### VS Code

Create `.vscode/settings.json`:
```json
{
    "python.linting.enabled": true,
    "python.linting.ruffEnabled": true,
    "python.formatting.provider": "black",
    "python.testing.pytestEnabled": true,
    "python.testing.unittestEnabled": false,
    "python.testing.pytestArgs": ["tests"]
}
```

#### PyCharm

1. Mark `src` as Sources Root
2. Configure pytest as test runner
3. Enable black formatter
4. Set Python interpreter to virtual environment

## Docker Installation

### Using Pre-built Image

```bash
# Pull latest image
docker pull anomaly-detection:latest

# Run container
docker run -p 8001:8001 anomaly-detection:latest
```

### Building Custom Image

Create `Dockerfile`:
```dockerfile
FROM python:3.11-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements
COPY pyproject.toml ./
COPY src/ ./src/

# Install package
RUN pip install --no-cache-dir -e ".[prod]"

# Expose ports
EXPOSE 8001

# Run server
CMD ["anomaly-detection-server", "--host", "0.0.0.0", "--port", "8001"]
```

Build and run:
```bash
docker build -t anomaly-detection:custom .
docker run -p 8001:8001 anomaly-detection:custom
```

### Docker Compose

Create `docker-compose.yml`:
```yaml
version: '3.8'

services:
  anomaly-detection:
    build: .
    ports:
      - "8001:8001"
    environment:
      - ANOMALY_DETECTION_ENV=production
      - ANOMALY_DETECTION_LOG_LEVEL=INFO
    volumes:
      - ./data:/app/data
      - ./models:/app/models
    healthcheck:
      test: ["CMD", "curl", "-f", "http://localhost:8001/health"]
      interval: 30s
      timeout: 10s
      retries: 3

  redis:
    image: redis:7-alpine
    ports:
      - "6379:6379"

  prometheus:
    image: prom/prometheus
    ports:
      - "9090:9090"
    volumes:
      - ./prometheus.yml:/etc/prometheus/prometheus.yml
```

## Cloud Deployments

### AWS

#### Using AWS Lambda

```bash
# Install with Lambda dependencies
pip install anomaly-detection[aws-lambda]

# Package for Lambda
pip install -t lambda_package anomaly-detection
cd lambda_package
zip -r ../anomaly_detection_lambda.zip .
```

#### Using ECS/Fargate

```bash
# Build and push to ECR
aws ecr get-login-password --region us-east-1 | docker login --username AWS --password-stdin $ECR_URI
docker build -t anomaly-detection .
docker tag anomaly-detection:latest $ECR_URI/anomaly-detection:latest
docker push $ECR_URI/anomaly-detection:latest
```

### Google Cloud

#### Using Cloud Run

```bash
# Build and deploy
gcloud builds submit --tag gcr.io/$PROJECT_ID/anomaly-detection
gcloud run deploy anomaly-detection \
    --image gcr.io/$PROJECT_ID/anomaly-detection \
    --platform managed \
    --allow-unauthenticated
```

### Azure

#### Using Container Instances

```bash
# Create container instance
az container create \
    --resource-group myResourceGroup \
    --name anomaly-detection \
    --image anomaly-detection:latest \
    --dns-name-label anomaly-detection-app \
    --ports 8001
```

### Kubernetes

#### Using Helm

```bash
# Add Helm repository
helm repo add anomaly-detection https://charts.anomaly-detection.io
helm repo update

# Install chart
helm install anomaly-detection anomaly-detection/anomaly-detection \
    --set image.tag=latest \
    --set service.type=LoadBalancer
```

#### Manual Deployment

Create `deployment.yaml`:
```yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: anomaly-detection
spec:
  replicas: 3
  selector:
    matchLabels:
      app: anomaly-detection
  template:
    metadata:
      labels:
        app: anomaly-detection
    spec:
      containers:
      - name: anomaly-detection
        image: anomaly-detection:latest
        ports:
        - containerPort: 8001
        resources:
          requests:
            memory: "512Mi"
            cpu: "500m"
          limits:
            memory: "2Gi"
            cpu: "2000m"
---
apiVersion: v1
kind: Service
metadata:
  name: anomaly-detection-service
spec:
  selector:
    app: anomaly-detection
  ports:
    - protocol: TCP
      port: 80
      targetPort: 8001
  type: LoadBalancer
```

Deploy:
```bash
kubectl apply -f deployment.yaml
```

## Troubleshooting

### Common Installation Issues

#### 1. Python Version Error

**Error:** `ERROR: Package 'anomaly-detection' requires a different Python: 3.8.0 not in '>=3.11'`

**Solution:**
```bash
# Install Python 3.11+
pyenv install 3.11.7
pyenv local 3.11.7

# Or using conda
conda create -n anomaly-detection python=3.11
conda activate anomaly-detection
```

#### 2. Missing System Dependencies

**Error:** `error: Microsoft Visual C++ 14.0 is required` (Windows)

**Solution:**
```bash
# Windows: Install Visual Studio Build Tools
# Download from: https://visualstudio.microsoft.com/downloads/

# Linux: Install build essentials
sudo apt-get update
sudo apt-get install build-essential

# macOS: Install Xcode Command Line Tools
xcode-select --install
```

#### 3. Memory Issues During Installation

**Error:** `MemoryError` during pip install

**Solution:**
```bash
# Install packages individually
pip install numpy
pip install pandas
pip install scikit-learn
pip install anomaly-detection --no-deps

# Or increase pip memory limit
pip install --no-cache-dir anomaly-detection
```

#### 4. Conflicting Dependencies

**Error:** `ERROR: pip's dependency resolver does not currently take into account all the packages that are installed`

**Solution:**
```bash
# Create clean environment
python -m venv clean_env
source clean_env/bin/activate
pip install --upgrade pip
pip install anomaly-detection
```

#### 5. Import Errors

**Error:** `ModuleNotFoundError: No module named 'anomaly_detection'`

**Solution:**
```bash
# Verify installation
pip list | grep anomaly-detection

# Check Python path
python -c "import sys; print(sys.path)"

# Reinstall
pip uninstall anomaly-detection
pip install anomaly-detection
```

### Performance Optimization

#### 1. Use Binary Wheels

```bash
# Ensure pip is updated
pip install --upgrade pip

# Install with binary wheels
pip install --only-binary :all: anomaly-detection
```

#### 2. Compile with Optimizations

```bash
# Install with compiler optimizations
CFLAGS="-O3 -march=native" pip install anomaly-detection
```

#### 3. GPU Support (Deep Learning)

```bash
# Install with CUDA support
pip install anomaly-detection[deeplearning-gpu]

# Verify GPU availability
python -c "import torch; print(torch.cuda.is_available())"
```

### Verification

After installation, verify everything is working:

```bash
# Check CLI
anomaly-detection --version
anomaly-detection algorithms list

# Check Python import
python -c "
from anomaly_detection import DetectionService
print('Core import successful')
"

# Run basic detection
python -c "
import numpy as np
from anomaly_detection import AnomalyDetector
data = np.random.randn(100, 5)
detector = AnomalyDetector('iforest')
result = detector.detect(data)
print(f'Detection successful: {result.anomaly_count} anomalies found')
"

# Check server
anomaly-detection-server --help
```

## Next Steps

1. Read the [Configuration Guide](configuration.md) to customize settings
2. Follow the [Quick Start Tutorial](../README.md#quick-start) to run your first detection
3. Explore [Algorithm Guide](algorithms.md) to choose the right algorithm
4. Check [API Reference](api.md) for detailed usage

## Support

If you encounter any issues:

1. Check the [Troubleshooting Guide](troubleshooting.md)
2. Search [existing issues](https://github.com/monorepo/anomaly_detection/issues)
3. Create a new issue with:
   - Python version (`python --version`)
   - Package version (`pip show anomaly-detection`)
   - Full error traceback
   - Steps to reproduce