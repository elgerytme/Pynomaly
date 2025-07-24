# 🔍 Anomaly Detection Platform

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/downloads/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.104+-green.svg)](https://fastapi.tiangolo.com/)
[![Code style: Black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)
[![Type checked: mypy](https://img.shields.io/badge/typed-mypy-blue.svg)](https://mypy-lang.org/)
[![scikit-learn](https://img.shields.io/badge/scikit--learn-1.3+-orange.svg)](https://scikit-learn.org/)
[![PyOD](https://img.shields.io/badge/PyOD-2.0+-red.svg)](https://pyod.readthedocs.io/)

🚀 **Enterprise-grade anomaly detection platform** with comprehensive CLI, REST API, and web interface. Built with **Domain-Driven Design (DDD)** architecture and production-ready features.

!!! info "🏗️ New Architecture"
    **Domain Migration Complete:** This package has been successfully migrated to a domain-driven architecture for improved scalability, maintainability, and future microservices support.

## 🎯 Overview

A complete anomaly detection solution providing multiple interfaces (CLI, API, Web) with advanced ML capabilities, real-time streaming, model management, and comprehensive monitoring. Built with **Domain-Driven Design** principles featuring clear domain boundaries and clean architecture.

### 🏗️ Domain-Driven Architecture

```
📦 Anomaly Detection Platform
├── 🤖 AI Domain
│   ├── machine_learning/     # Core ML algorithms and training
│   └── mlops/               # Model lifecycle and experiment tracking  
├── 📊 Data Domain
│   └── processing/          # Data entities and processing pipelines
├── 🔧 Shared Infrastructure
│   ├── infrastructure/      # Configuration, logging, security
│   └── observability/       # Monitoring, metrics, dashboards
└── 🎯 Application Layer      # Business logic and orchestration
```

📋 **[Complete Requirements Documentation](./requirements/)** - Business requirements, user stories, use cases, and feature roadmap

## ✨ Key Features

### 🔧 **Multiple Interfaces**
- **🖥️ Command Line Interface**: Full-featured Typer CLI with rich output
- **🌐 REST API**: FastAPI with OpenAPI documentation and async support
- **📱 Web Dashboard**: Interactive HTMX-based interface with real-time updates

### 🤖 **Advanced ML Capabilities**
- **Multiple Algorithms**: Isolation Forest, One-Class SVM, Local Outlier Factor, Ensemble Methods
- **Deep Learning Support**: TensorFlow and PyTorch integration via PyOD
- **Model Management**: Training, versioning, deployment, and performance monitoring
- **Auto-tuning**: Automatic hyperparameter optimization and threshold optimization

### 📊 **Real-time Processing** 
- **Streaming Detection**: WebSocket-based real-time anomaly detection
- **Concept Drift**: Automatic drift detection and model adaptation
- **Batch Processing**: Efficient processing of large datasets with job management
- **Performance Monitoring**: Real-time metrics, alerts, and health monitoring

### 🏢 **Enterprise Features**
- **Security**: Input validation, rate limiting, and secure model storage
- **Scalability**: Async processing, connection pooling, and horizontal scaling
- **Observability**: Structured logging, metrics collection, and distributed tracing
- **Deployment**: Docker containers, Kubernetes manifests, and CI/CD pipelines

## 🚀 Quick Start

### Installation

```bash
# Install the package
pip install -e .

# Install with additional dependencies
pip install -e .[algorithms,monitoring,all]
```

### 🖥️ Command Line Interface

```bash
# Basic anomaly detection
anomaly-detection detection run --data data.csv --algorithm isolation_forest

# Train a new model
anomaly-detection models train --data training.csv --algorithm lof --name my_model

# Stream processing
anomaly-detection streaming start --model my_model --source kafka://localhost:9092

# Generate reports
anomaly-detection reports detection results.json --format html --output report.html

# Health monitoring
anomaly-detection health system --detailed
```

### 🌐 REST API

```bash
# Start the API server
anomaly-detection-server

# Example API calls
curl -X POST "http://localhost:8000/api/v1/detection/run" \
  -H "Content-Type: application/json" \
  -d '{"data": [[1,2], [3,4]], "algorithm": "isolation_forest"}'
```

### 📱 Web Interface

```bash
# Start the web dashboard
anomaly-detection-web

# Open browser to http://localhost:8080
# Features:
# - Model management and training
# - Real-time monitoring dashboard  
# - Interactive detection interface
# - System health and metrics
```

### 🐍 Python API

```python
from anomaly_detection import DetectionService
from anomaly_detection.domain.entities import Dataset, DatasetMetadata
import numpy as np

# Create detection service
service = DetectionService()

# Prepare data
data = np.random.rand(1000, 5)
metadata = DatasetMetadata(name="test_data", feature_names=[f"f{i}" for i in range(5)])
dataset = Dataset(data=data, metadata=metadata)

# Run detection
result = await service.detect_anomalies(dataset, algorithm="isolation_forest")
print(f"Found {result.anomaly_count} anomalies")
```

## 🏗️ Architecture

The platform follows **Domain-Driven Design** principles with clean architecture:

```
anomaly_detection/
├── 🎯 domain/           # Core business logic
│   ├── entities/        # Dataset, Model, DetectionResult
│   └── services/        # DetectionService, StreamingService
├── 🚀 application/      # Use cases and facades
├── 🌐 presentation/     # CLI, API, and Web interfaces
├── 🔧 infrastructure/   # Adapters, monitoring, persistence
└── 📊 web/             # HTMX templates and static assets
```

### 🔍 Key Components

- **🎯 Domain Layer**: Pure business logic with no external dependencies
- **🚀 Application Layer**: Orchestrates domain services and use cases  
- **🌐 Presentation Layer**: Multiple interfaces (CLI/API/Web) 
- **🔧 Infrastructure Layer**: External integrations and cross-cutting concerns

## 🚢 Deployment

### 🐳 Docker

```bash
# Build images
docker build -t anomaly-detection:latest .

# Run with docker-compose
docker-compose up -d

# Scale services
docker-compose up --scale worker=3
```

### ☸️ Kubernetes

```bash
# Deploy to Kubernetes
kubectl apply -f k8s/

# Port forward for local access
kubectl port-forward svc/anomaly-detection-web 8080:8080
```

### 🌥️ Cloud Deployment

The platform supports deployment on:
- **AWS**: ECS, EKS, Lambda functions
- **GCP**: GKE, Cloud Run, Cloud Functions  
- **Azure**: AKS, Container Instances, Functions

## 📊 Use Cases

### 🔐 Fraud Detection
- Real-time transaction monitoring
- Pattern-based anomaly identification
- Risk scoring and alerting

### 🏭 Industrial IoT
- Equipment failure prediction
- Quality control monitoring
- Predictive maintenance

### 🌐 Network Security
- Intrusion detection
- Traffic anomaly identification
- Behavioral analysis

### 📈 Business Intelligence
- Customer behavior analysis
- Revenue anomaly detection
- Market trend identification

## 🤝 Contributing

```bash
# Development setup
git clone <repository>
cd anomaly_detection
python -m venv venv
source venv/bin/activate
pip install -e .[dev,test]

# Run tests
pytest tests/ -v --cov

# Code quality
ruff check src/
mypy src/
```

## 📄 License

MIT License - see [LICENSE](LICENSE) file for details.

---

🔗 **Links**: [Documentation](./docs/) | [Examples](./examples/) | [API Reference](./docs/api.md) | [Contributing](./CONTRIBUTING.md)