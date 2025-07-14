# Pynomaly - Anomaly Detection Platform

[![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Code Coverage](https://img.shields.io/badge/coverage-90%25-green.svg)](https://github.com/pynomaly/pynomaly)

State-of-the-art Python anomaly detection platform with clean architecture, enterprise features, and multiple interfaces.

## 🏗️ Monorepo Structure

This is a monorepo containing all Pynomaly packages, applications, and shared utilities organized for maximum modularity and reusability.

```
src/
├── packages/                  # Reusable packages
│   ├── core/                  # 🔴 Domain logic & business rules
│   ├── infrastructure/        # 🔧 Technical infrastructure  
│   ├── services/              # ⚙️ Application services
│   ├── api/                   # 🌐 REST API server
│   ├── cli/                   # 🖥️ Command-line interface
│   ├── web/                   # 📱 Web UI & dashboard
│   ├── enterprise/            # 🏢 Enterprise features
│   ├── algorithms/            # 🤖 ML algorithm adapters
│   ├── sdks/                  # 📚 Client SDKs
│   ├── testing/               # 🧪 Testing utilities
│   └── tools/                 # 🛠️ Development tools
├── apps/                      # Standalone applications
│   ├── anomaly-detector/      # Main application
│   ├── monitoring/            # Monitoring dashboard
│   └── admin/                 # Admin interface
└── shared/                    # Shared utilities
    ├── types/                 # Common types
    ├── constants/             # Shared constants
    └── utils/                 # Utility functions
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

### CLI Usage

```bash
# Detect anomalies from CSV
pynomaly detect --input data.csv --algorithm isolation_forest

# Start web dashboard
pynomaly web start

# Run API server
pynomaly api start --port 8000
```

## 📦 Package Overview

### Core Packages

| Package | Description | Dependencies |
|---------|-------------|--------------|
| **core** | Domain logic, entities, use cases | None (pure business logic) |
| **infrastructure** | Database, cache, monitoring, adapters | core |
| **services** | Application orchestration services | core, infrastructure |

### Interface Packages

| Package | Description | Dependencies |
|---------|-------------|--------------|
| **api** | FastAPI REST server with WebSocket | core, infrastructure, services |
| **cli** | Typer-based command interface | core, infrastructure, services |
| **web** | Progressive web app with dashboard | core, infrastructure, services |

### Specialized Packages

| Package | Description | Dependencies |
|---------|-------------|--------------|
| **enterprise** | Multi-tenancy, compliance, governance | core, infrastructure, services |
| **algorithms** | ML algorithm adapters (PyOD, PyTorch, etc.) | core |
| **sdks** | Client libraries (Python, TypeScript, Java) | API contracts |

## 🎯 Key Features

### 🤖 Algorithm Support
- **25+ algorithms**: Isolation Forest, Local Outlier Factor, One-Class SVM, LSTM, etc.
- **Deep Learning**: PyTorch, TensorFlow, JAX backends
- **Graph Anomalies**: Graph neural networks with PyGOD
- **AutoML**: Automated algorithm selection and hyperparameter tuning

### 🏢 Enterprise Ready
- **Multi-tenancy**: Complete data isolation and resource quotas
- **Security**: RBAC, audit logging, SOC2 compliance
- **Monitoring**: Prometheus metrics, distributed tracing
- **Governance**: ML model lifecycle and lineage tracking

### 🌐 Multiple Interfaces
- **REST API**: Production-ready FastAPI server
- **Web Dashboard**: Real-time monitoring and visualization
- **CLI**: Command-line tools for automation
- **SDKs**: Native client libraries

### ⚡ Performance
- **Streaming**: Real-time anomaly detection
- **Distributed**: Cluster processing with Dask/Ray
- **Caching**: Redis-based intelligent caching
- **Optimization**: Memory-efficient processing

## 🏗️ Architecture

Pynomaly follows clean architecture principles with clear dependency boundaries:

```
┌─────────────────────┐
│   Interfaces        │  ← API, CLI, Web
│  (Presentation)     │
├─────────────────────┤
│   Application       │  ← Services, Use Cases
│   (Orchestration)   │
├─────────────────────┤
│   Infrastructure    │  ← Adapters, Persistence
│   (Technical)       │
├─────────────────────┤
│   Domain            │  ← Entities, Business Logic
│   (Core)            │
└─────────────────────┘
```

**Dependency Rule**: Inner layers never depend on outer layers.

## 🚀 Development

### Workspace Setup

```bash
# Clone repository
git clone https://github.com/pynomaly/pynomaly.git
cd pynomaly

# Install development dependencies
pip install -e .[dev]

# Setup pre-commit hooks
pre-commit install
```

### Package Development

```bash
# Work on specific package
cd src/packages/core
pip install -e .[dev]
pytest tests/

# Run all tests
pytest src/packages/*/tests/

# Build all packages
hatch build
```

### Workspace Commands

```bash
# Build all packages
npm run build

# Test all packages  
npm run test

# Lint all packages
npm run lint

# Start development servers
npm run dev:api     # API server
npm run dev:web     # Web dashboard
```

## 🧪 Testing

```bash
# Run all tests
pytest

# Test specific package
pytest src/packages/core/tests/

# Integration tests
pytest -m integration

# Performance tests
pytest -m performance

# Coverage report
pytest --cov=src --cov-report=html
```

## 📊 Monitoring & Observability

- **Metrics**: Prometheus with Grafana dashboards
- **Tracing**: OpenTelemetry distributed tracing
- **Logging**: Structured logging with correlation IDs
- **Health Checks**: Comprehensive health monitoring

## 🔐 Security

- **Authentication**: JWT with refresh tokens
- **Authorization**: Role-based access control (RBAC)
- **Rate Limiting**: Adaptive rate limiting with DDoS protection
- **Audit**: Complete audit trail for compliance
- **Encryption**: Data encryption at rest and in transit

## 🌍 Deployment

### Docker

```bash
# Build images
docker-compose build

# Start services
docker-compose up -d

# Production deployment
docker-compose -f docker-compose.prod.yml up -d
```

### Kubernetes

```bash
# Deploy to Kubernetes
kubectl apply -f k8s/

# Monitor deployment
kubectl get pods -l app=pynomaly
```

### Cloud Platforms

- **AWS**: ECS, EKS, Lambda support
- **GCP**: GKE, Cloud Run, Cloud Functions
- **Azure**: AKS, Container Instances, Functions

## 🤝 Contributing

We welcome contributions! Please see our [Contributing Guide](CONTRIBUTING.md) for details.

1. **Fork** the repository
2. **Create** a feature branch
3. **Make** your changes
4. **Add** tests for new functionality
5. **Submit** a pull request

## 📝 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🔗 Links

- **Documentation**: [https://pynomaly.readthedocs.io](https://pynomaly.readthedocs.io)
- **PyPI**: [https://pypi.org/project/pynomaly/](https://pypi.org/project/pynomaly/)
- **GitHub**: [https://github.com/pynomaly/pynomaly](https://github.com/pynomaly/pynomaly)
- **Issues**: [https://github.com/pynomaly/pynomaly/issues](https://github.com/pynomaly/pynomaly/issues)

## 🙏 Acknowledgments

Built with ❤️ by the Pynomaly team and contributors. Special thanks to the open-source community for the amazing tools and libraries that make this project possible.

---

**Made with 🔍 for detecting the unexpected.**