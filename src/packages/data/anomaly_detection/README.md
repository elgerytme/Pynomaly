# 🔍 Anomaly Detection Platform

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/downloads/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.104+-green.svg)](https://fastapi.tiangolo.com/)
[![Code style: Black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)
[![Type checked: mypy](https://img.shields.io/badge/typed-mypy-blue.svg)](https://mypy-lang.org/)
[![scikit-learn](https://img.shields.io/badge/scikit--learn-1.3+-orange.svg)](https://scikit-learn.org/)
[![PyOD](https://img.shields.io/badge/PyOD-2.0+-red.svg)](https://pyod.readthedocs.io/)

🚀 **Enterprise-grade anomaly detection platform** with comprehensive CLI, REST API, and web interface. Built with modern architecture patterns and production-ready features.

## 🎯 Overview

A complete anomaly detection solution providing multiple interfaces (CLI, API, Web) with advanced ML capabilities, real-time streaming, model management, and comprehensive monitoring. Follows Domain-Driven Design principles with clean architecture.

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
```

### Streaming Detection

```python
from anomaly_detection import StreamingService
import numpy as np

# Create streaming detector
streaming = StreamingService(window_size=500, update_frequency=50)

# Process data stream
for i in range(1000):
    sample = np.random.rand(10)  # New data point
    result = streaming.process_sample(sample)
    
    if result.predictions[0] == 1:
        print(f"Anomaly detected at sample {i}")

# Check for concept drift
drift_info = streaming.detect_concept_drift()
print("Drift detected:", drift_info["drift_detected"])
```

## Documentation

### Requirements and Planning
- **[Requirements Documentation](./requirements/)** - Complete requirements specification
  - [Business Requirements](./requirements/business_requirements.md) - Business context and objectives
  - [Functional Requirements](./requirements/functional_requirements.md) - Feature specifications
  - [Non-Functional Requirements](./requirements/non_functional_requirements.md) - Performance and quality
  - [User Personas](./requirements/user_personas.md) - Target user profiles
  - [Use Cases](./requirements/use_cases.md) - Detailed use case specifications  
  - [User Stories](./requirements/user_stories.md) - Development backlog
  - [Story Mapping](./requirements/story_mapping.md) - Feature prioritization and roadmap

### Technical Documentation
- **[Architecture Guide](./docs/architecture.md)** - System design and patterns
- **[API Reference](./docs/api.md)** - Complete API documentation
- **[Algorithm Guide](./docs/algorithms.md)** - Algorithm selection and tuning
- **[Configuration Guide](./docs/configuration.md)** - System configuration
- **[Installation Guide](./docs/installation.md)** - Setup and deployment

## Architecture

### Package Structure

```
anomaly_detection/
├── __init__.py              # Main public API
├── core/                    # Core domain logic
│   └── services/            # Detection, ensemble, streaming services
├── algorithms/              # Algorithm implementations
│   └── adapters/            # Framework adapters (sklearn, PyOD, deep learning)
├── data/                    # Data processing utilities
├── monitoring/              # Performance monitoring
├── integrations/            # External integrations
└── utils/                   # Shared utilities
```

### Core Services

1. **DetectionService**: Main service for anomaly detection operations
2. **EnsembleService**: Combines multiple algorithms for robust detection  
3. **StreamingService**: Real-time detection with incremental learning

### Algorithm Adapters

1. **SklearnAdapter**: Scikit-learn algorithms (Isolation Forest, LOF, One-Class SVM, PCA)
2. **PyODAdapter**: 40+ algorithms from PyOD library
3. **DeepLearningAdapter**: Autoencoder-based detection with TensorFlow/PyTorch

## Integration with ML Infrastructure

This package integrates with the broader ML infrastructure:

```python
# AutoML integration
from anomaly_detection import get_automl_service
automl = get_automl_service()

# MLOps integration  
from anomaly_detection import ModelManagementService
model_mgmt = ModelManagementService()
```

## Installation

### Core Package
```bash
pip install -e .
```

### With PyOD Support
```bash
pip install -e ".[pyod]"
```

### With Deep Learning Support
```bash
pip install -e ".[deeplearning]"
```

### Full Installation
```bash
pip install -e ".[full]"
```

## Available Algorithms

### Built-in Algorithms
- `iforest`: Isolation Forest
- `lof`: Local Outlier Factor

### With SklearnAdapter
- `iforest`: Isolation Forest
- `lof`: Local Outlier Factor  
- `ocsvm`: One-Class SVM
- `pca`: PCA-based detection

### With PyODAdapter (40+ algorithms)
- `iforest`, `lof`, `ocsvm`, `pca`, `knn`, `hbos`, `abod`, `feature_bagging`
- And many more specialized algorithms

### With DeepLearningAdapter
- `autoencoder`: Autoencoder-based detection (TensorFlow/PyTorch)

## Configuration

The package supports flexible configuration:

```python
# Algorithm-specific parameters
detector = AnomalyDetector(
    algorithm="iforest",
    n_estimators=200,
    contamination=0.05
)

# Service configuration
service = DetectionService()
service.register_adapter("custom_algo", CustomAdapter())
```

## Testing

Run tests with:
```bash
pytest tests/
```

With coverage:
```bash
pytest --cov=anomaly_detection tests/
```

## Development

### Code Quality
```bash
# Linting
ruff check anomaly_detection/

# Type checking  
mypy anomaly_detection/

# Formatting
black anomaly_detection/
```

## Migration from Legacy Package

This consolidated package replaces the previous complex structure with:

- **Reduced complexity**: From 118+ services to 3 core services
- **Clear separation**: Domain logic vs ML infrastructure
- **Better integration**: Proper ML/MLOps delegation
- **Maintainable structure**: Standard Python package organization

### Key Changes
1. AutoML functionality moved to `@src/packages/ai/machine_learning`
2. MLOps features moved to `@src/packages/ai/machine_learning/mlops`
3. Algorithm adapters consolidated and simplified
4. Core detection logic streamlined

## 🔍 Troubleshooting

### Common Issues

#### Installation Issues

**Problem**: `ImportError` when importing anomaly detection components
```bash
# Solution: Ensure package is properly installed
cd src/packages/data/anomaly_detection/
pip install -e .

# Verify installation
python -c "import anomaly_detection; print('Package installed successfully')"
```

**Problem**: Missing ML dependencies
```bash
# Solution: Install with ML dependencies
pip install -e ".[algorithms]"  # For additional ML algorithms
pip install -e ".[all]"         # For all optional dependencies
```

#### Detection Issues

**Problem**: Poor detection performance on your dataset
```python
# Solution: Use ensemble detection for better results
from anomaly_detection.core.domain.services import EnsembleDetectionService

detector = EnsembleDetectionService()
detector.add_detector("isolation_forest", {"contamination": 0.1})
detector.add_detector("one_class_svm", {"gamma": "scale"})
detector.add_detector("local_outlier_factor", {"n_neighbors": 20})

anomalies = detector.detect(X)
```

**Problem**: High false positive rate
```python
# Solution: Adjust contamination parameter and use threshold tuning
from anomaly_detection.core.application.services import AnomalyDetectionService

service = AnomalyDetectionService()
results = service.detect_anomalies(
    data=X,
    algorithm="isolation_forest",
    hyperparameters={
        "contamination": 0.05,  # Lower contamination = fewer false positives
        "n_estimators": 200,    # More trees = more stable results
        "max_features": 1.0     # Use all features
    }
)
```

#### Streaming Detection Issues

**Problem**: Memory issues with streaming detection
```python
# Solution: Configure streaming buffer size
from anomaly_detection.core.domain.services import StreamingDetectionService

detector = StreamingDetectionService(
    buffer_size=1000,        # Smaller buffer for memory efficiency
    update_frequency=100,    # Update model every 100 samples
    drift_threshold=0.1      # Adjust drift sensitivity
)
```

**Problem**: Concept drift not detected
```python
# Solution: Enable drift monitoring and adjust sensitivity
from anomaly_detection.core.domain.services import StreamingDetectionService

detector = StreamingDetectionService(
    enable_drift_detection=True,
    drift_threshold=0.05,    # More sensitive drift detection
    drift_window_size=500,   # Larger window for drift calculation
    adaptation_strategy="retrain"  # Retrain on drift detection
)
```

#### Integration Issues

**Problem**: ML integration not working
```bash
# Solution: Ensure machine_learning package is available
cd ../ai/machine_learning/
pip install -e .

# Verify integration
python -c "
from anomaly_detection.core.application.services import AnomalyDetectionService
from machine_learning.training import ModelTrainer
print('Integration working')
"
```

### Performance Optimization

#### For Large Datasets
```python
# Use approximate algorithms for speed
service = AnomalyDetectionService()
results = service.detect_anomalies(
    data=X,
    algorithm="isolation_forest",
    hyperparameters={
        "n_estimators": 50,      # Fewer trees for speed
        "max_samples": "auto",   # Auto-sample for efficiency
        "n_jobs": -1            # Use all CPU cores
    }
)
```

#### For Real-time Detection
```python
# Configure for low latency
detector = StreamingDetectionService(
    algorithm="lof",          # Fast local outlier factor
    buffer_size=100,          # Small buffer
    update_frequency=50,      # Frequent updates
    batch_processing=True     # Process in batches
)
```

### FAQ

**Q: Which algorithm should I use for my data?**
A: Use the algorithm selection guide:
- **Tabular data**: Isolation Forest or Local Outlier Factor
- **High-dimensional data**: One-Class SVM or Elliptic Envelope
- **Time series**: Streaming detection with concept drift monitoring
- **Unsure**: Use EnsembleDetectionService with multiple algorithms

**Q: How do I handle categorical features?**
A: Preprocess categorical features before detection:
```python
from sklearn.preprocessing import LabelEncoder, StandardScaler
import pandas as pd

# Example preprocessing
df = pd.get_dummies(df, columns=['categorical_column'])
scaler = StandardScaler()
X_scaled = scaler.fit_transform(df)

# Then use with anomaly detection
results = service.detect_anomalies(X_scaled, algorithm="isolation_forest")
```

**Q: How do I tune hyperparameters?**
A: Use the integrated ML optimization:
```python
from machine_learning.optimization import HyperparameterOptimizer

optimizer = HyperparameterOptimizer()
best_params = optimizer.optimize(
    algorithm="isolation_forest",
    X_train=X_train,
    X_val=X_val,
    metric="f1_score"
)
```

## Contributing

1. Follow the existing code style and architecture
2. Add tests for new functionality
3. Update documentation for changes
4. Ensure integration with ML infrastructure works properly

## License

MIT License - see LICENSE file for details.