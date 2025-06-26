# Pynomaly Documentation

Welcome to Pynomaly - a comprehensive, production-ready Python anomaly detection platform. This documentation is organized by user journey to help you find exactly what you need.

## 🎯 Choose Your Path

### 👤 **I'm new to Pynomaly**
Start here for installation, basic concepts, and getting your first anomaly detection running.
- **[Getting Started →](getting-started/)** - Installation, quickstart, platform setup
- **[Web Interface Quickstart →](getting-started/web-interface-quickstart.md)** - 5-minute web UI tutorial

### 📊 **I want to use Pynomaly**
Learn how to use Pynomaly effectively for your anomaly detection needs.
- **[User Guides →](user-guides/)** - Basic usage, advanced features, troubleshooting
- **[Progressive Web App →](user-guides/progressive-web-app.md)** - Mobile and offline capabilities

### 💻 **I'm developing with Pynomaly**
Technical documentation for developers, integrators, and contributors.
- **[Developer Guides →](developer-guides/)** - Architecture, API integration, contributing

### 📚 **I need technical reference**
Comprehensive references for algorithms, APIs, and configuration.
- **[Reference →](reference/)** - Algorithms, API docs, configuration

### 🚀 **I'm deploying to production**
Production deployment, operations, and monitoring guidance.
- **[Deployment →](deployment/)** - Docker, Kubernetes, security, monitoring

### 📋 **I want examples**
Real-world examples, tutorials, and industry-specific guides.
- **[Examples →](examples/)** - Banking, manufacturing, tutorials

---

## 🏗️ Documentation Structure

```
docs/
├── 🏁 getting-started/        # New user onboarding
│   ├── installation.md       # Install Pynomaly
│   ├── quickstart.md         # First anomaly detection
│   └── platform-specific/    # Windows, macOS, Linux
├── 👤 user-guides/           # How to use Pynomaly
│   ├── basic-usage/          # Core functionality
│   ├── advanced-features/    # Advanced capabilities
│   └── troubleshooting/      # Problem solving
├── 💻 developer-guides/      # Technical development
│   ├── architecture/         # System design
│   ├── api-integration/      # API development
│   └── contributing/         # Development setup
├── 📚 reference/            # Technical reference
│   ├── algorithms/          # Algorithm documentation
│   ├── api/                 # API specifications
│   └── configuration/       # Config reference
├── 🚀 deployment/           # Production deployment
├── 📋 examples/             # Practical examples
│   ├── banking/            # Financial use cases
│   ├── manufacturing/      # Industrial use cases
│   └── tutorials/          # Step-by-step guides
└── 📁 project/             # Internal project docs
```

---

## 🚀 Quick Start

### 30-Second Setup
```bash
# Install Pynomaly
pip install pynomaly

# Quick anomaly detection
python -c "
from pynomaly import detect_anomalies
import pandas as pd

# Load your data
data = pd.read_csv('your_data.csv')

# Detect anomalies (one line!)
anomalies = detect_anomalies(data, contamination=0.1)
print(f'Found {anomalies.sum()} anomalies')
"
```

### Production Example
```python
from pynomaly.infrastructure.config import create_container
from pynomaly.application.use_cases import DetectAnomaliesUseCase, TrainDetectorUseCase
import pandas as pd

# Initialize enterprise-grade container
container = create_container()

# Create detector with advanced settings
train_use_case = container.train_detector_use_case()
detector = await train_use_case.execute(
    name="Production Fraud Detector",
    algorithm="IsolationForest",
    parameters={
        "contamination": 0.05,
        "n_estimators": 200,
        "max_samples": 'auto',
        "random_state": 42
    }
)

# Detect anomalies with monitoring
detect_use_case = container.detect_anomalies_use_case()
results = await detect_use_case.execute(
    detector_id=detector.id,
    dataset_id=dataset.id,
    enable_monitoring=True
)

print(f"Detected {results.n_anomalies} anomalies with {results.confidence:.2f} confidence")
```

---

## 🔥 Key Features

### 🎯 **Algorithm Excellence**
- **20+ Core Algorithms** - Essential algorithms for most use cases
- **Specialized Methods** - Time series, graphs, text, images  
- **Experimental Research** - Cutting-edge deep learning methods
- **Performance Guidance** - Algorithm comparison and selection

### 🏗️ **Production Architecture**
- **Clean Architecture** - Domain-driven design with hexagonal architecture
- **Async Performance** - High-throughput with async/await patterns
- **Enterprise Security** - JWT auth, encryption, audit logging
- **Observability** - Comprehensive monitoring and metrics

### 🔧 **Developer Experience**
- **Test-Driven Development** - TDD enforcement with 85% coverage threshold
- **Modern Tooling** - Hatch, Ruff, MyPy, pre-commit hooks
- **Rich APIs** - REST API, Python SDK, CLI, Progressive Web App
- **Type Safety** - 100% type hint coverage with strict MyPy

### 📊 **Business Intelligence**
- **Interactive Visualizations** - D3.js and Apache ECharts integration
- **Export Capabilities** - Excel, Power BI, CSV, JSON formats
- **Real-time Dashboards** - Live anomaly detection monitoring
- **Reporting** - Automated reports and alerting

---

## 🎯 Popular User Journeys

### **Data Scientist - First Time**
1. [Installation](getting-started/installation.md) → Install Pynomaly
2. [Quickstart](getting-started/quickstart.md) → Run first detection  
3. [Basic Usage](user-guides/basic-usage/) → Learn core concepts
4. [Algorithm Selection](reference/algorithms/) → Choose optimal algorithms

### **ML Engineer - Integration**
1. [API Integration](developer-guides/api-integration/) → Understand APIs
2. [Architecture](developer-guides/architecture/) → System design  
3. [Authentication](developer-guides/api-integration/authentication.md) → Security setup
4. [Deployment](deployment/) → Production deployment

### **DevOps - Production**
1. [Docker Deployment](deployment/DOCKER_DEPLOYMENT_GUIDE.md) → Containerization
2. [Kubernetes](deployment/kubernetes.md) → Orchestration
3. [Security](deployment/SECURITY.md) → Production security
4. [Monitoring](user-guides/basic-usage/monitoring.md) → Observability

### **Business Analyst - Domain Expert**
1. [Banking Examples](examples/banking/) → Industry use cases
2. [Data Quality](examples/Data_Quality_Anomaly_Detection_Guide.md) → Data preparation
3. [Explainability](user-guides/advanced-features/explainability.md) → Understanding results
4. [Performance Tuning](user-guides/advanced-features/performance-tuning.md) → Optimization

---

## 📈 Technology Stack

### **Core Framework**
- **Python 3.11+** - Modern Python with advanced type hints
- **FastAPI** - High-performance async web framework
- **SQLAlchemy 2.0** - Modern ORM with async support
- **Pydantic v2** - Data validation and settings management

### **Algorithm Libraries**
- **PyOD** - Comprehensive anomaly detection library
- **scikit-learn** - Machine learning fundamentals
- **PyTorch** - Deep learning and neural networks
- **TensorFlow** - Production ML deployment

### **Web Technologies**
- **HTMX** - Dynamic web interfaces without JavaScript complexity
- **Tailwind CSS** - Utility-first CSS framework
- **D3.js** - Custom interactive visualizations
- **Apache ECharts** - Statistical charts and dashboards

### **Development Tools**
- **Hatch** - Modern Python build system and environment management
- **Ruff** - Lightning-fast linting and formatting
- **MyPy** - Static type checking with strict mode
- **Playwright** - Cross-browser UI testing

---

## 🤝 Community & Support

### **Getting Help**
- **[Troubleshooting Guide](user-guides/troubleshooting/)** - Common issues and solutions
- **[GitHub Issues](https://github.com/your-org/pynomaly/issues)** - Bug reports and feature requests
- **[Discussions](https://github.com/your-org/pynomaly/discussions)** - Community Q&A

### **Contributing**
- **[Contributing Guide](developer-guides/contributing/CONTRIBUTING.md)** - How to contribute
- **[Development Setup](developer-guides/contributing/README.md)** - Local development
- **[Architecture](developer-guides/architecture/overview.md)** - System design principles

### **Governance**
- **[Documentation Standards](project/standards/)** - Writing and maintenance guidelines
- **[Code Organization](developer-guides/contributing/FILE_ORGANIZATION_STANDARDS.md)** - Project structure
- **[Development Process](developer-guides/contributing/HATCH_GUIDE.md)** - Workflow and tools

---

## 🎯 Next Steps

Ready to get started? Choose your path:

### **Quick Start** 
Jump right in with the essentials:
- **[Install Pynomaly →](getting-started/installation.md)**
- **[5-Minute Quickstart →](getting-started/quickstart.md)**

### **Deep Dive**
Comprehensive learning path:
- **[User Guides →](user-guides/)** - Master Pynomaly usage
- **[Algorithm Reference →](reference/algorithms/)** - Understand all algorithms
- **[Architecture →](developer-guides/architecture/)** - System design deep dive

### **Production**
Enterprise deployment:
- **[Production Deployment →](deployment/)** - Deploy to production
- **[Security →](deployment/SECURITY.md)** - Security best practices
- **[Monitoring →](user-guides/basic-usage/monitoring.md)** - Observability setup