# 🚀 Anomaly Detection Package Examples

This directory contains comprehensive examples demonstrating the full capabilities of the anomaly detection package, from basic usage to advanced production deployment scenarios.

## 📚 Example Categories

### 🟢 **Beginner Level**
Perfect for getting started with anomaly detection concepts.

| Example | Description | Runtime | Prerequisites |
|---------|-------------|---------|---------------|
| [`basic_usage.py`](basic_usage.py) | Mock demonstration of core concepts | ~5 min | None |
| [`01_quickstart.py`](01_quickstart.py) | Interactive quickstart with real functionality | ~10 min | Core dependencies |
| [`02_data_loading.py`](02_data_loading.py) | Loading data from various sources (CSV, JSON, DB, API) | ~15 min | pandas, requests |
| [`03_visualization.py`](03_visualization.py) | Comprehensive visualization techniques | ~20 min | matplotlib, seaborn, plotly |

### 🟡 **Intermediate Level**
Advanced algorithms and optimization techniques.

| Example | Description | Runtime | Prerequisites |
|---------|-------------|---------|---------------|
| [`04_advanced_algorithms.py`](04_advanced_algorithms.py) | Deep learning, PyOD, custom algorithms | ~25 min | PyTorch, PyOD, statsmodels |
| [`06_performance_optimization.py`](06_performance_optimization.py) | Benchmarking, parallel processing, GPU acceleration | ~30 min | memory-profiler, psutil, torch |
| [`07_explainable_ai.py`](07_explainable_ai.py) | SHAP, LIME, model interpretability | ~20 min | shap, lime, plotly |

### 🔴 **Advanced Level**
Production deployment and enterprise features.

| Example | Description | Runtime | Prerequisites |
|---------|-------------|---------|---------------|
| [`05_production_deployment.py`](05_production_deployment.py) | FastAPI, Docker, Kubernetes deployment | ~45 min | FastAPI, Docker, kubectl |
| [`08_streaming_realtime.py`](08_streaming_realtime.py) | Kafka, WebSocket, real-time processing | ~35 min | kafka-python, websockets |
| [`09_model_management.py`](09_model_management.py) | MLOps, versioning, A/B testing | ~30 min | MLflow, git |

### 🎯 **Domain-Specific**
Industry use cases and practical applications.

| Example | Description | Runtime | Prerequisites |
|---------|-------------|---------|---------------|
| [`10_industry_use_cases.py`](10_industry_use_cases.py) | Financial, IoT, healthcare, manufacturing | ~40 min | Domain-specific data |
| [`11_cli_and_integrations.py`](11_cli_and_integrations.py) | Command-line workflows and system integrations | ~25 min | CLI tools, APIs |

## 🚀 Quick Start

### 1. **First Time Users**
Start with the basic examples to understand core concepts:

```bash
# Run the interactive quickstart
python 01_quickstart.py

# Try data loading examples
python 02_data_loading.py

# Explore visualizations
python 03_visualization.py
```

### 2. **Developers & Data Scientists**
Dive into advanced algorithms and optimization:

```bash
# Advanced algorithms
python 04_advanced_algorithms.py

# Performance optimization
python 06_performance_optimization.py

# Explainable AI
python 07_explainable_ai.py
```

### 3. **MLOps Engineers**
Focus on production deployment and model management:

```bash
# Production deployment
python 05_production_deployment.py

# Streaming and real-time
python 08_streaming_realtime.py

# Model management
python 09_model_management.py
```

## 📋 Prerequisites

### Core Dependencies
All examples require these base packages:

```bash
pip install numpy pandas matplotlib seaborn scikit-learn
```

### Example-Specific Dependencies

#### For Advanced Algorithms (`04_advanced_algorithms.py`)
```bash
pip install torch pyod statsmodels memory-profiler
```

#### For Production Deployment (`05_production_deployment.py`)
```bash
pip install fastapi uvicorn redis psutil
# Docker and Kubernetes CLI tools
```

#### For Performance Optimization (`06_performance_optimization.py`)
```bash
pip install memory-profiler psutil numba torch
```

#### For Explainable AI (`07_explainable_ai.py`)
```bash
pip install shap lime plotly
```

#### For Streaming (`08_streaming_realtime.py`)
```bash
pip install kafka-python websockets asyncio-mqtt
```

#### For Model Management (`09_model_management.py`)
```bash
pip install mlflow gitpython wandb
```

#### For Visualizations (Multiple Examples)
```bash
pip install plotly bokeh dash streamlit
```

### Complete Installation
For all examples with full functionality:

```bash
# Create virtual environment
python -m venv anomaly_detection_examples
source anomaly_detection_examples/bin/activate  # Linux/macOS
# anomaly_detection_examples\Scripts\activate  # Windows

# Install all dependencies
pip install -r requirements-full.txt
```

## 🛠️ Setup Instructions

### 1. **Environment Setup**

```bash
# Clone and navigate to the examples directory
cd src/packages/data/anomaly_detection/examples/

# Install the anomaly detection package
cd ..
pip install -e .
cd examples/
```

### 2. **Data Setup**
Some examples use sample datasets. They will be generated automatically, but you can also use your own data:

```bash
# Create a data directory (optional)
mkdir data/
# Place your CSV, JSON, or other data files here
```

### 3. **Configuration**
Several examples support configuration files:

```bash
# Create config directory
mkdir config/

# Copy example configurations
cp config_templates/* config/
```

## 📊 Example Details

### Basic Usage (`basic_usage.py`)
- **Purpose**: Introduction to anomaly detection concepts
- **Key Features**: Mock demonstrations, conceptual overview
- **Best For**: Understanding the problem domain
- **Dependencies**: None (uses built-in libraries)

### Quickstart (`01_quickstart.py`)
- **Purpose**: Interactive introduction with real functionality
- **Key Features**: Multiple algorithms, visualization, parameter tuning
- **Best For**: First hands-on experience
- **Dependencies**: Core ML libraries

### Data Loading (`02_data_loading.py`)
- **Purpose**: Comprehensive data ingestion examples
- **Key Features**: CSV, JSON, database, API, streaming data
- **Best For**: Understanding data preprocessing
- **Dependencies**: pandas, requests, sqlite3

### Visualization (`03_visualization.py`)
- **Purpose**: Advanced visualization techniques
- **Key Features**: 2D/3D plots, interactive dashboards, time series
- **Best For**: Understanding detection results
- **Dependencies**: matplotlib, seaborn, plotly

### Advanced Algorithms (`04_advanced_algorithms.py`)
- **Purpose**: State-of-the-art detection techniques
- **Key Features**: Deep learning, PyOD algorithms, custom implementations
- **Best For**: Advanced users, researchers
- **Dependencies**: PyTorch, PyOD, advanced ML libraries

### Production Deployment (`05_production_deployment.py`)
- **Purpose**: Enterprise-grade deployment patterns
- **Key Features**: FastAPI, Docker, Kubernetes, monitoring
- **Best For**: DevOps engineers, production deployment
- **Dependencies**: FastAPI, Docker, cloud tools

### Performance Optimization (`06_performance_optimization.py`)
- **Purpose**: Scalability and efficiency optimization
- **Key Features**: Benchmarking, parallel processing, GPU acceleration
- **Best For**: Performance tuning, large-scale deployment
- **Dependencies**: Performance profiling tools, CUDA

### Explainable AI (`07_explainable_ai.py`)
- **Purpose**: Model interpretability and explanations
- **Key Features**: SHAP, LIME, counterfactual explanations
- **Best For**: Understanding model decisions, regulatory compliance
- **Dependencies**: SHAP, LIME, interpretability libraries

### Streaming & Real-time (`08_streaming_realtime.py`)
- **Purpose**: Real-time anomaly detection
- **Key Features**: Kafka integration, WebSockets, concept drift
- **Best For**: Real-time applications, streaming data
- **Dependencies**: Kafka, WebSocket libraries

### Model Management (`09_model_management.py`)
- **Purpose**: MLOps and model lifecycle management
- **Key Features**: Versioning, A/B testing, automated retraining
- **Best For**: ML engineering, model governance
- **Dependencies**: MLflow, versioning tools

### Industry Use Cases (`10_industry_use_cases.py`)
- **Purpose**: Domain-specific applications
- **Key Features**: Financial fraud, IoT monitoring, healthcare
- **Best For**: Industry-specific implementations
- **Dependencies**: Domain-specific libraries

### CLI & Integrations (`11_cli_and_integrations.py`)
- **Purpose**: Command-line workflows and system integration
- **Key Features**: Batch processing, API clients, workflow automation
- **Best For**: System integration, automation
- **Dependencies**: CLI tools, API clients

## 🎯 Learning Paths

### **Path 1: Data Scientist** 
`01_quickstart.py` → `02_data_loading.py` → `03_visualization.py` → `04_advanced_algorithms.py` → `07_explainable_ai.py`

### **Path 2: ML Engineer**
`01_quickstart.py` → `04_advanced_algorithms.py` → `06_performance_optimization.py` → `09_model_management.py` → `05_production_deployment.py`

### **Path 3: DevOps Engineer**
`01_quickstart.py` → `05_production_deployment.py` → `08_streaming_realtime.py` → `06_performance_optimization.py`

### **Path 4: Domain Expert**
`01_quickstart.py` → `02_data_loading.py` → `10_industry_use_cases.py` → `07_explainable_ai.py`

## 🔧 Troubleshooting

### Common Issues

#### **Import Errors**
```bash
# If you get import errors for the anomaly_detection package
cd ..  # Go to package root
pip install -e .
cd examples/
```

#### **Missing Dependencies**
```bash
# Install specific dependencies based on the error message
pip install <missing_package>

# Or install all dependencies
pip install -r requirements-full.txt
```

#### **Performance Issues**
- Start with smaller datasets for testing
- Check system resources (CPU, memory, GPU)
- Use appropriate algorithm for your data size

#### **Visualization Issues**
```bash
# For Jupyter notebooks
pip install jupyter ipywidgets

# For interactive plots
pip install plotly bokeh

# For inline plots in some environments
pip install ipympl
```

### Getting Help

1. **Check Example Documentation**: Each example has detailed docstrings
2. **Review Prerequisites**: Ensure all dependencies are installed
3. **Start Simple**: Begin with basic examples before advanced ones
4. **Check Logs**: Examples include detailed logging and error messages
5. **Community Support**: Join our community forums for help

## 📈 Performance Expectations

| Example | Dataset Size | Expected Runtime | Memory Usage |
|---------|--------------|------------------|--------------|
| Basic Usage | Small (100-1K) | < 1 minute | < 100MB |
| Quickstart | Medium (1K-10K) | 1-5 minutes | < 500MB |
| Data Loading | Variable | 2-10 minutes | < 1GB |
| Advanced Algorithms | Large (10K-100K) | 5-30 minutes | 1-8GB |
| Production Deployment | Variable | 10-60 minutes | Variable |
| Performance Optimization | Very Large (100K+) | 10-60 minutes | 2-16GB |

## 🎨 Customization

### Using Your Own Data
Replace the synthetic data generation in examples with your own data:

```python
# Instead of generated data
X, y = generate_synthetic_data()

# Use your data
X = pd.read_csv('your_data.csv').values
# or
X = np.load('your_data.npy')
```

### Modifying Algorithms
Examples are designed to be easily customizable:

```python
# Change algorithm parameters
detector = DetectionService()
results = detector.detect_anomalies(
    data=X,
    algorithm='your_preferred_algorithm',
    contamination=0.05,  # Adjust as needed
    **your_custom_parameters
)
```

### Adding New Examples
Follow the established pattern:
1. Create descriptive docstring
2. Include proper error handling
3. Add visualization where appropriate
4. Document prerequisites
5. Update this README

## 📄 License

These examples are part of the anomaly detection package and are subject to the same MIT license terms.

---

**Happy Learning!** 🎉

Start with the quickstart example and work your way through the examples that match your use case and expertise level. Each example is designed to be educational, practical, and directly applicable to real-world scenarios.