# Getting Started

Welcome to Pynomaly! This section will get you up and running with anomaly detection in minutes.

## 🚀 Quick Start Path

### **1. Installation** ⏱️ 2 minutes
**[→ Install Pynomaly](installation.md)**
- Multiple installation methods (pip, Hatch, Docker)
- System requirements and prerequisites
- Virtual environment setup

### **2. First Detection** ⏱️ 5 minutes  
**[→ Quickstart Guide](quickstart.md)**
- Your first anomaly detection in 5 lines of code
- Basic concepts and workflow
- Interactive examples

### **3. Platform Setup** ⏱️ 10 minutes
**[→ Platform-Specific Guides](platform-specific/)**
- Windows, macOS, Linux specific instructions
- Python version management
- IDE and development environment setup

### **4. CLI Setup** ⏱️ 3 minutes
**[→ Command-Line Interface](SETUP_CLI.md)**
- CLI installation and configuration
- Basic command usage
- Shell completion setup

---

## 🎯 Choose Your Installation Method

### **Quick Install (Recommended)**
```bash
pip install pynomaly
```
Best for: Quick evaluation, simple use cases

### **Development Install**
```bash
# Clone repository
git clone https://github.com/your-org/pynomaly.git
cd pynomaly

# Install with Hatch
pip install hatch
hatch env create
hatch shell
```
Best for: Contributors, advanced users, custom development

### **Docker Install**
```bash
docker pull pynomaly/pynomaly:latest
docker run -p 8000:8000 pynomaly/pynomaly
```
Best for: Production deployment, isolated environments

### **Enterprise Install**
```bash
# Install with all enterprise features
pip install "pynomaly[enterprise]"
```
Best for: Production systems, advanced features

---

## 📋 System Requirements

### **Minimum Requirements**
- **Python**: 3.11 or higher
- **Memory**: 2 GB RAM
- **Disk**: 1 GB free space
- **OS**: Windows 10+, macOS 10.15+, Linux (Ubuntu 20.04+)

### **Recommended Requirements**
- **Python**: 3.11 (latest stable)
- **Memory**: 8 GB RAM (for large datasets)
- **Disk**: 5 GB free space (including ML libraries)
- **GPU**: CUDA-compatible (optional, for deep learning)

### **Dependencies**
- **Core**: pandas, numpy, scikit-learn
- **ML Libraries**: PyOD, PyTorch (optional), TensorFlow (optional)
- **Web**: FastAPI, HTMX, Tailwind CSS
- **Build**: Hatch, Ruff, MyPy

---

## 🛣️ Learning Paths

### **Data Scientist (First Time)**
1. **[Installation](installation.md)** - Get Pynomaly installed
2. **[Quickstart](quickstart.md)** - First anomaly detection
3. **[User Guides](../user-guides/basic-usage/)** - Learn core concepts
4. **[Algorithm Selection](../reference/algorithms/)** - Choose algorithms

### **ML Engineer**
1. **[Development Install](installation.md#development-install)** - Full development setup
2. **[CLI Setup](SETUP_CLI.md)** - Command-line tools
3. **[API Integration](../developer-guides/api-integration/)** - Programming interfaces
4. **[Architecture](../developer-guides/architecture/)** - System design

### **DevOps Engineer**
1. **[Docker Install](installation.md#docker-install)** - Containerized deployment
2. **[Platform Setup](platform-specific/)** - Environment configuration
3. **[Deployment Guides](../deployment/)** - Production deployment
4. **[Monitoring](../user-guides/basic-usage/monitoring.md)** - System observability

### **Business Analyst**
1. **[Quick Install](installation.md#quick-install)** - Simple setup
2. **[Quickstart](quickstart.md)** - Basic usage
3. **[Examples](../examples/)** - Real-world use cases
4. **[User Guides](../user-guides/)** - Feature documentation

---

## 🔧 Platform-Specific Setup

### **[Windows](platform-specific/WINDOWS_SETUP_GUIDE.md)**
- Windows-specific installation steps
- PowerShell and Command Prompt usage
- WSL2 setup for Linux compatibility
- Visual Studio Code integration

### **[Multi-Python Versions](platform-specific/MULTI_PYTHON_VERSIONS.md)**
- Managing multiple Python installations
- pyenv for version management
- Virtual environment best practices
- Version compatibility matrix

---

## ✅ Verification Steps

After installation, verify your setup:

### **1. Python Package Import**
```python
import pynomaly
print(f"Pynomaly version: {pynomaly.__version__}")
```

### **2. CLI Command**
```bash
pynomaly --version
pynomaly --help
```

### **3. Quick Detection Test**
```python
from pynomaly import detect_anomalies
import pandas as pd
import numpy as np

# Generate test data
data = pd.DataFrame({
    'feature1': np.random.normal(0, 1, 1000),
    'feature2': np.random.normal(0, 1, 1000)
})

# Add some anomalies
data.iloc[990:] = data.iloc[990:] * 5

# Detect anomalies
anomalies = detect_anomalies(data, contamination=0.01)
print(f"Detected {anomalies.sum()} anomalies")
```

### **4. Web Interface (Optional)**
```bash
pynomaly server start
# Visit http://localhost:8000
```

---

## 🚨 Common Issues & Solutions

### **Installation Problems**
- **Python Version**: Ensure Python 3.11+
- **Permissions**: Use `--user` flag for pip install
- **Dependencies**: Install with `--no-deps` and resolve manually
- **Virtual Environment**: Always use isolated environments

### **Import Errors**
- **Missing Dependencies**: `pip install -r requirements.txt`
- **Path Issues**: Check PYTHONPATH environment variable
- **Version Conflicts**: Update to latest compatible versions

### **Performance Issues**
- **Memory**: Increase available RAM or use sampling
- **Speed**: Install optional accelerated libraries
- **GPU**: Install CUDA-compatible PyTorch/TensorFlow

### **Need Help?**
- **[Troubleshooting Guide](../user-guides/troubleshooting/)** - Detailed problem solving
- **[GitHub Issues](https://github.com/your-org/pynomaly/issues)** - Report bugs
- **[Discussions](https://github.com/your-org/pynomaly/discussions)** - Ask questions

---

## 🎯 Next Steps

### **After Installation**
1. **[Complete the Quickstart](quickstart.md)** - Your first detection
2. **[Explore Examples](../examples/)** - Real-world use cases  
3. **[Read User Guides](../user-guides/)** - Feature documentation
4. **[Check API Reference](../developer-guides/api-integration/)** - Programming interfaces

### **For Development**
1. **[Development Setup](../developer-guides/contributing/)** - Contributor guide
2. **[Architecture Overview](../developer-guides/architecture/)** - System design
3. **[Testing Guide](../developer-guides/contributing/COMPREHENSIVE_TEST_ANALYSIS.md)** - Test infrastructure

### **For Production**
1. **[Deployment Guides](../deployment/)** - Production deployment
2. **[Security Setup](../deployment/SECURITY.md)** - Security configuration
3. **[Monitoring](../user-guides/basic-usage/monitoring.md)** - System observability

---

**Ready to detect anomalies?** Start with the **[Quickstart Guide](quickstart.md)** → 🚀