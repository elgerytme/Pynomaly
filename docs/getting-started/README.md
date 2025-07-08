# Getting Started

🍞 **Breadcrumb:** 🏠 [Home](../index.md) > 🚀 [Getting Started](README.md)

---


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

## 📊 Benchmarking Your Setup

Once you have Pynomaly installed, you can benchmark your setup to ensure optimal performance:

### Prerequisites for Benchmarking

- **Python 3.11+** (confirmed working)
- **Sufficient RAM**: At least 2GB for basic benchmarking, 8GB+ recommended for larger datasets
- **Sample datasets**: Use included sample datasets or your own CSV files

### Installing Benchmarking Extras

For comprehensive benchmarking capabilities, install the performance monitoring extras:

```bash
# Install with performance monitoring and benchmarking features
pip install "pynomaly[monitoring,test]"

# Or install with all extras for full feature set
pip install "pynomaly[all]"

# For development/testing with benchmarking
pip install "pynomaly[test,ui-test]"
```

### Obtaining Sample Datasets

Pynomaly includes several sample datasets for testing and benchmarking:

```bash
# List available sample datasets
pynomaly dataset list-samples

# Download sample datasets (if not already included)
pynomaly dataset download-samples

# View sample dataset information
pynomaly dataset info examples/sample_datasets/synthetic/financial_fraud.csv
```

**Available Sample Datasets:**
- `examples/sample_datasets/synthetic/financial_fraud.csv` (10,000 samples, 9 features, 2% anomalies)
- `examples/sample_datasets/synthetic/network_intrusion.csv` (8,000 samples, 11 features, 5% anomalies)
- `examples/sample_datasets/synthetic/iot_sensors.csv` (12,000 samples, 10 features, 3% anomalies)
- `examples/sample_datasets/real_world/kdd_cup_1999.csv` (10,000 samples, 41 features, 3.3% anomalies)

You can also create a simple 1MB test dataset:

```bash
# Generate a 1MB test dataset
pynomaly dataset generate --size 1MB --name test_1mb --format csv --output test_1mb.csv
```

### Minimal Benchmark CLI Example

Here's a minimal example to benchmark anomaly detection on a 1MB CSV file:

```bash
# Step 1: Load your 1MB dataset
pynomaly dataset load test_1mb.csv --name "Test 1MB Dataset"

# Step 2: Run a simple benchmark with default algorithms
pynomaly benchmark comprehensive \
  --suite-name "Quick Benchmark" \
  --description "Basic performance test on 1MB dataset" \
  --algorithms IsolationForest LOF OneClassSVM \
  --dataset-sizes 1000 5000 \
  --iterations 3 \
  --output-dir ./benchmark_results \
  --export-format html

# Step 3: View benchmark results
pynomaly benchmark list-results

# Step 4: View system information for context
pynomaly benchmark system-info
```

**Expected Output:**
```bash
✓ Benchmark Suite Complete
Suite: Quick Benchmark
Duration: 45.32 seconds
Total Tests: 6
Algorithms: 3
Performance Grade: B+

┌─────────────────┬─────────────┬─────────────────┬─────────────────┬─────────────────┐
│ Algorithm       │ Avg Time (s)│ Avg Memory (MB) │ Avg Throughput  │ Avg Accuracy    │
├─────────────────┼─────────────┼─────────────────┼─────────────────┼─────────────────┤
│ IsolationForest │ 0.123       │ 45.2            │ 8130.5          │ 0.867           │
│ LOF             │ 0.456       │ 78.9            │ 2193.4          │ 0.834           │
│ OneClassSVM     │ 1.234       │ 67.3            │ 810.7           │ 0.792           │
└─────────────────┴─────────────┴─────────────────┴─────────────────┴─────────────────┘

Benchmark report saved to: ./benchmark_results/benchmark_report_quick_benchmark.html
```

### Advanced Benchmarking Examples

```bash
# Memory stress test
pynomaly benchmark memory-stress \
  --algorithm IsolationForest \
  --max-size 100000 \
  --memory-limit 4096.0 \
  --output-file memory_test.json

# Scalability test
pynomaly benchmark scalability \
  --algorithm LOF \
  --base-size 1000 \
  --scale-factors 2 4 8 16 \
  --output-file scalability_test.json

# Throughput benchmark
pynomaly benchmark throughput \
  --algorithms IsolationForest LOF \
  --dataset-sizes 1000 5000 10000 \
  --duration 60 \
  --output-file throughput_test.json
```

### Performance Optimization Tips

1. **Start Small**: Begin with smaller datasets (1,000-10,000 samples) to establish baseline performance
2. **Monitor Resources**: Use `pynomaly benchmark system-info` to check available resources
3. **Choose Appropriate Algorithms**: IsolationForest is generally fastest for initial testing
4. **Batch Processing**: For large datasets, use the `--dataset-sizes` parameter to test incremental sizes
5. **Export Results**: Always export benchmark results with `--export-format html` for detailed analysis

### Troubleshooting Benchmark Issues

**Memory Issues:**
```bash
# Check system resources
pynomaly benchmark system-info

# Run with smaller dataset sizes
pynomaly benchmark comprehensive --dataset-sizes 500 1000 2000
```

**Performance Issues:**
```bash
# Test single algorithm first
pynomaly benchmark comprehensive --algorithms IsolationForest --iterations 1

# Check for resource constraints
htop  # or Task Manager on Windows
```

**Command Not Found:**
```bash
# Verify CLI installation
pynomaly --version
pynomaly --help

# Check if benchmark commands are available
pynomaly benchmark --help
```

---

## 🎯 Next Steps

### **After Installation**
1. **[Complete the Quickstart](quickstart.md)** - Your first detection
2. **[Run Benchmarks](#-benchmarking-your-setup)** - Test your setup performance
3. **[Explore Examples](../examples/)** - Real-world use cases  
4. **[Read User Guides](../user-guides/)** - Feature documentation
5. **[Check API Reference](../developer-guides/api-integration/)** - Programming interfaces

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
