# Installation Guide

**Quick installation guide for Pynomaly anomaly detection framework**

## Quick Install

```bash
# Install from PyPI (recommended)
pip install pynomaly

# Or install from source
git clone https://github.com/elgerytme/pynomaly.git
cd pynomaly
pip install -e .
```

## System Requirements

### Minimum Requirements
- **Python**: 3.11+
- **Memory**: 4GB RAM
- **Storage**: 2GB free space
- **OS**: Windows, macOS, Linux

### Recommended Requirements
- **Python**: 3.11+ with virtual environment
- **Memory**: 8GB+ RAM
- **Storage**: 10GB+ free space
- **GPU**: CUDA-compatible GPU (optional, for deep learning models)

## Installation Methods

### 1. PyPI Installation (Recommended)

```bash
# Create virtual environment
python -m venv pynomaly-env
source pynomaly-env/bin/activate  # On Windows: pynomaly-env\Scripts\activate

# Install Pynomaly
pip install pynomaly

# Verify installation
python -c "import pynomaly; print(pynomaly.__version__)"
```

### 2. Development Installation

```bash
# Clone repository
git clone https://github.com/elgerytme/pynomaly.git
cd pynomaly

# Create development environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install in development mode
pip install -e ".[dev]"

# Run tests to verify
pytest tests/
```

### 3. Docker Installation

```bash
# Pull pre-built image
docker pull pynomaly/pynomaly:latest

# Or build from source
docker build -t pynomaly .

# Run container
docker run -p 8000:8000 pynomaly/pynomaly:latest
```

## Optional Dependencies

### Machine Learning Backends

```bash
# Scikit-learn (included by default)
pip install "pynomaly[sklearn]"

# PyOD for additional algorithms
pip install "pynomaly[pyod]"

# Deep learning support
pip install "pynomaly[deep]"

# Graph anomaly detection
pip install "pynomaly[graph]"

# All optional dependencies
pip install "pynomaly[all]"
```

### Web Interface

```bash
# Install web UI dependencies
pip install "pynomaly[web]"

# Start web interface
pynomaly web start
```

### Enterprise Features

```bash
# Install enterprise packages
pip install pynomaly-enterprise

# Configure enterprise features
pynomaly enterprise configure
```

## Configuration

### Environment Setup

```bash
# Create configuration directory
mkdir ~/.pynomaly

# Generate default configuration
pynomaly config init

# Edit configuration
nano ~/.pynomaly/config.yaml
```

### Basic Configuration

```yaml
# ~/.pynomaly/config.yaml
app:
  name: "Pynomaly"
  debug: false
  
detection:
  default_algorithm: "IsolationForest"
  threshold: 0.1
  
data:
  cache_enabled: true
  cache_ttl: 3600

logging:
  level: "INFO"
  format: "structured"
```

## Verification

### Quick Test

```python
from pynomaly import AnomalyDetector
import numpy as np

# Generate sample data
data = np.random.randn(1000, 5)
data[0] = [10, 10, 10, 10, 10]  # Add anomaly

# Create detector
detector = AnomalyDetector()

# Fit and detect
detector.fit(data)
results = detector.detect(data)

print(f"Detected {results.anomaly_count} anomalies")
```

### CLI Test

```bash
# Check CLI is working
pynomaly --help

# Run sample detection
pynomaly detect --algorithm IsolationForest --data sample_data.csv

# Check status
pynomaly status
```

### Web Interface Test

```bash
# Start web server
pynomaly web start --port 8000

# Open browser to http://localhost:8000
# Should see Pynomaly dashboard
```

## Troubleshooting

### Common Issues

1. **Python Version Error**:
   ```bash
   # Check Python version
   python --version
   # Should be 3.11+
   ```

2. **Permission Denied**:
   ```bash
   # Install for user only
   pip install --user pynomaly
   ```

3. **Memory Error**:
   ```bash
   # Install with minimal dependencies
   pip install pynomaly --no-deps
   pip install numpy pandas scikit-learn
   ```

4. **Import Error**:
   ```bash
   # Check installation
   pip show pynomaly
   
   # Reinstall if needed
   pip uninstall pynomaly
   pip install pynomaly
   ```

### Platform-Specific Issues

**Windows**:
```bash
# If Visual C++ errors occur
pip install --upgrade setuptools wheel
```

**macOS**:
```bash
# If Xcode command line tools missing
xcode-select --install
```

**Linux**:
```bash
# If compilation errors occur
sudo apt-get install build-essential python3-dev
# or
sudo yum install gcc python3-devel
```

## Next Steps

After successful installation:

1. **[Quick Start Tutorial](./quick-start.md)** - Your first anomaly detection
2. **[Basic Usage Guide](../user-guides/basic-usage.md)** - Core features and workflows
3. **[API Reference](../api-reference/core-api.md)** - Complete API documentation
4. **[Examples](../examples/basic-examples.md)** - Practical examples and use cases

## Getting Help

- 📖 [Documentation](../README.md)
- 🐛 [Report Issues](https://github.com/elgerytme/pynomaly/issues)
- 💬 [Community Discussions](https://github.com/elgerytme/pynomaly/discussions)
- 📧 [Support Email](mailto:support@pynomaly.io)

---

*Installation guide last updated: 2025-01-14*