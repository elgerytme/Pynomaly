# Environment Setup Guide

🍞 **Breadcrumb:** 🏠 [Home](../../index.md) > 📚 [User Guides](../README.md) > 🚀 [Onboarding](README.md) > ⚙️ Environment Setup

---

This guide will help you set up your development environment for Pynomaly. Choose the method that best fits your needs and experience level.

## 📋 Prerequisites

### System Requirements

- **Python 3.11+** (3.12 recommended)
- **Operating System**: Windows 10+, macOS 10.15+, Linux (Ubuntu 20.04+)
- **Memory**: 4 GB RAM minimum (8 GB recommended)
- **Disk Space**: 2 GB free space
- **Internet Connection**: Required for package downloads

### Required Software

1. **Python** - [Download from python.org](https://www.python.org/downloads/)
2. **Git** - [Download from git-scm.com](https://git-scm.com/downloads)
3. **Code Editor** - [VS Code](https://code.visualstudio.com/) (recommended) or your preferred editor

## 🚀 Installation Methods

### Method 1: Quick Install (Recommended for Beginners)

Perfect for evaluation and getting started quickly.

```bash
# Install Pynomaly with basic features
pip install pynomaly

# Verify installation
python -c "import pynomaly; print(f'✅ Pynomaly {pynomaly.__version__} installed successfully!')"

# Start the interactive tutorial
pynomaly tutorial
```

**Pros:**

- ✅ Fastest setup (2-3 minutes)
- ✅ No configuration needed
- ✅ Perfect for evaluation

**Cons:**

- ❌ Limited customization
- ❌ May have version conflicts

---

### Method 2: Development Install (Recommended for Contributors)

Best for development, customization, and contributing to the project.

```bash
# Clone the repository
git clone https://github.com/your-org/pynomaly.git
cd pynomaly

# Install Hatch (modern Python build tool)
pip install hatch

# Create development environment
hatch env create

# Activate the environment
hatch shell

# Install in development mode
hatch env run dev:install

# Verify installation
hatch env run dev:test
```

**Pros:**

- ✅ Latest features and fixes
- ✅ Easy to contribute back
- ✅ Full control over environment
- ✅ Integrated development tools

**Cons:**

- ❌ Requires more setup time
- ❌ May encounter development issues

---

### Method 3: Docker Install (Recommended for Production)

Ideal for containerized environments and production deployments.

```bash
# Pull the latest image
docker pull pynomaly/pynomaly:latest

# Run with default configuration
docker run -p 8000:8000 pynomaly/pynomaly

# Run with custom configuration
docker run -p 8000:8000 -v $(pwd)/config:/app/config pynomaly/pynomaly

# Access the web interface
open http://localhost:8000
```

**Pros:**

- ✅ Consistent environment
- ✅ Easy deployment
- ✅ Isolated dependencies
- ✅ Production-ready

**Cons:**

- ❌ Requires Docker knowledge
- ❌ Less flexible for development

---

### Method 4: Virtual Environment Install (Traditional)

Classic Python approach with full control.

```bash
# Create virtual environment
python -m venv pynomaly-env

# Activate environment
# On Windows:
pynomaly-env\Scripts\activate
# On macOS/Linux:
source pynomaly-env/bin/activate

# Upgrade pip
pip install --upgrade pip

# Install Pynomaly with extras
pip install "pynomaly[all]"

# Verify installation
python -c "import pynomaly; print('Installation successful!')"
```

**Pros:**

- ✅ Full control over dependencies
- ✅ Standard Python approach
- ✅ Works everywhere

**Cons:**

- ❌ Manual dependency management
- ❌ More commands to remember

## ⚙️ Configuration

### Environment Variables

Create a `.env` file in your project directory:

```bash
# Database Configuration
DATABASE_URL=sqlite:///./pynomaly.db

# Redis Configuration (optional)
REDIS_URL=redis://localhost:6379/0

# Security Settings
SECRET_KEY=your-secret-key-here-make-it-long-and-random
ALLOWED_HOSTS=localhost,127.0.0.1

# ML Settings
DEFAULT_MODEL_PATH=./models
ENABLE_GPU=false

# Logging
LOG_LEVEL=INFO
LOG_FORMAT=json

# API Settings
API_HOST=0.0.0.0
API_PORT=8000
API_WORKERS=1

# Feature Flags
ENABLE_WEB_UI=true
ENABLE_MONITORING=true
ENABLE_SECURITY_FEATURES=true
```

### Configuration File

Create `config/settings.yaml`:

```yaml
# Application Settings
app:
  name: "Pynomaly"
  version: "1.0.0"
  debug: false

# Database Settings
database:
  url: "sqlite:///./pynomaly.db"
  echo: false
  pool_size: 10

# Machine Learning Settings
ml:
  default_algorithm: "IsolationForest"
  model_cache_size: 100
  enable_gpu: false
  preprocessing:
    standardize: true
    handle_missing: true

# Security Settings
security:
  enable_csrf: true
  enable_rate_limiting: true
  max_request_size: 10485760  # 10MB
  session_timeout: 3600

# Monitoring Settings
monitoring:
  enable_metrics: true
  metrics_port: 9090
  enable_health_checks: true
```

## 🧪 Verification Steps

### 1. Basic Import Test

```python
# test_installation.py
import sys
import importlib

def test_imports():
    """Test that all core modules can be imported."""
    modules = [
        'pynomaly',
        'pynomaly.core',
        'pynomaly.algorithms',
        'pynomaly.preprocessing',
        'pynomaly.visualization'
    ]
    
    for module in modules:
        try:
            importlib.import_module(module)
            print(f"✅ {module}")
        except ImportError as e:
            print(f"❌ {module}: {e}")
            return False
    
    return True

if __name__ == "__main__":
    success = test_imports()
    sys.exit(0 if success else 1)
```

```bash
python test_installation.py
```

### 2. Quick Detection Test

```python
# test_detection.py
import numpy as np
import pandas as pd
from pynomaly import detect_anomalies

def test_detection():
    """Test basic anomaly detection functionality."""
    # Generate sample data
    np.random.seed(42)
    normal_data = np.random.normal(0, 1, (1000, 2))
    anomalous_data = np.random.normal(5, 1, (50, 2))
    
    # Combine data
    data = np.vstack([normal_data, anomalous_data])
    df = pd.DataFrame(data, columns=['feature1', 'feature2'])
    
    # Detect anomalies
    try:
        anomalies = detect_anomalies(df, contamination=0.05)
        
        print(f"✅ Detection successful!")
        print(f"   Data shape: {df.shape}")
        print(f"   Anomalies detected: {anomalies.sum()}")
        print(f"   Anomaly rate: {anomalies.mean():.1%}")
        
        return True
        
    except Exception as e:
        print(f"❌ Detection failed: {e}")
        return False

if __name__ == "__main__":
    test_detection()
```

```bash
python test_detection.py
```

### 3. Web Interface Test

```bash
# Start the web server
pynomaly server start

# Or with Hatch
hatch env run dev:serve

# Or with Python
python -m pynomaly.presentation.api.main
```

Open your browser and visit:

- **Main App**: <http://localhost:8000>
- **API Docs**: <http://localhost:8000/docs>
- **Health Check**: <http://localhost:8000/health>

### 4. CLI Test

```bash
# Test CLI commands
pynomaly --version
pynomaly --help

# Test dataset operations
pynomaly dataset list
pynomaly detector algorithms

# Test with sample data (if available)
pynomaly dataset load sample_data.csv --name "Test Data"
pynomaly detector create --name "Test Detector" --algorithm IsolationForest
```

## 🔧 Development Tools Setup

### VS Code Extensions

Install these extensions for the best development experience:

```json
{
  "recommendations": [
    "ms-python.python",
    "ms-python.vscode-pylance",
    "ms-python.black-formatter",
    "charliermarsh.ruff",
    "ms-python.mypy-type-checker",
    "ms-vscode.vscode-jupyter",
    "ms-vscode.vscode-json",
    "redhat.vscode-yaml",
    "ms-vscode.vscode-docker"
  ]
}
```

### Git Hooks (for Contributors)

```bash
# Install pre-commit hooks
pip install pre-commit
pre-commit install

# Test hooks
pre-commit run --all-files
```

### IDE Configuration

Create `.vscode/settings.json`:

```json
{
  "python.defaultInterpreterPath": "./venv/bin/python",
  "python.formatting.provider": "black",
  "python.linting.enabled": true,
  "python.linting.ruffEnabled": true,
  "python.linting.mypyEnabled": true,
  "python.testing.pytestEnabled": true,
  "python.testing.pytestArgs": ["tests/"],
  "files.associations": {
    "*.yaml": "yaml",
    "*.yml": "yaml"
  }
}
```

## 🚨 Troubleshooting

### Common Issues

#### Issue: "ModuleNotFoundError: No module named 'pynomaly'"

**Solutions:**

```bash
# Check if pip installed in correct environment
which pip
pip list | grep pynomaly

# Reinstall in current environment
pip uninstall pynomaly
pip install pynomaly

# Check Python path
python -c "import sys; print(sys.path)"
```

#### Issue: "Permission denied" during installation

**Solutions:**

```bash
# Use user installation
pip install --user pynomaly

# Or use virtual environment
python -m venv venv
source venv/bin/activate  # Linux/Mac
# venv\Scripts\activate  # Windows
pip install pynomaly
```

#### Issue: "SSL Certificate error"

**Solutions:**

```bash
# Upgrade certificates
pip install --upgrade certifi

# Use trusted hosts (temporary fix)
pip install --trusted-host pypi.org --trusted-host pypi.python.org pynomaly
```

#### Issue: "Hatch not found" or Hatch errors

**Solutions:**

```bash
# Install Hatch globally
pip install --user hatch

# Or use pipx
pipx install hatch

# Update PATH if needed
export PATH="$HOME/.local/bin:$PATH"
```

#### Issue: Port 8000 already in use

**Solutions:**

```bash
# Find process using port
lsof -i :8000  # Mac/Linux
netstat -ano | findstr :8000  # Windows

# Use different port
pynomaly server start --port 8080

# Or kill existing process
kill $(lsof -t -i:8000)  # Mac/Linux
```

### Performance Issues

#### Slow Import Times

```bash
# Check for conflicting packages
pip list | grep -E "(pandas|numpy|scikit-learn)"

# Update to latest versions
pip install --upgrade pandas numpy scikit-learn

# Clear Python cache
find . -name "*.pyc" -delete
find . -name "__pycache__" -type d -exec rm -rf {} +
```

#### Memory Issues

```python
# Check memory usage
import psutil
import os

process = psutil.Process(os.getpid())
print(f"Memory usage: {process.memory_info().rss / 1024 / 1024:.1f} MB")

# Reduce memory usage
import pynomaly
pynomaly.set_config(memory_limit="2GB")
```

## 🔄 Environment Management

### Switching Between Environments

```bash
# With Hatch
hatch env show
hatch env create test
hatch env run test:python --version

# With traditional venv
deactivate  # Exit current environment
source different-env/bin/activate
```

### Updating Dependencies

```bash
# With Hatch
hatch env prune  # Remove old environments
hatch env create  # Recreate with latest dependencies

# With pip
pip install --upgrade pynomaly
pip install --upgrade -r requirements.txt
```

### Environment Backup

```bash
# Export current environment
pip freeze > requirements.txt

# With Hatch
hatch env export requirements > requirements-hatch.txt

# Create backup
tar -czf pynomaly-env-backup.tar.gz venv/ .env config/
```

## 📊 Performance Optimization

### For Large Datasets

```python
# Configure for large datasets
import pynomaly

pynomaly.configure({
    'memory_optimization': True,
    'chunk_size': 10000,
    'parallel_processing': True,
    'gpu_acceleration': False  # Set to True if GPU available
})
```

### For Production

```bash
# Install with performance extras
pip install "pynomaly[performance,monitoring]"

# Set environment variables
export PYTHONOPTIMIZE=1
export PYTHONDONTWRITEBYTECODE=1
export OMP_NUM_THREADS=4
```

---

## 🎯 Next Steps

### After Successful Setup

1. **[Complete the Tutorial](first-detection.md)** - Run your first anomaly detection
2. **[Explore Examples](../../examples/README.md)** - See real-world use cases
3. **[Read the User Guide](../basic-usage/README.md)** - Learn core concepts
4. **[API Documentation](../../developer-guides/api-integration/README.md)** - Integrate with your systems

### For Development

1. **[Contributing Guide](../../developer-guides/contributing/README.md)** - Contribute to the project
2. **[Architecture Guide](../../developer-guides/architecture/README.md)** - Understand the system design
3. **[Testing Guide](../../developer-guides/testing/README.md)** - Write and run tests

### For Production

1. **[Deployment Guide](../../deployment/README.md)** - Deploy to production
2. **[Security Guide](../../deployment/SECURITY.md)** - Secure your installation
3. **[Monitoring Guide](../basic-usage/monitoring.md)** - Monitor your system

---

**Environment setup complete!** 🎉

Ready to detect some anomalies? Start with the **[First Detection Tutorial](first-detection.md)** →
