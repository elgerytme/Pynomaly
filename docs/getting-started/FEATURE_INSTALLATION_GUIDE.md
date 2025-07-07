# Feature Installation Guide

🍞 **Breadcrumb:** 🏠 [Home](../index.md) > 🚀 [Getting Started](README.md) > 📄 Feature Installation

---

This guide helps you install only the Pynomaly features you need, avoiding unnecessary dependencies and large downloads.

## 🎯 Quick Installation Options

### 1. **Complete Server** (Recommended for most users)
```bash
pip install -e ".[server]"
```
**Includes:** CLI, API, web interface, data processing  
**Size:** ~200MB  
**Use case:** Full anomaly detection platform

### 2. **Research & ML** (For data scientists)
```bash
pip install -e ".[server,automl,explainability]"
```
**Includes:** Server + AutoML + SHAP/LIME explanations  
**Size:** ~400MB  
**Use case:** ML research and model interpretability

### 3. **Production Deployment**
```bash
pip install -e ".[production]"
```
**Includes:** Authentication, monitoring, enterprise features  
**Size:** ~300MB  
**Use case:** Production-ready deployment

### 4. **Minimal Core** (Lightweight)
```bash
pip install -e ".[minimal]"
```
**Includes:** Core PyOD algorithms only  
**Size:** ~100MB  
**Use case:** Basic anomaly detection

## 🛠️ Interactive Installer

Use our interactive installer to guide you through the options:

```bash
python scripts/setup/install_features.py
```

This script will:
- Show all available features
- Recommend combinations based on your use case
- Install only what you need
- Provide usage examples

## 📋 Feature Categories

### **Core Features**
| Feature | Install Command | Description | Size |
|---------|----------------|-------------|------|
| `minimal` | `pip install -e ".[minimal]"` | Core PyOD algorithms | ~100MB |
| `standard` | `pip install -e ".[standard]"` | Core + data formats | ~150MB |

### **Interfaces**
| Feature | Install Command | Description | Size |
|---------|----------------|-------------|------|
| `cli` | `pip install -e ".[cli]"` | Command-line interface | ~50MB |
| `api` | `pip install -e ".[api]"` | REST API server | ~100MB |
| `server` | `pip install -e ".[server]"` | Complete server (CLI + API) | ~200MB |

### **Advanced ML**
| Feature | Install Command | Description | Size |
|---------|----------------|-------------|------|
| `automl` | `pip install -e ".[automl]"` | Optuna + auto-sklearn | ~300MB |
| `explainability` | `pip install -e ".[explainability]"` | SHAP + LIME | ~200MB |
| `torch` | `pip install -e ".[torch]"` | PyTorch deep learning | ~2GB |
| `tensorflow` | `pip install -e ".[tensorflow]"` | TensorFlow neural networks | ~2GB |
| `jax` | `pip install -e ".[jax]"` | JAX high-performance | ~500MB |
| `graph` | `pip install -e ".[graph]"` | Graph anomaly detection | ~1GB |

### **Production Features**
| Feature | Install Command | Description | Size |
|---------|----------------|-------------|------|
| `auth` | `pip install -e ".[auth]"` | JWT authentication | ~20MB |
| `monitoring` | `pip install -e ".[monitoring]"` | Prometheus metrics | ~50MB |
| `production` | `pip install -e ".[production]"` | Full production stack | ~300MB |

### **Development Tools**
| Feature | Install Command | Description | Size |
|---------|----------------|-------------|------|
| `test` | `pip install -e ".[test]"` | Testing dependencies | ~100MB |
| `dev` | `pip install -e ".[dev]"` | Development tools | ~150MB |
| `lint` | `pip install -e ".[lint]"` | Code quality tools | ~50MB |

## 🔄 Combining Features

You can install multiple feature groups at once:

```bash
# ML Research Setup
pip install -e ".[server,automl,explainability,torch]"

# Production with Monitoring
pip install -e ".[production,monitoring]"

# Development Environment
pip install -e ".[dev,test,lint]"

# Everything (not recommended unless needed)
pip install -e ".[all]"
```

## ⚠️ Important Notes

### **Large Dependencies**
- **PyTorch (`torch`)**: ~2GB download
- **TensorFlow (`tensorflow`)**: ~2GB download  
- **Graph ML (`graph`)**: ~1GB download

### **System Requirements**
Some features may require additional system dependencies:

- **Deep Learning**: CUDA for GPU support
- **Graph ML**: Additional C++ libraries
- **AutoML**: Significant computational resources

### **Optional Feature Warnings**
If you see warnings like "SHAP not available", it means optional features aren't installed:

```bash
# Fix SHAP warnings
pip install -e ".[explainability]"

# Fix deep learning warnings  
pip install -e ".[torch]"
```

## 🚀 Verification

After installation, verify your setup:

```bash
# Test CLI
python scripts/run/cli.py --help

# Test API (if installed)
python scripts/run/run_api.py --help

# Test features
python -c "import pynomaly; print('✅ Core installation working')"

# Test optional features
python -c "import shap; print('✅ SHAP available')" 2>/dev/null || echo "⚠️ SHAP not installed"
python -c "import torch; print('✅ PyTorch available')" 2>/dev/null || echo "⚠️ PyTorch not installed"
```

## 🆘 Troubleshooting

### **Installation Fails**
1. **Update pip**: `python -m pip install --upgrade pip`
2. **Check Python version**: Requires Python 3.11+
3. **Clear cache**: `pip cache purge`
4. **Virtual environment**: Always use a virtual environment

### **Import Errors**
```bash
# Reinstall in editable mode
pip install -e ".[your-features]"

# Check installation
pip show pynomaly
```

### **Large Download Issues**
For slow connections, install features separately:
```bash
# Install core first
pip install -e ".[server]"

# Add features one at a time
pip install -e ".[automl]"
pip install -e ".[explainability]"
```

## 🔗 Related Documentation

- [Installation Guide](installation.md) - Basic installation instructions
- [Quickstart Guide](quickstart.md) - Getting started after installation
- [Development Setup](../developer-guides/README.md) - Development environment

---

📍 **Location**: `docs/getting-started/`  
🏠 **Documentation Home**: [docs/](../README.md)
