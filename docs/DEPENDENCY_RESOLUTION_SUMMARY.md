# Dependency Resolution - Optional Feature Simplification

## Summary

This document describes the implementation of Step 2: Dependency Resolution for the Pynomaly project, which introduces a simplified and platform-aware dependency management system.

## 📋 Task Completion Status

✅ **COMPLETED**: All 5 subtasks have been successfully implemented

### 1. ✅ Audit Current Optional Requirements

**Implemented**: `scripts/audit_dependencies.py`

- Audited heavyweight dependencies: `torch`, `tensorflow`, `jax`, `optuna`, `hyperopt`, `auto-sklearn2`, `shap`, `lime`, `ray`
- Identified size impact: PyTorch (~2GB with CUDA), TensorFlow (~1.5GB with GPU), JAX (~500MB with CUDA)
- Current status: Deep learning frameworks installed, AutoML/explainability packages missing

### 2. ✅ Create New Extras in pyproject.toml

**Updated**: `pyproject.toml` with enhanced extras structure

#### New Simplified Extras:
- `pynomaly[automl]` - Basic AutoML with Optuna and HyperOpt
- `pynomaly[deep]` - All deep learning frameworks with platform markers
- `pynomaly[deep-cpu]` - CPU-only deep learning (lighter installations)
- `pynomaly[deep-gpu]` - GPU-enabled deep learning with CUDA support
- `pynomaly[explainability]` - SHAP and LIME model explanations
- `pynomaly[automl-advanced]` - AutoML with Ray Tune for distributed computing

#### Platform Markers Implemented:
```toml
"torch>=2.5.1; platform_machine != 'aarch64'"
"tensorflow-macos>=2.18.0,<2.20.0; sys_platform == 'darwin' and platform_machine == 'aarch64'"
"auto-sklearn2>=1.0.0; platform_system == 'Linux'"  # Linux only
```

### 3. ✅ Introduce Lightweight Stubs

**Created**: `src/pynomaly/utils/dependency_stubs.py`

#### Features:
- **Smart dependency checking** using `importlib.metadata`
- **Actionable error messages** with installation commands
- **Optional import stubs** that provide helpful guidance
- **Status reporting** with missing dependencies summary

#### Key Functions:
```python
def check_dependency(package: str) -> bool
def require_dependency(package: str) -> None
def create_optional_import_stub(package: str, feature_name: str)
def print_dependency_status()
```

#### Enhanced Adapter Integration:
Updated `src/pynomaly/infrastructure/adapters/deep_learning/__init__.py` to use the new dependency management system.

### 4. ✅ Update Dockerfiles & CI

#### New Dockerfiles:
- `deploy/docker/Dockerfile.deep-cpu` - CPU-only deep learning setup
- `deploy/docker/Dockerfile.automl-advanced` - AutoML with distributed computing
- Updated `deploy/docker/Dockerfile` with build arguments for extras

#### CI Enhancements (`.github/workflows/ci.yml`):
- **Dependency extras testing matrix** for Python 3.11 and 3.12
- **Docker build matrix** for different variants (standard, deep-cpu, automl-advanced)
- **Automated testing** of extras installation and dependency stubs

### 5. ✅ Add README Quick-Install Matrix and Badges

**Updated**: `README.md`

#### Quick Install Matrix:
| Feature | Installation Command | Description |
|---------|---------------------|-------------|
| **AutoML** | `pip install pynomaly[automl]` | Basic automated ML with Optuna and HyperOpt |
| **Deep Learning** | `pip install pynomaly[deep]` | PyTorch, TensorFlow, JAX |
| **Explainability** | `pip install pynomaly[explainability]` | SHAP and LIME-based model explanations |
| **AutoML Advanced** | `pip install pynomaly[automl-advanced]` | AutoML with Ray Tune for distributed computing |
| **Deep (CPU-only)** | `pip install pynomaly[deep-cpu]` | Lightweight CPU-only installations |

#### Code Compatibility Badges:
- ![PyTorch Compatible](https://img.shields.io/badge/PyTorch-Compatible-brightgreen)
- ![TensorFlow Compatible](https://img.shields.io/badge/TensorFlow-Compatible-brightgreen)
- ![JAX Compatible](https://img.shields.io/badge/JAX-Compatible-brightgreen)
- ![AutoML Toolkit](https://img.shields.io/badge/AutoML-Ready-brightgreen)

## 🔧 Testing Tools

### Dependency Audit Script
```bash
python scripts/audit_dependencies.py
```

### Dependency Status Check
```bash
python src/pynomaly/utils/dependency_stubs.py
```

### Comprehensive Extras Testing
```bash
python scripts/test_extras.py
```

## 🏗️ Architecture Benefits

### 1. **Platform Awareness**
- ARM64 Mac compatibility for TensorFlow
- Linux-specific packages for auto-sklearn2
- CPU vs GPU variants for deep learning frameworks

### 2. **Graceful Degradation**
- Missing dependencies don't break imports
- Actionable error messages guide users to correct installations
- Informative stubs explain what features require which packages

### 3. **Development Experience**
- Clear error messages with exact installation commands
- Status reporting shows what's available vs missing
- CI matrix testing ensures compatibility across Python versions

### 4. **Production Readiness**
- Docker variants for different deployment scenarios
- Minimal base installation with optional feature additions
- Platform-specific optimizations

## 📦 Usage Examples

### Basic Installation
```bash
pip install pynomaly
```

### Feature-Specific Installations
```bash
# AutoML features
pip install pynomaly[automl]

# Deep learning (CPU-only for lighter installations)
pip install pynomaly[deep-cpu]

# Full machine learning stack
pip install pynomaly[ml-all]

# Production deployment
pip install pynomaly[production]
```

### Docker Deployments
```bash
# Standard deployment
docker build -f deploy/docker/Dockerfile -t pynomaly:standard .

# CPU-only deep learning
docker build -f deploy/docker/Dockerfile.deep-cpu -t pynomaly:deep-cpu .

# AutoML with distributed computing
docker build -f deploy/docker/Dockerfile.automl-advanced -t pynomaly:automl-advanced .
```

## 🚀 Impact

1. **Reduced Installation Size**: Core package is now lightweight with optional heavy dependencies
2. **Better User Experience**: Clear guidance when features require additional dependencies
3. **Platform Compatibility**: Proper support for ARM64 Macs, Linux-specific packages
4. **CI/CD Efficiency**: Matrix testing ensures compatibility across different dependency combinations
5. **Production Flexibility**: Multiple Docker variants for different deployment scenarios

## 🎯 Next Steps

The dependency resolution system is now complete and ready for production use. Users can install exactly what they need, with clear guidance on adding additional capabilities as required.
