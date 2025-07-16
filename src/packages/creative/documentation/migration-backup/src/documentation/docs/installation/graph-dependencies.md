# Graph Anomaly Detection Dependencies

This guide covers the installation and setup of dependencies required for graph anomaly detection capabilities in Pynomaly.

## Overview

Graph anomaly detection in Pynomaly is powered by the PyGOD (Python Graph Outlier Detection) library, which requires additional dependencies including PyTorch and PyTorch Geometric.

## Quick Installation

### Option 1: Using Extras (Recommended)

```bash
# Install Pynomaly with graph dependencies
pip install "pynomaly[graph]"
```

### Option 2: Manual Installation

```bash
# Install core dependencies first
pip install pynomaly

# Install graph dependencies
pip install pygod torch torch-geometric
```

### Option 3: Full Installation with All Features

```bash
# Install everything including graph capabilities
pip install "pynomaly[all]"
```

## Detailed Installation Guide

### Prerequisites

- **Python**: 3.11 or higher
- **Operating System**: Linux, macOS, or Windows
- **Hardware**:
  - **CPU**: Any modern CPU (minimum 2 cores recommended)
  - **Memory**: 4GB RAM minimum, 8GB+ recommended for large graphs
  - **GPU**: Optional but recommended for deep learning algorithms (NVIDIA GPU with CUDA support)

### Step-by-Step Installation

#### 1. Create Virtual Environment (Recommended)

```bash
# Create virtual environment
python -m venv pynomaly-graph-env

# Activate virtual environment
# On Linux/macOS:
source pynomaly-graph-env/bin/activate
# On Windows:
pynomaly-graph-env\Scripts\activate
```

#### 2. Install PyTorch (Required)

PyTorch installation depends on your system configuration:

```bash
# CPU-only version
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu

# CUDA 11.8 (for GPU acceleration)
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

# CUDA 12.1 (latest)
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
```

For the most up-to-date PyTorch installation instructions, visit: <https://pytorch.org/get-started/locally/>

#### 3. Install PyTorch Geometric

```bash
pip install torch-geometric
```

#### 4. Install PyGOD

```bash
pip install pygod
```

#### 5. Install Pynomaly

```bash
pip install pynomaly
```

#### 6. Verify Installation

```python
# Test the installation
python -c "
from pynomaly.infrastructure.adapters.pygod_adapter import PyGODAdapter
algorithms = PyGODAdapter.get_supported_algorithms()
print(f'✅ Graph anomaly detection ready! Available algorithms: {len(algorithms)}')
for algo in algorithms[:3]:
    print(f'  - {algo}')
"
```

## Platform-Specific Instructions

### Linux (Ubuntu/Debian)

```bash
# Install system dependencies
sudo apt update
sudo apt install python3-dev python3-pip build-essential

# For GPU support (optional)
sudo apt install nvidia-cuda-toolkit

# Install Pynomaly with graph support
pip install "pynomaly[graph]"
```

### macOS

```bash
# Install Homebrew if not already installed
/bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"

# Install Python and dependencies
brew install python

# Install Pynomaly with graph support
pip install "pynomaly[graph]"
```

### Windows

```powershell
# Install Python from Microsoft Store or python.org
# Then install Pynomaly with graph support
pip install "pynomaly[graph]"

# For GPU support, install CUDA Toolkit from NVIDIA
# https://developer.nvidia.com/cuda-downloads
```

## GPU Support Configuration

### NVIDIA GPU Setup

#### 1. Install CUDA Toolkit

- Download from: <https://developer.nvidia.com/cuda-downloads>
- Follow platform-specific installation instructions
- Verify installation: `nvcc --version`

#### 2. Install cuDNN (Optional, for better performance)

- Download from: <https://developer.nvidia.com/cudnn>
- Follow installation instructions for your platform

#### 3. Verify GPU Support

```python
import torch
print(f"CUDA available: {torch.cuda.is_available()}")
print(f"CUDA version: {torch.version.cuda}")
print(f"GPU count: {torch.cuda.device_count()}")
if torch.cuda.is_available():
    print(f"GPU name: {torch.cuda.get_device_name(0)}")
```

### AMD GPU Support (ROCm)

```bash
# Install ROCm (Linux only)
# Follow instructions at: https://rocmdocs.amd.com/en/latest/Installation_Guide/Installation-Guide.html

# Install PyTorch with ROCm support
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/rocm5.4.2
```

## Dependency Versions

The following versions are tested and recommended:

```
pygod>=1.1.0
torch>=2.0.0
torch-geometric>=2.6.1
numpy>=1.21.0
scipy>=1.7.0
scikit-learn>=1.0.0
networkx>=2.6
```

### Version Compatibility Matrix

| Pynomaly | PyGOD | PyTorch | PyTorch Geometric | Python |
|----------|-------|---------|-------------------|--------|
| 0.1.0+   | 1.1.0+ | 2.0.0+  | 2.6.1+           | 3.11+  |

## Troubleshooting

### Common Installation Issues

#### ImportError: No module named 'torch'

```bash
# Install PyTorch first
pip install torch
```

#### ImportError: No module named 'torch_geometric'

```bash
# Install PyTorch Geometric
pip install torch-geometric
```

#### OSError: [WinError 126] The specified module could not be found

On Windows, this often indicates missing Visual C++ redistributables:

```powershell
# Download and install Microsoft Visual C++ Redistributable
# https://aka.ms/vs/17/release/vc_redist.x64.exe
```

#### CUDA out of memory errors

```python
# Reduce batch size and model complexity
adapter = PyGODAdapter(
    algorithm_name='DOMINANT',
    hidden_dim=32,  # Reduce from default
    batch_size=32   # Reduce batch size
)
```

#### Slow performance on CPU

```python
# Use statistical methods instead of deep learning
adapter = PyGODAdapter(
    algorithm_name='SCAN',  # Fast statistical method
    eps=0.5,
    mu=2
)
```

### Environment-Specific Issues

#### Conda Environments

```bash
# Create conda environment with graph dependencies
conda create -n pynomaly-graph python=3.11
conda activate pynomaly-graph

# Install dependencies via conda-forge
conda install -c conda-forge pytorch pytorch-geometric
pip install pygod pynomaly
```

#### Docker Setup

```dockerfile
FROM python:3.11-slim

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Install Python dependencies
RUN pip install "pynomaly[graph]"

# Verify installation
RUN python -c "from pynomaly.infrastructure.adapters.pygod_adapter import PyGODAdapter; print('Graph support ready!')"
```

#### Google Colab

```python
# Install in Google Colab
!pip install "pynomaly[graph]"

# Verify GPU availability
import torch
print(f"GPU available: {torch.cuda.is_available()}")
```

## Performance Optimization

### Memory Management

```python
# Configure memory-efficient settings
import torch
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

# For large graphs, use CPU with multiple cores
torch.set_num_threads(4)
```

### Parallel Processing

```python
# Enable parallel data loading
adapter = PyGODAdapter(
    algorithm_name='DOMINANT',
    num_workers=4,  # Parallel data loading
    pin_memory=True  # Faster GPU transfer
)
```

## Testing Your Installation

### Quick Test

```python
from pynomaly.infrastructure.adapters.pygod_adapter import PyGODAdapter
import pandas as pd
import numpy as np

# Create test data
data = pd.DataFrame({
    'source': [0, 1, 1, 2],
    'target': [1, 0, 2, 1],
    'feature_0': [1.0, 2.0, 3.0, 4.0],
    'feature_1': [0.5, 1.5, 2.5, 3.5]
})

# Test algorithm availability
algorithms = PyGODAdapter.get_supported_algorithms()
print(f"Available algorithms: {algorithms}")

# Test adapter creation
if algorithms:
    adapter = PyGODAdapter(algorithm_name=algorithms[0])
    print(f"✅ Successfully created {algorithms[0]} adapter")
else:
    print("❌ No algorithms available - check installation")
```

### Comprehensive Test

```python
import pytest
import subprocess
import sys

def test_dependencies():
    """Test all required dependencies."""
    dependencies = [
        'torch',
        'torch_geometric', 
        'pygod',
        'numpy',
        'scipy',
        'sklearn',
        'networkx'
    ]
    
    for dep in dependencies:
        try:
            __import__(dep)
            print(f"✅ {dep}")
        except ImportError:
            print(f"❌ {dep} - not installed")
            return False
    
    return True

def test_gpu_support():
    """Test GPU support if available."""
    import torch
    if torch.cuda.is_available():
        print(f"✅ GPU support available: {torch.cuda.get_device_name(0)}")
        return True
    else:
        print("ℹ️  CPU-only mode (GPU not available)")
        return True

# Run tests
if test_dependencies() and test_gpu_support():
    print("\\n🎉 All dependencies installed successfully!")
else:
    print("\\n❌ Some dependencies are missing. Please check installation.")
```

## Next Steps

After successful installation:

1. **Read the Graph Anomaly Detection Guide**: [graph-anomaly-detection.md](../guides/graph-anomaly-detection.md)
2. **Try the Tutorial**: [Graph Anomaly Detection Tutorial](../tutorials/graph-anomaly-detection-tutorial.md)
3. **Explore Examples**: Check the [examples](../../examples/) directory for graph anomaly detection examples
4. **Configure Production**: See [Production Deployment Guide](../deployment/production-guide.md) for production setup

## Support

If you encounter issues:

1. **Check the troubleshooting section** above
2. **Search existing issues**: [GitHub Issues](https://github.com/pynomaly/pynomaly/issues)
3. **Create a new issue** with:
   - Your operating system and version
   - Python version
   - Full error message
   - Installation method used
4. **Join the community**: [Discussions](https://github.com/pynomaly/pynomaly/discussions)
