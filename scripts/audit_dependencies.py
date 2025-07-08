#!/usr/bin/env python3
"""Audit current optional dependencies and their impact."""

import importlib.metadata
import sys
from pathlib import Path

def audit_dependencies():
    """Audit current heavyweight dependencies."""
    packages = {
        'torch': 'PyTorch',
        'tensorflow': 'TensorFlow', 
        'jax': 'JAX',
        'optuna': 'Optuna',
        'hyperopt': 'HyperOpt',
        'auto-sklearn2': 'auto-sklearn2',
        'shap': 'SHAP',
        'lime': 'LIME',
        'ray': 'Ray Tune'
    }
    
    print('🔍 Current Optional Dependencies Analysis')
    print('=' * 60)
    
    installed = []
    missing = []
    
    for pkg, display_name in packages.items():
        try:
            dist = importlib.metadata.distribution(pkg)
            print(f'✅ {display_name:15}: {dist.version:10} (installed)')
            installed.append(pkg)
        except importlib.metadata.PackageNotFoundError:
            print(f'❌ {display_name:15}: NOT FOUND   (not installed)')
            missing.append(pkg)
    
    print()
    print('📊 Heavyweight Dependencies Impact:')
    print('=' * 60)
    impact_data = {
        'torch': '~2GB with CUDA support, ~500MB CPU-only',
        'tensorflow': '~1.5GB with GPU support, ~400MB CPU-only',
        'jax': '~500MB with CUDA libs, ~200MB CPU-only',
        'optuna': '~50MB + dependencies (scikit-learn, scipy)',
        'hyperopt': '~30MB + dependencies (numpy, scipy)',
        'auto-sklearn2': '~200MB + scikit-learn dependencies',
        'shap': '~100MB + dependencies (pandas, scipy, scikit-learn)',
        'lime': '~20MB + dependencies (scikit-learn, scikit-image)',
        'ray': '~300MB + dependencies (distributed computing)'
    }
    
    for pkg, impact in impact_data.items():
        status = '✅' if pkg in installed else '❌'
        print(f'{status} {pkg:15}: {impact}')
    
    print()
    print('💡 Recommended Extras Structure:')
    print('=' * 60)
    print('• pynomaly[automl]    - Optuna, HyperOpt, Ray Tune')
    print('• pynomaly[deep]      - PyTorch, TensorFlow, JAX')
    print('• pynomaly[explain]   - SHAP, LIME')
    print('• pynomaly[deep-cpu]  - CPU-only versions of deep learning frameworks')
    print('• pynomaly[deep-gpu]  - GPU-enabled versions with CUDA support')
    
    return installed, missing

if __name__ == '__main__':
    audit_dependencies()
