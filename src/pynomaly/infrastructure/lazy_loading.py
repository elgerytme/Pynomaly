"""Lazy loading utilities for heavy ML backends.

This module provides lazy loading functionality to avoid importing heavy
dependencies until they are actually needed, improving startup time and
reducing memory usage.
"""

import importlib
import warnings
from functools import lru_cache
from typing import Any, Optional, Type, Union


class LazyImportError(ImportError):
    """Raised when a lazy import fails."""
    pass


class LazyModule:
    """Lazy module loader that imports modules only when accessed."""
    
    def __init__(self, module_name: str, package: Optional[str] = None):
        self._module_name = module_name
        self._package = package
        self._module = None
        self._import_error = None
        
    def __getattr__(self, name: str) -> Any:
        """Get attribute from the lazily imported module."""
        if self._module is None:
            self._import_module()
        
        if self._import_error:
            raise self._import_error
            
        return getattr(self._module, name)
    
    def _import_module(self):
        """Import the module and cache it."""
        try:
            self._module = importlib.import_module(self._module_name, self._package)
        except ImportError as e:
            self._import_error = LazyImportError(
                f"Failed to import {self._module_name}: {e}. "
                f"Install the required package with: pip install {self._get_install_command()}"
            )
    
    def _get_install_command(self) -> str:
        """Get the appropriate install command for the module."""
        install_commands = {
            'torch': 'torch>=2.5.1',
            'tensorflow': 'tensorflow>=2.18.0,<2.20.0',
            'jax': 'jax>=0.4.37 jaxlib>=0.4.37',
            'sklearn': 'scikit-learn>=1.6.0',
            'lightgbm': 'lightgbm>=4.0.0',
            'xgboost': 'xgboost>=2.0.0',
            'catboost': 'catboost>=1.2.0',
            'shap': 'shap>=0.46.0',
            'lime': 'lime>=0.2.0.1',
            'optuna': 'optuna>=4.1.0',
            'hyperopt': 'hyperopt>=0.2.7',
        }
        
        for key, command in install_commands.items():
            if key in self._module_name:
                return command
                
        return self._module_name


@lru_cache(maxsize=None)
def get_lazy_module(module_name: str, package: Optional[str] = None) -> LazyModule:
    """Get a cached lazy module instance."""
    return LazyModule(module_name, package)


def try_import(module_name: str, package: Optional[str] = None) -> Optional[Any]:
    """Try to import a module, returning None if it fails."""
    try:
        return importlib.import_module(module_name, package)
    except ImportError:
        return None


def require_module(module_name: str, package: Optional[str] = None) -> Any:
    """Require a module, raising LazyImportError if it fails."""
    module = try_import(module_name, package)
    if module is None:
        raise LazyImportError(
            f"Required module '{module_name}' is not available. "
            f"Install it with: pip install {module_name}"
        )
    return module


def warn_if_missing(module_name: str, feature_name: str) -> bool:
    """Warn if a module is missing and return availability status."""
    if try_import(module_name) is None:
        warnings.warn(
            f"{feature_name} requires {module_name} which is not installed. "
            f"Install it with: pip install {module_name}",
            UserWarning,
            stacklevel=2
        )
        return False
    return True


class LazyBackend:
    """Lazy backend loader for ML frameworks."""
    
    def __init__(self):
        self._backends = {}
        
    def register_backend(self, name: str, module_name: str, package: Optional[str] = None):
        """Register a backend for lazy loading."""
        self._backends[name] = {
            'module_name': module_name,
            'package': package,
            'lazy_module': None
        }
    
    def get_backend(self, name: str) -> LazyModule:
        """Get a lazy backend module."""
        if name not in self._backends:
            raise ValueError(f"Unknown backend: {name}")
            
        backend_info = self._backends[name]
        if backend_info['lazy_module'] is None:
            backend_info['lazy_module'] = get_lazy_module(
                backend_info['module_name'], 
                backend_info['package']
            )
            
        return backend_info['lazy_module']
    
    def is_available(self, name: str) -> bool:
        """Check if a backend is available."""
        if name not in self._backends:
            return False
            
        backend_info = self._backends[name]
        return try_import(backend_info['module_name'], backend_info['package']) is not None
    
    def list_available(self) -> list[str]:
        """List all available backends."""
        return [name for name in self._backends.keys() if self.is_available(name)]


# Global backend registry
_backend_registry = LazyBackend()

# Register common ML backends
_backend_registry.register_backend('torch', 'torch')
_backend_registry.register_backend('tensorflow', 'tensorflow')
_backend_registry.register_backend('jax', 'jax')
_backend_registry.register_backend('sklearn', 'sklearn')
_backend_registry.register_backend('lightgbm', 'lightgbm')
_backend_registry.register_backend('xgboost', 'xgboost')
_backend_registry.register_backend('catboost', 'catboost')
_backend_registry.register_backend('shap', 'shap')
_backend_registry.register_backend('lime', 'lime')
_backend_registry.register_backend('optuna', 'optuna')
_backend_registry.register_backend('hyperopt', 'hyperopt')


def get_backend(name: str) -> LazyModule:
    """Get a lazy backend module."""
    return _backend_registry.get_backend(name)


def is_backend_available(name: str) -> bool:
    """Check if a backend is available."""
    return _backend_registry.is_available(name)


def list_available_backends() -> list[str]:
    """List all available backends."""
    return _backend_registry.list_available()


# Lazy imports for common ML frameworks
torch = get_lazy_module('torch')
tensorflow = get_lazy_module('tensorflow')
jax = get_lazy_module('jax')
sklearn = get_lazy_module('sklearn')
lightgbm = get_lazy_module('lightgbm')
xgboost = get_lazy_module('xgboost')
catboost = get_lazy_module('catboost')
shap = get_lazy_module('shap')
lime = get_lazy_module('lime')
optuna = get_lazy_module('optuna')
hyperopt = get_lazy_module('hyperopt')


def check_ml_dependencies():
    """Check and report status of ML dependencies."""
    backends = [
        ('torch', 'PyTorch'),
        ('tensorflow', 'TensorFlow'),
        ('jax', 'JAX'),
        ('sklearn', 'scikit-learn'),
        ('lightgbm', 'LightGBM'),
        ('xgboost', 'XGBoost'),
        ('catboost', 'CatBoost'),
        ('shap', 'SHAP'),
        ('lime', 'LIME'),
        ('optuna', 'Optuna'),
        ('hyperopt', 'Hyperopt'),
    ]
    
    available = []
    unavailable = []
    
    for backend_name, display_name in backends:
        if is_backend_available(backend_name):
            available.append(display_name)
        else:
            unavailable.append(display_name)
    
    print("Available ML backends:")
    for backend in available:
        print(f"  ✓ {backend}")
        
    if unavailable:
        print("\nUnavailable ML backends:")
        for backend in unavailable:
            print(f"  ✗ {backend}")
    
    return {'available': available, 'unavailable': unavailable}
