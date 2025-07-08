"""Lightweight stubs for optional dependencies with actionable error messages."""

import importlib.metadata
import sys
from typing import Dict, List, Optional

from pynomaly.infrastructure.config.feature_flags import FeatureNotAvailableError


class DependencyNotFoundError(ImportError):
    """Raised when an optional dependency is required but not installed."""
    
    def __init__(self, package: str, extras: List[str], features: List[str]):
        self.package = package
        self.extras = extras
        self.features = features
        
        install_commands = [f"pip install pynomaly[{extra}]" for extra in extras]
        install_cmd = " or ".join(install_commands)
        
        feature_list = ", ".join(features)
        
        message = (
            f"The '{package}' package is required for: {feature_list}\n\n"
            f"Install it with: {install_cmd}\n\n"
            f"Or install specific dependencies:\n"
            f"  pip install {package}\n\n"
            f"For more information, see: "
            f"https://pynomaly.readthedocs.io/en/latest/installation/#optional-dependencies"
        )
        super().__init__(message)


# Mapping of packages to their extras and features
DEPENDENCY_MAP: Dict[str, Dict[str, List[str]]] = {
    "torch": {
        "extras": ["deep", "deep-cpu", "deep-gpu", "ml-all"],
        "features": ["PyTorch deep learning", "neural network models", "GPU acceleration"]
    },
    "tensorflow": {
        "extras": ["deep", "deep-cpu", "deep-gpu", "ml-all"],
        "features": ["TensorFlow neural networks", "Keras models", "distributed training"]
    },
    "jax": {
        "extras": ["deep", "deep-cpu", "deep-gpu", "ml-all"],
        "features": ["JAX high-performance computing", "automatic differentiation", "XLA compilation"]
    },
    "optuna": {
        "extras": ["automl", "automl-advanced", "ml-all"],
        "features": ["hyperparameter optimization", "automated model tuning", "Bayesian optimization"]
    },
    "hyperopt": {
        "extras": ["automl", "automl-advanced", "ml-all"],
        "features": ["hyperparameter search", "random search", "TPE optimization"]
    },
    "ray": {
        "extras": ["automl-advanced"],
        "features": ["distributed hyperparameter tuning", "Ray Tune", "parallel optimization"]
    },
    "auto-sklearn2": {
        "extras": ["automl-advanced", "ml-all"],
        "features": ["automated machine learning", "meta-learning", "ensemble optimization"]
    },
    "shap": {
        "extras": ["explainability", "ml-all"],
        "features": ["SHAP explanations", "feature importance", "model interpretability"]
    },
    "lime": {
        "extras": ["explainability", "ml-all"],
        "features": ["LIME explanations", "local interpretability", "black-box explanation"]
    },
    "pygod": {
        "extras": ["graph"],
        "features": ["graph anomaly detection", "network analysis", "graph neural networks"]
    },
    "torch-geometric": {
        "extras": ["graph"],
        "features": ["graph neural networks", "geometric deep learning", "graph convolutions"]
    }
}


def check_dependency(package: str) -> bool:
    """Check if a package is installed."""
    try:
        importlib.metadata.version(package)
        return True
    except importlib.metadata.PackageNotFoundError:
        return False


def require_dependency(package: str) -> None:
    """Require a dependency and raise informative error if missing."""
    if not check_dependency(package):
        if package in DEPENDENCY_MAP:
            extras = DEPENDENCY_MAP[package]["extras"]
            features = DEPENDENCY_MAP[package]["features"]
            install_commands = [f"pip install pynomaly[{extra}]" for extra in extras]
            install_hint = " or ".join(install_commands)
            feature_list = ", ".join(features)
            
            raise FeatureNotAvailableError(
                package,
                reason=f"Required for: {feature_list}",
                install_hint=install_hint
            )
        else:
            raise FeatureNotAvailableError(
                package,
                reason="Package is not installed",
                install_hint=f"Install with: pip install {package}"
            )


def create_optional_import_stub(package: str, feature_name: str):
    """Create a stub that raises an error when optional functionality is accessed."""
    
    class OptionalImportStub:
        """Stub for optional imports that provides actionable error messages."""
        
        def __init__(self, package_name: str, feature: str):
            self.package_name = package_name
            self.feature = feature
        
        def __getattr__(self, name: str):
            require_dependency(self.package_name)
        
        def __call__(self, *args, **kwargs):
            require_dependency(self.package_name)
        
        def __bool__(self):
            return False
        
        def __repr__(self):
            return f"<OptionalImportStub for {self.package_name} ({self.feature})>"
    
    return OptionalImportStub(package, feature_name)


def safe_import(package: str, feature_name: Optional[str] = None):
    """Safely import a package, returning a stub if not available."""
    try:
        return importlib.import_module(package)
    except ImportError:
        return create_optional_import_stub(package, feature_name or package)


def get_missing_dependencies() -> Dict[str, List[str]]:
    """Get a summary of missing optional dependencies."""
    missing = {}
    for package, info in DEPENDENCY_MAP.items():
        if not check_dependency(package):
            missing[package] = {
                "features": info["features"],
                "extras": info["extras"]
            }
    return missing


def print_dependency_status():
    """Print the status of all optional dependencies."""
    print("🔍 Optional Dependencies Status")
    print("=" * 50)
    
    for package, info in DEPENDENCY_MAP.items():
        status = "✅ Installed" if check_dependency(package) else "❌ Missing"
        features = ", ".join(info["features"][:2])  # Show first 2 features
        if len(info["features"]) > 2:
            features += "..."
        
        print(f"{package:15} {status:12} ({features})")
    
    missing = get_missing_dependencies()
    if missing:
        print("\n💡 Installation Commands:")
        print("-" * 30)
        
        # Group by extras
        extras_to_packages = {}
        for package, info in missing.items():
            for extra in info["extras"]:
                if extra not in extras_to_packages:
                    extras_to_packages[extra] = []
                extras_to_packages[extra].append(package)
        
        for extra, packages in extras_to_packages.items():
            pkg_list = ", ".join(packages)
            print(f"pip install pynomaly[{extra}]  # {pkg_list}")


if __name__ == "__main__":
    print_dependency_status()
