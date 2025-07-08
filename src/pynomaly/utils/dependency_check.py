import importlib.metadata
import warnings

# Check if optional dependencies are available and warn if not
optional_dependencies = {
    "torch": "PyTorch",
    "tensorflow": "TensorFlow",
    "jax": "JAX",
    "optuna": "Optuna",
    "hyperopt": "HyperOpt",
    "shap": "SHAP",
    "lime": "LIME",
    "ray": "Ray Tune",
}

def check_dependencies():
    for pkg, name in optional_dependencies.items():
        try:
            importlib.metadata.version(pkg)
        except importlib.metadata.PackageNotFoundError:
            warnings.warn(
                f"The '{name}' package is not installed and is required for specific functionalities.",
                ImportWarning,
            )


check_dependencies()
