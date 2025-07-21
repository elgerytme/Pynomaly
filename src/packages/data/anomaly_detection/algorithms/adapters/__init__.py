"""Algorithm adapters for various machine learning frameworks."""

from .sklearn_adapter import SklearnAdapter
from .pyod_adapter import PyODAdapter, PyODEnsemble

# Try to import deep learning adapter
try:
    from .deeplearning_adapter import DeepLearningAdapter
except ImportError:
    DeepLearningAdapter = None  # type: ignore

__all__ = [
    "SklearnAdapter",
    "PyODAdapter", 
    "PyODEnsemble",
    "DeepLearningAdapter",
]