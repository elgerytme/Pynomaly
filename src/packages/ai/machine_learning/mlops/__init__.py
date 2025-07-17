"""MLOps package for model lifecycle management."""

__version__ = "0.1.0"

try:
    from .mlops import *  # noqa: F403
except ImportError:
    # Graceful degradation if MLOps dependencies not available
    pass

__all__ = ["__version__"]
