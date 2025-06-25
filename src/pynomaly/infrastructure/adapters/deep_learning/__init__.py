"""Deep learning adapters for advanced anomaly detection frameworks."""

from __future__ import annotations

# Export deep learning adapters as they are created
try:
    from .pytorch_adapter import PyTorchAdapter
except ImportError:
    PyTorchAdapter = None

try:
    from .tensorflow_adapter import TensorFlowAdapter
except ImportError:
    TensorFlowAdapter = None

try:
    from .jax_adapter import JAXAdapter
except ImportError:
    JAXAdapter = None

__all__ = ["PyTorchAdapter", "TensorFlowAdapter", "JAXAdapter"]
