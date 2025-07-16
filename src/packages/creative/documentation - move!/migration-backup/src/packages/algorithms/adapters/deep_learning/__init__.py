"""Deep learning adapters for advanced anomaly detection frameworks."""

from __future__ import annotations

# Export deep learning adapters with fallbacks
try:
    from .pytorch_adapter import PyTorchAdapter
except (ImportError, SyntaxError, IndentationError):
    # Fallback to stub when PyTorch is not available or there are syntax issues
    from .pytorch_stub import PyTorchAdapter

try:
    from .tensorflow_adapter import TensorFlowAdapter
except (ImportError, SyntaxError, AttributeError):
    # Fallback to stub when TensorFlow is not available or there are issues
    from .tensorflow_stub import TensorFlowAdapter

try:
    from .jax_adapter import JAXAdapter
except (ImportError, SyntaxError, AttributeError):
    # Fallback to stub when JAX is not available or there are issues
    from .jax_stub import JAXAdapter

__all__ = ["PyTorchAdapter", "TensorFlowAdapter", "JAXAdapter"]
