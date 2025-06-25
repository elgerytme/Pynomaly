"""Deep learning adapters for advanced anomaly detection frameworks."""

from __future__ import annotations

# Export deep learning adapters with fallbacks
try:
    from .pytorch_adapter import PyTorchAdapter
except ImportError:
    # Fallback stub when PyTorch is not available
    class PyTorchAdapter:
        """Dummy PyTorchAdapter when PyTorch is not available."""
        
        def __init__(self, *args, **kwargs):
            raise ImportError("PyTorch is required for PyTorchAdapter. Install with: pip install torch")

try:
    from .tensorflow_adapter import TensorFlowAdapter
except ImportError:
    # Fallback stub when TensorFlow is not available
    class TensorFlowAdapter:
        """Dummy TensorFlowAdapter when TensorFlow is not available."""
        
        def __init__(self, *args, **kwargs):
            raise ImportError("TensorFlow is required for TensorFlowAdapter. Install with: pip install tensorflow")

try:
    from .jax_adapter import JAXAdapter
except ImportError:
    # Fallback stub when JAX is not available
    class JAXAdapter:
        """Dummy JAXAdapter when JAX is not available."""
        
        def __init__(self, *args, **kwargs):
            raise ImportError("JAX is required for JAXAdapter. Install with: pip install jax")

__all__ = ["PyTorchAdapter", "TensorFlowAdapter", "JAXAdapter"]
