"""PyTorch adapter stub when PyTorch is not available."""

from __future__ import annotations


class PyTorchAdapter:
    """Stub PyTorchAdapter when PyTorch is not available."""
    
    def __init__(self, *args, **kwargs):
        raise ImportError(
            "PyTorch is required for PyTorchAdapter. "
            "Install with: pip install torch torchvision"
        )


class AutoEncoder:
    """Stub AutoEncoder when PyTorch is not available."""
    
    def __init__(self, *args, **kwargs):
        raise ImportError(
            "PyTorch is required for AutoEncoder. "
            "Install with: pip install torch torchvision"
        )


class VAE:
    """Stub VAE when PyTorch is not available."""
    
    def __init__(self, *args, **kwargs):
        raise ImportError(
            "PyTorch is required for VAE. "
            "Install with: pip install torch torchvision"
        )


class LSTMAutoEncoder:
    """Stub LSTMAutoEncoder when PyTorch is not available."""
    
    def __init__(self, *args, **kwargs):
        raise ImportError(
            "PyTorch is required for LSTMAutoEncoder. "
            "Install with: pip install torch torchvision"
        )