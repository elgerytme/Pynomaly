"""JAX adapter stub when JAX is not available."""

from __future__ import annotations


class JAXAdapter:
    """Stub JAXAdapter when JAX is not available."""
    
    def __init__(self, *args, **kwargs):
        raise ImportError(
            "JAX is required for JAXAdapter. "
            "Install with: pip install jax jaxlib"
        )


class AutoEncoder:
    """Stub AutoEncoder when JAX is not available."""
    
    def __init__(self, *args, **kwargs):
        raise ImportError(
            "JAX is required for AutoEncoder. "
            "Install with: pip install jax jaxlib"
        )