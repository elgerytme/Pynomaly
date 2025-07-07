"""TensorFlow adapter stub when TensorFlow is not available."""

from __future__ import annotations
from typing import Any


class TensorFlowAdapter:
    """Stub TensorFlowAdapter when TensorFlow is not available."""

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        raise ImportError(
            "TensorFlow is required for TensorFlowAdapter. "
            "Install with: pip install tensorflow"
        )


class AutoEncoder:
    """Stub AutoEncoder when TensorFlow is not available."""

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        raise ImportError(
            "TensorFlow is required for AutoEncoder. "
            "Install with: pip install tensorflow"
        )
