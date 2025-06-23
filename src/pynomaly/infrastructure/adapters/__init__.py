"""Infrastructure adapters for anomaly detection algorithms."""

from .pyod_adapter import PyODAdapter
from .sklearn_adapter import SklearnAdapter

__all__ = [
    "PyODAdapter",
    "SklearnAdapter",
]

# Optional adapters - only import if dependencies are available
try:
    from .tods_adapter import TODSAdapter
    __all__.append("TODSAdapter")
except ImportError:
    pass

try:
    from .pygod_adapter import PyGODAdapter
    __all__.append("PyGODAdapter")
except ImportError:
    pass

try:
    from .pytorch_adapter import PyTorchAdapter
    __all__.append("PyTorchAdapter")
except ImportError:
    pass