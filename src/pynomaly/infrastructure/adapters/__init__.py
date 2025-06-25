"""Infrastructure adapters for anomaly detection algorithms and business intelligence platforms."""

from .pyod_adapter import PyODAdapter
from .sklearn_adapter import SklearnAdapter

# Optimized adapters for Phase 2
try:
    from .optimized_adapter import OptimizedAdapter, OptimizedEnsembleAdapter
    OPTIMIZED_ADAPTERS_AVAILABLE = True
except ImportError:
    OptimizedAdapter = None
    OptimizedEnsembleAdapter = None
    OPTIMIZED_ADAPTERS_AVAILABLE = False

__all__ = [
    "PyODAdapter",
    "SklearnAdapter",
]

if OPTIMIZED_ADAPTERS_AVAILABLE:
    __all__.extend(["OptimizedAdapter", "OptimizedEnsembleAdapter"])

# Optional ML adapters - only import if dependencies are available
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

try:
    from .tensorflow_adapter import TensorFlowAdapter
    __all__.append("TensorFlowAdapter")
except ImportError:
    pass

try:
    from .jax_adapter import JAXAdapter
    __all__.append("JAXAdapter")
except ImportError:
    pass

# Basic export adapters (BI integrations removed for simplification)
try:
    from .excel_adapter import ExcelAdapter
    __all__.append("ExcelAdapter")
except ImportError:
    pass

# Time series and ensemble adapters
try:
    from .time_series_adapter import TimeSeriesAdapter
    __all__.append("TimeSeriesAdapter")
except ImportError:
    pass

try:
    from .ensemble_adapter import EnsembleAdapter
    __all__.append("EnsembleAdapter")
except ImportError:
    pass