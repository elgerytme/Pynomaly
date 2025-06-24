"""Infrastructure adapters for anomaly detection algorithms and business intelligence platforms."""

from .pyod_adapter import PyODAdapter
from .sklearn_adapter import SklearnAdapter

__all__ = [
    "PyODAdapter",
    "SklearnAdapter",
]

# Optional ML adapters - only import if dependencies are available
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

# Business Intelligence and Export adapters
try:
    from .excel_adapter import ExcelAdapter
    __all__.append("ExcelAdapter")
except ImportError:
    pass

try:
    from .powerbi_adapter import PowerBIAdapter
    __all__.append("PowerBIAdapter")
except ImportError:
    pass

try:
    from .gsheets_adapter import GoogleSheetsAdapter
    __all__.append("GoogleSheetsAdapter")
except ImportError:
    pass

try:
    from .smartsheet_adapter import SmartsheetAdapter
    __all__.append("SmartsheetAdapter")
except ImportError:
    pass