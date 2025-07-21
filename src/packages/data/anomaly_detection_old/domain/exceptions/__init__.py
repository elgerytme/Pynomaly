"""Domain exceptions for anomaly detection."""

from .detector_exceptions import DetectorNotFittedError, DetectorConfigurationError
from .result_exceptions import ResultValidationError, ResultProcessingError

__all__ = [
    "DetectorNotFittedError",
    "DetectorConfigurationError", 
    "ResultValidationError",
    "ResultProcessingError",
]