"""Domain exceptions."""

from .base import (
    PynamolyError,
    DomainError,
    ValidationError,
    NotFittedError,
    ConfigurationError,
)
from .detector_exceptions import (
    DetectorError,
    DetectorNotFittedError,
    DetectorConfigurationError,
    InvalidAlgorithmError,
    FittingError,
)
from .dataset_exceptions import (
    DatasetError,
    DataValidationError,
    InsufficientDataError,
    DataTypeError,
    FeatureMismatchError,
)
from .result_exceptions import (
    ResultError,
    ScoreCalculationError,
    ThresholdError,
    InconsistentResultError,
)

__all__ = [
    # Base exceptions
    "PynamolyError",
    "DomainError",
    "ValidationError",
    "NotFittedError",
    "ConfigurationError",
    # Detector exceptions
    "DetectorError",
    "DetectorNotFittedError",
    "DetectorConfigurationError",
    "InvalidAlgorithmError",
    "FittingError",
    # Dataset exceptions
    "DatasetError",
    "DataValidationError",
    "InsufficientDataError",
    "DataTypeError",
    "FeatureMismatchError",
    # Result exceptions
    "ResultError",
    "ScoreCalculationError",
    "ThresholdError",
    "InconsistentResultError",
]