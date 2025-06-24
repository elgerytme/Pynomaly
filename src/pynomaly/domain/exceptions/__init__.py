"""Domain exceptions."""

from .base import (
    PynamolyError,
    DomainError,
    ValidationError,
    NotFittedError,
    ConfigurationError,
    AuthenticationError,
    AuthorizationError,
    CacheError,
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

# Import InvalidValueError from base
from .base import InvalidValueError

# Aliases for backward compatibility  
InvalidDataError = DataValidationError
AdapterError = DomainError
AlgorithmNotFoundError = InvalidAlgorithmError
AutoMLError = DomainError
InvalidParameterError = ValidationError
ProcessingError = DomainError
EntityNotFoundError = DomainError

__all__ = [
    # Base exceptions
    "PynamolyError",
    "DomainError",
    "ValidationError",
    "NotFittedError",
    "ConfigurationError",
    "AuthenticationError",
    "AuthorizationError",
    "CacheError",
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
    # Aliases
    "InvalidDataError",
    "InvalidValueError",
    "AdapterError",
    "AlgorithmNotFoundError",
    "AutoMLError",
    "InvalidParameterError",
    "ProcessingError",
    "EntityNotFoundError",
]