"""Domain exceptions."""

# Import InvalidValueError from base
from .base import (
    AuthenticationError,
    AuthorizationError,
    CacheError,
    ConfigurationError,
    DomainError,
    InfrastructureError,
    InvalidConfigurationError,
    InvalidValueError,
    NotFittedError,
    PynamolyError,
    ValidationError,
)
from .dataset_exceptions import (
    DataLoadError,
    DatasetError,
    DataTypeError,
    DataValidationError,
    FeatureMismatchError,
    InsufficientDataError,
)
from .detector_exceptions import (
    DetectorConfigurationError,
    DetectorError,
    DetectorNotFittedError,
    FittingError,
    InvalidAlgorithmError,
)
from .entity_exceptions import (
    AlertNotFoundError,
    EntityNotFoundError,
    ExperimentNotFoundError,
    InvalidAlertStateError,
    InvalidEntityStateError,
    InvalidExperimentStateError,
    InvalidModelStateError,
    InvalidPipelineStateError,
    ModelNotFoundError,
    PipelineNotFoundError,
)
from .result_exceptions import (
    InconsistentResultError,
    ResultError,
    ScoreCalculationError,
    ThresholdError,
)
from .storage_exceptions import (
    SerializationError,
    StorageConfigurationError,
    StorageConnectionError,
    StorageError,
    StorageNotFoundError,
    StoragePermissionError,
    StorageQuotaError,
    StorageTimeoutError,
)

# Aliases for backward compatibility
InvalidDataError = DataValidationError
AdapterError = DomainError
AlgorithmNotFoundError = InvalidAlgorithmError
AutoMLError = DomainError
ExplainabilityError = DomainError
InvalidParameterError = ValidationError
ProcessingError = DomainError
EntityNotFoundError = DomainError
PynomaliError = PynamolyError  # Common typo alias

__all__ = [
    # Base exceptions
    "PynamolyError",
    "DomainError",
    "ValidationError",
    "NotFittedError",
    "ConfigurationError",
    "InvalidConfigurationError",
    "AuthenticationError",
    "AuthorizationError",
    "CacheError",
    "InfrastructureError",
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
    "DataLoadError",
    # Result exceptions
    "ResultError",
    "ScoreCalculationError",
    "ThresholdError",
    "InconsistentResultError",
    # Entity exceptions
    "EntityNotFoundError",
    "InvalidEntityStateError",
    "ModelNotFoundError",
    "InvalidModelStateError",
    "ExperimentNotFoundError",
    "InvalidExperimentStateError",
    "PipelineNotFoundError",
    "InvalidPipelineStateError",
    "AlertNotFoundError",
    "InvalidAlertStateError",
    # Storage exceptions
    "StorageError",
    "StorageConnectionError",
    "StorageNotFoundError",
    "StoragePermissionError",
    "StorageTimeoutError",
    "StorageConfigurationError",
    "StorageQuotaError",
    "SerializationError",
    # Aliases
    "InvalidDataError",
    "InvalidValueError",
    "AdapterError",
    "AlgorithmNotFoundError",
    "AutoMLError",
    "ExplainabilityError",
    "InvalidParameterError",
    "ProcessingError",
    "PynomaliError",
]
