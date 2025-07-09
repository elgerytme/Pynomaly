"""Domain layer containing core business logic."""

from pynomaly.domain.entities import Anomaly, Dataset, DetectionResult, Detector
from pynomaly.domain.value_objects import (
    AnomalyScore,
    ConfidenceInterval,
    ContaminationRate,
)
from .validation import (
    ValidationSeverity,
    ValidationError,
    ValidationResult,
    DomainValidator,
    ValidationStrategy,
    validate_anomaly,
    validate_dataset,
    set_validation_strategy,
)

__all__ = [
    "Anomaly",
    "Dataset",
    "Detector",
    "DetectionResult",
    "AnomalyScore",
    "ConfidenceInterval",
    "ContaminationRate",
    "ValidationSeverity",
    "ValidationError",
    "ValidationResult",
    "DomainValidator",
    "ValidationStrategy",
    "validate_anomaly",
    "validate_dataset",
    "set_validation_strategy",
]
