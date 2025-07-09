"""Domain layer containing core business logic."""

from pynomaly.domain.abstractions import (
    BaseEntity,
    BaseRepository,
    BaseService,
    BaseValueObject,
    DomainEvent,
    DomainEventHandler,
    Specification,
)
from pynomaly.domain.entities import Anomaly, Dataset, DetectionResult, Detector
from pynomaly.domain.value_objects import (
    AnomalyScore,
    ConfidenceInterval,
    ContaminationRate,
)

__all__ = [
    "Anomaly",
    "Dataset",
    "Detector",
    "DetectionResult",
    "AnomalyScore",
    "ConfidenceInterval",
    "ContaminationRate",
    # Abstractions
    "BaseEntity",
    "BaseRepository",
    "BaseService",
    "BaseValueObject",
    "DomainEvent",
    "DomainEventHandler",
    "Specification",
]
