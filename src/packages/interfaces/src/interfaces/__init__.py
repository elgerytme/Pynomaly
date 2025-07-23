"""
Interfaces package for domain contracts and cross-domain communication.

This package defines stable interfaces that enable different domains to
communicate while maintaining proper boundaries and loose coupling.
"""

__version__ = "0.1.0"

from .dto import (
    DetectionRequest,
    DetectionResult,
    ModelTrainingRequest,
    ModelTrainingResult,
    DataQualityRequest,
    DataQualityResult,
)
from .events import (
    DomainEvent,
    ModelTrainingCompleted,
    DataQualityCheckCompleted,
    AnomalyDetected,
)
from .patterns import (
    Repository,
    Service,
    EventBus,
    AntiCorruptionLayer,
)

__all__ = [
    # DTOs
    "DetectionRequest",
    "DetectionResult", 
    "ModelTrainingRequest",
    "ModelTrainingResult",
    "DataQualityRequest",
    "DataQualityResult",
    # Events
    "DomainEvent",
    "ModelTrainingCompleted", 
    "DataQualityCheckCompleted",
    "AnomalyDetected",
    # Patterns
    "Repository",
    "Service",
    "EventBus",
    "AntiCorruptionLayer",
]