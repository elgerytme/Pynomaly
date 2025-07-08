"""Factory package for test data generation using factory-boy.

This package provides factory classes for creating test instances of domain entities,
DTOs, and other data structures used throughout the application.
"""

from .domain_factories import (
    AnomalyFactory,
    DatasetFactory,
    DetectionResultFactory,
    DetectorFactory,
    TrainingResultFactory,
    UserFactory,
)
from .dto_factories import (
    DetectionRequestDTOFactory,
    DetectionResultDTOFactory,
    ExplanationRequestDTOFactory,
    TrainingRequestDTOFactory,
    TrainingResultDTOFactory,
)
from .value_object_factories import (
    AnomalyScoreFactory,
    ContaminationRateFactory,
    PerformanceMetricsFactory,
    ThresholdConfigFactory,
)

__all__ = [
    # Domain entity factories
    "AnomalyFactory",
    "DatasetFactory",
    "DetectionResultFactory",
    "DetectorFactory",
    "TrainingResultFactory",
    "UserFactory",
    # DTO factories
    "DetectionRequestDTOFactory",
    "DetectionResultDTOFactory",
    "ExplanationRequestDTOFactory",
    "TrainingRequestDTOFactory",
    "TrainingResultDTOFactory",
    # Value object factories
    "AnomalyScoreFactory",
    "ContaminationRateFactory",
    "PerformanceMetricsFactory",
    "ThresholdConfigFactory",
]
