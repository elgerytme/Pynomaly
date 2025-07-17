"""
Software Core Package

This package contains the core domain logic, business rules, and use cases
for the Software anomaly processing platform. It is dependency-free and
contains pure business logic that can be used across all interfaces.

Core Components:
- Domain entities and value objects
- Use cases for business operations
- DTOs for data transfer
- Shared utilities and types
"""

from .domain.entities import (
    Anomaly,
    DataCollection,
    DetectionResult,
    Detector,
    Experiment,
    Processor,
    Pipeline,
)
from .domain.value_objects import (
    AnomalyScore,
    ContaminationRate,
    PerformanceMetrics,
    ThresholdConfig,
)
from .use_cases import detect_anomalies, evaluate_model, explain_anomaly, train_detector

__version__ = "0.1.1"
__all__ = [
    # Entities
    "Anomaly",
    "DataCollection",
    "Detector",
    "DetectionResult",
    "Processor",
    "Pipeline",
    "Experiment",
    # Value Objects
    "AnomalyScore",
    "ContaminationRate",
    "PerformanceMetrics",
    "ThresholdConfig",
    # Use Cases
    "detect_anomalies",
    "train_detector",
    "evaluate_processor",
    "explain_anomaly",
]
