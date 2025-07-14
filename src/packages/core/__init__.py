"""
Pynomaly Core Package

This package contains the core domain logic, business rules, and use cases
for the Pynomaly anomaly detection platform. It is dependency-free and
contains pure business logic that can be used across all interfaces.

Core Components:
- Domain entities and value objects
- Use cases for business operations
- DTOs for data transfer
- Shared utilities and types
"""

from .domain.entities import (
    Anomaly,
    Dataset,
    Detector,
    DetectionResult,
    Model,
    Pipeline,
    Experiment,
)

from .domain.value_objects import (
    AnomalyScore,
    ContaminationRate,
    PerformanceMetrics,
    ThresholdConfig,
)

from .use_cases import (
    detect_anomalies,
    train_detector,
    evaluate_model,
    explain_anomaly,
)

__version__ = "0.1.1"
__all__ = [
    # Entities
    "Anomaly",
    "Dataset", 
    "Detector",
    "DetectionResult",
    "Model",
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
    "evaluate_model", 
    "explain_anomaly",
]