"""Anomaly Detection Package.

A comprehensive anomaly detection package providing ML-based detection algorithms,
ensemble methods, streaming capabilities, and explainable AI features.
"""

__version__ = "0.3.0"
__author__ = "Anomaly Detection Team"
__email__ = "team@anomaly_detection.io"

# Domain layer exports
from .domain.services.detection_service import DetectionService
from .domain.services.ensemble_service import EnsembleService  
from .domain.services.processing_service import ProcessingService

# Infrastructure layer exports
from .infrastructure.adapters.algorithms.adapters.sklearn_adapter import SklearnAdapter
from .infrastructure.adapters.algorithms.adapters.pyod_adapter import PyODAdapter
from .infrastructure.adapters.algorithms.adapters.deeplearning_adapter import DeepLearningAdapter

# Application layer exports
from .application.services.explanation.analyzers import ExplanationAnalyzers
from .application.services.performance.optimization import PerformanceOptimizer
from .application.services.mlops import (
    UnifiedModelRegistry,
    ExperimentTrackingIntegration,
    initialize_unified_model_registry,
    initialize_experiment_tracking_integration,
)

__all__ = [
    "__version__",
    "__author__", 
    "__email__",
    # Domain services
    "DetectionService",
    "EnsembleService",
    "ProcessingService",
    # Infrastructure adapters
    "SklearnAdapter",
    "PyODAdapter", 
    "DeepLearningAdapter",
    # Application services
    "ExplanationAnalyzers",
    "PerformanceOptimizer",
    # MLOps integration
    "UnifiedModelRegistry",
    "ExperimentTrackingIntegration",
    "initialize_unified_model_registry",
    "initialize_experiment_tracking_integration",
]