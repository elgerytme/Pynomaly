"""Anomaly Detection Package - Domain-Driven Architecture.

A comprehensive anomaly detection package providing ML-based detection algorithms,
ensemble methods, streaming capabilities, and explainable AI features.

Built with Domain-Driven Design principles for scalability and maintainability.
"""

import logging
import os

__version__ = "0.3.0"
__author__ = "Anomaly Detection Team"
__email__ = "team@anomaly_detection.io"

# Configure lazy loading based on environment
_LAZY_LOADING = os.getenv("ANOMALY_DETECTION_LAZY_INIT", "true").lower() == "true"
_DOMAIN_MODE = os.getenv("ANOMALY_DETECTION_DOMAIN_MODE", "enabled").lower() == "enabled"

logger = logging.getLogger(__name__)

def _lazy_import(module_path: str, class_name: str):
    """Lazy import utility for performance optimization."""
    def _import():
        try:
            module = __import__(module_path, fromlist=[class_name])
            return getattr(module, class_name)
        except ImportError as e:
            logger.warning(f"Failed to import {class_name} from {module_path}: {e}")
            return None
    return _import

# Optimized imports - lazy loading for better startup performance
if _LAZY_LOADING:
    # Lazy imports for performance
    DetectionService = _lazy_import(".domain.services.detection_service", "DetectionService")
    EnsembleService = _lazy_import(".domain.services.ensemble_service", "EnsembleService")
    SklearnAdapter = _lazy_import(".infrastructure.adapters.algorithms.adapters.sklearn_adapter", "SklearnAdapter")
    PyODAdapter = _lazy_import(".infrastructure.adapters.algorithms.adapters.pyod_adapter", "PyODAdapter")
    DeepLearningAdapter = _lazy_import(".infrastructure.adapters.algorithms.adapters.deeplearning_adapter", "DeepLearningAdapter")
    ExplanationAnalyzers = _lazy_import(".application.services.explanation.analyzers", "ExplanationAnalyzers")
    PerformanceOptimizer = _lazy_import(".application.services.performance.optimization", "PerformanceOptimizer")
    UnifiedModelRegistry = _lazy_import(".application.services.mlops", "UnifiedModelRegistry")
    ExperimentTrackingIntegration = _lazy_import(".application.services.mlops", "ExperimentTrackingIntegration")
    initialize_unified_model_registry = _lazy_import(".application.services.mlops", "initialize_unified_model_registry")
    initialize_experiment_tracking_integration = _lazy_import(".application.services.mlops", "initialize_experiment_tracking_integration")
else:
    # Direct imports for immediate availability
    from .application.services.explanation.analyzers import ExplanationAnalyzers
    from .application.services.mlops import (
        ExperimentTrackingIntegration,
        UnifiedModelRegistry,
        initialize_experiment_tracking_integration,
        initialize_unified_model_registry,
    )
    from .application.services.performance.optimization import PerformanceOptimizer
    from .domain.services.detection_service import DetectionService
    from .domain.services.ensemble_service import EnsembleService
    from .infrastructure.adapters.algorithms.adapters.deeplearning_adapter import (
        DeepLearningAdapter,
    )
    from .infrastructure.adapters.algorithms.adapters.pyod_adapter import PyODAdapter
    from .infrastructure.adapters.algorithms.adapters.sklearn_adapter import (
        SklearnAdapter,
    )

__all__ = [
    "DeepLearningAdapter",
    # Domain services
    "DetectionService",
    "EnsembleService",
    "ExperimentTrackingIntegration",
    # Application services
    "ExplanationAnalyzers",
    "PerformanceOptimizer",
    "PyODAdapter",
    # Infrastructure adapters
    "SklearnAdapter",
    # MLOps integration
    "UnifiedModelRegistry",
    "__author__",
    "__email__",
    "__version__",
    "initialize_experiment_tracking_integration",
    "initialize_unified_model_registry",
]
