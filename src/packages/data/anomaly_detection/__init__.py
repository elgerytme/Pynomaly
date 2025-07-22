"""
Anomaly Detection - Domain-specific anomaly detection package.

This package provides clean, maintainable anomaly detection functionality 
focused on domain logic. It integrates with the broader ML infrastructure
in @src/packages/ai/machine_learning for AutoML, MLOps, and general ML capabilities.

Key Features:
- Core anomaly detection algorithms and services
- Algorithm adapters for various ML frameworks
- Domain-specific detection logic and entities
- Integration with enterprise ML/MLOps infrastructure
- Real-time streaming detection capabilities

Examples:
    Basic usage:
        >>> from anomaly_detection import AnomalyDetector
        >>> detector = AnomalyDetector()
        >>> results = detector.detect(data)

    Advanced usage:
        >>> from anomaly_detection.core.services import DetectionService, EnsembleService
        >>> service = DetectionService()
        >>> ensemble = EnsembleService()
"""

from __future__ import annotations

__version__ = "0.3.0"  # Updated for consolidation
__author__ = "Anomaly Detection Team"
__email__ = "team@anomaly-detection.io"

# Core exports
from .core.services.detection_service import DetectionService, DetectionResult
from .core.services.ensemble_service import EnsembleService
from .core.services.streaming_service import StreamingService

# Algorithm adapters
try:
    from .algorithms.adapters import SklearnAdapter, PyODAdapter, PyODEnsemble, DeepLearningAdapter
except ImportError:
    SklearnAdapter = None  # type: ignore
    PyODAdapter = None  # type: ignore
    PyODEnsemble = None  # type: ignore
    DeepLearningAdapter = None  # type: ignore

# ML integration - delegates to @src/packages/ai/machine_learning
try:
    # Import AutoML from the proper ML package
    from ...ai.machine_learning.domain.services.automl_service import AutoMLService
    from ...ai.machine_learning.mlops.services.model_management_service import ModelManagementService
except ImportError:
    AutoMLService = None  # type: ignore  
    ModelManagementService = None  # type: ignore

# Main public API
class AnomalyDetector:
    """Main entry point for anomaly detection with ML integration."""
    
    def __init__(self, algorithm: str = "iforest", **kwargs):
        """Initialize detector with specified algorithm."""
        self.algorithm = algorithm
        self.config = kwargs
        self._service = DetectionService()
        
    def detect(self, data, **kwargs):
        """Detect anomalies in data."""
        return self._service.detect_anomalies(data, self.algorithm, **kwargs)
        
    def fit(self, data, **kwargs):
        """Fit the detector on training data.""" 
        return self._service.fit(data, self.algorithm, **kwargs)
        
    def predict(self, data, **kwargs):
        """Predict anomalies on new data."""
        return self._service.predict(data, **kwargs)

# Factory functions
def get_detector(algorithm: str = "iforest", **kwargs) -> AnomalyDetector:
    """Get a detector instance."""
    return AnomalyDetector(algorithm, **kwargs)

def get_ensemble_detector(**kwargs) -> EnsembleService:
    """Get an ensemble detector."""
    return EnsembleService(**kwargs)

def get_streaming_detector(**kwargs) -> StreamingService:
    """Get a streaming detector."""
    return StreamingService(**kwargs)

def get_automl_service(**kwargs):
    """Get AutoML service from ML package."""
    if AutoMLService is None:
        raise ImportError("AutoML service not available. Check ML package installation.")
    return AutoMLService(**kwargs)

# Utility functions
def list_algorithms() -> list[str]:
    """List available algorithms."""
    service = DetectionService()
    return service.list_available_algorithms()

def get_version() -> str:
    """Get package version."""
    return __version__

# Public API
__all__ = [
    # Core services
    "DetectionService",
    "DetectionResult",
    "EnsembleService", 
    "StreamingService",
    
    # Main entry point
    "AnomalyDetector",
    
    # Algorithm adapters
    "SklearnAdapter",
    "PyODAdapter",
    "PyODEnsemble",
    "DeepLearningAdapter",
    
    # ML integration
    "AutoMLService",
    "ModelManagementService",
    
    # Factory functions
    "get_detector",
    "get_ensemble_detector",
    "get_streaming_detector", 
    "get_automl_service",
    
    # Utility functions
    "list_algorithms",
    "get_version",
    
    # Version info
    "__version__",
    "__author__",
    "__email__",
]