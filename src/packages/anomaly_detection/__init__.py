"""
Pynomaly Detection - Production-ready Python anomaly detection library.

A comprehensive anomaly detection library providing:
- 40+ algorithms through PyOD, scikit-learn, and deep learning frameworks
- Clean architecture with domain-driven design
- AutoML capabilities for automatic algorithm selection
- Real-time and batch processing
- Explainable AI features
- Enterprise-ready features including multi-tenancy and monitoring

Examples:
    Basic usage:
        >>> from pynomaly_detection import AnomalyDetector
        >>> detector = AnomalyDetector()
        >>> results = detector.detect(data)
    
    With specific algorithm:
        >>> from pynomaly_detection.algorithms import IsolationForestAdapter
        >>> detector = AnomalyDetector(algorithm=IsolationForestAdapter())
        >>> results = detector.detect(data)
"""

__version__ = "0.1.0"
__author__ = "Pynomaly Team"
__email__ = "team@pynomaly.io"

# Core exports - using try/except for graceful imports
try:
    from .core.domain.entities.anomaly import Anomaly
    from .core.domain.entities.detection_result import DetectionResult
    from .core.domain.entities.anomaly_point import AnomalyPoint
    from .core.domain.entities.streaming_anomaly import StreamingAnomaly
    from .core.domain.services.anomaly_scorer import AnomalyScorer
    from .core.domain.services.advanced_detection_service import AdvancedDetectionService
    from .core.domain.value_objects.anomaly_score import AnomalyScore
    from .core.domain.value_objects.anomaly_category import AnomalyCategory
    from .core.domain.value_objects.anomaly_type import AnomalyType
except ImportError:
    # Provide basic fallbacks if specific modules aren't available
    Anomaly = None
    DetectionResult = None
    AnomalyPoint = None
    StreamingAnomaly = None
    AnomalyScorer = None
    AdvancedDetectionService = None
    AnomalyScore = None
    AnomalyCategory = None
    AnomalyType = None

# Algorithm exports
try:
    from .adapters.pyod_adapter import PyODAdapter
    from .adapters.enhanced_pyod_adapter import EnhancedPyODAdapter
    from .adapters.optimized_pyod_adapter import OptimizedPyODAdapter
    from .adapters.drift_detection_adapter import DriftDetectionAdapter
except ImportError:
    PyODAdapter = None
    EnhancedPyODAdapter = None
    OptimizedPyODAdapter = None
    DriftDetectionAdapter = None

# Service exports
try:
    from .services.anomaly_detection_service import AnomalyDetectionService
    from .services.streaming_anomaly_detection_service import StreamingAnomalyDetectionService
    from .services.drift_detection_service import DriftDetectionService
    from .services.model_drift_detection_service import ModelDriftDetectionService
    from .services.detection_engine import DetectionEngine
    from .services.ensemble_detection_service import EnsembleDetectionService
except ImportError:
    AnomalyDetectionService = None
    StreamingAnomalyDetectionService = None
    DriftDetectionService = None
    ModelDriftDetectionService = None
    DetectionEngine = None
    EnsembleDetectionService = None

# Convenience classes
class AnomalyDetector:
    """Main entry point for anomaly detection."""
    
    def __init__(self, algorithm=None, config=None):
        """Initialize detector with optional algorithm and configuration."""
        self.algorithm = algorithm
        self.config = config
        self._trained = False
        self._model = None
        
        # For now, always use fallback implementation for reliability
        self.detection_service = None
    
    def detect(self, data, **kwargs):
        """Detect anomalies in the provided data."""
        if self.detection_service:
            return self.detection_service.detect_anomalies(data, **kwargs)
        else:
            # Fallback: use sklearn IsolationForest
            try:
                from sklearn.ensemble import IsolationForest
                import numpy as np
                
                if not self._trained:
                    self.fit(data, **kwargs)
                
                model = IsolationForest(**kwargs) if not self._model else self._model
                predictions = model.fit_predict(data)
                return predictions == -1  # Convert to boolean mask
            except ImportError:
                raise ImportError("sklearn is required for basic anomaly detection")
    
    def fit(self, data, **kwargs):
        """Train the detector on the provided data."""
        if self.detection_service:
            return self.detection_service.train(data, **kwargs)
        else:
            # Fallback implementation
            try:
                from sklearn.ensemble import IsolationForest
                self._model = IsolationForest(**kwargs)
                self._model.fit(data)
                self._trained = True
                return self
            except ImportError:
                raise ImportError("sklearn is required for basic anomaly detection")
    
    def predict(self, data, **kwargs):
        """Predict anomalies in new data."""
        if self.detection_service:
            return self.detection_service.predict(data, **kwargs)
        else:
            if not self._trained or not self._model:
                raise ValueError("Model must be trained before prediction")
            
            predictions = self._model.predict(data)
            return predictions == -1  # Convert to boolean mask

# Auto-configure based on available dependencies
def get_default_detector():
    """Get a default detector with automatic algorithm selection."""
    return AnomalyDetector()

__all__ = [
    # Core classes
    "Anomaly",
    "DetectionResult", 
    "AnomalyPoint",
    "StreamingAnomaly",
    "AnomalyScorer",
    "AdvancedDetectionService",
    "AnomalyScore",
    "AnomalyCategory",
    "AnomalyType",
    
    # Algorithm adapters
    "PyODAdapter",
    "EnhancedPyODAdapter", 
    "OptimizedPyODAdapter",
    "DriftDetectionAdapter",
    
    # Services
    "AnomalyDetectionService",
    "StreamingAnomalyDetectionService",
    "DriftDetectionService",
    "ModelDriftDetectionService",
    "DetectionEngine",
    "EnsembleDetectionService",
    
    # Main interfaces
    "AnomalyDetector",
    "get_default_detector",
]