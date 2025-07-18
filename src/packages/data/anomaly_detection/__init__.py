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

    Phase 2 simplified services:
        >>> from pynomaly_detection import CoreDetectionService, AutoMLService
        >>> detector = CoreDetectionService()
        >>> result = detector.detect_anomalies(data, algorithm="iforest")
        
    Enhanced features:
        >>> from pynomaly_detection import ModelPersistence, AdvancedExplainability
        >>> persistence = ModelPersistence()
        >>> explainer = AdvancedExplainability()
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import numpy as np
import numpy.typing as npt

if TYPE_CHECKING:
    from sklearn.ensemble import IsolationForest

__version__ = "0.2.0"  # Updated for Phase 2
__author__ = "Pynomaly Team"
__email__ = "team@anomaly-detection.io"

# ========== PHASE 2 SIMPLIFIED SERVICES (RECOMMENDED) ==========

# Core simplified services - these are the recommended interfaces
try:
    from simplified_services.core_detection_service import CoreDetectionService
    from simplified_services.automl_service import AutoMLService
    from simplified_services.ensemble_service import EnsembleService
    from simplified_services.explainability_service import ExplainabilityService
except ImportError:
    CoreDetectionService = None  # type: ignore
    AutoMLService = None  # type: ignore
    EnsembleService = None  # type: ignore
    ExplainabilityService = None  # type: ignore

# Performance optimizations
try:
    from performance.batch_processor import BatchProcessor
    from performance.streaming_detector import StreamingDetector
    from performance.memory_optimizer import MemoryOptimizer
except ImportError:
    BatchProcessor = None  # type: ignore
    StreamingDetector = None  # type: ignore
    MemoryOptimizer = None  # type: ignore

# Specialized algorithms
try:
    from specialized_algorithms.time_series_detector import TimeSeriesDetector
    from specialized_algorithms.text_anomaly_detector import TextAnomalyDetector
except ImportError:
    TimeSeriesDetector = None  # type: ignore
    TextAnomalyDetector = None  # type: ignore

# Enhanced features
try:
    from enhanced_features.model_persistence import ModelPersistence, ModelMetadata
    from enhanced_features.advanced_explainability import AdvancedExplainability, ExplanationResult
    from enhanced_features.integration_adapters import IntegrationManager, create_adapter
    from enhanced_features.monitoring_alerting import MonitoringAlertingSystem, Alert, AlertSeverity
except ImportError:
    ModelPersistence = None  # type: ignore
    ModelMetadata = None  # type: ignore
    AdvancedExplainability = None  # type: ignore
    ExplanationResult = None  # type: ignore
    IntegrationManager = None  # type: ignore
    create_adapter = None  # type: ignore
    MonitoringAlertingSystem = None  # type: ignore
    Alert = None  # type: ignore
    AlertSeverity = None  # type: ignore

# ========== BACKWARD COMPATIBILITY (LEGACY PHASE 1) ==========

# Legacy core exports - maintained for backward compatibility
try:
    from .application.use_cases.detect_anomalies import DetectAnomaliesUseCase
    from .domain.entities.anomaly import Anomaly
    from .domain.entities.detection_result import DetectionResult
    from .domain.entities.detector import Detector
    from .domain.services.detection_service import DetectionService
except ImportError:
    # Provide basic fallbacks if specific modules aren't available
    Anomaly = None  # type: ignore
    DetectionResult = None  # type: ignore
    Detector = None  # type: ignore
    DetectionService = None
    DetectAnomaliesUseCase = None  # type: ignore

# Legacy algorithm exports
try:
    from .algorithms.adapters.ensemble_adapter import EnsembleAdapter
    from .algorithms.adapters.pyod_adapter import PyODAdapter
    from .algorithms.adapters.sklearn_adapter import SklearnAdapter
    from .algorithms.adapters.simple_pyod_adapter import SimplePyODAdapter
except ImportError:
    PyODAdapter = None  # type: ignore
    SklearnAdapter = None  # type: ignore
    EnsembleAdapter = None  # type: ignore
    SimplePyODAdapter = None  # type: ignore

# Legacy service exports
try:
    from .services.anomaly_detection_service import AnomalyDetectionService
    from .services.automl_service import AutoMLService as LegacyAutoMLService
    from .services.explainability_service import ExplainabilityService as LegacyExplainabilityService
except ImportError:
    AnomalyDetectionService = None  # type: ignore
    LegacyAutoMLService = None  # type: ignore
    LegacyExplainabilityService = None  # type: ignore

# Type aliases for better readability
ArrayLike = npt.NDArray[np.floating[Any]] | list[float] | list[list[float]]
PredictionArray = npt.NDArray[np.integer[Any]] | list[int]


# ========== UNIFIED ENTRY POINTS ==========

class AnomalyDetector:
    """Main entry point for anomaly detection with Phase 2 integration.

    This class provides a unified interface that can use either the new
    simplified services (Phase 2) or fallback to basic sklearn implementation.

    Args:
        algorithm: Algorithm name or adapter
        config: Optional configuration dictionary
        use_phase2: Whether to use Phase 2 services (default: True)

    Attributes:
        algorithm: The algorithm adapter (if any)
        config: Configuration dictionary
        use_phase2: Whether using Phase 2 services
    """

    def __init__(
        self, 
        algorithm: Any | None = None, 
        config: dict[str, Any] | None = None,
        use_phase2: bool = True
    ) -> None:
        """Initialize detector with optional algorithm and configuration."""
        self.algorithm = algorithm
        self.config = config or {}
        self.use_phase2 = use_phase2
        self._trained: bool = False
        self._model: IsolationForest | None = None

        # Try to initialize Phase 2 services
        if use_phase2 and CoreDetectionService is not None:
            self.detection_service = CoreDetectionService()
            self._phase2_available = True
        else:
            self.detection_service = None
            self._phase2_available = False

    def detect(self, data: ArrayLike, **kwargs: Any) -> PredictionArray:
        """Detect anomalies in the provided data.

        This method combines fitting and prediction in a single call.

        Args:
            data: Input data array of shape (n_samples, n_features)
            **kwargs: Additional parameters passed to the underlying algorithm

        Returns:
            Array of predictions where 1 indicates anomaly, 0 indicates normal

        Raises:
            ImportError: If scikit-learn is not installed
        """
        if self._phase2_available and self.detection_service:
            # Use Phase 2 simplified services
            algorithm = kwargs.get('algorithm', self.algorithm or 'iforest')
            contamination = kwargs.get('contamination', 0.1)
            
            result = self.detection_service.detect_anomalies(
                np.array(data), 
                algorithm=algorithm, 
                contamination=contamination
            )
            return result.predictions
        else:
            # Fallback: use sklearn IsolationForest
            try:
                from sklearn.ensemble import IsolationForest

                if not self._trained:
                    self.fit(data, **kwargs)

                model = IsolationForest(**kwargs) if not self._model else self._model
                predictions = model.fit_predict(data)
                result: npt.NDArray[np.integer[Any]] = (predictions == -1).astype(int)
                return result
            except ImportError:
                msg = "sklearn is required for basic anomaly detection"
                raise ImportError(msg) from None

    def fit(self, data: ArrayLike, **kwargs: Any) -> 'AnomalyDetector':
        """Train the detector on the provided data.

        Args:
            data: Input data array of shape (n_samples, n_features)
            **kwargs: Additional parameters passed to the underlying algorithm.
                     Common parameters include:
                     - contamination: float or 'auto', expected proportion of outliers
                     - random_state: int, random seed for reproducibility

        Returns:
            self: Returns the fitted detector instance

        Raises:
            ImportError: If scikit-learn is not installed
        """
        if self._phase2_available and self.detection_service:
            # Phase 2 services don't require explicit fitting
            self._trained = True
            return self
        else:
            # Fallback implementation
            try:
                from sklearn.ensemble import IsolationForest

                # Set default contamination if not provided
                if "contamination" not in kwargs:
                    kwargs["contamination"] = "auto"
                self._model = IsolationForest(**kwargs)
                self._model.fit(data)
                self._trained = True
                return self
            except ImportError:
                msg = "sklearn is required for basic anomaly detection"
                raise ImportError(msg) from None

    def predict(self, data: ArrayLike, **kwargs: Any) -> PredictionArray:
        """Predict anomalies in new data.

        Args:
            data: Input data array of shape (n_samples, n_features)
            **kwargs: Additional parameters (currently unused)

        Returns:
            Array of predictions where 1 indicates anomaly, 0 indicates normal

        Raises:
            ValueError: If the model hasn't been trained yet
            ImportError: If scikit-learn is not installed
        """
        if self._phase2_available and self.detection_service:
            # Use Phase 2 services
            return self.detect(data, **kwargs)
        else:
            if not self._trained or not self._model:
                raise ValueError("Model must be trained before prediction")

            predictions = self._model.predict(data)
            result: npt.NDArray[np.integer[Any]] = (predictions == -1).astype(int)
            return result

    def get_feature_names(self) -> list[str] | None:
        """Get feature names if available."""
        if hasattr(self, '_feature_names'):
            return self._feature_names
        return None

    def is_phase2_available(self) -> bool:
        """Check if Phase 2 services are available."""
        return self._phase2_available


# ========== FACTORY FUNCTIONS ==========

def get_default_detector() -> AnomalyDetector:
    """Get a default detector with automatic algorithm selection.

    Returns:
        AnomalyDetector: A new detector instance with default configuration
    """
    return AnomalyDetector()


def get_core_detector() -> 'CoreDetectionService':
    """Get Phase 2 core detection service.
    
    Returns:
        CoreDetectionService: Core detection service instance
        
    Raises:
        ImportError: If Phase 2 services are not available
    """
    if CoreDetectionService is None:
        raise ImportError("Phase 2 services not available. Please check your installation.")
    return CoreDetectionService()


def get_automl_detector() -> 'AutoMLService':
    """Get Phase 2 AutoML service.
    
    Returns:
        AutoMLService: AutoML service instance
        
    Raises:
        ImportError: If Phase 2 services are not available
    """
    if AutoMLService is None:
        raise ImportError("Phase 2 services not available. Please check your installation.")
    return AutoMLService()


def get_ensemble_detector() -> 'EnsembleService':
    """Get Phase 2 ensemble service.
    
    Returns:
        EnsembleService: Ensemble service instance
        
    Raises:
        ImportError: If Phase 2 services are not available
    """
    if EnsembleService is None:
        raise ImportError("Phase 2 services not available. Please check your installation.")
    return EnsembleService()


def create_monitoring_system() -> 'MonitoringAlertingSystem':
    """Create a monitoring and alerting system.
    
    Returns:
        MonitoringAlertingSystem: Monitoring system instance
        
    Raises:
        ImportError: If Phase 2 enhanced features are not available
    """
    if MonitoringAlertingSystem is None:
        raise ImportError("Phase 2 enhanced features not available. Please check your installation.")
    return MonitoringAlertingSystem()


def create_model_persistence(storage_path: str = "models") -> 'ModelPersistence':
    """Create a model persistence system.
    
    Args:
        storage_path: Path for model storage
        
    Returns:
        ModelPersistence: Model persistence instance
        
    Raises:
        ImportError: If Phase 2 enhanced features are not available
    """
    if ModelPersistence is None:
        raise ImportError("Phase 2 enhanced features not available. Please check your installation.")
    return ModelPersistence(storage_path)


# ========== COMPATIBILITY UTILITIES ==========

def check_phase2_availability() -> dict[str, bool]:
    """Check availability of Phase 2 components.
    
    Returns:
        dict: Availability status of different Phase 2 components
    """
    return {
        "simplified_services": CoreDetectionService is not None,
        "performance_features": BatchProcessor is not None,
        "specialized_algorithms": TimeSeriesDetector is not None,
        "enhanced_features": ModelPersistence is not None,
        "monitoring": MonitoringAlertingSystem is not None,
        "integration": IntegrationManager is not None
    }


def get_version_info() -> dict[str, Any]:
    """Get version and feature information.
    
    Returns:
        dict: Version and feature availability information
    """
    return {
        "version": __version__,
        "phase2_available": check_phase2_availability(),
        "recommended_entry_points": [
            "CoreDetectionService",
            "AutoMLService", 
            "EnsembleService",
            "ModelPersistence",
            "MonitoringAlertingSystem"
        ]
    }


# ========== EXPORTS ==========

__all__ = [
    # ===== PHASE 2 SIMPLIFIED SERVICES (RECOMMENDED) =====
    # Core services
    "CoreDetectionService",
    "AutoMLService", 
    "EnsembleService",
    "ExplainabilityService",
    
    # Performance features
    "BatchProcessor",
    "StreamingDetector",
    "MemoryOptimizer",
    
    # Specialized algorithms
    "TimeSeriesDetector",
    "TextAnomalyDetector",
    
    # Enhanced features
    "ModelPersistence",
    "ModelMetadata",
    "AdvancedExplainability",
    "ExplanationResult",
    "IntegrationManager",
    "create_adapter",
    "MonitoringAlertingSystem",
    "Alert",
    "AlertSeverity",
    
    # ===== UNIFIED ENTRY POINTS =====
    "AnomalyDetector",
    "get_default_detector",
    "get_core_detector",
    "get_automl_detector",
    "get_ensemble_detector",
    "create_monitoring_system",
    "create_model_persistence",
    
    # ===== UTILITIES =====
    "check_phase2_availability",
    "get_version_info",
    
    # ===== BACKWARD COMPATIBILITY (LEGACY) =====
    # Legacy core classes
    "Anomaly",
    "DetectionResult",
    "Detector",
    "DetectionService",
    "DetectAnomaliesUseCase",
    
    # Legacy algorithm adapters
    "PyODAdapter",
    "SklearnAdapter",
    "EnsembleAdapter",
    "SimplePyODAdapter",
    
    # Legacy services
    "AnomalyDetectionService",
]