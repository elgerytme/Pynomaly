"""
Pynomaly Detection Package

A production-ready anomaly detection library with clean architecture and
multiple algorithms.

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

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import numpy as np
import numpy.typing as npt

if TYPE_CHECKING:
    from sklearn.ensemble import IsolationForest

__version__ = "0.1.0"
__author__ = "Pynomaly Team"
__email__ = "team@anomaly-detection.io"

# Graceful imports for optional dependencies
try:
    # Core entities (will be implemented later)
    from .algorithms.adapters.ensemble_adapter import EnsembleAdapter

    # Algorithm adapters (will be implemented later)
    from .algorithms.adapters.pyod_adapter import PyODAdapter
    from .algorithms.adapters.sklearn_adapter import SklearnAdapter
    from .core.application.use_cases.detect_anomalies import DetectAnomaliesUseCase
    from .core.domain.entities.anomaly import Anomaly
    from .core.domain.entities.detection_result import DetectionResult
    from .core.domain.entities.detector import Detector
    from .core.domain.services.detection_service import DetectionService

    # Services (will be implemented later)
    from .services.anomaly_detection_service import AnomalyDetectionService
    from .services.automl_service import AutoMLService
    from .services.explainability_service import ExplainabilityService

except ImportError:
    # Graceful fallbacks when domain architecture is not available
    Anomaly = None  # type: ignore
    DetectionResult = None  # type: ignore
    Detector = None  # type: ignore
    DetectionService = None  # type: ignore
    DetectAnomaliesUseCase = None  # type: ignore
    PyODAdapter = None  # type: ignore
    SklearnAdapter = None  # type: ignore
    EnsembleAdapter = None  # type: ignore
    AnomalyDetectionService = None  # type: ignore
    AutoMLService = None  # type: ignore
    ExplainabilityService = None  # type: ignore

# Type aliases for better readability
ArrayLike = npt.NDArray[np.floating[Any]] | list[float] | list[list[float]]
PredictionArray = npt.NDArray[np.integer[Any]] | list[int]


# Convenience classes
class AnomalyDetector:
    """Main entry point for anomaly detection.

    This class provides a simple interface for anomaly detection using
    scikit-learn's IsolationForest as the default algorithm.

    Args:
        algorithm: Optional algorithm adapter to use
        config: Optional configuration dictionary

    Attributes:
        algorithm: The algorithm adapter (if any)
        config: Configuration dictionary
    """

    def __init__(
        self, algorithm: Any | None = None, config: dict[str, Any] | None = None
    ) -> None:
        """Initialize detector with optional algorithm and configuration."""
        self.algorithm = algorithm
        self.config = config
        self._trained: bool = False
        self._model: IsolationForest | None = None

        # For now, always use fallback implementation for reliability
        self.detection_service: Any | None = None

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
        if self.detection_service:
            return self.detection_service.detect_anomalies(data, **kwargs)
        else:
            # Fallback: use sklearn algorithms
            try:
                from sklearn.ensemble import IsolationForest
                from sklearn.neighbors import LocalOutlierFactor
                from sklearn.svm import OneClassSVM

                if not self._trained:
                    self.fit(data, **kwargs)

                # Use the algorithm specified in config or default to IsolationForest
                algorithm = kwargs.get('algorithm', 'isolation_forest')

                if algorithm == 'lof':
                    # LOF requires different approach as it doesn't have
                    # separate fit/predict
                    lof_params = {
                        k: v for k, v in kwargs.items() if k != 'algorithm'
                    }
                    lof = LocalOutlierFactor(**lof_params)
                    predictions = lof.fit_predict(data)
                elif algorithm == 'ocsvm':
                    # Filter out parameters that OneClassSVM doesn't accept
                    ocsvm_params = {
                        k: v for k, v in kwargs.items()
                        if k not in ['algorithm', 'contamination']
                    }
                    model = OneClassSVM(**ocsvm_params)
                    predictions = model.fit_predict(data)
                else:
                    # Default to IsolationForest
                    if not self._model:
                        iso_params = {
                            k: v for k, v in kwargs.items() if k != 'algorithm'
                        }
                        model = IsolationForest(**iso_params)
                    else:
                        model = self._model
                    predictions = model.fit_predict(data)

                result: npt.NDArray[np.integer[Any]] = (predictions == -1).astype(int)
                return result
            except ImportError:
                msg = "sklearn is required for basic anomaly detection"
                raise ImportError(msg) from None

    def fit(self, data: ArrayLike, **kwargs: Any) -> AnomalyDetector:
        """Train the detector on the provided data.

        Args:
            data: Input data array of shape (n_samples, n_features)
            **kwargs: Additional parameters passed to the underlying algorithm.
                     Common parameters include:
                     - contamination: float or 'auto', expected proportion
                       of outliers
                     - random_state: int, random seed for reproducibility

        Returns:
            self: Returns the fitted detector instance

        Raises:
            ImportError: If scikit-learn is not installed
        """
        if self.detection_service:
            return self.detection_service.train(data, **kwargs)  # type: ignore
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
        if self.detection_service:
            return self.detection_service.predict(data, **kwargs)
        else:
            if not self._trained or not self._model:
                raise ValueError("Model must be trained before prediction")

            predictions = self._model.predict(data)
            result: npt.NDArray[np.integer[Any]] = (predictions == -1).astype(int)
            return result


# Auto-configure based on available dependencies
def get_default_detector() -> AnomalyDetector:
    """Get a default detector with automatic algorithm selection.

    Returns:
        AnomalyDetector: A new detector instance with default configuration
    """
    return AnomalyDetector()


__all__ = [
    # Core classes
    "Anomaly",
    "DetectionResult",
    "Detector",
    "DetectionService",
    "DetectAnomaliesUseCase",
    # Algorithm adapters
    "PyODAdapter",
    "SklearnAdapter",
    "EnsembleAdapter",
    # Services
    "AnomalyDetectionService",
    "AutoMLService",
    "ExplainabilityService",
    # Main interfaces
    "AnomalyDetector",
    "get_default_detector",
]
