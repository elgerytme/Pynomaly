"""Simplified detection service without complex monitoring dependencies."""

from __future__ import annotations

import logging
from typing import Any, Protocol
import numpy as np
import numpy.typing as npt

from ..entities.detection_result import DetectionResult

logger = logging.getLogger(__name__)


class AlgorithmAdapter(Protocol):
    """Protocol for algorithm adapters."""
    
    def fit(self, data: npt.NDArray[np.floating]) -> None:
        """Fit the algorithm on data."""
        ...
    
    def predict(self, data: npt.NDArray[np.floating]) -> npt.NDArray[np.integer]:
        """Predict anomalies in data."""
        ...
    
    def fit_predict(self, data: npt.NDArray[np.floating]) -> npt.NDArray[np.integer]:
        """Fit and predict in one step."""
        ...


class DetectionServiceSimple:
    """Simplified detection service for core functionality."""
    
    def __init__(self):
        """Initialize detection service."""
        self._adapters: dict[str, AlgorithmAdapter] = {}
        self._fitted_models: dict[str, Any] = {}
    
    def register_adapter(self, name: str, adapter: AlgorithmAdapter) -> None:
        """Register an algorithm adapter."""
        self._adapters[name] = adapter
        
    def detect_anomalies(
        self,
        data: npt.NDArray[np.floating],
        algorithm: str = "iforest",
        contamination: float = 0.1,
        **kwargs: Any
    ) -> DetectionResult:
        """Detect anomalies in data using specified algorithm."""
        try:
            # Try to use registered adapter first
            if algorithm in self._adapters:
                adapter = self._adapters[algorithm]
                predictions = adapter.fit_predict(data)
            else:
                # Fall back to built-in algorithms
                predictions = self._detect_with_builtin(
                    data, algorithm, contamination, **kwargs
                )
            
            # Get confidence scores if available
            confidence_scores = self._get_confidence_scores(data, algorithm, contamination, **kwargs)
            
            result = DetectionResult(
                predictions=predictions,
                confidence_scores=confidence_scores,
                algorithm=algorithm,
                metadata={
                    "contamination": contamination,
                    "data_shape": data.shape,
                    "algorithm_params": kwargs
                }
            )
            
            logger.info(f"Detection completed: {result.anomaly_count} anomalies in {result.total_samples} samples")
            return result
            
        except Exception as e:
            logger.error(f"Detection failed: {e}")
            raise
    
    def _detect_with_builtin(
        self,
        data: npt.NDArray[np.floating],
        algorithm: str,
        contamination: float,
        **kwargs: Any
    ) -> npt.NDArray[np.integer]:
        """Detect using built-in algorithm implementations."""
        if algorithm == "iforest":
            return self._isolation_forest(data, contamination, **kwargs)
        elif algorithm == "lof":
            return self._local_outlier_factor(data, contamination, **kwargs)
        else:
            raise ValueError(f"Unknown algorithm: {algorithm}")
    
    def _isolation_forest(
        self,
        data: npt.NDArray[np.floating],
        contamination: float,
        **kwargs: Any
    ) -> npt.NDArray[np.integer]:
        """Run Isolation Forest algorithm."""
        try:
            from sklearn.ensemble import IsolationForest
            
            model = IsolationForest(
                contamination=contamination,
                random_state=kwargs.get('random_state', 42),
                **kwargs
            )
            predictions = model.fit_predict(data)
            return predictions.astype(np.integer)
            
        except ImportError:
            raise ImportError("scikit-learn required for IsolationForest")
        except Exception as e:
            raise RuntimeError(f"Isolation Forest execution failed: {str(e)}")
    
    def _local_outlier_factor(
        self,
        data: npt.NDArray[np.floating],
        contamination: float,
        **kwargs: Any
    ) -> npt.NDArray[np.integer]:
        """Run Local Outlier Factor algorithm."""
        try:
            from sklearn.neighbors import LocalOutlierFactor
            
            model = LocalOutlierFactor(
                contamination=contamination,
                **kwargs
            )
            predictions = model.fit_predict(data)
            return predictions.astype(np.integer)
            
        except ImportError:
            raise ImportError("scikit-learn required for LocalOutlierFactor")
        except Exception as e:
            raise RuntimeError(f"Local Outlier Factor execution failed: {str(e)}")
    
    def _get_confidence_scores(
        self,
        data: npt.NDArray[np.floating],
        algorithm: str,
        contamination: float,
        **kwargs: Any
    ) -> npt.NDArray[np.floating] | None:
        """Get confidence scores for predictions if available."""
        try:
            if algorithm == "iforest":
                return self._isolation_forest_scores(data, contamination, **kwargs)
            elif algorithm == "lof":
                return self._local_outlier_factor_scores(data, contamination, **kwargs)
            else:
                return None
        except Exception as e:
            logger.warning(f"Failed to get confidence scores for {algorithm}: {e}")
            return None
    
    def _isolation_forest_scores(
        self,
        data: npt.NDArray[np.floating],
        contamination: float,
        **kwargs: Any
    ) -> npt.NDArray[np.floating]:
        """Get confidence scores from Isolation Forest."""
        try:
            from sklearn.ensemble import IsolationForest
            
            model = IsolationForest(
                contamination=contamination,
                random_state=kwargs.get('random_state', 42),
                **kwargs
            )
            model.fit(data)
            scores = model.decision_function(data)
            
            # Convert to confidence scores (0-1, higher = more anomalous)
            threshold = np.percentile(scores, contamination * 100)
            normalized_scores = np.clip((threshold - scores) / (np.max(scores) - threshold + 1e-10), 0, 1)
            return normalized_scores.astype(np.float64)
            
        except Exception as e:
            logger.warning(f"Failed to get Isolation Forest scores: {e}")
            return None
    
    def _local_outlier_factor_scores(
        self,
        data: npt.NDArray[np.floating],
        contamination: float,
        **kwargs: Any
    ) -> npt.NDArray[np.floating]:
        """Get confidence scores from Local Outlier Factor."""
        try:
            from sklearn.neighbors import LocalOutlierFactor
            
            model = LocalOutlierFactor(
                contamination=contamination,
                **kwargs
            )
            model.fit(data)
            scores = model.negative_outlier_factor_
            
            # Convert to confidence scores (0-1, higher = more anomalous)
            min_score = np.min(scores)
            max_score = np.max(scores)
            if min_score < max_score:
                normalized_scores = (max_score - scores) / (max_score - min_score)
            else:
                normalized_scores = np.zeros_like(scores)
            return normalized_scores.astype(np.float64)
            
        except Exception as e:
            logger.warning(f"Failed to get LOF scores: {e}")
            return None
    
    def list_available_algorithms(self) -> list[str]:
        """List all available algorithms."""
        builtin = ["iforest", "lof"]
        registered = list(self._adapters.keys())
        return builtin + registered