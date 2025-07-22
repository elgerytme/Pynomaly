"""Consolidated detection service for anomaly detection operations."""

from __future__ import annotations

import logging
from typing import Any, Protocol
import numpy as np
import numpy.typing as npt

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


class DetectionResult:
    """Detection result container."""
    
    def __init__(
        self,
        predictions: npt.NDArray[np.integer],
        scores: npt.NDArray[np.floating] | None = None,
        algorithm: str = "unknown",
        metadata: dict[str, Any] | None = None
    ):
        self.predictions = predictions
        self.scores = scores
        self.algorithm = algorithm
        self.metadata = metadata or {}
        self.anomaly_count = int(np.sum(predictions))
        self.normal_count = len(predictions) - self.anomaly_count


class DetectionService:
    """Consolidated service for anomaly detection operations.
    
    This service consolidates functionality from multiple detection services
    into a single, clean interface.
    """
    
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
        """Detect anomalies in data using specified algorithm.
        
        Args:
            data: Input data of shape (n_samples, n_features)
            algorithm: Algorithm name to use
            contamination: Expected proportion of anomalies
            **kwargs: Additional algorithm parameters
            
        Returns:
            DetectionResult with predictions and metadata
        """
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
            
            return DetectionResult(
                predictions=predictions,
                algorithm=algorithm,
                metadata={
                    "contamination": contamination,
                    "data_shape": data.shape,
                    "algorithm_params": kwargs
                }
            )
            
        except Exception as e:
            logger.error(f"Detection failed with algorithm {algorithm}: {e}")
            raise
    
    def fit(
        self,
        data: npt.NDArray[np.floating],
        algorithm: str = "iforest",
        **kwargs: Any
    ) -> DetectionService:
        """Fit a detector on training data.
        
        Args:
            data: Training data
            algorithm: Algorithm name
            **kwargs: Algorithm parameters
            
        Returns:
            Self for method chaining
        """
        if algorithm in self._adapters:
            adapter = self._adapters[algorithm]
            adapter.fit(data)
            self._fitted_models[algorithm] = adapter
        else:
            # Handle built-in algorithms
            model = self._fit_builtin(data, algorithm, **kwargs)
            self._fitted_models[algorithm] = model
            
        return self
    
    def predict(
        self,
        data: npt.NDArray[np.floating],
        algorithm: str = "iforest"
    ) -> DetectionResult:
        """Predict anomalies using fitted model.
        
        Args:
            data: Data to predict on
            algorithm: Algorithm name
            
        Returns:
            DetectionResult with predictions
        """
        if algorithm not in self._fitted_models:
            raise ValueError(f"Algorithm {algorithm} not fitted. Call fit() first.")
            
        model = self._fitted_models[algorithm]
        
        if hasattr(model, 'predict'):
            predictions = model.predict(data)
        else:
            # Handle adapter protocol
            predictions = model.predict(data)
            
        return DetectionResult(predictions=predictions, algorithm=algorithm)
    
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
    
    def _fit_builtin(
        self,
        data: npt.NDArray[np.floating],
        algorithm: str,
        **kwargs: Any
    ) -> Any:
        """Fit built-in algorithm."""
        if algorithm == "iforest":
            try:
                from sklearn.ensemble import IsolationForest
                model = IsolationForest(**kwargs)
                model.fit(data)
                return model
            except ImportError:
                raise ImportError("scikit-learn required for IsolationForest")
        else:
            raise ValueError(f"Unknown algorithm for fitting: {algorithm}")
    
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
            # Convert sklearn output (-1, 1) to (1, 0) for anomalies
            return (predictions == -1).astype(int)
            
        except ImportError:
            raise ImportError("scikit-learn required for IsolationForest")
    
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
            # Convert sklearn output (-1, 1) to (1, 0) for anomalies
            return (predictions == -1).astype(int)
            
        except ImportError:
            raise ImportError("scikit-learn required for LocalOutlierFactor")
    
    def list_available_algorithms(self) -> list[str]:
        """List all available algorithms."""
        builtin = ["iforest", "lof"]
        registered = list(self._adapters.keys())
        return builtin + registered
    
    def get_algorithm_info(self, algorithm: str) -> dict[str, Any]:
        """Get information about an algorithm."""
        info = {"name": algorithm, "type": "unknown"}
        
        if algorithm in ["iforest", "lof"]:
            info["type"] = "builtin"
            info["requires"] = ["scikit-learn"]
        elif algorithm in self._adapters:
            info["type"] = "registered_adapter"
            
        return info