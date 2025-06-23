"""Scikit-learn adapter for anomaly detection algorithms."""

from __future__ import annotations

import time
from typing import Any, Dict, List, Optional
from uuid import uuid4

import numpy as np
from sklearn.base import BaseEstimator

from pynomaly.domain.entities import (
    Anomaly, Dataset, Detector, DetectionResult
)
from pynomaly.domain.exceptions import (
    DetectorNotFittedError, FittingError, InvalidAlgorithmError
)
from pynomaly.domain.value_objects import AnomalyScore, ContaminationRate
from pynomaly.shared.protocols import DetectorProtocol


class SklearnAdapter(Detector, DetectorProtocol):
    """Adapter for scikit-learn anomaly detection algorithms."""
    
    # Mapping of algorithm names to sklearn classes
    ALGORITHM_MAPPING: Dict[str, tuple[str, str]] = {
        "IsolationForest": ("sklearn.ensemble", "IsolationForest"),
        "LocalOutlierFactor": ("sklearn.neighbors", "LocalOutlierFactor"),
        "OneClassSVM": ("sklearn.svm", "OneClassSVM"),
        "EllipticEnvelope": ("sklearn.covariance", "EllipticEnvelope"),
        "SGDOneClassSVM": ("sklearn.linear_model", "SGDOneClassSVM"),
    }
    
    def __init__(
        self,
        algorithm_name: str,
        name: Optional[str] = None,
        contamination_rate: Optional[ContaminationRate] = None,
        **kwargs: Any
    ):
        """Initialize sklearn adapter.
        
        Args:
            algorithm_name: Name of the sklearn algorithm
            name: Optional custom name for the detector
            contamination_rate: Expected contamination rate
            **kwargs: Algorithm-specific parameters
        """
        # Validate algorithm
        if algorithm_name not in self.ALGORITHM_MAPPING:
            raise InvalidAlgorithmError(
                algorithm_name,
                available_algorithms=list(self.ALGORITHM_MAPPING.keys())
            )
        
        # Initialize parent
        super().__init__(
            name=name or f"Sklearn_{algorithm_name}",
            algorithm_name=algorithm_name,
            contamination_rate=contamination_rate or ContaminationRate.auto(),
            parameters=kwargs
        )
        
        # Load sklearn model class
        self._model_class = self._load_model_class(algorithm_name)
        self._model: Optional[BaseEstimator] = None
        
        # Set metadata
        self._set_algorithm_metadata(algorithm_name)
    
    def _load_model_class(self, algorithm_name: str) -> type[BaseEstimator]:
        """Dynamically load sklearn model class."""
        module_path, class_name = self.ALGORITHM_MAPPING[algorithm_name]
        
        try:
            import importlib
            module = importlib.import_module(module_path)
            return getattr(module, class_name)
        except (ImportError, AttributeError) as e:
            raise InvalidAlgorithmError(
                algorithm_name,
                available_algorithms=list(self.ALGORITHM_MAPPING.keys())
            ) from e
    
    def _set_algorithm_metadata(self, algorithm_name: str) -> None:
        """Set metadata based on algorithm characteristics."""
        # Time/space complexity
        complexity_info = {
            "IsolationForest": ("O(n log n)", "O(n)"),
            "LocalOutlierFactor": ("O(n²)", "O(n)"),
            "OneClassSVM": ("O(n² × d)", "O(n²)"),
            "EllipticEnvelope": ("O(n × d²)", "O(d²)"),
            "SGDOneClassSVM": ("O(n × d)", "O(d)"),
        }
        
        if algorithm_name in complexity_info:
            time_complexity, space_complexity = complexity_info[algorithm_name]
            self.update_metadata("time_complexity", time_complexity)
            self.update_metadata("space_complexity", space_complexity)
        
        # Special characteristics
        if algorithm_name == "LocalOutlierFactor":
            self.update_metadata("supports_novelty", True)
            self.update_metadata("requires_neighbors", True)
        
        if algorithm_name == "SGDOneClassSVM":
            self.update_metadata("supports_streaming", True)
            self.update_metadata("is_online", True)
    
    def fit(self, dataset: Dataset) -> None:
        """Fit the sklearn detector on a dataset."""
        try:
            # Initialize model with parameters
            model_params = self.parameters.copy()
            
            # Handle contamination parameter
            if self.algorithm_name in ["IsolationForest", "LocalOutlierFactor", "EllipticEnvelope"]:
                model_params["contamination"] = self.contamination_rate.value
            
            # Special handling for LocalOutlierFactor
            if self.algorithm_name == "LocalOutlierFactor":
                # For training, we need novelty=True to use predict later
                model_params["novelty"] = True
            
            self._model = self._model_class(**model_params)
            
            # Fit on numeric features only
            X = dataset.features[dataset.get_numeric_features()].values
            
            start_time = time.time()
            self._model.fit(X)
            training_time = (time.time() - start_time) * 1000
            
            # Update detector state
            self.is_fitted = True
            self.trained_at = dataset.created_at
            self.update_metadata("training_time_ms", training_time)
            self.update_metadata("training_samples", dataset.n_samples)
            self.update_metadata("training_features", len(dataset.get_numeric_features()))
            
        except Exception as e:
            raise FittingError(
                detector_name=self.name,
                reason=str(e),
                dataset_name=dataset.name
            ) from e
    
    def detect(self, dataset: Dataset) -> DetectionResult:
        """Detect anomalies in a dataset."""
        if not self.is_fitted or self._model is None:
            raise DetectorNotFittedError(
                detector_name=self.name,
                operation="detect"
            )
        
        # Get features
        X = dataset.features[dataset.get_numeric_features()].values
        
        start_time = time.time()
        
        # Get predictions (-1 = anomaly, 1 = normal in sklearn)
        sklearn_labels = self._model.predict(X)
        # Convert to our convention (0 = normal, 1 = anomaly)
        labels = (sklearn_labels == -1).astype(int)
        
        # Get scores
        if hasattr(self._model, "score_samples"):
            # For IsolationForest and others
            # Lower scores = more anomalous in sklearn
            raw_scores = -self._model.score_samples(X)
        elif hasattr(self._model, "decision_function"):
            # For OneClassSVM and others
            # Negative scores = anomalous
            raw_scores = -self._model.decision_function(X)
        else:
            # Fallback: use predictions
            raw_scores = (1 - sklearn_labels) / 2
        
        # Normalize scores to [0, 1]
        min_score = raw_scores.min()
        max_score = raw_scores.max()
        if max_score > min_score:
            normalized_scores = (raw_scores - min_score) / (max_score - min_score)
        else:
            normalized_scores = np.ones_like(raw_scores) * 0.5
        
        # Create AnomalyScore objects
        anomaly_scores = [
            AnomalyScore(value=float(score), method="sklearn")
            for score in normalized_scores
        ]
        
        # Calculate threshold
        threshold_idx = int(len(scores) * (1 - self.contamination_rate.value))
        threshold = float(np.sort(normalized_scores)[threshold_idx])
        
        # Create Anomaly entities
        anomalies = []
        anomaly_indices = np.where(labels == 1)[0]
        
        for idx in anomaly_indices:
            anomaly = Anomaly(
                score=anomaly_scores[idx],
                data_point=dataset.data.iloc[idx].to_dict(),
                detector_name=self.name
            )
            anomaly.add_metadata("raw_score", float(raw_scores[idx]))
            anomaly.add_metadata("algorithm", self.algorithm_name)
            anomalies.append(anomaly)
        
        execution_time = (time.time() - start_time) * 1000
        
        # Create detection result
        result = DetectionResult(
            detector_id=self.id,
            dataset_id=dataset.id,
            anomalies=anomalies,
            scores=anomaly_scores,
            labels=labels,
            threshold=threshold,
            execution_time_ms=execution_time,
            metadata={
                "algorithm": self.algorithm_name,
                "sklearn_version": self._get_sklearn_version()
            }
        )
        
        return result
    
    def score(self, dataset: Dataset) -> List[AnomalyScore]:
        """Calculate anomaly scores for a dataset."""
        if not self.is_fitted or self._model is None:
            raise DetectorNotFittedError(
                detector_name=self.name,
                operation="score"
            )
        
        # Get features
        X = dataset.features[dataset.get_numeric_features()].values
        
        # Get scores
        if hasattr(self._model, "score_samples"):
            raw_scores = -self._model.score_samples(X)
        elif hasattr(self._model, "decision_function"):
            raw_scores = -self._model.decision_function(X)
        else:
            # Fallback
            predictions = self._model.predict(X)
            raw_scores = (1 - predictions) / 2
        
        # Normalize to [0, 1]
        min_score = raw_scores.min()
        max_score = raw_scores.max()
        if max_score > min_score:
            normalized_scores = (raw_scores - min_score) / (max_score - min_score)
        else:
            normalized_scores = np.ones_like(raw_scores) * 0.5
        
        return [
            AnomalyScore(value=float(score), method="sklearn")
            for score in normalized_scores
        ]
    
    def get_params(self) -> Dict[str, Any]:
        """Get current parameters."""
        if self._model is not None:
            return self._model.get_params()
        return self.parameters
    
    def set_params(self, **params: Any) -> None:
        """Set parameters."""
        self.update_parameters(**params)
        if self._model is not None:
            self._model.set_params(**params)
    
    def _get_sklearn_version(self) -> str:
        """Get scikit-learn version."""
        try:
            import sklearn
            return sklearn.__version__
        except:
            return "unknown"