"""Enhanced scikit-learn adapter for anomaly detection algorithms."""

from __future__ import annotations

import importlib
import time
import warnings
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator
from sklearn.preprocessing import StandardScaler

from pynomaly.domain.entities import Anomaly, Dataset, DetectionResult
from pynomaly.domain.exceptions import (
    DetectorNotFittedError,
    FittingError,
    InvalidAlgorithmError,
)
from pynomaly.domain.value_objects import AnomalyScore, ContaminationRate
from pynomaly.shared.protocols import DetectorProtocol


@dataclass
class SklearnAlgorithmInfo:
    """Information about scikit-learn algorithms."""
    
    category: str
    complexity_time: str
    complexity_space: str
    supports_streaming: bool
    requires_scaling: bool
    description: str
    hyperparameters: Dict[str, Any]


class EnhancedSklearnAdapter(DetectorProtocol):
    """Enhanced scikit-learn adapter for anomaly detection."""
    
    # Comprehensive algorithm mapping
    ALGORITHM_MAPPING: Dict[str, Tuple[str, str, SklearnAlgorithmInfo]] = {
        "IsolationForest": (
            "sklearn.ensemble", "IsolationForest",
            SklearnAlgorithmInfo(
                category="Ensemble",
                complexity_time="O(n log n)",
                complexity_space="O(n)",
                supports_streaming=False,
                requires_scaling=False,
                description="Isolation Forest for efficient anomaly detection",
                hyperparameters={
                    "n_estimators": [50, 100, 200],
                    "max_samples": ["auto", 256, 512],
                    "contamination": [0.1, 0.05, 0.2],
                    "max_features": [1.0, 0.8, 0.5],
                }
            )
        ),
        "OneClassSVM": (
            "sklearn.svm", "OneClassSVM",
            SklearnAlgorithmInfo(
                category="Support Vector",
                complexity_time="O(n²)",
                complexity_space="O(n²)",
                supports_streaming=False,
                requires_scaling=True,
                description="One-Class Support Vector Machine",
                hyperparameters={
                    "kernel": ["rbf", "linear", "poly", "sigmoid"],
                    "gamma": ["scale", "auto", 0.1, 0.01],
                    "nu": [0.05, 0.1, 0.2, 0.5],
                }
            )
        ),
        "LocalOutlierFactor": (
            "sklearn.neighbors", "LocalOutlierFactor",
            SklearnAlgorithmInfo(
                category="Proximity",
                complexity_time="O(n²)",
                complexity_space="O(n)",
                supports_streaming=False,
                requires_scaling=True,
                description="Local Outlier Factor for density-based detection",
                hyperparameters={
                    "n_neighbors": [5, 10, 20, 35],
                    "algorithm": ["auto", "ball_tree", "kd_tree", "brute"],
                    "leaf_size": [20, 30, 40],
                    "metric": ["minkowski", "euclidean", "manhattan"],
                    "contamination": [0.1, 0.05, 0.2],
                }
            )
        ),
        "EllipticEnvelope": (
            "sklearn.covariance", "EllipticEnvelope",
            SklearnAlgorithmInfo(
                category="Covariance",
                complexity_time="O(n*p²)",
                complexity_space="O(p²)",
                supports_streaming=False,
                requires_scaling=True,
                description="Elliptic Envelope for Gaussian data",
                hyperparameters={
                    "support_fraction": [None, 0.8, 0.9],
                    "contamination": [0.1, 0.05, 0.2],
                }
            )
        ),
    }
    
    def __init__(
        self,
        algorithm_name: str,
        name: Optional[str] = None,
        contamination_rate: Optional[ContaminationRate] = None,
        use_scaling: Optional[bool] = None,
        **kwargs: Any,
    ):
        """Initialize enhanced sklearn adapter.
        
        Args:
            algorithm_name: Name of the sklearn algorithm
            name: Optional custom name
            contamination_rate: Expected contamination rate
            use_scaling: Whether to apply feature scaling
            **kwargs: Algorithm-specific parameters
        """
        # Validate algorithm
        if algorithm_name not in self.ALGORITHM_MAPPING:
            available = list(self.ALGORITHM_MAPPING.keys())
            raise InvalidAlgorithmError(
                algorithm_name=algorithm_name,
                supported_algorithms=available
            )
        
        self._algorithm_name = algorithm_name
        self._name = name or f"Sklearn_{algorithm_name}"
        self._contamination_rate = contamination_rate or ContaminationRate.auto()
        self._parameters = kwargs
        
        # Load algorithm info
        module_path, class_name, info = self.ALGORITHM_MAPPING[algorithm_name]
        self._algorithm_info = info
        self._model_class = self._load_model_class(module_path, class_name)
        
        # Scaling setup
        self._use_scaling = use_scaling if use_scaling is not None else info.requires_scaling
        self._scaler: Optional[StandardScaler] = None
        
        # State
        self._model: Optional[BaseEstimator] = None
        self._is_fitted = False
        self._feature_names: Optional[List[str]] = None
        self._training_metadata: Dict[str, Any] = {}
        
        # Suppress sklearn warnings
        warnings.filterwarnings("ignore", category=UserWarning)
    
    def _load_model_class(self, module_path: str, class_name: str) -> type:
        """Load sklearn model class dynamically."""
        try:
            module = importlib.import_module(module_path)
            return getattr(module, class_name)
        except (ImportError, AttributeError) as e:
            raise InvalidAlgorithmError(
                algorithm_name=self._algorithm_name,
                supported_algorithms=list(self.ALGORITHM_MAPPING.keys()),
                details=f"Failed to load {module_path}.{class_name}: {e}"
            ) from e
    
    @property
    def name(self) -> str:
        """Get the detector name."""
        return self._name
    
    @property
    def contamination_rate(self) -> ContaminationRate:
        """Get the contamination rate."""
        return self._contamination_rate
    
    @property
    def is_fitted(self) -> bool:
        """Check if the detector has been fitted."""
        return self._is_fitted
    
    @property
    def parameters(self) -> Dict[str, Any]:
        """Get the current parameters."""
        return self._parameters.copy()
    
    @property
    def algorithm_info(self) -> SklearnAlgorithmInfo:
        """Get algorithm information."""
        return self._algorithm_info
    
    def fit(self, dataset: Dataset) -> None:
        """Fit the sklearn detector on a dataset.
        
        Args:
            dataset: Dataset to fit on
            
        Raises:
            FittingError: If fitting fails
        """
        start_time = time.perf_counter()
        
        try:
            # Prepare data
            X, feature_names = self._prepare_features(dataset)
            self._feature_names = feature_names
            
            # Apply scaling if needed
            if self._use_scaling:
                self._scaler = StandardScaler()
                X = self._scaler.fit_transform(X)
            
            # Initialize model
            model_params = self._prepare_model_parameters()
            self._model = self._model_class(**model_params)
            
            # Fit model
            if hasattr(self._model, 'fit_predict'):
                # For algorithms like LocalOutlierFactor that don't separate fit/predict
                self._model.fit(X)
            else:
                self._model.fit(X)
            
            # Update state
            self._is_fitted = True
            training_time = time.perf_counter() - start_time
            
            # Store training metadata
            self._training_metadata = {
                "training_time_seconds": training_time,
                "n_samples": len(X),
                "n_features": X.shape[1],
                "feature_names": feature_names,
                "algorithm": self._algorithm_name,
                "parameters": model_params,
                "scaling_used": self._use_scaling,
                "sklearn_version": self._get_sklearn_version(),
            }
            
        except Exception as e:
            raise FittingError(
                detector_name=self._name,
                reason=str(e),
                dataset_name=dataset.name
            ) from e
    
    def detect(self, dataset: Dataset) -> DetectionResult:
        """Detect anomalies in the dataset.
        
        Args:
            dataset: Dataset to analyze
            
        Returns:
            Detection result with anomalies, scores, and metadata
        """
        if not self._is_fitted:
            raise DetectorNotFittedError(
                detector_name=self._name,
                operation="detect"
            )
        
        start_time = time.perf_counter()
        
        try:
            # Prepare data
            X, _ = self._prepare_features(dataset)
            
            # Apply scaling if used during training
            if self._scaler is not None:
                X = self._scaler.transform(X)
            
            # Get predictions
            predictions = self._model.predict(X)
            
            # Convert sklearn predictions (-1/1) to standard format (0/1)
            labels = np.where(predictions == -1, 1, 0)
            
            # Get anomaly scores
            raw_scores = self._get_anomaly_scores(X)
            normalized_scores = self._normalize_scores(raw_scores)
            
            # Create anomaly score objects
            anomaly_scores = [
                AnomalyScore(value=float(score), method=f"sklearn_{self._algorithm_name}")
                for score in normalized_scores
            ]
            
            # Create anomaly entities
            anomalies = self._create_anomaly_entities(
                dataset, labels, anomaly_scores, raw_scores
            )
            
            # Calculate threshold
            threshold = self._calculate_threshold(normalized_scores)
            
            execution_time = (time.perf_counter() - start_time) * 1000
            
            # Create result
            return DetectionResult(
                detector_id=hash(self._name),
                dataset_id=dataset.id,
                anomalies=anomalies,
                scores=anomaly_scores,
                labels=labels.tolist(),
                threshold=threshold,
                execution_time_ms=execution_time,
                metadata={
                    "algorithm": self._algorithm_name,
                    "category": self._algorithm_info.category,
                    "detection_time_seconds": execution_time / 1000,
                    "n_anomalies": len(anomalies),
                    "contamination_rate": self._contamination_rate.value,
                    **self._training_metadata,
                }
            )
            
        except Exception as e:
            execution_time = (time.perf_counter() - start_time) * 1000
            
            # Return empty result with error
            return DetectionResult(
                detector_id=hash(self._name),
                dataset_id=dataset.id,
                anomalies=[],
                scores=[],
                labels=[],
                threshold=0.5,
                execution_time_ms=execution_time,
                metadata={
                    "algorithm": self._algorithm_name,
                    "error": str(e),
                    "status": "failed"
                }
            )
    
    def score(self, dataset: Dataset) -> List[AnomalyScore]:
        """Calculate anomaly scores for the dataset.
        
        Args:
            dataset: Dataset to score
            
        Returns:
            List of anomaly scores
        """
        if not self._is_fitted:
            raise DetectorNotFittedError(
                detector_name=self._name,
                operation="score"
            )
        
        X, _ = self._prepare_features(dataset)
        
        if self._scaler is not None:
            X = self._scaler.transform(X)
        
        raw_scores = self._get_anomaly_scores(X)
        normalized_scores = self._normalize_scores(raw_scores)
        
        return [
            AnomalyScore(value=float(score), method=f"sklearn_{self._algorithm_name}")
            for score in normalized_scores
        ]
    
    def fit_detect(self, dataset: Dataset) -> DetectionResult:
        """Fit detector and detect anomalies in one step.
        
        Args:
            dataset: Dataset to fit and analyze
            
        Returns:
            Detection result
        """
        self.fit(dataset)
        return self.detect(dataset)
    
    def get_params(self) -> Dict[str, Any]:
        """Get parameters of the detector."""
        if self._model is not None:
            return self._model.get_params()
        return self._parameters
    
    def set_params(self, **params: Any) -> None:
        """Set parameters of the detector."""
        self._parameters.update(params)
        if self._model is not None:
            # Update model parameters
            valid_params = {
                k: v for k, v in params.items() 
                if k in self._model.get_params()
            }
            if valid_params:
                self._model.set_params(**valid_params)
    
    def _prepare_features(self, dataset: Dataset) -> Tuple[np.ndarray, List[str]]:
        """Prepare features for algorithm."""
        # Get numeric features
        numeric_data = dataset.data.select_dtypes(include=[np.number])
        
        if numeric_data.empty:
            raise ValueError("No numeric features found in dataset")
        
        # Handle missing values
        numeric_data = numeric_data.fillna(numeric_data.median())
        
        feature_names = numeric_data.columns.tolist()
        return numeric_data.values, feature_names
    
    def _prepare_model_parameters(self) -> Dict[str, Any]:
        """Prepare parameters for model initialization."""
        params = self._parameters.copy()
        
        # Handle contamination parameter for algorithms that support it
        if self._algorithm_name in ["LocalOutlierFactor", "EllipticEnvelope"]:
            if "contamination" not in params:
                params["contamination"] = self._contamination_rate.value
        
        # Algorithm-specific parameter handling
        if self._algorithm_name == "OneClassSVM":
            # OneClassSVM uses nu instead of contamination
            if "nu" not in params:
                params["nu"] = self._contamination_rate.value
        
        return params
    
    def _get_anomaly_scores(self, X: np.ndarray) -> np.ndarray:
        """Get anomaly scores from the model."""
        if hasattr(self._model, 'decision_function'):
            scores = self._model.decision_function(X)
            # Convert to positive anomaly scores (higher = more anomalous)
            return -scores
        elif hasattr(self._model, 'score_samples'):
            scores = self._model.score_samples(X)
            return -scores
        elif hasattr(self._model, 'negative_outlier_factor_'):
            # For LocalOutlierFactor
            return -self._model.negative_outlier_factor_
        else:
            # Fallback: use predictions as scores
            predictions = self._model.predict(X)
            return np.where(predictions == -1, 1.0, 0.0)
    
    def _normalize_scores(self, scores: np.ndarray) -> np.ndarray:
        """Normalize scores to [0, 1] range."""
        if len(scores) == 0:
            return scores
        
        # Handle constant scores
        min_score = np.min(scores)
        max_score = np.max(scores)
        
        if max_score == min_score:
            return np.full_like(scores, 0.5)
        
        # Normalize to [0, 1]
        normalized = (scores - min_score) / (max_score - min_score)
        return np.clip(normalized, 0.0, 1.0)
    
    def _create_anomaly_entities(
        self,
        dataset: Dataset,
        labels: np.ndarray,
        anomaly_scores: List[AnomalyScore],
        raw_scores: np.ndarray
    ) -> List[Anomaly]:
        """Create anomaly entities for detected anomalies."""
        anomalies = []
        anomaly_indices = np.where(labels == 1)[0]
        
        for idx in anomaly_indices:
            if idx >= len(dataset.data):
                continue
            
            anomaly = Anomaly(
                score=anomaly_scores[idx],
                data_point=dataset.data.iloc[idx].to_dict(),
                detector_name=self._name,
                metadata={
                    "index": int(idx),
                    "raw_score": float(raw_scores[idx]),
                    "algorithm": self._algorithm_name,
                    "category": self._algorithm_info.category,
                    "library": "sklearn",
                }
            )
            
            anomalies.append(anomaly)
        
        return anomalies
    
    def _calculate_threshold(self, scores: np.ndarray) -> float:
        """Calculate anomaly threshold based on contamination rate."""
        if len(scores) == 0:
            return 0.5
        
        threshold_idx = int(len(scores) * (1 - self._contamination_rate.value))
        threshold_idx = max(0, min(threshold_idx, len(scores) - 1))
        
        sorted_scores = np.sort(scores)
        return float(sorted_scores[threshold_idx])
    
    def _get_sklearn_version(self) -> str:
        """Get scikit-learn version."""
        try:
            import sklearn
            return sklearn.__version__
        except (ImportError, AttributeError):
            return "unknown"
    
    @classmethod
    def list_algorithms(cls) -> List[str]:
        """List all available algorithms."""
        return list(cls.ALGORITHM_MAPPING.keys())
    
    @classmethod
    def get_algorithm_info(cls, algorithm_name: str) -> Optional[SklearnAlgorithmInfo]:
        """Get information for a specific algorithm."""
        if algorithm_name in cls.ALGORITHM_MAPPING:
            return cls.ALGORITHM_MAPPING[algorithm_name][2]
        return None
    
    @classmethod
    def get_algorithms_by_category(cls) -> Dict[str, List[str]]:
        """Get algorithms grouped by category."""
        categories = {}
        for name, (_, _, info) in cls.ALGORITHM_MAPPING.items():
            category = info.category
            if category not in categories:
                categories[category] = []
            categories[category].append(name)
        return categories
    
    @classmethod
    def recommend_algorithms(
        cls, 
        n_samples: int,
        n_features: int,
        prefer_interpretable: bool = False
    ) -> List[str]:
        """Recommend algorithms based on dataset characteristics.
        
        Args:
            n_samples: Number of samples
            n_features: Number of features
            prefer_interpretable: Whether to prefer interpretable algorithms
            
        Returns:
            List of recommended algorithm names
        """
        recommendations = []
        
        for name, (_, _, info) in cls.ALGORITHM_MAPPING.items():
            # Small datasets: any algorithm
            if n_samples < 1000:
                recommendations.append(name)
            # Medium datasets: avoid O(n²) for large n
            elif n_samples < 5000:
                if "n²" not in info.complexity_time:
                    recommendations.append(name)
            # Large datasets: only fast algorithms
            else:
                if info.complexity_time in ["O(n log n)", "O(n*p)"]:
                    recommendations.append(name)
        
        # Prioritize interpretable algorithms if requested
        if prefer_interpretable:
            interpretable_order = ["IsolationForest", "EllipticEnvelope", "LocalOutlierFactor", "OneClassSVM"]
            recommendations.sort(key=lambda x: interpretable_order.index(x) if x in interpretable_order else len(interpretable_order))
        
        # Default fallback
        if not recommendations:
            recommendations = ["IsolationForest"]
        
        return recommendations[:3]  # Return top 3 recommendations