"""Enhanced PyOD service adapter following clean architecture principles."""

from __future__ import annotations

import logging
import time
from typing import Any

import numpy as np
import pandas as pd

from pynomaly_detection.domain.entities.anomaly import Anomaly
from pynomaly_detection.domain.entities.dataset import Dataset
from pynomaly_detection.domain.entities.detector import Detector
from pynomaly_detection.shared.protocols.detector_protocol import DetectorProtocol

logger = logging.getLogger(__name__)


class PyODAlgorithmRegistry:
    """Registry for PyOD algorithms with metadata."""

    ALGORITHMS = {
        # Linear models
        "PCA": {
            "module": "pyod.models.pca",
            "class": "PCA",
            "category": "linear",
            "complexity": "O(n*p²)",
            "description": "Principal Component Analysis outlier detection",
            "supports_streaming": False,
        },
        "MCD": {
            "module": "pyod.models.mcd",
            "class": "MCD",
            "category": "linear",
            "complexity": "O(n²)",
            "description": "Minimum Covariance Determinant",
            "supports_streaming": False,
        },
        "OCSVM": {
            "module": "pyod.models.ocsvm",
            "class": "OCSVM",
            "category": "linear",
            "complexity": "O(n²)",
            "description": "One-Class Support Vector Machine",
            "supports_streaming": False,
        },
        # Proximity-based
        "IsolationForest": {
            "module": "pyod.models.iforest",
            "class": "IForest",
            "category": "ensemble",
            "complexity": "O(n log n)",
            "description": "Isolation Forest anomaly detection",
            "supports_streaming": False,
        },
        "LOF": {
            "module": "pyod.models.lof",
            "class": "LOF",
            "category": "proximity",
            "complexity": "O(n²)",
            "description": "Local Outlier Factor",
            "supports_streaming": True,
        },
        "KNN": {
            "module": "pyod.models.knn",
            "class": "KNN",
            "category": "proximity",
            "complexity": "O(n log n)",
            "description": "k-Nearest Neighbors outlier detection",
            "supports_streaming": True,
        },
        "CBLOF": {
            "module": "pyod.models.cblof",
            "class": "CBLOF",
            "category": "proximity",
            "complexity": "O(n²)",
            "description": "Clustering-Based Local Outlier Factor",
            "supports_streaming": False,
        },
        "HBOS": {
            "module": "pyod.models.hbos",
            "class": "HBOS",
            "category": "proximity",
            "complexity": "O(n*p)",
            "description": "Histogram-Based Outlier Score",
            "supports_streaming": False,
        },
        # Probabilistic
        "ABOD": {
            "module": "pyod.models.abod",
            "class": "ABOD",
            "category": "probabilistic",
            "complexity": "O(n³)",
            "description": "Angle-Based Outlier Detection",
            "supports_streaming": False,
        },
        "COPOD": {
            "module": "pyod.models.copod",
            "class": "COPOD",
            "category": "probabilistic",
            "complexity": "O(n*p)",
            "description": "Copula-Based Outlier Detection",
            "supports_streaming": False,
        },
        # Neural networks
        "AutoEncoder": {
            "module": "pyod.models.auto_encoder",
            "class": "AutoEncoder",
            "category": "neural_network",
            "complexity": "O(n*epochs)",
            "description": "AutoEncoder-based anomaly detection",
            "supports_streaming": False,
        },
    }

    @classmethod
    def get_algorithm_info(cls, algorithm: str) -> dict[str, Any]:
        """Get algorithm information."""
        if algorithm not in cls.ALGORITHMS:
            raise ValueError(f"Algorithm {algorithm} not supported")
        return cls.ALGORITHMS[algorithm].copy()

    @classmethod
    def get_supported_algorithms(cls) -> list[str]:
        """Get list of supported algorithms."""
        return list(cls.ALGORITHMS.keys())

    @classmethod
    def load_algorithm_class(cls, algorithm: str) -> type[Any]:
        """Dynamically load PyOD algorithm class."""
        if algorithm not in cls.ALGORITHMS:
            raise ValueError(f"Algorithm {algorithm} not supported")

        algo_info = cls.ALGORITHMS[algorithm]

        try:
            import importlib

            module = importlib.import_module(algo_info["module"])
            return getattr(module, algo_info["class"])
        except ImportError as e:
            raise ImportError(
                f"PyOD is required for {algorithm}. Install with: pip install pyod"
            ) from e
        except AttributeError as e:
            raise AttributeError(
                f"Algorithm class {algo_info['class']} not found in {algo_info['module']}"
            ) from e


class PyODService:
    """Service for PyOD algorithm operations."""

    def __init__(self) -> None:
        """Initialize PyOD service."""
        self._registry = PyODAlgorithmRegistry()

    def create_model(
        self, algorithm: str, parameters: dict[str, Any] | None = None
    ) -> Any:
        """Create PyOD model instance."""
        parameters = parameters or {}

        # Get algorithm class
        algorithm_class = self._registry.load_algorithm_class(algorithm)

        # Set default parameters
        default_params = self._get_default_parameters(algorithm)
        default_params.update(parameters)

        logger.info(f"Creating PyOD {algorithm} with parameters: {default_params}")

        try:
            return algorithm_class(**default_params)
        except Exception as e:
            logger.error(f"Failed to create PyOD model {algorithm}: {e}")
            raise RuntimeError(f"Failed to create {algorithm} model: {e}") from e

    def train_model(self, model: Any, data: pd.DataFrame) -> dict[str, Any]:
        """Train PyOD model and return metadata."""
        if data.empty:
            raise ValueError("Training data cannot be empty")

        # Prepare data
        X = self._prepare_data(data)

        # Train model
        start_time = time.time()
        try:
            model.fit(X)
            training_time = time.time() - start_time

            logger.info(f"PyOD model trained successfully in {training_time:.3f}s")

            return {
                "training_time": training_time,
                "training_samples": X.shape[0],
                "training_features": X.shape[1],
                "data_shape": X.shape,
            }

        except Exception as e:
            logger.error(f"PyOD model training failed: {e}")
            raise RuntimeError(f"Model training failed: {e}") from e

    def predict_anomalies(
        self, model: Any, data: pd.DataFrame, threshold: float = 0.5
    ) -> list[Anomaly]:
        """Predict anomalies using trained PyOD model."""
        if data.empty:
            return []

        # Prepare data
        X = self._prepare_data(data)

        try:
            # Get predictions and scores
            predictions = model.predict(X)  # 0 = normal, 1 = anomaly
            scores = model.decision_function(X)  # Raw anomaly scores

            # Normalize scores to [0,1] range
            normalized_scores = self._normalize_scores(scores)

            # Create anomaly objects for detected anomalies
            anomalies = []
            anomaly_indices = np.where(predictions == 1)[0]

            for idx in anomaly_indices:
                # Get original data row
                data_row = data.iloc[idx].to_dict()

                anomaly = Anomaly(
                    data_point=data_row,
                    anomaly_score=float(normalized_scores[idx]),
                    confidence=min(float(normalized_scores[idx]), 1.0),
                    feature_contributions={},  # To be populated by explainability
                    metadata={
                        "algorithm": "PyOD",
                        "raw_score": float(scores[idx]),
                        "prediction": int(predictions[idx]),
                        "row_index": idx,
                        "threshold": threshold,
                    },
                )
                anomalies.append(anomaly)

            logger.info(
                f"PyOD prediction completed: {len(anomalies)} anomalies from {len(data)} samples"
            )
            return anomalies

        except Exception as e:
            logger.error(f"PyOD prediction failed: {e}")
            raise RuntimeError(f"Prediction failed: {e}") from e

    def get_anomaly_scores(self, model: Any, data: pd.DataFrame) -> np.ndarray:
        """Get anomaly scores for all data points."""
        if data.empty:
            return np.array([])

        X = self._prepare_data(data)

        try:
            scores = model.decision_function(X)
            return self._normalize_scores(scores)
        except Exception as e:
            logger.error(f"Failed to get anomaly scores: {e}")
            raise RuntimeError(f"Failed to get scores: {e}") from e

    def get_supported_algorithms(self) -> list[str]:
        """Get list of supported algorithms."""
        return self._registry.get_supported_algorithms()

    def get_algorithm_info(self, algorithm: str) -> dict[str, Any]:
        """Get algorithm metadata."""
        return self._registry.get_algorithm_info(algorithm)

    def _prepare_data(self, data: pd.DataFrame) -> np.ndarray:
        """Prepare data for PyOD processing."""
        # Select only numeric columns
        numeric_data = data.select_dtypes(include=[np.number])

        if numeric_data.empty:
            raise ValueError("No numeric columns found in data")

        # Convert to numpy array
        X = numeric_data.values

        # Handle NaN values
        if np.isnan(X).any():
            logger.warning("Data contains NaN values, filling with column means")
            # Simple NaN handling - fill with column means
            col_means = np.nanmean(X, axis=0)
            nan_mask = np.isnan(X)
            X[nan_mask] = np.take(col_means, np.where(nan_mask)[1])

        return X

    def _normalize_scores(self, scores: np.ndarray) -> np.ndarray:
        """Normalize scores to [0,1] range."""
        min_score = scores.min()
        max_score = scores.max()

        if max_score > min_score:
            return (scores - min_score) / (max_score - min_score)
        else:
            return np.ones_like(scores) * 0.5

    def _get_default_parameters(self, algorithm: str) -> dict[str, Any]:
        """Get default parameters for algorithm."""
        defaults = {
            "IsolationForest": {
                "n_estimators": 100,
                "contamination": 0.1,
                "random_state": 42,
            },
            "LOF": {
                "n_neighbors": 20,
                "contamination": 0.1,
            },
            "KNN": {
                "n_neighbors": 5,
                "contamination": 0.1,
                "method": "largest",
            },
            "OCSVM": {
                "kernel": "rbf",
                "gamma": "scale",
                "nu": 0.5,
                "contamination": 0.1,
            },
            "PCA": {
                "n_components": None,
                "contamination": 0.1,
            },
            "HBOS": {
                "n_bins": 10,
                "alpha": 0.1,
                "contamination": 0.1,
            },
            "CBLOF": {
                "n_clusters": 8,
                "contamination": 0.1,
                "alpha": 0.9,
                "beta": 5,
            },
            "ABOD": {
                "contamination": 0.1,
                "n_neighbors": 10,
            },
            "COPOD": {
                "contamination": 0.1,
            },
            "AutoEncoder": {
                "hidden_neurons": [64, 32, 32, 64],
                "contamination": 0.1,
                "epochs": 100,
                "batch_size": 32,
            },
        }

        return defaults.get(algorithm, {"contamination": 0.1})


class PyODDetectorAdapter(DetectorProtocol):
    """Detector adapter using PyOD service."""

    def __init__(self, service: PyODService | None = None) -> None:
        """Initialize adapter with PyOD service."""
        self._service = service or PyODService()
        self._model: Any | None = None
        self._is_fitted = False
        self._algorithm = ""

    def train(self, detector: Detector, dataset: Dataset) -> None:
        """Train detector using PyOD."""
        if not dataset.data or dataset.data.empty:
            raise ValueError("Dataset cannot be empty for training")

        # Store algorithm for reference
        self._algorithm = detector.algorithm

        # Create model
        self._model = self._service.create_model(
            detector.algorithm, detector.parameters
        )

        # Train model
        training_metadata = self._service.train_model(self._model, dataset.data)

        # Update detector
        detector.is_fitted = True
        detector.fitted_model = self._model
        detector.metadata.update(training_metadata)
        detector.metadata.update(
            {
                "algorithm_info": self._service.get_algorithm_info(detector.algorithm),
                "adapter": "PyOD",
            }
        )

        self._is_fitted = True

        logger.info(f"Detector {detector.name} trained with {detector.algorithm}")

    def predict(self, detector: Detector, dataset: Dataset) -> list[Anomaly]:
        """Predict anomalies using trained detector."""
        if not self._is_fitted or self._model is None:
            raise ValueError("Detector must be trained before prediction")

        if not dataset.data or dataset.data.empty:
            return []

        # Get threshold from detector parameters or use default
        threshold = detector.parameters.get("contamination", 0.1)

        return self._service.predict_anomalies(self._model, dataset.data, threshold)

    def get_anomaly_scores(self, detector: Detector, dataset: Dataset) -> np.ndarray:
        """Get anomaly scores for all data points."""
        if not self._is_fitted or self._model is None:
            raise ValueError("Detector must be trained before getting scores")

        if not dataset.data or dataset.data.empty:
            return np.array([])

        return self._service.get_anomaly_scores(self._model, dataset.data)

    def supports_algorithm(self, algorithm: str) -> bool:
        """Check if algorithm is supported."""
        return algorithm in self._service.get_supported_algorithms()

    def get_supported_algorithms(self) -> list[str]:
        """Get list of supported algorithms."""
        return self._service.get_supported_algorithms()

    def get_algorithm_info(self, algorithm: str) -> dict[str, Any]:
        """Get algorithm information."""
        return self._service.get_algorithm_info(algorithm)
