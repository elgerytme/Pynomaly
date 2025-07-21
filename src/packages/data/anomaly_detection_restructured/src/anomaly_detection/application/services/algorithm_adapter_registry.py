"""Algorithm adapter registry for managing different detection algorithms."""

from __future__ import annotations

import importlib
from abc import ABC, abstractmethod
from typing import Any, Protocol

import numpy as np
import pandas as pd

# TODO: Create local Dataset entity, Detector
from monorepo.domain.exceptions import FittingError, InvalidAlgorithmError
from monorepo.domain.value_objects import AnomalyScore


class AlgorithmAdapter(Protocol):
    """Protocol for algorithm adapters."""

    def fit(self, detector: Detector, dataset: Dataset) -> None:
        """Fit the algorithm on training data."""
        ...

    def predict(self, detector: Detector, dataset: Dataset) -> list[int]:
        """Predict anomaly labels (0=normal, 1=anomaly)."""
        ...

    def score(self, detector: Detector, dataset: Dataset) -> list[AnomalyScore]:
        """Calculate anomaly scores."""
        ...

    def get_algorithm_info(self) -> dict[str, Any]:
        """Get information about the algorithm."""
        ...


class BaseAlgorithmAdapter(ABC):
    """Base class for algorithm adapters."""

    def __init__(self):
        """Initialize the adapter."""
        self._fitted_models: dict[str, Any] = {}

    @abstractmethod
    def _create_algorithm_instance(self, detector: Detector) -> Any:
        """Create an instance of the underlying algorithm."""
        pass

    @abstractmethod
    def _fit_algorithm(self, algorithm: Any, data: pd.DataFrame) -> Any:
        """Fit the algorithm on data."""
        pass

    @abstractmethod
    def _predict_algorithm(self, algorithm: Any, data: pd.DataFrame) -> np.ndarray:
        """Predict with the algorithm."""
        pass

    @abstractmethod
    def _score_algorithm(self, algorithm: Any, data: pd.DataFrame) -> np.ndarray:
        """Score with the algorithm."""
        pass

    def fit(self, detector: Detector, dataset: Dataset) -> None:
        """Fit the algorithm on training data."""
        try:
            # Create algorithm instance
            algorithm = self._create_algorithm_instance(detector)

            # Prepare data
            feature_data = self._prepare_data(dataset)

            # Fit algorithm
            fitted_algorithm = self._fit_algorithm(algorithm, feature_data)

            # Store fitted model
            self._fitted_models[str(detector.id)] = fitted_algorithm

        except Exception as e:
            raise FittingError(
                detector_name=detector.name, reason=str(e), dataset_name=dataset.name
            ) from e

    def predict(self, detector: Detector, dataset: Dataset) -> list[int]:
        """Predict anomaly labels."""
        if str(detector.id) not in self._fitted_models:
            raise FittingError(
                detector_name=detector.name,
                reason="Detector not fitted",
                dataset_name=dataset.name,
            )

        algorithm = self._fitted_models[str(detector.id)]
        feature_data = self._prepare_data(dataset)

        predictions = self._predict_algorithm(algorithm, feature_data)
        return predictions.tolist()

    def score(self, detector: Detector, dataset: Dataset) -> list[AnomalyScore]:
        """Calculate anomaly scores."""
        if str(detector.id) not in self._fitted_models:
            raise FittingError(
                detector_name=detector.name,
                reason="Detector not fitted",
                dataset_name=dataset.name,
            )

        algorithm = self._fitted_models[str(detector.id)]
        feature_data = self._prepare_data(dataset)

        scores = self._score_algorithm(algorithm, feature_data)

        # Normalize scores to [0, 1] range for AnomalyScore
        normalized_scores = self._normalize_scores(scores)

        return [
            AnomalyScore(value=float(score), method=detector.algorithm_name)
            for score in normalized_scores
        ]

    def _prepare_data(self, dataset: Dataset) -> pd.DataFrame:
        """Prepare data for algorithm."""
        # Get numeric features only
        numeric_data = dataset.data.select_dtypes(include=[np.number])

        if numeric_data.empty:
            raise ValueError("No numeric features found in dataset")

        # Handle missing values
        numeric_data = numeric_data.fillna(numeric_data.mean())

        return numeric_data

    def _normalize_scores(self, scores: np.ndarray) -> np.ndarray:
        """Normalize scores to [0, 1] range."""
        min_score = np.min(scores)
        max_score = np.max(scores)

        # Avoid division by zero
        if max_score == min_score:
            return np.ones_like(scores) * 0.5

        # Normalize to [0, 1]
        normalized = (scores - min_score) / (max_score - min_score)

        # Ensure values are within [0, 1]
        return np.clip(normalized, 0.0, 1.0)

    def is_fitted(self, detector: Detector) -> bool:
        """Check if detector is fitted."""
        return str(detector.id) in self._fitted_models


class PyODAlgorithmAdapter(BaseAlgorithmAdapter):
    """Adapter for PyOD algorithms."""

    # Algorithm mapping
    ALGORITHM_MAPPING = {
        "IsolationForest": ("pyod.models.iforest", "IForest"),
        "LOF": ("pyod.models.lof", "LOF"),
        "OneClassSVM": ("pyod.models.ocsvm", "OCSVM"),
        "ABOD": ("pyod.models.abod", "ABOD"),
        "CBLOF": ("pyod.models.cblof", "CBLOF"),
        "HBOS": ("pyod.models.hbos", "HBOS"),
        "KNN": ("pyod.models.knn", "KNN"),
        "PCA": ("pyod.models.pca", "PCA"),
        "MCD": ("pyod.models.mcd", "MCD"),
        "COPOD": ("pyod.models.copod", "COPOD"),
    }

    def _create_algorithm_instance(self, detector: Detector) -> Any:
        """Create PyOD algorithm instance."""
        algorithm_name = detector.algorithm_name

        if algorithm_name not in self.ALGORITHM_MAPPING:
            raise InvalidAlgorithmError(
                algorithm_name=algorithm_name,
                supported_algorithms=list(self.ALGORITHM_MAPPING.keys()),
            )

        module_name, class_name = self.ALGORITHM_MAPPING[algorithm_name]

        try:
            # Import the module
            module = importlib.import_module(module_name)
            algorithm_class = getattr(module, class_name)

            # Create instance with parameters
            params = detector.parameters.copy()

            # Handle contamination parameter
            if "contamination" not in params:
                params["contamination"] = detector.contamination_rate.value

            return algorithm_class(**params)

        except ImportError as e:
            raise InvalidAlgorithmError(
                algorithm_name=algorithm_name,
                supported_algorithms=list(self.ALGORITHM_MAPPING.keys()),
                details=f"Failed to import {module_name}: {e}",
            ) from e
        except Exception as e:
            raise InvalidAlgorithmError(
                algorithm_name=algorithm_name,
                supported_algorithms=list(self.ALGORITHM_MAPPING.keys()),
                details=f"Failed to create algorithm: {e}",
            ) from e

    def _fit_algorithm(self, algorithm: Any, data: pd.DataFrame) -> Any:
        """Fit PyOD algorithm."""
        algorithm.fit(data.values)
        return algorithm

    def _predict_algorithm(self, algorithm: Any, data: pd.DataFrame) -> np.ndarray:
        """Predict with PyOD algorithm."""
        return algorithm.predict(data.values)

    def _score_algorithm(self, algorithm: Any, data: pd.DataFrame) -> np.ndarray:
        """Score with PyOD algorithm."""
        return algorithm.decision_function(data.values)

    def get_algorithm_info(self) -> dict[str, Any]:
        """Get PyOD algorithm information."""
        return {
            "name": "PyOD",
            "description": "Python Outlier Detection Library",
            "supported_algorithms": list(self.ALGORITHM_MAPPING.keys()),
            "type": "unsupervised",
            "supports_streaming": False,
            "supports_multivariate": True,
        }


class SklearnAlgorithmAdapter(BaseAlgorithmAdapter):
    """Adapter for scikit-learn algorithms."""

    ALGORITHM_MAPPING = {
        "IsolationForest": ("sklearn.ensemble", "IsolationForest"),
        "OneClassSVM": ("sklearn.svm", "OneClassSVM"),
        "LocalOutlierFactor": ("sklearn.neighbors", "LocalOutlierFactor"),
        "EllipticEnvelope": ("sklearn.covariance", "EllipticEnvelope"),
    }

    def _create_algorithm_instance(self, detector: Detector) -> Any:
        """Create sklearn algorithm instance."""
        algorithm_name = detector.algorithm_name

        if algorithm_name not in self.ALGORITHM_MAPPING:
            raise InvalidAlgorithmError(
                algorithm_name=algorithm_name,
                supported_algorithms=list(self.ALGORITHM_MAPPING.keys()),
            )

        module_name, class_name = self.ALGORITHM_MAPPING[algorithm_name]

        try:
            module = importlib.import_module(module_name)
            algorithm_class = getattr(module, class_name)

            params = detector.parameters.copy()
            return algorithm_class(**params)

        except Exception as e:
            raise InvalidAlgorithmError(
                algorithm_name=algorithm_name,
                supported_algorithms=list(self.ALGORITHM_MAPPING.keys()),
                details=str(e),
            ) from e

    def _fit_algorithm(self, algorithm: Any, data: pd.DataFrame) -> Any:
        """Fit sklearn algorithm."""
        algorithm.fit(data.values)
        return algorithm

    def _predict_algorithm(self, algorithm: Any, data: pd.DataFrame) -> np.ndarray:
        """Predict with sklearn algorithm."""
        predictions = algorithm.predict(data.values)
        # Convert to 0/1 labels (sklearn returns -1/1)
        return np.where(predictions == -1, 1, 0)

    def _score_algorithm(self, algorithm: Any, data: pd.DataFrame) -> np.ndarray:
        """Score with sklearn algorithm."""
        if hasattr(algorithm, "decision_function"):
            scores = algorithm.decision_function(data.values)
            # Convert to positive anomaly scores
            return -scores
        elif hasattr(algorithm, "score_samples"):
            scores = algorithm.score_samples(data.values)
            return -scores
        else:
            # Fallback: use negative log likelihood
            return np.ones(len(data))


class AlgorithmAdapterRegistry:
    """Registry for managing algorithm adapters."""

    def __init__(self):
        """Initialize the registry."""
        self._adapters: dict[str, AlgorithmAdapter] = {}
        self._algorithm_map: dict[str, str] = {}

        # Register default adapters
        self._register_default_adapters()

    def _register_default_adapters(self) -> None:
        """Register default algorithm adapters."""
        # Register PyOD adapter
        pyod_adapter = PyODAlgorithmAdapter()
        self.register_adapter("pyod", pyod_adapter)

        # Map PyOD algorithms
        for algorithm in PyODAlgorithmAdapter.ALGORITHM_MAPPING.keys():
            self._algorithm_map[algorithm] = "pyod"

        # Register sklearn adapter
        sklearn_adapter = SklearnAlgorithmAdapter()
        self.register_adapter("sklearn", sklearn_adapter)

        # Map sklearn algorithms (with sklearn prefix to avoid conflicts)
        for algorithm in SklearnAlgorithmAdapter.ALGORITHM_MAPPING.keys():
            sklearn_name = f"sklearn_{algorithm}"
            self._algorithm_map[sklearn_name] = "sklearn"

    def register_adapter(self, name: str, adapter: AlgorithmAdapter) -> None:
        """Register an algorithm adapter."""
        self._adapters[name] = adapter

    def get_adapter_for_algorithm(self, algorithm_name: str) -> AlgorithmAdapter | None:
        """Get the appropriate adapter for an algorithm."""
        adapter_name = self._algorithm_map.get(algorithm_name)
        if adapter_name:
            return self._adapters.get(adapter_name)
        return None

    def get_supported_algorithms(self) -> list[str]:
        """Get list of all supported algorithms."""
        return list(self._algorithm_map.keys())

    def get_adapter_info(self, adapter_name: str) -> dict[str, Any] | None:
        """Get information about an adapter."""
        adapter = self._adapters.get(adapter_name)
        if adapter:
            return adapter.get_algorithm_info()
        return None

    def fit_detector(self, detector: Detector, dataset: Dataset) -> None:
        """Fit a detector using the appropriate adapter."""
        adapter = self.get_adapter_for_algorithm(detector.algorithm_name)
        if not adapter:
            raise InvalidAlgorithmError(
                algorithm_name=detector.algorithm_name,
                supported_algorithms=self.get_supported_algorithms(),
            )

        adapter.fit(detector, dataset)

    def predict_with_detector(self, detector: Detector, dataset: Dataset) -> list[int]:
        """Predict with a detector using the appropriate adapter."""
        adapter = self.get_adapter_for_algorithm(detector.algorithm_name)
        if not adapter:
            raise InvalidAlgorithmError(
                algorithm_name=detector.algorithm_name,
                supported_algorithms=self.get_supported_algorithms(),
            )

        return adapter.predict(detector, dataset)

    def score_with_detector(
        self, detector: Detector, dataset: Dataset
    ) -> list[AnomalyScore]:
        """Score with a detector using the appropriate adapter."""
        adapter = self.get_adapter_for_algorithm(detector.algorithm_name)
        if not adapter:
            raise InvalidAlgorithmError(
                algorithm_name=detector.algorithm_name,
                supported_algorithms=self.get_supported_algorithms(),
            )

        return adapter.score(detector, dataset)
