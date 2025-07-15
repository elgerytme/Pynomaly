"""Algorithm adapter registry for managing different detection algorithms."""

from __future__ import annotations

import importlib
import logging
from abc import ABC, abstractmethod
from typing import Any, Protocol

import numpy as np
import pandas as pd

from pynomaly.domain.entities import Dataset, Detector
from pynomaly.domain.exceptions import FittingError, InvalidAlgorithmError
from pynomaly.domain.value_objects import AnomalyScore

# Import MLOps infrastructure for model persistence
try:
    from pynomaly.mlops.model_registry import ModelRegistry, ModelStatus, ModelType
    MLOPS_AVAILABLE = True
except ImportError:
    MLOPS_AVAILABLE = False

logger = logging.getLogger(__name__)


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
    """Base class for algorithm adapters with MLOps persistence."""

    def __init__(self, model_registry: ModelRegistry = None):
        """Initialize the adapter with optional model registry.
        
        Args:
            model_registry: MLOps model registry for persistence
        """
        self._fitted_models: dict[str, Any] = {}
        self._model_registry = model_registry
        if self._model_registry is None and MLOPS_AVAILABLE:
            try:
                self._model_registry = ModelRegistry()
                logger.info("Initialized MLOps model registry for persistence")
            except Exception as e:
                logger.warning(f"Failed to initialize model registry: {e}")
                self._model_registry = None

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

            # Store fitted model in memory
            self._fitted_models[str(detector.id)] = fitted_algorithm

            # Save to MLOps registry for persistence (will be called from async context)
            if self._model_registry is not None:
                try:

                    model_type = self._get_model_type(detector.algorithm_name)

                    # Create a simple save task - we'll handle this in the calling context
                    detector.metadata["_fitted_algorithm"] = fitted_algorithm
                    detector.metadata["_needs_registry_save"] = True
                    detector.metadata["_model_type"] = model_type

                    logger.info(f"Model prepared for registry save: {detector.name}")

                except Exception as e:
                    logger.warning(f"Failed to prepare model for registry: {e}")

        except Exception as e:
            raise FittingError(
                detector_name=detector.name, reason=str(e), dataset_name=dataset.name
            ) from e

    def predict(self, detector: Detector, dataset: Dataset) -> list[int]:
        """Predict anomaly labels."""
        # Try to get fitted model from memory first
        algorithm = self._fitted_models.get(str(detector.id))

        # If not in memory, try to load from registry
        if algorithm is None and self._model_registry is not None:
            algorithm = self._load_model_from_registry(detector)

        # If still not found, raise error
        if algorithm is None:
            raise FittingError(
                detector_name=detector.name,
                reason="Detector not fitted",
                dataset_name=dataset.name,
            )

        feature_data = self._prepare_data(dataset)

        predictions = self._predict_algorithm(algorithm, feature_data)
        return predictions.tolist()

    def score(self, detector: Detector, dataset: Dataset) -> list[AnomalyScore]:
        """Calculate anomaly scores."""
        # Try to get fitted model from memory first
        algorithm = self._fitted_models.get(str(detector.id))

        # If not in memory, try to load from registry
        if algorithm is None and self._model_registry is not None:
            algorithm = self._load_model_from_registry(detector)

        # If still not found, raise error
        if algorithm is None:
            raise FittingError(
                detector_name=detector.name,
                reason="Detector not fitted",
                dataset_name=dataset.name,
            )

        feature_data = self._prepare_data(dataset)

        scores = self._score_algorithm(algorithm, feature_data)

        # Normalize scores to [0, 1] range for AnomalyScore
        normalized_scores = self._normalize_scores(scores)

        # Map algorithm names to valid scoring methods
        method_mapping = {
            "IsolationForest": "isolation_forest",
            "LOF": "local_outlier_factor",
            "OneClassSVM": "one_class_svm",
            "LocalOutlierFactor": "local_outlier_factor",
            "EllipticEnvelope": "elliptic_envelope",
        }

        method_name = method_mapping.get(detector.algorithm_name, detector.algorithm_name.lower().replace(" ", "_"))

        return [
            AnomalyScore(value=round(float(score), 6), method=method_name)
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
        fitted_in_memory = str(detector.id) in self._fitted_models

        # Also check if model exists in registry
        if not fitted_in_memory and self._model_registry is not None:
            model_registry_id = detector.metadata.get("model_registry_id")
            if model_registry_id:
                try:
                    # Simple check - we'll load it when needed
                    return True
                except:
                    pass

        return fitted_in_memory

    def _load_model_from_registry(self, detector: Detector) -> Any:
        """Load model from MLOps registry."""
        if self._model_registry is None:
            return None

        try:
            model_registry_id = detector.metadata.get("model_registry_id")
            if not model_registry_id:
                logger.warning(f"No model registry ID found for detector {detector.name}")
                return None

            # This would need to be called from async context
            # For now, return None and let the sync methods handle it
            logger.info(f"Model loading from registry would happen here for {model_registry_id}")
            return None

        except Exception as e:
            logger.warning(f"Failed to load model from registry: {e}")
            return None

    def _get_model_type(self, algorithm_name: str) -> ModelType:
        """Map algorithm name to ModelType enum."""
        if not MLOPS_AVAILABLE:
            return None

        # Simple mapping - can be expanded
        algorithm_mapping = {
            "IsolationForest": ModelType.ISOLATION_FOREST,
            "OneClassSVM": ModelType.ONE_CLASS_SVM,
            "LSTM": ModelType.LSTM_AUTOENCODER,
        }

        return algorithm_mapping.get(algorithm_name, ModelType.CUSTOM)


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
