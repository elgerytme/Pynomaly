"""Enhanced PyOD adapter with comprehensive algorithm support and optimizations."""

from __future__ import annotations

import importlib
import time
import warnings
from dataclasses import dataclass
from typing import Any

import numpy as np

from pynomaly.domain.entities import Anomaly, Dataset, DetectionResult
from pynomaly.domain.exceptions import (
    DetectorNotFittedError,
    FittingError,
    InvalidAlgorithmError,
)
from pynomaly.domain.value_objects import AnomalyScore, ContaminationRate
from pynomaly.shared.protocols.detector_protocol import (
    ExplainableDetectorProtocol,
)


@dataclass
class AlgorithmMetadata:
    """Metadata for PyOD algorithms."""

    category: str
    complexity_time: str
    complexity_space: str
    supports_streaming: bool
    supports_multivariate: bool
    requires_gpu: bool
    description: str
    paper_reference: str | None = None
    typical_use_cases: list[str] | None = None


class EnhancedPyODAdapter(ExplainableDetectorProtocol):
    """Enhanced PyOD adapter with comprehensive algorithm support."""

    # Comprehensive algorithm mapping with metadata
    ALGORITHM_MAPPING: dict[str, tuple[str, str, AlgorithmMetadata]] = {
        # Linear Models
        "PCA": (
            "pyod.models.pca",
            "PCA",
            AlgorithmMetadata(
                category="Linear",
                complexity_time="O(n*p²)",
                complexity_space="O(p²)",
                supports_streaming=False,
                supports_multivariate=True,
                requires_gpu=False,
                description="Principal Component Analysis for outlier detection",
                paper_reference="Shyu et al. (2003)",
                typical_use_cases=["Dimensionality reduction", "Noise detection"],
            ),
        ),
        "MCD": (
            "pyod.models.mcd",
            "MCD",
            AlgorithmMetadata(
                category="Linear",
                complexity_time="O(n*p²)",
                complexity_space="O(p²)",
                supports_streaming=False,
                supports_multivariate=True,
                requires_gpu=False,
                description="Minimum Covariance Determinant for robust covariance estimation",
                paper_reference="Rousseeuw & Driessen (1999)",
                typical_use_cases=["Robust statistics", "Financial fraud detection"],
            ),
        ),
        "OCSVM": (
            "pyod.models.ocsvm",
            "OCSVM",
            AlgorithmMetadata(
                category="Linear",
                complexity_time="O(n²)",
                complexity_space="O(n²)",
                supports_streaming=False,
                supports_multivariate=True,
                requires_gpu=False,
                description="One-Class Support Vector Machine",
                paper_reference="Schölkopf et al. (2001)",
                typical_use_cases=["Complex boundaries", "High-dimensional data"],
            ),
        ),
        # Proximity-Based Models
        "LOF": (
            "pyod.models.lof",
            "LOF",
            AlgorithmMetadata(
                category="Proximity",
                complexity_time="O(n²)",
                complexity_space="O(n)",
                supports_streaming=True,
                supports_multivariate=True,
                requires_gpu=False,
                description="Local Outlier Factor for density-based outlier detection",
                paper_reference="Breunig et al. (2000)",
                typical_use_cases=["Density-based detection", "Local anomalies"],
            ),
        ),
        "KNN": (
            "pyod.models.knn",
            "KNN",
            AlgorithmMetadata(
                category="Proximity",
                complexity_time="O(n log n)",
                complexity_space="O(n)",
                supports_streaming=True,
                supports_multivariate=True,
                requires_gpu=False,
                description="k-Nearest Neighbors outlier detection",
                typical_use_cases=["Simple baseline", "Distance-based detection"],
            ),
        ),
        "COF": (
            "pyod.models.cof",
            "COF",
            AlgorithmMetadata(
                category="Proximity",
                complexity_time="O(n²)",
                complexity_space="O(n)",
                supports_streaming=False,
                supports_multivariate=True,
                requires_gpu=False,
                description="Connectivity-Based Outlier Factor",
                paper_reference="Tang et al. (2002)",
                typical_use_cases=["Path-based detection", "Chain-like clusters"],
            ),
        ),
        "CBLOF": (
            "pyod.models.cblof",
            "CBLOF",
            AlgorithmMetadata(
                category="Proximity",
                complexity_time="O(n log n)",
                complexity_space="O(n)",
                supports_streaming=False,
                supports_multivariate=True,
                requires_gpu=False,
                description="Clustering-Based Local Outlier Factor",
                paper_reference="He et al. (2003)",
                typical_use_cases=["Cluster-based detection", "Mixed data"],
            ),
        ),
        "HBOS": (
            "pyod.models.hbos",
            "HBOS",
            AlgorithmMetadata(
                category="Proximity",
                complexity_time="O(n*p)",
                complexity_space="O(p)",
                supports_streaming=False,
                supports_multivariate=True,
                requires_gpu=False,
                description="Histogram-Based Outlier Score",
                paper_reference="Goldstein & Dengel (2012)",
                typical_use_cases=["Fast detection", "Feature-wise analysis"],
            ),
        ),
        # Ensemble Methods
        "IsolationForest": (
            "pyod.models.iforest",
            "IForest",
            AlgorithmMetadata(
                category="Ensemble",
                complexity_time="O(n log n)",
                complexity_space="O(n)",
                supports_streaming=False,
                supports_multivariate=True,
                requires_gpu=False,
                description="Isolation Forest for path-based outlier detection",
                paper_reference="Liu et al. (2008)",
                typical_use_cases=[
                    "General purpose",
                    "Large datasets",
                    "Mixed data types",
                ],
            ),
        ),
        "FeatureBagging": (
            "pyod.models.feature_bagging",
            "FeatureBagging",
            AlgorithmMetadata(
                category="Ensemble",
                complexity_time="O(n*p)",
                complexity_space="O(n*p)",
                supports_streaming=False,
                supports_multivariate=True,
                requires_gpu=False,
                description="Feature Bagging for ensemble outlier detection",
                paper_reference="Lazarevic & Kumar (2005)",
                typical_use_cases=["High-dimensional data", "Feature selection"],
            ),
        ),
        "LSCP": (
            "pyod.models.lscp",
            "LSCP",
            AlgorithmMetadata(
                category="Ensemble",
                complexity_time="O(n*k)",
                complexity_space="O(n*k)",
                supports_streaming=False,
                supports_multivariate=True,
                requires_gpu=False,
                description="Locally Selective Combination of Parallel Outlier Ensembles",
                paper_reference="Zhao et al. (2019)",
                typical_use_cases=["Model combination", "Robust ensembles"],
            ),
        ),
        "SUOD": (
            "pyod.models.suod",
            "SUOD",
            AlgorithmMetadata(
                category="Ensemble",
                complexity_time="O(n)",
                complexity_space="O(n)",
                supports_streaming=False,
                supports_multivariate=True,
                requires_gpu=False,
                description="Scalable Unsupervised Outlier Detection",
                paper_reference="Zhao et al. (2021)",
                typical_use_cases=["Large datasets", "Parallel processing"],
            ),
        ),
        # Neural Networks & Deep Learning
        "AutoEncoder": (
            "pyod.models.auto_encoder",
            "AutoEncoder",
            AlgorithmMetadata(
                category="Neural Network",
                complexity_time="O(n*epochs)",
                complexity_space="O(params)",
                supports_streaming=False,
                supports_multivariate=True,
                requires_gpu=True,
                description="AutoEncoder for reconstruction-based anomaly detection",
                typical_use_cases=[
                    "Complex patterns",
                    "High-dimensional data",
                    "Images",
                ],
            ),
        ),
        "VAE": (
            "pyod.models.vae",
            "VAE",
            AlgorithmMetadata(
                category="Neural Network",
                complexity_time="O(n*epochs)",
                complexity_space="O(params)",
                supports_streaming=False,
                supports_multivariate=True,
                requires_gpu=True,
                description="Variational AutoEncoder for probabilistic anomaly detection",
                typical_use_cases=["Generative modeling", "Uncertainty quantification"],
            ),
        ),
        "DeepSVDD": (
            "pyod.models.deep_svdd",
            "DeepSVDD",
            AlgorithmMetadata(
                category="Neural Network",
                complexity_time="O(n*epochs)",
                complexity_space="O(params)",
                supports_streaming=False,
                supports_multivariate=True,
                requires_gpu=True,
                description="Deep Support Vector Data Description",
                paper_reference="Ruff et al. (2018)",
                typical_use_cases=["Deep learning", "Complex manifolds"],
            ),
        ),
        # Probabilistic Models
        "COPOD": (
            "pyod.models.copod",
            "COPOD",
            AlgorithmMetadata(
                category="Probabilistic",
                complexity_time="O(n*p)",
                complexity_space="O(n*p)",
                supports_streaming=False,
                supports_multivariate=True,
                requires_gpu=False,
                description="Copula-Based Outlier Detection",
                paper_reference="Li et al. (2020)",
                typical_use_cases=["Parameter-free", "Fast detection", "Interpretable"],
            ),
        ),
        "ECOD": (
            "pyod.models.ecod",
            "ECOD",
            AlgorithmMetadata(
                category="Probabilistic",
                complexity_time="O(n*p)",
                complexity_space="O(p)",
                supports_streaming=False,
                supports_multivariate=True,
                requires_gpu=False,
                description="Empirical Cumulative Distribution Functions for Outlier Detection",
                paper_reference="Li et al. (2022)",
                typical_use_cases=["Parameter-free", "Interpretable", "Fast"],
            ),
        ),
        "ABOD": (
            "pyod.models.abod",
            "ABOD",
            AlgorithmMetadata(
                category="Probabilistic",
                complexity_time="O(n³)",
                complexity_space="O(n²)",
                supports_streaming=False,
                supports_multivariate=True,
                requires_gpu=False,
                description="Angle-Based Outlier Detection",
                paper_reference="Kriegel et al. (2008)",
                typical_use_cases=["High-dimensional data", "Geometric outliers"],
            ),
        ),
    }

    def __init__(
        self,
        algorithm_name: str,
        name: str | None = None,
        contamination_rate: ContaminationRate | None = None,
        **kwargs: Any,
    ):
        """Initialize enhanced PyOD adapter.

        Args:
            algorithm_name: Name of the PyOD algorithm
            name: Optional custom name
            contamination_rate: Expected contamination rate
            **kwargs: Algorithm-specific parameters
        """
        # Validate algorithm
        if algorithm_name not in self.ALGORITHM_MAPPING:
            available = list(self.ALGORITHM_MAPPING.keys())
            raise InvalidAlgorithmError(
                algorithm_name=algorithm_name, supported_algorithms=available
            )

        self._algorithm_name = algorithm_name
        self._name = name or f"Enhanced_PyOD_{algorithm_name}"
        self._contamination_rate = contamination_rate or ContaminationRate.auto()
        self._parameters = kwargs
        self._model: Any | None = None
        self._is_fitted = False
        self._feature_names: list[str] | None = None
        self._training_metadata: dict[str, Any] = {}

        # Load algorithm class and metadata
        module_path, class_name, metadata = self.ALGORITHM_MAPPING[algorithm_name]
        self._metadata = metadata
        self._model_class = self._load_model_class(module_path, class_name)

        # Suppress warnings for cleaner output
        warnings.filterwarnings("ignore", category=UserWarning)

    def _load_model_class(self, module_path: str, class_name: str) -> type:
        """Load PyOD model class dynamically."""
        try:
            module = importlib.import_module(module_path)
            return getattr(module, class_name)
        except (ImportError, AttributeError) as e:
            raise InvalidAlgorithmError(
                algorithm_name=self._algorithm_name,
                supported_algorithms=list(self.ALGORITHM_MAPPING.keys()),
                details=f"Failed to load {module_path}.{class_name}: {e}",
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
    def parameters(self) -> dict[str, Any]:
        """Get the current parameters."""
        return self._parameters.copy()

    @property
    def algorithm_metadata(self) -> AlgorithmMetadata:
        """Get algorithm metadata."""
        return self._metadata

    def fit(self, dataset: Dataset) -> None:
        """Fit the PyOD detector on a dataset.

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

            # Initialize model with parameters
            model_params = self._prepare_model_parameters()
            self._model = self._model_class(**model_params)

            # Fit model
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
                "pyod_version": self._get_pyod_version(),
            }

        except Exception as e:
            raise FittingError(
                detector_name=self._name, reason=str(e), dataset_name=dataset.name
            ) from e

    def detect(self, dataset: Dataset) -> DetectionResult:
        """Detect anomalies in the dataset.

        Args:
            dataset: Dataset to analyze

        Returns:
            Detection result with anomalies, scores, and metadata

        Raises:
            DetectorNotFittedError: If detector is not fitted
        """
        if not self._is_fitted:
            raise DetectorNotFittedError(detector_name=self._name, operation="detect")

        start_time = time.perf_counter()

        try:
            # Prepare data
            X, _ = self._prepare_features(dataset)

            # Get predictions and scores
            labels = self._model.predict(X)  # 0=normal, 1=anomaly
            raw_scores = self._model.decision_function(X)

            # Normalize scores
            normalized_scores = self._normalize_scores(raw_scores)

            # Create anomaly score objects
            anomaly_scores = [
                AnomalyScore(value=float(score), method=f"pyod_{self._algorithm_name}")
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
                detector_id=hash(self._name),  # Simple hash for demo
                dataset_id=dataset.id,
                anomalies=anomalies,
                scores=anomaly_scores,
                labels=labels.tolist(),
                threshold=threshold,
                execution_time_ms=execution_time,
                metadata={
                    "algorithm": self._algorithm_name,
                    "category": self._metadata.category,
                    "detection_time_seconds": execution_time / 1000,
                    "n_anomalies": len(anomalies),
                    "contamination_rate": self._contamination_rate.value,
                    **self._training_metadata,
                },
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
                    "status": "failed",
                },
            )

    def score(self, dataset: Dataset) -> list[AnomalyScore]:
        """Calculate anomaly scores for the dataset.

        Args:
            dataset: Dataset to score

        Returns:
            List of anomaly scores
        """
        if not self._is_fitted:
            raise DetectorNotFittedError(detector_name=self._name, operation="score")

        X, _ = self._prepare_features(dataset)
        raw_scores = self._model.decision_function(X)
        normalized_scores = self._normalize_scores(raw_scores)

        return [
            AnomalyScore(value=float(score), method=f"pyod_{self._algorithm_name}")
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

    def get_params(self) -> dict[str, Any]:
        """Get parameters of the detector."""
        if self._model is not None:
            return self._model.get_params()
        return self._parameters

    def set_params(self, **params: Any) -> None:
        """Set parameters of the detector."""
        self._parameters.update(params)
        if self._model is not None:
            # Update model parameters if possible
            valid_params = {
                k: v for k, v in params.items() if k in self._model.get_params()
            }
            if valid_params:
                self._model.set_params(**valid_params)

    def explain(
        self, dataset: Dataset, indices: list[int] | None = None
    ) -> dict[int, dict[str, Any]]:
        """Explain why certain points are anomalous.

        Args:
            dataset: Dataset containing the points
            indices: Specific indices to explain

        Returns:
            Dictionary mapping indices to explanations
        """
        if not self._is_fitted:
            raise DetectorNotFittedError(detector_name=self._name, operation="explain")

        X, feature_names = self._prepare_features(dataset)

        if indices is None:
            # Explain all anomalies
            labels = self._model.predict(X)
            indices = np.where(labels == 1)[0].tolist()

        explanations = {}

        for idx in indices:
            if idx >= len(X):
                continue

            # Get feature contributions
            feature_contributions = self._calculate_feature_contributions(
                X[idx : idx + 1], feature_names
            )

            explanations[idx] = {
                "feature_contributions": feature_contributions,
                "data_point": dataset.data.iloc[idx].to_dict(),
                "anomaly_score": float(
                    self._model.decision_function(X[idx : idx + 1])[0]
                ),
                "algorithm": self._algorithm_name,
                "explanation_method": "feature_deviation",
            }

        return explanations

    def feature_importances(self) -> dict[str, float]:
        """Get feature importances for anomaly detection.

        Returns:
            Dictionary mapping feature names to importance scores
        """
        if not self._is_fitted or self._feature_names is None:
            return {}

        # This is a simplified feature importance calculation
        # In practice, this would use SHAP or LIME for better explanations

        importances = {}
        for i, feature_name in enumerate(self._feature_names):
            # Placeholder: equal importance for now
            # Real implementation would calculate based on algorithm specifics
            importances[feature_name] = 1.0 / len(self._feature_names)

        return importances

    def _prepare_features(self, dataset: Dataset) -> tuple[np.ndarray, list[str]]:
        """Prepare features for algorithm.

        Args:
            dataset: Input dataset

        Returns:
            Tuple of (feature array, feature names)
        """
        # Get numeric features
        numeric_data = dataset.data.select_dtypes(include=[np.number])

        if numeric_data.empty:
            raise ValueError("No numeric features found in dataset")

        # Handle missing values
        numeric_data = numeric_data.fillna(numeric_data.mean())

        # Get feature names
        feature_names = numeric_data.columns.tolist()

        return numeric_data.values, feature_names

    def _prepare_model_parameters(self) -> dict[str, Any]:
        """Prepare parameters for model initialization."""
        params = self._parameters.copy()

        # Set contamination if not provided
        if "contamination" not in params:
            params["contamination"] = self._contamination_rate.value

        # Algorithm-specific parameter handling
        if self._algorithm_name in ["AvgKNN", "MedKNN"]:
            params["method"] = "mean" if "Avg" in self._algorithm_name else "median"

        # Handle neural network parameters
        if self._metadata.category == "Neural Network":
            # Set reasonable defaults for neural networks
            if "epochs" not in params:
                params["epochs"] = 100
            if "batch_size" not in params:
                params["batch_size"] = 32
            if "verbose" not in params:
                params["verbose"] = 0  # Suppress training output

        return params

    def _normalize_scores(self, scores: np.ndarray) -> np.ndarray:
        """Normalize scores to [0, 1] range."""
        if len(scores) == 0:
            return scores

        min_score = np.min(scores)
        max_score = np.max(scores)

        # Avoid division by zero
        if max_score == min_score:
            return np.full_like(scores, 0.5)

        # Normalize to [0, 1]
        normalized = (scores - min_score) / (max_score - min_score)
        return np.clip(normalized, 0.0, 1.0)

    def _create_anomaly_entities(
        self,
        dataset: Dataset,
        labels: np.ndarray,
        anomaly_scores: list[AnomalyScore],
        raw_scores: np.ndarray,
    ) -> list[Anomaly]:
        """Create anomaly entities for detected anomalies."""
        anomalies = []
        anomaly_indices = np.where(labels == 1)[0]

        for idx in anomaly_indices:
            if idx >= len(dataset.data):
                continue

            # Create anomaly entity
            anomaly = Anomaly(
                score=anomaly_scores[idx],
                data_point=dataset.data.iloc[idx].to_dict(),
                detector_name=self._name,
                metadata={
                    "index": int(idx),
                    "raw_score": float(raw_scores[idx]),
                    "algorithm": self._algorithm_name,
                    "category": self._metadata.category,
                },
            )

            anomalies.append(anomaly)

        return anomalies

    def _calculate_threshold(self, scores: np.ndarray) -> float:
        """Calculate anomaly threshold based on contamination rate."""
        if len(scores) == 0:
            return 0.5

        # Use contamination rate to determine threshold
        threshold_idx = int(len(scores) * (1 - self._contamination_rate.value))
        threshold_idx = max(0, min(threshold_idx, len(scores) - 1))

        sorted_scores = np.sort(scores)
        return float(sorted_scores[threshold_idx])

    def _calculate_feature_contributions(
        self, data_point: np.ndarray, feature_names: list[str]
    ) -> dict[str, float]:
        """Calculate feature contributions for explanation.

        This is a simplified version. In practice, use SHAP or LIME.
        """
        contributions = {}

        if len(data_point.shape) == 1:
            data_point = data_point.reshape(1, -1)

        # Simple feature deviation-based explanation
        for i, feature_name in enumerate(feature_names):
            # Placeholder: contribution based on feature value deviation
            value = data_point[0, i]
            # In real implementation, this would be more sophisticated
            contribution = abs(value) / (abs(value) + 1.0)  # Simple normalization
            contributions[feature_name] = float(contribution)

        return contributions

    def _get_pyod_version(self) -> str:
        """Get PyOD version."""
        try:
            import pyod

            return pyod.__version__
        except (ImportError, AttributeError):
            return "unknown"

    @classmethod
    def list_algorithms(cls) -> list[str]:
        """List all available algorithms."""
        return list(cls.ALGORITHM_MAPPING.keys())

    @classmethod
    def get_algorithm_metadata(cls, algorithm_name: str) -> AlgorithmMetadata | None:
        """Get metadata for a specific algorithm."""
        if algorithm_name in cls.ALGORITHM_MAPPING:
            return cls.ALGORITHM_MAPPING[algorithm_name][2]
        return None

    @classmethod
    def get_algorithms_by_category(cls) -> dict[str, list[str]]:
        """Get algorithms grouped by category."""
        categories = {}
        for name, (_, _, metadata) in cls.ALGORITHM_MAPPING.items():
            category = metadata.category
            if category not in categories:
                categories[category] = []
            categories[category].append(name)
        return categories

    @classmethod
    def recommend_algorithms(
        cls,
        n_samples: int,
        n_features: int,
        has_gpu: bool = False,
        prefer_fast: bool = False,
    ) -> list[str]:
        """Recommend algorithms based on dataset characteristics.

        Args:
            n_samples: Number of samples
            n_features: Number of features
            has_gpu: Whether GPU is available
            prefer_fast: Whether to prefer faster algorithms

        Returns:
            List of recommended algorithm names
        """
        recommendations = []

        for name, (_, _, metadata) in cls.ALGORITHM_MAPPING.items():
            # Skip GPU algorithms if no GPU
            if metadata.requires_gpu and not has_gpu:
                continue

            # Fast algorithms for large datasets
            if prefer_fast and n_samples > 10000:
                if metadata.complexity_time in ["O(n*p)", "O(n log n)"]:
                    recommendations.append(name)
            # General recommendations
            elif n_samples < 1000:
                # Small datasets: any algorithm is fine
                recommendations.append(name)
            elif n_samples < 10000:
                # Medium datasets: avoid O(n²) and higher
                if (
                    "n²" not in metadata.complexity_time
                    and "n³" not in metadata.complexity_time
                ):
                    recommendations.append(name)
            else:
                # Large datasets: only fast algorithms
                if metadata.complexity_time in ["O(n*p)", "O(n log n)", "O(n)"]:
                    recommendations.append(name)

        # Default recommendations if none match
        if not recommendations:
            recommendations = ["IsolationForest", "COPOD", "ECOD"]

        return recommendations[:5]  # Return top 5 recommendations
