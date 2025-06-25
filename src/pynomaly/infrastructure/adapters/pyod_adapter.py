"""PyOD adapter for anomaly detection algorithms."""

from __future__ import annotations

import time
from typing import Any

import numpy as np

from pynomaly.domain.entities import Anomaly, Dataset, DetectionResult, Detector
from pynomaly.domain.exceptions import (
    DetectorNotFittedError,
    FittingError,
    InvalidAlgorithmError,
)
from pynomaly.domain.value_objects import AnomalyScore, ContaminationRate


class PyODAdapter(Detector):
    """Adapter for PyOD anomaly detection algorithms."""

    # Mapping of algorithm names to PyOD classes
    ALGORITHM_MAPPING: dict[str, tuple[str, type[Any]]] = {
        # Linear models
        "PCA": ("pyod.models.pca", "PCA"),
        "MCD": ("pyod.models.mcd", "MCD"),
        "OCSVM": ("pyod.models.ocsvm", "OCSVM"),
        "LMDD": ("pyod.models.lmdd", "LMDD"),
        # Proximity-based
        "LOF": ("pyod.models.lof", "LOF"),
        "COF": ("pyod.models.cof", "COF"),
        "CBLOF": ("pyod.models.cblof", "CBLOF"),
        "LOCI": ("pyod.models.loci", "LOCI"),
        "HBOS": ("pyod.models.hbos", "HBOS"),
        "KNN": ("pyod.models.knn", "KNN"),
        "AvgKNN": ("pyod.models.knn", "KNN"),
        "MedKNN": ("pyod.models.knn", "KNN"),
        "SOD": ("pyod.models.sod", "SOD"),
        "ROD": ("pyod.models.rod", "ROD"),
        # Probabilistic
        "ABOD": ("pyod.models.abod", "ABOD"),
        "FastABOD": ("pyod.models.abod", "FastABOD"),
        "COPOD": ("pyod.models.copod", "COPOD"),
        "MAD": ("pyod.models.mad", "MAD"),
        "SOS": ("pyod.models.sos", "SOS"),
        # Ensemble
        "IsolationForest": ("pyod.models.iforest", "IForest"),
        "IForest": ("pyod.models.iforest", "IForest"),
        "FeatureBagging": ("pyod.models.feature_bagging", "FeatureBagging"),
        "LSCP": ("pyod.models.lscp", "LSCP"),
        "XGBOD": ("pyod.models.xgbod", "XGBOD"),
        "LODA": ("pyod.models.loda", "LODA"),
        "SUOD": ("pyod.models.suod", "SUOD"),
        # Neural networks
        "AutoEncoder": ("pyod.models.auto_encoder", "AutoEncoder"),
        "VAE": ("pyod.models.vae", "VAE"),
        "Beta-VAE": ("pyod.models.vae", "BetaVAE"),
        "SO_GAAL": ("pyod.models.so_gaal", "SO_GAAL"),
        "MO_GAAL": ("pyod.models.mo_gaal", "MO_GAAL"),
        "DeepSVDD": ("pyod.models.deep_svdd", "DeepSVDD"),
        # Graph-based
        "R-Graph": ("pyod.models.rgraph", "RGraph"),
        "LUNAR": ("pyod.models.lunar", "LUNAR"),
        # Deep Learning (Additional)
        "ALAD": ("pyod.models.alad", "ALAD"),
        "AnoGAN": ("pyod.models.anogan", "AnoGAN"),
        "DIF": ("pyod.models.dif", "DIF"),
        # Statistical/Other
        "CLF": ("pyod.models.clf", "CLF"),
        "KPCA": ("pyod.models.kpca", "KPCA"),
        "PCA-MAD": ("pyod.models.pca", "PCA"),  # With MAD option
        "QMCD": ("pyod.models.qmcd", "QMCD"),
        # Other
        "INNE": ("pyod.models.inne", "INNE"),
        "ECOD": ("pyod.models.ecod", "ECOD"),
        "CD": ("pyod.models.cd", "CD"),
        "KDE": ("pyod.models.kde", "KDE"),
        "Sampling": ("pyod.models.sampling", "Sampling"),
        "GMM": ("pyod.models.gmm", "GMM"),
    }

    def __init__(
        self,
        algorithm_name: str,
        name: str | None = None,
        contamination_rate: ContaminationRate | None = None,
        **kwargs: Any,
    ):
        """Initialize PyOD adapter.

        Args:
            algorithm_name: Name of the PyOD algorithm
            name: Optional custom name for the detector
            contamination_rate: Expected contamination rate
            **kwargs: Algorithm-specific parameters
        """
        # Validate algorithm
        if algorithm_name not in self.ALGORITHM_MAPPING:
            raise InvalidAlgorithmError(
                algorithm_name, available_algorithms=list(self.ALGORITHM_MAPPING.keys())
            )

        # Initialize parent
        super().__init__(
            name=name or f"PyOD_{algorithm_name}",
            algorithm_name=algorithm_name,
            contamination_rate=contamination_rate or ContaminationRate.auto(),
            parameters=kwargs,
        )

        # Load PyOD model class
        self._model_class = self._load_model_class(algorithm_name)
        self._model: Any | None = None

        # Set metadata based on algorithm
        self._set_algorithm_metadata(algorithm_name)

    def _load_model_class(self, algorithm_name: str) -> type[Any]:
        """Dynamically load PyOD model class."""
        module_path, class_name = self.ALGORITHM_MAPPING[algorithm_name]

        try:
            import importlib

            module = importlib.import_module(module_path)
            return getattr(module, class_name)
        except (ImportError, AttributeError) as e:
            raise InvalidAlgorithmError(
                algorithm_name, available_algorithms=list(self.ALGORITHM_MAPPING.keys())
            ) from e

    def _set_algorithm_metadata(self, algorithm_name: str) -> None:
        """Set metadata based on algorithm characteristics."""
        # Time/space complexity for common algorithms
        complexity_info = {
            "IsolationForest": ("O(n log n)", "O(n)"),
            "LOF": ("O(n²)", "O(n)"),
            "KNN": ("O(n log n)", "O(n)"),
            "OCSVM": ("O(n²)", "O(n²)"),
            "PCA": ("O(n*p²)", "O(p²)"),
            "AutoEncoder": ("O(n*epochs)", "O(params)"),
            "COPOD": ("O(n*p)", "O(n*p)"),
            "ECOD": ("O(n*p)", "O(n*p)"),
            "HBOS": ("O(n*p)", "O(p)"),
        }

        if algorithm_name in complexity_info:
            time_complexity, space_complexity = complexity_info[algorithm_name]
            self.update_metadata("time_complexity", time_complexity)
            self.update_metadata("space_complexity", space_complexity)

        # Streaming support
        streaming_algorithms = {"LOF", "KNN", "LODA"}
        self.update_metadata(
            "supports_streaming", algorithm_name in streaming_algorithms
        )

        # Algorithm categories
        if algorithm_name in {"IsolationForest", "LODA", "FeatureBagging"}:
            self.update_metadata("category", "ensemble")
        elif algorithm_name in {"LOF", "KNN", "COF", "SOD"}:
            self.update_metadata("category", "proximity")
        elif algorithm_name in {"AutoEncoder", "VAE", "DeepSVDD"}:
            self.update_metadata("category", "neural_network")
        elif algorithm_name in {"PCA", "MCD", "OCSVM"}:
            self.update_metadata("category", "linear")

    def fit(self, dataset: Dataset) -> None:
        """Fit the PyOD detector on a dataset."""
        try:
            # Initialize model with parameters
            model_params = {
                "contamination": self.contamination_rate.value,
                **self.parameters,
            }

            # Special handling for certain algorithms
            if self.algorithm_name in ["AvgKNN", "MedKNN"]:
                model_params["method"] = (
                    "mean" if "Avg" in self.algorithm_name else "median"
                )

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
            self.update_metadata(
                "training_features", len(dataset.get_numeric_features())
            )

        except Exception as e:
            raise FittingError(
                detector_name=self.name, reason=str(e), dataset_name=dataset.name
            ) from e

    def detect(self, dataset: Dataset) -> DetectionResult:
        """Detect anomalies in a dataset."""
        if not self.is_fitted or self._model is None:
            raise DetectorNotFittedError(detector_name=self.name, operation="detect")

        # Get features
        X = dataset.features[dataset.get_numeric_features()].values

        start_time = time.time()

        # Get predictions and scores
        labels = self._model.predict(X)  # 0 = normal, 1 = anomaly
        scores = self._model.decision_function(X)  # Raw anomaly scores

        # Normalize scores to [0, 1] range
        # PyOD scores: higher = more anomalous, but not normalized
        min_score = scores.min()
        max_score = scores.max()
        if max_score > min_score:
            normalized_scores = (scores - min_score) / (max_score - min_score)
        else:
            normalized_scores = np.ones_like(scores) * 0.5

        # Create AnomalyScore objects
        anomaly_scores = [
            AnomalyScore(value=float(score), method="pyod")
            for score in normalized_scores
        ]

        # Calculate threshold from normalized scores
        threshold_idx = int(len(scores) * (1 - self.contamination_rate.value))
        threshold = float(np.sort(normalized_scores)[threshold_idx])

        # Create Anomaly entities for detected anomalies
        anomalies = []
        anomaly_indices = np.where(labels == 1)[0]

        for idx in anomaly_indices:
            anomaly = Anomaly(
                score=anomaly_scores[idx],
                data_point=dataset.data.iloc[idx].to_dict(),
                detector_name=self.name,
            )
            # Add algorithm-specific metadata
            anomaly.add_metadata("raw_score", float(scores[idx]))
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
                "pyod_version": self._get_pyod_version(),
            },
        )

        return result

    def score(self, dataset: Dataset) -> list[AnomalyScore]:
        """Calculate anomaly scores for a dataset."""
        if not self.is_fitted or self._model is None:
            raise DetectorNotFittedError(detector_name=self.name, operation="score")

        # Get features
        X = dataset.features[dataset.get_numeric_features()].values

        # Get raw scores
        scores = self._model.decision_function(X)

        # Normalize to [0, 1]
        min_score = scores.min()
        max_score = scores.max()
        if max_score > min_score:
            normalized_scores = (scores - min_score) / (max_score - min_score)
        else:
            normalized_scores = np.ones_like(scores) * 0.5

        # Create AnomalyScore objects
        return [
            AnomalyScore(value=float(score), method="pyod")
            for score in normalized_scores
        ]

    def get_params(self) -> dict[str, Any]:
        """Get current parameters."""
        if self._model is not None:
            return self._model.get_params()
        return self.parameters

    def set_params(self, **params: Any) -> None:
        """Set parameters."""
        self.update_parameters(**params)
        if self._model is not None:
            # Update model parameters
            valid_params = {
                k: v for k, v in params.items() if k in self._model.get_params()
            }
            self._model.set_params(**valid_params)

    def _get_pyod_version(self) -> str:
        """Get PyOD version."""
        try:
            import pyod

            return pyod.__version__
        except ImportError:
            return "unknown"

    @classmethod
    def list_algorithms(cls) -> list[str]:
        """List all available PyOD algorithms.

        Returns:
            List of algorithm names
        """
        return list(cls.ALGORITHM_MAPPING.keys())

    @classmethod
    def get_algorithm_info(cls) -> dict[str, dict[str, str]]:
        """Get detailed information about PyOD algorithms.

        Returns:
            Dictionary mapping algorithm names to their information
        """
        algorithm_info = {
            # Linear models
            "PCA": {
                "category": "Linear",
                "type": "Unsupervised",
                "description": "Principal Component Analysis based outlier detection",
            },
            "MCD": {
                "category": "Linear",
                "type": "Unsupervised",
                "description": "Minimum Covariance Determinant outlier detection",
            },
            "OCSVM": {
                "category": "Linear",
                "type": "Unsupervised",
                "description": "One-Class Support Vector Machine",
            },
            "LMDD": {
                "category": "Linear",
                "type": "Unsupervised",
                "description": "Linear Model with Decorrelation",
            },
            # Proximity-based
            "LOF": {
                "category": "Proximity",
                "type": "Unsupervised",
                "description": "Local Outlier Factor",
            },
            "COF": {
                "category": "Proximity",
                "type": "Unsupervised",
                "description": "Connectivity-Based Outlier Factor",
            },
            "CBLOF": {
                "category": "Proximity",
                "type": "Unsupervised",
                "description": "Clustering-Based Local Outlier Factor",
            },
            "LOCI": {
                "category": "Proximity",
                "type": "Unsupervised",
                "description": "Local Correlation Integral",
            },
            "HBOS": {
                "category": "Proximity",
                "type": "Unsupervised",
                "description": "Histogram-Based Outlier Score",
            },
            "KNN": {
                "category": "Proximity",
                "type": "Unsupervised",
                "description": "k-Nearest Neighbors outlier detection",
            },
            "AvgKNN": {
                "category": "Proximity",
                "type": "Unsupervised",
                "description": "Average k-Nearest Neighbors",
            },
            "MedKNN": {
                "category": "Proximity",
                "type": "Unsupervised",
                "description": "Median k-Nearest Neighbors",
            },
            "SOD": {
                "category": "Proximity",
                "type": "Unsupervised",
                "description": "Subspace Outlier Detection",
            },
            "ROD": {
                "category": "Proximity",
                "type": "Unsupervised",
                "description": "Rotation-Based Outlier Detection",
            },
            # Probabilistic
            "ABOD": {
                "category": "Probabilistic",
                "type": "Unsupervised",
                "description": "Angle-Based Outlier Detection",
            },
            "FastABOD": {
                "category": "Probabilistic",
                "type": "Unsupervised",
                "description": "Fast Angle-Based Outlier Detection",
            },
            "COPOD": {
                "category": "Probabilistic",
                "type": "Unsupervised",
                "description": "Copula-Based Outlier Detection",
            },
            "MAD": {
                "category": "Probabilistic",
                "type": "Unsupervised",
                "description": "Median Absolute Deviation",
            },
            "SOS": {
                "category": "Probabilistic",
                "type": "Unsupervised",
                "description": "Stochastic Outlier Selection",
            },
            # Ensemble
            "IsolationForest": {
                "category": "Ensemble",
                "type": "Unsupervised",
                "description": "Isolation Forest for anomaly detection",
            },
            "IForest": {
                "category": "Ensemble",
                "type": "Unsupervised",
                "description": "Isolation Forest (alias)",
            },
            "FeatureBagging": {
                "category": "Ensemble",
                "type": "Unsupervised",
                "description": "Feature Bagging for outlier detection",
            },
            "LSCP": {
                "category": "Ensemble",
                "type": "Unsupervised",
                "description": "Locally Selective Combination",
            },
            "XGBOD": {
                "category": "Ensemble",
                "type": "Supervised",
                "description": "Extreme Gradient Boosting Outlier Detection",
            },
            "LODA": {
                "category": "Ensemble",
                "type": "Unsupervised",
                "description": "Lightweight On-line Detector of Anomalies",
            },
            "SUOD": {
                "category": "Ensemble",
                "type": "Unsupervised",
                "description": "Scalable Unsupervised Outlier Detection",
            },
            # Neural networks
            "AutoEncoder": {
                "category": "Neural Network",
                "type": "Unsupervised",
                "description": "AutoEncoder-based anomaly detection",
            },
            "VAE": {
                "category": "Neural Network",
                "type": "Unsupervised",
                "description": "Variational AutoEncoder",
            },
            "Beta-VAE": {
                "category": "Neural Network",
                "type": "Unsupervised",
                "description": "Beta Variational AutoEncoder",
            },
            "SO_GAAL": {
                "category": "Neural Network",
                "type": "Semi-supervised",
                "description": "Single-Objective Generative Adversarial Active Learning",
            },
            "MO_GAAL": {
                "category": "Neural Network",
                "type": "Semi-supervised",
                "description": "Multiple-Objective Generative Adversarial Active Learning",
            },
            "DeepSVDD": {
                "category": "Neural Network",
                "type": "Unsupervised",
                "description": "Deep Support Vector Data Description",
            },
            # Graph-based
            "R-Graph": {
                "category": "Graph",
                "type": "Unsupervised",
                "description": "Random Graph-based outlier detection",
            },
            "LUNAR": {
                "category": "Graph",
                "type": "Unsupervised",
                "description": "Locally Uniform Network Anomaly Ranking",
            },
            # Deep Learning (Additional)
            "ALAD": {
                "category": "Deep Learning",
                "type": "Unsupervised",
                "description": "Adversarially Learned Anomaly Detection",
            },
            "AnoGAN": {
                "category": "Deep Learning",
                "type": "Unsupervised",
                "description": "Anomaly Detection with Generative Adversarial Networks",
            },
            "DIF": {
                "category": "Deep Learning",
                "type": "Unsupervised",
                "description": "Deep Isolation Forest",
            },
        }

        # Return info for algorithms that actually exist in our mapping
        return {
            algo: algorithm_info.get(
                algo,
                {
                    "category": "Unknown",
                    "type": "Unknown",
                    "description": "No description available",
                },
            )
            for algo in cls.ALGORITHM_MAPPING.keys()
        }
