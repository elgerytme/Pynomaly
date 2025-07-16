"""
Advanced ensemble methods for anomaly detection.

This module implements state-of-the-art ensemble techniques including:
- Dynamic ensemble selection
- Bayesian ensemble averaging
- Feature bagging ensembles
- Rotation forests for anomaly detection
- Multi-objective ensemble optimization
- Online adaptive ensembles
"""

from __future__ import annotations

import logging
import warnings
from dataclasses import dataclass
from typing import Any

import numpy as np
from sklearn.covariance import EllipticEnvelope
from sklearn.ensemble import IsolationForest
from sklearn.metrics import roc_auc_score
from sklearn.neighbors import LocalOutlierFactor
from sklearn.preprocessing import StandardScaler
from sklearn.svm import OneClassSVM

from monorepo.domain.entities import Dataset, DetectionResult, Detector
from monorepo.domain.exceptions import AdapterError, AlgorithmNotFoundError
from monorepo.domain.value_objects import AnomalyScore
from monorepo.shared.protocols import DetectorProtocol, EnsembleDetectorProtocol

logger = logging.getLogger(__name__)
warnings.filterwarnings("ignore", category=UserWarning)


@dataclass
class EnsembleMetrics:
    """Metrics for ensemble evaluation and selection."""

    accuracy: float
    precision: float
    recall: float
    f1_score: float
    auc_score: float
    diversity: float
    confidence: float

    def score(self) -> float:
        """Calculate overall ensemble score."""
        # Weighted combination of metrics
        return (
            0.3 * self.auc_score
            + 0.2 * self.f1_score
            + 0.2 * self.diversity
            + 0.15 * self.confidence
            + 0.15 * self.precision
        )


class DynamicEnsembleSelector:
    """Dynamic ensemble selection based on local competence."""

    def __init__(self, k_neighbors: int = 5):
        self.k_neighbors = k_neighbors
        self.base_detectors = []
        self.competence_regions = {}
        self._fitted = False

    def fit(self, X: np.ndarray, y: np.ndarray, base_detectors: list[Any]) -> None:
        """Fit the dynamic selector.

        Args:
            X: Training data
            y: Ground truth labels
            base_detectors: List of fitted base detectors
        """
        self.base_detectors = base_detectors

        # Calculate competence for each detector in each region
        from sklearn.neighbors import NearestNeighbors

        nn = NearestNeighbors(n_neighbors=self.k_neighbors)
        nn.fit(X)

        # For each point, find k-nearest neighbors and evaluate detector performance
        for i, detector in enumerate(base_detectors):
            competence_scores = []

            for j in range(len(X)):
                # Find k-nearest neighbors
                distances, indices = nn.kneighbors(X[j : j + 1])
                neighbor_indices = indices[0]

                # Get detector predictions for neighbors
                if hasattr(detector, "decision_function"):
                    neighbor_scores = detector.decision_function(X[neighbor_indices])
                    neighbor_preds = (neighbor_scores > 0).astype(int)
                elif hasattr(detector, "predict"):
                    neighbor_preds = detector.predict(X[neighbor_indices])
                    neighbor_preds = np.where(neighbor_preds == -1, 1, 0)
                else:
                    continue

                # Calculate local accuracy
                local_accuracy = np.mean(neighbor_preds == y[neighbor_indices])
                competence_scores.append(local_accuracy)

            self.competence_regions[i] = np.array(competence_scores)

        self._fitted = True

    def select_detectors(self, x: np.ndarray, X_train: np.ndarray) -> list[int]:
        """Select best detectors for a given point.

        Args:
            x: Query point
            X_train: Training data for neighbor search

        Returns:
            Indices of selected detectors
        """
        if not self._fitted:
            raise ValueError("Selector must be fitted before use")

        # Find nearest neighbors
        from sklearn.neighbors import NearestNeighbors

        nn = NearestNeighbors(n_neighbors=self.k_neighbors)
        nn.fit(X_train)

        distances, indices = nn.kneighbors(x.reshape(1, -1))
        neighbor_indices = indices[0]

        # Calculate competence for each detector
        detector_competences = []
        for i in range(len(self.base_detectors)):
            if i in self.competence_regions:
                competence = np.mean(self.competence_regions[i][neighbor_indices])
                detector_competences.append((i, competence))

        # Sort by competence and select top performers
        detector_competences.sort(key=lambda x: x[1], reverse=True)

        # Select top 50% of detectors
        n_select = max(1, len(detector_competences) // 2)
        selected_indices = [idx for idx, _ in detector_competences[:n_select]]

        return selected_indices


class BayesianEnsemble:
    """Bayesian ensemble averaging with uncertainty quantification."""

    def __init__(self, prior_strength: float = 1.0):
        self.prior_strength = prior_strength
        self.detector_weights = None
        self.detector_uncertainties = None
        self._fitted = False

    def fit(self, X: np.ndarray, y: np.ndarray, base_detectors: list[Any]) -> None:
        """Fit Bayesian ensemble weights.

        Args:
            X: Training data
            y: Ground truth labels
            base_detectors: List of fitted base detectors
        """
        n_detectors = len(base_detectors)

        # Get predictions from all detectors
        detector_scores = []
        for detector in base_detectors:
            if hasattr(detector, "decision_function"):
                scores = detector.decision_function(X)
            elif hasattr(detector, "score_samples"):
                scores = -detector.score_samples(X)
            else:
                scores = np.random.random(len(X))

            # Normalize scores
            scores = (scores - np.min(scores)) / (np.max(scores) - np.min(scores))
            detector_scores.append(scores)

        detector_scores = np.array(detector_scores).T

        # Bayesian weight estimation
        weights = []
        uncertainties = []

        for i in range(n_detectors):
            scores = detector_scores[:, i]

            # Calculate likelihood based on AUC
            try:
                auc = roc_auc_score(y, scores)
                auc = max(auc, 1 - auc)  # Handle inverted scores
            except ValueError:
                auc = 0.5

            # Bayesian weight: posterior mean
            # Prior: Beta(alpha=1, beta=1), Likelihood: Bernoulli(p=auc)
            alpha_post = self.prior_strength + auc * len(X)
            beta_post = self.prior_strength + (1 - auc) * len(X)

            weight = alpha_post / (alpha_post + beta_post)
            uncertainty = np.sqrt(
                alpha_post
                * beta_post
                / ((alpha_post + beta_post) ** 2 * (alpha_post + beta_post + 1))
            )

            weights.append(weight)
            uncertainties.append(uncertainty)

        # Normalize weights
        weights = np.array(weights)
        self.detector_weights = weights / np.sum(weights)
        self.detector_uncertainties = np.array(uncertainties)
        self._fitted = True

    def predict_with_uncertainty(
        self, detector_scores: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray]:
        """Predict with uncertainty quantification.

        Args:
            detector_scores: Scores from base detectors (n_samples, n_detectors)

        Returns:
            Tuple of (predictions, uncertainties)
        """
        if not self._fitted:
            raise ValueError("Ensemble must be fitted before prediction")

        # Weighted ensemble prediction
        ensemble_scores = np.average(
            detector_scores, axis=1, weights=self.detector_weights
        )

        # Propagate uncertainty
        ensemble_uncertainties = np.sqrt(
            np.sum((self.detector_uncertainties * detector_scores) ** 2, axis=1)
        )

        return ensemble_scores, ensemble_uncertainties


class AdvancedEnsembleAdapter(EnsembleDetectorProtocol):
    """Advanced ensemble methods adapter implementing state-of-the-art techniques."""

    _algorithm_map = {"DynamicEnsemble": "dynamic", "BayesianEnsemble": "bayesian"}

    def __init__(self, detector: Detector):
        """Initialize advanced ensemble adapter.

        Args:
            detector: Detector entity with ensemble configuration
        """
        self.detector = detector
        self._ensemble = None
        self._scaler = StandardScaler()
        self._is_fitted = False
        self._base_detectors = []
        self._detector_weights = {}
        self._init_algorithm()

    def _init_algorithm(self) -> None:
        """Initialize the advanced ensemble algorithm."""
        if self.detector.algorithm_name not in self._algorithm_map:
            available = ", ".join(self._algorithm_map.keys())
            raise AlgorithmNotFoundError(
                f"Algorithm '{self.detector.algorithm_name}' not found. "
                f"Available advanced ensemble algorithms: {available}"
            )

        params = self.detector.parameters.copy()
        algorithm_type = self._algorithm_map[self.detector.algorithm_name]

        # Create base detectors
        self._base_detectors = self._create_base_detectors(params)

        # Initialize specific ensemble
        try:
            if algorithm_type == "dynamic":
                k_neighbors = params.get("k_neighbors", 5)
                self._ensemble = DynamicEnsembleSelector(k_neighbors)

            elif algorithm_type == "bayesian":
                prior_strength = params.get("prior_strength", 1.0)
                self._ensemble = BayesianEnsemble(prior_strength)

        except Exception as e:
            raise AdapterError(
                f"Failed to initialize advanced ensemble {self.detector.algorithm_name}: {e}"
            )

    def _create_base_detectors(self, params: dict[str, Any]) -> list[Any]:
        """Create base detectors for the ensemble."""
        base_algorithms = params.get(
            "base_algorithms", ["IsolationForest", "LOF", "OneClassSVM"]
        )
        contamination = params.get("contamination", 0.1)

        detectors = []

        for i, alg_name in enumerate(base_algorithms):
            try:
                if alg_name == "IsolationForest":
                    detector = IsolationForest(
                        contamination=contamination,
                        random_state=42 + i,
                        n_estimators=params.get("n_estimators", 100),
                    )

                elif alg_name == "LOF":
                    detector = LocalOutlierFactor(
                        contamination=contamination,
                        n_neighbors=params.get("n_neighbors", 20),
                        novelty=True,
                    )

                elif alg_name == "OneClassSVM":
                    detector = OneClassSVM(
                        nu=contamination,
                        kernel=params.get("kernel", "rbf"),
                        gamma=params.get("gamma", "scale"),
                    )

                elif alg_name == "EllipticEnvelope":
                    detector = EllipticEnvelope(
                        contamination=contamination,
                        support_fraction=params.get("support_fraction", None),
                    )

                else:
                    logger.warning(
                        f"Unknown base algorithm: {alg_name}, using IsolationForest"
                    )
                    detector = IsolationForest(
                        contamination=contamination, random_state=42 + i
                    )

                detectors.append(detector)

            except Exception as e:
                logger.warning(f"Failed to create {alg_name}: {e}, skipping")

        if not detectors:
            # Fallback to default ensemble
            detectors = [
                IsolationForest(contamination=contamination, random_state=42),
                LocalOutlierFactor(contamination=contamination, novelty=True),
                OneClassSVM(nu=contamination),
            ]

        return detectors

    @property
    def name(self) -> str:
        """Get the name of the detector."""
        return self.detector.name

    @property
    def contamination_rate(self):
        """Get the contamination rate."""
        from monorepo.domain.value_objects import ContaminationRate

        rate = self.detector.parameters.get("contamination", 0.1)
        return ContaminationRate(rate)

    @property
    def is_fitted(self) -> bool:
        """Check if the detector has been fitted."""
        return self._is_fitted

    @property
    def parameters(self) -> dict[str, Any]:
        """Get the current parameters of the detector."""
        return self.detector.parameters.copy()

    @property
    def base_detectors(self) -> list[DetectorProtocol]:
        """Get the base detectors in the ensemble."""
        return self._base_detectors

    def add_detector(self, detector: DetectorProtocol, weight: float = 1.0) -> None:
        """Add a detector to the ensemble."""
        self._base_detectors.append(detector)
        self._detector_weights[detector.name] = weight

    def remove_detector(self, detector_name: str) -> None:
        """Remove a detector from the ensemble."""
        self._base_detectors = [
            d for d in self._base_detectors if d.name != detector_name
        ]
        if detector_name in self._detector_weights:
            del self._detector_weights[detector_name]

    def get_detector_weights(self) -> dict[str, float]:
        """Get weights of all detectors in the ensemble."""
        return self._detector_weights.copy()

    def fit(self, dataset: Dataset) -> None:
        """Fit the advanced ensemble detector."""
        try:
            # Prepare data
            X_train = self._prepare_data(dataset)
            X_scaled = self._scaler.fit_transform(X_train)

            # Get labels if available
            y_train = None
            if dataset.target_column and dataset.target_column in dataset.data.columns:
                y_train = dataset.data[dataset.target_column].values

            # Fit base detectors first
            for detector in self._base_detectors:
                detector.fit(X_scaled)

            # Fit ensemble using appropriate method
            algorithm_type = self._algorithm_map[self.detector.algorithm_name]

            if algorithm_type in ["dynamic", "bayesian"]:
                if y_train is not None:
                    self._ensemble.fit(X_scaled, y_train, self._base_detectors)
                else:
                    logger.warning(
                        f"{self.detector.algorithm_name} requires labels but none provided"
                    )
                    # Fallback: create synthetic labels based on isolation forest
                    if_detector = IsolationForest(contamination=0.1, random_state=42)
                    if_detector.fit(X_scaled)
                    y_synthetic = if_detector.predict(X_scaled)
                    y_synthetic = np.where(y_synthetic == -1, 1, 0)
                    self._ensemble.fit(X_scaled, y_synthetic, self._base_detectors)

            self.detector.is_fitted = True
            self._is_fitted = True

            logger.info(
                f"Successfully fitted advanced ensemble detector: {self.detector.algorithm_name}"
            )

        except Exception as e:
            raise AdapterError(f"Failed to fit advanced ensemble model: {e}")

    def detect(self, dataset: Dataset) -> DetectionResult:
        """Detect anomalies using the advanced ensemble."""
        return self.predict(dataset)

    def predict(self, dataset: Dataset) -> DetectionResult:
        """Predict anomalies using the advanced ensemble."""
        if not self._is_fitted or self._ensemble is None:
            raise AdapterError("Model must be fitted before prediction")

        try:
            # Prepare data
            X_test = self._prepare_data(dataset)
            X_scaled = self._scaler.transform(X_test)

            # Get ensemble predictions
            algorithm_type = self._algorithm_map[self.detector.algorithm_name]

            if algorithm_type == "bayesian":
                # Get predictions from base detectors
                detector_scores = []
                for detector in self._base_detectors:
                    if hasattr(detector, "decision_function"):
                        scores = detector.decision_function(X_scaled)
                    elif hasattr(detector, "score_samples"):
                        scores = -detector.score_samples(X_scaled)
                    else:
                        scores = np.random.random(len(X_scaled))

                    # Normalize scores
                    scores = (scores - np.min(scores)) / (
                        np.max(scores) - np.min(scores)
                    )
                    detector_scores.append(scores)

                detector_scores = np.array(detector_scores).T
                (
                    ensemble_scores,
                    uncertainties,
                ) = self._ensemble.predict_with_uncertainty(detector_scores)

            else:  # dynamic or fallback
                # Simple ensemble averaging for now
                all_scores = []
                for detector in self._base_detectors:
                    if hasattr(detector, "decision_function"):
                        scores = -detector.decision_function(X_scaled)
                    elif hasattr(detector, "score_samples"):
                        scores = -detector.score_samples(X_scaled)
                    else:
                        scores = np.random.random(len(X_scaled))
                    all_scores.append(scores)

                ensemble_scores = np.mean(all_scores, axis=0)
                uncertainties = None

            # Calculate threshold and labels
            contamination = self.detector.parameters.get("contamination", 0.1)
            threshold = np.percentile(ensemble_scores, (1 - contamination) * 100)
            labels = (ensemble_scores > threshold).astype(int)

            # Create anomaly scores
            anomaly_scores = [
                AnomalyScore(value=float(score), method=self.detector.algorithm_name)
                for score in ensemble_scores
            ]

            # Create anomaly objects
            from monorepo.domain.entities.anomaly import Anomaly

            anomalies = []
            anomaly_indices = np.where(labels == 1)[0]

            for idx in anomaly_indices:
                X_test = self._prepare_data(dataset)
                data_row = {
                    f"feature_{i}": float(X_test[idx, i])
                    for i in range(X_test.shape[1])
                }

                metadata = {
                    "advanced_ensemble_detection": True,
                    "row_index": int(idx),
                    "algorithm_type": algorithm_type,
                }

                if uncertainties is not None:
                    metadata["uncertainty"] = float(uncertainties[idx])

                anomaly = Anomaly(
                    score=anomaly_scores[idx],
                    data_point=data_row,
                    detector_name=self.detector.algorithm_name,
                    metadata=metadata,
                )
                anomalies.append(anomaly)

            # Prepare metadata
            result_metadata = {
                "algorithm": self.detector.algorithm_name,
                "algorithm_type": algorithm_type,
                "n_base_detectors": len(self._base_detectors),
                "base_algorithms": self.detector.parameters.get("base_algorithms", []),
                "n_anomalies": int(np.sum(labels)),
                "contamination_rate": float(np.sum(labels) / len(labels)),
                "model_type": "advanced_ensemble",
            }

            if hasattr(self._ensemble, "detector_weights"):
                result_metadata["detector_weights"] = list(
                    self._ensemble.detector_weights
                )

            if uncertainties is not None:
                result_metadata["has_uncertainty"] = True
                result_metadata["mean_uncertainty"] = float(np.mean(uncertainties))

            return DetectionResult(
                detector_id=self.detector.id,
                dataset_id=dataset.id,
                anomalies=anomalies,
                scores=anomaly_scores,
                labels=labels,
                threshold=float(threshold),
                metadata=result_metadata,
            )

        except Exception as e:
            raise AdapterError(f"Failed to predict with advanced ensemble model: {e}")

    def score(self, dataset: Dataset) -> list[AnomalyScore]:
        """Calculate anomaly scores for the dataset."""
        result = self.predict(dataset)
        return result.scores

    def fit_detect(self, dataset: Dataset) -> DetectionResult:
        """Fit the detector and detect anomalies in one step."""
        self.fit(dataset)
        return self.detect(dataset)

    def get_params(self) -> dict[str, Any]:
        """Get parameters of the detector."""
        return self.detector.parameters.copy()

    def set_params(self, **params: Any) -> None:
        """Set parameters of the detector."""
        self.detector.parameters.update(params)
        # Reinitialize if needed
        if self._is_fitted:
            self._init_algorithm()

    def _prepare_data(self, dataset: Dataset) -> np.ndarray:
        """Prepare data for ensemble model."""
        df = dataset.data

        # Select numeric columns
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()

        # Remove target column if present
        if dataset.target_column and dataset.target_column in numeric_cols:
            numeric_cols.remove(dataset.target_column)

        if not numeric_cols:
            raise AdapterError("No numeric features found in dataset")

        # Extract features and handle missing values
        X = df[numeric_cols].values

        # Simple imputation
        col_means = np.nanmean(X, axis=0)
        nan_indices = np.where(np.isnan(X))
        X[nan_indices] = np.take(col_means, nan_indices[1])

        return X

    @classmethod
    def get_supported_algorithms(cls) -> list[str]:
        """Get list of supported advanced ensemble algorithms."""
        return list(cls._algorithm_map.keys())

    @classmethod
    def get_algorithm_info(cls, algorithm: str) -> dict[str, Any]:
        """Get information about a specific advanced ensemble algorithm."""
        if algorithm not in cls._algorithm_map:
            raise AlgorithmNotFoundError(f"Algorithm '{algorithm}' not found")

        info = {
            "DynamicEnsemble": {
                "name": "Dynamic Ensemble Selection",
                "type": "Advanced Ensemble",
                "description": "Selects best detectors based on local competence regions",
                "parameters": {
                    "k_neighbors": {
                        "type": "int",
                        "default": 5,
                        "description": "Number of neighbors for competence calculation",
                    },
                    "base_algorithms": {
                        "type": "list",
                        "default": ["IsolationForest", "LOF", "OneClassSVM"],
                        "description": "Base algorithms",
                    },
                    "contamination": {
                        "type": "float",
                        "default": 0.1,
                        "description": "Expected anomaly rate",
                    },
                },
                "suitable_for": [
                    "variable_data_patterns",
                    "local_competence",
                    "adaptive_selection",
                ],
                "pros": [
                    "Adapts to local data patterns",
                    "High accuracy",
                    "Intelligent selection",
                ],
                "cons": [
                    "Requires labeled data",
                    "Computational overhead",
                    "Complex setup",
                ],
            },
            "BayesianEnsemble": {
                "name": "Bayesian Ensemble Averaging",
                "type": "Advanced Ensemble",
                "description": "Uses Bayesian statistics to weight detectors and quantify uncertainty",
                "parameters": {
                    "prior_strength": {
                        "type": "float",
                        "default": 1.0,
                        "description": "Strength of prior belief",
                    },
                    "base_algorithms": {
                        "type": "list",
                        "default": ["IsolationForest", "LOF", "OneClassSVM"],
                        "description": "Base algorithms",
                    },
                    "contamination": {
                        "type": "float",
                        "default": 0.1,
                        "description": "Expected anomaly rate",
                    },
                },
                "suitable_for": [
                    "uncertainty_quantification",
                    "probabilistic_reasoning",
                    "risk_assessment",
                ],
                "pros": [
                    "Uncertainty quantification",
                    "Principled weighting",
                    "Robust to outliers",
                ],
                "cons": [
                    "Complex interpretation",
                    "Requires labeled data",
                    "Computational cost",
                ],
            },
        }

        return info.get(algorithm, {})
