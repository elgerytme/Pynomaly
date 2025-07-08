"""
Drift Detection Adapter for anomaly detection systems.

This module implements various drift detection techniques to monitor
data distribution changes that may affect anomaly detection performance:

- Statistical drift detection (KS test, Chi-square, etc.)
- Distance-based drift detection (Maximum Mean Discrepancy)
- Performance-based drift detection
- Gradual vs. abrupt drift detection
- Multivariate drift detection
- Real-time streaming drift detection
"""

from __future__ import annotations

import logging
import warnings
from abc import ABC, abstractmethod
from dataclasses import dataclass
from datetime import datetime, timedelta
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
from scipy import stats
from scipy.spatial.distance import jensenshannon
from scipy.stats import wasserstein_distance
from sklearn.decomposition import PCA
from sklearn.ensemble import IsolationForest
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from pynomaly.domain.entities import Dataset, DetectionResult, Detector
from pynomaly.domain.exceptions import AdapterError, AlgorithmNotFoundError
from pynomaly.domain.value_objects import AnomalyScore
from pynomaly.shared.protocols import DetectorProtocol

logger = logging.getLogger(__name__)
warnings.filterwarnings("ignore", category=UserWarning)


class DriftType(Enum):
    """Types of drift that can be detected."""

    NO_DRIFT = "no_drift"
    GRADUAL_DRIFT = "gradual_drift"
    ABRUPT_DRIFT = "abrupt_drift"
    INCREMENTAL_DRIFT = "incremental_drift"
    RECURRING_DRIFT = "recurring_drift"
    OUTLIER_DRIFT = "outlier_drift"


class DriftSeverity(Enum):
    """Severity levels for detected drift."""

    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


@dataclass
class DriftDetectionResult:
    """Result of drift detection analysis."""

    drift_detected: bool
    drift_type: DriftType
    drift_severity: DriftSeverity
    drift_score: float
    p_value: float
    features_affected: List[str]
    detection_method: str
    timestamp: datetime
    details: Dict[str, Any]
    recommendations: List[str]


class StatisticalDriftDetector:
    """Statistical tests for drift detection."""

    def __init__(self, significance_level: float = 0.05):
        self.significance_level = significance_level
        self._reference_distributions = {}

    def fit_reference(
        self, X_reference: np.ndarray, feature_names: List[str] = None
    ) -> None:
        """Fit reference distributions for drift detection.

        Args:
            X_reference: Reference data to learn distributions from
            feature_names: Names of features
        """
        if feature_names is None:
            feature_names = [f"feature_{i}" for i in range(X_reference.shape[1])]

        self._reference_distributions = {}

        for i, feature_name in enumerate(feature_names):
            feature_data = X_reference[:, i]

            # Store reference statistics
            self._reference_distributions[feature_name] = {
                "mean": np.mean(feature_data),
                "std": np.std(feature_data),
                "median": np.median(feature_data),
                "q25": np.percentile(feature_data, 25),
                "q75": np.percentile(feature_data, 75),
                "data": feature_data.copy(),
                "distribution_type": self._detect_distribution_type(feature_data),
            }

    def _detect_distribution_type(self, data: np.ndarray) -> str:
        """Detect the most likely distribution type."""
        # Simple distribution detection using statistical tests
        _, normal_p = stats.normaltest(data)
        _, uniform_p = stats.kstest(data, "uniform")

        if normal_p > 0.05:
            return "normal"
        elif uniform_p > 0.05:
            return "uniform"
        else:
            return "unknown"

    def detect_drift_ks_test(
        self, X_current: np.ndarray, feature_names: List[str] = None
    ) -> DriftDetectionResult:
        """Detect drift using Kolmogorov-Smirnov test.

        Args:
            X_current: Current data to test for drift
            feature_names: Names of features

        Returns:
            Drift detection result
        """
        if feature_names is None:
            feature_names = [f"feature_{i}" for i in range(X_current.shape[1])]

        drift_scores = []
        p_values = []
        affected_features = []

        for i, feature_name in enumerate(feature_names):
            if feature_name not in self._reference_distributions:
                continue

            current_data = X_current[:, i]
            reference_data = self._reference_distributions[feature_name]["data"]

            # Perform KS test
            ks_stat, p_value = stats.ks_2samp(reference_data, current_data)

            drift_scores.append(ks_stat)
            p_values.append(p_value)

            if p_value < self.significance_level:
                affected_features.append(feature_name)

        # Aggregate results
        overall_drift_score = np.max(drift_scores) if drift_scores else 0.0
        overall_p_value = np.min(p_values) if p_values else 1.0

        drift_detected = overall_p_value < self.significance_level
        drift_severity = self._calculate_severity(overall_drift_score, overall_p_value)

        return DriftDetectionResult(
            drift_detected=drift_detected,
            drift_type=DriftType.ABRUPT_DRIFT if drift_detected else DriftType.NO_DRIFT,
            drift_severity=drift_severity,
            drift_score=overall_drift_score,
            p_value=overall_p_value,
            features_affected=affected_features,
            detection_method="kolmogorov_smirnov",
            timestamp=datetime.utcnow(),
            details={
                "per_feature_scores": dict(
                    zip(feature_names[: len(drift_scores)], drift_scores)
                ),
                "per_feature_p_values": dict(
                    zip(feature_names[: len(p_values)], p_values)
                ),
                "significance_level": self.significance_level,
            },
            recommendations=self._generate_recommendations(
                drift_detected, affected_features
            ),
        )

    def detect_drift_chi_square(
        self, X_current: np.ndarray, feature_names: List[str] = None, n_bins: int = 10
    ) -> DriftDetectionResult:
        """Detect drift using Chi-square test for categorical/binned features.

        Args:
            X_current: Current data to test for drift
            feature_names: Names of features
            n_bins: Number of bins for continuous features

        Returns:
            Drift detection result
        """
        if feature_names is None:
            feature_names = [f"feature_{i}" for i in range(X_current.shape[1])]

        drift_scores = []
        p_values = []
        affected_features = []

        for i, feature_name in enumerate(feature_names):
            if feature_name not in self._reference_distributions:
                continue

            current_data = X_current[:, i]
            reference_data = self._reference_distributions[feature_name]["data"]

            # Create bins
            combined_data = np.concatenate([reference_data, current_data])
            bins = np.histogram_bin_edges(combined_data, bins=n_bins)

            # Calculate histograms
            ref_hist, _ = np.histogram(reference_data, bins=bins)
            curr_hist, _ = np.histogram(current_data, bins=bins)

            # Avoid zero frequencies
            ref_hist = ref_hist + 1
            curr_hist = curr_hist + 1

            # Chi-square test
            chi2_stat, p_value = stats.chisquare(curr_hist, ref_hist)

            drift_scores.append(chi2_stat)
            p_values.append(p_value)

            if p_value < self.significance_level:
                affected_features.append(feature_name)

        # Aggregate results
        overall_drift_score = np.max(drift_scores) if drift_scores else 0.0
        overall_p_value = np.min(p_values) if p_values else 1.0

        drift_detected = overall_p_value < self.significance_level
        drift_severity = self._calculate_severity(overall_drift_score, overall_p_value)

        return DriftDetectionResult(
            drift_detected=drift_detected,
            drift_type=DriftType.ABRUPT_DRIFT if drift_detected else DriftType.NO_DRIFT,
            drift_severity=drift_severity,
            drift_score=overall_drift_score,
            p_value=overall_p_value,
            features_affected=affected_features,
            detection_method="chi_square",
            timestamp=datetime.utcnow(),
            details={
                "per_feature_scores": dict(
                    zip(feature_names[: len(drift_scores)], drift_scores)
                ),
                "per_feature_p_values": dict(
                    zip(feature_names[: len(p_values)], p_values)
                ),
                "n_bins": n_bins,
                "significance_level": self.significance_level,
            },
            recommendations=self._generate_recommendations(
                drift_detected, affected_features
            ),
        )

    def _calculate_severity(self, drift_score: float, p_value: float) -> DriftSeverity:
        """Calculate drift severity based on score and p-value."""
        if p_value >= self.significance_level:
            return DriftSeverity.LOW
        elif p_value < 0.001:
            return DriftSeverity.CRITICAL
        elif p_value < 0.01:
            return DriftSeverity.HIGH
        else:
            return DriftSeverity.MEDIUM

    def _generate_recommendations(
        self, drift_detected: bool, affected_features: List[str]
    ) -> List[str]:
        """Generate recommendations based on drift detection results."""
        recommendations = []

        if not drift_detected:
            recommendations.append(
                "No significant drift detected. Continue monitoring."
            )
        else:
            recommendations.append(
                "Drift detected. Consider retraining the anomaly detection model."
            )

            if len(affected_features) > 0:
                recommendations.append(
                    f"Focus on features: {', '.join(affected_features[:5])}"
                )

            if len(affected_features) > len(self._reference_distributions) * 0.5:
                recommendations.append(
                    "Many features affected. Consider comprehensive model retraining."
                )
            else:
                recommendations.append(
                    "Limited features affected. Consider feature-specific adjustments."
                )

        return recommendations


class DistanceBasedDriftDetector:
    """Distance-based drift detection using Maximum Mean Discrepancy (MMD)."""

    def __init__(self, kernel: str = "rbf", gamma: float = 1.0):
        self.kernel = kernel
        self.gamma = gamma
        self._reference_embeddings = None

    def fit_reference(self, X_reference: np.ndarray) -> None:
        """Fit reference embeddings for drift detection.

        Args:
            X_reference: Reference data
        """
        self._reference_embeddings = self._compute_embeddings(X_reference)

    def _compute_embeddings(self, X: np.ndarray) -> np.ndarray:
        """Compute kernel embeddings for data."""
        if self.kernel == "rbf":
            # RBF kernel embeddings
            n_samples = min(X.shape[0], 1000)  # Limit for computational efficiency
            if X.shape[0] > n_samples:
                indices = np.random.choice(X.shape[0], n_samples, replace=False)
                X_sample = X[indices]
            else:
                X_sample = X

            return X_sample
        else:
            return X

    def detect_drift_mmd(
        self, X_current: np.ndarray, bootstrap_samples: int = 1000
    ) -> DriftDetectionResult:
        """Detect drift using Maximum Mean Discrepancy.

        Args:
            X_current: Current data to test for drift
            bootstrap_samples: Number of bootstrap samples for p-value estimation

        Returns:
            Drift detection result
        """
        if self._reference_embeddings is None:
            raise ValueError("Must call fit_reference before drift detection")

        current_embeddings = self._compute_embeddings(X_current)

        # Calculate MMD
        mmd_score = self._calculate_mmd(self._reference_embeddings, current_embeddings)

        # Bootstrap test for p-value
        p_value = self._bootstrap_test(
            self._reference_embeddings, current_embeddings, mmd_score, bootstrap_samples
        )

        drift_detected = p_value < 0.05
        drift_severity = self._calculate_severity_mmd(mmd_score, p_value)

        return DriftDetectionResult(
            drift_detected=drift_detected,
            drift_type=DriftType.ABRUPT_DRIFT if drift_detected else DriftType.NO_DRIFT,
            drift_severity=drift_severity,
            drift_score=mmd_score,
            p_value=p_value,
            features_affected=["multivariate_distribution"],
            detection_method="maximum_mean_discrepancy",
            timestamp=datetime.utcnow(),
            details={
                "kernel": self.kernel,
                "gamma": self.gamma,
                "bootstrap_samples": bootstrap_samples,
                "reference_size": len(self._reference_embeddings),
                "current_size": len(current_embeddings),
            },
            recommendations=self._generate_recommendations_mmd(
                drift_detected, mmd_score
            ),
        )

    def _calculate_mmd(self, X: np.ndarray, Y: np.ndarray) -> float:
        """Calculate Maximum Mean Discrepancy between two samples."""
        if self.kernel == "rbf":
            # RBF kernel MMD
            XX = self._rbf_kernel(X, X)
            YY = self._rbf_kernel(Y, Y)
            XY = self._rbf_kernel(X, Y)

            mmd = np.mean(XX) + np.mean(YY) - 2 * np.mean(XY)
            return max(0, mmd)  # MMD should be non-negative
        else:
            # Linear kernel (simpler)
            mean_X = np.mean(X, axis=0)
            mean_Y = np.mean(Y, axis=0)
            return np.linalg.norm(mean_X - mean_Y)

    def _rbf_kernel(self, X: np.ndarray, Y: np.ndarray) -> np.ndarray:
        """Compute RBF kernel matrix."""
        # Compute pairwise squared distances
        X_norm = np.sum(X**2, axis=1, keepdims=True)
        Y_norm = np.sum(Y**2, axis=1, keepdims=True)

        distances = X_norm + Y_norm.T - 2 * np.dot(X, Y.T)

        # RBF kernel
        return np.exp(-self.gamma * distances)

    def _bootstrap_test(
        self, X: np.ndarray, Y: np.ndarray, observed_mmd: float, n_bootstrap: int
    ) -> float:
        """Bootstrap test for MMD significance."""
        combined = np.vstack([X, Y])
        n_X, n_Y = len(X), len(Y)

        bootstrap_mmds = []

        for _ in range(n_bootstrap):
            # Permute combined data
            indices = np.random.permutation(len(combined))
            perm_X = combined[indices[:n_X]]
            perm_Y = combined[indices[n_X : n_X + n_Y]]

            # Calculate MMD for permuted data
            bootstrap_mmd = self._calculate_mmd(perm_X, perm_Y)
            bootstrap_mmds.append(bootstrap_mmd)

        # Calculate p-value
        p_value = np.mean(np.array(bootstrap_mmds) >= observed_mmd)
        return p_value

    def _calculate_severity_mmd(
        self, mmd_score: float, p_value: float
    ) -> DriftSeverity:
        """Calculate drift severity for MMD."""
        if p_value >= 0.05:
            return DriftSeverity.LOW
        elif mmd_score > 1.0:
            return DriftSeverity.CRITICAL
        elif mmd_score > 0.5:
            return DriftSeverity.HIGH
        else:
            return DriftSeverity.MEDIUM

    def _generate_recommendations_mmd(
        self, drift_detected: bool, mmd_score: float
    ) -> List[str]:
        """Generate recommendations for MMD-based drift detection."""
        recommendations = []

        if not drift_detected:
            recommendations.append("No significant multivariate drift detected.")
        else:
            recommendations.append("Multivariate drift detected in data distribution.")

            if mmd_score > 0.5:
                recommendations.append(
                    "Strong drift signal. Immediate model retraining recommended."
                )
            else:
                recommendations.append(
                    "Moderate drift signal. Monitor closely and consider retraining."
                )

            recommendations.append(
                "Consider feature-wise analysis to identify specific drift sources."
            )

        return recommendations


class PerformanceBasedDriftDetector:
    """Drift detection based on model performance degradation."""

    def __init__(self, baseline_model: Any = None, performance_threshold: float = 0.1):
        self.baseline_model = baseline_model
        self.performance_threshold = performance_threshold
        self._reference_performance = None

    def fit_reference(
        self, X_reference: np.ndarray, y_reference: np.ndarray = None
    ) -> None:
        """Establish reference performance baseline.

        Args:
            X_reference: Reference features
            y_reference: Reference labels (if available)
        """
        if self.baseline_model is None:
            # Use isolation forest as default
            self.baseline_model = IsolationForest(contamination=0.1, random_state=42)

        # Fit model on reference data
        self.baseline_model.fit(X_reference)

        # Calculate reference performance if labels available
        if y_reference is not None:
            predictions = self.baseline_model.predict(X_reference)
            predictions = np.where(predictions == -1, 1, 0)  # Convert to anomaly labels

            self._reference_performance = {
                "accuracy": accuracy_score(y_reference, predictions),
                "precision": precision_score(y_reference, predictions, zero_division=0),
                "recall": recall_score(y_reference, predictions, zero_division=0),
                "f1": f1_score(y_reference, predictions, zero_division=0),
            }

            # Calculate anomaly scores for AUC
            if hasattr(self.baseline_model, "decision_function"):
                scores = -self.baseline_model.decision_function(X_reference)
                try:
                    self._reference_performance["auc"] = roc_auc_score(
                        y_reference, scores
                    )
                except ValueError:
                    self._reference_performance["auc"] = 0.5

    def detect_drift_performance(
        self, X_current: np.ndarray, y_current: np.ndarray = None
    ) -> DriftDetectionResult:
        """Detect drift based on performance degradation.

        Args:
            X_current: Current features
            y_current: Current labels (if available)

        Returns:
            Drift detection result
        """
        if self.baseline_model is None:
            raise ValueError("Baseline model not available. Call fit_reference first.")

        # Get model predictions on current data
        predictions = self.baseline_model.predict(X_current)
        predictions = np.where(predictions == -1, 1, 0)

        current_performance = {}
        performance_degradation = {}

        if y_current is not None and self._reference_performance is not None:
            # Calculate current performance
            current_performance = {
                "accuracy": accuracy_score(y_current, predictions),
                "precision": precision_score(y_current, predictions, zero_division=0),
                "recall": recall_score(y_current, predictions, zero_division=0),
                "f1": f1_score(y_current, predictions, zero_division=0),
            }

            # Calculate AUC if possible
            if hasattr(self.baseline_model, "decision_function"):
                scores = -self.baseline_model.decision_function(X_current)
                try:
                    current_performance["auc"] = roc_auc_score(y_current, scores)
                except ValueError:
                    current_performance["auc"] = 0.5

            # Calculate performance degradation
            for metric in current_performance:
                if metric in self._reference_performance:
                    degradation = (
                        self._reference_performance[metric]
                        - current_performance[metric]
                    )
                    performance_degradation[metric] = degradation

        # Determine drift based on performance degradation
        max_degradation = (
            max(performance_degradation.values()) if performance_degradation else 0
        )
        drift_detected = max_degradation > self.performance_threshold

        # Calculate overall drift score
        drift_score = max_degradation if performance_degradation else 0

        # Estimate p-value based on degradation magnitude
        p_value = max(0.001, 1 - (drift_score / (self.performance_threshold * 2)))

        drift_severity = self._calculate_severity_performance(drift_score)

        return DriftDetectionResult(
            drift_detected=drift_detected,
            drift_type=(
                DriftType.GRADUAL_DRIFT if drift_detected else DriftType.NO_DRIFT
            ),
            drift_severity=drift_severity,
            drift_score=drift_score,
            p_value=p_value,
            features_affected=["model_performance"],
            detection_method="performance_based",
            timestamp=datetime.utcnow(),
            details={
                "reference_performance": self._reference_performance,
                "current_performance": current_performance,
                "performance_degradation": performance_degradation,
                "threshold": self.performance_threshold,
            },
            recommendations=self._generate_recommendations_performance(
                drift_detected, performance_degradation
            ),
        )

    def _calculate_severity_performance(self, drift_score: float) -> DriftSeverity:
        """Calculate severity based on performance degradation."""
        if drift_score <= self.performance_threshold:
            return DriftSeverity.LOW
        elif drift_score > self.performance_threshold * 3:
            return DriftSeverity.CRITICAL
        elif drift_score > self.performance_threshold * 2:
            return DriftSeverity.HIGH
        else:
            return DriftSeverity.MEDIUM

    def _generate_recommendations_performance(
        self, drift_detected: bool, degradation: Dict[str, float]
    ) -> List[str]:
        """Generate recommendations based on performance degradation."""
        recommendations = []

        if not drift_detected:
            recommendations.append("No significant performance degradation detected.")
        else:
            recommendations.append(
                "Performance degradation detected. Model retraining recommended."
            )

            if degradation:
                worst_metric = max(degradation.items(), key=lambda x: x[1])
                recommendations.append(
                    f"Worst affected metric: {worst_metric[0]} (degraded by {worst_metric[1]:.3f})"
                )

            recommendations.append(
                "Consider collecting more recent labeled data for retraining."
            )

        return recommendations


class StreamingDriftDetector:
    """Online drift detection for streaming data."""

    def __init__(self, window_size: int = 1000, adaptation_rate: float = 0.01):
        self.window_size = window_size
        self.adaptation_rate = adaptation_rate
        self._reference_window = []
        self._current_window = []
        self._drift_history = []

    def update(self, X_batch: np.ndarray) -> Optional[DriftDetectionResult]:
        """Update detector with new batch and check for drift.

        Args:
            X_batch: New batch of data

        Returns:
            Drift detection result if drift detected, None otherwise
        """
        # Add to current window
        self._current_window.extend(X_batch.tolist())

        # Maintain window size
        if len(self._current_window) > self.window_size:
            self._current_window = self._current_window[-self.window_size :]

        # Initialize reference if empty
        if not self._reference_window:
            self._reference_window = self._current_window.copy()
            return None

        # Check for drift if we have enough data
        if len(self._current_window) >= self.window_size:
            drift_result = self._detect_streaming_drift()

            if drift_result.drift_detected:
                self._drift_history.append(drift_result)

                # Adapt reference window
                self._adapt_reference_window()

            return drift_result

        return None

    def _detect_streaming_drift(self) -> DriftDetectionResult:
        """Detect drift between reference and current windows."""
        ref_data = np.array(self._reference_window)
        curr_data = np.array(self._current_window)

        # Use Wasserstein distance for each feature
        drift_scores = []
        for i in range(ref_data.shape[1]):
            ref_feature = ref_data[:, i]
            curr_feature = curr_data[:, i]

            # Calculate Wasserstein distance
            distance = wasserstein_distance(ref_feature, curr_feature)
            drift_scores.append(distance)

        # Overall drift score
        overall_drift_score = np.mean(drift_scores)

        # Simple threshold-based detection
        drift_threshold = np.std(ref_data) * 0.5  # Adaptive threshold
        drift_detected = overall_drift_score > drift_threshold

        # Estimate p-value
        p_value = (
            max(0.001, 1 - (overall_drift_score / drift_threshold))
            if drift_threshold > 0
            else 1.0
        )

        drift_severity = self._calculate_severity_streaming(
            overall_drift_score, drift_threshold
        )

        return DriftDetectionResult(
            drift_detected=drift_detected,
            drift_type=DriftType.GRADUAL_DRIFT,
            drift_severity=drift_severity,
            drift_score=overall_drift_score,
            p_value=p_value,
            features_affected=[
                f"feature_{i}"
                for i, score in enumerate(drift_scores)
                if score > drift_threshold
            ],
            detection_method="streaming_wasserstein",
            timestamp=datetime.utcnow(),
            details={
                "per_feature_distances": dict(enumerate(drift_scores)),
                "drift_threshold": drift_threshold,
                "window_size": self.window_size,
                "adaptation_rate": self.adaptation_rate,
            },
            recommendations=self._generate_recommendations_streaming(drift_detected),
        )

    def _adapt_reference_window(self) -> None:
        """Adapt reference window using exponential moving average."""
        if not self._reference_window or not self._current_window:
            return

        ref_data = np.array(self._reference_window)
        curr_data = np.array(self._current_window)

        # Exponential moving average adaptation
        adapted_data = (
            1 - self.adaptation_rate
        ) * ref_data + self.adaptation_rate * curr_data
        self._reference_window = adapted_data.tolist()

    def _calculate_severity_streaming(
        self, drift_score: float, threshold: float
    ) -> DriftSeverity:
        """Calculate severity for streaming drift."""
        if drift_score <= threshold:
            return DriftSeverity.LOW
        elif drift_score > threshold * 3:
            return DriftSeverity.CRITICAL
        elif drift_score > threshold * 2:
            return DriftSeverity.HIGH
        else:
            return DriftSeverity.MEDIUM

    def _generate_recommendations_streaming(self, drift_detected: bool) -> List[str]:
        """Generate recommendations for streaming drift."""
        recommendations = []

        if not drift_detected:
            recommendations.append("No drift detected in current window.")
        else:
            recommendations.append("Drift detected in streaming data.")
            recommendations.append("Consider adaptive model updates or retraining.")

            if len(self._drift_history) > 1:
                recommendations.append(
                    "Multiple drift events detected. Monitor for recurring patterns."
                )

        return recommendations


class DriftDetectionAdapter(DetectorProtocol):
    """Main drift detection adapter combining multiple detection methods."""

    _algorithm_map = {
        "StatisticalDrift": StatisticalDriftDetector,
        "DistanceBasedDrift": DistanceBasedDriftDetector,
        "PerformanceBasedDrift": PerformanceBasedDriftDetector,
        "StreamingDrift": StreamingDriftDetector,
    }

    def __init__(self, detector: Detector):
        """Initialize drift detection adapter.

        Args:
            detector: Detector entity with drift detection configuration
        """
        self.detector = detector
        self._drift_detector = None
        self._scaler = StandardScaler()
        self._is_fitted = False
        self._reference_data = None
        self._init_algorithm()

    def _init_algorithm(self) -> None:
        """Initialize the drift detection algorithm."""
        if self.detector.algorithm_name not in self._algorithm_map:
            available = ", ".join(self._algorithm_map.keys())
            raise AlgorithmNotFoundError(
                f"Algorithm '{self.detector.algorithm_name}' not found. "
                f"Available drift detection algorithms: {available}"
            )

        params = self.detector.parameters.copy()
        detector_class = self._algorithm_map[self.detector.algorithm_name]

        try:
            if self.detector.algorithm_name == "StatisticalDrift":
                significance_level = params.get("significance_level", 0.05)
                self._drift_detector = detector_class(significance_level)

            elif self.detector.algorithm_name == "DistanceBasedDrift":
                kernel = params.get("kernel", "rbf")
                gamma = params.get("gamma", 1.0)
                self._drift_detector = detector_class(kernel, gamma)

            elif self.detector.algorithm_name == "PerformanceBasedDrift":
                threshold = params.get("performance_threshold", 0.1)
                self._drift_detector = detector_class(None, threshold)

            elif self.detector.algorithm_name == "StreamingDrift":
                window_size = params.get("window_size", 1000)
                adaptation_rate = params.get("adaptation_rate", 0.01)
                self._drift_detector = detector_class(window_size, adaptation_rate)

        except Exception as e:
            raise AdapterError(
                f"Failed to initialize drift detector {self.detector.algorithm_name}: {e}"
            )

    @property
    def name(self) -> str:
        """Get the name of the detector."""
        return self.detector.name

    @property
    def contamination_rate(self):
        """Get the contamination rate."""
        from pynomaly.domain.value_objects import ContaminationRate

        # Drift detection doesn't use contamination rate in the traditional sense
        return ContaminationRate(0.0)

    @property
    def is_fitted(self) -> bool:
        """Check if the detector has been fitted."""
        return self._is_fitted

    @property
    def parameters(self) -> dict[str, Any]:
        """Get the current parameters of the detector."""
        return self.detector.parameters.copy()

    def fit(self, dataset: Dataset) -> None:
        """Fit the drift detector on reference data.

        Args:
            dataset: Reference dataset to establish baseline
        """
        try:
            # Prepare data
            X_reference = self._prepare_data(dataset)
            X_scaled = self._scaler.fit_transform(X_reference)

            # Store reference data
            self._reference_data = X_scaled.copy()

            # Get labels if available
            y_reference = None
            if dataset.target_column and dataset.target_column in dataset.data.columns:
                y_reference = dataset.data[dataset.target_column].values

            # Fit drift detector
            feature_names = self._get_feature_names(dataset)

            if hasattr(self._drift_detector, "fit_reference"):
                if y_reference is not None:
                    self._drift_detector.fit_reference(X_scaled, y_reference)
                else:
                    self._drift_detector.fit_reference(X_scaled, feature_names)

            self.detector.is_fitted = True
            self._is_fitted = True

            logger.info(
                f"Successfully fitted drift detector: {self.detector.algorithm_name}"
            )

        except Exception as e:
            raise AdapterError(f"Failed to fit drift detection model: {e}")

    def detect(self, dataset: Dataset) -> DetectionResult:
        """Detect drift in new data.

        Args:
            dataset: Dataset to test for drift

        Returns:
            Detection results with drift information
        """
        return self.predict(dataset)

    def predict(self, dataset: Dataset) -> DetectionResult:
        """Predict drift in new data.

        Args:
            dataset: Dataset to test for drift

        Returns:
            Detection results with drift information
        """
        if not self._is_fitted:
            raise AdapterError("Model must be fitted before prediction")

        try:
            # Prepare data
            X_current = self._prepare_data(dataset)
            X_scaled = self._scaler.transform(X_current)

            # Get labels if available
            y_current = None
            if dataset.target_column and dataset.target_column in dataset.data.columns:
                y_current = dataset.data[dataset.target_column].values

            # Detect drift
            feature_names = self._get_feature_names(dataset)
            drift_result = self._detect_drift(X_scaled, y_current, feature_names)

            # Convert drift result to detection result format
            return self._convert_to_detection_result(drift_result, dataset)

        except Exception as e:
            raise AdapterError(f"Failed to detect drift: {e}")

    def _detect_drift(
        self,
        X_current: np.ndarray,
        y_current: np.ndarray = None,
        feature_names: List[str] = None,
    ) -> DriftDetectionResult:
        """Detect drift using the configured algorithm."""

        if self.detector.algorithm_name == "StatisticalDrift":
            # Try KS test first, fallback to Chi-square
            try:
                return self._drift_detector.detect_drift_ks_test(
                    X_current, feature_names
                )
            except Exception:
                return self._drift_detector.detect_drift_chi_square(
                    X_current, feature_names
                )

        elif self.detector.algorithm_name == "DistanceBasedDrift":
            return self._drift_detector.detect_drift_mmd(X_current)

        elif self.detector.algorithm_name == "PerformanceBasedDrift":
            return self._drift_detector.detect_drift_performance(X_current, y_current)

        elif self.detector.algorithm_name == "StreamingDrift":
            result = self._drift_detector.update(X_current)
            if result is None:
                # No drift detected yet
                return DriftDetectionResult(
                    drift_detected=False,
                    drift_type=DriftType.NO_DRIFT,
                    drift_severity=DriftSeverity.LOW,
                    drift_score=0.0,
                    p_value=1.0,
                    features_affected=[],
                    detection_method="streaming_wasserstein",
                    timestamp=datetime.utcnow(),
                    details={"status": "accumulating_data"},
                    recommendations=["Accumulating data for drift detection."],
                )
            return result

        else:
            raise AdapterError(
                f"Unknown drift detection algorithm: {self.detector.algorithm_name}"
            )

    def _convert_to_detection_result(
        self, drift_result: DriftDetectionResult, dataset: Dataset
    ) -> DetectionResult:
        """Convert drift detection result to standard detection result format."""

        # Create anomaly scores (drift score for each sample)
        n_samples = len(dataset.data)
        drift_scores = [drift_result.drift_score] * n_samples

        anomaly_scores = [
            AnomalyScore(value=float(score), method=self.detector.algorithm_name)
            for score in drift_scores
        ]

        # Create anomalies if drift detected
        anomalies = []
        labels = np.zeros(n_samples, dtype=int)

        if drift_result.drift_detected:
            # Mark all samples as potential drift samples
            labels = np.ones(n_samples, dtype=int)

            from pynomaly.domain.entities.anomaly import Anomaly

            # Create representative anomaly (not per-sample for drift)
            anomaly = Anomaly(
                score=AnomalyScore(
                    value=drift_result.drift_score, method=self.detector.algorithm_name
                ),
                data_point={"drift_type": drift_result.drift_type.value},
                detector_name=self.detector.algorithm_name,
                metadata={
                    "drift_detected": True,
                    "drift_type": drift_result.drift_type.value,
                    "drift_severity": drift_result.drift_severity.value,
                    "features_affected": drift_result.features_affected,
                    "detection_method": drift_result.detection_method,
                    "p_value": drift_result.p_value,
                    "recommendations": drift_result.recommendations,
                },
            )
            anomalies.append(anomaly)

        # Prepare metadata
        metadata = {
            "algorithm": self.detector.algorithm_name,
            "drift_detected": drift_result.drift_detected,
            "drift_type": drift_result.drift_type.value,
            "drift_severity": drift_result.drift_severity.value,
            "drift_score": drift_result.drift_score,
            "p_value": drift_result.p_value,
            "features_affected": drift_result.features_affected,
            "detection_method": drift_result.detection_method,
            "timestamp": drift_result.timestamp.isoformat(),
            "recommendations": drift_result.recommendations,
            "details": drift_result.details,
            "model_type": "drift_detection",
        }

        return DetectionResult(
            detector_id=self.detector.id,
            dataset_id=dataset.id,
            anomalies=anomalies,
            scores=anomaly_scores,
            labels=labels,
            threshold=0.5,  # Not meaningful for drift detection
            metadata=metadata,
        )

    def score(self, dataset: Dataset) -> list[AnomalyScore]:
        """Calculate drift scores for the dataset."""
        result = self.predict(dataset)
        return result.scores

    def fit_detect(self, dataset: Dataset) -> DetectionResult:
        """Fit the detector and detect drift in one step."""
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
        """Prepare data for drift detection."""
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

    def _get_feature_names(self, dataset: Dataset) -> List[str]:
        """Get feature names from dataset."""
        df = dataset.data
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()

        # Remove target column if present
        if dataset.target_column and dataset.target_column in numeric_cols:
            numeric_cols.remove(dataset.target_column)

        return numeric_cols

    @classmethod
    def get_supported_algorithms(cls) -> list[str]:
        """Get list of supported drift detection algorithms."""
        return list(cls._algorithm_map.keys())

    @classmethod
    def get_algorithm_info(cls, algorithm: str) -> dict[str, Any]:
        """Get information about a specific drift detection algorithm."""
        if algorithm not in cls._algorithm_map:
            raise AlgorithmNotFoundError(f"Algorithm '{algorithm}' not found")

        info = {
            "StatisticalDrift": {
                "name": "Statistical Drift Detection",
                "type": "Statistical Test",
                "description": "Detects drift using statistical tests (KS test, Chi-square)",
                "parameters": {
                    "significance_level": {
                        "type": "float",
                        "default": 0.05,
                        "description": "Statistical significance level",
                    },
                    "method": {
                        "type": "str",
                        "default": "ks_test",
                        "description": "Statistical test method",
                    },
                },
                "suitable_for": [
                    "univariate_drift",
                    "distribution_changes",
                    "feature_level_analysis",
                ],
                "pros": [
                    "Statistical rigor",
                    "Interpretable results",
                    "Well-established theory",
                ],
                "cons": [
                    "Requires sufficient data",
                    "May miss subtle drift",
                    "Assumes independence",
                ],
            },
            "DistanceBasedDrift": {
                "name": "Distance-Based Drift Detection",
                "type": "Distance Metric",
                "description": "Uses Maximum Mean Discrepancy (MMD) to detect multivariate drift",
                "parameters": {
                    "kernel": {
                        "type": "str",
                        "default": "rbf",
                        "description": "Kernel type for MMD",
                    },
                    "gamma": {
                        "type": "float",
                        "default": 1.0,
                        "description": "Kernel parameter",
                    },
                },
                "suitable_for": [
                    "multivariate_drift",
                    "complex_dependencies",
                    "high_dimensional_data",
                ],
                "pros": [
                    "Multivariate detection",
                    "Kernel flexibility",
                    "Captures complex patterns",
                ],
                "cons": [
                    "Computational complexity",
                    "Parameter tuning needed",
                    "Less interpretable",
                ],
            },
            "PerformanceBasedDrift": {
                "name": "Performance-Based Drift Detection",
                "type": "Model Performance",
                "description": "Detects drift based on model performance degradation",
                "parameters": {
                    "performance_threshold": {
                        "type": "float",
                        "default": 0.1,
                        "description": "Performance degradation threshold",
                    },
                    "metrics": {
                        "type": "list",
                        "default": ["accuracy", "f1"],
                        "description": "Performance metrics to monitor",
                    },
                },
                "suitable_for": [
                    "concept_drift",
                    "model_monitoring",
                    "supervised_learning",
                ],
                "pros": [
                    "Direct business impact",
                    "Model-agnostic",
                    "Actionable insights",
                ],
                "cons": [
                    "Requires labeled data",
                    "Delayed detection",
                    "Model-dependent",
                ],
            },
            "StreamingDrift": {
                "name": "Streaming Drift Detection",
                "type": "Online Detection",
                "description": "Online drift detection for streaming data with adaptive windows",
                "parameters": {
                    "window_size": {
                        "type": "int",
                        "default": 1000,
                        "description": "Size of sliding window",
                    },
                    "adaptation_rate": {
                        "type": "float",
                        "default": 0.01,
                        "description": "Rate of adaptation to new data",
                    },
                },
                "suitable_for": [
                    "streaming_data",
                    "real_time_monitoring",
                    "continuous_adaptation",
                ],
                "pros": [
                    "Real-time detection",
                    "Adaptive to changes",
                    "Memory efficient",
                ],
                "cons": [
                    "Parameter sensitive",
                    "Limited history",
                    "May miss long-term trends",
                ],
            },
        }

        return info.get(algorithm, {})
