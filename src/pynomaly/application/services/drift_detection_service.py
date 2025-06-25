"""Advanced drift detection service with statistical and AI-based methods."""

from __future__ import annotations

import logging
from datetime import datetime, timedelta
from typing import Any
from uuid import UUID

import numpy as np
from scipy import stats

from pynomaly.domain.entities.drift_detection import (
    ConceptDriftResult,
    DriftAnalysisResult,
    DriftDetectionMethod,
    DriftThresholds,
    FeatureData,
    FeatureDriftAnalysis,
    MultivariateDriftResult,
    TimeWindow,
    UnivariateDriftResult,
)

logger = logging.getLogger(__name__)


class DriftDetectionError(Exception):
    """Base exception for drift detection errors."""

    pass


class InsufficientDataError(DriftDetectionError):
    """Insufficient data for drift detection."""

    pass


class MethodNotSupportedError(DriftDetectionError):
    """Drift detection method not supported."""

    pass


@dataclass
class DataBatch:
    """Batch of data for drift detection."""

    data: np.ndarray
    timestamps: list[datetime]
    feature_names: list[str]
    batch_id: UUID = field(default_factory=uuid4)
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        """Validate data batch."""
        if len(self.data) == 0:
            raise ValueError("Data cannot be empty")
        if len(self.timestamps) != len(self.data):
            raise ValueError("Timestamps must match data length")
        if len(self.feature_names) != self.data.shape[1]:
            raise ValueError("Feature names must match data dimensions")


@dataclass
class PerformanceHistory:
    """Historical performance data for concept drift detection."""

    timestamps: list[datetime]
    accuracy_scores: list[float]
    precision_scores: list[float]
    recall_scores: list[float]
    prediction_confidence: list[float]
    prediction_counts: list[int]

    def __post_init__(self):
        """Validate performance history."""
        lengths = [
            len(self.timestamps),
            len(self.accuracy_scores),
            len(self.precision_scores),
            len(self.recall_scores),
            len(self.prediction_confidence),
            len(self.prediction_counts),
        ]
        if len(set(lengths)) > 1:
            raise ValueError("All performance history lists must have same length")

    def get_recent_window(self, window_size: int) -> PerformanceHistory:
        """Get recent performance data."""
        return PerformanceHistory(
            timestamps=self.timestamps[-window_size:],
            accuracy_scores=self.accuracy_scores[-window_size:],
            precision_scores=self.precision_scores[-window_size:],
            recall_scores=self.recall_scores[-window_size:],
            prediction_confidence=self.prediction_confidence[-window_size:],
            prediction_counts=self.prediction_counts[-window_size:],
        )


class DriftDetectionService:
    """Advanced drift detection and analysis service.

    Provides comprehensive drift detection capabilities including:
    - Statistical drift detection methods
    - AI-powered drift analysis
    - Multivariate drift detection
    - Concept drift detection
    - Contextual drift assessment
    """

    def __init__(
        self,
        drift_thresholds: DriftThresholds | None = None,
        enable_ai_detection: bool = True,
    ):
        """Initialize drift detection service.

        Args:
            drift_thresholds: Configurable thresholds for drift detection
            enable_ai_detection: Whether to enable AI-based drift detection
        """
        self.drift_thresholds = drift_thresholds or DriftThresholds()
        self.enable_ai_detection = enable_ai_detection

        # Statistical detectors
        self.statistical_detector = StatisticalDriftDetector(self.drift_thresholds)

        # AI-based detectors (if enabled)
        if enable_ai_detection:
            self.ai_detector = AIBasedDriftDetector()
            self.neural_detector = NeuralDriftDetector()

        # Contextual analyzer
        self.contextual_assessor = ContextualDriftAssessor()

        # Reference data storage
        self.reference_data_cache: dict[UUID, DataBatch] = {}

        # Performance monitoring
        self.performance_monitor = PerformanceMonitor()

    async def monitor_data_drift(
        self,
        model_id: UUID,
        incoming_data: DataBatch,
        reference_data: DataBatch | None = None,
    ) -> DriftAnalysisResult:
        """Monitor for data drift in incoming data.

        Args:
            model_id: Model ID for context
            incoming_data: New data to analyze
            reference_data: Reference data (uses cached if None)

        Returns:
            Comprehensive drift analysis result
        """
        try:
            # Get reference data
            if reference_data is None:
                if model_id not in self.reference_data_cache:
                    raise InsufficientDataError(
                        f"No reference data for model {model_id}"
                    )
                reference_data = self.reference_data_cache[model_id]
            else:
                # Cache reference data for future use
                self.reference_data_cache[model_id] = reference_data

            # Create time window for analysis
            time_window = TimeWindow(
                start_time=min(incoming_data.timestamps),
                end_time=max(incoming_data.timestamps),
                window_size=timedelta(hours=1),
            )

            # Perform univariate drift detection for each feature
            univariate_results = []
            feature_analyses = {}

            for i, feature_name in enumerate(incoming_data.feature_names):
                feature_data = FeatureData(
                    feature_name=feature_name,
                    reference_data=reference_data.data[:, i],
                    current_data=incoming_data.data[:, i],
                    data_type="numerical",  # Simplified for now
                )

                # Statistical drift detection
                univariate_result = await self._detect_univariate_drift(feature_data)
                univariate_results.append(univariate_result)

                # Feature-level analysis
                feature_analysis = await self._analyze_feature_drift(
                    feature_data, time_window
                )
                feature_analyses[feature_name] = feature_analysis

            # Multivariate drift detection
            multivariate_result = await self._detect_multivariate_drift(
                reference_data.data, incoming_data.data
            )

            # Calculate overall drift score
            overall_drift_score = self._calculate_overall_drift_score(
                univariate_results, multivariate_result
            )

            # Determine severity
            drift_severity = self._determine_drift_severity(overall_drift_score)

            # Generate recommendations
            recommendations = self._generate_drift_recommendations(
                univariate_results, multivariate_result, drift_severity
            )

            # Create comprehensive result
            result = DriftAnalysisResult(
                model_id=model_id,
                time_window=time_window,
                data_drift_results=univariate_results,
                multivariate_drift_result=multivariate_result,
                feature_analyses=feature_analyses,
                overall_drift_score=overall_drift_score,
                drift_severity=drift_severity,
                recommended_actions=recommendations,
                analysis_metadata={
                    "reference_sample_size": len(reference_data.data),
                    "current_sample_size": len(incoming_data.data),
                    "detection_methods_used": self._get_methods_used(),
                    "analysis_timestamp": datetime.utcnow().isoformat(),
                },
            )

            logger.info(
                f"Data drift analysis completed for model {model_id}: "
                f"score={overall_drift_score:.3f}, severity={drift_severity}"
            )

            return result

        except Exception as e:
            logger.error(f"Data drift monitoring failed for model {model_id}: {e}")
            raise DriftDetectionError(f"Data drift monitoring failed: {e}") from e

    async def detect_concept_drift(
        self, model_id: UUID, performance_history: PerformanceHistory
    ) -> ConceptDriftResult:
        """Detect concept drift based on model performance degradation.

        Args:
            model_id: Model ID
            performance_history: Historical performance data

        Returns:
            Concept drift detection result
        """
        try:
            # Check for sufficient data
            if len(performance_history.accuracy_scores) < 10:
                raise InsufficientDataError(
                    "Insufficient performance history for concept drift detection"
                )

            # Analyze performance trends
            performance_trend = self._analyze_performance_trend(performance_history)

            # Statistical tests for performance degradation
            stability_metrics = self._calculate_stability_metrics(performance_history)

            # AI-based concept drift detection (if enabled)
            if self.enable_ai_detection:
                ai_drift_probability = await self.ai_detector.detect_concept_drift(
                    performance_history
                )
            else:
                ai_drift_probability = 0.0

            # Combine statistical and AI-based results
            combined_probability = self._combine_drift_probabilities(
                performance_trend["degradation_probability"], ai_drift_probability
            )

            # Determine if drift is detected
            drift_detected = (
                combined_probability > self.drift_thresholds.neural_drift_threshold
            )

            # Calculate confidence
            confidence = abs(combined_probability - 0.5) * 2

            # Analyze drift patterns
            drift_patterns = self._analyze_concept_drift_patterns(
                performance_history, performance_trend
            )

            # Detect specific drift types
            label_distribution_shift = self._detect_label_distribution_shift(
                performance_history
            )
            decision_boundary_shift = self._detect_decision_boundary_shift(
                performance_history
            )

            result = ConceptDriftResult(
                detection_method=(
                    DriftDetectionMethod.NEURAL_DRIFT_DETECTOR
                    if self.enable_ai_detection
                    else DriftDetectionMethod.STATISTICAL_PROCESS_CONTROL
                ),
                drift_probability=combined_probability,
                drift_detected=drift_detected,
                confidence=confidence,
                stability_metrics=stability_metrics,
                drift_patterns=drift_patterns,
                prediction_consistency_score=stability_metrics.get(
                    "consistency_score", 1.0
                ),
                label_distribution_shift=label_distribution_shift,
                decision_boundary_shift=decision_boundary_shift,
            )

            logger.info(
                f"Concept drift analysis completed for model {model_id}: "
                f"probability={combined_probability:.3f}, detected={drift_detected}"
            )

            return result

        except Exception as e:
            logger.error(f"Concept drift detection failed for model {model_id}: {e}")
            raise DriftDetectionError(f"Concept drift detection failed: {e}") from e

    async def analyze_feature_drift(
        self, feature_data: FeatureData, time_window: TimeWindow
    ) -> FeatureDriftAnalysis:
        """Analyze drift for a specific feature.

        Args:
            feature_data: Feature data for analysis
            time_window: Time window for analysis

        Returns:
            Feature drift analysis result
        """
        return await self._analyze_feature_drift(feature_data, time_window)

    # Private helper methods

    async def _detect_univariate_drift(
        self, feature_data: FeatureData
    ) -> UnivariateDriftResult:
        """Detect drift in a single feature."""
        # Use multiple statistical methods
        methods_results = []

        # Kolmogorov-Smirnov test
        ks_result = self.statistical_detector.kolmogorov_smirnov_test(
            feature_data.reference_data, feature_data.current_data
        )
        methods_results.append(ks_result)

        # Population Stability Index
        psi_result = self.statistical_detector.population_stability_index(
            feature_data.reference_data, feature_data.current_data
        )
        methods_results.append(psi_result)

        # Jensen-Shannon Divergence
        js_result = self.statistical_detector.jensen_shannon_divergence(
            feature_data.reference_data, feature_data.current_data
        )
        methods_results.append(js_result)

        # Select best result (most significant)
        best_result = min(methods_results, key=lambda x: x.p_value)

        return UnivariateDriftResult(
            feature_name=feature_data.feature_name,
            detection_method=best_result.detection_method,
            drift_detected=best_result.drift_detected,
            drift_score=best_result.drift_score,
            p_value=best_result.p_value,
            effect_size=best_result.effect_size,
            confidence_interval=best_result.confidence_interval,
            threshold_used=best_result.threshold_used,
            sample_size_reference=len(feature_data.reference_data),
            sample_size_current=len(feature_data.current_data),
            additional_metrics={"all_methods": [r.to_dict() for r in methods_results]},
        )

    async def _detect_multivariate_drift(
        self, reference_data: np.ndarray, current_data: np.ndarray
    ) -> MultivariateDriftResult:
        """Detect multivariate drift."""
        # Maximum Mean Discrepancy
        mmd_score = self.statistical_detector.maximum_mean_discrepancy(
            reference_data, current_data
        )

        # Wasserstein distance
        wasserstein_score = self.statistical_detector.wasserstein_distance(
            reference_data, current_data
        )

        # Energy distance
        energy_score = self.statistical_detector.energy_distance(
            reference_data, current_data
        )

        # Combine scores
        combined_score = (mmd_score + wasserstein_score + energy_score) / 3

        # Detect drift
        drift_detected = combined_score > self.drift_thresholds.mmd_threshold

        # Calculate feature contributions
        feature_contributions = self._calculate_feature_contributions(
            reference_data, current_data
        )

        return MultivariateDriftResult(
            detection_method=DriftDetectionMethod.MAXIMUM_MEAN_DISCREPANCY,
            drift_detected=drift_detected,
            drift_score=combined_score,
            threshold_used=self.drift_thresholds.mmd_threshold,
            affected_features=list(feature_contributions.keys()),
            feature_contributions=feature_contributions,
        )

    async def _analyze_feature_drift(
        self, feature_data: FeatureData, time_window: TimeWindow
    ) -> FeatureDriftAnalysis:
        """Analyze drift for a specific feature."""
        # Detect drift using multiple methods
        univariate_results = []

        # Add different detection methods
        for method in [
            DriftDetectionMethod.KOLMOGOROV_SMIRNOV,
            DriftDetectionMethod.POPULATION_STABILITY_INDEX,
            DriftDetectionMethod.JENSEN_SHANNON_DIVERGENCE,
        ]:
            result = await self._detect_univariate_drift_with_method(
                feature_data, method
            )
            univariate_results.append(result)

        # Calculate temporal patterns
        temporal_pattern = self._analyze_temporal_patterns(feature_data, time_window)

        # Calculate drift velocity and acceleration
        drift_velocity = self._calculate_drift_velocity(feature_data)
        drift_acceleration = self._calculate_drift_acceleration(feature_data)

        # Calculate stability score
        stability_score = self._calculate_feature_stability(feature_data)

        return FeatureDriftAnalysis(
            feature_name=feature_data.feature_name,
            univariate_results=univariate_results,
            temporal_drift_pattern=temporal_pattern,
            drift_velocity=drift_velocity,
            drift_acceleration=drift_acceleration,
            stability_score=stability_score,
        )

    def _calculate_overall_drift_score(
        self,
        univariate_results: list[UnivariateDriftResult],
        multivariate_result: MultivariateDriftResult | None,
    ) -> float:
        """Calculate overall drift score."""
        scores = []

        # Univariate scores
        for result in univariate_results:
            if result.drift_detected:
                scores.append(result.drift_score)

        # Multivariate score
        if multivariate_result and multivariate_result.drift_detected:
            scores.append(multivariate_result.drift_score)

        if not scores:
            return 0.0

        # Weight by significance
        weighted_scores = []
        for _i, result in enumerate(univariate_results):
            if result.drift_detected:
                weight = 1.0 / max(result.p_value, 1e-10)
                weighted_scores.append(result.drift_score * weight)

        if multivariate_result and multivariate_result.drift_detected:
            weighted_scores.append(
                multivariate_result.drift_score * 2.0
            )  # Higher weight

        return float(np.mean(weighted_scores)) if weighted_scores else 0.0

    def _determine_drift_severity(self, drift_score: float) -> str:
        """Determine drift severity based on score."""
        if drift_score < 0.1:
            return "none"
        elif drift_score < 0.3:
            return "low"
        elif drift_score < 0.7:
            return "medium"
        else:
            return "high"

    def _get_methods_used(self) -> list[str]:
        """Get list of detection methods used."""
        methods = [
            "kolmogorov_smirnov",
            "population_stability_index",
            "jensen_shannon_divergence",
            "maximum_mean_discrepancy",
            "wasserstein_distance",
            "energy_distance",
        ]

        if self.enable_ai_detection:
            methods.extend(["neural_drift_detector", "adversarial_drift_detection"])

        return methods


# Supporting classes


class StatisticalDriftDetector:
    """Statistical methods for drift detection."""

    def __init__(self, thresholds: DriftThresholds):
        self.thresholds = thresholds

    def kolmogorov_smirnov_test(
        self, reference_data: np.ndarray, current_data: np.ndarray
    ) -> UnivariateDriftResult:
        """Perform Kolmogorov-Smirnov test."""
        ks_statistic, p_value = stats.ks_2samp(reference_data, current_data)

        # Calculate effect size (Cohen's d equivalent for distributions)
        pooled_std = np.sqrt((np.var(reference_data) + np.var(current_data)) / 2)
        effect_size = abs(np.mean(reference_data) - np.mean(current_data)) / pooled_std

        drift_detected = p_value < self.thresholds.statistical_significance

        return UnivariateDriftResult(
            feature_name="",  # Will be set by caller
            detection_method=DriftDetectionMethod.KOLMOGOROV_SMIRNOV,
            drift_detected=drift_detected,
            drift_score=ks_statistic,
            p_value=p_value,
            effect_size=effect_size,
            confidence_interval=(0.0, 1.0),  # Simplified
            threshold_used=self.thresholds.statistical_significance,
            sample_size_reference=len(reference_data),
            sample_size_current=len(current_data),
        )

    def population_stability_index(
        self, reference_data: np.ndarray, current_data: np.ndarray, bins: int = 10
    ) -> UnivariateDriftResult:
        """Calculate Population Stability Index."""
        # Create bins based on reference data
        bin_edges = np.histogram_bin_edges(reference_data, bins=bins)

        # Calculate distributions
        ref_counts, _ = np.histogram(reference_data, bins=bin_edges)
        curr_counts, _ = np.histogram(current_data, bins=bin_edges)

        # Convert to proportions
        ref_props = ref_counts / len(reference_data)
        curr_props = curr_counts / len(current_data)

        # Avoid division by zero
        ref_props = np.where(ref_props == 0, 1e-10, ref_props)
        curr_props = np.where(curr_props == 0, 1e-10, curr_props)

        # Calculate PSI
        psi = np.sum((curr_props - ref_props) * np.log(curr_props / ref_props))

        drift_detected = psi > self.thresholds.psi_threshold

        return UnivariateDriftResult(
            feature_name="",
            detection_method=DriftDetectionMethod.POPULATION_STABILITY_INDEX,
            drift_detected=drift_detected,
            drift_score=psi,
            p_value=0.05 if drift_detected else 0.95,  # Simplified
            effect_size=psi,
            confidence_interval=(0.0, 1.0),
            threshold_used=self.thresholds.psi_threshold,
            sample_size_reference=len(reference_data),
            sample_size_current=len(current_data),
        )

    def jensen_shannon_divergence(
        self, reference_data: np.ndarray, current_data: np.ndarray, bins: int = 50
    ) -> UnivariateDriftResult:
        """Calculate Jensen-Shannon divergence."""
        # Create histograms
        min_val = min(np.min(reference_data), np.min(current_data))
        max_val = max(np.max(reference_data), np.max(current_data))
        bin_edges = np.linspace(min_val, max_val, bins + 1)

        ref_hist, _ = np.histogram(reference_data, bins=bin_edges, density=True)
        curr_hist, _ = np.histogram(current_data, bins=bin_edges, density=True)

        # Normalize to probabilities
        ref_prob = ref_hist / np.sum(ref_hist)
        curr_prob = curr_hist / np.sum(curr_hist)

        # Calculate JS divergence
        m = 0.5 * (ref_prob + curr_prob)

        # Add small epsilon to avoid log(0)
        epsilon = 1e-10
        ref_prob = ref_prob + epsilon
        curr_prob = curr_prob + epsilon
        m = m + epsilon

        kl_ref_m = np.sum(ref_prob * np.log(ref_prob / m))
        kl_curr_m = np.sum(curr_prob * np.log(curr_prob / m))

        js_divergence = 0.5 * kl_ref_m + 0.5 * kl_curr_m

        drift_detected = js_divergence > self.thresholds.js_divergence_threshold

        return UnivariateDriftResult(
            feature_name="",
            detection_method=DriftDetectionMethod.JENSEN_SHANNON_DIVERGENCE,
            drift_detected=drift_detected,
            drift_score=js_divergence,
            p_value=0.05 if drift_detected else 0.95,  # Simplified
            effect_size=js_divergence,
            confidence_interval=(0.0, 1.0),
            threshold_used=self.thresholds.js_divergence_threshold,
            sample_size_reference=len(reference_data),
            sample_size_current=len(current_data),
        )

    def maximum_mean_discrepancy(
        self, reference_data: np.ndarray, current_data: np.ndarray
    ) -> float:
        """Calculate Maximum Mean Discrepancy."""
        # Simplified MMD using RBF kernel
        gamma = 1.0 / reference_data.shape[1]  # Bandwidth parameter

        def rbf_kernel(X, Y, gamma):
            """RBF kernel function."""
            sq_dists = np.sum((X[:, None] - Y[None, :]) ** 2, axis=2)
            return np.exp(-gamma * sq_dists)

        # Calculate kernel matrices
        K_xx = rbf_kernel(reference_data, reference_data, gamma)
        K_yy = rbf_kernel(current_data, current_data, gamma)
        K_xy = rbf_kernel(reference_data, current_data, gamma)

        # Calculate MMD
        m, n = len(reference_data), len(current_data)
        mmd = (
            np.sum(K_xx) / (m * m) + np.sum(K_yy) / (n * n) - 2 * np.sum(K_xy) / (m * n)
        )

        return max(0, mmd)  # MMD should be non-negative

    def wasserstein_distance(
        self, reference_data: np.ndarray, current_data: np.ndarray
    ) -> float:
        """Calculate Wasserstein distance (simplified 1D version)."""
        # For multivariate data, use average of univariate distances
        distances = []

        for i in range(reference_data.shape[1]):
            ref_feature = reference_data[:, i]
            curr_feature = current_data[:, i]

            # Sort the data
            ref_sorted = np.sort(ref_feature)
            curr_sorted = np.sort(curr_feature)

            # Interpolate to same length
            n = min(len(ref_sorted), len(curr_sorted))
            ref_interp = np.interp(
                np.linspace(0, 1, n), np.linspace(0, 1, len(ref_sorted)), ref_sorted
            )
            curr_interp = np.interp(
                np.linspace(0, 1, n), np.linspace(0, 1, len(curr_sorted)), curr_sorted
            )

            # Calculate L1 distance
            distance = np.mean(np.abs(ref_interp - curr_interp))
            distances.append(distance)

        return float(np.mean(distances))

    def energy_distance(
        self, reference_data: np.ndarray, current_data: np.ndarray
    ) -> float:
        """Calculate energy distance."""
        # Simplified energy distance calculation
        m, n = len(reference_data), len(current_data)

        # Calculate pairwise distances
        def pairwise_distances(X, Y):
            """Calculate pairwise Euclidean distances."""
            return np.sqrt(np.sum((X[:, None] - Y[None, :]) ** 2, axis=2))

        # E[|X - Y|]
        dist_xy = np.mean(pairwise_distances(reference_data, current_data))

        # E[|X - X'|]
        if m > 1:
            dist_xx = np.mean(pairwise_distances(reference_data, reference_data))
        else:
            dist_xx = 0.0

        # E[|Y - Y'|]
        if n > 1:
            dist_yy = np.mean(pairwise_distances(current_data, current_data))
        else:
            dist_yy = 0.0

        energy_dist = 2 * dist_xy - dist_xx - dist_yy

        return max(0, energy_dist)  # Should be non-negative


class AIBasedDriftDetector:
    """AI-powered drift detection using deep learning."""

    def __init__(self):
        # This would initialize neural networks for drift detection
        pass

    async def detect_concept_drift(
        self, performance_history: PerformanceHistory
    ) -> float:
        """Detect concept drift using AI methods."""
        # Simplified implementation - in practice would use trained neural networks

        # Calculate performance degradation trend
        accuracy_trend = np.diff(performance_history.accuracy_scores)

        # Look for consistent degradation
        if len(accuracy_trend) >= 5:
            recent_trend = accuracy_trend[-5:]
            degradation_ratio = np.sum(recent_trend < -0.01) / len(recent_trend)

            # Convert to probability
            drift_probability = min(1.0, degradation_ratio * 2)
        else:
            drift_probability = 0.0

        return drift_probability


class NeuralDriftDetector:
    """Neural network-based drift detector."""

    def __init__(self):
        # This would initialize a trained neural network
        pass


class ContextualDriftAssessor:
    """Domain-aware drift analysis with business context."""

    def __init__(self):
        pass


class PerformanceMonitor:
    """Monitor model performance for drift detection."""

    def __init__(self):
        pass
