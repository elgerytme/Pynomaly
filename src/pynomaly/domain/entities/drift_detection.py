"""Domain entities for drift detection framework."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from typing import Any
from uuid import UUID, uuid4

import numpy as np


class DriftDetectionMethod(Enum):
    """Methods for drift detection."""

    KOLMOGOROV_SMIRNOV = "kolmogorov_smirnov"
    POPULATION_STABILITY_INDEX = "population_stability_index"
    JENSEN_SHANNON_DIVERGENCE = "jensen_shannon_divergence"
    MAXIMUM_MEAN_DISCREPANCY = "maximum_mean_discrepancy"
    WASSERSTEIN_DISTANCE = "wasserstein_distance"
    ENERGY_DISTANCE = "energy_distance"
    ADVERSARIAL_DRIFT_DETECTION = "adversarial_drift_detection"
    NEURAL_DRIFT_DETECTOR = "neural_drift_detector"
    STATISTICAL_PROCESS_CONTROL = "statistical_process_control"


class DriftScope(Enum):
    """Scope of drift detection."""

    UNIVARIATE = "univariate"
    MULTIVARIATE = "multivariate"
    GLOBAL = "global"
    FEATURE_SPECIFIC = "feature_specific"
    TEMPORAL = "temporal"
    CONCEPTUAL = "conceptual"


class SeasonalPattern(Enum):
    """Types of seasonal patterns in data."""

    DAILY = "daily"
    WEEKLY = "weekly"
    MONTHLY = "monthly"
    QUARTERLY = "quarterly"
    YEARLY = "yearly"
    BUSINESS_CYCLE = "business_cycle"
    CUSTOM = "custom"


@dataclass
class DriftThresholds:
    """Configurable thresholds for drift detection."""

    statistical_significance: float = 0.05
    effect_size_threshold: float = 0.2
    psi_threshold: float = 0.1
    js_divergence_threshold: float = 0.1
    mmd_threshold: float = 0.1
    wasserstein_threshold: float = 0.1
    energy_threshold: float = 0.1
    neural_drift_threshold: float = 0.7

    def __post_init__(self):
        """Validate drift thresholds."""
        if not (0.0 < self.statistical_significance < 1.0):
            raise ValueError("Statistical significance must be between 0.0 and 1.0")
        if self.effect_size_threshold < 0.0:
            raise ValueError("Effect size threshold must be non-negative")

    def get_threshold_for_method(self, method: DriftDetectionMethod) -> float:
        """Get threshold for specific drift detection method."""
        threshold_map = {
            DriftDetectionMethod.KOLMOGOROV_SMIRNOV: self.statistical_significance,
            DriftDetectionMethod.POPULATION_STABILITY_INDEX: self.psi_threshold,
            DriftDetectionMethod.JENSEN_SHANNON_DIVERGENCE: self.js_divergence_threshold,
            DriftDetectionMethod.MAXIMUM_MEAN_DISCREPANCY: self.mmd_threshold,
            DriftDetectionMethod.WASSERSTEIN_DISTANCE: self.wasserstein_threshold,
            DriftDetectionMethod.ENERGY_DISTANCE: self.energy_threshold,
            DriftDetectionMethod.NEURAL_DRIFT_DETECTOR: self.neural_drift_threshold,
        }
        return threshold_map.get(method, self.statistical_significance)


@dataclass
class TimeWindow:
    """Time window configuration for drift detection."""

    start_time: datetime
    end_time: datetime
    window_size: timedelta
    overlap_percentage: float = 0.0

    def __post_init__(self):
        """Validate time window."""
        if self.start_time >= self.end_time:
            raise ValueError("Start time must be before end time")
        if not (0.0 <= self.overlap_percentage < 1.0):
            raise ValueError("Overlap percentage must be between 0.0 and 1.0")
        if self.window_size <= timedelta(0):
            raise ValueError("Window size must be positive")

    def get_duration(self) -> timedelta:
        """Get total duration of the time window."""
        return self.end_time - self.start_time

    def get_overlap_duration(self) -> timedelta:
        """Get overlap duration for sliding windows."""
        return timedelta(
            seconds=self.window_size.total_seconds() * self.overlap_percentage
        )

    def generate_sliding_windows(self) -> list[TimeWindow]:
        """Generate sliding windows within this time range."""
        windows = []
        current_start = self.start_time
        step_size = self.window_size - self.get_overlap_duration()

        while current_start + self.window_size <= self.end_time:
            window_end = current_start + self.window_size
            windows.append(
                TimeWindow(
                    start_time=current_start,
                    end_time=window_end,
                    window_size=self.window_size,
                    overlap_percentage=self.overlap_percentage,
                )
            )
            current_start += step_size

        return windows


@dataclass
class FeatureData:
    """Data for individual features in drift detection."""

    feature_name: str
    reference_data: np.ndarray
    current_data: np.ndarray
    data_type: str  # numerical, categorical, text
    missing_value_rate: float = 0.0
    outlier_rate: float = 0.0
    cardinality: int | None = None
    statistical_properties: dict[str, float] = field(default_factory=dict)

    def __post_init__(self):
        """Validate and compute feature data properties."""
        if len(self.reference_data) == 0 or len(self.current_data) == 0:
            raise ValueError("Reference and current data cannot be empty")

        if not (0.0 <= self.missing_value_rate <= 1.0):
            raise ValueError("Missing value rate must be between 0.0 and 1.0")

        if not (0.0 <= self.outlier_rate <= 1.0):
            raise ValueError("Outlier rate must be between 0.0 and 1.0")

        # Compute statistical properties if not provided
        if not self.statistical_properties:
            self.statistical_properties = self._compute_statistical_properties()

    def _compute_statistical_properties(self) -> dict[str, float]:
        """Compute statistical properties for the feature."""
        if self.data_type == "numerical":
            return {
                "ref_mean": float(np.mean(self.reference_data)),
                "ref_std": float(np.std(self.reference_data)),
                "ref_median": float(np.median(self.reference_data)),
                "ref_skew": float(self._safe_skew(self.reference_data)),
                "ref_kurtosis": float(self._safe_kurtosis(self.reference_data)),
                "curr_mean": float(np.mean(self.current_data)),
                "curr_std": float(np.std(self.current_data)),
                "curr_median": float(np.median(self.current_data)),
                "curr_skew": float(self._safe_skew(self.current_data)),
                "curr_kurtosis": float(self._safe_kurtosis(self.current_data)),
            }
        else:
            return {
                "ref_unique_count": float(len(np.unique(self.reference_data))),
                "curr_unique_count": float(len(np.unique(self.current_data))),
                "ref_mode_frequency": float(
                    self._get_mode_frequency(self.reference_data)
                ),
                "curr_mode_frequency": float(
                    self._get_mode_frequency(self.current_data)
                ),
            }

    def _safe_skew(self, data: np.ndarray) -> float:
        """Safely compute skewness."""
        try:
            from scipy.stats import skew

            return skew(data)
        except Exception:
            return 0.0

    def _safe_kurtosis(self, data: np.ndarray) -> float:
        """Safely compute kurtosis."""
        try:
            from scipy.stats import kurtosis

            return kurtosis(data)
        except Exception:
            return 0.0

    def _get_mode_frequency(self, data: np.ndarray) -> float:
        """Get frequency of the most common value."""
        unique, counts = np.unique(data, return_counts=True)
        return float(np.max(counts)) / len(data)

    def get_sample_size_ratio(self) -> float:
        """Get ratio of current to reference sample sizes."""
        return len(self.current_data) / len(self.reference_data)


@dataclass
class UnivariateDriftResult:
    """Result of univariate drift detection."""

    feature_name: str
    detection_method: DriftDetectionMethod
    drift_detected: bool
    drift_score: float
    p_value: float
    effect_size: float
    confidence_interval: tuple[float, float]
    threshold_used: float
    sample_size_reference: int
    sample_size_current: int
    detection_timestamp: datetime = field(default_factory=datetime.utcnow)
    additional_metrics: dict[str, float] = field(default_factory=dict)

    def __post_init__(self):
        """Validate univariate drift result."""
        if not (0.0 <= self.p_value <= 1.0):
            raise ValueError("P-value must be between 0.0 and 1.0")
        if self.sample_size_reference <= 0 or self.sample_size_current <= 0:
            raise ValueError("Sample sizes must be positive")

    def get_drift_severity(self) -> str:
        """Get qualitative assessment of drift severity."""
        if not self.drift_detected:
            return "none"
        elif self.effect_size < 0.2:
            return "negligible"
        elif self.effect_size < 0.5:
            return "small"
        elif self.effect_size < 0.8:
            return "medium"
        else:
            return "large"

    def is_statistically_significant(self, alpha: float = 0.05) -> bool:
        """Check if drift is statistically significant."""
        return self.p_value < alpha

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary representation."""
        return {
            "feature_name": self.feature_name,
            "detection_method": self.detection_method.value,
            "drift_detected": self.drift_detected,
            "drift_score": self.drift_score,
            "p_value": self.p_value,
            "effect_size": self.effect_size,
            "confidence_interval": list(self.confidence_interval),
            "threshold_used": self.threshold_used,
            "sample_sizes": {
                "reference": self.sample_size_reference,
                "current": self.sample_size_current,
            },
            "drift_severity": self.get_drift_severity(),
            "statistically_significant": self.is_statistically_significant(),
            "detection_timestamp": self.detection_timestamp.isoformat(),
            "additional_metrics": self.additional_metrics,
        }


@dataclass
class MultivariateDriftResult:
    """Result of multivariate drift detection."""

    detection_method: DriftDetectionMethod
    drift_detected: bool
    drift_score: float
    threshold_used: float
    affected_features: list[str]
    feature_contributions: dict[str, float] = field(default_factory=dict)
    spatial_drift_map: np.ndarray | None = None
    covariance_shift_detected: bool = False
    marginal_drift_components: dict[str, float] = field(default_factory=dict)
    detection_timestamp: datetime = field(default_factory=datetime.utcnow)
    computation_time_seconds: float = 0.0

    def __post_init__(self):
        """Validate multivariate drift result."""
        if self.computation_time_seconds < 0:
            raise ValueError("Computation time must be non-negative")

    def get_most_affected_features(self, top_k: int = 5) -> list[tuple[str, float]]:
        """Get features most affected by drift."""
        if not self.feature_contributions:
            return []

        sorted_features = sorted(
            self.feature_contributions.items(), key=lambda x: x[1], reverse=True
        )
        return sorted_features[:top_k]

    def get_drift_pattern_summary(self) -> dict[str, Any]:
        """Get summary of drift patterns detected."""
        return {
            "overall_drift": self.drift_detected,
            "drift_score": self.drift_score,
            "covariance_shift": self.covariance_shift_detected,
            "most_affected_features": self.get_most_affected_features(3),
            "detection_method": self.detection_method.value,
            "computation_time": self.computation_time_seconds,
        }


@dataclass
class FeatureDriftAnalysis:
    """Comprehensive analysis of feature-level drift."""

    feature_name: str
    analysis_timestamp: datetime = field(default_factory=datetime.utcnow)
    univariate_results: list[UnivariateDriftResult] = field(default_factory=list)
    temporal_drift_pattern: dict[str, Any] | None = None
    seasonal_components: list[SeasonalPattern] = field(default_factory=list)
    drift_velocity: float = 0.0  # Rate of drift over time
    drift_acceleration: float = 0.0  # Change in drift rate
    stability_score: float = (
        1.0  # Overall stability (1.0 = very stable, 0.0 = very unstable)
    )
    anomalous_periods: list[TimeWindow] = field(default_factory=list)

    def __post_init__(self):
        """Validate feature drift analysis."""
        if not (0.0 <= self.stability_score <= 1.0):
            raise ValueError("Stability score must be between 0.0 and 1.0")

    def has_drift(self) -> bool:
        """Check if any drift was detected."""
        return any(result.drift_detected for result in self.univariate_results)

    def get_strongest_drift_signal(self) -> UnivariateDriftResult | None:
        """Get the strongest drift signal detected."""
        if not self.univariate_results:
            return None

        drift_results = [r for r in self.univariate_results if r.drift_detected]
        if not drift_results:
            return None

        return max(drift_results, key=lambda x: x.effect_size)

    def get_consensus_drift_score(self) -> float:
        """Get consensus drift score across all methods."""
        if not self.univariate_results:
            return 0.0

        # Weight by inverse p-value for significant results
        weighted_scores = []
        weights = []

        for result in self.univariate_results:
            if result.drift_detected:
                weight = 1.0 / max(result.p_value, 1e-10)  # Avoid division by zero
                weighted_scores.append(result.drift_score * weight)
                weights.append(weight)

        if not weighted_scores:
            return 0.0

        return sum(weighted_scores) / sum(weights)

    def is_trend_drift(self) -> bool:
        """Check if drift follows a trend pattern."""
        return abs(self.drift_velocity) > 0.1

    def is_sudden_drift(self) -> bool:
        """Check if drift appears suddenly."""
        return abs(self.drift_acceleration) > 0.1

    def get_drift_characterization(self) -> dict[str, Any]:
        """Get comprehensive characterization of the drift."""
        return {
            "has_drift": self.has_drift(),
            "consensus_score": self.get_consensus_drift_score(),
            "is_trend": self.is_trend_drift(),
            "is_sudden": self.is_sudden_drift(),
            "stability": self.stability_score,
            "velocity": self.drift_velocity,
            "acceleration": self.drift_acceleration,
            "seasonal_patterns": [p.value for p in self.seasonal_components],
            "anomalous_periods_count": len(self.anomalous_periods),
        }


@dataclass
class ConceptDriftResult:
    """Result of concept drift detection."""

    detection_method: DriftDetectionMethod
    drift_probability: float
    drift_detected: bool
    confidence: float
    affected_concepts: list[str] = field(default_factory=list)
    stability_metrics: dict[str, float] = field(default_factory=dict)
    drift_patterns: dict[str, Any] = field(default_factory=dict)
    temporal_analysis: dict[str, Any] | None = None
    prediction_consistency_score: float = 1.0
    label_distribution_shift: bool = False
    decision_boundary_shift: bool = False
    detection_timestamp: datetime = field(default_factory=datetime.utcnow)

    def __post_init__(self):
        """Validate concept drift result."""
        if not (0.0 <= self.drift_probability <= 1.0):
            raise ValueError("Drift probability must be between 0.0 and 1.0")
        if not (0.0 <= self.confidence <= 1.0):
            raise ValueError("Confidence must be between 0.0 and 1.0")
        if not (0.0 <= self.prediction_consistency_score <= 1.0):
            raise ValueError("Prediction consistency score must be between 0.0 and 1.0")

    def get_drift_severity(self) -> str:
        """Get qualitative assessment of concept drift severity."""
        if not self.drift_detected:
            return "none"
        elif self.drift_probability < 0.3:
            return "low"
        elif self.drift_probability < 0.7:
            return "medium"
        else:
            return "high"

    def has_multiple_drift_types(self) -> bool:
        """Check if multiple types of concept drift are detected."""
        drift_types = [
            self.label_distribution_shift,
            self.decision_boundary_shift,
            len(self.affected_concepts) > 1,
        ]
        return sum(drift_types) > 1

    def get_drift_summary(self) -> dict[str, Any]:
        """Get summary of concept drift detection."""
        return {
            "drift_detected": self.drift_detected,
            "probability": self.drift_probability,
            "severity": self.get_drift_severity(),
            "confidence": self.confidence,
            "consistency_score": self.prediction_consistency_score,
            "multiple_types": self.has_multiple_drift_types(),
            "label_shift": self.label_distribution_shift,
            "boundary_shift": self.decision_boundary_shift,
            "affected_concepts": self.affected_concepts,
            "detection_method": self.detection_method.value,
        }


@dataclass
class DriftDetectionResult:
    """General drift detection result."""

    drift_detected: bool
    drift_score: float
    method: DriftDetectionMethod
    timestamp: datetime = field(default_factory=datetime.utcnow)
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        """Validate drift detection result."""
        if not (0.0 <= self.drift_score <= 1.0):
            raise ValueError("Drift score must be between 0.0 and 1.0")


class DriftType(Enum):
    """Types of drift that can be detected."""

    DATA_DRIFT = "data_drift"
    CONCEPT_DRIFT = "concept_drift"
    COVARIATE_SHIFT = "covariate_shift"
    LABEL_SHIFT = "label_shift"


class DriftSeverity(Enum):
    """Severity levels for detected drift."""

    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


class MonitoringStatus(Enum):
    """Status of drift monitoring."""

    ACTIVE = "active"
    PAUSED = "paused"
    STOPPED = "stopped"
    ERROR = "error"


@dataclass
class ModelMonitoringConfig:
    """Configuration for model monitoring."""

    monitoring_enabled: bool = True
    check_interval_minutes: int = 60
    drift_threshold: float = 0.1
    severity_threshold: DriftSeverity = DriftSeverity.MEDIUM
    notification_enabled: bool = True
    auto_retrain_enabled: bool = False

    def __post_init__(self):
        """Validate monitoring configuration."""
        if self.check_interval_minutes <= 0:
            raise ValueError("Check interval must be positive")
        if not (0.0 <= self.drift_threshold <= 1.0):
            raise ValueError("Drift threshold must be between 0.0 and 1.0")


@dataclass
class DriftAlert:
    """Alert for drift detection events."""

    alert_id: UUID = field(default_factory=uuid4)
    drift_type: str = "data_drift"
    severity: str = "medium"
    model_id: UUID = field(default_factory=uuid4)
    feature_name: str = ""
    drift_score: float = 0.0
    threshold: float = 0.1
    detection_timestamp: datetime = field(default_factory=datetime.utcnow)
    alert_message: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class DriftAnalysisResult:
    """Comprehensive drift analysis result."""

    analysis_id: UUID = field(default_factory=uuid4)
    model_id: UUID = field(default_factory=uuid4)
    analysis_timestamp: datetime = field(default_factory=datetime.utcnow)
    time_window: TimeWindow = field(
        default_factory=lambda: TimeWindow(
            start_time=datetime.utcnow() - timedelta(days=1),
            end_time=datetime.utcnow(),
            window_size=timedelta(hours=1),
        )
    )
    data_drift_results: list[UnivariateDriftResult] = field(default_factory=list)
    multivariate_drift_result: MultivariateDriftResult | None = None
    concept_drift_result: ConceptDriftResult | None = None
    feature_analyses: dict[str, FeatureDriftAnalysis] = field(default_factory=dict)
    overall_drift_score: float = 0.0
    drift_severity: str = "none"
    recommended_actions: list[str] = field(default_factory=list)
    analysis_metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        """Validate and compute overall drift analysis."""
        if not (0.0 <= self.overall_drift_score <= 1.0):
            raise ValueError("Overall drift score must be between 0.0 and 1.0")

        # Compute overall drift score if not provided
        if self.overall_drift_score == 0.0:
            self.overall_drift_score = self._compute_overall_drift_score()

        # Determine drift severity if not provided
        if self.drift_severity == "none":
            self.drift_severity = self._determine_overall_severity()

    def _compute_overall_drift_score(self) -> float:
        """Compute overall drift score from individual results."""
        scores = []

        # Data drift scores
        for result in self.data_drift_results:
            if result.drift_detected:
                scores.append(result.drift_score)

        # Multivariate drift score
        if (
            self.multivariate_drift_result
            and self.multivariate_drift_result.drift_detected
        ):
            scores.append(self.multivariate_drift_result.drift_score)

        # Concept drift score
        if self.concept_drift_result and self.concept_drift_result.drift_detected:
            scores.append(self.concept_drift_result.drift_probability)

        # Feature analysis scores
        for analysis in self.feature_analyses.values():
            if analysis.has_drift():
                scores.append(analysis.get_consensus_drift_score())

        return np.mean(scores) if scores else 0.0

    def _determine_overall_severity(self) -> str:
        """Determine overall drift severity."""
        if self.overall_drift_score < 0.1:
            return "none"
        elif self.overall_drift_score < 0.3:
            return "low"
        elif self.overall_drift_score < 0.7:
            return "medium"
        else:
            return "high"

    def has_any_drift(self) -> bool:
        """Check if any type of drift was detected."""
        has_data_drift = any(r.drift_detected for r in self.data_drift_results)
        has_multivariate_drift = (
            self.multivariate_drift_result
            and self.multivariate_drift_result.drift_detected
        )
        has_concept_drift = (
            self.concept_drift_result and self.concept_drift_result.drift_detected
        )
        has_feature_drift = any(a.has_drift() for a in self.feature_analyses.values())

        return any(
            [
                has_data_drift,
                has_multivariate_drift,
                has_concept_drift,
                has_feature_drift,
            ]
        )

    def get_critical_features(self) -> list[str]:
        """Get features with critical drift levels."""
        critical_features = []

        for result in self.data_drift_results:
            if result.drift_detected and result.effect_size > 0.8:
                critical_features.append(result.feature_name)

        for feature_name, analysis in self.feature_analyses.items():
            if analysis.has_drift() and analysis.get_consensus_drift_score() > 0.8:
                critical_features.append(feature_name)

        return list(set(critical_features))

    def needs_immediate_action(self) -> bool:
        """Check if drift requires immediate action."""
        return (
            self.drift_severity in ["high"]
            or len(self.get_critical_features()) > 0
            or (
                self.concept_drift_result
                and self.concept_drift_result.drift_probability > 0.8
            )
        )

    def get_comprehensive_summary(self) -> dict[str, Any]:
        """Get comprehensive summary of all drift analysis."""
        return {
            "analysis_id": str(self.analysis_id),
            "model_id": str(self.model_id),
            "timestamp": self.analysis_timestamp.isoformat(),
            "time_window": {
                "start": self.time_window.start_time.isoformat(),
                "end": self.time_window.end_time.isoformat(),
                "duration_hours": self.time_window.get_duration().total_seconds()
                / 3600,
            },
            "overall": {
                "has_drift": self.has_any_drift(),
                "drift_score": self.overall_drift_score,
                "severity": self.drift_severity,
                "needs_action": self.needs_immediate_action(),
            },
            "data_drift": {
                "features_analyzed": len(self.data_drift_results),
                "features_with_drift": sum(
                    1 for r in self.data_drift_results if r.drift_detected
                ),
                "critical_features": self.get_critical_features(),
            },
            "multivariate_drift": {
                "detected": (
                    self.multivariate_drift_result.drift_detected
                    if self.multivariate_drift_result
                    else False
                ),
                "score": (
                    self.multivariate_drift_result.drift_score
                    if self.multivariate_drift_result
                    else 0.0
                ),
            },
            "concept_drift": {
                "detected": (
                    self.concept_drift_result.drift_detected
                    if self.concept_drift_result
                    else False
                ),
                "probability": (
                    self.concept_drift_result.drift_probability
                    if self.concept_drift_result
                    else 0.0
                ),
            },
            "recommended_actions": self.recommended_actions,
            "analysis_metadata": self.analysis_metadata,
        }


@dataclass
class DriftMetrics:
    """Comprehensive metrics for drift monitoring and analysis."""

    metrics_id: UUID = field(default_factory=uuid4)
    model_id: UUID = field(default_factory=uuid4)
    timestamp: datetime = field(default_factory=datetime.utcnow)
    monitoring_window_hours: int = 24
    
    # Detection statistics
    total_detections: int = 0
    drift_detections: int = 0
    false_positive_rate: float = 0.0
    false_negative_rate: float = 0.0
    detection_accuracy: float = 1.0
    
    # Drift characteristics
    avg_drift_score: float = 0.0
    max_drift_score: float = 0.0
    drift_frequency: float = 0.0  # Drifts per hour
    drift_duration_minutes: float = 0.0  # Average duration
    
    # Feature-level metrics
    features_monitored: int = 0
    features_with_drift: int = 0
    most_unstable_features: list[str] = field(default_factory=list)
    feature_stability_scores: dict[str, float] = field(default_factory=dict)
    
    # Performance impact
    prediction_degradation: float = 0.0
    model_confidence_drop: float = 0.0
    retraining_triggered: bool = False
    last_retrain_timestamp: datetime | None = None
    
    # Alert metrics
    alerts_generated: int = 0
    alerts_resolved: int = 0
    avg_resolution_time_minutes: float = 0.0
    critical_alerts: int = 0
    
    # System metrics
    monitoring_coverage: float = 1.0  # Percentage of data monitored
    data_quality_score: float = 1.0
    monitoring_latency_ms: float = 0.0
    computational_cost: float = 0.0
    
    # Trend analysis
    drift_trend: str = "stable"  # stable, increasing, decreasing
    seasonality_detected: bool = False
    seasonal_patterns: list[str] = field(default_factory=list)
    
    # Additional metadata
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        """Validate drift metrics."""
        if self.monitoring_window_hours <= 0:
            raise ValueError("Monitoring window must be positive")
        
        rates = [self.false_positive_rate, self.false_negative_rate, self.detection_accuracy]
        for rate in rates:
            if not (0.0 <= rate <= 1.0):
                raise ValueError("Rates must be between 0.0 and 1.0")
        
        scores = [self.avg_drift_score, self.max_drift_score]
        for score in scores:
            if not (0.0 <= score <= 1.0):
                raise ValueError("Drift scores must be between 0.0 and 1.0")

    @property
    def drift_detection_rate(self) -> float:
        """Calculate drift detection rate."""
        if self.total_detections == 0:
            return 0.0
        return self.drift_detections / self.total_detections

    @property
    def feature_drift_rate(self) -> float:
        """Calculate feature drift rate."""
        if self.features_monitored == 0:
            return 0.0
        return self.features_with_drift / self.features_monitored

    @property
    def alert_resolution_rate(self) -> float:
        """Calculate alert resolution rate."""
        if self.alerts_generated == 0:
            return 1.0
        return self.alerts_resolved / self.alerts_generated

    @property
    def monitoring_health_score(self) -> float:
        """Calculate overall monitoring health score."""
        scores = [
            self.detection_accuracy,
            1.0 - self.false_positive_rate,
            1.0 - self.false_negative_rate,
            self.monitoring_coverage,
            self.data_quality_score,
            self.alert_resolution_rate,
        ]
        return sum(scores) / len(scores)

    def is_monitoring_healthy(self) -> bool:
        """Check if monitoring system is healthy."""
        return (
            self.monitoring_health_score > 0.8
            and self.false_positive_rate < 0.1
            and self.data_quality_score > 0.9
        )

    def needs_attention(self) -> bool:
        """Check if monitoring needs attention."""
        return (
            self.drift_detection_rate > 0.5
            or self.critical_alerts > 0
            or self.prediction_degradation > 0.2
            or not self.is_monitoring_healthy()
        )

    def update_from_detection_result(self, result: DriftAnalysisResult):
        """Update metrics from a drift detection result."""
        self.total_detections += 1
        
        if result.has_any_drift():
            self.drift_detections += 1
            self.max_drift_score = max(self.max_drift_score, result.overall_drift_score)
            
            # Update running average
            n = self.drift_detections
            self.avg_drift_score = ((n - 1) * self.avg_drift_score + result.overall_drift_score) / n
            
            # Update feature metrics
            critical_features = result.get_critical_features()
            self.most_unstable_features.extend(critical_features)
            # Keep only top 10 most recent unstable features
            self.most_unstable_features = list(set(self.most_unstable_features))[-10:]
            
            if result.needs_immediate_action():
                self.critical_alerts += 1

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary representation."""
        return {
            "metrics_id": str(self.metrics_id),
            "model_id": str(self.model_id),
            "timestamp": self.timestamp.isoformat(),
            "monitoring_window_hours": self.monitoring_window_hours,
            "detection_stats": {
                "total_detections": self.total_detections,
                "drift_detections": self.drift_detections,
                "detection_rate": self.drift_detection_rate,
                "accuracy": self.detection_accuracy,
                "false_positive_rate": self.false_positive_rate,
                "false_negative_rate": self.false_negative_rate,
            },
            "drift_characteristics": {
                "avg_score": self.avg_drift_score,
                "max_score": self.max_drift_score,
                "frequency": self.drift_frequency,
                "avg_duration_minutes": self.drift_duration_minutes,
            },
            "feature_metrics": {
                "monitored": self.features_monitored,
                "with_drift": self.features_with_drift,
                "drift_rate": self.feature_drift_rate,
                "most_unstable": self.most_unstable_features,
            },
            "performance_impact": {
                "prediction_degradation": self.prediction_degradation,
                "confidence_drop": self.model_confidence_drop,
                "retraining_needed": self.retraining_triggered,
                "last_retrain": (
                    self.last_retrain_timestamp.isoformat()
                    if self.last_retrain_timestamp
                    else None
                ),
            },
            "alert_metrics": {
                "generated": self.alerts_generated,
                "resolved": self.alerts_resolved,
                "resolution_rate": self.alert_resolution_rate,
                "avg_resolution_time": self.avg_resolution_time_minutes,
                "critical": self.critical_alerts,
            },
            "system_health": {
                "monitoring_coverage": self.monitoring_coverage,
                "data_quality": self.data_quality_score,
                "latency_ms": self.monitoring_latency_ms,
                "cost": self.computational_cost,
                "health_score": self.monitoring_health_score,
                "is_healthy": self.is_monitoring_healthy(),
                "needs_attention": self.needs_attention(),
            },
            "trend_analysis": {
                "trend": self.drift_trend,
                "drift_frequency": self.drift_frequency,
                "avg_duration": self.drift_duration_minutes,
                "seasonality": self.seasonality_detected,
                "patterns": self.seasonal_patterns,
                "unstable_features": self.most_unstable_features[:5],
            },
            "metadata": self.metadata,
        }
