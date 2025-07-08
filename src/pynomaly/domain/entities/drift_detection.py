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
    
    # Aliases for backwards compatibility
    KS_TEST = "kolmogorov_smirnov"
    PSI = "population_stability_index"


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
    """General drift detection result with additional attributes for use case compatibility."""

    drift_detected: bool
    drift_score: float
    method: DriftDetectionMethod
    timestamp: datetime = field(default_factory=datetime.utcnow)
    metadata: dict[str, Any] = field(default_factory=dict)
    
    # Additional fields for use case compatibility
    detector_id: str | None = None
    severity: DriftSeverity = field(default_factory=lambda: DriftSeverity.NONE)
    
    # Test compatibility fields
    model_id: str | None = None
    detector_name: str | None = None
    p_value: float | None = None
    reference_window_size: int | None = None
    current_window_size: int | None = None
    drift_type: DriftType | None = None
    
    def __post_init__(self):
        """Validate drift detection result."""
        if not (0.0 <= self.drift_score <= 1.0):
            raise ValueError("Drift score must be between 0.0 and 1.0")
        
        # Auto-assign severity based on drift score if not provided
        if self.severity == DriftSeverity.NONE and self.drift_detected:
            if self.drift_score >= 0.8:
                self.severity = DriftSeverity.CRITICAL
            elif self.drift_score >= 0.6:
                self.severity = DriftSeverity.HIGH
            elif self.drift_score >= 0.3:
                self.severity = DriftSeverity.MEDIUM
            else:
                self.severity = DriftSeverity.LOW


class DriftType(Enum):
    """Types of drift that can be detected."""

    DATA_DRIFT = "data_drift"
    CONCEPT_DRIFT = "concept_drift"
    COVARIATE_SHIFT = "covariate_shift"
    LABEL_SHIFT = "label_shift"


class DriftSeverity(Enum):
    """Severity levels for detected drift."""

    NONE = "none"
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


# Alias for backwards compatibility
DriftMonitoringStatus = MonitoringStatus


@dataclass
class DriftDetectionConfig:
    """Configuration for drift detection."""

    # Detection methods
    enabled_methods: list[DriftDetectionMethod] = field(
        default_factory=lambda: [DriftDetectionMethod.KOLMOGOROV_SMIRNOV]
    )

    # Thresholds
    drift_threshold: float = 0.05
    method_thresholds: dict[str, float] = field(default_factory=dict)
    thresholds: DriftThresholds = field(default_factory=DriftThresholds)

    # Severity mapping
    severity_thresholds: dict[str, float] = field(
        default_factory=lambda: {
            "low": 0.1,
            "medium": 0.3,
            "high": 0.6,
            "critical": 0.8,
        }
    )

    # Detection settings
    min_sample_size: int = 100
    reference_window_size: int | None = None
    detection_window_size: int = 1000
    time_window: TimeWindow | None = None
    drift_scope: DriftScope = DriftScope.GLOBAL

    # Feature selection
    features_to_monitor: list[str] | None = None
    exclude_features: list[str] = field(default_factory=list)

    # Advanced settings
    enable_multivariate_detection: bool = True
    enable_concept_drift: bool = True
    enable_univariate_detection: bool = True
    adaptive_thresholds: bool = False
    seasonal_patterns: list[SeasonalPattern] = field(default_factory=list)

    # Alerting
    alert_on_drift: bool = True
    alert_severity_threshold: DriftSeverity = field(default_factory=lambda: DriftSeverity.MEDIUM)
    monitoring_config: ModelMonitoringConfig | None = None

    # Metadata
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        """Validate drift detection configuration."""
        if not (0.0 <= self.drift_threshold <= 1.0):
            raise ValueError("Drift threshold must be between 0.0 and 1.0")
        if self.min_sample_size <= 0:
            raise ValueError("Minimum sample size must be positive")
        if self.detection_window_size <= 0:
            raise ValueError("Detection window size must be positive")
        if self.reference_window_size is not None and self.reference_window_size <= 0:
            raise ValueError("Reference window size must be positive")

        # Initialize monitoring config if not provided
        if self.monitoring_config is None:
            self.monitoring_config = ModelMonitoringConfig(
                drift_threshold=self.drift_threshold,
                severity_threshold=self.alert_severity_threshold,
                notification_enabled=self.alert_on_drift,
            )

    def get_threshold_for_method(self, method: DriftDetectionMethod) -> float:
        """Get threshold for specific drift detection method."""
        method_name = method.value if isinstance(method, DriftDetectionMethod) else str(method)
        return self.method_thresholds.get(method_name, self.thresholds.get_threshold_for_method(method))

    def get_severity_for_score(self, score: float) -> DriftSeverity:
        """Get severity level for drift score."""
        if score >= self.severity_thresholds["critical"]:
            return DriftSeverity.CRITICAL
        elif score >= self.severity_thresholds["high"]:
            return DriftSeverity.HIGH
        elif score >= self.severity_thresholds["medium"]:
            return DriftSeverity.MEDIUM
        elif score >= self.severity_thresholds["low"]:
            return DriftSeverity.LOW
        else:
            return DriftSeverity.NONE

    def is_method_enabled(self, method: DriftDetectionMethod) -> bool:
        """Check if a detection method is enabled."""
        return method in self.enabled_methods

    def should_monitor_feature(self, feature_name: str) -> bool:
        """Check if a feature should be monitored."""
        if feature_name in self.exclude_features:
            return False
        if self.features_to_monitor is None:
            return True
        return feature_name in self.features_to_monitor

    def get_effective_window_size(self, data_size: int) -> int:
        """Get effective window size based on data size and configuration."""
        if self.reference_window_size is not None:
            return min(self.reference_window_size, data_size)
        return min(self.detection_window_size, data_size)


@dataclass
class FeatureDrift:
    """Drift information for a single feature."""

    feature_name: str
    drift_score: float
    threshold: float
    is_drifted: bool
    severity: DriftSeverity
    method: DriftDetectionMethod

    # Statistical information
    p_value: float | None = None
    reference_mean: float | None = None
    current_mean: float | None = None
    reference_std: float | None = None
    current_std: float | None = None

    # Distribution information
    reference_distribution: dict[str, Any] = field(default_factory=dict)
    current_distribution: dict[str, Any] = field(default_factory=dict)

    # Visualization data
    histogram_data: dict[str, Any] = field(default_factory=dict)
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class DriftConfiguration:
    """Configuration for drift detection (legacy alias for DriftDetectionConfig)."""

    # Detection methods
    enabled_methods: list[DriftDetectionMethod] = field(
        default_factory=lambda: [DriftDetectionMethod.KOLMOGOROV_SMIRNOV]
    )

    # Thresholds
    drift_threshold: float = 0.05
    method_thresholds: dict[str, float] = field(default_factory=dict)

    # Severity mapping
    severity_thresholds: dict[str, float] = field(
        default_factory=lambda: {
            "low": 0.1,
            "medium": 0.3,
            "high": 0.6,
            "critical": 0.8,
        }
    )

    # Detection settings
    min_sample_size: int = 100
    reference_window_size: int | None = None
    detection_window_size: int = 1000

    # Feature selection
    features_to_monitor: list[str] | None = None
    exclude_features: list[str] = field(default_factory=list)

    # Advanced settings
    enable_multivariate_detection: bool = True
    enable_concept_drift: bool = True
    adaptive_thresholds: bool = False

    # Alerting
    alert_on_drift: bool = True
    alert_severity_threshold: DriftSeverity = field(default_factory=lambda: DriftSeverity.MEDIUM)

    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        """Validate drift configuration."""
        if not (0.0 <= self.drift_threshold <= 1.0):
            raise ValueError("Drift threshold must be between 0.0 and 1.0")
        if self.min_sample_size <= 0:
            raise ValueError("Minimum sample size must be positive")
        if self.detection_window_size <= 0:
            raise ValueError("Detection window size must be positive")


@dataclass
class DriftReport:
    """Comprehensive drift detection report."""

    # Required fields
    model_id: UUID
    reference_sample_size: int
    current_sample_size: int
    overall_drift_detected: bool
    overall_drift_severity: DriftSeverity
    drift_types_detected: list[DriftType]
    feature_drift: dict[str, FeatureDrift]
    drifted_features: list[str]
    configuration: DriftConfiguration
    detection_start_time: datetime
    detection_end_time: datetime

    # Auto-generated fields
    id: UUID = field(default_factory=uuid4)
    created_at: datetime = field(default_factory=datetime.utcnow)

    # Data information
    reference_data_id: UUID | None = None
    current_data_id: UUID | None = None
    reference_period: str | None = None
    detection_period: str | None = None

    # Multivariate drift
    multivariate_drift_score: float | None = None
    multivariate_drift_detected: bool = False

    # Concept drift
    concept_drift_score: float | None = None
    concept_drift_detected: bool = False

    # Model performance impact
    performance_degradation: dict[str, float] = field(default_factory=dict)

    # Recommendations
    recommendations: list[str] = field(default_factory=list)

    # Metadata
    created_by: str | None = None
    tags: list[str] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)

    def get_high_priority_features(self) -> list[str]:
        """Get features with high or critical drift severity."""
        return [
            feature_name
            for feature_name, drift in self.feature_drift.items()
            if drift.severity in [DriftSeverity.HIGH, DriftSeverity.CRITICAL]
        ]

    def get_drift_summary(self) -> dict[str, Any]:
        """Get a summary of drift detection results."""
        severity_counts = {}
        for severity in DriftSeverity:
            severity_counts[severity.value] = len(
                [f for f in self.feature_drift.values() if f.severity == severity]
            )

        return {
            "total_features": len(self.feature_drift),
            "drifted_features": len(self.drifted_features),
            "drift_percentage": len(self.drifted_features)
            / len(self.feature_drift)
            * 100,
            "severity_distribution": severity_counts,
            "multivariate_drift": self.multivariate_drift_detected,
            "concept_drift": self.concept_drift_detected,
            "overall_severity": self.overall_drift_severity.value,
        }

    def requires_immediate_attention(self) -> bool:
        """Check if the drift requires immediate attention."""
        return (
            self.overall_drift_severity in [DriftSeverity.HIGH, DriftSeverity.CRITICAL]
            or self.concept_drift_detected
            or len(self.get_high_priority_features()) > 0
        )

    def get_recommended_actions(self) -> list[str]:
        """Get recommended actions based on drift severity."""
        actions = []

        if self.overall_drift_severity == DriftSeverity.CRITICAL:
            actions.append("URGENT: Consider immediate model retraining")
            actions.append("Investigate data pipeline for potential issues")

        elif self.overall_drift_severity == DriftSeverity.HIGH:
            actions.append("Schedule model retraining within next maintenance window")
            actions.append("Monitor model performance closely")

        elif self.overall_drift_severity == DriftSeverity.MEDIUM:
            actions.append("Plan model retraining for next release cycle")
            actions.append("Increase monitoring frequency")

        if self.concept_drift_detected:
            actions.append("Investigate changes in target variable distribution")

        if self.multivariate_drift_detected:
            actions.append("Analyze feature interactions and correlations")

        return actions


@dataclass
class DriftMonitor:
    """Drift monitoring configuration and state."""

    # Required fields
    model_id: UUID
    name: str
    configuration: DriftConfiguration
    created_by: str

    # Auto-generated fields
    id: UUID = field(default_factory=uuid4)
    created_at: datetime = field(default_factory=datetime.utcnow)
    updated_at: datetime = field(default_factory=datetime.utcnow)

    # Optional fields
    description: str | None = None

    # Monitoring schedule
    monitoring_enabled: bool = True
    monitoring_frequency: str = "daily"
    last_check_time: datetime | None = None
    next_check_time: datetime | None = None

    # State
    consecutive_drift_detections: int = 0
    last_drift_detection: datetime | None = None
    current_drift_severity: DriftSeverity = field(default_factory=lambda: DriftSeverity.NONE)

    # Alerts
    alert_enabled: bool = True
    alert_recipients: list[str] = field(default_factory=list)
    last_alert_time: datetime | None = None

    # History
    recent_reports: list[UUID] = field(default_factory=list)

    # Metadata
    tags: list[str] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        """Validate drift monitor configuration."""
        valid_frequencies = {"hourly", "daily", "weekly"}
        if self.monitoring_frequency not in valid_frequencies:
            raise ValueError(
                f"Monitoring frequency must be one of: {valid_frequencies}"
            )

    def should_check_now(self) -> bool:
        """Check if drift detection should be performed now."""
        if not self.monitoring_enabled:
            return False

        if self.next_check_time is None:
            return True

        return datetime.utcnow() >= self.next_check_time

    def record_drift_detection(self, severity: DriftSeverity, report_id: UUID) -> None:
        """Record a drift detection result."""
        self.last_check_time = datetime.utcnow()
        self.updated_at = datetime.utcnow()

        if severity != DriftSeverity.NONE:
            self.consecutive_drift_detections += 1
            self.last_drift_detection = datetime.utcnow()
            self.current_drift_severity = severity
        else:
            self.consecutive_drift_detections = 0
            self.current_drift_severity = DriftSeverity.NONE

        # Add to recent reports (keep last 10)
        self.recent_reports.append(report_id)
        if len(self.recent_reports) > 10:
            self.recent_reports = self.recent_reports[-10:]

    def needs_alert(self, current_severity: DriftSeverity) -> bool:
        """Check if an alert should be sent."""
        if not self.alert_enabled:
            return False

        # Alert if severity meets threshold
        severity_order = [
            DriftSeverity.NONE,
            DriftSeverity.LOW,
            DriftSeverity.MEDIUM,
            DriftSeverity.HIGH,
            DriftSeverity.CRITICAL,
        ]

        current_level = severity_order.index(current_severity)
        threshold_level = severity_order.index(
            self.configuration.alert_severity_threshold
        )

        return current_level >= threshold_level


@dataclass
class DriftMetrics:
    """Comprehensive drift detection metrics."""

    statistical_distance: float
    p_value: float
    effect_size: float
    confidence_interval: tuple[float, float]
    power: float
    sample_size: int
    feature_importance_shift: dict[str, float] | None = None
    distribution_shift_score: float | None = None
    temporal_stability_score: float | None = None

    def __post_init__(self):
        """Validate drift metrics."""
        if not (0.0 <= self.p_value <= 1.0):
            raise ValueError("P-value must be between 0.0 and 1.0")
        if not (0.0 <= self.power <= 1.0):
            raise ValueError("Power must be between 0.0 and 1.0")
        if self.sample_size <= 0:
            raise ValueError("Sample size must be positive")

    def is_significant(self, alpha: float = 0.05) -> bool:
        """Check if drift is statistically significant."""
        return self.p_value < alpha

    def get_drift_magnitude(self) -> str:
        """Get qualitative description of drift magnitude."""
        if self.effect_size < 0.2:
            return "negligible"
        elif self.effect_size < 0.5:
            return "small"
        elif self.effect_size < 0.8:
            return "medium"
        else:
            return "large"


@dataclass
class RecommendedAction:
    """Recommended action for addressing drift or performance issues."""

    action_type: str
    priority: str  # HIGH, MEDIUM, LOW
    description: str
    estimated_effort: str  # HOURS, DAYS, WEEKS
    expected_impact: str  # HIGH, MEDIUM, LOW
    prerequisites: list[str] = field(default_factory=list)
    implementation_steps: list[str] = field(default_factory=list)
    success_criteria: list[str] = field(default_factory=list)
    risks: list[str] = field(default_factory=list)
    alternatives: list[str] = field(default_factory=list)

    def get_priority_score(self) -> int:
        """Get numeric priority score."""
        priority_map = {"HIGH": 3, "MEDIUM": 2, "LOW": 1}
        return priority_map.get(self.priority, 1)

    def get_impact_score(self) -> int:
        """Get numeric impact score."""
        impact_map = {"HIGH": 3, "MEDIUM": 2, "LOW": 1}
        return impact_map.get(self.expected_impact, 1)

    def get_urgency_score(self) -> float:
        """Calculate urgency score based on priority and impact."""
        return (self.get_priority_score() + self.get_impact_score()) / 2


@dataclass
class DriftEvent:
    """Represents a detected drift event."""

    drift_id: UUID = field(default_factory=uuid4)
    detected_at: datetime = field(default_factory=datetime.utcnow)
    drift_type: DriftType = DriftType.DATA_DRIFT
    severity: DriftSeverity = field(default_factory=lambda: DriftSeverity.MEDIUM)
    affected_features: list[str] = field(default_factory=list)
    drift_metrics: DriftMetrics | None = None
    recommended_actions: list[RecommendedAction] = field(default_factory=list)
    detection_method: str = "statistical"
    confidence: float = 0.5
    business_impact_assessment: dict[str, Any] | None = None
    resolution_status: str = "OPEN"  # OPEN, IN_PROGRESS, RESOLVED, IGNORED
    resolution_notes: str | None = None
    resolved_at: datetime | None = None

    def __post_init__(self):
        """Validate drift event."""
        if not (0.0 <= self.confidence <= 1.0):
            raise ValueError("Confidence must be between 0.0 and 1.0")

    def get_time_since_detection(self) -> timedelta:
        """Get time elapsed since drift detection."""
        return datetime.utcnow() - self.detected_at

    def is_critical(self) -> bool:
        """Check if drift event is critical."""
        return self.severity == DriftSeverity.CRITICAL

    def needs_immediate_attention(self) -> bool:
        """Check if drift needs immediate attention."""
        return (
            self.severity in [DriftSeverity.HIGH, DriftSeverity.CRITICAL]
            and self.resolution_status == "OPEN"
        )

    def add_recommended_action(self, action: RecommendedAction) -> None:
        """Add a recommended action."""
        self.recommended_actions.append(action)
        # Sort by urgency score
        self.recommended_actions.sort(key=lambda x: x.get_urgency_score(), reverse=True)

    def resolve(self, resolution_notes: str) -> None:
        """Mark drift event as resolved."""
        self.resolution_status = "RESOLVED"
        self.resolution_notes = resolution_notes
        self.resolved_at = datetime.utcnow()


@dataclass
class ModelMonitoringConfig:
    """Configuration for model monitoring."""

    monitoring_enabled: bool = True
    check_interval_minutes: int = 60
    drift_threshold: float = 0.1
    severity_threshold: DriftSeverity = field(default_factory=lambda: DriftSeverity.MEDIUM)
    notification_enabled: bool = True
    auto_retrain_enabled: bool = False
    
    # Additional attributes expected by use cases
    enabled: bool = True
    check_interval_hours: int = 1
    min_sample_size: int = 100
    enabled_methods: list[DriftDetectionMethod] = field(
        default_factory=lambda: [DriftDetectionMethod.KOLMOGOROV_SMIRNOV]
    )
    notification_channels: list[str] = field(
        default_factory=lambda: ["email"]
    )
    
    # Test compatibility fields
    model_id: str | None = None
    check_frequency_hours: int = 1
    alert_threshold: DriftSeverity = field(default_factory=lambda: DriftSeverity.MEDIUM)
    methods: list[DriftDetectionMethod] = field(
        default_factory=lambda: [DriftDetectionMethod.KOLMOGOROV_SMIRNOV]
    )
    monitoring_features: list[str] = field(default_factory=list)

    def __post_init__(self):
        """Validate monitoring configuration."""
        if self.check_interval_minutes <= 0:
            raise ValueError("Check interval must be positive")
        if not (0.0 <= self.drift_threshold <= 1.0):
            raise ValueError("Drift threshold must be between 0.0 and 1.0")
        
        # Sync enabled with monitoring_enabled
        if not hasattr(self, 'enabled') or self.enabled is None:
            self.enabled = self.monitoring_enabled
        
        # Convert minutes to hours if needed
        if not hasattr(self, 'check_interval_hours') or self.check_interval_hours <= 0:
            self.check_interval_hours = max(1, self.check_interval_minutes // 60)
    
    def should_alert(self, severity: DriftSeverity) -> bool:
        """Check if an alert should be sent for the given severity."""
        if not self.notification_enabled:
            return False
        
        # Define severity order
        severity_order = {
            DriftSeverity.NONE: 0,
            DriftSeverity.LOW: 1,
            DriftSeverity.MEDIUM: 2,
            DriftSeverity.HIGH: 3,
            DriftSeverity.CRITICAL: 4,
        }
        
        return severity_order.get(severity, 0) >= severity_order.get(self.severity_threshold, 2)


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
    
    # Additional attributes expected by use cases
    detector_id: str = ""
    alert_type: DriftType = DriftType.DATA_DRIFT
    severity_enum: DriftSeverity = field(default_factory=lambda: DriftSeverity.MEDIUM)
    title: str = ""
    message: str = ""
    id: str = field(default_factory=lambda: str(uuid4()))
    
    def __post_init__(self):
        """Synchronize string and enum fields."""
        # Sync detector_id with model_id if not provided
        if not self.detector_id and self.model_id:
            self.detector_id = str(self.model_id)
        
        # Sync alert_type with drift_type
        if isinstance(self.alert_type, DriftType):
            self.drift_type = self.alert_type.value
        else:
            try:
                self.alert_type = DriftType(self.drift_type)
            except ValueError:
                self.alert_type = DriftType.DATA_DRIFT
        
        # Sync severity fields
        if isinstance(self.severity_enum, DriftSeverity):
            self.severity = self.severity_enum.value
        else:
            try:
                self.severity_enum = DriftSeverity(self.severity)
            except ValueError:
                self.severity_enum = DriftSeverity.MEDIUM
        
        # Sync message fields
        if not self.message and self.alert_message:
            self.message = self.alert_message
        elif not self.alert_message and self.message:
            self.alert_message = self.message
        
        # Sync id with alert_id
        if not self.id and self.alert_id:
            self.id = str(self.alert_id)


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


# Export all drift detection entities
__all__ = [
    # Enums
    "DriftDetectionMethod",
    "DriftScope",
    "SeasonalPattern",
    "DriftType",
    "DriftSeverity",
    "MonitoringStatus",
    "DriftMonitoringStatus",
    
    # Configuration classes
    "DriftDetectionConfig",
    "DriftConfiguration",
    "DriftThresholds",
    "ModelMonitoringConfig",
    
    # Core data classes
    "TimeWindow",
    "FeatureData",
    "FeatureDrift",
    
    # Result classes
    "UnivariateDriftResult",
    "MultivariateDriftResult",
    "ConceptDriftResult",
    "DriftDetectionResult",
    
    # Analysis classes
    "FeatureDriftAnalysis",
    "DriftAnalysisResult",
    
    # Reporting classes
    "DriftReport",
    "DriftMonitor",
    "DriftAlert",
    
    # Metrics and Events
    "DriftMetrics",
    "DriftEvent",
    "RecommendedAction",
]
