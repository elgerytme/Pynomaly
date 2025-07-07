"""Model drift detection domain entities."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any
from uuid import UUID, uuid4


class DriftType(str, Enum):
    """Types of drift detection."""

    DATA_DRIFT = "data_drift"
    CONCEPT_DRIFT = "concept_drift"
    PREDICTION_DRIFT = "prediction_drift"
    LABEL_DRIFT = "label_drift"
    COVARIATE_SHIFT = "covariate_shift"
    PRIOR_PROBABILITY_SHIFT = "prior_probability_shift"


class DriftSeverity(str, Enum):
    """Drift severity levels."""

    NONE = "none"
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


class DriftDetectionMethod(str, Enum):
    """Drift detection methods."""

    KOLMOGOROV_SMIRNOV = "kolmogorov_smirnov"
    POPULATION_STABILITY_INDEX = "population_stability_index"
    JENSEN_SHANNON_DIVERGENCE = "jensen_shannon_divergence"
    WASSERSTEIN_DISTANCE = "wasserstein_distance"
    ENERGY_DISTANCE = "energy_distance"
    MAXIMUM_MEAN_DISCREPANCY = "maximum_mean_discrepancy"
    CHI_SQUARE = "chi_square"
    STATISTICAL_DISTANCE = "statistical_distance"
    ADVERSARIAL_DETECTION = "adversarial_detection"
    AUTOENCODER_RECONSTRUCTION = "autoencoder_reconstruction"
    ISOLATION_FOREST = "isolation_forest"
    CUSTOM = "custom"


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
    """Configuration for drift detection."""

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
    alert_severity_threshold: DriftSeverity = DriftSeverity.MEDIUM

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
    current_drift_severity: DriftSeverity = DriftSeverity.NONE

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
            raise ValueError(f"Monitoring frequency must be one of: {valid_frequencies}")

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
