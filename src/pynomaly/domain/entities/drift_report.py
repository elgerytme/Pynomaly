"""Model drift detection domain entities."""

from __future__ import annotations

from datetime import datetime
from enum import Enum
from typing import Any
from uuid import UUID, uuid4

from pydantic import BaseModel, Field


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


class FeatureDrift(BaseModel):
    """Drift information for a single feature."""

    feature_name: str = Field(..., description="Feature name")
    drift_score: float = Field(..., description="Drift score (0-1 or method-specific)")
    p_value: float | None = Field(None, description="Statistical p-value")
    threshold: float = Field(..., description="Drift threshold used")
    is_drifted: bool = Field(..., description="Whether drift was detected")
    severity: DriftSeverity = Field(..., description="Drift severity level")
    method: DriftDetectionMethod = Field(..., description="Detection method used")
    
    # Statistical information
    reference_mean: float | None = Field(None, description="Reference data mean")
    current_mean: float | None = Field(None, description="Current data mean")
    reference_std: float | None = Field(None, description="Reference data standard deviation")
    current_std: float | None = Field(None, description="Current data standard deviation")
    
    # Distribution information
    reference_distribution: dict[str, Any] = Field(
        default_factory=dict, description="Reference distribution parameters"
    )
    current_distribution: dict[str, Any] = Field(
        default_factory=dict, description="Current distribution parameters"
    )
    
    # Visualization data
    histogram_data: dict[str, Any] = Field(
        default_factory=dict, description="Histogram data for visualization"
    )
    
    metadata: dict[str, Any] = Field(default_factory=dict, description="Additional metadata")


class DriftConfiguration(BaseModel):
    """Configuration for drift detection."""

    # Detection methods
    enabled_methods: list[DriftDetectionMethod] = Field(
        default=[DriftDetectionMethod.KOLMOGOROV_SMIRNOV],
        description="Enabled drift detection methods"
    )
    
    # Thresholds
    drift_threshold: float = Field(0.05, description="Primary drift threshold")
    method_thresholds: dict[str, float] = Field(
        default_factory=dict, description="Method-specific thresholds"
    )
    
    # Severity mapping
    severity_thresholds: dict[str, float] = Field(
        default_factory=lambda: {
            "low": 0.1,
            "medium": 0.3,
            "high": 0.6,
            "critical": 0.8
        },
        description="Severity level thresholds"
    )
    
    # Detection settings
    min_sample_size: int = Field(100, description="Minimum sample size for detection")
    reference_window_size: int | None = Field(None, description="Reference window size")
    detection_window_size: int = Field(1000, description="Detection window size")
    
    # Feature selection
    features_to_monitor: list[str] | None = Field(
        None, description="Specific features to monitor (None for all)"
    )
    exclude_features: list[str] = Field(
        default_factory=list, description="Features to exclude from monitoring"
    )
    
    # Advanced settings
    enable_multivariate_detection: bool = Field(
        True, description="Enable multivariate drift detection"
    )
    enable_concept_drift: bool = Field(
        True, description="Enable concept drift detection"
    )
    adaptive_thresholds: bool = Field(
        False, description="Use adaptive thresholds based on historical data"
    )
    
    # Alerting
    alert_on_drift: bool = Field(True, description="Send alerts when drift is detected")
    alert_severity_threshold: DriftSeverity = Field(
        DriftSeverity.MEDIUM, description="Minimum severity for alerts"
    )
    
    metadata: dict[str, Any] = Field(default_factory=dict, description="Additional configuration")


class DriftReport(BaseModel):
    """Comprehensive drift detection report."""

    id: UUID = Field(default_factory=uuid4, description="Report identifier")
    model_id: UUID = Field(..., description="Model identifier")
    
    # Data information
    reference_data_id: UUID | None = Field(None, description="Reference dataset identifier")
    current_data_id: UUID | None = Field(None, description="Current dataset identifier")
    reference_period: str | None = Field(None, description="Reference data period")
    detection_period: str | None = Field(None, description="Detection data period")
    
    # Sample information
    reference_sample_size: int = Field(..., description="Reference data sample size")
    current_sample_size: int = Field(..., description="Current data sample size")
    
    # Overall drift assessment
    overall_drift_detected: bool = Field(..., description="Whether any drift was detected")
    overall_drift_severity: DriftSeverity = Field(..., description="Overall drift severity")
    drift_types_detected: list[DriftType] = Field(..., description="Types of drift detected")
    
    # Feature-level drift
    feature_drift: dict[str, FeatureDrift] = Field(..., description="Per-feature drift analysis")
    drifted_features: list[str] = Field(..., description="List of features with detected drift")
    
    # Multivariate drift
    multivariate_drift_score: float | None = Field(None, description="Multivariate drift score")
    multivariate_drift_detected: bool = Field(False, description="Multivariate drift detected")
    
    # Concept drift
    concept_drift_score: float | None = Field(None, description="Concept drift score")
    concept_drift_detected: bool = Field(False, description="Concept drift detected")
    
    # Model performance impact
    performance_degradation: dict[str, float] = Field(
        default_factory=dict, description="Performance degradation metrics"
    )
    
    # Configuration used
    configuration: DriftConfiguration = Field(..., description="Configuration used for detection")
    
    # Timing information
    created_at: datetime = Field(default_factory=datetime.utcnow, description="Report creation time")
    detection_start_time: datetime = Field(..., description="Detection start time")
    detection_end_time: datetime = Field(..., description="Detection end time")
    
    # Recommendations
    recommendations: list[str] = Field(
        default_factory=list, description="Recommendations based on drift analysis"
    )
    
    # Metadata
    created_by: str | None = Field(None, description="User who initiated the detection")
    tags: list[str] = Field(default_factory=list, description="Report tags")
    metadata: dict[str, Any] = Field(default_factory=dict, description="Additional metadata")

    def get_high_priority_features(self) -> list[str]:
        """Get features with high or critical drift severity."""
        return [
            feature_name for feature_name, drift in self.feature_drift.items()
            if drift.severity in [DriftSeverity.HIGH, DriftSeverity.CRITICAL]
        ]

    def get_drift_summary(self) -> dict[str, Any]:
        """Get a summary of drift detection results."""
        severity_counts = {}
        for severity in DriftSeverity:
            severity_counts[severity.value] = len([
                f for f in self.feature_drift.values()
                if f.severity == severity
            ])

        return {
            "total_features": len(self.feature_drift),
            "drifted_features": len(self.drifted_features),
            "drift_percentage": len(self.drifted_features) / len(self.feature_drift) * 100,
            "severity_distribution": severity_counts,
            "multivariate_drift": self.multivariate_drift_detected,
            "concept_drift": self.concept_drift_detected,
            "overall_severity": self.overall_drift_severity.value,
        }

    def requires_immediate_attention(self) -> bool:
        """Check if the drift requires immediate attention."""
        return (
            self.overall_drift_severity in [DriftSeverity.HIGH, DriftSeverity.CRITICAL] or
            self.concept_drift_detected or
            len(self.get_high_priority_features()) > 0
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

    class Config:
        """Pydantic model configuration."""
        
        validate_assignment = True
        use_enum_values = True
        json_encoders = {
            datetime: lambda v: v.isoformat(),
        }


class DriftMonitor(BaseModel):
    """Drift monitoring configuration and state."""

    id: UUID = Field(default_factory=uuid4, description="Monitor identifier")
    model_id: UUID = Field(..., description="Model being monitored")
    name: str = Field(..., description="Monitor name")
    description: str | None = Field(None, description="Monitor description")
    
    # Configuration
    configuration: DriftConfiguration = Field(..., description="Drift detection configuration")
    
    # Monitoring schedule
    monitoring_enabled: bool = Field(True, description="Whether monitoring is enabled")
    monitoring_frequency: str = Field("daily", description="Monitoring frequency (hourly, daily, weekly)")
    last_check_time: datetime | None = Field(None, description="Last drift check time")
    next_check_time: datetime | None = Field(None, description="Next scheduled check time")
    
    # State
    consecutive_drift_detections: int = Field(0, description="Consecutive drift detections")
    last_drift_detection: datetime | None = Field(None, description="Last drift detection time")
    current_drift_severity: DriftSeverity = Field(
        DriftSeverity.NONE, description="Current drift severity"
    )
    
    # Alerts
    alert_enabled: bool = Field(True, description="Whether alerts are enabled")
    alert_recipients: list[str] = Field(default_factory=list, description="Alert recipients")
    last_alert_time: datetime | None = Field(None, description="Last alert time")
    
    # History
    recent_reports: list[UUID] = Field(
        default_factory=list, description="Recent drift report IDs"
    )
    
    # Metadata
    created_at: datetime = Field(default_factory=datetime.utcnow, description="Creation time")
    updated_at: datetime = Field(default_factory=datetime.utcnow, description="Last update time")
    created_by: str = Field(..., description="Creator")
    tags: list[str] = Field(default_factory=list, description="Monitor tags")
    metadata: dict[str, Any] = Field(default_factory=dict, description="Additional metadata")

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
            DriftSeverity.NONE, DriftSeverity.LOW, DriftSeverity.MEDIUM,
            DriftSeverity.HIGH, DriftSeverity.CRITICAL
        ]
        
        current_level = severity_order.index(current_severity)
        threshold_level = severity_order.index(self.configuration.alert_severity_threshold)
        
        return current_level >= threshold_level

    class Config:
        """Pydantic model configuration."""
        
        validate_assignment = True
        use_enum_values = True
        json_encoders = {
            datetime: lambda v: v.isoformat(),
        }