"""Domain entities for model performance degradation detection.

This module implements the domain logic for D-003: Model Performance Degradation Detection.
It provides entities and value objects for tracking and detecting when model performance
drops below acceptable thresholds.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from typing import Any, Dict, List, Optional, Union
from uuid import UUID, uuid4

from pynomaly.domain.value_objects import AnomalyScore


class PerformanceMetricType(Enum):
    """Types of performance metrics."""
    
    ACCURACY = "accuracy"
    PRECISION = "precision"
    RECALL = "recall"
    F1_SCORE = "f1_score"
    AUC_ROC = "auc_roc"
    AUC_PR = "auc_pr"
    PRECISION_AT_K = "precision_at_k"
    RECALL_AT_K = "recall_at_k"
    MEAN_AVERAGE_PRECISION = "mean_average_precision"
    SPECIFICITY = "specificity"
    SENSITIVITY = "sensitivity"
    NPV = "negative_predictive_value"
    FPR = "false_positive_rate"
    FNR = "false_negative_rate"
    MATTHEWS_CORRELATION = "matthews_correlation"
    BALANCED_ACCURACY = "balanced_accuracy"
    COHEN_KAPPA = "cohen_kappa"
    BRIER_SCORE = "brier_score"
    LOG_LOSS = "log_loss"
    DETECTION_RATE = "detection_rate"
    FALSE_ALARM_RATE = "false_alarm_rate"
    MEAN_TIME_TO_DETECTION = "mean_time_to_detection"
    
    # Custom metrics
    CUSTOM = "custom"


class DegradationSeverity(Enum):
    """Severity levels for performance degradation."""
    
    NONE = "none"
    MINOR = "minor"
    MODERATE = "moderate"
    MAJOR = "major"
    CRITICAL = "critical"


class DegradationStatus(Enum):
    """Status of performance degradation."""
    
    NORMAL = "normal"
    WARNING = "warning"
    DEGRADED = "degraded"
    CRITICAL = "critical"
    RECOVERING = "recovering"


@dataclass(frozen=True)
class PerformanceThreshold:
    """Value object representing a performance threshold."""
    
    metric_type: PerformanceMetricType
    threshold_value: float
    comparison_operator: str  # ">=", "<=", ">", "<", "==", "!="
    severity: DegradationSeverity
    description: Optional[str] = None
    
    def __post_init__(self):
        """Validate threshold configuration."""
        if self.comparison_operator not in [">=", "<=", ">", "<", "==", "!="]:
            raise ValueError(f"Invalid comparison operator: {self.comparison_operator}")
        
        if not isinstance(self.threshold_value, (int, float)):
            raise ValueError("Threshold value must be numeric")
        
        if self.metric_type in [
            PerformanceMetricType.ACCURACY,
            PerformanceMetricType.PRECISION,
            PerformanceMetricType.RECALL,
            PerformanceMetricType.F1_SCORE,
            PerformanceMetricType.AUC_ROC,
            PerformanceMetricType.AUC_PR,
            PerformanceMetricType.SPECIFICITY,
            PerformanceMetricType.SENSITIVITY,
            PerformanceMetricType.NPV,
            PerformanceMetricType.BALANCED_ACCURACY,
        ]:
            if not (0.0 <= self.threshold_value <= 1.0):
                raise ValueError(f"Threshold for {self.metric_type.value} must be between 0.0 and 1.0")
    
    def evaluate(self, value: float) -> bool:
        """Evaluate if the value violates the threshold."""
        if self.comparison_operator == ">=":
            return value >= self.threshold_value
        elif self.comparison_operator == "<=":
            return value <= self.threshold_value
        elif self.comparison_operator == ">":
            return value > self.threshold_value
        elif self.comparison_operator == "<":
            return value < self.threshold_value
        elif self.comparison_operator == "==":
            return abs(value - self.threshold_value) < 1e-10
        elif self.comparison_operator == "!=":
            return abs(value - self.threshold_value) >= 1e-10
        
        return False


@dataclass
class PerformanceMetric:
    """Entity representing a performance metric measurement."""
    
    id: UUID = field(default_factory=uuid4)
    detector_id: UUID = field(default=None)
    metric_type: PerformanceMetricType = field(default=None)
    value: float = field(default=0.0)
    timestamp: datetime = field(default_factory=datetime.utcnow)
    
    # Metadata
    dataset_id: Optional[str] = None
    sample_count: Optional[int] = None
    evaluation_window: Optional[timedelta] = None
    confidence_interval: Optional[tuple[float, float]] = None
    
    # Context information
    model_version: Optional[str] = None
    experiment_id: Optional[str] = None
    environment: Optional[str] = None
    
    # Additional metrics context
    true_positives: Optional[int] = None
    false_positives: Optional[int] = None
    true_negatives: Optional[int] = None
    false_negatives: Optional[int] = None
    
    # Custom metadata
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def __post_init__(self):
        """Validate metric data."""
        if self.detector_id is None:
            raise ValueError("Detector ID is required")
        
        if self.metric_type is None:
            raise ValueError("Metric type is required")
        
        if not isinstance(self.value, (int, float)):
            raise ValueError("Metric value must be numeric")
        
        # Validate value ranges for specific metrics
        if self.metric_type in [
            PerformanceMetricType.ACCURACY,
            PerformanceMetricType.PRECISION,
            PerformanceMetricType.RECALL,
            PerformanceMetricType.F1_SCORE,
            PerformanceMetricType.AUC_ROC,
            PerformanceMetricType.AUC_PR,
            PerformanceMetricType.SPECIFICITY,
            PerformanceMetricType.SENSITIVITY,
            PerformanceMetricType.NPV,
            PerformanceMetricType.BALANCED_ACCURACY,
        ]:
            if not (0.0 <= self.value <= 1.0):
                raise ValueError(f"Value for {self.metric_type.value} must be between 0.0 and 1.0")
    
    def is_recent(self, time_window: timedelta) -> bool:
        """Check if metric is within the specified time window."""
        return datetime.utcnow() - self.timestamp <= time_window
    
    def get_confusion_matrix_metrics(self) -> Optional[Dict[str, int]]:
        """Get confusion matrix metrics if available."""
        if all(x is not None for x in [
            self.true_positives, self.false_positives,
            self.true_negatives, self.false_negatives
        ]):
            return {
                "true_positives": self.true_positives,
                "false_positives": self.false_positives,
                "true_negatives": self.true_negatives,
                "false_negatives": self.false_negatives
            }
        return None


@dataclass
class PerformanceDegradationEvent:
    """Entity representing a performance degradation event."""
    
    id: UUID = field(default_factory=uuid4)
    detector_id: UUID = field(default=None)
    detected_at: datetime = field(default_factory=datetime.utcnow)
    severity: DegradationSeverity = field(default=DegradationSeverity.MINOR)
    status: DegradationStatus = field(default=DegradationStatus.DEGRADED)
    
    # Threshold information
    violated_threshold: PerformanceThreshold = field(default=None)
    trigger_metric: PerformanceMetric = field(default=None)
    
    # Degradation details
    baseline_value: Optional[float] = None
    current_value: Optional[float] = None
    degradation_percentage: Optional[float] = None
    
    # Time window information
    evaluation_window: timedelta = field(default=timedelta(days=7))
    baseline_window: timedelta = field(default=timedelta(days=30))
    
    # Historical context
    historical_metrics: List[PerformanceMetric] = field(default_factory=list)
    trend_analysis: Optional[Dict[str, Any]] = None
    
    # Resolution information
    resolved_at: Optional[datetime] = None
    resolution_notes: Optional[str] = None
    
    # Actions taken
    actions_triggered: List[str] = field(default_factory=list)
    retraining_triggered: bool = False
    
    # Metadata
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def __post_init__(self):
        """Validate degradation event."""
        if self.detector_id is None:
            raise ValueError("Detector ID is required")
        
        if self.violated_threshold is None:
            raise ValueError("Violated threshold is required")
        
        if self.trigger_metric is None:
            raise ValueError("Trigger metric is required")
        
        # Calculate degradation percentage if baseline available
        if self.baseline_value is not None and self.current_value is not None:
            if self.baseline_value != 0:
                self.degradation_percentage = (
                    (self.current_value - self.baseline_value) / self.baseline_value * 100
                )
    
    def is_resolved(self) -> bool:
        """Check if the degradation event has been resolved."""
        return self.resolved_at is not None
    
    def get_duration(self) -> timedelta:
        """Get the duration of the degradation event."""
        end_time = self.resolved_at or datetime.utcnow()
        return end_time - self.detected_at
    
    def add_action(self, action: str) -> None:
        """Add an action taken in response to this degradation."""
        if action not in self.actions_triggered:
            self.actions_triggered.append(action)
    
    def resolve(self, notes: Optional[str] = None) -> None:
        """Mark the degradation event as resolved."""
        self.resolved_at = datetime.utcnow()
        self.status = DegradationStatus.NORMAL
        if notes:
            self.resolution_notes = notes
    
    def update_trend_analysis(self, analysis: Dict[str, Any]) -> None:
        """Update trend analysis information."""
        self.trend_analysis = analysis


@dataclass
class PerformanceMonitoringConfiguration:
    """Entity representing performance monitoring configuration."""
    
    id: UUID = field(default_factory=uuid4)
    detector_id: UUID = field(default=None)
    created_at: datetime = field(default_factory=datetime.utcnow)
    updated_at: datetime = field(default_factory=datetime.utcnow)
    
    # Monitoring settings
    enabled: bool = True
    monitoring_interval: timedelta = field(default=timedelta(hours=1))
    evaluation_window: timedelta = field(default=timedelta(days=7))
    baseline_window: timedelta = field(default=timedelta(days=30))
    
    # Thresholds
    performance_thresholds: List[PerformanceThreshold] = field(default_factory=list)
    
    # Data requirements
    min_samples_required: int = 100
    confidence_level: float = 0.95
    
    # Alerting configuration
    alert_on_degradation: bool = True
    alert_channels: List[str] = field(default_factory=list)
    escalation_rules: Dict[str, Any] = field(default_factory=dict)
    
    # Auto-recovery settings
    auto_trigger_retraining: bool = True
    retraining_threshold_severity: DegradationSeverity = DegradationSeverity.MODERATE
    
    # Metadata
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def __post_init__(self):
        """Validate configuration."""
        if self.detector_id is None:
            raise ValueError("Detector ID is required")
        
        if self.min_samples_required < 1:
            raise ValueError("Minimum samples required must be at least 1")
        
        if not (0.0 < self.confidence_level < 1.0):
            raise ValueError("Confidence level must be between 0.0 and 1.0")
        
        if self.monitoring_interval.total_seconds() <= 0:
            raise ValueError("Monitoring interval must be positive")
    
    def add_threshold(self, threshold: PerformanceThreshold) -> None:
        """Add a performance threshold."""
        if threshold not in self.performance_thresholds:
            self.performance_thresholds.append(threshold)
            self.updated_at = datetime.utcnow()
    
    def remove_threshold(self, threshold: PerformanceThreshold) -> bool:
        """Remove a performance threshold."""
        if threshold in self.performance_thresholds:
            self.performance_thresholds.remove(threshold)
            self.updated_at = datetime.utcnow()
            return True
        return False
    
    def get_thresholds_by_severity(self, severity: DegradationSeverity) -> List[PerformanceThreshold]:
        """Get thresholds by severity level."""
        return [t for t in self.performance_thresholds if t.severity == severity]
    
    def should_trigger_retraining(self, severity: DegradationSeverity) -> bool:
        """Check if retraining should be triggered for the given severity."""
        severity_levels = [
            DegradationSeverity.NONE,
            DegradationSeverity.MINOR,
            DegradationSeverity.MODERATE,
            DegradationSeverity.MAJOR,
            DegradationSeverity.CRITICAL
        ]
        
        return (
            self.auto_trigger_retraining and
            severity_levels.index(severity) >= severity_levels.index(self.retraining_threshold_severity)
        )


@dataclass
class PerformanceBaseline:
    """Entity representing performance baseline for degradation detection."""
    
    id: UUID = field(default_factory=uuid4)
    detector_id: UUID = field(default=None)
    created_at: datetime = field(default_factory=datetime.utcnow)
    
    # Baseline period
    baseline_start: datetime = field(default=None)
    baseline_end: datetime = field(default=None)
    
    # Baseline metrics
    baseline_metrics: Dict[PerformanceMetricType, float] = field(default_factory=dict)
    baseline_statistics: Dict[str, Any] = field(default_factory=dict)
    
    # Data quality
    sample_count: int = 0
    confidence_intervals: Dict[PerformanceMetricType, tuple[float, float]] = field(default_factory=dict)
    
    # Validation
    is_valid: bool = True
    validation_notes: Optional[str] = None
    
    # Metadata
    model_version: Optional[str] = None
    dataset_id: Optional[str] = None
    environment: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def __post_init__(self):
        """Validate baseline."""
        if self.detector_id is None:
            raise ValueError("Detector ID is required")
        
        if self.baseline_start is None or self.baseline_end is None:
            raise ValueError("Baseline start and end times are required")
        
        if self.baseline_start >= self.baseline_end:
            raise ValueError("Baseline start must be before baseline end")
        
        if self.sample_count < 1:
            raise ValueError("Sample count must be at least 1")
    
    def get_baseline_value(self, metric_type: PerformanceMetricType) -> Optional[float]:
        """Get baseline value for a specific metric type."""
        return self.baseline_metrics.get(metric_type)
    
    def get_confidence_interval(self, metric_type: PerformanceMetricType) -> Optional[tuple[float, float]]:
        """Get confidence interval for a specific metric type."""
        return self.confidence_intervals.get(metric_type)
    
    def is_within_confidence_interval(self, metric_type: PerformanceMetricType, value: float) -> bool:
        """Check if a value is within the confidence interval."""
        interval = self.get_confidence_interval(metric_type)
        if interval is None:
            return True  # No interval available, assume within
        
        lower, upper = interval
        return lower <= value <= upper
    
    def get_duration(self) -> timedelta:
        """Get the duration of the baseline period."""
        return self.baseline_end - self.baseline_start
    
    def is_recent(self, max_age: timedelta) -> bool:
        """Check if the baseline is recent enough."""
        return datetime.utcnow() - self.baseline_end <= max_age
    
    def update_statistics(self, statistics: Dict[str, Any]) -> None:
        """Update baseline statistics."""
        self.baseline_statistics.update(statistics)
        
    def invalidate(self, reason: str) -> None:
        """Mark baseline as invalid."""
        self.is_valid = False
        self.validation_notes = reason
