"""Performance degradation metrics value object for model monitoring."""

from __future__ import annotations

from enum import Enum
from typing import Optional

from pydantic import Field, validator

from packages.core.domain.abstractions.base_value_object import BaseValueObject


class DegradationSeverity(str, Enum):
    """Severity levels for performance degradation."""
    
    MINOR = "minor"
    MODERATE = "moderate"
    MAJOR = "major"
    CRITICAL = "critical"


class DegradationMetricType(str, Enum):
    """Types of performance degradation metrics."""
    
    ACCURACY_DROP = "accuracy_drop"
    PRECISION_DROP = "precision_drop"
    RECALL_DROP = "recall_drop"
    F1_SCORE_DROP = "f1_score_drop"
    AUC_DROP = "auc_drop"
    MSE_INCREASE = "mse_increase"
    RMSE_INCREASE = "rmse_increase"
    MAE_INCREASE = "mae_increase"
    R2_SCORE_DROP = "r2_score_drop"
    STABILITY_DECREASE = "stability_decrease"
    CONFIDENCE_DROP = "confidence_drop"
    PREDICTION_TIME_INCREASE = "prediction_time_increase"
    THROUGHPUT_DECREASE = "throughput_decrease"


class PerformanceDegradationMetrics(BaseValueObject):
    """Value object representing performance degradation metrics and thresholds.
    
    This immutable value object defines the metrics and thresholds used to
    detect when model performance has degraded below acceptable levels.
    
    Attributes:
        metric_type: Type of performance metric being monitored
        threshold_value: The threshold value that triggers degradation alert
        baseline_value: The baseline performance value for comparison
        current_value: Current performance value
        degradation_percentage: Percentage of degradation from baseline
        severity: Severity level of the degradation
        consecutive_breaches: Number of consecutive threshold breaches
        time_window_minutes: Time window for degradation detection
        min_samples_required: Minimum number of samples needed for valid detection
        confidence_level: Confidence level for statistical significance
        alert_enabled: Whether alerts are enabled for this metric
        auto_recovery_enabled: Whether auto-recovery is enabled
    """
    
    metric_type: DegradationMetricType
    threshold_value: float
    baseline_value: float
    current_value: Optional[float] = None
    degradation_percentage: Optional[float] = None
    severity: DegradationSeverity = DegradationSeverity.MINOR
    consecutive_breaches: int = Field(default=0, ge=0)
    time_window_minutes: int = Field(default=30, gt=0)
    min_samples_required: int = Field(default=100, gt=0)
    confidence_level: float = Field(default=0.95, gt=0, lt=1)
    alert_enabled: bool = Field(default=True)
    auto_recovery_enabled: bool = Field(default=False)
    
    @validator('threshold_value', 'baseline_value')
    def validate_threshold_and_baseline(cls, v: float) -> float:
        """Validate threshold and baseline values are reasonable."""
        if v < 0:
            raise ValueError("Threshold and baseline values must be non-negative")
        return v
    
    @validator('degradation_percentage')
    def validate_degradation_percentage(cls, v: Optional[float]) -> Optional[float]:
        """Validate degradation percentage is within reasonable bounds."""
        if v is not None:
            if v < 0 or v > 100:
                raise ValueError("Degradation percentage must be between 0 and 100")
        return v
    
    @validator('severity')
    def validate_severity(cls, v: DegradationSeverity, values: dict) -> DegradationSeverity:
        """Validate severity based on degradation percentage."""
        degradation_pct = values.get('degradation_percentage')
        if degradation_pct is not None:
            if degradation_pct >= 50 and v != DegradationSeverity.CRITICAL:
                return DegradationSeverity.CRITICAL
            elif degradation_pct >= 30 and v == DegradationSeverity.MINOR:
                return DegradationSeverity.MAJOR
            elif degradation_pct >= 15 and v == DegradationSeverity.MINOR:
                return DegradationSeverity.MODERATE
        return v
    
    def calculate_degradation_percentage(self) -> float:
        """Calculate degradation percentage from baseline."""
        if self.current_value is None:
            return 0.0
        
        if self.baseline_value == 0:
            return 0.0
        
        # For metrics where lower is better (MSE, MAE, RMSE)
        if self.metric_type in [
            DegradationMetricType.MSE_INCREASE,
            DegradationMetricType.RMSE_INCREASE,
            DegradationMetricType.MAE_INCREASE,
            DegradationMetricType.PREDICTION_TIME_INCREASE,
        ]:
            if self.current_value > self.baseline_value:
                return ((self.current_value - self.baseline_value) / self.baseline_value) * 100
        else:
            # For metrics where higher is better (accuracy, precision, recall, etc.)
            if self.current_value < self.baseline_value:
                return ((self.baseline_value - self.current_value) / self.baseline_value) * 100
        
        return 0.0
    
    def is_degraded(self) -> bool:
        """Check if performance has degraded below threshold."""
        if self.current_value is None:
            return False
        
        # For metrics where lower is better
        if self.metric_type in [
            DegradationMetricType.MSE_INCREASE,
            DegradationMetricType.RMSE_INCREASE,
            DegradationMetricType.MAE_INCREASE,
            DegradationMetricType.PREDICTION_TIME_INCREASE,
        ]:
            return self.current_value > self.threshold_value
        else:
            # For metrics where higher is better
            return self.current_value < self.threshold_value
    
    def get_severity_threshold(self) -> DegradationSeverity:
        """Get severity based on current degradation level."""
        degradation_pct = self.calculate_degradation_percentage()
        
        if degradation_pct >= 50:
            return DegradationSeverity.CRITICAL
        elif degradation_pct >= 30:
            return DegradationSeverity.MAJOR
        elif degradation_pct >= 15:
            return DegradationSeverity.MODERATE
        else:
            return DegradationSeverity.MINOR
    
    def should_alert(self) -> bool:
        """Check if an alert should be triggered."""
        return (
            self.alert_enabled 
            and self.is_degraded() 
            and self.consecutive_breaches >= 1
        )
    
    def get_alert_message(self) -> str:
        """Get formatted alert message."""
        if not self.should_alert():
            return ""
        
        degradation_pct = self.calculate_degradation_percentage()
        return (
            f"Performance degradation detected: {self.metric_type.value} "
            f"has degraded by {degradation_pct:.1f}% from baseline "
            f"({self.baseline_value:.3f} → {self.current_value:.3f}). "
            f"Severity: {self.severity.value.upper()}"
        )
    
    def get_recovery_recommendation(self) -> str:
        """Get recovery recommendation based on degradation type."""
        recommendations = {
            DegradationMetricType.ACCURACY_DROP: "Consider retraining with recent data or reviewing feature quality",
            DegradationMetricType.PRECISION_DROP: "Check for data drift or adjust decision threshold",
            DegradationMetricType.RECALL_DROP: "Review training data distribution or model sensitivity",
            DegradationMetricType.F1_SCORE_DROP: "Evaluate both precision and recall components",
            DegradationMetricType.AUC_DROP: "Assess model discriminative power and feature relevance",
            DegradationMetricType.MSE_INCREASE: "Check for outliers or consider regularization",
            DegradationMetricType.RMSE_INCREASE: "Evaluate prediction variance and model stability",
            DegradationMetricType.MAE_INCREASE: "Review prediction accuracy and error distribution",
            DegradationMetricType.R2_SCORE_DROP: "Assess model fit and feature engineering",
            DegradationMetricType.STABILITY_DECREASE: "Check for concept drift or model overfitting",
            DegradationMetricType.CONFIDENCE_DROP: "Review model calibration and uncertainty estimation",
            DegradationMetricType.PREDICTION_TIME_INCREASE: "Optimize model inference or infrastructure",
            DegradationMetricType.THROUGHPUT_DECREASE: "Scale resources or optimize batch processing",
        }
        
        return recommendations.get(
            self.metric_type, 
            "Review model performance and consider retraining"
        )
    
    def update_current_value(self, new_value: float) -> PerformanceDegradationMetrics:
        """Update current value and recalculate degradation metrics."""
        updated_data = self.dict()
        updated_data['current_value'] = new_value
        updated_data['degradation_percentage'] = self.calculate_degradation_percentage()
        updated_data['severity'] = self.get_severity_threshold()
        
        # Update consecutive breaches
        if self.is_degraded():
            updated_data['consecutive_breaches'] = self.consecutive_breaches + 1
        else:
            updated_data['consecutive_breaches'] = 0
        
        return PerformanceDegradationMetrics(**updated_data)
    
    def to_monitoring_dict(self) -> dict:
        """Convert to dictionary for monitoring systems."""
        return {
            "metric_type": self.metric_type.value,
            "threshold_value": self.threshold_value,
            "baseline_value": self.baseline_value,
            "current_value": self.current_value,
            "degradation_percentage": self.calculate_degradation_percentage(),
            "severity": self.severity.value,
            "is_degraded": self.is_degraded(),
            "should_alert": self.should_alert(),
            "consecutive_breaches": self.consecutive_breaches,
            "time_window_minutes": self.time_window_minutes,
            "alert_enabled": self.alert_enabled,
            "recovery_recommendation": self.get_recovery_recommendation(),
        }
    
    @classmethod
    def create_accuracy_degradation_metric(
        cls, 
        baseline_accuracy: float, 
        threshold_percentage: float = 5.0,
        **kwargs
    ) -> PerformanceDegradationMetrics:
        """Create accuracy degradation metric with sensible defaults."""
        threshold_value = baseline_accuracy * (1 - threshold_percentage / 100)
        return cls(
            metric_type=DegradationMetricType.ACCURACY_DROP,
            threshold_value=threshold_value,
            baseline_value=baseline_accuracy,
            **kwargs
        )
    
    @classmethod
    def create_mse_degradation_metric(
        cls, 
        baseline_mse: float, 
        threshold_percentage: float = 20.0,
        **kwargs
    ) -> PerformanceDegradationMetrics:
        """Create MSE degradation metric with sensible defaults."""
        threshold_value = baseline_mse * (1 + threshold_percentage / 100)
        return cls(
            metric_type=DegradationMetricType.MSE_INCREASE,
            threshold_value=threshold_value,
            baseline_value=baseline_mse,
            **kwargs
        )
    
    @classmethod
    def create_latency_degradation_metric(
        cls, 
        baseline_latency: float, 
        threshold_percentage: float = 50.0,
        **kwargs
    ) -> PerformanceDegradationMetrics:
        """Create latency degradation metric with sensible defaults."""
        threshold_value = baseline_latency * (1 + threshold_percentage / 100)
        return cls(
            metric_type=DegradationMetricType.PREDICTION_TIME_INCREASE,
            threshold_value=threshold_value,
            baseline_value=baseline_latency,
            **kwargs
        )