"""Performance degradation metrics value objects for model performance monitoring."""

from __future__ import annotations

from datetime import datetime
from enum import Enum
from typing import Any, Optional

from pydantic import Field, validator

from pynomaly.domain.value_objects.base_value_object import BaseValueObject


class DegradationSeverity(str, Enum):
    """Severity levels for performance degradation."""
    
    NONE = "none"
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


class DegradationType(str, Enum):
    """Types of performance degradation."""
    
    ACCURACY_DROP = "accuracy_drop"
    PRECISION_DROP = "precision_drop"
    RECALL_DROP = "recall_drop"
    F1_DROP = "f1_drop"
    AUC_DROP = "auc_drop"
    RMSE_INCREASE = "rmse_increase"
    MAE_INCREASE = "mae_increase"
    R2_DROP = "r2_drop"
    LATENCY_INCREASE = "latency_increase"
    MEMORY_INCREASE = "memory_increase"
    THROUGHPUT_DROP = "throughput_drop"
    ERROR_RATE_INCREASE = "error_rate_increase"
    CONFIDENCE_DROP = "confidence_drop"
    STABILITY_DROP = "stability_drop"
    FAIRNESS_DEGRADATION = "fairness_degradation"
    BUSINESS_VALUE_DROP = "business_value_drop"


class MetricThreshold(BaseValueObject):
    """Threshold configuration for a specific metric."""
    
    metric_name: str = Field(..., min_length=1)
    warning_threshold: float = Field(..., description="Threshold for warning level degradation")
    critical_threshold: float = Field(..., description="Threshold for critical level degradation")
    threshold_type: str = Field(..., description="Type of threshold: 'absolute', 'percentage', 'standard_deviation'")
    direction: str = Field(..., description="Direction of degradation: 'increase' or 'decrease'")
    min_samples: int = Field(default=5, ge=1, description="Minimum samples needed for detection")
    
    @validator('threshold_type')
    def validate_threshold_type(cls, v: str) -> str:
        """Validate threshold type."""
        valid_types = ['absolute', 'percentage', 'standard_deviation']
        if v not in valid_types:
            raise ValueError(f"threshold_type must be one of {valid_types}")
        return v
    
    @validator('direction')
    def validate_direction(cls, v: str) -> str:
        """Validate direction."""
        valid_directions = ['increase', 'decrease']
        if v not in valid_directions:
            raise ValueError(f"direction must be one of {valid_directions}")
        return v
    
    @validator('critical_threshold')
    def validate_critical_threshold(cls, v: float, values: dict[str, Any]) -> float:
        """Validate critical threshold is more severe than warning."""
        warning = values.get('warning_threshold')
        if warning is not None:
            direction = values.get('direction', 'decrease')
            if direction == 'decrease' and v >= warning:
                raise ValueError("Critical threshold must be lower than warning for decrease direction")
            elif direction == 'increase' and v <= warning:
                raise ValueError("Critical threshold must be higher than warning for increase direction")
        return v


class PerformanceDegradation(BaseValueObject):
    """Represents a detected performance degradation."""
    
    degradation_type: DegradationType
    severity: DegradationSeverity
    metric_name: str = Field(..., min_length=1)
    current_value: float
    baseline_value: float
    degradation_amount: float
    degradation_percentage: float
    threshold_violated: str = Field(..., description="Which threshold was violated")
    confidence_level: float = Field(..., ge=0, le=1, description="Confidence in the degradation detection")
    detection_method: str = Field(..., description="Method used to detect degradation")
    detected_at: datetime = Field(default_factory=datetime.utcnow)
    samples_used: int = Field(..., ge=1, description="Number of samples used in detection")
    
    @validator('degradation_percentage')
    def validate_degradation_percentage(cls, v: float, values: dict[str, Any]) -> float:
        """Validate degradation percentage calculation."""
        current = values.get('current_value')
        baseline = values.get('baseline_value')
        if current is not None and baseline is not None and baseline != 0:
            expected_percentage = ((current - baseline) / baseline) * 100
            if abs(v - expected_percentage) > 1e-6:
                raise ValueError("Degradation percentage inconsistent with values")
        return v


class PerformanceBaseline(BaseValueObject):
    """Baseline performance metrics for comparison."""
    
    metric_name: str = Field(..., min_length=1)
    baseline_value: float
    standard_deviation: float = Field(..., ge=0)
    sample_count: int = Field(..., ge=1)
    established_at: datetime = Field(default_factory=datetime.utcnow)
    last_updated: datetime = Field(default_factory=datetime.utcnow)
    confidence_interval_lower: Optional[float] = None
    confidence_interval_upper: Optional[float] = None
    confidence_level: float = Field(default=0.95, ge=0, le=1)
    min_value: Optional[float] = None
    max_value: Optional[float] = None
    percentile_25: Optional[float] = None
    percentile_75: Optional[float] = None
    median_value: Optional[float] = None
    trend_slope: Optional[float] = None
    trend_r_squared: Optional[float] = None
    is_stable: bool = Field(default=True, description="Whether the baseline is considered stable")
    
    def is_degraded(self, current_value: float, threshold: MetricThreshold) -> tuple[bool, DegradationSeverity]:
        """Check if current value represents degradation."""
        if threshold.threshold_type == 'absolute':
            if threshold.direction == 'decrease':
                if current_value <= threshold.critical_threshold:
                    return True, DegradationSeverity.CRITICAL
                elif current_value <= threshold.warning_threshold:
                    return True, DegradationSeverity.HIGH
            else:  # increase
                if current_value >= threshold.critical_threshold:
                    return True, DegradationSeverity.CRITICAL
                elif current_value >= threshold.warning_threshold:
                    return True, DegradationSeverity.HIGH
        
        elif threshold.threshold_type == 'percentage':
            percentage_change = ((current_value - self.baseline_value) / self.baseline_value) * 100
            if threshold.direction == 'decrease':
                if percentage_change <= -threshold.critical_threshold:
                    return True, DegradationSeverity.CRITICAL
                elif percentage_change <= -threshold.warning_threshold:
                    return True, DegradationSeverity.HIGH
            else:  # increase
                if percentage_change >= threshold.critical_threshold:
                    return True, DegradationSeverity.CRITICAL
                elif percentage_change >= threshold.warning_threshold:
                    return True, DegradationSeverity.HIGH
        
        elif threshold.threshold_type == 'standard_deviation':
            std_deviations = abs(current_value - self.baseline_value) / self.standard_deviation
            if threshold.direction == 'decrease' and current_value < self.baseline_value:
                if std_deviations >= threshold.critical_threshold:
                    return True, DegradationSeverity.CRITICAL
                elif std_deviations >= threshold.warning_threshold:
                    return True, DegradationSeverity.HIGH
            elif threshold.direction == 'increase' and current_value > self.baseline_value:
                if std_deviations >= threshold.critical_threshold:
                    return True, DegradationSeverity.CRITICAL
                elif std_deviations >= threshold.warning_threshold:
                    return True, DegradationSeverity.HIGH
        
        return False, DegradationSeverity.NONE
    
    def update_baseline(self, new_values: list[float], update_timestamp: Optional[datetime] = None) -> PerformanceBaseline:
        """Update baseline with new values."""
        import numpy as np
        
        all_values = new_values
        new_mean = float(np.mean(all_values))
        new_std = float(np.std(all_values))
        new_count = len(all_values)
        
        # Calculate confidence interval
        confidence_margin = 1.96 * (new_std / np.sqrt(new_count))  # 95% confidence
        
        return PerformanceBaseline(
            metric_name=self.metric_name,
            baseline_value=new_mean,
            standard_deviation=new_std,
            sample_count=new_count,
            established_at=self.established_at,
            last_updated=update_timestamp or datetime.utcnow(),
            confidence_interval_lower=new_mean - confidence_margin,
            confidence_interval_upper=new_mean + confidence_margin,
            confidence_level=self.confidence_level,
            min_value=float(np.min(all_values)),
            max_value=float(np.max(all_values)),
            percentile_25=float(np.percentile(all_values, 25)),
            percentile_75=float(np.percentile(all_values, 75)),
            median_value=float(np.median(all_values)),
            is_stable=new_std <= (new_mean * 0.1)  # Consider stable if CV < 10%
        )


class DegradationAlert(BaseValueObject):
    """Alert for performance degradation."""
    
    alert_id: str = Field(..., min_length=1)
    model_id: str = Field(..., min_length=1)
    degradation: PerformanceDegradation
    alert_level: DegradationSeverity
    message: str = Field(..., min_length=1)
    recommended_actions: list[str] = Field(default_factory=list)
    created_at: datetime = Field(default_factory=datetime.utcnow)
    acknowledged: bool = Field(default=False)
    acknowledged_by: Optional[str] = None
    acknowledged_at: Optional[datetime] = None
    resolved: bool = Field(default=False)
    resolved_by: Optional[str] = None
    resolved_at: Optional[datetime] = None
    tags: list[str] = Field(default_factory=list)
    
    def acknowledge(self, acknowledged_by: str, timestamp: Optional[datetime] = None) -> DegradationAlert:
        """Acknowledge the alert."""
        return DegradationAlert(
            **{
                **self.dict(),
                'acknowledged': True,
                'acknowledged_by': acknowledged_by,
                'acknowledged_at': timestamp or datetime.utcnow()
            }
        )
    
    def resolve(self, resolved_by: str, timestamp: Optional[datetime] = None) -> DegradationAlert:
        """Resolve the alert."""
        return DegradationAlert(
            **{
                **self.dict(),
                'resolved': True,
                'resolved_by': resolved_by,
                'resolved_at': timestamp or datetime.utcnow()
            }
        )


class DegradationReport(BaseValueObject):
    """Comprehensive performance degradation report."""
    
    report_id: str = Field(..., min_length=1)
    model_id: str = Field(..., min_length=1)
    generated_at: datetime = Field(default_factory=datetime.utcnow)
    time_period_start: datetime
    time_period_end: datetime
    degradations: list[PerformanceDegradation] = Field(default_factory=list)
    baselines_used: list[PerformanceBaseline] = Field(default_factory=list)
    overall_health_score: float = Field(..., ge=0, le=1)
    degradation_summary: dict[str, Any] = Field(default_factory=dict)
    recommendations: list[str] = Field(default_factory=list)
    trend_analysis: dict[str, Any] = Field(default_factory=dict)
    
    @property
    def has_critical_degradations(self) -> bool:
        """Check if report contains critical degradations."""
        return any(d.severity == DegradationSeverity.CRITICAL for d in self.degradations)
    
    @property
    def has_any_degradations(self) -> bool:
        """Check if report contains any degradations."""
        return len(self.degradations) > 0
    
    @property
    def degradation_count_by_severity(self) -> dict[str, int]:
        """Count degradations by severity."""
        counts = {severity.value: 0 for severity in DegradationSeverity}
        for degradation in self.degradations:
            counts[degradation.severity.value] += 1
        return counts
    
    @property
    def most_degraded_metrics(self) -> list[str]:
        """Get list of most degraded metrics."""
        return [d.metric_name for d in sorted(
            self.degradations, 
            key=lambda x: x.degradation_percentage, 
            reverse=True
        )[:5]]
    
    def generate_summary(self) -> dict[str, Any]:
        """Generate degradation summary."""
        return {
            "total_degradations": len(self.degradations),
            "severity_breakdown": self.degradation_count_by_severity,
            "most_degraded_metrics": self.most_degraded_metrics,
            "health_score": self.overall_health_score,
            "has_critical_issues": self.has_critical_degradations,
            "period": {
                "start": self.time_period_start.isoformat(),
                "end": self.time_period_end.isoformat(),
                "duration_hours": (self.time_period_end - self.time_period_start).total_seconds() / 3600
            }
        }