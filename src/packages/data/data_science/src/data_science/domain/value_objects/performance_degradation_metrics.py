"""Performance degradation metrics value object."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Dict, Any, Optional


class DegradationMetricType(Enum):
    """Types of metrics used for degradation detection."""
    
    ACCURACY = "accuracy"
    PRECISION = "precision"
    RECALL = "recall"
    F1_SCORE = "f1_score"
    AUC_ROC = "auc_roc"
    AUC_PR = "auc_pr"
    MAE = "mae"  # Mean Absolute Error
    MSE = "mse"  # Mean Squared Error
    RMSE = "rmse"  # Root Mean Squared Error
    R2_SCORE = "r2_score"
    LATENCY = "latency"
    THROUGHPUT = "throughput"
    ERROR_RATE = "error_rate"
    MEMORY_USAGE = "memory_usage"
    CPU_USAGE = "cpu_usage"
    CUSTOM = "custom"


@dataclass(frozen=True)
class PerformanceDegradationMetrics:
    """Value object representing performance metrics for degradation analysis."""
    
    # Primary metric (main metric being monitored)
    primary_metric_type: DegradationMetricType = DegradationMetricType.ACCURACY
    primary_metric_value: float = 0.0
    primary_metric_name: str = ""
    
    # Additional metrics for comprehensive analysis
    secondary_metrics: Dict[str, float] = field(default_factory=dict)
    
    # Statistical information
    sample_size: int = 0
    confidence_interval_lower: float = 0.0
    confidence_interval_upper: float = 0.0
    standard_deviation: float = 0.0
    variance: float = 0.0
    
    # Temporal information
    measurement_timestamp: Optional[datetime] = None
    measurement_period_start: Optional[datetime] = None
    measurement_period_end: Optional[datetime] = None
    
    # Data quality indicators
    data_quality_score: float = 1.0  # 0.0 to 1.0
    missing_data_percentage: float = 0.0
    outlier_percentage: float = 0.0
    
    # Contextual metadata
    model_version: str = ""
    dataset_version: str = ""
    environment: str = ""  # e.g., "production", "staging", "test"
    region: str = ""
    
    # Additional metadata
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def __post_init__(self) -> None:
        """Validate metrics after initialization."""
        if self.sample_size < 0:
            raise ValueError("Sample size cannot be negative")
        
        if not (0.0 <= self.data_quality_score <= 1.0):
            raise ValueError("Data quality score must be between 0.0 and 1.0")
        
        if not (0.0 <= self.missing_data_percentage <= 100.0):
            raise ValueError("Missing data percentage must be between 0.0 and 100.0")
        
        if not (0.0 <= self.outlier_percentage <= 100.0):
            raise ValueError("Outlier percentage must be between 0.0 and 100.0")
        
        if self.confidence_interval_lower > self.confidence_interval_upper:
            raise ValueError("Confidence interval lower bound cannot be greater than upper bound")
    
    def get_metric_value(self, metric_name: str) -> Optional[float]:
        """Get value for a specific metric."""
        if metric_name == self.primary_metric_name or metric_name == self.primary_metric_type.value:
            return self.primary_metric_value
        
        return self.secondary_metrics.get(metric_name)
    
    def has_sufficient_data(self, min_sample_size: int = 100) -> bool:
        """Check if metrics have sufficient data for reliable analysis."""
        return (
            self.sample_size >= min_sample_size and
            self.data_quality_score >= 0.7 and
            self.missing_data_percentage <= 30.0
        )
    
    def get_confidence_width(self) -> float:
        """Get width of confidence interval."""
        return self.confidence_interval_upper - self.confidence_interval_lower
    
    def is_within_expected_range(self, expected_min: float, expected_max: float) -> bool:
        """Check if primary metric is within expected range."""
        return expected_min <= self.primary_metric_value <= expected_max
    
    def compare_to(self, other: PerformanceDegradationMetrics) -> Dict[str, float]:
        """Compare metrics with another metrics object."""
        if self.primary_metric_type != other.primary_metric_type:
            raise ValueError("Cannot compare metrics of different types")
        
        comparison = {
            "primary_metric_diff": self.primary_metric_value - other.primary_metric_value,
            "primary_metric_pct_change": (
                ((self.primary_metric_value - other.primary_metric_value) / other.primary_metric_value * 100)
                if other.primary_metric_value != 0 else 0.0
            ),
            "data_quality_diff": self.data_quality_score - other.data_quality_score,
            "sample_size_diff": self.sample_size - other.sample_size
        }
        
        # Compare secondary metrics
        for metric_name in set(self.secondary_metrics.keys()) | set(other.secondary_metrics.keys()):
            self_value = self.secondary_metrics.get(metric_name, 0.0)
            other_value = other.secondary_metrics.get(metric_name, 0.0)
            
            comparison[f"{metric_name}_diff"] = self_value - other_value
            if other_value != 0:
                comparison[f"{metric_name}_pct_change"] = (self_value - other_value) / other_value * 100
        
        return comparison
    
    def get_degradation_indicators(self, baseline: PerformanceDegradationMetrics) -> Dict[str, Any]:
        """Get degradation indicators compared to baseline."""
        comparison = self.compare_to(baseline)
        
        # Determine if metrics indicate degradation based on metric type
        is_degraded = False
        degradation_severity = "none"
        
        primary_pct_change = comparison["primary_metric_pct_change"]
        
        # For accuracy-like metrics (higher is better)
        if self.primary_metric_type in [
            DegradationMetricType.ACCURACY,
            DegradationMetricType.PRECISION,
            DegradationMetricType.RECALL,
            DegradationMetricType.F1_SCORE,
            DegradationMetricType.AUC_ROC,
            DegradationMetricType.AUC_PR,
            DegradationMetricType.R2_SCORE,
            DegradationMetricType.THROUGHPUT
        ]:
            if primary_pct_change < -10:  # 10% decrease
                is_degraded = True
                degradation_severity = "critical" if primary_pct_change < -25 else "high" if primary_pct_change < -20 else "medium"
            elif primary_pct_change < -5:  # 5% decrease
                is_degraded = True
                degradation_severity = "low"
        
        # For error-like metrics (lower is better)
        elif self.primary_metric_type in [
            DegradationMetricType.MAE,
            DegradationMetricType.MSE,
            DegradationMetricType.RMSE,
            DegradationMetricType.LATENCY,
            DegradationMetricType.ERROR_RATE,
            DegradationMetricType.MEMORY_USAGE,
            DegradationMetricType.CPU_USAGE
        ]:
            if primary_pct_change > 10:  # 10% increase
                is_degraded = True
                degradation_severity = "critical" if primary_pct_change > 50 else "high" if primary_pct_change > 25 else "medium"
            elif primary_pct_change > 5:  # 5% increase
                is_degraded = True
                degradation_severity = "low"
        
        return {
            "is_degraded": is_degraded,
            "degradation_severity": degradation_severity,
            "primary_metric_change_pct": primary_pct_change,
            "data_quality_degraded": self.data_quality_score < baseline.data_quality_score - 0.1,
            "insufficient_data": not self.has_sufficient_data(),
            "comparison_details": comparison
        }
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary representation."""
        return {
            "primary_metric_type": self.primary_metric_type.value,
            "primary_metric_value": self.primary_metric_value,
            "primary_metric_name": self.primary_metric_name,
            "secondary_metrics": self.secondary_metrics,
            "sample_size": self.sample_size,
            "confidence_interval_lower": self.confidence_interval_lower,
            "confidence_interval_upper": self.confidence_interval_upper,
            "standard_deviation": self.standard_deviation,
            "variance": self.variance,
            "measurement_timestamp": self.measurement_timestamp.isoformat() if self.measurement_timestamp else None,
            "measurement_period_start": self.measurement_period_start.isoformat() if self.measurement_period_start else None,
            "measurement_period_end": self.measurement_period_end.isoformat() if self.measurement_period_end else None,
            "data_quality_score": self.data_quality_score,
            "missing_data_percentage": self.missing_data_percentage,
            "outlier_percentage": self.outlier_percentage,
            "model_version": self.model_version,
            "dataset_version": self.dataset_version,
            "environment": self.environment,
            "region": self.region,
            "metadata": self.metadata
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> PerformanceDegradationMetrics:
        """Create from dictionary representation."""
        # Parse datetime fields
        measurement_timestamp = (
            datetime.fromisoformat(data["measurement_timestamp"]) 
            if data.get("measurement_timestamp") else None
        )
        measurement_period_start = (
            datetime.fromisoformat(data["measurement_period_start"]) 
            if data.get("measurement_period_start") else None
        )
        measurement_period_end = (
            datetime.fromisoformat(data["measurement_period_end"]) 
            if data.get("measurement_period_end") else None
        )
        
        # Parse enum
        primary_metric_type = DegradationMetricType(data.get("primary_metric_type", "accuracy"))
        
        return cls(
            primary_metric_type=primary_metric_type,
            primary_metric_value=data.get("primary_metric_value", 0.0),
            primary_metric_name=data.get("primary_metric_name", ""),
            secondary_metrics=data.get("secondary_metrics", {}),
            sample_size=data.get("sample_size", 0),
            confidence_interval_lower=data.get("confidence_interval_lower", 0.0),
            confidence_interval_upper=data.get("confidence_interval_upper", 0.0),
            standard_deviation=data.get("standard_deviation", 0.0),
            variance=data.get("variance", 0.0),
            measurement_timestamp=measurement_timestamp,
            measurement_period_start=measurement_period_start,
            measurement_period_end=measurement_period_end,
            data_quality_score=data.get("data_quality_score", 1.0),
            missing_data_percentage=data.get("missing_data_percentage", 0.0),
            outlier_percentage=data.get("outlier_percentage", 0.0),
            model_version=data.get("model_version", ""),
            dataset_version=data.get("dataset_version", ""),
            environment=data.get("environment", ""),
            region=data.get("region", ""),
            metadata=data.get("metadata", {})
        )
    
    @classmethod
    def create_baseline(
        cls,
        metric_type: DegradationMetricType,
        metric_value: float,
        sample_size: int,
        model_version: str = "",
        environment: str = "production"
    ) -> PerformanceDegradationMetrics:
        """Create a baseline metrics object."""
        return cls(
            primary_metric_type=metric_type,
            primary_metric_value=metric_value,
            primary_metric_name=metric_type.value,
            sample_size=sample_size,
            measurement_timestamp=datetime.utcnow(),
            model_version=model_version,
            environment=environment,
            data_quality_score=1.0
        )