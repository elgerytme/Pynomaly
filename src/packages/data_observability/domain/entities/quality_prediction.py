"""
Quality Prediction Domain Entities

Defines the domain model for predictive data quality monitoring,
including quality predictions, trends, and forecasting.
"""

from dataclasses import dataclass
from datetime import datetime, timedelta
from enum import Enum
from typing import Dict, List, Optional, Any, Union
from uuid import UUID, uuid4

from pydantic import BaseModel, ConfigDict, Enum, Field


class PredictionType(str):
    """Types of quality predictions."""
    
    QUALITY_DEGRADATION = "quality_degradation"
    FRESHNESS_DECAY = "freshness_decay"
    VOLUME_ANOMALY = "volume_anomaly"
    SCHEMA_DRIFT = "schema_drift"
    COMPLETENESS_DROP = "completeness_drop"
    ACCURACY_DECLINE = "accuracy_decline"
    CONSISTENCY_BREACH = "consistency_breach"


class PredictionConfidence(str, Enum):
    """Confidence levels for predictions."""
    
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    VERY_HIGH = "very_high"


class TrendDirection(str, Enum):
    """Direction of quality trends."""
    
    IMPROVING = "improving"
    STABLE = "stable"
    DEGRADING = "degrading"
    VOLATILE = "volatile"


class SeasonalPattern(str, Enum):
    """Types of seasonal patterns in data quality."""
    
    DAILY = "daily"
    WEEKLY = "weekly"
    MONTHLY = "monthly"
    QUARTERLY = "quarterly"
    YEARLY = "yearly"
    NONE = "none"


@dataclass
class QualityMetricPoint:
    """A single point in quality metric time series."""
    
    timestamp: datetime
    value: float
    metric_type: str
    context: Dict[str, Any] = None
    
    def __post_init__(self):
        if self.context is None:
            self.context = {}


class QualityTrend(BaseModel):
    """Represents a trend in data quality metrics."""
    
    id: UUID = Field(default_factory=uuid4)
    asset_id: UUID = Field(..., description="ID of the data asset")
    metric_type: str = Field(..., description="Type of quality metric")
    
    # Trend characteristics
    direction: TrendDirection = Field(..., description="Direction of the trend")
    slope: float = Field(..., description="Rate of change (positive or negative)")
    r_squared: float = Field(..., description="Coefficient of determination (0-1)")
    
    # Time period
    start_time: datetime = Field(..., description="Start of trend period")
    end_time: datetime = Field(..., description="End of trend period")
    data_points: int = Field(..., description="Number of data points in trend")
    
    # Statistical measures
    mean_value: float = Field(..., description="Mean value over the period")
    std_deviation: float = Field(..., description="Standard deviation")
    min_value: float = Field(..., description="Minimum value")
    max_value: float = Field(..., description="Maximum value")
    
    # Seasonal patterns
    seasonal_pattern: SeasonalPattern = Field(default=SeasonalPattern.NONE)
    seasonal_strength: float = Field(default=0.0, ge=0, le=1)
    
    # Anomalies in the trend
    anomaly_count: int = Field(default=0)
    anomaly_timestamps: List[datetime] = Field(default_factory=list)        use_enum_values = True
    
    def is_significant_trend(self, min_r_squared: float = 0.7) -> bool:
        """Check if trend is statistically significant."""
        return self.r_squared >= min_r_squared and self.data_points >= 10
    
    def get_trend_strength(self) -> str:
        """Get qualitative assessment of trend strength."""
        if self.r_squared >= 0.9:
            return "very_strong"
        elif self.r_squared >= 0.7:
            return "strong"
        elif self.r_squared >= 0.5:
            return "moderate"
        elif self.r_squared >= 0.3:
            return "weak"
        else:
            return "very_weak"


class QualityPrediction(BaseModel):
    """Represents a prediction about future data quality."""
    
    id: UUID = Field(default_factory=uuid4)
    asset_id: UUID = Field(..., description="ID of the data asset")
    prediction_type: PredictionType = Field(..., description="Type of prediction")
    
    # Prediction details
    predicted_value: float = Field(..., description="Predicted quality metric value")
    current_value: Optional[float] = None
    threshold_value: Optional[float] = None
    
    # Timing
    prediction_time: datetime = Field(default_factory=datetime.utcnow)
    target_time: datetime = Field(..., description="When the prediction applies")
    time_horizon: timedelta = Field(..., description="How far into the future")
    
    # Confidence and accuracy
    confidence: PredictionConfidence = Field(..., description="Confidence level")
    confidence_score: float = Field(..., ge=0, le=1, description="Numerical confidence (0-1)")
    prediction_interval: Optional[Dict[str, float]] = None  # {"lower": x, "upper": y}
    
    # Model information
    model_name: str = Field(..., description="Name of prediction model used")
    model_version: str = Field(default="1.0")
    features_used: List[str] = Field(default_factory=list)
    
    # Impact assessment
    impact_score: float = Field(default=0.0, ge=0, le=1)
    affected_systems: List[str] = Field(default_factory=list)
    
    # Recommendations
    recommended_actions: List[str] = Field(default_factory=list)
    
    # Tracking
    is_validated: bool = False
    actual_value: Optional[float] = None
    validation_time: Optional[datetime] = None
    prediction_error: Optional[float] = None        use_enum_values = True
    
    def validate_prediction(self, actual_value: float) -> None:
        """Validate the prediction against actual observed value."""
        self.actual_value = actual_value
        self.validation_time = datetime.utcnow()
        self.is_validated = True
        
        # Calculate prediction error
        self.prediction_error = abs(self.predicted_value - actual_value)
    
    def get_accuracy(self) -> Optional[float]:
        """Get prediction accuracy (0-1, higher is better)."""
        if not self.is_validated or self.actual_value is None:
            return None
        
        # Avoid division by zero
        if self.actual_value == 0:
            return 1.0 if self.predicted_value == 0 else 0.0
        
        # Calculate relative accuracy
        relative_error = abs(self.predicted_value - self.actual_value) / abs(self.actual_value)
        return max(0.0, 1.0 - relative_error)
    
    def is_within_interval(self) -> Optional[bool]:
        """Check if actual value falls within prediction interval."""
        if not self.is_validated or not self.prediction_interval or self.actual_value is None:
            return None
        
        lower = self.prediction_interval.get("lower")
        upper = self.prediction_interval.get("upper")
        
        if lower is not None and upper is not None:
            return lower <= self.actual_value <= upper
        
        return None
    
    def get_severity_level(self) -> str:
        """Get severity level based on prediction type and impact."""
        if self.prediction_type in [
            PredictionType.QUALITY_DEGRADATION,
            PredictionType.ACCURACY_DECLINE,
            PredictionType.CONSISTENCY_BREACH
        ]:
            if self.impact_score >= 0.8:
                return "critical"
            elif self.impact_score >= 0.6:
                return "high"
            elif self.impact_score >= 0.4:
                return "medium"
            else:
                return "low"
        
        return "info"


class QualityForecast(BaseModel):
    """Represents a forecast of quality metrics over a time period."""
    
    id: UUID = Field(default_factory=uuid4)
    asset_id: UUID = Field(..., description="ID of the data asset")
    metric_type: str = Field(..., description="Type of quality metric")
    
    # Forecast details
    forecast_time: datetime = Field(default_factory=datetime.utcnow)
    forecast_horizon: timedelta = Field(..., description="How far into the future")
    resolution: timedelta = Field(default=timedelta(hours=1), description="Time resolution of forecast")
    
    # Forecast data
    forecasted_values: List[QualityMetricPoint] = Field(default_factory=list)
    confidence_intervals: Dict[str, List[QualityMetricPoint]] = Field(default_factory=dict)
    
    # Model information
    model_name: str = Field(..., description="Forecasting model used")
    model_accuracy: Optional[float] = None
    
    # Trend analysis
    overall_trend: TrendDirection = Field(..., description="Overall trend direction")
    trend_strength: float = Field(default=0.0, ge=0, le=1)
    
    # Seasonal components
    seasonal_component: List[float] = Field(default_factory=list)
    seasonal_pattern: SeasonalPattern = Field(default=SeasonalPattern.NONE)
    
    # Anomaly predictions
    predicted_anomalies: List[datetime] = Field(default_factory=list)
    anomaly_probabilities: List[float] = Field(default_factory=list)        use_enum_values = True
    
    def get_value_at_time(self, target_time: datetime) -> Optional[float]:
        """Get forecasted value at a specific time."""
        # Find the closest forecasted point
        closest_point = None
        min_diff = float('inf')
        
        for point in self.forecasted_values:
            diff = abs((target_time - point.timestamp).total_seconds())
            if diff < min_diff:
                min_diff = diff
                closest_point = point
        
        return closest_point.value if closest_point else None
    
    def get_trend_summary(self) -> Dict[str, Any]:
        """Get summary of forecast trends."""
        if not self.forecasted_values:
            return {}
        
        values = [point.value for point in self.forecasted_values]
        
        return {
            "overall_trend": self.overall_trend,
            "trend_strength": self.trend_strength,
            "forecast_range": {
                "min": min(values),
                "max": max(values),
                "start": values[0],
                "end": values[-1]
            },
            "volatility": self._calculate_volatility(values),
            "seasonal_pattern": self.seasonal_pattern,
            "anomaly_count": len(self.predicted_anomalies)
        }
    
    def _calculate_volatility(self, values: List[float]) -> float:
        """Calculate volatility of forecasted values."""
        if len(values) < 2:
            return 0.0
        
        # Calculate standard deviation of percentage changes
        changes = []
        for i in range(1, len(values)):
            if values[i-1] != 0:
                change = abs(values[i] - values[i-1]) / abs(values[i-1])
                changes.append(change)
        
        if not changes:
            return 0.0
        
        mean_change = sum(changes) / len(changes)
        variance = sum((c - mean_change) ** 2 for c in changes) / len(changes)
        
        return variance ** 0.5


class QualityAlert(BaseModel):
    """Represents an alert based on quality predictions."""
    
    id: UUID = Field(default_factory=uuid4)
    prediction_id: UUID = Field(..., description="ID of the triggering prediction")
    asset_id: UUID = Field(..., description="ID of the data asset")
    
    # Alert details
    alert_type: str = Field(..., description="Type of alert")
    severity: str = Field(..., description="Alert severity level")
    title: str = Field(..., description="Alert title")
    description: str = Field(..., description="Alert description")
    
    # Timing
    created_at: datetime = Field(default_factory=datetime.utcnow)
    expected_time: datetime = Field(..., description="When issue is expected to occur")
    
    # Actions
    recommended_actions: List[str] = Field(default_factory=list)
    assigned_to: Optional[str] = None
    status: str = Field(default="open")  # open, acknowledged, resolved, false_positive
    
    # Tracking
    acknowledged_at: Optional[datetime] = None
    resolved_at: Optional[datetime] = None        use_enum_values = True
    
    def acknowledge(self, user: str) -> None:
        """Acknowledge the alert."""
        self.status = "acknowledged"
        self.assigned_to = user
        self.acknowledged_at = datetime.utcnow()
    
    def resolve(self, user: str = None) -> None:
        """Resolve the alert."""
        self.status = "resolved"
        if user:
            self.assigned_to = user
        self.resolved_at = datetime.utcnow()
    
    def mark_false_positive(self, user: str = None) -> None:
        """Mark alert as false positive."""
        self.status = "false_positive"
        if user:
            self.assigned_to = user
        self.resolved_at = datetime.utcnow()