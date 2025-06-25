"""A/B testing domain entities for model experimentation."""

from __future__ import annotations

from datetime import datetime, timedelta
from enum import Enum
from typing import Any
from uuid import UUID, uuid4

from pydantic import BaseModel, Field


class ABTestStatus(str, Enum):
    """A/B test status enumeration."""

    DRAFT = "draft"
    RUNNING = "running"
    PAUSED = "paused"
    COMPLETED = "completed"
    CANCELLED = "cancelled"


class ABTestResult(str, Enum):
    """A/B test result enumeration."""

    PENDING = "pending"
    CONTROL_WINS = "control_wins"
    TREATMENT_WINS = "treatment_wins"
    NO_SIGNIFICANT_DIFFERENCE = "no_significant_difference"
    INCONCLUSIVE = "inconclusive"


class TrafficSplit(BaseModel):
    """Traffic split configuration for A/B testing."""

    control_percentage: float = Field(
        ..., ge=0, le=100, description="Percentage of traffic to control model"
    )
    treatment_percentage: float = Field(
        ..., ge=0, le=100, description="Percentage of traffic to treatment model"
    )

    def model_post_init(self, __context: Any) -> None:
        """Validate traffic split percentages sum to 100."""
        total = self.control_percentage + self.treatment_percentage
        if abs(total - 100.0) > 0.01:  # Allow for floating point precision
            raise ValueError("Traffic split percentages must sum to 100")


class SuccessMetric(BaseModel):
    """Success metric definition for A/B testing."""

    name: str = Field(..., description="Metric name")
    type: str = Field(..., description="Metric type (accuracy, precision, recall, f1, custom)")
    target_value: float | None = Field(None, description="Target value for success")
    improvement_threshold: float = Field(
        0.05, description="Minimum improvement threshold for statistical significance"
    )
    is_primary: bool = Field(False, description="Whether this is the primary success metric")


class ABTestConfiguration(BaseModel):
    """A/B test configuration."""

    traffic_split: TrafficSplit = Field(..., description="Traffic allocation between models")
    duration: timedelta = Field(..., description="Test duration")
    min_sample_size: int = Field(100, description="Minimum sample size for valid results")
    confidence_level: float = Field(0.95, ge=0.8, le=0.99, description="Statistical confidence level")
    success_metrics: list[SuccessMetric] = Field(..., description="Success metrics to evaluate")
    early_stopping_enabled: bool = Field(True, description="Enable early stopping for clear results")
    randomization_unit: str = Field("user", description="Unit of randomization (user, session, request)")


class ABTestMetrics(BaseModel):
    """A/B test performance metrics."""

    control_sample_size: int = Field(0, description="Control group sample size")
    treatment_sample_size: int = Field(0, description="Treatment group sample size")
    control_metrics: dict[str, float] = Field(default_factory=dict, description="Control group metrics")
    treatment_metrics: dict[str, float] = Field(default_factory=dict, description="Treatment group metrics")
    statistical_significance: dict[str, bool] = Field(
        default_factory=dict, description="Statistical significance per metric"
    )
    p_values: dict[str, float] = Field(default_factory=dict, description="P-values per metric")
    confidence_intervals: dict[str, tuple[float, float]] = Field(
        default_factory=dict, description="Confidence intervals per metric"
    )
    effect_sizes: dict[str, float] = Field(default_factory=dict, description="Effect sizes per metric")


class ABTest(BaseModel):
    """A/B test entity for model experimentation."""

    id: UUID = Field(default_factory=uuid4, description="Unique test identifier")
    name: str = Field(..., description="Test name")
    description: str | None = Field(None, description="Test description")
    
    # Model references
    control_model_id: UUID = Field(..., description="Control model identifier")
    treatment_model_id: UUID = Field(..., description="Treatment model identifier")
    
    # Test configuration
    configuration: ABTestConfiguration = Field(..., description="Test configuration")
    
    # Test status and timing
    status: ABTestStatus = Field(default=ABTestStatus.DRAFT, description="Test status")
    created_at: datetime = Field(default_factory=datetime.utcnow, description="Creation timestamp")
    started_at: datetime | None = Field(None, description="Start timestamp")
    ended_at: datetime | None = Field(None, description="End timestamp")
    updated_at: datetime = Field(default_factory=datetime.utcnow, description="Last update timestamp")
    
    # Results and metrics
    current_metrics: ABTestMetrics = Field(
        default_factory=ABTestMetrics, description="Current test metrics"
    )
    result: ABTestResult = Field(default=ABTestResult.PENDING, description="Test result")
    conclusion: str | None = Field(None, description="Test conclusion and recommendation")
    
    # Metadata
    created_by: str = Field(..., description="User who created the test")
    tags: list[str] = Field(default_factory=list, description="Test tags")
    metadata: dict[str, Any] = Field(default_factory=dict, description="Additional metadata")

    def start_test(self) -> None:
        """Start the A/B test."""
        if self.status != ABTestStatus.DRAFT:
            raise ValueError(f"Cannot start test in {self.status} status")
        
        self.status = ABTestStatus.RUNNING
        self.started_at = datetime.utcnow()
        self.updated_at = datetime.utcnow()

    def pause_test(self) -> None:
        """Pause the A/B test."""
        if self.status != ABTestStatus.RUNNING:
            raise ValueError(f"Cannot pause test in {self.status} status")
        
        self.status = ABTestStatus.PAUSED
        self.updated_at = datetime.utcnow()

    def resume_test(self) -> None:
        """Resume the A/B test."""
        if self.status != ABTestStatus.PAUSED:
            raise ValueError(f"Cannot resume test in {self.status} status")
        
        self.status = ABTestStatus.RUNNING
        self.updated_at = datetime.utcnow()

    def complete_test(self, result: ABTestResult, conclusion: str | None = None) -> None:
        """Complete the A/B test with results."""
        if self.status not in [ABTestStatus.RUNNING, ABTestStatus.PAUSED]:
            raise ValueError(f"Cannot complete test in {self.status} status")
        
        self.status = ABTestStatus.COMPLETED
        self.ended_at = datetime.utcnow()
        self.updated_at = datetime.utcnow()
        self.result = result
        self.conclusion = conclusion

    def cancel_test(self, reason: str | None = None) -> None:
        """Cancel the A/B test."""
        if self.status == ABTestStatus.COMPLETED:
            raise ValueError("Cannot cancel completed test")
        
        self.status = ABTestStatus.CANCELLED
        self.ended_at = datetime.utcnow()
        self.updated_at = datetime.utcnow()
        if reason:
            self.metadata["cancellation_reason"] = reason

    def update_metrics(self, metrics: ABTestMetrics) -> None:
        """Update test metrics."""
        self.current_metrics = metrics
        self.updated_at = datetime.utcnow()

    def is_active(self) -> bool:
        """Check if test is currently active."""
        return self.status == ABTestStatus.RUNNING

    def is_completed(self) -> bool:
        """Check if test is completed."""
        return self.status == ABTestStatus.COMPLETED

    def get_duration(self) -> timedelta | None:
        """Get actual test duration."""
        if not self.started_at:
            return None
        
        end_time = self.ended_at or datetime.utcnow()
        return end_time - self.started_at

    def has_sufficient_sample_size(self) -> bool:
        """Check if test has sufficient sample size."""
        total_samples = (
            self.current_metrics.control_sample_size +
            self.current_metrics.treatment_sample_size
        )
        return total_samples >= self.configuration.min_sample_size

    def get_winning_model_id(self) -> UUID | None:
        """Get the ID of the winning model."""
        if self.result == ABTestResult.CONTROL_WINS:
            return self.control_model_id
        elif self.result == ABTestResult.TREATMENT_WINS:
            return self.treatment_model_id
        return None

    class Config:
        """Pydantic model configuration."""
        
        validate_assignment = True
        use_enum_values = True
        json_encoders = {
            datetime: lambda v: v.isoformat(),
            timedelta: lambda v: v.total_seconds(),
        }


class ABTestSummary(BaseModel):
    """A/B test summary for reporting."""

    test_id: UUID = Field(..., description="Test identifier")
    name: str = Field(..., description="Test name")
    status: ABTestStatus = Field(..., description="Test status")
    result: ABTestResult = Field(..., description="Test result")
    control_model_id: UUID = Field(..., description="Control model identifier")
    treatment_model_id: UUID = Field(..., description="Treatment model identifier")
    started_at: datetime | None = Field(None, description="Start timestamp")
    ended_at: datetime | None = Field(None, description="End timestamp")
    sample_size: int = Field(..., description="Total sample size")
    primary_metric_improvement: float | None = Field(
        None, description="Primary metric improvement percentage"
    )
    statistical_significance: bool = Field(
        False, description="Whether results are statistically significant"
    )
    winning_model_id: UUID | None = Field(None, description="Winning model identifier")