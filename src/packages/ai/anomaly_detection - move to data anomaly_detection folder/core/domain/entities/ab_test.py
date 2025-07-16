"""A/B testing domain entities for model experimentation."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from typing import Any
from uuid import UUID, uuid4


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


@dataclass
class TrafficSplit:
    """Traffic split configuration for A/B testing."""

    control_percentage: float
    treatment_percentage: float

    def __post_init__(self) -> None:
        """Validate traffic split percentages sum to 100."""
        if not (0 <= self.control_percentage <= 100):
            raise ValueError("Control percentage must be between 0 and 100")
        if not (0 <= self.treatment_percentage <= 100):
            raise ValueError("Treatment percentage must be between 0 and 100")

        total = self.control_percentage + self.treatment_percentage
        if abs(total - 100.0) > 0.01:  # Allow for floating point precision
            raise ValueError("Traffic split percentages must sum to 100")


@dataclass
class SuccessMetric:
    """Success metric definition for A/B testing."""

    name: str
    type: str  # accuracy, precision, recall, f1, custom
    target_value: float | None = None
    improvement_threshold: float = 0.05
    is_primary: bool = False


@dataclass
class ABTestConfiguration:
    """A/B test configuration."""

    traffic_split: TrafficSplit
    duration: timedelta
    success_metrics: list[SuccessMetric]
    min_sample_size: int = 100
    confidence_level: float = 0.95
    early_stopping_enabled: bool = True
    randomization_unit: str = "user"

    def __post_init__(self) -> None:
        """Validate configuration parameters."""
        if not (0.8 <= self.confidence_level <= 0.99):
            raise ValueError("Confidence level must be between 0.8 and 0.99")
        if self.min_sample_size < 1:
            raise ValueError("Minimum sample size must be at least 1")
        if not self.success_metrics:
            raise ValueError("At least one success metric must be defined")


@dataclass
class ABTestMetrics:
    """A/B test performance metrics."""

    control_sample_size: int = 0
    treatment_sample_size: int = 0
    control_metrics: dict[str, float] = field(default_factory=dict)
    treatment_metrics: dict[str, float] = field(default_factory=dict)
    statistical_significance: dict[str, bool] = field(default_factory=dict)
    p_values: dict[str, float] = field(default_factory=dict)
    confidence_intervals: dict[str, tuple[float, float]] = field(default_factory=dict)
    effect_sizes: dict[str, float] = field(default_factory=dict)


@dataclass
class ABTest:
    """A/B test entity for model experimentation."""

    name: str
    control_model_id: UUID
    treatment_model_id: UUID
    configuration: ABTestConfiguration
    created_by: str
    id: UUID = field(default_factory=uuid4)
    description: str | None = None
    status: ABTestStatus = ABTestStatus.DRAFT
    created_at: datetime = field(default_factory=datetime.utcnow)
    started_at: datetime | None = None
    ended_at: datetime | None = None
    updated_at: datetime = field(default_factory=datetime.utcnow)
    current_metrics: ABTestMetrics = field(default_factory=ABTestMetrics)
    result: ABTestResult = ABTestResult.PENDING
    conclusion: str | None = None
    tags: list[str] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)

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

    def complete_test(
        self, result: ABTestResult, conclusion: str | None = None
    ) -> None:
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
            self.current_metrics.control_sample_size
            + self.current_metrics.treatment_sample_size
        )
        return total_samples >= self.configuration.min_sample_size

    def get_winning_model_id(self) -> UUID | None:
        """Get the ID of the winning model."""
        if self.result == ABTestResult.CONTROL_WINS:
            return self.control_model_id
        elif self.result == ABTestResult.TREATMENT_WINS:
            return self.treatment_model_id
        return None


@dataclass
class ABTestSummary:
    """A/B test summary for reporting."""

    test_id: UUID
    name: str
    status: ABTestStatus
    result: ABTestResult
    control_model_id: UUID
    treatment_model_id: UUID
    sample_size: int

    started_at: datetime | None = None
    ended_at: datetime | None = None
    primary_metric_improvement: float | None = None
    statistical_significance: bool = False
    winning_model_id: UUID | None = None
