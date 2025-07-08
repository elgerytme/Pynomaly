"""Base schemas for metrics frames.

These schemas provide the foundation for real-time metric collection
and analysis, supporting various use cases such as anomaly detection,
system health checks, financial impact assessment, and more.

Schemas:
    MetricFrame: Base schema for all metric frames
    RealTimeMetricFrame: Enhanced metric frame with real-time properties
    MetricMetadata: Metadata associated with each metric
    TimestampedMetric: Base class including timestamp information
"""

from __future__ import annotations

from datetime import datetime

from pydantic import BaseModel, Field, validator


class TimestampedMetric(BaseModel):
    """Schema including timestamp information for metrics."""

    timestamp: datetime = Field(default_factory=datetime.utcnow)


class MetricMetadata(BaseModel):
    """Metadata associated with a metric, used to describe its context."""

    source: str | None
    labels: dict[str, str] | None
    unit: str | None


class MetricFrame(TimestampedMetric):
    """Base schema for all metric frames."""

    metric_id: str
    name: str
    value: float
    metadata: MetricMetadata | None = None


class RealTimeMetricFrame(MetricFrame):
    """Enhanced metric frame with real-time properties."""

    delay: float | None = Field(
        None,
        description="Time delay in seconds for metric collection"
    )

    @validator('value')
    def validate_value(cls, v: float) -> float:
        """Ensure the metric value is non-negative."""
        if v < 0:
            raise ValueError('Metric value must be non-negative')
        return v
