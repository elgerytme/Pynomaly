"""Data Transfer Objects for real-time monitoring system."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any
from uuid import UUID


class MetricType(Enum):
    """Types of metrics that can be monitored."""

    COUNTER = "counter"
    GAUGE = "gauge"
    HISTOGRAM = "histogram"
    SUMMARY = "summary"
    TIMER = "timer"


class AlertLevel(Enum):
    """Alert severity levels."""

    INFO = "info"
    WARNING = "warning"
    ERROR = "error"
    CRITICAL = "critical"


class SystemHealth(Enum):
    """System health status."""

    HEALTHY = "healthy"
    DEGRADED = "degraded"
    UNHEALTHY = "unhealthy"
    CRITICAL = "critical"


@dataclass
class MetricDataPoint:
    """Single metric data point."""

    timestamp: datetime
    value: float
    labels: dict[str, str] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "timestamp": self.timestamp.isoformat(),
            "value": self.value,
            "labels": self.labels,
        }


@dataclass
class MetricSeries:
    """Time series of metric data points."""

    metric_name: str
    metric_type: MetricType
    description: str = ""
    unit: str = ""
    data_points: list[MetricDataPoint] = field(default_factory=list)

    def add_data_point(self, value: float, timestamp: datetime | None = None, labels: dict[str, str] | None = None) -> None:
        """Add a new data point to the series."""
        if timestamp is None:
            timestamp = datetime.utcnow()
        if labels is None:
            labels = {}

        self.data_points.append(MetricDataPoint(timestamp=timestamp, value=value, labels=labels))

        # Keep only last 1000 points to prevent memory issues
        if len(self.data_points) > 1000:
            self.data_points = self.data_points[-1000:]

    def get_latest_value(self) -> float | None:
        """Get the most recent value."""
        if not self.data_points:
            return None
        return self.data_points[-1].value

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "metric_name": self.metric_name,
            "metric_type": self.metric_type.value,
            "description": self.description,
            "unit": self.unit,
            "data_points": [dp.to_dict() for dp in self.data_points],
            "latest_value": self.get_latest_value(),
        }


@dataclass
class SystemMetrics:
    """System-level metrics."""

    cpu_usage_percent: float = 0.0
    memory_usage_percent: float = 0.0
    disk_usage_percent: float = 0.0
    network_io_bytes: float = 0.0
    active_connections: int = 0
    uptime_seconds: float = 0.0
    timestamp: datetime = field(default_factory=datetime.utcnow)

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "cpu_usage_percent": self.cpu_usage_percent,
            "memory_usage_percent": self.memory_usage_percent,
            "disk_usage_percent": self.disk_usage_percent,
            "network_io_bytes": self.network_io_bytes,
            "active_connections": self.active_connections,
            "uptime_seconds": self.uptime_seconds,
            "timestamp": self.timestamp.isoformat(),
        }


@dataclass
class DetectionMetrics:
    """Anomaly detection specific metrics."""

    active_detections: int = 0
    total_detections_today: int = 0
    avg_detection_time_ms: float = 0.0
    anomaly_rate_percent: float = 0.0
    false_positive_rate_percent: float = 0.0
    precision: float = 0.0
    recall: float = 0.0
    f1_score: float = 0.0
    timestamp: datetime = field(default_factory=datetime.utcnow)

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "active_detections": self.active_detections,
            "total_detections_today": self.total_detections_today,
            "avg_detection_time_ms": self.avg_detection_time_ms,
            "anomaly_rate_percent": self.anomaly_rate_percent,
            "false_positive_rate_percent": self.false_positive_rate_percent,
            "precision": self.precision,
            "recall": self.recall,
            "f1_score": self.f1_score,
            "timestamp": self.timestamp.isoformat(),
        }


@dataclass
class AlertMessage:
    """Alert message for real-time notifications."""

    alert_id: UUID
    level: AlertLevel
    title: str
    message: str
    source: str
    timestamp: datetime = field(default_factory=datetime.utcnow)
    metadata: dict[str, Any] = field(default_factory=dict)
    acknowledged: bool = False

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "alert_id": str(self.alert_id),
            "level": self.level.value,
            "title": self.title,
            "message": self.message,
            "source": self.source,
            "timestamp": self.timestamp.isoformat(),
            "metadata": self.metadata,
            "acknowledged": self.acknowledged,
        }


@dataclass
class MonitoringDashboardRequest:
    """Request for monitoring dashboard data."""

    time_range_minutes: int = 60
    include_system_metrics: bool = True
    include_detection_metrics: bool = True
    include_alerts: bool = True
    metric_names: list[str] = field(default_factory=list)

    def validate(self) -> None:
        """Validate the request parameters."""
        if self.time_range_minutes <= 0:
            raise ValueError("Time range must be positive")
        if self.time_range_minutes > 1440:  # 24 hours
            raise ValueError("Time range cannot exceed 24 hours")


@dataclass
class MonitoringDashboardResponse:
    """Response with monitoring dashboard data."""

    system_health: SystemHealth
    system_metrics: SystemMetrics
    detection_metrics: DetectionMetrics
    metric_series: list[MetricSeries] = field(default_factory=list)
    active_alerts: list[AlertMessage] = field(default_factory=list)
    generated_at: datetime = field(default_factory=datetime.utcnow)

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "system_health": self.system_health.value,
            "system_metrics": self.system_metrics.to_dict(),
            "detection_metrics": self.detection_metrics.to_dict(),
            "metric_series": [series.to_dict() for series in self.metric_series],
            "active_alerts": [alert.to_dict() for alert in self.active_alerts],
            "generated_at": self.generated_at.isoformat(),
        }


@dataclass
class RealTimeUpdate:
    """Real-time update message for WebSocket."""

    update_type: str  # "metrics", "alert", "health", "detection"
    data: dict[str, Any]
    timestamp: datetime = field(default_factory=datetime.utcnow)

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "update_type": self.update_type,
            "data": self.data,
            "timestamp": self.timestamp.isoformat(),
        }
