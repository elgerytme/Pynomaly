"""Anomaly event domain entities for real-time processing."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any
from uuid import UUID, uuid4


class EventType(str, Enum):
    """Event type enumeration."""

    ANOMALY_DETECTED = "anomaly_detected"
    ANOMALY_RESOLVED = "anomaly_resolved"
    ANOMALY_ESCALATED = "anomaly_escalated"
    DATA_QUALITY_ISSUE = "data_quality_issue"
    MODEL_DRIFT_DETECTED = "model_drift_detected"
    PERFORMANCE_DEGRADATION = "performance_degradation"
    SYSTEM_ALERT = "system_alert"
    THRESHOLD_BREACH = "threshold_breach"
    PATTERN_CHANGE = "pattern_change"
    BATCH_COMPLETED = "batch_completed"
    SESSION_STARTED = "session_started"
    SESSION_STOPPED = "session_stopped"
    CUSTOM = "custom"


class EventSeverity(str, Enum):
    """Event severity enumeration."""

    INFO = "info"
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


class EventStatus(str, Enum):
    """Event processing status."""

    PENDING = "pending"
    PROCESSING = "processing"
    PROCESSED = "processed"
    FAILED = "failed"
    ACKNOWLEDGED = "acknowledged"
    RESOLVED = "resolved"
    IGNORED = "ignored"


@dataclass
class AnomalyEventData:
    """Specific data for anomaly events."""

    anomaly_score: float
    confidence: float
    feature_contributions: dict[str, float] = field(default_factory=dict)
    predicted_class: str | None = None
    expected_range: dict[str, Any] = field(default_factory=dict)
    actual_values: dict[str, Any] = field(default_factory=dict)
    detection_method: str | None = None
    model_version: str | None = None
    explanation: str | None = None

    def __post_init__(self) -> None:
        """Validate anomaly event data."""
        if not (0.0 <= self.anomaly_score <= 1.0):
            raise ValueError("Anomaly score must be between 0.0 and 1.0")
        if not (0.0 <= self.confidence <= 1.0):
            raise ValueError("Confidence must be between 0.0 and 1.0")


@dataclass
class DataQualityEventData:
    """Specific data for data quality events."""

    issue_type: str
    affected_fields: list[str]
    severity_score: float
    missing_percentage: float | None = None
    outlier_percentage: float | None = None
    schema_violations: list[str] = field(default_factory=list)
    data_drift_score: float | None = None
    recommendations: list[str] = field(default_factory=list)

    def __post_init__(self) -> None:
        """Validate data quality event data."""
        if not (0.0 <= self.severity_score <= 1.0):
            raise ValueError("Severity score must be between 0.0 and 1.0")
        if self.missing_percentage is not None and not (0.0 <= self.missing_percentage <= 100.0):
            raise ValueError("Missing percentage must be between 0.0 and 100.0")
        if self.outlier_percentage is not None and not (0.0 <= self.outlier_percentage <= 100.0):
            raise ValueError("Outlier percentage must be between 0.0 and 100.0")


@dataclass
class PerformanceEventData:
    """Specific data for performance events."""

    metric_name: str
    current_value: float
    baseline_value: float
    degradation_percentage: float
    threshold_value: float
    trend_direction: str
    affected_components: list[str] = field(default_factory=list)
    potential_causes: list[str] = field(default_factory=list)

    def __post_init__(self) -> None:
        """Validate performance event data."""
        valid_trends = {"increasing", "decreasing", "stable"}
        if self.trend_direction not in valid_trends:
            raise ValueError(f"Trend direction must be one of: {valid_trends}")


@dataclass
class AnomalyEvent:
    """Real-time anomaly event for streaming processing."""

    # Required fields
    event_type: EventType
    severity: EventSeverity
    title: str
    description: str
    raw_data: dict[str, Any]
    event_time: datetime

    # Auto-generated fields
    id: UUID = field(default_factory=uuid4)
    status: EventStatus = EventStatus.PENDING
    ingestion_time: datetime = field(default_factory=datetime.utcnow)

    # Source information
    source_session_id: UUID | None = None
    detector_id: UUID | None = None
    data_source: str | None = None

    # Specific event data (polymorphic based on event type)
    anomaly_data: AnomalyEventData | None = None
    data_quality_data: DataQualityEventData | None = None
    performance_data: PerformanceEventData | None = None

    # Context information
    business_context: dict[str, Any] = field(default_factory=dict)
    technical_context: dict[str, Any] = field(default_factory=dict)
    correlation_id: str | None = None
    parent_event_id: UUID | None = None

    # Timing information
    processing_time: datetime | None = None

    # Geographic information
    location: dict[str, Any] = field(default_factory=dict)
    timezone: str = "UTC"

    # Notification and routing
    notification_sent: bool = False
    notification_channels: list[str] = field(default_factory=list)
    routing_key: str | None = None

    # Acknowledgment and resolution
    acknowledged_by: str | None = None
    acknowledged_at: datetime | None = None
    resolved_by: str | None = None
    resolved_at: datetime | None = None
    resolution_notes: str | None = None

    # Processing metadata
    processing_attempts: int = 0
    last_error: str | None = None
    retry_after: datetime | None = None

    # Metadata
    tags: list[str] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)
    custom_fields: dict[str, Any] = field(default_factory=dict)

    def acknowledge(self, user: str, notes: str | None = None) -> None:
        """Acknowledge the event."""
        self.status = EventStatus.ACKNOWLEDGED
        self.acknowledged_by = user
        self.acknowledged_at = datetime.utcnow()
        if notes:
            self.metadata["acknowledgment_notes"] = notes

    def resolve(self, user: str, notes: str | None = None) -> None:
        """Resolve the event."""
        self.status = EventStatus.RESOLVED
        self.resolved_by = user
        self.resolved_at = datetime.utcnow()
        if notes:
            self.resolution_notes = notes

    def ignore(self, user: str, reason: str | None = None) -> None:
        """Ignore the event."""
        self.status = EventStatus.IGNORED
        self.metadata["ignored_by"] = user
        self.metadata["ignored_at"] = datetime.utcnow().isoformat()
        if reason:
            self.metadata["ignore_reason"] = reason

    def mark_processing(self) -> None:
        """Mark event as being processed."""
        self.status = EventStatus.PROCESSING
        self.processing_time = datetime.utcnow()
        self.processing_attempts += 1

    def mark_processed(self) -> None:
        """Mark event as successfully processed."""
        self.status = EventStatus.PROCESSED

    def mark_failed(self, error: str, retry_after: datetime | None = None) -> None:
        """Mark event as failed."""
        self.status = EventStatus.FAILED
        self.last_error = error
        self.retry_after = retry_after

    def is_actionable(self) -> bool:
        """Check if event requires action."""
        return self.severity in [
            EventSeverity.HIGH,
            EventSeverity.CRITICAL,
        ] and self.status in [EventStatus.PENDING, EventStatus.PROCESSING]

    def is_resolved(self) -> bool:
        """Check if event is resolved."""
        return self.status in [EventStatus.RESOLVED, EventStatus.IGNORED]

    def get_age(self) -> float:
        """Get event age in seconds."""
        return (datetime.utcnow() - self.event_time).total_seconds()

    def get_processing_duration(self) -> float | None:
        """Get processing duration in seconds."""
        if not self.processing_time:
            return None
        end_time = self.resolved_at or datetime.utcnow()
        return (end_time - self.processing_time).total_seconds()

    def add_correlation(self, correlation_id: str) -> None:
        """Add correlation ID to the event."""
        self.correlation_id = correlation_id

    def add_tag(self, tag: str) -> None:
        """Add a tag to the event."""
        if tag not in self.tags:
            self.tags.append(tag)

    def remove_tag(self, tag: str) -> None:
        """Remove a tag from the event."""
        if tag in self.tags:
            self.tags.remove(tag)



@dataclass
class EventFilter:
    """Filter criteria for querying events."""

    # Event classification filters
    event_types: list[EventType] | None = None
    severities: list[EventSeverity] | None = None
    statuses: list[EventStatus] | None = None
    detector_ids: list[UUID] | None = None
    session_ids: list[UUID] | None = None
    data_sources: list[str] | None = None

    # Time-based filters
    event_time_start: datetime | None = None
    event_time_end: datetime | None = None
    ingestion_time_start: datetime | None = None
    ingestion_time_end: datetime | None = None

    # Content filters
    title_contains: str | None = None
    description_contains: str | None = None
    tags: list[str] | None = None
    correlation_id: str | None = None

    # User filters
    acknowledged_by: str | None = None
    resolved_by: str | None = None

    # Anomaly-specific filters
    min_anomaly_score: float | None = None
    max_anomaly_score: float | None = None
    min_confidence: float | None = None

    # Geographic filters
    location_bounds: dict[str, Any] | None = None

    # Pagination
    limit: int = 100
    offset: int = 0
    sort_by: str = "event_time"
    sort_order: str = "desc"

    def __post_init__(self) -> None:
        """Validate filter parameters."""
        if self.limit <= 0:
            raise ValueError("Limit must be positive")
        if self.offset < 0:
            raise ValueError("Offset must be non-negative")
        if self.sort_order not in ["asc", "desc"]:
            raise ValueError("Sort order must be 'asc' or 'desc'")
        if self.min_anomaly_score is not None and not (0.0 <= self.min_anomaly_score <= 1.0):
            raise ValueError("Min anomaly score must be between 0.0 and 1.0")
        if self.max_anomaly_score is not None and not (0.0 <= self.max_anomaly_score <= 1.0):
            raise ValueError("Max anomaly score must be between 0.0 and 1.0")


@dataclass
class EventAggregation:
    """Event aggregation result."""

    group_key: str
    count: int
    min_severity: EventSeverity
    max_severity: EventSeverity
    first_event_time: datetime
    last_event_time: datetime
    unique_detectors: int
    unique_sessions: int
    resolved_count: int
    acknowledged_count: int
    avg_anomaly_score: float | None = None


@dataclass
class EventPattern:
    """Detected event pattern."""

    name: str
    description: str
    pattern_type: str
    conditions: dict[str, Any]
    time_window: int
    confidence: float
    created_by: str

    id: UUID = field(default_factory=uuid4)
    match_count: int = 0
    last_matched: datetime | None = None
    alert_enabled: bool = True
    alert_threshold: int = 1
    created_at: datetime = field(default_factory=datetime.utcnow)
    tags: list[str] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        """Validate event pattern."""
        valid_pattern_types = {"frequency", "sequence", "correlation"}
        if self.pattern_type not in valid_pattern_types:
            raise ValueError(f"Pattern type must be one of: {valid_pattern_types}")
        if self.time_window <= 0:
            raise ValueError("Time window must be positive")
        if not (0.0 <= self.confidence <= 1.0):
            raise ValueError("Confidence must be between 0.0 and 1.0")
        if self.alert_threshold <= 0:
            raise ValueError("Alert threshold must be positive")


@dataclass
class EventSummary:
    """Summary statistics for events."""

    total_events: int
    events_by_type: dict[str, int]
    events_by_severity: dict[str, int]
    events_by_status: dict[str, int]
    anomaly_rate: float
    resolution_rate: float
    top_detectors: list[dict[str, Any]]
    top_data_sources: list[dict[str, Any]]
    time_range: dict[str, datetime]

    avg_anomaly_score: float | None = None
    avg_resolution_time: float | None = None
    summary_generated_at: datetime = field(default_factory=datetime.utcnow)

    def __post_init__(self) -> None:
        """Validate event summary."""
        if self.total_events < 0:
            raise ValueError("Total events must be non-negative")
        if not (0.0 <= self.anomaly_rate <= 1.0):
            raise ValueError("Anomaly rate must be between 0.0 and 1.0")
        if not (0.0 <= self.resolution_rate <= 1.0):
            raise ValueError("Resolution rate must be between 0.0 and 1.0")
