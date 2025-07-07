"""Anomaly event domain entities for real-time processing."""

from __future__ import annotations

from datetime import datetime
from enum import Enum
from typing import Any
from uuid import UUID, uuid4

from pydantic import BaseModel, Field


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


class AnomalyEventData(BaseModel):
    """Specific data for anomaly events."""

    anomaly_score: float = Field(..., description="Anomaly score (0.0 to 1.0)")
    confidence: float = Field(..., description="Detection confidence (0.0 to 1.0)")
    feature_contributions: dict[str, float] = Field(
        default_factory=dict, description="Feature contribution scores"
    )
    predicted_class: str | None = Field(None, description="Predicted anomaly class")
    expected_range: dict[str, Any] = Field(
        default_factory=dict, description="Expected value ranges"
    )
    actual_values: dict[str, Any] = Field(
        default_factory=dict, description="Actual observed values"
    )
    detection_method: str | None = Field(None, description="Detection method used")
    model_version: str | None = Field(
        None, description="Model version that detected the anomaly"
    )
    explanation: str | None = Field(None, description="Human-readable explanation")


class DataQualityEventData(BaseModel):
    """Specific data for data quality events."""

    issue_type: str = Field(..., description="Type of data quality issue")
    affected_fields: list[str] = Field(..., description="Fields affected by the issue")
    severity_score: float = Field(..., description="Severity score (0.0 to 1.0)")
    missing_percentage: float | None = Field(
        None, description="Percentage of missing values"
    )
    outlier_percentage: float | None = Field(
        None, description="Percentage of outlier values"
    )
    schema_violations: list[str] = Field(
        default_factory=list, description="Schema violations detected"
    )
    data_drift_score: float | None = Field(
        None, description="Data drift score if applicable"
    )
    recommendations: list[str] = Field(
        default_factory=list, description="Recommended actions"
    )


class PerformanceEventData(BaseModel):
    """Specific data for performance events."""

    metric_name: str = Field(..., description="Performance metric name")
    current_value: float = Field(..., description="Current metric value")
    baseline_value: float = Field(..., description="Baseline metric value")
    degradation_percentage: float = Field(
        ..., description="Performance degradation percentage"
    )
    threshold_value: float = Field(..., description="Alert threshold value")
    trend_direction: str = Field(
        ..., description="Trend direction (increasing, decreasing, stable)"
    )
    affected_components: list[str] = Field(
        default_factory=list, description="Affected system components"
    )
    potential_causes: list[str] = Field(
        default_factory=list, description="Potential root causes"
    )


class AnomalyEvent(BaseModel):
    """Real-time anomaly event for streaming processing."""

    id: UUID = Field(default_factory=uuid4, description="Event identifier")

    # Event classification
    event_type: EventType = Field(..., description="Type of event")
    severity: EventSeverity = Field(..., description="Event severity")
    status: EventStatus = Field(
        default=EventStatus.PENDING, description="Processing status"
    )

    # Source information
    source_session_id: UUID | None = Field(
        None, description="Source streaming session ID"
    )
    detector_id: UUID | None = Field(
        None, description="Detector that generated the event"
    )
    data_source: str | None = Field(None, description="Original data source")

    # Event data
    title: str = Field(..., description="Event title")
    description: str = Field(..., description="Event description")
    raw_data: dict[str, Any] = Field(
        ..., description="Raw input data that triggered the event"
    )

    # Specific event data (polymorphic based on event type)
    anomaly_data: AnomalyEventData | None = Field(
        None, description="Anomaly-specific data"
    )
    data_quality_data: DataQualityEventData | None = Field(
        None, description="Data quality-specific data"
    )
    performance_data: PerformanceEventData | None = Field(
        None, description="Performance-specific data"
    )

    # Context information
    business_context: dict[str, Any] = Field(
        default_factory=dict, description="Business context data"
    )
    technical_context: dict[str, Any] = Field(
        default_factory=dict, description="Technical context data"
    )
    correlation_id: str | None = Field(
        None, description="Correlation ID for related events"
    )
    parent_event_id: UUID | None = Field(
        None, description="Parent event ID for hierarchical events"
    )

    # Timing information
    event_time: datetime = Field(..., description="When the event occurred")
    ingestion_time: datetime = Field(
        default_factory=datetime.utcnow, description="When event was ingested"
    )
    processing_time: datetime | None = Field(
        None, description="When event was processed"
    )

    # Geographic information
    location: dict[str, Any] = Field(
        default_factory=dict, description="Geographic location data"
    )
    timezone: str = Field(default="UTC", description="Event timezone")

    # Notification and routing
    notification_sent: bool = Field(
        default=False, description="Whether notification was sent"
    )
    notification_channels: list[str] = Field(
        default_factory=list, description="Notification channels used"
    )
    routing_key: str | None = Field(
        None, description="Routing key for event processing"
    )

    # Acknowledgment and resolution
    acknowledged_by: str | None = Field(
        None, description="User who acknowledged the event"
    )
    acknowledged_at: datetime | None = Field(
        None, description="Acknowledgment timestamp"
    )
    resolved_by: str | None = Field(None, description="User who resolved the event")
    resolved_at: datetime | None = Field(None, description="Resolution timestamp")
    resolution_notes: str | None = Field(None, description="Resolution notes")

    # Processing metadata
    processing_attempts: int = Field(
        default=0, description="Number of processing attempts"
    )
    last_error: str | None = Field(None, description="Last processing error")
    retry_after: datetime | None = Field(None, description="Next retry timestamp")

    # Metadata
    tags: list[str] = Field(default_factory=list, description="Event tags")
    metadata: dict[str, Any] = Field(
        default_factory=dict, description="Additional metadata"
    )
    custom_fields: dict[str, Any] = Field(
        default_factory=dict, description="Custom application fields"
    )

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

    class Config:
        """Pydantic model configuration."""

        validate_assignment = True
        use_enum_values = True
        json_encoders = {
            datetime: lambda v: v.isoformat(),
        }


class EventFilter(BaseModel):
    """Filter criteria for querying events."""

    event_types: list[EventType] | None = Field(
        None, description="Filter by event types"
    )
    severities: list[EventSeverity] | None = Field(
        None, description="Filter by severities"
    )
    statuses: list[EventStatus] | None = Field(None, description="Filter by statuses")
    detector_ids: list[UUID] | None = Field(None, description="Filter by detector IDs")
    session_ids: list[UUID] | None = Field(None, description="Filter by session IDs")
    data_sources: list[str] | None = Field(None, description="Filter by data sources")

    # Time-based filters
    event_time_start: datetime | None = Field(None, description="Event time start")
    event_time_end: datetime | None = Field(None, description="Event time end")
    ingestion_time_start: datetime | None = Field(
        None, description="Ingestion time start"
    )
    ingestion_time_end: datetime | None = Field(None, description="Ingestion time end")

    # Content filters
    title_contains: str | None = Field(None, description="Filter by title content")
    description_contains: str | None = Field(
        None, description="Filter by description content"
    )
    tags: list[str] | None = Field(None, description="Filter by tags")
    correlation_id: str | None = Field(None, description="Filter by correlation ID")

    # User filters
    acknowledged_by: str | None = Field(None, description="Filter by acknowledger")
    resolved_by: str | None = Field(None, description="Filter by resolver")

    # Anomaly-specific filters
    min_anomaly_score: float | None = Field(None, description="Minimum anomaly score")
    max_anomaly_score: float | None = Field(None, description="Maximum anomaly score")
    min_confidence: float | None = Field(None, description="Minimum confidence")

    # Geographic filters
    location_bounds: dict[str, Any] | None = Field(
        None, description="Geographic bounds filter"
    )

    # Pagination
    limit: int = Field(default=100, description="Maximum number of results")
    offset: int = Field(default=0, description="Result offset")
    sort_by: str = Field(default="event_time", description="Sort field")
    sort_order: str = Field(default="desc", description="Sort order (asc, desc)")


class EventAggregation(BaseModel):
    """Event aggregation result."""

    group_key: str = Field(..., description="Aggregation group key")
    count: int = Field(..., description="Number of events in group")
    min_severity: EventSeverity = Field(..., description="Minimum severity in group")
    max_severity: EventSeverity = Field(..., description="Maximum severity in group")
    first_event_time: datetime = Field(..., description="First event time in group")
    last_event_time: datetime = Field(..., description="Last event time in group")
    avg_anomaly_score: float | None = Field(None, description="Average anomaly score")
    unique_detectors: int = Field(..., description="Number of unique detectors")
    unique_sessions: int = Field(..., description="Number of unique sessions")
    resolved_count: int = Field(..., description="Number of resolved events")
    acknowledged_count: int = Field(..., description="Number of acknowledged events")


class EventPattern(BaseModel):
    """Detected event pattern."""

    id: UUID = Field(default_factory=uuid4, description="Pattern identifier")
    name: str = Field(..., description="Pattern name")
    description: str = Field(..., description="Pattern description")

    # Pattern definition
    pattern_type: str = Field(
        ..., description="Type of pattern (frequency, sequence, correlation)"
    )
    conditions: dict[str, Any] = Field(..., description="Pattern matching conditions")
    time_window: int = Field(
        ..., description="Time window for pattern detection (seconds)"
    )

    # Pattern statistics
    match_count: int = Field(default=0, description="Number of times pattern matched")
    confidence: float = Field(..., description="Pattern confidence score")
    last_matched: datetime | None = Field(None, description="Last time pattern matched")

    # Actions
    alert_enabled: bool = Field(
        default=True, description="Whether to alert on pattern match"
    )
    alert_threshold: int = Field(
        default=1, description="Number of matches before alerting"
    )

    # Metadata
    created_at: datetime = Field(
        default_factory=datetime.utcnow, description="Pattern creation time"
    )
    created_by: str = Field(..., description="Pattern creator")
    tags: list[str] = Field(default_factory=list, description="Pattern tags")
    metadata: dict[str, Any] = Field(
        default_factory=dict, description="Additional metadata"
    )


class EventSummary(BaseModel):
    """Summary statistics for events."""

    total_events: int = Field(..., description="Total number of events")
    events_by_type: dict[str, int] = Field(..., description="Events by type")
    events_by_severity: dict[str, int] = Field(..., description="Events by severity")
    events_by_status: dict[str, int] = Field(..., description="Events by status")
    anomaly_rate: float = Field(..., description="Anomaly detection rate")
    avg_anomaly_score: float | None = Field(None, description="Average anomaly score")
    resolution_rate: float = Field(..., description="Event resolution rate")
    avg_resolution_time: float | None = Field(
        None, description="Average resolution time (seconds)"
    )
    top_detectors: list[dict[str, Any]] = Field(
        ..., description="Top detectors by event count"
    )
    top_data_sources: list[dict[str, Any]] = Field(
        ..., description="Top data sources by event count"
    )
    time_range: dict[str, datetime] = Field(..., description="Time range of events")
    summary_generated_at: datetime = Field(
        default_factory=datetime.utcnow, description="Summary generation time"
    )
