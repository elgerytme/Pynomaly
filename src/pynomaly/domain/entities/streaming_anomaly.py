"""Real-time streaming anomaly detection domain entities."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from typing import Any
from uuid import UUID


class StreamingMode(str, Enum):
    """Streaming processing modes."""

    REAL_TIME = "real_time"
    NEAR_REAL_TIME = "near_real_time"
    BATCH = "batch"
    MICRO_BATCH = "micro_batch"


class StreamState(str, Enum):
    """Stream processing states."""

    IDLE = "idle"
    STARTING = "starting"
    RUNNING = "running"
    PAUSING = "pausing"
    PAUSED = "paused"
    STOPPING = "stopping"
    STOPPED = "stopped"
    ERROR = "error"
    RECOVERING = "recovering"


class BackpressureStrategy(str, Enum):
    """Backpressure handling strategies."""

    DROP_OLDEST = "drop_oldest"
    DROP_NEWEST = "drop_newest"
    BLOCK = "block"
    BUFFER = "buffer"
    SAMPLE = "sample"


class AlertSeverity(str, Enum):
    """Alert severity levels."""

    INFO = "info"
    WARNING = "warning"
    ERROR = "error"
    CRITICAL = "critical"


@dataclass
class StreamingMetrics:
    """Real-time metrics for streaming anomaly detection."""

    # Throughput metrics
    records_processed: int = 0
    records_per_second: float = 0.0
    bytes_processed: int = 0
    bytes_per_second: float = 0.0

    # Detection metrics
    anomalies_detected: int = 0
    anomaly_rate: float = 0.0
    false_positive_rate: float = 0.0
    false_negative_rate: float = 0.0

    # Performance metrics
    avg_processing_latency: float = 0.0
    max_processing_latency: float = 0.0
    min_processing_latency: float = 0.0
    memory_usage: float = 0.0
    cpu_usage: float = 0.0

    # Quality metrics
    model_accuracy: float | None = None
    model_confidence: float | None = None
    data_quality_score: float | None = None

    # Buffer and queue metrics
    input_queue_size: int = 0
    output_queue_size: int = 0
    max_queue_size: int = 1000
    dropped_records: int = 0

    # Time tracking
    last_updated: datetime = field(default_factory=datetime.utcnow)

    def __post_init__(self) -> None:
        """Validate streaming metrics."""
        if self.records_processed < 0:
            raise ValueError("Records processed must be non-negative")
        if self.anomaly_rate < 0 or self.anomaly_rate > 1:
            raise ValueError("Anomaly rate must be between 0 and 1")
        if self.memory_usage < 0:
            raise ValueError("Memory usage must be non-negative")

    def update_throughput(self, records: int, duration: float) -> None:
        """Update throughput metrics."""
        if duration > 0:
            self.records_per_second = records / duration
        self.records_processed += records
        self.last_updated = datetime.utcnow()

    def update_latency(self, latency: float) -> None:
        """Update latency metrics."""
        if self.avg_processing_latency == 0:
            self.avg_processing_latency = latency
        else:
            # Simple moving average
            self.avg_processing_latency = (self.avg_processing_latency + latency) / 2

        self.max_processing_latency = max(self.max_processing_latency, latency)
        if self.min_processing_latency == 0:
            self.min_processing_latency = latency
        else:
            self.min_processing_latency = min(self.min_processing_latency, latency)

    def calculate_anomaly_rate(self) -> float:
        """Calculate current anomaly detection rate."""
        if self.records_processed == 0:
            return 0.0
        return self.anomalies_detected / self.records_processed

    def is_healthy(self) -> bool:
        """Check if streaming metrics indicate healthy processing."""
        return (
            self.cpu_usage < 90.0
            and self.memory_usage < 85.0
            and self.input_queue_size < self.max_queue_size * 0.8
            and self.avg_processing_latency < 1000.0  # 1 second threshold
        )


@dataclass
class StreamingWindow:
    """Time-based or count-based window for streaming data."""

    window_type: str  # "time" or "count"
    size: int | timedelta
    slide: int | timedelta | None = None

    # Current window state
    current_data: list[Any] = field(default_factory=list)
    window_start: datetime | None = None
    window_end: datetime | None = None

    def __post_init__(self) -> None:
        """Validate streaming window configuration."""
        if self.window_type not in ["time", "count"]:
            raise ValueError("Window type must be 'time' or 'count'")

        if self.window_type == "time":
            if not isinstance(self.size, timedelta):
                raise ValueError("Time window size must be a timedelta")
            if self.slide and not isinstance(self.slide, timedelta):
                raise ValueError("Time window slide must be a timedelta")
        else:  # count window
            if not isinstance(self.size, int) or self.size <= 0:
                raise ValueError("Count window size must be a positive integer")
            if self.slide and (not isinstance(self.slide, int) or self.slide <= 0):
                raise ValueError("Count window slide must be a positive integer")

    def add_data(self, data: Any, timestamp: datetime | None = None) -> bool:
        """Add data to window and return True if window is complete."""
        if timestamp is None:
            timestamp = datetime.utcnow()

        self.current_data.append((data, timestamp))

        if self.window_start is None:
            self.window_start = timestamp

        return self.is_complete()

    def is_complete(self) -> bool:
        """Check if window is complete and ready for processing."""
        if not self.current_data:
            return False

        if self.window_type == "count":
            return len(self.current_data) >= self.size
        else:  # time window
            if self.window_start is None:
                return False
            latest_time = self.current_data[-1][1]
            return (latest_time - self.window_start) >= self.size

    def get_data(self) -> list[Any]:
        """Get data from current window."""
        return [item[0] for item in self.current_data]

    def slide_window(self) -> None:
        """Slide the window forward."""
        if not self.slide:
            # No sliding, clear entire window
            self.current_data.clear()
            self.window_start = None
            return

        if self.window_type == "count":
            # Remove slide number of oldest items
            slide_count = min(self.slide, len(self.current_data))
            self.current_data = self.current_data[slide_count:]
        else:  # time window
            # Remove items older than slide duration
            if self.window_start:
                new_start = self.window_start + self.slide
                self.current_data = [
                    (data, ts) for data, ts in self.current_data if ts >= new_start
                ]
                self.window_start = new_start


@dataclass
class StreamingAlert:
    """Alert for streaming anomaly detection."""

    # Identity
    alert_id: UUID
    stream_id: UUID
    detector_id: UUID

    # Alert details
    severity: AlertSeverity
    title: str
    message: str
    anomaly_score: float

    # Timing
    triggered_at: datetime

    # Data context
    triggering_data: dict[str, Any]
    window_data: list[Any] | None = None
    alert_window_start: datetime | None = None
    alert_window_end: datetime | None = None

    # Metadata
    tags: list[str] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        """Validate streaming alert."""
        if not (0.0 <= self.anomaly_score <= 1.0):
            raise ValueError("Anomaly score must be between 0.0 and 1.0")

    def get_duration(self) -> timedelta | None:
        """Get alert duration if window times are available."""
        if self.alert_window_start and self.alert_window_end:
            return self.alert_window_end - self.alert_window_start
        return None


@dataclass
class StreamingConfiguration:
    """Configuration for real-time streaming anomaly detection."""

    # Stream settings
    stream_id: UUID
    stream_name: str
    detector_id: UUID
    processing_mode: StreamingMode = StreamingMode.REAL_TIME
    detector_config: dict[str, Any] = field(default_factory=dict)

    # Window configuration
    window_config: StreamingWindow | None = None

    # Performance settings
    batch_size: int = 100
    max_latency: timedelta = field(default_factory=lambda: timedelta(seconds=1))
    buffer_size: int = 10000
    backpressure_strategy: BackpressureStrategy = BackpressureStrategy.DROP_OLDEST

    # Alert settings
    alert_threshold: float = 0.7
    alert_cooldown: timedelta = field(default_factory=lambda: timedelta(minutes=5))

    # Quality settings
    enable_drift_detection: bool = True
    enable_model_updates: bool = False
    update_frequency: timedelta = field(default_factory=lambda: timedelta(hours=1))

    # Monitoring
    metrics_collection_interval: timedelta = field(
        default_factory=lambda: timedelta(seconds=30)
    )

    def __post_init__(self) -> None:
        """Validate streaming configuration."""
        if self.batch_size <= 0:
            raise ValueError("Batch size must be positive")
        if self.buffer_size <= 0:
            raise ValueError("Buffer size must be positive")
        if not (0.0 <= self.alert_threshold <= 1.0):
            raise ValueError("Alert threshold must be between 0.0 and 1.0")


@dataclass
class StreamingSession:
    """Active streaming anomaly detection session."""

    # Identity
    session_id: UUID
    configuration: StreamingConfiguration

    # Session state
    state: StreamState = StreamState.IDLE
    start_time: datetime | None = None
    end_time: datetime | None = None

    # Runtime data
    current_window: StreamingWindow | None = None
    metrics: StreamingMetrics = field(default_factory=StreamingMetrics)
    recent_alerts: list[StreamingAlert] = field(default_factory=list)

    # Error handling
    last_error: str | None = None
    error_count: int = 0
    recovery_attempts: int = 0

    # Callbacks and handlers
    alert_handlers: list[Callable[[StreamingAlert], None]] = field(default_factory=list)

    # Metadata
    created_at: datetime = field(default_factory=datetime.utcnow)
    updated_at: datetime = field(default_factory=datetime.utcnow)
    tags: list[str] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)

    def start_session(self) -> None:
        """Start the streaming session."""
        if self.state != StreamState.IDLE:
            raise ValueError(f"Cannot start session in {self.state} state")

        self.state = StreamState.STARTING
        self.start_time = datetime.utcnow()
        self.updated_at = datetime.utcnow()

        # Initialize window if configured
        if self.configuration.window_config:
            self.current_window = StreamingWindow(
                window_type=self.configuration.window_config.window_type,
                size=self.configuration.window_config.size,
                slide=self.configuration.window_config.slide,
            )

    def pause_session(self) -> None:
        """Pause the streaming session."""
        if self.state != StreamState.RUNNING:
            raise ValueError(f"Cannot pause session in {self.state} state")

        self.state = StreamState.PAUSING
        self.updated_at = datetime.utcnow()

    def resume_session(self) -> None:
        """Resume the streaming session."""
        if self.state != StreamState.PAUSED:
            raise ValueError(f"Cannot resume session in {self.state} state")

        self.state = StreamState.RUNNING
        self.updated_at = datetime.utcnow()

    def stop_session(self) -> None:
        """Stop the streaming session."""
        if self.state in [StreamState.STOPPED, StreamState.STOPPING]:
            return

        self.state = StreamState.STOPPING
        self.end_time = datetime.utcnow()
        self.updated_at = datetime.utcnow()

    def mark_running(self) -> None:
        """Mark session as running."""
        self.state = StreamState.RUNNING
        self.updated_at = datetime.utcnow()

    def mark_error(self, error_message: str) -> None:
        """Mark session as in error state."""
        self.state = StreamState.ERROR
        self.last_error = error_message
        self.error_count += 1
        self.updated_at = datetime.utcnow()

    def mark_stopped(self) -> None:
        """Mark session as stopped."""
        self.state = StreamState.STOPPED
        if not self.end_time:
            self.end_time = datetime.utcnow()
        self.updated_at = datetime.utcnow()

    def add_alert(self, alert: StreamingAlert) -> None:
        """Add alert to session."""
        self.recent_alerts.append(alert)

        # Keep only recent alerts (last 100)
        if len(self.recent_alerts) > 100:
            self.recent_alerts = self.recent_alerts[-100:]

        # Trigger alert handlers
        for handler in self.alert_handlers:
            try:
                handler(alert)
            except Exception:
                # Log error but don't fail the session
                pass

    def get_duration(self) -> timedelta | None:
        """Get session duration."""
        if not self.start_time:
            return None

        end_time = self.end_time or datetime.utcnow()
        return end_time - self.start_time

    def is_active(self) -> bool:
        """Check if session is actively processing."""
        return self.state in [
            StreamState.RUNNING,
            StreamState.STARTING,
            StreamState.PAUSING,
        ]

    def get_recent_alerts(
        self, severity: AlertSeverity | None = None
    ) -> list[StreamingAlert]:
        """Get recent alerts, optionally filtered by severity."""
        if severity is None:
            return self.recent_alerts
        return [alert for alert in self.recent_alerts if alert.severity == severity]

    def get_health_status(self) -> dict[str, Any]:
        """Get session health status."""
        return {
            "state": self.state.value,
            "is_healthy": self.metrics.is_healthy(),
            "error_count": self.error_count,
            "last_error": self.last_error,
            "duration": (
                self.get_duration().total_seconds() if self.get_duration() else None
            ),
            "metrics": {
                "records_processed": self.metrics.records_processed,
                "anomalies_detected": self.metrics.anomalies_detected,
                "anomaly_rate": self.metrics.anomaly_rate,
                "avg_latency": self.metrics.avg_processing_latency,
                "cpu_usage": self.metrics.cpu_usage,
                "memory_usage": self.metrics.memory_usage,
            },
            "alerts": {
                "total": len(self.recent_alerts),
                "critical": len(self.get_recent_alerts(AlertSeverity.CRITICAL)),
                "error": len(self.get_recent_alerts(AlertSeverity.ERROR)),
            },
        }


@dataclass
class StreamingQueryFilter:
    """Filter for querying streaming sessions and alerts."""

    # Session filters
    session_ids: list[UUID] | None = None
    states: list[StreamState] | None = None
    detector_ids: list[UUID] | None = None

    # Time filters
    created_after: datetime | None = None
    created_before: datetime | None = None
    active_during: tuple[datetime, datetime] | None = None

    # Performance filters
    min_records_processed: int | None = None
    min_anomalies_detected: int | None = None
    min_anomaly_rate: float | None = None

    # Alert filters
    alert_severities: list[AlertSeverity] | None = None
    min_alert_count: int | None = None

    # Pagination
    limit: int = 100
    offset: int = 0
    sort_by: str = "created_at"
    sort_order: str = "desc"

    def __post_init__(self) -> None:
        """Validate streaming query filter."""
        if self.limit <= 0:
            raise ValueError("Limit must be positive")
        if self.offset < 0:
            raise ValueError("Offset must be non-negative")
        if self.sort_order not in ["asc", "desc"]:
            raise ValueError("Sort order must be 'asc' or 'desc'")

        if self.min_anomaly_rate is not None and not (
            0.0 <= self.min_anomaly_rate <= 1.0
        ):
            raise ValueError("Min anomaly rate must be between 0.0 and 1.0")


@dataclass
class StreamingSummary:
    """Summary of streaming anomaly detection activity."""

    total_sessions: int
    active_sessions: int
    sessions_by_state: dict[str, int]
    total_records_processed: int
    total_anomalies_detected: float
    average_anomaly_rate: float

    alert_counts_by_severity: dict[str, int]
    top_detectors: list[dict[str, Any]] = field(default_factory=list)
    performance_metrics: dict[str, float] = field(default_factory=dict)

    time_range: dict[str, datetime] = field(default_factory=dict)
    summary_generated_at: datetime = field(default_factory=datetime.utcnow)

    def __post_init__(self) -> None:
        """Validate streaming summary."""
        if self.total_sessions < 0:
            raise ValueError("Total sessions must be non-negative")
        if self.active_sessions < 0:
            raise ValueError("Active sessions must be non-negative")
        if not (0.0 <= self.average_anomaly_rate <= 1.0):
            raise ValueError("Average anomaly rate must be between 0.0 and 1.0")
