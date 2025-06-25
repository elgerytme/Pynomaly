"""Real-time streaming session domain entities."""

from __future__ import annotations

from datetime import datetime, timedelta
from enum import Enum
from typing import Any
from uuid import UUID, uuid4

from pydantic import BaseModel, Field


class SessionStatus(str, Enum):
    """Streaming session status enumeration."""

    PENDING = "pending"
    STARTING = "starting"
    ACTIVE = "active"
    PAUSED = "paused"
    STOPPING = "stopping"
    STOPPED = "stopped"
    ERROR = "error"
    EXPIRED = "expired"


class ProcessingMode(str, Enum):
    """Data processing mode enumeration."""

    REAL_TIME = "real_time"
    MICRO_BATCH = "micro_batch"
    BATCH = "batch"
    SLIDING_WINDOW = "sliding_window"
    TUMBLING_WINDOW = "tumbling_window"


class DataFormat(str, Enum):
    """Supported data formats for streaming."""

    JSON = "json"
    AVRO = "avro"
    PROTOBUF = "protobuf"
    CSV = "csv"
    PARQUET = "parquet"
    BINARY = "binary"


class StreamingConfiguration(BaseModel):
    """Configuration for streaming session."""

    # Processing configuration
    processing_mode: ProcessingMode = Field(default=ProcessingMode.REAL_TIME, description="Data processing mode")
    batch_size: int = Field(default=1000, description="Batch size for micro-batch processing")
    window_size: timedelta = Field(default=timedelta(minutes=5), description="Window size for windowed processing")
    window_slide: timedelta = Field(default=timedelta(minutes=1), description="Window slide interval")
    
    # Data configuration
    input_format: DataFormat = Field(default=DataFormat.JSON, description="Input data format")
    output_format: DataFormat = Field(default=DataFormat.JSON, description="Output data format")
    schema_validation: bool = Field(default=True, description="Enable schema validation")
    compression: str | None = Field(None, description="Compression algorithm (gzip, lz4, snappy)")
    
    # Performance configuration
    max_throughput: int = Field(default=10000, description="Maximum messages per second")
    buffer_size: int = Field(default=10000, description="Internal buffer size")
    parallelism: int = Field(default=4, description="Number of parallel processing threads")
    
    # Reliability configuration
    enable_checkpointing: bool = Field(default=True, description="Enable periodic checkpointing")
    checkpoint_interval: timedelta = Field(default=timedelta(minutes=1), description="Checkpoint interval")
    max_retries: int = Field(default=3, description="Maximum retry attempts for failed messages")
    retry_backoff: timedelta = Field(default=timedelta(seconds=1), description="Retry backoff interval")
    
    # Quality configuration
    enable_deduplication: bool = Field(default=True, description="Enable message deduplication")
    late_data_tolerance: timedelta = Field(default=timedelta(minutes=10), description="Late data tolerance")
    watermark_interval: timedelta = Field(default=timedelta(seconds=30), description="Watermark generation interval")
    
    # Monitoring configuration
    enable_metrics: bool = Field(default=True, description="Enable metrics collection")
    metrics_interval: timedelta = Field(default=timedelta(seconds=30), description="Metrics reporting interval")
    enable_profiling: bool = Field(default=False, description="Enable performance profiling")
    
    # Advanced configuration
    custom_config: dict[str, Any] = Field(default_factory=dict, description="Custom configuration parameters")


class StreamingMetrics(BaseModel):
    """Real-time metrics for streaming session."""

    # Throughput metrics
    messages_processed: int = Field(default=0, description="Total messages processed")
    messages_per_second: float = Field(default=0.0, description="Current messages per second")
    avg_processing_time: float = Field(default=0.0, description="Average processing time per message (ms)")
    
    # Quality metrics
    anomalies_detected: int = Field(default=0, description="Total anomalies detected")
    anomaly_rate: float = Field(default=0.0, description="Current anomaly detection rate")
    false_positive_rate: float = Field(default=0.0, description="Estimated false positive rate")
    
    # Error metrics
    failed_messages: int = Field(default=0, description="Total failed messages")
    error_rate: float = Field(default=0.0, description="Current error rate")
    retried_messages: int = Field(default=0, description="Total retried messages")
    
    # Latency metrics
    end_to_end_latency: float = Field(default=0.0, description="End-to-end latency (ms)")
    p50_latency: float = Field(default=0.0, description="50th percentile latency (ms)")
    p95_latency: float = Field(default=0.0, description="95th percentile latency (ms)")
    p99_latency: float = Field(default=0.0, description="99th percentile latency (ms)")
    
    # Resource metrics
    cpu_usage: float = Field(default=0.0, description="CPU usage percentage")
    memory_usage: float = Field(default=0.0, description="Memory usage (MB)")
    network_io: dict[str, float] = Field(default_factory=dict, description="Network I/O statistics")
    
    # Backpressure metrics
    buffer_utilization: float = Field(default=0.0, description="Buffer utilization percentage")
    backpressure_events: int = Field(default=0, description="Number of backpressure events")
    
    # Window metrics (for windowed processing)
    active_windows: int = Field(default=0, description="Number of active windows")
    completed_windows: int = Field(default=0, description="Number of completed windows")
    late_messages: int = Field(default=0, description="Number of late messages")
    
    # Checkpoint metrics
    last_checkpoint: datetime | None = Field(None, description="Last successful checkpoint time")
    checkpoint_size: int = Field(default=0, description="Last checkpoint size (bytes)")
    
    # Custom metrics
    custom_metrics: dict[str, float] = Field(default_factory=dict, description="Custom application metrics")
    
    # Metadata
    measurement_time: datetime = Field(default_factory=datetime.utcnow, description="Metrics measurement time")


class StreamingDataSource(BaseModel):
    """Configuration for streaming data source."""

    source_type: str = Field(..., description="Source type (kafka, kinesis, pubsub, websocket, file)")
    connection_config: dict[str, Any] = Field(..., description="Source-specific connection configuration")
    topic_pattern: str | None = Field(None, description="Topic/stream pattern to consume")
    consumer_group: str | None = Field(None, description="Consumer group identifier")
    start_position: str = Field(default="latest", description="Start position (latest, earliest, timestamp)")
    authentication: dict[str, Any] = Field(default_factory=dict, description="Authentication configuration")
    ssl_config: dict[str, Any] = Field(default_factory=dict, description="SSL/TLS configuration")


class StreamingDataSink(BaseModel):
    """Configuration for streaming data sink."""

    sink_type: str = Field(..., description="Sink type (kafka, kinesis, pubsub, websocket, database, file)")
    connection_config: dict[str, Any] = Field(..., description="Sink-specific connection configuration")
    output_topic: str | None = Field(None, description="Output topic/stream")
    partitioning_key: str | None = Field(None, description="Partitioning key field")
    serialization_config: dict[str, Any] = Field(default_factory=dict, description="Serialization configuration")
    authentication: dict[str, Any] = Field(default_factory=dict, description="Authentication configuration")
    ssl_config: dict[str, Any] = Field(default_factory=dict, description="SSL/TLS configuration")


class StreamingSession(BaseModel):
    """Real-time streaming session for anomaly detection."""

    id: UUID = Field(default_factory=uuid4, description="Session identifier")
    name: str = Field(..., description="Session name")
    description: str | None = Field(None, description="Session description")
    
    # Model configuration
    detector_id: UUID = Field(..., description="Anomaly detector identifier")
    model_version: str | None = Field(None, description="Specific model version to use")
    
    # Stream configuration
    data_source: StreamingDataSource = Field(..., description="Input data source configuration")
    data_sink: StreamingDataSink | None = Field(None, description="Output data sink configuration")
    configuration: StreamingConfiguration = Field(..., description="Streaming configuration")
    
    # Session state
    status: SessionStatus = Field(default=SessionStatus.PENDING, description="Session status")
    current_metrics: StreamingMetrics = Field(default_factory=StreamingMetrics, description="Current session metrics")
    
    # Timing information
    created_at: datetime = Field(default_factory=datetime.utcnow, description="Session creation time")
    started_at: datetime | None = Field(None, description="Session start time")
    stopped_at: datetime | None = Field(None, description="Session stop time")
    last_activity: datetime | None = Field(None, description="Last activity timestamp")
    
    # Session management
    max_duration: timedelta | None = Field(None, description="Maximum session duration")
    auto_stop_on_idle: bool = Field(default=True, description="Auto-stop on idle")
    idle_timeout: timedelta = Field(default=timedelta(hours=1), description="Idle timeout duration")
    
    # Error handling
    error_message: str | None = Field(None, description="Error message if status is ERROR")
    error_details: dict[str, Any] = Field(default_factory=dict, description="Detailed error information")
    
    # Metadata
    created_by: str = Field(..., description="User who created the session")
    tags: list[str] = Field(default_factory=list, description="Session tags")
    metadata: dict[str, Any] = Field(default_factory=dict, description="Additional metadata")

    def start_session(self) -> None:
        """Start the streaming session."""
        if self.status != SessionStatus.PENDING:
            raise ValueError(f"Cannot start session in {self.status} status")
        
        self.status = SessionStatus.STARTING
        self.started_at = datetime.utcnow()
        self.last_activity = datetime.utcnow()

    def activate_session(self) -> None:
        """Activate the streaming session."""
        if self.status != SessionStatus.STARTING:
            raise ValueError(f"Cannot activate session in {self.status} status")
        
        self.status = SessionStatus.ACTIVE
        self.last_activity = datetime.utcnow()

    def pause_session(self) -> None:
        """Pause the streaming session."""
        if self.status != SessionStatus.ACTIVE:
            raise ValueError(f"Cannot pause session in {self.status} status")
        
        self.status = SessionStatus.PAUSED

    def resume_session(self) -> None:
        """Resume the streaming session."""
        if self.status != SessionStatus.PAUSED:
            raise ValueError(f"Cannot resume session in {self.status} status")
        
        self.status = SessionStatus.ACTIVE
        self.last_activity = datetime.utcnow()

    def stop_session(self, error_message: str | None = None) -> None:
        """Stop the streaming session."""
        if self.status in [SessionStatus.STOPPED, SessionStatus.ERROR]:
            return
        
        if error_message:
            self.status = SessionStatus.ERROR
            self.error_message = error_message
        else:
            self.status = SessionStatus.STOPPING
        
        self.stopped_at = datetime.utcnow()

    def complete_stop(self) -> None:
        """Complete the session stop process."""
        if self.status == SessionStatus.STOPPING:
            self.status = SessionStatus.STOPPED

    def update_activity(self) -> None:
        """Update last activity timestamp."""
        self.last_activity = datetime.utcnow()

    def update_metrics(self, metrics: StreamingMetrics) -> None:
        """Update session metrics."""
        self.current_metrics = metrics
        self.update_activity()

    def is_active(self) -> bool:
        """Check if session is active."""
        return self.status == SessionStatus.ACTIVE

    def is_expired(self) -> bool:
        """Check if session is expired."""
        if self.max_duration and self.started_at:
            return datetime.utcnow() > self.started_at + self.max_duration
        return False

    def is_idle(self) -> bool:
        """Check if session is idle."""
        if not self.last_activity:
            return False
        return datetime.utcnow() > self.last_activity + self.idle_timeout

    def get_uptime(self) -> timedelta | None:
        """Get session uptime."""
        if not self.started_at:
            return None
        
        end_time = self.stopped_at or datetime.utcnow()
        return end_time - self.started_at

    def get_throughput_summary(self) -> dict[str, Any]:
        """Get throughput summary."""
        uptime = self.get_uptime()
        if not uptime or uptime.total_seconds() == 0:
            return {"avg_throughput": 0.0, "total_processed": 0}
        
        avg_throughput = self.current_metrics.messages_processed / uptime.total_seconds()
        
        return {
            "avg_throughput": avg_throughput,
            "current_throughput": self.current_metrics.messages_per_second,
            "total_processed": self.current_metrics.messages_processed,
            "uptime_seconds": uptime.total_seconds(),
        }

    class Config:
        """Pydantic model configuration."""
        
        validate_assignment = True
        use_enum_values = True
        json_encoders = {
            datetime: lambda v: v.isoformat(),
            timedelta: lambda v: v.total_seconds(),
        }


class StreamingAlert(BaseModel):
    """Alert configuration for streaming sessions."""

    id: UUID = Field(default_factory=uuid4, description="Alert identifier")
    session_id: UUID = Field(..., description="Associated session identifier")
    name: str = Field(..., description="Alert name")
    description: str | None = Field(None, description="Alert description")
    
    # Alert conditions
    metric_name: str = Field(..., description="Metric to monitor")
    threshold_value: float = Field(..., description="Alert threshold value")
    comparison_operator: str = Field(..., description="Comparison operator (>, <, >=, <=, ==)")
    duration_threshold: timedelta = Field(default=timedelta(minutes=1), description="Duration threshold")
    
    # Alert configuration
    severity: str = Field(default="medium", description="Alert severity (low, medium, high, critical)")
    enabled: bool = Field(default=True, description="Whether alert is enabled")
    
    # Notification configuration
    notification_channels: list[str] = Field(default_factory=list, description="Notification channels")
    notification_template: str | None = Field(None, description="Custom notification template")
    
    # State tracking
    is_triggered: bool = Field(default=False, description="Whether alert is currently triggered")
    trigger_count: int = Field(default=0, description="Number of times alert has been triggered")
    last_triggered: datetime | None = Field(None, description="Last trigger timestamp")
    last_resolved: datetime | None = Field(None, description="Last resolution timestamp")
    
    # Metadata
    created_at: datetime = Field(default_factory=datetime.utcnow, description="Alert creation time")
    created_by: str = Field(..., description="User who created the alert")
    tags: list[str] = Field(default_factory=list, description="Alert tags")
    metadata: dict[str, Any] = Field(default_factory=dict, description="Additional metadata")

    def evaluate_condition(self, current_value: float, duration_met: bool = True) -> bool:
        """Evaluate alert condition."""
        if not self.enabled:
            return False
        
        # Evaluate threshold condition
        if self.comparison_operator == ">":
            threshold_met = current_value > self.threshold_value
        elif self.comparison_operator == "<":
            threshold_met = current_value < self.threshold_value
        elif self.comparison_operator == ">=":
            threshold_met = current_value >= self.threshold_value
        elif self.comparison_operator == "<=":
            threshold_met = current_value <= self.threshold_value
        elif self.comparison_operator == "==":
            threshold_met = abs(current_value - self.threshold_value) < 1e-6
        else:
            threshold_met = False
        
        return threshold_met and duration_met

    def trigger_alert(self) -> None:
        """Trigger the alert."""
        if not self.is_triggered:
            self.is_triggered = True
            self.trigger_count += 1
            self.last_triggered = datetime.utcnow()

    def resolve_alert(self) -> None:
        """Resolve the alert."""
        if self.is_triggered:
            self.is_triggered = False
            self.last_resolved = datetime.utcnow()


class SessionSummary(BaseModel):
    """Summary information for streaming session."""

    session_id: UUID = Field(..., description="Session identifier")
    name: str = Field(..., description="Session name")
    status: SessionStatus = Field(..., description="Session status")
    detector_id: UUID = Field(..., description="Detector identifier")
    created_at: datetime = Field(..., description="Creation timestamp")
    started_at: datetime | None = Field(None, description="Start timestamp")
    stopped_at: datetime | None = Field(None, description="Stop timestamp")
    uptime_seconds: float = Field(..., description="Session uptime in seconds")
    messages_processed: int = Field(..., description="Total messages processed")
    anomalies_detected: int = Field(..., description="Total anomalies detected")
    current_throughput: float = Field(..., description="Current throughput (msg/sec)")
    avg_throughput: float = Field(..., description="Average throughput (msg/sec)")
    error_rate: float = Field(..., description="Current error rate")
    anomaly_rate: float = Field(..., description="Current anomaly rate")
    created_by: str = Field(..., description="Session creator")