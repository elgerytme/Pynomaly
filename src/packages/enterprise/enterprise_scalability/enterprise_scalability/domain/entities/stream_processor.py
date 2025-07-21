"""
Stream Processing domain entities for real-time data processing.
"""

from datetime import datetime, timedelta
from enum import Enum
from typing import Dict, List, Optional, Any, Union, Callable
from uuid import UUID, uuid4

from pydantic import BaseModel, Field, validator


class StreamType(str, Enum):
    """Types of data streams."""
    KAFKA = "kafka"
    KINESIS = "kinesis"
    PUBSUB = "pubsub"
    REDIS_STREAMS = "redis_streams"
    RABBITMQ = "rabbitmq"
    WEBSOCKET = "websocket"
    FILE_STREAM = "file_stream"
    CUSTOM = "custom"


class ProcessorStatus(str, Enum):
    """Stream processor status."""
    PENDING = "pending"
    STARTING = "starting"
    RUNNING = "running"
    PAUSED = "paused"
    STOPPING = "stopping"
    STOPPED = "stopped"
    ERROR = "error"
    SCALING = "scaling"


class ProcessingMode(str, Enum):
    """Stream processing modes."""
    EXACTLY_ONCE = "exactly_once"
    AT_LEAST_ONCE = "at_least_once"
    AT_MOST_ONCE = "at_most_once"
    BEST_EFFORT = "best_effort"


class WindowType(str, Enum):
    """Stream windowing types."""
    TUMBLING = "tumbling"  # Fixed-size, non-overlapping windows
    SLIDING = "sliding"    # Fixed-size, overlapping windows
    SESSION = "session"    # Dynamic windows based on activity
    GLOBAL = "global"      # Single window for all data


class StreamSource(BaseModel):
    """
    Stream data source configuration.
    
    Defines the source of streaming data including
    connection details, serialization, and partitioning.
    """
    
    id: UUID = Field(default_factory=uuid4, description="Source identifier")
    
    # Source identification
    name: str = Field(..., description="Source name")
    stream_type: StreamType = Field(..., description="Type of stream")
    description: str = Field(default="", description="Source description")
    
    # Connection configuration
    connection_string: str = Field(..., description="Connection string/URL")
    topics: List[str] = Field(default_factory=list, description="Topics/channels to consume")
    partition_key: Optional[str] = Field(None, description="Partition key field")
    
    # Serialization
    data_format: str = Field(default="json", description="Data serialization format")
    schema_registry_url: Optional[str] = Field(None, description="Schema registry URL")
    schema_id: Optional[str] = Field(None, description="Schema identifier")
    
    # Consumer configuration
    consumer_group: str = Field(..., description="Consumer group ID")
    max_poll_records: int = Field(default=1000, ge=1, description="Max records per poll")
    poll_timeout_ms: int = Field(default=5000, ge=100, description="Poll timeout")
    auto_offset_reset: str = Field(default="latest", description="Auto offset reset policy")
    
    # Performance settings
    enable_auto_commit: bool = Field(default=False, description="Enable auto commit")
    batch_size: int = Field(default=100, ge=1, description="Processing batch size")
    buffer_size: int = Field(default=10000, ge=100, description="Internal buffer size")
    
    # Security
    security_protocol: str = Field(default="PLAINTEXT", description="Security protocol")
    auth_config: Dict[str, str] = Field(default_factory=dict, description="Authentication config")
    
    # Monitoring
    metrics_enabled: bool = Field(default=True, description="Enable metrics collection")
    health_check_interval_seconds: int = Field(default=30, description="Health check interval")
    
    # Metadata
    tags: Dict[str, str] = Field(default_factory=dict)
    created_at: datetime = Field(default_factory=datetime.utcnow)
    updated_at: datetime = Field(default_factory=datetime.utcnow)
    
    class Config:
        """Pydantic configuration."""
        use_enum_values = True
        json_encoders = {
            datetime: lambda v: v.isoformat(),
            UUID: lambda v: str(v),
        }
    
    def validate_connection(self) -> bool:
        """Validate source connection configuration."""
        # Basic validation - could be extended with actual connection testing
        return bool(self.connection_string and self.consumer_group)
    
    def get_consumer_config(self) -> Dict[str, Any]:
        """Get consumer configuration dictionary."""
        return {
            "bootstrap_servers": self.connection_string,
            "group_id": self.consumer_group,
            "auto_offset_reset": self.auto_offset_reset,
            "enable_auto_commit": self.enable_auto_commit,
            "max_poll_records": self.max_poll_records,
            "security_protocol": self.security_protocol,
            **self.auth_config
        }


class StreamSink(BaseModel):
    """
    Stream data sink configuration.
    
    Defines the destination for processed streaming data
    including batching, error handling, and delivery guarantees.
    """
    
    id: UUID = Field(default_factory=uuid4, description="Sink identifier")
    
    # Sink identification
    name: str = Field(..., description="Sink name")
    sink_type: str = Field(..., description="Type of sink")
    description: str = Field(default="", description="Sink description")
    
    # Connection configuration
    connection_string: str = Field(..., description="Connection string/URL")
    destination: str = Field(..., description="Destination topic/table/file")
    
    # Output configuration
    data_format: str = Field(default="json", description="Output data format")
    compression: Optional[str] = Field(None, description="Compression type")
    partition_by: Optional[str] = Field(None, description="Partitioning field")
    
    # Batching configuration
    batch_size: int = Field(default=1000, ge=1, description="Batch size")
    batch_timeout_ms: int = Field(default=10000, ge=100, description="Batch timeout")
    max_batch_bytes: int = Field(default=1048576, ge=1024, description="Max batch size in bytes")
    
    # Delivery settings
    delivery_guarantee: ProcessingMode = Field(default=ProcessingMode.AT_LEAST_ONCE)
    retry_attempts: int = Field(default=3, ge=0, description="Retry attempts")
    retry_backoff_ms: int = Field(default=1000, ge=100, description="Retry backoff")
    
    # Error handling
    error_handling: str = Field(default="log", description="Error handling strategy")
    dead_letter_topic: Optional[str] = Field(None, description="Dead letter queue/topic")
    
    # Performance settings
    parallelism: int = Field(default=1, ge=1, description="Parallel writers")
    buffer_size: int = Field(default=10000, ge=100, description="Output buffer size")
    
    # Security
    auth_config: Dict[str, str] = Field(default_factory=dict, description="Authentication config")
    
    # Metadata
    tags: Dict[str, str] = Field(default_factory=dict)
    created_at: datetime = Field(default_factory=datetime.utcnow)
    updated_at: datetime = Field(default_factory=datetime.utcnow)
    
    class Config:
        """Pydantic configuration."""
        use_enum_values = True
        json_encoders = {
            datetime: lambda v: v.isoformat(),
            UUID: lambda v: str(v),
        }
    
    def get_producer_config(self) -> Dict[str, Any]:
        """Get producer configuration dictionary."""
        return {
            "bootstrap_servers": self.connection_string,
            "batch_size": self.batch_size,
            "linger_ms": self.batch_timeout_ms,
            "retries": self.retry_attempts,
            "retry_backoff_ms": self.retry_backoff_ms,
            **self.auth_config
        }


class ProcessingWindow(BaseModel):
    """
    Stream processing window configuration.
    
    Defines time or count-based windows for stream aggregation
    and processing with configurable triggers and watermarks.
    """
    
    id: UUID = Field(default_factory=uuid4, description="Window identifier")
    
    # Window configuration
    window_type: WindowType = Field(..., description="Type of window")
    size_seconds: Optional[int] = Field(None, ge=1, description="Window size in seconds")
    slide_seconds: Optional[int] = Field(None, ge=1, description="Slide interval in seconds")
    session_timeout_seconds: Optional[int] = Field(None, ge=1, description="Session timeout")
    
    # Count-based windowing
    size_count: Optional[int] = Field(None, ge=1, description="Window size in record count")
    slide_count: Optional[int] = Field(None, ge=1, description="Slide interval in count")
    
    # Watermark configuration
    watermark_delay_seconds: int = Field(default=10, ge=0, description="Watermark delay")
    late_data_handling: str = Field(default="drop", description="Late data handling strategy")
    
    # Trigger configuration
    trigger_type: str = Field(default="processing_time", description="Trigger type")
    trigger_interval_seconds: int = Field(default=60, ge=1, description="Trigger interval")
    early_firing: bool = Field(default=False, description="Enable early firing")
    
    # Aggregation functions
    aggregation_functions: List[str] = Field(default_factory=list, description="Aggregation functions")
    group_by_fields: List[str] = Field(default_factory=list, description="Grouping fields")
    
    class Config:
        """Pydantic configuration."""
        use_enum_values = True
        json_encoders = {
            datetime: lambda v: v.isoformat(),
            UUID: lambda v: str(v),
        }
    
    @validator('slide_seconds')
    def slide_not_greater_than_size(cls, v, values):
        """Ensure slide <= size for time windows."""
        if v and 'size_seconds' in values and values['size_seconds']:
            if v > values['size_seconds']:
                raise ValueError('Slide interval cannot be greater than window size')
        return v
    
    def is_time_based(self) -> bool:
        """Check if this is a time-based window."""
        return self.size_seconds is not None
    
    def is_count_based(self) -> bool:
        """Check if this is a count-based window."""
        return self.size_count is not None


class StreamProcessor(BaseModel):
    """
    Stream processor for real-time data processing.
    
    Manages stream processing pipelines with configurable sources,
    sinks, processing logic, and scaling parameters.
    """
    
    id: UUID = Field(default_factory=uuid4, description="Processor identifier")
    
    # Processor identification
    name: str = Field(..., description="Processor name")
    description: str = Field(default="", description="Processor description")
    version: str = Field(default="1.0.0", description="Processor version")
    
    # Ownership and access
    tenant_id: UUID = Field(..., description="Owning tenant")
    created_by: UUID = Field(..., description="User who created processor")
    
    # Pipeline configuration
    sources: List[UUID] = Field(..., description="Source stream IDs")
    sinks: List[UUID] = Field(..., description="Sink stream IDs")
    processing_logic: str = Field(..., description="Processing logic/code")
    processing_language: str = Field(default="python", description="Processing language")
    
    # Processing configuration
    processing_mode: ProcessingMode = Field(default=ProcessingMode.AT_LEAST_ONCE)
    parallelism: int = Field(default=1, ge=1, description="Processing parallelism")
    max_parallelism: int = Field(default=10, ge=1, description="Maximum parallelism")
    checkpoint_interval_ms: int = Field(default=60000, description="Checkpoint interval")
    
    # Windowing configuration
    windowing_enabled: bool = Field(default=False, description="Enable windowing")
    windows: List[UUID] = Field(default_factory=list, description="Processing windows")
    
    # Resource configuration
    cpu_request: float = Field(default=0.5, ge=0.1, description="CPU request")
    memory_request_gb: float = Field(default=1.0, ge=0.1, description="Memory request GB")
    cpu_limit: float = Field(default=2.0, ge=0.1, description="CPU limit")
    memory_limit_gb: float = Field(default=4.0, ge=0.1, description="Memory limit GB")
    
    # Scaling configuration
    auto_scaling_enabled: bool = Field(default=False, description="Enable auto scaling")
    scale_up_threshold: float = Field(default=80.0, ge=0.0, le=100.0)
    scale_down_threshold: float = Field(default=20.0, ge=0.0, le=100.0)
    scale_up_cooldown_seconds: int = Field(default=300, ge=60)
    scale_down_cooldown_seconds: int = Field(default=600, ge=60)
    
    # Current state
    status: ProcessorStatus = Field(default=ProcessorStatus.PENDING)
    current_parallelism: int = Field(default=1, ge=1)
    current_throughput_per_second: float = Field(default=0.0, ge=0.0)
    current_latency_ms: float = Field(default=0.0, ge=0.0)
    
    # Performance metrics
    records_processed: int = Field(default=0, ge=0)
    records_failed: int = Field(default=0, ge=0)
    bytes_processed: int = Field(default=0, ge=0)
    processing_errors: int = Field(default=0, ge=0)
    avg_processing_time_ms: float = Field(default=0.0, ge=0.0)
    
    # Health monitoring
    health_score: float = Field(default=100.0, ge=0.0, le=100.0)
    last_checkpoint: Optional[datetime] = Field(None, description="Last checkpoint time")
    last_scaling_action: Optional[datetime] = Field(None, description="Last scaling action")
    error_message: Optional[str] = Field(None, description="Current error message")
    
    # Configuration
    environment_variables: Dict[str, str] = Field(default_factory=dict)
    secrets: List[str] = Field(default_factory=list, description="Secret references")
    tags: Dict[str, str] = Field(default_factory=dict)
    
    # Lifecycle
    created_at: datetime = Field(default_factory=datetime.utcnow)
    updated_at: datetime = Field(default_factory=datetime.utcnow)
    started_at: Optional[datetime] = Field(None, description="Processor start time")
    stopped_at: Optional[datetime] = Field(None, description="Processor stop time")
    
    class Config:
        """Pydantic configuration."""
        use_enum_values = True
        json_encoders = {
            datetime: lambda v: v.isoformat(),
            UUID: lambda v: str(v),
        }
    
    @validator('max_parallelism')
    def max_parallelism_valid(cls, v, values):
        """Ensure max_parallelism >= parallelism."""
        if 'parallelism' in values and v < values['parallelism']:
            raise ValueError('max_parallelism must be >= parallelism')
        return v
    
    @validator('cpu_limit')
    def cpu_limit_valid(cls, v, values):
        """Ensure CPU limit >= request."""
        if 'cpu_request' in values and v < values['cpu_request']:
            raise ValueError('cpu_limit must be >= cpu_request')
        return v
    
    @validator('memory_limit_gb')
    def memory_limit_valid(cls, v, values):
        """Ensure memory limit >= request."""
        if 'memory_request_gb' in values and v < values['memory_request_gb']:
            raise ValueError('memory_limit_gb must be >= memory_request_gb')
        return v
    
    def is_running(self) -> bool:
        """Check if processor is running."""
        return self.status == ProcessorStatus.RUNNING
    
    def is_healthy(self) -> bool:
        """Check if processor is healthy."""
        if not self.is_running():
            return False
        return self.health_score >= 70.0
    
    def get_success_rate(self) -> float:
        """Calculate processing success rate."""
        total_records = self.records_processed + self.records_failed
        if total_records == 0:
            return 100.0
        return (self.records_processed / total_records) * 100.0
    
    def should_scale_up(self) -> bool:
        """Check if processor should scale up."""
        if not self.auto_scaling_enabled or self.current_parallelism >= self.max_parallelism:
            return False
        
        # Check resource utilization or backlog
        cpu_usage = self.get_current_cpu_usage()
        return cpu_usage > self.scale_up_threshold
    
    def should_scale_down(self) -> bool:
        """Check if processor should scale down."""
        if not self.auto_scaling_enabled or self.current_parallelism <= 1:
            return False
        
        cpu_usage = self.get_current_cpu_usage()
        return cpu_usage < self.scale_down_threshold
    
    def get_current_cpu_usage(self) -> float:
        """Get current CPU usage percentage (placeholder)."""
        # This would be implemented with actual metrics collection
        return 50.0
    
    def can_scale(self, action: str) -> bool:
        """Check if scaling action is allowed based on cooldown."""
        if not self.last_scaling_action:
            return True
        
        time_since_scaling = datetime.utcnow() - self.last_scaling_action
        cooldown_seconds = (
            self.scale_up_cooldown_seconds if action == "up" 
            else self.scale_down_cooldown_seconds
        )
        
        return time_since_scaling >= timedelta(seconds=cooldown_seconds)
    
    def start(self) -> None:
        """Start the stream processor."""
        self.status = ProcessorStatus.RUNNING
        self.started_at = datetime.utcnow()
        self.updated_at = datetime.utcnow()
        self.error_message = None
    
    def stop(self) -> None:
        """Stop the stream processor."""
        self.status = ProcessorStatus.STOPPED
        self.stopped_at = datetime.utcnow()
        self.updated_at = datetime.utcnow()
    
    def pause(self) -> None:
        """Pause the stream processor."""
        self.status = ProcessorStatus.PAUSED
        self.updated_at = datetime.utcnow()
    
    def set_error(self, error_message: str) -> None:
        """Set processor error state."""
        self.status = ProcessorStatus.ERROR
        self.error_message = error_message
        self.updated_at = datetime.utcnow()
    
    def update_metrics(
        self,
        throughput: Optional[float] = None,
        latency: Optional[float] = None,
        processed_count: Optional[int] = None,
        failed_count: Optional[int] = None
    ) -> None:
        """Update processor performance metrics."""
        if throughput is not None:
            self.current_throughput_per_second = throughput
        if latency is not None:
            self.current_latency_ms = latency
        if processed_count is not None:
            self.records_processed += processed_count
        if failed_count is not None:
            self.records_failed += failed_count
        
        self.updated_at = datetime.utcnow()
    
    def record_checkpoint(self) -> None:
        """Record successful checkpoint."""
        self.last_checkpoint = datetime.utcnow()
        self.updated_at = datetime.utcnow()
    
    def scale(self, new_parallelism: int) -> None:
        """Scale processor to new parallelism level."""
        if new_parallelism < 1 or new_parallelism > self.max_parallelism:
            raise ValueError(f"Invalid parallelism: {new_parallelism}")
        
        self.current_parallelism = new_parallelism
        self.last_scaling_action = datetime.utcnow()
        self.updated_at = datetime.utcnow()
    
    def get_processor_summary(self) -> Dict[str, Any]:
        """Get processor summary information."""
        uptime_hours = (
            (datetime.utcnow() - self.started_at).total_seconds() / 3600.0
            if self.started_at else 0.0
        )
        
        return {
            "id": str(self.id),
            "name": self.name,
            "status": self.status,
            "parallelism": {
                "current": self.current_parallelism,
                "max": self.max_parallelism,
                "auto_scaling": self.auto_scaling_enabled
            },
            "performance": {
                "throughput_per_second": self.current_throughput_per_second,
                "latency_ms": self.current_latency_ms,
                "success_rate": self.get_success_rate(),
                "records_processed": self.records_processed,
                "records_failed": self.records_failed
            },
            "resources": {
                "cpu_request": self.cpu_request,
                "memory_request_gb": self.memory_request_gb,
                "cpu_limit": self.cpu_limit,
                "memory_limit_gb": self.memory_limit_gb
            },
            "health_score": self.health_score,
            "uptime_hours": uptime_hours,
            "sources_count": len(self.sources),
            "sinks_count": len(self.sinks)
        }