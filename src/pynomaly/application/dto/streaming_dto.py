"""Data Transfer Objects for streaming anomaly detection."""

from datetime import datetime
from typing import Any

from pydantic import BaseModel, Field, field_validator, ConfigDict


class StreamDataPointDTO(BaseModel):
    """DTO for individual stream data point."""
    
    model_config = ConfigDict(extra="forbid")
    timestamp: datetime | None = Field(default=None, description="Data point timestamp")
    features: dict[str, float] = Field(description="Feature values for the data point")
    metadata: dict[str, Any] | None = Field(
        default_factory=dict, description="Additional metadata"
    )
    anomaly_score: float | None = Field(
        default=None, description="Anomaly score if computed"
    )
    is_anomaly: bool | None = Field(
        default=None, description="Whether this point is an anomaly"
    )

    @field_validator("features")
    @classmethod
    def validate_features(cls, v):
        """Validate feature values."""
        if not v:
            raise ValueError("Features cannot be empty")
        if not all(isinstance(val, (int, float)) for val in v.values()):
            raise ValueError("All feature values must be numeric")
        return v

    @field_validator("anomaly_score")
    @classmethod
    def validate_anomaly_score(cls, v):
        """Validate anomaly score range."""
        if v is not None and not (0.0 <= v <= 1.0):
            raise ValueError("Anomaly score must be between 0.0 and 1.0")
        return v

    def to_dict(self) -> dict:
        """Convert to dictionary with proper timestamp serialization."""
        return {
            "timestamp": self.timestamp.isoformat() if self.timestamp else None,
            "features": self.features,
            "metadata": self.metadata,
            "anomaly_score": self.anomaly_score,
            "is_anomaly": self.is_anomaly,
        }

    @classmethod
    def from_dict(cls, data: dict) -> "StreamDataPointDTO":
        """Create from dictionary with proper timestamp deserialization."""
        # Handle timestamp deserialization
        timestamp = None
        if data.get("timestamp"):
            from datetime import datetime

            timestamp = datetime.fromisoformat(data["timestamp"].replace("Z", "+00:00"))

        return cls(
            timestamp=timestamp,
            features=data["features"],
            metadata=data.get("metadata", {}),
            anomaly_score=data.get("anomaly_score"),
            is_anomaly=data.get("is_anomaly"),
        )


class StreamDataBatchDTO(BaseModel):
    """DTO for batch of stream data points."""
    
    model_config = ConfigDict(extra="forbid")
    batch_id: str | None = Field(default=None, description="Batch identifier")
    data_points: list[StreamDataPointDTO] = Field(
        description="List of data points in the batch"
    )
    timestamp: datetime | None = Field(default=None, description="Batch timestamp")
    window_start: datetime | None = Field(default=None, description="Window start time")
    window_end: datetime | None = Field(default=None, description="Window end time")

    @field_validator("data_points")
    @classmethod
    def validate_data_points(cls, v):
        """Validate data points."""
        if not v:
            raise ValueError("Batch cannot be empty")
        return v

    @property
    def batch_size(self) -> int:
        """Get the number of data points in the batch."""
        return len(self.data_points)

    def to_pandas(self):
        """Convert to pandas DataFrame."""
        try:
            import pandas as pd
        except ImportError:
            raise ImportError("pandas is required for to_pandas() method")

        if not self.data_points:
            return pd.DataFrame()

        # Convert data points to records
        records = []
        for point in self.data_points:
            record = {
                "timestamp": point.timestamp,
                "anomaly_score": point.anomaly_score,
                "is_anomaly": point.is_anomaly,
            }
            # Add feature columns
            if point.features:
                record.update(point.features)
            # Add metadata columns with prefix
            if point.metadata:
                for k, v in point.metadata.items():
                    record[f"meta_{k}"] = v
            records.append(record)

        return pd.DataFrame(records)


class StreamDetectionRequestDTO(BaseModel):
    """DTO for stream detection request."""
    
    model_config = ConfigDict(extra="forbid")
    detector_id: str = Field(description="Detector identifier")
    data_batch: StreamDataBatchDTO = Field(description="Data batch to process")
    configuration: dict[str, Any] | None = Field(
        default_factory=dict, description="Additional configuration"
    )


class StreamDetectionResponseDTO(BaseModel):
    """DTO for stream detection response."""
    
    model_config = ConfigDict(extra="forbid")
    request_id: str = Field(description="Request identifier")
    predictions: list[int] = Field(description="Anomaly predictions")
    scores: list[float] = Field(description="Anomaly scores")
    processing_time: float = Field(description="Processing time in seconds")
    timestamp: datetime = Field(description="Response timestamp")


class StreamConfigurationDTO(BaseModel):
    """DTO for stream configuration."""
    
    model_config = ConfigDict(extra="forbid")
    buffer_size: int = Field(default=1000, description="Buffer size")
    batch_size: int = Field(default=100, description="Batch size")
    timeout_ms: int = Field(default=1000, description="Timeout in milliseconds")


class StreamMetricsDTO(BaseModel):
    """DTO for stream metrics."""
    
    model_config = ConfigDict(extra="forbid")
    stream_id: str = Field(description="Stream identifier")
    throughput: float = Field(description="Throughput in samples/second")
    latency_ms: float = Field(description="Average latency in milliseconds")
    error_rate: float = Field(description="Error rate percentage")
    buffer_utilization: float = Field(description="Buffer utilization percentage")


class StreamStatusDTO(BaseModel):
    """DTO for stream status."""
    
    model_config = ConfigDict(extra="forbid")
    stream_id: str = Field(description="Stream identifier")
    status: str = Field(description="Current status")
    last_updated: datetime = Field(description="Last update timestamp")
    message: str | None = Field(default=None, description="Status message")


class StreamErrorDTO(BaseModel):
    """DTO for stream errors."""
    
    model_config = ConfigDict(extra="forbid")
    stream_id: str = Field(description="Stream identifier")
    error_code: str = Field(description="Error code")
    error_message: str = Field(description="Error message")
    timestamp: datetime = Field(description="Error timestamp")
    severity: str = Field(description="Error severity level")


class BackpressureConfigDTO(BaseModel):
    """DTO for backpressure configuration."""
    
    model_config = ConfigDict(extra="forbid")
    enabled: bool = Field(default=True, description="Enable backpressure")
    high_watermark: float = Field(default=0.8, description="High watermark threshold")
    low_watermark: float = Field(default=0.3, description="Low watermark threshold")
    strategy: str = Field(default="drop_oldest", description="Backpressure strategy")


class WindowConfigDTO(BaseModel):
    """DTO for window configuration."""
    
    model_config = ConfigDict(extra="forbid")
    window_type: str = Field(description="Window type (sliding, tumbling, session)")
    size_ms: int = Field(description="Window size in milliseconds")
    advance_ms: int | None = Field(
        default=None, description="Window advance in milliseconds"
    )
    allowed_lateness_ms: int = Field(
        default=0, description="Allowed lateness in milliseconds"
    )


class CheckpointConfigDTO(BaseModel):
    """DTO for checkpoint configuration."""
    
    model_config = ConfigDict(extra="forbid")
    enabled: bool = Field(default=True, description="Enable checkpointing")
    interval_ms: int = Field(
        default=10000, description="Checkpoint interval in milliseconds"
    )
    storage_path: str = Field(description="Checkpoint storage path")
    retention_count: int = Field(
        default=5, description="Number of checkpoints to retain"
    )


class StreamingConfigurationDTO(BaseModel):
    """DTO for streaming detection configuration."""
    
    model_config = ConfigDict(extra="forbid")
    strategy: str = Field(
        default="adaptive_batch",
        description="Streaming strategy (real_time, micro_batch, adaptive_batch, windowed, ensemble_stream)",
    )
    backpressure_strategy: str = Field(
        default="adaptive_sampling", description="Backpressure handling strategy"
    )
    mode: str = Field(
        default="continuous",
        description="Streaming mode (continuous, burst, scheduled, event_driven)",
    )

    # Buffer configuration
    max_buffer_size: int = Field(
        default=10000,
        ge=100,
        le=100000,
        description="Maximum buffer size for incoming samples",
    )
    min_batch_size: int = Field(
        default=1, ge=1, description="Minimum batch size for processing"
    )
    max_batch_size: int = Field(
        default=100, ge=1, description="Maximum batch size for processing"
    )
    batch_timeout_ms: int = Field(
        default=100, ge=1, le=10000, description="Batch timeout in milliseconds"
    )

    # Backpressure thresholds
    high_watermark: float = Field(
        default=0.8,
        ge=0.1,
        le=1.0,
        description="High watermark for backpressure activation",
    )
    low_watermark: float = Field(
        default=0.3,
        ge=0.0,
        le=0.9,
        description="Low watermark for backpressure deactivation",
    )

    # Performance settings
    max_processing_time_ms: int = Field(
        default=1000,
        ge=10,
        le=60000,
        description="Maximum processing time per batch in milliseconds",
    )
    max_concurrent_batches: int = Field(
        default=5, ge=1, le=50, description="Maximum concurrent batch processing"
    )
    adaptive_scaling_enabled: bool = Field(
        default=True, description="Enable adaptive scaling of resources"
    )

    # Quality settings
    enable_quality_monitoring: bool = Field(
        default=True, description="Enable quality monitoring"
    )
    quality_check_interval_ms: int = Field(
        default=5000,
        ge=1000,
        le=60000,
        description="Quality check interval in milliseconds",
    )
    drift_detection_enabled: bool = Field(
        default=True, description="Enable data drift detection"
    )

    # Output configuration
    enable_result_buffering: bool = Field(
        default=True, description="Enable result buffering"
    )
    result_buffer_size: int = Field(
        default=1000, ge=10, le=10000, description="Size of result buffer"
    )
    enable_metrics_collection: bool = Field(
        default=True, description="Enable metrics collection"
    )

    @field_validator("strategy")
    @classmethod
    def validate_strategy(cls, v):
        """Validate streaming strategy."""
        valid_strategies = {
            "real_time",
            "micro_batch",
            "adaptive_batch",
            "windowed",
            "ensemble_stream",
        }
        if v not in valid_strategies:
            raise ValueError(f"Invalid strategy. Must be one of: {valid_strategies}")
        return v

    @field_validator("backpressure_strategy")
    @classmethod
    def validate_backpressure_strategy(cls, v):
        """Validate backpressure strategy."""
        valid_strategies = {
            "drop_oldest",
            "drop_newest",
            "adaptive_sampling",
            "circuit_breaker",
            "elastic_scaling",
        }
        if v not in valid_strategies:
            raise ValueError(
                f"Invalid backpressure strategy. Must be one of: {valid_strategies}"
            )
        return v

    @field_validator("mode")
    @classmethod
    def validate_mode(cls, v):
        """Validate streaming mode."""
        valid_modes = {"continuous", "burst", "scheduled", "event_driven"}
        if v not in valid_modes:
            raise ValueError(f"Invalid mode. Must be one of: {valid_modes}")
        return v

    @field_validator("max_batch_size")
    @classmethod
    def validate_batch_sizes(cls, v, info):
        """Validate batch size consistency."""
        if (
            hasattr(info, "data")
            and "min_batch_size" in info.data
            and v < info.data["min_batch_size"]
        ):
            raise ValueError("max_batch_size must be >= min_batch_size")
        return v

    @field_validator("high_watermark")
    @classmethod
    def validate_watermarks(cls, v, info):
        """Validate watermark consistency."""
        if (
            hasattr(info, "data")
            and "low_watermark" in info.data
            and v <= info.data["low_watermark"]
        ):
            raise ValueError("high_watermark must be > low_watermark")
        return v


class StreamingSampleDTO(BaseModel):
    """DTO for streaming sample data."""
    
    model_config = ConfigDict(extra="forbid")
    id: str | None = Field(
        default=None, description="Sample ID (auto-generated if not provided)"
    )
    data: list[float] | dict[str, Any] = Field(
        description="Sample data as array or key-value pairs"
    )
    timestamp: float | None = Field(
        default=None, description="Sample timestamp (auto-generated if not provided)"
    )
    metadata: dict[str, Any] = Field(
        default_factory=dict, description="Additional sample metadata"
    )
    priority: int = Field(
        default=0, ge=0, le=10, description="Sample priority (0=lowest, 10=highest)"
    )

    @field_validator("data")
    @classmethod
    def validate_data(cls, v):
        """Validate sample data format."""
        if isinstance(v, list):
            if not v:
                raise ValueError("Data array cannot be empty")
            if not all(isinstance(x, (int, float)) for x in v):
                raise ValueError("All data array elements must be numeric")
        elif isinstance(v, dict):
            if not v:
                raise ValueError("Data dictionary cannot be empty")
            if not all(isinstance(val, (int, float)) for val in v.values()):
                raise ValueError("All data dictionary values must be numeric")
        else:
            raise ValueError(
                "Data must be a list of numbers or dictionary of numeric values"
            )
        return v


class StreamingRequestDTO(BaseModel):
    """DTO for streaming detection request."""
    
    model_config = ConfigDict(extra="forbid")
    detector_id: str = Field(description="ID of detector to use for streaming")
    configuration: StreamingConfigurationDTO = Field(
        description="Streaming configuration"
    )
    enable_ensemble: bool = Field(
        default=False, description="Enable ensemble streaming with multiple detectors"
    )
    ensemble_detector_ids: list[str] = Field(
        default_factory=list, description="List of detector IDs for ensemble streaming"
    )
    callback_settings: dict[str, Any] = Field(
        default_factory=dict, description="Callback handler settings"
    )

    @field_validator("ensemble_detector_ids")
    @classmethod
    def validate_ensemble_detectors(cls, v, info):
        """Validate ensemble detector configuration."""
        if hasattr(info, "data") and info.data.get("enable_ensemble", False) and not v:
            raise ValueError(
                "ensemble_detector_ids required when enable_ensemble is True"
            )
        return v


class StreamingResponseDTO(BaseModel):
    """DTO for streaming detection response."""
    
    model_config = ConfigDict(extra="forbid")
    success: bool = Field(description="Whether streaming setup was successful")
    stream_id: str = Field(
        default="", description="Unique identifier for the streaming session"
    )
    configuration: StreamingConfigurationDTO | None = Field(
        default=None, description="Applied streaming configuration"
    )
    performance_metrics: dict[str, Any] = Field(
        default_factory=dict, description="Initial performance metrics"
    )
    error_message: str | None = Field(
        default=None, description="Error message if setup failed"
    )
    estimated_throughput: float | None = Field(
        default=None, description="Estimated throughput (samples/second)"
    )
    resource_allocation: dict[str, Any] = Field(
        default_factory=dict, description="Allocated system resources"
    )


class StreamingResultDTO(BaseModel):
    """DTO for streaming detection result."""
    
    model_config = ConfigDict(extra="forbid")
    sample_id: str = Field(description="ID of processed sample")
    prediction: int = Field(
        ge=0, le=1, description="Anomaly prediction (0=normal, 1=anomaly)"
    )
    anomaly_score: float = Field(
        ge=0.0, le=1.0, description="Anomaly score between 0.0 and 1.0"
    )
    confidence: float = Field(
        ge=0.0, le=1.0, description="Prediction confidence between 0.0 and 1.0"
    )
    processing_time: float = Field(ge=0.0, description="Processing time in seconds")
    detector_id: str = Field(description="ID of detector that made the prediction")
    timestamp: float = Field(description="Result timestamp")
    metadata: dict[str, Any] = Field(
        default_factory=dict, description="Additional result metadata"
    )
    quality_indicators: dict[str, float] = Field(
        default_factory=dict, description="Quality indicators for the prediction"
    )


class StreamingMetricsDTO(BaseModel):
    """DTO for streaming performance metrics."""
    
    model_config = ConfigDict(extra="forbid")
    stream_id: str = Field(description="Stream identifier")
    samples_processed: int = Field(ge=0, description="Total samples processed")
    samples_dropped: int = Field(
        ge=0, description="Total samples dropped due to backpressure"
    )
    anomalies_detected: int = Field(ge=0, description="Total anomalies detected")
    average_processing_time: float = Field(
        ge=0.0, description="Average processing time per sample (seconds)"
    )
    current_buffer_size: int = Field(ge=0, description="Current buffer size")
    buffer_utilization: float = Field(
        ge=0.0, le=1.0, description="Buffer utilization percentage"
    )
    throughput_per_second: float = Field(
        ge=0.0, description="Current throughput (samples/second)"
    )
    backpressure_active: bool = Field(
        description="Whether backpressure is currently active"
    )
    circuit_breaker_open: bool = Field(
        description="Whether circuit breaker is currently open"
    )
    error_rate: float = Field(ge=0.0, le=1.0, description="Current error rate")
    quality_score: float = Field(ge=0.0, le=1.0, description="Overall quality score")
    last_updated: float = Field(description="Last metrics update timestamp")

    # Advanced metrics
    latency_percentiles: dict[str, float] = Field(
        default_factory=dict, description="Latency percentiles (p50, p95, p99)"
    )
    detector_performance: dict[str, float] = Field(
        default_factory=dict, description="Individual detector performance metrics"
    )
    resource_utilization: dict[str, float] = Field(
        default_factory=dict, description="System resource utilization"
    )


class StreamingStatusDTO(BaseModel):
    """DTO for streaming system status."""
    
    model_config = ConfigDict(extra="forbid")
    active_streams: list[str] = Field(description="List of active stream IDs")
    total_streams_created: int = Field(
        ge=0, description="Total number of streams created"
    )
    system_capacity: dict[str, Any] = Field(
        description="Current system capacity and limits"
    )
    performance_summary: dict[str, float] = Field(
        description="Aggregate performance metrics across all streams"
    )
    health_status: str = Field(description="Overall system health status")
    available_strategies: list[dict[str, str]] = Field(
        description="Available streaming strategies with descriptions"
    )
    resource_usage: dict[str, float] = Field(
        description="Current resource usage statistics"
    )


class StreamingBatchResultDTO(BaseModel):
    """DTO for batch streaming results."""
    
    model_config = ConfigDict(extra="forbid")
    stream_id: str = Field(description="Stream identifier")
    batch_id: str = Field(description="Batch identifier")
    results: list[StreamingResultDTO] = Field(
        description="List of individual results in the batch"
    )
    batch_processing_time: float = Field(
        ge=0.0, description="Total batch processing time"
    )
    batch_size: int = Field(ge=0, description="Number of samples in the batch")
    anomaly_count: int = Field(
        ge=0, description="Number of anomalies detected in the batch"
    )
    quality_metrics: dict[str, float] = Field(
        default_factory=dict, description="Batch quality metrics"
    )
    timestamp: float = Field(description="Batch completion timestamp")


class StreamingControlDTO(BaseModel):
    """DTO for streaming control operations."""
    
    model_config = ConfigDict(extra="forbid")
    action: str = Field(
        description="Control action (start, stop, pause, resume, configure)"
    )
    stream_id: str | None = Field(
        default=None, description="Target stream ID (required for stop, pause, resume)"
    )
    configuration_updates: dict[str, Any] | None = Field(
        default=None, description="Configuration updates for configure action"
    )
    force: bool = Field(
        default=False, description="Force the action even if it may cause data loss"
    )

    @field_validator("action")
    @classmethod
    def validate_action(cls, v):
        """Validate control action."""
        valid_actions = {"start", "stop", "pause", "resume", "configure"}
        if v not in valid_actions:
            raise ValueError(f"Invalid action. Must be one of: {valid_actions}")
        return v

    @field_validator("stream_id")
    @classmethod
    def validate_stream_id_requirement(cls, v, info):
        """Validate stream_id requirement for certain actions."""
        action = info.data.get("action") if hasattr(info, "data") else None
        if action in {"stop", "pause", "resume", "configure"} and not v:
            raise ValueError(f"stream_id is required for action '{action}'")
        return v


class StreamingHealthCheckDTO(BaseModel):
    """DTO for streaming health check response."""
    
    model_config = ConfigDict(extra="forbid")
    status: str = Field(description="Health status (healthy, degraded, unhealthy)")
    timestamp: datetime = Field(
        default_factory=datetime.now, description="Health check timestamp"
    )
    active_streams_count: int = Field(ge=0, description="Number of active streams")
    total_throughput: float = Field(ge=0.0, description="Total system throughput")
    error_rates: dict[str, float] = Field(description="Error rates by category")
    resource_health: dict[str, str] = Field(
        description="Health status of system resources"
    )
    performance_indicators: dict[str, float] = Field(
        description="Key performance indicators"
    )
    recommendations: list[str] = Field(
        default_factory=list, description="Health improvement recommendations"
    )
    alerts: list[dict[str, Any]] = Field(
        default_factory=list, description="Active system alerts"
    )
