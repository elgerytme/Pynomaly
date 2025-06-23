"""Data Transfer Objects for streaming operations."""

from __future__ import annotations

from typing import Optional, List, Dict, Any, Union
from pydantic import BaseModel, Field, ConfigDict
from datetime import datetime


class StreamRecordDTO(BaseModel):
    """DTO for stream record data."""
    model_config = ConfigDict(from_attributes=True)
    
    id: str = Field(..., description="Unique record identifier")
    timestamp: str = Field(..., description="Record timestamp (ISO format)")
    data: Dict[str, Any] = Field(..., description="Record data as key-value pairs")
    metadata: Optional[Dict[str, Any]] = Field(default=None, description="Additional metadata")


class StreamingResultDTO(BaseModel):
    """DTO for streaming anomaly detection result."""
    model_config = ConfigDict(from_attributes=True)
    
    record_id: str = Field(..., description="Original record identifier")
    timestamp: str = Field(..., description="Processing timestamp (ISO format)")
    anomaly_score: float = Field(..., description="Anomaly score (0.0 to 1.0)")
    is_anomaly: bool = Field(..., description="Whether record is classified as anomaly")
    confidence: float = Field(..., description="Confidence in the prediction")
    explanation: Optional[Dict[str, Any]] = Field(default=None, description="Explanation data")
    metadata: Optional[Dict[str, Any]] = Field(default=None, description="Processing metadata")


class WindowConfigurationDTO(BaseModel):
    """DTO for sliding window configuration."""
    model_config = ConfigDict(from_attributes=True)
    
    window_type: str = Field(..., description="Type of window (count_based, time_based, session_based, adaptive_size)")
    size: Union[int, float] = Field(..., description="Window size (count or time in seconds)")
    step: Union[int, float] = Field(..., description="Step size for sliding window")
    overlap: float = Field(default=0.0, ge=0.0, le=1.0, description="Overlap percentage (0.0 to 1.0)")
    min_size: Optional[int] = Field(default=None, description="Minimum window size")
    max_size: Optional[int] = Field(default=None, description="Maximum window size")


class StreamingConfigurationDTO(BaseModel):
    """DTO for streaming processing configuration."""
    model_config = ConfigDict(from_attributes=True)
    
    window_config: WindowConfigurationDTO = Field(..., description="Window configuration")
    mode: str = Field(default="real_time", description="Streaming mode (real_time, batch, micro_batch, adaptive)")
    buffer_size: int = Field(default=1000, ge=1, description="Buffer size for processing")
    anomaly_threshold: float = Field(default=0.5, ge=0.0, le=1.0, description="Anomaly detection threshold")
    enable_online_learning: bool = Field(default=False, description="Enable online model learning")
    enable_callbacks: bool = Field(default=True, description="Enable result callbacks")
    max_processing_time: float = Field(default=1.0, gt=0.0, description="Maximum processing time per record/batch")


class StartStreamingRequestDTO(BaseModel):
    """DTO for starting streaming session."""
    model_config = ConfigDict(from_attributes=True)
    
    detector_ids: List[str] = Field(..., description="List of detector IDs to use")
    feature_names: List[str] = Field(..., description="List of feature names expected in data")
    window_type: str = Field(default="count_based", description="Window type")
    window_size: int = Field(default=100, ge=1, description="Window size")
    streaming_mode: str = Field(default="real_time", description="Streaming processing mode")
    anomaly_threshold: float = Field(default=0.5, ge=0.0, le=1.0, description="Anomaly threshold")
    enable_online_learning: bool = Field(default=False, description="Enable online learning")
    output_stream: Optional[str] = Field(default=None, description="Output stream identifier")


class ProcessStreamRequestDTO(BaseModel):
    """DTO for processing streaming data."""
    model_config = ConfigDict(from_attributes=True)
    
    session_id: str = Field(..., description="Streaming session identifier")
    records: List[StreamRecordDTO] = Field(..., description="List of records to process")


class StreamingSessionInfoDTO(BaseModel):
    """DTO for streaming session information."""
    model_config = ConfigDict(from_attributes=True)
    
    session_id: str = Field(..., description="Session identifier")
    status: str = Field(..., description="Session status (active, stopped, error)")
    detector_count: int = Field(..., description="Number of detectors in session")
    feature_names: List[str] = Field(..., description="List of feature names")
    configuration: Dict[str, Any] = Field(..., description="Session configuration")
    statistics: Optional[Dict[str, Any]] = Field(default=None, description="Session statistics")
    created_at: Optional[str] = Field(default=None, description="Session creation timestamp")
    stopped_at: Optional[str] = Field(default=None, description="Session stop timestamp")


class StreamingResponseDTO(BaseModel):
    """DTO for streaming operation responses."""
    model_config = ConfigDict(from_attributes=True)
    
    success: bool = Field(..., description="Whether operation succeeded")
    session_id: Optional[str] = Field(default=None, description="Session identifier")
    session_info: Optional[StreamingSessionInfoDTO] = Field(default=None, description="Session information")
    results: Optional[List[StreamingResultDTO]] = Field(default=None, description="Processing results")
    statistics: Optional[Dict[str, Any]] = Field(default=None, description="Operation statistics")
    message: str = Field(..., description="Response message")
    error: Optional[str] = Field(default=None, description="Error message if failed")


class StreamingStatisticsDTO(BaseModel):
    """DTO for streaming statistics."""
    model_config = ConfigDict(from_attributes=True)
    
    processed_records: int = Field(..., description="Number of processed records")
    detected_anomalies: int = Field(..., description="Number of detected anomalies")
    processing_errors: int = Field(..., description="Number of processing errors")
    anomaly_rate: float = Field(..., description="Rate of anomalies detected")
    error_rate: float = Field(..., description="Rate of processing errors")
    last_processing_time: Optional[str] = Field(default=None, description="Last processing timestamp")
    session_info: Optional[Dict[str, Any]] = Field(default=None, description="Session metadata")


class KafkaConnectorConfigDTO(BaseModel):
    """DTO for Kafka connector configuration."""
    model_config = ConfigDict(from_attributes=True)
    
    bootstrap_servers: List[str] = Field(..., description="Kafka bootstrap servers")
    input_topic: str = Field(..., description="Input topic to consume from")
    output_topic: Optional[str] = Field(default=None, description="Output topic to publish to")
    consumer_group: str = Field(default="pynomaly-streaming", description="Consumer group ID")
    auto_offset_reset: str = Field(default="latest", description="Auto offset reset strategy")
    enable_auto_commit: bool = Field(default=True, description="Enable auto commit")


class RedisConnectorConfigDTO(BaseModel):
    """DTO for Redis connector configuration."""
    model_config = ConfigDict(from_attributes=True)
    
    redis_url: str = Field(default="redis://localhost:6379", description="Redis connection URL")
    input_stream: str = Field(default="anomaly_input", description="Input stream name")
    output_stream: Optional[str] = Field(default=None, description="Output stream name")
    consumer_group: str = Field(default="pynomaly-group", description="Consumer group name")
    consumer_name: str = Field(default="pynomaly-consumer", description="Consumer name")
    block_time: int = Field(default=1000, ge=0, description="Blocking time for reading (ms)")
    count: int = Field(default=10, ge=1, description="Max messages to read at once")


class CreateTestStreamRequestDTO(BaseModel):
    """DTO for creating test stream."""
    model_config = ConfigDict(from_attributes=True)
    
    session_id: str = Field(..., description="Session identifier")
    count: int = Field(default=100, ge=1, le=10000, description="Number of records to generate")
    delay: float = Field(default=0.1, ge=0.0, le=10.0, description="Delay between records (seconds)")
    anomaly_rate: float = Field(default=0.05, ge=0.0, le=1.0, description="Rate of anomalies to inject")


class StreamingHealthCheckDTO(BaseModel):
    """DTO for streaming service health check."""
    model_config = ConfigDict(from_attributes=True)
    
    service_healthy: bool = Field(..., description="Overall service health")
    active_sessions: int = Field(..., description="Number of active sessions")
    total_sessions: int = Field(..., description="Total number of sessions")
    connector_status: Optional[Dict[str, Any]] = Field(default=None, description="Connector health status")
    last_error: Optional[str] = Field(default=None, description="Last error encountered")
    uptime_seconds: Optional[float] = Field(default=None, description="Service uptime in seconds")


class StreamingMetricsDTO(BaseModel):
    """DTO for streaming performance metrics."""
    model_config = ConfigDict(from_attributes=True)
    
    total_records_processed: int = Field(..., description="Total records processed across all sessions")
    total_anomalies_detected: int = Field(..., description="Total anomalies detected")
    average_processing_time: float = Field(..., description="Average processing time per record (ms)")
    throughput_records_per_second: float = Field(..., description="Processing throughput (records/sec)")
    memory_usage_mb: Optional[float] = Field(default=None, description="Current memory usage (MB)")
    cpu_usage_percent: Optional[float] = Field(default=None, description="Current CPU usage (%)")
    error_rate_percent: float = Field(..., description="Error rate percentage")


class BatchProcessingResultDTO(BaseModel):
    """DTO for batch processing results."""
    model_config = ConfigDict(from_attributes=True)
    
    batch_id: str = Field(..., description="Batch identifier")
    session_id: str = Field(..., description="Session identifier")
    processed_count: int = Field(..., description="Number of records processed")
    anomaly_count: int = Field(..., description="Number of anomalies detected")
    error_count: int = Field(..., description="Number of processing errors")
    processing_time_ms: float = Field(..., description="Total processing time (ms)")
    results: List[StreamingResultDTO] = Field(..., description="Individual processing results")
    statistics: Optional[Dict[str, Any]] = Field(default=None, description="Batch statistics")


class StreamingConfigurationUpdateDTO(BaseModel):
    """DTO for updating streaming configuration."""
    model_config = ConfigDict(from_attributes=True)
    
    session_id: str = Field(..., description="Session to update")
    anomaly_threshold: Optional[float] = Field(default=None, ge=0.0, le=1.0, description="New anomaly threshold")
    enable_online_learning: Optional[bool] = Field(default=None, description="Enable/disable online learning")
    buffer_size: Optional[int] = Field(default=None, ge=1, description="New buffer size")
    max_processing_time: Optional[float] = Field(default=None, gt=0.0, description="New max processing time")


class StreamingCallbackConfigDTO(BaseModel):
    """DTO for streaming callback configuration."""
    model_config = ConfigDict(from_attributes=True)
    
    session_id: str = Field(..., description="Session identifier")
    callback_url: str = Field(..., description="URL to call for results")
    callback_method: str = Field(default="POST", description="HTTP method for callback")
    callback_headers: Optional[Dict[str, str]] = Field(default=None, description="Headers for callback request")
    anomaly_only: bool = Field(default=True, description="Only send callbacks for anomalies")
    batch_size: int = Field(default=1, ge=1, description="Batch size for callbacks")
    retry_attempts: int = Field(default=3, ge=0, description="Number of retry attempts for failed callbacks")