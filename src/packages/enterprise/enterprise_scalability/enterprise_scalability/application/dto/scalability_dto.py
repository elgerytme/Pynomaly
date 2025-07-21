"""
Scalability Data Transfer Objects (DTOs) for API communication.
"""

from datetime import datetime
from typing import Dict, List, Optional, Any, Union
from uuid import UUID

from pydantic import BaseModel, Field, validator


# Compute Cluster DTOs

class ClusterCreateRequest(BaseModel):
    """Request to create a compute cluster."""
    
    name: str = Field(..., description="Cluster name")
    cluster_type: str = Field(..., description="Type of cluster")
    description: Optional[str] = Field(None, description="Cluster description")
    
    # Scaling configuration
    min_nodes: int = Field(default=1, ge=1, description="Minimum nodes")
    max_nodes: int = Field(default=10, ge=1, description="Maximum nodes")
    auto_scaling: bool = Field(default=True, description="Enable auto-scaling")
    
    # Node configuration
    node_type: str = Field(default="standard", description="Node type/size")
    cpu_cores: int = Field(default=4, ge=1, description="CPU cores per node")
    memory_gb: float = Field(default=16.0, ge=0.1, description="Memory per node")
    gpu_count: int = Field(default=0, ge=0, description="GPUs per node")
    
    # Advanced configuration
    configuration: Dict[str, Any] = Field(default_factory=dict, description="Advanced config")
    
    @validator('max_nodes')
    def max_nodes_valid(cls, v, values):
        """Ensure max_nodes >= min_nodes."""
        if 'min_nodes' in values and v < values['min_nodes']:
            raise ValueError('max_nodes must be >= min_nodes')
        return v


class ClusterScaleRequest(BaseModel):
    """Request to scale a cluster."""
    
    target_nodes: int = Field(..., ge=1, description="Target number of nodes")
    reason: str = Field(default="manual", description="Scaling reason")


class ClusterResponse(BaseModel):
    """Response with cluster information."""
    
    id: UUID
    name: str
    cluster_type: str
    status: str
    
    # Node counts
    current_nodes: int
    active_nodes: int
    min_nodes: int
    max_nodes: int
    
    # Resource totals
    total_cpu_cores: int
    total_memory_gb: float
    used_cpu_cores: float
    used_memory_gb: float
    
    # Configuration
    auto_scaling: bool
    health_score: float
    
    # Timestamps
    created_at: datetime
    started_at: Optional[datetime]
    
    class Config:
        """Pydantic configuration."""
        json_encoders = {
            datetime: lambda v: v.isoformat(),
            UUID: lambda v: str(v),
        }


class NodeResponse(BaseModel):
    """Response with compute node information."""
    
    id: UUID
    name: str
    hostname: str
    ip_address: str
    status: str
    
    # Resources
    cpu_cores: int
    memory_gb: float
    gpu_count: int
    
    # Utilization
    cpu_usage_percent: float
    memory_usage_gb: float
    gpu_usage_percent: float
    
    # Performance
    tasks_running: int
    tasks_completed: int
    health_score: float
    uptime_seconds: float
    
    class Config:
        """Pydantic configuration."""
        json_encoders = {
            datetime: lambda v: v.isoformat(),
            UUID: lambda v: str(v),
        }


class ClusterMetricsResponse(BaseModel):
    """Response with cluster performance metrics."""
    
    cluster_id: UUID
    cluster_name: str
    status: str
    
    # Node metrics
    nodes: Dict[str, Any] = Field(default_factory=dict)
    
    # Resource metrics
    resources: Dict[str, Any] = Field(default_factory=dict)
    
    # Performance metrics
    performance: Dict[str, Any] = Field(default_factory=dict)
    
    # Health metrics
    health_score: float
    uptime_hours: float
    
    # Timestamp
    collected_at: datetime = Field(default_factory=datetime.utcnow)
    
    class Config:
        """Pydantic configuration."""
        json_encoders = {
            datetime: lambda v: v.isoformat(),
            UUID: lambda v: str(v),
        }


# Stream Processing DTOs

class StreamSourceRequest(BaseModel):
    """Request to create a stream source."""
    
    name: str = Field(..., description="Source name")
    stream_type: str = Field(..., description="Stream type")
    connection_string: str = Field(..., description="Connection string")
    
    # Stream configuration
    topics: List[str] = Field(default_factory=list, description="Topics to consume")
    consumer_group: str = Field(..., description="Consumer group ID")
    data_format: str = Field(default="json", description="Data format")
    
    # Consumer settings
    batch_size: int = Field(default=100, ge=1, description="Batch size")
    poll_timeout_ms: int = Field(default=5000, ge=100, description="Poll timeout")
    
    # Security
    auth_config: Dict[str, str] = Field(default_factory=dict, description="Auth configuration")


class StreamSinkRequest(BaseModel):
    """Request to create a stream sink."""
    
    name: str = Field(..., description="Sink name")
    sink_type: str = Field(..., description="Sink type")
    connection_string: str = Field(..., description="Connection string")
    destination: str = Field(..., description="Destination topic/table")
    
    # Output configuration
    data_format: str = Field(default="json", description="Output format")
    batch_size: int = Field(default=1000, ge=1, description="Output batch size")
    
    # Delivery settings
    delivery_guarantee: str = Field(default="at_least_once", description="Delivery guarantee")
    retry_attempts: int = Field(default=3, ge=0, description="Retry attempts")


class StreamProcessorRequest(BaseModel):
    """Request to create a stream processor."""
    
    name: str = Field(..., description="Processor name")
    description: Optional[str] = Field(None, description="Processor description")
    
    # Sources and sinks
    sources: List[Dict[str, Any]] = Field(..., description="Stream sources")
    sinks: List[Dict[str, Any]] = Field(..., description="Stream sinks")
    
    # Processing logic
    processing_logic: str = Field(..., description="Processing logic/code")
    processing_language: str = Field(default="python", description="Processing language")
    
    # Configuration
    parallelism: int = Field(default=1, ge=1, description="Processing parallelism")
    max_parallelism: int = Field(default=10, ge=1, description="Maximum parallelism")
    processing_mode: str = Field(default="at_least_once", description="Processing mode")
    
    # Resources
    cpu_request: float = Field(default=0.5, ge=0.1, description="CPU request")
    memory_request_gb: float = Field(default=1.0, ge=0.1, description="Memory request")
    cpu_limit: float = Field(default=2.0, ge=0.1, description="CPU limit")
    memory_limit_gb: float = Field(default=4.0, ge=0.1, description="Memory limit")
    
    # Scaling
    auto_scaling: bool = Field(default=False, description="Enable auto-scaling")
    
    # Environment
    environment_variables: Dict[str, str] = Field(default_factory=dict)


class StreamProcessorResponse(BaseModel):
    """Response with stream processor information."""
    
    id: UUID
    name: str
    status: str
    
    # Configuration
    parallelism: Dict[str, Any] = Field(default_factory=dict)
    processing_mode: str
    
    # Performance
    performance: Dict[str, Any] = Field(default_factory=dict)
    
    # Resources
    resources: Dict[str, Any] = Field(default_factory=dict)
    
    # Health
    health_score: float
    uptime_hours: float
    
    # Counts
    sources_count: int
    sinks_count: int
    
    # Timestamps
    created_at: datetime
    started_at: Optional[datetime]
    
    class Config:
        """Pydantic configuration."""
        json_encoders = {
            datetime: lambda v: v.isoformat(),
            UUID: lambda v: str(v),
        }


# Distributed Task DTOs

class TaskSubmitRequest(BaseModel):
    """Request to submit a distributed task."""
    
    # Task identification
    function_name: str = Field(..., description="Function to execute")
    module_name: str = Field(..., description="Module containing function")
    task_type: str = Field(default="custom", description="Type of task")
    
    # Function parameters
    args: List[Any] = Field(default_factory=list, description="Function arguments")
    kwargs: Dict[str, Any] = Field(default_factory=dict, description="Function kwargs")
    
    # Execution settings
    priority: str = Field(default="normal", description="Task priority")
    timeout_seconds: Optional[int] = Field(None, ge=1, description="Task timeout")
    max_retries: int = Field(default=3, ge=0, le=10, description="Maximum retries")
    
    # Resources
    cpu_cores: float = Field(default=1.0, ge=0.1, description="Required CPU cores")
    memory_gb: float = Field(default=1.0, ge=0.1, description="Required memory")
    gpu_count: int = Field(default=0, ge=0, description="Required GPUs")
    
    # Scheduling
    cluster_id: Optional[UUID] = Field(None, description="Specific cluster to use")
    node_selector: Dict[str, str] = Field(default_factory=dict, description="Node selector")
    
    # Dependencies
    depends_on: List[UUID] = Field(default_factory=list, description="Task dependencies")
    
    @validator('priority')
    def validate_priority(cls, v):
        """Validate task priority."""
        allowed = ["low", "normal", "high", "critical"]
        if v not in allowed:
            raise ValueError(f'Priority must be one of: {", ".join(allowed)}')
        return v


class TaskBatchSubmitRequest(BaseModel):
    """Request to submit a batch of tasks."""
    
    batch_name: str = Field(..., description="Batch name")
    tasks: List[Dict[str, Any]] = Field(..., description="List of tasks")
    
    # Batch configuration
    max_concurrent: int = Field(default=10, ge=1, description="Max concurrent tasks")
    stop_on_failure: bool = Field(default=False, description="Stop on first failure")
    batch_timeout_seconds: Optional[int] = Field(None, description="Batch timeout")


class TaskResponse(BaseModel):
    """Response with task information."""
    
    id: UUID
    name: str
    task_type: str
    status: str
    priority: str
    progress: float
    
    # Dependencies
    dependencies: int
    retries: int
    
    # Timing
    timing: Dict[str, Any] = Field(default_factory=dict)
    
    # Resources
    resources: Dict[str, Any] = Field(default_factory=dict)
    
    # Results
    result: Optional[Dict[str, Any]] = None
    
    class Config:
        """Pydantic configuration."""
        json_encoders = {
            datetime: lambda v: v.isoformat(),
            UUID: lambda v: str(v),
        }


class TaskBatchResponse(BaseModel):
    """Response with task batch information."""
    
    id: UUID
    name: str
    status: str
    progress: float
    
    # Task counts
    tasks: Dict[str, Any] = Field(default_factory=dict)
    
    # Timing
    timing: Dict[str, Any] = Field(default_factory=dict)
    
    # Configuration
    configuration: Dict[str, Any] = Field(default_factory=dict)
    
    class Config:
        """Pydantic configuration."""
        json_encoders = {
            datetime: lambda v: v.isoformat(),
            UUID: lambda v: str(v),
        }


class TaskResultResponse(BaseModel):
    """Response with task execution result."""
    
    task_id: UUID
    success: bool
    return_value: Optional[Any] = None
    error_message: Optional[str] = None
    
    # Execution metrics
    execution_time_seconds: float
    cpu_time_seconds: float
    memory_peak_gb: float
    
    # Metadata
    computed_at: datetime
    computed_by: Optional[str] = None
    
    class Config:
        """Pydantic configuration."""
        json_encoders = {
            datetime: lambda v: v.isoformat(),
            UUID: lambda v: str(v),
        }


# General DTOs

class ScalabilityOverviewResponse(BaseModel):
    """Response with tenant scalability overview."""
    
    tenant_id: UUID
    
    # Clusters summary
    clusters: Dict[str, Any] = Field(default_factory=dict)
    
    # Stream processors summary
    stream_processors: Dict[str, Any] = Field(default_factory=dict)
    
    # Tasks summary
    tasks: Dict[str, Any] = Field(default_factory=dict)
    
    # Generation timestamp
    generated_at: datetime = Field(default_factory=datetime.utcnow)
    
    class Config:
        """Pydantic configuration."""
        json_encoders = {
            datetime: lambda v: v.isoformat(),
            UUID: lambda v: str(v),
        }


class ResourceUtilizationResponse(BaseModel):
    """Response with resource utilization metrics."""
    
    # Resource totals
    total_cpu_cores: int
    total_memory_gb: float
    total_gpu_count: int
    total_storage_gb: float
    
    # Current usage
    used_cpu_cores: float
    used_memory_gb: float
    used_gpu_count: int
    used_storage_gb: float
    
    # Utilization percentages
    cpu_utilization_percent: float
    memory_utilization_percent: float
    gpu_utilization_percent: float
    storage_utilization_percent: float
    
    # Capacity planning
    recommended_scaling: Optional[Dict[str, Any]] = None
    
    # Timestamp
    measured_at: datetime = Field(default_factory=datetime.utcnow)
    
    class Config:
        """Pydantic configuration."""
        json_encoders = {
            datetime: lambda v: v.isoformat(),
        }


class PerformanceMetricsResponse(BaseModel):
    """Response with performance metrics."""
    
    # Throughput metrics
    tasks_per_second: float = Field(default=0.0)
    records_per_second: float = Field(default=0.0)
    bytes_per_second: float = Field(default=0.0)
    
    # Latency metrics
    avg_task_latency_ms: float = Field(default=0.0)
    p95_task_latency_ms: float = Field(default=0.0)
    avg_stream_latency_ms: float = Field(default=0.0)
    
    # Success rates
    task_success_rate_percent: float = Field(default=100.0)
    stream_success_rate_percent: float = Field(default=100.0)
    
    # Error rates
    task_error_rate_percent: float = Field(default=0.0)
    stream_error_rate_percent: float = Field(default=0.0)
    
    # Time period
    measurement_period_seconds: int = Field(default=3600)
    measured_at: datetime = Field(default_factory=datetime.utcnow)
    
    class Config:
        """Pydantic configuration."""
        json_encoders = {
            datetime: lambda v: v.isoformat(),
        }


class ScalingRecommendationResponse(BaseModel):
    """Response with scaling recommendations."""
    
    # Current state
    current_resources: Dict[str, Any] = Field(default_factory=dict)
    current_utilization: Dict[str, float] = Field(default_factory=dict)
    
    # Recommendations
    recommended_actions: List[Dict[str, Any]] = Field(default_factory=list)
    
    # Scaling targets
    target_cpu_utilization: float = Field(default=70.0)
    target_memory_utilization: float = Field(default=80.0)
    
    # Cost implications
    estimated_cost_change: Optional[float] = None
    cost_savings_opportunities: List[str] = Field(default_factory=list)
    
    # Justification
    reasoning: str = Field(default="")
    confidence_score: float = Field(default=0.8, ge=0.0, le=1.0)
    
    # Timestamp
    generated_at: datetime = Field(default_factory=datetime.utcnow)
    valid_until: datetime = Field(default_factory=lambda: datetime.utcnow() + timedelta(hours=1))
    
    class Config:
        """Pydantic configuration."""
        json_encoders = {
            datetime: lambda v: v.isoformat(),
        }


class HealthCheckResponse(BaseModel):
    """Response with system health information."""
    
    # Overall health
    status: str = Field(default="healthy")
    health_score: float = Field(default=100.0, ge=0.0, le=100.0)
    
    # Component health
    cluster_health: Dict[str, Any] = Field(default_factory=dict)
    stream_processor_health: Dict[str, Any] = Field(default_factory=dict)
    task_system_health: Dict[str, Any] = Field(default_factory=dict)
    
    # Issues and warnings
    issues: List[Dict[str, Any]] = Field(default_factory=list)
    warnings: List[Dict[str, Any]] = Field(default_factory=list)
    
    # Recommendations
    recommendations: List[str] = Field(default_factory=list)
    
    # Check metadata
    checked_at: datetime = Field(default_factory=datetime.utcnow)
    check_duration_ms: float = Field(default=0.0)
    
    class Config:
        """Pydantic configuration."""
        json_encoders = {
            datetime: lambda v: v.isoformat(),
        }