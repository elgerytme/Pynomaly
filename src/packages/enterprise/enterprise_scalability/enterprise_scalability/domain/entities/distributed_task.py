"""
Distributed Task domain entities for scalable task execution and scheduling.
"""

from datetime import datetime, timedelta
from enum import Enum
from typing import Dict, List, Optional, Any, Union, Callable
from uuid import UUID, uuid4

from pydantic import BaseModel, Field, validator


class TaskStatus(str, Enum):
    """Task execution status."""
    PENDING = "pending"
    QUEUED = "queued"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"
    RETRY = "retry"
    TIMEOUT = "timeout"


class TaskPriority(str, Enum):
    """Task execution priority."""
    LOW = "low"
    NORMAL = "normal"
    HIGH = "high"
    CRITICAL = "critical"


class TaskType(str, Enum):
    """Types of distributed tasks."""
    ANOMALY_DETECTION = "anomaly_detection"
    MODEL_TRAINING = "model_training"
    BATCH_PROCESSING = "batch_processing"
    DATA_PIPELINE = "data_pipeline"
    FEATURE_ENGINEERING = "feature_engineering"
    MODEL_INFERENCE = "model_inference"
    DATA_VALIDATION = "data_validation"
    CUSTOM = "custom"


class ResourceRequirements(BaseModel):
    """Task resource requirements."""
    
    cpu_cores: float = Field(default=1.0, ge=0.1, description="Required CPU cores")
    memory_gb: float = Field(default=1.0, ge=0.1, description="Required memory in GB")
    gpu_count: int = Field(default=0, ge=0, description="Required GPU count")
    gpu_memory_gb: Optional[float] = Field(None, ge=0.1, description="GPU memory requirement")
    storage_gb: float = Field(default=1.0, ge=0.1, description="Required storage in GB")
    network_mbps: Optional[float] = Field(None, ge=1.0, description="Network bandwidth requirement")
    
    # Constraints
    node_selector: Dict[str, str] = Field(default_factory=dict, description="Node selector labels")
    tolerations: List[Dict[str, Any]] = Field(default_factory=list, description="Node tolerations")
    affinity_rules: List[Dict[str, Any]] = Field(default_factory=list, description="Affinity rules")
    
    class Config:
        """Pydantic configuration."""
        json_encoders = {
            datetime: lambda v: v.isoformat(),
        }


class TaskResult(BaseModel):
    """Task execution result."""
    
    # Execution summary
    success: bool = Field(..., description="Task success status")
    return_value: Optional[Any] = Field(None, description="Task return value")
    error_message: Optional[str] = Field(None, description="Error message if failed")
    error_traceback: Optional[str] = Field(None, description="Full error traceback")
    
    # Execution metrics
    execution_time_seconds: float = Field(default=0.0, ge=0.0, description="Execution time")
    cpu_time_seconds: float = Field(default=0.0, ge=0.0, description="CPU time used")
    memory_peak_gb: float = Field(default=0.0, ge=0.0, description="Peak memory usage")
    disk_io_gb: float = Field(default=0.0, ge=0.0, description="Disk I/O volume")
    network_io_gb: float = Field(default=0.0, ge=0.0, description="Network I/O volume")
    
    # Output metadata
    output_size_bytes: int = Field(default=0, ge=0, description="Output data size")
    artifacts_created: List[str] = Field(default_factory=list, description="Created artifacts")
    logs: List[str] = Field(default_factory=list, description="Execution logs")
    
    # Metadata
    computed_at: datetime = Field(default_factory=datetime.utcnow)
    computed_by: Optional[str] = Field(None, description="Node that computed the task")
    
    class Config:
        """Pydantic configuration."""
        json_encoders = {
            datetime: lambda v: v.isoformat(),
        }


class DistributedTask(BaseModel):
    """
    Distributed task for scalable processing across compute clusters.
    
    Represents a unit of work that can be executed on distributed
    infrastructure with resource requirements, dependencies, and results.
    """
    
    id: UUID = Field(default_factory=uuid4, description="Task identifier")
    
    # Task identification
    name: str = Field(..., description="Task name")
    task_type: TaskType = Field(..., description="Type of task")
    function_name: str = Field(..., description="Function to execute")
    module_name: str = Field(..., description="Module containing function")
    description: str = Field(default="", description="Task description")
    
    # Ownership and context
    tenant_id: UUID = Field(..., description="Owning tenant")
    user_id: UUID = Field(..., description="User who submitted task")
    job_id: Optional[UUID] = Field(None, description="Parent job ID")
    workflow_id: Optional[UUID] = Field(None, description="Parent workflow ID")
    
    # Task configuration
    function_args: List[Any] = Field(default_factory=list, description="Function arguments")
    function_kwargs: Dict[str, Any] = Field(default_factory=dict, description="Function keyword arguments")
    environment_vars: Dict[str, str] = Field(default_factory=dict, description="Environment variables")
    
    # Execution settings
    priority: TaskPriority = Field(default=TaskPriority.NORMAL)
    timeout_seconds: Optional[int] = Field(None, ge=1, description="Task timeout")
    max_retries: int = Field(default=3, ge=0, description="Maximum retry attempts")
    retry_delay_seconds: int = Field(default=60, ge=1, description="Retry delay")
    
    # Resource requirements
    resources: ResourceRequirements = Field(default_factory=ResourceRequirements)
    
    # Dependencies
    depends_on: List[UUID] = Field(default_factory=list, description="Task dependencies")
    blocks: List[UUID] = Field(default_factory=list, description="Tasks blocked by this task")
    
    # Execution state
    status: TaskStatus = Field(default=TaskStatus.PENDING)
    current_retry: int = Field(default=0, ge=0, description="Current retry attempt")
    assigned_node: Optional[str] = Field(None, description="Assigned compute node")
    cluster_id: Optional[UUID] = Field(None, description="Assigned cluster")
    
    # Timing
    submitted_at: datetime = Field(default_factory=datetime.utcnow)
    scheduled_at: Optional[datetime] = Field(None, description="When task was scheduled")
    started_at: Optional[datetime] = Field(None, description="Task start time")
    completed_at: Optional[datetime] = Field(None, description="Task completion time")
    
    # Results and metrics
    result: Optional[TaskResult] = Field(None, description="Task execution result")
    progress_percent: float = Field(default=0.0, ge=0.0, le=100.0, description="Task progress")
    
    # Metadata
    tags: Dict[str, str] = Field(default_factory=dict, description="Task tags")
    labels: Dict[str, str] = Field(default_factory=dict, description="Task labels")
    
    class Config:
        """Pydantic configuration."""
        use_enum_values = True
        json_encoders = {
            datetime: lambda v: v.isoformat(),
            UUID: lambda v: str(v),
        }
    
    @validator('max_retries')
    def validate_max_retries(cls, v):
        """Validate max retries is reasonable."""
        if v > 10:
            raise ValueError("max_retries cannot exceed 10")
        return v
    
    def is_ready_to_run(self) -> bool:
        """Check if task is ready to run (no pending dependencies)."""
        return (
            self.status == TaskStatus.PENDING and
            len(self.depends_on) == 0  # Dependencies resolved
        )
    
    def is_running(self) -> bool:
        """Check if task is currently running."""
        return self.status == TaskStatus.RUNNING
    
    def is_completed(self) -> bool:
        """Check if task completed successfully."""
        return self.status == TaskStatus.COMPLETED
    
    def is_failed(self) -> bool:
        """Check if task failed."""
        return self.status in [TaskStatus.FAILED, TaskStatus.TIMEOUT, TaskStatus.CANCELLED]
    
    def can_retry(self) -> bool:
        """Check if task can be retried."""
        return (
            self.status == TaskStatus.FAILED and
            self.current_retry < self.max_retries
        )
    
    def get_execution_time(self) -> Optional[float]:
        """Get task execution time in seconds."""
        if self.started_at and self.completed_at:
            return (self.completed_at - self.started_at).total_seconds()
        return None
    
    def get_wait_time(self) -> float:
        """Get time task waited in queue."""
        if self.scheduled_at:
            return (self.scheduled_at - self.submitted_at).total_seconds()
        return (datetime.utcnow() - self.submitted_at).total_seconds()
    
    def schedule(self, cluster_id: UUID, node_id: Optional[str] = None) -> None:
        """Schedule task for execution."""
        self.status = TaskStatus.QUEUED
        self.cluster_id = cluster_id
        self.assigned_node = node_id
        self.scheduled_at = datetime.utcnow()
    
    def start(self, node_id: str) -> None:
        """Start task execution."""
        self.status = TaskStatus.RUNNING
        self.assigned_node = node_id
        self.started_at = datetime.utcnow()
    
    def complete(self, result: TaskResult) -> None:
        """Complete task with result."""
        self.status = TaskStatus.COMPLETED
        self.completed_at = datetime.utcnow()
        self.result = result
        self.progress_percent = 100.0
    
    def fail(self, error_message: str, traceback: Optional[str] = None) -> None:
        """Mark task as failed."""
        self.status = TaskStatus.FAILED
        self.completed_at = datetime.utcnow()
        
        # Create error result
        self.result = TaskResult(
            success=False,
            error_message=error_message,
            error_traceback=traceback,
            computed_at=datetime.utcnow(),
            computed_by=self.assigned_node
        )
    
    def retry(self) -> None:
        """Retry task execution."""
        if not self.can_retry():
            raise ValueError("Task cannot be retried")
        
        self.current_retry += 1
        self.status = TaskStatus.RETRY
        self.assigned_node = None
        self.started_at = None
        self.completed_at = None
        self.result = None
    
    def cancel(self) -> None:
        """Cancel task execution."""
        self.status = TaskStatus.CANCELLED
        self.completed_at = datetime.utcnow()
    
    def timeout(self) -> None:
        """Mark task as timed out."""
        self.status = TaskStatus.TIMEOUT
        self.completed_at = datetime.utcnow()
        
        self.result = TaskResult(
            success=False,
            error_message="Task execution timed out",
            computed_at=datetime.utcnow(),
            computed_by=self.assigned_node
        )
    
    def update_progress(self, progress_percent: float) -> None:
        """Update task progress."""
        self.progress_percent = max(0.0, min(100.0, progress_percent))
    
    def add_dependency(self, task_id: UUID) -> None:
        """Add task dependency."""
        if task_id not in self.depends_on:
            self.depends_on.append(task_id)
    
    def remove_dependency(self, task_id: UUID) -> None:
        """Remove task dependency."""
        if task_id in self.depends_on:
            self.depends_on.remove(task_id)
    
    def get_task_summary(self) -> Dict[str, Any]:
        """Get task summary information."""
        execution_time = self.get_execution_time()
        wait_time = self.get_wait_time()
        
        summary = {
            "id": str(self.id),
            "name": self.name,
            "type": self.task_type,
            "status": self.status,
            "priority": self.priority,
            "progress": self.progress_percent,
            "dependencies": len(self.depends_on),
            "retries": self.current_retry,
            "timing": {
                "submitted_at": self.submitted_at.isoformat(),
                "wait_time_seconds": wait_time,
                "execution_time_seconds": execution_time
            },
            "resources": {
                "cpu_cores": self.resources.cpu_cores,
                "memory_gb": self.resources.memory_gb,
                "gpu_count": self.resources.gpu_count
            }
        }
        
        # Add result summary if available
        if self.result:
            summary["result"] = {
                "success": self.result.success,
                "execution_time": self.result.execution_time_seconds,
                "memory_peak": self.result.memory_peak_gb,
                "cpu_time": self.result.cpu_time_seconds
            }
        
        return summary


class TaskBatch(BaseModel):
    """
    Batch of related distributed tasks.
    
    Groups tasks for coordinated execution with shared
    configuration and batch-level operations.
    """
    
    id: UUID = Field(default_factory=uuid4, description="Batch identifier")
    
    # Batch identification
    name: str = Field(..., description="Batch name")
    description: str = Field(default="", description="Batch description")
    batch_type: str = Field(default="general", description="Type of batch")
    
    # Ownership
    tenant_id: UUID = Field(..., description="Owning tenant")
    user_id: UUID = Field(..., description="User who created batch")
    
    # Batch configuration
    task_ids: List[UUID] = Field(default_factory=list, description="Tasks in batch")
    total_tasks: int = Field(default=0, ge=0, description="Total number of tasks")
    
    # Execution settings
    max_concurrent_tasks: int = Field(default=10, ge=1, description="Max concurrent tasks")
    batch_timeout_seconds: Optional[int] = Field(None, description="Batch timeout")
    stop_on_first_failure: bool = Field(default=False, description="Stop on first failure")
    
    # Progress tracking
    tasks_pending: int = Field(default=0, ge=0)
    tasks_running: int = Field(default=0, ge=0)
    tasks_completed: int = Field(default=0, ge=0)
    tasks_failed: int = Field(default=0, ge=0)
    tasks_cancelled: int = Field(default=0, ge=0)
    
    # Status
    status: str = Field(default="pending", description="Batch status")
    progress_percent: float = Field(default=0.0, ge=0.0, le=100.0)
    
    # Timing
    created_at: datetime = Field(default_factory=datetime.utcnow)
    started_at: Optional[datetime] = Field(None, description="Batch start time")
    completed_at: Optional[datetime] = Field(None, description="Batch completion time")
    
    # Metadata
    tags: Dict[str, str] = Field(default_factory=dict)
    
    class Config:
        """Pydantic configuration."""
        json_encoders = {
            datetime: lambda v: v.isoformat(),
            UUID: lambda v: str(v),
        }
    
    def update_task_counts(self, tasks: List[DistributedTask]) -> None:
        """Update task counts from task list."""
        self.tasks_pending = sum(1 for t in tasks if t.status == TaskStatus.PENDING)
        self.tasks_running = sum(1 for t in tasks if t.status == TaskStatus.RUNNING)
        self.tasks_completed = sum(1 for t in tasks if t.status == TaskStatus.COMPLETED)
        self.tasks_failed = sum(1 for t in tasks if t.is_failed())
        
        # Update progress
        if self.total_tasks > 0:
            completed_and_failed = self.tasks_completed + self.tasks_failed
            self.progress_percent = (completed_and_failed / self.total_tasks) * 100.0
        
        # Update batch status
        if self.tasks_completed == self.total_tasks:
            self.status = "completed"
            if not self.completed_at:
                self.completed_at = datetime.utcnow()
        elif self.tasks_failed > 0 and self.stop_on_first_failure:
            self.status = "failed"
            if not self.completed_at:
                self.completed_at = datetime.utcnow()
        elif self.tasks_running > 0:
            self.status = "running"
            if not self.started_at:
                self.started_at = datetime.utcnow()
    
    def get_success_rate(self) -> float:
        """Get batch success rate percentage."""
        completed_tasks = self.tasks_completed + self.tasks_failed
        if completed_tasks == 0:
            return 0.0
        return (self.tasks_completed / completed_tasks) * 100.0
    
    def is_completed(self) -> bool:
        """Check if batch is completed."""
        return self.status in ["completed", "failed"]
    
    def get_batch_summary(self) -> Dict[str, Any]:
        """Get batch summary information."""
        execution_time = None
        if self.started_at and self.completed_at:
            execution_time = (self.completed_at - self.started_at).total_seconds()
        
        return {
            "id": str(self.id),
            "name": self.name,
            "status": self.status,
            "progress": self.progress_percent,
            "tasks": {
                "total": self.total_tasks,
                "pending": self.tasks_pending,
                "running": self.tasks_running,
                "completed": self.tasks_completed,
                "failed": self.tasks_failed,
                "success_rate": self.get_success_rate()
            },
            "timing": {
                "created_at": self.created_at.isoformat(),
                "started_at": self.started_at.isoformat() if self.started_at else None,
                "completed_at": self.completed_at.isoformat() if self.completed_at else None,
                "execution_time_seconds": execution_time
            },
            "configuration": {
                "max_concurrent": self.max_concurrent_tasks,
                "stop_on_failure": self.stop_on_first_failure
            }
        }