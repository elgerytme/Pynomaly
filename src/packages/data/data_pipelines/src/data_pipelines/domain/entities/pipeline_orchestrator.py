"""Pipeline Orchestrator domain entity for pipeline management and execution."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional
from uuid import UUID, uuid4


class OrchestrationStatus(str, Enum):
    """Status of pipeline orchestration."""
    
    IDLE = "idle"
    SCHEDULING = "scheduling"
    RUNNING = "running"
    PAUSED = "paused"
    STOPPED = "stopped"
    ERROR = "error"
    MAINTENANCE = "maintenance"


class ExecutionStrategy(str, Enum):
    """Strategy for pipeline execution."""
    
    SEQUENTIAL = "sequential"
    PARALLEL = "parallel"
    CONDITIONAL = "conditional"
    BATCH = "batch"
    STREAMING = "streaming"
    HYBRID = "hybrid"


class PipelineExecutionMode(str, Enum):
    """Mode for pipeline execution."""
    
    IMMEDIATE = "immediate"
    SCHEDULED = "scheduled"
    TRIGGERED = "triggered"
    MANUAL = "manual"


@dataclass
class ExecutionMetrics:
    """Metrics for pipeline execution tracking."""
    
    total_executions: int = 0
    successful_executions: int = 0
    failed_executions: int = 0
    cancelled_executions: int = 0
    
    # Timing metrics
    average_duration_seconds: float = 0.0
    min_duration_seconds: float = 0.0
    max_duration_seconds: float = 0.0
    
    # Resource metrics
    average_cpu_usage: float = 0.0
    average_memory_usage_mb: float = 0.0
    peak_memory_usage_mb: float = 0.0
    
    # Queue metrics
    average_queue_time_seconds: float = 0.0
    max_queue_time_seconds: float = 0.0
    
    def __post_init__(self) -> None:
        """Validate metrics after initialization."""
        if self.total_executions < 0:
            raise ValueError("Total executions cannot be negative")
        
        if self.successful_executions < 0:
            raise ValueError("Successful executions cannot be negative")
        
        if self.failed_executions < 0:
            raise ValueError("Failed executions cannot be negative")
    
    @property
    def success_rate(self) -> float:
        """Get success rate as percentage."""
        if self.total_executions == 0:
            return 0.0
        return (self.successful_executions / self.total_executions) * 100
    
    @property
    def failure_rate(self) -> float:
        """Get failure rate as percentage."""
        if self.total_executions == 0:
            return 0.0
        return (self.failed_executions / self.total_executions) * 100
    
    def update_execution(self, success: bool, duration_seconds: float, 
                        cpu_usage: float = 0.0, memory_usage_mb: float = 0.0) -> None:
        """Update metrics with new execution data."""
        self.total_executions += 1
        
        if success:
            self.successful_executions += 1
        else:
            self.failed_executions += 1
        
        # Update duration metrics
        if self.total_executions == 1:
            self.average_duration_seconds = duration_seconds
            self.min_duration_seconds = duration_seconds
            self.max_duration_seconds = duration_seconds
        else:
            # Update average using cumulative moving average
            self.average_duration_seconds = (
                (self.average_duration_seconds * (self.total_executions - 1) + duration_seconds) 
                / self.total_executions
            )
            self.min_duration_seconds = min(self.min_duration_seconds, duration_seconds)
            self.max_duration_seconds = max(self.max_duration_seconds, duration_seconds)
        
        # Update resource metrics
        if cpu_usage > 0:
            if self.total_executions == 1:
                self.average_cpu_usage = cpu_usage
            else:
                self.average_cpu_usage = (
                    (self.average_cpu_usage * (self.total_executions - 1) + cpu_usage)
                    / self.total_executions
                )
        
        if memory_usage_mb > 0:
            if self.total_executions == 1:
                self.average_memory_usage_mb = memory_usage_mb
            else:
                self.average_memory_usage_mb = (
                    (self.average_memory_usage_mb * (self.total_executions - 1) + memory_usage_mb)
                    / self.total_executions
                )
            self.peak_memory_usage_mb = max(self.peak_memory_usage_mb, memory_usage_mb)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert metrics to dictionary."""
        return {
            "total_executions": self.total_executions,
            "successful_executions": self.successful_executions,
            "failed_executions": self.failed_executions,
            "cancelled_executions": self.cancelled_executions,
            "success_rate": self.success_rate,
            "failure_rate": self.failure_rate,
            "average_duration_seconds": self.average_duration_seconds,
            "min_duration_seconds": self.min_duration_seconds,
            "max_duration_seconds": self.max_duration_seconds,
            "average_cpu_usage": self.average_cpu_usage,
            "average_memory_usage_mb": self.average_memory_usage_mb,
            "peak_memory_usage_mb": self.peak_memory_usage_mb,
            "average_queue_time_seconds": self.average_queue_time_seconds,
            "max_queue_time_seconds": self.max_queue_time_seconds,
        }


@dataclass
class PipelineOrchestrator:
    """Pipeline orchestrator domain entity for managing pipeline execution."""
    
    id: UUID = field(default_factory=uuid4)
    name: str = ""
    description: str = ""
    version: str = "1.0.0"
    
    # Orchestration configuration
    status: OrchestrationStatus = OrchestrationStatus.IDLE
    execution_strategy: ExecutionStrategy = ExecutionStrategy.SEQUENTIAL
    execution_mode: PipelineExecutionMode = PipelineExecutionMode.SCHEDULED
    
    # Pipeline management
    registered_pipelines: List[UUID] = field(default_factory=list)
    active_executions: Dict[UUID, Dict[str, Any]] = field(default_factory=dict)
    execution_queue: List[Dict[str, Any]] = field(default_factory=list)
    
    # Resource management
    max_concurrent_executions: int = 5
    max_queue_size: int = 100
    worker_pool_size: int = 10
    memory_limit_mb: Optional[int] = None
    cpu_limit_cores: Optional[float] = None
    
    # Scheduling configuration
    schedule_enabled: bool = True
    schedule_interval_minutes: int = 60
    retry_policy: Dict[str, Any] = field(default_factory=lambda: {
        "max_retries": 3,
        "retry_delay_seconds": 300,
        "exponential_backoff": True
    })
    
    # Monitoring and alerts
    monitoring_enabled: bool = True
    alert_thresholds: Dict[str, Any] = field(default_factory=lambda: {
        "failure_rate_threshold": 0.1,
        "queue_size_threshold": 50,
        "execution_time_threshold_seconds": 3600
    })
    
    # Execution tracking
    metrics: ExecutionMetrics = field(default_factory=ExecutionMetrics)
    last_execution_at: Optional[datetime] = None
    next_scheduled_at: Optional[datetime] = None
    
    # Metadata
    created_at: datetime = field(default_factory=datetime.utcnow)
    created_by: str = ""
    updated_at: datetime = field(default_factory=datetime.utcnow)
    updated_by: str = ""
    
    # Configuration
    config: Dict[str, Any] = field(default_factory=dict)
    environment_vars: Dict[str, str] = field(default_factory=dict)
    tags: List[str] = field(default_factory=list)
    
    def __post_init__(self) -> None:
        """Validate orchestrator after initialization."""
        if not self.name:
            raise ValueError("Orchestrator name cannot be empty")
        
        if len(self.name) > 100:
            raise ValueError("Orchestrator name cannot exceed 100 characters")
        
        if self.max_concurrent_executions <= 0:
            raise ValueError("Max concurrent executions must be positive")
        
        if self.max_queue_size <= 0:
            raise ValueError("Max queue size must be positive")
        
        if self.worker_pool_size <= 0:
            raise ValueError("Worker pool size must be positive")
        
        if self.schedule_interval_minutes <= 0:
            raise ValueError("Schedule interval must be positive")
    
    @property
    def is_running(self) -> bool:
        """Check if orchestrator is running."""
        return self.status == OrchestrationStatus.RUNNING
    
    @property
    def is_available(self) -> bool:
        """Check if orchestrator is available to accept new executions."""
        return self.status in [
            OrchestrationStatus.IDLE,
            OrchestrationStatus.RUNNING,
            OrchestrationStatus.SCHEDULING
        ]
    
    @property
    def current_queue_size(self) -> int:
        """Get current queue size."""
        return len(self.execution_queue)
    
    @property
    def current_active_executions(self) -> int:
        """Get current number of active executions."""
        return len(self.active_executions)
    
    @property
    def is_queue_full(self) -> bool:
        """Check if execution queue is full."""
        return self.current_queue_size >= self.max_queue_size
    
    @property
    def can_accept_execution(self) -> bool:
        """Check if orchestrator can accept new execution."""
        return (
            self.is_available and
            not self.is_queue_full and
            self.current_active_executions < self.max_concurrent_executions
        )
    
    @property
    def resource_utilization(self) -> Dict[str, float]:
        """Get current resource utilization."""
        execution_utilization = (
            self.current_active_executions / self.max_concurrent_executions
        ) * 100
        
        queue_utilization = (
            self.current_queue_size / self.max_queue_size
        ) * 100
        
        return {
            "execution_utilization": execution_utilization,
            "queue_utilization": queue_utilization,
        }
    
    def register_pipeline(self, pipeline_id: UUID) -> None:
        """Register a pipeline with the orchestrator."""
        if pipeline_id in self.registered_pipelines:
            raise ValueError(f"Pipeline {pipeline_id} is already registered")
        
        self.registered_pipelines.append(pipeline_id)
        self.updated_at = datetime.utcnow()
    
    def unregister_pipeline(self, pipeline_id: UUID) -> bool:
        """Unregister a pipeline from the orchestrator."""
        if pipeline_id not in self.registered_pipelines:
            return False
        
        # Check if pipeline is currently executing
        if pipeline_id in self.active_executions:
            raise ValueError(f"Cannot unregister pipeline {pipeline_id} - currently executing")
        
        self.registered_pipelines.remove(pipeline_id)
        self.updated_at = datetime.utcnow()
        return True
    
    def queue_execution(self, pipeline_id: UUID, execution_config: Optional[Dict[str, Any]] = None) -> str:
        """Queue a pipeline for execution."""
        if not self.can_accept_execution:
            raise ValueError("Cannot accept new execution - orchestrator unavailable or at capacity")
        
        if pipeline_id not in self.registered_pipelines:
            raise ValueError(f"Pipeline {pipeline_id} is not registered")
        
        execution_id = str(uuid4())
        execution_request = {
            "execution_id": execution_id,
            "pipeline_id": pipeline_id,
            "queued_at": datetime.utcnow(),
            "config": execution_config or {},
            "priority": execution_config.get("priority", 5) if execution_config else 5,
            "retry_count": 0
        }
        
        self.execution_queue.append(execution_request)
        
        # Sort queue by priority (lower number = higher priority)
        self.execution_queue.sort(key=lambda x: x["priority"])
        
        self.updated_at = datetime.utcnow()
        return execution_id
    
    def start_execution(self, execution_id: str) -> bool:
        """Start execution from queue."""
        # Find execution in queue
        execution_request = None
        for i, req in enumerate(self.execution_queue):
            if req["execution_id"] == execution_id:
                execution_request = self.execution_queue.pop(i)
                break
        
        if not execution_request:
            return False
        
        if self.current_active_executions >= self.max_concurrent_executions:
            # Put back in queue if at capacity
            self.execution_queue.append(execution_request)
            return False
        
        # Start execution
        execution_info = {
            **execution_request,
            "started_at": datetime.utcnow(),
            "status": "running"
        }
        
        self.active_executions[execution_id] = execution_info
        self.last_execution_at = execution_info["started_at"]
        self.updated_at = execution_info["started_at"]
        
        return True
    
    def complete_execution(self, execution_id: str, success: bool, 
                          duration_seconds: float, result_data: Optional[Dict[str, Any]] = None) -> None:
        """Complete an active execution."""
        if execution_id not in self.active_executions:
            raise ValueError(f"Execution {execution_id} is not active")
        
        execution_info = self.active_executions.pop(execution_id)
        completion_time = datetime.utcnow()
        
        # Update metrics
        self.metrics.update_execution(success, duration_seconds)
        
        # Store completion info in config for history
        if "completed_executions" not in self.config:
            self.config["completed_executions"] = []
        
        completion_record = {
            "execution_id": execution_id,
            "pipeline_id": execution_info["pipeline_id"],
            "started_at": execution_info["started_at"].isoformat(),
            "completed_at": completion_time.isoformat(),
            "duration_seconds": duration_seconds,
            "success": success,
            "result_data": result_data or {}
        }
        
        self.config["completed_executions"].append(completion_record)
        
        # Keep only last 100 completion records
        if len(self.config["completed_executions"]) > 100:
            self.config["completed_executions"] = self.config["completed_executions"][-100:]
        
        self.updated_at = completion_time
    
    def cancel_execution(self, execution_id: str) -> bool:
        """Cancel an active execution."""
        if execution_id not in self.active_executions:
            return False
        
        execution_info = self.active_executions.pop(execution_id)
        self.metrics.cancelled_executions += 1
        
        # Log cancellation
        if "cancelled_executions" not in self.config:
            self.config["cancelled_executions"] = []
        
        cancellation_record = {
            "execution_id": execution_id,
            "pipeline_id": execution_info["pipeline_id"],
            "started_at": execution_info["started_at"].isoformat(),
            "cancelled_at": datetime.utcnow().isoformat()
        }
        
        self.config["cancelled_executions"].append(cancellation_record)
        
        # Keep only last 50 cancellation records
        if len(self.config["cancelled_executions"]) > 50:
            self.config["cancelled_executions"] = self.config["cancelled_executions"][-50:]
        
        self.updated_at = datetime.utcnow()
        return True
    
    def start_orchestration(self) -> None:
        """Start the orchestrator."""
        if self.status == OrchestrationStatus.RUNNING:
            raise ValueError("Orchestrator is already running")
        
        self.status = OrchestrationStatus.RUNNING
        self.updated_at = datetime.utcnow()
    
    def stop_orchestration(self) -> None:
        """Stop the orchestrator."""
        if self.status == OrchestrationStatus.STOPPED:
            raise ValueError("Orchestrator is already stopped")
        
        self.status = OrchestrationStatus.STOPPED
        self.updated_at = datetime.utcnow()
    
    def pause_orchestration(self) -> None:
        """Pause the orchestrator."""
        if self.status != OrchestrationStatus.RUNNING:
            raise ValueError("Can only pause a running orchestrator")
        
        self.status = OrchestrationStatus.PAUSED
        self.updated_at = datetime.utcnow()
    
    def resume_orchestration(self) -> None:
        """Resume the orchestrator."""
        if self.status != OrchestrationStatus.PAUSED:
            raise ValueError("Can only resume a paused orchestrator")
        
        self.status = OrchestrationStatus.RUNNING
        self.updated_at = datetime.utcnow()
    
    def enter_maintenance_mode(self) -> None:
        """Enter maintenance mode."""
        self.status = OrchestrationStatus.MAINTENANCE
        self.updated_at = datetime.utcnow()
    
    def exit_maintenance_mode(self) -> None:
        """Exit maintenance mode."""
        if self.status != OrchestrationStatus.MAINTENANCE:
            raise ValueError("Not in maintenance mode")
        
        self.status = OrchestrationStatus.IDLE
        self.updated_at = datetime.utcnow()
    
    def update_resource_limits(self, max_concurrent: Optional[int] = None,
                              max_queue_size: Optional[int] = None,
                              memory_limit_mb: Optional[int] = None,
                              cpu_limit_cores: Optional[float] = None) -> None:
        """Update resource limits."""
        if max_concurrent is not None:
            if max_concurrent <= 0:
                raise ValueError("Max concurrent executions must be positive")
            self.max_concurrent_executions = max_concurrent
        
        if max_queue_size is not None:
            if max_queue_size <= 0:
                raise ValueError("Max queue size must be positive")
            self.max_queue_size = max_queue_size
        
        if memory_limit_mb is not None:
            if memory_limit_mb <= 0:
                raise ValueError("Memory limit must be positive")
            self.memory_limit_mb = memory_limit_mb
        
        if cpu_limit_cores is not None:
            if cpu_limit_cores <= 0:
                raise ValueError("CPU limit must be positive")
            self.cpu_limit_cores = cpu_limit_cores
        
        self.updated_at = datetime.utcnow()
    
    def add_tag(self, tag: str) -> None:
        """Add a tag to the orchestrator."""
        if tag and tag not in self.tags:
            self.tags.append(tag)
            self.updated_at = datetime.utcnow()
    
    def remove_tag(self, tag: str) -> None:
        """Remove a tag from the orchestrator."""
        if tag in self.tags:
            self.tags.remove(tag)
            self.updated_at = datetime.utcnow()
    
    def has_tag(self, tag: str) -> bool:
        """Check if orchestrator has a specific tag."""
        return tag in self.tags
    
    def update_config(self, key: str, value: Any) -> None:
        """Update configuration key-value pair."""
        self.config[key] = value
        self.updated_at = datetime.utcnow()
    
    def get_health_status(self) -> Dict[str, Any]:
        """Get health status of the orchestrator."""
        resource_util = self.resource_utilization
        
        # Determine health based on various factors
        health_issues = []
        
        if self.metrics.failure_rate > (self.alert_thresholds["failure_rate_threshold"] * 100):
            health_issues.append("High failure rate")
        
        if resource_util["queue_utilization"] > 80:
            health_issues.append("Queue near capacity")
        
        if resource_util["execution_utilization"] > 90:
            health_issues.append("High execution load")
        
        if self.status == OrchestrationStatus.ERROR:
            health_issues.append("Orchestrator in error state")
        
        health_score = max(0, 100 - (len(health_issues) * 25))
        
        return {
            "status": self.status.value,
            "health_score": health_score,
            "health_issues": health_issues,
            "resource_utilization": resource_util,
            "active_executions": self.current_active_executions,
            "queued_executions": self.current_queue_size,
            "registered_pipelines": len(self.registered_pipelines),
            "metrics": self.metrics.to_dict(),
            "is_healthy": len(health_issues) == 0
        }
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert orchestrator to dictionary."""
        return {
            "id": str(self.id),
            "name": self.name,
            "description": self.description,
            "version": self.version,
            "status": self.status.value,
            "execution_strategy": self.execution_strategy.value,
            "execution_mode": self.execution_mode.value,
            "registered_pipelines": [str(pid) for pid in self.registered_pipelines],
            "active_executions": len(self.active_executions),
            "execution_queue": self.current_queue_size,
            "max_concurrent_executions": self.max_concurrent_executions,
            "max_queue_size": self.max_queue_size,
            "worker_pool_size": self.worker_pool_size,
            "memory_limit_mb": self.memory_limit_mb,
            "cpu_limit_cores": self.cpu_limit_cores,
            "schedule_enabled": self.schedule_enabled,
            "schedule_interval_minutes": self.schedule_interval_minutes,
            "retry_policy": self.retry_policy,
            "monitoring_enabled": self.monitoring_enabled,
            "alert_thresholds": self.alert_thresholds,
            "metrics": self.metrics.to_dict(),
            "last_execution_at": (
                self.last_execution_at.isoformat()
                if self.last_execution_at else None
            ),
            "next_scheduled_at": (
                self.next_scheduled_at.isoformat()
                if self.next_scheduled_at else None
            ),
            "created_at": self.created_at.isoformat(),
            "created_by": self.created_by,
            "updated_at": self.updated_at.isoformat(),
            "updated_by": self.updated_by,
            "config": self.config,
            "environment_vars": self.environment_vars,
            "tags": self.tags,
            "is_running": self.is_running,
            "is_available": self.is_available,
            "can_accept_execution": self.can_accept_execution,
            "resource_utilization": self.resource_utilization,
        }
    
    def __str__(self) -> str:
        """Human-readable string representation."""
        return f"PipelineOrchestrator('{self.name}', {self.status.value}, pipelines={len(self.registered_pipelines)})"
    
    def __repr__(self) -> str:
        """Developer string representation."""
        return (
            f"PipelineOrchestrator(id={self.id}, name='{self.name}', "
            f"status={self.status.value}, executions={self.current_active_executions})"
        )