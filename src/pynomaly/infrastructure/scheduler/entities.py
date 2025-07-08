"""Core entities for the scheduler system."""

from __future__ import annotations

import uuid
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from typing import Any, Dict, List, Optional, Set, Union


class ScheduleStatus(Enum):
    """Status of a schedule."""
    ACTIVE = "active"
    PAUSED = "paused"
    DISABLED = "disabled"
    DELETED = "deleted"


class JobStatus(Enum):
    """Status of a job instance."""
    PENDING = "pending"
    WAITING = "waiting"  # Waiting for dependencies
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"
    TIMEOUT = "timeout"
    RETRY = "retry"


class ExecutionStatus(Enum):
    """Status of a schedule execution."""
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"
    PARTIAL = "partial"  # Some jobs completed, others failed


@dataclass
class ResourceRequirement:
    """Resource requirements for a job."""
    cpu_cores: float = 1.0
    memory_gb: float = 1.0
    workers: int = 1
    gpu_count: int = 0
    max_runtime_minutes: int = 60
    
    def __post_init__(self):
        """Validate resource requirements."""
        if self.cpu_cores <= 0:
            raise ValueError("CPU cores must be positive")
        if self.memory_gb <= 0:
            raise ValueError("Memory must be positive")
        if self.workers <= 0:
            raise ValueError("Workers must be positive")
        if self.gpu_count < 0:
            raise ValueError("GPU count cannot be negative")
        if self.max_runtime_minutes <= 0:
            raise ValueError("Max runtime must be positive")


@dataclass
class JobDefinition:
    """Definition of a job that can be executed."""
    
    job_id: str
    name: str
    description: str = ""
    
    # Processing configuration
    processing_config: Dict[str, Any] = field(default_factory=dict)
    
    # Dependencies
    depends_on: Set[str] = field(default_factory=set)
    
    # Resource requirements
    resource_requirements: ResourceRequirement = field(default_factory=ResourceRequirement)
    
    # Retry configuration
    max_retries: int = 3
    retry_delay_seconds: int = 60
    
    # Timeout configuration
    timeout_minutes: int = 60
    
    # Metadata
    tags: Dict[str, str] = field(default_factory=dict)
    created_at: datetime = field(default_factory=datetime.now)
    updated_at: datetime = field(default_factory=datetime.now)
    
    def __post_init__(self):
        """Validate job definition."""
        if not self.job_id:
            raise ValueError("Job ID is required")
        if not self.name:
            raise ValueError("Job name is required")
        if self.max_retries < 0:
            raise ValueError("Max retries cannot be negative")
        if self.retry_delay_seconds < 0:
            raise ValueError("Retry delay cannot be negative")
        if self.timeout_minutes <= 0:
            raise ValueError("Timeout must be positive")
    
    def add_dependency(self, job_id: str) -> None:
        """Add a dependency to this job."""
        self.depends_on.add(job_id)
        self.updated_at = datetime.now()
    
    def remove_dependency(self, job_id: str) -> None:
        """Remove a dependency from this job."""
        self.depends_on.discard(job_id)
        self.updated_at = datetime.now()
    
    def has_dependency(self, job_id: str) -> bool:
        """Check if this job depends on another job."""
        return job_id in self.depends_on


@dataclass
class Schedule:
    """A schedule defines when and how to execute a DAG of jobs."""
    
    schedule_id: str
    name: str
    description: str = ""
    
    # DAG definition
    jobs: Dict[str, JobDefinition] = field(default_factory=dict)
    
    # Schedule configuration
    cron_expression: Optional[str] = None
    interval_seconds: Optional[int] = None
    
    # Status
    status: ScheduleStatus = ScheduleStatus.ACTIVE
    
    # Execution history
    last_execution_id: Optional[str] = None
    last_execution_at: Optional[datetime] = None
    last_execution_status: Optional[ExecutionStatus] = None
    
    # Schedule metadata
    next_execution_at: Optional[datetime] = None
    total_executions: int = 0
    successful_executions: int = 0
    failed_executions: int = 0
    
    # Configuration
    max_concurrent_jobs: int = 10
    global_timeout_minutes: int = 240  # 4 hours
    
    # Metadata
    tags: Dict[str, str] = field(default_factory=dict)
    created_at: datetime = field(default_factory=datetime.now)
    updated_at: datetime = field(default_factory=datetime.now)
    
    def __post_init__(self):
        """Validate schedule."""
        if not self.schedule_id:
            raise ValueError("Schedule ID is required")
        if not self.name:
            raise ValueError("Schedule name is required")
        if not self.cron_expression and not self.interval_seconds:
            raise ValueError("Either cron expression or interval must be provided")
        if self.max_concurrent_jobs <= 0:
            raise ValueError("Max concurrent jobs must be positive")
        if self.global_timeout_minutes <= 0:
            raise ValueError("Global timeout must be positive")
    
    def add_job(self, job_definition: JobDefinition) -> None:
        """Add a job to this schedule."""
        self.jobs[job_definition.job_id] = job_definition
        self.updated_at = datetime.now()
    
    def remove_job(self, job_id: str) -> None:
        """Remove a job from this schedule."""
        if job_id in self.jobs:
            del self.jobs[job_id]
            # Remove dependencies referencing this job
            for job in self.jobs.values():
                job.remove_dependency(job_id)
            self.updated_at = datetime.now()
    
    def get_job(self, job_id: str) -> Optional[JobDefinition]:
        """Get a job definition by ID."""
        return self.jobs.get(job_id)
    
    def get_job_ids(self) -> List[str]:
        """Get all job IDs in this schedule."""
        return list(self.jobs.keys())
    
    def is_active(self) -> bool:
        """Check if schedule is active."""
        return self.status == ScheduleStatus.ACTIVE
