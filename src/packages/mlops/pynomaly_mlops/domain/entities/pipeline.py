"""MLOps Pipeline Domain Entity

Represents the core pipeline abstraction for MLOps workflows with advanced DAG-based execution,
step dependencies, resource management, and comprehensive lifecycle orchestration.
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from enum import Enum
from typing import Dict, Any, Optional, List, Set, Union
from uuid import UUID, uuid4

from pydantic import BaseModel, Field, validator


class PipelineStatus(str, Enum):
    """Pipeline execution status enumeration."""
    
    DRAFT = "draft"
    VALIDATING = "validating"
    READY = "ready"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"
    PAUSED = "paused"
    ARCHIVED = "archived"


class StepStatus(str, Enum):
    """Pipeline step execution status enumeration."""
    
    PENDING = "pending"
    WAITING = "waiting"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    SKIPPED = "skipped"
    RETRYING = "retrying"
    CANCELLED = "cancelled"


class StepType(str, Enum):
    """Pipeline step type enumeration."""
    
    DATA_INGESTION = "data_ingestion"
    DATA_VALIDATION = "data_validation"
    DATA_PREPROCESSING = "data_preprocessing"
    FEATURE_ENGINEERING = "feature_engineering"
    MODEL_TRAINING = "model_training"
    MODEL_EVALUATION = "model_evaluation"
    MODEL_VALIDATION = "model_validation"
    MODEL_DEPLOYMENT = "model_deployment"
    MONITORING = "monitoring"
    NOTIFICATION = "notification"
    CUSTOM = "custom"


class RetryPolicy(BaseModel):
    """Retry policy configuration for pipeline steps."""
    
    max_attempts: int = Field(default=3, ge=1, le=10)
    delay_seconds: float = Field(default=1.0, ge=0.1, le=3600.0)
    backoff_multiplier: float = Field(default=2.0, ge=1.0, le=10.0)
    max_delay_seconds: float = Field(default=300.0, ge=1.0, le=3600.0)


class ResourceRequirements(BaseModel):
    """Resource requirements for pipeline step execution."""
    
    cpu_cores: float = Field(default=1.0, ge=0.1, le=64.0)
    memory_gb: float = Field(default=1.0, ge=0.1, le=512.0)
    gpu_count: int = Field(default=0, ge=0, le=8)
    disk_gb: float = Field(default=10.0, ge=1.0, le=1000.0)
    timeout_minutes: int = Field(default=60, ge=1, le=1440)


class ScheduleType(Enum):
    """Pipeline schedule type."""
    MANUAL = "manual"
    CRON = "cron"
    INTERVAL = "interval"
    EVENT_DRIVEN = "event_driven"


class PipelineStep(BaseModel):
    """Enhanced pipeline step with advanced resource management and execution tracking."""
    
    id: UUID = Field(default_factory=uuid4)
    name: str = Field(..., min_length=1, max_length=100)
    step_type: StepType
    description: Optional[str] = Field(None, max_length=500)
    
    # Execution configuration
    command: str = Field(..., min_length=1)
    working_directory: Optional[str] = None
    environment_variables: Dict[str, str] = Field(default_factory=dict)
    parameters: Dict[str, Any] = Field(default_factory=dict)
    
    # Dependencies and ordering
    depends_on: Set[UUID] = Field(default_factory=set)
    
    # Resource and retry configuration
    resource_requirements: ResourceRequirements = Field(default_factory=ResourceRequirements)
    retry_policy: RetryPolicy = Field(default_factory=RetryPolicy)
    
    # Runtime state
    status: StepStatus = Field(default=StepStatus.PENDING)
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    attempt_count: int = Field(default=0, ge=0)
    
    # Execution results
    exit_code: Optional[int] = None
    stdout: Optional[str] = None
    stderr: Optional[str] = None
    artifacts: Dict[str, str] = Field(default_factory=dict)
    metrics: Dict[str, float] = Field(default_factory=dict)
    
    # Metadata
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    updated_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    created_by: Optional[str] = None
    tags: Set[str] = Field(default_factory=set)
    
    class Config:
        use_enum_values = True
        json_encoders = {
            datetime: lambda v: v.isoformat(),
            UUID: str,
            set: list
        }
    
    @validator('depends_on', pre=True)
    def convert_depends_on_to_set(cls, v):
        """Convert depends_on to set if provided as list."""
        if isinstance(v, list):
            return set(v)
        return v
    
    def start_execution(self) -> None:
        """Mark step as started."""
        self.status = StepStatus.RUNNING
        self.started_at = datetime.now(timezone.utc)
        self.attempt_count += 1
        self.updated_at = datetime.now(timezone.utc)
    
    def complete_execution(self, exit_code: int = 0, stdout: str = "", stderr: str = "") -> None:
        """Mark step as completed."""
        self.status = StepStatus.COMPLETED
        self.completed_at = datetime.now(timezone.utc)
        self.exit_code = exit_code
        self.stdout = stdout
        self.stderr = stderr
        self.updated_at = datetime.now(timezone.utc)
    
    def fail_execution(self, exit_code: int = 1, stdout: str = "", stderr: str = "") -> None:
        """Mark step as failed."""
        self.status = StepStatus.FAILED
        self.completed_at = datetime.now(timezone.utc)
        self.exit_code = exit_code
        self.stdout = stdout
        self.stderr = stderr
        self.updated_at = datetime.now(timezone.utc)
    
    def should_retry(self) -> bool:
        """Check if step should be retried based on retry policy."""
        return (
            self.status == StepStatus.FAILED and
            self.attempt_count < self.retry_policy.max_attempts
        )
    
    def get_retry_delay(self) -> float:
        """Calculate retry delay based on retry policy."""
        delay = self.retry_policy.delay_seconds * (
            self.retry_policy.backoff_multiplier ** (self.attempt_count - 1)
        )
        return min(delay, self.retry_policy.max_delay_seconds)
    
    @property
    def execution_duration(self) -> Optional[float]:
        """Get execution duration in seconds."""
        if self.started_at and self.completed_at:
            return (self.completed_at - self.started_at).total_seconds()
        return None
    
    @property
    def is_terminal_status(self) -> bool:
        """Check if step is in a terminal status."""
        return self.status in {StepStatus.COMPLETED, StepStatus.FAILED, StepStatus.SKIPPED}


class PipelineSchedule(BaseModel):
    """Pipeline scheduling configuration."""
    
    enabled: bool = False
    cron_expression: Optional[str] = None
    timezone: str = "UTC"
    max_concurrent_runs: int = Field(default=1, ge=1, le=10)
    
    @validator('cron_expression')
    def validate_cron(cls, v):
        """Basic cron expression validation."""
        if v is not None:
            parts = v.split()
            if len(parts) != 5:
                raise ValueError("Cron expression must have 5 parts")
        return v


class Pipeline(BaseModel):
    """MLOps Pipeline for workflow orchestration with advanced DAG execution."""
    
    id: UUID = Field(default_factory=uuid4)
    name: str = Field(..., min_length=1, max_length=100)
    description: Optional[str] = Field(None, max_length=1000)
    version: str = Field(default="1.0.0")
    
    # Pipeline configuration
    steps: Dict[UUID, PipelineStep] = Field(default_factory=dict)
    schedule: Optional[PipelineSchedule] = None
    
    # Pipeline state
    status: PipelineStatus = Field(default=PipelineStatus.DRAFT)
    
    # Execution tracking
    current_run_id: Optional[UUID] = None
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    
    # Metadata
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    updated_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    created_by: Optional[str] = None
    tags: Set[str] = Field(default_factory=set)
    
    # Configuration
    max_parallel_steps: int = Field(default=5, ge=1, le=50)
    global_timeout_minutes: int = Field(default=480, ge=1, le=2880)  # 8 hours default
    
    class Config:
        use_enum_values = True
        json_encoders = {
            datetime: lambda v: v.isoformat(),
            UUID: str,
            set: list
        }
    
    def add_step(self, step: PipelineStep) -> None:
        """Add a step to the pipeline."""
        self.steps[step.id] = step
        self.updated_at = datetime.now(timezone.utc)
    
    def remove_step(self, step_id: UUID) -> None:
        """Remove a step from the pipeline."""
        if step_id in self.steps:
            # Remove dependencies to this step from other steps
            for step in self.steps.values():
                step.depends_on.discard(step_id)
            
            del self.steps[step_id]
            self.updated_at = datetime.now(timezone.utc)
    
    def validate_dag(self) -> List[str]:
        """Validate the pipeline forms a valid DAG (no cycles)."""
        errors = []
        
        # Check for cycles using DFS
        visited = set()
        rec_stack = set()
        
        def has_cycle(step_id: UUID) -> bool:
            if step_id in rec_stack:
                return True
            if step_id in visited:
                return False
            
            visited.add(step_id)
            rec_stack.add(step_id)
            
            step = self.steps.get(step_id)
            if step:
                for dep_id in step.depends_on:
                    if dep_id not in self.steps:
                        errors.append(f"Step {step.name} depends on non-existent step {dep_id}")
                        continue
                    
                    if has_cycle(dep_id):
                        return True
            
            rec_stack.remove(step_id)
            return False
        
        # Check each step for cycles
        for step_id in self.steps:
            if step_id not in visited:
                if has_cycle(step_id):
                    errors.append("Pipeline contains circular dependencies")
                    break
        
        return errors
    
    def get_execution_order(self) -> List[List[UUID]]:
        """Get step execution order as levels (parallel groups)."""
        if not self.steps:
            return []
        
        # Calculate in-degree for each step
        in_degree = {step_id: 0 for step_id in self.steps}
        for step in self.steps.values():
            for dep_id in step.depends_on:
                if dep_id in in_degree:
                    in_degree[dep_id] += 1
        
        execution_levels = []
        remaining_steps = set(self.steps.keys())
        
        while remaining_steps:
            # Find steps with no dependencies (in-degree 0)
            ready_steps = [
                step_id for step_id in remaining_steps
                if in_degree[step_id] == 0
            ]
            
            if not ready_steps:
                # Should not happen if DAG is valid
                break
            
            execution_levels.append(ready_steps)
            
            # Remove ready steps and update in-degrees
            for step_id in ready_steps:
                remaining_steps.remove(step_id)
                step = self.steps[step_id]
                for dep_id in step.depends_on:
                    if dep_id in in_degree:
                        in_degree[dep_id] -= 1
        
        return execution_levels
    
    def get_root_steps(self) -> List[UUID]:
        """Get steps with no dependencies (can start immediately)."""
        return [
            step_id for step_id, step in self.steps.items()
            if not step.depends_on
        ]
    
    def get_leaf_steps(self) -> List[UUID]:
        """Get steps that no other steps depend on."""
        dependent_steps = set()
        for step in self.steps.values():
            dependent_steps.update(step.depends_on)
        
        return [
            step_id for step_id in self.steps.keys()
            if step_id not in dependent_steps
        ]
    
    def can_start_step(self, step_id: UUID) -> bool:
        """Check if a step can be started based on dependencies."""
        step = self.steps.get(step_id)
        if not step or step.status != StepStatus.PENDING:
            return False
        
        # Check all dependencies are completed
        for dep_id in step.depends_on:
            dep_step = self.steps.get(dep_id)
            if not dep_step or dep_step.status != StepStatus.COMPLETED:
                return False
        
        return True
    
    def start_pipeline(self, run_id: Optional[UUID] = None) -> None:
        """Start pipeline execution."""
        self.status = PipelineStatus.RUNNING
        self.current_run_id = run_id or uuid4()
        self.started_at = datetime.now(timezone.utc)
        self.updated_at = datetime.now(timezone.utc)
        
        # Reset all steps to pending
        for step in self.steps.values():
            step.status = StepStatus.PENDING
            step.started_at = None
            step.completed_at = None
            step.attempt_count = 0
    
    def complete_pipeline(self) -> None:
        """Mark pipeline as completed."""
        self.status = PipelineStatus.COMPLETED
        self.completed_at = datetime.now(timezone.utc)
        self.updated_at = datetime.now(timezone.utc)
    
    def fail_pipeline(self) -> None:
        """Mark pipeline as failed."""
        self.status = PipelineStatus.FAILED
        self.completed_at = datetime.now(timezone.utc)
        self.updated_at = datetime.now(timezone.utc)
    
    def cancel_pipeline(self) -> None:
        """Cancel pipeline execution."""
        self.status = PipelineStatus.CANCELLED
        self.completed_at = datetime.now(timezone.utc)
        self.updated_at = datetime.now(timezone.utc)
    
    @property
    def execution_duration(self) -> Optional[float]:
        """Get pipeline execution duration in seconds."""
        if self.started_at and self.completed_at:
            return (self.completed_at - self.started_at).total_seconds()
        return None
    
    @property
    def is_running(self) -> bool:
        """Check if pipeline is currently running."""
        return self.status == PipelineStatus.RUNNING
    
    @property
    def is_terminal_status(self) -> bool:
        """Check if pipeline is in a terminal status."""
        return self.status in {
            PipelineStatus.COMPLETED,
            PipelineStatus.FAILED,
            PipelineStatus.CANCELLED
        }
    
    def get_progress(self) -> Dict[str, Any]:
        """Get pipeline execution progress."""
        total_steps = len(self.steps)
        if total_steps == 0:
            return {"total_steps": 0, "completed_steps": 0, "progress_percentage": 100.0}
        
        completed_steps = sum(
            1 for step in self.steps.values()
            if step.status == StepStatus.COMPLETED
        )
        
        failed_steps = sum(
            1 for step in self.steps.values()
            if step.status == StepStatus.FAILED
        )
        
        running_steps = sum(
            1 for step in self.steps.values()
            if step.status == StepStatus.RUNNING
        )
        
        progress_percentage = (completed_steps / total_steps) * 100.0
        
        return {
            "total_steps": total_steps,
            "completed_steps": completed_steps,
            "failed_steps": failed_steps,
            "running_steps": running_steps,
            "progress_percentage": progress_percentage,
            "status": self.status
        }
    


class PipelineRun(BaseModel):
    """Individual execution instance of a pipeline."""
    
    id: UUID = Field(default_factory=uuid4)
    pipeline_id: UUID
    pipeline_version: str
    
    # Execution tracking
    status: PipelineStatus = Field(default=PipelineStatus.RUNNING)
    started_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    completed_at: Optional[datetime] = None
    
    # Step execution state
    step_runs: Dict[UUID, Dict[str, Any]] = Field(default_factory=dict)
    
    # Execution context
    triggered_by: Optional[str] = None
    trigger_type: str = "manual"  # manual, scheduled, webhook, etc.
    parameters: Dict[str, Any] = Field(default_factory=dict)
    
    # Results
    artifacts: Dict[str, str] = Field(default_factory=dict)
    metrics: Dict[str, float] = Field(default_factory=dict)
    
    class Config:
        use_enum_values = True
        json_encoders = {
            datetime: lambda v: v.isoformat(),
            UUID: str
        }
    
    def complete_run(self) -> None:
        """Mark run as completed."""
        self.status = PipelineStatus.COMPLETED
        self.completed_at = datetime.now(timezone.utc)
    
    def fail_run(self) -> None:
        """Mark run as failed."""
        self.status = PipelineStatus.FAILED
        self.completed_at = datetime.now(timezone.utc)
    
    @property
    def execution_duration(self) -> Optional[float]:
        """Get run execution duration in seconds."""
        if self.started_at and self.completed_at:
            return (self.completed_at - self.started_at).total_seconds()
        return None