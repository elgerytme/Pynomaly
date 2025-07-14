"""Pipeline Entities

Domain entities for training and inference pipeline management with DAG support.
"""

from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from typing import Dict, Any, Optional, List, Set
from uuid import UUID, uuid4


class PipelineStatus(Enum):
    """Pipeline execution status."""
    DRAFT = "draft"
    ACTIVE = "active"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    PAUSED = "paused"
    CANCELLED = "cancelled"
    ARCHIVED = "archived"


class PipelineStepStatus(Enum):
    """Pipeline step execution status."""
    PENDING = "pending"
    READY = "ready"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    SKIPPED = "skipped"
    CANCELLED = "cancelled"


class PipelineStepType(Enum):
    """Pipeline step type classification."""
    DATA_INGESTION = "data_ingestion"
    DATA_VALIDATION = "data_validation"
    DATA_PREPROCESSING = "data_preprocessing"
    FEATURE_ENGINEERING = "feature_engineering"
    MODEL_TRAINING = "model_training"
    MODEL_VALIDATION = "model_validation"
    MODEL_EVALUATION = "model_evaluation"
    MODEL_REGISTRATION = "model_registration"
    DEPLOYMENT = "deployment"
    NOTIFICATION = "notification"
    CUSTOM = "custom"


class ScheduleType(Enum):
    """Pipeline schedule type."""
    MANUAL = "manual"
    CRON = "cron"
    INTERVAL = "interval"
    EVENT_DRIVEN = "event_driven"


@dataclass
class PipelineStep:
    """Pipeline step entity representing a single operation in a pipeline.
    
    Steps can have dependencies and form a Directed Acyclic Graph (DAG).
    """
    
    # Identity
    id: UUID = field(default_factory=uuid4)
    name: str = field()
    pipeline_id: UUID = field()
    
    # Step Configuration
    step_type: PipelineStepType = field()
    implementation: str = field()  # Class or function to execute
    parameters: Dict[str, Any] = field(default_factory=dict)
    environment_variables: Dict[str, str] = field(default_factory=dict)
    
    # Dependencies
    depends_on: List[UUID] = field(default_factory=list)  # Step IDs this step depends on
    
    # Execution
    status: PipelineStepStatus = field(default=PipelineStepStatus.PENDING)
    retry_count: int = field(default=0)
    max_retries: int = field(default=3)
    timeout_seconds: Optional[int] = field(default=None)
    
    # Execution History
    started_at: Optional[datetime] = field(default=None)
    completed_at: Optional[datetime] = field(default=None)
    duration_seconds: Optional[float] = field(default=None)
    
    # Results
    outputs: Dict[str, Any] = field(default_factory=dict)
    artifacts: Dict[str, str] = field(default_factory=dict)
    error_message: Optional[str] = field(default=None)
    logs: List[str] = field(default_factory=list)
    
    # Metadata
    description: Optional[str] = field(default=None)
    tags: List[str] = field(default_factory=list)
    
    def __post_init__(self):
        """Post-initialization validation."""
        if not self.name:
            raise ValueError("Pipeline step name cannot be empty")
        
        if not self.implementation:
            raise ValueError("Pipeline step must have an implementation")
        
        # Ensure tags are unique
        self.tags = list(set(self.tags))
    
    def add_dependency(self, step_id: UUID) -> None:
        """Add a dependency to this step."""
        if step_id not in self.depends_on:
            self.depends_on.append(step_id)
    
    def remove_dependency(self, step_id: UUID) -> None:
        """Remove a dependency from this step."""
        if step_id in self.depends_on:
            self.depends_on.remove(step_id)
    
    def start_execution(self) -> None:
        """Mark step as started."""
        self.status = PipelineStepStatus.RUNNING
        self.started_at = datetime.utcnow()
    
    def complete_execution(self, outputs: Optional[Dict[str, Any]] = None) -> None:
        """Mark step as completed successfully."""
        self.status = PipelineStepStatus.COMPLETED
        self.completed_at = datetime.utcnow()
        if self.started_at:
            self.duration_seconds = (self.completed_at - self.started_at).total_seconds()
        if outputs:
            self.outputs.update(outputs)
    
    def fail_execution(self, error_message: str) -> None:
        """Mark step as failed."""
        self.status = PipelineStepStatus.FAILED
        self.completed_at = datetime.utcnow()
        self.error_message = error_message
        if self.started_at:
            self.duration_seconds = (self.completed_at - self.started_at).total_seconds()
    
    def retry_execution(self) -> bool:
        """Attempt to retry step execution.
        
        Returns:
            True if retry is allowed, False if max retries exceeded
        """
        if self.retry_count >= self.max_retries:
            return False
        
        self.retry_count += 1
        self.status = PipelineStepStatus.READY
        self.error_message = None
        return True
    
    def can_execute(self, completed_steps: Set[UUID]) -> bool:
        """Check if step can execute based on dependencies.
        
        Args:
            completed_steps: Set of completed step IDs
            
        Returns:
            True if all dependencies are satisfied
        """
        return all(dep_id in completed_steps for dep_id in self.depends_on)
    
    @property
    def is_finished(self) -> bool:
        """Check if step has finished execution."""
        return self.status in [
            PipelineStepStatus.COMPLETED,
            PipelineStepStatus.FAILED,
            PipelineStepStatus.SKIPPED,
            PipelineStepStatus.CANCELLED
        ]


@dataclass 
class Pipeline:
    """Pipeline entity for orchestrating ML workflows.
    
    Represents a complete ML pipeline with steps organized as a DAG.
    """
    
    # Identity
    id: UUID = field(default_factory=uuid4)
    name: str = field()
    version: str = field(default="1.0.0")
    
    # Pipeline Configuration
    description: Optional[str] = field(default=None)
    pipeline_type: str = field(default="training")  # training, inference, retraining
    
    # Lifecycle
    status: PipelineStatus = field(default=PipelineStatus.DRAFT)
    created_at: datetime = field(default_factory=datetime.utcnow)
    updated_at: datetime = field(default_factory=datetime.utcnow)
    created_by: str = field()
    
    # Execution Configuration
    steps: List[PipelineStep] = field(default_factory=list)
    global_parameters: Dict[str, Any] = field(default_factory=dict)
    environment_config: Dict[str, Any] = field(default_factory=dict)
    
    # Scheduling
    schedule_type: ScheduleType = field(default=ScheduleType.MANUAL)
    cron_expression: Optional[str] = field(default=None)
    interval_minutes: Optional[int] = field(default=None)
    next_run_at: Optional[datetime] = field(default=None)
    
    # Execution History
    last_run_id: Optional[UUID] = field(default=None)
    last_run_at: Optional[datetime] = field(default=None)
    last_run_status: Optional[PipelineStatus] = field(default=None)
    execution_count: int = field(default=0)
    
    # Settings
    max_parallel_steps: int = field(default=5)
    default_timeout_seconds: int = field(default=3600)  # 1 hour
    retry_policy: Dict[str, Any] = field(default_factory=dict)
    
    # Metadata
    tags: List[str] = field(default_factory=list)
    labels: Dict[str, str] = field(default_factory=dict)
    
    def __post_init__(self):
        """Post-initialization validation."""
        if not self.name:
            raise ValueError("Pipeline name cannot be empty")
        
        if not self.created_by:
            raise ValueError("Pipeline must have a creator")
        
        # Ensure tags are unique
        self.tags = list(set(self.tags))
        
        # Validate schedule configuration
        if self.schedule_type == ScheduleType.CRON and not self.cron_expression:
            raise ValueError("Cron schedule requires cron_expression")
        
        if self.schedule_type == ScheduleType.INTERVAL and not self.interval_minutes:
            raise ValueError("Interval schedule requires interval_minutes")
    
    def add_step(self, step: PipelineStep) -> None:
        """Add a step to the pipeline."""
        step.pipeline_id = self.id
        self.steps.append(step)
        self.updated_at = datetime.utcnow()
    
    def remove_step(self, step_id: UUID) -> None:
        """Remove a step from the pipeline."""
        self.steps = [step for step in self.steps if step.id != step_id]
        
        # Remove dependencies on the removed step
        for step in self.steps:
            step.remove_dependency(step_id)
        
        self.updated_at = datetime.utcnow()
    
    def get_step(self, step_id: UUID) -> Optional[PipelineStep]:
        """Get a step by ID."""
        return next((step for step in self.steps if step.id == step_id), None)
    
    def validate_dag(self) -> List[str]:
        """Validate that steps form a valid DAG.
        
        Returns:
            List of validation errors
        """
        errors = []
        step_ids = {step.id for step in self.steps}
        
        # Check for invalid dependencies
        for step in self.steps:
            for dep_id in step.depends_on:
                if dep_id not in step_ids:
                    errors.append(f"Step {step.name} depends on non-existent step {dep_id}")
        
        # Check for cycles using DFS
        visited = set()
        rec_stack = set()
        
        def has_cycle(step_id: UUID) -> bool:
            visited.add(step_id)
            rec_stack.add(step_id)
            
            step = self.get_step(step_id)
            if step:
                for dep_id in step.depends_on:
                    if dep_id not in visited:
                        if has_cycle(dep_id):
                            return True
                    elif dep_id in rec_stack:
                        return True
            
            rec_stack.remove(step_id)
            return False
        
        for step in self.steps:
            if step.id not in visited:
                if has_cycle(step.id):
                    errors.append(f"Circular dependency detected involving step {step.name}")
                    break
        
        return errors
    
    def get_ready_steps(self, completed_steps: Set[UUID]) -> List[PipelineStep]:
        """Get steps that are ready to execute.
        
        Args:
            completed_steps: Set of completed step IDs
            
        Returns:
            List of steps ready for execution
        """
        ready_steps = []
        for step in self.steps:
            if (step.status == PipelineStepStatus.PENDING and 
                step.can_execute(completed_steps)):
                ready_steps.append(step)
        
        return ready_steps
    
    def activate(self) -> None:
        """Activate the pipeline for scheduling."""
        validation_errors = self.validate_dag()
        if validation_errors:
            raise ValueError(f"Cannot activate pipeline with validation errors: {validation_errors}")
        
        self.status = PipelineStatus.ACTIVE
        self.updated_at = datetime.utcnow()
        
        # Schedule next run if applicable
        if self.schedule_type == ScheduleType.INTERVAL and self.interval_minutes:
            self.next_run_at = datetime.utcnow() + timedelta(minutes=self.interval_minutes)
    
    def start_execution(self, run_id: UUID) -> None:
        """Start pipeline execution."""
        self.status = PipelineStatus.RUNNING
        self.last_run_id = run_id
        self.last_run_at = datetime.utcnow()
        self.execution_count += 1
        
        # Reset all steps to pending
        for step in self.steps:
            step.status = PipelineStepStatus.PENDING
            step.retry_count = 0
            step.error_message = None
    
    def complete_execution(self) -> None:
        """Mark pipeline execution as completed."""
        self.status = PipelineStatus.COMPLETED
        self.last_run_status = PipelineStatus.COMPLETED
        self.updated_at = datetime.utcnow()
        
        # Schedule next run if applicable
        if self.schedule_type == ScheduleType.INTERVAL and self.interval_minutes:
            self.next_run_at = datetime.utcnow() + timedelta(minutes=self.interval_minutes)
    
    def fail_execution(self, error_message: str) -> None:
        """Mark pipeline execution as failed."""
        self.status = PipelineStatus.FAILED
        self.last_run_status = PipelineStatus.FAILED
        self.updated_at = datetime.utcnow()
    
    def pause(self) -> None:
        """Pause the pipeline."""
        self.status = PipelineStatus.PAUSED
        self.updated_at = datetime.utcnow()
    
    def archive(self) -> None:
        """Archive the pipeline."""
        self.status = PipelineStatus.ARCHIVED
        self.updated_at = datetime.utcnow()
    
    @property
    def is_active(self) -> bool:
        """Check if pipeline is active."""
        return self.status == PipelineStatus.ACTIVE
    
    @property
    def is_running(self) -> bool:
        """Check if pipeline is currently running."""
        return self.status == PipelineStatus.RUNNING
    
    @property
    def step_count(self) -> int:
        """Get total number of steps."""
        return len(self.steps)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert pipeline to dictionary representation."""
        return {
            "id": str(self.id),
            "name": self.name,
            "version": self.version,
            "description": self.description,
            "pipeline_type": self.pipeline_type,
            "status": self.status.value,
            "created_at": self.created_at.isoformat(),
            "updated_at": self.updated_at.isoformat(),
            "created_by": self.created_by,
            "steps": [step.id for step in self.steps],
            "global_parameters": self.global_parameters,
            "environment_config": self.environment_config,
            "schedule_type": self.schedule_type.value,
            "cron_expression": self.cron_expression,
            "interval_minutes": self.interval_minutes,
            "next_run_at": self.next_run_at.isoformat() if self.next_run_at else None,
            "last_run_id": str(self.last_run_id) if self.last_run_id else None,
            "last_run_at": self.last_run_at.isoformat() if self.last_run_at else None,
            "last_run_status": self.last_run_status.value if self.last_run_status else None,
            "execution_count": self.execution_count,
            "max_parallel_steps": self.max_parallel_steps,
            "default_timeout_seconds": self.default_timeout_seconds,
            "retry_policy": self.retry_policy,
            "tags": self.tags,
            "labels": self.labels,
        }