"""Pipeline Workflow domain entity for workflow management and step orchestration."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional, Set
from uuid import UUID, uuid4


class WorkflowStatus(str, Enum):
    """Status of pipeline workflow."""
    
    DRAFT = "draft"
    ACTIVE = "active"
    RUNNING = "running"
    PAUSED = "paused"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"
    ARCHIVED = "archived"


class StepType(str, Enum):
    """Type of workflow step."""
    
    EXTRACT = "extract"
    TRANSFORM = "transform"
    LOAD = "load"
    VALIDATE = "validate"
    AGGREGATE = "aggregate"
    ENRICH = "enrich"
    FILTER = "filter"
    JOIN = "join"
    SPLIT = "split"
    MERGE = "merge"
    CUSTOM = "custom"
    CONDITIONAL = "conditional"
    PARALLEL = "parallel"
    SUBPROCESS = "subprocess"


class StepStatus(str, Enum):
    """Status of individual workflow step."""
    
    PENDING = "pending"
    READY = "ready"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    SKIPPED = "skipped"
    CANCELLED = "cancelled"
    RETRYING = "retrying"


class ExecutionMode(str, Enum):
    """Execution mode for workflow steps."""
    
    SEQUENTIAL = "sequential"
    PARALLEL = "parallel"
    CONDITIONAL = "conditional"
    LOOP = "loop"


@dataclass
class StepCondition:
    """Condition for conditional step execution."""
    
    id: UUID = field(default_factory=uuid4)
    expression: str = ""
    operator: str = "=="  # ==, !=, >, <, >=, <=, in, not_in
    value: Any = None
    source_step_id: Optional[UUID] = None
    source_field: Optional[str] = None
    
    def __post_init__(self) -> None:
        """Validate condition after initialization."""
        if not self.expression and not self.source_step_id:
            raise ValueError("Condition must have expression or source step")
        
        valid_operators = ["==", "!=", ">", "<", ">=", "<=", "in", "not_in", "exists", "not_exists"]
        if self.operator not in valid_operators:
            raise ValueError(f"Invalid operator: {self.operator}")
    
    def evaluate(self, context: Dict[str, Any]) -> bool:
        """Evaluate the condition against provided context."""
        if self.expression:
            # Simplified expression evaluation - would use proper parser in real implementation
            try:
                return eval(self.expression, {"__builtins__": {}}, context)
            except Exception:
                return False
        
        if self.source_step_id and self.source_field:
            source_value = context.get(f"step_{self.source_step_id}_{self.source_field}")
            
            if self.operator == "==":
                return source_value == self.value
            elif self.operator == "!=":
                return source_value != self.value
            elif self.operator == ">":
                return source_value > self.value
            elif self.operator == "<":
                return source_value < self.value
            elif self.operator == ">=":
                return source_value >= self.value
            elif self.operator == "<=":
                return source_value <= self.value
            elif self.operator == "in":
                return source_value in self.value if self.value else False
            elif self.operator == "not_in":
                return source_value not in self.value if self.value else True
            elif self.operator == "exists":
                return f"step_{self.source_step_id}_{self.source_field}" in context
            elif self.operator == "not_exists":
                return f"step_{self.source_step_id}_{self.source_field}" not in context
        
        return False
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert condition to dictionary."""
        return {
            "id": str(self.id),
            "expression": self.expression,
            "operator": self.operator,
            "value": self.value,
            "source_step_id": str(self.source_step_id) if self.source_step_id else None,
            "source_field": self.source_field,
        }


@dataclass
class WorkflowStep:
    """Individual step in a pipeline workflow."""
    
    id: UUID = field(default_factory=uuid4)
    name: str = ""
    description: str = ""
    step_type: StepType = StepType.TRANSFORM
    
    # Step configuration
    config: Dict[str, Any] = field(default_factory=dict)
    command: Optional[str] = None
    script_path: Optional[str] = None
    function_name: Optional[str] = None
    
    # Dependencies and execution
    depends_on: List[UUID] = field(default_factory=list)
    parallel_group: Optional[str] = None
    execution_mode: ExecutionMode = ExecutionMode.SEQUENTIAL
    
    # Conditions
    conditions: List[StepCondition] = field(default_factory=list)
    skip_on_failure: bool = False
    continue_on_failure: bool = False
    
    # Resource requirements
    cpu_limit: Optional[float] = None
    memory_limit_mb: Optional[int] = None
    timeout_seconds: int = 3600
    
    # Retry configuration
    max_retries: int = 0
    retry_delay_seconds: int = 60
    exponential_backoff: bool = False
    
    # Execution tracking
    status: StepStatus = StepStatus.PENDING
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    retry_count: int = 0
    error_message: Optional[str] = None
    
    # Input/Output data
    input_data: Dict[str, Any] = field(default_factory=dict)
    output_data: Dict[str, Any] = field(default_factory=dict)
    
    # Metadata
    created_at: datetime = field(default_factory=datetime.utcnow)
    updated_at: datetime = field(default_factory=datetime.utcnow)
    tags: List[str] = field(default_factory=list)
    
    def __post_init__(self) -> None:
        """Validate step after initialization."""
        if not self.name:
            raise ValueError("Step name cannot be empty")
        
        if len(self.name) > 100:
            raise ValueError("Step name cannot exceed 100 characters")
        
        if self.timeout_seconds <= 0:
            raise ValueError("Timeout must be positive")
        
        if self.max_retries < 0:
            raise ValueError("Max retries cannot be negative")
        
        if self.retry_delay_seconds < 0:
            raise ValueError("Retry delay cannot be negative")
    
    @property
    def is_completed(self) -> bool:
        """Check if step is completed."""
        return self.status in [
            StepStatus.COMPLETED,
            StepStatus.FAILED,
            StepStatus.SKIPPED,
            StepStatus.CANCELLED
        ]
    
    @property
    def is_successful(self) -> bool:
        """Check if step completed successfully."""
        return self.status == StepStatus.COMPLETED
    
    @property
    def duration_seconds(self) -> Optional[float]:
        """Get step execution duration in seconds."""
        if self.started_at and self.completed_at:
            return (self.completed_at - self.started_at).total_seconds()
        return None
    
    @property
    def can_retry(self) -> bool:
        """Check if step can be retried."""
        return (
            self.status == StepStatus.FAILED and
            self.retry_count < self.max_retries
        )
    
    def add_dependency(self, step_id: UUID) -> None:
        """Add a dependency on another step."""
        if step_id == self.id:
            raise ValueError("Step cannot depend on itself")
        
        if step_id not in self.depends_on:
            self.depends_on.append(step_id)
            self.updated_at = datetime.utcnow()
    
    def remove_dependency(self, step_id: UUID) -> None:
        """Remove a dependency."""
        if step_id in self.depends_on:
            self.depends_on.remove(step_id)
            self.updated_at = datetime.utcnow()
    
    def add_condition(self, condition: StepCondition) -> None:
        """Add a condition for step execution."""
        self.conditions.append(condition)
        self.updated_at = datetime.utcnow()
    
    def remove_condition(self, condition_id: UUID) -> bool:
        """Remove a condition by ID."""
        for i, condition in enumerate(self.conditions):
            if condition.id == condition_id:
                self.conditions.pop(i)
                self.updated_at = datetime.utcnow()
                return True
        return False
    
    def evaluate_conditions(self, context: Dict[str, Any]) -> bool:
        """Evaluate all conditions for the step."""
        if not self.conditions:
            return True
        
        # All conditions must be true for step to execute
        return all(condition.evaluate(context) for condition in self.conditions)
    
    def start(self) -> None:
        """Start step execution."""
        if self.status not in [StepStatus.PENDING, StepStatus.READY]:
            raise ValueError(f"Cannot start step in {self.status} status")
        
        self.status = StepStatus.RUNNING
        self.started_at = datetime.utcnow()
        self.updated_at = self.started_at
    
    def complete(self, output_data: Optional[Dict[str, Any]] = None) -> None:
        """Complete step execution successfully."""
        if self.status != StepStatus.RUNNING:
            raise ValueError(f"Cannot complete step in {self.status} status")
        
        self.status = StepStatus.COMPLETED
        self.completed_at = datetime.utcnow()
        self.updated_at = self.completed_at
        
        if output_data:
            self.output_data.update(output_data)
    
    def fail(self, error_message: str) -> None:
        """Mark step as failed."""
        self.status = StepStatus.FAILED
        self.completed_at = datetime.utcnow()
        self.updated_at = self.completed_at
        self.error_message = error_message
    
    def skip(self, reason: Optional[str] = None) -> None:
        """Skip step execution."""
        self.status = StepStatus.SKIPPED
        self.completed_at = datetime.utcnow()
        self.updated_at = self.completed_at
        if reason:
            self.error_message = f"Skipped: {reason}"
    
    def cancel(self) -> None:
        """Cancel step execution."""
        if self.status in [StepStatus.COMPLETED]:
            raise ValueError("Cannot cancel completed step")
        
        self.status = StepStatus.CANCELLED
        self.completed_at = datetime.utcnow()
        self.updated_at = self.completed_at
    
    def retry(self) -> None:
        """Retry failed step."""
        if not self.can_retry:
            raise ValueError("Step cannot be retried")
        
        self.retry_count += 1
        self.status = StepStatus.RETRYING
        self.started_at = None
        self.completed_at = None
        self.error_message = None
        self.updated_at = datetime.utcnow()
    
    def reset(self) -> None:
        """Reset step to pending state."""
        self.status = StepStatus.PENDING
        self.started_at = None
        self.completed_at = None
        self.retry_count = 0
        self.error_message = None
        self.input_data = {}
        self.output_data = {}
        self.updated_at = datetime.utcnow()
    
    def add_tag(self, tag: str) -> None:
        """Add a tag to the step."""
        if tag and tag not in self.tags:
            self.tags.append(tag)
            self.updated_at = datetime.utcnow()
    
    def remove_tag(self, tag: str) -> None:
        """Remove a tag from the step."""
        if tag in self.tags:
            self.tags.remove(tag)
            self.updated_at = datetime.utcnow()
    
    def has_tag(self, tag: str) -> bool:
        """Check if step has a specific tag."""
        return tag in self.tags
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert step to dictionary."""
        return {
            "id": str(self.id),
            "name": self.name,
            "description": self.description,
            "step_type": self.step_type.value,
            "config": self.config,
            "command": self.command,
            "script_path": self.script_path,
            "function_name": self.function_name,
            "depends_on": [str(dep_id) for dep_id in self.depends_on],
            "parallel_group": self.parallel_group,
            "execution_mode": self.execution_mode.value,
            "conditions": [c.to_dict() for c in self.conditions],
            "skip_on_failure": self.skip_on_failure,
            "continue_on_failure": self.continue_on_failure,
            "cpu_limit": self.cpu_limit,
            "memory_limit_mb": self.memory_limit_mb,
            "timeout_seconds": self.timeout_seconds,
            "max_retries": self.max_retries,
            "retry_delay_seconds": self.retry_delay_seconds,
            "exponential_backoff": self.exponential_backoff,
            "status": self.status.value,
            "started_at": self.started_at.isoformat() if self.started_at else None,
            "completed_at": self.completed_at.isoformat() if self.completed_at else None,
            "retry_count": self.retry_count,
            "error_message": self.error_message,
            "input_data": self.input_data,
            "output_data": self.output_data,
            "created_at": self.created_at.isoformat(),
            "updated_at": self.updated_at.isoformat(),
            "tags": self.tags,
            "is_completed": self.is_completed,
            "is_successful": self.is_successful,
            "duration_seconds": self.duration_seconds,
            "can_retry": self.can_retry,
        }


@dataclass
class PipelineWorkflow:
    """Pipeline workflow domain entity for managing complex data processing workflows."""
    
    id: UUID = field(default_factory=uuid4)
    name: str = ""
    description: str = ""
    version: str = "1.0.0"
    
    # Workflow configuration
    status: WorkflowStatus = WorkflowStatus.DRAFT
    execution_mode: ExecutionMode = ExecutionMode.SEQUENTIAL
    
    # Steps and dependencies
    steps: List[WorkflowStep] = field(default_factory=list)
    step_groups: Dict[str, List[UUID]] = field(default_factory=dict)
    
    # Execution settings
    max_parallel_steps: int = 5
    global_timeout_seconds: int = 14400  # 4 hours
    stop_on_failure: bool = True
    
    # Resource limits
    total_cpu_limit: Optional[float] = None
    total_memory_limit_mb: Optional[int] = None
    
    # Execution tracking
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    current_step_ids: Set[UUID] = field(default_factory=set)
    completed_steps: int = 0
    failed_steps: int = 0
    
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
        """Validate workflow after initialization."""
        if not self.name:
            raise ValueError("Workflow name cannot be empty")
        
        if len(self.name) > 100:
            raise ValueError("Workflow name cannot exceed 100 characters")
        
        if self.max_parallel_steps <= 0:
            raise ValueError("Max parallel steps must be positive")
        
        if self.global_timeout_seconds <= 0:
            raise ValueError("Global timeout must be positive")
    
    @property
    def total_steps(self) -> int:
        """Get total number of steps."""
        return len(self.steps)
    
    @property
    def is_running(self) -> bool:
        """Check if workflow is running."""
        return self.status == WorkflowStatus.RUNNING
    
    @property
    def is_completed(self) -> bool:
        """Check if workflow is completed."""
        return self.status in [
            WorkflowStatus.COMPLETED,
            WorkflowStatus.FAILED,
            WorkflowStatus.CANCELLED
        ]
    
    @property
    def success_rate(self) -> float:
        """Get step success rate."""
        if self.total_steps == 0:
            return 0.0
        successful_steps = sum(1 for step in self.steps if step.is_successful)
        return (successful_steps / self.total_steps) * 100
    
    @property
    def duration_seconds(self) -> Optional[float]:
        """Get workflow duration in seconds."""
        if self.started_at and self.completed_at:
            return (self.completed_at - self.started_at).total_seconds()
        return None
    
    @property
    def current_running_steps(self) -> int:
        """Get count of currently running steps."""
        return len(self.current_step_ids)
    
    def add_step(self, step: WorkflowStep) -> None:
        """Add a step to the workflow."""
        # Check for duplicate step names
        existing_names = [s.name for s in self.steps]
        if step.name in existing_names:
            raise ValueError(f"Step with name '{step.name}' already exists")
        
        self.steps.append(step)
        self.updated_at = datetime.utcnow()
    
    def remove_step(self, step_id: UUID) -> bool:
        """Remove a step from the workflow."""
        # Check if other steps depend on this one
        dependent_steps = [s for s in self.steps if step_id in s.depends_on]
        if dependent_steps:
            raise ValueError(f"Cannot remove step: {len(dependent_steps)} steps depend on it")
        
        for i, step in enumerate(self.steps):
            if step.id == step_id:
                self.steps.pop(i)
                self.updated_at = datetime.utcnow()
                return True
        return False
    
    def get_step(self, step_id: UUID) -> Optional[WorkflowStep]:
        """Get a step by ID."""
        for step in self.steps:
            if step.id == step_id:
                return step
        return None
    
    def get_step_by_name(self, name: str) -> Optional[WorkflowStep]:
        """Get a step by name."""
        for step in self.steps:
            if step.name == name:
                return step
        return None
    
    def get_runnable_steps(self) -> List[WorkflowStep]:
        """Get steps that are ready to run."""
        runnable = []
        
        for step in self.steps:
            if step.status not in [StepStatus.PENDING, StepStatus.READY]:
                continue
            
            # Check dependencies
            dependencies_satisfied = True
            for dep_id in step.depends_on:
                dep_step = self.get_step(dep_id)
                if not dep_step or not dep_step.is_successful:
                    dependencies_satisfied = False
                    break
            
            if dependencies_satisfied:
                # Check conditions
                context = self._build_execution_context()
                if step.evaluate_conditions(context):
                    runnable.append(step)
        
        return runnable
    
    def get_parallel_groups(self) -> Dict[str, List[WorkflowStep]]:
        """Get steps grouped by parallel execution groups."""
        groups = {}
        for step in self.steps:
            if step.parallel_group:
                if step.parallel_group not in groups:
                    groups[step.parallel_group] = []
                groups[step.parallel_group].append(step)
        return groups
    
    def validate_dependencies(self) -> List[str]:
        """Validate workflow dependencies for cycles and missing references."""
        issues = []
        step_ids = {step.id for step in self.steps}
        
        # Check for missing dependencies
        for step in self.steps:
            for dep_id in step.depends_on:
                if dep_id not in step_ids:
                    issues.append(f"Step '{step.name}' depends on non-existent step {dep_id}")
        
        # Check for circular dependencies using DFS
        def has_cycle(step_id: UUID, visiting: Set[UUID], visited: Set[UUID]) -> bool:
            if step_id in visiting:
                return True
            if step_id in visited:
                return False
            
            visiting.add(step_id)
            step = self.get_step(step_id)
            
            if step:
                for dep_id in step.depends_on:
                    if has_cycle(dep_id, visiting, visited):
                        return True
            
            visiting.remove(step_id)
            visited.add(step_id)
            return False
        
        visited = set()
        for step in self.steps:
            if step.id not in visited:
                if has_cycle(step.id, set(), visited):
                    issues.append(f"Circular dependency detected involving step '{step.name}'")
        
        return issues
    
    def _build_execution_context(self) -> Dict[str, Any]:
        """Build execution context for condition evaluation."""
        context = {}
        
        # Add step outputs to context
        for step in self.steps:
            if step.is_successful and step.output_data:
                for key, value in step.output_data.items():
                    context[f"step_{step.id}_{key}"] = value
        
        # Add environment variables
        context.update(self.environment_vars)
        
        # Add workflow metadata
        context.update({
            "workflow_id": str(self.id),
            "workflow_name": self.name,
            "total_steps": self.total_steps,
            "completed_steps": self.completed_steps,
        })
        
        return context
    
    def start(self) -> None:
        """Start workflow execution."""
        if self.status != WorkflowStatus.ACTIVE:
            raise ValueError(f"Cannot start workflow in {self.status} status")
        
        # Validate workflow before starting
        validation_issues = self.validate_dependencies()
        if validation_issues:
            raise ValueError(f"Workflow validation failed: {'; '.join(validation_issues)}")
        
        self.status = WorkflowStatus.RUNNING
        self.started_at = datetime.utcnow()
        self.updated_at = self.started_at
        
        # Reset step counters
        self.completed_steps = 0
        self.failed_steps = 0
        self.current_step_ids.clear()
        
        # Reset all steps to pending
        for step in self.steps:
            step.reset()
    
    def complete(self) -> None:
        """Complete workflow execution."""
        if self.status != WorkflowStatus.RUNNING:
            raise ValueError(f"Cannot complete workflow in {self.status} status")
        
        self.status = WorkflowStatus.COMPLETED
        self.completed_at = datetime.utcnow()
        self.updated_at = self.completed_at
        self.current_step_ids.clear()
    
    def fail(self, error_message: Optional[str] = None) -> None:
        """Mark workflow as failed."""
        self.status = WorkflowStatus.FAILED
        self.completed_at = datetime.utcnow()
        self.updated_at = self.completed_at
        self.current_step_ids.clear()
        
        if error_message:
            self.config["error_message"] = error_message
    
    def cancel(self) -> None:
        """Cancel workflow execution."""
        if self.status not in [WorkflowStatus.RUNNING, WorkflowStatus.PAUSED]:
            raise ValueError(f"Cannot cancel workflow in {self.status} status")
        
        self.status = WorkflowStatus.CANCELLED
        self.completed_at = datetime.utcnow()
        self.updated_at = self.completed_at
        self.current_step_ids.clear()
        
        # Cancel running steps
        for step in self.steps:
            if step.status == StepStatus.RUNNING:
                step.cancel()
    
    def pause(self) -> None:
        """Pause workflow execution."""
        if self.status != WorkflowStatus.RUNNING:
            raise ValueError(f"Cannot pause workflow in {self.status} status")
        
        self.status = WorkflowStatus.PAUSED
        self.updated_at = datetime.utcnow()
    
    def resume(self) -> None:
        """Resume workflow execution."""
        if self.status != WorkflowStatus.PAUSED:
            raise ValueError(f"Cannot resume workflow in {self.status} status")
        
        self.status = WorkflowStatus.RUNNING
        self.updated_at = datetime.utcnow()
    
    def activate(self) -> None:
        """Activate the workflow."""
        if self.status != WorkflowStatus.DRAFT:
            raise ValueError(f"Cannot activate workflow in {self.status} status")
        
        # Validate workflow before activation
        validation_issues = self.validate_dependencies()
        if validation_issues:
            raise ValueError(f"Cannot activate workflow with validation issues: {'; '.join(validation_issues)}")
        
        self.status = WorkflowStatus.ACTIVE
        self.updated_at = datetime.utcnow()
    
    def archive(self) -> None:
        """Archive the workflow."""
        if self.status == WorkflowStatus.RUNNING:
            raise ValueError("Cannot archive running workflow")
        
        self.status = WorkflowStatus.ARCHIVED
        self.updated_at = datetime.utcnow()
    
    def step_started(self, step_id: UUID) -> None:
        """Record that a step has started."""
        self.current_step_ids.add(step_id)
        self.updated_at = datetime.utcnow()
    
    def step_completed(self, step_id: UUID, success: bool) -> None:
        """Record that a step has completed."""
        self.current_step_ids.discard(step_id)
        
        if success:
            self.completed_steps += 1
        else:
            self.failed_steps += 1
        
        self.updated_at = datetime.utcnow()
        
        # Check if workflow should complete or fail
        if self.failed_steps > 0 and self.stop_on_failure:
            self.fail("Workflow failed due to step failure")
        elif self.completed_steps + self.failed_steps == self.total_steps:
            if self.failed_steps == 0:
                self.complete()
            else:
                self.fail("Workflow completed with failures")
    
    def add_tag(self, tag: str) -> None:
        """Add a tag to the workflow."""
        if tag and tag not in self.tags:
            self.tags.append(tag)
            self.updated_at = datetime.utcnow()
    
    def remove_tag(self, tag: str) -> None:
        """Remove a tag from the workflow."""
        if tag in self.tags:
            self.tags.remove(tag)
            self.updated_at = datetime.utcnow()
    
    def has_tag(self, tag: str) -> bool:
        """Check if workflow has a specific tag."""
        return tag in self.tags
    
    def update_config(self, key: str, value: Any) -> None:
        """Update configuration key-value pair."""
        self.config[key] = value
        self.updated_at = datetime.utcnow()
    
    def get_workflow_summary(self) -> Dict[str, Any]:
        """Get workflow summary information."""
        runnable_steps = len(self.get_runnable_steps())
        
        return {
            "id": str(self.id),
            "name": self.name,
            "status": self.status.value,
            "total_steps": self.total_steps,
            "completed_steps": self.completed_steps,
            "failed_steps": self.failed_steps,
            "running_steps": self.current_running_steps,
            "runnable_steps": runnable_steps,
            "success_rate": self.success_rate,
            "duration_seconds": self.duration_seconds,
            "is_running": self.is_running,
            "is_completed": self.is_completed,
        }
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert workflow to dictionary."""
        return {
            "id": str(self.id),
            "name": self.name,
            "description": self.description,
            "version": self.version,
            "status": self.status.value,
            "execution_mode": self.execution_mode.value,
            "steps": [step.to_dict() for step in self.steps],
            "step_groups": {k: [str(sid) for sid in v] for k, v in self.step_groups.items()},
            "max_parallel_steps": self.max_parallel_steps,
            "global_timeout_seconds": self.global_timeout_seconds,
            "stop_on_failure": self.stop_on_failure,
            "total_cpu_limit": self.total_cpu_limit,
            "total_memory_limit_mb": self.total_memory_limit_mb,
            "started_at": self.started_at.isoformat() if self.started_at else None,
            "completed_at": self.completed_at.isoformat() if self.completed_at else None,
            "current_step_ids": [str(sid) for sid in self.current_step_ids],
            "completed_steps": self.completed_steps,
            "failed_steps": self.failed_steps,
            "created_at": self.created_at.isoformat(),
            "created_by": self.created_by,
            "updated_at": self.updated_at.isoformat(),
            "updated_by": self.updated_by,
            "config": self.config,
            "environment_vars": self.environment_vars,
            "tags": self.tags,
            "total_steps": self.total_steps,
            "is_running": self.is_running,
            "is_completed": self.is_completed,
            "success_rate": self.success_rate,
            "duration_seconds": self.duration_seconds,
            "current_running_steps": self.current_running_steps,
        }
    
    def __str__(self) -> str:
        """Human-readable string representation."""
        return f"PipelineWorkflow('{self.name}', {self.status.value}, steps={self.total_steps})"
    
    def __repr__(self) -> str:
        """Developer string representation."""
        return (
            f"PipelineWorkflow(id={self.id}, name='{self.name}', "
            f"status={self.status.value}, steps={self.total_steps})"
        )