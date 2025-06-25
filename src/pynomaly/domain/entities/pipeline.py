"""Pipeline entity for representing ML pipelines and workflows."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional, Union
from uuid import UUID, uuid4


class PipelineType(Enum):
    """Type of ML pipeline."""
    TRAINING = "training"
    INFERENCE = "inference"
    BATCH_PROCESSING = "batch_processing"
    STREAMING = "streaming"
    RETRAINING = "retraining"
    VALIDATION = "validation"
    ETL = "etl"
    MONITORING = "monitoring"


class PipelineStatus(Enum):
    """Status of a pipeline."""
    DRAFT = "draft"
    ACTIVE = "active"
    PAUSED = "paused"
    FAILED = "failed"
    DEPRECATED = "deprecated"
    ARCHIVED = "archived"


class StepType(Enum):
    """Type of pipeline step."""
    DATA_LOADING = "data_loading"
    DATA_VALIDATION = "data_validation"
    DATA_PREPROCESSING = "data_preprocessing"
    FEATURE_ENGINEERING = "feature_engineering"
    MODEL_TRAINING = "model_training"
    MODEL_VALIDATION = "model_validation"
    MODEL_DEPLOYMENT = "model_deployment"
    PREDICTION = "prediction"
    POST_PROCESSING = "post_processing"
    MONITORING = "monitoring"
    ALERTING = "alerting"
    CUSTOM = "custom"


@dataclass
class PipelineStep:
    """Represents a single step in a pipeline."""
    
    id: UUID = field(default_factory=uuid4)
    name: str = ""
    step_type: StepType = StepType.CUSTOM
    description: str = ""
    order: int = 0
    is_enabled: bool = True
    configuration: Dict[str, Any] = field(default_factory=dict)
    inputs: List[str] = field(default_factory=list)  # Names of input artifacts
    outputs: List[str] = field(default_factory=list)  # Names of output artifacts
    dependencies: List[UUID] = field(default_factory=list)  # Step IDs this depends on
    timeout_seconds: Optional[int] = None
    retry_count: int = 0
    retry_delay_seconds: int = 60
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def __post_init__(self) -> None:
        """Validate step after initialization."""
        if not self.name:
            raise ValueError("Step name cannot be empty")
        
        if not isinstance(self.step_type, StepType):
            raise TypeError(f"Step type must be StepType, got {type(self.step_type)}")
        
        if self.order < 0:
            raise ValueError("Step order must be non-negative")
    
    @property
    def has_dependencies(self) -> bool:
        """Check if step has dependencies."""
        return len(self.dependencies) > 0
    
    @property
    def is_ready(self) -> bool:
        """Check if step is ready to execute (enabled and has name)."""
        return self.is_enabled and bool(self.name)
    
    def add_dependency(self, step_id: UUID) -> None:
        """Add a dependency to this step."""
        if step_id not in self.dependencies:
            self.dependencies.append(step_id)
    
    def remove_dependency(self, step_id: UUID) -> None:
        """Remove a dependency from this step."""
        if step_id in self.dependencies:
            self.dependencies.remove(step_id)
    
    def update_configuration(self, config: Dict[str, Any]) -> None:
        """Update step configuration."""
        self.configuration.update(config)
        self.metadata["config_updated_at"] = datetime.utcnow().isoformat()
    
    def get_info(self) -> Dict[str, Any]:
        """Get comprehensive information about the step."""
        return {
            "id": str(self.id),
            "name": self.name,
            "step_type": self.step_type.value,
            "description": self.description,
            "order": self.order,
            "is_enabled": self.is_enabled,
            "is_ready": self.is_ready,
            "inputs": self.inputs.copy(),
            "outputs": self.outputs.copy(),
            "dependencies": [str(d) for d in self.dependencies],
            "timeout_seconds": self.timeout_seconds,
            "retry_count": self.retry_count,
            "configuration": self.configuration.copy(),
            "metadata": self.metadata.copy(),
        }


@dataclass
class PipelineRun:
    """Represents an execution of a pipeline."""
    
    id: UUID = field(default_factory=uuid4)
    pipeline_id: UUID = field(default=uuid4)
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    status: str = "pending"
    triggered_by: str = ""
    trigger_reason: str = ""
    step_results: Dict[str, Dict[str, Any]] = field(default_factory=dict)  # step_id -> result
    artifacts: Dict[str, str] = field(default_factory=dict)  # artifact_name -> path
    error_message: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    @property
    def duration_seconds(self) -> Optional[float]:
        """Get run duration in seconds."""
        if self.started_at and self.completed_at:
            return (self.completed_at - self.started_at).total_seconds()
        return None
    
    @property
    def is_running(self) -> bool:
        """Check if run is currently executing."""
        return self.status == "running"
    
    @property
    def is_completed(self) -> bool:
        """Check if run completed successfully."""
        return self.status == "completed"
    
    @property
    def is_failed(self) -> bool:
        """Check if run failed."""
        return self.status == "failed"
    
    def start(self, triggered_by: str, reason: str = "") -> None:
        """Start the pipeline run."""
        self.started_at = datetime.utcnow()
        self.status = "running"
        self.triggered_by = triggered_by
        self.trigger_reason = reason
    
    def complete(self) -> None:
        """Mark run as completed."""
        self.completed_at = datetime.utcnow()
        self.status = "completed"
    
    def fail(self, error_message: str) -> None:
        """Mark run as failed."""
        self.completed_at = datetime.utcnow()
        self.status = "failed"
        self.error_message = error_message
    
    def add_step_result(self, step_id: str, result: Dict[str, Any]) -> None:
        """Add result for a step."""
        self.step_results[step_id] = result
    
    def get_step_result(self, step_id: str) -> Optional[Dict[str, Any]]:
        """Get result for a specific step."""
        return self.step_results.get(step_id)


@dataclass
class Pipeline:
    """Represents an ML pipeline for automated workflows.
    
    A Pipeline defines a sequence of steps that automate machine learning
    workflows, from data loading to model deployment and monitoring.
    It can be executed on schedules or triggered by events.
    
    Attributes:
        id: Unique identifier for the pipeline
        name: Human-readable name for the pipeline
        description: Detailed description of the pipeline's purpose
        pipeline_type: Type of pipeline (training, inference, etc.)
        created_at: When the pipeline was created
        created_by: User who created the pipeline
        status: Current status of the pipeline
        steps: List of pipeline steps in execution order
        schedule: Cron-style schedule for automatic execution
        triggers: Event triggers that can start the pipeline
        environment: Target environment (dev, staging, prod)
        tags: Semantic tags for organization
        metadata: Additional metadata and configuration
        model_id: Associated model (if applicable)
        datasets: Dataset IDs used by this pipeline
        version: Pipeline version for change tracking
    """
    
    name: str
    description: str
    pipeline_type: PipelineType
    created_by: str
    id: UUID = field(default_factory=uuid4)
    created_at: datetime = field(default_factory=datetime.utcnow)
    status: PipelineStatus = PipelineStatus.DRAFT
    steps: List[PipelineStep] = field(default_factory=list)
    schedule: Optional[str] = None  # Cron expression
    triggers: List[Dict[str, Any]] = field(default_factory=list)
    environment: str = "development"
    tags: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)
    model_id: Optional[UUID] = None
    datasets: List[UUID] = field(default_factory=list)
    version: str = "1.0.0"
    
    def __post_init__(self) -> None:
        """Validate pipeline after initialization."""
        if not self.name:
            raise ValueError("Pipeline name cannot be empty")
        
        if not self.description:
            raise ValueError("Pipeline description cannot be empty")
        
        if not isinstance(self.pipeline_type, PipelineType):
            raise TypeError(f"Pipeline type must be PipelineType, got {type(self.pipeline_type)}")
        
        if not isinstance(self.status, PipelineStatus):
            raise TypeError(f"Status must be PipelineStatus, got {type(self.status)}")
        
        if not self.created_by:
            raise ValueError("Created by cannot be empty")
        
        if not self.environment:
            raise ValueError("Environment cannot be empty")
    
    @property
    def is_active(self) -> bool:
        """Check if pipeline is active."""
        return self.status == PipelineStatus.ACTIVE
    
    @property
    def is_scheduled(self) -> bool:
        """Check if pipeline has a schedule."""
        return self.schedule is not None
    
    @property
    def step_count(self) -> int:
        """Get number of steps in pipeline."""
        return len(self.steps)
    
    @property
    def enabled_steps(self) -> List[PipelineStep]:
        """Get list of enabled steps."""
        return [step for step in self.steps if step.is_enabled]
    
    @property
    def execution_order(self) -> List[PipelineStep]:
        """Get steps in execution order."""
        return sorted(self.steps, key=lambda s: s.order)
    
    def add_step(self, step: PipelineStep) -> None:
        """Add a step to the pipeline."""
        # Check for duplicate step IDs
        if step.id in [s.id for s in self.steps]:
            raise ValueError(f"Step {step.id} already exists in pipeline")
        
        # Check for duplicate step names
        if step.name in [s.name for s in self.steps]:
            raise ValueError(f"Step with name '{step.name}' already exists")
        
        # If no order specified, put at end
        if step.order == 0:
            step.order = max([s.order for s in self.steps], default=0) + 1
        
        self.steps.append(step)
        self.metadata["last_step_added"] = datetime.utcnow().isoformat()
    
    def remove_step(self, step_id: UUID) -> bool:
        """Remove a step from the pipeline."""
        for i, step in enumerate(self.steps):
            if step.id == step_id:
                removed_step = self.steps.pop(i)
                
                # Remove dependencies on this step from other steps
                for other_step in self.steps:
                    other_step.remove_dependency(step_id)
                
                self.metadata["last_step_removed"] = datetime.utcnow().isoformat()
                self.metadata["removed_step_name"] = removed_step.name
                return True
        return False
    
    def get_step(self, step_id: UUID) -> Optional[PipelineStep]:
        """Get a step by ID."""
        for step in self.steps:
            if step.id == step_id:
                return step
        return None
    
    def get_step_by_name(self, step_name: str) -> Optional[PipelineStep]:
        """Get a step by name."""
        for step in self.steps:
            if step.name == step_name:
                return step
        return None
    
    def reorder_steps(self) -> None:
        """Reorder steps based on dependencies."""
        # Topological sort based on dependencies
        sorted_steps = []
        remaining_steps = self.steps.copy()
        
        while remaining_steps:
            # Find steps with no unresolved dependencies
            ready_steps = []
            for step in remaining_steps:
                if not step.dependencies or all(
                    dep_id in [s.id for s in sorted_steps] 
                    for dep_id in step.dependencies
                ):
                    ready_steps.append(step)
            
            if not ready_steps:
                # Circular dependency detected
                raise ValueError("Circular dependency detected in pipeline steps")
            
            # Sort ready steps by current order, then add to sorted list
            ready_steps.sort(key=lambda s: s.order)
            sorted_steps.extend(ready_steps)
            
            # Remove from remaining
            for step in ready_steps:
                remaining_steps.remove(step)
        
        # Update orders
        for i, step in enumerate(sorted_steps):
            step.order = i + 1
        
        self.steps = sorted_steps
        self.metadata["last_reordered"] = datetime.utcnow().isoformat()
    
    def validate_pipeline(self) -> tuple[bool, List[str]]:
        """Validate pipeline configuration."""
        issues = []
        
        # Check for enabled steps
        if not self.enabled_steps:
            issues.append("No enabled steps in pipeline")
        
        # Check for circular dependencies
        try:
            self.reorder_steps()
        except ValueError as e:
            issues.append(str(e))
        
        # Check step dependencies exist
        step_ids = {step.id for step in self.steps}
        for step in self.steps:
            for dep_id in step.dependencies:
                if dep_id not in step_ids:
                    issues.append(f"Step '{step.name}' depends on non-existent step {dep_id}")
        
        # Check for required configurations based on pipeline type
        if self.pipeline_type == PipelineType.TRAINING:
            training_steps = [s for s in self.steps if s.step_type == StepType.MODEL_TRAINING]
            if not training_steps:
                issues.append("Training pipeline must have at least one model training step")
        
        if self.pipeline_type == PipelineType.INFERENCE:
            prediction_steps = [s for s in self.steps if s.step_type == StepType.PREDICTION]
            if not prediction_steps:
                issues.append("Inference pipeline must have at least one prediction step")
        
        return len(issues) == 0, issues
    
    def activate(self) -> None:
        """Activate the pipeline."""
        is_valid, issues = self.validate_pipeline()
        if not is_valid:
            raise ValueError(f"Cannot activate invalid pipeline: {'; '.join(issues)}")
        
        self.status = PipelineStatus.ACTIVE
        self.metadata["activated_at"] = datetime.utcnow().isoformat()
    
    def pause(self) -> None:
        """Pause the pipeline."""
        self.status = PipelineStatus.PAUSED
        self.metadata["paused_at"] = datetime.utcnow().isoformat()
    
    def deprecate(self, reason: str = "") -> None:
        """Deprecate the pipeline."""
        self.status = PipelineStatus.DEPRECATED
        self.metadata["deprecated_at"] = datetime.utcnow().isoformat()
        if reason:
            self.metadata["deprecation_reason"] = reason
    
    def add_trigger(self, trigger_type: str, configuration: Dict[str, Any]) -> None:
        """Add an event trigger to the pipeline."""
        trigger = {
            "id": str(uuid4()),
            "type": trigger_type,
            "configuration": configuration,
            "created_at": datetime.utcnow().isoformat()
        }
        self.triggers.append(trigger)
    
    def remove_trigger(self, trigger_id: str) -> bool:
        """Remove a trigger from the pipeline."""
        for i, trigger in enumerate(self.triggers):
            if trigger["id"] == trigger_id:
                self.triggers.pop(i)
                return True
        return False
    
    def set_schedule(self, cron_expression: str) -> None:
        """Set a cron schedule for the pipeline."""
        # Basic validation of cron expression (5 or 6 fields)
        cron_parts = cron_expression.split()
        if len(cron_parts) not in [5, 6]:
            raise ValueError("Invalid cron expression")
        
        self.schedule = cron_expression
        self.metadata["schedule_set_at"] = datetime.utcnow().isoformat()
    
    def clear_schedule(self) -> None:
        """Clear the pipeline schedule."""
        self.schedule = None
        self.metadata["schedule_cleared_at"] = datetime.utcnow().isoformat()
    
    def add_tag(self, tag: str) -> None:
        """Add a tag to the pipeline."""
        if tag and tag not in self.tags:
            self.tags.append(tag)
    
    def remove_tag(self, tag: str) -> None:
        """Remove a tag from the pipeline."""
        if tag in self.tags:
            self.tags.remove(tag)
    
    def increment_version(self, version_type: str = "patch") -> None:
        """Increment pipeline version."""
        parts = self.version.split(".")
        if len(parts) != 3:
            self.version = "1.0.0"
            return
        
        major, minor, patch = map(int, parts)
        
        if version_type == "major":
            major += 1
            minor = 0
            patch = 0
        elif version_type == "minor":
            minor += 1
            patch = 0
        else:  # patch
            patch += 1
        
        self.version = f"{major}.{minor}.{patch}"
        self.metadata["version_updated_at"] = datetime.utcnow().isoformat()
    
    def get_info(self) -> Dict[str, Any]:
        """Get comprehensive information about the pipeline."""
        return {
            "id": str(self.id),
            "name": self.name,
            "description": self.description,
            "pipeline_type": self.pipeline_type.value,
            "created_at": self.created_at.isoformat(),
            "created_by": self.created_by,
            "status": self.status.value,
            "version": self.version,
            "environment": self.environment,
            "step_count": self.step_count,
            "enabled_steps": len(self.enabled_steps),
            "is_scheduled": self.is_scheduled,
            "schedule": self.schedule,
            "triggers": len(self.triggers),
            "model_id": str(self.model_id) if self.model_id else None,
            "datasets": [str(d) for d in self.datasets],
            "tags": self.tags.copy(),
            "metadata": self.metadata.copy(),
        }
    
    def __str__(self) -> str:
        """Human-readable representation."""
        return (
            f"Pipeline('{self.name}', {self.pipeline_type.value}, "
            f"v{self.version}, steps={self.step_count})"
        )
    
    def __repr__(self) -> str:
        """Developer representation."""
        return (
            f"Pipeline(id={self.id}, name='{self.name}', "
            f"type={self.pipeline_type.value}, status={self.status.value})"
        )