"""Data Pipeline domain entity for data engineering."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional
from uuid import UUID, uuid4


class PipelineStatus(str, Enum):
    """Pipeline execution status."""
    
    CREATED = "created"
    QUEUED = "queued"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"
    PAUSED = "paused"


class StepStatus(str, Enum):
    """Individual step status."""
    
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    SKIPPED = "skipped"


@dataclass
class PipelineStep:
    """Individual pipeline step."""
    
    id: UUID = field(default_factory=uuid4)
    name: str = ""
    step_type: str = ""
    config: Dict[str, Any] = field(default_factory=dict)
    dependencies: List[UUID] = field(default_factory=list)
    status: StepStatus = StepStatus.PENDING
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    error_message: Optional[str] = None
    output_data: Dict[str, Any] = field(default_factory=dict)
    
    def __post_init__(self) -> None:
        """Validate step after initialization."""
        if not self.name:
            raise ValueError("Step name cannot be empty")
        if not self.step_type:
            raise ValueError("Step type cannot be empty")
    
    @property
    def duration(self) -> Optional[float]:
        """Get step duration in seconds."""
        if self.started_at and self.completed_at:
            return (self.completed_at - self.started_at).total_seconds()
        return None
    
    @property
    def is_completed(self) -> bool:
        """Check if step is completed."""
        return self.status in [StepStatus.COMPLETED, StepStatus.FAILED, StepStatus.SKIPPED]
    
    def start(self) -> None:
        """Start the step execution."""
        if self.status != StepStatus.PENDING:
            raise ValueError(f"Cannot start step in {self.status} status")
        
        self.status = StepStatus.RUNNING
        self.started_at = datetime.utcnow()
    
    def complete(self, output_data: Optional[Dict[str, Any]] = None) -> None:
        """Complete the step successfully."""
        if self.status != StepStatus.RUNNING:
            raise ValueError(f"Cannot complete step in {self.status} status")
        
        self.status = StepStatus.COMPLETED
        self.completed_at = datetime.utcnow()
        
        if output_data:
            self.output_data.update(output_data)
    
    def fail(self, error_message: str) -> None:
        """Mark step as failed."""
        self.status = StepStatus.FAILED
        self.completed_at = datetime.utcnow()
        self.error_message = error_message
    
    def skip(self, reason: str) -> None:
        """Skip the step."""
        self.status = StepStatus.SKIPPED
        self.completed_at = datetime.utcnow()
        self.error_message = f"Skipped: {reason}"


@dataclass  
class DataPipeline:
    """Data pipeline domain entity representing an ETL/ELT workflow."""
    
    id: UUID = field(default_factory=uuid4)
    name: str = ""
    description: str = ""
    version: str = "1.0.0"
    status: PipelineStatus = PipelineStatus.CREATED
    steps: List[PipelineStep] = field(default_factory=list)
    
    # Scheduling and execution
    schedule_cron: Optional[str] = None
    max_retry_attempts: int = 3
    timeout_seconds: Optional[int] = None
    
    # Metadata
    created_at: datetime = field(default_factory=datetime.utcnow)
    created_by: str = ""
    updated_at: datetime = field(default_factory=datetime.utcnow) 
    updated_by: str = ""
    
    # Execution tracking
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    last_run_at: Optional[datetime] = None
    next_run_at: Optional[datetime] = None
    retry_count: int = 0
    
    # Configuration
    config: Dict[str, Any] = field(default_factory=dict)
    environment_vars: Dict[str, str] = field(default_factory=dict)
    tags: List[str] = field(default_factory=list)
    
    def __post_init__(self) -> None:
        """Validate pipeline after initialization."""
        if not self.name:
            raise ValueError("Pipeline name cannot be empty")
        
        if len(self.name) > 100:
            raise ValueError("Pipeline name cannot exceed 100 characters")
        
        if self.max_retry_attempts < 0:
            raise ValueError("Max retry attempts cannot be negative")
        
        if self.timeout_seconds is not None and self.timeout_seconds <= 0:
            raise ValueError("Timeout must be positive")
    
    @property
    def duration(self) -> Optional[float]:
        """Get pipeline duration in seconds."""
        if self.started_at and self.completed_at:
            return (self.completed_at - self.started_at).total_seconds()
        return None
    
    @property
    def is_running(self) -> bool:
        """Check if pipeline is currently running."""
        return self.status == PipelineStatus.RUNNING
    
    @property
    def is_completed(self) -> bool:
        """Check if pipeline is completed (success or failure)."""
        return self.status in [
            PipelineStatus.COMPLETED,
            PipelineStatus.FAILED,
            PipelineStatus.CANCELLED
        ]
    
    @property
    def success_rate(self) -> float:
        """Calculate step success rate."""
        if not self.steps:
            return 0.0
        
        completed_steps = sum(1 for step in self.steps if step.status == StepStatus.COMPLETED)
        return completed_steps / len(self.steps)
    
    def add_step(self, step: PipelineStep) -> None:
        """Add a step to the pipeline."""
        # Check for duplicate step names
        if any(s.name == step.name for s in self.steps):
            raise ValueError(f"Step with name '{step.name}' already exists")
        
        self.steps.append(step)
        self.updated_at = datetime.utcnow()
    
    def remove_step(self, step_id: UUID) -> bool:
        """Remove a step from the pipeline."""
        for i, step in enumerate(self.steps):
            if step.id == step_id:
                # Check if other steps depend on this one
                dependent_steps = [s for s in self.steps if step_id in s.dependencies]
                if dependent_steps:
                    raise ValueError(f"Cannot remove step: {len(dependent_steps)} steps depend on it")
                
                self.steps.pop(i)
                self.updated_at = datetime.utcnow()
                return True
        return False
    
    def get_step(self, step_id: UUID) -> Optional[PipelineStep]:
        """Get a step by ID."""
        for step in self.steps:
            if step.id == step_id:
                return step
        return None
    
    def get_step_by_name(self, name: str) -> Optional[PipelineStep]:
        """Get a step by name."""
        for step in self.steps:
            if step.name == name:
                return step
        return None
    
    def start(self) -> None:
        """Start pipeline execution."""
        if self.status not in [PipelineStatus.CREATED, PipelineStatus.QUEUED]:
            raise ValueError(f"Cannot start pipeline in {self.status} status")
        
        self.status = PipelineStatus.RUNNING
        self.started_at = datetime.utcnow()
        self.updated_at = self.started_at
        self.last_run_at = self.started_at
    
    def complete(self) -> None:
        """Complete pipeline execution successfully."""
        if self.status != PipelineStatus.RUNNING:
            raise ValueError(f"Cannot complete pipeline in {self.status} status")
        
        self.status = PipelineStatus.COMPLETED
        self.completed_at = datetime.utcnow()
        self.updated_at = self.completed_at
    
    def fail(self, error_message: Optional[str] = None) -> None:
        """Mark pipeline as failed."""
        self.status = PipelineStatus.FAILED
        self.completed_at = datetime.utcnow()
        self.updated_at = self.completed_at
        
        if error_message:
            self.config["last_error"] = error_message
    
    def cancel(self) -> None:
        """Cancel pipeline execution."""
        if self.status not in [PipelineStatus.QUEUED, PipelineStatus.RUNNING, PipelineStatus.PAUSED]:
            raise ValueError(f"Cannot cancel pipeline in {self.status} status")
        
        self.status = PipelineStatus.CANCELLED
        self.completed_at = datetime.utcnow()
        self.updated_at = self.completed_at
    
    def pause(self) -> None:
        """Pause pipeline execution."""
        if self.status != PipelineStatus.RUNNING:
            raise ValueError(f"Cannot pause pipeline in {self.status} status")
        
        self.status = PipelineStatus.PAUSED
        self.updated_at = datetime.utcnow()
    
    def resume(self) -> None:
        """Resume pipeline execution."""
        if self.status != PipelineStatus.PAUSED:
            raise ValueError(f"Cannot resume pipeline in {self.status} status")
        
        self.status = PipelineStatus.RUNNING
        self.updated_at = datetime.utcnow()
    
    def can_retry(self) -> bool:
        """Check if pipeline can be retried."""
        return (
            self.status == PipelineStatus.FAILED and
            self.retry_count < self.max_retry_attempts
        )
    
    def retry(self) -> None:
        """Retry failed pipeline."""
        if not self.can_retry():
            raise ValueError("Pipeline cannot be retried")
        
        self.retry_count += 1
        self.status = PipelineStatus.QUEUED
        self.started_at = None
        self.completed_at = None
        self.updated_at = datetime.utcnow()
        
        # Reset step statuses
        for step in self.steps:
            if step.status == StepStatus.FAILED:
                step.status = StepStatus.PENDING
                step.started_at = None
                step.completed_at = None
                step.error_message = None
    
    def add_tag(self, tag: str) -> None:
        """Add a tag to the pipeline."""
        if tag and tag not in self.tags:
            self.tags.append(tag)
            self.updated_at = datetime.utcnow()
    
    def remove_tag(self, tag: str) -> None:
        """Remove a tag from the pipeline."""
        if tag in self.tags:
            self.tags.remove(tag)
            self.updated_at = datetime.utcnow()
    
    def has_tag(self, tag: str) -> bool:
        """Check if pipeline has a specific tag."""
        return tag in self.tags
    
    def get_failed_steps(self) -> List[PipelineStep]:
        """Get all failed steps."""
        return [step for step in self.steps if step.status == StepStatus.FAILED]
    
    def get_completed_steps(self) -> List[PipelineStep]:
        """Get all completed steps."""
        return [step for step in self.steps if step.status == StepStatus.COMPLETED]
    
    def get_runnable_steps(self) -> List[PipelineStep]:
        """Get steps that can be executed (dependencies satisfied)."""
        runnable = []
        
        for step in self.steps:
            if step.status != StepStatus.PENDING:
                continue
            
            # Check if all dependencies are completed
            dependencies_satisfied = True
            for dep_id in step.dependencies:
                dep_step = self.get_step(dep_id)
                if not dep_step or dep_step.status != StepStatus.COMPLETED:
                    dependencies_satisfied = False
                    break
            
            if dependencies_satisfied:
                runnable.append(step)
        
        return runnable
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert pipeline to dictionary."""
        return {
            "id": str(self.id),
            "name": self.name,
            "description": self.description,
            "version": self.version,
            "status": self.status.value,
            "steps": [step.id for step in self.steps],
            "schedule_cron": self.schedule_cron,
            "max_retry_attempts": self.max_retry_attempts,
            "timeout_seconds": self.timeout_seconds,
            "created_at": self.created_at.isoformat(),
            "created_by": self.created_by,
            "updated_at": self.updated_at.isoformat(),
            "updated_by": self.updated_by,
            "started_at": self.started_at.isoformat() if self.started_at else None,
            "completed_at": self.completed_at.isoformat() if self.completed_at else None,
            "last_run_at": self.last_run_at.isoformat() if self.last_run_at else None,
            "next_run_at": self.next_run_at.isoformat() if self.next_run_at else None,
            "retry_count": self.retry_count,
            "config": self.config,
            "environment_vars": self.environment_vars,
            "tags": self.tags,
            "duration": self.duration,
            "success_rate": self.success_rate,
            "is_running": self.is_running,
            "is_completed": self.is_completed,
        }
    
    def __str__(self) -> str:
        """Human-readable string representation."""
        return f"DataPipeline('{self.name}', status={self.status.value}, steps={len(self.steps)})"
    
    def __repr__(self) -> str:
        """Developer string representation."""
        return (
            f"DataPipeline(id={self.id}, name='{self.name}', "
            f"status={self.status.value}, steps={len(self.steps)})"
        )