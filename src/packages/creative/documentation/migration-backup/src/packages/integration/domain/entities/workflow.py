"""
Workflow entity for orchestrating data science operations.
"""

from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import List, Dict, Any, Optional
from uuid import UUID, uuid4

from core.domain.abstractions.base_entity import BaseEntity


class WorkflowStatus(Enum):
    """Workflow execution status."""
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


class WorkflowStepType(Enum):
    """Types of workflow steps."""
    DATA_PROFILING = "data_profiling"
    DATA_QUALITY = "data_quality"
    FEATURE_ENGINEERING = "feature_engineering"
    MODEL_TRAINING = "model_training"
    MODEL_VALIDATION = "model_validation"
    DEPLOYMENT = "deployment"
    MONITORING = "monitoring"


@dataclass
class WorkflowStep:
    """Individual step in a workflow."""
    id: str
    name: str
    step_type: WorkflowStepType
    config: Dict[str, Any] = field(default_factory=dict)
    dependencies: List[str] = field(default_factory=list)
    outputs: List[str] = field(default_factory=list)
    timeout_seconds: int = 3600
    retry_count: int = 3
    status: WorkflowStatus = WorkflowStatus.PENDING
    start_time: Optional[datetime] = None
    end_time: Optional[datetime] = None
    error_message: Optional[str] = None
    metrics: Dict[str, Any] = field(default_factory=dict)


@dataclass
class Workflow(BaseEntity):
    """Workflow entity for orchestrating data science operations."""
    
    name: str
    description: str
    steps: List[WorkflowStep] = field(default_factory=list)
    status: WorkflowStatus = WorkflowStatus.PENDING
    config: Dict[str, Any] = field(default_factory=dict)
    created_at: datetime = field(default_factory=datetime.utcnow)
    updated_at: datetime = field(default_factory=datetime.utcnow)
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    created_by: Optional[str] = None
    tags: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def __post_init__(self):
        """Initialize workflow after creation."""
        if not self.id:
            self.id = uuid4()
            
    def add_step(self, step: WorkflowStep) -> None:
        """Add a step to the workflow."""
        self.steps.append(step)
        self.updated_at = datetime.utcnow()
        
    def get_step(self, step_id: str) -> Optional[WorkflowStep]:
        """Get a step by ID."""
        for step in self.steps:
            if step.id == step_id:
                return step
        return None
        
    def get_ready_steps(self) -> List[WorkflowStep]:
        """Get steps that are ready to execute (all dependencies completed)."""
        ready_steps = []
        
        for step in self.steps:
            if step.status != WorkflowStatus.PENDING:
                continue
                
            # Check if all dependencies are completed
            dependencies_completed = True
            for dep_id in step.dependencies:
                dep_step = self.get_step(dep_id)
                if not dep_step or dep_step.status != WorkflowStatus.COMPLETED:
                    dependencies_completed = False
                    break
                    
            if dependencies_completed:
                ready_steps.append(step)
                
        return ready_steps
        
    def update_step_status(self, step_id: str, status: WorkflowStatus, 
                          error_message: Optional[str] = None) -> None:
        """Update the status of a specific step."""
        step = self.get_step(step_id)
        if step:
            step.status = status
            if status == WorkflowStatus.RUNNING:
                step.start_time = datetime.utcnow()
            elif status in [WorkflowStatus.COMPLETED, WorkflowStatus.FAILED, WorkflowStatus.CANCELLED]:
                step.end_time = datetime.utcnow()
                
            if error_message:
                step.error_message = error_message
                
            self.updated_at = datetime.utcnow()
            
    def is_completed(self) -> bool:
        """Check if all steps are completed."""
        return all(step.status == WorkflowStatus.COMPLETED for step in self.steps)
        
    def has_failed(self) -> bool:
        """Check if any step has failed."""
        return any(step.status == WorkflowStatus.FAILED for step in self.steps)
        
    def get_execution_time(self) -> Optional[float]:
        """Get total execution time in seconds."""
        if not self.started_at:
            return None
            
        end_time = self.completed_at or datetime.utcnow()
        return (end_time - self.started_at).total_seconds()
        
    def get_step_metrics(self) -> Dict[str, Any]:
        """Get aggregated metrics from all steps."""
        metrics = {
            "total_steps": len(self.steps),
            "completed_steps": sum(1 for step in self.steps if step.status == WorkflowStatus.COMPLETED),
            "failed_steps": sum(1 for step in self.steps if step.status == WorkflowStatus.FAILED),
            "running_steps": sum(1 for step in self.steps if step.status == WorkflowStatus.RUNNING),
            "pending_steps": sum(1 for step in self.steps if step.status == WorkflowStatus.PENDING),
            "execution_time": self.get_execution_time()
        }
        
        # Aggregate step-specific metrics
        for step in self.steps:
            if step.metrics:
                for key, value in step.metrics.items():
                    metrics[f"step_{step.id}_{key}"] = value
                    
        return metrics