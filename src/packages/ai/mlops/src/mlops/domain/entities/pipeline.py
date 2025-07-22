"""Pipeline domain entity for MLOps package."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Dict, Any, Optional, List
from uuid import UUID, uuid4


class PipelineStatus(str, Enum):
    """Pipeline execution status."""
    
    CREATED = "created"
    SCHEDULED = "scheduled"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"
    PAUSED = "paused"


class PipelineType(str, Enum):
    """Pipeline type enumeration."""
    
    TRAINING = "training"
    INFERENCE = "inference"
    BATCH_PREDICTION = "batch_prediction"
    DATA_PROCESSING = "data_processing"
    MODEL_VALIDATION = "model_validation"
    DEPLOYMENT = "deployment"


@dataclass
class PipelineStage:
    """Individual pipeline stage."""
    
    name: str
    stage_type: str
    config: Dict[str, Any] = field(default_factory=dict)
    status: str = "pending"
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    error_message: Optional[str] = None
    outputs: Dict[str, Any] = field(default_factory=dict)
    
    def start(self) -> None:
        """Start the pipeline stage."""
        self.status = "running"
        self.started_at = datetime.utcnow()
    
    def complete(self, outputs: Optional[Dict[str, Any]] = None) -> None:
        """Complete the pipeline stage."""
        self.status = "completed"
        self.completed_at = datetime.utcnow()
        if outputs:
            self.outputs.update(outputs)
    
    def fail(self, error_message: str) -> None:
        """Mark the pipeline stage as failed."""
        self.status = "failed"
        self.completed_at = datetime.utcnow()
        self.error_message = error_message
    
    @property
    def duration(self) -> Optional[float]:
        """Get stage duration in seconds."""
        if not self.started_at:
            return None
        
        end_time = self.completed_at or datetime.utcnow()
        return (end_time - self.started_at).total_seconds()


@dataclass
class Pipeline:
    """Pipeline domain entity.
    
    Represents an ML pipeline with orchestration capabilities,
    following Domain-Driven Design principles.
    """
    
    id: UUID = field(default_factory=uuid4)
    name: str = ""
    description: str = ""
    pipeline_type: PipelineType = PipelineType.TRAINING
    status: PipelineStatus = PipelineStatus.CREATED
    
    # Pipeline configuration
    stages: List[PipelineStage] = field(default_factory=list)
    config: Dict[str, Any] = field(default_factory=dict)
    environment: str = "development"
    
    # Scheduling
    schedule: Optional[str] = None  # Cron expression
    is_scheduled: bool = False
    
    # Execution tracking
    execution_count: int = 0
    last_execution_id: Optional[UUID] = None
    
    # Metadata
    created_at: datetime = field(default_factory=datetime.utcnow)
    updated_at: datetime = field(default_factory=datetime.utcnow)
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    
    # User tracking
    created_by: str = ""
    team: str = ""
    
    # Associated resources
    model_id: Optional[UUID] = None
    experiment_id: Optional[UUID] = None
    
    def __post_init__(self) -> None:
        """Post-initialization validation and setup."""
        if not self.name:
            self.name = f"pipeline_{self.id.hex[:8]}"
        
        self._validate_pipeline()
    
    def _validate_pipeline(self) -> None:
        """Validate pipeline state and business rules."""
        if not self.name or len(self.name.strip()) == 0:
            raise ValueError("Pipeline name cannot be empty")
        
        if len(self.name) > 100:
            raise ValueError("Pipeline name cannot exceed 100 characters")
        
        if self.pipeline_type not in PipelineType:
            raise ValueError(f"Invalid pipeline type: {self.pipeline_type}")
        
        if self.status not in PipelineStatus:
            raise ValueError(f"Invalid pipeline status: {self.status}")
    
    def add_stage(self, name: str, stage_type: str, config: Optional[Dict[str, Any]] = None) -> None:
        """Add a stage to the pipeline.
        
        Args:
            name: Stage name
            stage_type: Type of stage
            config: Stage configuration
        """
        if not name or not isinstance(name, str):
            raise ValueError("Stage name must be a non-empty string")
        
        if not stage_type or not isinstance(stage_type, str):
            raise ValueError("Stage type must be a non-empty string")
        
        # Check for duplicate stage names
        if any(stage.name == name for stage in self.stages):
            raise ValueError(f"Stage with name '{name}' already exists")
        
        stage = PipelineStage(
            name=name,
            stage_type=stage_type,
            config=config or {}
        )
        
        self.stages.append(stage)
        self.updated_at = datetime.utcnow()
    
    def remove_stage(self, name: str) -> None:
        """Remove a stage from the pipeline.
        
        Args:
            name: Name of stage to remove
        """
        original_count = len(self.stages)
        self.stages = [stage for stage in self.stages if stage.name != name]
        
        if len(self.stages) == original_count:
            raise ValueError(f"Stage '{name}' not found")
        
        self.updated_at = datetime.utcnow()
    
    def get_stage(self, name: str) -> Optional[PipelineStage]:
        """Get a stage by name.
        
        Args:
            name: Stage name
            
        Returns:
            PipelineStage if found, None otherwise
        """
        return next((stage for stage in self.stages if stage.name == name), None)
    
    def start(self) -> None:
        """Start pipeline execution.
        
        Raises:
            ValueError: If pipeline cannot be started
        """
        if self.status not in [PipelineStatus.CREATED, PipelineStatus.SCHEDULED]:
            raise ValueError(f"Cannot start pipeline in {self.status} status")
        
        if not self.stages:
            raise ValueError("Cannot start pipeline without stages")
        
        self.status = PipelineStatus.RUNNING
        self.started_at = datetime.utcnow()
        self.updated_at = datetime.utcnow()
        self.execution_count += 1
    
    def complete(self) -> None:
        """Complete pipeline execution.
        
        Raises:
            ValueError: If pipeline is not in RUNNING status
        """
        if self.status != PipelineStatus.RUNNING:
            raise ValueError(f"Cannot complete pipeline in {self.status} status")
        
        self.status = PipelineStatus.COMPLETED
        self.completed_at = datetime.utcnow()
        self.updated_at = datetime.utcnow()
    
    def fail(self, error_message: str = "") -> None:
        """Mark pipeline as failed.
        
        Args:
            error_message: Optional error message
        """
        self.status = PipelineStatus.FAILED
        self.completed_at = datetime.utcnow()
        self.updated_at = datetime.utcnow()
        
        if error_message:
            self.config["error_message"] = error_message
    
    def cancel(self) -> None:
        """Cancel pipeline execution.
        
        Raises:
            ValueError: If pipeline cannot be cancelled
        """
        if self.status in [PipelineStatus.COMPLETED, PipelineStatus.FAILED]:
            raise ValueError(f"Cannot cancel pipeline in {self.status} status")
        
        self.status = PipelineStatus.CANCELLED
        self.completed_at = datetime.utcnow()
        self.updated_at = datetime.utcnow()
    
    def pause(self) -> None:
        """Pause pipeline execution.
        
        Raises:
            ValueError: If pipeline is not in RUNNING status
        """
        if self.status != PipelineStatus.RUNNING:
            raise ValueError(f"Cannot pause pipeline in {self.status} status")
        
        self.status = PipelineStatus.PAUSED
        self.updated_at = datetime.utcnow()
    
    def resume(self) -> None:
        """Resume pipeline execution.
        
        Raises:
            ValueError: If pipeline is not in PAUSED status
        """
        if self.status != PipelineStatus.PAUSED:
            raise ValueError(f"Cannot resume pipeline in {self.status} status")
        
        self.status = PipelineStatus.RUNNING
        self.updated_at = datetime.utcnow()
    
    def schedule(self, cron_expression: str) -> None:
        """Schedule the pipeline with cron expression.
        
        Args:
            cron_expression: Cron expression for scheduling
        """
        if not cron_expression or not isinstance(cron_expression, str):
            raise ValueError("Cron expression must be a non-empty string")
        
        self.schedule = cron_expression
        self.is_scheduled = True
        self.status = PipelineStatus.SCHEDULED
        self.updated_at = datetime.utcnow()
    
    def unschedule(self) -> None:
        """Remove scheduling from the pipeline."""
        self.schedule = None
        self.is_scheduled = False
        if self.status == PipelineStatus.SCHEDULED:
            self.status = PipelineStatus.CREATED
        self.updated_at = datetime.utcnow()
    
    @property
    def duration(self) -> Optional[float]:
        """Get pipeline duration in seconds."""
        if not self.started_at:
            return None
        
        end_time = self.completed_at or datetime.utcnow()
        return (end_time - self.started_at).total_seconds()
    
    @property
    def is_running(self) -> bool:
        """Check if pipeline is currently running."""
        return self.status == PipelineStatus.RUNNING
    
    @property
    def is_finished(self) -> bool:
        """Check if pipeline is finished."""
        return self.status in [
            PipelineStatus.COMPLETED,
            PipelineStatus.FAILED,
            PipelineStatus.CANCELLED
        ]
    
    @property
    def stage_count(self) -> int:
        """Get the number of stages."""
        return len(self.stages)
    
    @property
    def completed_stages(self) -> List[PipelineStage]:
        """Get completed stages."""
        return [stage for stage in self.stages if stage.status == "completed"]
    
    @property
    def failed_stages(self) -> List[PipelineStage]:
        """Get failed stages."""
        return [stage for stage in self.stages if stage.status == "failed"]
    
    @property
    def success_rate(self) -> float:
        """Calculate pipeline success rate based on execution history."""
        if self.execution_count == 0:
            return 0.0
        
        # This would typically be calculated from execution history
        # For now, return 1.0 if last execution was successful
        if self.status == PipelineStatus.COMPLETED:
            return 1.0
        elif self.status == PipelineStatus.FAILED:
            return 0.0
        else:
            return 0.5  # Unknown/in progress
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert pipeline to dictionary representation."""
        return {
            "id": str(self.id),
            "name": self.name,
            "description": self.description,
            "pipeline_type": self.pipeline_type.value,
            "status": self.status.value,
            "stages": [
                {
                    "name": stage.name,
                    "stage_type": stage.stage_type,
                    "config": stage.config,
                    "status": stage.status,
                    "started_at": stage.started_at.isoformat() if stage.started_at else None,
                    "completed_at": stage.completed_at.isoformat() if stage.completed_at else None,
                    "error_message": stage.error_message,
                    "outputs": stage.outputs,
                    "duration": stage.duration,
                }
                for stage in self.stages
            ],
            "config": self.config,
            "environment": self.environment,
            "schedule": self.schedule,
            "is_scheduled": self.is_scheduled,
            "execution_count": self.execution_count,
            "last_execution_id": str(self.last_execution_id) if self.last_execution_id else None,
            "created_at": self.created_at.isoformat(),
            "updated_at": self.updated_at.isoformat(),
            "started_at": self.started_at.isoformat() if self.started_at else None,
            "completed_at": self.completed_at.isoformat() if self.completed_at else None,
            "created_by": self.created_by,
            "team": self.team,
            "model_id": str(self.model_id) if self.model_id else None,
            "experiment_id": str(self.experiment_id) if self.experiment_id else None,
            "duration": self.duration,
            "is_running": self.is_running,
            "is_finished": self.is_finished,
            "stage_count": self.stage_count,
            "success_rate": self.success_rate,
        }
    
    def __str__(self) -> str:
        """String representation of the pipeline."""
        return f"Pipeline(id={self.id.hex[:8]}, name='{self.name}', status={self.status.value})"
    
    def __repr__(self) -> str:
        """Detailed string representation of the pipeline."""
        return (
            f"Pipeline("
            f"id={self.id}, "
            f"name='{self.name}', "
            f"type={self.pipeline_type.value}, "
            f"status={self.status.value}, "
            f"stages={len(self.stages)}"
            f")"
        )