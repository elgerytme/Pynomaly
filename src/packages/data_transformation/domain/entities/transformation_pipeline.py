"""Transformation pipeline entity."""

from __future__ import annotations

from typing import Any, Dict, List, Optional
from uuid import UUID, uuid4
from datetime import datetime
from enum import Enum

from pydantic import BaseModel, ConfigDict, Enum, Field

from ..value_objects.pipeline_config import PipelineConfig
from ..value_objects.transformation_step import TransformationStep


class PipelineStatus(str):
    """Pipeline execution status."""
    CREATED = "created"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


class TransformationPipeline(BaseModel):
    """
    Core entity representing a data transformation pipeline.
    
    Encapsulates the complete transformation workflow including configuration,
    steps, execution state, and results.
    """
    
    id: UUID = Field(default_factory=uuid4)
    name: str
    description: Optional[str] = None
    config: PipelineConfig
    steps: List[TransformationStep] = Field(default_factory=list)
    status: PipelineStatus = PipelineStatus.CREATED
    
    # Execution metadata
    created_at: datetime = Field(default_factory=datetime.utcnow)
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    
    # Execution results
    records_processed: Optional[int] = None
    features_created: Optional[int] = None
    execution_time_seconds: Optional[float] = None
    memory_usage_mb: Optional[float] = None
    
    # Error handling
    error_message: Optional[str] = None
    failed_step: Optional[str] = None
    
    # Metadata
    metadata: Dict[str, Any] = Field(default_factory=dict)
    tags: List[str] = Field(default_factory=list)        """Pydantic configuration."""
        arbitrary_types_allowed = True
        use_enum_values = True
    
    def add_step(self, step: TransformationStep) -> None:
        """Add a transformation step to the pipeline."""
        self.steps.append(step)
    
    def start_execution(self) -> None:
        """Mark pipeline as started."""
        self.status = PipelineStatus.RUNNING
        self.started_at = datetime.utcnow()
    
    def complete_execution(
        self,
        records_processed: int,
        features_created: int,
        execution_time: float,
        memory_usage: float
    ) -> None:
        """Mark pipeline as completed with results."""
        self.status = PipelineStatus.COMPLETED
        self.completed_at = datetime.utcnow()
        self.records_processed = records_processed
        self.features_created = features_created
        self.execution_time_seconds = execution_time
        self.memory_usage_mb = memory_usage
    
    def fail_execution(self, error_message: str, failed_step: Optional[str] = None) -> None:
        """Mark pipeline as failed with error details."""
        self.status = PipelineStatus.FAILED
        self.completed_at = datetime.utcnow()
        self.error_message = error_message
        self.failed_step = failed_step
    
    def cancel_execution(self) -> None:
        """Cancel pipeline execution."""
        self.status = PipelineStatus.CANCELLED
        self.completed_at = datetime.utcnow()
    
    @property
    def is_running(self) -> bool:
        """Check if pipeline is currently running."""
        return self.status == PipelineStatus.RUNNING
    
    @property
    def is_completed(self) -> bool:
        """Check if pipeline completed successfully."""
        return self.status == PipelineStatus.COMPLETED
    
    @property
    def has_failed(self) -> bool:
        """Check if pipeline failed."""
        return self.status == PipelineStatus.FAILED
    
    @property
    def total_steps(self) -> int:
        """Get total number of transformation steps."""
        return len(self.steps)
    
    def get_step_by_name(self, name: str) -> Optional[TransformationStep]:
        """Get transformation step by name."""
        for step in self.steps:
            if step.name == name:
                return step
        return None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert pipeline to dictionary representation."""
        return {
            "id": str(self.id),
            "name": self.name,
            "description": self.description,
            "status": self.status,
            "config": self.config.model_dump(),
            "steps": [step.model_dump() for step in self.steps],
            "created_at": self.created_at.isoformat(),
            "started_at": self.started_at.isoformat() if self.started_at else None,
            "completed_at": self.completed_at.isoformat() if self.completed_at else None,
            "records_processed": self.records_processed,
            "features_created": self.features_created,
            "execution_time_seconds": self.execution_time_seconds,
            "memory_usage_mb": self.memory_usage_mb,
            "error_message": self.error_message,
            "failed_step": self.failed_step,
            "metadata": self.metadata,
            "tags": self.tags,
        }