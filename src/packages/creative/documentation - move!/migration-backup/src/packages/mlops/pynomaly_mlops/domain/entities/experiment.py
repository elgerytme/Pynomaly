"""Experiment Entities

Domain entities for experiment tracking and management in MLOps.
"""

from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Dict, Any, Optional, List
from uuid import UUID, uuid4


class ExperimentStatus(Enum):
    """Experiment status enumeration."""
    ACTIVE = "active"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"
    ARCHIVED = "archived"


@dataclass
class Experiment:
    """Experiment entity for organizing related experiment runs.
    
    An experiment groups multiple runs that are testing variations
    of the same hypothesis or approach.
    """
    
    # Identity
    id: UUID = field(default_factory=uuid4)
    name: str = field()
    
    # Metadata
    description: Optional[str] = field(default=None)
    tags: List[str] = field(default_factory=list)
    
    # Lifecycle
    status: ExperimentStatus = field(default=ExperimentStatus.ACTIVE)
    created_at: datetime = field(default_factory=datetime.utcnow)
    updated_at: datetime = field(default_factory=datetime.utcnow)
    created_by: str = field()
    
    # Organization
    project: Optional[str] = field(default=None)
    team: Optional[str] = field(default=None)
    
    # Configuration
    baseline_config: Dict[str, Any] = field(default_factory=dict)
    search_space: Dict[str, Any] = field(default_factory=dict)
    
    # Statistics
    run_count: int = field(default=0)
    best_run_id: Optional[UUID] = field(default=None)
    best_metric_value: Optional[float] = field(default=None)
    optimization_metric: Optional[str] = field(default=None)
    
    def __post_init__(self):
        """Post-initialization validation."""
        if not self.name:
            raise ValueError("Experiment name cannot be empty")
        
        if not self.created_by:
            raise ValueError("Experiment must have a creator")
        
        # Ensure tags are unique
        self.tags = list(set(self.tags))
    
    def add_tag(self, tag: str) -> None:
        """Add a tag to the experiment."""
        if tag and tag not in self.tags:
            self.tags.append(tag)
    
    def remove_tag(self, tag: str) -> None:
        """Remove a tag from the experiment."""
        if tag in self.tags:
            self.tags.remove(tag)
    
    def update_best_run(self, run_id: UUID, metric_value: float) -> None:
        """Update the best performing run for this experiment."""
        if (self.best_metric_value is None or 
            metric_value > self.best_metric_value):
            self.best_run_id = run_id
            self.best_metric_value = metric_value
            self.updated_at = datetime.utcnow()
    
    def increment_run_count(self) -> None:
        """Increment the run count for this experiment."""
        self.run_count += 1
        self.updated_at = datetime.utcnow()
    
    def complete(self) -> None:
        """Mark experiment as completed."""
        self.status = ExperimentStatus.COMPLETED
        self.updated_at = datetime.utcnow()
    
    def archive(self) -> None:
        """Archive the experiment."""
        self.status = ExperimentStatus.ARCHIVED
        self.updated_at = datetime.utcnow()


@dataclass
class ExperimentRun:
    """Individual experiment run entity.
    
    Represents a single execution of an experiment with specific
    parameters, metrics, and artifacts.
    """
    
    # Identity
    id: UUID = field(default_factory=uuid4)
    experiment_id: UUID = field()
    name: Optional[str] = field(default=None)
    
    # Execution
    status: ExperimentStatus = field(default=ExperimentStatus.ACTIVE)
    started_at: datetime = field(default_factory=datetime.utcnow)
    ended_at: Optional[datetime] = field(default=None)
    duration_seconds: Optional[float] = field(default=None)
    
    # User and Environment
    created_by: str = field()
    source_file: Optional[str] = field(default=None)
    source_version: Optional[str] = field(default=None)
    environment_info: Dict[str, Any] = field(default_factory=dict)
    
    # Parameters and Configuration
    parameters: Dict[str, Any] = field(default_factory=dict)
    hyperparameters: Dict[str, Any] = field(default_factory=dict)
    config: Dict[str, Any] = field(default_factory=dict)
    
    # Results
    metrics: Dict[str, float] = field(default_factory=dict)
    final_metrics: Dict[str, float] = field(default_factory=dict)
    intermediate_metrics: List[Dict[str, Any]] = field(default_factory=list)
    
    # Artifacts
    artifacts: Dict[str, str] = field(default_factory=dict)  # name -> uri
    model_artifacts: Dict[str, str] = field(default_factory=dict)
    
    # Tracking
    tags: List[str] = field(default_factory=list)
    notes: Optional[str] = field(default=None)
    
    # System Information
    hostname: Optional[str] = field(default=None)
    git_commit: Optional[str] = field(default=None)
    python_version: Optional[str] = field(default=None)
    
    def __post_init__(self):
        """Post-initialization validation."""
        if not self.created_by:
            raise ValueError("ExperimentRun must have a creator")
        
        # Generate name if not provided
        if not self.name:
            self.name = f"run_{self.started_at.strftime('%Y%m%d_%H%M%S')}"
        
        # Ensure tags are unique
        self.tags = list(set(self.tags))
    
    def log_parameter(self, key: str, value: Any) -> None:
        """Log a parameter for this run."""
        self.parameters[key] = value
    
    def log_metric(self, key: str, value: float, step: Optional[int] = None) -> None:
        """Log a metric for this run."""
        self.metrics[key] = value
        
        if step is not None:
            # Store intermediate metric
            metric_entry = {
                "key": key,
                "value": value,
                "step": step,
                "timestamp": datetime.utcnow().isoformat()
            }
            self.intermediate_metrics.append(metric_entry)
    
    def log_artifact(self, name: str, uri: str) -> None:
        """Log an artifact for this run."""
        self.artifacts[name] = uri
    
    def log_model_artifact(self, name: str, uri: str) -> None:
        """Log a model artifact for this run."""
        self.model_artifacts[name] = uri
    
    def add_tag(self, tag: str) -> None:
        """Add a tag to the run."""
        if tag and tag not in self.tags:
            self.tags.append(tag)
    
    def set_status(self, status: ExperimentStatus) -> None:
        """Set the run status."""
        self.status = status
        
        if status in [ExperimentStatus.COMPLETED, ExperimentStatus.FAILED, ExperimentStatus.CANCELLED]:
            self.end_run()
    
    def end_run(self) -> None:
        """End the experiment run and calculate duration."""
        self.ended_at = datetime.utcnow()
        self.duration_seconds = (self.ended_at - self.started_at).total_seconds()
        
        # Copy current metrics to final metrics
        self.final_metrics = self.metrics.copy()
    
    def complete(self) -> None:
        """Mark the run as completed successfully."""
        self.set_status(ExperimentStatus.COMPLETED)
    
    def fail(self, error_message: Optional[str] = None) -> None:
        """Mark the run as failed."""
        self.set_status(ExperimentStatus.FAILED)
        if error_message:
            self.notes = f"Failed: {error_message}"
    
    @property
    def is_active(self) -> bool:
        """Check if the run is currently active."""
        return self.status == ExperimentStatus.ACTIVE
    
    @property
    def is_finished(self) -> bool:
        """Check if the run has finished."""
        return self.status in [
            ExperimentStatus.COMPLETED,
            ExperimentStatus.FAILED,
            ExperimentStatus.CANCELLED
        ]
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert run to dictionary representation."""
        return {
            "id": str(self.id),
            "experiment_id": str(self.experiment_id),
            "name": self.name,
            "status": self.status.value,
            "started_at": self.started_at.isoformat(),
            "ended_at": self.ended_at.isoformat() if self.ended_at else None,
            "duration_seconds": self.duration_seconds,
            "created_by": self.created_by,
            "source_file": self.source_file,
            "source_version": self.source_version,
            "environment_info": self.environment_info,
            "parameters": self.parameters,
            "hyperparameters": self.hyperparameters,
            "config": self.config,
            "metrics": self.metrics,
            "final_metrics": self.final_metrics,
            "intermediate_metrics": self.intermediate_metrics,
            "artifacts": self.artifacts,
            "model_artifacts": self.model_artifacts,
            "tags": self.tags,
            "notes": self.notes,
            "hostname": self.hostname,
            "git_commit": self.git_commit,
            "python_version": self.python_version,
        }