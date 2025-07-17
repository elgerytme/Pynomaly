"""Value objects for ML pipelines."""

from dataclasses import dataclass, field
from datetime import datetime
from typing import Dict, List, Optional, Any
from uuid import UUID, uuid4
from enum import Enum


class PipelineStatus(Enum):
    """Pipeline status enumeration."""
    CREATED = "created"
    SCHEDULED = "scheduled"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"
    PAUSED = "paused"


class PipelineType(Enum):
    """Pipeline type enumeration."""
    TRAINING = "training"
    INFERENCE = "inference"
    DATA_PROCESSING = "data_processing"
    MODEL_VALIDATION = "model_validation"
    DEPLOYMENT = "deployment"
    MONITORING = "monitoring"
    BATCH_INFERENCE = "batch_inference"
    REAL_TIME_INFERENCE = "real_time_inference"


class StepType(Enum):
    """Pipeline step type enumeration."""
    DATA_INGESTION = "data_ingestion"
    DATA_VALIDATION = "data_validation"
    DATA_PREPROCESSING = "data_preprocessing"
    FEATURE_ENGINEERING = "feature_engineering"
    MODEL_TRAINING = "model_training"
    MODEL_EVALUATION = "model_evaluation"
    MODEL_VALIDATION = "model_validation"
    MODEL_DEPLOYMENT = "model_deployment"
    MODEL_MONITORING = "model_monitoring"
    CUSTOM = "custom"


@dataclass(frozen=True)
class PipelineId:
    """Unique identifier for ML pipelines."""
    value: UUID = field(default_factory=uuid4)
    
    def __str__(self) -> str:
        return str(self.value)


@dataclass(frozen=True)
class PipelineStep:
    """Individual step in ML pipeline."""
    name: str
    step_type: StepType
    inputs: List[str] = field(default_factory=list)
    outputs: List[str] = field(default_factory=list)
    parameters: Dict[str, Any] = field(default_factory=dict)
    dependencies: List[str] = field(default_factory=list)
    timeout_seconds: Optional[int] = None
    retries: int = 0
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert step to dictionary."""
        return {
            "name": self.name,
            "step_type": self.step_type.value,
            "inputs": self.inputs,
            "outputs": self.outputs,
            "parameters": self.parameters,
            "dependencies": self.dependencies,
            "timeout_seconds": self.timeout_seconds,
            "retries": self.retries,
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "PipelineStep":
        """Create step from dictionary."""
        return cls(
            name=data["name"],
            step_type=StepType(data["step_type"]),
            inputs=data.get("inputs", []),
            outputs=data.get("outputs", []),
            parameters=data.get("parameters", {}),
            dependencies=data.get("dependencies", []),
            timeout_seconds=data.get("timeout_seconds"),
            retries=data.get("retries", 0),
        )


@dataclass(frozen=True)
class PipelineConfiguration:
    """Configuration for ML pipelines."""
    name: str
    description: Optional[str] = None
    steps: List[PipelineStep] = field(default_factory=list)
    schedule: Optional[str] = None  # cron expression
    timeout_seconds: Optional[int] = None
    max_retries: int = 3
    environment: Dict[str, str] = field(default_factory=dict)
    resources: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert configuration to dictionary."""
        return {
            "name": self.name,
            "description": self.description,
            "steps": [step.to_dict() for step in self.steps],
            "schedule": self.schedule,
            "timeout_seconds": self.timeout_seconds,
            "max_retries": self.max_retries,
            "environment": self.environment,
            "resources": self.resources,
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "PipelineConfiguration":
        """Create configuration from dictionary."""
        return cls(
            name=data["name"],
            description=data.get("description"),
            steps=[PipelineStep.from_dict(step) for step in data.get("steps", [])],
            schedule=data.get("schedule"),
            timeout_seconds=data.get("timeout_seconds"),
            max_retries=data.get("max_retries", 3),
            environment=data.get("environment", {}),
            resources=data.get("resources", {}),
        )


@dataclass(frozen=True)
class PipelineExecution:
    """Execution information for ML pipelines."""
    execution_id: str
    started_at: datetime
    completed_at: Optional[datetime] = None
    status: PipelineStatus = PipelineStatus.CREATED
    step_executions: Dict[str, Dict[str, Any]] = field(default_factory=dict)
    logs: List[str] = field(default_factory=list)
    artifacts: List[str] = field(default_factory=list)
    metrics: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert execution to dictionary."""
        return {
            "execution_id": self.execution_id,
            "started_at": self.started_at.isoformat(),
            "completed_at": self.completed_at.isoformat() if self.completed_at else None,
            "status": self.status.value,
            "step_executions": self.step_executions,
            "logs": self.logs,
            "artifacts": self.artifacts,
            "metrics": self.metrics,
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "PipelineExecution":
        """Create execution from dictionary."""
        return cls(
            execution_id=data["execution_id"],
            started_at=datetime.fromisoformat(data["started_at"]),
            completed_at=datetime.fromisoformat(data["completed_at"]) if data.get("completed_at") else None,
            status=PipelineStatus(data.get("status", PipelineStatus.CREATED.value)),
            step_executions=data.get("step_executions", {}),
            logs=data.get("logs", []),
            artifacts=data.get("artifacts", []),
            metrics=data.get("metrics", {}),
        )


@dataclass(frozen=True)
class PipelineMetadata:
    """Metadata for ML pipelines."""
    created_by: str
    project_name: str
    version: str = "1.0.0"
    tags: List[str] = field(default_factory=list)
    git_commit: Optional[str] = None
    git_branch: Optional[str] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert metadata to dictionary."""
        return {
            "created_by": self.created_by,
            "project_name": self.project_name,
            "version": self.version,
            "tags": self.tags,
            "git_commit": self.git_commit,
            "git_branch": self.git_branch,
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "PipelineMetadata":
        """Create metadata from dictionary."""
        return cls(
            created_by=data["created_by"],
            project_name=data["project_name"],
            version=data.get("version", "1.0.0"),
            tags=data.get("tags", []),
            git_commit=data.get("git_commit"),
            git_branch=data.get("git_branch"),
        )