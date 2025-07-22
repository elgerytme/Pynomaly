"""
Training Job Domain Entity

Represents an automated machine learning training job with comprehensive
tracking of progress, results, and metadata for anomaly detection models.
"""

from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any

from monorepo.application.dto.training_dto import TrainingConfigDTO
from monorepo.domain.value_objects.hyperparameters import HyperparameterSet


class TrainingStatus(Enum):
    """Training job status enumeration."""

    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"
    PAUSED = "paused"


class TrainingPriority(Enum):
    """Training job priority levels."""

    LOW = "low"
    NORMAL = "normal"
    HIGH = "high"
    URGENT = "urgent"


@dataclass
class ResourceUsage:
    """Resource usage tracking for training jobs."""

    cpu_time: float = 0.0
    memory_peak: float = 0.0  # MB
    gpu_memory_peak: float = 0.0  # MB
    disk_io: float = 0.0  # MB
    network_io: float = 0.0  # MB

    def to_dict(self) -> dict[str, float]:
        """Convert to dictionary."""
        return {
            "cpu_time": self.cpu_time,
            "memory_peak": self.memory_peak,
            "gpu_memory_peak": self.gpu_memory_peak,
            "disk_io": self.disk_io,
            "network_io": self.network_io,
        }


@dataclass
class TrainingMetrics:
    """Training metrics and performance indicators."""

    best_score: float | None = None
    final_score: float | None = None
    validation_scores: list[float] = field(default_factory=list)
    training_loss: list[float] = field(default_factory=list)
    validation_loss: list[float] = field(default_factory=list)
    epochs_completed: int = 0
    early_stopping_epoch: int | None = None
    convergence_metrics: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "best_score": self.best_score,
            "final_score": self.final_score,
            "validation_scores": self.validation_scores,
            "training_loss": self.training_loss,
            "validation_loss": self.validation_loss,
            "epochs_completed": self.epochs_completed,
            "early_stopping_epoch": self.early_stopping_epoch,
            "convergence_metrics": self.convergence_metrics,
        }


@dataclass
class AlgorithmResult:
    """Result for a single algorithm in the training job."""

    algorithm_name: str
    model_id: str | None = None
    hyperparameters: HyperparameterSet | None = None
    metrics: TrainingMetrics | None = None
    training_time: float = 0.0
    optimization_history: list[dict[str, Any]] = field(default_factory=list)
    status: TrainingStatus = TrainingStatus.PENDING
    error_message: str | None = None

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "algorithm_name": self.algorithm_name,
            "model_id": self.model_id,
            "hyperparameters": (
                self.hyperparameters.to_dict() if self.hyperparameters else None
            ),
            "metrics": self.metrics.to_dict() if self.metrics else None,
            "training_time": self.training_time,
            "optimization_history": self.optimization_history,
            "status": self.status.value,
            "error_message": self.error_message,
        }


@dataclass
class TrainingJob:
    """
    Training job entity representing an automated ML training process.

    A training job encompasses the complete process of training multiple
    algorithms with hyperparameter optimization, model evaluation, and
    selection for anomaly detection tasks.
    """

    # Core identification
    id: str
    dataset_id: str
    name: str | None = None
    description: str | None = None

    # Configuration
    algorithms: list[str] = field(default_factory=list)
    config: TrainingConfigDTO | None = None
    priority: TrainingPriority = TrainingPriority.NORMAL

    # Status tracking
    status: TrainingStatus = TrainingStatus.PENDING
    progress: float = 0.0  # 0.0 to 100.0
    current_algorithm: str | None = None
    current_trial: int | None = None

    # Timing
    created_at: datetime = field(default_factory=datetime.utcnow)
    started_at: datetime | None = None
    completed_at: datetime | None = None
    updated_at: datetime | None = None
    estimated_completion: datetime | None = None

    # Results
    algorithm_results: list[AlgorithmResult] = field(default_factory=list)
    best_model_id: str | None = None
    best_algorithm: str | None = None
    final_metrics: TrainingMetrics | None = None

    # Resource tracking
    resource_usage: ResourceUsage = field(default_factory=ResourceUsage)
    worker_id: str | None = None
    gpu_ids: list[int] = field(default_factory=list)

    # Error handling
    error_message: str | None = None
    error_details: dict[str, Any] = field(default_factory=dict)
    retry_count: int = 0
    max_retries: int = 3

    # Metadata
    tags: list[str] = field(default_factory=list)
    user_id: str | None = None
    experiment_id: str | None = None
    parent_job_id: str | None = None
    child_job_ids: list[str] = field(default_factory=list)

    # Legacy compatibility (for backward compatibility with existing code)
    results: list[dict[str, Any]] | None = None

    def __post_init__(self):
        """Post-initialization processing."""
        if self.updated_at is None:
            self.updated_at = self.created_at

        # Convert legacy results to algorithm_results if needed
        if self.results and not self.algorithm_results:
            self.algorithm_results = [
                AlgorithmResult(
                    algorithm_name=result.get("algorithm", "unknown"),
                    model_id=result.get("model_id"),
                    training_time=result.get("training_time", 0.0),
                    optimization_history=result.get("optimization_history", []),
                    status=TrainingStatus.COMPLETED,
                )
                for result in self.results
            ]

    @property
    def duration(self) -> float | None:
        """Get job duration in seconds."""
        if self.started_at:
            end_time = self.completed_at or datetime.utcnow()
            return (end_time - self.started_at).total_seconds()
        return None

    @property
    def is_running(self) -> bool:
        """Check if job is currently running."""
        return self.status == TrainingStatus.RUNNING

    @property
    def is_completed(self) -> bool:
        """Check if job is completed (successfully or failed)."""
        return self.status in [
            TrainingStatus.COMPLETED,
            TrainingStatus.FAILED,
            TrainingStatus.CANCELLED,
        ]

    @property
    def success_rate(self) -> float:
        """Get success rate of algorithm training attempts."""
        if not self.algorithm_results:
            return 0.0

        successful = sum(
            1
            for result in self.algorithm_results
            if result.status == TrainingStatus.COMPLETED
        )
        return successful / len(self.algorithm_results)

    @property
    def total_training_time(self) -> float:
        """Get total training time across all algorithms."""
        return sum(result.training_time for result in self.algorithm_results)

    def start(self) -> None:
        """Mark job as started."""
        self.status = TrainingStatus.RUNNING
        self.started_at = datetime.utcnow()
        self.updated_at = self.started_at

    def complete(self) -> None:
        """Mark job as completed."""
        self.status = TrainingStatus.COMPLETED
        self.completed_at = datetime.utcnow()
        self.updated_at = self.completed_at
        self.progress = 100.0

    def fail(
        self, error_message: str, error_details: dict[str, Any] | None = None
    ) -> None:
        """Mark job as failed."""
        self.status = TrainingStatus.FAILED
        self.completed_at = datetime.utcnow()
        self.updated_at = self.completed_at
        self.error_message = error_message
        if error_details:
            self.error_details = error_details

    def cancel(self) -> None:
        """Cancel the job."""
        self.status = TrainingStatus.CANCELLED
        self.completed_at = datetime.utcnow()
        self.updated_at = self.completed_at

    def pause(self) -> None:
        """Pause the job."""
        if self.status == TrainingStatus.RUNNING:
            self.status = TrainingStatus.PAUSED
            self.updated_at = datetime.utcnow()

    def resume(self) -> None:
        """Resume the job."""
        if self.status == TrainingStatus.PAUSED:
            self.status = TrainingStatus.RUNNING
            self.updated_at = datetime.utcnow()

    def update_progress(
        self, progress: float, current_algorithm: str | None = None
    ) -> None:
        """Update job progress."""
        self.progress = max(0.0, min(100.0, progress))
        if current_algorithm:
            self.current_algorithm = current_algorithm
        self.updated_at = datetime.utcnow()

    def add_algorithm_result(self, result: AlgorithmResult) -> None:
        """Add an algorithm training result."""
        # Remove any existing result for the same algorithm
        self.algorithm_results = [
            r
            for r in self.algorithm_results
            if r.algorithm_name != result.algorithm_name
        ]
        self.algorithm_results.append(result)
        self.updated_at = datetime.utcnow()

        # Update best model if this result is better
        if (
            result.status == TrainingStatus.COMPLETED
            and result.metrics
            and result.metrics.best_score is not None
        ):
            current_best_score = (
                self.final_metrics.best_score
                if self.final_metrics and self.final_metrics.best_score
                else 0.0
            )

            if result.metrics.best_score > current_best_score:
                self.best_model_id = result.model_id
                self.best_algorithm = result.algorithm_name
                self.final_metrics = result.metrics

    def get_algorithm_result(self, algorithm_name: str) -> AlgorithmResult | None:
        """Get result for a specific algorithm."""
        for result in self.algorithm_results:
            if result.algorithm_name == algorithm_name:
                return result
        return None

    def get_successful_results(self) -> list[AlgorithmResult]:
        """Get all successful algorithm results."""
        return [
            result
            for result in self.algorithm_results
            if result.status == TrainingStatus.COMPLETED
        ]

    def get_failed_results(self) -> list[AlgorithmResult]:
        """Get all failed algorithm results."""
        return [
            result
            for result in self.algorithm_results
            if result.status == TrainingStatus.FAILED
        ]

    def add_tag(self, tag: str) -> None:
        """Add a tag to the job."""
        if tag not in self.tags:
            self.tags.append(tag)
            self.updated_at = datetime.utcnow()

    def remove_tag(self, tag: str) -> None:
        """Remove a tag from the job."""
        if tag in self.tags:
            self.tags.remove(tag)
            self.updated_at = datetime.utcnow()

    def can_retry(self) -> bool:
        """Check if job can be retried."""
        return (
            self.status == TrainingStatus.FAILED and self.retry_count < self.max_retries
        )

    def retry(self) -> None:
        """Retry the job."""
        if self.can_retry():
            self.retry_count += 1
            self.status = TrainingStatus.PENDING
            self.error_message = None
            self.error_details = {}
            self.updated_at = datetime.utcnow()

    def estimate_completion_time(self) -> datetime | None:
        """Estimate job completion time based on current progress."""
        if not self.started_at or self.progress <= 0:
            return None

        elapsed = (datetime.utcnow() - self.started_at).total_seconds()
        estimated_total = elapsed / (self.progress / 100.0)
        remaining = estimated_total - elapsed

        self.estimated_completion = datetime.utcnow() + timedelta(seconds=remaining)
        return self.estimated_completion

    def to_dict(self) -> dict[str, Any]:
        """Convert job to dictionary."""
        return {
            "id": self.id,
            "dataset_id": self.dataset_id,
            "name": self.name,
            "description": self.description,
            "algorithms": self.algorithms,
            "config": self.config.to_dict() if self.config else None,
            "priority": self.priority.value,
            "status": self.status.value,
            "progress": self.progress,
            "current_algorithm": self.current_algorithm,
            "current_trial": self.current_trial,
            "created_at": self.created_at.isoformat() if self.created_at else None,
            "started_at": self.started_at.isoformat() if self.started_at else None,
            "completed_at": (
                self.completed_at.isoformat() if self.completed_at else None
            ),
            "updated_at": self.updated_at.isoformat() if self.updated_at else None,
            "estimated_completion": (
                self.estimated_completion.isoformat()
                if self.estimated_completion
                else None
            ),
            "algorithm_results": [
                result.to_dict() for result in self.algorithm_results
            ],
            "best_model_id": self.best_model_id,
            "best_algorithm": self.best_algorithm,
            "final_metrics": (
                self.final_metrics.to_dict() if self.final_metrics else None
            ),
            "resource_usage": self.resource_usage.to_dict(),
            "worker_id": self.worker_id,
            "gpu_ids": self.gpu_ids,
            "error_message": self.error_message,
            "error_details": self.error_details,
            "retry_count": self.retry_count,
            "max_retries": self.max_retries,
            "tags": self.tags,
            "user_id": self.user_id,
            "experiment_id": self.experiment_id,
            "parent_job_id": self.parent_job_id,
            "child_job_ids": self.child_job_ids,
            "duration": self.duration,
            "success_rate": self.success_rate,
            "total_training_time": self.total_training_time,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "TrainingJob":
        """Create job from dictionary."""
        # Handle datetime fields
        datetime_fields = [
            "created_at",
            "started_at",
            "completed_at",
            "updated_at",
            "estimated_completion",
        ]
        for field in datetime_fields:
            if data.get(field):
                data[field] = datetime.fromisoformat(data[field])

        # Handle enum fields
        if "status" in data:
            data["status"] = TrainingStatus(data["status"])
        if "priority" in data:
            data["priority"] = TrainingPriority(data["priority"])

        # Handle algorithm results
        if "algorithm_results" in data:
            results = []
            for result_data in data["algorithm_results"]:
                result = AlgorithmResult(
                    algorithm_name=result_data["algorithm_name"],
                    model_id=result_data.get("model_id"),
                    hyperparameters=(
                        HyperparameterSet.from_dict(result_data["hyperparameters"])
                        if result_data.get("hyperparameters")
                        else None
                    ),
                    metrics=(
                        TrainingMetrics(**result_data["metrics"])
                        if result_data.get("metrics")
                        else None
                    ),
                    training_time=result_data.get("training_time", 0.0),
                    optimization_history=result_data.get("optimization_history", []),
                    status=TrainingStatus(result_data.get("status", "pending")),
                    error_message=result_data.get("error_message"),
                )
                results.append(result)
            data["algorithm_results"] = results

        # Handle other complex fields
        if "resource_usage" in data:
            data["resource_usage"] = ResourceUsage(**data["resource_usage"])

        if "final_metrics" in data and data["final_metrics"]:
            data["final_metrics"] = TrainingMetrics(**data["final_metrics"])

        # Remove calculated fields
        calculated_fields = ["duration", "success_rate", "total_training_time"]
        for field in calculated_fields:
            data.pop(field, None)

        return cls(**data)


# Import timedelta for completion time estimation
from datetime import timedelta
