"""Training domain protocols for dependency inversion.

This module defines the interfaces/protocols that the domain layer expects
from infrastructure implementations for training automation services.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Protocol

from pynomaly.domain.entities import Dataset, DetectionResult, Detector
from pynomaly.domain.entities.training_job import TrainingJob, TrainingStatus


class TrainingJobRepositoryProtocol(Protocol):
    """Repository protocol for training job persistence."""

    async def save_job(self, job: TrainingJob) -> None:
        """Save training job."""
        ...

    async def get_job(self, job_id: str) -> TrainingJob | None:
        """Get training job by ID."""
        ...

    async def list_jobs(
        self, status: TrainingStatus | None = None, limit: int = 100
    ) -> list[TrainingJob]:
        """List training jobs."""
        ...

    async def update_job_status(self, job_id: str, status: TrainingStatus) -> None:
        """Update job status."""
        ...

    async def delete_job(self, job_id: str) -> None:
        """Delete training job."""
        ...

    async def get_jobs_by_dataset(self, dataset_id: str) -> list[TrainingJob]:
        """Get all jobs for a specific dataset."""
        ...

    async def get_jobs_by_user(self, user_id: str) -> list[TrainingJob]:
        """Get all jobs for a specific user."""
        ...


class ModelTrainerProtocol(Protocol):
    """Protocol for model training implementations."""

    async def train(
        self, detector: Detector, dataset: Dataset, parameters: dict[str, Any]
    ) -> DetectionResult:
        """Train model with given parameters."""
        ...

    async def evaluate(
        self,
        detector: Detector,
        dataset: Dataset,
        validation_data: Dataset | None = None,
    ) -> dict[str, float]:
        """Evaluate trained model."""
        ...

    async def save_model(self, detector: Detector, path: str) -> None:
        """Save trained model to file."""
        ...

    async def load_model(self, detector: Detector, path: str) -> None:
        """Load trained model from file."""
        ...


class DatasetRepositoryProtocol(Protocol):
    """Protocol for dataset persistence and retrieval."""

    async def get_dataset(self, dataset_id: str) -> Dataset | None:
        """Get dataset by ID."""
        ...

    async def save_dataset(self, dataset: Dataset) -> None:
        """Save dataset."""
        ...

    async def list_datasets(self, limit: int = 100) -> list[Dataset]:
        """List available datasets."""
        ...


class ExperimentTrackerProtocol(Protocol):
    """Protocol for experiment tracking systems."""

    async def create_experiment(self, name: str, description: str = "") -> str:
        """Create new experiment and return experiment ID."""
        ...

    async def log_parameters(self, experiment_id: str, parameters: dict[str, Any]) -> None:
        """Log experiment parameters."""
        ...

    async def log_metrics(self, experiment_id: str, metrics: dict[str, float]) -> None:
        """Log experiment metrics."""
        ...

    async def log_artifact(self, experiment_id: str, artifact_path: str) -> None:
        """Log experiment artifact."""
        ...

    async def end_experiment(self, experiment_id: str) -> None:
        """End experiment logging."""
        ...


class OptimizationEngineProtocol(Protocol):
    """Protocol for hyperparameter optimization engines."""

    async def create_study(
        self,
        study_name: str,
        direction: str = "maximize",
        algorithm: str = "tpe",
    ) -> str:
        """Create optimization study and return study ID."""
        ...

    async def suggest_parameters(
        self, study_id: str, parameter_space: dict[str, Any]
    ) -> dict[str, Any]:
        """Suggest parameters for next trial."""
        ...

    async def report_trial_result(
        self, study_id: str, trial_id: str, score: float, parameters: dict[str, Any]
    ) -> None:
        """Report trial result to optimization engine."""
        ...

    async def get_best_parameters(self, study_id: str) -> dict[str, Any] | None:
        """Get best parameters from completed study."""
        ...

    async def get_study_history(self, study_id: str) -> list[dict[str, Any]]:
        """Get optimization history for study."""
        ...


class NotificationServiceProtocol(Protocol):
    """Protocol for training job notifications."""

    async def notify_job_started(self, job: TrainingJob) -> None:
        """Notify that training job has started."""
        ...

    async def notify_job_completed(self, job: TrainingJob) -> None:
        """Notify that training job has completed."""
        ...

    async def notify_job_failed(self, job: TrainingJob, error: str) -> None:
        """Notify that training job has failed."""
        ...

    async def notify_progress_update(self, job: TrainingJob, progress: float) -> None:
        """Notify of training progress update."""
        ...


class ResourceManagerProtocol(Protocol):
    """Protocol for training resource management."""

    async def allocate_resources(
        self, job_id: str, cpu_cores: int | None = None, memory_gb: float | None = None, gpu_ids: list[int] | None = None
    ) -> dict[str, Any]:
        """Allocate resources for training job."""
        ...

    async def release_resources(self, job_id: str) -> None:
        """Release resources allocated to training job."""
        ...

    async def get_resource_usage(self, job_id: str) -> dict[str, float]:
        """Get current resource usage for training job."""
        ...

    async def check_resource_availability(
        self, cpu_cores: int | None = None, memory_gb: float | None = None, gpu_ids: list[int] | None = None
    ) -> bool:
        """Check if requested resources are available."""
        ...