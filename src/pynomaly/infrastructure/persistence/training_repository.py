"""
Training Repository Implementation

Persistent storage implementation for training jobs, optimization trials,
and training history with support for multiple storage backends including
in-memory, file-based, and database storage.
"""

import asyncio
import json
import logging
import pickle
from abc import ABC, abstractmethod
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional
from uuid import UUID

from pynomaly.domain.entities.optimization_trial import OptimizationTrial, TrialStatus
from pynomaly.domain.entities.training_job import (
    TrainingJob,
    TrainingPriority,
    TrainingStatus,
)
from pynomaly.shared.exceptions import RepositoryError

logger = logging.getLogger(__name__)


class TrainingRepositoryProtocol(ABC):
    """Protocol for training job repositories."""

    @abstractmethod
    async def save_job(self, job: TrainingJob) -> None:
        """Save a training job."""
        pass

    @abstractmethod
    async def get_job(self, job_id: str) -> Optional[TrainingJob]:
        """Get a training job by ID."""
        pass

    @abstractmethod
    async def update_job(self, job: TrainingJob) -> None:
        """Update a training job."""
        pass

    @abstractmethod
    async def delete_job(self, job_id: str) -> bool:
        """Delete a training job."""
        pass

    @abstractmethod
    async def list_jobs(
        self,
        dataset_id: Optional[str] = None,
        status: Optional[TrainingStatus] = None,
        priority: Optional[TrainingPriority] = None,
        limit: int = 100,
        offset: int = 0,
    ) -> List[TrainingJob]:
        """List training jobs with optional filtering."""
        pass

    @abstractmethod
    async def get_active_jobs(self) -> List[TrainingJob]:
        """Get all active (running or pending) training jobs."""
        pass

    @abstractmethod
    async def get_jobs_by_detector(self, detector_id: UUID) -> List[TrainingJob]:
        """Get all training jobs for a specific detector."""
        pass


class OptimizationTrialRepositoryProtocol(ABC):
    """Protocol for optimization trial repositories."""

    @abstractmethod
    async def save_trial(self, trial: OptimizationTrial) -> None:
        """Save an optimization trial."""
        pass

    @abstractmethod
    async def get_trial(self, trial_id: int) -> Optional[OptimizationTrial]:
        """Get an optimization trial by ID."""
        pass

    @abstractmethod
    async def update_trial(self, trial: OptimizationTrial) -> None:
        """Update an optimization trial."""
        pass

    @abstractmethod
    async def get_trials_by_job(self, job_id: str) -> List[OptimizationTrial]:
        """Get all trials for a training job."""
        pass

    @abstractmethod
    async def get_trials_by_study(self, study_id: str) -> List[OptimizationTrial]:
        """Get all trials for an optimization study."""
        pass


class InMemoryTrainingRepository(TrainingRepositoryProtocol):
    """In-memory implementation of training repository for testing and development."""

    def __init__(self):
        self._jobs: Dict[str, TrainingJob] = {}
        self._lock = asyncio.Lock()

    async def save_job(self, job: TrainingJob) -> None:
        """Save a training job."""
        async with self._lock:
            self._jobs[job.id] = job
            logger.debug(f"Saved training job {job.id}")

    async def get_job(self, job_id: str) -> Optional[TrainingJob]:
        """Get a training job by ID."""
        return self._jobs.get(job_id)

    async def update_job(self, job: TrainingJob) -> None:
        """Update a training job."""
        async with self._lock:
            if job.id in self._jobs:
                self._jobs[job.id] = job
                logger.debug(f"Updated training job {job.id}")
            else:
                raise RepositoryError(f"Training job {job.id} not found for update")

    async def delete_job(self, job_id: str) -> bool:
        """Delete a training job."""
        async with self._lock:
            if job_id in self._jobs:
                del self._jobs[job_id]
                logger.debug(f"Deleted training job {job_id}")
                return True
            return False

    async def list_jobs(
        self,
        dataset_id: Optional[str] = None,
        status: Optional[TrainingStatus] = None,
        priority: Optional[TrainingPriority] = None,
        limit: int = 100,
        offset: int = 0,
    ) -> List[TrainingJob]:
        """List training jobs with optional filtering."""
        jobs = list(self._jobs.values())

        # Apply filters
        if dataset_id:
            jobs = [job for job in jobs if job.dataset_id == dataset_id]

        if status:
            jobs = [job for job in jobs if job.status == status]

        if priority:
            jobs = [job for job in jobs if job.priority == priority]

        # Sort by creation time (newest first)
        jobs.sort(key=lambda j: j.created_at, reverse=True)

        # Apply pagination
        return jobs[offset : offset + limit]

    async def get_active_jobs(self) -> List[TrainingJob]:
        """Get all active (running or pending) training jobs."""
        active_statuses = {TrainingStatus.PENDING, TrainingStatus.RUNNING}
        return [job for job in self._jobs.values() if job.status in active_statuses]

    async def get_jobs_by_detector(self, detector_id: UUID) -> List[TrainingJob]:
        """Get all training jobs for a specific detector."""
        # Note: This assumes detector_id is stored in job metadata
        # In a real implementation, this would be a proper field
        jobs = []
        for job in self._jobs.values():
            # Check if detector_id is in job metadata or config
            if (hasattr(job, "detector_id") and job.detector_id == detector_id) or (
                job.config and getattr(job.config, "detector_id", None) == detector_id
            ):
                jobs.append(job)

        jobs.sort(key=lambda j: j.created_at, reverse=True)
        return jobs


class InMemoryOptimizationTrialRepository(OptimizationTrialRepositoryProtocol):
    """In-memory implementation of optimization trial repository."""

    def __init__(self):
        self._trials: Dict[int, OptimizationTrial] = {}
        self._job_trials: Dict[str, List[int]] = {}  # job_id -> trial_ids
        self._study_trials: Dict[str, List[int]] = {}  # study_id -> trial_ids
        self._lock = asyncio.Lock()

    async def save_trial(self, trial: OptimizationTrial) -> None:
        """Save an optimization trial."""
        async with self._lock:
            self._trials[trial.trial_id] = trial

            # Update job mapping if metadata contains job_id
            if trial.metadata and trial.metadata.experiment_id:
                job_id = trial.metadata.experiment_id
                if job_id not in self._job_trials:
                    self._job_trials[job_id] = []
                if trial.trial_id not in self._job_trials[job_id]:
                    self._job_trials[job_id].append(trial.trial_id)

            # Update study mapping
            if trial.study_id:
                if trial.study_id not in self._study_trials:
                    self._study_trials[trial.study_id] = []
                if trial.trial_id not in self._study_trials[trial.study_id]:
                    self._study_trials[trial.study_id].append(trial.trial_id)

            logger.debug(f"Saved optimization trial {trial.trial_id}")

    async def get_trial(self, trial_id: int) -> Optional[OptimizationTrial]:
        """Get an optimization trial by ID."""
        return self._trials.get(trial_id)

    async def update_trial(self, trial: OptimizationTrial) -> None:
        """Update an optimization trial."""
        async with self._lock:
            if trial.trial_id in self._trials:
                self._trials[trial.trial_id] = trial
                logger.debug(f"Updated optimization trial {trial.trial_id}")
            else:
                raise RepositoryError(
                    f"Optimization trial {trial.trial_id} not found for update"
                )

    async def get_trials_by_job(self, job_id: str) -> List[OptimizationTrial]:
        """Get all trials for a training job."""
        trial_ids = self._job_trials.get(job_id, [])
        trials = [
            self._trials[trial_id] for trial_id in trial_ids if trial_id in self._trials
        ]
        trials.sort(key=lambda t: t.start_time)
        return trials

    async def get_trials_by_study(self, study_id: str) -> List[OptimizationTrial]:
        """Get all trials for an optimization study."""
        trial_ids = self._study_trials.get(study_id, [])
        trials = [
            self._trials[trial_id] for trial_id in trial_ids if trial_id in self._trials
        ]
        trials.sort(key=lambda t: t.start_time)
        return trials


class FileBasedTrainingRepository(TrainingRepositoryProtocol):
    """File-based implementation of training repository using JSON storage."""

    def __init__(self, storage_path: Path):
        self.storage_path = Path(storage_path)
        self.storage_path.mkdir(parents=True, exist_ok=True)
        self._jobs_file = self.storage_path / "training_jobs.json"
        self._lock = asyncio.Lock()

        # Load existing jobs
        self._jobs: Dict[str, TrainingJob] = {}
        asyncio.create_task(self._load_jobs())

    async def _load_jobs(self) -> None:
        """Load jobs from storage file."""
        if not self._jobs_file.exists():
            return

        try:
            with open(self._jobs_file, "r") as f:
                jobs_data = json.load(f)

            for job_id, job_data in jobs_data.items():
                try:
                    job = TrainingJob.from_dict(job_data)
                    self._jobs[job_id] = job
                except Exception as e:
                    logger.error(f"Failed to load training job {job_id}: {e}")

            logger.info(f"Loaded {len(self._jobs)} training jobs from storage")

        except Exception as e:
            logger.error(f"Failed to load training jobs: {e}")

    async def _save_jobs(self) -> None:
        """Save jobs to storage file."""
        try:
            jobs_data = {job_id: job.to_dict() for job_id, job in self._jobs.items()}

            # Write to temporary file first
            temp_file = self._jobs_file.with_suffix(".tmp")
            with open(temp_file, "w") as f:
                json.dump(jobs_data, f, indent=2, default=str)

            # Atomic rename
            temp_file.replace(self._jobs_file)

        except Exception as e:
            logger.error(f"Failed to save training jobs: {e}")
            raise RepositoryError(f"Failed to save training jobs: {e}")

    async def save_job(self, job: TrainingJob) -> None:
        """Save a training job."""
        async with self._lock:
            self._jobs[job.id] = job
            await self._save_jobs()
            logger.debug(f"Saved training job {job.id} to file storage")

    async def get_job(self, job_id: str) -> Optional[TrainingJob]:
        """Get a training job by ID."""
        return self._jobs.get(job_id)

    async def update_job(self, job: TrainingJob) -> None:
        """Update a training job."""
        async with self._lock:
            if job.id in self._jobs:
                self._jobs[job.id] = job
                await self._save_jobs()
                logger.debug(f"Updated training job {job.id} in file storage")
            else:
                raise RepositoryError(f"Training job {job.id} not found for update")

    async def delete_job(self, job_id: str) -> bool:
        """Delete a training job."""
        async with self._lock:
            if job_id in self._jobs:
                del self._jobs[job_id]
                await self._save_jobs()
                logger.debug(f"Deleted training job {job_id} from file storage")
                return True
            return False

    async def list_jobs(
        self,
        dataset_id: Optional[str] = None,
        status: Optional[TrainingStatus] = None,
        priority: Optional[TrainingPriority] = None,
        limit: int = 100,
        offset: int = 0,
    ) -> List[TrainingJob]:
        """List training jobs with optional filtering."""
        jobs = list(self._jobs.values())

        # Apply filters
        if dataset_id:
            jobs = [job for job in jobs if job.dataset_id == dataset_id]

        if status:
            jobs = [job for job in jobs if job.status == status]

        if priority:
            jobs = [job for job in jobs if job.priority == priority]

        # Sort by creation time (newest first)
        jobs.sort(key=lambda j: j.created_at, reverse=True)

        # Apply pagination
        return jobs[offset : offset + limit]

    async def get_active_jobs(self) -> List[TrainingJob]:
        """Get all active (running or pending) training jobs."""
        active_statuses = {TrainingStatus.PENDING, TrainingStatus.RUNNING}
        return [job for job in self._jobs.values() if job.status in active_statuses]

    async def get_jobs_by_detector(self, detector_id: UUID) -> List[TrainingJob]:
        """Get all training jobs for a specific detector."""
        jobs = []
        for job in self._jobs.values():
            # Check if detector_id is in job metadata or config
            if (hasattr(job, "detector_id") and job.detector_id == detector_id) or (
                job.config and getattr(job.config, "detector_id", None) == detector_id
            ):
                jobs.append(job)

        jobs.sort(key=lambda j: j.created_at, reverse=True)
        return jobs


class SQLTrainingRepository(TrainingRepositoryProtocol):
    """SQL database implementation of training repository (placeholder)."""

    def __init__(self, connection_string: str):
        self.connection_string = connection_string
        # In a real implementation, this would initialize database connection
        # For now, delegate to in-memory implementation
        self._in_memory_repo = InMemoryTrainingRepository()
        logger.warning(
            "SQLTrainingRepository is not implemented, using in-memory fallback"
        )

    async def save_job(self, job: TrainingJob) -> None:
        """Save a training job."""
        return await self._in_memory_repo.save_job(job)

    async def get_job(self, job_id: str) -> Optional[TrainingJob]:
        """Get a training job by ID."""
        return await self._in_memory_repo.get_job(job_id)

    async def update_job(self, job: TrainingJob) -> None:
        """Update a training job."""
        return await self._in_memory_repo.update_job(job)

    async def delete_job(self, job_id: str) -> bool:
        """Delete a training job."""
        return await self._in_memory_repo.delete_job(job_id)

    async def list_jobs(
        self,
        dataset_id: Optional[str] = None,
        status: Optional[TrainingStatus] = None,
        priority: Optional[TrainingPriority] = None,
        limit: int = 100,
        offset: int = 0,
    ) -> List[TrainingJob]:
        """List training jobs with optional filtering."""
        return await self._in_memory_repo.list_jobs(
            dataset_id, status, priority, limit, offset
        )

    async def get_active_jobs(self) -> List[TrainingJob]:
        """Get all active (running or pending) training jobs."""
        return await self._in_memory_repo.get_active_jobs()

    async def get_jobs_by_detector(self, detector_id: UUID) -> List[TrainingJob]:
        """Get all training jobs for a specific detector."""
        return await self._in_memory_repo.get_jobs_by_detector(detector_id)


class TrainingRepository:
    """
    Main training repository that provides a unified interface to different storage backends.

    Supports in-memory, file-based, and database storage with automatic fallback
    and configuration-based backend selection.
    """

    def __init__(self, storage_type: str = "memory", **kwargs):
        """
        Initialize training repository.

        Args:
            storage_type: Storage backend type ("memory", "file", "sql")
            **kwargs: Backend-specific configuration
        """
        self.storage_type = storage_type

        if storage_type == "memory":
            self._repo = InMemoryTrainingRepository()
        elif storage_type == "file":
            storage_path = kwargs.get("storage_path", "./data/training")
            self._repo = FileBasedTrainingRepository(storage_path)
        elif storage_type == "sql":
            connection_string = kwargs.get("connection_string", "sqlite:///training.db")
            self._repo = SQLTrainingRepository(connection_string)
        else:
            raise ValueError(f"Unknown storage type: {storage_type}")

        # Also create trial repository
        if storage_type == "memory":
            self._trial_repo = InMemoryOptimizationTrialRepository()
        else:
            # For now, use in-memory for trials regardless of main storage
            self._trial_repo = InMemoryOptimizationTrialRepository()

        logger.info(f"Initialized training repository with {storage_type} backend")

    # Delegate training job methods
    async def save_job(self, job: TrainingJob) -> None:
        return await self._repo.save_job(job)

    async def get_job(self, job_id: str) -> Optional[TrainingJob]:
        return await self._repo.get_job(job_id)

    async def update_job(self, job: TrainingJob) -> None:
        return await self._repo.update_job(job)

    async def delete_job(self, job_id: str) -> bool:
        return await self._repo.delete_job(job_id)

    async def list_jobs(
        self,
        dataset_id: Optional[str] = None,
        status: Optional[TrainingStatus] = None,
        priority: Optional[TrainingPriority] = None,
        limit: int = 100,
        offset: int = 0,
    ) -> List[TrainingJob]:
        return await self._repo.list_jobs(dataset_id, status, priority, limit, offset)

    async def get_active_jobs(self) -> List[TrainingJob]:
        return await self._repo.get_active_jobs()

    async def get_jobs_by_detector(self, detector_id: UUID) -> List[TrainingJob]:
        return await self._repo.get_jobs_by_detector(detector_id)

    # Trial repository methods
    async def save_trial(self, trial: OptimizationTrial) -> None:
        return await self._trial_repo.save_trial(trial)

    async def get_trial(self, trial_id: int) -> Optional[OptimizationTrial]:
        return await self._trial_repo.get_trial(trial_id)

    async def update_trial(self, trial: OptimizationTrial) -> None:
        return await self._trial_repo.update_trial(trial)

    async def get_trials_by_job(self, job_id: str) -> List[OptimizationTrial]:
        return await self._trial_repo.get_trials_by_job(job_id)

    async def get_trials_by_study(self, study_id: str) -> List[OptimizationTrial]:
        return await self._trial_repo.get_trials_by_study(study_id)

    async def cleanup_old_jobs(self, days: int = 30) -> int:
        """
        Clean up old training jobs.

        Args:
            days: Number of days to keep jobs

        Returns:
            Number of jobs deleted
        """
        from datetime import timedelta

        cutoff_date = datetime.utcnow() - timedelta(days=days)
        jobs = await self.list_jobs(limit=1000)

        deleted_count = 0
        for job in jobs:
            if job.created_at < cutoff_date and job.is_completed:
                if await self.delete_job(job.id):
                    deleted_count += 1

        logger.info(f"Cleaned up {deleted_count} old training jobs")
        return deleted_count

    async def get_statistics(self) -> Dict[str, Any]:
        """Get repository statistics."""
        jobs = await self.list_jobs(limit=1000)

        stats = {
            "total_jobs": len(jobs),
            "active_jobs": len([j for j in jobs if not j.is_completed]),
            "completed_jobs": len(
                [j for j in jobs if j.status == TrainingStatus.COMPLETED]
            ),
            "failed_jobs": len([j for j in jobs if j.status == TrainingStatus.FAILED]),
            "cancelled_jobs": len(
                [j for j in jobs if j.status == TrainingStatus.CANCELLED]
            ),
            "storage_type": self.storage_type,
        }

        if jobs:
            # Calculate average training time for completed jobs
            completed_jobs = [
                j for j in jobs if j.status == TrainingStatus.COMPLETED and j.duration
            ]
            if completed_jobs:
                avg_duration = sum(j.duration for j in completed_jobs) / len(
                    completed_jobs
                )
                stats["average_training_time_seconds"] = avg_duration

        return stats
