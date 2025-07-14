"""Analysis Job repository interface."""

from abc import ABC, abstractmethod
from typing import List, Optional, Dict, Any
from datetime import datetime
from uuid import UUID

from ..entities.analysis_job import AnalysisJob, JobId, UserId, DatasetId


class IAnalysisJobRepository(ABC):
    """Repository interface for analysis job persistence."""
    
    @abstractmethod
    async def save(self, job: AnalysisJob) -> None:
        """Save an analysis job."""
        pass
    
    @abstractmethod
    async def get_by_id(self, job_id: JobId) -> Optional[AnalysisJob]:
        """Get job by ID."""
        pass
    
    @abstractmethod
    async def get_by_user_id(self, user_id: UserId) -> List[AnalysisJob]:
        """Get all jobs for a user."""
        pass
    
    @abstractmethod
    async def get_by_dataset_id(self, dataset_id: DatasetId) -> List[AnalysisJob]:
        """Get all jobs for a dataset."""
        pass
    
    @abstractmethod
    async def get_by_status(self, status: str) -> List[AnalysisJob]:
        """Get jobs by status."""
        pass
    
    @abstractmethod
    async def get_by_job_type(self, job_type: str) -> List[AnalysisJob]:
        """Get jobs by type."""
        pass
    
    @abstractmethod
    async def get_running_jobs(self) -> List[AnalysisJob]:
        """Get all currently running jobs."""
        pass
    
    @abstractmethod
    async def get_pending_jobs(self) -> List[AnalysisJob]:
        """Get all pending jobs."""
        pass
    
    @abstractmethod
    async def get_failed_jobs(self, since: Optional[datetime] = None) -> List[AnalysisJob]:
        """Get failed jobs, optionally since a date."""
        pass
    
    @abstractmethod
    async def get_completed_jobs(self, since: Optional[datetime] = None) -> List[AnalysisJob]:
        """Get completed jobs, optionally since a date."""
        pass
    
    @abstractmethod
    async def get_jobs_by_priority(self, priority: str) -> List[AnalysisJob]:
        """Get jobs by priority level."""
        pass
    
    @abstractmethod
    async def get_scheduled_jobs(self) -> List[AnalysisJob]:
        """Get all scheduled jobs."""
        pass
    
    @abstractmethod
    async def get_jobs_by_date_range(self, start_date: datetime, end_date: datetime) -> List[AnalysisJob]:
        """Get jobs created within date range."""
        pass
    
    @abstractmethod
    async def get_jobs_with_dependencies(self) -> List[AnalysisJob]:
        """Get jobs that have dependencies."""
        pass
    
    @abstractmethod
    async def get_dependent_jobs(self, job_id: JobId) -> List[AnalysisJob]:
        """Get jobs that depend on the given job."""
        pass
    
    @abstractmethod
    async def search_jobs(self, query: str, filters: Optional[Dict[str, Any]] = None) -> List[AnalysisJob]:
        """Search jobs by query and optional filters."""
        pass
    
    @abstractmethod
    async def update_status(self, job_id: JobId, status: str, progress: Optional[float] = None) -> None:
        """Update job status and progress."""
        pass
    
    @abstractmethod
    async def update_progress(self, job_id: JobId, progress: float, current_step: Optional[str] = None) -> None:
        """Update job progress."""
        pass
    
    @abstractmethod
    async def add_log_entry(self, job_id: JobId, log_entry: str, level: str = "INFO") -> None:
        """Add log entry to job."""
        pass
    
    @abstractmethod
    async def get_job_logs(self, job_id: JobId) -> List[Dict[str, Any]]:
        """Get job execution logs."""
        pass
    
    @abstractmethod
    async def cancel_job(self, job_id: JobId, reason: Optional[str] = None) -> None:
        """Cancel a running or pending job."""
        pass
    
    @abstractmethod
    async def retry_job(self, job_id: JobId) -> None:
        """Retry a failed job."""
        pass
    
    @abstractmethod
    async def delete(self, job_id: JobId) -> None:
        """Delete a job."""
        pass
    
    @abstractmethod
    async def list_all(self, limit: Optional[int] = None, offset: Optional[int] = None) -> List[AnalysisJob]:
        """List all jobs with pagination."""
        pass
    
    @abstractmethod
    async def count(self) -> int:
        """Count total number of jobs."""
        pass
    
    @abstractmethod
    async def get_job_statistics(self) -> Dict[str, Any]:
        """Get job execution statistics."""
        pass
    
    @abstractmethod
    async def cleanup_old_jobs(self, days_old: int) -> int:
        """Clean up jobs older than specified days."""
        pass