"""Repository interface for ProfilingJob entities."""

from __future__ import annotations

from abc import ABC, abstractmethod
from datetime import datetime
from typing import Any, Optional, List, Dict
from uuid import UUID

from packages.core.domain.abstractions.repository_interface import RepositoryInterface
from ..entities.data_profile import ProfilingJob, ProfilingStatus
from ..value_objects.profiling_metadata import ProfilingStrategy, ExecutionPhase


class ProfilingJobRepository(RepositoryInterface[ProfilingJob], ABC):
    """Repository interface for profiling job persistence operations."""

    @abstractmethod
    async def find_by_status(self, status: ProfilingStatus) -> List[ProfilingJob]:
        """Find profiling jobs by status.
        
        Args:
            status: Profiling status to search for
            
        Returns:
            List of profiling jobs with the specified status
        """
        pass

    @abstractmethod
    async def find_active_jobs(self) -> List[ProfilingJob]:
        """Find all active profiling jobs.
        
        Returns:
            List of active profiling jobs (pending or in progress)
        """
        pass

    @abstractmethod
    async def find_by_dataset_id(self, dataset_id: str) -> List[ProfilingJob]:
        """Find profiling jobs for a specific dataset.
        
        Args:
            dataset_id: Dataset ID to search for
            
        Returns:
            List of profiling jobs for the dataset
        """
        pass

    @abstractmethod
    async def find_by_profile_id(self, profile_id: str) -> Optional[ProfilingJob]:
        """Find profiling job by profile ID.
        
        Args:
            profile_id: Profile ID to search for
            
        Returns:
            ProfilingJob if found, None otherwise
        """
        pass

    @abstractmethod
    async def find_by_date_range(
        self, start_date: datetime, end_date: datetime
    ) -> List[ProfilingJob]:
        """Find profiling jobs created within a date range.
        
        Args:
            start_date: Start of date range
            end_date: End of date range
            
        Returns:
            List of profiling jobs created within the date range
        """
        pass

    @abstractmethod
    async def find_long_running_jobs(self, threshold_minutes: int = 60) -> List[ProfilingJob]:
        """Find jobs that have been running longer than threshold.
        
        Args:
            threshold_minutes: Threshold in minutes
            
        Returns:
            List of long-running profiling jobs
        """
        pass

    @abstractmethod
    async def find_failed_jobs(
        self, since: Optional[datetime] = None
    ) -> List[ProfilingJob]:
        """Find failed profiling jobs.
        
        Args:
            since: Optional date to filter jobs since
            
        Returns:
            List of failed profiling jobs
        """
        pass

    @abstractmethod
    async def find_jobs_by_strategy(self, strategy: ProfilingStrategy) -> List[ProfilingJob]:
        """Find jobs by profiling strategy.
        
        Args:
            strategy: Profiling strategy to search for
            
        Returns:
            List of jobs using the specified strategy
        """
        pass

    @abstractmethod
    async def get_job_statistics(self) -> Dict[str, Any]:
        """Get statistics about profiling jobs.
        
        Returns:
            Dictionary containing job statistics
        """
        pass

    @abstractmethod
    async def get_performance_metrics(
        self, since: Optional[datetime] = None
    ) -> Dict[str, Any]:
        """Get performance metrics for profiling jobs.
        
        Args:
            since: Optional date to filter metrics since
            
        Returns:
            Dictionary containing performance metrics
        """
        pass

    @abstractmethod
    async def cancel_job(self, job_id: str, reason: str = "") -> bool:
        """Cancel a profiling job.
        
        Args:
            job_id: Job ID to cancel
            reason: Reason for cancellation
            
        Returns:
            True if job was cancelled, False otherwise
        """
        pass

    @abstractmethod
    async def retry_failed_job(self, job_id: str) -> Optional[ProfilingJob]:
        """Retry a failed profiling job.
        
        Args:
            job_id: Job ID to retry
            
        Returns:
            New ProfilingJob instance if retry was successful, None otherwise
        """
        pass

    @abstractmethod
    async def update_job_progress(
        self, 
        job_id: str, 
        progress_percentage: float,
        current_step: str,
        phase: Optional[ExecutionPhase] = None
    ) -> bool:
        """Update job progress information.
        
        Args:
            job_id: Job ID to update
            progress_percentage: Current progress percentage
            current_step: Description of current step
            phase: Current execution phase
            
        Returns:
            True if update was successful, False otherwise
        """
        pass

    @abstractmethod
    async def update_resource_usage(
        self,
        job_id: str,
        memory_usage_mb: float,
        cpu_usage_percentage: float
    ) -> bool:
        """Update resource usage for a job.
        
        Args:
            job_id: Job ID to update
            memory_usage_mb: Current memory usage in MB
            cpu_usage_percentage: Current CPU usage percentage
            
        Returns:
            True if update was successful, False otherwise
        """
        pass

    @abstractmethod
    async def add_job_warning(self, job_id: str, warning_message: str) -> bool:
        """Add warning to a profiling job.
        
        Args:
            job_id: Job ID to update
            warning_message: Warning message to add
            
        Returns:
            True if warning was added, False otherwise
        """
        pass

    @abstractmethod
    async def add_job_error(self, job_id: str, error_message: str) -> bool:
        """Add error to a profiling job.
        
        Args:
            job_id: Job ID to update
            error_message: Error message to add
            
        Returns:
            True if error was added, False otherwise
        """
        pass

    @abstractmethod
    async def complete_job(
        self,
        job_id: str,
        result_profile_id: str,
        execution_summary: Dict[str, Any]
    ) -> bool:
        """Mark job as completed with results.
        
        Args:
            job_id: Job ID to complete
            result_profile_id: ID of the resulting profile
            execution_summary: Summary of execution metrics
            
        Returns:
            True if job was completed, False otherwise
        """
        pass

    @abstractmethod
    async def fail_job(self, job_id: str, error_details: List[str]) -> bool:
        """Mark job as failed with error details.
        
        Args:
            job_id: Job ID to fail
            error_details: List of error messages
            
        Returns:
            True if job was marked as failed, False otherwise
        """
        pass

    @abstractmethod
    async def cleanup_old_jobs(self, older_than_days: int = 30) -> int:
        """Clean up old completed jobs.
        
        Args:
            older_than_days: Remove jobs older than this many days
            
        Returns:
            Number of jobs cleaned up
        """
        pass

    @abstractmethod
    async def get_job_queue_status(self) -> Dict[str, Any]:
        """Get status of the job queue.
        
        Returns:
            Dictionary containing queue status information
        """
        pass

    @abstractmethod
    async def find_jobs_needing_retry(
        self, max_retries: int = 3
    ) -> List[ProfilingJob]:
        """Find jobs that need to be retried.
        
        Args:
            max_retries: Maximum number of retry attempts
            
        Returns:
            List of jobs that should be retried
        """
        pass

    @abstractmethod
    async def get_job_dependency_chain(self, job_id: str) -> List[str]:
        """Get dependency chain for a job.
        
        Args:
            job_id: Job ID to analyze
            
        Returns:
            List of job IDs in dependency order
        """
        pass

    @abstractmethod
    async def find_duplicate_jobs(self, job_id: str) -> List[ProfilingJob]:
        """Find jobs with similar configuration that might be duplicates.
        
        Args:
            job_id: Reference job ID
            
        Returns:
            List of potentially duplicate jobs
        """
        pass

    @abstractmethod
    async def estimate_completion_time(self, job_id: str) -> Optional[datetime]:
        """Estimate completion time for a running job.
        
        Args:
            job_id: Job ID to analyze
            
        Returns:
            Estimated completion time if available
        """
        pass

    @abstractmethod
    async def get_resource_utilization_trends(
        self, days: int = 7
    ) -> Dict[str, List[float]]:
        """Get resource utilization trends over time.
        
        Args:
            days: Number of days to analyze
            
        Returns:
            Dictionary with resource utilization trends
        """
        pass

    @abstractmethod
    async def archive_completed_jobs(
        self, older_than_days: int = 90
    ) -> Dict[str, Any]:
        """Archive old completed jobs.
        
        Args:
            older_than_days: Archive jobs older than this many days
            
        Returns:
            Dictionary with archival results
        """
        pass

    @abstractmethod
    async def validate_job_integrity(self, job_id: str) -> Dict[str, Any]:
        """Validate job data integrity.
        
        Args:
            job_id: Job ID to validate
            
        Returns:
            Dictionary containing validation results
        """
        pass