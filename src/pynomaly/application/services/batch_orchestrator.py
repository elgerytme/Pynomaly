"""Batch Processing Orchestrator.

This module provides a high-level orchestration service that coordinates
batch processing jobs across different domains and packages.
"""

from __future__ import annotations

import asyncio
import logging
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Set

import pandas as pd
from pydantic import BaseModel

from ...infrastructure.messaging.services.queue_manager import QueueManager
from ...infrastructure.messaging.models.tasks import Task, TaskType, TaskStatus
from ...infrastructure.config.settings import Settings
from .batch_processing_service import (
    BatchProcessingService, BatchJob, BatchStatus, BatchPriority,
    anomaly_detection_batch_processor, data_quality_batch_processor
)
from .batch_configuration_manager import BatchConfigurationManager, BatchConfig

logger = logging.getLogger(__name__)


class BatchJobRequest(BaseModel):
    """Request to create a batch processing job."""
    
    name: str
    description: Optional[str] = None
    processor_type: str
    data_source: Any
    priority: BatchPriority = BatchPriority.MEDIUM
    config_overrides: Dict[str, Any] = {}
    processor_kwargs: Dict[str, Any] = {}
    
    # Scheduling options
    schedule_immediately: bool = True
    depends_on: List[str] = []  # Job IDs this job depends on


class BatchJobResult(BaseModel):
    """Result of a completed batch job."""
    
    job_id: str
    status: BatchStatus
    results: List[Dict[str, Any]] = []
    error_message: Optional[str] = None
    execution_time_seconds: float = 0.0
    items_processed: int = 0
    items_failed: int = 0


class JobDependencyManager:
    """Manages dependencies between batch jobs."""
    
    def __init__(self):
        self.dependencies: Dict[str, Set[str]] = {}  # job_id -> set of dependency job_ids
        self.dependents: Dict[str, Set[str]] = {}   # job_id -> set of jobs that depend on it
        self.completed_jobs: Set[str] = set()
    
    def add_dependency(self, job_id: str, depends_on: str) -> None:
        """Add a dependency relationship."""
        if job_id not in self.dependencies:
            self.dependencies[job_id] = set()
        self.dependencies[job_id].add(depends_on)
        
        if depends_on not in self.dependents:
            self.dependents[depends_on] = set()
        self.dependents[depends_on].add(job_id)
        
        logger.debug(f"Job {job_id} now depends on {depends_on}")
    
    def mark_completed(self, job_id: str) -> Set[str]:
        """Mark a job as completed and return jobs that can now run."""
        self.completed_jobs.add(job_id)
        
        ready_jobs = set()
        if job_id in self.dependents:
            for dependent_job in self.dependents[job_id]:
                if self.can_run(dependent_job):
                    ready_jobs.add(dependent_job)
        
        return ready_jobs
    
    def can_run(self, job_id: str) -> bool:
        """Check if a job can run (all dependencies completed)."""
        if job_id not in self.dependencies:
            return True
        
        return all(dep_id in self.completed_jobs for dep_id in self.dependencies[job_id])
    
    def get_pending_dependencies(self, job_id: str) -> Set[str]:
        """Get pending dependencies for a job."""
        if job_id not in self.dependencies:
            return set()
        
        return self.dependencies[job_id] - self.completed_jobs


class BatchOrchestrator:
    """High-level orchestrator for batch processing jobs."""
    
    def __init__(self,
                 queue_manager: Optional[QueueManager] = None,
                 settings: Optional[Settings] = None):
        """Initialize the batch orchestrator.
        
        Args:
            queue_manager: Queue manager for distributed processing
            settings: Application settings
        """
        self.settings = settings or Settings()
        self.logger = logging.getLogger(__name__)
        
        # Core services
        self.batch_service = BatchProcessingService(queue_manager, settings)
        self.config_manager = BatchConfigurationManager(settings)
        
        # Job management
        self.dependency_manager = JobDependencyManager()
        self.scheduled_jobs: Dict[str, BatchJobRequest] = {}
        self.running_jobs: Dict[str, BatchJob] = {}
        self.completed_jobs: Dict[str, BatchJobResult] = {}
        
        # Resource management
        self.max_concurrent_jobs = 5
        self.current_running_count = 0
        
        # Register standard processors
        self._register_standard_processors()
    
    def _register_standard_processors(self) -> None:
        """Register standard batch processors."""
        self.batch_service.register_processor(
            "anomaly_detection", 
            anomaly_detection_batch_processor
        )
        self.batch_service.register_processor(
            "data_quality", 
            data_quality_batch_processor
        )
        
        # Register data profiling processor
        async def data_profiling_batch_processor(batch_data: pd.DataFrame, 
                                               batch_context: Dict[str, Any]) -> Dict[str, Any]:
            """Batch processor for data profiling."""
            await asyncio.sleep(0.08)  # Simulate processing
            
            # Basic profiling metrics
            numeric_cols = batch_data.select_dtypes(include=['number']).columns
            text_cols = batch_data.select_dtypes(include=['object', 'string']).columns
            
            profile_data = {
                'batch_index': batch_context['batch_index'],
                'processed_rows': len(batch_data),
                'numeric_columns': len(numeric_cols),
                'text_columns': len(text_cols),
                'missing_values': batch_data.isnull().sum().sum(),
                'memory_usage_mb': batch_data.memory_usage(deep=True).sum() / (1024**2)
            }
            
            # Add basic statistics for numeric columns
            if len(numeric_cols) > 0:
                numeric_stats = batch_data[numeric_cols].describe()
                profile_data['numeric_statistics'] = numeric_stats.to_dict()
            
            return profile_data
        
        self.batch_service.register_processor(
            "data_profiling",
            data_profiling_batch_processor
        )
        
        # Register feature engineering processor
        async def feature_engineering_batch_processor(batch_data: pd.DataFrame, 
                                                    batch_context: Dict[str, Any]) -> Dict[str, Any]:
            """Batch processor for feature engineering."""
            await asyncio.sleep(0.15)  # Simulate processing
            
            features_created = []
            
            # Create simple engineered features
            numeric_cols = batch_data.select_dtypes(include=['number']).columns
            if len(numeric_cols) >= 2:
                # Create interaction features
                col1, col2 = numeric_cols[0], numeric_cols[1]
                batch_data[f'{col1}_{col2}_interaction'] = batch_data[col1] * batch_data[col2]
                features_created.append(f'{col1}_{col2}_interaction')
                
                # Create ratio features
                batch_data[f'{col1}_{col2}_ratio'] = batch_data[col1] / (batch_data[col2] + 1e-8)
                features_created.append(f'{col1}_{col2}_ratio')
            
            return {
                'batch_index': batch_context['batch_index'],
                'processed_rows': len(batch_data),
                'features_created': features_created,
                'original_features': len(batch_data.columns) - len(features_created),
                'total_features': len(batch_data.columns)
            }
        
        self.batch_service.register_processor(
            "feature_engineering",
            feature_engineering_batch_processor
        )
    
    async def submit_job(self, request: BatchJobRequest) -> str:
        """Submit a batch processing job.
        
        Args:
            request: Job request specification
            
        Returns:
            Job ID
        """
        # Validate processor type
        if request.processor_type not in self.batch_service._processors:
            raise ValueError(f"Unknown processor type: {request.processor_type}")
        
        # Create optimized configuration
        optimized_config = self.config_manager.create_optimized_config(
            data=request.data_source,
            processor_name=request.processor_type,
            **request.config_overrides
        )
        
        # Create batch job
        job = await self.batch_service.create_batch_job(
            name=request.name,
            data=request.data_source,
            processor_name=request.processor_type,
            config=optimized_config,
            processor_kwargs=request.processor_kwargs,
            description=request.description,
            priority=request.priority
        )
        
        # Handle dependencies
        for dep_job_id in request.depends_on:
            self.dependency_manager.add_dependency(job.id, dep_job_id)
        
        # Store request for scheduling
        self.scheduled_jobs[job.id] = request
        
        # Schedule immediately if requested and no dependencies
        if request.schedule_immediately and self.dependency_manager.can_run(job.id):
            await self._schedule_job(job.id)
        
        self.logger.info(f"Submitted batch job {job.id}: {request.name}")
        return job.id
    
    async def _schedule_job(self, job_id: str) -> None:
        """Schedule a job for execution."""
        if job_id not in self.scheduled_jobs:
            raise ValueError(f"Job {job_id} not found in scheduled jobs")
        
        if self.current_running_count >= self.max_concurrent_jobs:
            self.logger.info(f"Job {job_id} queued - max concurrent jobs reached")
            return
        
        # Check dependencies
        if not self.dependency_manager.can_run(job_id):
            pending_deps = self.dependency_manager.get_pending_dependencies(job_id)
            self.logger.info(f"Job {job_id} waiting for dependencies: {pending_deps}")
            return
        
        # Move to running
        job = self.batch_service._active_jobs[job_id]
        self.running_jobs[job_id] = job
        del self.scheduled_jobs[job_id]
        self.current_running_count += 1
        
        self.logger.info(f"Starting job {job_id}")
        
        # Start job in background
        asyncio.create_task(self._execute_job(job_id))
    
    async def _execute_job(self, job_id: str) -> None:
        """Execute a batch job and handle completion."""
        try:
            # Start the job
            await self.batch_service.start_batch_job(job_id)
            
            # Job completed successfully
            job = self.running_jobs[job_id]
            result = BatchJobResult(
                job_id=job_id,
                status=job.status,
                execution_time_seconds=(
                    (job.completed_at - job.started_at).total_seconds()
                    if job.completed_at and job.started_at else 0.0
                ),
                items_processed=job.metrics.processed_items,
                items_failed=job.metrics.failed_batches * job.config.batch_size
            )
            
        except Exception as e:
            # Job failed
            self.logger.error(f"Job {job_id} failed: {e}")
            job = self.running_jobs[job_id]
            result = BatchJobResult(
                job_id=job_id,
                status=BatchStatus.FAILED,
                error_message=str(e),
                execution_time_seconds=(
                    (datetime.now(timezone.utc) - job.started_at).total_seconds()
                    if job.started_at else 0.0
                ),
                items_processed=job.metrics.processed_items,
                items_failed=job.metrics.failed_batches * job.config.batch_size
            )
        
        finally:
            # Clean up
            if job_id in self.running_jobs:
                del self.running_jobs[job_id]
            self.current_running_count -= 1
            
            # Store result
            self.completed_jobs[job_id] = result
            
            # Mark as completed and check for dependent jobs
            ready_jobs = self.dependency_manager.mark_completed(job_id)
            
            # Schedule any jobs that are now ready
            for ready_job_id in ready_jobs:
                if ready_job_id in self.scheduled_jobs:
                    await self._schedule_job(ready_job_id)
            
            # Try to schedule next job from queue
            await self._schedule_next_job()
    
    async def _schedule_next_job(self) -> None:
        """Schedule the next available job from the queue."""
        if self.current_running_count >= self.max_concurrent_jobs:
            return
        
        # Find highest priority job that can run
        available_jobs = []
        for job_id, request in self.scheduled_jobs.items():
            if self.dependency_manager.can_run(job_id):
                job = self.batch_service._active_jobs[job_id]
                available_jobs.append((job_id, job.priority, job.created_at))
        
        if not available_jobs:
            return
        
        # Sort by priority (urgent > high > medium > low) then by creation time
        priority_order = {
            BatchPriority.URGENT: 0,
            BatchPriority.HIGH: 1, 
            BatchPriority.MEDIUM: 2,
            BatchPriority.LOW: 3
        }
        
        available_jobs.sort(key=lambda x: (priority_order[x[1]], x[2]))
        next_job_id = available_jobs[0][0]
        
        await self._schedule_job(next_job_id)
    
    def get_job_status(self, job_id: str) -> Dict[str, Any]:
        """Get comprehensive status of a job.
        
        Args:
            job_id: Job ID
            
        Returns:
            Job status information
        """
        # Check running jobs
        if job_id in self.running_jobs:
            job = self.running_jobs[job_id]
            return {
                'status': 'running',
                'progress_percentage': job.calculate_progress_percentage(),
                'processed_items': job.metrics.processed_items,
                'total_items': job.metrics.total_items,
                'estimated_completion': job.metrics.estimated_completion_time,
                'processing_rate': job.metrics.processing_rate_items_per_second,
                'error_rate': job.metrics.error_rate,
                'started_at': job.started_at,
                'last_checkpoint': job.metrics.last_checkpoint
            }
        
        # Check completed jobs
        if job_id in self.completed_jobs:
            result = self.completed_jobs[job_id]
            return {
                'status': result.status.value,
                'execution_time': result.execution_time_seconds,
                'items_processed': result.items_processed,
                'items_failed': result.items_failed,
                'error_message': result.error_message
            }
        
        # Check scheduled jobs
        if job_id in self.scheduled_jobs:
            pending_deps = self.dependency_manager.get_pending_dependencies(job_id)
            return {
                'status': 'scheduled',
                'pending_dependencies': list(pending_deps),
                'can_run': self.dependency_manager.can_run(job_id)
            }
        
        # Check if it exists in batch service
        job = self.batch_service.get_job_status(job_id)
        if job:
            return {
                'status': job.status.value,
                'progress_percentage': job.calculate_progress_percentage(),
                'processed_items': job.metrics.processed_items,
                'total_items': job.metrics.total_items
            }
        
        return {'status': 'not_found'}
    
    def list_jobs(self, 
                  status_filter: Optional[str] = None,
                  limit: Optional[int] = None) -> List[Dict[str, Any]]:
        """List all jobs with their status.
        
        Args:
            status_filter: Filter by status ('running', 'scheduled', 'completed')
            limit: Maximum number of jobs to return
            
        Returns:
            List of job information
        """
        jobs = []
        
        # Add running jobs
        if not status_filter or status_filter == 'running':
            for job_id, job in self.running_jobs.items():
                jobs.append({
                    'job_id': job_id,
                    'name': job.name,
                    'status': 'running',
                    'progress': job.calculate_progress_percentage(),
                    'created_at': job.created_at,
                    'started_at': job.started_at
                })
        
        # Add scheduled jobs
        if not status_filter or status_filter == 'scheduled':
            for job_id, request in self.scheduled_jobs.items():
                job = self.batch_service._active_jobs[job_id]
                jobs.append({
                    'job_id': job_id,
                    'name': request.name,
                    'status': 'scheduled',
                    'can_run': self.dependency_manager.can_run(job_id),
                    'created_at': job.created_at,
                    'priority': request.priority.value
                })
        
        # Add completed jobs
        if not status_filter or status_filter == 'completed':
            for job_id, result in self.completed_jobs.items():
                # Get original job info from batch service
                job = self.batch_service._active_jobs.get(job_id)
                jobs.append({
                    'job_id': job_id,
                    'name': job.name if job else f'Job {job_id}',
                    'status': result.status.value,
                    'execution_time': result.execution_time_seconds,
                    'items_processed': result.items_processed,
                    'completed_at': job.completed_at if job else None
                })
        
        # Sort by creation time (newest first)
        jobs.sort(key=lambda j: j.get('created_at', datetime.min), reverse=True)
        
        if limit:
            jobs = jobs[:limit]
        
        return jobs
    
    async def cancel_job(self, job_id: str) -> bool:
        """Cancel a batch job.
        
        Args:
            job_id: Job ID to cancel
            
        Returns:
            True if cancelled successfully
        """
        # Cancel running job
        if job_id in self.running_jobs:
            await self.batch_service.cancel_job(job_id)
            del self.running_jobs[job_id]
            self.current_running_count -= 1
            
            # Mark as completed to free up dependencies
            self.dependency_manager.mark_completed(job_id)
            
            self.logger.info(f"Cancelled running job {job_id}")
            await self._schedule_next_job()
            return True
        
        # Cancel scheduled job
        if job_id in self.scheduled_jobs:
            del self.scheduled_jobs[job_id]
            await self.batch_service.cancel_job(job_id)
            
            # Mark as completed to free up dependencies
            self.dependency_manager.mark_completed(job_id)
            
            self.logger.info(f"Cancelled scheduled job {job_id}")
            return True
        
        return False
    
    def get_system_status(self) -> Dict[str, Any]:
        """Get overall system status and recommendations.
        
        Returns:
            System status information
        """
        return {
            'running_jobs': len(self.running_jobs),
            'scheduled_jobs': len(self.scheduled_jobs),
            'completed_jobs': len(self.completed_jobs),
            'max_concurrent_jobs': self.max_concurrent_jobs,
            'system_recommendations': self.config_manager.get_system_recommendations()
        }
    
    async def cleanup_old_jobs(self, hours: int = 24) -> int:
        """Clean up old completed jobs.
        
        Args:
            hours: Remove jobs completed more than this many hours ago
            
        Returns:
            Number of jobs cleaned up
        """
        # Clean up from batch service
        cleaned_from_service = await self.batch_service.cleanup_completed_jobs(hours)
        
        # Clean up from orchestrator
        cutoff_time = datetime.now(timezone.utc) - pd.Timedelta(hours=hours)
        
        jobs_to_remove = []
        for job_id, result in self.completed_jobs.items():
            # Check if job is old enough to clean up
            job = self.batch_service._active_jobs.get(job_id)
            if job and job.completed_at and job.completed_at < cutoff_time:
                jobs_to_remove.append(job_id)
        
        for job_id in jobs_to_remove:
            del self.completed_jobs[job_id]
        
        total_cleaned = cleaned_from_service + len(jobs_to_remove)
        if total_cleaned > 0:
            self.logger.info(f"Cleaned up {total_cleaned} old jobs")
        
        return total_cleaned