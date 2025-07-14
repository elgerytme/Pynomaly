"""Batch Processing Orchestration Service.

This service provides comprehensive batch processing capabilities for large datasets,
including configurable batch sizes, progress tracking, error handling, and recovery.
"""

from __future__ import annotations

import asyncio
import logging
import time
import uuid
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Protocol, TypeVar, Union

import pandas as pd
from pydantic import BaseModel, Field

from ...domain.entities.dataset import Dataset
from ...infrastructure.config.settings import Settings
from ...infrastructure.messaging.services.queue_manager import QueueManager
from ...infrastructure.messaging.models.tasks import Task, TaskType, TaskStatus
from ...infrastructure.resilience.retry import retry_async, DATABASE_RETRY_POLICY

logger = logging.getLogger(__name__)

T = TypeVar('T')


class BatchStatus(str, Enum):
    """Status of a batch job."""
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"
    PAUSED = "paused"
    RETRYING = "retrying"


class BatchPriority(str, Enum):
    """Priority levels for batch jobs."""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    URGENT = "urgent"


class ProgressCallback(Protocol):
    """Protocol for progress callback functions."""
    
    def __call__(self, 
                 batch_id: str,
                 current: int, 
                 total: int, 
                 message: str = "") -> None:
        """Called to report progress."""
        ...


class BatchProcessor(Protocol):
    """Protocol for batch processing functions."""
    
    async def __call__(self, 
                      batch_data: Any,
                      batch_context: Dict[str, Any]) -> Any:
        """Process a single batch of data."""
        ...


class BatchConfig(BaseModel):
    """Configuration for batch processing jobs."""
    
    batch_size: int = Field(default=1000, ge=1, description="Number of items per batch")
    max_concurrent_batches: int = Field(default=4, ge=1, description="Maximum concurrent batches")
    memory_limit_mb: float = Field(default=1000.0, gt=0, description="Memory limit in MB")
    timeout_seconds: int = Field(default=3600, gt=0, description="Job timeout in seconds")
    retry_attempts: int = Field(default=3, ge=0, description="Number of retry attempts")
    retry_delay_seconds: float = Field(default=30.0, ge=0, description="Delay between retries")
    checkpoint_interval: int = Field(default=10, ge=1, description="Batches between checkpoints")
    enable_progress_tracking: bool = Field(default=True, description="Enable progress tracking")
    preserve_order: bool = Field(default=False, description="Preserve batch processing order")
    auto_optimize: bool = Field(default=True, description="Auto-optimize batch sizes")


class BatchMetrics(BaseModel):
    """Metrics for batch processing jobs."""
    
    total_batches: int = 0
    processed_batches: int = 0
    failed_batches: int = 0
    skipped_batches: int = 0
    total_items: int = 0
    processed_items: int = 0
    
    start_time: Optional[datetime] = None
    end_time: Optional[datetime] = None
    last_checkpoint: Optional[datetime] = None
    
    processing_rate_items_per_second: float = 0.0
    estimated_completion_time: Optional[datetime] = None
    memory_usage_mb: float = 0.0
    cpu_usage_percent: float = 0.0
    
    error_rate: float = 0.0
    success_rate: float = 0.0


class BatchCheckpoint(BaseModel):
    """Checkpoint data for resumable batch processing."""
    
    job_id: str
    last_processed_batch: int
    processed_items: int
    checkpoint_time: datetime
    checkpoint_data: Dict[str, Any] = Field(default_factory=dict)
    metrics_snapshot: BatchMetrics


class BatchJob(BaseModel):
    """Represents a batch processing job."""
    
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    name: str
    description: Optional[str] = None
    
    # Status and priority
    status: BatchStatus = BatchStatus.PENDING
    priority: BatchPriority = BatchPriority.MEDIUM
    
    # Configuration
    config: BatchConfig = Field(default_factory=BatchConfig)
    
    # Data and processing
    input_data: Any = None
    processor_name: str
    processor_kwargs: Dict[str, Any] = Field(default_factory=dict)
    
    # Progress and metrics
    metrics: BatchMetrics = Field(default_factory=BatchMetrics)
    
    # Checkpointing and recovery
    checkpoints: List[BatchCheckpoint] = Field(default_factory=list)
    resume_from_checkpoint: bool = False
    
    # Timestamps
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    
    # Error handling
    last_error: Optional[str] = None
    retry_count: int = 0
    
    def calculate_progress_percentage(self) -> float:
        """Calculate completion percentage."""
        if self.metrics.total_batches == 0:
            return 0.0
        return (self.metrics.processed_batches / self.metrics.total_batches) * 100.0
    
    def estimate_remaining_time(self) -> Optional[float]:
        """Estimate remaining processing time in seconds."""
        if (self.metrics.processing_rate_items_per_second == 0 or 
            self.metrics.total_items == 0):
            return None
        
        remaining_items = self.metrics.total_items - self.metrics.processed_items
        return remaining_items / self.metrics.processing_rate_items_per_second
    
    def update_metrics(self) -> None:
        """Update computed metrics."""
        # Calculate rates
        if self.metrics.start_time and self.metrics.processed_items > 0:
            elapsed = (datetime.now(timezone.utc) - self.metrics.start_time).total_seconds()
            if elapsed > 0:
                self.metrics.processing_rate_items_per_second = self.metrics.processed_items / elapsed
        
        # Calculate error rate
        if self.metrics.total_batches > 0:
            self.metrics.error_rate = self.metrics.failed_batches / self.metrics.total_batches
            self.metrics.success_rate = self.metrics.processed_batches / self.metrics.total_batches
        
        # Estimate completion time
        remaining_time = self.estimate_remaining_time()
        if remaining_time:
            self.metrics.estimated_completion_time = (
                datetime.now(timezone.utc) + 
                pd.Timedelta(seconds=remaining_time)
            )


class BatchProcessingService:
    """Service for orchestrating batch processing of large datasets."""
    
    def __init__(self, 
                 queue_manager: Optional[QueueManager] = None,
                 settings: Optional[Settings] = None):
        """Initialize the batch processing service.
        
        Args:
            queue_manager: Queue manager for distributed processing
            settings: Application settings
        """
        self.queue_manager = queue_manager
        self.settings = settings or Settings()
        self.logger = logging.getLogger(__name__)
        
        # Registry of available processors
        self._processors: Dict[str, BatchProcessor] = {}
        
        # Active jobs
        self._active_jobs: Dict[str, BatchJob] = {}
        
        # Progress callbacks
        self._progress_callbacks: Dict[str, List[ProgressCallback]] = {}
        
        # Checkpoint storage
        self._checkpoint_storage: Dict[str, List[BatchCheckpoint]] = {}
    
    def register_processor(self, name: str, processor: BatchProcessor) -> None:
        """Register a batch processor function.
        
        Args:
            name: Name of the processor
            processor: Processor function
        """
        self._processors[name] = processor
        self.logger.info(f"Registered batch processor: {name}")
    
    def add_progress_callback(self, job_id: str, callback: ProgressCallback) -> None:
        """Add a progress callback for a job.
        
        Args:
            job_id: Job ID
            callback: Progress callback function
        """
        if job_id not in self._progress_callbacks:
            self._progress_callbacks[job_id] = []
        self._progress_callbacks[job_id].append(callback)
    
    def _notify_progress(self, job_id: str, current: int, total: int, message: str = "") -> None:
        """Notify progress callbacks."""
        if job_id in self._progress_callbacks:
            for callback in self._progress_callbacks[job_id]:
                try:
                    callback(job_id, current, total, message)
                except Exception as e:
                    self.logger.warning(f"Progress callback failed: {e}")
    
    async def create_batch_job(self,
                              name: str,
                              data: Any,
                              processor_name: str,
                              config: Optional[BatchConfig] = None,
                              processor_kwargs: Optional[Dict[str, Any]] = None,
                              description: Optional[str] = None,
                              priority: BatchPriority = BatchPriority.MEDIUM) -> BatchJob:
        """Create a new batch processing job.
        
        Args:
            name: Job name
            data: Input data to process
            processor_name: Name of registered processor
            config: Batch processing configuration
            processor_kwargs: Additional processor arguments
            description: Job description
            priority: Job priority
            
        Returns:
            Created batch job
            
        Raises:
            ValueError: If processor is not registered
        """
        if processor_name not in self._processors:
            raise ValueError(f"Processor '{processor_name}' not registered")
        
        job = BatchJob(
            name=name,
            description=description,
            processor_name=processor_name,
            processor_kwargs=processor_kwargs or {},
            config=config or BatchConfig(),
            priority=priority,
            input_data=data
        )
        
        # Calculate total items and batches
        if hasattr(data, '__len__'):
            job.metrics.total_items = len(data)
            job.metrics.total_batches = (
                (job.metrics.total_items + job.config.batch_size - 1) // 
                job.config.batch_size
            )
        
        self._active_jobs[job.id] = job
        
        self.logger.info(
            f"Created batch job '{name}' (ID: {job.id}) with "
            f"{job.metrics.total_items} items in {job.metrics.total_batches} batches"
        )
        
        return job
    
    async def start_batch_job(self, job_id: str) -> None:
        """Start processing a batch job.
        
        Args:
            job_id: Job ID to start
            
        Raises:
            ValueError: If job not found
        """
        if job_id not in self._active_jobs:
            raise ValueError(f"Job {job_id} not found")
        
        job = self._active_jobs[job_id]
        
        if job.status != BatchStatus.PENDING:
            raise ValueError(f"Job {job_id} is not in pending state")
        
        job.status = BatchStatus.RUNNING
        job.started_at = datetime.now(timezone.utc)
        job.metrics.start_time = job.started_at
        
        self.logger.info(f"Starting batch job {job_id}")
        
        try:
            await self._process_job(job)
        except Exception as e:
            self.logger.error(f"Batch job {job_id} failed: {e}")
            job.status = BatchStatus.FAILED
            job.last_error = str(e)
            raise
    
    async def _process_job(self, job: BatchJob) -> None:
        """Process a batch job."""
        processor = self._processors[job.processor_name]
        
        # Create batches
        batches = self._create_batches(job.input_data, job.config.batch_size)
        
        # Process batches with concurrency control
        semaphore = asyncio.Semaphore(job.config.max_concurrent_batches)
        
        async def process_batch_with_semaphore(batch_index: int, batch_data: Any) -> Any:
            async with semaphore:
                return await self._process_single_batch(job, batch_index, batch_data, processor)
        
        # Create processing tasks
        tasks = [
            process_batch_with_semaphore(i, batch)
            for i, batch in enumerate(batches)
        ]
        
        # Process batches
        if job.config.preserve_order:
            # Process in order
            for task in tasks:
                await task
        else:
            # Process concurrently
            await asyncio.gather(*tasks, return_exceptions=True)
        
        # Finalize job
        job.status = BatchStatus.COMPLETED
        job.completed_at = datetime.now(timezone.utc)
        job.metrics.end_time = job.completed_at
        job.update_metrics()
        
        self.logger.info(f"Completed batch job {job.id}")
    
    @retry_async(DATABASE_RETRY_POLICY)
    async def _process_single_batch(self,
                                   job: BatchJob,
                                   batch_index: int,
                                   batch_data: Any,
                                   processor: BatchProcessor) -> Any:
        """Process a single batch with error handling and retry logic."""
        batch_context = {
            'job_id': job.id,
            'batch_index': batch_index,
            'batch_size': len(batch_data) if hasattr(batch_data, '__len__') else job.config.batch_size,
            'total_batches': job.metrics.total_batches,
            **job.processor_kwargs
        }
        
        try:
            self.logger.debug(f"Processing batch {batch_index + 1}/{job.metrics.total_batches}")
            
            # Process the batch
            result = await processor(batch_data, batch_context)
            
            # Update metrics
            job.metrics.processed_batches += 1
            job.metrics.processed_items += batch_context['batch_size']
            job.update_metrics()
            
            # Notify progress
            if job.config.enable_progress_tracking:
                self._notify_progress(
                    job.id,
                    job.metrics.processed_batches,
                    job.metrics.total_batches,
                    f"Processed batch {batch_index + 1}"
                )
            
            # Create checkpoint if needed
            if (batch_index + 1) % job.config.checkpoint_interval == 0:
                await self._create_checkpoint(job, batch_index)
            
            return result
            
        except Exception as e:
            job.metrics.failed_batches += 1
            job.last_error = str(e)
            self.logger.error(f"Batch {batch_index + 1} failed: {e}")
            raise
    
    def _create_batches(self, data: Any, batch_size: int) -> List[Any]:
        """Create batches from input data."""
        if isinstance(data, pd.DataFrame):
            return [data.iloc[i:i + batch_size] for i in range(0, len(data), batch_size)]
        elif isinstance(data, list):
            return [data[i:i + batch_size] for i in range(0, len(data), batch_size)]
        elif hasattr(data, '__iter__'):
            # For other iterables
            batches = []
            iterator = iter(data)
            while True:
                batch = list(itertools.islice(iterator, batch_size))
                if not batch:
                    break
                batches.append(batch)
            return batches
        else:
            # Single item
            return [data]
    
    async def _create_checkpoint(self, job: BatchJob, batch_index: int) -> None:
        """Create a checkpoint for the job."""
        checkpoint = BatchCheckpoint(
            job_id=job.id,
            last_processed_batch=batch_index,
            processed_items=job.metrics.processed_items,
            checkpoint_time=datetime.now(timezone.utc),
            metrics_snapshot=job.metrics.copy()
        )
        
        job.checkpoints.append(checkpoint)
        job.metrics.last_checkpoint = checkpoint.checkpoint_time
        
        # Store checkpoint
        if job.id not in self._checkpoint_storage:
            self._checkpoint_storage[job.id] = []
        self._checkpoint_storage[job.id].append(checkpoint)
        
        self.logger.debug(f"Created checkpoint for job {job.id} at batch {batch_index + 1}")
    
    async def pause_job(self, job_id: str) -> None:
        """Pause a running batch job.
        
        Args:
            job_id: Job ID to pause
        """
        if job_id not in self._active_jobs:
            raise ValueError(f"Job {job_id} not found")
        
        job = self._active_jobs[job_id]
        if job.status == BatchStatus.RUNNING:
            job.status = BatchStatus.PAUSED
            await self._create_checkpoint(job, job.metrics.processed_batches - 1)
            self.logger.info(f"Paused job {job_id}")
    
    async def resume_job(self, job_id: str) -> None:
        """Resume a paused batch job.
        
        Args:
            job_id: Job ID to resume
        """
        if job_id not in self._active_jobs:
            raise ValueError(f"Job {job_id} not found")
        
        job = self._active_jobs[job_id]
        if job.status == BatchStatus.PAUSED:
            job.status = BatchStatus.RUNNING
            job.resume_from_checkpoint = True
            self.logger.info(f"Resumed job {job_id}")
            await self._process_job(job)
    
    async def cancel_job(self, job_id: str) -> None:
        """Cancel a batch job.
        
        Args:
            job_id: Job ID to cancel
        """
        if job_id not in self._active_jobs:
            raise ValueError(f"Job {job_id} not found")
        
        job = self._active_jobs[job_id]
        job.status = BatchStatus.CANCELLED
        job.completed_at = datetime.now(timezone.utc)
        self.logger.info(f"Cancelled job {job_id}")
    
    def get_job_status(self, job_id: str) -> Optional[BatchJob]:
        """Get the status of a batch job.
        
        Args:
            job_id: Job ID
            
        Returns:
            Job status or None if not found
        """
        return self._active_jobs.get(job_id)
    
    def list_jobs(self, 
                  status_filter: Optional[BatchStatus] = None,
                  priority_filter: Optional[BatchPriority] = None) -> List[BatchJob]:
        """List batch jobs with optional filtering.
        
        Args:
            status_filter: Filter by status
            priority_filter: Filter by priority
            
        Returns:
            List of matching jobs
        """
        jobs = list(self._active_jobs.values())
        
        if status_filter:
            jobs = [job for job in jobs if job.status == status_filter]
        
        if priority_filter:
            jobs = [job for job in jobs if job.priority == priority_filter]
        
        return sorted(jobs, key=lambda j: j.created_at, reverse=True)
    
    async def cleanup_completed_jobs(self, 
                                   older_than_hours: int = 24) -> int:
        """Clean up completed jobs older than specified time.
        
        Args:
            older_than_hours: Remove jobs completed more than this many hours ago
            
        Returns:
            Number of jobs cleaned up
        """
        cutoff_time = datetime.now(timezone.utc) - pd.Timedelta(hours=older_than_hours)
        cleaned_up = 0
        
        jobs_to_remove = []
        for job_id, job in self._active_jobs.items():
            if (job.status in [BatchStatus.COMPLETED, BatchStatus.FAILED, BatchStatus.CANCELLED] and
                job.completed_at and job.completed_at < cutoff_time):
                jobs_to_remove.append(job_id)
        
        for job_id in jobs_to_remove:
            del self._active_jobs[job_id]
            if job_id in self._progress_callbacks:
                del self._progress_callbacks[job_id]
            if job_id in self._checkpoint_storage:
                del self._checkpoint_storage[job_id]
            cleaned_up += 1
        
        if cleaned_up > 0:
            self.logger.info(f"Cleaned up {cleaned_up} completed jobs")
        
        return cleaned_up


# Example batch processors
async def anomaly_detection_batch_processor(batch_data: pd.DataFrame, 
                                          batch_context: Dict[str, Any]) -> Dict[str, Any]:
    """Example batch processor for anomaly detection."""
    import asyncio
    
    # Simulate processing time
    await asyncio.sleep(0.1)
    
    algorithm = batch_context.get('algorithm', 'isolation_forest')
    
    # Simulate anomaly detection
    anomaly_count = len(batch_data) // 20  # 5% anomalies
    
    return {
        'batch_index': batch_context['batch_index'],
        'processed_rows': len(batch_data),
        'anomalies_detected': anomaly_count,
        'algorithm': algorithm,
        'processing_time': 0.1
    }


async def data_quality_batch_processor(batch_data: pd.DataFrame, 
                                     batch_context: Dict[str, Any]) -> Dict[str, Any]:
    """Example batch processor for data quality assessment."""
    import asyncio
    
    # Simulate processing time
    await asyncio.sleep(0.05)
    
    # Calculate basic quality metrics
    null_count = batch_data.isnull().sum().sum()
    duplicate_count = batch_data.duplicated().sum()
    
    return {
        'batch_index': batch_context['batch_index'],
        'processed_rows': len(batch_data),
        'null_values': int(null_count),
        'duplicates': int(duplicate_count),
        'quality_score': max(0, 1.0 - (null_count + duplicate_count) / (len(batch_data) * len(batch_data.columns)))
    }