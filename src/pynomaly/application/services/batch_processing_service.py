"""Batch processing service providing an application-level interface."""

import asyncio
import logging
import os
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Union

from ...infrastructure.batch.batch_processor import (
    BatchProcessor, BatchConfig, BatchEngine, BatchStatus, DataFormat,
    create_batch_processor
)
from ...domain.services.advanced_detection_service import DetectionAlgorithm


logger = logging.getLogger(__name__)


@dataclass
class JobSubmissionOptions:
    """Options for job submission."""
    name: str
    description: str
    input_path: str
    output_path: str
    detection_algorithm: DetectionAlgorithm = DetectionAlgorithm.ISOLATION_FOREST
    chunk_size: Optional[int] = None
    engine: Optional[BatchEngine] = None
    config_overrides: Optional[Dict[str, Any]] = None


@dataclass
class ProcessingHooks:
    """Hooks for processing events."""
    pre_processing: Optional[Callable[[Dict[str, Any]], Dict[str, Any]]] = None
    post_processing: Optional[Callable[[Dict[str, Any]], Dict[str, Any]]] = None
    on_job_submitted: Optional[Callable[[str, Dict[str, Any]], None]] = None
    on_job_started: Optional[Callable[[str, Dict[str, Any]], None]] = None
    on_job_completed: Optional[Callable[[str, Dict[str, Any]], None]] = None
    on_job_failed: Optional[Callable[[str, Dict[str, Any]], None]] = None
    on_job_cancelled: Optional[Callable[[str, Dict[str, Any]], None]] = None
    on_chunk_processed: Optional[Callable[[str, int, Dict[str, Any]], None]] = None
    on_progress_update: Optional[Callable[[str, float], None]] = None


class BatchProcessingService:
    """Service for batch processing in the application layer."""

    def __init__(self, config: Optional[BatchConfig] = None):
        """Initialize the batch processing service."""
        self.config = config or BatchConfig()
        self.processor = create_batch_processor(self.config)
        self.hooks = ProcessingHooks()
        self._active_jobs: Dict[str, Dict[str, Any]] = {}
        self._job_monitoring_task: Optional[asyncio.Task] = None
        self._monitoring_active = False

    def set_hooks(self, hooks: ProcessingHooks) -> None:
        """Set processing hooks."""
        self.hooks = hooks
        logger.info("Processing hooks updated")

    def get_file_size_mb(self, file_path: str) -> float:
        """Get file size in MB."""
        try:
            return os.path.getsize(file_path) / (1024 * 1024)
        except OSError:
            logger.warning(f"Could not get size for {file_path}")
            return 0.0

    def choose_engine(self, data_size_mb: float, complexity: str = "medium") -> BatchEngine:
        """Choose the appropriate batch processing engine based on data characteristics."""
        if data_size_mb > 10000:  # >10GB
            return BatchEngine.DASK
        elif data_size_mb > 1000:  # >1GB
            if complexity == "high":
                return BatchEngine.DASK
            else:
                return BatchEngine.MULTIPROCESSING
        elif data_size_mb > 100:  # >100MB
            return BatchEngine.MULTIPROCESSING
        else:
            return BatchEngine.SEQUENTIAL

    def choose_chunk_size(self, data_size_mb: float, engine: BatchEngine) -> int:
        """Choose optimal chunk size based on data size and engine."""
        if engine == BatchEngine.DASK:
            if data_size_mb > 10000:
                return 100000  # Large chunks for very large datasets
            else:
                return 50000
        elif engine == BatchEngine.MULTIPROCESSING:
            if data_size_mb > 1000:
                return 25000
            else:
                return 10000
        else:
            return 5000  # Sequential processing

    async def submit_job(
        self,
        options: JobSubmissionOptions
    ) -> str:
        """Submit a batch processing job with comprehensive options."""
        
        # Pre-processing hook
        job_data = {
            'name': options.name,
            'description': options.description,
            'input_path': options.input_path,
            'output_path': options.output_path,
            'detection_algorithm': options.detection_algorithm,
            'submitted_at': datetime.now().isoformat()
        }
        
        if self.hooks.pre_processing:
            job_data = self.hooks.pre_processing(job_data)

        # Auto-configure engine and chunk size if not specified
        data_size_mb = self.get_file_size_mb(options.input_path)
        
        config = self.config
        if options.engine or options.chunk_size or options.config_overrides:
            # Create a new config with overrides
            config = BatchConfig(
                engine=options.engine or self.choose_engine(data_size_mb),
                chunk_size=options.chunk_size or self.choose_chunk_size(data_size_mb, options.engine or self.choose_engine(data_size_mb)),
                detection_algorithm=options.detection_algorithm,
                **(options.config_overrides or {})
            )

        # Submit job
        job_id = await self.processor.submit_job(
            name=options.name,
            description=options.description,
            input_path=options.input_path,
            output_path=options.output_path,
            config=config
        )

        # Track active job
        self._active_jobs[job_id] = {
            'options': options,
            'config': config,
            'submitted_at': datetime.now(),
            'data_size_mb': data_size_mb
        }

        # Start monitoring if not already active
        if not self._monitoring_active:
            await self.start_job_monitoring()

        logger.info(f"Job {job_id} submitted with engine {config.engine.value} and chunk size {config.chunk_size}")

        # Call submission hook
        if self.hooks.on_job_submitted:
            self.hooks.on_job_submitted(job_id, job_data)

        return job_id

    async def submit_simple_job(
        self,
        name: str,
        description: str,
        input_path: str,
        output_path: str,
        detection_algorithm: DetectionAlgorithm = DetectionAlgorithm.ISOLATION_FOREST
    ) -> str:
        """Submit a simple batch processing job (convenience method)."""
        options = JobSubmissionOptions(
            name=name,
            description=description,
            input_path=input_path,
            output_path=output_path,
            detection_algorithm=detection_algorithm
        )
        return await self.submit_job(options)

    async def get_job_status(self, job_id: str) -> Optional[Dict[str, Any]]:
        """Retrieve the status of a specific job."""
        status = await self.processor.get_job_status(job_id)
        if status:
            # Add additional metadata from our tracking
            if job_id in self._active_jobs:
                job_info = self._active_jobs[job_id]
                status['data_size_mb'] = job_info['data_size_mb']
                status['submitted_at'] = job_info['submitted_at'].isoformat()
                status['engine_chosen'] = job_info['config'].engine.value
                status['chunk_size_chosen'] = job_info['config'].chunk_size
            
            logger.debug(f"Status for job {job_id}: {status['status']} ({status.get('progress_percentage', 0):.1f}%)")
        return status

    async def cancel_job(self, job_id: str) -> bool:
        """Cancel a job by ID."""
        cancelled = await self.processor.cancel_job(job_id)
        
        if cancelled:
            logger.info(f"Job {job_id} cancelled")
            
            # Remove from active jobs
            if job_id in self._active_jobs:
                del self._active_jobs[job_id]
            
            # Call cancellation hook
            if self.hooks.on_job_cancelled:
                self.hooks.on_job_cancelled(job_id, {'cancelled_at': datetime.now().isoformat()})

        return cancelled

    async def list_jobs(self, status: Optional[BatchStatus] = None) -> List[Dict[str, Any]]:
        """List all jobs with optional status filter."""
        jobs = await self.processor.list_jobs(status)
        
        # Enhance with our tracking data
        for job in jobs:
            job_id = job['job_id']
            if job_id in self._active_jobs:
                job_info = self._active_jobs[job_id]
                job['data_size_mb'] = job_info['data_size_mb']
                job['engine_chosen'] = job_info['config'].engine.value
                job['chunk_size_chosen'] = job_info['config'].chunk_size
        
        return jobs

    async def get_job_metrics(self, job_id: str) -> Dict[str, Any]:
        """Get comprehensive metrics for a job."""
        status = await self.get_job_status(job_id)
        if not status:
            return {}
        
        metrics = {
            'basic_metrics': {
                'job_id': job_id,
                'status': status['status'],
                'progress_percentage': status.get('progress_percentage', 0),
                'total_samples': status.get('total_samples', 0),
                'total_anomalies': status.get('total_anomalies', 0),
                'execution_time': status.get('execution_time', 0)
            },
            'performance_metrics': {
                'samples_per_second': 0,
                'anomaly_detection_rate': 0,
                'memory_efficiency': 0
            },
            'resource_metrics': {
                'engine': status.get('engine_chosen', 'unknown'),
                'chunk_size': status.get('chunk_size_chosen', 0),
                'data_size_mb': status.get('data_size_mb', 0)
            }
        }
        
        # Calculate derived metrics
        if status.get('execution_time', 0) > 0:
            metrics['performance_metrics']['samples_per_second'] = status.get('total_samples', 0) / status['execution_time']
        
        if status.get('total_samples', 0) > 0:
            metrics['performance_metrics']['anomaly_detection_rate'] = status.get('total_anomalies', 0) / status['total_samples']
        
        return metrics

    async def start_job_monitoring(self) -> None:
        """Start background job monitoring."""
        if self._monitoring_active:
            return
        
        self._monitoring_active = True
        self._job_monitoring_task = asyncio.create_task(self._monitoring_loop())
        logger.info("Started job monitoring")

    async def stop_job_monitoring(self) -> None:
        """Stop background job monitoring."""
        if not self._monitoring_active:
            return
        
        self._monitoring_active = False
        if self._job_monitoring_task:
            self._job_monitoring_task.cancel()
            try:
                await self._job_monitoring_task
            except asyncio.CancelledError:
                pass
        
        logger.info("Stopped job monitoring")

    async def _monitoring_loop(self) -> None:
        """Background monitoring loop for job status updates."""
        while self._monitoring_active:
            try:
                await self._check_job_status_updates()
                await asyncio.sleep(10)  # Check every 10 seconds
            except Exception as e:
                logger.error(f"Error in job monitoring loop: {e}")
                await asyncio.sleep(30)  # Wait longer on error

    async def _check_job_status_updates(self) -> None:
        """Check for job status updates and trigger hooks."""
        for job_id in list(self._active_jobs.keys()):
            try:
                status = await self.get_job_status(job_id)
                if not status:
                    continue
                
                current_status = status['status']
                progress = status.get('progress_percentage', 0)
                
                # Check for status changes and trigger appropriate hooks
                if current_status == 'running' and self.hooks.on_job_started:
                    self.hooks.on_job_started(job_id, status)
                
                elif current_status == 'completed':
                    if self.hooks.on_job_completed:
                        self.hooks.on_job_completed(job_id, status)
                    
                    # Post-processing hook
                    if self.hooks.post_processing:
                        processed_status = self.hooks.post_processing(status)
                        if processed_status:
                            status = processed_status
                    
                    # Remove from active jobs
                    del self._active_jobs[job_id]
                    
                elif current_status == 'failed':
                    if self.hooks.on_job_failed:
                        self.hooks.on_job_failed(job_id, status)
                    
                    # Remove from active jobs
                    del self._active_jobs[job_id]
                
                # Progress update hook
                if self.hooks.on_progress_update:
                    self.hooks.on_progress_update(job_id, progress)
                    
            except Exception as e:
                logger.error(f"Error checking status for job {job_id}: {e}")

    async def get_system_metrics(self) -> Dict[str, Any]:
        """Get system metrics for the batch processing service."""
        processor_metrics = await self.processor.get_system_metrics()
        
        service_metrics = {
            'service_metrics': {
                'active_jobs': len(self._active_jobs),
                'monitoring_active': self._monitoring_active,
                'hooks_configured': {
                    'pre_processing': self.hooks.pre_processing is not None,
                    'post_processing': self.hooks.post_processing is not None,
                    'on_job_submitted': self.hooks.on_job_submitted is not None,
                    'on_job_completed': self.hooks.on_job_completed is not None,
                    'on_job_failed': self.hooks.on_job_failed is not None,
                    'on_progress_update': self.hooks.on_progress_update is not None
                }
            },
            'processor_metrics': processor_metrics
        }
        
        return service_metrics

    async def cleanup_completed_jobs(self, older_than_days: int = 7) -> int:
        """Clean up completed jobs older than specified days."""
        return await self.processor.cleanup_completed_jobs(older_than_days)

    async def shutdown(self) -> None:
        """Shutdown the batch processing service."""
        logger.info("Shutting down batch processing service...")
        
        # Stop monitoring
        await self.stop_job_monitoring()
        
        # Shutdown processor
        await self.processor.shutdown()
        
        # Clear active jobs
        self._active_jobs.clear()
        
        logger.info("Batch processing service shutdown complete")


# Factory function
def create_batch_processing_service(config: Optional[BatchConfig] = None) -> BatchProcessingService:
    """Create a batch processing service with the given configuration."""
    return BatchProcessingService(config)

