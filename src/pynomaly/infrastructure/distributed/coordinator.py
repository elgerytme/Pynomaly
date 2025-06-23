"""Detection coordinator for orchestrating distributed anomaly detection workflows."""

import asyncio
import logging
from typing import List, Dict, Any, Optional, Set
from dataclasses import dataclass
from datetime import datetime
import json
import uuid

from pynomaly.domain.entities import Detector, Dataset, DetectionResult, Anomaly
from pynomaly.domain.value_objects import AnomalyScore
from pynomaly.domain.exceptions import ProcessingError
from .manager import DistributedProcessingManager


logger = logging.getLogger(__name__)


@dataclass
class DistributedDetectionJob:
    """Represents a distributed detection job."""
    id: str
    detector: Detector
    dataset: Dataset
    chunks: List[Dataset]
    task_ids: List[str]
    results: Dict[str, DetectionResult]
    status: str = "pending"  # pending, running, completed, failed
    created_at: datetime = None
    completed_at: Optional[datetime] = None
    error: Optional[str] = None
    
    def __post_init__(self):
        if self.created_at is None:
            self.created_at = datetime.now()
    
    @property
    def progress(self) -> float:
        """Get job completion progress (0.0 to 1.0)."""
        if not self.task_ids:
            return 0.0
        return len(self.results) / len(self.task_ids)
    
    @property
    def is_complete(self) -> bool:
        """Check if all tasks are completed."""
        return len(self.results) == len(self.task_ids)


class DetectionCoordinator:
    """Coordinator for orchestrating distributed anomaly detection workflows."""
    
    def __init__(self, processing_manager: DistributedProcessingManager):
        self.processing_manager = processing_manager
        self.active_jobs: Dict[str, DistributedDetectionJob] = {}
        self.completed_jobs: Dict[str, DistributedDetectionJob] = {}
        
        # Job monitoring
        self._job_monitor: Optional[asyncio.Task] = None
        self._running = False
        
        logger.info("Detection coordinator initialized")
    
    async def start(self) -> None:
        """Start the detection coordinator."""
        if self._running:
            return
        
        self._running = True
        self._job_monitor = asyncio.create_task(self._monitor_jobs())
        
        logger.info("Detection coordinator started")
    
    async def stop(self) -> None:
        """Stop the detection coordinator."""
        if not self._running:
            return
        
        self._running = False
        
        if self._job_monitor:
            self._job_monitor.cancel()
            try:
                await self._job_monitor
            except asyncio.CancelledError:
                pass
        
        # Wait for active jobs to complete
        if self.active_jobs:
            logger.info(f"Waiting for {len(self.active_jobs)} active jobs to complete")
            try:
                await asyncio.wait_for(self._wait_for_jobs(), timeout=120)
            except asyncio.TimeoutError:
                logger.warning("Timeout waiting for jobs to complete")
        
        logger.info("Detection coordinator stopped")
    
    async def submit_distributed_detection(self,
                                         detector: Detector,
                                         dataset: Dataset,
                                         chunk_size: Optional[int] = None,
                                         priority: int = 5) -> str:
        """Submit a distributed detection job."""
        job_id = f"job_{uuid.uuid4().hex[:8]}"
        
        # Split dataset into chunks
        chunks = await self._split_dataset(dataset, chunk_size)
        
        # Create job
        job = DistributedDetectionJob(
            id=job_id,
            detector=detector,
            dataset=dataset,
            chunks=chunks,
            task_ids=[],
            results={}
        )
        
        # Submit tasks to processing manager
        task_ids = []
        for chunk in chunks:
            task_id = await self.processing_manager.submit_detection_task(
                detector=detector,
                dataset=chunk,
                priority=priority
            )
            task_ids.append(task_id)
        
        job.task_ids = task_ids
        job.status = "running"
        
        # Track job
        self.active_jobs[job_id] = job
        
        logger.info(f"Submitted distributed detection job {job_id} with {len(chunks)} chunks")
        return job_id
    
    async def get_job_status(self, job_id: str) -> Optional[Dict[str, Any]]:
        """Get status of a distributed detection job."""
        # Check active jobs
        if job_id in self.active_jobs:
            job = self.active_jobs[job_id]
            return {
                "id": job.id,
                "status": job.status,
                "progress": job.progress,
                "chunks": len(job.chunks),
                "completed_chunks": len(job.results),
                "created_at": job.created_at.isoformat(),
                "detector_id": job.detector.id,
                "dataset_id": job.dataset.id
            }
        
        # Check completed jobs
        if job_id in self.completed_jobs:
            job = self.completed_jobs[job_id]
            return {
                "id": job.id,
                "status": job.status,
                "progress": 1.0,
                "chunks": len(job.chunks),
                "completed_chunks": len(job.results),
                "created_at": job.created_at.isoformat(),
                "completed_at": job.completed_at.isoformat() if job.completed_at else None,
                "detector_id": job.detector.id,
                "dataset_id": job.dataset.id,
                "error": job.error
            }
        
        return None
    
    async def get_job_result(self, job_id: str) -> Optional[DetectionResult]:
        """Get the combined result of a distributed detection job."""
        job = self.completed_jobs.get(job_id)
        if not job or job.status != "completed":
            return None
        
        # Combine results from all chunks
        combined_result = await self._combine_detection_results(job)
        return combined_result
    
    async def cancel_job(self, job_id: str) -> bool:
        """Cancel a distributed detection job."""
        if job_id not in self.active_jobs:
            return False
        
        job = self.active_jobs[job_id]
        
        # TODO: Cancel tasks in processing manager
        job.status = "cancelled"
        
        # Move to completed jobs
        self.completed_jobs[job_id] = job
        del self.active_jobs[job_id]
        
        logger.info(f"Cancelled job {job_id}")
        return True
    
    async def list_jobs(self, status: Optional[str] = None) -> List[Dict[str, Any]]:
        """List all jobs with optional status filter."""
        jobs = []
        
        # Add active jobs
        for job in self.active_jobs.values():
            if status is None or job.status == status:
                jobs.append({
                    "id": job.id,
                    "status": job.status,
                    "progress": job.progress,
                    "created_at": job.created_at.isoformat(),
                    "detector_id": job.detector.id,
                    "dataset_id": job.dataset.id
                })
        
        # Add completed jobs
        for job in self.completed_jobs.values():
            if status is None or job.status == status:
                jobs.append({
                    "id": job.id,
                    "status": job.status,
                    "progress": 1.0,
                    "created_at": job.created_at.isoformat(),
                    "completed_at": job.completed_at.isoformat() if job.completed_at else None,
                    "detector_id": job.detector.id,
                    "dataset_id": job.dataset.id
                })
        
        # Sort by creation time (newest first)
        jobs.sort(key=lambda x: x["created_at"], reverse=True)
        return jobs
    
    async def get_system_metrics(self) -> Dict[str, Any]:
        """Get coordinator system metrics."""
        return {
            "jobs": {
                "active": len(self.active_jobs),
                "completed": len(self.completed_jobs),
                "total": len(self.active_jobs) + len(self.completed_jobs)
            },
            "processing_manager": await self.processing_manager.get_system_metrics()
        }
    
    async def _monitor_jobs(self) -> None:
        """Monitor job progress and completion."""
        while self._running:
            try:
                completed_jobs = []
                
                for job_id, job in self.active_jobs.items():
                    # Check task completion status
                    await self._update_job_progress(job)
                    
                    # Check if job is complete
                    if job.is_complete:
                        job.status = "completed"
                        job.completed_at = datetime.now()
                        completed_jobs.append(job_id)
                        logger.info(f"Job {job_id} completed")
                
                # Move completed jobs
                for job_id in completed_jobs:
                    self.completed_jobs[job_id] = self.active_jobs[job_id]
                    del self.active_jobs[job_id]
                
                await asyncio.sleep(5)  # Check every 5 seconds
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Error in job monitor: {e}")
                await asyncio.sleep(5)
    
    async def _update_job_progress(self, job: DistributedDetectionJob) -> None:
        """Update job progress by checking task statuses."""
        for task_id in job.task_ids:
            if task_id not in job.results:
                # Check task status
                task_status = await self.processing_manager.get_task_status(task_id)
                
                if task_status and task_status["status"] == "completed":
                    # TODO: Get actual result from task
                    # For now, create a mock result
                    result = self._create_mock_result(job.detector, job.dataset, task_id)
                    job.results[task_id] = result
    
    async def _split_dataset(self, dataset: Dataset, chunk_size: Optional[int] = None) -> List[Dataset]:
        """Split dataset into chunks for distributed processing."""
        if chunk_size is None:
            # Determine optimal chunk size based on dataset size
            n_samples = dataset.n_samples
            if n_samples <= 1000:
                chunk_size = n_samples  # Single chunk for small datasets
            elif n_samples <= 10000:
                chunk_size = 2500  # 4 chunks for medium datasets
            else:
                chunk_size = 5000  # Multiple chunks for large datasets
        
        chunks = []
        n_samples = dataset.n_samples
        
        for i in range(0, n_samples, chunk_size):
            end_idx = min(i + chunk_size, n_samples)
            
            # Create chunk dataset (simplified)
            # TODO: Implement proper dataset slicing
            chunk_id = f"{dataset.id}_chunk_{i}_{end_idx}"
            chunk = Dataset(
                id=chunk_id,
                name=f"{dataset.name}_chunk_{len(chunks)}",
                data=dataset.data.iloc[i:end_idx] if hasattr(dataset.data, 'iloc') else dataset.data
            )
            chunks.append(chunk)
        
        logger.info(f"Split dataset {dataset.id} into {len(chunks)} chunks")
        return chunks
    
    async def _combine_detection_results(self, job: DistributedDetectionJob) -> DetectionResult:
        """Combine detection results from multiple chunks."""
        all_anomalies = []
        total_execution_time = 0.0
        
        # Combine anomalies from all chunks
        chunk_offset = 0
        for i, chunk in enumerate(job.chunks):
            task_id = job.task_ids[i]
            if task_id in job.results:
                chunk_result = job.results[task_id]
                
                # Adjust anomaly indices based on chunk offset
                for anomaly in chunk_result.anomalies:
                    adjusted_anomaly = Anomaly(
                        index=anomaly.index + chunk_offset,
                        score=anomaly.score,
                        timestamp=anomaly.timestamp,
                        feature_names=anomaly.feature_names
                    )
                    all_anomalies.append(adjusted_anomaly)
                
                total_execution_time += chunk_result.execution_time
            
            chunk_offset += chunk.n_samples
        
        # Create combined result
        combined_result = DetectionResult(
            id=f"combined_{job.id}",
            detector_id=job.detector.id,
            dataset_id=job.dataset.id,
            anomalies=all_anomalies,
            n_anomalies=len(all_anomalies),
            anomaly_rate=len(all_anomalies) / job.dataset.n_samples,
            threshold=0.5,  # TODO: Calculate proper threshold
            execution_time=total_execution_time
        )
        
        return combined_result
    
    def _create_mock_result(self, detector: Detector, dataset: Dataset, task_id: str) -> DetectionResult:
        """Create a mock detection result for testing."""
        # Create some mock anomalies
        anomalies = [
            Anomaly(
                index=i,
                score=AnomalyScore(0.8),
                timestamp=None,
                feature_names=["feature_1", "feature_2"]
            )
            for i in range(2)  # Mock 2 anomalies per chunk
        ]
        
        return DetectionResult(
            id=f"result_{task_id}",
            detector_id=detector.id,
            dataset_id=dataset.id,
            anomalies=anomalies,
            n_anomalies=len(anomalies),
            anomaly_rate=len(anomalies) / 100,  # Assume 100 samples per chunk
            threshold=0.5,
            execution_time=2.0
        )
    
    async def _wait_for_jobs(self) -> None:
        """Wait for all active jobs to complete."""
        while self.active_jobs:
            await asyncio.sleep(0.5)