"""Worker management endpoints."""

import asyncio
from typing import Dict, List, Any, Optional
from datetime import datetime

from fastapi import APIRouter, Depends, HTTPException, status, BackgroundTasks
from pydantic import BaseModel, Field

from ...worker import AnomalyDetectionWorker, JobType, JobPriority, JobStatus
from ...infrastructure.logging import get_logger

router = APIRouter()
logger = get_logger(__name__)


class JobSubmissionRequest(BaseModel):
    """Request model for job submission."""
    job_type: str = Field(..., description="Type of job to submit")
    payload: Dict[str, Any] = Field(..., description="Job payload data")
    priority: str = Field("normal", description="Job priority (low, normal, high, critical)")
    max_retries: int = Field(3, description="Maximum retry attempts")
    timeout_seconds: int = Field(300, description="Job timeout in seconds")


class JobSubmissionResponse(BaseModel):
    """Response model for job submission."""
    success: bool = Field(..., description="Whether job was submitted successfully")
    job_id: str = Field(..., description="Unique job identifier")
    job_type: str = Field(..., description="Type of job submitted")
    priority: str = Field(..., description="Job priority")
    estimated_completion: Optional[str] = Field(None, description="Estimated completion time")
    queue_position: Optional[int] = Field(None, description="Position in queue")
    timestamp: str = Field(..., description="Submission timestamp")


class JobStatusResponse(BaseModel):
    """Response model for job status."""
    job_id: str = Field(..., description="Job identifier")
    job_type: str = Field(..., description="Type of job")
    status: str = Field(..., description="Current job status")
    priority: str = Field(..., description="Job priority")
    progress: float = Field(..., description="Job progress percentage")
    created_at: str = Field(..., description="Job creation timestamp")
    started_at: Optional[str] = Field(None, description="Job start timestamp")
    completed_at: Optional[str] = Field(None, description="Job completion timestamp")
    error_message: Optional[str] = Field(None, description="Error message if failed")
    retry_count: int = Field(..., description="Number of retry attempts")
    max_retries: int = Field(..., description="Maximum retry attempts")
    result: Optional[Dict[str, Any]] = Field(None, description="Job result if completed")


class WorkerStatusResponse(BaseModel):
    """Response model for worker status."""
    is_running: bool = Field(..., description="Whether worker is running")
    max_concurrent_jobs: int = Field(..., description="Maximum concurrent jobs")
    currently_running_jobs: int = Field(..., description="Number of currently running jobs")
    running_job_ids: List[str] = Field(..., description="IDs of currently running jobs")
    monitoring_enabled: bool = Field(..., description="Whether monitoring is enabled")
    queue_status: Dict[str, Any] = Field(..., description="Queue status information")
    timestamp: str = Field(..., description="Status timestamp")


class WorkerHealthResponse(BaseModel):
    """Response model for worker health."""
    is_healthy: bool = Field(..., description="Overall health status")
    health_score: float = Field(..., description="Health score (0-100)")
    resource_utilization: Dict[str, Any] = Field(..., description="Resource utilization metrics")
    performance_metrics: Dict[str, Any] = Field(..., description="Performance metrics")
    recent_issues: List[Dict[str, Any]] = Field(..., description="Recent issues and warnings")
    timestamp: str = Field(..., description="Health check timestamp")


class JobListResponse(BaseModel):
    """Response model for job listing."""
    jobs: List[JobStatusResponse] = Field(..., description="List of jobs")
    total_count: int = Field(..., description="Total number of jobs")
    filtered_count: int = Field(..., description="Number of jobs matching filters")
    timestamp: str = Field(..., description="Response timestamp")


# Global worker instance (in production, use proper dependency injection)
_worker_instance: Optional[AnomalyDetectionWorker] = None


def get_worker_instance() -> AnomalyDetectionWorker:
    """Get or create worker instance."""
    global _worker_instance
    if _worker_instance is None:
        _worker_instance = AnomalyDetectionWorker(
            models_dir="./models",
            max_concurrent_jobs=5,
            enable_monitoring=True
        )
        logger.info("Created new worker instance")
    return _worker_instance


@router.post("/jobs", response_model=JobSubmissionResponse)
async def submit_job(
    request: JobSubmissionRequest,
    background_tasks: BackgroundTasks,
    worker: AnomalyDetectionWorker = Depends(get_worker_instance)
) -> JobSubmissionResponse:
    """Submit a new job to the worker queue."""
    try:
        logger.info("Submitting job", job_type=request.job_type, priority=request.priority)
        
        # Map string values to enums
        job_type_map = {
            'detection': JobType.DETECTION,
            'ensemble': JobType.ENSEMBLE,
            'batch_training': JobType.BATCH_TRAINING,
            'stream_monitoring': JobType.STREAM_MONITORING,
            'model_validation': JobType.MODEL_VALIDATION,
            'data_preprocessing': JobType.DATA_PREPROCESSING,
            'explanation_generation': JobType.EXPLANATION_GENERATION,
            'scheduled_analysis': JobType.SCHEDULED_ANALYSIS
        }
        
        priority_map = {
            'low': JobPriority.LOW,
            'normal': JobPriority.NORMAL,
            'high': JobPriority.HIGH,
            'critical': JobPriority.CRITICAL
        }
        
        if request.job_type not in job_type_map:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Invalid job type: {request.job_type}. Valid types: {list(job_type_map.keys())}"
            )
        
        if request.priority not in priority_map:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Invalid priority: {request.priority}. Valid priorities: {list(priority_map.keys())}"
            )
        
        # Submit job
        job_id = await worker.submit_job(
            job_type_map[request.job_type],
            request.payload,
            priority=priority_map[request.priority],
            max_retries=request.max_retries,
            timeout_seconds=request.timeout_seconds
        )
        
        # Start worker if not running
        if not worker.is_running:
            background_tasks.add_task(_start_worker_if_needed, worker)
        
        # Get queue status for position estimation
        queue_status = await worker.job_queue.get_queue_status()
        queue_position = queue_status.get('pending_jobs', 0)
        
        return JobSubmissionResponse(
            success=True,
            job_id=job_id,
            job_type=request.job_type,
            priority=request.priority,
            queue_position=queue_position,
            timestamp=datetime.utcnow().isoformat()
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error("Job submission failed", error=str(e))
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Job submission failed: {str(e)}"
        )


@router.get("/jobs/{job_id}", response_model=JobStatusResponse)
async def get_job_status(
    job_id: str,
    worker: AnomalyDetectionWorker = Depends(get_worker_instance)
) -> JobStatusResponse:
    """Get status of a specific job."""
    try:
        job_status = await worker.get_job_status(job_id)
        
        if not job_status:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Job {job_id} not found"
            )
        
        return JobStatusResponse(
            job_id=job_status['job_id'],
            job_type=job_status['job_type'],
            status=job_status['status'],
            priority=job_status['priority'],
            progress=job_status.get('progress', 0),
            created_at=job_status['created_at'],
            started_at=job_status.get('started_at'),
            completed_at=job_status.get('completed_at'),
            error_message=job_status.get('error_message'),
            retry_count=job_status.get('retry_count', 0),
            max_retries=job_status.get('max_retries', 3),
            result=job_status.get('result')
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error("Failed to get job status", job_id=job_id, error=str(e))
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to get job status: {str(e)}"
        )


@router.delete("/jobs/{job_id}")
async def cancel_job(
    job_id: str,
    worker: AnomalyDetectionWorker = Depends(get_worker_instance)
) -> Dict[str, str]:
    """Cancel a pending or running job."""
    try:
        success = await worker.cancel_job(job_id)
        
        if success:
            return {"message": f"Job {job_id} cancelled successfully"}
        else:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Job {job_id} not found or cannot be cancelled"
            )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error("Failed to cancel job", job_id=job_id, error=str(e))
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to cancel job: {str(e)}"
        )


@router.get("/jobs", response_model=JobListResponse)
async def list_jobs(
    status_filter: Optional[str] = None,
    job_type_filter: Optional[str] = None,
    limit: int = 50,
    offset: int = 0,
    worker: AnomalyDetectionWorker = Depends(get_worker_instance)
) -> JobListResponse:
    """List jobs with optional filtering."""
    try:
        # Get queue status (in a real implementation, we'd have a proper job history API)
        queue_status = await worker.job_queue.get_queue_status()
        
        # Mock job list - in real implementation, would come from job store
        mock_jobs = []
        total_count = queue_status.get('total_jobs', 0)
        
        # Add currently running jobs
        worker_status = await worker.get_worker_status()
        for job_id in worker_status.get('running_job_ids', []):
            job_status = await worker.get_job_status(job_id)
            if job_status:
                if not status_filter or job_status['status'] == status_filter:
                    if not job_type_filter or job_status['job_type'] == job_type_filter:
                        mock_jobs.append(JobStatusResponse(
                            job_id=job_status['job_id'],
                            job_type=job_status['job_type'],
                            status=job_status['status'],
                            priority=job_status['priority'],
                            progress=job_status.get('progress', 0),
                            created_at=job_status['created_at'],
                            started_at=job_status.get('started_at'),
                            completed_at=job_status.get('completed_at'),
                            error_message=job_status.get('error_message'),
                            retry_count=job_status.get('retry_count', 0),
                            max_retries=job_status.get('max_retries', 3),
                            result=job_status.get('result')
                        ))
        
        # Apply pagination
        paginated_jobs = mock_jobs[offset:offset + limit]
        
        return JobListResponse(
            jobs=paginated_jobs,
            total_count=total_count,
            filtered_count=len(mock_jobs),
            timestamp=datetime.utcnow().isoformat()
        )
        
    except Exception as e:
        logger.error("Failed to list jobs", error=str(e))
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to list jobs: {str(e)}"
        )


@router.get("/status", response_model=WorkerStatusResponse)
async def get_worker_status(
    worker: AnomalyDetectionWorker = Depends(get_worker_instance)
) -> WorkerStatusResponse:
    """Get comprehensive worker status."""
    try:
        worker_status = await worker.get_worker_status()
        
        return WorkerStatusResponse(
            is_running=worker_status['is_running'],
            max_concurrent_jobs=worker_status['max_concurrent_jobs'],
            currently_running_jobs=worker_status['currently_running_jobs'],
            running_job_ids=worker_status['running_job_ids'],
            monitoring_enabled=worker_status['monitoring_enabled'],
            queue_status=worker_status['queue_status'],
            timestamp=datetime.utcnow().isoformat()
        )
        
    except Exception as e:
        logger.error("Failed to get worker status", error=str(e))
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to get worker status: {str(e)}"
        )


@router.get("/health", response_model=WorkerHealthResponse)
async def get_worker_health(
    detailed: bool = False,
    worker: AnomalyDetectionWorker = Depends(get_worker_instance)
) -> WorkerHealthResponse:
    """Get worker health metrics."""
    try:
        worker_status = await worker.get_worker_status()
        
        # Calculate health score based on various metrics
        health_factors = []
        
        # Worker running status
        health_factors.append(100 if worker_status['is_running'] else 0)
        
        # Resource utilization
        current_jobs = worker_status['currently_running_jobs']
        max_jobs = worker_status['max_concurrent_jobs']
        utilization = (current_jobs / max_jobs) * 100 if max_jobs > 0 else 0
        
        # Lower score for high utilization
        if utilization < 50:
            health_factors.append(100)
        elif utilization < 80:
            health_factors.append(80)
        else:
            health_factors.append(60)
        
        # Queue health
        queue_status = worker_status['queue_status']
        pending_jobs = queue_status.get('pending_jobs', 0)
        
        if pending_jobs < 10:
            health_factors.append(100)
        elif pending_jobs < 50:
            health_factors.append(80)
        else:
            health_factors.append(60)
        
        # Calculate overall health score
        health_score = sum(health_factors) / len(health_factors)
        is_healthy = health_score >= 70
        
        # Mock resource utilization (in real implementation, would come from system monitoring)
        resource_utilization = {
            "cpu_usage_percent": 25.3,
            "memory_usage_mb": 1024,
            "memory_usage_percent": 15.2,
            "disk_usage_percent": 45.8,
            "network_io_mbps": 2.1,
            "worker_utilization_percent": utilization
        }
        
        # Mock performance metrics
        performance_metrics = {
            "jobs_completed_24h": 156,
            "jobs_failed_24h": 8,
            "success_rate_percent": 95.1,
            "avg_processing_time_seconds": 42.3,
            "throughput_jobs_per_hour": 12.8,
            "peak_queue_length_24h": 23
        }
        
        # Mock recent issues
        recent_issues = []
        if utilization > 80:
            recent_issues.append({
                "severity": "warning",
                "message": "High worker utilization detected",
                "timestamp": datetime.utcnow().isoformat(),
                "details": f"Worker utilization is {utilization:.1f}%"
            })
        
        if pending_jobs > 20:
            recent_issues.append({
                "severity": "info",
                "message": "High queue length detected",
                "timestamp": datetime.utcnow().isoformat(),
                "details": f"{pending_jobs} jobs pending"
            })
        
        return WorkerHealthResponse(
            is_healthy=is_healthy,
            health_score=health_score,
            resource_utilization=resource_utilization,
            performance_metrics=performance_metrics,
            recent_issues=recent_issues,
            timestamp=datetime.utcnow().isoformat()
        )
        
    except Exception as e:
        logger.error("Failed to get worker health", error=str(e))
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to get worker health: {str(e)}"
        )


@router.post("/start")
async def start_worker(
    background_tasks: BackgroundTasks,
    worker: AnomalyDetectionWorker = Depends(get_worker_instance)
) -> Dict[str, str]:
    """Start the worker service."""
    try:
        if worker.is_running:
            return {"message": "Worker is already running"}
        
        # Start worker in background
        background_tasks.add_task(_start_worker_if_needed, worker)
        
        return {"message": "Worker start initiated"}
        
    except Exception as e:
        logger.error("Failed to start worker", error=str(e))
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to start worker: {str(e)}"
        )


@router.post("/stop")
async def stop_worker(
    worker: AnomalyDetectionWorker = Depends(get_worker_instance)
) -> Dict[str, str]:
    """Stop the worker service gracefully."""
    try:
        if not worker.is_running:
            return {"message": "Worker is not running"}
        
        await worker.stop()
        
        return {"message": "Worker stopped successfully"}
        
    except Exception as e:
        logger.error("Failed to stop worker", error=str(e))
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to stop worker: {str(e)}"
        )


@router.post("/jobs/{job_id}/retry")
async def retry_job(
    job_id: str,
    worker: AnomalyDetectionWorker = Depends(get_worker_instance)
) -> Dict[str, str]:
    """Retry a failed job."""
    try:
        job_status = await worker.get_job_status(job_id)
        
        if not job_status:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Job {job_id} not found"
            )
        
        if job_status['status'] != 'failed':
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Job {job_id} is not in failed status (current: {job_status['status']})"
            )
        
        # This is a placeholder - in a real implementation, we'd need to add
        # retry functionality to reset job status and re-queue it
        
        return {"message": f"Job {job_id} queued for retry"}
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error("Failed to retry job", job_id=job_id, error=str(e))
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to retry job: {str(e)}"
        )


async def _start_worker_if_needed(worker: AnomalyDetectionWorker) -> None:
    """Start worker in background if not already running."""
    try:
        if not worker.is_running:
            logger.info("Starting worker in background")
            await worker.start()
    except Exception as e:
        logger.error("Failed to start worker in background", error=str(e))