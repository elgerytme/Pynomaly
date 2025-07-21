"""REST API endpoints for training automation.

This module provides comprehensive REST API endpoints for:
- Training job management and monitoring
- Hyperparameter optimization configuration
- Experiment tracking and results
- Model lifecycle management
"""




from __future__ import annotations

import logging
from datetime import datetime
from typing import Any

from fastapi import APIRouter, BackgroundTasks, Depends, HTTPException, Query, status
from pydantic import BaseModel, Field

from ...core.dependency_injection import inject, ServiceContainer, get_container
from ...core.domain_entities import (
    TrainingJob,
    DetectorConfiguration,
    ModelStatus
)

# Service interfaces - these should be properly defined protocols
from abc import ABC, abstractmethod
from typing import Protocol, List, Optional, Dict, Any
from datetime import datetime
from enum import Enum

class OptimizationStrategy(Enum):
    """Optimization strategy enumeration."""
    TPE = "tpe"
    RANDOM = "random"
    GRID = "grid"
    BAYESIAN = "bayesian"

class PruningStrategy(Enum):
    """Pruning strategy enumeration."""  
    MEDIAN = "median"
    NONE = "none"
    HYPERBAND = "hyperband"

class TrainingStatus(Enum):
    """Training status enumeration."""
    PENDING = "pending"
    RUNNING = "running" 
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"

class TrainingConfiguration:
    """Training configuration class."""
    def __init__(self, **kwargs):
        for key, value in kwargs.items():
            setattr(self, key, value)

class TrainingAutomationServiceProtocol(Protocol):
    """Protocol for training automation service."""
    
    @abstractmethod
    async def create_training_job(self, name: str, dataset_id: str, 
                                configuration: TrainingConfiguration,
                                target_algorithms: Optional[List[str]] = None) -> TrainingJob:
        """Create a new training job."""
        pass
    
    @abstractmethod 
    async def start_training_job(self, job_id: str) -> None:
        """Start a training job."""
        pass
        
    @abstractmethod
    async def get_job_status(self, job_id: str) -> Optional[TrainingJob]:
        """Get job status."""
        pass
        
    @abstractmethod
    async def list_training_jobs(self, status_filter: Optional[TrainingStatus] = None,
                               limit: int = 100) -> List[TrainingJob]:
        """List training jobs."""
        pass
        
    @abstractmethod
    async def cancel_training_job(self, job_id: str) -> None:
        """Cancel training job."""
        pass
        
    @abstractmethod
    async def get_training_metrics(self, job_id: str) -> Dict[str, Any]:
        """Get training metrics."""
        pass
        
    @abstractmethod
    async def cleanup_old_jobs(self, days: int) -> int:
        """Cleanup old jobs."""
        pass
        
    @property
    @abstractmethod
    def active_jobs(self) -> List[str]:
        """Get active job IDs."""
        pass

# For now, we'll create a stub implementation that uses the dependency injection container
class TrainingAutomationService(TrainingAutomationServiceProtocol):
    """Training automation service implementation."""
    
    def __init__(self, repository=None, trainer=None):
        self.repository = repository
        self.trainer = trainer
        self._active_jobs = []
    
    async def create_training_job(self, name: str, dataset_id: str,
                                configuration: TrainingConfiguration,
                                target_algorithms: Optional[List[str]] = None) -> TrainingJob:
        """Create training job stub."""
        job = TrainingJob(
            name=name,
            dataset_id=dataset_id,
            configuration=DetectorConfiguration(),
            status="pending"
        )
        return job
    
    async def start_training_job(self, job_id: str) -> None:
        """Start training job stub."""
        self._active_jobs.append(job_id)
    
    async def get_job_status(self, job_id: str) -> Optional[TrainingJob]:
        """Get job status stub."""
        return None
    
    async def list_training_jobs(self, status_filter: Optional[TrainingStatus] = None,
                               limit: int = 100) -> List[TrainingJob]:
        """List jobs stub."""
        return []
    
    async def cancel_training_job(self, job_id: str) -> None:
        """Cancel job stub."""
        if job_id in self._active_jobs:
            self._active_jobs.remove(job_id)
    
    async def get_training_metrics(self, job_id: str) -> Dict[str, Any]:
        """Get metrics stub.""" 
        return {
            "job_id": job_id,
            "status": "unknown",
            "execution_time": 0.0,
            "total_trials": 0,
            "successful_trials": 0,
            "failed_trials": 0,
            "success_rate": 0.0,
            "best_score": None,
            "best_algorithm": None,
            "trial_history": []
        }
    
    async def cleanup_old_jobs(self, days: int) -> int:
        """Cleanup stub."""
        return 0
        
    @property
    def active_jobs(self) -> List[str]:
        """Get active jobs."""
        return self._active_jobs.copy()

logger = logging.getLogger(__name__)

# Create router
router = APIRouter(prefix="/api/v1/training", tags=["Training Automation"])


# Pydantic models for API
class TrainingConfigurationRequest(BaseModel):
    """Request model for training configuration."""

    max_trials: int = Field(
        default=100, ge=10, le=1000, description="Maximum number of optimization trials"
    )
    timeout_minutes: int | None = Field(
        default=60, ge=1, le=1440, description="Timeout in minutes"
    )
    optimization_strategy: OptimizationStrategy = Field(
        default=OptimizationStrategy.TPE, description="Optimization strategy"
    )
    pruning_strategy: PruningStrategy = Field(
        default=PruningStrategy.MEDIAN,
        description="Pruning strategy for early stopping",
    )

    # Optimization objectives
    primary_metric: str = Field(
        default="roc_auc", description="Primary optimization metric"
    )
    secondary_metrics: list[str] = Field(
        default=["precision", "recall", "f1"], description="Secondary metrics to track"
    )
    optimization_direction: str = Field(
        default="maximize",
        regex="^(maximize|minimize)$",
        description="Optimization direction",
    )

    # Validation settings
    cross_validation_folds: int = Field(
        default=5, ge=2, le=10, description="Number of CV folds"
    )
    validation_split: float = Field(
        default=0.2, ge=0.1, le=0.5, description="Validation split ratio"
    )
    early_stopping_patience: int = Field(
        default=10, ge=1, le=100, description="Early stopping patience"
    )
    min_improvement_threshold: float = Field(
        default=0.001, ge=0.0, le=0.1, description="Minimum improvement threshold"
    )

    # Resource constraints
    max_memory_gb: float | None = Field(
        default=None, ge=0.1, le=1024, description="Maximum memory usage in GB"
    )
    max_cpu_cores: int | None = Field(
        default=None, ge=1, le=64, description="Maximum CPU cores"
    )
    use_gpu: bool = Field(default=False, description="Enable GPU acceleration")

    # Experiment tracking
    experiment_name: str | None = Field(
        default=None, description="Experiment name for tracking"
    )
    track_artifacts: bool = Field(default=True, description="Enable artifact tracking")
    save_models: bool = Field(default=True, description="Save trained models")

    # Algorithm selection
    algorithm_whitelist: list[str] | None = Field(
        default=None, description="Allowed algorithms"
    )
    algorithm_blacklist: list[str] | None = Field(
        default=None, description="Excluded algorithms"
    )
    ensemble_methods: list[str] = Field(
        default=["voting", "stacking"], description="Ensemble methods to try"
    )


class CreateTrainingJobRequest(BaseModel):
    """Request model for creating training jobs."""

    name: str = Field(..., min_length=1, max_length=255, description="Job name")
    dataset_id: str = Field(..., min_length=1, description="Dataset ID")
    target_algorithms: list[str] | None = Field(
        default=None, description="Target algorithms"
    )
    configuration: TrainingConfigurationRequest = Field(
        ..., description="Training configuration"
    )


class TrainingJobResponse(BaseModel):
    """Response model for training job information."""

    job_id: str
    name: str
    status: TrainingStatus
    dataset_id: str
    target_algorithms: list[str]

    created_at: datetime
    started_at: datetime | None = None
    completed_at: datetime | None = None
    error_message: str | None = None

    best_score: float | None = None
    best_parameters: dict[str, Any] | None = None

    total_trials: int
    successful_trials: int
    failed_trials: int
    execution_time_seconds: float

    model_path: str | None = None
    experiment_id: str | None = None


class TrainingJobMetricsResponse(BaseModel):
    """Response model for training job metrics."""

    job_id: str
    status: str
    execution_time: float
    total_trials: int
    successful_trials: int
    failed_trials: int
    success_rate: float
    best_score: float | None = None
    best_algorithm: str | None = None
    trial_history: list[dict[str, Any]] = []


class QuickOptimizeRequest(BaseModel):
    """Request model for quick optimization."""

    dataset_id: str = Field(..., description="Dataset ID")
    algorithms: list[str] | None = Field(default=None, description="Target algorithms")
    max_trials: int = Field(default=50, ge=10, le=200, description="Maximum trials")
    timeout_minutes: int = Field(
        default=30, ge=5, le=120, description="Timeout in minutes"
    )


# Dependency injection
def get_training_service() -> TrainingAutomationService:
    """Get training automation service via dependency injection."""
    container = get_container()
    
    # Try to resolve from container first
    service = container.try_resolve(TrainingAutomationServiceProtocol)
    if service:
        return service
    
    # Fallback to creating a default service with stub implementations
    return TrainingAutomationService()


# API Endpoints


@router.post(
    "/jobs", response_model=TrainingJobResponse, status_code=status.HTTP_201_CREATED
)
async def create_training_job(
    request: CreateTrainingJobRequest,
    service: TrainingAutomationService = Depends(get_training_service),
) -> TrainingJobResponse:
    """Create a new training job."""
    try:
        # Convert request to domain objects
        config = TrainingConfiguration(
            max_trials=request.configuration.max_trials,
            timeout_minutes=request.configuration.timeout_minutes,
            optimization_strategy=request.configuration.optimization_strategy,
            pruning_strategy=request.configuration.pruning_strategy,
            primary_metric=request.configuration.primary_metric,
            secondary_metrics=request.configuration.secondary_metrics,
            optimization_direction=request.configuration.optimization_direction,
            cross_validation_folds=request.configuration.cross_validation_folds,
            validation_split=request.configuration.validation_split,
            early_stopping_patience=request.configuration.early_stopping_patience,
            min_improvement_threshold=request.configuration.min_improvement_threshold,
            max_memory_gb=request.configuration.max_memory_gb,
            max_cpu_cores=request.configuration.max_cpu_cores,
            use_gpu=request.configuration.use_gpu,
            experiment_name=request.configuration.experiment_name,
            track_artifacts=request.configuration.track_artifacts,
            save_models=request.configuration.save_models,
            algorithm_whitelist=request.configuration.algorithm_whitelist,
            algorithm_blacklist=request.configuration.algorithm_blacklist,
            ensemble_methods=request.configuration.ensemble_methods,
        )

        job = await service.create_training_job(
            name=request.name,
            dataset_id=request.dataset_id,
            configuration=config,
            target_algorithms=request.target_algorithms,
        )

        return TrainingJobResponse(
            job_id=job.job_id,
            name=job.name,
            status=job.status,
            dataset_id=job.dataset_id,
            target_algorithms=job.target_algorithms,
            created_at=job.created_at,
            started_at=job.started_at,
            completed_at=job.completed_at,
            error_message=job.error_message,
            best_score=job.best_score,
            best_parameters=job.best_parameters,
            total_trials=job.total_trials,
            successful_trials=job.successful_trials,
            failed_trials=job.failed_trials,
            execution_time_seconds=job.execution_time_seconds,
            model_path=job.model_path,
            experiment_id=job.experiment_id,
        )

    except Exception as e:
        logger.error(f"Failed to create training job: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to create training job: {str(e)}",
        )


@router.post("/jobs/{job_id}/start", status_code=status.HTTP_202_ACCEPTED)
async def start_training_job(
    job_id: str,
    background_tasks: BackgroundTasks,
    service: TrainingAutomationService = Depends(get_training_service),
):
    """Start a training job."""
    try:
        # Start job in background
        background_tasks.add_task(service.start_training_job, job_id)

        return {"message": f"Training job {job_id} started successfully"}

    except Exception as e:
        logger.error(f"Failed to start training job {job_id}: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to start training job: {str(e)}",
        )


@router.get("/jobs/{job_id}", response_model=TrainingJobResponse)
async def get_training_job(
    job_id: str, service: TrainingAutomationService = Depends(get_training_service)
) -> TrainingJobResponse:
    """Get training job details."""
    try:
        job = await service.get_job_status(job_id)

        if not job:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Training job {job_id} not found",
            )

        return TrainingJobResponse(
            job_id=job.job_id,
            name=job.name,
            status=job.status,
            dataset_id=job.dataset_id,
            target_algorithms=job.target_algorithms,
            created_at=job.created_at,
            started_at=job.started_at,
            completed_at=job.completed_at,
            error_message=job.error_message,
            best_score=job.best_score,
            best_parameters=job.best_parameters,
            total_trials=job.total_trials,
            successful_trials=job.successful_trials,
            failed_trials=job.failed_trials,
            execution_time_seconds=job.execution_time_seconds,
            model_path=job.model_path,
            experiment_id=job.experiment_id,
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to get training job {job_id}: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to get training job: {str(e)}",
        )


@router.get("/jobs", response_model=list[TrainingJobResponse])
async def list_training_jobs(
    status_filter: TrainingStatus | None = Query(
        None, alias="status", description="Filter by status"
    ),
    limit: int = Query(
        100, ge=1, le=1000, description="Maximum number of jobs to return"
    ),
    service: TrainingAutomationService = Depends(get_training_service),
) -> list[TrainingJobResponse]:
    """List training jobs."""
    try:
        jobs = await service.list_training_jobs(status_filter, limit)

        return [
            TrainingJobResponse(
                job_id=job.job_id,
                name=job.name,
                status=job.status,
                dataset_id=job.dataset_id,
                target_algorithms=job.target_algorithms,
                created_at=job.created_at,
                started_at=job.started_at,
                completed_at=job.completed_at,
                error_message=job.error_message,
                best_score=job.best_score,
                best_parameters=job.best_parameters,
                total_trials=job.total_trials,
                successful_trials=job.successful_trials,
                failed_trials=job.failed_trials,
                execution_time_seconds=job.execution_time_seconds,
                model_path=job.model_path,
                experiment_id=job.experiment_id,
            )
            for job in jobs
        ]

    except Exception as e:
        logger.error(f"Failed to list training jobs: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to list training jobs: {str(e)}",
        )


@router.delete("/jobs/{job_id}/cancel", status_code=status.HTTP_200_OK)
async def cancel_training_job(
    job_id: str, service: TrainingAutomationService = Depends(get_training_service)
):
    """Cancel a running training job."""
    try:
        await service.cancel_training_job(job_id)
        return {"message": f"Training job {job_id} cancelled successfully"}

    except Exception as e:
        logger.error(f"Failed to cancel training job {job_id}: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to cancel training job: {str(e)}",
        )


@router.get("/jobs/{job_id}/metrics", response_model=TrainingJobMetricsResponse)
async def get_training_metrics(
    job_id: str, service: TrainingAutomationService = Depends(get_training_service)
) -> TrainingJobMetricsResponse:
    """Get comprehensive training metrics for a job."""
    try:
        metrics = await service.get_training_metrics(job_id)

        return TrainingJobMetricsResponse(
            job_id=metrics["job_id"],
            status=metrics["status"],
            execution_time=metrics["execution_time"],
            total_trials=metrics["total_trials"],
            successful_trials=metrics["successful_trials"],
            failed_trials=metrics["failed_trials"],
            success_rate=metrics["success_rate"],
            best_score=metrics["best_score"],
            best_algorithm=metrics["best_algorithm"],
            trial_history=metrics["trial_history"],
        )

    except Exception as e:
        logger.error(f"Failed to get training metrics for {job_id}: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to get training metrics: {str(e)}",
        )


@router.post(
    "/quick-optimize",
    response_model=TrainingJobResponse,
    status_code=status.HTTP_201_CREATED,
)
async def quick_optimize(
    request: QuickOptimizeRequest,
    background_tasks: BackgroundTasks,
    service: TrainingAutomationService = Depends(get_training_service),
) -> TrainingJobResponse:
    """Quick optimization with sensible defaults."""
    try:
        # Create quick optimization configuration
        config = TrainingConfiguration(
            max_trials=request.max_trials,
            timeout_minutes=request.timeout_minutes,
            optimization_strategy=OptimizationStrategy.TPE,
            pruning_strategy=PruningStrategy.MEDIAN,
            experiment_name=f"Quick optimization {datetime.now().strftime('%Y%m%d_%H%M%S')}",
        )

        # Create job
        job = await service.create_training_job(
            name=f"Quick optimization {datetime.now().strftime('%Y%m%d_%H%M%S')}",
            dataset_id=request.dataset_id,
            configuration=config,
            target_algorithms=request.algorithms,
        )

        # Start job in background
        background_tasks.add_task(service.start_training_job, job.job_id)

        return TrainingJobResponse(
            job_id=job.job_id,
            name=job.name,
            status=job.status,
            dataset_id=job.dataset_id,
            target_algorithms=job.target_algorithms,
            created_at=job.created_at,
            started_at=job.started_at,
            completed_at=job.completed_at,
            error_message=job.error_message,
            best_score=job.best_score,
            best_parameters=job.best_parameters,
            total_trials=job.total_trials,
            successful_trials=job.successful_trials,
            failed_trials=job.failed_trials,
            execution_time_seconds=job.execution_time_seconds,
            model_path=job.model_path,
            experiment_id=job.experiment_id,
        )

    except Exception as e:
        logger.error(f"Failed to start quick optimization: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to start quick optimization: {str(e)}",
        )


@router.get("/algorithms", response_model=list[str])
async def get_supported_algorithms(
    service: TrainingAutomationService = Depends(get_training_service),
) -> list[str]:
    """Get list of supported algorithms."""
    try:
        # Get algorithms via dependency injection
        container = get_container()
        # For now return a default set of algorithms
        return [
            "isolation_forest", "local_outlier_factor", "one_class_svm",
            "elliptic_envelope", "robust_covariance", "dbscan",
            "autoencoder", "variational_autoencoder", "lstm_autoencoder"
        ]

    except Exception as e:
        logger.error(f"Failed to get supported algorithms: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to get supported algorithms: {str(e)}",
        )


@router.get("/algorithms/{algorithm_name}")
async def get_algorithm_info(
    algorithm_name: str,
    service: TrainingAutomationService = Depends(get_training_service),
) -> dict[str, Any]:
    """Get detailed information about a specific algorithm."""
    try:
        # Return default algorithm info via dependency injection pattern
        container = get_container()
        # For now return default algorithm info
        return {
            "name": algorithm_name,
            "description": f"Information for {algorithm_name} algorithm",
            "parameters": {},
            "supported_data_types": ["numerical", "categorical"],
            "performance_characteristics": {}
        }

    except Exception as e:
        logger.error(f"Failed to get algorithm info for {algorithm_name}: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to get algorithm info: {str(e)}",
        )


@router.delete("/cleanup", status_code=status.HTTP_200_OK)
async def cleanup_old_jobs(
    days: int = Query(30, ge=1, le=365, description="Age threshold in days"),
    service: TrainingAutomationService = Depends(get_training_service),
):
    """Clean up old training jobs and artifacts."""
    try:
        cleaned_count = await service.cleanup_old_jobs(days)
        return {
            "message": f"Cleaned up {cleaned_count} old training jobs",
            "days_threshold": days,
            "cleaned_count": cleaned_count,
        }

    except Exception as e:
        logger.error(f"Failed to cleanup old jobs: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to cleanup old jobs: {str(e)}",
        )


@router.get("/status")
async def get_training_system_status(
    service: TrainingAutomationService = Depends(get_training_service),
) -> dict[str, Any]:
    """Get training system status and statistics."""
    try:
        # Get system information via dependency injection pattern
        container = get_container()
        
        # Default stats for now
        stats = {
            "total_jobs": 0,
            "completed_jobs": 0,
            "failed_jobs": 0,
            "success_rate": 0.0
        }

        # Add system information
        status_info = {
            "system_status": "healthy",
            "active_jobs": len(service.active_jobs),
            "job_statistics": stats,
            "supported_algorithms": 9,  # From our default algorithm list
            "timestamp": datetime.now().isoformat(),
        }

        return status_info

    except Exception as e:
        logger.error(f"Failed to get training system status: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to get system status: {str(e)}",
        )
