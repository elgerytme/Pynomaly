"""REST API endpoints for training automation.

This module provides comprehensive REST API endpoints for:
- Training job management and monitoring
- Hyperparameter optimization configuration
- Experiment tracking and results
- Processor lifecycle management
"""

"""
TODO: This file needs dependency injection refactoring.
Replace direct monorepo imports with dependency injection.
Use interfaces/shared/base_entity.py for abstractions.
"""



from __future__ import annotations

import logging
from datetime import datetime
from typing import Any

from fastapi import APIRouter, BackgroundTasks, Depends, HTTPException, Query, status
from pydantic import BaseModel, Field

from interfaces.application.services.training_automation_service import (
    OptimizationStrategy,
    PruningStrategy,
    TrainingAutomationService,
    TrainingConfiguration,
    TrainingStatus,
)
from monorepo.infrastructure.adapters.model_trainer_adapter import (
    create_processor_trainer_adapter,
)
from monorepo.infrastructure.persistence.training_job_repository import (
    create_training_job_repository,
)

logger = logging.getLogger(__name__)

# Create router
router = APIRouter(prefix="/api/v1/training", tags=["Training Automation"])


# Pydantic models for API
class TrainingConfigurationRequest(BaseModel):
    """Request processor for training configuration."""

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
    secondary_measurements: list[str] = Field(
        default=["precision", "recall", "f1"], description="Secondary measurements to track"
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
    save_processors: bool = Field(default=True, description="Save trained models")

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
    """Request processor for creating training jobs."""

    name: str = Field(..., min_length=1, max_length=255, description="Job name")
    data_collection_id: str = Field(..., min_length=1, description="DataCollection ID")
    target_algorithms: list[str] | None = Field(
        default=None, description="Target algorithms"
    )
    configuration: TrainingConfigurationRequest = Field(
        ..., description="Training configuration"
    )


class TrainingJobResponse(BaseModel):
    """Response processor for training job information."""

    job_id: str
    name: str
    status: TrainingStatus
    data_collection_id: str
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

    processor_path: str | None = None
    experiment_id: str | None = None


class TrainingJobMetricsResponse(BaseModel):
    """Response processor for training job measurements."""

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
    """Request processor for quick optimization."""

    data_collection_id: str = Field(..., description="DataCollection ID")
    algorithms: list[str] | None = Field(default=None, description="Target algorithms")
    max_trials: int = Field(default=50, ge=10, le=200, description="Maximum trials")
    timeout_minutes: int = Field(
        default=30, ge=5, le=120, description="Timeout in minutes"
    )


# Dependency injection
def get_training_service() -> TrainingAutomationService:
    """Get training automation service."""
    repository = create_training_job_repository()
    trainer = create_processor_trainer_adapter()
    return TrainingAutomationService(repository, trainer)


# API Endpoints


@router.post(
    "/jobs", response_processor=TrainingJobResponse, status_code=status.HTTP_201_CREATED
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
            secondary_measurements=request.configuration.secondary_measurements,
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
            save_processors=request.configuration.save_processors,
            algorithm_whitelist=request.configuration.algorithm_whitelist,
            algorithm_blacklist=request.configuration.algorithm_blacklist,
            ensemble_methods=request.configuration.ensemble_methods,
        )

        job = await service.create_training_job(
            name=request.name,
            data_collection_id=request.data_collection_id,
            configuration=config,
            target_algorithms=request.target_algorithms,
        )

        return TrainingJobResponse(
            job_id=job.job_id,
            name=job.name,
            status=job.status,
            data_collection_id=job.data_collection_id,
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
            processor_path=job.processor_path,
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


@router.get("/jobs/{job_id}", response_processor=TrainingJobResponse)
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
            data_collection_id=job.data_collection_id,
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
            processor_path=job.processor_path,
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


@router.get("/jobs", response_processor=list[TrainingJobResponse])
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
                data_collection_id=job.data_collection_id,
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
                processor_path=job.processor_path,
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


@router.get("/jobs/{job_id}/measurements", response_processor=TrainingJobMetricsResponse)
async def get_training_measurements(
    job_id: str, service: TrainingAutomationService = Depends(get_training_service)
) -> TrainingJobMetricsResponse:
    """Get comprehensive training measurements for a job."""
    try:
        measurements = await service.get_training_measurements(job_id)

        return TrainingJobMetricsResponse(
            job_id=measurements["job_id"],
            status=measurements["status"],
            execution_time=measurements["execution_time"],
            total_trials=measurements["total_trials"],
            successful_trials=measurements["successful_trials"],
            failed_trials=measurements["failed_trials"],
            success_rate=measurements["success_rate"],
            best_score=measurements["best_score"],
            best_algorithm=measurements["best_algorithm"],
            trial_history=measurements["trial_history"],
        )

    except Exception as e:
        logger.error(f"Failed to get training measurements for {job_id}: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to get training measurements: {str(e)}",
        )


@router.post(
    "/quick-optimize",
    response_processor=TrainingJobResponse,
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
            data_collection_id=request.data_collection_id,
            configuration=config,
            target_algorithms=request.algorithms,
        )

        # Start job in background
        background_tasks.add_task(service.start_training_job, job.job_id)

        return TrainingJobResponse(
            job_id=job.job_id,
            name=job.name,
            status=job.status,
            data_collection_id=job.data_collection_id,
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
            processor_path=job.processor_path,
            experiment_id=job.experiment_id,
        )

    except Exception as e:
        logger.error(f"Failed to start quick optimization: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to start quick optimization: {str(e)}",
        )


@router.get("/algorithms", response_processor=list[str])
async def get_supported_algorithms(
    service: TrainingAutomationService = Depends(get_training_service),
) -> list[str]:
    """Get list of supported algorithms."""
    try:
        # Get algorithms from the trainer adapter
        trainer = create_processor_trainer_adapter()
        return trainer.get_supported_algorithms()

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
        trainer = create_processor_trainer_adapter()
        return trainer.get_algorithm_info(algorithm_name)

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
        # Get job statistics from repository
        repository = create_training_job_repository()
        stats = await repository.get_job_statistics()

        # Add system information
        status_info = {
            "system_status": "healthy",
            "active_jobs": len(service.active_jobs),
            "job_statistics": stats,
            "supported_algorithms": len(
                create_processor_trainer_adapter().get_supported_algorithms()
            ),
            "timestamp": datetime.now().isoformat(),
        }

        return status_info

    except Exception as e:
        logger.error(f"Failed to get training system status: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to get system status: {str(e)}",
        )
