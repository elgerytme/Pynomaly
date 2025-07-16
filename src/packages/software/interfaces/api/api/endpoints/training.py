"""REST API endpoints for automated training management.

Provides endpoints for:
- Starting and managing training pipelines
- Monitoring training progress and history
- Configuring automated retraining schedules
- Performance threshold management
"""

import logging
from datetime import UTC, datetime
from typing import Any
from uuid import UUID

from fastapi import APIRouter, Depends, HTTPException, status
from pydantic import BaseModel, Field

from pynomaly_detection.application.services.automated_training_service import (
    AutomatedTrainingService,
    TrainingConfig,
    TriggerType,
)
from pynomaly_detection.application.services.automl_service import OptimizationObjective
from pynomaly_detection.presentation.api.dependencies import (
    get_current_user,
    get_training_service,
)

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/training", tags=["automated-training"])


# Request/Response Models


class StartTrainingRequest(BaseModel):
    """Request to start automated training."""

    detector_id: UUID = Field(description="ID of detector to train")
    dataset_id: str = Field(description="ID of dataset to use for training")
    experiment_name: str | None = Field(None, description="Optional experiment name")

    # AutoML settings
    enable_automl: bool = Field(True, description="Enable AutoML optimization")
    optimization_objective: str = Field(
        "auc", description="Optimization objective (auc, precision, recall, f1_score)"
    )
    max_algorithms: int = Field(
        3, ge=1, le=10, description="Maximum algorithms to test"
    )
    enable_ensemble: bool = Field(True, description="Enable ensemble creation")
    max_optimization_time: int = Field(
        3600, ge=60, le=86400, description="Maximum optimization time in seconds"
    )
    n_trials: int = Field(
        100, ge=10, le=1000, description="Number of optimization trials"
    )

    # Training settings
    validation_split: float = Field(
        0.2, ge=0.0, le=0.5, description="Validation split ratio"
    )
    cv_folds: int = Field(3, ge=2, le=10, description="Cross-validation folds")
    enable_early_stopping: bool = Field(True, description="Enable early stopping")
    max_training_time: int | None = Field(
        None, ge=60, description="Maximum training time in seconds"
    )

    # Resource constraints
    max_memory_mb: int | None = Field(
        None, ge=512, description="Maximum memory usage in MB"
    )
    max_cpu_cores: int | None = Field(
        None, ge=1, description="Maximum CPU cores to use"
    )
    enable_gpu: bool = Field(False, description="Enable GPU acceleration")


class ScheduleTrainingRequest(BaseModel):
    """Request to schedule automated training."""

    detector_id: UUID = Field(description="ID of detector to train")
    dataset_id: str = Field(description="ID of dataset to use for training")
    experiment_name: str | None = Field(None, description="Optional experiment name")

    # Scheduling settings
    schedule_cron: str | None = Field(
        None, description="Cron expression for scheduling"
    )
    retrain_threshold: float = Field(
        0.05, ge=0.001, le=0.5, description="Performance drop threshold for retraining"
    )
    performance_window: int = Field(
        7, ge=1, le=30, description="Days to monitor performance"
    )

    # Training configuration (inherit from StartTrainingRequest)
    enable_automl: bool = Field(True, description="Enable AutoML optimization")
    optimization_objective: str = Field("auc", description="Optimization objective")
    max_algorithms: int = Field(
        3, ge=1, le=10, description="Maximum algorithms to test"
    )
    enable_ensemble: bool = Field(True, description="Enable ensemble creation")
    max_optimization_time: int = Field(
        3600, ge=60, le=86400, description="Maximum optimization time in seconds"
    )


class UpdatePerformanceRequest(BaseModel):
    """Request to update detector performance metrics."""

    detector_id: UUID = Field(description="ID of detector")
    score: float = Field(ge=0.0, le=1.0, description="Performance score (0-1)")
    metric_name: str = Field("auc", description="Name of the performance metric")
    timestamp: datetime | None = Field(None, description="Timestamp of the measurement")


class TrainingProgressResponse(BaseModel):
    """Response containing training progress information."""

    training_id: str
    status: str
    current_step: str
    progress_percentage: float
    start_time: datetime
    estimated_completion: datetime | None = None

    # Current metrics
    current_algorithm: str | None = None
    current_trial: int | None = None
    total_trials: int | None = None
    best_score: float | None = None
    current_score: float | None = None

    # Resource usage
    memory_usage_mb: float | None = None
    cpu_usage_percent: float | None = None

    # Messages and logs
    current_message: str | None = None
    warnings: list[str] = Field(default_factory=list)


class TrainingResultResponse(BaseModel):
    """Response containing training result information."""

    training_id: str
    detector_id: UUID
    status: str
    trigger_type: str

    # Training metrics
    best_algorithm: str | None = None
    best_params: dict[str, Any] | None = None
    best_score: float | None = None
    training_time_seconds: float | None = None
    trials_completed: int | None = None

    # Model information
    model_version: str | None = None
    model_path: str | None = None
    model_size_mb: float | None = None

    # Performance comparison
    previous_score: float | None = None
    performance_improvement: float | None = None

    # Metadata
    dataset_id: str | None = None
    experiment_name: str | None = None
    start_time: datetime | None = None
    completion_time: datetime | None = None
    error_message: str | None = None
    warnings: list[str] = Field(default_factory=list)


class TrainingHistoryResponse(BaseModel):
    """Response containing training history."""

    trainings: list[TrainingResultResponse]
    total_count: int
    page: int
    page_size: int


# API Endpoints


@router.post("/start", response_model=dict[str, str])
async def start_training(
    request: StartTrainingRequest,
    training_service: AutomatedTrainingService = Depends(get_training_service),
    current_user: dict = Depends(get_current_user),
):
    """Start a new automated training pipeline.

    Args:
        request: Training configuration
        training_service: Automated training service
        current_user: Current authenticated user

    Returns:
        Training ID and status
    """
    try:
        logger.info(
            f"Starting training for detector {request.detector_id} by user {current_user.get('id')}"
        )

        # Parse optimization objective
        objective_map = {
            "auc": OptimizationObjective.AUC,
            "precision": OptimizationObjective.PRECISION,
            "recall": OptimizationObjective.RECALL,
            "f1_score": OptimizationObjective.F1_SCORE,
            "balanced_accuracy": OptimizationObjective.BALANCED_ACCURACY,
            "detection_rate": OptimizationObjective.DETECTION_RATE,
        }

        objective = objective_map.get(
            request.optimization_objective.lower(), OptimizationObjective.AUC
        )

        # Create training configuration
        config = TrainingConfig(
            detector_id=request.detector_id,
            dataset_id=request.dataset_id,
            experiment_name=request.experiment_name,
            enable_automl=request.enable_automl,
            optimization_objective=objective,
            max_algorithms=request.max_algorithms,
            enable_ensemble=request.enable_ensemble,
            max_optimization_time=request.max_optimization_time,
            n_trials=request.n_trials,
            validation_split=request.validation_split,
            cv_folds=request.cv_folds,
            enable_early_stopping=request.enable_early_stopping,
            max_training_time=request.max_training_time,
            max_memory_mb=request.max_memory_mb,
            max_cpu_cores=request.max_cpu_cores,
            enable_gpu=request.enable_gpu,
        )

        # Start training
        training_id = await training_service.schedule_training(
            config, TriggerType.MANUAL
        )

        return {
            "training_id": training_id,
            "status": "started",
            "message": "Training pipeline started successfully",
        }

    except Exception as e:
        logger.error(f"Failed to start training: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to start training: {str(e)}",
        )


@router.post("/schedule", response_model=dict[str, str])
async def schedule_training(
    request: ScheduleTrainingRequest,
    training_service: AutomatedTrainingService = Depends(get_training_service),
    current_user: dict = Depends(get_current_user),
):
    """Schedule automated training with performance monitoring.

    Args:
        request: Scheduling configuration
        training_service: Automated training service
        current_user: Current authenticated user

    Returns:
        Training ID and schedule status
    """
    try:
        logger.info(
            f"Scheduling training for detector {request.detector_id} by user {current_user.get('id')}"
        )

        # Parse optimization objective
        objective_map = {
            "auc": OptimizationObjective.AUC,
            "precision": OptimizationObjective.PRECISION,
            "recall": OptimizationObjective.RECALL,
            "f1_score": OptimizationObjective.F1_SCORE,
            "balanced_accuracy": OptimizationObjective.BALANCED_ACCURACY,
            "detection_rate": OptimizationObjective.DETECTION_RATE,
        }

        objective = objective_map.get(
            request.optimization_objective.lower(), OptimizationObjective.AUC
        )

        # Create training configuration
        config = TrainingConfig(
            detector_id=request.detector_id,
            dataset_id=request.dataset_id,
            experiment_name=request.experiment_name,
            schedule_cron=request.schedule_cron,
            retrain_threshold=request.retrain_threshold,
            performance_window=request.performance_window,
            enable_automl=request.enable_automl,
            optimization_objective=objective,
            max_algorithms=request.max_algorithms,
            enable_ensemble=request.enable_ensemble,
            max_optimization_time=request.max_optimization_time,
        )

        # Schedule training
        training_id = await training_service.schedule_training(
            config, TriggerType.SCHEDULED
        )

        return {
            "training_id": training_id,
            "status": "scheduled",
            "message": "Training pipeline scheduled successfully",
        }

    except Exception as e:
        logger.error(f"Failed to schedule training: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to schedule training: {str(e)}",
        )


@router.post("/cancel/{training_id}")
async def cancel_training(
    training_id: str,
    training_service: AutomatedTrainingService = Depends(get_training_service),
    current_user: dict = Depends(get_current_user),
):
    """Cancel an active training pipeline.

    Args:
        training_id: ID of training to cancel
        training_service: Automated training service
        current_user: Current authenticated user

    Returns:
        Cancellation status
    """
    try:
        logger.info(
            f"Cancelling training {training_id} by user {current_user.get('id')}"
        )

        success = await training_service.cancel_training(training_id)

        if success:
            return {
                "training_id": training_id,
                "status": "cancelled",
                "message": "Training cancelled successfully",
            }
        else:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Training {training_id} not found or cannot be cancelled",
            )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to cancel training: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to cancel training: {str(e)}",
        )


@router.get("/status/{training_id}", response_model=TrainingProgressResponse)
async def get_training_status(
    training_id: str,
    training_service: AutomatedTrainingService = Depends(get_training_service),
    current_user: dict = Depends(get_current_user),
):
    """Get current status of a training pipeline.

    Args:
        training_id: ID of training
        training_service: Automated training service
        current_user: Current authenticated user

    Returns:
        Training progress information
    """
    try:
        progress = await training_service.get_training_status(training_id)

        if not progress:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Training {training_id} not found",
            )

        return TrainingProgressResponse(
            training_id=progress.training_id,
            status=progress.status.value,
            current_step=progress.current_step,
            progress_percentage=progress.progress_percentage,
            start_time=progress.start_time,
            estimated_completion=progress.estimated_completion,
            current_algorithm=progress.current_algorithm,
            current_trial=progress.current_trial,
            total_trials=progress.total_trials,
            best_score=progress.best_score,
            current_score=progress.current_score,
            memory_usage_mb=progress.memory_usage_mb,
            cpu_usage_percent=progress.cpu_usage_percent,
            current_message=progress.current_message,
            warnings=progress.warnings,
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to get training status: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to get training status: {str(e)}",
        )


@router.get("/result/{training_id}", response_model=TrainingResultResponse)
async def get_training_result(
    training_id: str,
    training_service: AutomatedTrainingService = Depends(get_training_service),
    current_user: dict = Depends(get_current_user),
):
    """Get result of a completed training pipeline.

    Args:
        training_id: ID of training
        training_service: Automated training service
        current_user: Current authenticated user

    Returns:
        Training result information
    """
    try:
        result = await training_service.get_training_result(training_id)

        if not result:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Training result {training_id} not found",
            )

        return TrainingResultResponse(
            training_id=result.training_id,
            detector_id=result.detector_id,
            status=result.status.value,
            trigger_type=result.trigger_type.value,
            best_algorithm=result.best_algorithm,
            best_params=result.best_params,
            best_score=result.best_score,
            training_time_seconds=result.training_time_seconds,
            trials_completed=result.trials_completed,
            model_version=result.model_version,
            model_path=result.model_path,
            model_size_mb=result.model_size_mb,
            previous_score=result.previous_score,
            performance_improvement=result.performance_improvement,
            dataset_id=result.dataset_id,
            experiment_name=result.experiment_name,
            start_time=result.start_time,
            completion_time=result.completion_time,
            error_message=result.error_message,
            warnings=result.warnings,
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to get training result: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to get training result: {str(e)}",
        )


@router.get("/active", response_model=list[TrainingProgressResponse])
async def get_active_trainings(
    training_service: AutomatedTrainingService = Depends(get_training_service),
    current_user: dict = Depends(get_current_user),
):
    """Get all currently active training pipelines.

    Args:
        training_service: Automated training service
        current_user: Current authenticated user

    Returns:
        List of active training progress
    """
    try:
        active_trainings = await training_service.get_active_trainings()

        return [
            TrainingProgressResponse(
                training_id=progress.training_id,
                status=progress.status.value,
                current_step=progress.current_step,
                progress_percentage=progress.progress_percentage,
                start_time=progress.start_time,
                estimated_completion=progress.estimated_completion,
                current_algorithm=progress.current_algorithm,
                current_trial=progress.current_trial,
                total_trials=progress.total_trials,
                best_score=progress.best_score,
                current_score=progress.current_score,
                memory_usage_mb=progress.memory_usage_mb,
                cpu_usage_percent=progress.cpu_usage_percent,
                current_message=progress.current_message,
                warnings=progress.warnings,
            )
            for progress in active_trainings
        ]

    except Exception as e:
        logger.error(f"Failed to get active trainings: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to get active trainings: {str(e)}",
        )


@router.get("/history", response_model=TrainingHistoryResponse)
async def get_training_history(
    detector_id: UUID | None = None,
    page: int = 1,
    page_size: int = 50,
    training_service: AutomatedTrainingService = Depends(get_training_service),
    current_user: dict = Depends(get_current_user),
):
    """Get training history.

    Args:
        detector_id: Optional detector ID filter
        page: Page number (1-based)
        page_size: Number of results per page
        training_service: Automated training service
        current_user: Current authenticated user

    Returns:
        Paginated training history
    """
    try:
        # Calculate offset
        offset = (page - 1) * page_size
        limit = page_size

        # Get training history
        history = await training_service.get_training_history(
            detector_id, limit + offset
        )

        # Apply pagination
        paginated_history = history[offset : offset + limit]

        training_responses = [
            TrainingResultResponse(
                training_id=result.training_id,
                detector_id=result.detector_id,
                status=result.status.value,
                trigger_type=result.trigger_type.value,
                best_algorithm=result.best_algorithm,
                best_params=result.best_params,
                best_score=result.best_score,
                training_time_seconds=result.training_time_seconds,
                trials_completed=result.trials_completed,
                model_version=result.model_version,
                model_path=result.model_path,
                model_size_mb=result.model_size_mb,
                previous_score=result.previous_score,
                performance_improvement=result.performance_improvement,
                dataset_id=result.dataset_id,
                experiment_name=result.experiment_name,
                start_time=result.start_time,
                completion_time=result.completion_time,
                error_message=result.error_message,
                warnings=result.warnings,
            )
            for result in paginated_history
        ]

        return TrainingHistoryResponse(
            trainings=training_responses,
            total_count=len(history),
            page=page,
            page_size=page_size,
        )

    except Exception as e:
        logger.error(f"Failed to get training history: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to get training history: {str(e)}",
        )


@router.post("/performance/update")
async def update_performance(
    request: UpdatePerformanceRequest,
    training_service: AutomatedTrainingService = Depends(get_training_service),
    current_user: dict = Depends(get_current_user),
):
    """Update detector performance metrics for retraining evaluation.

    Args:
        request: Performance update data
        training_service: Automated training service
        current_user: Current authenticated user

    Returns:
        Update status and retraining recommendation
    """
    try:
        logger.info(
            f"Updating performance for detector {request.detector_id}: {request.score}"
        )

        # Update performance
        await training_service.update_performance(request.detector_id, request.score)

        # Check if retraining is needed
        needs_retraining = await training_service.check_retraining_needed(
            request.detector_id
        )

        return {
            "detector_id": str(request.detector_id),
            "score": request.score,
            "metric_name": request.metric_name,
            "timestamp": request.timestamp or datetime.now(UTC),
            "needs_retraining": needs_retraining,
            "message": "Performance updated successfully",
        }

    except Exception as e:
        logger.error(f"Failed to update performance: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to update performance: {str(e)}",
        )
