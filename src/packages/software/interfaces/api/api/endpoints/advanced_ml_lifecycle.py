"""REST API endpoints for advanced ML lifecycle management."""

from datetime import UTC, datetime
from typing import Any
from uuid import UUID

from fastapi import APIRouter, Depends, HTTPException, status
from pydantic import BaseModel, Field

from monorepo.application.services.advanced_ml_lifecycle_service import (
    AdvancedMLLifecycleService,
)
from monorepo.domain.entities import ExperimentType
from monorepo.presentation.api.deps import (
    get_advanced_ml_lifecycle_service,
    get_current_user,
    require_write,
)

router = APIRouter(prefix="/advanced-ml-lifecycle", tags=["Advanced ML Lifecycle"])


# ==================== Request/Response Models ====================


class StartExperimentRequest(BaseModel):
    """Request model for starting an experiment."""

    name: str = Field(..., description="Experiment name")
    description: str = Field(..., description="Experiment description")
    experiment_type: ExperimentType = Field(..., description="Type of experiment")
    objective: str = Field(..., description="Primary objective")
    auto_log_parameters: bool = Field(True, description="Auto-log parameters")
    auto_log_metrics: bool = Field(True, description="Auto-log metrics")
    auto_log_artifacts: bool = Field(True, description="Auto-log artifacts")
    tags: list[str] | None = Field(None, description="Experiment tags")
    metadata: dict[str, Any] | None = Field(None, description="Additional metadata")


class StartExperimentResponse(BaseModel):
    """Response model for starting an experiment."""

    experiment_id: str = Field(..., description="Created experiment ID")
    message: str = Field(..., description="Success message")


class StartRunRequest(BaseModel):
    """Request model for starting a run."""

    run_name: str = Field(..., description="Run name")
    detector_id: UUID = Field(..., description="Detector ID")
    dataset_id: UUID = Field(..., description="Dataset ID")
    parameters: dict[str, Any] = Field(..., description="Run parameters")
    parent_run_id: str | None = Field(None, description="Parent run ID")
    tags: list[str] | None = Field(None, description="Run tags")
    description: str = Field("", description="Run description")


class StartRunResponse(BaseModel):
    """Response model for starting a run."""

    run_id: str = Field(..., description="Created run ID")
    experiment_id: str = Field(..., description="Parent experiment ID")
    message: str = Field(..., description="Success message")


class LogParameterRequest(BaseModel):
    """Request model for logging a parameter."""

    key: str = Field(..., description="Parameter name")
    value: Any = Field(..., description="Parameter value")


class LogMetricRequest(BaseModel):
    """Request model for logging a metric."""

    key: str = Field(..., description="Metric name")
    value: float = Field(..., description="Metric value")
    step: int | None = Field(None, description="Step number")
    timestamp: datetime | None = Field(None, description="Metric timestamp")


class LogArtifactRequest(BaseModel):
    """Request model for logging an artifact."""

    artifact_name: str = Field(..., description="Artifact name")
    artifact_data: Any = Field(..., description="Artifact data")
    artifact_type: str = Field("pickle", description="Artifact type")
    metadata: dict[str, Any] | None = Field(None, description="Artifact metadata")


class LogArtifactResponse(BaseModel):
    """Response model for logging an artifact."""

    artifact_path: str = Field(..., description="Path to stored artifact")
    message: str = Field(..., description="Success message")


class LogModelRequest(BaseModel):
    """Request model for logging a model."""

    model_name: str = Field(..., description="Model name")
    model_signature: dict[str, Any] | None = Field(None, description="Model signature")
    input_example: Any | None = Field(None, description="Input example")
    registered_model_name: str | None = Field(None, description="Registered model name")


class LogModelResponse(BaseModel):
    """Response model for logging a model."""

    model_version_id: str = Field(..., description="Model version ID or path")
    message: str = Field(..., description="Success message")


class EndRunRequest(BaseModel):
    """Request model for ending a run."""

    status: str = Field("FINISHED", description="Run status")
    end_time: datetime | None = Field(None, description="End time")


class CreateModelVersionRequest(BaseModel):
    """Request model for creating a model version."""

    model_name: str = Field(..., description="Model name")
    run_id: str = Field(..., description="Associated run ID")
    model_path: str = Field(..., description="Model artifact path")
    performance_metrics: dict[str, float] = Field(
        ..., description="Performance metrics"
    )
    description: str = Field("", description="Version description")
    tags: list[str] | None = Field(None, description="Version tags")
    auto_version: bool = Field(True, description="Auto-determine version")


class CreateModelVersionResponse(BaseModel):
    """Response model for creating a model version."""

    model_version_id: str = Field(..., description="Created model version ID")
    message: str = Field(..., description="Success message")


class PromoteModelRequest(BaseModel):
    """Request model for promoting a model version."""

    stage: str = Field(..., description="Target stage")
    approval_workflow: bool = Field(True, description="Use approval workflow")
    validation_tests: list[str] | None = Field(None, description="Validation tests")


class PromoteModelResponse(BaseModel):
    """Response model for promoting a model version."""

    success: bool = Field(..., description="Promotion success")
    model_version_id: str = Field(..., description="Model version ID")
    new_stage: str = Field(..., description="New stage")
    new_status: str = Field(..., description="New status")
    validation_results: dict[str, Any] = Field(..., description="Validation results")
    promoted_by: str = Field(..., description="User who promoted")
    promoted_at: str = Field(..., description="Promotion timestamp")


class SearchModelsRequest(BaseModel):
    """Request model for searching models."""

    query: str = Field(..., description="Search query")
    max_results: int = Field(50, description="Maximum results")
    filter_dict: dict[str, Any] | None = Field(None, description="Additional filters")
    order_by: list[str] | None = Field(None, description="Ordering criteria")


class ModelRegistryStatsResponse(BaseModel):
    """Response model for model registry statistics."""

    total_models: int = Field(..., description="Total number of models")
    total_versions: int = Field(..., description="Total number of versions")
    average_versions_per_model: float = Field(
        ..., description="Average versions per model"
    )
    model_status_distribution: dict[str, int] = Field(
        ..., description="Model status counts"
    )
    version_status_distribution: dict[str, int] = Field(
        ..., description="Version status counts"
    )
    recent_models: list[dict[str, Any]] = Field(
        ..., description="Recently created models"
    )
    recent_versions: list[dict[str, Any]] = Field(
        ..., description="Recently created versions"
    )
    performance_trends: dict[str, Any] = Field(..., description="Performance trends")
    registry_health: dict[str, Any] = Field(..., description="Registry health metrics")


# ==================== Experiment Tracking Endpoints ====================


@router.post("/experiments/start", response_model=StartExperimentResponse)
async def start_experiment(
    request: StartExperimentRequest,
    current_user: dict = Depends(get_current_user),
    ml_service: AdvancedMLLifecycleService = Depends(get_advanced_ml_lifecycle_service),
    _: None = Depends(require_write),
) -> StartExperimentResponse:
    """Start a new ML experiment with advanced tracking capabilities.

    Creates a new experiment with comprehensive tracking features including
    automatic parameter, metric, and artifact logging.
    """
    try:
        experiment_id = await ml_service.start_experiment(
            name=request.name,
            description=request.description,
            experiment_type=request.experiment_type,
            objective=request.objective,
            created_by=current_user["username"],
            auto_log_parameters=request.auto_log_parameters,
            auto_log_metrics=request.auto_log_metrics,
            auto_log_artifacts=request.auto_log_artifacts,
            tags=request.tags,
            metadata=request.metadata,
        )

        return StartExperimentResponse(
            experiment_id=experiment_id,
            message=f"Experiment '{request.name}' started successfully",
        )
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Failed to start experiment: {str(e)}",
        )


@router.post("/experiments/{experiment_id}/runs/start", response_model=StartRunResponse)
async def start_run(
    experiment_id: str,
    request: StartRunRequest,
    current_user: dict = Depends(get_current_user),
    ml_service: AdvancedMLLifecycleService = Depends(get_advanced_ml_lifecycle_service),
    _: None = Depends(require_write),
) -> StartRunResponse:
    """Start a new experiment run with comprehensive tracking.

    Creates a new run within an experiment with automatic environment
    capture and tracking setup.
    """
    try:
        run_id = await ml_service.start_run(
            experiment_id=experiment_id,
            run_name=request.run_name,
            detector_id=request.detector_id,
            dataset_id=request.dataset_id,
            parameters=request.parameters,
            created_by=current_user["username"],
            parent_run_id=request.parent_run_id,
            tags=request.tags,
            description=request.description,
        )

        return StartRunResponse(
            run_id=run_id,
            experiment_id=experiment_id,
            message=f"Run '{request.run_name}' started successfully",
        )
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Failed to start run: {str(e)}",
        )


@router.post("/runs/{run_id}/parameters")
async def log_parameter(
    run_id: str,
    request: LogParameterRequest,
    ml_service: AdvancedMLLifecycleService = Depends(get_advanced_ml_lifecycle_service),
    _: None = Depends(require_write),
) -> dict[str, str]:
    """Log a parameter for the specified run."""
    try:
        await ml_service.log_parameter(run_id, request.key, request.value)
        return {"message": f"Parameter '{request.key}' logged successfully"}
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Failed to log parameter: {str(e)}",
        )


@router.post("/runs/{run_id}/metrics")
async def log_metric(
    run_id: str,
    request: LogMetricRequest,
    ml_service: AdvancedMLLifecycleService = Depends(get_advanced_ml_lifecycle_service),
    _: None = Depends(require_write),
) -> dict[str, str]:
    """Log a metric for the specified run."""
    try:
        await ml_service.log_metric(
            run_id, request.key, request.value, request.step, request.timestamp
        )
        return {"message": f"Metric '{request.key}' logged successfully"}
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Failed to log metric: {str(e)}",
        )


@router.post("/runs/{run_id}/artifacts", response_model=LogArtifactResponse)
async def log_artifact(
    run_id: str,
    request: LogArtifactRequest,
    ml_service: AdvancedMLLifecycleService = Depends(get_advanced_ml_lifecycle_service),
    _: None = Depends(require_write),
) -> LogArtifactResponse:
    """Log an artifact for the specified run."""
    try:
        artifact_path = await ml_service.log_artifact(
            run_id=run_id,
            artifact_name=request.artifact_name,
            artifact_data=request.artifact_data,
            artifact_type=request.artifact_type,
            metadata=request.metadata,
        )

        return LogArtifactResponse(
            artifact_path=artifact_path,
            message=f"Artifact '{request.artifact_name}' logged successfully",
        )
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Failed to log artifact: {str(e)}",
        )


@router.post("/runs/{run_id}/models", response_model=LogModelResponse)
async def log_model(
    run_id: str,
    request: LogModelRequest,
    ml_service: AdvancedMLLifecycleService = Depends(get_advanced_ml_lifecycle_service),
    _: None = Depends(require_write),
) -> LogModelResponse:
    """Log a trained model with the specified run."""
    try:
        # Get ML service instance
        from monorepo.infrastructure.config import get_container

        container = get_container()
        ml_service_instance = container.get_ml_service()

        # Convert request to model data
        model_data = {
            "name": request.model_name,
            "version": request.model_version,
            "algorithm": request.algorithm,
            "parameters": request.parameters,
            "metrics": request.metrics,
            "metadata": request.metadata,
            "tags": request.tags,
        }

        # Log model with the ML service
        model_id = await ml_service_instance.log_model(run_id, model_data)

        return LogModelResponse(
            model_id=model_id,
            model_name=request.model_name,
            model_version=request.model_version,
            logged_at=datetime.now(),
            status="logged",
            artifact_path=f"/models/{model_id}",
            metadata=request.metadata,
        )
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Failed to log model: {str(e)}",
        ) from e


@router.post("/runs/{run_id}/end")
async def end_run(
    run_id: str,
    request: EndRunRequest,
    ml_service: AdvancedMLLifecycleService = Depends(get_advanced_ml_lifecycle_service),
    _: None = Depends(require_write),
) -> dict[str, str]:
    """End the specified experiment run."""
    try:
        await ml_service.end_run(run_id, request.status, request.end_time)
        return {"message": f"Run '{run_id}' ended successfully"}
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Failed to end run: {str(e)}",
        )


# ==================== Model Versioning Endpoints ====================


@router.post("/model-versions", response_model=CreateModelVersionResponse)
async def create_model_version(
    request: CreateModelVersionRequest,
    current_user: dict = Depends(get_current_user),
    ml_service: AdvancedMLLifecycleService = Depends(get_advanced_ml_lifecycle_service),
    _: None = Depends(require_write),
) -> CreateModelVersionResponse:
    """Create a new model version with intelligent versioning."""
    try:
        version_id = await ml_service.create_model_version(
            model_name=request.model_name,
            run_id=request.run_id,
            model_path=request.model_path,
            performance_metrics=request.performance_metrics,
            description=request.description,
            tags=request.tags,
            auto_version=request.auto_version,
        )

        return CreateModelVersionResponse(
            model_version_id=version_id,
            message=f"Model version created for '{request.model_name}'",
        )
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Failed to create model version: {str(e)}",
        )


@router.post(
    "/model-versions/{model_version_id}/promote", response_model=PromoteModelResponse
)
async def promote_model_version(
    model_version_id: str,
    request: PromoteModelRequest,
    current_user: dict = Depends(get_current_user),
    ml_service: AdvancedMLLifecycleService = Depends(get_advanced_ml_lifecycle_service),
    _: None = Depends(require_write),
) -> PromoteModelResponse:
    """Promote a model version to a specific stage with validation."""
    try:
        result = await ml_service.promote_model_version(
            model_version_id=model_version_id,
            stage=request.stage,
            promoted_by=current_user["username"],
            approval_workflow=request.approval_workflow,
            validation_tests=request.validation_tests,
        )

        return PromoteModelResponse(**result)
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Failed to promote model version: {str(e)}",
        )


# ==================== Model Registry Endpoints ====================


@router.post("/models/search")
async def search_models(
    request: SearchModelsRequest,
    ml_service: AdvancedMLLifecycleService = Depends(get_advanced_ml_lifecycle_service),
) -> list[dict[str, Any]]:
    """Search models in the registry with advanced filtering."""
    try:
        results = await ml_service.search_models(
            query=request.query,
            max_results=request.max_results,
            filter_dict=request.filter_dict,
            order_by=request.order_by,
        )
        return results
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Failed to search models: {str(e)}",
        )


@router.get("/registry/stats", response_model=ModelRegistryStatsResponse)
async def get_model_registry_stats(
    ml_service: AdvancedMLLifecycleService = Depends(get_advanced_ml_lifecycle_service),
) -> ModelRegistryStatsResponse:
    """Get comprehensive model registry statistics and health metrics."""
    try:
        stats = await ml_service.get_model_registry_stats()
        return ModelRegistryStatsResponse(**stats)
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to get registry stats: {str(e)}",
        )


# ==================== Health and Status Endpoints ====================


@router.get("/health")
async def get_ml_lifecycle_health() -> dict[str, Any]:
    """Get ML lifecycle service health status."""
    return {
        "status": "healthy",
        "service": "advanced_ml_lifecycle",
        "version": "1.0.0",
        "features": [
            "experiment_tracking",
            "model_versioning",
            "model_registry",
            "validation_pipeline",
            "performance_monitoring",
        ],
        "timestamp": datetime.now(UTC).isoformat(),
    }


@router.get("/capabilities")
async def get_capabilities() -> dict[str, Any]:
    """Get advanced ML lifecycle service capabilities."""
    return {
        "experiment_tracking": {
            "auto_logging": True,
            "nested_runs": True,
            "artifact_storage": True,
            "environment_capture": True,
            "system_monitoring": True,
        },
        "model_versioning": {
            "semantic_versioning": True,
            "automatic_versioning": True,
            "performance_based_versioning": True,
            "lineage_tracking": True,
        },
        "model_registry": {
            "search": True,
            "filtering": True,
            "tagging": True,
            "performance_tracking": True,
            "statistics": True,
        },
        "validation": {
            "performance_baseline": True,
            "data_drift": True,
            "model_signature": True,
            "resource_usage": True,
            "custom_tests": True,
        },
        "promotion": {
            "stage_management": True,
            "approval_workflow": True,
            "validation_pipeline": True,
            "rollback_support": True,
        },
    }
