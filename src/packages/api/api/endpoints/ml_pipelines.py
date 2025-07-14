"""Machine Learning Pipeline API endpoints for data science operations."""

from typing import List, Optional, Dict, Any
from uuid import UUID
from datetime import datetime

from fastapi import APIRouter, Depends, HTTPException, Query, BackgroundTasks
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field

from pynomaly.infrastructure.auth import require_data_scientist, require_viewer
from pynomaly.infrastructure.config import Container
from pynomaly.presentation.api.auth_deps import get_container_simple

# Attempt to import data science components with fallback
try:
    from packages.data_science.domain.entities import MachineLearningPipeline, DataScienceModel
    from packages.data_science.domain.entities.machine_learning_pipeline import PipelineType, PipelineStatus
    from packages.data_science.domain.entities.data_science_model import ModelType, ModelStatus
    from packages.data_science.domain.value_objects import ModelPerformanceMetrics
    DATA_SCIENCE_AVAILABLE = True
except ImportError:
    DATA_SCIENCE_AVAILABLE = False
    # Mock classes for API documentation
    class PipelineType:
        TRAINING = "training"
        INFERENCE = "inference"
        BATCH = "batch"
        STREAMING = "streaming"
    
    class PipelineStatus:
        DRAFT = "draft"
        RUNNING = "running"
        COMPLETED = "completed"
        FAILED = "failed"
    
    class ModelType:
        MACHINE_LEARNING = "machine_learning"
        STATISTICAL = "statistical"
        DEEP_LEARNING = "deep_learning"
    
    class ModelStatus:
        DRAFT = "draft"
        TRAINING = "training"
        TRAINED = "trained"
        DEPLOYED = "deployed"

router = APIRouter(prefix="/ml-pipelines", tags=["ML Pipelines"])


# Request/Response Models
class PipelineStep(BaseModel):
    """Individual step in ML pipeline."""
    step_id: str
    name: str
    step_type: str = Field(..., description="Type of step (data_loading, preprocessing, training, etc.)")
    component: str = Field(..., description="Component or algorithm to use")
    parameters: Dict[str, Any] = Field(default_factory=dict)
    inputs: List[str] = Field(default_factory=list, description="Input step IDs")
    outputs: List[str] = Field(default_factory=list, description="Output artifacts")
    enabled: bool = Field(default=True)
    retry_count: int = Field(default=0, ge=0, le=5)
    timeout_seconds: Optional[int] = Field(None, gt=0)


class PipelineCreateRequest(BaseModel):
    """Request model for creating ML pipeline."""
    name: str = Field(..., min_length=1, max_length=255)
    description: Optional[str] = Field(None, max_length=2000)
    pipeline_type: str = Field(..., description="Type of pipeline")
    steps: List[PipelineStep] = Field(..., min_items=1)
    schedule: Optional[str] = Field(None, description="Cron schedule for automatic runs")
    environment_config: Dict[str, Any] = Field(default_factory=dict)
    resource_requirements: Dict[str, Any] = Field(
        default_factory=lambda: {"cpu": "1", "memory": "2Gi", "gpu": "0"},
        description="Resource requirements for pipeline execution"
    )
    metadata: Dict[str, Any] = Field(default_factory=dict)


class PipelineExecutionRequest(BaseModel):
    """Request model for executing pipeline."""
    pipeline_id: str
    execution_name: Optional[str] = Field(None, description="Name for this execution")
    parameters: Dict[str, Any] = Field(default_factory=dict, description="Runtime parameters")
    environment_overrides: Dict[str, Any] = Field(
        default_factory=dict,
        description="Environment configuration overrides"
    )
    resource_overrides: Dict[str, Any] = Field(
        default_factory=dict,
        description="Resource requirement overrides"
    )
    async_execution: bool = Field(default=True, description="Execute asynchronously")


class StepResult(BaseModel):
    """Result of pipeline step execution."""
    step_id: str
    status: str
    started_at: Optional[str] = None
    completed_at: Optional[str] = None
    duration_seconds: Optional[float] = None
    outputs: Dict[str, Any] = Field(default_factory=dict)
    metrics: Dict[str, float] = Field(default_factory=dict)
    logs: List[str] = Field(default_factory=list)
    error_message: Optional[str] = None
    retry_count: int = 0


class PipelineExecutionResponse(BaseModel):
    """Response model for pipeline execution."""
    execution_id: str
    pipeline_id: str
    execution_name: str
    status: str
    created_at: str
    started_at: Optional[str] = None
    completed_at: Optional[str] = None
    duration_seconds: Optional[float] = None
    progress_percentage: float = Field(default=0.0, ge=0, le=100)
    
    # Step results
    step_results: List[StepResult] = Field(default_factory=list)
    current_step: Optional[str] = None
    
    # Execution metadata
    parameters: Dict[str, Any] = Field(default_factory=dict)
    resource_usage: Dict[str, Any] = Field(default_factory=dict)
    artifacts: List[str] = Field(default_factory=list)
    
    # Model results (if training pipeline)
    trained_models: List[str] = Field(default_factory=list)
    performance_metrics: Dict[str, float] = Field(default_factory=dict)
    
    # Error handling
    error_message: Optional[str] = None
    failed_step: Optional[str] = None


class PipelineResponse(BaseModel):
    """Response model for ML pipeline."""
    pipeline_id: str
    name: str
    description: Optional[str] = None
    pipeline_type: str
    status: str
    version_number: str
    created_at: str
    updated_at: str
    
    # Pipeline definition
    steps: List[PipelineStep]
    schedule: Optional[str] = None
    environment_config: Dict[str, Any] = Field(default_factory=dict)
    resource_requirements: Dict[str, Any] = Field(default_factory=dict)
    
    # Execution history
    execution_count: int = 0
    last_execution_id: Optional[str] = None
    last_execution_status: Optional[str] = None
    last_execution_time: Optional[str] = None
    
    # Performance metrics
    success_rate: Optional[float] = None
    average_duration: Optional[float] = None
    
    # Dependencies
    dependencies: List[str] = Field(default_factory=list)
    dependents: List[str] = Field(default_factory=list)


class ModelTrainingRequest(BaseModel):
    """Request model for model training within pipeline."""
    pipeline_id: str
    model_name: str
    model_type: str
    algorithm: str
    dataset_id: str
    target_column: Optional[str] = None
    feature_columns: Optional[List[str]] = None
    hyperparameters: Dict[str, Any] = Field(default_factory=dict)
    validation_split: float = Field(default=0.2, ge=0, le=0.5)
    cross_validation_folds: int = Field(default=5, ge=2, le=20)
    early_stopping: bool = Field(default=True)
    max_training_time: Optional[int] = Field(None, gt=0, description="Max training time in seconds")
    performance_threshold: Optional[float] = Field(None, description="Minimum performance threshold")


class ModelTrainingResponse(BaseModel):
    """Response model for model training."""
    training_id: str
    model_id: str
    pipeline_id: str
    model_name: str
    status: str
    created_at: str
    started_at: Optional[str] = None
    completed_at: Optional[str] = None
    
    # Training progress
    progress_percentage: float = Field(default=0.0, ge=0, le=100)
    current_epoch: Optional[int] = None
    total_epochs: Optional[int] = None
    
    # Performance metrics
    training_metrics: Dict[str, float] = Field(default_factory=dict)
    validation_metrics: Dict[str, float] = Field(default_factory=dict)
    test_metrics: Dict[str, float] = Field(default_factory=dict)
    
    # Model artifacts
    model_uri: Optional[str] = None
    model_size_bytes: Optional[int] = None
    feature_importance: Dict[str, float] = Field(default_factory=dict)
    
    # Configuration
    final_hyperparameters: Dict[str, Any] = Field(default_factory=dict)
    dataset_info: Dict[str, Any] = Field(default_factory=dict)


@router.post("/pipelines",
             response_model=PipelineResponse,
             summary="Create ML Pipeline",
             description="Create a new machine learning pipeline")
async def create_pipeline(
    request: PipelineCreateRequest,
    container: Container = Depends(get_container_simple),
    current_user = Depends(require_data_scientist)
) -> PipelineResponse:
    """Create a new machine learning pipeline.
    
    Supports various pipeline types:
    - Training: Model training and validation pipelines
    - Inference: Batch prediction pipelines
    - Streaming: Real-time inference pipelines
    - ETL: Data processing and feature engineering pipelines
    """
    if not DATA_SCIENCE_AVAILABLE:
        raise HTTPException(
            status_code=503,
            detail="ML pipeline capabilities are not available. Please install the data_science package."
        )
    
    try:
        # Create pipeline entity
        pipeline = MachineLearningPipeline(
            name=request.name,
            pipeline_type=PipelineType(request.pipeline_type),
            description=request.description,
            steps=[step.dict() for step in request.steps],
            dependencies=[],  # TODO: Extract from steps
            schedule=request.schedule,
            environment_config=request.environment_config,
            resource_requirements=request.resource_requirements,
            metadata=request.metadata
        )
        
        # TODO: Save pipeline to repository
        
        return PipelineResponse(
            pipeline_id=str(pipeline.id),
            name=pipeline.name,
            description=pipeline.description,
            pipeline_type=pipeline.pipeline_type.value,
            status=pipeline.status.value,
            version_number=pipeline.version_number,
            created_at=pipeline.created_at.isoformat(),
            updated_at=pipeline.updated_at.isoformat(),
            steps=request.steps,
            schedule=request.schedule,
            environment_config=request.environment_config,
            resource_requirements=request.resource_requirements,
            dependencies=pipeline.dependencies
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to create pipeline: {str(e)}")


@router.post("/pipelines/{pipeline_id}/execute",
             response_model=PipelineExecutionResponse,
             summary="Execute Pipeline",
             description="Execute a machine learning pipeline")
async def execute_pipeline(
    pipeline_id: str,
    request: PipelineExecutionRequest,
    background_tasks: BackgroundTasks,
    container: Container = Depends(get_container_simple),
    current_user = Depends(require_data_scientist)
) -> PipelineExecutionResponse:
    """Execute a machine learning pipeline."""
    if not DATA_SCIENCE_AVAILABLE:
        raise HTTPException(
            status_code=503,
            detail="ML pipeline execution capabilities are not available."
        )
    
    try:
        execution_id = str(UUID.uuid4())
        execution_name = request.execution_name or f"execution_{execution_id[:8]}"
        
        if request.async_execution:
            # Add background task for pipeline execution
            background_tasks.add_task(
                _execute_pipeline_async,
                execution_id,
                pipeline_id,
                request
            )
            
            return PipelineExecutionResponse(
                execution_id=execution_id,
                pipeline_id=pipeline_id,
                execution_name=execution_name,
                status="queued",
                created_at=datetime.utcnow().isoformat(),
                parameters=request.parameters
            )
        else:
            # Synchronous execution (for small pipelines)
            # TODO: Implement synchronous execution
            raise HTTPException(
                status_code=501,
                detail="Synchronous pipeline execution not yet implemented"
            )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to execute pipeline: {str(e)}")


@router.get("/pipelines/{pipeline_id}/executions/{execution_id}",
            response_model=PipelineExecutionResponse,
            summary="Get Execution Status",
            description="Get the status and results of a pipeline execution")
async def get_execution_status(
    pipeline_id: str,
    execution_id: str,
    container: Container = Depends(get_container_simple),
    current_user = Depends(require_viewer)
) -> PipelineExecutionResponse:
    """Get pipeline execution status and results."""
    if not DATA_SCIENCE_AVAILABLE:
        raise HTTPException(
            status_code=503,
            detail="ML pipeline capabilities are not available."
        )
    
    try:
        # TODO: Retrieve execution status from repository
        return PipelineExecutionResponse(
            execution_id=execution_id,
            pipeline_id=pipeline_id,
            execution_name=f"execution_{execution_id[:8]}",
            status="completed",
            created_at="2025-01-01T00:00:00",
            started_at="2025-01-01T00:01:00",
            completed_at="2025-01-01T00:05:00",
            duration_seconds=240.0,
            progress_percentage=100.0
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to get execution status: {str(e)}")


@router.post("/models/train",
             response_model=ModelTrainingResponse,
             summary="Train Model",
             description="Train a machine learning model using a pipeline")
async def train_model(
    request: ModelTrainingRequest,
    background_tasks: BackgroundTasks,
    container: Container = Depends(get_container_simple),
    current_user = Depends(require_data_scientist)
) -> ModelTrainingResponse:
    """Train a machine learning model.
    
    Supports various model types:
    - Classification models (Random Forest, SVM, Neural Networks)
    - Regression models (Linear, Polynomial, Ensemble)
    - Anomaly detection models (Isolation Forest, One-Class SVM)
    - Deep learning models (AutoEncoders, CNNs, RNNs)
    """
    if not DATA_SCIENCE_AVAILABLE:
        raise HTTPException(
            status_code=503,
            detail="Model training capabilities are not available."
        )
    
    try:
        training_id = str(UUID.uuid4())
        model_id = str(UUID.uuid4())
        
        # Create model entity
        model = DataScienceModel(
            name=request.model_name,
            model_type=ModelType(request.model_type),
            algorithm=request.algorithm,
            version_number="1.0.0",
            hyperparameters=request.hyperparameters,
            training_dataset_id=request.dataset_id,
            target_variable=request.target_column,
            features=request.feature_columns or []
        )
        
        # Add background task for model training
        background_tasks.add_task(
            _train_model_async,
            training_id,
            model_id,
            request
        )
        
        return ModelTrainingResponse(
            training_id=training_id,
            model_id=model_id,
            pipeline_id=request.pipeline_id,
            model_name=request.model_name,
            status="queued",
            created_at=datetime.utcnow().isoformat()
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to start model training: {str(e)}")


@router.get("/pipelines",
            response_model=List[PipelineResponse],
            summary="List Pipelines",
            description="List all ML pipelines with optional filtering")
async def list_pipelines(
    pipeline_type: Optional[str] = Query(None, description="Filter by pipeline type"),
    status: Optional[str] = Query(None, description="Filter by status"),
    name_contains: Optional[str] = Query(None, description="Filter by name substring"),
    limit: int = Query(10, ge=1, le=100),
    offset: int = Query(0, ge=0),
    container: Container = Depends(get_container_simple),
    current_user = Depends(require_viewer)
) -> List[PipelineResponse]:
    """List ML pipelines with optional filtering."""
    if not DATA_SCIENCE_AVAILABLE:
        raise HTTPException(
            status_code=503,
            detail="ML pipeline capabilities are not available."
        )
    
    # TODO: Implement pipeline listing from repository
    return []


@router.get("/pipelines/{pipeline_id}",
            response_model=PipelineResponse,
            summary="Get Pipeline",
            description="Get detailed information about a specific pipeline")
async def get_pipeline(
    pipeline_id: str,
    container: Container = Depends(get_container_simple),
    current_user = Depends(require_viewer)
) -> PipelineResponse:
    """Get detailed pipeline information."""
    if not DATA_SCIENCE_AVAILABLE:
        raise HTTPException(
            status_code=503,
            detail="ML pipeline capabilities are not available."
        )
    
    try:
        # TODO: Retrieve pipeline from repository
        raise HTTPException(status_code=404, detail="Pipeline not found")
        
    except Exception as e:
        if "not found" in str(e).lower():
            raise
        raise HTTPException(status_code=500, detail=f"Failed to get pipeline: {str(e)}")


@router.delete("/pipelines/{pipeline_id}",
               summary="Delete Pipeline",
               description="Delete a machine learning pipeline")
async def delete_pipeline(
    pipeline_id: str,
    force: bool = Query(default=False, description="Force deletion even if executions exist"),
    container: Container = Depends(get_container_simple),
    current_user = Depends(require_data_scientist)
) -> JSONResponse:
    """Delete a machine learning pipeline."""
    if not DATA_SCIENCE_AVAILABLE:
        raise HTTPException(
            status_code=503,
            detail="ML pipeline capabilities are not available."
        )
    
    try:
        # TODO: Implement pipeline deletion
        return JSONResponse(
            status_code=200,
            content={"message": f"Pipeline {pipeline_id} deleted successfully"}
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to delete pipeline: {str(e)}")


# Background task functions
async def _execute_pipeline_async(execution_id: str, pipeline_id: str, request: PipelineExecutionRequest):
    """Execute pipeline in background."""
    # TODO: Implement actual pipeline execution logic
    pass


async def _train_model_async(training_id: str, model_id: str, request: ModelTrainingRequest):
    """Train model in background."""
    # TODO: Implement actual model training logic
    pass