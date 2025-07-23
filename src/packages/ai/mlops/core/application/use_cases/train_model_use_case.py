"""Train model use case."""

from typing import Dict, Any, Optional
from uuid import UUID
from pydantic import BaseModel, Field

from ...domain.entities.model import Model
from ...domain.entities.experiment import Experiment
from ...domain.services.model_management_service import ModelManagementService
from ...domain.services.experiment_tracking_service import ExperimentTrackingService
from ...domain.services.model_optimization_service import ModelOptimizationService


class TrainModelRequest(BaseModel):
    """Request for training a model."""
    model_id: UUID
    training_config: Dict[str, Any]
    optimization_config: Optional[Dict[str, Any]] = None
    experiment_name: Optional[str] = None
    dataset_path: str
    validation_split: float = Field(default=0.2, ge=0.0, le=1.0)
    test_split: float = Field(default=0.1, ge=0.0, le=1.0)
    epochs: int = Field(default=10, ge=1)
    batch_size: int = Field(default=32, ge=1)
    learning_rate: float = Field(default=0.001, gt=0.0)
    save_checkpoints: bool = True
    early_stopping: bool = True
    

class TrainModelResponse(BaseModel):
    """Response for model training."""
    model_id: UUID
    experiment_id: UUID
    training_metrics: Dict[str, Any]
    validation_metrics: Dict[str, Any]
    model_path: str
    training_duration_seconds: float
    final_loss: float
    best_epoch: int
    status: str


class TrainModelUseCase:
    """Use case for training models."""
    
    def __init__(
        self,
        model_service: ModelManagementService,
        experiment_service: ExperimentTrackingService,
        optimization_service: ModelOptimizationService,
    ):
        self.model_service = model_service
        self.experiment_service = experiment_service
        self.optimization_service = optimization_service
    
    async def execute(self, request: TrainModelRequest) -> TrainModelResponse:
        """Execute model training use case."""
        # Get the model
        model = await self.model_service.get_model(request.model_id)
        if not model:
            raise ValueError(f"Model {request.model_id} not found")
        
        # Create experiment for tracking
        experiment_name = request.experiment_name or f"{model.name}_training"
        experiment = await self.experiment_service.create_experiment(
            name=experiment_name,
            description=f"Training experiment for model {model.name}",
            parameters={
                "model_id": str(request.model_id),
                "dataset_path": request.dataset_path,
                "epochs": request.epochs,
                "batch_size": request.batch_size,
                "learning_rate": request.learning_rate,
                "validation_split": request.validation_split,
                "test_split": request.test_split,
            },
        )
        
        # Start training process
        training_result = await self._train_model(
            model=model,
            experiment=experiment,
            request=request,
        )
        
        # Run optimization if configured
        if request.optimization_config:
            optimization_result = await self.optimization_service.optimize_hyperparameters(
                model_id=request.model_id,
                optimization_config=request.optimization_config,
            )
            training_result["optimization_metrics"] = optimization_result
        
        # Log final metrics
        await self.experiment_service.log_metrics(
            experiment_id=experiment.id,
            metrics=training_result["training_metrics"],
        )
        
        # Update model with training info
        await self.model_service.update_model(
            model_id=request.model_id,
            updates={
                "last_trained": training_result["completion_time"],
                "training_metrics": training_result["training_metrics"],
            },
        )
        
        return TrainModelResponse(
            model_id=request.model_id,
            experiment_id=experiment.id,
            training_metrics=training_result["training_metrics"],
            validation_metrics=training_result["validation_metrics"],
            model_path=training_result["model_path"],
            training_duration_seconds=training_result["duration_seconds"],
            final_loss=training_result["final_loss"],
            best_epoch=training_result["best_epoch"],
            status="completed",
        )
    
    async def _train_model(
        self,
        model: Model,
        experiment: Experiment,
        request: TrainModelRequest,
    ) -> Dict[str, Any]:
        """Execute the actual model training."""
        import time
        
        start_time = time.time()
        
        # Mock training process - in real implementation would use actual ML framework
        training_metrics = {
            "accuracy": 0.95,
            "precision": 0.93,
            "recall": 0.94,
            "f1_score": 0.935,
        }
        
        validation_metrics = {
            "accuracy": 0.92,
            "precision": 0.90,
            "recall": 0.91,
            "f1_score": 0.905,
        }
        
        # Simulate training duration
        end_time = time.time()
        duration = end_time - start_time
        
        return {
            "training_metrics": training_metrics,
            "validation_metrics": validation_metrics,
            "model_path": f"/models/{model.name}/trained_{int(end_time)}.pkl",
            "duration_seconds": duration,
            "final_loss": 0.05,
            "best_epoch": 8,
            "completion_time": end_time,
        }