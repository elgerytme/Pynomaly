"""UC-002: Train Anomaly Detection Model use case implementation."""

from typing import Dict, Any
from dataclasses import dataclass

from ...domain.entities.dataset import Dataset
from ...domain.entities.model import Model
from ...domain.services.detection_service import DetectionService
from ...infrastructure.repositories.model_repository import ModelRepository


@dataclass
class TrainModelRequest:
    """Request for model training."""
    dataset: Dataset
    algorithm: str = "isolation_forest"
    parameters: Dict[str, Any] = None
    model_name: str = None


@dataclass
class TrainModelResponse:
    """Response from model training."""
    model_id: str = None
    model: Model = None
    success: bool = False
    error_message: str = None


class TrainModelUseCase:
    """Use case for training anomaly detection models."""
    
    def __init__(
        self,
        detection_service: DetectionService,
        model_repository: ModelRepository
    ):
        self._detection_service = detection_service
        self._model_repository = model_repository
    
    def execute(self, request: TrainModelRequest) -> TrainModelResponse:
        """Execute model training.
        
        Args:
            request: Training request
            
        Returns:
            Training response
        """
        try:
            # Validate training data
            if not request.dataset.is_valid():
                return TrainModelResponse(
                    success=False,
                    error_message="Invalid dataset format or quality"
                )
            
            if len(request.dataset.samples) < 100:
                return TrainModelResponse(
                    success=False,
                    error_message="Insufficient training data (minimum 100 samples required)"
                )
            
            # Train model
            model = self._detection_service.train(
                request.dataset,
                request.algorithm,
                request.parameters or {}
            )
            
            # Set model name if provided
            if request.model_name:
                model.name = request.model_name
            
            # Save model
            model_id = self._model_repository.save(model)
            
            return TrainModelResponse(
                model_id=model_id,
                model=model,
                success=True
            )
            
        except Exception as e:
            return TrainModelResponse(
                success=False,
                error_message=str(e)
            )