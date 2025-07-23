"""Model management endpoints."""

import numpy as np
from datetime import datetime
from typing import List, Dict, Any, Optional

from fastapi import APIRouter, Depends, HTTPException, status
from pydantic import BaseModel, Field

from ...infrastructure.repositories.model_repository import ModelRepository
from ...infrastructure.logging import get_logger

router = APIRouter()
logger = get_logger(__name__)


class ModelPredictionRequest(BaseModel):
    """Request model for predictions using saved models."""
    data: List[List[float]] = Field(..., description="Input data as list of feature vectors")
    model_id: str = Field(..., description="ID of the model to use for prediction")


class ModelListResponse(BaseModel):
    """Response model for listing models."""
    models: List[Dict[str, Any]] = Field(..., description="List of available models")
    total_count: int = Field(..., description="Total number of models")


class PredictionResponse(BaseModel):
    """Response model for model predictions."""
    success: bool = Field(..., description="Whether prediction completed successfully")
    anomalies: List[int] = Field(..., description="Indices of detected anomalies")
    scores: Optional[List[float]] = Field(None, description="Anomaly confidence scores")
    algorithm: str = Field(..., description="Algorithm used for prediction")
    model_id: str = Field(..., description="Model ID used")
    total_samples: int = Field(..., description="Total number of samples processed")
    anomalies_detected: int = Field(..., description="Number of anomalies detected")
    anomaly_rate: float = Field(..., description="Ratio of anomalies to total samples")
    timestamp: str = Field(..., description="Prediction timestamp")
    processing_time_ms: float = Field(..., description="Processing time in milliseconds")


# Dependency injection for model repository
_model_repository: Optional[ModelRepository] = None


def get_model_repository() -> ModelRepository:
    """Get model repository instance."""
    global _model_repository
    if _model_repository is None:
        _model_repository = ModelRepository()
    return _model_repository


@router.get("", response_model=ModelListResponse)
async def list_models(
    algorithm: Optional[str] = None,
    status: Optional[str] = None,
    model_repository: ModelRepository = Depends(get_model_repository)
) -> ModelListResponse:
    """List available trained models with optional filtering."""
    try:
        from ...domain.entities.model import ModelStatus
        
        status_filter = None
        if status:
            try:
                status_filter = ModelStatus(status)
            except ValueError:
                raise HTTPException(
                    status_code=status.HTTP_400_BAD_REQUEST,
                    detail=f"Invalid status: {status}"
                )
        
        models = model_repository.list_models(
            status=status_filter,
            algorithm=algorithm
        )
        
        return ModelListResponse(
            models=models,
            total_count=len(models)
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error("Error listing models", error=str(e))
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to list models: {str(e)}"
        )


@router.get("/{model_id}")
async def get_model_info(
    model_id: str,
    model_repository: ModelRepository = Depends(get_model_repository)
) -> Dict[str, Any]:
    """Get detailed information about a specific model."""
    try:
        metadata = model_repository.get_model_metadata(model_id)
        return metadata
        
    except FileNotFoundError:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Model with ID '{model_id}' not found"
        )
    except Exception as e:
        logger.error("Error getting model info", error=str(e))
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to get model info: {str(e)}"
        )


@router.post("/predict", response_model=PredictionResponse)
async def predict_with_model(
    request: ModelPredictionRequest,
    model_repository: ModelRepository = Depends(get_model_repository)
) -> PredictionResponse:
    """Make predictions using a saved model."""
    start_time = datetime.utcnow()
    
    logger.info("Processing prediction request with saved model",
                model_id=request.model_id,
                samples=len(request.data))
    
    try:
        # Validate input
        if not request.data or not request.data[0]:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Input data cannot be empty"
            )
        
        # Load model
        try:
            model = model_repository.load(request.model_id)
        except FileNotFoundError:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Model with ID '{request.model_id}' not found"
            )
        
        # Convert to numpy array
        data_array = np.array(request.data, dtype=np.float64)
        
        # Make predictions
        predictions = model.predict(data_array)
        
        try:
            scores = model.get_anomaly_scores(data_array)
        except:
            scores = None
        
        # Calculate statistics
        anomaly_count = int(np.sum(predictions == -1))
        total_samples = len(predictions)
        anomaly_rate = anomaly_count / total_samples if total_samples > 0 else 0.0
        anomaly_indices = np.where(predictions == -1)[0].tolist()
        
        end_time = datetime.utcnow()
        processing_time_ms = (end_time - start_time).total_seconds() * 1000
        
        return PredictionResponse(
            success=True,
            anomalies=anomaly_indices,
            scores=scores.tolist() if scores is not None else None,
            algorithm=model.metadata.algorithm,
            model_id=request.model_id,
            total_samples=total_samples,
            anomalies_detected=anomaly_count,
            anomaly_rate=anomaly_rate,
            timestamp=end_time.isoformat(),
            processing_time_ms=processing_time_ms
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error("Model prediction error", error=str(e))
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Prediction failed: {str(e)}"
        )


@router.delete("/{model_id}")
async def delete_model(
    model_id: str,
    model_repository: ModelRepository = Depends(get_model_repository)
) -> Dict[str, str]:
    """Delete a saved model."""
    try:
        if model_repository.delete(model_id):
            return {"message": f"Model '{model_id}' deleted successfully"}
        else:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Model '{model_id}' not found"
            )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error("Error deleting model", model_id=model_id, error=str(e))
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to delete model: {str(e)}"
        )


@router.get("/stats/repository")
async def get_repository_stats(
    model_repository: ModelRepository = Depends(get_model_repository)
) -> Dict[str, Any]:
    """Get model repository statistics."""
    try:
        stats = model_repository.get_repository_stats()
        return {
            "repository_stats": stats,
            "timestamp": datetime.utcnow().isoformat()
        }
        
    except Exception as e:
        logger.error("Error getting repository stats", error=str(e))
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to get repository stats: {str(e)}"
        )