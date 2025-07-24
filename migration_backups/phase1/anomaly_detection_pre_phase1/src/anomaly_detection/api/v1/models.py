"""Model management endpoints."""

import numpy as np
import pandas as pd
from datetime import datetime
from typing import List, Dict, Any, Optional
import uuid

from fastapi import APIRouter, Depends, HTTPException, status
from pydantic import BaseModel, Field

from ...infrastructure.repositories.model_repository import ModelRepository
from ...domain.entities.model import Model, ModelMetadata, ModelStatus, SerializationFormat
from ...domain.entities.dataset import Dataset, DatasetType, DatasetMetadata
from ...domain.services.detection_service import DetectionService
from ...infrastructure.logging import get_logger
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

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


class TrainingRequest(BaseModel):
    """Request model for training a new model."""
    model_name: str = Field(..., description="Name for the trained model")
    algorithm: str = Field("isolation_forest", description="Algorithm to train (isolation_forest, one_class_svm, lof)")
    contamination: float = Field(0.1, ge=0.0, le=0.5, description="Contamination rate (0.0-0.5)")
    data: List[List[float]] = Field(..., description="Training data as list of feature vectors")
    labels: Optional[List[int]] = Field(None, description="Ground truth labels (-1 for anomaly, 1 for normal)")
    feature_names: Optional[List[str]] = Field(None, description="Names of features")
    description: Optional[str] = Field(None, description="Model description")
    hyperparameters: Optional[Dict[str, Any]] = Field(None, description="Additional hyperparameters")


class TrainingResponse(BaseModel):
    """Response model for model training."""
    success: bool = Field(..., description="Whether training completed successfully")
    model_id: str = Field(..., description="ID of the trained model")
    model_name: str = Field(..., description="Name of the trained model")
    algorithm: str = Field(..., description="Algorithm used for training")
    training_samples: int = Field(..., description="Number of training samples")
    training_features: int = Field(..., description="Number of features")
    contamination_rate: float = Field(..., description="Contamination rate used")
    training_duration_seconds: float = Field(..., description="Training time in seconds")
    accuracy: Optional[float] = Field(None, description="Model accuracy (if labels provided)")
    precision: Optional[float] = Field(None, description="Model precision (if labels provided)")
    recall: Optional[float] = Field(None, description="Model recall (if labels provided)")
    f1_score: Optional[float] = Field(None, description="Model F1-score (if labels provided)")
    timestamp: str = Field(..., description="Training completion timestamp")


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


@router.post("/train", response_model=TrainingResponse)
async def train_model(
    request: TrainingRequest,
    model_repository: ModelRepository = Depends(get_model_repository)
) -> TrainingResponse:
    """Train and save a new anomaly detection model."""
    start_time = datetime.utcnow()
    
    logger.info("Starting model training",
                model_name=request.model_name,
                algorithm=request.algorithm,
                samples=len(request.data))
    
    try:
        # Validate input data
        if not request.data or not request.data[0]:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Training data cannot be empty"
            )
        
        # Convert to numpy array
        data_array = np.array(request.data, dtype=np.float64)
        labels_array = np.array(request.labels) if request.labels else None
        
        # Validate data dimensions
        if len(data_array.shape) != 2:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Training data must be a 2D array"
            )
        
        # Validate labels if provided
        if labels_array is not None and len(labels_array) != len(data_array):
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Labels length must match data length"
            )
        
        # Create dataset entity
        df = pd.DataFrame(data_array, columns=request.feature_names or [f"feature_{i}" for i in range(data_array.shape[1])])
        dataset = Dataset(
            data=df,
            dataset_type=DatasetType.TRAINING,
            labels=labels_array,
            metadata=DatasetMetadata(
                name=f"{request.model_name}_training_data",
                source="api_upload",
                description=request.description or f"Training dataset for {request.model_name}"
            )
        )
        
        # Validate dataset
        validation_issues = dataset.validate()
        if validation_issues:
            logger.warning("Dataset validation issues", issues=validation_issues)
        
        # Initialize detection service and train model
        service = DetectionService()
        
        # Algorithm mapping
        algorithm_map = {
            'isolation_forest': 'iforest',
            'one_class_svm': 'ocsvm',
            'lof': 'lof'
        }
        
        mapped_algorithm = algorithm_map.get(request.algorithm, request.algorithm)
        
        # Fit the model
        service.fit(data_array, mapped_algorithm, contamination=request.contamination)
        
        # Get predictions for evaluation
        detection_result = service.detect_anomalies(
            data=data_array,
            algorithm=mapped_algorithm,
            contamination=request.contamination
        )
        
        end_time = datetime.utcnow()
        training_duration = (end_time - start_time).total_seconds()
        
        # Calculate metrics if labels available
        accuracy, precision, recall, f1_score_val = None, None, None, None
        if labels_array is not None:
            pred_labels = detection_result.predictions
            accuracy = float(accuracy_score(labels_array, pred_labels))
            precision = float(precision_score(labels_array, pred_labels, pos_label=-1, zero_division=0, average='binary'))
            recall = float(recall_score(labels_array, pred_labels, pos_label=-1, zero_division=0, average='binary'))
            f1_score_val = float(f1_score(labels_array, pred_labels, pos_label=-1, zero_division=0, average='binary'))
        
        # Create model entity
        model_id = str(uuid.uuid4())
        metadata = ModelMetadata(
            model_id=model_id,
            name=request.model_name,
            algorithm=request.algorithm,
            status=ModelStatus.TRAINED,
            training_samples=dataset.n_samples,
            training_features=dataset.n_features,
            contamination_rate=request.contamination,
            training_duration_seconds=training_duration,
            accuracy=accuracy,
            precision=precision,
            recall=recall,
            f1_score=f1_score_val,
            feature_names=dataset.feature_names,
            hyperparameters=request.hyperparameters or {'contamination': request.contamination},
            description=request.description or f"Trained {request.algorithm} model via API",
        )
        
        # Get the trained model object from the service
        trained_model_obj = service._fitted_models.get(mapped_algorithm)
        
        model = Model(
            metadata=metadata,
            model_object=trained_model_obj
        )
        
        # Save model using repository
        saved_model_id = model_repository.save(model, SerializationFormat.PICKLE)
        
        logger.info("Model training completed",
                   model_id=saved_model_id,
                   duration_seconds=training_duration,
                   accuracy=accuracy)
        
        return TrainingResponse(
            success=True,
            model_id=saved_model_id,
            model_name=request.model_name,
            algorithm=request.algorithm,
            training_samples=dataset.n_samples,
            training_features=dataset.n_features,
            contamination_rate=request.contamination,
            training_duration_seconds=training_duration,
            accuracy=accuracy,
            precision=precision,
            recall=recall,
            f1_score=f1_score_val,
            timestamp=end_time.isoformat()
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error("Model training error", error=str(e))
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Training failed: {str(e)}"
        )