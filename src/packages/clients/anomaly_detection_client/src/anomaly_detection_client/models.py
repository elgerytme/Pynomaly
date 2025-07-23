"""Data models for anomaly detection client."""

from datetime import datetime
from typing import Any, Dict, List, Optional, Union
from pydantic import BaseModel, Field

from sdk_core.models import BaseResponse, ModelInfo as BaseModelInfo


class DetectionRequest(BaseModel):
    """Request for anomaly detection."""
    
    data: List[List[float]] = Field(description="Input data points as 2D array")
    algorithm: str = Field(default="isolation_forest", description="Detection algorithm")
    contamination: float = Field(default=0.1, ge=0.0, le=0.5, description="Expected proportion of outliers")
    parameters: Dict[str, Any] = Field(default_factory=dict, description="Algorithm-specific parameters")
    
    class Config:
        schema_extra = {
            "example": {
                "data": [[1.0, 2.0], [2.0, 3.0], [100.0, 200.0]],
                "algorithm": "isolation_forest",
                "contamination": 0.1,
                "parameters": {"n_estimators": 100}
            }
        }


class DetectionResponse(BaseResponse):
    """Response from anomaly detection."""
    
    anomalies: List[int] = Field(description="Indices of detected anomalies")
    scores: Optional[List[float]] = Field(None, description="Anomaly scores for each data point")
    algorithm: str = Field(description="Algorithm used for detection")
    total_samples: int = Field(description="Total number of data points processed")
    anomaly_count: int = Field(description="Number of anomalies detected")
    processing_time_ms: float = Field(description="Processing time in milliseconds")


class EnsembleDetectionRequest(BaseModel):
    """Request for ensemble anomaly detection."""
    
    data: List[List[float]] = Field(description="Input data points as 2D array")
    algorithms: List[str] = Field(description="List of algorithms to use")
    voting_strategy: str = Field(default="majority", description="Voting strategy (majority, average, max)")
    contamination: float = Field(default=0.1, ge=0.0, le=0.5, description="Expected proportion of outliers")
    algorithm_parameters: Dict[str, Dict[str, Any]] = Field(
        default_factory=dict,
        description="Algorithm-specific parameters"
    )
    
    class Config:
        schema_extra = {
            "example": {
                "data": [[1.0, 2.0], [2.0, 3.0], [100.0, 200.0]],
                "algorithms": ["isolation_forest", "one_class_svm"],
                "voting_strategy": "majority",
                "contamination": 0.1
            }
        }


class EnsembleDetectionResponse(BaseResponse):
    """Response from ensemble anomaly detection."""
    
    anomalies: List[int] = Field(description="Indices of detected anomalies")
    ensemble_scores: List[float] = Field(description="Ensemble anomaly scores")
    individual_results: Dict[str, Dict[str, Any]] = Field(
        description="Results from individual algorithms"
    )
    voting_strategy: str = Field(description="Voting strategy used")
    algorithms_used: List[str] = Field(description="Algorithms used in ensemble")
    total_samples: int = Field(description="Total number of data points processed")
    anomaly_count: int = Field(description="Number of anomalies detected")
    processing_time_ms: float = Field(description="Processing time in milliseconds")


class ModelInfo(BaseModelInfo):
    """Information about a trained anomaly detection model."""
    
    algorithm: str = Field(description="Algorithm used")
    contamination: float = Field(description="Contamination parameter used during training")
    training_samples: int = Field(description="Number of samples used for training")
    parameters: Dict[str, Any] = Field(description="Model parameters")
    performance_metrics: Optional[Dict[str, float]] = Field(None, description="Model performance metrics")


class TrainingRequest(BaseModel):
    """Request to train an anomaly detection model."""
    
    data: List[List[float]] = Field(description="Training data points as 2D array")
    algorithm: str = Field(default="isolation_forest", description="Algorithm to train")
    name: str = Field(description="Name for the trained model")
    description: Optional[str] = Field(None, description="Optional model description")
    contamination: float = Field(default=0.1, ge=0.0, le=0.5, description="Expected proportion of outliers")
    parameters: Dict[str, Any] = Field(default_factory=dict, description="Algorithm-specific parameters")
    validation_split: float = Field(default=0.2, ge=0.0, le=0.5, description="Fraction of data for validation")
    
    class Config:
        schema_extra = {
            "example": {
                "data": [[1.0, 2.0], [2.0, 3.0], [3.0, 4.0]],
                "algorithm": "isolation_forest",
                "name": "production_model_v1",
                "contamination": 0.1,
                "parameters": {"n_estimators": 100}
            }
        }


class TrainingResponse(BaseResponse):
    """Response from model training."""
    
    model: ModelInfo = Field(description="Information about the trained model")
    training_metrics: Dict[str, float] = Field(description="Training performance metrics")
    validation_metrics: Optional[Dict[str, float]] = Field(None, description="Validation performance metrics")
    training_time_ms: float = Field(description="Training time in milliseconds")


class PredictionRequest(BaseModel):
    """Request for prediction using a trained model."""
    
    data: List[List[float]] = Field(description="Input data points as 2D array")
    model_id: str = Field(description="ID of the trained model to use")
    
    class Config:
        schema_extra = {
            "example": {
                "data": [[1.0, 2.0], [2.0, 3.0], [100.0, 200.0]],
                "model_id": "model_123"
            }
        }


class PredictionResponse(BaseResponse):
    """Response from model prediction."""
    
    anomalies: List[int] = Field(description="Indices of detected anomalies")
    scores: List[float] = Field(description="Anomaly scores for each data point")
    model_id: str = Field(description="ID of the model used")
    model_name: str = Field(description="Name of the model used")
    total_samples: int = Field(description="Total number of data points processed")
    anomaly_count: int = Field(description="Number of anomalies detected")
    processing_time_ms: float = Field(description="Processing time in milliseconds")


class AlgorithmInfo(BaseModel):
    """Information about an available algorithm."""
    
    name: str = Field(description="Algorithm name")
    display_name: str = Field(description="Human-readable algorithm name")
    description: str = Field(description="Algorithm description")
    parameters: Dict[str, Dict[str, Any]] = Field(description="Available parameters and their types")
    supports_online_learning: bool = Field(description="Whether algorithm supports online learning")
    supports_feature_importance: bool = Field(description="Whether algorithm provides feature importance")
    computational_complexity: str = Field(description="Computational complexity description")
    
    class Config:
        schema_extra = {
            "example": {
                "name": "isolation_forest",
                "display_name": "Isolation Forest",
                "description": "Isolation-based anomaly detection algorithm",
                "parameters": {
                    "n_estimators": {"type": "int", "default": 100, "range": [10, 1000]},
                    "contamination": {"type": "float", "default": 0.1, "range": [0.0, 0.5]}
                },
                "supports_online_learning": False,
                "supports_feature_importance": True,
                "computational_complexity": "O(n log n)"
            }
        }


class BatchDetectionRequest(BaseModel):
    """Request for batch anomaly detection."""
    
    datasets: List[Dict[str, Any]] = Field(description="List of datasets to process")
    algorithm: str = Field(default="isolation_forest", description="Detection algorithm")
    contamination: float = Field(default=0.1, description="Expected proportion of outliers")
    parameters: Dict[str, Any] = Field(default_factory=dict, description="Algorithm parameters")
    parallel_processing: bool = Field(default=True, description="Enable parallel processing")
    
    class Config:
        schema_extra = {
            "example": {
                "datasets": [
                    {"id": "dataset1", "data": [[1.0, 2.0], [2.0, 3.0]]},
                    {"id": "dataset2", "data": [[3.0, 4.0], [4.0, 5.0]]}
                ],
                "algorithm": "isolation_forest",
                "contamination": 0.1
            }
        }


class BatchDetectionResponse(BaseResponse):
    """Response from batch anomaly detection."""
    
    results: List[Dict[str, Any]] = Field(description="Detection results for each dataset")
    total_datasets: int = Field(description="Total number of datasets processed")
    successful_count: int = Field(description="Number of successfully processed datasets")
    failed_count: int = Field(description="Number of failed datasets")
    total_processing_time_ms: float = Field(description="Total processing time in milliseconds")