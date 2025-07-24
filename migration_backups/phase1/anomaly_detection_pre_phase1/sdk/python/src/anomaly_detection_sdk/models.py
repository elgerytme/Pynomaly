"""Data models for the Anomaly Detection SDK."""

from typing import Dict, List, Optional, Any, Union
from datetime import datetime
from enum import Enum
from pydantic import BaseModel, Field


class AlgorithmType(str, Enum):
    """Supported anomaly detection algorithms."""
    ISOLATION_FOREST = "isolation_forest"
    LOCAL_OUTLIER_FACTOR = "local_outlier_factor"
    ONE_CLASS_SVM = "one_class_svm"
    ELLIPTIC_ENVELOPE = "elliptic_envelope"
    AUTOENCODER = "autoencoder"
    ENSEMBLE = "ensemble"


class AnomalyData(BaseModel):
    """Represents an individual anomaly detection."""
    index: int = Field(..., description="Index of the anomalous data point")
    score: float = Field(..., description="Anomaly score (higher = more anomalous)")
    data_point: List[float] = Field(..., description="Original data point values")
    confidence: Optional[float] = Field(None, description="Confidence level of detection")
    timestamp: Optional[datetime] = Field(None, description="Detection timestamp")


class DetectionResult(BaseModel):
    """Result of anomaly detection operation."""
    anomalies: List[AnomalyData] = Field(..., description="List of detected anomalies")
    total_points: int = Field(..., description="Total number of data points analyzed")
    anomaly_count: int = Field(..., description="Number of anomalies detected")
    algorithm_used: AlgorithmType = Field(..., description="Algorithm used for detection")
    execution_time: float = Field(..., description="Processing time in seconds")
    model_version: Optional[str] = Field(None, description="Version of the model used")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Additional metadata")


class ModelInfo(BaseModel):
    """Information about a trained model."""
    model_id: str = Field(..., description="Unique model identifier")
    algorithm: AlgorithmType = Field(..., description="Algorithm type")
    created_at: datetime = Field(..., description="Model creation timestamp")
    training_data_size: int = Field(..., description="Size of training dataset")
    performance_metrics: Dict[str, float] = Field(default_factory=dict, description="Model performance metrics")
    hyperparameters: Dict[str, Any] = Field(default_factory=dict, description="Model hyperparameters")
    version: str = Field(..., description="Model version")
    status: str = Field(..., description="Model status (trained, deployed, etc.)")


class StreamingConfig(BaseModel):
    """Configuration for streaming detection."""
    buffer_size: int = Field(100, description="Size of the streaming buffer")
    detection_threshold: float = Field(0.5, description="Threshold for anomaly detection")
    batch_size: int = Field(10, description="Batch size for processing")
    algorithm: AlgorithmType = Field(AlgorithmType.ISOLATION_FOREST, description="Algorithm to use")
    auto_retrain: bool = Field(False, description="Enable automatic model retraining")


class ExplanationResult(BaseModel):
    """Result of anomaly explanation."""
    anomaly_index: int = Field(..., description="Index of the explained anomaly")
    feature_importance: Dict[str, float] = Field(..., description="Feature importance scores")
    shap_values: Optional[List[float]] = Field(None, description="SHAP values for features")
    lime_explanation: Optional[Dict[str, Any]] = Field(None, description="LIME explanation")
    explanation_text: str = Field(..., description="Human-readable explanation")
    confidence: float = Field(..., description="Confidence in the explanation")


class HealthStatus(BaseModel):
    """Health status of the service."""
    status: str = Field(..., description="Overall health status")
    timestamp: datetime = Field(..., description="Health check timestamp")
    version: str = Field(..., description="Service version")
    uptime: float = Field(..., description="Service uptime in seconds")
    components: Dict[str, str] = Field(default_factory=dict, description="Component health status")
    metrics: Dict[str, Union[int, float, str]] = Field(default_factory=dict, description="Health metrics")


class BatchProcessingRequest(BaseModel):
    """Request for batch processing."""
    data: List[List[float]] = Field(..., description="Batch data to process")
    algorithm: AlgorithmType = Field(AlgorithmType.ISOLATION_FOREST, description="Algorithm to use")
    parameters: Dict[str, Any] = Field(default_factory=dict, description="Algorithm parameters")
    return_explanations: bool = Field(False, description="Whether to return explanations")


class TrainingRequest(BaseModel):
    """Request for model training."""
    data: List[List[float]] = Field(..., description="Training data")
    algorithm: AlgorithmType = Field(..., description="Algorithm to train")
    hyperparameters: Dict[str, Any] = Field(default_factory=dict, description="Training hyperparameters")
    validation_split: float = Field(0.2, description="Validation data split ratio")
    model_name: Optional[str] = Field(None, description="Name for the trained model")


class TrainingResult(BaseModel):
    """Result of model training."""
    model_id: str = Field(..., description="ID of the trained model")
    training_time: float = Field(..., description="Training time in seconds")
    performance_metrics: Dict[str, float] = Field(..., description="Training performance metrics")
    validation_metrics: Dict[str, float] = Field(..., description="Validation performance metrics")
    model_info: ModelInfo = Field(..., description="Detailed model information")