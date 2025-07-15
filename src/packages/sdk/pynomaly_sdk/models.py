"""
Pynomaly SDK Data Models

Pydantic models for API requests, responses, and data structures
used throughout the SDK.
"""

from datetime import datetime
from typing import Any, Dict, List, Optional, Union
from uuid import UUID

from pydantic import BaseModel, Field, ConfigDict
import pandas as pd
import numpy as np


class APIResponse(BaseModel):
    """Generic API response wrapper."""
    
    model_config = ConfigDict(arbitrary_types_allowed=True)
    
    status_code: int
    data: Dict[str, Any]
    headers: Dict[str, str] = Field(default_factory=dict)
    success: bool = True


class DetectorConfig(BaseModel):
    """Configuration for anomaly detectors."""
    
    algorithm_name: str = Field(..., description="Name of the detection algorithm")
    hyperparameters: Dict[str, Any] = Field(default_factory=dict, description="Algorithm hyperparameters")
    contamination_rate: float = Field(0.1, ge=0.0, le=1.0, description="Expected contamination rate")
    random_state: Optional[int] = Field(None, description="Random seed for reproducibility")
    n_jobs: int = Field(1, description="Number of parallel jobs")
    
    class Config:
        json_schema_extra = {
            "example": {
                "algorithm_name": "IsolationForest",
                "hyperparameters": {"n_estimators": 100, "max_samples": "auto"},
                "contamination_rate": 0.1,
                "random_state": 42,
                "n_jobs": -1
            }
        }


class Dataset(BaseModel):
    """Dataset representation for API operations."""
    
    model_config = ConfigDict(arbitrary_types_allowed=True)
    
    name: str = Field(..., description="Dataset name")
    data: Union[pd.DataFrame, Dict[str, Any], List[Dict[str, Any]]] = Field(..., description="Dataset content")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Dataset metadata")
    feature_names: Optional[List[str]] = Field(None, description="Feature column names")
    target_column: Optional[str] = Field(None, description="Target column name")
    
    @classmethod
    def from_dataframe(
        cls, 
        name: str, 
        df: pd.DataFrame, 
        target_column: Optional[str] = None,
        **metadata
    ) -> "Dataset":
        """Create Dataset from pandas DataFrame."""
        return cls(
            name=name,
            data=df,
            feature_names=df.columns.tolist(),
            target_column=target_column,
            metadata=metadata
        )
    
    @classmethod
    def from_numpy(
        cls,
        name: str,
        data: np.ndarray,
        feature_names: Optional[List[str]] = None,
        **metadata
    ) -> "Dataset":
        """Create Dataset from numpy array."""
        df = pd.DataFrame(data, columns=feature_names)
        return cls.from_dataframe(name, df, **metadata)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for API serialization."""
        if isinstance(self.data, pd.DataFrame):
            data_dict = self.data.to_dict(orient="records")
        elif isinstance(self.data, np.ndarray):
            data_dict = self.data.tolist()
        else:
            data_dict = self.data
        
        return {
            "name": self.name,
            "data": data_dict,
            "metadata": self.metadata,
            "feature_names": self.feature_names,
            "target_column": self.target_column
        }


class DetectionResult(BaseModel):
    """Results from anomaly detection."""
    
    anomaly_scores: List[float] = Field(..., description="Anomaly scores for each sample")
    anomaly_labels: List[int] = Field(..., description="Binary anomaly labels (1=anomaly, 0=normal)")
    n_anomalies: int = Field(..., description="Total number of anomalies detected")
    n_samples: int = Field(..., description="Total number of samples")
    contamination_rate: float = Field(..., description="Actual contamination rate")
    threshold: float = Field(..., description="Decision threshold used")
    execution_time: float = Field(..., description="Execution time in seconds")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Additional result metadata")
    
    @property
    def anomaly_indices(self) -> List[int]:
        """Get indices of anomalous samples."""
        return [i for i, label in enumerate(self.anomaly_labels) if label == 1]
    
    @property
    def normal_indices(self) -> List[int]:
        """Get indices of normal samples."""
        return [i for i, label in enumerate(self.anomaly_labels) if label == 0]


class ModelMetrics(BaseModel):
    """Model performance metrics."""
    
    accuracy: Optional[float] = Field(None, ge=0.0, le=1.0, description="Classification accuracy")
    precision: Optional[float] = Field(None, ge=0.0, le=1.0, description="Precision score")
    recall: Optional[float] = Field(None, ge=0.0, le=1.0, description="Recall score")
    f1_score: Optional[float] = Field(None, ge=0.0, le=1.0, description="F1 score")
    auc_roc: Optional[float] = Field(None, ge=0.0, le=1.0, description="AUC-ROC score")
    auc_pr: Optional[float] = Field(None, ge=0.0, le=1.0, description="AUC-PR score")
    
    # Anomaly detection specific metrics
    contamination_rate: Optional[float] = Field(None, ge=0.0, le=1.0, description="Contamination rate")
    anomaly_threshold: Optional[float] = Field(None, description="Anomaly decision threshold")
    
    # Additional metrics
    custom_metrics: Dict[str, float] = Field(default_factory=dict, description="Custom metrics")
    
    class Config:
        json_schema_extra = {
            "example": {
                "accuracy": 0.95,
                "precision": 0.87,
                "recall": 0.82,
                "f1_score": 0.84,
                "auc_roc": 0.92,
                "contamination_rate": 0.1,
                "custom_metrics": {"silhouette_score": 0.65}
            }
        }


class ExperimentConfig(BaseModel):
    """Configuration for ML experiments."""
    
    name: str = Field(..., description="Experiment name")
    description: Optional[str] = Field(None, description="Experiment description")
    algorithm_configs: List[DetectorConfig] = Field(..., description="List of algorithm configurations to test")
    evaluation_metrics: List[str] = Field(default_factory=lambda: ["auc_roc", "precision", "recall"], description="Metrics to evaluate")
    cross_validation_folds: int = Field(5, ge=2, le=20, description="Number of CV folds")
    random_state: Optional[int] = Field(None, description="Random seed for reproducibility")
    parallel_jobs: int = Field(1, description="Number of parallel jobs")
    
    # Hyperparameter optimization
    optimization_enabled: bool = Field(False, description="Enable hyperparameter optimization")
    optimization_trials: int = Field(100, ge=10, le=1000, description="Number of optimization trials")
    optimization_timeout: Optional[int] = Field(None, description="Optimization timeout in seconds")
    
    class Config:
        json_schema_extra = {
            "example": {
                "name": "IsolationForest_Comparison",
                "description": "Compare IsolationForest with different parameters",
                "algorithm_configs": [
                    {
                        "algorithm_name": "IsolationForest",
                        "hyperparameters": {"n_estimators": 100},
                        "contamination_rate": 0.1
                    }
                ],
                "evaluation_metrics": ["auc_roc", "precision", "recall"],
                "cross_validation_folds": 5,
                "optimization_enabled": True,
                "optimization_trials": 50
            }
        }


class TrainingJob(BaseModel):
    """Training job information."""
    
    job_id: UUID = Field(..., description="Unique job identifier")
    name: str = Field(..., description="Job name")
    status: str = Field(..., description="Job status (pending, running, completed, failed)")
    detector_config: DetectorConfig = Field(..., description="Detector configuration")
    dataset_name: str = Field(..., description="Dataset name used for training")
    
    # Timestamps
    created_at: datetime = Field(..., description="Job creation timestamp")
    started_at: Optional[datetime] = Field(None, description="Job start timestamp")
    completed_at: Optional[datetime] = Field(None, description="Job completion timestamp")
    
    # Results
    metrics: Optional[ModelMetrics] = Field(None, description="Training metrics")
    model_path: Optional[str] = Field(None, description="Path to trained model")
    logs: List[str] = Field(default_factory=list, description="Training logs")
    error_message: Optional[str] = Field(None, description="Error message if failed")
    
    # Resource usage
    execution_time: Optional[float] = Field(None, description="Execution time in seconds")
    memory_usage: Optional[float] = Field(None, description="Peak memory usage in MB")
    cpu_usage: Optional[float] = Field(None, description="Average CPU usage percentage")
    
    @property
    def is_completed(self) -> bool:
        """Check if job is completed."""
        return self.status in ["completed", "failed"]
    
    @property
    def is_successful(self) -> bool:
        """Check if job completed successfully."""
        return self.status == "completed"


class PaginatedResponse(BaseModel):
    """Paginated response wrapper."""
    
    items: List[Dict[str, Any]] = Field(..., description="Response items")
    total: int = Field(..., description="Total number of items")
    page: int = Field(..., description="Current page number")
    page_size: int = Field(..., description="Page size")
    has_next: bool = Field(..., description="Whether there are more pages")
    has_previous: bool = Field(..., description="Whether there are previous pages")
    
    @property
    def total_pages(self) -> int:
        """Calculate total number of pages."""
        return (self.total + self.page_size - 1) // self.page_size


class HealthStatus(BaseModel):
    """API health status information."""
    
    status: str = Field(..., description="Overall health status")
    version: str = Field(..., description="API version")
    timestamp: datetime = Field(..., description="Health check timestamp")
    services: Dict[str, str] = Field(default_factory=dict, description="Service health status")
    metrics: Dict[str, Any] = Field(default_factory=dict, description="Performance metrics")
    
    @property
    def is_healthy(self) -> bool:
        """Check if all services are healthy."""
        return self.status == "healthy" and all(
            status == "healthy" for status in self.services.values()
        )