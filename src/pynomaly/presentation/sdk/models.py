"""
SDK Data Models

Pydantic models for all SDK data structures providing type safety,
validation, and serialization for API interactions.
"""

from datetime import datetime
from enum import Enum
from typing import Any

import numpy as np
from pydantic import BaseModel, ConfigDict, Enum, Field, field_validator, model_validator


class TaskStatus(str):
    """Task execution status."""

    PENDING = "pending"
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


class DetectorStatus(str, Enum):
    """Detector training and deployment status."""

    DRAFT = "draft"
    TRAINING = "training"
    TRAINED = "trained"
    DEPLOYED = "deployed"
    FAILED = "failed"
    ARCHIVED = "archived"


class AlgorithmType(str, Enum):
    """Supported algorithm types."""

    ISOLATION_FOREST = "IsolationForest"
    LOCAL_OUTLIER_FACTOR = "LocalOutlierFactor"
    ONE_CLASS_SVM = "OneClassSVM"
    ELLIPTIC_ENVELOPE = "EllipticEnvelope"
    AUTOENCODER = "Autoencoder"
    VARIATIONAL_AUTOENCODER = "VariationalAutoencoder"
    LSTM_AUTOENCODER = "LSTMAutoencoder"
    GAN_ANOMALY = "GANAnomaly"


class DataFormat(str, Enum):
    """Supported data formats."""

    CSV = "csv"
    JSON = "json"
    PARQUET = "parquet"
    NUMPY = "numpy"
    PANDAS = "pandas"


class BaseSDKModel(BaseModel):
    """Base model for all SDK models."""
    
    model_config = ConfigDict(
        arbitrary_types_allowed=True,
        use_enum_values=True,
        allow_population_by_field_name=True
    )


class AnomalyScore(BaseSDKModel):
    """Anomaly score representation."""

    value: float = Field(
        ..., ge=0.0, le=1.0, description="Anomaly score between 0 and 1"
    )
    confidence: float | None = Field(
        None, ge=0.0, le=1.0, description="Confidence in the score"
    )
    percentile: float | None = Field(
        None, ge=0.0, le=100.0, description="Percentile rank"
    )

    @field_validator("value")
    @classmethod
    def validate_score(cls, v):
        if not 0.0 <= v <= 1.0:
            raise ValueError("Anomaly score must be between 0 and 1")
        return v


class PerformanceMetrics(BaseSDKModel):
    """Model performance metrics."""

    accuracy: float | None = Field(None, ge=0.0, le=1.0)
    precision: float | None = Field(None, ge=0.0, le=1.0)
    recall: float | None = Field(None, ge=0.0, le=1.0)
    f1_score: float | None = Field(None, ge=0.0, le=1.0)
    auc_roc: float | None = Field(None, ge=0.0, le=1.0)
    auc_pr: float | None = Field(None, ge=0.0, le=1.0)

    # Anomaly detection specific metrics
    contamination_rate: float | None = Field(None, ge=0.0, le=1.0)
    outlier_fraction: float | None = Field(None, ge=0.0, le=1.0)

    # Training metrics
    training_time: float | None = Field(
        None, ge=0.0, description="Training time in seconds"
    )
    inference_time: float | None = Field(
        None, ge=0.0, description="Average inference time per sample"
    )

    # Additional metrics
    custom_metrics: dict[str, float] | None = Field(default_factory=dict)


class Dataset(BaseSDKModel):
    """Dataset representation."""

    id: str = Field(..., description="Unique dataset identifier")
    name: str = Field(..., min_length=1, description="Dataset name")
    description: str | None = Field(None, description="Dataset description")

    # Data characteristics
    num_samples: int = Field(..., ge=0, description="Number of samples")
    num_features: int = Field(..., ge=1, description="Number of features")
    feature_names: list[str] = Field(..., description="Feature column names")
    data_types: dict[str, str] | None = Field(None, description="Feature data types")

    # Metadata
    format: DataFormat = Field(DataFormat.CSV, description="Data format")
    size_bytes: int | None = Field(None, ge=0, description="Dataset size in bytes")
    checksum: str | None = Field(None, description="Data checksum for integrity")

    # Timestamps
    created_at: datetime = Field(default_factory=datetime.now)
    updated_at: datetime | None = None

    # Data quality metrics
    missing_values: dict[str, int] | None = Field(
        None, description="Missing values per feature"
    )
    outlier_count: int | None = Field(
        None, ge=0, description="Number of detected outliers"
    )

    # Storage information
    storage_path: str | None = Field(None, description="Storage location")
    is_public: bool = Field(False, description="Whether dataset is publicly accessible")

    @field_validator("feature_names")
    @classmethod
    def validate_feature_names(cls, v, info):
        if (
            hasattr(info, "data")
            and "num_features" in info.data
            and len(v) != info.data["num_features"]
        ):
            raise ValueError("Number of feature names must match num_features")
        return v


class Detector(BaseSDKModel):
    """Anomaly detector representation."""

    id: str = Field(..., description="Unique detector identifier")
    name: str = Field(..., min_length=1, description="Detector name")
    description: str | None = Field(None, description="Detector description")

    # Algorithm configuration
    algorithm: AlgorithmType = Field(..., description="Algorithm type")
    parameters: dict[str, Any] = Field(
        default_factory=dict, description="Algorithm parameters"
    )

    # Training information
    dataset_id: str | None = Field(None, description="Training dataset ID")
    status: DetectorStatus = Field(DetectorStatus.DRAFT, description="Detector status")

    # Model metadata
    version: str = Field("1.0", description="Model version")
    framework: str | None = Field(None, description="ML framework used")
    model_size_bytes: int | None = Field(None, ge=0, description="Model size in bytes")

    # Performance
    performance_metrics: PerformanceMetrics | None = None
    training_duration: float | None = Field(
        None, ge=0.0, description="Training time in seconds"
    )

    # Timestamps
    created_at: datetime = Field(default_factory=datetime.now)
    trained_at: datetime | None = None
    deployed_at: datetime | None = None

    # Configuration
    contamination_rate: float = Field(
        0.1, ge=0.0, le=1.0, description="Expected contamination rate"
    )
    threshold: float | None = Field(None, description="Anomaly threshold")

    # Deployment information
    endpoint_url: str | None = Field(None, description="Deployed model endpoint")
    is_active: bool = Field(False, description="Whether detector is active")


class DetectionResult(BaseSDKModel):
    """Anomaly detection result."""

    id: str = Field(..., description="Unique result identifier")
    detector_id: str = Field(..., description="Detector used for detection")

    # Predictions
    predictions: list[int] = Field(
        ..., description="Binary predictions (0=normal, 1=anomaly)"
    )
    anomaly_scores: list[AnomalyScore] = Field(
        ..., description="Anomaly scores for each sample"
    )

    # Summary statistics
    num_samples: int = Field(..., ge=0, description="Number of samples processed")
    num_anomalies: int = Field(..., ge=0, description="Number of detected anomalies")
    anomaly_rate: float = Field(
        ..., ge=0.0, le=1.0, description="Fraction of anomalies detected"
    )

    # Timing information
    processing_time: float = Field(
        ..., ge=0.0, description="Processing time in seconds"
    )
    created_at: datetime = Field(default_factory=datetime.now)

    # Optional explanations
    feature_importances: dict[str, float] | None = Field(
        None, description="Feature importance scores"
    )
    explanations: list[dict[str, Any]] | None = Field(
        None, description="Per-sample explanations"
    )

    # Confidence and uncertainty
    confidence_intervals: list[list[float]] | None = Field(
        None, description="Confidence intervals"
    )
    uncertainty_scores: list[float] | None = Field(
        None, description="Uncertainty estimates"
    )

    @field_validator("predictions")
    @classmethod
    def validate_predictions(cls, v):
        if any(pred not in [0, 1] for pred in v):
            raise ValueError("Predictions must be 0 or 1")
        return v

    @model_validator(mode="before")
    @classmethod
    def validate_consistency(cls, values):
        predictions = values.get("predictions", [])
        anomaly_scores = values.get("anomaly_scores", [])
        num_samples = values.get("num_samples", 0)

        if len(predictions) != num_samples:
            raise ValueError("Number of predictions must match num_samples")

        if len(anomaly_scores) != num_samples:
            raise ValueError("Number of anomaly scores must match num_samples")

        num_anomalies = values.get("num_anomalies", 0)
        if num_anomalies != sum(predictions):
            raise ValueError("num_anomalies must equal sum of predictions")

        return values


class TrainingJob(BaseSDKModel):
    """Training job representation."""

    id: str = Field(..., description="Unique job identifier")
    detector_id: str = Field(..., description="Detector being trained")
    dataset_id: str = Field(..., description="Training dataset")

    # Job status
    status: TaskStatus = Field(TaskStatus.PENDING, description="Job status")
    progress: float = Field(
        0.0, ge=0.0, le=100.0, description="Training progress percentage"
    )

    # Configuration
    algorithm: AlgorithmType = Field(..., description="Algorithm being used")
    parameters: dict[str, Any] = Field(
        default_factory=dict, description="Training parameters"
    )

    # Timing
    created_at: datetime = Field(default_factory=datetime.now)
    started_at: datetime | None = None
    completed_at: datetime | None = None
    estimated_completion: datetime | None = None

    # Results
    performance_metrics: PerformanceMetrics | None = None
    error_message: str | None = Field(None, description="Error message if failed")
    logs: list[str] | None = Field(default_factory=list, description="Training logs")

    # Resource usage
    cpu_usage: float | None = Field(None, ge=0.0, description="CPU usage percentage")
    memory_usage: float | None = Field(None, ge=0.0, description="Memory usage in MB")
    gpu_usage: float | None = Field(None, ge=0.0, description="GPU usage percentage")


class ExperimentResult(BaseSDKModel):
    """Experiment result representation."""

    id: str = Field(..., description="Unique experiment identifier")
    name: str = Field(..., description="Experiment name")
    description: str | None = Field(None, description="Experiment description")

    # Configuration
    dataset_id: str = Field(..., description="Dataset used")
    algorithms: list[AlgorithmType] = Field(..., description="Algorithms compared")

    # Results
    results: dict[str, PerformanceMetrics] = Field(
        ..., description="Results per algorithm"
    )
    best_algorithm: AlgorithmType | None = Field(
        None, description="Best performing algorithm"
    )

    # Metadata
    created_at: datetime = Field(default_factory=datetime.now)
    completed_at: datetime | None = None
    duration: float | None = Field(
        None, ge=0.0, description="Experiment duration in seconds"
    )

    # Statistical analysis
    statistical_significance: dict[str, float] | None = Field(
        None, description="P-values for comparisons"
    )
    confidence_intervals: dict[str, list[float]] | None = Field(
        None, description="CI for metrics"
    )


class BatchDetectionRequest(BaseSDKModel):
    """Batch detection request."""

    detector_id: str = Field(..., description="Detector to use")
    data: list[list[float]] | str = Field(
        ..., description="Data to analyze or dataset ID"
    )

    # Options
    return_scores: bool = Field(True, description="Whether to return anomaly scores")
    return_explanations: bool = Field(
        False, description="Whether to return explanations"
    )
    batch_size: int | None = Field(None, gt=0, description="Batch processing size")

    # Output format
    output_format: DataFormat = Field(DataFormat.JSON, description="Output format")
    include_metadata: bool = Field(True, description="Include result metadata")


class StreamingDetectionConfig(BaseSDKModel):
    """Streaming detection configuration."""

    detector_id: str = Field(..., description="Detector to use")

    # Stream settings
    buffer_size: int = Field(1000, gt=0, description="Buffer size for batching")
    processing_interval: float = Field(
        1.0, gt=0.0, description="Processing interval in seconds"
    )

    # Alert settings
    alert_threshold: float = Field(
        0.8, ge=0.0, le=1.0, description="Threshold for alerts"
    )
    alert_window: int = Field(10, gt=0, description="Window size for alert aggregation")

    # Output settings
    output_format: DataFormat = Field(DataFormat.JSON, description="Output format")
    include_scores: bool = Field(True, description="Include anomaly scores")
    include_explanations: bool = Field(False, description="Include explanations")


# Utility functions for model conversion


def numpy_to_list(arr: np.ndarray) -> list:
    """Convert numpy array to list for JSON serialization."""
    return arr.tolist()


def list_to_numpy(data: list, dtype: str | None = None) -> np.ndarray:
    """Convert list to numpy array."""
    if dtype:
        return np.array(data, dtype=dtype)
    return np.array(data)


def validate_data_shape(
    data: list | np.ndarray, expected_features: int | None = None
) -> None:
    """Validate data shape for consistency."""
    if isinstance(data, list):
        if not data:
            raise ValueError("Data cannot be empty")

        if isinstance(data[0], list):
            # 2D data
            feature_counts = [len(row) for row in data]
            if len(set(feature_counts)) > 1:
                raise ValueError("All samples must have the same number of features")

            if expected_features and feature_counts[0] != expected_features:
                raise ValueError(
                    f"Expected {expected_features} features, got {feature_counts[0]}"
                )

        elif expected_features and expected_features != 1:
            raise ValueError(f"Expected {expected_features} features, got 1D data")

    elif isinstance(data, np.ndarray):
        if data.ndim == 1 and expected_features and expected_features != 1:
            raise ValueError(f"Expected {expected_features} features, got 1D data")

        elif (
            data.ndim == 2 and expected_features and data.shape[1] != expected_features
        ):
            raise ValueError(
                f"Expected {expected_features} features, got {data.shape[1]}"
            )

        elif data.ndim > 2:
            raise ValueError("Data cannot have more than 2 dimensions")
