"""DTOs for detection-related operations."""

from __future__ import annotations

from datetime import datetime
from typing import Any
from uuid import UUID

from pydantic import BaseModel, ConfigDict, Field


class DetectionRequestDTO(BaseModel):
    """DTO for anomaly detection requests."""
    
    model_config = ConfigDict(
        extra="forbid",
        json_schema_extra={
            "example": {
                "detector_id": "123e4567-e89b-12d3-a456-426614174000",
                "dataset_id": "456e7890-e89b-12d3-a456-426614174000",
                "threshold": 0.5,
                "return_scores": True,
                "return_explanations": False,
                "validate_features": True,
                "save_results": True,
            }
        }
    )
    detector_id: UUID
    dataset_id: UUID | None = None
    data: list[dict[str, Any]] | None = None  # Inline data alternative to dataset_id
    threshold: float | None = Field(None, ge=0, le=1)
    return_scores: bool = True
    return_explanations: bool = False
    validate_features: bool = True
    save_results: bool = True

    def model_post_init(self, __context: Any) -> None:
        """Validate that either dataset_id or data is provided."""
        if self.dataset_id is None and self.data is None:
            raise ValueError("Either dataset_id or data must be provided")
        if self.dataset_id is not None and self.data is not None:
            raise ValueError("Provide either dataset_id or data, not both")


class TrainingRequestDTO(BaseModel):
    """DTO for detector training requests."""
    
    model_config = ConfigDict(
        extra="forbid",
        json_schema_extra={
            "example": {
                "detector_id": "123e4567-e89b-12d3-a456-426614174000",
                "dataset_id": "456e7890-e89b-12d3-a456-426614174000",
                "validation_split": 0.2,
                "cross_validation": False,
                "save_model": True,
                "parameters": {"n_estimators": 100, "max_samples": "auto"},
            }
        }
    )

    detector_id: UUID
    dataset_id: UUID
    validation_split: float | None = Field(None, ge=0, le=0.5)
    cross_validation: bool = False
    save_model: bool = True
    parameters: dict[str, Any] = Field(default_factory=dict)


class AnomalyDTO(BaseModel):
    """DTO for individual anomaly information."""
    
    model_config = ConfigDict(from_attributes=True, extra="forbid")

    id: UUID
    score: float = Field(ge=0, le=1)
    detector_name: str
    timestamp: datetime
    data_point: dict[str, Any]
    metadata: dict[str, Any] = Field(default_factory=dict)
    explanation: str | None = None
    severity: str = "medium"  # low, medium, high, critical
    confidence_lower: float | None = None
    confidence_upper: float | None = None


class DetectionResultDTO(BaseModel):
    """DTO for detection results."""
    
    model_config = ConfigDict(
        from_attributes=True,
        extra="forbid",
        json_schema_extra={
            "example": {
                "id": "789e1234-e89b-12d3-a456-426614174000",
                "detector_id": "123e4567-e89b-12d3-a456-426614174000",
                "dataset_id": "456e7890-e89b-12d3-a456-426614174000",
                "timestamp": "2024-01-01T12:00:00",
                "n_samples": 1000,
                "n_anomalies": 45,
                "anomaly_rate": 0.045,
                "threshold": 0.6,
                "execution_time_ms": 1500.0,
                "anomalies": [],
                "score_statistics": {
                    "min": 0.0,
                    "max": 1.0,
                    "mean": 0.3,
                    "median": 0.28,
                    "std": 0.22,
                },
            }
        },
    )

    id: UUID
    detector_id: UUID
    dataset_id: UUID
    timestamp: datetime
    n_samples: int
    n_anomalies: int
    anomaly_rate: float = Field(ge=0, le=1)
    threshold: float
    execution_time_ms: float | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)

    # Results
    anomalies: list[AnomalyDTO] = Field(default_factory=list)
    predictions: list[int] | None = None  # 0=normal, 1=anomaly
    scores: list[float] | None = None
    score_statistics: dict[str, float] = Field(default_factory=dict)

    # Quality metrics
    has_confidence_intervals: bool = False
    quality_warnings: list[str] = Field(default_factory=list)


class TrainingResultDTO(BaseModel):
    """DTO for training results."""
    
    model_config = ConfigDict(from_attributes=True, extra="forbid")

    detector_id: UUID
    dataset_id: UUID
    timestamp: datetime
    training_time_ms: float
    model_path: str | None = None
    training_warnings: list[str] = Field(default_factory=list)

    # Metrics
    training_metrics: dict[str, Any] = Field(default_factory=dict)
    validation_metrics: dict[str, Any] | None = None
    dataset_summary: dict[str, Any] = Field(default_factory=dict)
    parameters_used: dict[str, Any] = Field(default_factory=dict)


class ExplanationRequestDTO(BaseModel):
    """DTO for anomaly explanation requests."""
    
    model_config = ConfigDict(extra="forbid")
    
    detector_id: UUID
    instance: dict[str, Any]  # Single data point to explain
    method: str = Field(default="shap", pattern="^(shap|lime)$")
    feature_names: list[str] | None = None
    n_features: int = Field(default=10, ge=1, le=50)


class ExplanationResultDTO(BaseModel):
    """DTO for anomaly explanation results."""
    
    model_config = ConfigDict(from_attributes=True, extra="forbid")

    method_used: str
    prediction: float
    confidence: float | None = None
    feature_importance: dict[str, float] = Field(default_factory=dict)
    explanation_text: str | None = None
    visualization_data: dict[str, Any] | None = None


class DetectionSummaryDTO(BaseModel):
    """DTO for detection summary statistics."""
    
    model_config = ConfigDict(extra="forbid")
    total_detections: int
    recent_detections: int  # Last 24 hours
    average_anomaly_rate: float
    most_active_detector: str | None = None
    top_algorithms: list[dict[str, Any]] = Field(default_factory=list)
    performance_metrics: dict[str, float] = Field(default_factory=dict)
