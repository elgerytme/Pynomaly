"""DTOs for dataset-related operations."""

from __future__ import annotations

from datetime import datetime
from typing import Any
from uuid import UUID

from pydantic import BaseModel, ConfigDict, Field


class DatasetDTO(BaseModel):
    """DTO for dataset information."""

    model_config = ConfigDict(
        from_attributes=True,
        extra="forbid",
        json_schema_extra={
            "example": {
                "id": "123e4567-e89b-12d3-a456-426614174000",
                "name": "credit_card_transactions",
                "shape": [10000, 30],
                "n_samples": 10000,
                "n_features": 30,
                "feature_names": ["amount", "merchant_id", "time", "..."],
                "has_target": True,
                "target_column": "is_fraud",
                "created_at": "2024-01-01T00:00:00",
                "memory_usage_mb": 2.4,
                "numeric_features": 25,
                "categorical_features": 5,
            }
        },
    )

    id: UUID
    name: str
    shape: tuple[int, int]
    n_samples: int
    n_features: int
    feature_names: list[str]
    has_target: bool
    target_column: str | None = None
    created_at: datetime
    metadata: dict[str, Any] = Field(default_factory=dict)
    description: str | None = None
    memory_usage_mb: float

    # Data quality metrics
    numeric_features: int
    categorical_features: int


class CreateDatasetDTO(BaseModel):
    """DTO for creating/uploading a dataset."""

    model_config = ConfigDict(extra="forbid")

    name: str = Field(min_length=1, max_length=100)
    description: str | None = Field(None, max_length=500)
    target_column: str | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)

    # Data source options
    file_path: str | None = None
    file_format: str | None = Field(None, pattern="^(csv|parquet|json|excel)$")

    # Data loading options
    delimiter: str = Field(default=",")
    encoding: str = Field(default="utf-8")
    parse_dates: list[str] | None = None
    dtype: dict[str, str] | None = None


class DataQualityReportDTO(BaseModel):
    """DTO for data quality report."""

    model_config = ConfigDict(extra="forbid")

    quality_score: float = Field(ge=0, le=1)
    n_missing_values: int
    n_duplicates: int
    n_outliers: int
    missing_columns: list[str] = Field(default_factory=list)
    constant_columns: list[str] = Field(default_factory=list)
    high_cardinality_columns: list[str] = Field(default_factory=list)
    highly_correlated_features: list[tuple[str, str, float]] = Field(
        default_factory=list
    )
    recommendations: list[str] = Field(default_factory=list)


class DatasetResponseDTO(BaseModel):
    """DTO for dataset API response."""

    model_config = ConfigDict(
        from_attributes=True,
        extra="forbid",
        json_schema_extra={
            "example": {
                "id": "123e4567-e89b-12d3-a456-426614174000",
                "name": "credit_card_transactions",
                "description": "Credit card transaction dataset for fraud detection",
                "shape": [10000, 30],
                "n_samples": 10000,
                "n_features": 30,
                "feature_names": ["amount", "merchant_id", "time"],
                "has_target": True,
                "target_column": "is_fraud",
                "created_at": "2024-01-01T00:00:00",
                "memory_usage_mb": 2.4,
                "numeric_features": 25,
                "categorical_features": 5,
                "quality_score": 0.85,
                "status": "ready",
            }
        },
    )

    id: UUID
    name: str
    description: str | None = None
    shape: tuple[int, int]
    n_samples: int
    n_features: int
    feature_names: list[str]
    has_target: bool
    target_column: str | None = None
    created_at: datetime
    memory_usage_mb: float
    numeric_features: int
    categorical_features: int
    quality_score: float | None = None
    status: str = "ready"
    metadata: dict[str, Any] = Field(default_factory=dict)
