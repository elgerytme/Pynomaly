"""DTOs for dataset-related operations."""

from __future__ import annotations

from datetime import datetime
from typing import Any, Dict, List, Optional
from uuid import UUID

from pydantic import BaseModel, Field, ConfigDict


class DatasetDTO(BaseModel):
    """DTO for dataset information."""
    
    model_config = ConfigDict(
        from_attributes=True,
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
                "categorical_features": 5
            }
        }
    )
    
    id: UUID
    name: str
    shape: tuple[int, int]
    n_samples: int
    n_features: int
    feature_names: List[str]
    has_target: bool
    target_column: Optional[str] = None
    created_at: datetime
    metadata: Dict[str, Any] = Field(default_factory=dict)
    description: Optional[str] = None
    memory_usage_mb: float
    
    # Data quality metrics
    numeric_features: int
    categorical_features: int


class CreateDatasetDTO(BaseModel):
    """DTO for creating/uploading a dataset."""
    
    name: str = Field(min_length=1, max_length=100)
    description: Optional[str] = Field(None, max_length=500)
    target_column: Optional[str] = None
    metadata: Dict[str, Any] = Field(default_factory=dict)
    
    # Data source options
    file_path: Optional[str] = None
    file_format: Optional[str] = Field(None, pattern="^(csv|parquet|json|excel)$")
    
    # Data loading options
    delimiter: str = Field(default=",")
    encoding: str = Field(default="utf-8")
    parse_dates: Optional[List[str]] = None
    dtype: Optional[Dict[str, str]] = None


class DataQualityReportDTO(BaseModel):
    """DTO for data quality report."""
    
    quality_score: float = Field(ge=0, le=1)
    n_missing_values: int
    n_duplicates: int
    n_outliers: int
    missing_columns: List[str] = Field(default_factory=list)
    constant_columns: List[str] = Field(default_factory=list)
    high_cardinality_columns: List[str] = Field(default_factory=list)
    highly_correlated_features: List[tuple[str, str, float]] = Field(default_factory=list)
    recommendations: List[str] = Field(default_factory=list)