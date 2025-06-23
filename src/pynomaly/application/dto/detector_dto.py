"""DTOs for detector-related operations."""

from __future__ import annotations

from datetime import datetime
from typing import Any, Dict, Optional
from uuid import UUID

from pydantic import BaseModel, Field, ConfigDict


class DetectorDTO(BaseModel):
    """DTO for detector information."""
    
    model_config = ConfigDict(from_attributes=True)
    
    id: UUID
    name: str
    algorithm_name: str
    contamination_rate: float = Field(ge=0, le=1)
    is_fitted: bool
    created_at: datetime
    trained_at: Optional[datetime] = None
    parameters: Dict[str, Any] = Field(default_factory=dict)
    metadata: Dict[str, Any] = Field(default_factory=dict)
    
    # Computed fields
    requires_fitting: bool = True
    supports_streaming: bool = False
    supports_multivariate: bool = True
    time_complexity: Optional[str] = None
    space_complexity: Optional[str] = None


class CreateDetectorDTO(BaseModel):
    """DTO for creating a new detector."""
    
    name: str = Field(min_length=1, max_length=100)
    algorithm_name: str = Field(min_length=1)
    contamination_rate: float = Field(default=0.1, ge=0, le=1)
    parameters: Dict[str, Any] = Field(default_factory=dict)
    metadata: Dict[str, Any] = Field(default_factory=dict)


class UpdateDetectorDTO(BaseModel):
    """DTO for updating an existing detector."""
    
    name: Optional[str] = Field(None, min_length=1, max_length=100)
    contamination_rate: Optional[float] = Field(None, ge=0, le=1)
    parameters: Optional[Dict[str, Any]] = None
    metadata: Optional[Dict[str, Any]] = None


class DetectorResponseDTO(BaseModel):
    """DTO for detector API responses."""
    
    model_config = ConfigDict(from_attributes=True)
    
    id: UUID
    name: str
    algorithm_name: str
    contamination_rate: float
    is_fitted: bool
    created_at: datetime
    trained_at: Optional[datetime] = None
    parameters: Dict[str, Any] = Field(default_factory=dict)
    metadata: Dict[str, Any] = Field(default_factory=dict)
    status: str = "active"
    version: str = "1.0.0"


class DetectionRequestDTO(BaseModel):
    """DTO for detection requests."""
    
    model_config = ConfigDict(from_attributes=True)
    
    detector_id: UUID
    data: list[list[float]]  # 2D array of features
    return_scores: bool = True
    return_feature_importance: bool = False
    threshold: Optional[float] = None