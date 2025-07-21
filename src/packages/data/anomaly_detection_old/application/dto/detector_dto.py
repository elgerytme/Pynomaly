"""DTOs for detector-related operations."""

from __future__ import annotations

from datetime import datetime
from typing import Any
from uuid import UUID

from pydantic import BaseModel, ConfigDict, Field, field_validator


class DetectorDTO(BaseModel):
    """DTO for detector information."""

    model_config = ConfigDict(from_attributes=True, extra="forbid")

    id: UUID
    name: str
    algorithm_name: str
    contamination_rate: float = Field(ge=0, le=1)
    is_fitted: bool
    created_at: datetime
    trained_at: datetime | None = None
    parameters: dict[str, Any] = Field(default_factory=dict)
    metadata: dict[str, Any] = Field(default_factory=dict)

    # Computed fields
    requires_fitting: bool = True
    supports_streaming: bool = False
    supports_multivariate: bool = True
    time_complexity: str | None = None
    space_complexity: str | None = None


class CreateDetectorDTO(BaseModel):
    """DTO for creating a new detector."""

    model_config = ConfigDict(extra="forbid")

    name: str = Field(min_length=1, max_length=100)
    algorithm_name: str = Field(min_length=1)
    contamination_rate: float = Field(default=0.1, ge=0, le=1)
    parameters: dict[str, Any] = Field(default_factory=dict)
    metadata: dict[str, Any] = Field(default_factory=dict)

    @field_validator("name")
    @classmethod
    def validate_name(cls, v: str) -> str:
        """Validate name is not empty or whitespace only."""
        if not v or not v.strip():
            raise ValueError("Name cannot be empty or whitespace only")
        return v.strip()


class UpdateDetectorDTO(BaseModel):
    """DTO for updating an existing detector."""

    model_config = ConfigDict(extra="forbid")

    name: str | None = Field(None, min_length=1, max_length=100)
    contamination_rate: float | None = Field(None, ge=0, le=1)
    parameters: dict[str, Any] | None = None
    metadata: dict[str, Any] | None = None


class DetectorResponseDTO(BaseModel):
    """DTO for detector API responses."""

    model_config = ConfigDict(from_attributes=True, extra="forbid")

    id: UUID
    name: str
    algorithm_name: str
    contamination_rate: float
    is_fitted: bool
    created_at: datetime
    trained_at: datetime | None = None
    parameters: dict[str, Any] = Field(default_factory=dict)
    metadata: dict[str, Any] = Field(default_factory=dict)
    status: str = "active"
    version: str = "1.0.0"


class DetectionRequestDTO(BaseModel):
    """DTO for detection requests."""

    model_config = ConfigDict(from_attributes=True, extra="forbid")

    detector_id: UUID
    data: list[list[float]]  # 2D array of features
    return_scores: bool = True
    return_feature_importance: bool = False
    threshold: float | None = None