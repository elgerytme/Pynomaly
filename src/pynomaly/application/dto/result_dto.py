"""DTOs for detection result operations."""

from __future__ import annotations

from datetime import datetime
from typing import Any
from uuid import UUID

from pydantic import BaseModel, ConfigDict, Field

from pynomaly.application.dto.detection_dto import AnomalyDTO


class DetectionResultDTO(BaseModel):
    """DTO for detection result."""
    
    model_config = ConfigDict(extra="forbid")
     model_config = ConfigDict(from_attributes=True, extra="forbid")

    id: UUID
    detector_id: UUID
    dataset_id: UUID
    run_id: UUID | None = None
    created_at: datetime
    duration_seconds: float

    # Results
    anomalies: list[AnomalyDTO]
    total_samples: int
    anomaly_count: int
    contamination_rate: float = Field(ge=0, le=1)

    # Performance metrics
    mean_score: float
    max_score: float
    min_score: float
    threshold: float

    # Metadata
    metadata: dict[str, Any] = Field(default_factory=dict)
    parameters: dict[str, Any] = Field(default_factory=dict)


class AnomalyDTO(BaseModel):
    """DTO for individual anomaly."""
    
    model_config = ConfigDict(extra="forbid")
     model_config = ConfigDict(from_attributes=True, extra="forbid")

    id: UUID
    score: float = Field(ge=0, le=1)
    data_point: dict[str, Any]
    detector_name: str
    timestamp: datetime
    severity: str = Field(pattern="^(low|medium|high|critical)$")
    explanation: str | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)

    # Confidence information
    confidence_lower: float | None = Field(None, ge=0, le=1)
    confidence_upper: float | None = Field(None, ge=0, le=1)
