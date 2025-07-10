"""DTOs for detection result operations."""

from __future__ import annotations

from datetime import datetime
from typing import Any
from uuid import UUID

from pydantic import BaseModel, ConfigDict, Field

from pynomaly.application.dto.detection_dto import AnomalyDTO


class DetectionResultDTO(BaseModel):
    """DTO for detection result."""

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


# AnomalyDTO is imported from detection_dto above
