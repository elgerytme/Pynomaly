"""DTOs for experiment tracking."""

from __future__ import annotations

from datetime import datetime
from typing import Any

from pydantic import BaseModel, ConfigDict, Field


class RunDTO(BaseModel):
    """DTO for experiment run."""
    
    model_config = ConfigDict(extra="forbid")
    id: str
    detector_name: str
    dataset_name: str
    parameters: dict[str, Any]
    metrics: dict[str, float]
    artifacts: dict[str, str] = Field(default_factory=dict)
    timestamp: datetime


class ExperimentDTO(BaseModel):
    """DTO for experiment."""
    
    model_config = ConfigDict(from_attributes=True, extra="forbid")

    id: str
    name: str
    description: str | None = None
    runs: list[RunDTO] = Field(default_factory=list)
    created_at: datetime
    updated_at: datetime
    metadata: dict[str, Any] = Field(default_factory=dict)


class CreateExperimentDTO(BaseModel):
    """DTO for creating experiments."""
    
    model_config = ConfigDict(extra="forbid")
    name: str = Field(min_length=1, max_length=100)
    description: str | None = Field(None, max_length=500)
    metadata: dict[str, Any] = Field(default_factory=dict)


class LeaderboardEntryDTO(BaseModel):
    """DTO for leaderboard entries."""
    
    model_config = ConfigDict(from_attributes=True, extra="forbid")

    rank: int
    experiment_id: str
    run_id: str
    detector_name: str
    dataset_name: str
    score: float
    metric_name: str
    parameters: dict[str, Any] = Field(default_factory=dict)
    timestamp: datetime


class ExperimentResponseDTO(BaseModel):
    """DTO for experiment API responses."""
    
    model_config = ConfigDict(from_attributes=True, extra="forbid")

    id: str
    name: str
    description: str | None = None
    status: str = "active"
    total_runs: int = 0
    best_score: float | None = None
    best_metric: str | None = None
    created_at: datetime
    updated_at: datetime
    metadata: dict[str, Any] = Field(default_factory=dict)
