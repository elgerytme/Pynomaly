"""DTOs for experiment tracking."""

from __future__ import annotations

from datetime import datetime
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field, ConfigDict


class RunDTO(BaseModel):
    """DTO for experiment run."""
    
    id: str
    detector_name: str
    dataset_name: str
    parameters: Dict[str, Any]
    metrics: Dict[str, float]
    artifacts: Dict[str, str] = Field(default_factory=dict)
    timestamp: datetime


class ExperimentDTO(BaseModel):
    """DTO for experiment."""
    
    model_config = ConfigDict(from_attributes=True)
    
    id: str
    name: str
    description: Optional[str] = None
    runs: List[RunDTO] = Field(default_factory=list)
    created_at: datetime
    updated_at: datetime
    metadata: Dict[str, Any] = Field(default_factory=dict)


class CreateExperimentDTO(BaseModel):
    """DTO for creating experiments."""
    
    name: str = Field(min_length=1, max_length=100)
    description: Optional[str] = Field(None, max_length=500)
    metadata: Dict[str, Any] = Field(default_factory=dict)


class LeaderboardEntryDTO(BaseModel):
    """DTO for leaderboard entries."""
    
    model_config = ConfigDict(from_attributes=True)
    
    rank: int
    experiment_id: str
    run_id: str
    detector_name: str
    dataset_name: str
    score: float
    metric_name: str
    parameters: Dict[str, Any] = Field(default_factory=dict)
    timestamp: datetime