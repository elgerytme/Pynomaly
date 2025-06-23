"""DTOs for detection result operations."""

from __future__ import annotations

from datetime import datetime
from typing import Any, Dict, List, Optional
from uuid import UUID

from pydantic import BaseModel, Field, ConfigDict


class DetectionResultDTO(BaseModel):
    """DTO for detection result."""
    
    model_config = ConfigDict(from_attributes=True)
    
    id: UUID
    detector_id: UUID
    dataset_id: UUID
    run_id: Optional[UUID] = None
    created_at: datetime
    duration_seconds: float
    
    # Results
    anomalies: List['AnomalyDTO']
    total_samples: int
    anomaly_count: int
    contamination_rate: float = Field(ge=0, le=1)
    
    # Performance metrics
    mean_score: float
    max_score: float
    min_score: float
    threshold: float
    
    # Metadata
    metadata: Dict[str, Any] = Field(default_factory=dict)
    parameters: Dict[str, Any] = Field(default_factory=dict)


class AnomalyDTO(BaseModel):
    """DTO for individual anomaly."""
    
    model_config = ConfigDict(from_attributes=True)
    
    id: UUID
    score: float = Field(ge=0, le=1)
    data_point: Dict[str, Any]
    detector_name: str
    timestamp: datetime
    severity: str = Field(pattern="^(low|medium|high|critical)$")
    explanation: Optional[str] = None
    metadata: Dict[str, Any] = Field(default_factory=dict)
    
    # Confidence information
    confidence_lower: Optional[float] = Field(None, ge=0, le=1)
    confidence_upper: Optional[float] = Field(None, ge=0, le=1)