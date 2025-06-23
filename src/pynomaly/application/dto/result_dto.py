"""DTOs for detection result operations."""

from __future__ import annotations

from datetime import datetime
from typing import Any, Dict, List, Optional
from uuid import UUID

from pydantic import BaseModel, Field, ConfigDict


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