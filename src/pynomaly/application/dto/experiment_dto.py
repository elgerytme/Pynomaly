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
    timestamp: datetime        }