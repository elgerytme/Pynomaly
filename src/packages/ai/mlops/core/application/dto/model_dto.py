"""Data Transfer Objects for ML models."""

from dataclasses import dataclass
from typing import Dict, List, Optional, Any
from datetime import datetime


@dataclass
class ModelDTO:
    """Data Transfer Object for ML models."""
    id: Optional[str] = None
    name: str = ""
    description: str = ""
    model_type: str = ""
    algorithm: str = ""
    version: str = "1.0.0"
    status: str = "draft"
    metrics: Optional[Dict[str, Any]] = None
    metadata: Optional[Dict[str, Any]] = None
    tags: List[str] = None
    created_by: str = ""
    model_path: Optional[str] = None
    experiment_id: Optional[str] = None
    parent_model_id: Optional[str] = None
    created_at: Optional[datetime] = None
    updated_at: Optional[datetime] = None
    
    def __post_init__(self):
        if self.tags is None:
            self.tags = []
        if self.metrics is None:
            self.metrics = {}
        if self.metadata is None:
            self.metadata = {}


@dataclass
class ModelCreateDTO:
    """DTO for creating ML models."""
    name: str
    description: str
    model_type: str
    algorithm: str
    created_by: str
    experiment_id: Optional[str] = None
    parent_model_id: Optional[str] = None
    tags: List[str] = None
    
    def __post_init__(self):
        if self.tags is None:
            self.tags = []


@dataclass
class ModelUpdateDTO:
    """DTO for updating ML models."""
    description: Optional[str] = None
    status: Optional[str] = None
    metrics: Optional[Dict[str, Any]] = None
    metadata: Optional[Dict[str, Any]] = None
    tags: Optional[List[str]] = None
    model_path: Optional[str] = None