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
    
    class Config:
        json_schema_extra = {
            "example": {
                "id": "123e4567-e89b-12d3-a456-426614174000",
                "score": 0.85,
                "data_point": {
                    "amount": 5000.0,
                    "merchant_id": "M123",
                    "time": "02:30:00"
                },
                "detector_name": "IsolationForest",
                "timestamp": "2024-01-01T00:00:00",
                "severity": "high",
                "explanation": "Unusual transaction amount and time"
            }
        }


class DetectionResultDTO(BaseModel):
    """DTO for detection results."""
    
    model_config = ConfigDict(from_attributes=True)
    
    id: UUID
    detector_id: UUID
    dataset_id: UUID
    timestamp: datetime
    n_samples: int
    n_anomalies: int
    anomaly_rate: float = Field(ge=0, le=1)
    threshold: float
    execution_time_ms: Optional[float] = None
    
    # Summary statistics
    score_statistics: Dict[str, float]
    
    # Optional detailed results
    anomalies: Optional[List[AnomalyDTO]] = None
    scores: Optional[List[float]] = None
    labels: Optional[List[int]] = None
    
    # Metadata
    metadata: Dict[str, Any] = Field(default_factory=dict)
    has_confidence_intervals: bool = False
    
    class Config:
        json_schema_extra = {
            "example": {
                "id": "123e4567-e89b-12d3-a456-426614174000",
                "detector_id": "456e7890-e89b-12d3-a456-426614174000",
                "dataset_id": "789e0123-e89b-12d3-a456-426614174000",
                "timestamp": "2024-01-01T00:00:00",
                "n_samples": 10000,
                "n_anomalies": 150,
                "anomaly_rate": 0.015,
                "threshold": 0.75,
                "execution_time_ms": 1234.5,
                "score_statistics": {
                    "min": 0.0,
                    "max": 0.95,
                    "mean": 0.15,
                    "median": 0.10,
                    "std": 0.18
                }
            }
        }


class DetectionComparisonDTO(BaseModel):
    """DTO for comparing detection results."""
    
    detectors: Dict[str, Dict[str, Any]]
    summary: Dict[str, float]
    dataset_name: str
    comparison_date: datetime
    
    class Config:
        json_schema_extra = {
            "example": {
                "detectors": {
                    "detector_1": {
                        "name": "IsolationForest",
                        "n_anomalies": 150,
                        "precision": 0.85,
                        "recall": 0.78,
                        "f1": 0.81
                    },
                    "detector_2": {
                        "name": "LocalOutlierFactor",
                        "n_anomalies": 180,
                        "precision": 0.82,
                        "recall": 0.84,
                        "f1": 0.83
                    }
                },
                "summary": {
                    "f1_mean": 0.82,
                    "f1_std": 0.01,
                    "f1_best": 0.83
                },
                "dataset_name": "credit_card_transactions",
                "comparison_date": "2024-01-01T00:00:00"
            }
        }