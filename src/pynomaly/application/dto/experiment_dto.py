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
    
    class Config:
        json_schema_extra = {
            "example": {
                "id": "run_123",
                "detector_name": "IsolationForest",
                "dataset_name": "credit_card_transactions",
                "parameters": {
                    "n_estimators": 100,
                    "contamination": 0.1
                },
                "metrics": {
                    "f1": 0.85,
                    "precision": 0.88,
                    "recall": 0.82,
                    "auc_roc": 0.91
                },
                "artifacts": {
                    "model": "/models/run_123/model.pkl",
                    "confusion_matrix": "/results/run_123/cm.png"
                },
                "timestamp": "2024-01-01T00:00:00"
            }
        }


class ExperimentDTO(BaseModel):
    """DTO for ML experiment."""
    
    model_config = ConfigDict(from_attributes=True)
    
    id: str
    name: str
    description: Optional[str] = None
    tags: List[str] = Field(default_factory=list)
    created_at: datetime
    runs: List[RunDTO] = Field(default_factory=list)
    
    # Computed fields
    n_runs: int = 0
    best_metric: Optional[float] = None
    best_run_id: Optional[str] = None
    
    def model_post_init(self, __context: Any) -> None:
        """Calculate computed fields after initialization."""
        self.n_runs = len(self.runs)
        
        if self.runs:
            # Find best F1 score
            best_f1 = 0.0
            for run in self.runs:
                if "f1" in run.metrics and run.metrics["f1"] > best_f1:
                    best_f1 = run.metrics["f1"]
                    self.best_metric = best_f1
                    self.best_run_id = run.id
    
    class Config:
        json_schema_extra = {
            "example": {
                "id": "exp_123",
                "name": "Credit Card Fraud Detection Benchmark",
                "description": "Comparing different algorithms for fraud detection",
                "tags": ["fraud", "benchmark", "production"],
                "created_at": "2024-01-01T00:00:00",
                "n_runs": 5,
                "best_metric": 0.91,
                "best_run_id": "run_456"
            }
        }


class CreateExperimentDTO(BaseModel):
    """DTO for creating a new experiment."""
    
    name: str = Field(min_length=1, max_length=200)
    description: Optional[str] = Field(None, max_length=1000)
    tags: List[str] = Field(default_factory=list, max_items=10)
    
    class Config:
        json_schema_extra = {
            "example": {
                "name": "Algorithm Comparison - Q1 2024",
                "description": "Comparing PyOD algorithms on production data",
                "tags": ["comparison", "q1-2024", "production"]
            }
        }


class LeaderboardEntryDTO(BaseModel):
    """DTO for leaderboard entry."""
    
    rank: int
    experiment: str
    run_id: str
    detector: str
    dataset: str
    metric_value: float
    metric_name: str
    timestamp: datetime
    
    class Config:
        json_schema_extra = {
            "example": {
                "rank": 1,
                "experiment": "Q1 2024 Benchmark",
                "run_id": "run_123",
                "detector": "IsolationForest",
                "dataset": "credit_card_transactions",
                "metric_value": 0.95,
                "metric_name": "f1",
                "timestamp": "2024-01-01T00:00:00"
            }
        }