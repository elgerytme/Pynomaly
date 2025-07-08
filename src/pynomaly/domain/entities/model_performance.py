"""Model performance entities for tracking model metrics and baselines."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Dict, List, Optional
from uuid import UUID, uuid4


@dataclass
class ModelPerformanceMetrics:
    """Entity representing model performance metrics."""
    
    # Identification
    id: UUID = field(default_factory=uuid4)
    model_id: str = ""
    dataset_id: str = ""
    
    # Core metrics
    accuracy: float = 0.0
    precision: float = 0.0
    recall: float = 0.0
    f1_score: float = 0.0
    roc_auc: float = 0.0
    
    # Additional metrics
    false_positive_rate: float = 0.0
    false_negative_rate: float = 0.0
    specificity: float = 0.0
    sensitivity: float = 0.0
    
    # Performance characteristics
    training_time: float = 0.0
    inference_time: float = 0.0
    memory_usage: float = 0.0
    
    # Metadata
    timestamp: datetime = field(default_factory=datetime.utcnow)
    version: str = "1.0.0"
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert metrics to dictionary."""
        return {
            "id": str(self.id),
            "model_id": self.model_id,
            "dataset_id": self.dataset_id,
            "accuracy": self.accuracy,
            "precision": self.precision,
            "recall": self.recall,
            "f1_score": self.f1_score,
            "roc_auc": self.roc_auc,
            "false_positive_rate": self.false_positive_rate,
            "false_negative_rate": self.false_negative_rate,
            "specificity": self.specificity,
            "sensitivity": self.sensitivity,
            "training_time": self.training_time,
            "inference_time": self.inference_time,
            "memory_usage": self.memory_usage,
            "timestamp": self.timestamp.isoformat(),
            "version": self.version,
            "metadata": self.metadata,
        }


@dataclass
class ModelPerformanceBaseline:
    """Entity representing a performance baseline for a model."""
    
    # Identification
    id: UUID = field(default_factory=uuid4)
    model_id: str = ""
    version: str = "1.0.0"
    
    # Baseline statistics
    mean: float = 0.0
    std: float = 0.0
    min_value: float = 0.0
    max_value: float = 0.0
    
    # Percentile thresholds
    pct_thresholds: Dict[str, float] = field(default_factory=dict)
    
    # Metadata
    created_at: datetime = field(default_factory=datetime.utcnow)
    updated_at: datetime = field(default_factory=datetime.utcnow)
    description: Optional[str] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert baseline to dictionary."""
        return {
            "id": str(self.id),
            "model_id": self.model_id,
            "version": self.version,
            "mean": self.mean,
            "std": self.std,
            "min_value": self.min_value,
            "max_value": self.max_value,
            "pct_thresholds": self.pct_thresholds,
            "created_at": self.created_at.isoformat(),
            "updated_at": self.updated_at.isoformat(),
            "description": self.description,
        }
    
    def is_degraded(self, current_value: float, threshold_factor: float = 2.0) -> bool:
        """Check if current value indicates performance degradation."""
        if self.std == 0:
            return False
        
        # Check if the value is significantly below the mean
        z_score = (current_value - self.mean) / self.std
        return z_score < -threshold_factor
    
    def get_threshold(self, percentile: str) -> Optional[float]:
        """Get threshold for a specific percentile."""
        return self.pct_thresholds.get(percentile)
