"""Model performance entities for tracking model quality and baselines."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import Dict, Any, Optional
from uuid import UUID, uuid4


@dataclass
class ModelPerformanceMetrics:
    """Performance metrics for a model."""
    
    model_id: str
    metrics: Dict[str, float]
    timestamp: datetime = field(default_factory=datetime.utcnow)
    id: UUID = field(default_factory=uuid4)
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def __post_init__(self):
        """Validate metrics."""
        if not self.model_id:
            raise ValueError("Model ID cannot be empty")
        if not isinstance(self.metrics, dict):
            raise TypeError("Metrics must be a dictionary")
    
    def get_metric(self, name: str) -> Optional[float]:
        """Get a specific metric value."""
        return self.metrics.get(name)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "id": str(self.id),
            "model_id": self.model_id,
            "metrics": self.metrics,
            "timestamp": self.timestamp.isoformat(),
            "metadata": self.metadata,
        }


@dataclass
class ModelPerformanceBaseline:
    """Baseline performance for a model."""
    
    model_id: str
    baseline_metrics: Dict[str, float]
    baseline_date: datetime = field(default_factory=datetime.utcnow)
    id: UUID = field(default_factory=uuid4)
    description: str = ""
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def __post_init__(self):
        """Validate baseline."""
        if not self.model_id:
            raise ValueError("Model ID cannot be empty")
        if not isinstance(self.baseline_metrics, dict):
            raise TypeError("Baseline metrics must be a dictionary")
    
    def get_baseline_metric(self, name: str) -> Optional[float]:
        """Get a specific baseline metric value."""
        return self.baseline_metrics.get(name)
    
    def is_performance_degraded(self, current_metrics: Dict[str, float], threshold: float = 0.05) -> bool:
        """Check if performance has degraded compared to baseline."""
        for metric_name, baseline_value in self.baseline_metrics.items():
            if metric_name in current_metrics:
                current_value = current_metrics[metric_name]
                # Assuming higher values are better for now
                if (baseline_value - current_value) / baseline_value > threshold:
                    return True
        return False
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "id": str(self.id),
            "model_id": self.model_id,
            "baseline_metrics": self.baseline_metrics,
            "baseline_date": self.baseline_date.isoformat(),
            "description": self.description,
            "metadata": self.metadata,
        }
