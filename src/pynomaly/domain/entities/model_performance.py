"""Model performance entities for monitoring and alerting."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import Dict, Any
from uuid import uuid4

from .alert import Alert, AlertType


@dataclass
class ModelPerformanceMetrics:
    """Captures model performance metrics for monitoring.
    
    Attributes:
        accuracy: Model accuracy score
        precision: Model precision score
        recall: Model recall score
        f1: Model F1 score
        timestamp: When the metrics were recorded
        model_id: Unique identifier for the model
        dataset_id: Unique identifier for the dataset
        metadata: Additional metadata about the metrics
    """
    accuracy: float
    precision: float
    recall: float
    f1: float
    timestamp: datetime
    model_id: str
    dataset_id: str
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def __post_init__(self) -> None:
        """Validate metrics after initialization."""
        # Validate score ranges
        for metric_name, score in [
            ("accuracy", self.accuracy),
            ("precision", self.precision),
            ("recall", self.recall),
            ("f1", self.f1)
        ]:
            if not (0.0 <= score <= 1.0):
                raise ValueError(f"{metric_name} must be between 0.0 and 1.0, got {score}")
        
        # Validate required fields
        if not self.model_id:
            raise ValueError("model_id cannot be empty")
        if not self.dataset_id:
            raise ValueError("dataset_id cannot be empty")
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert metrics to dictionary format."""
        return {
            "accuracy": self.accuracy,
            "precision": self.precision,
            "recall": self.recall,
            "f1": self.f1,
            "timestamp": self.timestamp.isoformat(),
            "model_id": self.model_id,
            "dataset_id": self.dataset_id,
            "metadata": self.metadata.copy()
        }


@dataclass
class ModelPerformanceBaseline:
    """Stores baseline statistics for model performance monitoring.
    
    Attributes:
        model_id: Unique identifier for the model
        version: Model version
        mean: Mean performance score
        std: Standard deviation of performance scores
        pct_thresholds: Percentile thresholds for performance degradation detection
    """
    model_id: str
    version: str
    mean: float
    std: float
    pct_thresholds: Dict[str, float] = field(default_factory=dict)
    
    def __post_init__(self) -> None:
        """Validate baseline after initialization."""
        if not self.model_id:
            raise ValueError("model_id cannot be empty")
        if not self.version:
            raise ValueError("version cannot be empty")
        if self.std < 0:
            raise ValueError("std must be non-negative")
        
        # Validate percentile thresholds
        for pct_name, threshold in self.pct_thresholds.items():
            if not (0.0 <= threshold <= 1.0):
                raise ValueError(f"Percentile threshold {pct_name} must be between 0.0 and 1.0, got {threshold}")
    
    def get_threshold(self, percentile: str) -> float | None:
        """Get threshold for a specific percentile."""
        return self.pct_thresholds.get(percentile)
    
    def set_threshold(self, percentile: str, threshold: float) -> None:
        """Set threshold for a specific percentile."""
        if not (0.0 <= threshold <= 1.0):
            raise ValueError(f"Threshold must be between 0.0 and 1.0, got {threshold}")
        self.pct_thresholds[percentile] = threshold
    
    def is_degraded(self, metric_value: float, percentile: str = "p95") -> bool:
        """Check if a metric value indicates performance degradation."""
        threshold = self.get_threshold(percentile)
        if threshold is None:
            # Fallback to using mean - 2*std as threshold
            threshold = max(0.0, self.mean - 2 * self.std)
        return metric_value < threshold
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert baseline to dictionary format."""
        return {
            "model_id": self.model_id,
            "version": self.version,
            "mean": self.mean,
            "std": self.std,
            "pct_thresholds": self.pct_thresholds.copy()
        }


class PerformanceDegradationAlert(Alert):
    """Specialized alert for model performance degradation issues.
    
    This is a subclass of Alert that is specifically designed for
    model performance degradation alerts. It enforces that the alert
    type is MODEL_PERFORMANCE.
    """
    
    def __post_init__(self) -> None:
        """Validate alert after initialization."""
        super().__post_init__()
        if self.alert_type != AlertType.MODEL_PERFORMANCE:
            raise TypeError(
                "PerformanceDegradationAlert must be of type MODEL_PERFORMANCE"
            )
    
    def add_performance_context(self, 
                              current_metrics: ModelPerformanceMetrics,
                              baseline: ModelPerformanceBaseline,
                              degradation_details: Dict[str, Any]) -> None:
        """Add performance-specific context to the alert."""
        self.metadata.update({
            "current_metrics": current_metrics.to_dict(),
            "baseline": baseline.to_dict(),
            "degradation_details": degradation_details
        })
    
    def get_performance_summary(self) -> Dict[str, Any]:
        """Get a summary of the performance degradation."""
        current_metrics = self.metadata.get("current_metrics", {})
        baseline = self.metadata.get("baseline", {})
        degradation_details = self.metadata.get("degradation_details", {})
        
        return {
            "model_id": current_metrics.get("model_id"),
            "dataset_id": current_metrics.get("dataset_id"),
            "current_performance": {
                "accuracy": current_metrics.get("accuracy"),
                "precision": current_metrics.get("precision"),
                "recall": current_metrics.get("recall"),
                "f1": current_metrics.get("f1")
            },
            "baseline_performance": {
                "mean": baseline.get("mean"),
                "std": baseline.get("std")
            },
            "degradation_severity": degradation_details.get("severity"),
            "affected_metrics": degradation_details.get("affected_metrics", [])
        }
