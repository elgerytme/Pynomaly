"""
Model Performance Metrics Entity

This module defines the ModelPerformanceMetrics entity for tracking
and comparing model performance across different metrics.
"""

from dataclasses import dataclass
from datetime import datetime
from typing import Any


@dataclass
class ModelPerformanceMetrics:
    """
    Entity representing performance metrics for a trained model.

    This entity encapsulates all performance-related data for a model,
    including standard metrics, custom metrics, and metadata.
    """

    model_id: str
    metrics: dict[str, Any]
    algorithm: str
    dataset_id: str
    training_job_id: str | None = None
    evaluation_timestamp: datetime | None = None
    hyperparameters: dict[str, Any] | None = None
    cross_validation_scores: dict[str, list] | None = None
    training_time: float | None = None
    inference_time: float | None = None
    model_size: int | None = None
    memory_usage: float | None = None

    def __post_init__(self):
        """Post-initialization validation and setup."""
        if self.evaluation_timestamp is None:
            self.evaluation_timestamp = datetime.utcnow()

        # Ensure metrics have the expected structure
        if not isinstance(self.metrics, dict):
            raise ValueError("Metrics must be a dictionary")

        # Normalize metrics to ensure consistent structure
        self._normalize_metrics()

    def _normalize_metrics(self) -> None:
        """Normalize metrics to ensure consistent structure."""
        normalized_metrics = {}

        for metric_name, metric_value in self.metrics.items():
            if isinstance(metric_value, dict):
                # Already in structured format
                normalized_metrics[metric_name] = metric_value
            else:
                # Convert to structured format
                normalized_metrics[metric_name] = {
                    "value": metric_value,
                    "mean": metric_value,
                    "std": 0.0,
                }

        self.metrics = normalized_metrics

    def get_metric_value(self, metric_name: str) -> float | None:
        """
        Get the primary value for a specific metric.

        Args:
            metric_name: Name of the metric

        Returns:
            The metric value or None if not found
        """
        if metric_name not in self.metrics:
            return None

        metric_data = self.metrics[metric_name]

        if isinstance(metric_data, dict):
            return metric_data.get("value", metric_data.get("mean"))

        return metric_data

    def get_primary_metrics(self) -> dict[str, float]:
        """
        Get primary performance metrics.

        Returns:
            Dictionary of primary metric values
        """
        primary_metric_names = [
            "accuracy",
            "precision",
            "recall",
            "f1_score",
            "roc_auc",
        ]

        primary_metrics = {}
        for metric_name in primary_metric_names:
            value = self.get_metric_value(metric_name)
            if value is not None:
                primary_metrics[metric_name] = value

        return primary_metrics

    def compare_with(
        self, other: "ModelPerformanceMetrics", metric_name: str = "f1_score"
    ) -> dict[str, Any]:
        """
        Compare this model's performance with another model.

        Args:
            other: Another ModelPerformanceMetrics instance
            metric_name: Primary metric for comparison

        Returns:
            Dictionary with comparison results
        """
        self_value = self.get_metric_value(metric_name)
        other_value = other.get_metric_value(metric_name)

        if self_value is None or other_value is None:
            return {
                "comparable": False,
                "reason": f"Missing {metric_name} metric in one or both models",
            }

        difference = self_value - other_value
        relative_improvement = (
            (difference / other_value) * 100 if other_value != 0 else 0
        )

        return {
            "comparable": True,
            "metric_name": metric_name,
            "self_value": self_value,
            "other_value": other_value,
            "difference": difference,
            "relative_improvement_percent": relative_improvement,
            "better_model": (
                self.model_id if self_value > other_value else other.model_id
            ),
            "self_algorithm": self.algorithm,
            "other_algorithm": other.algorithm,
        }

    def get_efficiency_score(self) -> float:
        """
        Calculate an efficiency score considering performance and resource usage.

        Returns:
            Efficiency score (higher is better)
        """
        # Get primary performance metric (defaulting to F1 score)
        performance = (
            self.get_metric_value("f1_score")
            or self.get_metric_value("accuracy")
            or 0.0
        )

        # Factor in training time (penalize slower training)
        time_penalty = 1.0
        if self.training_time and self.training_time > 0:
            # Normalize training time penalty (assuming 60 seconds as baseline)
            time_penalty = 60.0 / (60.0 + self.training_time)

        # Factor in model size (penalize larger models)
        size_penalty = 1.0
        if self.model_size and self.model_size > 0:
            # Normalize model size penalty (assuming 1MB as baseline)
            baseline_size_mb = 1024 * 1024  # 1MB in bytes
            size_penalty = baseline_size_mb / (baseline_size_mb + self.model_size)

        # Calculate composite efficiency score
        efficiency_score = performance * time_penalty * size_penalty

        return efficiency_score

    def to_dict(self) -> dict[str, Any]:
        """
        Convert to dictionary representation.

        Returns:
            Dictionary representation of the metrics
        """
        return {
            "model_id": self.model_id,
            "algorithm": self.algorithm,
            "dataset_id": self.dataset_id,
            "training_job_id": self.training_job_id,
            "evaluation_timestamp": (
                self.evaluation_timestamp.isoformat()
                if self.evaluation_timestamp
                else None
            ),
            "metrics": self.metrics,
            "hyperparameters": self.hyperparameters,
            "cross_validation_scores": self.cross_validation_scores,
            "training_time": self.training_time,
            "inference_time": self.inference_time,
            "model_size": self.model_size,
            "memory_usage": self.memory_usage,
            "primary_metrics": self.get_primary_metrics(),
            "efficiency_score": self.get_efficiency_score(),
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "ModelPerformanceMetrics":
        """
        Create instance from dictionary representation.

        Args:
            data: Dictionary containing model performance data

        Returns:
            ModelPerformanceMetrics instance
        """
        evaluation_timestamp = None
        if data.get("evaluation_timestamp"):
            evaluation_timestamp = datetime.fromisoformat(data["evaluation_timestamp"])

        return cls(
            model_id=data["model_id"],
            metrics=data["metrics"],
            algorithm=data["algorithm"],
            dataset_id=data["dataset_id"],
            training_job_id=data.get("training_job_id"),
            evaluation_timestamp=evaluation_timestamp,
            hyperparameters=data.get("hyperparameters"),
            cross_validation_scores=data.get("cross_validation_scores"),
            training_time=data.get("training_time"),
            inference_time=data.get("inference_time"),
            model_size=data.get("model_size"),
            memory_usage=data.get("memory_usage"),
        )
