"""Model performance metrics entity."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any
from uuid import UUID, uuid4

from pynomaly.domain.value_objects import PerformanceMetrics


@dataclass
class ModelPerformanceMetrics:
    """Entity representing model performance metrics."""

    model_id: UUID | str
    metrics: PerformanceMetrics | dict[str, Any]
    id: UUID = None
    metadata: dict[str, Any] = None

    def __post_init__(self):
        if self.id is None:
            self.id = uuid4()
        if self.metadata is None:
            self.metadata = {}

        # Convert string model_id to ensure consistency
        if isinstance(self.model_id, str):
            # Keep as string for test compatibility
            pass

    def get_accuracy(self) -> float:
        """Get accuracy metric."""
        if isinstance(self.metrics, dict):
            return self.metrics.get("accuracy", 0.0)
        return self.metrics.accuracy

    def get_precision(self) -> float:
        """Get precision metric."""
        if isinstance(self.metrics, dict):
            return self.metrics.get("precision", 0.0)
        return self.metrics.precision

    def get_recall(self) -> float:
        """Get recall metric."""
        if isinstance(self.metrics, dict):
            return self.metrics.get("recall", 0.0)
        return self.metrics.recall

    def get_f1_score(self) -> float:
        """Get F1 score metric."""
        if isinstance(self.metrics, dict):
            return self.metrics.get("f1_score", 0.0)
        return self.metrics.f1_score
