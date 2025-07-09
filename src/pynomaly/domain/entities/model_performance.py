"""Model performance metrics entity."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any
from uuid import UUID, uuid4

from pynomaly.domain.value_objects import PerformanceMetrics


@dataclass
class ModelPerformanceMetrics:
    """Entity representing model performance metrics."""

    model_id: UUID
    metrics: PerformanceMetrics
    id: UUID = None
    metadata: dict[str, Any] = None

    def __post_init__(self):
        if self.id is None:
            self.id = uuid4()
        if self.metadata is None:
            self.metadata = {}

    def get_accuracy(self) -> float:
        """Get accuracy metric."""
        return self.metrics.accuracy

    def get_precision(self) -> float:
        """Get precision metric."""
        return self.metrics.precision

    def get_recall(self) -> float:
        """Get recall metric."""
        return self.metrics.recall

    def get_f1_score(self) -> float:
        """Get F1 score metric."""
        return self.metrics.f1_score
