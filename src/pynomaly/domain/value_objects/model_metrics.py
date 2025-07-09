"""Model metrics value object."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class ModelMetrics:
    """Value object representing model metrics."""

    accuracy: float
    precision: float
    recall: float
    f1_score: float
    auc_score: float = 0.0
    metadata: dict[str, Any] = None

    def __post_init__(self):
        if self.metadata is None:
            object.__setattr__(self, "metadata", {})

        # Validate metric ranges
        for metric_name, value in [
            ("accuracy", self.accuracy),
            ("precision", self.precision),
            ("recall", self.recall),
            ("f1_score", self.f1_score),
            ("auc_score", self.auc_score),
        ]:
            if not 0.0 <= value <= 1.0:
                raise ValueError(
                    f"{metric_name} must be between 0.0 and 1.0, got {value}"
                )

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "accuracy": self.accuracy,
            "precision": self.precision,
            "recall": self.recall,
            "f1_score": self.f1_score,
            "auc_score": self.auc_score,
            "metadata": self.metadata,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> ModelMetrics:
        """Create from dictionary."""
        return cls(
            accuracy=data["accuracy"],
            precision=data["precision"],
            recall=data["recall"],
            f1_score=data["f1_score"],
            auc_score=data.get("auc_score", 0.0),
            metadata=data.get("metadata", {}),
        )
