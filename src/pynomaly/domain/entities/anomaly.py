"""Simple anomaly entity without circular imports."""

from __future__ import annotations

from datetime import datetime
from typing import Any
from uuid import UUID

from pydantic import Field

from pynomaly.domain.abstractions import BaseEntity
from pynomaly.domain.value_objects import AnomalyScore


class Anomaly(BaseEntity):
    """Simple anomaly entity."""

    score: float | AnomalyScore
    data_point: dict[str, Any]
    detector_name: str
    timestamp: datetime = Field(default_factory=datetime.utcnow)
    explanation: str | dict[str, Any] | None = None

    def __init__(self, **data: Any) -> None:
        """Initialize anomaly with validation."""
        super().__init__(**data)
        self.validate_invariants()

    def validate_invariants(self) -> None:
        """Validate anomaly invariants."""
        super().validate_invariants()
        
        if not isinstance(self.score, (int, float, AnomalyScore)):
            raise TypeError(
                f"Score must be a number or AnomalyScore, got {type(self.score)}"
            )

        if not self.detector_name:
            raise ValueError("Detector name cannot be empty")

        if not isinstance(self.data_point, dict):
            raise TypeError(
                f"Data point must be a dictionary, got {type(self.data_point)}"
            )

    @property
    def severity(self) -> str:
        """Categorize anomaly severity based on score."""
        # Get the numeric value from score
        score_value = (
            self.score.value if isinstance(self.score, AnomalyScore) else self.score
        )

        if score_value > 0.9:
            return "critical"
        elif score_value > 0.7:
            return "high"
        elif score_value > 0.5:
            return "medium"
        else:
            return "low"

    def add_metadata(self, key: str, value: Any) -> None:
        """Add metadata to the anomaly."""
        self.metadata[key] = value
        self.mark_as_updated()

    def to_dict(self) -> dict[str, Any]:
        """Convert anomaly to dictionary representation."""
        # Get the numeric value from score
        score_value = (
            self.score.value if isinstance(self.score, AnomalyScore) else self.score
        )

        return {
            "id": str(self.id),
            "score": score_value,
            "detector_name": self.detector_name,
            "timestamp": self.timestamp.isoformat(),
            "data_point": self.data_point,
            "metadata": self.metadata,
            "severity": self.severity,
            "explanation": self.explanation,
            "created_at": self.created_at.isoformat(),
            "updated_at": self.updated_at.isoformat(),
            "version": self.version,
        }
