"""Simple anomaly entity without circular imports."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import Any
from uuid import UUID, uuid4

from monorepo_detection.domain.value_objects import AnomalyScore


@dataclass
class Anomaly:
    """Simple anomaly entity."""

    score: float | AnomalyScore
    data_point: dict[str, Any]
    detector_name: str
    id: UUID = field(default_factory=uuid4)
    timestamp: datetime = field(default_factory=datetime.utcnow)
    metadata: dict[str, Any] = field(default_factory=dict)
    explanation: str | None = None

    def __post_init__(self) -> None:
        """Validate anomaly after initialization."""
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
        }

    def __eq__(self, other: object) -> bool:
        """Check equality based on ID."""
        if not isinstance(other, Anomaly):
            return NotImplemented
        return self.id == other.id

    def __hash__(self) -> int:
        """Hash based on ID."""
        return hash(self.id)