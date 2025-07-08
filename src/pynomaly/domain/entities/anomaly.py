"""Anomaly entity."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import Any
from uuid import UUID, uuid4

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from pynomaly.domain.value_objects import AnomalyScore, ConfidenceInterval
else:
    from pynomaly.domain.value_objects import AnomalyScore, ConfidenceInterval


@dataclass
class Anomaly:
    """Entity representing a detected anomaly.

    Attributes:
        id: Unique identifier for the anomaly
        score: The anomaly score
        data_point: The original data point
        timestamp: When the anomaly was detected
        detector_name: Name of the detector that found this anomaly
        metadata: Additional metadata about the anomaly
        explanation: Optional explanation for why this is anomalous
        confidence_interval: Optional confidence interval for the anomaly
    """

    score: AnomalyScore  # Deprecated, use severity_score instead
    data_point: dict[str, Any]
    detector_name: str
    id: UUID = field(default_factory=uuid4)
    timestamp: datetime = field(default_factory=datetime.utcnow)
    metadata: dict[str, Any] = field(default_factory=dict)
    explanation: str | None = None
    confidence_interval: ConfidenceInterval | None = None
    anomaly_type: AnomalyType = field(default_factory=AnomalyType.get_default)
    anomaly_category: AnomalyCategory = field(default_factory=AnomalyCategory.get_default)
    severity_score: SeverityScore = field(default_factory=SeverityScore.create_minimal)

    def __post_init__(self) -> None:
        """Validate anomaly after initialization."""
        if self.score is not None and not isinstance(self.score, AnomalyScore):
            raise TypeError(
                f"Score must be AnomalyScore instance, got {type(self.score)}"
            )

        if not self.detector_name:
            raise ValueError("Detector name cannot be empty")

        if not isinstance(self.data_point, dict):
            raise TypeError(
                f"Data point must be a dictionary, got {type(self.data_point)}"
            )

        if not isinstance(self.severity_score, SeverityScore):
            raise TypeError(
                f"SeverityScore must be an instance of SeverityScore, got {type(self.severity_score)}"
            )

    @property
    def is_high_confidence(self) -> bool:
        """Check if anomaly has high confidence (narrow interval)."""
        if self.confidence_interval is None:
            return False

        # Consider high confidence if interval width is < 10% of score
        relative_width = self.confidence_interval.width() / self.score.value
        return relative_width < 0.1

    @property
    def severity(self) -> str:
        """Categorize anomaly severity based on score."""
        # This is a simple example - could be customized per detector
        if self.score.value > 0.9:
            return "critical"
        elif self.score.value > 0.7:
            return "high"
        elif self.score.value > 0.5:
            return "medium"
        else:
            return "low"

    @property
    def severity_level(self) -> SeverityLevel:
        """Get the severity level from severity score."""
        return self.severity_score.severity_level

    def add_metadata(self, key: str, value: Any) -> None:
        """Add metadata to the anomaly."""
        self.metadata[key] = value

    def to_dict(self) -> dict[str, Any]:
        """Convert anomaly to dictionary representation."""
        result = {
            "id": str(self.id),
            "score": self.score.value,
            "detector_name": self.detector_name,
            "timestamp": self.timestamp.isoformat(),
            "data_point": self.data_point,
            "metadata": self.metadata,
            "severity": self.severity,
        }

        if self.score.confidence_lower is not None:
            result["score_confidence_lower"] = self.score.confidence_lower

        if self.score.confidence_upper is not None:
            result["score_confidence_upper"] = self.score.confidence_upper

        if self.explanation:
            result["explanation"] = self.explanation

        if self.confidence_interval:
            result["confidence_interval"] = {
                "lower": self.confidence_interval.lower,
                "upper": self.confidence_interval.upper,
                "level": self.confidence_interval.confidence_level,
            }

        return result

    def __eq__(self, other: object) -> bool:
        """Check equality based on ID."""
        if not isinstance(other, Anomaly):
            return NotImplemented
        return self.id == other.id

    def __hash__(self) -> int:
        """Hash based on ID."""
        return hash(self.id)
