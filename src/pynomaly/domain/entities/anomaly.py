"""Anomaly entity."""

from __future__ import annotations

import warnings
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any
from uuid import UUID, uuid4

from pynomaly.domain.value_objects.anomaly_score import AnomalyScore
from pynomaly.domain.value_objects.anomaly_type import AnomalyType
from pynomaly.domain.value_objects.confidence_interval import ConfidenceInterval
from pynomaly.domain.value_objects.severity_level import SeverityLevel


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
        anomaly_type: Type of anomaly detected
        severity_level: Severity level of the anomaly
    """

    score: AnomalyScore
    data_point: dict[str, Any]
    detector_name: str
    id: UUID = field(default_factory=uuid4)
    timestamp: datetime = field(default_factory=datetime.utcnow)
    metadata: dict[str, Any] = field(default_factory=dict)
    explanation: str | None = None
    confidence_interval: ConfidenceInterval | None = None
    anomaly_type: AnomalyType = AnomalyType.UNKNOWN
    severity_level: SeverityLevel | None = None

    def __post_init__(self) -> None:
        """Validate anomaly after initialization."""
        if not isinstance(self.score, AnomalyScore):
            raise TypeError(
                f"Score must be AnomalyScore instance, got {type(self.score)}"
            )

        if not self.detector_name:
            raise ValueError("Detector name cannot be empty")

        if not isinstance(self.data_point, dict):
            raise TypeError(
                f"Data point must be a dictionary, got {type(self.data_point)}"
            )

        if not isinstance(self.anomaly_type, AnomalyType):
            raise TypeError(
                f"Anomaly type must be AnomalyType instance, got {type(self.anomaly_type)}"
            )

        if self.severity_level is not None and not isinstance(self.severity_level, SeverityLevel):
            raise TypeError(
                f"Severity level must be SeverityLevel instance, got {type(self.severity_level)}"
            )

        # Auto-derive severity_level from score if not provided
        if self.severity_level is None:
            self.severity_level = SeverityLevel.from_score(self.score.value)

        # Validate consistency between score, anomaly_type, and severity_level
        self._validate_consistency()

    def _validate_consistency(self) -> None:
        """Validate consistency between score, anomaly_type, and severity_level."""
        # Validate that severity_level is consistent with score
        expected_severity = SeverityLevel.from_score(self.score.value)
        if self.severity_level != expected_severity:
            warnings.warn(
                f"Severity level {self.severity_level} may not be consistent with score {self.score.value} "
                f"(expected {expected_severity})",
                UserWarning,
                stacklevel=2,
            )

        # Additional consistency checks can be added here
        # For example, certain anomaly types might require specific severity ranges

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
        """DEPRECATED: Use severity_level instead. Redirects to severity_level."""
        warnings.warn(
            "The 'severity' property is deprecated. Use 'severity_level' instead.",
            DeprecationWarning,
            stacklevel=2,
        )
        return str(self.severity_level)

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
            "anomaly_type": self.anomaly_type.value,
            "severity_level": str(self.severity_level),
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
