"""Simple anomaly entity without circular imports."""

from __future__ import annotations

import warnings
from dataclasses import dataclass, field
from datetime import datetime
from typing import TYPE_CHECKING, Any
from uuid import UUID, uuid4

if TYPE_CHECKING:
    pass


@dataclass
class Anomaly:
    """Simple anomaly entity."""

    score: Any  # Can be AnomalyScore or float
    data_point: dict[str, Any]
    detector_name: str
    id: UUID = field(default_factory=uuid4)
    timestamp: datetime = field(default_factory=datetime.utcnow)
    metadata: dict[str, Any] = field(default_factory=dict)
    explanation: str | None = None

    def __post_init__(self) -> None:
        """Validate anomaly after initialization."""
        from pynomaly.domain.value_objects import AnomalyScore

        # Only accept AnomalyScore objects
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

    @property
    def severity(self) -> str:
        """Categorize anomaly severity based on score.
        
        .. deprecated:: 1.2.0
            The `severity` property is deprecated and will be removed in v2.0.0.
            Use `AnomalyClassificationService` instead for more flexible and 
            extensible severity classification.
            
            Migration example:
            
            .. code-block:: python
            
                # Old way (deprecated)
                severity = anomaly.severity
                
                # New way (recommended)
                from pynomaly.application.services.anomaly_classification_service import AnomalyClassificationService
                
                service = AnomalyClassificationService()
                service.classify(anomaly)
                severity = anomaly.metadata.get('severity')
        """
        warnings.warn(
            "The 'severity' property is deprecated and will be removed in v2.0.0. "
            "Use AnomalyClassificationService for more flexible severity classification. "
            "See migration guide: https://github.com/yourusername/pynomaly/blob/main/docs/migration/v2.0-migration-guide.md",
            DeprecationWarning,
            stacklevel=2
        )
        
        # Handle both AnomalyScore and float
        score_value = self.score.value if hasattr(self.score, "value") else self.score

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
        return {
            "id": str(self.id),
            "score": self.score,
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
