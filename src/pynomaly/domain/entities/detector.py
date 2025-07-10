"""Detector entity."""

from __future__ import annotations

# Removed ABC and abstractmethod - Detector is a concrete domain entity
from dataclasses import dataclass, field
from datetime import datetime
from typing import TYPE_CHECKING, Any
from uuid import UUID, uuid4

from pynomaly.domain.value_objects import ContaminationRate

if TYPE_CHECKING:
    from pynomaly.domain.entities.dataset import Dataset
    from pynomaly.domain.entities.detection_result import DetectionResult
    from pynomaly.domain.entities.training_result import TrainingResult


@dataclass
class Detector:
    """Domain entity for anomaly detectors.

    This is a concrete domain entity that represents the concept of an anomaly detector,
    independent of any specific implementation or algorithm.

    Attributes:
        id: Unique identifier for the detector
        name: Name of the detector
        algorithm_name: Name of the underlying algorithm
        contamination_rate: Expected proportion of anomalies
        parameters: Algorithm-specific parameters
        metadata: Additional metadata
        created_at: When the detector was created
        trained_at: When the detector was last trained
        is_fitted: Whether the detector has been fitted
    """

    name: str
    algorithm_name: str
    contamination_rate: ContaminationRate = field(
        default_factory=ContaminationRate.auto
    )
    id: UUID = field(default_factory=uuid4)
    parameters: dict[str, Any] = field(default_factory=dict)
    metadata: dict[str, Any] = field(default_factory=dict)
    created_at: datetime = field(default_factory=datetime.utcnow)
    trained_at: datetime | None = None
    is_fitted: bool = False

    def __post_init__(self) -> None:
        """Validate detector after initialization."""
        if not self.name:
            raise ValueError("Detector name cannot be empty")

        if not self.algorithm_name:
            raise ValueError("Algorithm name cannot be empty")

        if not isinstance(self.contamination_rate, ContaminationRate):
            raise TypeError(
                f"Contamination rate must be ContaminationRate instance, "
                f"got {type(self.contamination_rate)}"
            )

    # Note: fit, detect, and score methods are implemented by infrastructure adapters
    # The domain entity only contains data and domain logic

    @property
    def requires_fitting(self) -> bool:
        """Check if detector requires fitting before detection."""
        # Most detectors require fitting, but some (like statistical tests) don't
        return self.metadata.get("requires_fitting", True)

    @property
    def supports_streaming(self) -> bool:
        """Check if detector supports streaming/online detection."""
        return self.metadata.get("supports_streaming", False)

    @property
    def supports_multivariate(self) -> bool:
        """Check if detector supports multivariate data."""
        return self.metadata.get("supports_multivariate", True)

    @property
    def time_complexity(self) -> str | None:
        """Get time complexity of the algorithm."""
        return self.metadata.get("time_complexity")

    @property
    def space_complexity(self) -> str | None:
        """Get space complexity of the algorithm."""
        return self.metadata.get("space_complexity")

    def update_metadata(self, key: str, value: Any) -> None:
        """Update detector metadata."""
        self.metadata[key] = value

    def update_parameters(self, **params: Any) -> None:
        """Update algorithm parameters."""
        self.parameters.update(params)
        # Reset fitted state when parameters change
        self.is_fitted = False
        self.trained_at = None

    def get_info(self) -> dict[str, Any]:
        """Get comprehensive information about the detector."""
        return {
            "id": str(self.id),
            "name": self.name,
            "algorithm": self.algorithm_name,
            "contamination_rate": self.contamination_rate.value,
            "is_fitted": self.is_fitted,
            "created_at": self.created_at.isoformat(),
            "trained_at": self.trained_at.isoformat() if self.trained_at else None,
            "parameters": self.parameters,
            "metadata": self.metadata,
            "requires_fitting": self.requires_fitting,
            "supports_streaming": self.supports_streaming,
            "supports_multivariate": self.supports_multivariate,
            "time_complexity": self.time_complexity,
            "space_complexity": self.space_complexity,
        }

    def mark_as_fitted(self) -> None:
        """Mark the detector as fitted."""
        self.is_fitted = True
        self.trained_at = datetime.utcnow()

    def mark_as_unfitted(self) -> None:
        """Mark the detector as not fitted."""
        self.is_fitted = False
        self.trained_at = None

    def validate_for_detection(self) -> None:
        """Validate that the detector is ready for detection.
        
        Raises:
            ValueError: If detector is not fitted
        """
        if not self.is_fitted:
            raise ValueError("Detector must be fitted before detection")

    def __repr__(self) -> str:
        """Developer representation."""
        return (
            f"Detector(id={self.id}, name='{self.name}', "
            f"algorithm='{self.algorithm_name}', is_fitted={self.is_fitted})"
        )
