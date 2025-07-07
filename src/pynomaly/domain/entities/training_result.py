"""Training result domain entity."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Dict, Optional
from uuid import UUID, uuid4


@dataclass
class TrainingResult:
    """Result of detector training operation.

    Contains comprehensive information about the training process,
    including metrics, performance indicators, and metadata.

    Attributes:
        id: Unique identifier for the training result
        detector_id: ID of the detector that was trained
        dataset_id: ID of the dataset used for training
        success: Whether training completed successfully
        metrics: Training metrics and performance indicators
        training_duration: Time taken for training in seconds
        created_at: When the training was completed
        error_message: Error message if training failed
        metadata: Additional training metadata
    """

    detector_id: UUID
    dataset_id: UUID
    success: bool
    metrics: Dict[str, Any] = field(default_factory=dict)
    training_duration: Optional[float] = None
    id: UUID = field(default_factory=uuid4)
    created_at: datetime = field(default_factory=datetime.utcnow)
    error_message: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        """Validate training result after initialization."""
        if self.success and self.error_message:
            raise ValueError("Successful training cannot have error message")

        if not self.success and not self.error_message:
            raise ValueError("Failed training must have error message")

    @property
    def training_time_minutes(self) -> Optional[float]:
        """Get training duration in minutes."""
        if self.training_duration is None:
            return None
        return self.training_duration / 60.0

    @property
    def samples_processed(self) -> Optional[int]:
        """Get number of samples processed during training."""
        return self.metrics.get("n_samples")

    @property
    def model_parameters(self) -> Optional[Dict[str, Any]]:
        """Get final model parameters after training."""
        return self.metrics.get("model_parameters")

    @property
    def validation_score(self) -> Optional[float]:
        """Get validation score if available."""
        return self.metrics.get("validation_score")

    def add_metric(self, key: str, value: Any) -> None:
        """Add a training metric."""
        self.metrics[key] = value

    def add_metadata(self, key: str, value: Any) -> None:
        """Add training metadata."""
        self.metadata[key] = value

    def get_summary(self) -> Dict[str, Any]:
        """Get a summary of the training result."""
        return {
            "id": str(self.id),
            "detector_id": str(self.detector_id),
            "dataset_id": str(self.dataset_id),
            "success": self.success,
            "training_duration": self.training_duration,
            "training_time_minutes": self.training_time_minutes,
            "samples_processed": self.samples_processed,
            "validation_score": self.validation_score,
            "created_at": self.created_at.isoformat(),
            "error_message": self.error_message,
            "metrics_count": len(self.metrics),
            "metadata_count": len(self.metadata),
        }

    @classmethod
    def success_result(
        cls,
        detector_id: UUID,
        dataset_id: UUID,
        metrics: Dict[str, Any],
        training_duration: Optional[float] = None,
        **metadata: Any,
    ) -> TrainingResult:
        """Create a successful training result."""
        return cls(
            detector_id=detector_id,
            dataset_id=dataset_id,
            success=True,
            metrics=metrics,
            training_duration=training_duration,
            metadata=metadata,
        )

    @classmethod
    def failure_result(
        cls,
        detector_id: UUID,
        dataset_id: UUID,
        error_message: str,
        training_duration: Optional[float] = None,
        **metadata: Any,
    ) -> TrainingResult:
        """Create a failed training result."""
        return cls(
            detector_id=detector_id,
            dataset_id=dataset_id,
            success=False,
            error_message=error_message,
            training_duration=training_duration,
            metadata=metadata,
        )

    def __repr__(self) -> str:
        """Developer representation."""
        status = "SUCCESS" if self.success else "FAILED"
        duration_str = (
            f"{self.training_duration:.2f}s" if self.training_duration else "N/A"
        )
        return (
            f"TrainingResult(id={self.id}, status={status}, "
            f"duration={duration_str}, samples={self.samples_processed})"
        )
