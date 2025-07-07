"""Detector entity."""

from __future__ import annotations

import time

# Removed ABC and abstractmethod - Detector is a concrete domain entity
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import TYPE_CHECKING, Any, Protocol
from uuid import UUID, uuid4

import pandas as pd

from pynomaly.domain.value_objects import ContaminationRate

if TYPE_CHECKING:
    from pynomaly.domain.entities.dataset import Dataset
    from pynomaly.domain.entities.detection_result import DetectionResult
    from pynomaly.domain.entities.training_result import TrainingResult


class DetectorAlgorithm(Protocol):
    """Protocol for detector algorithm implementations."""

    def fit(self, data: pd.DataFrame) -> None:
        """Fit the detector on training data."""
        ...

    def predict(self, data: pd.DataFrame) -> list[int]:
        """Predict anomaly labels (0=normal, 1=anomaly)."""
        ...

    def score(self, data: pd.DataFrame) -> list[float]:
        """Calculate anomaly scores."""
        ...


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
    created_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    updated_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
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
        self.updated_at = datetime.now(timezone.utc)

    def update_parameters(self, **params: Any) -> None:
        """Update algorithm parameters."""
        self.parameters.update(params)
        # Reset fitted state when parameters change
        self.is_fitted = False
        self.trained_at = None
        self.updated_at = datetime.now(timezone.utc)

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

    def train(
        self, dataset: "Dataset", algorithm_adapter: DetectorAlgorithm
    ) -> "TrainingResult":
        """Train the detector with provided dataset.

        Args:
            dataset: The dataset to train on
            algorithm_adapter: The algorithm implementation to use

        Returns:
            TrainingResult: Comprehensive training metrics and status

        Raises:
            ValueError: If dataset is empty or invalid
            RuntimeError: If training fails
        """
        from pynomaly.domain.entities.training_result import TrainingResult

        # Validate inputs
        if dataset.data.empty:
            raise ValueError("Dataset cannot be empty")

        start_time = time.time()

        try:
            # Perform training
            algorithm_adapter.fit(dataset.data)

            # Calculate training metrics
            training_duration = time.time() - start_time

            # Get anomaly scores for validation
            scores = algorithm_adapter.score(dataset.data)

            metrics = {
                "n_samples": len(dataset.data),
                "n_features": len(dataset.data.columns),
                "training_time": training_duration,
                "algorithm": self.algorithm_name,
                "parameters": self.parameters.copy(),
                "mean_score": sum(scores) / len(scores) if scores else 0.0,
                "max_score": max(scores) if scores else 0.0,
                "min_score": min(scores) if scores else 0.0,
            }

            # Update detector state
            self.is_fitted = True
            self.trained_at = datetime.now(timezone.utc)
            self.updated_at = datetime.now(timezone.utc)

            return TrainingResult.success_result(
                detector_id=self.id,
                dataset_id=dataset.id,
                metrics=metrics,
                training_duration=training_duration,
                algorithm=self.algorithm_name,
                contamination_rate=self.contamination_rate.value,
            )

        except Exception as e:
            training_duration = time.time() - start_time

            return TrainingResult.failure_result(
                detector_id=self.id,
                dataset_id=dataset.id,
                error_message=str(e),
                training_duration=training_duration,
                algorithm=self.algorithm_name,
            )

    def detect(
        self, dataset: "Dataset", algorithm_adapter: DetectorAlgorithm
    ) -> "DetectionResult":
        """Detect anomalies in provided dataset.

        Args:
            dataset: The dataset to analyze
            algorithm_adapter: The algorithm implementation to use

        Returns:
            DetectionResult: Anomaly predictions, scores, and metadata

        Raises:
            ValueError: If detector is not fitted
        """
        from pynomaly.domain.entities.detection_result import DetectionResult

        # Validate detector state
        if not self.is_fitted:
            raise ValueError("Detector must be fitted before detection")

        # Handle empty dataset
        if dataset.data.empty:
            from pynomaly.domain.entities.anomaly import Anomaly

            return DetectionResult(
                detector_id=self.id,
                dataset_id=dataset.id,
                anomalies=[],
                scores=[],
                labels=[],
                threshold=0.5,
                execution_time_ms=0.0,
                metadata={
                    "n_samples": 0,
                    "algorithm": self.algorithm_name,
                    "detection_time": 0.0,
                },
            )

        start_time = time.time()

        try:
            # Get predictions and scores
            predictions = algorithm_adapter.predict(dataset.data)
            raw_scores = algorithm_adapter.score(dataset.data)

            detection_time = time.time() - start_time

            # Convert to domain objects
            from pynomaly.domain.entities.anomaly import Anomaly
            from pynomaly.domain.value_objects import AnomalyScore

            scores = [AnomalyScore(score) for score in raw_scores]

            # Create anomaly objects for positive predictions
            anomalies = []
            for i, (pred, score) in enumerate(zip(predictions, scores)):
                if pred == 1:  # Anomaly detected
                    # Extract data point as dictionary
                    data_point = dataset.data.iloc[i].to_dict()

                    anomaly = Anomaly(
                        score=score,
                        data_point=data_point,
                        detector_name=self.name,
                        metadata={"index": i, "detector_id": str(self.id)},
                    )
                    anomalies.append(anomaly)

            # Use contamination rate as threshold
            threshold = self.contamination_rate.value

            metadata = {
                "n_samples": len(dataset.data),
                "n_features": len(dataset.data.columns),
                "algorithm": self.algorithm_name,
                "detection_time": detection_time,
                "contamination_rate": self.contamination_rate.value,
            }

            return DetectionResult(
                detector_id=self.id,
                dataset_id=dataset.id,
                anomalies=anomalies,
                scores=scores,
                labels=predictions,
                threshold=threshold,
                execution_time_ms=detection_time * 1000,  # Convert to milliseconds
                metadata=metadata,
            )

        except Exception as e:
            # Return failed detection result
            return DetectionResult(
                detector_id=self.id,
                dataset_id=dataset.id,
                anomalies=[],
                scores=[],
                labels=[],
                threshold=0.5,
                execution_time_ms=(time.time() - start_time) * 1000,
                metadata={
                    "algorithm": self.algorithm_name,
                    "detection_time": time.time() - start_time,
                    "error": str(e),
                },
            )

    def __repr__(self) -> str:
        """Developer representation."""
        return (
            f"Detector(id={self.id}, name='{self.name}', "
            f"algorithm='{self.algorithm_name}', is_fitted={self.is_fitted})"
        )
