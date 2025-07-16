"""Generic detection protocol for any detection algorithm."""

from __future__ import annotations

from typing import Any, Generic, Protocol, TypeVar, runtime_checkable

# Generic type for input data
T = TypeVar('T')
# Generic type for detection results  
R = TypeVar('R')
# Generic type for scoring results
S = TypeVar('S')


@runtime_checkable
class GenericDetectionProtocol(Protocol, Generic[T, R, S]):
    """Protocol defining the interface for any detection algorithm.

    This protocol can be implemented by infrastructure adapters
    for any type of detection (anomaly, fraud, intrusion, malware, etc.).
    """

    @property
    def name(self) -> str:
        """Get the name of the detector."""
        ...

    @property
    def algorithm_type(self) -> str:
        """Get the type of detection algorithm (e.g., 'anomaly', 'fraud')."""
        ...

    @property
    def is_fitted(self) -> bool:
        """Check if the detector has been fitted."""
        ...

    @property
    def parameters(self) -> dict[str, Any]:
        """Get the current parameters of the detector."""
        ...

    def fit(self, data: T) -> None:
        """Fit the detector on training data.

        Args:
            data: The training data
        """
        ...

    def detect(self, data: T) -> R:
        """Detect patterns in the data.

        Args:
            data: The data to analyze

        Returns:
            Detection results
        """
        ...

    def score(self, data: T) -> S:
        """Calculate scores for the data.

        Args:
            data: The data to score

        Returns:
            Scores for each data point
        """
        ...

    def fit_detect(self, data: T) -> R:
        """Fit the detector and detect patterns in one step.

        Args:
            data: The data to fit and analyze

        Returns:
            Detection results
        """
        ...

    def get_params(self) -> dict[str, Any]:
        """Get parameters of the detector.

        Returns:
            Dictionary of parameters
        """
        ...

    def set_params(self, **params: Any) -> None:
        """Set parameters of the detector.

        Args:
            **params: Parameters to set
        """
        ...


@runtime_checkable
class StreamingDetectionProtocol(GenericDetectionProtocol[T, R, S]):
    """Protocol for detectors that support streaming/online detection."""

    def partial_fit(self, data: T) -> None:
        """Incrementally fit the detector on new data.

        Args:
            data: New data to fit on
        """
        ...

    def detect_online(self, data_point: Any) -> tuple[bool, float]:
        """Detect patterns in a single data point.

        Args:
            data_point: Single observation to analyze

        Returns:
            Tuple of (has_pattern, confidence_score)
        """
        ...


@runtime_checkable
class ExplainableDetectionProtocol(GenericDetectionProtocol[T, R, S]):
    """Protocol for detectors that provide explanations."""

    def explain(
        self, data: T, indices: list[int] | None = None
    ) -> dict[int, dict[str, Any]]:
        """Explain why certain patterns were detected.

        Args:
            data: The data containing the points
            indices: Specific indices to explain (None = explain all detections)

        Returns:
            Dictionary mapping indices to their explanations
        """
        ...

    def feature_importances(self) -> dict[str, float]:
        """Get feature importances for pattern detection.

        Returns:
            Dictionary mapping feature names to importance scores
        """
        ...


@runtime_checkable
class EnsembleDetectionProtocol(GenericDetectionProtocol[T, R, S]):
    """Protocol for ensemble detectors."""

    @property
    def base_detectors(self) -> list[GenericDetectionProtocol[T, R, S]]:
        """Get the base detectors in the ensemble."""
        ...

    def add_detector(
        self, detector: GenericDetectionProtocol[T, R, S], weight: float = 1.0
    ) -> None:
        """Add a detector to the ensemble.

        Args:
            detector: Detector to add
            weight: Weight for this detector's votes
        """
        ...

    def remove_detector(self, detector_name: str) -> None:
        """Remove a detector from the ensemble.

        Args:
            detector_name: Name of detector to remove
        """
        ...

    def get_detector_weights(self) -> dict[str, float]:
        """Get weights of all detectors in the ensemble.

        Returns:
            Dictionary mapping detector names to weights
        """
        ...


@runtime_checkable
class BatchDetectionProtocol(GenericDetectionProtocol[T, R, S]):
    """Protocol for detectors that support batch processing."""

    def detect_batch(self, data_batches: list[T]) -> list[R]:
        """Detect patterns in multiple data batches.

        Args:
            data_batches: List of data batches to analyze

        Returns:
            List of detection results for each batch
        """
        ...

    def score_batch(self, data_batches: list[T]) -> list[S]:
        """Score multiple data batches.

        Args:
            data_batches: List of data batches to score

        Returns:
            List of scores for each batch
        """
        ...