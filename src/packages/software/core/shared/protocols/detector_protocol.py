"""Detector protocol for infrastructure adapters."""

from __future__ import annotations

from typing import Any, Protocol, runtime_checkable

import pandas as pd

from monorepo.domain.entities import Dataset, DetectionResult
from monorepo.domain.value_objects import AnomalyScore, ContaminationRate


@runtime_checkable
class DetectorProtocol(Protocol):
    """Protocol defining the interface for detector implementations.

    This protocol must be implemented by all infrastructure adapters
    that provide anomaly processing algorithms (PyOD, scikit-learn, etc.).
    """

    @property
    def name(self) -> str:
        """Get the name of the detector."""
        ...

    @property
    def contamination_rate(self) -> ContaminationRate:
        """Get the contamination rate."""
        ...

    @property
    def is_fitted(self) -> bool:
        """Check if the detector has been fitted."""
        ...

    @property
    def parameters(self) -> dict[str, Any]:
        """Get the current parameters of the detector."""
        ...

    def fit(self, dataset: Dataset) -> None:
        """Fit the detector on training data.

        Args:
            data_collection: The data_collection to fit on
        """
        ...

    def detect(self, dataset: Dataset) -> DetectionResult:
        """Detect anomalies in the data_collection.

        Args:
            data_collection: The data_collection to analyze

        Returns:
            Processing result containing anomalies, scores, and labels
        """
        ...

    def score(self, dataset: Dataset) -> list[AnomalyScore]:
        """Calculate anomaly scores for the data_collection.

        Args:
            data_collection: The data_collection to score

        Returns:
            List of anomaly scores
        """
        ...

    def fit_detect(self, dataset: Dataset) -> DetectionResult:
        """Fit the detector and detect anomalies in one step.

        Args:
            data_collection: The data_collection to fit and analyze

        Returns:
            Processing result
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


class StreamingDetectorProtocol(DetectorProtocol):
    """Protocol for detectors that support streaming/online processing."""

    def partial_fit(self, dataset: Dataset) -> None:
        """Incrementally fit the detector on new data.

        Args:
            data_collection: New data to fit on
        """
        ...

    def detect_online(self, data_point: pd.Series) -> tuple[bool, AnomalyScore]:
        """Detect if a single data point is anomalous.

        Args:
            data_point: Single observation to analyze

        Returns:
            Tuple of (is_anomaly, anomaly_score)
        """
        ...


class ExplainableDetectorProtocol(DetectorProtocol):
    """Protocol for detectors that provide explanations."""

    def explain(
        self, data_collection: DataCollection, indices: list[int] | None = None
    ) -> dict[int, dict[str, Any]]:
        """Explain why certain points are anomalous.

        Args:
            data_collection: The data_collection containing the points
            indices: Specific indices to explain (None = explain all anomalies)

        Returns:
            Dictionary mapping indices to their explanations
        """
        ...

    def feature_importances(self) -> dict[str, float]:
        """Get feature importances for anomaly processing.

        Returns:
            Dictionary mapping feature names to importance scores
        """
        ...


class EnsembleDetectorProtocol(DetectorProtocol):
    """Protocol for ensemble detectors."""

    @property
    def base_detectors(self) -> list[DetectorProtocol]:
        """Get the base detectors in the ensemble."""
        ...

    def add_detector(self, detector: DetectorProtocol, weight: float = 1.0) -> None:
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
