"""Detection result entity."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import Any
from uuid import UUID, uuid4

import numpy as np
import pandas as pd

from monorepo.domain.entities.anomaly import Anomaly
from monorepo.domain.value_objects import AnomalyScore, ConfidenceInterval


@dataclass
class DetectionResult:
    """Entity representing the result of anomaly detection.

    Attributes:
        id: Unique identifier for the result
        detector_id: ID of the detector that produced this result
        dataset_id: ID of the dataset that was analyzed
        anomalies: List of detected anomalies
        scores: Anomaly scores for all data points
        labels: Binary labels (0=normal, 1=anomaly) for all data points
        threshold: Score threshold used for classification
        execution_time_ms: Time taken to perform detection (milliseconds)
        timestamp: When the detection was performed
        metadata: Additional metadata about the detection
        confidence_intervals: Optional confidence intervals for predictions
    """

    detector_id: UUID
    dataset_id: UUID
    anomalies: list[Anomaly]
    scores: list[AnomalyScore]
    labels: np.ndarray
    threshold: float
    id: UUID = field(default_factory=uuid4)
    execution_time_ms: float | None = None
    timestamp: datetime = field(default_factory=datetime.utcnow)
    metadata: dict[str, Any] = field(default_factory=dict)
    confidence_intervals: list[ConfidenceInterval] | None = None

    def __post_init__(self) -> None:
        """Validate detection result after initialization."""
        # Convert labels to numpy array if needed
        if not isinstance(self.labels, np.ndarray):
            self.labels = np.array(self.labels)

        # Validate dimensions
        n_samples = len(self.labels)

        if len(self.scores) != n_samples:
            raise ValueError(
                f"Number of scores ({len(self.scores)}) doesn't match "
                f"number of labels ({n_samples})"
            )

        if self.confidence_intervals and len(self.confidence_intervals) != n_samples:
            raise ValueError(
                f"Number of confidence intervals ({len(self.confidence_intervals)}) "
                f"doesn't match number of samples ({n_samples})"
            )

        # Validate labels are binary
        unique_labels = np.unique(self.labels)
        if not set(unique_labels).issubset({0, 1}):
            raise ValueError(
                f"Labels must be binary (0 or 1), got unique values: {unique_labels}"
            )

        # Validate anomalies match labels
        n_anomaly_labels = np.sum(self.labels == 1)
        if len(self.anomalies) != n_anomaly_labels:
            raise ValueError(
                f"Number of anomalies ({len(self.anomalies)}) doesn't match "
                f"number of anomaly labels ({n_anomaly_labels})"
            )

    @property
    def n_samples(self) -> int:
        """Get total number of samples analyzed."""
        return len(self.labels)

    @property
    def n_anomalies(self) -> int:
        """Get number of detected anomalies."""
        return len(self.anomalies)

    @property
    def n_normal(self) -> int:
        """Get number of normal samples."""
        return self.n_samples - self.n_anomalies

    @property
    def anomaly_rate(self) -> float:
        """Get proportion of samples classified as anomalies."""
        if self.n_samples == 0:
            return 0.0
        return self.n_anomalies / self.n_samples

    @property
    def anomaly_indices(self) -> np.ndarray:
        """Get indices of anomalous samples."""
        return np.where(self.labels == 1)[0]

    @property
    def normal_indices(self) -> np.ndarray:
        """Get indices of normal samples."""
        return np.where(self.labels == 0)[0]

    @property
    def score_statistics(self) -> dict[str, float]:
        """Get statistics of anomaly scores."""
        score_values = [s.value for s in self.scores]
        return {
            "min": float(np.min(score_values)),
            "max": float(np.max(score_values)),
            "mean": float(np.mean(score_values)),
            "median": float(np.median(score_values)),
            "std": float(np.std(score_values)),
            "q25": float(np.percentile(score_values, 25)),
            "q75": float(np.percentile(score_values, 75)),
        }

    @property
    def has_confidence_intervals(self) -> bool:
        """Check if result includes confidence intervals."""
        return self.confidence_intervals is not None

    def get_top_anomalies(self, n: int = 10) -> list[Anomaly]:
        """Get top N anomalies by score."""
        sorted_anomalies = sorted(
            self.anomalies, key=lambda a: a.score.value, reverse=True
        )
        return sorted_anomalies[:n]

    def get_scores_dataframe(self) -> pd.DataFrame:
        """Get scores as a DataFrame for analysis."""
        data = {
            "score": [s.value for s in self.scores],
            "label": self.labels,
        }

        if self.has_confidence_intervals:
            data["ci_lower"] = [ci.lower for ci in self.confidence_intervals]  # type: ignore
            data["ci_upper"] = [ci.upper for ci in self.confidence_intervals]  # type: ignore
            data["ci_width"] = [ci.width for ci in self.confidence_intervals]  # type: ignore

        return pd.DataFrame(data)

    def filter_by_score(self, min_score: float) -> list[Anomaly]:
        """Get anomalies with score above threshold."""
        return [a for a in self.anomalies if a.score.value >= min_score]

    def filter_by_confidence(self, min_level: float = 0.95) -> list[Anomaly]:
        """Get anomalies with high confidence."""
        if not self.has_confidence_intervals:
            return []

        return [
            a
            for a in self.anomalies
            if a.confidence_interval and a.confidence_interval.level >= min_level
        ]

    def add_metadata(self, key: str, value: Any) -> None:
        """Add metadata to the result."""
        self.metadata[key] = value

    def summary(self) -> dict[str, Any]:
        """Get summary of detection results."""
        summary_dict = {
            "id": str(self.id),
            "detector_id": str(self.detector_id),
            "dataset_id": str(self.dataset_id),
            "timestamp": self.timestamp.isoformat(),
            "n_samples": self.n_samples,
            "n_anomalies": self.n_anomalies,
            "anomaly_rate": self.anomaly_rate,
            "threshold": self.threshold,
            "score_statistics": self.score_statistics,
            "has_confidence_intervals": self.has_confidence_intervals,
        }

        if self.execution_time_ms is not None:
            summary_dict["execution_time_ms"] = self.execution_time_ms

        if self.metadata:
            summary_dict["metadata"] = self.metadata

        return summary_dict

    def __repr__(self) -> str:
        """Developer representation."""
        return (
            f"DetectionResult(id={self.id}, n_samples={self.n_samples}, "
            f"n_anomalies={self.n_anomalies}, anomaly_rate={self.anomaly_rate:.2%})"
        )
