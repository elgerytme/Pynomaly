"""Domain entity for anomaly point representation."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import Any
from uuid import UUID, uuid4

import numpy as np

from monorepo.domain.value_objects import AnomalyScore, SeverityScore


@dataclass
class AnomalyPoint:
    """Represents a single anomaly point in a data stream.

    This entity represents a specific point in time and space where an anomaly
    was detected, with associated metadata and scoring information.
    """

    point_id: UUID = field(default_factory=uuid4)
    timestamp: datetime = field(default_factory=datetime.utcnow)
    coordinates: dict[str, float] = field(default_factory=dict)
    feature_values: dict[str, Any] = field(default_factory=dict)
    anomaly_score: AnomalyScore = field(default_factory=lambda: AnomalyScore(0.0))
    severity_score: SeverityScore = field(default_factory=SeverityScore.create_minimal)
    detector_name: str = "unknown"
    algorithm_name: str = "unknown"
    confidence: float = 0.0
    context_metadata: dict[str, Any] = field(default_factory=dict)
    raw_data_point: dict[str, Any] | None = None

    def __post_init__(self):
        """Validate anomaly point data."""
        if not (0.0 <= self.confidence <= 1.0):
            raise ValueError("Confidence must be between 0.0 and 1.0")
        if not self.feature_values:
            raise ValueError("Feature values cannot be empty")

    def get_spatial_distance(self, other: AnomalyPoint) -> float:
        """Calculate spatial distance to another anomaly point."""
        if not self.coordinates or not other.coordinates:
            return float("inf")

        # Calculate Euclidean distance using common coordinates
        common_coords = set(self.coordinates.keys()) & set(other.coordinates.keys())
        if not common_coords:
            return float("inf")

        distance_sq = sum(
            (self.coordinates[coord] - other.coordinates[coord]) ** 2
            for coord in common_coords
        )
        return np.sqrt(distance_sq)

    def get_temporal_distance(self, other: AnomalyPoint) -> float:
        """Calculate temporal distance to another anomaly point in seconds."""
        time_diff = abs((self.timestamp - other.timestamp).total_seconds())
        return time_diff

    def is_similar_to(
        self,
        other: AnomalyPoint,
        spatial_threshold: float = 1.0,
        temporal_threshold: float = 60.0,
    ) -> bool:
        """Check if this anomaly point is similar to another.

        Args:
            other: Another anomaly point to compare with
            spatial_threshold: Maximum spatial distance for similarity
            temporal_threshold: Maximum temporal distance in seconds for similarity

        Returns:
            True if points are similar within thresholds
        """
        spatial_dist = self.get_spatial_distance(other)
        temporal_dist = self.get_temporal_distance(other)

        return spatial_dist <= spatial_threshold and temporal_dist <= temporal_threshold

    def add_context(self, key: str, value: Any) -> None:
        """Add context metadata to the anomaly point."""
        self.context_metadata[key] = value

    def get_feature_vector(self) -> np.ndarray:
        """Get feature values as a numpy array."""
        if not self.feature_values:
            return np.array([])

        # Extract numeric values only
        numeric_values = []
        for value in self.feature_values.values():
            if isinstance(value, (int, float)):
                numeric_values.append(float(value))
            elif isinstance(value, np.number):
                numeric_values.append(float(value))

        return np.array(numeric_values)

    def get_composite_score(self) -> float:
        """Calculate composite score combining anomaly score and severity."""
        return (self.anomaly_score.value + self.severity_score.value) / 2.0

    def is_high_severity(self) -> bool:
        """Check if this is a high severity anomaly."""
        return self.severity_score.value > 0.7

    def is_high_confidence(self) -> bool:
        """Check if this anomaly has high confidence."""
        return self.confidence > 0.8

    def to_dict(self) -> dict[str, Any]:
        """Convert anomaly point to dictionary representation."""
        return {
            "point_id": str(self.point_id),
            "timestamp": self.timestamp.isoformat(),
            "coordinates": self.coordinates,
            "feature_values": self.feature_values,
            "anomaly_score": self.anomaly_score.value,
            "severity_score": self.severity_score.value,
            "detector_name": self.detector_name,
            "algorithm_name": self.algorithm_name,
            "confidence": self.confidence,
            "context_metadata": self.context_metadata,
            "raw_data_point": self.raw_data_point,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> AnomalyPoint:
        """Create anomaly point from dictionary representation."""
        return cls(
            point_id=UUID(data["point_id"]) if "point_id" in data else uuid4(),
            timestamp=datetime.fromisoformat(data["timestamp"])
            if "timestamp" in data
            else datetime.utcnow(),
            coordinates=data.get("coordinates", {}),
            feature_values=data.get("feature_values", {}),
            anomaly_score=AnomalyScore(data.get("anomaly_score", 0.0)),
            severity_score=SeverityScore.from_score(data.get("severity_score", 0.0)),
            detector_name=data.get("detector_name", "unknown"),
            algorithm_name=data.get("algorithm_name", "unknown"),
            confidence=data.get("confidence", 0.0),
            context_metadata=data.get("context_metadata", {}),
            raw_data_point=data.get("raw_data_point"),
        )
