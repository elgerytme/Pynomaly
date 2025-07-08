"""Anomaly category value object."""

from enum import Enum


class AnomalyCategory(str, Enum):
    """Enumeration for different categories of anomalies."""

    STATISTICAL = "statistical"
    THRESHOLD = "threshold"
    CLUSTERING = "clustering"
    DISTANCE = "distance"
    DENSITY = "density"
    NEURAL = "neural"
    ENSEMBLE = "ensemble"

    @classmethod
    def get_default(cls) -> "AnomalyCategory":
        """Get default anomaly category."""
        return cls.STATISTICAL
