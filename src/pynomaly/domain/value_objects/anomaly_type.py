"""Anomaly type value object."""

from enum import Enum


class AnomalyType(str, Enum):
    """Enumeration for different types of anomalies."""

    POINT = "point"
    CONTEXTUAL = "contextual"
    COLLECTIVE = "collective"
    GLOBAL = "global"
    LOCAL = "local"

    def __str__(self) -> str:
        """Return the string value of the enum."""
        return self.value

    @classmethod
    def get_default(cls) -> "AnomalyType":
        """Get default anomaly type."""
        return cls.POINT
