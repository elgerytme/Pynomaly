"""Anomaly category enumeration."""

from enum import Enum


class AnomalyCategory(Enum):
    """Enumeration for different categories of anomalies."""
    
    STATISTICAL = "statistical"
    BEHAVIORAL = "behavioral"
    CONTEXTUAL = "contextual"
    COLLECTIVE = "collective"
    TEMPORAL = "temporal"
    SPATIAL = "spatial"
    NETWORK = "network"
    SYSTEM = "system"
    BUSINESS_RULE = "business_rule"
    DATA_QUALITY = "data_quality"
    PERFORMANCE = "performance"
    SECURITY = "security"
    PATTERN = "pattern"
    DRIFT = "drift"
    CONCEPT_DRIFT = "concept_drift"
    SEASONAL = "seasonal"
    TREND = "trend"
    OUTLIER = "outlier"
    POINT = "point"
    UNKNOWN = "unknown"
    
    @classmethod
    def get_default(cls) -> "AnomalyCategory":
        """Get the default anomaly category."""
        return cls.UNKNOWN
