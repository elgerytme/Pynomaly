"""Anomaly type enumeration."""

from enum import Enum


class AnomalyType(Enum):
    """Enumeration for different types of anomalies."""
    
    UNKNOWN = "unknown"
    POINT = "point"
    CONTEXTUAL = "contextual"
    COLLECTIVE = "collective"
    SEASONAL = "seasonal"
    TREND = "trend"
    OUTLIER = "outlier"
    DRIFT = "drift"
    CONCEPT_DRIFT = "concept_drift"
    DATA_QUALITY = "data_quality"
    PERFORMANCE = "performance"
    SECURITY = "security"
    BUSINESS_RULE = "business_rule"
    STATISTICAL = "statistical"
    PATTERN = "pattern"
    BEHAVIORAL = "behavioral"
    TEMPORAL = "temporal"
    SPATIAL = "spatial"
    NETWORK = "network"
    SYSTEM = "system"
    
    @classmethod
    def get_default(cls) -> "AnomalyType":
        """Get the default anomaly type."""
        return cls.UNKNOWN
