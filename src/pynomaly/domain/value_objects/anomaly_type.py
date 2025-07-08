"""Anomaly type value objects."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Any, Dict


class AnomalyTypeEnum(Enum):
    """Enumeration of anomaly types."""
    
    OUTLIER = "outlier"
    NOVELTY = "novelty"
    DRIFT = "drift"
    SEASONAL = "seasonal"
    TREND = "trend"
    CONTEXTUAL = "contextual"
    COLLECTIVE = "collective"
    POINT = "point"
    UNKNOWN = "unknown"


class AnomalyCategoryEnum(Enum):
    """Enumeration of anomaly categories."""
    
    STATISTICAL = "statistical"
    BEHAVIORAL = "behavioral"
    TEMPORAL = "temporal"
    SPATIAL = "spatial"
    NETWORK = "network"
    SECURITY = "security"
    BUSINESS = "business"
    TECHNICAL = "technical"
    UNKNOWN = "unknown"


class SeverityLevel(Enum):
    """Severity levels for anomalies."""
    
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


@dataclass(frozen=True)
class AnomalyType:
    """Value object representing an anomaly type."""
    
    type_name: AnomalyTypeEnum
    description: str = ""
    metadata: Dict[str, Any] = None
    
    def __post_init__(self):
        """Initialize metadata if None."""
        if self.metadata is None:
            object.__setattr__(self, 'metadata', {})
    
    @classmethod
    def get_default(cls) -> AnomalyType:
        """Get default anomaly type."""
        return cls(
            type_name=AnomalyTypeEnum.UNKNOWN,
            description="Unknown anomaly type"
        )
    
    @classmethod
    def outlier(cls) -> AnomalyType:
        """Create outlier anomaly type."""
        return cls(
            type_name=AnomalyTypeEnum.OUTLIER,
            description="Data point that deviates significantly from the norm"
        )
    
    @classmethod
    def novelty(cls) -> AnomalyType:
        """Create novelty anomaly type."""
        return cls(
            type_name=AnomalyTypeEnum.NOVELTY,
            description="Previously unseen pattern or behavior"
        )
    
    @classmethod
    def drift(cls) -> AnomalyType:
        """Create drift anomaly type."""
        return cls(
            type_name=AnomalyTypeEnum.DRIFT,
            description="Gradual change in data distribution"
        )
    
    @property
    def is_unknown(self) -> bool:
        """Check if anomaly type is unknown."""
        return self.type_name == AnomalyTypeEnum.UNKNOWN
    
    def __str__(self) -> str:
        """String representation."""
        return f"{self.type_name.value}"


@dataclass(frozen=True)
class AnomalyCategory:
    """Value object representing an anomaly category."""
    
    category_name: AnomalyCategoryEnum
    description: str = ""
    metadata: Dict[str, Any] = None
    
    def __post_init__(self):
        """Initialize metadata if None."""
        if self.metadata is None:
            object.__setattr__(self, 'metadata', {})
    
    @classmethod
    def get_default(cls) -> AnomalyCategory:
        """Get default anomaly category."""
        return cls(
            category_name=AnomalyCategoryEnum.UNKNOWN,
            description="Unknown anomaly category"
        )
    
    @classmethod
    def statistical(cls) -> AnomalyCategory:
        """Create statistical anomaly category."""
        return cls(
            category_name=AnomalyCategoryEnum.STATISTICAL,
            description="Statistical deviation from expected patterns"
        )
    
    @classmethod
    def behavioral(cls) -> AnomalyCategory:
        """Create behavioral anomaly category."""
        return cls(
            category_name=AnomalyCategoryEnum.BEHAVIORAL,
            description="Unusual behavior patterns"
        )
    
    @classmethod
    def temporal(cls) -> AnomalyCategory:
        """Create temporal anomaly category."""
        return cls(
            category_name=AnomalyCategoryEnum.TEMPORAL,
            description="Time-based anomalies"
        )
    
    @property
    def is_unknown(self) -> bool:
        """Check if anomaly category is unknown."""
        return self.category_name == AnomalyCategoryEnum.UNKNOWN
    
    def __str__(self) -> str:
        """String representation."""
        return f"{self.category_name.value}"


@dataclass(frozen=True)
class SeverityScore:
    """Value object representing anomaly severity."""
    
    severity_level: SeverityLevel
    numeric_score: float
    description: str = ""
    metadata: Dict[str, Any] = None
    
    def __post_init__(self):
        """Validate and initialize."""
        if not 0.0 <= self.numeric_score <= 1.0:
            raise ValueError("Numeric score must be between 0.0 and 1.0")
        
        if self.metadata is None:
            object.__setattr__(self, 'metadata', {})
    
    @classmethod
    def create_minimal(cls) -> SeverityScore:
        """Create minimal severity score."""
        return cls(
            severity_level=SeverityLevel.LOW,
            numeric_score=0.0,
            description="Minimal severity"
        )
    
    @classmethod
    def low(cls, score: float = 0.25) -> SeverityScore:
        """Create low severity score."""
        return cls(
            severity_level=SeverityLevel.LOW,
            numeric_score=score,
            description="Low severity anomaly"
        )
    
    @classmethod
    def medium(cls, score: float = 0.5) -> SeverityScore:
        """Create medium severity score."""
        return cls(
            severity_level=SeverityLevel.MEDIUM,
            numeric_score=score,
            description="Medium severity anomaly"
        )
    
    @classmethod
    def high(cls, score: float = 0.75) -> SeverityScore:
        """Create high severity score."""
        return cls(
            severity_level=SeverityLevel.HIGH,
            numeric_score=score,
            description="High severity anomaly"
        )
    
    @classmethod
    def critical(cls, score: float = 1.0) -> SeverityScore:
        """Create critical severity score."""
        return cls(
            severity_level=SeverityLevel.CRITICAL,
            numeric_score=score,
            description="Critical severity anomaly"
        )
    
    @property
    def is_critical(self) -> bool:
        """Check if severity is critical."""
        return self.severity_level == SeverityLevel.CRITICAL
    
    @property
    def is_high_or_above(self) -> bool:
        """Check if severity is high or above."""
        return self.severity_level in [SeverityLevel.HIGH, SeverityLevel.CRITICAL]
    
    def __str__(self) -> str:
        """String representation."""
        return f"{self.severity_level.value} ({self.numeric_score:.2f})"
