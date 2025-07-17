"""
Anomaly Score value object for machine learning domain.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class AnomalyScore:
    """
    Represents an anomaly score value object.
    
    Attributes:
        value: The anomaly score (0.0 to 1.0)
        method: Method used to calculate the score
        confidence: Confidence in the score (0.0 to 1.0)
        metadata: Additional metadata about the score
    """
    
    value: float
    method: str | None = None
    confidence: float | None = None
    metadata: dict[str, Any] | None = None
    
    def __post_init__(self) -> None:
        """Validate the anomaly score."""
        if not 0.0 <= self.value <= 1.0:
            raise ValueError(f"Anomaly score must be between 0.0 and 1.0, got {self.value}")
        
        if self.confidence is not None and not 0.0 <= self.confidence <= 1.0:
            raise ValueError(f"Confidence must be between 0.0 and 1.0, got {self.confidence}")
    
    @classmethod
    def create(
        cls,
        value: float,
        method: str | None = None,
        confidence: float | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> AnomalyScore:
        """Create a new anomaly score."""
        return cls(
            value=value,
            method=method,
            confidence=confidence,
            metadata=metadata or {},
        )
    
    def is_anomaly(self, threshold: float = 0.5) -> bool:
        """Check if the score indicates an anomaly."""
        return self.value > threshold
    
    def is_high_confidence(self, threshold: float = 0.8) -> bool:
        """Check if the score has high confidence."""
        return self.confidence is not None and self.confidence >= threshold
    
    def is_low_score(self, threshold: float = 0.2) -> bool:
        """Check if the score is low (likely normal)."""
        return self.value < threshold
    
    def is_high_score(self, threshold: float = 0.8) -> bool:
        """Check if the score is high (likely anomaly)."""
        return self.value > threshold
    
    def is_uncertain(self, lower: float = 0.3, upper: float = 0.7) -> bool:
        """Check if the score is in the uncertain range."""
        return lower <= self.value <= upper
    
    def with_confidence(self, confidence: float) -> AnomalyScore:
        """Create a new score with updated confidence."""
        return AnomalyScore(
            value=self.value,
            method=self.method,
            confidence=confidence,
            metadata=self.metadata,
        )
    
    def with_metadata(self, metadata: dict[str, Any]) -> AnomalyScore:
        """Create a new score with updated metadata."""
        return AnomalyScore(
            value=self.value,
            method=self.method,
            confidence=self.confidence,
            metadata=metadata,
        )
    
    def __str__(self) -> str:
        """String representation of the score."""
        parts = [f"score={self.value:.3f}"]
        if self.method:
            parts.append(f"method={self.method}")
        if self.confidence is not None:
            parts.append(f"confidence={self.confidence:.3f}")
        return f"AnomalyScore({', '.join(parts)})"
    
    def __lt__(self, other: AnomalyScore) -> bool:
        """Compare scores for sorting."""
        return self.value < other.value
    
    def __le__(self, other: AnomalyScore) -> bool:
        """Compare scores for sorting."""
        return self.value <= other.value
    
    def __gt__(self, other: AnomalyScore) -> bool:
        """Compare scores for sorting."""
        return self.value > other.value
    
    def __ge__(self, other: AnomalyScore) -> bool:
        """Compare scores for sorting."""
        return self.value >= other.value