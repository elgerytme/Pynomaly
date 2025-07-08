"""Severity level enumeration."""

from enum import Enum


class SeverityLevel(Enum):
    """Enumeration for different severity levels of anomalies."""
    
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"
    
    def __str__(self) -> str:
        """Return the string representation of the severity level."""
        return self.value
    
    @classmethod
    def from_score(cls, score: float) -> "SeverityLevel":
        """Convert a score to a severity level.
        
        Args:
            score: Anomaly score (0.0 to 1.0)
            
        Returns:
            Corresponding severity level
        """
        if score > 0.9:
            return cls.CRITICAL
        elif score > 0.7:
            return cls.HIGH
        elif score > 0.5:
            return cls.MEDIUM
        else:
            return cls.LOW
