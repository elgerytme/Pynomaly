"""Value object for neuro-symbolic reasoning results."""

from typing import Any, List, Optional, Dict
from dataclasses import dataclass
from datetime import datetime


@dataclass(frozen=True)
class ReasoningResult:
    """
    Immutable value object representing the result of neuro-symbolic reasoning.
    Contains both neural predictions and symbolic explanations.
    """
    
    prediction: Any
    confidence: float
    symbolic_explanation: List[str]
    neural_features: Optional[Dict[str, float]] = None
    reasoning_trace: Optional[List[str]] = None
    timestamp: Optional[datetime] = None
    
    def __post_init__(self):
        if self.timestamp is None:
            object.__setattr__(self, 'timestamp', datetime.now())
    
    @classmethod
    def create(
        cls,
        prediction: Any,
        confidence: float,
        symbolic_explanation: List[str],
        neural_features: Optional[Dict[str, float]] = None,
        reasoning_trace: Optional[List[str]] = None
    ) -> "ReasoningResult":
        """Create a new reasoning result."""
        return cls(
            prediction=prediction,
            confidence=confidence,
            symbolic_explanation=symbolic_explanation,
            neural_features=neural_features,
            reasoning_trace=reasoning_trace,
            timestamp=datetime.now()
        )
    
    def is_confident(self, threshold: float = 0.8) -> bool:
        """Check if the result meets a confidence threshold."""
        return self.confidence >= threshold
    
    def get_explanation_summary(self) -> str:
        """Get a human-readable explanation summary."""
        if not self.symbolic_explanation:
            return "No symbolic explanation available"
        
        return " -> ".join(self.symbolic_explanation)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary representation."""
        return {
            "prediction": self.prediction,
            "confidence": self.confidence,
            "symbolic_explanation": self.symbolic_explanation,
            "neural_features": self.neural_features,
            "reasoning_trace": self.reasoning_trace,
            "timestamp": self.timestamp.isoformat() if self.timestamp else None
        }