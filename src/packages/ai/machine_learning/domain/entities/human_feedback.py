"""
Human Feedback entity for machine learning domain.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from enum import Enum
from typing import Any
from uuid import UUID, uuid4


class FeedbackType(Enum):
    """Types of feedback that can be provided."""
    BINARY_CLASSIFICATION = "binary_classification"
    CONFIDENCE_RATING = "confidence_rating"
    SCORE_CORRECTION = "score_correction"
    EXPLANATION = "explanation"
    FEATURE_IMPORTANCE = "feature_importance"
    RANKING = "ranking"


class FeedbackConfidence(Enum):
    """Confidence levels for feedback."""
    VERY_LOW = "very_low"
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    VERY_HIGH = "very_high"


@dataclass
class HumanFeedback:
    """
    Represents human feedback on a detection result.
    
    Attributes:
        feedback_id: Unique feedback identifier
        session_id: ID of the active learning session
        sample_id: ID of the sample being annotated
        annotator_id: ID of the human annotator
        feedback_type: Type of feedback being provided
        feedback_value: The actual feedback value
        confidence: Annotator's confidence in the feedback
        original_prediction: Original model prediction for comparison
        time_spent_seconds: Time spent on annotation
        created_at: Timestamp when feedback was created
        metadata: Additional feedback metadata
        quality_score: Quality score for this feedback
        is_validated: Whether feedback has been validated
    """
    
    feedback_id: UUID
    session_id: UUID
    sample_id: str
    annotator_id: str
    feedback_type: FeedbackType
    feedback_value: bool | float | str | dict[str, Any]
    confidence: FeedbackConfidence
    original_prediction: float | None = None
    time_spent_seconds: float | None = None
    created_at: datetime | None = None
    metadata: dict[str, Any] | None = None
    quality_score: float | None = None
    is_validated: bool = False
    
    def __post_init__(self) -> None:
        """Initialize default values."""
        if self.created_at is None:
            self.created_at = datetime.now()
        if self.metadata is None:
            self.metadata = {}
    
    @classmethod
    def create(
        cls,
        session_id: UUID,
        sample_id: str,
        annotator_id: str,
        feedback_type: FeedbackType,
        feedback_value: bool | float | str | dict[str, Any],
        confidence: FeedbackConfidence,
        original_prediction: float | None = None,
        time_spent_seconds: float | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> HumanFeedback:
        """Create a new human feedback."""
        return cls(
            feedback_id=uuid4(),
            session_id=session_id,
            sample_id=sample_id,
            annotator_id=annotator_id,
            feedback_type=feedback_type,
            feedback_value=feedback_value,
            confidence=confidence,
            original_prediction=original_prediction,
            time_spent_seconds=time_spent_seconds,
            created_at=datetime.now(),
            metadata=metadata or {},
        )
    
    def is_high_confidence(self) -> bool:
        """Check if feedback has high confidence."""
        return self.confidence in [FeedbackConfidence.HIGH, FeedbackConfidence.VERY_HIGH]
    
    def is_binary_feedback(self) -> bool:
        """Check if feedback is binary classification."""
        return self.feedback_type == FeedbackType.BINARY_CLASSIFICATION
    
    def is_score_correction(self) -> bool:
        """Check if feedback is a score correction."""
        return self.feedback_type == FeedbackType.SCORE_CORRECTION
    
    def get_corrected_score(self) -> float | None:
        """Get the corrected score if this is a score correction."""
        if self.is_score_correction() and isinstance(self.feedback_value, (int, float)):
            return float(self.feedback_value)
        return None
    
    def get_binary_value(self) -> bool | None:
        """Get the binary value if this is binary feedback."""
        if self.is_binary_feedback() and isinstance(self.feedback_value, bool):
            return self.feedback_value
        return None
    
    def validate(self) -> None:
        """Mark feedback as validated."""
        self.is_validated = True
    
    def set_quality_score(self, score: float) -> None:
        """Set the quality score for this feedback."""
        self.quality_score = max(0.0, min(1.0, score))
    
    def agrees_with_prediction(self, threshold: float = 0.5) -> bool | None:
        """Check if feedback agrees with original prediction."""
        if self.original_prediction is None:
            return None
        
        if self.is_binary_feedback():
            binary_value = self.get_binary_value()
            if binary_value is not None:
                predicted_binary = self.original_prediction > threshold
                return binary_value == predicted_binary
        
        return None