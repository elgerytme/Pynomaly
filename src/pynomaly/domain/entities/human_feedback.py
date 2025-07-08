"""
Human feedback entity for active learning.

This module defines the HumanFeedback entity that captures human expert
annotations and corrections for active learning scenarios.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from enum import Enum

from pynomaly.domain.value_objects.anomaly_score import AnomalyScore


class FeedbackType(Enum):
    """Type of human feedback provided."""

    BINARY_CLASSIFICATION = "binary_classification"  # Normal/Anomaly
    CONFIDENCE_RATING = "confidence_rating"  # Confidence in prediction
    SCORE_CORRECTION = "score_correction"  # Corrected anomaly score
    EXPLANATION = "explanation"  # Text explanation
    FEATURE_IMPORTANCE = "feature_importance"  # Important features


class FeedbackConfidence(Enum):
    """Human annotator's confidence in their feedback."""

    LOW = "low"  # Uncertain about the label
    MEDIUM = "medium"  # Moderately confident
    HIGH = "high"  # Very confident
    EXPERT = "expert"  # Domain expert level confidence


@dataclass
class HumanFeedback:
    """
    Represents human expert feedback for active learning.

    This entity captures various types of human annotations that can be
    used to improve model performance through active learning.
    """

    feedback_id: str
    sample_id: str
    annotator_id: str
    feedback_type: FeedbackType
    feedback_value: bool | float | str | dict
    confidence: FeedbackConfidence
    timestamp: datetime
    metadata: dict
    original_prediction: AnomalyScore | None = None
    session_id: str | None = None
    time_spent_seconds: float | None = None

    def __post_init__(self) -> None:
        """Validate feedback after initialization."""
        if not self.feedback_id:
            raise ValueError("Feedback ID cannot be empty")

        if not self.sample_id:
            raise ValueError("Sample ID cannot be empty")

        if not self.annotator_id:
            raise ValueError("Annotator ID cannot be empty")

        self._validate_feedback_value()

    def _validate_feedback_value(self) -> None:
        """Validate feedback value based on feedback type."""
        if self.feedback_type == FeedbackType.BINARY_CLASSIFICATION:
            if not isinstance(self.feedback_value, bool):
                raise ValueError("Binary classification feedback must be boolean")

        elif self.feedback_type == FeedbackType.CONFIDENCE_RATING:
            if not isinstance(self.feedback_value, (int, float)):
                raise ValueError("Confidence rating must be numeric")
            if not 0.0 <= self.feedback_value <= 1.0:
                raise ValueError("Confidence rating must be between 0 and 1")

        elif self.feedback_type == FeedbackType.SCORE_CORRECTION:
            if not isinstance(self.feedback_value, (int, float)):
                raise ValueError("Score correction must be numeric")
            if not 0.0 <= self.feedback_value <= 1.0:
                raise ValueError("Score correction must be between 0 and 1")

        elif self.feedback_type == FeedbackType.EXPLANATION:
            if not isinstance(self.feedback_value, str):
                raise ValueError("Explanation feedback must be string")
            if not self.feedback_value.strip():
                raise ValueError("Explanation cannot be empty")

        elif self.feedback_type == FeedbackType.FEATURE_IMPORTANCE:
            if not isinstance(self.feedback_value, dict):
                raise ValueError("Feature importance must be dictionary")
            if not self.feedback_value:
                raise ValueError("Feature importance dictionary cannot be empty")

    def is_correction(self) -> bool:
        """Check if feedback represents a correction to the original prediction."""
        if self.original_prediction is None:
            return False

        if self.feedback_type == FeedbackType.BINARY_CLASSIFICATION:
            original_binary = self.original_prediction.value > 0.5
            return original_binary != self.feedback_value

        elif self.feedback_type == FeedbackType.SCORE_CORRECTION:
            # Consider correction if difference is significant
            threshold = 0.1
            return abs(self.original_prediction.value - self.feedback_value) > threshold

        return False

    def get_corrected_score(self) -> AnomalyScore | None:
        """Get the corrected anomaly score based on feedback."""
        if self.feedback_type == FeedbackType.SCORE_CORRECTION:
            return AnomalyScore(value=float(self.feedback_value))

        elif self.feedback_type == FeedbackType.BINARY_CLASSIFICATION:
            # Convert binary feedback to score
            score_value = 0.9 if self.feedback_value else 0.1
            return AnomalyScore(value=score_value)

        return None

    def get_feedback_weight(self) -> float:
        """Calculate weight for this feedback based on confidence and time spent."""
        # Base weight from confidence level
        confidence_weights = {
            FeedbackConfidence.LOW: 0.3,
            FeedbackConfidence.MEDIUM: 0.6,
            FeedbackConfidence.HIGH: 0.9,
            FeedbackConfidence.EXPERT: 1.0,
        }

        base_weight = confidence_weights[self.confidence]

        # Adjust weight based on time spent (if available)
        if self.time_spent_seconds is not None:
            # More time spent generally indicates more careful consideration
            time_factor = min(
                1.0, self.time_spent_seconds / 30.0
            )  # 30 seconds as baseline
            base_weight *= 0.7 + 0.3 * time_factor  # Scale between 0.7 and 1.0

        return base_weight

    def to_dict(self) -> dict:
        """Convert feedback to dictionary representation."""
        return {
            "feedback_id": self.feedback_id,
            "sample_id": self.sample_id,
            "annotator_id": self.annotator_id,
            "feedback_type": self.feedback_type.value,
            "feedback_value": self.feedback_value,
            "confidence": self.confidence.value,
            "timestamp": self.timestamp.isoformat(),
            "metadata": self.metadata,
            "original_prediction": (
                self.original_prediction.value if self.original_prediction else None
            ),
            "session_id": self.session_id,
            "time_spent_seconds": self.time_spent_seconds,
            "is_correction": self.is_correction(),
            "feedback_weight": self.get_feedback_weight(),
        }

    @classmethod
    def create_binary_feedback(
        cls,
        feedback_id: str,
        sample_id: str,
        annotator_id: str,
        is_anomaly: bool,
        confidence: FeedbackConfidence,
        original_prediction: AnomalyScore | None = None,
        session_id: str | None = None,
        time_spent_seconds: float | None = None,
        metadata: dict | None = None,
    ) -> HumanFeedback:
        """Create binary classification feedback."""
        return cls(
            feedback_id=feedback_id,
            sample_id=sample_id,
            annotator_id=annotator_id,
            feedback_type=FeedbackType.BINARY_CLASSIFICATION,
            feedback_value=is_anomaly,
            confidence=confidence,
            timestamp=datetime.now(),
            metadata=metadata or {},
            original_prediction=original_prediction,
            session_id=session_id,
            time_spent_seconds=time_spent_seconds,
        )

    @classmethod
    def create_score_correction(
        cls,
        feedback_id: str,
        sample_id: str,
        annotator_id: str,
        corrected_score: float,
        confidence: FeedbackConfidence,
        original_prediction: AnomalyScore | None = None,
        session_id: str | None = None,
        time_spent_seconds: float | None = None,
        metadata: dict | None = None,
    ) -> HumanFeedback:
        """Create score correction feedback."""
        return cls(
            feedback_id=feedback_id,
            sample_id=sample_id,
            annotator_id=annotator_id,
            feedback_type=FeedbackType.SCORE_CORRECTION,
            feedback_value=corrected_score,
            confidence=confidence,
            timestamp=datetime.now(),
            metadata=metadata or {},
            original_prediction=original_prediction,
            session_id=session_id,
            time_spent_seconds=time_spent_seconds,
        )

    @classmethod
    def create_explanation_feedback(
        cls,
        feedback_id: str,
        sample_id: str,
        annotator_id: str,
        explanation: str,
        confidence: FeedbackConfidence,
        session_id: str | None = None,
        time_spent_seconds: float | None = None,
        metadata: dict | None = None,
    ) -> HumanFeedback:
        """Create explanation feedback."""
        return cls(
            feedback_id=feedback_id,
            sample_id=sample_id,
            annotator_id=annotator_id,
            feedback_type=FeedbackType.EXPLANATION,
            feedback_value=explanation,
            confidence=confidence,
            timestamp=datetime.now(),
            metadata=metadata or {},
            session_id=session_id,
            time_spent_seconds=time_spent_seconds,
        )

    def __str__(self) -> str:
        """String representation of feedback."""
        return (
            f"HumanFeedback(id={self.feedback_id}, "
            f"type={self.feedback_type.value}, "
            f"value={self.feedback_value}, "
            f"confidence={self.confidence.value})"
        )
