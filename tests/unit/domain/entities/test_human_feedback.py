"""
Unit tests for human feedback entity.

Tests the HumanFeedback entity functionality including
validation, feedback types, and quality assessment.
"""

from datetime import datetime

import pytest

from pynomaly.domain.entities.human_feedback import (
    FeedbackConfidence,
    FeedbackType,
    HumanFeedback,
)
from pynomaly.domain.value_objects.anomaly_score import AnomalyScore


class TestHumanFeedback:
    """Test cases for HumanFeedback entity."""

    def test_create_binary_feedback(self):
        """Test creating binary classification feedback."""
        original_prediction = AnomalyScore(value=0.3)

        feedback = HumanFeedback.create_binary_feedback(
            feedback_id="fb_001",
            sample_id="sample_123",
            annotator_id="annotator_456",
            is_anomaly=True,
            confidence=FeedbackConfidence.HIGH,
            original_prediction=original_prediction,
            session_id="session_789",
            time_spent_seconds=45.5,
            metadata={"source": "expert_review"},
        )

        assert feedback.feedback_id == "fb_001"
        assert feedback.sample_id == "sample_123"
        assert feedback.annotator_id == "annotator_456"
        assert feedback.feedback_type == FeedbackType.BINARY_CLASSIFICATION
        assert feedback.feedback_value is True
        assert feedback.confidence == FeedbackConfidence.HIGH
        assert feedback.original_prediction == original_prediction
        assert feedback.session_id == "session_789"
        assert feedback.time_spent_seconds == 45.5
        assert feedback.metadata["source"] == "expert_review"

    def test_create_score_correction(self):
        """Test creating score correction feedback."""
        original_prediction = AnomalyScore(value=0.2)

        feedback = HumanFeedback.create_score_correction(
            feedback_id="fb_002",
            sample_id="sample_124",
            annotator_id="annotator_456",
            corrected_score=0.8,
            confidence=FeedbackConfidence.EXPERT,
            original_prediction=original_prediction,
            time_spent_seconds=120.0,
        )

        assert feedback.feedback_type == FeedbackType.SCORE_CORRECTION
        assert feedback.feedback_value == 0.8
        assert feedback.confidence == FeedbackConfidence.EXPERT
        assert feedback.is_correction()

    def test_create_explanation_feedback(self):
        """Test creating explanation feedback."""
        feedback = HumanFeedback.create_explanation_feedback(
            feedback_id="fb_003",
            sample_id="sample_125",
            annotator_id="annotator_456",
            explanation="This pattern indicates network intrusion based on traffic anomalies",
            confidence=FeedbackConfidence.HIGH,
            time_spent_seconds=180.0,
        )

        assert feedback.feedback_type == FeedbackType.EXPLANATION
        assert "network intrusion" in feedback.feedback_value
        assert feedback.confidence == FeedbackConfidence.HIGH

    def test_feedback_validation_empty_fields(self):
        """Test validation of required fields."""
        with pytest.raises(ValueError, match="Feedback ID cannot be empty"):
            HumanFeedback(
                feedback_id="",
                sample_id="sample_123",
                annotator_id="annotator_456",
                feedback_type=FeedbackType.BINARY_CLASSIFICATION,
                feedback_value=True,
                confidence=FeedbackConfidence.HIGH,
                timestamp=datetime.now(),
                metadata={},
            )

        with pytest.raises(ValueError, match="Sample ID cannot be empty"):
            HumanFeedback(
                feedback_id="fb_001",
                sample_id="",
                annotator_id="annotator_456",
                feedback_type=FeedbackType.BINARY_CLASSIFICATION,
                feedback_value=True,
                confidence=FeedbackConfidence.HIGH,
                timestamp=datetime.now(),
                metadata={},
            )

        with pytest.raises(ValueError, match="Annotator ID cannot be empty"):
            HumanFeedback(
                feedback_id="fb_001",
                sample_id="sample_123",
                annotator_id="",
                feedback_type=FeedbackType.BINARY_CLASSIFICATION,
                feedback_value=True,
                confidence=FeedbackConfidence.HIGH,
                timestamp=datetime.now(),
                metadata={},
            )

    def test_binary_classification_validation(self):
        """Test validation of binary classification feedback."""
        # Valid boolean value
        feedback = HumanFeedback(
            feedback_id="fb_001",
            sample_id="sample_123",
            annotator_id="annotator_456",
            feedback_type=FeedbackType.BINARY_CLASSIFICATION,
            feedback_value=True,
            confidence=FeedbackConfidence.HIGH,
            timestamp=datetime.now(),
            metadata={},
        )
        assert feedback.feedback_value is True

        # Invalid non-boolean value
        with pytest.raises(
            ValueError, match="Binary classification feedback must be boolean"
        ):
            HumanFeedback(
                feedback_id="fb_001",
                sample_id="sample_123",
                annotator_id="annotator_456",
                feedback_type=FeedbackType.BINARY_CLASSIFICATION,
                feedback_value="true",  # String instead of boolean
                confidence=FeedbackConfidence.HIGH,
                timestamp=datetime.now(),
                metadata={},
            )

    def test_confidence_rating_validation(self):
        """Test validation of confidence rating feedback."""
        # Valid numeric value in range
        feedback = HumanFeedback(
            feedback_id="fb_001",
            sample_id="sample_123",
            annotator_id="annotator_456",
            feedback_type=FeedbackType.CONFIDENCE_RATING,
            feedback_value=0.75,
            confidence=FeedbackConfidence.HIGH,
            timestamp=datetime.now(),
            metadata={},
        )
        assert feedback.feedback_value == 0.75

        # Invalid non-numeric value
        with pytest.raises(ValueError, match="Confidence rating must be numeric"):
            HumanFeedback(
                feedback_id="fb_001",
                sample_id="sample_123",
                annotator_id="annotator_456",
                feedback_type=FeedbackType.CONFIDENCE_RATING,
                feedback_value="high",
                confidence=FeedbackConfidence.HIGH,
                timestamp=datetime.now(),
                metadata={},
            )

        # Invalid value out of range
        with pytest.raises(
            ValueError, match="Confidence rating must be between 0 and 1"
        ):
            HumanFeedback(
                feedback_id="fb_001",
                sample_id="sample_123",
                annotator_id="annotator_456",
                feedback_type=FeedbackType.CONFIDENCE_RATING,
                feedback_value=1.5,
                confidence=FeedbackConfidence.HIGH,
                timestamp=datetime.now(),
                metadata={},
            )

    def test_score_correction_validation(self):
        """Test validation of score correction feedback."""
        # Valid score correction
        feedback = HumanFeedback(
            feedback_id="fb_001",
            sample_id="sample_123",
            annotator_id="annotator_456",
            feedback_type=FeedbackType.SCORE_CORRECTION,
            feedback_value=0.9,
            confidence=FeedbackConfidence.HIGH,
            timestamp=datetime.now(),
            metadata={},
        )
        assert feedback.feedback_value == 0.9

        # Invalid value out of range
        with pytest.raises(
            ValueError, match="Score correction must be between 0 and 1"
        ):
            HumanFeedback(
                feedback_id="fb_001",
                sample_id="sample_123",
                annotator_id="annotator_456",
                feedback_type=FeedbackType.SCORE_CORRECTION,
                feedback_value=-0.1,
                confidence=FeedbackConfidence.HIGH,
                timestamp=datetime.now(),
                metadata={},
            )

    def test_explanation_validation(self):
        """Test validation of explanation feedback."""
        # Valid explanation
        feedback = HumanFeedback(
            feedback_id="fb_001",
            sample_id="sample_123",
            annotator_id="annotator_456",
            feedback_type=FeedbackType.EXPLANATION,
            feedback_value="This is a detailed explanation",
            confidence=FeedbackConfidence.HIGH,
            timestamp=datetime.now(),
            metadata={},
        )
        assert feedback.feedback_value == "This is a detailed explanation"

        # Invalid non-string value
        with pytest.raises(ValueError, match="Explanation feedback must be string"):
            HumanFeedback(
                feedback_id="fb_001",
                sample_id="sample_123",
                annotator_id="annotator_456",
                feedback_type=FeedbackType.EXPLANATION,
                feedback_value=123,
                confidence=FeedbackConfidence.HIGH,
                timestamp=datetime.now(),
                metadata={},
            )

        # Invalid empty explanation
        with pytest.raises(ValueError, match="Explanation cannot be empty"):
            HumanFeedback(
                feedback_id="fb_001",
                sample_id="sample_123",
                annotator_id="annotator_456",
                feedback_type=FeedbackType.EXPLANATION,
                feedback_value="   ",  # Only whitespace
                confidence=FeedbackConfidence.HIGH,
                timestamp=datetime.now(),
                metadata={},
            )

    def test_feature_importance_validation(self):
        """Test validation of feature importance feedback."""
        # Valid feature importance
        feedback = HumanFeedback(
            feedback_id="fb_001",
            sample_id="sample_123",
            annotator_id="annotator_456",
            feedback_type=FeedbackType.FEATURE_IMPORTANCE,
            feedback_value={"feature1": 0.8, "feature2": 0.3},
            confidence=FeedbackConfidence.HIGH,
            timestamp=datetime.now(),
            metadata={},
        )
        assert feedback.feedback_value == {"feature1": 0.8, "feature2": 0.3}

        # Invalid non-dict value
        with pytest.raises(ValueError, match="Feature importance must be dictionary"):
            HumanFeedback(
                feedback_id="fb_001",
                sample_id="sample_123",
                annotator_id="annotator_456",
                feedback_type=FeedbackType.FEATURE_IMPORTANCE,
                feedback_value="important features",
                confidence=FeedbackConfidence.HIGH,
                timestamp=datetime.now(),
                metadata={},
            )

        # Invalid empty dictionary
        with pytest.raises(
            ValueError, match="Feature importance dictionary cannot be empty"
        ):
            HumanFeedback(
                feedback_id="fb_001",
                sample_id="sample_123",
                annotator_id="annotator_456",
                feedback_type=FeedbackType.FEATURE_IMPORTANCE,
                feedback_value={},
                confidence=FeedbackConfidence.HIGH,
                timestamp=datetime.now(),
                metadata={},
            )

    def test_is_correction_binary(self):
        """Test correction detection for binary feedback."""
        original_prediction = AnomalyScore(value=0.3)

        # Correction: original prediction was low (normal), feedback says anomaly
        correction_feedback = HumanFeedback.create_binary_feedback(
            feedback_id="fb_001",
            sample_id="sample_123",
            annotator_id="annotator_456",
            is_anomaly=True,
            confidence=FeedbackConfidence.HIGH,
            original_prediction=original_prediction,
        )
        assert correction_feedback.is_correction()

        # No correction: original prediction was low (normal), feedback agrees
        agreement_feedback = HumanFeedback.create_binary_feedback(
            feedback_id="fb_002",
            sample_id="sample_124",
            annotator_id="annotator_456",
            is_anomaly=False,
            confidence=FeedbackConfidence.HIGH,
            original_prediction=original_prediction,
        )
        assert not agreement_feedback.is_correction()

    def test_is_correction_score(self):
        """Test correction detection for score feedback."""
        original_prediction = AnomalyScore(value=0.2)

        # Significant correction
        correction_feedback = HumanFeedback.create_score_correction(
            feedback_id="fb_001",
            sample_id="sample_123",
            annotator_id="annotator_456",
            corrected_score=0.8,  # Large difference from 0.2
            confidence=FeedbackConfidence.HIGH,
            original_prediction=original_prediction,
        )
        assert correction_feedback.is_correction()

        # Minor adjustment (not considered correction)
        minor_feedback = HumanFeedback.create_score_correction(
            feedback_id="fb_002",
            sample_id="sample_124",
            annotator_id="annotator_456",
            corrected_score=0.25,  # Small difference from 0.2
            confidence=FeedbackConfidence.HIGH,
            original_prediction=original_prediction,
        )
        assert not minor_feedback.is_correction()

    def test_get_corrected_score(self):
        """Test getting corrected anomaly score from feedback."""
        original_prediction = AnomalyScore(value=0.2)

        # Score correction feedback
        score_feedback = HumanFeedback.create_score_correction(
            feedback_id="fb_001",
            sample_id="sample_123",
            annotator_id="annotator_456",
            corrected_score=0.85,
            confidence=FeedbackConfidence.HIGH,
            original_prediction=original_prediction,
        )
        corrected_score = score_feedback.get_corrected_score()
        assert corrected_score is not None
        assert corrected_score.value == 0.85

        # Binary feedback (converted to score)
        binary_feedback = HumanFeedback.create_binary_feedback(
            feedback_id="fb_002",
            sample_id="sample_124",
            annotator_id="annotator_456",
            is_anomaly=True,
            confidence=FeedbackConfidence.HIGH,
            original_prediction=original_prediction,
        )
        corrected_score = binary_feedback.get_corrected_score()
        assert corrected_score is not None
        assert corrected_score.value == 0.9  # High confidence anomaly

        # Explanation feedback (no corrected score)
        explanation_feedback = HumanFeedback.create_explanation_feedback(
            feedback_id="fb_003",
            sample_id="sample_125",
            annotator_id="annotator_456",
            explanation="Detailed explanation",
            confidence=FeedbackConfidence.HIGH,
        )
        corrected_score = explanation_feedback.get_corrected_score()
        assert corrected_score is None

    def test_get_feedback_weight(self):
        """Test feedback weight calculation."""
        # High confidence feedback
        high_confidence = HumanFeedback.create_binary_feedback(
            feedback_id="fb_001",
            sample_id="sample_123",
            annotator_id="annotator_456",
            is_anomaly=True,
            confidence=FeedbackConfidence.HIGH,
            time_spent_seconds=60.0,
        )
        high_weight = high_confidence.get_feedback_weight()
        assert high_weight > 0.8

        # Low confidence feedback
        low_confidence = HumanFeedback.create_binary_feedback(
            feedback_id="fb_002",
            sample_id="sample_124",
            annotator_id="annotator_456",
            is_anomaly=True,
            confidence=FeedbackConfidence.LOW,
            time_spent_seconds=60.0,
        )
        low_weight = low_confidence.get_feedback_weight()
        assert low_weight < 0.5

        # Expert feedback should have highest weight
        expert_feedback = HumanFeedback.create_binary_feedback(
            feedback_id="fb_003",
            sample_id="sample_125",
            annotator_id="annotator_456",
            is_anomaly=True,
            confidence=FeedbackConfidence.EXPERT,
            time_spent_seconds=120.0,
        )
        expert_weight = expert_feedback.get_feedback_weight()
        assert expert_weight >= high_weight

        # Time spent should affect weight
        quick_feedback = HumanFeedback.create_binary_feedback(
            feedback_id="fb_004",
            sample_id="sample_126",
            annotator_id="annotator_456",
            is_anomaly=True,
            confidence=FeedbackConfidence.HIGH,
            time_spent_seconds=5.0,  # Very quick
        )
        quick_weight = quick_feedback.get_feedback_weight()
        assert quick_weight < high_weight

    def test_to_dict(self):
        """Test converting feedback to dictionary."""
        original_prediction = AnomalyScore(value=0.3)

        feedback = HumanFeedback.create_binary_feedback(
            feedback_id="fb_001",
            sample_id="sample_123",
            annotator_id="annotator_456",
            is_anomaly=True,
            confidence=FeedbackConfidence.HIGH,
            original_prediction=original_prediction,
            session_id="session_789",
            time_spent_seconds=45.5,
            metadata={"source": "expert_review"},
        )

        feedback_dict = feedback.to_dict()

        assert feedback_dict["feedback_id"] == "fb_001"
        assert feedback_dict["sample_id"] == "sample_123"
        assert feedback_dict["annotator_id"] == "annotator_456"
        assert feedback_dict["feedback_type"] == "binary_classification"
        assert feedback_dict["feedback_value"] is True
        assert feedback_dict["confidence"] == "high"
        assert feedback_dict["original_prediction"] == 0.3
        assert feedback_dict["session_id"] == "session_789"
        assert feedback_dict["time_spent_seconds"] == 45.5
        assert feedback_dict["metadata"]["source"] == "expert_review"
        assert "is_correction" in feedback_dict
        assert "feedback_weight" in feedback_dict
        assert "timestamp" in feedback_dict

    def test_string_representation(self):
        """Test string representation of feedback."""
        feedback = HumanFeedback.create_binary_feedback(
            feedback_id="fb_001",
            sample_id="sample_123",
            annotator_id="annotator_456",
            is_anomaly=True,
            confidence=FeedbackConfidence.HIGH,
        )

        str_repr = str(feedback)
        assert "fb_001" in str_repr
        assert "binary_classification" in str_repr
        assert "True" in str_repr
        assert "high" in str_repr

    def test_feedback_consistency(self):
        """Test that feedback maintains consistency across operations."""
        original_prediction = AnomalyScore(value=0.7)

        feedback = HumanFeedback.create_score_correction(
            feedback_id="fb_001",
            sample_id="sample_123",
            annotator_id="annotator_456",
            corrected_score=0.9,
            confidence=FeedbackConfidence.EXPERT,
            original_prediction=original_prediction,
            time_spent_seconds=180.0,
        )

        # Test that all operations are consistent
        assert feedback.is_correction()  # Should be correction (0.7 -> 0.9)
        corrected_score = feedback.get_corrected_score()
        assert corrected_score is not None
        assert corrected_score.value == 0.9

        weight = feedback.get_feedback_weight()
        assert weight == 1.0  # Expert confidence with good time

        feedback_dict = feedback.to_dict()
        assert feedback_dict["is_correction"] is True
        assert feedback_dict["feedback_weight"] == weight
