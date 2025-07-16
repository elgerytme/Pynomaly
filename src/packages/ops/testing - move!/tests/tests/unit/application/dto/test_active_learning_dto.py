"""Tests for Active Learning DTOs."""

from datetime import datetime
from uuid import uuid4

import numpy as np
import pytest

from monorepo.application.dto.active_learning_dto import (
    AnnotationTaskRequest,
    AnnotationTaskResponse,
    BatchFeedbackRequest,
    BatchFeedbackResponse,
    CreateSessionRequest,
    CreateSessionResponse,
    LearningProgressRequest,
    LearningProgressResponse,
    SelectSamplesRequest,
    SelectSamplesResponse,
    SessionStatusRequest,
    SessionStatusResponse,
    SubmitFeedbackRequest,
    SubmitFeedbackResponse,
    UpdateModelRequest,
    UpdateModelResponse,
)
from monorepo.domain.entities.active_learning_session import (
    SamplingStrategy,
    SessionStatus,
)
from monorepo.domain.entities.detection_result import DetectionResult
from monorepo.domain.entities.human_feedback import (
    FeedbackConfidence,
    FeedbackType,
    HumanFeedback,
)
from monorepo.domain.value_objects.anomaly_score import AnomalyScore


class TestCreateSessionRequest:
    """Test suite for CreateSessionRequest."""

    def test_valid_creation(self):
        """Test creating a valid create session request."""
        request = CreateSessionRequest(
            annotator_id="annotator_123",
            model_version="v1.0.0",
            sampling_strategy=SamplingStrategy.UNCERTAINTY_SAMPLING,
            max_samples=50,
            timeout_minutes=120,
            min_feedback_quality=0.8,
            target_corrections=10,
            metadata={"experiment": "fraud_detection", "priority": "high"},
        )

        assert request.annotator_id == "annotator_123"
        assert request.model_version == "v1.0.0"
        assert request.sampling_strategy == SamplingStrategy.UNCERTAINTY_SAMPLING
        assert request.max_samples == 50
        assert request.timeout_minutes == 120
        assert request.min_feedback_quality == 0.8
        assert request.target_corrections == 10
        assert request.metadata == {"experiment": "fraud_detection", "priority": "high"}

    def test_default_values(self):
        """Test default values."""
        request = CreateSessionRequest(
            annotator_id="annotator_456",
            model_version="v2.0.0",
            sampling_strategy=SamplingStrategy.RANDOM_SAMPLING,
        )

        assert request.annotator_id == "annotator_456"
        assert request.model_version == "v2.0.0"
        assert request.sampling_strategy == SamplingStrategy.RANDOM_SAMPLING
        assert request.max_samples == 20
        assert request.timeout_minutes == 60
        assert request.min_feedback_quality == 0.7
        assert request.target_corrections is None
        assert request.metadata == {}

    def test_empty_annotator_id_validation(self):
        """Test validation for empty annotator ID."""
        with pytest.raises(ValueError, match="Annotator ID cannot be empty"):
            CreateSessionRequest(
                annotator_id="",
                model_version="v1.0.0",
                sampling_strategy=SamplingStrategy.UNCERTAINTY_SAMPLING,
            )

    def test_empty_model_version_validation(self):
        """Test validation for empty model version."""
        with pytest.raises(ValueError, match="Model version cannot be empty"):
            CreateSessionRequest(
                annotator_id="annotator_123",
                model_version="",
                sampling_strategy=SamplingStrategy.UNCERTAINTY_SAMPLING,
            )

    def test_negative_max_samples_validation(self):
        """Test validation for negative max samples."""
        with pytest.raises(ValueError, match="Max samples must be positive"):
            CreateSessionRequest(
                annotator_id="annotator_123",
                model_version="v1.0.0",
                sampling_strategy=SamplingStrategy.UNCERTAINTY_SAMPLING,
                max_samples=-5,
            )

    def test_zero_max_samples_validation(self):
        """Test validation for zero max samples."""
        with pytest.raises(ValueError, match="Max samples must be positive"):
            CreateSessionRequest(
                annotator_id="annotator_123",
                model_version="v1.0.0",
                sampling_strategy=SamplingStrategy.UNCERTAINTY_SAMPLING,
                max_samples=0,
            )

    def test_negative_timeout_validation(self):
        """Test validation for negative timeout."""
        with pytest.raises(ValueError, match="Timeout must be positive"):
            CreateSessionRequest(
                annotator_id="annotator_123",
                model_version="v1.0.0",
                sampling_strategy=SamplingStrategy.UNCERTAINTY_SAMPLING,
                timeout_minutes=-10,
            )

    def test_invalid_min_feedback_quality_validation(self):
        """Test validation for invalid min feedback quality."""
        # Too low
        with pytest.raises(
            ValueError, match="Min feedback quality must be between 0 and 1"
        ):
            CreateSessionRequest(
                annotator_id="annotator_123",
                model_version="v1.0.0",
                sampling_strategy=SamplingStrategy.UNCERTAINTY_SAMPLING,
                min_feedback_quality=-0.1,
            )

        # Too high
        with pytest.raises(
            ValueError, match="Min feedback quality must be between 0 and 1"
        ):
            CreateSessionRequest(
                annotator_id="annotator_123",
                model_version="v1.0.0",
                sampling_strategy=SamplingStrategy.UNCERTAINTY_SAMPLING,
                min_feedback_quality=1.1,
            )

    def test_none_metadata_initialization(self):
        """Test that None metadata is initialized to empty dict."""
        request = CreateSessionRequest(
            annotator_id="annotator_123",
            model_version="v1.0.0",
            sampling_strategy=SamplingStrategy.UNCERTAINTY_SAMPLING,
            metadata=None,
        )

        assert request.metadata == {}


class TestCreateSessionResponse:
    """Test suite for CreateSessionResponse."""

    def test_valid_creation(self):
        """Test creating a valid create session response."""
        created_at = datetime.now()
        configuration = {
            "max_samples": 50,
            "timeout_minutes": 120,
            "sampling_strategy": "uncertainty_sampling",
        }

        response = CreateSessionResponse(
            session_id="session_123",
            status=SessionStatus.ACTIVE,
            created_at=created_at,
            configuration=configuration,
            message="Session created successfully",
        )

        assert response.session_id == "session_123"
        assert response.status == SessionStatus.ACTIVE
        assert response.created_at == created_at
        assert response.configuration == configuration
        assert response.message == "Session created successfully"


class TestSelectSamplesRequest:
    """Test suite for SelectSamplesRequest."""

    def test_valid_creation(self):
        """Test creating a valid select samples request."""
        # Create mock detection results
        detection_results = [
            DetectionResult(
                id=str(uuid4()),
                detector_id=str(uuid4()),
                dataset_id=str(uuid4()),
                anomaly_score=AnomalyScore(0.8),
                is_anomaly=True,
                sample_index=i,
            )
            for i in range(10)
        ]

        features = np.random.rand(10, 5)
        strategy_params = {"threshold": 0.5, "diversity_weight": 0.3}

        request = SelectSamplesRequest(
            session_id="session_123",
            detection_results=detection_results,
            n_samples=5,
            sampling_strategy=SamplingStrategy.UNCERTAINTY_SAMPLING,
            features=features,
            strategy_params=strategy_params,
        )

        assert request.session_id == "session_123"
        assert len(request.detection_results) == 10
        assert request.n_samples == 5
        assert request.sampling_strategy == SamplingStrategy.UNCERTAINTY_SAMPLING
        assert request.features.shape == (10, 5)
        assert request.strategy_params == strategy_params

    def test_default_values(self):
        """Test default values."""
        detection_results = [
            DetectionResult(
                id=str(uuid4()),
                detector_id=str(uuid4()),
                dataset_id=str(uuid4()),
                anomaly_score=AnomalyScore(0.7),
                is_anomaly=False,
                sample_index=0,
            )
        ]

        request = SelectSamplesRequest(
            session_id="session_456",
            detection_results=detection_results,
            n_samples=1,
            sampling_strategy=SamplingStrategy.RANDOM_SAMPLING,
        )

        assert request.session_id == "session_456"
        assert len(request.detection_results) == 1
        assert request.n_samples == 1
        assert request.sampling_strategy == SamplingStrategy.RANDOM_SAMPLING
        assert request.features is None
        assert request.strategy_params == {}
        assert request.ensemble_results is None
        assert request.model_gradients is None
        assert request.existing_feedback is None

    def test_empty_session_id_validation(self):
        """Test validation for empty session ID."""
        with pytest.raises(ValueError, match="Session ID cannot be empty"):
            SelectSamplesRequest(
                session_id="",
                detection_results=[],
                n_samples=1,
                sampling_strategy=SamplingStrategy.RANDOM_SAMPLING,
            )

    def test_empty_detection_results_validation(self):
        """Test validation for empty detection results."""
        with pytest.raises(ValueError, match="Detection results cannot be empty"):
            SelectSamplesRequest(
                session_id="session_123",
                detection_results=[],
                n_samples=1,
                sampling_strategy=SamplingStrategy.RANDOM_SAMPLING,
            )

    def test_negative_n_samples_validation(self):
        """Test validation for negative n_samples."""
        detection_results = [
            DetectionResult(
                id=str(uuid4()),
                detector_id=str(uuid4()),
                dataset_id=str(uuid4()),
                anomaly_score=AnomalyScore(0.7),
                is_anomaly=False,
                sample_index=0,
            )
        ]

        with pytest.raises(ValueError, match="Number of samples must be positive"):
            SelectSamplesRequest(
                session_id="session_123",
                detection_results=detection_results,
                n_samples=-1,
                sampling_strategy=SamplingStrategy.RANDOM_SAMPLING,
            )

    def test_too_many_samples_validation(self):
        """Test validation for requesting more samples than available."""
        detection_results = [
            DetectionResult(
                id=str(uuid4()),
                detector_id=str(uuid4()),
                dataset_id=str(uuid4()),
                anomaly_score=AnomalyScore(0.7),
                is_anomaly=False,
                sample_index=0,
            )
        ]

        with pytest.raises(
            ValueError, match="Cannot select more samples than available"
        ):
            SelectSamplesRequest(
                session_id="session_123",
                detection_results=detection_results,
                n_samples=5,
                sampling_strategy=SamplingStrategy.RANDOM_SAMPLING,
            )


class TestSelectSamplesResponse:
    """Test suite for SelectSamplesResponse."""

    def test_valid_creation(self):
        """Test creating a valid select samples response."""
        selected_samples = [
            {"sample_id": "sample_1", "score": 0.9, "uncertainty": 0.8},
            {"sample_id": "sample_2", "score": 0.7, "uncertainty": 0.9},
            {"sample_id": "sample_3", "score": 0.85, "uncertainty": 0.75},
        ]

        selection_metadata = {
            "strategy_used": "uncertainty_sampling",
            "selection_time_ms": 150,
            "diversity_score": 0.7,
        }

        response = SelectSamplesResponse(
            session_id="session_123",
            selected_samples=selected_samples,
            sampling_strategy=SamplingStrategy.UNCERTAINTY_SAMPLING,
            selection_metadata=selection_metadata,
        )

        assert response.session_id == "session_123"
        assert len(response.selected_samples) == 3
        assert response.sampling_strategy == SamplingStrategy.UNCERTAINTY_SAMPLING
        assert response.selection_metadata == selection_metadata


class TestSubmitFeedbackRequest:
    """Test suite for SubmitFeedbackRequest."""

    def test_valid_binary_feedback_creation(self):
        """Test creating valid binary feedback request."""
        request = SubmitFeedbackRequest(
            session_id="session_123",
            sample_id="sample_456",
            annotator_id="annotator_789",
            feedback_type=FeedbackType.BINARY_CLASSIFICATION,
            feedback_value=True,
            confidence=FeedbackConfidence.HIGH,
            original_prediction=AnomalyScore(0.8),
            time_spent_seconds=45.5,
            metadata={"notes": "Clear anomaly pattern"},
        )

        assert request.session_id == "session_123"
        assert request.sample_id == "sample_456"
        assert request.annotator_id == "annotator_789"
        assert request.feedback_type == FeedbackType.BINARY_CLASSIFICATION
        assert request.feedback_value is True
        assert request.confidence == FeedbackConfidence.HIGH
        assert request.original_prediction.value == 0.8
        assert request.time_spent_seconds == 45.5
        assert request.metadata == {"notes": "Clear anomaly pattern"}

    def test_valid_confidence_rating_feedback_creation(self):
        """Test creating valid confidence rating feedback request."""
        request = SubmitFeedbackRequest(
            session_id="session_123",
            sample_id="sample_456",
            annotator_id="annotator_789",
            feedback_type=FeedbackType.CONFIDENCE_RATING,
            feedback_value=0.75,
            confidence=FeedbackConfidence.MEDIUM,
        )

        assert request.feedback_type == FeedbackType.CONFIDENCE_RATING
        assert request.feedback_value == 0.75
        assert request.confidence == FeedbackConfidence.MEDIUM

    def test_valid_score_correction_feedback_creation(self):
        """Test creating valid score correction feedback request."""
        request = SubmitFeedbackRequest(
            session_id="session_123",
            sample_id="sample_456",
            annotator_id="annotator_789",
            feedback_type=FeedbackType.SCORE_CORRECTION,
            feedback_value=0.9,
            confidence=FeedbackConfidence.HIGH,
        )

        assert request.feedback_type == FeedbackType.SCORE_CORRECTION
        assert request.feedback_value == 0.9
        assert request.confidence == FeedbackConfidence.HIGH

    def test_valid_explanation_feedback_creation(self):
        """Test creating valid explanation feedback request."""
        request = SubmitFeedbackRequest(
            session_id="session_123",
            sample_id="sample_456",
            annotator_id="annotator_789",
            feedback_type=FeedbackType.EXPLANATION,
            feedback_value="This pattern indicates fraudulent behavior",
            confidence=FeedbackConfidence.HIGH,
        )

        assert request.feedback_type == FeedbackType.EXPLANATION
        assert request.feedback_value == "This pattern indicates fraudulent behavior"
        assert request.confidence == FeedbackConfidence.HIGH

    def test_default_values(self):
        """Test default values."""
        request = SubmitFeedbackRequest(
            session_id="session_123",
            sample_id="sample_456",
            annotator_id="annotator_789",
            feedback_type=FeedbackType.BINARY_CLASSIFICATION,
            feedback_value=False,
            confidence=FeedbackConfidence.LOW,
        )

        assert request.original_prediction is None
        assert request.time_spent_seconds is None
        assert request.metadata == {}

    def test_empty_session_id_validation(self):
        """Test validation for empty session ID."""
        with pytest.raises(ValueError, match="Session ID cannot be empty"):
            SubmitFeedbackRequest(
                session_id="",
                sample_id="sample_456",
                annotator_id="annotator_789",
                feedback_type=FeedbackType.BINARY_CLASSIFICATION,
                feedback_value=True,
                confidence=FeedbackConfidence.HIGH,
            )

    def test_empty_sample_id_validation(self):
        """Test validation for empty sample ID."""
        with pytest.raises(ValueError, match="Sample ID cannot be empty"):
            SubmitFeedbackRequest(
                session_id="session_123",
                sample_id="",
                annotator_id="annotator_789",
                feedback_type=FeedbackType.BINARY_CLASSIFICATION,
                feedback_value=True,
                confidence=FeedbackConfidence.HIGH,
            )

    def test_empty_annotator_id_validation(self):
        """Test validation for empty annotator ID."""
        with pytest.raises(ValueError, match="Annotator ID cannot be empty"):
            SubmitFeedbackRequest(
                session_id="session_123",
                sample_id="sample_456",
                annotator_id="",
                feedback_type=FeedbackType.BINARY_CLASSIFICATION,
                feedback_value=True,
                confidence=FeedbackConfidence.HIGH,
            )

    def test_invalid_binary_feedback_validation(self):
        """Test validation for invalid binary feedback value."""
        with pytest.raises(
            ValueError, match="Binary classification feedback must be boolean"
        ):
            SubmitFeedbackRequest(
                session_id="session_123",
                sample_id="sample_456",
                annotator_id="annotator_789",
                feedback_type=FeedbackType.BINARY_CLASSIFICATION,
                feedback_value="invalid",
                confidence=FeedbackConfidence.HIGH,
            )

    def test_invalid_confidence_rating_type_validation(self):
        """Test validation for invalid confidence rating type."""
        with pytest.raises(ValueError, match="Confidence rating must be numeric"):
            SubmitFeedbackRequest(
                session_id="session_123",
                sample_id="sample_456",
                annotator_id="annotator_789",
                feedback_type=FeedbackType.CONFIDENCE_RATING,
                feedback_value="invalid",
                confidence=FeedbackConfidence.HIGH,
            )

    def test_invalid_confidence_rating_range_validation(self):
        """Test validation for confidence rating out of range."""
        # Too low
        with pytest.raises(
            ValueError, match="Confidence rating must be between 0 and 1"
        ):
            SubmitFeedbackRequest(
                session_id="session_123",
                sample_id="sample_456",
                annotator_id="annotator_789",
                feedback_type=FeedbackType.CONFIDENCE_RATING,
                feedback_value=-0.1,
                confidence=FeedbackConfidence.HIGH,
            )

        # Too high
        with pytest.raises(
            ValueError, match="Confidence rating must be between 0 and 1"
        ):
            SubmitFeedbackRequest(
                session_id="session_123",
                sample_id="sample_456",
                annotator_id="annotator_789",
                feedback_type=FeedbackType.CONFIDENCE_RATING,
                feedback_value=1.1,
                confidence=FeedbackConfidence.HIGH,
            )

    def test_invalid_score_correction_type_validation(self):
        """Test validation for invalid score correction type."""
        with pytest.raises(ValueError, match="Score correction must be numeric"):
            SubmitFeedbackRequest(
                session_id="session_123",
                sample_id="sample_456",
                annotator_id="annotator_789",
                feedback_type=FeedbackType.SCORE_CORRECTION,
                feedback_value="invalid",
                confidence=FeedbackConfidence.HIGH,
            )

    def test_invalid_score_correction_range_validation(self):
        """Test validation for score correction out of range."""
        # Too low
        with pytest.raises(
            ValueError, match="Score correction must be between 0 and 1"
        ):
            SubmitFeedbackRequest(
                session_id="session_123",
                sample_id="sample_456",
                annotator_id="annotator_789",
                feedback_type=FeedbackType.SCORE_CORRECTION,
                feedback_value=-0.1,
                confidence=FeedbackConfidence.HIGH,
            )

        # Too high
        with pytest.raises(
            ValueError, match="Score correction must be between 0 and 1"
        ):
            SubmitFeedbackRequest(
                session_id="session_123",
                sample_id="sample_456",
                annotator_id="annotator_789",
                feedback_type=FeedbackType.SCORE_CORRECTION,
                feedback_value=1.1,
                confidence=FeedbackConfidence.HIGH,
            )

    def test_invalid_explanation_type_validation(self):
        """Test validation for invalid explanation type."""
        with pytest.raises(ValueError, match="Explanation feedback must be string"):
            SubmitFeedbackRequest(
                session_id="session_123",
                sample_id="sample_456",
                annotator_id="annotator_789",
                feedback_type=FeedbackType.EXPLANATION,
                feedback_value=123,
                confidence=FeedbackConfidence.HIGH,
            )

    def test_empty_explanation_validation(self):
        """Test validation for empty explanation."""
        with pytest.raises(ValueError, match="Explanation cannot be empty"):
            SubmitFeedbackRequest(
                session_id="session_123",
                sample_id="sample_456",
                annotator_id="annotator_789",
                feedback_type=FeedbackType.EXPLANATION,
                feedback_value="   ",
                confidence=FeedbackConfidence.HIGH,
            )


class TestSubmitFeedbackResponse:
    """Test suite for SubmitFeedbackResponse."""

    def test_valid_creation(self):
        """Test creating a valid submit feedback response."""
        feedback_summary = {
            "feedback_type": "binary_classification",
            "value": True,
            "confidence": "high",
            "processing_time_ms": 250,
        }

        quality_assessment = {
            "consistency_score": 0.9,
            "confidence_alignment": 0.85,
            "time_efficiency": 0.8,
        }

        next_recommendations = ["sample_789", "sample_101", "sample_202"]

        response = SubmitFeedbackResponse(
            feedback_id="feedback_123",
            session_id="session_456",
            feedback_summary=feedback_summary,
            quality_assessment=quality_assessment,
            next_recommendations=next_recommendations,
        )

        assert response.feedback_id == "feedback_123"
        assert response.session_id == "session_456"
        assert response.feedback_summary == feedback_summary
        assert response.quality_assessment == quality_assessment
        assert response.next_recommendations == next_recommendations


class TestSessionStatusRequest:
    """Test suite for SessionStatusRequest."""

    def test_valid_creation(self):
        """Test creating a valid session status request."""
        request = SessionStatusRequest(
            session_id="session_123", include_details=True, include_feedback=True
        )

        assert request.session_id == "session_123"
        assert request.include_details is True
        assert request.include_feedback is True

    def test_default_values(self):
        """Test default values."""
        request = SessionStatusRequest(session_id="session_456")

        assert request.session_id == "session_456"
        assert request.include_details is True
        assert request.include_feedback is False

    def test_empty_session_id_validation(self):
        """Test validation for empty session ID."""
        with pytest.raises(ValueError, match="Session ID cannot be empty"):
            SessionStatusRequest(session_id="")


class TestSessionStatusResponse:
    """Test suite for SessionStatusResponse."""

    def test_valid_creation(self):
        """Test creating a valid session status response."""
        progress = {
            "samples_annotated": 15,
            "target_samples": 50,
            "completion_percentage": 30.0,
            "time_elapsed_minutes": 45,
        }

        quality_metrics = {
            "average_confidence": 0.8,
            "consistency_score": 0.85,
            "annotation_speed": 2.5,
        }

        recent_activity = [
            {"timestamp": "2023-01-01T10:00:00Z", "action": "feedback_submitted"},
            {"timestamp": "2023-01-01T09:55:00Z", "action": "sample_selected"},
        ]

        response = SessionStatusResponse(
            session_id="session_123",
            status=SessionStatus.ACTIVE,
            progress=progress,
            quality_metrics=quality_metrics,
            recent_activity=recent_activity,
            message="Session progressing well",
        )

        assert response.session_id == "session_123"
        assert response.status == SessionStatus.ACTIVE
        assert response.progress == progress
        assert response.quality_metrics == quality_metrics
        assert response.recent_activity == recent_activity
        assert response.message == "Session progressing well"

    def test_default_values(self):
        """Test default values."""
        response = SessionStatusResponse(
            session_id="session_456",
            status=SessionStatus.COMPLETED,
            progress={"completion_percentage": 100.0},
            quality_metrics={"average_confidence": 0.9},
        )

        assert response.session_id == "session_456"
        assert response.status == SessionStatus.COMPLETED
        assert response.recent_activity is None
        assert response.message is None


class TestUpdateModelRequest:
    """Test suite for UpdateModelRequest."""

    def test_valid_creation(self):
        """Test creating a valid update model request."""
        feedback_list = [
            HumanFeedback(
                id=str(uuid4()),
                session_id="session_123",
                sample_id="sample_1",
                annotator_id="annotator_1",
                feedback_type=FeedbackType.BINARY_CLASSIFICATION,
                feedback_value=True,
                confidence=FeedbackConfidence.HIGH,
                timestamp=datetime.now(),
            ),
            HumanFeedback(
                id=str(uuid4()),
                session_id="session_123",
                sample_id="sample_2",
                annotator_id="annotator_1",
                feedback_type=FeedbackType.BINARY_CLASSIFICATION,
                feedback_value=False,
                confidence=FeedbackConfidence.MEDIUM,
                timestamp=datetime.now(),
            ),
        ]

        request = UpdateModelRequest(
            session_id="session_123",
            feedback_list=feedback_list,
            learning_rate=0.05,
            validation_split=0.3,
            update_strategy="batch",
        )

        assert request.session_id == "session_123"
        assert len(request.feedback_list) == 2
        assert request.learning_rate == 0.05
        assert request.validation_split == 0.3
        assert request.update_strategy == "batch"

    def test_default_values(self):
        """Test default values."""
        feedback_list = [
            HumanFeedback(
                id=str(uuid4()),
                session_id="session_123",
                sample_id="sample_1",
                annotator_id="annotator_1",
                feedback_type=FeedbackType.BINARY_CLASSIFICATION,
                feedback_value=True,
                confidence=FeedbackConfidence.HIGH,
                timestamp=datetime.now(),
            )
        ]

        request = UpdateModelRequest(
            session_id="session_123", feedback_list=feedback_list
        )

        assert request.learning_rate == 0.1
        assert request.validation_split == 0.2
        assert request.update_strategy == "incremental"

    def test_empty_session_id_validation(self):
        """Test validation for empty session ID."""
        with pytest.raises(ValueError, match="Session ID cannot be empty"):
            UpdateModelRequest(session_id="", feedback_list=[])

    def test_empty_feedback_list_validation(self):
        """Test validation for empty feedback list."""
        with pytest.raises(ValueError, match="Feedback list cannot be empty"):
            UpdateModelRequest(session_id="session_123", feedback_list=[])

    def test_invalid_learning_rate_validation(self):
        """Test validation for invalid learning rate."""
        feedback_list = [
            HumanFeedback(
                id=str(uuid4()),
                session_id="session_123",
                sample_id="sample_1",
                annotator_id="annotator_1",
                feedback_type=FeedbackType.BINARY_CLASSIFICATION,
                feedback_value=True,
                confidence=FeedbackConfidence.HIGH,
                timestamp=datetime.now(),
            )
        ]

        # Too low
        with pytest.raises(ValueError, match="Learning rate must be between 0 and 1"):
            UpdateModelRequest(
                session_id="session_123", feedback_list=feedback_list, learning_rate=0.0
            )

        # Too high
        with pytest.raises(ValueError, match="Learning rate must be between 0 and 1"):
            UpdateModelRequest(
                session_id="session_123", feedback_list=feedback_list, learning_rate=1.1
            )

    def test_invalid_validation_split_validation(self):
        """Test validation for invalid validation split."""
        feedback_list = [
            HumanFeedback(
                id=str(uuid4()),
                session_id="session_123",
                sample_id="sample_1",
                annotator_id="annotator_1",
                feedback_type=FeedbackType.BINARY_CLASSIFICATION,
                feedback_value=True,
                confidence=FeedbackConfidence.HIGH,
                timestamp=datetime.now(),
            )
        ]

        # Too low
        with pytest.raises(
            ValueError, match="Validation split must be between 0 and 1"
        ):
            UpdateModelRequest(
                session_id="session_123",
                feedback_list=feedback_list,
                validation_split=-0.1,
            )

        # Too high
        with pytest.raises(
            ValueError, match="Validation split must be between 0 and 1"
        ):
            UpdateModelRequest(
                session_id="session_123",
                feedback_list=feedback_list,
                validation_split=1.0,
            )


class TestUpdateModelResponse:
    """Test suite for UpdateModelResponse."""

    def test_valid_creation(self):
        """Test creating a valid update model response."""
        update_statistics = {
            "feedback_processed": 25,
            "model_updates_applied": 8,
            "validation_accuracy": 0.92,
            "training_time_seconds": 45.5,
        }

        feedback_analysis = {
            "feedback_types": {"binary": 20, "score_correction": 5},
            "confidence_distribution": {"high": 15, "medium": 8, "low": 2},
            "annotation_quality": 0.88,
        }

        performance_impact = {
            "accuracy_change": 0.03,
            "precision_change": 0.02,
            "recall_change": 0.04,
            "f1_change": 0.025,
        }

        recommendations = [
            "Continue with current sampling strategy",
            "Consider increasing learning rate",
            "Add more diverse samples",
        ]

        next_session_suggestions = {
            "suggested_samples": 30,
            "focus_areas": ["edge_cases", "boundary_samples"],
            "estimated_improvement": 0.05,
        }

        response = UpdateModelResponse(
            session_id="session_123",
            update_applied=True,
            update_statistics=update_statistics,
            feedback_analysis=feedback_analysis,
            performance_impact=performance_impact,
            recommendations=recommendations,
            next_session_suggestions=next_session_suggestions,
        )

        assert response.session_id == "session_123"
        assert response.update_applied is True
        assert response.update_statistics == update_statistics
        assert response.feedback_analysis == feedback_analysis
        assert response.performance_impact == performance_impact
        assert response.recommendations == recommendations
        assert response.next_session_suggestions == next_session_suggestions


class TestAnnotationTaskRequest:
    """Test suite for AnnotationTaskRequest."""

    def test_valid_creation(self):
        """Test creating a valid annotation task request."""
        sample_data = {
            "features": {"amount": 1500.0, "location": "NYC", "time": "02:30"},
            "raw_data": {"transaction_id": "tx_123"},
        }

        current_prediction = {
            "anomaly_score": 0.85,
            "is_anomaly": True,
            "confidence": 0.9,
        }

        context = {
            "similar_samples": ["sample_1", "sample_2"],
            "historical_patterns": {"frequency": "rare", "region": "high_risk"},
        }

        request = AnnotationTaskRequest(
            sample_id="sample_123",
            sample_data=sample_data,
            current_prediction=current_prediction,
            context=context,
            instruction="Verify if this transaction pattern is anomalous",
            expected_time=120,
        )

        assert request.sample_id == "sample_123"
        assert request.sample_data == sample_data
        assert request.current_prediction == current_prediction
        assert request.context == context
        assert request.instruction == "Verify if this transaction pattern is anomalous"
        assert request.expected_time == 120

    def test_default_values(self):
        """Test default values."""
        request = AnnotationTaskRequest(
            sample_id="sample_456",
            sample_data={"feature1": 1.0},
            current_prediction={"score": 0.7},
        )

        assert request.context is None
        assert request.instruction is None
        assert request.expected_time is None

    def test_empty_sample_id_validation(self):
        """Test validation for empty sample ID."""
        with pytest.raises(ValueError, match="Sample ID cannot be empty"):
            AnnotationTaskRequest(
                sample_id="",
                sample_data={"feature1": 1.0},
                current_prediction={"score": 0.7},
            )

    def test_empty_sample_data_validation(self):
        """Test validation for empty sample data."""
        with pytest.raises(ValueError, match="Sample data cannot be empty"):
            AnnotationTaskRequest(
                sample_id="sample_123",
                sample_data={},
                current_prediction={"score": 0.7},
            )

    def test_empty_current_prediction_validation(self):
        """Test validation for empty current prediction."""
        with pytest.raises(ValueError, match="Current prediction cannot be empty"):
            AnnotationTaskRequest(
                sample_id="sample_123",
                sample_data={"feature1": 1.0},
                current_prediction={},
            )


class TestAnnotationTaskResponse:
    """Test suite for AnnotationTaskResponse."""

    def test_valid_creation(self):
        """Test creating a valid annotation task response."""
        annotation_interface = {
            "type": "binary_classification",
            "options": ["Normal", "Anomalous"],
            "additional_fields": ["confidence", "explanation"],
        }

        guidance = [
            "Consider the transaction amount relative to user history",
            "Check for unusual timing patterns",
            "Look for geographical anomalies",
        ]

        shortcuts = {
            "n": "Mark as Normal",
            "a": "Mark as Anomalous",
            "?": "Show help",
            "s": "Save and continue",
        }

        validation_rules = [
            "Confidence level must be selected",
            "Explanation required for anomalous samples",
            "Review similar samples before deciding",
        ]

        response = AnnotationTaskResponse(
            task_id="task_123",
            sample_id="sample_456",
            annotation_interface=annotation_interface,
            guidance=guidance,
            shortcuts=shortcuts,
            validation_rules=validation_rules,
        )

        assert response.task_id == "task_123"
        assert response.sample_id == "sample_456"
        assert response.annotation_interface == annotation_interface
        assert response.guidance == guidance
        assert response.shortcuts == shortcuts
        assert response.validation_rules == validation_rules


class TestLearningProgressRequest:
    """Test suite for LearningProgressRequest."""

    def test_valid_creation(self):
        """Test creating a valid learning progress request."""
        request = LearningProgressRequest(
            annotator_id="annotator_123",
            time_period_days=60,
            include_sessions=True,
            include_trends=True,
        )

        assert request.annotator_id == "annotator_123"
        assert request.time_period_days == 60
        assert request.include_sessions is True
        assert request.include_trends is True

    def test_default_values(self):
        """Test default values."""
        request = LearningProgressRequest(annotator_id="annotator_456")

        assert request.annotator_id == "annotator_456"
        assert request.time_period_days == 30
        assert request.include_sessions is True
        assert request.include_trends is True

    def test_empty_annotator_id_validation(self):
        """Test validation for empty annotator ID."""
        with pytest.raises(ValueError, match="Annotator ID cannot be empty"):
            LearningProgressRequest(annotator_id="")

    def test_negative_time_period_validation(self):
        """Test validation for negative time period."""
        with pytest.raises(ValueError, match="Time period must be positive"):
            LearningProgressRequest(annotator_id="annotator_123", time_period_days=-5)


class TestLearningProgressResponse:
    """Test suite for LearningProgressResponse."""

    def test_valid_creation(self):
        """Test creating a valid learning progress response."""
        analysis_period = {
            "start_date": "2023-01-01",
            "end_date": "2023-01-31",
            "duration_days": "30",
        }

        overall_metrics = {
            "total_annotations": 150,
            "average_confidence": 0.85,
            "consistency_score": 0.9,
            "improvement_rate": 0.15,
        }

        session_history = [
            {"session_id": "session_1", "date": "2023-01-01", "annotations": 25},
            {"session_id": "session_2", "date": "2023-01-15", "annotations": 30},
        ]

        trends = {
            "confidence_trend": "increasing",
            "speed_trend": "stable",
            "accuracy_trend": "improving",
        }

        recommendations = [
            "Continue with current annotation pace",
            "Focus on boundary cases for improved accuracy",
            "Consider advanced annotation techniques",
        ]

        response = LearningProgressResponse(
            annotator_id="annotator_123",
            analysis_period=analysis_period,
            overall_metrics=overall_metrics,
            session_history=session_history,
            trends=trends,
            recommendations=recommendations,
        )

        assert response.annotator_id == "annotator_123"
        assert response.analysis_period == analysis_period
        assert response.overall_metrics == overall_metrics
        assert response.session_history == session_history
        assert response.trends == trends
        assert response.recommendations == recommendations

    def test_default_values(self):
        """Test default values."""
        response = LearningProgressResponse(
            annotator_id="annotator_456",
            analysis_period={"start_date": "2023-01-01", "end_date": "2023-01-31"},
            overall_metrics={"total_annotations": 100},
        )

        assert response.session_history is None
        assert response.trends is None
        assert response.recommendations is None


class TestBatchFeedbackRequest:
    """Test suite for BatchFeedbackRequest."""

    def test_valid_creation(self):
        """Test creating a valid batch feedback request."""
        feedback_batch = [
            SubmitFeedbackRequest(
                session_id="session_123",
                sample_id="sample_1",
                annotator_id="annotator_1",
                feedback_type=FeedbackType.BINARY_CLASSIFICATION,
                feedback_value=True,
                confidence=FeedbackConfidence.HIGH,
            ),
            SubmitFeedbackRequest(
                session_id="session_123",
                sample_id="sample_2",
                annotator_id="annotator_1",
                feedback_type=FeedbackType.BINARY_CLASSIFICATION,
                feedback_value=False,
                confidence=FeedbackConfidence.MEDIUM,
            ),
        ]

        request = BatchFeedbackRequest(
            session_id="session_123",
            feedback_batch=feedback_batch,
            validate_consistency=True,
            auto_quality_check=True,
        )

        assert request.session_id == "session_123"
        assert len(request.feedback_batch) == 2
        assert request.validate_consistency is True
        assert request.auto_quality_check is True

    def test_default_values(self):
        """Test default values."""
        feedback_batch = [
            SubmitFeedbackRequest(
                session_id="session_456",
                sample_id="sample_1",
                annotator_id="annotator_1",
                feedback_type=FeedbackType.BINARY_CLASSIFICATION,
                feedback_value=True,
                confidence=FeedbackConfidence.HIGH,
            )
        ]

        request = BatchFeedbackRequest(
            session_id="session_456", feedback_batch=feedback_batch
        )

        assert request.validate_consistency is True
        assert request.auto_quality_check is True

    def test_empty_session_id_validation(self):
        """Test validation for empty session ID."""
        with pytest.raises(ValueError, match="Session ID cannot be empty"):
            BatchFeedbackRequest(session_id="", feedback_batch=[])

    def test_empty_feedback_batch_validation(self):
        """Test validation for empty feedback batch."""
        with pytest.raises(ValueError, match="Feedback batch cannot be empty"):
            BatchFeedbackRequest(session_id="session_123", feedback_batch=[])

    def test_mismatched_session_id_validation(self):
        """Test validation for mismatched session IDs in batch."""
        feedback_batch = [
            SubmitFeedbackRequest(
                session_id="session_123",
                sample_id="sample_1",
                annotator_id="annotator_1",
                feedback_type=FeedbackType.BINARY_CLASSIFICATION,
                feedback_value=True,
                confidence=FeedbackConfidence.HIGH,
            ),
            SubmitFeedbackRequest(
                session_id="session_456",  # Different session ID
                sample_id="sample_2",
                annotator_id="annotator_1",
                feedback_type=FeedbackType.BINARY_CLASSIFICATION,
                feedback_value=False,
                confidence=FeedbackConfidence.MEDIUM,
            ),
        ]

        with pytest.raises(
            ValueError, match="All feedback must belong to the same session"
        ):
            BatchFeedbackRequest(
                session_id="session_123", feedback_batch=feedback_batch
            )


class TestBatchFeedbackResponse:
    """Test suite for BatchFeedbackResponse."""

    def test_valid_creation(self):
        """Test creating a valid batch feedback response."""
        feedback_ids = ["feedback_1", "feedback_2", "feedback_3"]

        batch_quality = {
            "average_confidence": 0.8,
            "consistency_score": 0.85,
            "annotation_speed": 2.5,
            "quality_score": 0.82,
        }

        consistency_analysis = {
            "internal_consistency": 0.9,
            "cross_annotator_agreement": 0.85,
            "temporal_consistency": 0.88,
            "outlier_samples": ["sample_5"],
        }

        warnings = [
            "Low confidence on sample_3",
            "Potential annotation fatigue detected",
        ]

        response = BatchFeedbackResponse(
            session_id="session_123",
            processed_count=3,
            feedback_ids=feedback_ids,
            batch_quality=batch_quality,
            consistency_analysis=consistency_analysis,
            warnings=warnings,
        )

        assert response.session_id == "session_123"
        assert response.processed_count == 3
        assert response.feedback_ids == feedback_ids
        assert response.batch_quality == batch_quality
        assert response.consistency_analysis == consistency_analysis
        assert response.warnings == warnings


class TestActiveLearningIntegration:
    """Test suite for active learning integration scenarios."""

    def test_complete_active_learning_workflow(self):
        """Test complete active learning workflow."""
        # 1. Create session
        create_request = CreateSessionRequest(
            annotator_id="annotator_expert_1",
            model_version="fraud_detection_v2.1",
            sampling_strategy=SamplingStrategy.UNCERTAINTY_SAMPLING,
            max_samples=100,
            timeout_minutes=180,
            min_feedback_quality=0.8,
            target_corrections=20,
            metadata={"domain": "financial_fraud", "priority": "high"},
        )

        create_response = CreateSessionResponse(
            session_id="session_fraud_001",
            status=SessionStatus.ACTIVE,
            created_at=datetime.now(),
            configuration={
                "max_samples": 100,
                "timeout_minutes": 180,
                "sampling_strategy": "uncertainty_sampling",
            },
            message="Fraud detection session created successfully",
        )

        # 2. Select samples for annotation
        detection_results = [
            DetectionResult(
                id=str(uuid4()),
                detector_id="fraud_detector_v2",
                dataset_id="transactions_2023",
                anomaly_score=AnomalyScore(0.95),
                is_anomaly=True,
                sample_index=i,
            )
            for i in range(50)
        ]

        select_request = SelectSamplesRequest(
            session_id="session_fraud_001",
            detection_results=detection_results,
            n_samples=10,
            sampling_strategy=SamplingStrategy.UNCERTAINTY_SAMPLING,
            features=np.random.rand(50, 15),
            strategy_params={"diversity_weight": 0.3, "uncertainty_threshold": 0.1},
        )

        select_response = SelectSamplesResponse(
            session_id="session_fraud_001",
            selected_samples=[
                {
                    "sample_id": f"sample_{i}",
                    "score": 0.9 - i * 0.05,
                    "uncertainty": 0.8 + i * 0.02,
                }
                for i in range(10)
            ],
            sampling_strategy=SamplingStrategy.UNCERTAINTY_SAMPLING,
            selection_metadata={"selection_time_ms": 200, "diversity_score": 0.75},
        )

        # 3. Submit feedback
        feedback_request = SubmitFeedbackRequest(
            session_id="session_fraud_001",
            sample_id="sample_0",
            annotator_id="annotator_expert_1",
            feedback_type=FeedbackType.BINARY_CLASSIFICATION,
            feedback_value=True,
            confidence=FeedbackConfidence.HIGH,
            original_prediction=AnomalyScore(0.95),
            time_spent_seconds=75.5,
            metadata={"notes": "Clear fraudulent pattern", "difficulty": "easy"},
        )

        feedback_response = SubmitFeedbackResponse(
            feedback_id="feedback_001",
            session_id="session_fraud_001",
            feedback_summary={
                "feedback_type": "binary_classification",
                "value": True,
                "confidence": "high",
                "agreement_with_model": True,
            },
            quality_assessment={
                "consistency_score": 0.95,
                "confidence_alignment": 0.9,
                "time_efficiency": 0.85,
            },
            next_recommendations=["sample_1", "sample_2", "sample_3"],
        )

        # 4. Check session status
        status_request = SessionStatusRequest(
            session_id="session_fraud_001", include_details=True, include_feedback=True
        )

        status_response = SessionStatusResponse(
            session_id="session_fraud_001",
            status=SessionStatus.ACTIVE,
            progress={
                "samples_annotated": 1,
                "target_samples": 100,
                "completion_percentage": 1.0,
                "estimated_time_remaining_minutes": 170,
            },
            quality_metrics={
                "average_confidence": 0.9,
                "annotation_speed": 1.2,
                "model_agreement_rate": 0.85,
            },
            recent_activity=[
                {
                    "timestamp": datetime.now().isoformat(),
                    "action": "feedback_submitted",
                    "sample_id": "sample_0",
                }
            ],
            message="Session progressing well with high-quality annotations",
        )

        # Verify workflow consistency
        assert create_request.annotator_id == feedback_request.annotator_id
        assert create_response.session_id == select_request.session_id
        assert select_response.session_id == feedback_request.session_id
        assert feedback_response.session_id == status_request.session_id
        assert status_response.session_id == create_response.session_id
        assert len(select_response.selected_samples) == select_request.n_samples
        assert feedback_response.feedback_summary["agreement_with_model"] is True

    def test_batch_feedback_workflow(self):
        """Test batch feedback submission workflow."""
        # Create multiple feedback requests
        feedback_batch = [
            SubmitFeedbackRequest(
                session_id="session_batch_001",
                sample_id=f"sample_{i}",
                annotator_id="annotator_batch_1",
                feedback_type=FeedbackType.BINARY_CLASSIFICATION,
                feedback_value=i % 2 == 0,  # Alternate True/False
                confidence=FeedbackConfidence.HIGH
                if i < 3
                else FeedbackConfidence.MEDIUM,
                time_spent_seconds=60 + i * 10,
            )
            for i in range(5)
        ]

        batch_request = BatchFeedbackRequest(
            session_id="session_batch_001",
            feedback_batch=feedback_batch,
            validate_consistency=True,
            auto_quality_check=True,
        )

        batch_response = BatchFeedbackResponse(
            session_id="session_batch_001",
            processed_count=5,
            feedback_ids=[f"feedback_batch_{i}" for i in range(5)],
            batch_quality={
                "average_confidence": 0.85,
                "consistency_score": 0.9,
                "annotation_speed": 2.2,
                "quality_score": 0.88,
            },
            consistency_analysis={
                "internal_consistency": 0.92,
                "temporal_consistency": 0.88,
                "confidence_stability": 0.85,
                "outliers_detected": [],
            },
            warnings=["Slight decrease in annotation speed towards end of batch"],
        )

        # Verify batch processing
        assert len(batch_request.feedback_batch) == batch_response.processed_count
        assert len(batch_response.feedback_ids) == batch_response.processed_count
        assert batch_response.batch_quality["consistency_score"] > 0.8
        assert batch_response.consistency_analysis["internal_consistency"] > 0.9

    def test_model_update_workflow(self):
        """Test model update with collected feedback workflow."""
        # Create feedback for model update
        feedback_list = [
            HumanFeedback(
                id=str(uuid4()),
                session_id="session_update_001",
                sample_id=f"sample_{i}",
                annotator_id="annotator_update_1",
                feedback_type=FeedbackType.SCORE_CORRECTION,
                feedback_value=0.8 + i * 0.05,
                confidence=FeedbackConfidence.HIGH,
                timestamp=datetime.now(),
                original_prediction=AnomalyScore(0.7 + i * 0.05),
            )
            for i in range(10)
        ]

        update_request = UpdateModelRequest(
            session_id="session_update_001",
            feedback_list=feedback_list,
            learning_rate=0.05,
            validation_split=0.25,
            update_strategy="batch_update",
        )

        update_response = UpdateModelResponse(
            session_id="session_update_001",
            update_applied=True,
            update_statistics={
                "feedback_processed": 10,
                "model_parameters_updated": 150,
                "validation_accuracy": 0.94,
                "training_time_seconds": 120.5,
            },
            feedback_analysis={
                "feedback_types": {"score_correction": 10},
                "confidence_distribution": {"high": 10},
                "average_correction_magnitude": 0.08,
                "prediction_improvement": 0.12,
            },
            performance_impact={
                "accuracy_change": 0.08,
                "precision_change": 0.06,
                "recall_change": 0.10,
                "f1_change": 0.08,
            },
            recommendations=[
                "Performance improved significantly",
                "Continue with current learning rate",
                "Focus on boundary cases in next session",
            ],
            next_session_suggestions={
                "suggested_samples": 50,
                "focus_strategy": "boundary_sampling",
                "estimated_improvement": 0.04,
            },
        )

        # Verify model update
        assert (
            len(update_request.feedback_list)
            == update_response.update_statistics["feedback_processed"]
        )
        assert update_response.update_applied is True
        assert update_response.performance_impact["accuracy_change"] > 0
        assert len(update_response.recommendations) > 0
        assert "suggested_samples" in update_response.next_session_suggestions

    def test_annotation_task_workflow(self):
        """Test annotation task creation and guidance workflow."""
        # Create annotation task
        task_request = AnnotationTaskRequest(
            sample_id="complex_sample_001",
            sample_data={
                "transaction_amount": 15000.0,
                "merchant_category": "electronics",
                "location": "foreign_country",
                "time_of_day": "3:15 AM",
                "user_history": {"avg_transaction": 250.0, "location_frequency": 0.01},
            },
            current_prediction={
                "anomaly_score": 0.78,
                "is_anomaly": True,
                "confidence": 0.65,
                "contributing_factors": ["amount", "location", "time"],
            },
            context={
                "similar_samples": ["sample_A", "sample_B"],
                "historical_patterns": {
                    "amount_percentile": 99.5,
                    "location_risk": "high",
                },
            },
            instruction="Evaluate if this transaction represents fraudulent activity considering all contextual factors",
            expected_time=180,
        )

        task_response = AnnotationTaskResponse(
            task_id="task_complex_001",
            sample_id="complex_sample_001",
            annotation_interface={
                "type": "enhanced_classification",
                "primary_options": ["Legitimate", "Fraudulent"],
                "confidence_scale": {"min": 0, "max": 100, "step": 5},
                "required_fields": ["classification", "confidence", "explanation"],
                "optional_fields": ["contributing_factors", "uncertainty_notes"],
            },
            guidance=[
                "Consider transaction amount relative to user's historical behavior",
                "Evaluate geographical risk factors and user travel patterns",
                "Assess timing patterns and merchant category consistency",
                "Review similar cases for comparison",
            ],
            shortcuts={
                "l": "Mark as Legitimate",
                "f": "Mark as Fraudulent",
                "h": "Show historical data",
                "s": "Show similar samples",
                "?": "Show help",
            },
            validation_rules=[
                "Confidence level must be >= 70 for final submission",
                "Explanation required for all fraudulent classifications",
                "Review similar samples before making decision",
                "Consider all contributing factors in explanation",
            ],
        )

        # Verify annotation task setup
        assert task_request.sample_id == task_response.sample_id
        assert "transaction_amount" in task_request.sample_data
        assert task_request.current_prediction["anomaly_score"] == 0.78
        assert task_response.annotation_interface["type"] == "enhanced_classification"
        assert len(task_response.guidance) > 0
        assert len(task_response.validation_rules) > 0

    def test_learning_progress_analysis_workflow(self):
        """Test learning progress analysis workflow."""
        # Request learning progress analysis
        progress_request = LearningProgressRequest(
            annotator_id="annotator_progress_1",
            time_period_days=90,
            include_sessions=True,
            include_trends=True,
        )

        progress_response = LearningProgressResponse(
            annotator_id="annotator_progress_1",
            analysis_period={
                "start_date": "2023-10-01",
                "end_date": "2023-12-30",
                "duration_days": "90",
            },
            overall_metrics={
                "total_annotations": 450,
                "total_sessions": 18,
                "average_session_length_minutes": 85,
                "average_confidence": 0.87,
                "consistency_score": 0.91,
                "accuracy_improvement": 0.15,
                "speed_improvement": 0.12,
            },
            session_history=[
                {
                    "session_id": f"session_{i}",
                    "date": f"2023-{10 + i//10}-{(i%10)+1:02d}",
                    "annotations": 25 + i,
                    "quality_score": 0.75 + i * 0.01,
                    "duration_minutes": 80 + i * 2,
                }
                for i in range(18)
            ],
            trends={
                "confidence_trend": "steadily_increasing",
                "speed_trend": "improving",
                "accuracy_trend": "strong_improvement",
                "consistency_trend": "stable_high",
                "complexity_handling": "improving",
            },
            recommendations=[
                "Excellent progress over 90-day period",
                "Ready for more complex annotation tasks",
                "Consider mentoring junior annotators",
                "Focus on edge cases to further improve accuracy",
            ],
        )

        # Verify progress analysis
        assert progress_request.annotator_id == progress_response.annotator_id
        assert progress_response.overall_metrics["total_annotations"] > 400
        assert progress_response.overall_metrics["consistency_score"] > 0.9
        assert len(progress_response.session_history) == 18
        assert progress_response.trends["accuracy_trend"] == "strong_improvement"
        assert len(progress_response.recommendations) > 0

    def test_dataclass_serialization(self):
        """Test serialization and field access for dataclass DTOs."""
        # Test CreateSessionRequest
        request = CreateSessionRequest(
            annotator_id="test_annotator",
            model_version="v1.0",
            sampling_strategy=SamplingStrategy.UNCERTAINTY_SAMPLING,
            max_samples=25,
            metadata={"test": "value"},
        )

        # Verify field access
        assert hasattr(request, "annotator_id")
        assert hasattr(request, "model_version")
        assert hasattr(request, "sampling_strategy")
        assert hasattr(request, "max_samples")
        assert hasattr(request, "metadata")

        # Test field modification
        request.max_samples = 30
        assert request.max_samples == 30

        # Test metadata access
        assert request.metadata["test"] == "value"
        request.metadata["new_key"] = "new_value"
        assert request.metadata["new_key"] == "new_value"

    def test_edge_cases_and_boundary_conditions(self):
        """Test edge cases and boundary conditions."""
        # Minimum valid values
        min_request = CreateSessionRequest(
            annotator_id="a",
            model_version="v",
            sampling_strategy=SamplingStrategy.RANDOM_SAMPLING,
            max_samples=1,
            timeout_minutes=1,
            min_feedback_quality=0.0,
        )
        assert min_request.max_samples == 1
        assert min_request.timeout_minutes == 1
        assert min_request.min_feedback_quality == 0.0

        # Maximum valid values
        max_request = CreateSessionRequest(
            annotator_id="a" * 100,
            model_version="v" * 50,
            sampling_strategy=SamplingStrategy.COMMITTEE_DISAGREEMENT,
            max_samples=10000,
            timeout_minutes=1440,  # 24 hours
            min_feedback_quality=1.0,
        )
        assert max_request.max_samples == 10000
        assert max_request.timeout_minutes == 1440
        assert max_request.min_feedback_quality == 1.0

        # Boundary feedback values
        boundary_feedback = SubmitFeedbackRequest(
            session_id="session_boundary",
            sample_id="sample_boundary",
            annotator_id="annotator_boundary",
            feedback_type=FeedbackType.CONFIDENCE_RATING,
            feedback_value=0.0,  # Minimum valid value
            confidence=FeedbackConfidence.LOW,
        )
        assert boundary_feedback.feedback_value == 0.0

        boundary_feedback.feedback_value = 1.0  # Maximum valid value
        assert boundary_feedback.feedback_value == 1.0
