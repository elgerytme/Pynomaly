"""
FastAPI endpoints for active learning functionality.

This module provides REST API endpoints for managing human-in-the-loop
active learning sessions, sample selection, and feedback collection.
"""

from fastapi import APIRouter, BackgroundTasks, Depends, HTTPException, status
from pydantic import BaseModel, Field

from pynomaly.application.dto.active_learning_dto import CreateSessionRequest
from pynomaly.application.use_cases.manage_active_learning import (
    ManageActiveLearningUseCase,
)
from pynomaly.domain.entities.active_learning_session import SamplingStrategy
from pynomaly.domain.entities.human_feedback import FeedbackConfidence, FeedbackType
from pynomaly.domain.services.active_learning_service import ActiveLearningService

router = APIRouter(prefix="/active-learning", tags=["active-learning"])


class CreateSessionModel(BaseModel):
    """Pydantic model for session creation request."""

    annotator_id: str = Field(..., description="ID of the human annotator")
    model_version: str = Field(..., description="Version of the model being improved")
    sampling_strategy: str = Field(
        "uncertainty",
        description="Strategy for selecting samples",
        regex="^(uncertainty|diversity|disagreement|margin|entropy|committee_disagreement|expected_model_change|random)$",
    )
    max_samples: int = Field(20, ge=1, le=1000, description="Maximum number of samples")
    timeout_minutes: int | None = Field(60, ge=1, le=480, description="Session timeout")
    min_feedback_quality: float = Field(
        0.7, ge=0.0, le=1.0, description="Minimum feedback quality"
    )
    target_corrections: int | None = Field(
        None, ge=1, description="Target number of corrections"
    )
    metadata: dict = Field(default_factory=dict, description="Additional metadata")


class DetectionResultModel(BaseModel):
    """Pydantic model for detection result."""

    sample_id: str = Field(..., description="Sample identifier")
    score: float = Field(..., ge=0.0, le=1.0, description="Anomaly score")
    is_anomaly: bool = Field(..., description="Anomaly classification")
    timestamp: str | None = Field(None, description="Detection timestamp")
    model_version: str = Field(..., description="Model version")
    metadata: dict = Field(default_factory=dict, description="Additional metadata")


class SelectSamplesModel(BaseModel):
    """Pydantic model for sample selection request."""

    session_id: str = Field(..., description="Active learning session ID")
    detection_results: list[DetectionResultModel] = Field(
        ..., min_items=1, description="Available detection results"
    )
    n_samples: int = Field(..., ge=1, le=100, description="Number of samples to select")
    sampling_strategy: str = Field(
        "uncertainty",
        description="Strategy for sample selection",
        regex="^(uncertainty|diversity|disagreement|margin|entropy|committee_disagreement|expected_model_change|random)$",
    )
    strategy_params: dict = Field(
        default_factory=dict, description="Strategy parameters"
    )


class SubmitFeedbackModel(BaseModel):
    """Pydantic model for feedback submission."""

    session_id: str = Field(..., description="Session ID")
    sample_id: str = Field(..., description="Sample ID")
    annotator_id: str = Field(..., description="Annotator ID")
    feedback_type: str = Field(
        "binary_classification",
        description="Type of feedback",
        regex="^(binary_classification|confidence_rating|score_correction|explanation|feature_importance)$",
    )
    feedback_value: bool | float | str | dict = Field(..., description="Feedback value")
    confidence: str = Field(
        "medium", description="Confidence level", regex="^(low|medium|high|expert)$"
    )
    original_score: float | None = Field(
        None, ge=0.0, le=1.0, description="Original prediction score"
    )
    time_spent_seconds: float | None = Field(
        None, ge=0.0, description="Time spent on annotation"
    )
    metadata: dict = Field(default_factory=dict, description="Additional metadata")


class SessionStatusModel(BaseModel):
    """Pydantic model for session status request."""

    session_id: str = Field(..., description="Session ID")
    include_details: bool = Field(True, description="Include detailed information")
    include_feedback: bool = Field(False, description="Include feedback history")


class UpdateModelModel(BaseModel):
    """Pydantic model for model update request."""

    session_id: str = Field(..., description="Session ID")
    learning_rate: float = Field(0.1, ge=0.001, le=1.0, description="Learning rate")
    validation_split: float = Field(0.2, ge=0.0, lt=1.0, description="Validation split")
    update_strategy: str = Field("incremental", description="Update strategy")


def get_active_learning_use_case() -> ManageActiveLearningUseCase:
    """Dependency injection for active learning use case."""
    active_learning_service = ActiveLearningService(random_seed=42)
    return ManageActiveLearningUseCase(active_learning_service=active_learning_service)


def _convert_to_sampling_strategy(strategy_str: str) -> SamplingStrategy:
    """Convert string to SamplingStrategy enum."""
    strategy_map = {
        "uncertainty": SamplingStrategy.UNCERTAINTY,
        "diversity": SamplingStrategy.DIVERSITY,
        "disagreement": SamplingStrategy.DISAGREEMENT,
        "margin": SamplingStrategy.MARGIN,
        "entropy": SamplingStrategy.ENTROPY,
        "committee_disagreement": SamplingStrategy.COMMITTEE_DISAGREEMENT,
        "expected_model_change": SamplingStrategy.EXPECTED_MODEL_CHANGE,
        "random": SamplingStrategy.RANDOM,
    }

    if strategy_str not in strategy_map:
        raise ValueError(f"Unknown sampling strategy: {strategy_str}")

    return strategy_map[strategy_str]


def _convert_to_feedback_type(feedback_type_str: str) -> FeedbackType:
    """Convert string to FeedbackType enum."""
    type_map = {
        "binary_classification": FeedbackType.BINARY_CLASSIFICATION,
        "confidence_rating": FeedbackType.CONFIDENCE_RATING,
        "score_correction": FeedbackType.SCORE_CORRECTION,
        "explanation": FeedbackType.EXPLANATION,
        "feature_importance": FeedbackType.FEATURE_IMPORTANCE,
    }

    if feedback_type_str not in type_map:
        raise ValueError(f"Unknown feedback type: {feedback_type_str}")

    return type_map[feedback_type_str]


def _convert_to_feedback_confidence(confidence_str: str) -> FeedbackConfidence:
    """Convert string to FeedbackConfidence enum."""
    confidence_map = {
        "low": FeedbackConfidence.LOW,
        "medium": FeedbackConfidence.MEDIUM,
        "high": FeedbackConfidence.HIGH,
        "expert": FeedbackConfidence.EXPERT,
    }

    if confidence_str not in confidence_map:
        raise ValueError(f"Unknown confidence level: {confidence_str}")

    return confidence_map[confidence_str]


@router.post(
    "/sessions",
    response_model=dict,
    status_code=status.HTTP_201_CREATED,
    summary="Create new active learning session",
    description="""
    Create a new active learning session for human-in-the-loop training.

    The session will manage sample selection, feedback collection, and model updates
    based on the specified sampling strategy and configuration.

    **Sampling Strategies:**
    - **uncertainty**: Select samples with highest prediction uncertainty
    - **diversity**: Select diverse samples to cover feature space
    - **margin**: Select samples close to decision boundary
    - **committee_disagreement**: Select samples where ensemble models disagree
    - **expected_model_change**: Select samples likely to cause large model updates
    - **random**: Random selection baseline
    """,
)
async def create_session(
    request: CreateSessionModel,
    use_case: ManageActiveLearningUseCase = Depends(get_active_learning_use_case),
) -> dict:
    """Create a new active learning session."""
    try:
        # Convert to domain request
        domain_request = CreateSessionRequest(
            annotator_id=request.annotator_id,
            model_version=request.model_version,
            sampling_strategy=_convert_to_sampling_strategy(request.sampling_strategy),
            max_samples=request.max_samples,
            timeout_minutes=request.timeout_minutes,
            min_feedback_quality=request.min_feedback_quality,
            target_corrections=request.target_corrections,
            metadata=request.metadata,
        )

        # Execute use case
        response = use_case.create_session(domain_request)

        # Convert to API response
        return {
            "session_id": response.session_id,
            "status": response.status.value,
            "created_at": response.created_at.isoformat(),
            "configuration": response.configuration,
            "message": response.message,
        }

    except ValueError as e:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST, detail=f"Invalid request: {str(e)}"
        )
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Internal error: {str(e)}",
        )


@router.post(
    "/sessions/{session_id}/start",
    response_model=dict,
    status_code=status.HTTP_200_OK,
    summary="Start active learning session",
    description="""
    Start an active learning session. This transitions the session from
    CREATED to ACTIVE status and begins the annotation workflow.
    """,
)
async def start_session(
    session_id: str,
    use_case: ManageActiveLearningUseCase = Depends(get_active_learning_use_case),
) -> dict:
    """Start an active learning session."""
    try:
        response = use_case.start_session(session_id)

        return {
            "session_id": response.session_id,
            "status": response.status.value,
            "progress": response.progress,
            "quality_metrics": response.quality_metrics,
            "message": response.message,
        }

    except ValueError as e:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST, detail=f"Invalid session: {str(e)}"
        )
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Internal error: {str(e)}",
        )


@router.post(
    "/sessions/{session_id}/select-samples",
    response_model=dict,
    status_code=status.HTTP_200_OK,
    summary="Select samples for annotation",
    description="""
    Select the most informative samples for human annotation based on
    the specified sampling strategy.

    **Returns samples with:**
    - Sample identification and metadata
    - Current model predictions
    - Annotation value score
    - Selection reasoning

    The selection algorithm considers uncertainty, diversity, expected model
    impact, and other factors depending on the chosen strategy.
    """,
)
async def select_samples(
    session_id: str,
    request: SelectSamplesModel,
    use_case: ManageActiveLearningUseCase = Depends(get_active_learning_use_case),
) -> dict:
    """Select samples for annotation in an active learning session."""
    try:
        # Validate session ID matches
        if request.session_id != session_id:
            raise ValueError("Session ID mismatch")

        # Convert detection results to domain entities
        from pynomaly.domain.entities.detection_result import DetectionResult
        from pynomaly.domain.value_objects.anomaly_score import AnomalyScore

        detection_results = []
        for result_model in request.detection_results:
            score = AnomalyScore(value=result_model.score)
            result = DetectionResult(
                sample_id=result_model.sample_id,
                score=score,
                is_anomaly=result_model.is_anomaly,
                timestamp=result_model.timestamp,
                model_version=result_model.model_version,
                metadata=result_model.metadata,
            )
            detection_results.append(result)

        # Create domain request
        from pynomaly.application.dto.active_learning_dto import SelectSamplesRequest

        domain_request = SelectSamplesRequest(
            session_id=request.session_id,
            detection_results=detection_results,
            n_samples=request.n_samples,
            sampling_strategy=_convert_to_sampling_strategy(request.sampling_strategy),
            strategy_params=request.strategy_params,
        )

        # Execute use case
        response = use_case.select_samples(domain_request)

        # Convert to API response
        return {
            "session_id": response.session_id,
            "selected_samples": response.selected_samples,
            "sampling_strategy": response.sampling_strategy.value,
            "selection_metadata": response.selection_metadata,
        }

    except ValueError as e:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST, detail=f"Invalid request: {str(e)}"
        )
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Internal error: {str(e)}",
        )


@router.post(
    "/sessions/{session_id}/feedback",
    response_model=dict,
    status_code=status.HTTP_201_CREATED,
    summary="Submit human feedback",
    description="""
    Submit human feedback for a sample in an active learning session.

    **Feedback Types:**
    - **binary_classification**: True/False anomaly classification
    - **score_correction**: Corrected anomaly score (0.0 to 1.0)
    - **explanation**: Text explanation of reasoning
    - **confidence_rating**: Confidence in original prediction
    - **feature_importance**: Important features for decision

    **Confidence Levels:**
    - **low**: Uncertain about the annotation
    - **medium**: Moderately confident
    - **high**: Very confident in the annotation
    - **expert**: Domain expert level confidence

    The system tracks annotation time and quality to improve future
    sample selection and model updates.
    """,
)
async def submit_feedback(
    session_id: str,
    request: SubmitFeedbackModel,
    use_case: ManageActiveLearningUseCase = Depends(get_active_learning_use_case),
) -> dict:
    """Submit human feedback for a sample."""
    try:
        # Validate session ID matches
        if request.session_id != session_id:
            raise ValueError("Session ID mismatch")

        # Convert to domain request
        from pynomaly.application.dto.active_learning_dto import SubmitFeedbackRequest
        from pynomaly.domain.value_objects.anomaly_score import AnomalyScore

        original_prediction = None
        if request.original_score is not None:
            original_prediction = AnomalyScore(value=request.original_score)

        domain_request = SubmitFeedbackRequest(
            session_id=request.session_id,
            sample_id=request.sample_id,
            annotator_id=request.annotator_id,
            feedback_type=_convert_to_feedback_type(request.feedback_type),
            feedback_value=request.feedback_value,
            confidence=_convert_to_feedback_confidence(request.confidence),
            original_prediction=original_prediction,
            time_spent_seconds=request.time_spent_seconds,
            metadata=request.metadata,
        )

        # Execute use case
        response = use_case.submit_feedback(domain_request)

        # Convert to API response
        return {
            "feedback_id": response.feedback_id,
            "session_id": response.session_id,
            "feedback_summary": response.feedback_summary,
            "quality_assessment": response.quality_assessment,
            "next_recommendations": response.next_recommendations,
        }

    except ValueError as e:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Invalid feedback: {str(e)}",
        )
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Internal error: {str(e)}",
        )


@router.get(
    "/sessions/{session_id}/status",
    response_model=dict,
    status_code=status.HTTP_200_OK,
    summary="Get session status",
    description="""
    Get current status and progress of an active learning session.

    **Returns:**
    - Session status and progress metrics
    - Feedback quality indicators
    - Recent session activity
    - Completion percentage and timing

    Use this endpoint to monitor session progress and quality.
    """,
)
async def get_session_status(
    session_id: str,
    include_details: bool = True,
    include_feedback: bool = False,
    use_case: ManageActiveLearningUseCase = Depends(get_active_learning_use_case),
) -> dict:
    """Get status of an active learning session."""
    try:
        # Create domain request
        from pynomaly.application.dto.active_learning_dto import SessionStatusRequest

        domain_request = SessionStatusRequest(
            session_id=session_id,
            include_details=include_details,
            include_feedback=include_feedback,
        )

        # Execute use case
        response = use_case.get_session_status(domain_request)

        # Convert to API response
        result = {
            "session_id": response.session_id,
            "status": response.status.value,
            "progress": response.progress,
            "quality_metrics": response.quality_metrics,
        }

        if response.recent_activity:
            result["recent_activity"] = response.recent_activity

        if response.message:
            result["message"] = response.message

        return result

    except ValueError as e:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST, detail=f"Invalid session: {str(e)}"
        )
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Internal error: {str(e)}",
        )


@router.post(
    "/sessions/{session_id}/update-model",
    response_model=dict,
    status_code=status.HTTP_200_OK,
    summary="Update model with feedback",
    description="""
    Update the anomaly detection model using collected human feedback.

    **Process:**
    1. Analyzes collected feedback patterns
    2. Calculates model update parameters
    3. Applies incremental learning updates
    4. Validates performance impact
    5. Provides recommendations for next session

    **Returns:**
    - Update statistics and performance impact
    - Feedback pattern analysis
    - Recommendations for model improvement
    - Suggestions for next active learning session
    """,
)
async def update_model(
    session_id: str,
    request: UpdateModelModel,
    background_tasks: BackgroundTasks,
    use_case: ManageActiveLearningUseCase = Depends(get_active_learning_use_case),
) -> dict:
    """Update model with collected feedback from session."""
    try:
        # Validate session ID matches
        if request.session_id != session_id:
            raise ValueError("Session ID mismatch")

        # This would typically load feedback from repository
        # For demonstration, create empty feedback list
        from pynomaly.application.dto.active_learning_dto import UpdateModelRequest

        domain_request = UpdateModelRequest(
            session_id=request.session_id,
            feedback_list=[],  # Would be loaded from repository
            learning_rate=request.learning_rate,
            validation_split=request.validation_split,
            update_strategy=request.update_strategy,
        )

        # Execute use case
        response = use_case.update_model_with_feedback(domain_request)

        # Schedule background model retraining if needed
        if response.update_statistics.get("total_corrections", 0) > 5:
            background_tasks.add_task(
                _schedule_model_retraining, session_id, response.update_statistics
            )

        # Convert to API response
        return {
            "session_id": response.session_id,
            "update_applied": response.update_applied,
            "update_statistics": response.update_statistics,
            "feedback_analysis": response.feedback_analysis,
            "performance_impact": response.performance_impact,
            "recommendations": response.recommendations,
            "next_session_suggestions": response.next_session_suggestions,
        }

    except ValueError as e:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Invalid update request: {str(e)}",
        )
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Internal error: {str(e)}",
        )


@router.get(
    "/strategies",
    response_model=dict[str, dict[str, str]],
    status_code=status.HTTP_200_OK,
    summary="Get available sampling strategies",
    description="""
    Get information about available active learning sampling strategies
    and their characteristics.
    """,
)
async def get_sampling_strategies() -> dict[str, dict[str, str]]:
    """Get available sampling strategies for active learning."""
    return {
        "uncertainty": {
            "description": "Select samples with highest prediction uncertainty",
            "best_for": "General purpose, works well for most scenarios",
            "computational_cost": "Low",
            "requires": "Only prediction scores",
        },
        "diversity": {
            "description": "Select diverse samples to cover feature space",
            "best_for": "Ensuring broad coverage of data distribution",
            "computational_cost": "Medium",
            "requires": "Feature vectors",
        },
        "margin": {
            "description": "Select samples close to decision boundary",
            "best_for": "Binary classification tasks",
            "computational_cost": "Low",
            "requires": "Prediction scores",
        },
        "committee_disagreement": {
            "description": "Select samples where ensemble models disagree",
            "best_for": "When multiple models are available",
            "computational_cost": "Low",
            "requires": "Ensemble predictions",
        },
        "expected_model_change": {
            "description": "Select samples likely to cause large model updates",
            "best_for": "Maximizing learning efficiency",
            "computational_cost": "High",
            "requires": "Model gradients (optional)",
        },
        "entropy": {
            "description": "Select samples with high prediction entropy",
            "best_for": "Multi-class scenarios",
            "computational_cost": "Low",
            "requires": "Prediction probabilities",
        },
        "random": {
            "description": "Random selection baseline",
            "best_for": "Baseline comparison",
            "computational_cost": "Very Low",
            "requires": "Nothing",
        },
    }


@router.get(
    "/feedback-types",
    response_model=dict[str, dict[str, str]],
    status_code=status.HTTP_200_OK,
    summary="Get available feedback types",
    description="""
    Get information about available feedback types for human annotation.
    """,
)
async def get_feedback_types() -> dict[str, dict[str, str]]:
    """Get available feedback types for human annotation."""
    return {
        "binary_classification": {
            "description": "True/False anomaly classification",
            "input_type": "boolean",
            "use_case": "Simple anomaly labeling",
            "example": "true (is anomaly) or false (is normal)",
        },
        "score_correction": {
            "description": "Corrected anomaly score",
            "input_type": "float (0.0 to 1.0)",
            "use_case": "Fine-grained score adjustment",
            "example": "0.85 (high anomaly probability)",
        },
        "confidence_rating": {
            "description": "Confidence in model prediction",
            "input_type": "float (0.0 to 1.0)",
            "use_case": "Assessing prediction reliability",
            "example": "0.3 (low confidence in model)",
        },
        "explanation": {
            "description": "Text explanation of reasoning",
            "input_type": "string",
            "use_case": "Capturing domain knowledge",
            "example": "Unusual pattern in sensor readings",
        },
        "feature_importance": {
            "description": "Important features for decision",
            "input_type": "object",
            "use_case": "Feature-level feedback",
            "example": "{'temperature': 0.8, 'pressure': 0.3}",
        },
    }


async def _schedule_model_retraining(session_id: str, update_stats: dict):
    """Background task for scheduling model retraining."""
    # This would typically trigger a background ML pipeline
    # For now, just log the event
    print(
        f"Scheduling model retraining for session {session_id} with stats: {update_stats}"
    )


@router.delete(
    "/sessions/{session_id}",
    status_code=status.HTTP_204_NO_CONTENT,
    summary="Cancel active learning session",
    description="""
    Cancel an active learning session. This will stop the session
    and mark it as cancelled, preserving any collected feedback.
    """,
)
async def cancel_session(
    session_id: str,
    use_case: ManageActiveLearningUseCase = Depends(get_active_learning_use_case),
) -> None:
    """Cancel an active learning session."""
    try:
        # This would typically call a cancel method on the use case
        # For now, just validate the session exists
        if not session_id:
            raise ValueError("Session ID cannot be empty")

        # Session cancellation logic would go here

    except ValueError as e:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST, detail=f"Invalid session: {str(e)}"
        )
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Internal error: {str(e)}",
        )
