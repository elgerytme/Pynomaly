"""
Data Transfer Objects for active learning operations.

This module defines the request and response DTOs for active learning
and human-in-the-loop use cases.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from typing import Any, Dict, List, Optional, Union

import numpy as np

from pynomaly.domain.entities.active_learning_session import SessionStatus, SamplingStrategy
from pynomaly.domain.entities.human_feedback import (
    HumanFeedback, 
    FeedbackType, 
    FeedbackConfidence
)
from pynomaly.domain.entities.detection_result import DetectionResult
from pynomaly.domain.value_objects.anomaly_score import AnomalyScore


@dataclass
class CreateSessionRequest:
    """
    Request for creating a new active learning session.
    
    Attributes:
        annotator_id: ID of the human annotator
        model_version: Version of the model being improved
        sampling_strategy: Strategy for selecting samples
        max_samples: Maximum number of samples to annotate
        timeout_minutes: Session timeout in minutes
        min_feedback_quality: Minimum required feedback quality
        target_corrections: Target number of corrections
        metadata: Additional session metadata
    """
    
    annotator_id: str
    model_version: str
    sampling_strategy: SamplingStrategy
    max_samples: int = 20
    timeout_minutes: Optional[int] = 60
    min_feedback_quality: float = 0.7
    target_corrections: Optional[int] = None
    metadata: Dict = None
    
    def __post_init__(self) -> None:
        """Validate request parameters."""
        if not self.annotator_id:
            raise ValueError("Annotator ID cannot be empty")
        
        if not self.model_version:
            raise ValueError("Model version cannot be empty")
        
        if self.max_samples <= 0:
            raise ValueError("Max samples must be positive")
        
        if self.timeout_minutes and self.timeout_minutes <= 0:
            raise ValueError("Timeout must be positive")
        
        if not 0.0 <= self.min_feedback_quality <= 1.0:
            raise ValueError("Min feedback quality must be between 0 and 1")
        
        if self.metadata is None:
            self.metadata = {}


@dataclass
class CreateSessionResponse:
    """
    Response from creating a new active learning session.
    
    Attributes:
        session_id: Generated session ID
        status: Current session status
        created_at: Session creation timestamp
        configuration: Session configuration details
        message: Success or information message
    """
    
    session_id: str
    status: SessionStatus
    created_at: datetime
    configuration: Dict[str, Any]
    message: str


@dataclass
class SelectSamplesRequest:
    """
    Request for selecting samples for annotation.
    
    Attributes:
        session_id: Active learning session ID
        detection_results: Available detection results
        features: Feature matrix for samples
        n_samples: Number of samples to select
        sampling_strategy: Strategy for sample selection
        strategy_params: Additional parameters for strategy
        ensemble_results: Results from ensemble models (for committee disagreement)
        model_gradients: Model gradients for expected change strategy
        existing_feedback: Previously collected feedback
    """
    
    session_id: str
    detection_results: List[DetectionResult]
    n_samples: int
    sampling_strategy: SamplingStrategy
    features: Optional[np.ndarray] = None
    strategy_params: Dict[str, Any] = None
    ensemble_results: Optional[List[List[DetectionResult]]] = None
    model_gradients: Optional[np.ndarray] = None
    existing_feedback: Optional[List[HumanFeedback]] = None
    
    def __post_init__(self) -> None:
        """Validate request parameters."""
        if not self.session_id:
            raise ValueError("Session ID cannot be empty")
        
        if not self.detection_results:
            raise ValueError("Detection results cannot be empty")
        
        if self.n_samples <= 0:
            raise ValueError("Number of samples must be positive")
        
        if self.n_samples > len(self.detection_results):
            raise ValueError("Cannot select more samples than available")
        
        if self.strategy_params is None:
            self.strategy_params = {}


@dataclass
class SelectSamplesResponse:
    """
    Response from sample selection.
    
    Attributes:
        session_id: Session ID
        selected_samples: Information about selected samples
        sampling_strategy: Strategy used for selection
        selection_metadata: Additional metadata about selection
    """
    
    session_id: str
    selected_samples: List[Dict[str, Any]]
    sampling_strategy: SamplingStrategy
    selection_metadata: Dict[str, Any]


@dataclass
class SubmitFeedbackRequest:
    """
    Request for submitting human feedback.
    
    Attributes:
        session_id: Active learning session ID
        sample_id: ID of the annotated sample
        annotator_id: ID of the human annotator
        feedback_type: Type of feedback being provided
        feedback_value: The actual feedback value
        confidence: Annotator's confidence in the feedback
        original_prediction: Original model prediction for comparison
        time_spent_seconds: Time spent on annotation
        metadata: Additional feedback metadata
    """
    
    session_id: str
    sample_id: str
    annotator_id: str
    feedback_type: FeedbackType
    feedback_value: Union[bool, float, str, Dict]
    confidence: FeedbackConfidence
    original_prediction: Optional[AnomalyScore] = None
    time_spent_seconds: Optional[float] = None
    metadata: Dict = None
    
    def __post_init__(self) -> None:
        """Validate request parameters."""
        if not self.session_id:
            raise ValueError("Session ID cannot be empty")
        
        if not self.sample_id:
            raise ValueError("Sample ID cannot be empty")
        
        if not self.annotator_id:
            raise ValueError("Annotator ID cannot be empty")
        
        if self.metadata is None:
            self.metadata = {}
        
        # Validate feedback value based on type
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


@dataclass
class SubmitFeedbackResponse:
    """
    Response from feedback submission.
    
    Attributes:
        feedback_id: Generated feedback ID
        session_id: Session ID
        feedback_summary: Summary of submitted feedback
        quality_assessment: Assessment of feedback quality
        next_recommendations: Recommendations for next samples
    """
    
    feedback_id: str
    session_id: str
    feedback_summary: Dict[str, Any]
    quality_assessment: Dict[str, float]
    next_recommendations: List[str]


@dataclass
class SessionStatusRequest:
    """
    Request for session status information.
    
    Attributes:
        session_id: Active learning session ID
        include_details: Whether to include detailed information
        include_feedback: Whether to include feedback history
    """
    
    session_id: str
    include_details: bool = True
    include_feedback: bool = False
    
    def __post_init__(self) -> None:
        """Validate request parameters."""
        if not self.session_id:
            raise ValueError("Session ID cannot be empty")


@dataclass
class SessionStatusResponse:
    """
    Response with session status information.
    
    Attributes:
        session_id: Session ID
        status: Current session status
        progress: Progress statistics
        quality_metrics: Feedback quality metrics
        recent_activity: Recent session activity
        message: Status message
    """
    
    session_id: str
    status: SessionStatus
    progress: Dict[str, float]
    quality_metrics: Dict[str, float]
    recent_activity: Optional[List[Dict[str, Any]]] = None
    message: Optional[str] = None


@dataclass
class UpdateModelRequest:
    """
    Request for updating model with collected feedback.
    
    Attributes:
        session_id: Session ID
        feedback_list: List of collected feedback
        learning_rate: Learning rate for updates
        validation_split: Fraction of feedback to use for validation
        update_strategy: Strategy for applying updates
    """
    
    session_id: str
    feedback_list: List[HumanFeedback]
    learning_rate: float = 0.1
    validation_split: float = 0.2
    update_strategy: str = "incremental"
    
    def __post_init__(self) -> None:
        """Validate request parameters."""
        if not self.session_id:
            raise ValueError("Session ID cannot be empty")
        
        if not self.feedback_list:
            raise ValueError("Feedback list cannot be empty")
        
        if not 0.0 < self.learning_rate <= 1.0:
            raise ValueError("Learning rate must be between 0 and 1")
        
        if not 0.0 <= self.validation_split < 1.0:
            raise ValueError("Validation split must be between 0 and 1")


@dataclass
class UpdateModelResponse:
    """
    Response from model update.
    
    Attributes:
        session_id: Session ID
        update_applied: Whether update was successfully applied
        update_statistics: Statistics about the update
        feedback_analysis: Analysis of feedback patterns
        performance_impact: Expected performance impact
        recommendations: Recommendations for future sessions
        next_session_suggestions: Suggestions for next session
    """
    
    session_id: str
    update_applied: bool
    update_statistics: Dict[str, float]
    feedback_analysis: Dict[str, Any]
    performance_impact: Dict[str, float]
    recommendations: List[str]
    next_session_suggestions: Dict[str, Any]


@dataclass
class AnnotationTaskRequest:
    """
    Request for creating an annotation task.
    
    Attributes:
        sample_id: ID of sample to annotate
        sample_data: Raw sample data
        current_prediction: Current model prediction
        context: Additional context for annotation
        instruction: Specific instruction for annotator
        expected_time: Expected time for annotation
    """
    
    sample_id: str
    sample_data: Dict[str, Any]
    current_prediction: Dict[str, Any]
    context: Optional[Dict[str, Any]] = None
    instruction: Optional[str] = None
    expected_time: Optional[int] = None
    
    def __post_init__(self) -> None:
        """Validate request parameters."""
        if not self.sample_id:
            raise ValueError("Sample ID cannot be empty")
        
        if not self.sample_data:
            raise ValueError("Sample data cannot be empty")
        
        if not self.current_prediction:
            raise ValueError("Current prediction cannot be empty")


@dataclass
class AnnotationTaskResponse:
    """
    Response with annotation task details.
    
    Attributes:
        task_id: Generated task ID
        sample_id: Sample ID
        annotation_interface: Interface configuration
        guidance: Guidance for annotator
        shortcuts: Available keyboard shortcuts
        validation_rules: Rules for validating annotations
    """
    
    task_id: str
    sample_id: str
    annotation_interface: Dict[str, Any]
    guidance: List[str]
    shortcuts: Dict[str, str]
    validation_rules: List[str]


@dataclass
class LearningProgressRequest:
    """
    Request for learning progress analysis.
    
    Attributes:
        annotator_id: ID of annotator
        time_period_days: Time period for analysis in days
        include_sessions: Whether to include session details
        include_trends: Whether to include trend analysis
    """
    
    annotator_id: str
    time_period_days: int = 30
    include_sessions: bool = True
    include_trends: bool = True
    
    def __post_init__(self) -> None:
        """Validate request parameters."""
        if not self.annotator_id:
            raise ValueError("Annotator ID cannot be empty")
        
        if self.time_period_days <= 0:
            raise ValueError("Time period must be positive")


@dataclass
class LearningProgressResponse:
    """
    Response with learning progress analysis.
    
    Attributes:
        annotator_id: Annotator ID
        analysis_period: Period analyzed
        overall_metrics: Overall progress metrics
        session_history: Historical session data
        trends: Trend analysis
        recommendations: Recommendations for improvement
    """
    
    annotator_id: str
    analysis_period: Dict[str, str]
    overall_metrics: Dict[str, float]
    session_history: Optional[List[Dict[str, Any]]] = None
    trends: Optional[Dict[str, Any]] = None
    recommendations: Optional[List[str]] = None


@dataclass
class BatchFeedbackRequest:
    """
    Request for submitting multiple feedback items.
    
    Attributes:
        session_id: Session ID
        feedback_batch: List of feedback submissions
        validate_consistency: Whether to validate consistency
        auto_quality_check: Whether to perform automatic quality checks
    """
    
    session_id: str
    feedback_batch: List[SubmitFeedbackRequest]
    validate_consistency: bool = True
    auto_quality_check: bool = True
    
    def __post_init__(self) -> None:
        """Validate request parameters."""
        if not self.session_id:
            raise ValueError("Session ID cannot be empty")
        
        if not self.feedback_batch:
            raise ValueError("Feedback batch cannot be empty")
        
        # Validate that all feedback belongs to the same session
        for feedback_req in self.feedback_batch:
            if feedback_req.session_id != self.session_id:
                raise ValueError("All feedback must belong to the same session")


@dataclass
class BatchFeedbackResponse:
    """
    Response from batch feedback submission.
    
    Attributes:
        session_id: Session ID
        processed_count: Number of feedback items processed
        feedback_ids: List of generated feedback IDs
        batch_quality: Overall batch quality metrics
        consistency_analysis: Analysis of feedback consistency
        warnings: Any warnings about the batch
    """
    
    session_id: str
    processed_count: int
    feedback_ids: List[str]
    batch_quality: Dict[str, float]
    consistency_analysis: Dict[str, Any]
    warnings: List[str]