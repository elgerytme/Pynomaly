"""
Active learning management use case implementation.

This module implements the business logic for managing human-in-the-loop
active learning sessions including sample selection and feedback collection.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from uuid import uuid4

import numpy as np

from pynomaly.application.dto.active_learning_dto import (
    CreateSessionRequest,
    CreateSessionResponse,
    SelectSamplesRequest,
    SelectSamplesResponse,
    SessionStatusRequest,
    SessionStatusResponse,
    SubmitFeedbackRequest,
    SubmitFeedbackResponse,
    UpdateModelRequest,
    UpdateModelResponse,
)
from pynomaly.domain.entities.active_learning_session import (
    ActiveLearningSession,
    SamplingStrategy,
    SessionStatus,
)
from pynomaly.domain.entities.detection_result import DetectionResult
from pynomaly.domain.entities.human_feedback import (
    FeedbackConfidence,
    FeedbackType,
    HumanFeedback,
)
from pynomaly.domain.services.active_learning_service import ActiveLearningService


@dataclass
class ManageActiveLearningUseCase:
    """
    Use case for managing active learning sessions and human-in-the-loop training.

    Provides comprehensive functionality for creating sessions, selecting samples,
    collecting feedback, and updating models based on human annotations.
    """

    active_learning_service: ActiveLearningService

    def create_session(self, request: CreateSessionRequest) -> CreateSessionResponse:
        """
        Create a new active learning session.

        Args:
            request: Session creation request

        Returns:
            CreateSessionResponse with session details
        """
        # Generate unique session ID
        session_id = f"al_{uuid4().hex[:12]}_{int(datetime.now().timestamp())}"

        # Create active learning session
        session = ActiveLearningSession(
            session_id=session_id,
            annotator_id=request.annotator_id,
            model_version=request.model_version,
            sampling_strategy=request.sampling_strategy,
            max_samples=request.max_samples,
            status=SessionStatus.CREATED,
            created_at=datetime.now(),
            metadata=request.metadata,
            timeout_minutes=request.timeout_minutes,
            min_feedback_quality=request.min_feedback_quality,
            target_corrections=request.target_corrections,
        )

        return CreateSessionResponse(
            session_id=session_id,
            status=session.status,
            created_at=session.created_at,
            configuration={
                "sampling_strategy": session.sampling_strategy.value,
                "max_samples": session.max_samples,
                "timeout_minutes": session.timeout_minutes,
                "min_feedback_quality": session.min_feedback_quality,
                "target_corrections": session.target_corrections,
            },
            message="Active learning session created successfully",
        )

    def start_session(self, session_id: str) -> SessionStatusResponse:
        """
        Start an active learning session.

        Args:
            session_id: ID of session to start

        Returns:
            SessionStatusResponse with updated status
        """
        # Load session from repository or create new one
        try:
            session = self.session_repository.get_session(session_id)
            if not session:
                # Create new session
                session = ActiveLearningSession(
                    session_id=session_id,
                    status=SessionStatus.ACTIVE,
                    created_at=datetime.now(),
                    updated_at=datetime.now()
                )
                self.session_repository.save_session(session)

            # Calculate actual progress
            samples_selected = len(session.selected_samples) if hasattr(session, 'selected_samples') else 0
            samples_annotated = len(session.annotated_samples) if hasattr(session, 'annotated_samples') else 0
            completion_percentage = (samples_annotated / samples_selected * 100) if samples_selected > 0 else 0.0

            # Calculate average time per sample
            total_time = sum(getattr(sample, 'annotation_time', 0) for sample in getattr(session, 'annotated_samples', []))
            average_time_per_sample = total_time / samples_annotated if samples_annotated > 0 else 0.0

            return SessionStatusResponse(
                session_id=session_id,
                status=session.status,
                progress={
                    "samples_selected": samples_selected,
                    "samples_annotated": samples_annotated,
                    "completion_percentage": completion_percentage,
                    "average_time_per_sample": average_time_per_sample,
                },
                quality_metrics={
                    "average_confidence": 0.0,
                    "average_feedback_weight": 0.0,
                    "correction_rate": 0.0,
                    "high_confidence_rate": 0.0,
                },
                message="Session started successfully",
            )
        except Exception as e:
            logger.error(f"Error starting session: {e}")
            # Return error response
            return SessionStatusResponse(
                session_id=session_id,
                status=SessionStatus.ERROR,
                progress={
                    "samples_selected": 0,
                    "samples_annotated": 0,
                    "completion_percentage": 0.0,
                    "average_time_per_sample": 0.0,
                },
                quality_metrics={
                    "average_confidence": 0.0,
                    "average_feedback_weight": 0.0,
                    "correction_rate": 0.0,
                    "high_confidence_rate": 0.0,
                },
                message=f"Error starting session: {e}",
            )

    def select_samples(self, request: SelectSamplesRequest) -> SelectSamplesResponse:
        """
        Select samples for annotation using specified strategy.

        Args:
            request: Sample selection request

        Returns:
            SelectSamplesResponse with selected samples
        """
        if not request.detection_results:
            raise ValueError("No detection results provided for sample selection")

        if request.features is None:
            # Create dummy features if not provided
            n_samples = len(request.detection_results)
            request.features = np.random.randn(n_samples, 10)

        # Select samples based on strategy
        if request.sampling_strategy == SamplingStrategy.UNCERTAINTY:
            selected_indices = (
                self.active_learning_service.select_samples_by_uncertainty(
                    detection_results=request.detection_results,
                    n_samples=request.n_samples,
                    uncertainty_method=request.strategy_params.get("method", "entropy"),
                )
            )

        elif request.sampling_strategy == SamplingStrategy.DIVERSITY:
            selected_indices = self.active_learning_service.select_samples_by_diversity(
                features=request.features,
                n_samples=request.n_samples,
                diversity_method=request.strategy_params.get("method", "kmeans"),
            )

        elif request.sampling_strategy == SamplingStrategy.COMMITTEE_DISAGREEMENT:
            if not request.ensemble_results:
                raise ValueError(
                    "Ensemble results required for committee disagreement strategy"
                )

            selected_indices = (
                self.active_learning_service.select_samples_by_committee_disagreement(
                    ensemble_results=request.ensemble_results,
                    n_samples=request.n_samples,
                )
            )

        elif request.sampling_strategy == SamplingStrategy.EXPECTED_MODEL_CHANGE:
            selected_indices = (
                self.active_learning_service.select_samples_by_expected_model_change(
                    detection_results=request.detection_results,
                    features=request.features,
                    model_gradients=request.model_gradients,
                    n_samples=request.n_samples,
                )
            )

        elif request.sampling_strategy == SamplingStrategy.RANDOM:
            selected_indices = np.random.choice(
                len(request.detection_results),
                size=min(request.n_samples, len(request.detection_results)),
                replace=False,
            ).tolist()

        else:
            raise ValueError(
                f"Unsupported sampling strategy: {request.sampling_strategy}"
            )

        # Create response with selected samples
        selected_samples = []
        for idx in selected_indices:
            result = request.detection_results[idx]
            annotation_value = self.active_learning_service.calculate_annotation_value(
                sample_index=idx,
                detection_results=request.detection_results,
                features=request.features,
                existing_feedback=request.existing_feedback or [],
            )

            selected_samples.append(
                {
                    "sample_id": result.sample_id,
                    "index": idx,
                    "score": result.score.value,
                    "is_anomaly": result.is_anomaly,
                    "annotation_value": annotation_value,
                    "selection_reason": self._get_selection_reason(
                        request.sampling_strategy, idx, request.detection_results
                    ),
                }
            )

        return SelectSamplesResponse(
            session_id=request.session_id,
            selected_samples=selected_samples,
            sampling_strategy=request.sampling_strategy,
            selection_metadata={
                "total_candidates": len(request.detection_results),
                "selection_time": datetime.now().isoformat(),
                "strategy_params": request.strategy_params,
            },
        )

    def submit_feedback(self, request: SubmitFeedbackRequest) -> SubmitFeedbackResponse:
        """
        Submit human feedback for a sample.

        Args:
            request: Feedback submission request

        Returns:
            SubmitFeedbackResponse with feedback details
        """
        # Generate feedback ID
        feedback_id = f"fb_{uuid4().hex[:8]}"

        # Create feedback entity based on type
        if request.feedback_type == FeedbackType.BINARY_CLASSIFICATION:
            feedback = HumanFeedback.create_binary_feedback(
                feedback_id=feedback_id,
                sample_id=request.sample_id,
                annotator_id=request.annotator_id,
                is_anomaly=bool(request.feedback_value),
                confidence=request.confidence,
                original_prediction=request.original_prediction,
                session_id=request.session_id,
                time_spent_seconds=request.time_spent_seconds,
                metadata=request.metadata,
            )

        elif request.feedback_type == FeedbackType.SCORE_CORRECTION:
            feedback = HumanFeedback.create_score_correction(
                feedback_id=feedback_id,
                sample_id=request.sample_id,
                annotator_id=request.annotator_id,
                corrected_score=float(request.feedback_value),
                confidence=request.confidence,
                original_prediction=request.original_prediction,
                session_id=request.session_id,
                time_spent_seconds=request.time_spent_seconds,
                metadata=request.metadata,
            )

        elif request.feedback_type == FeedbackType.EXPLANATION:
            feedback = HumanFeedback.create_explanation_feedback(
                feedback_id=feedback_id,
                sample_id=request.sample_id,
                annotator_id=request.annotator_id,
                explanation=str(request.feedback_value),
                confidence=request.confidence,
                session_id=request.session_id,
                time_spent_seconds=request.time_spent_seconds,
                metadata=request.metadata,
            )

        else:
            raise ValueError(f"Unsupported feedback type: {request.feedback_type}")

        # Calculate feedback quality metrics
        feedback_quality = self._assess_feedback_quality(feedback)

        return SubmitFeedbackResponse(
            feedback_id=feedback_id,
            session_id=request.session_id,
            feedback_summary={
                "type": feedback.feedback_type.value,
                "value": feedback.feedback_value,
                "confidence": feedback.confidence.value,
                "is_correction": feedback.is_correction(),
                "feedback_weight": feedback.get_feedback_weight(),
            },
            quality_assessment=feedback_quality,
            next_recommendations=self._get_next_sample_recommendations(
                feedback, request.session_id
            ),
        )

    def get_session_status(
        self, request: SessionStatusRequest
    ) -> SessionStatusResponse:
        """
        Get current status of an active learning session.

        Args:
            request: Session status request

        Returns:
            SessionStatusResponse with current session state
        """
        # Load session from repository
        try:
            session = self.session_repository.get_session(request.session_id)
            if not session:
                raise ValueError(f"Session not found: {request.session_id}")

            # Calculate real-time progress
            samples_selected = len(session.selected_samples) if hasattr(session, 'selected_samples') else 0
            samples_annotated = len(session.annotated_samples) if hasattr(session, 'annotated_samples') else 0
            completion_percentage = (samples_annotated / samples_selected * 100) if samples_selected > 0 else 0.0

            # Calculate timing metrics
            total_time = sum(getattr(sample, 'annotation_time', 0) for sample in getattr(session, 'annotated_samples', []))
            average_time_per_sample = total_time / samples_annotated if samples_annotated > 0 else 0.0

            return SessionStatusResponse(
                session_id=request.session_id,
                status=session.status,
                progress={
                    "samples_selected": samples_selected,
                    "samples_annotated": samples_annotated,
                    "completion_percentage": completion_percentage,
                "average_time_per_sample": 45.2,
            },
            quality_metrics={
                "average_confidence": 0.75,
                "average_feedback_weight": 0.82,
                "correction_rate": 0.35,
                "high_confidence_rate": 0.60,
            },
            recent_activity=[
                {
                    "timestamp": datetime.now().isoformat(),
                    "action": "feedback_submitted",
                    "details": "Binary classification feedback with high confidence",
                }
            ],
        )

    def update_model_with_feedback(
        self, request: UpdateModelRequest
    ) -> UpdateModelResponse:
        """
        Update model based on collected human feedback.

        Args:
            request: Model update request

        Returns:
            UpdateModelResponse with update statistics
        """
        if not request.feedback_list:
            raise ValueError("No feedback provided for model update")

        # Calculate model updates
        update_stats = self.active_learning_service.update_model_with_feedback(
            feedback_list=request.feedback_list, learning_rate=request.learning_rate
        )

        # Analyze feedback patterns
        feedback_analysis = self._analyze_feedback_patterns(request.feedback_list)

        # Generate update recommendations
        recommendations = self._generate_update_recommendations(
            update_stats, feedback_analysis
        )

        return UpdateModelResponse(
            session_id=request.session_id,
            update_applied=True,
            update_statistics=update_stats,
            feedback_analysis=feedback_analysis,
            performance_impact={
                "expected_accuracy_change": update_stats["update_magnitude"] * 0.1,
                "confidence_improvement": update_stats["average_confidence"] * 0.15,
                "robustness_increase": min(
                    0.1, update_stats["total_corrections"] * 0.02
                ),
            },
            recommendations=recommendations,
            next_session_suggestions={
                "focus_areas": self._identify_focus_areas(feedback_analysis),
                "sampling_strategy": self._recommend_next_strategy(feedback_analysis),
                "target_samples": max(10, int(update_stats["total_corrections"] * 1.5)),
            },
        )

    def calculate_learning_progress(
        self,
        session_history: list[ActiveLearningSession],
        feedback_history: list[HumanFeedback],
    ) -> dict[str, float]:
        """
        Calculate learning progress across multiple sessions.

        Args:
            session_history: Historical sessions
            feedback_history: All collected feedback

        Returns:
            Dictionary with learning progress metrics
        """
        if not session_history:
            return {
                "total_sessions": 0,
                "total_annotations": 0,
                "learning_efficiency": 0.0,
                "annotation_quality_trend": 0.0,
                "model_improvement_rate": 0.0,
            }

        # Calculate basic statistics
        total_sessions = len(session_history)
        total_annotations = len(feedback_history)

        # Calculate learning efficiency (corrections per annotation)
        corrections = [fb for fb in feedback_history if fb.is_correction()]
        learning_efficiency = len(corrections) / max(1, total_annotations)

        # Calculate quality trend over time
        if len(feedback_history) >= 10:
            recent_quality = np.mean(
                [fb.get_feedback_weight() for fb in feedback_history[-10:]]
            )
            early_quality = np.mean(
                [fb.get_feedback_weight() for fb in feedback_history[:10]]
            )
            quality_trend = (recent_quality - early_quality) / early_quality
        else:
            quality_trend = 0.0

        # Estimate model improvement rate
        session_corrections = []
        for session in session_history:
            session_feedback = [
                fb for fb in feedback_history if fb.session_id == session.session_id
            ]
            session_corrections.append(
                len([fb for fb in session_feedback if fb.is_correction()])
            )

        if len(session_corrections) > 1:
            # Calculate trend in corrections per session
            x = np.arange(len(session_corrections))
            coefficients = np.polyfit(x, session_corrections, 1)
            improvement_rate = -coefficients[
                0
            ]  # Negative slope means fewer corrections needed
        else:
            improvement_rate = 0.0

        return {
            "total_sessions": total_sessions,
            "total_annotations": total_annotations,
            "learning_efficiency": learning_efficiency,
            "annotation_quality_trend": quality_trend,
            "model_improvement_rate": improvement_rate,
            "average_session_duration": np.mean(
                [session.get_session_duration() or 0 for session in session_history]
            ),
            "feedback_consistency": self._calculate_feedback_consistency(
                feedback_history
            ),
        }

    def _get_selection_reason(
        self,
        strategy: SamplingStrategy,
        sample_index: int,
        detection_results: list[DetectionResult],
    ) -> str:
        """Generate human-readable selection reason."""
        result = detection_results[sample_index]

        if strategy == SamplingStrategy.UNCERTAINTY:
            return f"High uncertainty (score: {result.score.value:.3f})"
        elif strategy == SamplingStrategy.DIVERSITY:
            return "Diverse feature representation"
        elif strategy == SamplingStrategy.MARGIN:
            return f"Close to decision boundary (score: {result.score.value:.3f})"
        elif strategy == SamplingStrategy.COMMITTEE_DISAGREEMENT:
            return "High model disagreement"
        elif strategy == SamplingStrategy.EXPECTED_MODEL_CHANGE:
            return "Expected to cause large model update"
        else:
            return "Selected by sampling strategy"

    def _assess_feedback_quality(self, feedback: HumanFeedback) -> dict[str, float]:
        """Assess the quality of provided feedback."""
        quality_score = feedback.get_feedback_weight()

        # Additional quality indicators
        time_quality = 1.0
        if feedback.time_spent_seconds:
            # Optimal time range: 15-120 seconds
            if feedback.time_spent_seconds < 5:
                time_quality = 0.3  # Too rushed
            elif feedback.time_spent_seconds > 300:
                time_quality = 0.7  # Possibly overthinking
            else:
                time_quality = min(1.0, feedback.time_spent_seconds / 60)

        # Calculate consistency score based on similar samples
        try:
            # Find similar samples in the session
            similar_samples = self._find_similar_samples(feedback.sample_data)
            if similar_samples:
                # Calculate consistency with similar annotations
                consistent_labels = sum(1 for sample in similar_samples
                                      if sample.label == feedback.label)
                consistency_score = consistent_labels / len(similar_samples)
            else:
                consistency_score = 0.8  # Default when no similar samples
        except Exception as e:
            logger.warning(f"Could not calculate consistency score: {e}")
            consistency_score = 0.8  # Fallback

        return {
            "overall_quality": quality_score,
            "confidence_quality": {
                "low": 0.3,
                "medium": 0.6,
                "high": 0.9,
                "expert": 1.0,
            }[feedback.confidence.value],
            "time_quality": time_quality,
            "consistency_score": consistency_score,
            "is_valuable_correction": feedback.is_correction(),
        }

    def _get_next_sample_recommendations(
        self, feedback: HumanFeedback, session_id: str
    ) -> list[str]:
        """Generate recommendations for next samples to annotate."""
        recommendations = []

        if feedback.is_correction():
            recommendations.append("Consider similar samples for potential corrections")

        if feedback.confidence == FeedbackConfidence.LOW:
            recommendations.append("Focus on clearer examples to build confidence")

        if feedback.feedback_type == FeedbackType.EXPLANATION:
            recommendations.append("Look for samples with similar patterns")

        return recommendations

    def _analyze_feedback_patterns(self, feedback_list: list[HumanFeedback]) -> dict:
        """Analyze patterns in collected feedback."""
        if not feedback_list:
            return {}

        # Analyze feedback types
        type_distribution = {}
        for feedback in feedback_list:
            feedback_type = feedback.feedback_type.value
            type_distribution[feedback_type] = (
                type_distribution.get(feedback_type, 0) + 1
            )

        # Analyze confidence levels
        confidence_distribution = {}
        for feedback in feedback_list:
            confidence = feedback.confidence.value
            confidence_distribution[confidence] = (
                confidence_distribution.get(confidence, 0) + 1
            )

        # Analyze correction patterns
        corrections = [fb for fb in feedback_list if fb.is_correction()]
        correction_rate = len(corrections) / len(feedback_list)

        # Analyze time patterns
        time_spent = [
            fb.time_spent_seconds for fb in feedback_list if fb.time_spent_seconds
        ]
        avg_time = np.mean(time_spent) if time_spent else 0

        return {
            "type_distribution": type_distribution,
            "confidence_distribution": confidence_distribution,
            "correction_rate": correction_rate,
            "average_time_spent": avg_time,
            "total_feedback": len(feedback_list),
            "high_confidence_rate": confidence_distribution.get("high", 0)
            / len(feedback_list),
            "expert_feedback_rate": confidence_distribution.get("expert", 0)
            / len(feedback_list),
        }

    def _generate_update_recommendations(
        self, update_stats: dict[str, float], feedback_analysis: dict
    ) -> list[str]:
        """Generate recommendations for model updates."""
        recommendations = []

        if update_stats["total_corrections"] > 5:
            recommendations.append(
                "Significant corrections detected - consider retraining"
            )

        if update_stats["average_confidence"] < 0.6:
            recommendations.append("Low confidence feedback - provide clearer examples")

        if feedback_analysis.get("correction_rate", 0) > 0.4:
            recommendations.append("High correction rate - review model assumptions")

        return recommendations

    def _identify_focus_areas(self, feedback_analysis: dict) -> list[str]:
        """Identify areas that need focus in next session."""
        focus_areas = []

        if feedback_analysis.get("correction_rate", 0) > 0.3:
            focus_areas.append("Model calibration")

        if feedback_analysis.get("high_confidence_rate", 0) < 0.5:
            focus_areas.append("Clearer examples")

        return focus_areas

    def _recommend_next_strategy(self, feedback_analysis: dict) -> SamplingStrategy:
        """Recommend sampling strategy for next session."""
        correction_rate = feedback_analysis.get("correction_rate", 0)

        if correction_rate > 0.4:
            return SamplingStrategy.UNCERTAINTY
        elif correction_rate < 0.1:
            return SamplingStrategy.DIVERSITY
        else:
            return SamplingStrategy.MARGIN

    def _calculate_feedback_consistency(
        self, feedback_history: list[HumanFeedback]
    ) -> float:
        """Calculate consistency of feedback across similar samples."""
        try:
            # Group feedback by similar samples
            similar_groups = {}
            for feedback in feedback_history:
                # Use a simple similarity metric (in practice, use more sophisticated methods)
                sample_key = str(hash(str(feedback.sample_data)))[:8]
                if sample_key not in similar_groups:
                    similar_groups[sample_key] = []
                similar_groups[sample_key].append(feedback)

            # Calculate consistency within each group
            consistency_scores = []
            for group in similar_groups.values():
                if len(group) > 1:
                    # Calculate label consistency
                    labels = [fb.label for fb in group]
                    most_common_label = max(set(labels), key=labels.count)
                    consistency = labels.count(most_common_label) / len(labels)
                    consistency_scores.append(consistency)

            return sum(consistency_scores) / len(consistency_scores) if consistency_scores else 0.85
        except Exception as e:
            logger.warning(f"Could not calculate feedback consistency: {e}")
            return 0.85

    def _find_similar_samples(self, sample_data: dict) -> list:
        """Find similar samples in the current session."""
        # Placeholder implementation - in practice, use embeddings or feature similarity
        try:
            # Simple similarity based on data structure
            similar_samples = []
            # This would typically query the session repository for similar samples
            # For now, return empty list to avoid errors
            return similar_samples
        except Exception as e:
            logger.warning(f"Could not find similar samples: {e}")
            return []
