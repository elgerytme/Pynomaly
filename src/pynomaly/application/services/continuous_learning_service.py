"""Continuous learning service for autonomous model adaptation."""

from __future__ import annotations

import asyncio
import json
import logging
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
from uuid import UUID, uuid4

import numpy as np
from sklearn.base import BaseEstimator

from pynomaly.domain.entities.continuous_learning import (
    LearningSession, ModelAdaptation, PerformanceBaseline, ConvergenceCriteria,
    LearningStrategy, EvolutionTrigger, UserFeedback, FeedbackType,
    PerformanceDelta, KnowledgeTransferMetrics, ModelEvolution
)
from pynomaly.domain.entities.drift_detection import DriftEvent


logger = logging.getLogger(__name__)


class ContinuousLearningError(Exception):
    """Base exception for continuous learning errors."""
    pass


class LearningSessionNotFoundError(ContinuousLearningError):
    """Learning session not found error."""
    pass


class ModelAdaptationError(ContinuousLearningError):
    """Model adaptation error."""
    pass


class FeedbackProcessingError(ContinuousLearningError):
    """Feedback processing error."""
    pass


@dataclass
class LearningConfiguration:
    """Configuration for continuous learning."""
    learning_strategy: LearningStrategy = LearningStrategy.INCREMENTAL
    learning_rate: float = 0.01
    batch_size: int = 1000
    max_memory_samples: int = 100000
    adaptation_frequency: int = 10  # Adapt every N batches
    performance_window_size: int = 100
    convergence_criteria: Optional[ConvergenceCriteria] = None
    enable_active_learning: bool = True
    uncertainty_threshold: float = 0.3
    feedback_weight: float = 1.0
    
    def __post_init__(self):
        """Validate configuration."""
        if not (0.0 < self.learning_rate <= 1.0):
            raise ValueError("Learning rate must be between 0.0 and 1.0")
        if self.batch_size <= 0:
            raise ValueError("Batch size must be positive")
        if self.convergence_criteria is None:
            self.convergence_criteria = ConvergenceCriteria()


@dataclass
class FeedbackBatch:
    """Batch of user feedback for processing."""
    feedback_items: List[UserFeedback]
    batch_id: UUID = field(default_factory=uuid4)
    received_at: datetime = field(default_factory=datetime.utcnow)
    source: str = "user_interface"
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def get_high_confidence_feedback(self) -> List[UserFeedback]:
        """Get high confidence feedback items."""
        return [f for f in self.feedback_items if f.is_high_confidence()]
    
    def get_corrections(self) -> List[UserFeedback]:
        """Get feedback items that are corrections."""
        return [f for f in self.feedback_items if f.is_correction()]
    
    def get_feedback_by_type(self, feedback_type: FeedbackType) -> List[UserFeedback]:
        """Get feedback items of specific type."""
        return [f for f in self.feedback_items if f.feedback_type == feedback_type]


@dataclass
class ModelUpdateResult:
    """Result of model update operation."""
    success: bool
    adaptation: Optional[ModelAdaptation] = None
    performance_change: Optional[PerformanceDelta] = None
    samples_processed: int = 0
    processing_time_seconds: float = 0.0
    error_message: Optional[str] = None
    convergence_status: str = "continuing"  # continuing, converged, diverged
    
    def was_beneficial(self) -> bool:
        """Check if update was beneficial."""
        if not self.success or not self.performance_change:
            return False
        return self.performance_change.is_improvement()


@dataclass
class AdaptationAssessment:
    """Assessment of adaptation effectiveness."""
    session_id: UUID
    assessment_timestamp: datetime = field(default_factory=datetime.utcnow)
    overall_effectiveness: float = 0.0
    performance_trend: List[float] = field(default_factory=list)
    adaptation_success_rate: float = 0.0
    learning_velocity: float = 0.0
    stability_score: float = 1.0
    convergence_probability: float = 0.0
    recommendations: List[str] = field(default_factory=list)
    
    def is_learning_effectively(self) -> bool:
        """Check if learning is effective."""
        return (self.overall_effectiveness > 0.6 and 
                self.adaptation_success_rate > 0.5 and
                self.stability_score > 0.3)
    
    def needs_intervention(self) -> bool:
        """Check if manual intervention is needed."""
        return (self.overall_effectiveness < 0.3 or
                self.adaptation_success_rate < 0.2 or
                self.stability_score < 0.1)


class ContinuousLearningService:
    """Service for managing continuous learning processes.
    
    This service orchestrates the continuous learning lifecycle including:
    - Learning session management
    - Model adaptation based on new data and feedback
    - Performance monitoring and assessment
    - Convergence detection and optimization
    """
    
    def __init__(
        self,
        storage_path: Path,
        default_config: Optional[LearningConfiguration] = None
    ):
        """Initialize continuous learning service.
        
        Args:
            storage_path: Path for storing learning sessions and models
            default_config: Default learning configuration
        """
        self.storage_path = Path(storage_path)
        self.storage_path.mkdir(parents=True, exist_ok=True)
        
        self.default_config = default_config or LearningConfiguration()
        
        # Active learning sessions
        self.active_sessions: Dict[UUID, LearningSession] = {}
        
        # Model registry for adaptive models
        self.adaptive_models: Dict[UUID, BaseEstimator] = {}
        
        # Feedback processors
        self.feedback_processors: Dict[str, Any] = {}
        
        # Performance trackers
        self.performance_trackers: Dict[UUID, Any] = {}
        
        # Background tasks
        self._background_tasks: List[asyncio.Task] = []
        
        # Load existing sessions
        asyncio.create_task(self._load_sessions())
    
    async def _load_sessions(self) -> None:
        """Load existing learning sessions from storage."""
        session_files = self.storage_path.glob("session_*.json")
        
        for session_file in session_files:
            try:
                session = await self._load_session_from_file(session_file)
                if session.is_active:
                    self.active_sessions[session.session_id] = session
                    logger.info(f"Loaded active learning session {session.session_id}")
            except Exception as e:
                logger.warning(f"Failed to load session from {session_file}: {e}")
    
    async def initiate_learning_session(
        self,
        model_id: UUID,
        learning_config: Optional[LearningConfiguration] = None,
        initial_baseline: Optional[PerformanceBaseline] = None
    ) -> LearningSession:
        """Initiate a new continuous learning session.
        
        Args:
            model_id: ID of the model to adapt
            learning_config: Learning configuration
            initial_baseline: Initial performance baseline
            
        Returns:
            Created learning session
        """
        config = learning_config or self.default_config
        
        # Create learning session
        session = LearningSession(
            model_version_id=model_id,
            learning_strategy=config.learning_strategy,
            performance_baseline=initial_baseline,
            convergence_criteria=config.convergence_criteria,
            learning_rate=config.learning_rate
        )
        
        # Initialize adaptive model
        if model_id not in self.adaptive_models:
            adaptive_model = await self._create_adaptive_model(model_id, config)
            self.adaptive_models[model_id] = adaptive_model
        
        # Initialize performance tracker
        self.performance_trackers[session.session_id] = PerformanceTracker(
            window_size=config.performance_window_size
        )
        
        # Initialize feedback processor
        self.feedback_processors[session.session_id] = FeedbackProcessor(
            session.session_id, config
        )
        
        # Store session
        self.active_sessions[session.session_id] = session
        await self._save_session(session)
        
        # Start background monitoring
        if config.adaptation_frequency > 0:
            task = asyncio.create_task(
                self._monitor_session(session.session_id, config)
            )
            self._background_tasks.append(task)
        
        logger.info(f"Initiated learning session {session.session_id} for model {model_id}")
        return session
    
    async def process_feedback_batch(
        self,
        session_id: UUID,
        feedback_data: FeedbackBatch
    ) -> ModelUpdateResult:
        """Process a batch of user feedback for model improvement.
        
        Args:
            session_id: Learning session ID
            feedback_data: Batch of feedback data
            
        Returns:
            Model update result
            
        Raises:
            LearningSessionNotFoundError: If session not found
            FeedbackProcessingError: If feedback processing fails
        """
        if session_id not in self.active_sessions:
            raise LearningSessionNotFoundError(f"Session {session_id} not found")
        
        session = self.active_sessions[session_id]
        
        try:
            start_time = datetime.utcnow()
            
            # Process feedback through processor
            processor = self.feedback_processors[session_id]
            processed_feedback = await processor.process_feedback_batch(feedback_data)
            
            # Get adaptive model
            adaptive_model = self.adaptive_models[session.model_version_id]
            
            # Apply feedback to model
            performance_before = await self._get_current_performance(session_id)
            
            adaptation_result = await self._apply_feedback_to_model(
                adaptive_model, processed_feedback, session
            )
            
            performance_after = await self._get_current_performance(session_id)
            
            # Calculate performance change
            performance_change = PerformanceDelta(
                overall_improvement=performance_after.get_performance_score() - 
                                  performance_before.get_performance_score(),
                sample_size=len(feedback_data.feedback_items),
                statistical_significance=await self._test_significance(
                    performance_before, performance_after
                )
            )
            
            # Create adaptation record
            adaptation = ModelAdaptation(
                trigger=EvolutionTrigger.USER_FEEDBACK,
                adaptation_type="feedback_integration",
                performance_before=performance_before.compare_with(performance_before),
                performance_after=performance_after.compare_with(performance_before),
                samples_processed=len(feedback_data.feedback_items),
                adaptation_time_seconds=(datetime.utcnow() - start_time).total_seconds(),
                success=adaptation_result.success,
                error_message=adaptation_result.error_message
            )
            
            # Update session
            session.add_adaptation(adaptation)
            await self._save_session(session)
            
            # Update performance tracker
            tracker = self.performance_trackers[session_id]
            tracker.add_performance_point(performance_after.get_performance_score())
            
            result = ModelUpdateResult(
                success=adaptation_result.success,
                adaptation=adaptation,
                performance_change=performance_change,
                samples_processed=len(feedback_data.feedback_items),
                processing_time_seconds=adaptation.adaptation_time_seconds,
                convergence_status=self._check_convergence_status(session)
            )
            
            logger.info(f"Processed feedback batch for session {session_id}: "
                       f"{'success' if result.success else 'failed'}")
            
            return result
            
        except Exception as e:
            error_msg = f"Failed to process feedback batch: {e}"
            logger.error(error_msg)
            
            return ModelUpdateResult(
                success=False,
                error_message=error_msg,
                processing_time_seconds=(datetime.utcnow() - start_time).total_seconds()
            )
    
    async def evaluate_adaptation_performance(
        self,
        session_id: UUID
    ) -> AdaptationAssessment:
        """Evaluate the performance of adaptation in a learning session.
        
        Args:
            session_id: Learning session ID
            
        Returns:
            Adaptation assessment
            
        Raises:
            LearningSessionNotFoundError: If session not found
        """
        if session_id not in self.active_sessions:
            raise LearningSessionNotFoundError(f"Session {session_id} not found")
        
        session = self.active_sessions[session_id]
        tracker = self.performance_trackers[session_id]
        
        # Calculate effectiveness metrics
        performance_trend = session.get_performance_trend()
        adaptation_success_rate = session.get_adaptation_success_rate()
        
        # Calculate learning velocity (rate of improvement)
        learning_velocity = self._calculate_learning_velocity(performance_trend)
        
        # Calculate stability score
        stability_score = self._calculate_stability_score(performance_trend)
        
        # Calculate overall effectiveness
        overall_effectiveness = self._calculate_overall_effectiveness(
            adaptation_success_rate, learning_velocity, stability_score
        )
        
        # Calculate convergence probability
        convergence_probability = self._estimate_convergence_probability(
            session, performance_trend
        )
        
        # Generate recommendations
        recommendations = self._generate_learning_recommendations(
            session, overall_effectiveness, stability_score
        )
        
        assessment = AdaptationAssessment(
            session_id=session_id,
            overall_effectiveness=overall_effectiveness,
            performance_trend=performance_trend,
            adaptation_success_rate=adaptation_success_rate,
            learning_velocity=learning_velocity,
            stability_score=stability_score,
            convergence_probability=convergence_probability,
            recommendations=recommendations
        )
        
        logger.info(f"Evaluated adaptation performance for session {session_id}: "
                   f"effectiveness={overall_effectiveness:.3f}")
        
        return assessment
    
    async def adapt_to_drift(
        self,
        session_id: UUID,
        drift_event: DriftEvent,
        adaptation_data: Optional[np.ndarray] = None
    ) -> ModelUpdateResult:
        """Adapt model in response to detected drift.
        
        Args:
            session_id: Learning session ID
            drift_event: Detected drift event
            adaptation_data: Optional data for adaptation
            
        Returns:
            Model update result
        """
        if session_id not in self.active_sessions:
            raise LearningSessionNotFoundError(f"Session {session_id} not found")
        
        session = self.active_sessions[session_id]
        
        try:
            start_time = datetime.utcnow()
            
            # Get adaptive model
            adaptive_model = self.adaptive_models[session.model_version_id]
            
            # Determine adaptation strategy based on drift type and severity
            adaptation_strategy = self._select_drift_adaptation_strategy(drift_event)
            
            # Get current performance
            performance_before = await self._get_current_performance(session_id)
            
            # Apply drift adaptation
            adaptation_result = await self._apply_drift_adaptation(
                adaptive_model, drift_event, adaptation_data, adaptation_strategy
            )
            
            # Get updated performance
            performance_after = await self._get_current_performance(session_id)
            
            # Calculate performance change
            performance_change = PerformanceDelta(
                overall_improvement=performance_after.get_performance_score() - 
                                  performance_before.get_performance_score(),
                sample_size=len(adaptation_data) if adaptation_data is not None else 0
            )
            
            # Create adaptation record
            adaptation = ModelAdaptation(
                trigger=EvolutionTrigger.DRIFT_DETECTION,
                adaptation_type=f"drift_adaptation_{drift_event.drift_type.value}",
                performance_before=performance_before.compare_with(performance_before),
                performance_after=performance_after.compare_with(performance_before),
                samples_processed=len(adaptation_data) if adaptation_data is not None else 0,
                adaptation_time_seconds=(datetime.utcnow() - start_time).total_seconds(),
                success=adaptation_result.success,
                metadata={
                    "drift_id": str(drift_event.drift_id),
                    "drift_type": drift_event.drift_type.value,
                    "drift_severity": drift_event.severity.value,
                    "adaptation_strategy": adaptation_strategy
                }
            )
            
            # Update session
            session.add_adaptation(adaptation)
            await self._save_session(session)
            
            result = ModelUpdateResult(
                success=adaptation_result.success,
                adaptation=adaptation,
                performance_change=performance_change,
                samples_processed=adaptation.samples_processed,
                processing_time_seconds=adaptation.adaptation_time_seconds,
                convergence_status=self._check_convergence_status(session)
            )
            
            logger.info(f"Adapted to drift for session {session_id}: "
                       f"type={drift_event.drift_type.value}, "
                       f"success={result.success}")
            
            return result
            
        except Exception as e:
            error_msg = f"Failed to adapt to drift: {e}"
            logger.error(error_msg)
            
            return ModelUpdateResult(
                success=False,
                error_message=error_msg,
                processing_time_seconds=(datetime.utcnow() - start_time).total_seconds()
            )
    
    async def get_session_status(self, session_id: UUID) -> Dict[str, Any]:
        """Get comprehensive status of a learning session.
        
        Args:
            session_id: Learning session ID
            
        Returns:
            Session status information
        """
        if session_id not in self.active_sessions:
            raise LearningSessionNotFoundError(f"Session {session_id} not found")
        
        session = self.active_sessions[session_id]
        tracker = self.performance_trackers.get(session_id)
        
        status = {
            "session_id": str(session_id),
            "model_id": str(session.model_version_id),
            "learning_strategy": session.learning_strategy.value,
            "is_active": session.is_active,
            "started_at": session.started_at.isoformat(),
            "last_updated": session.last_updated.isoformat(),
            "duration": str(session.get_session_duration()),
            "current_epoch": session.current_epoch,
            "total_samples_processed": session.total_samples_processed,
            "total_adaptations": len(session.adaptation_history),
            "adaptation_success_rate": session.get_adaptation_success_rate(),
            "is_converged": session.is_converged(),
            "learning_rate": session.learning_rate
        }
        
        if tracker:
            recent_performance = tracker.get_recent_performance()
            status.update({
                "recent_performance": recent_performance,
                "performance_trend": tracker.get_trend(),
                "performance_variance": tracker.get_variance()
            })
        
        if session.performance_baseline:
            status["baseline_performance"] = {
                "accuracy": session.performance_baseline.accuracy,
                "established_at": session.performance_baseline.established_at.isoformat()
            }
        
        return status
    
    async def stop_learning_session(
        self,
        session_id: UUID,
        reason: str = "manual_stop"
    ) -> bool:
        """Stop an active learning session.
        
        Args:
            session_id: Learning session ID
            reason: Reason for stopping
            
        Returns:
            True if successfully stopped
        """
        if session_id not in self.active_sessions:
            return False
        
        session = self.active_sessions[session_id]
        session.is_active = False
        session.session_metadata["stop_reason"] = reason
        session.session_metadata["stopped_at"] = datetime.utcnow().isoformat()
        
        # Save final session state
        await self._save_session(session)
        
        # Clean up resources
        if session_id in self.performance_trackers:
            del self.performance_trackers[session_id]
        
        if session_id in self.feedback_processors:
            del self.feedback_processors[session_id]
        
        # Remove from active sessions
        del self.active_sessions[session_id]
        
        # Cancel background tasks
        for task in self._background_tasks:
            if not task.done():
                task.cancel()
        
        logger.info(f"Stopped learning session {session_id}: {reason}")
        return True
    
    # Private helper methods
    
    async def _create_adaptive_model(
        self,
        model_id: UUID,
        config: LearningConfiguration
    ) -> BaseEstimator:
        """Create adaptive model wrapper."""
        # This would integrate with the model registry to get the base model
        # For now, return a placeholder
        from sklearn.ensemble import IsolationForest
        return IsolationForest(contamination=0.1, random_state=42)
    
    async def _get_current_performance(self, session_id: UUID) -> PerformanceBaseline:
        """Get current performance metrics."""
        # Placeholder implementation
        return PerformanceBaseline(
            accuracy=0.85,
            precision=0.83,
            recall=0.87,
            f1_score=0.85,
            roc_auc=0.89,
            pr_auc=0.84,
            false_positive_rate=0.15,
            false_negative_rate=0.13,
            detection_rate=0.87,
            established_at=datetime.utcnow(),
            sample_size=1000
        )
    
    def _check_convergence_status(self, session: LearningSession) -> str:
        """Check convergence status of learning session."""
        if session.is_converged():
            return "converged"
        
        performance_trend = session.get_performance_trend()
        if len(performance_trend) >= 10:
            recent_trend = np.array(performance_trend[-10:])
            if np.all(np.diff(recent_trend) < -0.01):  # Consistent degradation
                return "diverged"
        
        return "continuing"
    
    def _calculate_learning_velocity(self, performance_trend: List[float]) -> float:
        """Calculate learning velocity (rate of improvement)."""
        if len(performance_trend) < 2:
            return 0.0
        
        # Simple linear regression slope
        x = np.arange(len(performance_trend))
        y = np.array(performance_trend)
        
        if len(x) > 1:
            slope = np.polyfit(x, y, 1)[0]
            return float(slope)
        
        return 0.0
    
    def _calculate_stability_score(self, performance_trend: List[float]) -> float:
        """Calculate stability score based on performance variance."""
        if len(performance_trend) < 2:
            return 1.0
        
        variance = np.var(performance_trend)
        # Convert variance to stability score (lower variance = higher stability)
        stability = 1.0 / (1.0 + variance * 10)  # Scale factor
        return float(np.clip(stability, 0.0, 1.0))
    
    def _calculate_overall_effectiveness(
        self,
        success_rate: float,
        velocity: float,
        stability: float
    ) -> float:
        """Calculate overall learning effectiveness."""
        # Weighted combination of factors
        effectiveness = (
            0.4 * success_rate +
            0.3 * max(0, velocity * 10) +  # Scale velocity
            0.3 * stability
        )
        return float(np.clip(effectiveness, 0.0, 1.0))
    
    async def _save_session(self, session: LearningSession) -> None:
        """Save learning session to storage."""
        session_file = self.storage_path / f"session_{session.session_id}.json"
        
        # Convert session to serializable format
        session_data = {
            "session_id": str(session.session_id),
            "model_version_id": str(session.model_version_id),
            "learning_strategy": session.learning_strategy.value,
            "learning_rate": session.learning_rate,
            "started_at": session.started_at.isoformat(),
            "last_updated": session.last_updated.isoformat(),
            "current_epoch": session.current_epoch,
            "total_samples_processed": session.total_samples_processed,
            "is_active": session.is_active,
            "adaptation_count": len(session.adaptation_history),
            "session_metadata": session.session_metadata
        }
        
        with open(session_file, "w") as f:
            json.dump(session_data, f, indent=2)
    
    async def _load_session_from_file(self, session_file: Path) -> LearningSession:
        """Load learning session from file."""
        with open(session_file, "r") as f:
            data = json.load(f)
        
        # Simplified loading - in practice would reconstruct full session
        session = LearningSession(
            session_id=UUID(data["session_id"]),
            model_version_id=UUID(data["model_version_id"]),
            learning_strategy=LearningStrategy(data["learning_strategy"]),
            learning_rate=data["learning_rate"],
            current_epoch=data["current_epoch"],
            total_samples_processed=data["total_samples_processed"],
            is_active=data["is_active"]
        )
        
        return session


# Supporting classes

class PerformanceTracker:
    """Tracks performance metrics over time."""
    
    def __init__(self, window_size: int = 100):
        self.window_size = window_size
        self.performance_history: List[float] = []
        self.timestamps: List[datetime] = []
    
    def add_performance_point(self, performance: float) -> None:
        """Add a performance data point."""
        self.performance_history.append(performance)
        self.timestamps.append(datetime.utcnow())
        
        # Maintain window size
        if len(self.performance_history) > self.window_size:
            self.performance_history.pop(0)
            self.timestamps.pop(0)
    
    def get_recent_performance(self, n: int = 10) -> List[float]:
        """Get recent performance values."""
        return self.performance_history[-n:] if len(self.performance_history) >= n else self.performance_history
    
    def get_trend(self) -> str:
        """Get performance trend."""
        if len(self.performance_history) < 5:
            return "insufficient_data"
        
        recent = self.performance_history[-5:]
        slope = np.polyfit(range(len(recent)), recent, 1)[0]
        
        if slope > 0.01:
            return "improving"
        elif slope < -0.01:
            return "declining"
        else:
            return "stable"
    
    def get_variance(self) -> float:
        """Get performance variance."""
        if len(self.performance_history) < 2:
            return 0.0
        return float(np.var(self.performance_history))


class FeedbackProcessor:
    """Processes user feedback for model improvement."""
    
    def __init__(self, session_id: UUID, config: LearningConfiguration):
        self.session_id = session_id
        self.config = config
        self.feedback_buffer: List[UserFeedback] = []
    
    async def process_feedback_batch(self, batch: FeedbackBatch) -> Dict[str, Any]:
        """Process a batch of user feedback."""
        processed_feedback = {
            "high_confidence": batch.get_high_confidence_feedback(),
            "corrections": batch.get_corrections(),
            "total_items": len(batch.feedback_items),
            "processing_timestamp": datetime.utcnow()
        }
        
        # Store in buffer
        self.feedback_buffer.extend(batch.feedback_items)
        
        # Maintain buffer size
        max_buffer_size = 10000
        if len(self.feedback_buffer) > max_buffer_size:
            self.feedback_buffer = self.feedback_buffer[-max_buffer_size:]
        
        return processed_feedback