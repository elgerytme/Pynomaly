"""
Active learning session entity.

This module defines the ActiveLearningSession entity that manages
human-in-the-loop learning sessions with sample selection and feedback collection.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Dict, List, Optional, Set

from pynomaly.domain.entities.human_feedback import HumanFeedback
from pynomaly.domain.entities.detection_result import DetectionResult


class SessionStatus(Enum):
    """Status of an active learning session."""
    
    CREATED = "created"           # Session created but not started
    ACTIVE = "active"             # Session in progress
    PAUSED = "paused"             # Session temporarily paused
    COMPLETED = "completed"       # Session finished successfully
    CANCELLED = "cancelled"       # Session cancelled
    EXPIRED = "expired"           # Session expired due to timeout


class SamplingStrategy(Enum):
    """Strategy for selecting samples for human annotation."""
    
    UNCERTAINTY = "uncertainty"                    # Select most uncertain samples
    DIVERSITY = "diversity"                       # Select diverse samples
    DISAGREEMENT = "disagreement"                 # Select samples with model disagreement
    MARGIN = "margin"                            # Select samples close to decision boundary
    ENTROPY = "entropy"                          # Select samples with high entropy
    COMMITTEE_DISAGREEMENT = "committee_disagreement"  # Ensemble disagreement
    EXPECTED_MODEL_CHANGE = "expected_model_change"    # Expected gradient length
    RANDOM = "random"                            # Random sampling baseline


@dataclass
class ActiveLearningSession:
    """
    Represents an active learning session for human-in-the-loop training.
    
    Manages the lifecycle of annotation sessions including sample selection,
    feedback collection, and session statistics.
    """
    
    session_id: str
    annotator_id: str
    model_version: str
    sampling_strategy: SamplingStrategy
    max_samples: int
    status: SessionStatus
    created_at: datetime
    metadata: Dict
    
    # Session state
    selected_samples: List[str] = field(default_factory=list)
    annotated_samples: Set[str] = field(default_factory=set)
    feedback_collection: List[HumanFeedback] = field(default_factory=list)
    
    # Session configuration
    timeout_minutes: Optional[int] = None
    min_feedback_quality: float = 0.7
    target_corrections: Optional[int] = None
    
    # Session statistics
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    total_time_spent_seconds: float = 0.0
    
    def __post_init__(self) -> None:
        """Validate session after initialization."""
        if not self.session_id:
            raise ValueError("Session ID cannot be empty")
        
        if not self.annotator_id:
            raise ValueError("Annotator ID cannot be empty")
        
        if not self.model_version:
            raise ValueError("Model version cannot be empty")
        
        if self.max_samples <= 0:
            raise ValueError("Max samples must be positive")
        
        if self.min_feedback_quality < 0.0 or self.min_feedback_quality > 1.0:
            raise ValueError("Min feedback quality must be between 0 and 1")
    
    def start_session(self) -> None:
        """Start the active learning session."""
        if self.status != SessionStatus.CREATED:
            raise ValueError(f"Cannot start session in {self.status} status")
        
        self.status = SessionStatus.ACTIVE
        self.started_at = datetime.now()
    
    def pause_session(self) -> None:
        """Pause the active learning session."""
        if self.status != SessionStatus.ACTIVE:
            raise ValueError(f"Cannot pause session in {self.status} status")
        
        self.status = SessionStatus.PAUSED
    
    def resume_session(self) -> None:
        """Resume a paused session."""
        if self.status != SessionStatus.PAUSED:
            raise ValueError(f"Cannot resume session in {self.status} status")
        
        self.status = SessionStatus.ACTIVE
    
    def complete_session(self) -> None:
        """Mark session as completed."""
        if self.status not in [SessionStatus.ACTIVE, SessionStatus.PAUSED]:
            raise ValueError(f"Cannot complete session in {self.status} status")
        
        self.status = SessionStatus.COMPLETED
        self.completed_at = datetime.now()
    
    def cancel_session(self) -> None:
        """Cancel the session."""
        if self.status in [SessionStatus.COMPLETED, SessionStatus.CANCELLED]:
            raise ValueError(f"Cannot cancel session in {self.status} status")
        
        self.status = SessionStatus.CANCELLED
        self.completed_at = datetime.now()
    
    def expire_session(self) -> None:
        """Mark session as expired due to timeout."""
        self.status = SessionStatus.EXPIRED
        self.completed_at = datetime.now()
    
    def add_sample_for_annotation(self, sample_id: str) -> None:
        """Add a sample to the annotation queue."""
        if len(self.selected_samples) >= self.max_samples:
            raise ValueError("Session has reached maximum number of samples")
        
        if sample_id in self.selected_samples:
            raise ValueError(f"Sample {sample_id} already selected for annotation")
        
        self.selected_samples.append(sample_id)
    
    def add_feedback(self, feedback: HumanFeedback) -> None:
        """Add human feedback to the session."""
        if feedback.sample_id not in self.selected_samples:
            raise ValueError(f"Sample {feedback.sample_id} not selected for this session")
        
        if feedback.session_id and feedback.session_id != self.session_id:
            raise ValueError("Feedback session ID does not match this session")
        
        # Update feedback with session ID if not set
        if not feedback.session_id:
            object.__setattr__(feedback, 'session_id', self.session_id)
        
        self.feedback_collection.append(feedback)
        self.annotated_samples.add(feedback.sample_id)
        
        # Update total time spent
        if feedback.time_spent_seconds:
            self.total_time_spent_seconds += feedback.time_spent_seconds
    
    def get_pending_samples(self) -> List[str]:
        """Get samples that still need annotation."""
        return [
            sample_id for sample_id in self.selected_samples
            if sample_id not in self.annotated_samples
        ]
    
    def get_progress(self) -> Dict[str, float]:
        """Get session progress statistics."""
        total_selected = len(self.selected_samples)
        total_annotated = len(self.annotated_samples)
        
        return {
            "samples_selected": total_selected,
            "samples_annotated": total_annotated,
            "completion_percentage": (
                total_annotated / total_selected * 100 if total_selected > 0 else 0
            ),
            "average_time_per_sample": (
                self.total_time_spent_seconds / total_annotated 
                if total_annotated > 0 else 0
            )
        }
    
    def get_feedback_quality_metrics(self) -> Dict[str, float]:
        """Calculate feedback quality metrics."""
        if not self.feedback_collection:
            return {
                "average_confidence": 0.0,
                "average_feedback_weight": 0.0,
                "correction_rate": 0.0,
                "high_confidence_rate": 0.0
            }
        
        confidence_values = {
            "low": 0.3, "medium": 0.6, "high": 0.9, "expert": 1.0
        }
        
        confidence_scores = [
            confidence_values[feedback.confidence.value] 
            for feedback in self.feedback_collection
        ]
        
        feedback_weights = [
            feedback.get_feedback_weight() 
            for feedback in self.feedback_collection
        ]
        
        corrections = [
            feedback for feedback in self.feedback_collection 
            if feedback.is_correction()
        ]
        
        high_confidence_feedback = [
            feedback for feedback in self.feedback_collection
            if feedback.confidence.value in ["high", "expert"]
        ]
        
        return {
            "average_confidence": sum(confidence_scores) / len(confidence_scores),
            "average_feedback_weight": sum(feedback_weights) / len(feedback_weights),
            "correction_rate": len(corrections) / len(self.feedback_collection),
            "high_confidence_rate": len(high_confidence_feedback) / len(self.feedback_collection)
        }
    
    def is_session_complete(self) -> bool:
        """Check if session meets completion criteria."""
        # Check if all samples are annotated
        if len(self.annotated_samples) >= len(self.selected_samples):
            return True
        
        # Check if target corrections reached
        if self.target_corrections:
            corrections = sum(
                1 for feedback in self.feedback_collection 
                if feedback.is_correction()
            )
            if corrections >= self.target_corrections:
                return True
        
        return False
    
    def is_session_expired(self) -> bool:
        """Check if session has expired due to timeout."""
        if not self.timeout_minutes or not self.started_at:
            return False
        
        elapsed_minutes = (datetime.now() - self.started_at).total_seconds() / 60
        return elapsed_minutes > self.timeout_minutes
    
    def get_session_duration(self) -> Optional[float]:
        """Get session duration in minutes."""
        if not self.started_at:
            return None
        
        end_time = self.completed_at or datetime.now()
        return (end_time - self.started_at).total_seconds() / 60
    
    def get_feedback_by_sample(self, sample_id: str) -> List[HumanFeedback]:
        """Get all feedback for a specific sample."""
        return [
            feedback for feedback in self.feedback_collection
            if feedback.sample_id == sample_id
        ]
    
    def get_corrections(self) -> List[HumanFeedback]:
        """Get all feedback that represents corrections."""
        return [
            feedback for feedback in self.feedback_collection
            if feedback.is_correction()
        ]
    
    def to_dict(self) -> Dict:
        """Convert session to dictionary representation."""
        return {
            "session_id": self.session_id,
            "annotator_id": self.annotator_id,
            "model_version": self.model_version,
            "sampling_strategy": self.sampling_strategy.value,
            "max_samples": self.max_samples,
            "status": self.status.value,
            "created_at": self.created_at.isoformat(),
            "started_at": self.started_at.isoformat() if self.started_at else None,
            "completed_at": self.completed_at.isoformat() if self.completed_at else None,
            "metadata": self.metadata,
            "selected_samples": self.selected_samples,
            "annotated_samples": list(self.annotated_samples),
            "feedback_count": len(self.feedback_collection),
            "timeout_minutes": self.timeout_minutes,
            "min_feedback_quality": self.min_feedback_quality,
            "target_corrections": self.target_corrections,
            "total_time_spent_seconds": self.total_time_spent_seconds,
            "progress": self.get_progress(),
            "quality_metrics": self.get_feedback_quality_metrics(),
            "session_duration_minutes": self.get_session_duration(),
            "is_complete": self.is_session_complete(),
            "is_expired": self.is_session_expired()
        }
    
    def __str__(self) -> str:
        """String representation of session."""
        return (
            f"ActiveLearningSession(id={self.session_id}, "
            f"status={self.status.value}, "
            f"samples={len(self.annotated_samples)}/{len(self.selected_samples)}, "
            f"strategy={self.sampling_strategy.value})"
        )