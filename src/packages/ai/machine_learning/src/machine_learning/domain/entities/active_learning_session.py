"""
Active Learning Session entity for machine learning domain.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from enum import Enum
from typing import Any
from uuid import uuid4


class SessionStatus(Enum):
    """Status of an active learning session."""
    CREATED = "created"
    ACTIVE = "active"
    PAUSED = "paused"
    COMPLETED = "completed"
    EXPIRED = "expired"
    CANCELLED = "cancelled"


class SamplingStrategy(Enum):
    """Strategies for selecting samples in active learning."""
    RANDOM = "random"
    UNCERTAINTY = "uncertainty"
    QUERY_BY_COMMITTEE = "query_by_committee"
    COMMITTEE_DISAGREEMENT = "committee_disagreement"
    EXPECTED_MODEL_CHANGE = "expected_model_change"
    EXPECTED_ERROR_REDUCTION = "expected_error_reduction"
    DIVERSITY = "diversity"
    DENSITY_WEIGHTED = "density_weighted"
    MARGIN = "margin"


@dataclass
class ActiveLearningSession:
    """
    Represents an active learning session.
    
    Attributes:
        session_id: Unique session identifier
        annotator_id: ID of the human annotator
        model_version: Version of the model being improved
        sampling_strategy: Strategy for selecting samples
        status: Current session status
        created_at: Session creation timestamp
        updated_at: Last update timestamp
        max_samples: Maximum number of samples to annotate
        timeout_minutes: Session timeout in minutes
        min_feedback_quality: Minimum required feedback quality
        target_corrections: Target number of corrections
        metadata: Additional session metadata
        feedback_count: Number of feedback items collected
        samples_annotated: Number of samples annotated
        quality_score: Current quality score
    """
    
    session_id: str
    annotator_id: str
    model_version: str
    sampling_strategy: SamplingStrategy
    status: SessionStatus
    created_at: datetime
    updated_at: datetime
    max_samples: int = 20
    timeout_minutes: int | None = 60
    min_feedback_quality: float = 0.7
    target_corrections: int | None = None
    metadata: dict[str, Any] = None
    feedback_count: int = 0
    samples_annotated: int = 0
    quality_score: float = 0.0
    selected_samples: list[Any] = None
    annotated_samples: list[Any] = None
    
    def __post_init__(self) -> None:
        """Initialize default values."""
        if self.metadata is None:
            self.metadata = {}
        if self.selected_samples is None:
            self.selected_samples = []
        if self.annotated_samples is None:
            self.annotated_samples = []
    
    @classmethod
    def create(
        cls,
        annotator_id: str,
        model_version: str,
        sampling_strategy: SamplingStrategy,
        max_samples: int = 20,
        timeout_minutes: int | None = 60,
        min_feedback_quality: float = 0.7,
        target_corrections: int | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> ActiveLearningSession:
        """Create a new active learning session."""
        now = datetime.now()
        return cls(
            session_id=f"al_{uuid4().hex[:12]}_{int(datetime.now().timestamp())}",
            annotator_id=annotator_id,
            model_version=model_version,
            sampling_strategy=sampling_strategy,
            status=SessionStatus.CREATED,
            created_at=now,
            updated_at=now,
            max_samples=max_samples,
            timeout_minutes=timeout_minutes,
            min_feedback_quality=min_feedback_quality,
            target_corrections=target_corrections,
            metadata=metadata or {},
        )
    
    def is_active(self) -> bool:
        """Check if session is active."""
        return self.status == SessionStatus.ACTIVE
    
    def is_completed(self) -> bool:
        """Check if session is completed."""
        return self.status in [SessionStatus.COMPLETED, SessionStatus.EXPIRED, SessionStatus.CANCELLED]
    
    def can_accept_feedback(self) -> bool:
        """Check if session can accept more feedback."""
        return (
            self.status in [SessionStatus.CREATED, SessionStatus.ACTIVE] and
            self.feedback_count < self.max_samples
        )
    
    def update_status(self, new_status: SessionStatus) -> None:
        """Update session status."""
        self.status = new_status
        self.updated_at = datetime.now()
    
    def add_feedback(self) -> None:
        """Record that feedback was added."""
        self.feedback_count += 1
        self.samples_annotated += 1
        self.updated_at = datetime.now()
        
        # Auto-complete if max samples reached
        if self.feedback_count >= self.max_samples:
            self.status = SessionStatus.COMPLETED
    
    def update_quality_score(self, quality_score: float) -> None:
        """Update the quality score."""
        self.quality_score = quality_score
        self.updated_at = datetime.now()
    
    def get_session_duration(self) -> float | None:
        """Get session duration in minutes."""
        if self.status in [SessionStatus.COMPLETED, SessionStatus.EXPIRED, SessionStatus.CANCELLED]:
            return (self.updated_at - self.created_at).total_seconds() / 60.0
        elif self.status in [SessionStatus.ACTIVE, SessionStatus.PAUSED]:
            return (datetime.now() - self.created_at).total_seconds() / 60.0
        return None