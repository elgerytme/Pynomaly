"""Domain entities for continuous learning framework."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from typing import Any
from uuid import UUID, uuid4

import numpy as np


class LearningStrategy(Enum):
    """Strategy for continuous learning."""

    INCREMENTAL = "incremental"
    SLIDING_WINDOW = "sliding_window"
    FORGETTING_FACTOR = "forgetting_factor"
    ACTIVE_LEARNING = "active_learning"
    ENSEMBLE_EVOLUTION = "ensemble_evolution"


class EvolutionTrigger(Enum):
    """Triggers for model evolution."""

    PERFORMANCE_DEGRADATION = "performance_degradation"
    DRIFT_DETECTION = "drift_detection"
    DATA_VOLUME_THRESHOLD = "data_volume_threshold"
    TIME_BASED = "time_based"
    USER_FEEDBACK = "user_feedback"
    BUSINESS_RULE = "business_rule"


class DriftType(Enum):
    """Types of drift in machine learning."""

    DATA_DRIFT = "data_drift"
    CONCEPT_DRIFT = "concept_drift"
    LABEL_DRIFT = "label_drift"
    PRIOR_PROBABILITY_DRIFT = "prior_probability_drift"
    FEATURE_DRIFT = "feature_drift"


class DriftSeverity(Enum):
    """Severity levels for drift events."""

    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"

    def __lt__(self, other):
        severity_order = {
            DriftSeverity.LOW: 1,
            DriftSeverity.MEDIUM: 2,
            DriftSeverity.HIGH: 3,
            DriftSeverity.CRITICAL: 4,
        }
        return severity_order[self] < severity_order[other]

    def __gt__(self, other):
        severity_order = {
            DriftSeverity.LOW: 1,
            DriftSeverity.MEDIUM: 2,
            DriftSeverity.HIGH: 3,
            DriftSeverity.CRITICAL: 4,
        }
        return severity_order[self] > severity_order[other]


class FeedbackType(Enum):
    """Types of user feedback."""

    TRUE_POSITIVE = "true_positive"
    FALSE_POSITIVE = "false_positive"
    TRUE_NEGATIVE = "true_negative"
    FALSE_NEGATIVE = "false_negative"
    SEVERITY_CORRECTION = "severity_correction"
    CONTEXT_ANNOTATION = "context_annotation"


@dataclass
class PerformanceBaseline:
    """Baseline performance metrics for learning sessions."""

    accuracy: float
    precision: float
    recall: float
    f1_score: float
    roc_auc: float
    pr_auc: float
    false_positive_rate: float
    false_negative_rate: float
    detection_rate: float
    established_at: datetime
    sample_size: int
    confidence_interval: float = 0.95

    def __post_init__(self):
        """Validate baseline metrics."""
        if not (0.0 <= self.accuracy <= 1.0):
            raise ValueError("Accuracy must be between 0.0 and 1.0")
        if not (0.0 <= self.precision <= 1.0):
            raise ValueError("Precision must be between 0.0 and 1.0")
        if not (0.0 <= self.recall <= 1.0):
            raise ValueError("Recall must be between 0.0 and 1.0")
        if self.sample_size <= 0:
            raise ValueError("Sample size must be positive")

    def get_performance_score(self) -> float:
        """Calculate overall performance score."""
        return (self.accuracy + self.precision + self.recall + self.f1_score) / 4

    def compare_with(self, other: PerformanceBaseline) -> dict[str, float]:
        """Compare with another baseline."""
        return {
            "accuracy_delta": self.accuracy - other.accuracy,
            "precision_delta": self.precision - other.precision,
            "recall_delta": self.recall - other.recall,
            "f1_delta": self.f1_score - other.f1_score,
            "overall_delta": self.get_performance_score()
            - other.get_performance_score(),
        }


@dataclass
class ConvergenceCriteria:
    """Criteria for learning convergence."""

    min_improvement_threshold: float = 0.001
    patience_epochs: int = 10
    max_epochs: int = 1000
    stability_window: int = 5
    performance_tolerance: float = 0.005
    min_sample_size: int = 100

    def __post_init__(self):
        """Validate convergence criteria."""
        if self.min_improvement_threshold <= 0:
            raise ValueError("Improvement threshold must be positive")
        if self.patience_epochs <= 0:
            raise ValueError("Patience epochs must be positive")
        if self.max_epochs <= 0:
            raise ValueError("Max epochs must be positive")

    def is_converged(
        self, performance_history: list[float], current_epoch: int
    ) -> bool:
        """Check if learning has converged."""
        if current_epoch >= self.max_epochs:
            return True

        if len(performance_history) < self.stability_window:
            return False

        # Check for stability in recent performance
        recent_performance = performance_history[-self.stability_window :]
        performance_variance = np.var(recent_performance)

        if performance_variance <= self.performance_tolerance:
            return True

        # Check for lack of improvement
        if len(performance_history) >= self.patience_epochs:
            recent_max = max(performance_history[-self.patience_epochs :])
            if len(performance_history) > self.patience_epochs:
                earlier_max = max(performance_history[: -self.patience_epochs])
                improvement = recent_max - earlier_max
                if improvement < self.min_improvement_threshold:
                    return True

        return False


@dataclass
class ModelAdaptation:
    """Represents a single model adaptation event."""

    adaptation_id: UUID = field(default_factory=uuid4)
    timestamp: datetime = field(default_factory=datetime.utcnow)
    trigger: EvolutionTrigger = EvolutionTrigger.PERFORMANCE_DEGRADATION
    adaptation_type: str = "incremental_update"
    performance_before: dict[str, float] | None = None
    performance_after: dict[str, float] | None = None
    samples_processed: int = 0
    adaptation_time_seconds: float = 0.0
    success: bool = True
    error_message: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def get_performance_improvement(self) -> float | None:
        """Calculate performance improvement from adaptation."""
        if not self.performance_before or not self.performance_after:
            return None

        before_score = sum(self.performance_before.values()) / len(
            self.performance_before
        )
        after_score = sum(self.performance_after.values()) / len(self.performance_after)

        return after_score - before_score

    def was_successful(self) -> bool:
        """Check if adaptation was successful."""
        if not self.success:
            return False

        improvement = self.get_performance_improvement()
        return improvement is not None and improvement >= 0


@dataclass
class LearningSession:
    """Represents a continuous learning session."""

    session_id: UUID = field(default_factory=uuid4)
    model_version_id: UUID = field(default_factory=uuid4)
    learning_strategy: LearningStrategy = LearningStrategy.INCREMENTAL
    performance_baseline: PerformanceBaseline | None = None
    convergence_criteria: ConvergenceCriteria = field(
        default_factory=ConvergenceCriteria
    )
    learning_rate: float = 0.01
    started_at: datetime = field(default_factory=datetime.utcnow)
    last_updated: datetime = field(default_factory=datetime.utcnow)
    adaptation_history: list[ModelAdaptation] = field(default_factory=list)
    current_epoch: int = 0
    total_samples_processed: int = 0
    is_active: bool = True
    session_metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        """Validate learning session parameters."""
        if not (0.0 < self.learning_rate <= 1.0):
            raise ValueError("Learning rate must be between 0.0 and 1.0")
        if self.current_epoch < 0:
            raise ValueError("Current epoch must be non-negative")

    def add_adaptation(self, adaptation: ModelAdaptation) -> None:
        """Add an adaptation event to the session."""
        self.adaptation_history.append(adaptation)
        self.last_updated = datetime.utcnow()
        self.total_samples_processed += adaptation.samples_processed
        if adaptation.trigger != EvolutionTrigger.TIME_BASED:
            self.current_epoch += 1

    def get_performance_trend(self) -> list[float]:
        """Get performance trend over time."""
        performance_scores = []
        for adaptation in self.adaptation_history:
            if adaptation.performance_after:
                score = sum(adaptation.performance_after.values()) / len(
                    adaptation.performance_after
                )
                performance_scores.append(score)
        return performance_scores

    def is_converged(self) -> bool:
        """Check if the learning session has converged."""
        performance_trend = self.get_performance_trend()
        return self.convergence_criteria.is_converged(
            performance_trend, self.current_epoch
        )

    def get_session_duration(self) -> timedelta:
        """Get total session duration."""
        return self.last_updated - self.started_at

    def get_adaptation_success_rate(self) -> float:
        """Calculate success rate of adaptations."""
        if not self.adaptation_history:
            return 0.0

        successful_adaptations = sum(
            1 for adaptation in self.adaptation_history if adaptation.was_successful()
        )
        return successful_adaptations / len(self.adaptation_history)


@dataclass
class DriftMetrics:
    """Comprehensive drift detection metrics."""

    statistical_distance: float
    p_value: float
    effect_size: float
    confidence_interval: tuple[float, float]
    power: float
    sample_size: int
    feature_importance_shift: dict[str, float] | None = None
    distribution_shift_score: float | None = None
    temporal_stability_score: float | None = None

    def __post_init__(self):
        """Validate drift metrics."""
        if not (0.0 <= self.p_value <= 1.0):
            raise ValueError("P-value must be between 0.0 and 1.0")
        if not (0.0 <= self.power <= 1.0):
            raise ValueError("Power must be between 0.0 and 1.0")
        if self.sample_size <= 0:
            raise ValueError("Sample size must be positive")

    def is_significant(self, alpha: float = 0.05) -> bool:
        """Check if drift is statistically significant."""
        return self.p_value < alpha

    def get_drift_magnitude(self) -> str:
        """Get qualitative description of drift magnitude."""
        if self.effect_size < 0.2:
            return "negligible"
        elif self.effect_size < 0.5:
            return "small"
        elif self.effect_size < 0.8:
            return "medium"
        else:
            return "large"


@dataclass
class RecommendedAction:
    """Recommended action for addressing drift or performance issues."""

    action_type: str
    priority: str  # HIGH, MEDIUM, LOW
    description: str
    estimated_effort: str  # HOURS, DAYS, WEEKS
    expected_impact: str  # HIGH, MEDIUM, LOW
    prerequisites: list[str] = field(default_factory=list)
    implementation_steps: list[str] = field(default_factory=list)
    success_criteria: list[str] = field(default_factory=list)
    risks: list[str] = field(default_factory=list)
    alternatives: list[str] = field(default_factory=list)

    def get_priority_score(self) -> int:
        """Get numeric priority score."""
        priority_map = {"HIGH": 3, "MEDIUM": 2, "LOW": 1}
        return priority_map.get(self.priority, 1)

    def get_impact_score(self) -> int:
        """Get numeric impact score."""
        impact_map = {"HIGH": 3, "MEDIUM": 2, "LOW": 1}
        return impact_map.get(self.expected_impact, 1)

    def get_urgency_score(self) -> float:
        """Calculate urgency score based on priority and impact."""
        return (self.get_priority_score() + self.get_impact_score()) / 2


@dataclass
class DriftEvent:
    """Represents a detected drift event."""

    drift_id: UUID = field(default_factory=uuid4)
    detected_at: datetime = field(default_factory=datetime.utcnow)
    drift_type: DriftType = DriftType.DATA_DRIFT
    severity: DriftSeverity = DriftSeverity.MEDIUM
    affected_features: list[str] = field(default_factory=list)
    drift_metrics: DriftMetrics | None = None
    recommended_actions: list[RecommendedAction] = field(default_factory=list)
    detection_method: str = "statistical"
    confidence: float = 0.5
    business_impact_assessment: dict[str, Any] | None = None
    resolution_status: str = "OPEN"  # OPEN, IN_PROGRESS, RESOLVED, IGNORED
    resolution_notes: str | None = None
    resolved_at: datetime | None = None

    def __post_init__(self):
        """Validate drift event."""
        if not (0.0 <= self.confidence <= 1.0):
            raise ValueError("Confidence must be between 0.0 and 1.0")

    def get_time_since_detection(self) -> timedelta:
        """Get time elapsed since drift detection."""
        return datetime.utcnow() - self.detected_at

    def is_critical(self) -> bool:
        """Check if drift event is critical."""
        return self.severity == DriftSeverity.CRITICAL

    def needs_immediate_attention(self) -> bool:
        """Check if drift needs immediate attention."""
        return (
            self.severity in [DriftSeverity.HIGH, DriftSeverity.CRITICAL]
            and self.resolution_status == "OPEN"
        )

    def add_recommended_action(self, action: RecommendedAction) -> None:
        """Add a recommended action."""
        self.recommended_actions.append(action)
        # Sort by urgency score
        self.recommended_actions.sort(key=lambda x: x.get_urgency_score(), reverse=True)

    def resolve(self, resolution_notes: str) -> None:
        """Mark drift event as resolved."""
        self.resolution_status = "RESOLVED"
        self.resolution_notes = resolution_notes
        self.resolved_at = datetime.utcnow()


@dataclass
class UserFeedback:
    """User feedback for model predictions."""

    feedback_id: UUID = field(default_factory=uuid4)
    prediction_id: UUID = field(default_factory=uuid4)
    user_id: str = "anonymous"
    feedback_type: FeedbackType = FeedbackType.TRUE_POSITIVE
    original_prediction: float = 0.0
    model_confidence: float = 0.0
    true_label: bool | None = None
    severity_score: float | None = None
    context_annotations: dict[str, Any] = field(default_factory=dict)
    feedback_timestamp: datetime = field(default_factory=datetime.utcnow)
    confidence: float = 1.0
    sample: dict[str, Any] | None = None
    explanation: str | None = None

    def __post_init__(self):
        """Validate user feedback."""
        if not (0.0 <= self.original_prediction <= 1.0):
            raise ValueError("Original prediction must be between 0.0 and 1.0")
        if not (0.0 <= self.model_confidence <= 1.0):
            raise ValueError("Model confidence must be between 0.0 and 1.0")
        if not (0.0 <= self.confidence <= 1.0):
            raise ValueError("Feedback confidence must be between 0.0 and 1.0")

    def is_correction(self) -> bool:
        """Check if feedback is a correction of model prediction."""
        return self.feedback_type in [
            FeedbackType.FALSE_POSITIVE,
            FeedbackType.FALSE_NEGATIVE,
        ]

    def is_high_confidence(self) -> bool:
        """Check if feedback has high confidence."""
        return self.confidence >= 0.8

    def get_feedback_quality_score(self) -> float:
        """Calculate quality score for this feedback."""
        quality_factors = []

        # Confidence factor
        quality_factors.append(self.confidence)

        # Consistency factor (alignment between model and user)
        if self.true_label is not None:
            predicted_anomaly = self.original_prediction > 0.5
            consistency = 1.0 if predicted_anomaly == self.true_label else 0.0
            quality_factors.append(consistency)

        # Context richness factor
        context_richness = min(1.0, len(self.context_annotations) / 5.0)
        quality_factors.append(context_richness)

        return sum(quality_factors) / len(quality_factors)


@dataclass
class PerformanceDelta:
    """Performance change between model versions."""

    metric_deltas: dict[str, float] = field(default_factory=dict)
    overall_improvement: float = 0.0
    statistical_significance: bool = False
    confidence_interval: tuple[float, float] = (0.0, 0.0)
    sample_size: int = 0
    test_duration: timedelta | None = None

    def __post_init__(self):
        """Validate performance delta."""
        if self.sample_size < 0:
            raise ValueError("Sample size must be non-negative")

    def is_improvement(self) -> bool:
        """Check if overall performance improved."""
        return self.overall_improvement > 0

    def is_significant_improvement(self) -> bool:
        """Check if improvement is statistically significant."""
        return self.is_improvement() and self.statistical_significance

    def get_improvement_magnitude(self) -> str:
        """Get qualitative description of improvement magnitude."""
        abs_improvement = abs(self.overall_improvement)
        if abs_improvement < 0.01:
            return "negligible"
        elif abs_improvement < 0.05:
            return "small"
        elif abs_improvement < 0.1:
            return "medium"
        else:
            return "large"


@dataclass
class KnowledgeTransferMetrics:
    """Metrics for knowledge transfer between models."""

    source_model_performance: dict[str, float] = field(default_factory=dict)
    target_model_performance: dict[str, float] = field(default_factory=dict)
    knowledge_retention_score: float = 0.0
    transfer_efficiency: float = 0.0
    catastrophic_forgetting_score: float = 0.0
    transfer_time_seconds: float = 0.0
    transferred_parameters: int = 0

    def __post_init__(self):
        """Validate knowledge transfer metrics."""
        if not (0.0 <= self.knowledge_retention_score <= 1.0):
            raise ValueError("Knowledge retention score must be between 0.0 and 1.0")
        if not (0.0 <= self.transfer_efficiency <= 1.0):
            raise ValueError("Transfer efficiency must be between 0.0 and 1.0")
        if self.transfer_time_seconds < 0:
            raise ValueError("Transfer time must be non-negative")

    def is_successful_transfer(self) -> bool:
        """Check if knowledge transfer was successful."""
        return (
            self.knowledge_retention_score > 0.7
            and self.catastrophic_forgetting_score < 0.3
        )

    def get_transfer_quality_score(self) -> float:
        """Calculate overall transfer quality score."""
        return (
            self.knowledge_retention_score
            + self.transfer_efficiency
            + (1 - self.catastrophic_forgetting_score)
        ) / 3


@dataclass
class ModelEvolution:
    """Tracks model evolution over time."""

    evolution_id: UUID = field(default_factory=uuid4)
    original_model_id: UUID = field(default_factory=uuid4)
    evolved_model_id: UUID = field(default_factory=uuid4)
    evolution_trigger: EvolutionTrigger = EvolutionTrigger.PERFORMANCE_DEGRADATION
    evolution_timestamp: datetime = field(default_factory=datetime.utcnow)
    performance_delta: PerformanceDelta | None = None
    knowledge_transfer_metrics: KnowledgeTransferMetrics | None = None
    evolution_metadata: dict[str, Any] = field(default_factory=dict)
    success: bool = True
    rollback_available: bool = True

    def get_evolution_summary(self) -> dict[str, Any]:
        """Get summary of evolution event."""
        summary = {
            "evolution_id": str(self.evolution_id),
            "trigger": self.evolution_trigger.value,
            "timestamp": self.evolution_timestamp.isoformat(),
            "success": self.success,
            "rollback_available": self.rollback_available,
        }

        if self.performance_delta:
            summary["performance_change"] = {
                "overall_improvement": self.performance_delta.overall_improvement,
                "magnitude": self.performance_delta.get_improvement_magnitude(),
                "significant": self.performance_delta.is_significant_improvement(),
            }

        if self.knowledge_transfer_metrics:
            summary["knowledge_transfer"] = {
                "retention_score": self.knowledge_transfer_metrics.knowledge_retention_score,
                "transfer_quality": self.knowledge_transfer_metrics.get_transfer_quality_score(),
                "successful": self.knowledge_transfer_metrics.is_successful_transfer(),
            }

        return summary

    def was_beneficial(self) -> bool:
        """Check if evolution was beneficial overall."""
        if not self.success:
            return False

        if (
            self.performance_delta
            and self.performance_delta.is_significant_improvement()
        ):
            return True

        if (
            self.knowledge_transfer_metrics
            and self.knowledge_transfer_metrics.is_successful_transfer()
        ):
            return True

        return False



@dataclass
class ContinuousLearning:
    """Main entity for continuous learning system management."""

    learning_id: UUID = field(default_factory=uuid4)
    model_id: UUID = field(default_factory=uuid4)
    strategy: LearningStrategy = LearningStrategy.INCREMENTAL
    created_at: datetime = field(default_factory=datetime.utcnow)
    last_updated: datetime = field(default_factory=datetime.utcnow)
    
    # Configuration
    learning_enabled: bool = True
    adaptation_threshold: float = 0.05
    evolution_triggers: list[EvolutionTrigger] = field(default_factory=list)
    
    # Learning history
    learning_sessions: list[LearningSession] = field(default_factory=list)
    model_evolutions: list[ModelEvolution] = field(default_factory=list)
    drift_events: list[DriftEvent] = field(default_factory=list)
    user_feedback_history: list[UserFeedback] = field(default_factory=list)
    
    # Current state
    current_performance_baseline: PerformanceBaseline  < /dev/null |  None = None
    consecutive_improvements: int = 0
    consecutive_degradations: int = 0
    
    # Adaptation metrics
    total_adaptations: int = 0
    successful_adaptations: int = 0
    adaptation_frequency: float = 0.0
    average_improvement: float = 0.0
    knowledge_retention_score: float = 1.0
    
    # System health
    learning_health_score: float = 1.0
    adaptation_latency_ms: float = 0.0
    resource_efficiency: float = 1.0
    
    # Metadata
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        """Validate continuous learning configuration."""
        if not (0.0 <= self.adaptation_threshold <= 1.0):
            raise ValueError("Adaptation threshold must be between 0.0 and 1.0")

    @property
    def adaptation_success_rate(self) -> float:
        """Calculate adaptation success rate."""
        if self.total_adaptations == 0:
            return 1.0
        return self.successful_adaptations / self.total_adaptations

    @property
    def is_learning_healthy(self) -> bool:
        """Check if learning system is healthy."""
        return (
            self.learning_health_score > 0.8
            and self.adaptation_success_rate > 0.7
            and self.knowledge_retention_score > 0.8
        )

    def get_learning_summary(self) -> dict[str, Any]:
        """Get comprehensive learning system summary."""
        return {
            "learning_id": str(self.learning_id),
            "model_id": str(self.model_id),
            "strategy": self.strategy.value,
            "enabled": self.learning_enabled,
            "health": {
                "score": self.learning_health_score,
                "is_healthy": self.is_learning_healthy(),
            },
            "adaptation_metrics": {
                "total": self.total_adaptations,
                "success_rate": self.adaptation_success_rate,
                "frequency": self.adaptation_frequency,
                "average_improvement": self.average_improvement,
            },
            "last_updated": self.last_updated.isoformat(),
            "metadata": self.metadata,
        }

