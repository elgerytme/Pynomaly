"""
Optimization Trial Domain Entity

Represents a single hyperparameter optimization trial with comprehensive
tracking of parameters, results, and metadata for machine learning experiments.
"""

from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any

from ..value_objects.hyperparameters import HyperparameterSet


class TrialStatus(Enum):
    """Optimization trial status enumeration."""

    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    PRUNED = "pruned"
    CANCELLED = "cancelled"


class TrialState(Enum):
    """Trial state for multi-step optimization."""

    CREATED = "created"
    STARTED = "started"
    EVALUATING = "evaluating"
    FINISHED = "finished"
    ERROR = "error"


@dataclass
class TrialMetadata:
    """Metadata for optimization trials."""

    algorithm_name: str
    optimization_strategy: str
    sampler_name: str | None = None
    pruner_name: str | None = None
    study_name: str | None = None
    experiment_id: str | None = None
    user_id: str | None = None
    worker_id: str | None = None

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "algorithm_name": self.algorithm_name,
            "optimization_strategy": self.optimization_strategy,
            "sampler_name": self.sampler_name,
            "pruner_name": self.pruner_name,
            "study_name": self.study_name,
            "experiment_id": self.experiment_id,
            "user_id": self.user_id,
            "worker_id": self.worker_id,
        }


@dataclass
class TrialResourceUsage:
    """Resource usage tracking for optimization trials."""

    cpu_time: float = 0.0
    memory_peak: float = 0.0  # MB
    gpu_memory_peak: float = 0.0  # MB
    evaluation_time: float = 0.0

    def to_dict(self) -> dict[str, float]:
        """Convert to dictionary."""
        return {
            "cpu_time": self.cpu_time,
            "memory_peak": self.memory_peak,
            "gpu_memory_peak": self.gpu_memory_peak,
            "evaluation_time": self.evaluation_time,
        }


@dataclass
class OptimizationTrial:
    """
    Optimization trial entity representing a single hyperparameter evaluation.

    An optimization trial represents one evaluation of a specific hyperparameter
    configuration, including the parameters tested, the resulting score, timing
    information, and metadata about the optimization process.
    """

    # Core identification
    trial_id: int
    study_id: str | None = None
    number: int | None = None  # Trial number within study

    # Parameters and results
    parameters: HyperparameterSet = field(default_factory=lambda: HyperparameterSet({}))
    score: float = 0.0
    objective_values: list[float] = field(
        default_factory=list
    )  # For multi-objective optimization

    # Status and state
    status: TrialStatus = TrialStatus.PENDING
    state: TrialState = TrialState.CREATED

    # Timing information
    start_time: datetime = field(default_factory=datetime.utcnow)
    end_time: datetime | None = None
    duration: float = 0.0  # seconds

    # Intermediate values for pruning
    intermediate_values: list[float] = field(default_factory=list)
    intermediate_steps: list[int] = field(default_factory=list)

    # Metadata and attributes
    metadata: TrialMetadata | None = None
    user_attrs: dict[str, Any] = field(default_factory=dict)
    system_attrs: dict[str, Any] = field(default_factory=dict)

    # Resource tracking
    resource_usage: TrialResourceUsage = field(default_factory=TrialResourceUsage)

    # Error handling
    error_message: str | None = None
    error_traceback: str | None = None

    # Pruning information
    pruned_at_step: int | None = None
    pruning_reason: str | None = None

    # Validation and cross-validation scores
    validation_scores: list[float] = field(default_factory=list)
    cv_scores: list[float] = field(default_factory=list)
    cv_std: float | None = None

    @property
    def is_complete(self) -> bool:
        """Check if trial is complete."""
        return self.status in [
            TrialStatus.COMPLETED,
            TrialStatus.FAILED,
            TrialStatus.PRUNED,
            TrialStatus.CANCELLED,
        ]

    @property
    def is_successful(self) -> bool:
        """Check if trial completed successfully."""
        return self.status == TrialStatus.COMPLETED

    @property
    def was_pruned(self) -> bool:
        """Check if trial was pruned."""
        return self.status == TrialStatus.PRUNED

    def start(self) -> None:
        """Mark trial as started."""
        self.status = TrialStatus.RUNNING
        self.state = TrialState.STARTED
        self.start_time = datetime.utcnow()

    def complete(self, score: float) -> None:
        """Mark trial as completed with final score."""
        self.status = TrialStatus.COMPLETED
        self.state = TrialState.FINISHED
        self.score = score
        self.end_time = datetime.utcnow()
        if self.end_time:
            self.duration = (self.end_time - self.start_time).total_seconds()

    def fail(self, error_message: str, error_traceback: str | None = None) -> None:
        """Mark trial as failed."""
        self.status = TrialStatus.FAILED
        self.state = TrialState.ERROR
        self.error_message = error_message
        self.error_traceback = error_traceback
        self.end_time = datetime.utcnow()
        if self.end_time:
            self.duration = (self.end_time - self.start_time).total_seconds()

    def prune(self, step: int, reason: str = "Pruned by optimizer") -> None:
        """Mark trial as pruned."""
        self.status = TrialStatus.PRUNED
        self.state = TrialState.FINISHED
        self.pruned_at_step = step
        self.pruning_reason = reason
        self.end_time = datetime.utcnow()
        if self.end_time:
            self.duration = (self.end_time - self.start_time).total_seconds()

    def cancel(self) -> None:
        """Cancel the trial."""
        self.status = TrialStatus.CANCELLED
        self.state = TrialState.FINISHED
        self.end_time = datetime.utcnow()
        if self.end_time:
            self.duration = (self.end_time - self.start_time).total_seconds()

    def add_intermediate_value(self, step: int, value: float) -> None:
        """Add an intermediate value for pruning decisions."""
        self.intermediate_steps.append(step)
        self.intermediate_values.append(value)

    def add_user_attr(self, key: str, value: Any) -> None:
        """Add a user attribute."""
        self.user_attrs[key] = value

    def add_system_attr(self, key: str, value: Any) -> None:
        """Add a system attribute."""
        self.system_attrs[key] = value

    def get_best_intermediate_value(self) -> float | None:
        """Get the best intermediate value."""
        if not self.intermediate_values:
            return None
        return max(self.intermediate_values)

    def get_parameter_value(self, param_name: str) -> Any:
        """Get the value of a specific parameter."""
        return self.parameters.get(param_name)

    def set_parameter_value(self, param_name: str, value: Any) -> None:
        """Set the value of a specific parameter."""
        self.parameters.set(param_name, value)

    def add_validation_score(self, score: float) -> None:
        """Add a validation score."""
        self.validation_scores.append(score)

    def add_cv_scores(self, scores: list[float]) -> None:
        """Add cross-validation scores."""
        self.cv_scores = scores
        if scores:
            import numpy as np

            self.cv_std = float(np.std(scores))

    def get_summary_stats(self) -> dict[str, Any]:
        """Get summary statistics for the trial."""
        stats = {
            "trial_id": self.trial_id,
            "status": self.status.value,
            "score": self.score,
            "duration": self.duration,
            "parameter_count": len(self.parameters.parameters),
            "intermediate_values_count": len(self.intermediate_values),
        }

        if self.validation_scores:
            import numpy as np

            stats.update(
                {
                    "validation_mean": float(np.mean(self.validation_scores)),
                    "validation_std": float(np.std(self.validation_scores)),
                    "validation_min": float(np.min(self.validation_scores)),
                    "validation_max": float(np.max(self.validation_scores)),
                }
            )

        if self.cv_scores:
            import numpy as np

            stats.update(
                {
                    "cv_mean": float(np.mean(self.cv_scores)),
                    "cv_std": self.cv_std,
                    "cv_scores": self.cv_scores,
                }
            )

        if self.was_pruned:
            stats.update(
                {
                    "pruned_at_step": self.pruned_at_step,
                    "pruning_reason": self.pruning_reason,
                }
            )

        return stats

    def to_dict(self) -> dict[str, Any]:
        """Convert trial to dictionary."""
        return {
            "trial_id": self.trial_id,
            "study_id": self.study_id,
            "number": self.number,
            "parameters": self.parameters.to_dict(),
            "score": self.score,
            "objective_values": self.objective_values,
            "status": self.status.value,
            "state": self.state.value,
            "start_time": self.start_time.isoformat(),
            "end_time": self.end_time.isoformat() if self.end_time else None,
            "duration": self.duration,
            "intermediate_values": self.intermediate_values,
            "intermediate_steps": self.intermediate_steps,
            "metadata": self.metadata.to_dict() if self.metadata else None,
            "user_attrs": self.user_attrs,
            "system_attrs": self.system_attrs,
            "resource_usage": self.resource_usage.to_dict(),
            "error_message": self.error_message,
            "error_traceback": self.error_traceback,
            "pruned_at_step": self.pruned_at_step,
            "pruning_reason": self.pruning_reason,
            "validation_scores": self.validation_scores,
            "cv_scores": self.cv_scores,
            "cv_std": self.cv_std,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "OptimizationTrial":
        """Create trial from dictionary."""
        # Handle datetime fields
        if "start_time" in data:
            data["start_time"] = datetime.fromisoformat(data["start_time"])
        if "end_time" in data and data["end_time"]:
            data["end_time"] = datetime.fromisoformat(data["end_time"])

        # Handle enum fields
        if "status" in data:
            data["status"] = TrialStatus(data["status"])
        if "state" in data:
            data["state"] = TrialState(data["state"])

        # Handle complex fields
        if "parameters" in data:
            data["parameters"] = HyperparameterSet.from_dict(data["parameters"])

        if "metadata" in data and data["metadata"]:
            data["metadata"] = TrialMetadata(**data["metadata"])

        if "resource_usage" in data:
            data["resource_usage"] = TrialResourceUsage(**data["resource_usage"])

        return cls(**data)

    def compare_to(self, other: "OptimizationTrial", metric: str = "score") -> int:
        """
        Compare this trial to another trial.

        Args:
            other: Other trial to compare to
            metric: Metric to compare ('score', 'duration', 'parameter_count')

        Returns:
            -1 if this trial is worse, 0 if equal, 1 if this trial is better
        """
        if metric == "score":
            if self.score < other.score:
                return -1
            elif self.score > other.score:
                return 1
            else:
                return 0
        elif metric == "duration":
            if self.duration > other.duration:
                return -1
            elif self.duration < other.duration:
                return 1
            else:
                return 0
        elif metric == "parameter_count":
            self_count = len(self.parameters.parameters)
            other_count = len(other.parameters.parameters)
            if self_count > other_count:
                return -1
            elif self_count < other_count:
                return 1
            else:
                return 0
        else:
            raise ValueError(f"Unknown comparison metric: {metric}")

    def is_pareto_optimal(self, other_trials: list["OptimizationTrial"]) -> bool:
        """
        Check if this trial is Pareto optimal compared to other trials.

        Args:
            other_trials: List of other trials to compare against

        Returns:
            True if this trial is Pareto optimal, False otherwise
        """
        for other in other_trials:
            if other.trial_id == self.trial_id:
                continue

            # Check if other trial dominates this trial
            # (better or equal in all objectives, strictly better in at least one)
            other_better_score = other.score >= self.score
            other_better_duration = other.duration <= self.duration

            if other_better_score and other_better_duration:
                # Check if strictly better in at least one objective
                if other.score > self.score or other.duration < self.duration:
                    return False

        return True

    def calculate_improvement_rate(self) -> float | None:
        """Calculate the improvement rate from intermediate values."""
        if len(self.intermediate_values) < 2:
            return None

        # Calculate linear regression slope
        import numpy as np

        steps = np.array(self.intermediate_steps)
        values = np.array(self.intermediate_values)

        if len(steps) != len(values):
            return None

        # Simple linear regression
        n = len(steps)
        slope = (n * np.sum(steps * values) - np.sum(steps) * np.sum(values)) / (
            n * np.sum(steps**2) - np.sum(steps) ** 2
        )

        return float(slope)

    def should_continue(self, min_improvement_rate: float = 0.001) -> bool:
        """
        Determine if trial should continue based on improvement rate.

        Args:
            min_improvement_rate: Minimum required improvement rate

        Returns:
            True if trial should continue, False if it should be stopped
        """
        if self.status != TrialStatus.RUNNING:
            return False

        improvement_rate = self.calculate_improvement_rate()
        if improvement_rate is None:
            return True  # Not enough data to decide

        return improvement_rate >= min_improvement_rate
