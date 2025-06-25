"""Experiment entity for tracking ML experiments and comparisons."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any
from uuid import UUID, uuid4


class ExperimentStatus(Enum):
    """Status of an experiment."""

    DRAFT = "draft"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"
    ARCHIVED = "archived"


class ExperimentType(Enum):
    """Type of experiment."""

    ALGORITHM_COMPARISON = "algorithm_comparison"
    HYPERPARAMETER_TUNING = "hyperparameter_tuning"
    FEATURE_ENGINEERING = "feature_engineering"
    DATA_PREPROCESSING = "data_preprocessing"
    ENSEMBLE_ANALYSIS = "ensemble_analysis"
    CROSS_VALIDATION = "cross_validation"
    A_B_TESTING = "a_b_testing"
    PERFORMANCE_ANALYSIS = "performance_analysis"


@dataclass
class ExperimentRun:
    """Represents a single run within an experiment."""

    id: UUID = field(default_factory=uuid4)
    name: str = ""
    detector_id: UUID | None = None
    dataset_id: UUID | None = None
    parameters: dict[str, Any] = field(default_factory=dict)
    metrics: dict[str, float] = field(default_factory=dict)
    artifacts: dict[str, str] = field(default_factory=dict)  # artifact_name -> path
    started_at: datetime | None = None
    completed_at: datetime | None = None
    status: str = "pending"
    error_message: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def duration_seconds(self) -> float | None:
        """Get run duration in seconds."""
        if self.started_at and self.completed_at:
            return (self.completed_at - self.started_at).total_seconds()
        return None

    @property
    def is_successful(self) -> bool:
        """Check if run completed successfully."""
        return self.status == "completed" and self.error_message is None

    def start(self) -> None:
        """Mark run as started."""
        self.started_at = datetime.utcnow()
        self.status = "running"

    def complete(self, metrics: dict[str, float]) -> None:
        """Mark run as completed with metrics."""
        self.completed_at = datetime.utcnow()
        self.status = "completed"
        self.metrics.update(metrics)

    def fail(self, error_message: str) -> None:
        """Mark run as failed with error."""
        self.completed_at = datetime.utcnow()
        self.status = "failed"
        self.error_message = error_message


@dataclass
class Experiment:
    """Represents an ML experiment for tracking model development.

    An Experiment captures a systematic investigation of model performance,
    algorithm comparisons, or hyperparameter optimization. It contains
    multiple runs and tracks their results for analysis.

    Attributes:
        id: Unique identifier for the experiment
        name: Human-readable name for the experiment
        description: Detailed description of the experiment's purpose
        experiment_type: Type of experiment being conducted
        objective: Primary objective or hypothesis being tested
        created_at: When the experiment was created
        created_by: User who created the experiment
        status: Current status of the experiment
        runs: List of experiment runs
        baseline_run_id: ID of the baseline run for comparison
        best_run_id: ID of the best performing run
        metrics_to_optimize: List of metrics to optimize (with direction)
        tags: Semantic tags for organization
        dataset_ids: Datasets used in this experiment
        metadata: Additional metadata and configuration
    """

    name: str
    description: str
    experiment_type: ExperimentType
    objective: str
    created_by: str
    id: UUID = field(default_factory=uuid4)
    created_at: datetime = field(default_factory=datetime.utcnow)
    status: ExperimentStatus = ExperimentStatus.DRAFT
    runs: list[ExperimentRun] = field(default_factory=list)
    baseline_run_id: UUID | None = None
    best_run_id: UUID | None = None
    metrics_to_optimize: list[dict[str, str]] = field(
        default_factory=list
    )  # [{"metric": "f1", "direction": "maximize"}]
    tags: list[str] = field(default_factory=list)
    dataset_ids: set[UUID] = field(default_factory=set)
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        """Validate experiment after initialization."""
        if not self.name:
            raise ValueError("Experiment name cannot be empty")

        if not self.description:
            raise ValueError("Experiment description cannot be empty")

        if not isinstance(self.experiment_type, ExperimentType):
            raise TypeError(
                f"Experiment type must be ExperimentType, got {type(self.experiment_type)}"
            )

        if not isinstance(self.status, ExperimentStatus):
            raise TypeError(f"Status must be ExperimentStatus, got {type(self.status)}")

        if not self.created_by:
            raise ValueError("Created by cannot be empty")

        if not self.objective:
            raise ValueError("Experiment objective cannot be empty")

    @property
    def is_running(self) -> bool:
        """Check if experiment is currently running."""
        return self.status == ExperimentStatus.RUNNING

    @property
    def is_completed(self) -> bool:
        """Check if experiment is completed."""
        return self.status == ExperimentStatus.COMPLETED

    @property
    def run_count(self) -> int:
        """Get total number of runs."""
        return len(self.runs)

    @property
    def successful_runs(self) -> list[ExperimentRun]:
        """Get list of successful runs."""
        return [run for run in self.runs if run.is_successful]

    @property
    def failed_runs(self) -> list[ExperimentRun]:
        """Get list of failed runs."""
        return [run for run in self.runs if run.status == "failed"]

    @property
    def success_rate(self) -> float:
        """Get success rate of runs."""
        if not self.runs:
            return 0.0
        return len(self.successful_runs) / len(self.runs)

    def add_run(self, run: ExperimentRun) -> None:
        """Add a run to the experiment."""
        if run.id in [r.id for r in self.runs]:
            raise ValueError(f"Run {run.id} already exists in experiment")

        self.runs.append(run)

        # Add dataset to experiment datasets if specified
        if run.dataset_id:
            self.dataset_ids.add(run.dataset_id)

        self.metadata["last_run_added"] = datetime.utcnow().isoformat()

    def get_run(self, run_id: UUID) -> ExperimentRun | None:
        """Get a specific run by ID."""
        for run in self.runs:
            if run.id == run_id:
                return run
        return None

    def remove_run(self, run_id: UUID) -> bool:
        """Remove a run from the experiment."""
        for i, run in enumerate(self.runs):
            if run.id == run_id:
                self.runs.pop(i)

                # Update best/baseline if they were removed
                if self.best_run_id == run_id:
                    self.best_run_id = None
                if self.baseline_run_id == run_id:
                    self.baseline_run_id = None

                return True
        return False

    def set_baseline_run(self, run_id: UUID) -> None:
        """Set the baseline run for comparison."""
        if self.get_run(run_id) is None:
            raise ValueError(f"Run {run_id} not found in experiment")

        self.baseline_run_id = run_id
        self.metadata["baseline_set_at"] = datetime.utcnow().isoformat()

    def update_best_run(self) -> None:
        """Update the best run based on optimization metrics."""
        if not self.successful_runs or not self.metrics_to_optimize:
            return

        best_run = None
        best_score = None

        for run in self.successful_runs:
            score = self._calculate_run_score(run)
            if score is not None and (best_score is None or score > best_score):
                best_score = score
                best_run = run

        if best_run:
            self.best_run_id = best_run.id
            self.metadata["best_run_updated"] = datetime.utcnow().isoformat()
            self.metadata["best_score"] = best_score

    def _calculate_run_score(self, run: ExperimentRun) -> float | None:
        """Calculate a composite score for a run based on optimization metrics."""
        if not self.metrics_to_optimize:
            return None

        score = 0.0
        weight_per_metric = 1.0 / len(self.metrics_to_optimize)

        for metric_config in self.metrics_to_optimize:
            metric_name = metric_config["metric"]
            direction = metric_config.get("direction", "maximize")

            if metric_name not in run.metrics:
                return None  # Missing required metric

            metric_value = run.metrics[metric_name]

            # Normalize metric value (simple approach - could be more sophisticated)
            if direction == "maximize":
                score += metric_value * weight_per_metric
            else:  # minimize
                score += (1.0 - metric_value) * weight_per_metric

        return score

    def start_experiment(self) -> None:
        """Start the experiment."""
        self.status = ExperimentStatus.RUNNING
        self.metadata["started_at"] = datetime.utcnow().isoformat()

    def complete_experiment(self) -> None:
        """Mark experiment as completed."""
        self.status = ExperimentStatus.COMPLETED
        self.metadata["completed_at"] = datetime.utcnow().isoformat()

        # Update best run
        self.update_best_run()

        # Calculate summary statistics
        successful_runs = self.successful_runs
        if successful_runs:
            self.metadata["total_runs"] = len(self.runs)
            self.metadata["successful_runs"] = len(successful_runs)
            self.metadata["success_rate"] = self.success_rate

    def fail_experiment(self, reason: str) -> None:
        """Mark experiment as failed."""
        self.status = ExperimentStatus.FAILED
        self.metadata["failed_at"] = datetime.utcnow().isoformat()
        self.metadata["failure_reason"] = reason

    def cancel_experiment(self, reason: str = "") -> None:
        """Cancel the experiment."""
        self.status = ExperimentStatus.CANCELLED
        self.metadata["cancelled_at"] = datetime.utcnow().isoformat()
        if reason:
            self.metadata["cancellation_reason"] = reason

    def add_optimization_metric(
        self, metric_name: str, direction: str = "maximize"
    ) -> None:
        """Add a metric to optimize."""
        if direction not in ["maximize", "minimize"]:
            raise ValueError("Direction must be 'maximize' or 'minimize'")

        metric_config = {"metric": metric_name, "direction": direction}
        if metric_config not in self.metrics_to_optimize:
            self.metrics_to_optimize.append(metric_config)

    def remove_optimization_metric(self, metric_name: str) -> None:
        """Remove a metric from optimization."""
        self.metrics_to_optimize = [
            m for m in self.metrics_to_optimize if m["metric"] != metric_name
        ]

    def add_tag(self, tag: str) -> None:
        """Add a tag to the experiment."""
        if tag and tag not in self.tags:
            self.tags.append(tag)

    def remove_tag(self, tag: str) -> None:
        """Remove a tag from the experiment."""
        if tag in self.tags:
            self.tags.remove(tag)

    def get_metric_summary(self, metric_name: str) -> dict[str, float]:
        """Get statistical summary of a metric across all successful runs."""
        successful_runs = self.successful_runs
        metric_values = [
            run.metrics[metric_name]
            for run in successful_runs
            if metric_name in run.metrics
        ]

        if not metric_values:
            return {}

        import statistics

        return {
            "count": len(metric_values),
            "mean": statistics.mean(metric_values),
            "median": statistics.median(metric_values),
            "std": statistics.stdev(metric_values) if len(metric_values) > 1 else 0.0,
            "min": min(metric_values),
            "max": max(metric_values),
        }

    def compare_runs(self, run_id_1: UUID, run_id_2: UUID) -> dict[str, Any]:
        """Compare two runs within the experiment."""
        run1 = self.get_run(run_id_1)
        run2 = self.get_run(run_id_2)

        if not run1 or not run2:
            raise ValueError("One or both runs not found")

        # Compare metrics
        all_metrics = set(run1.metrics.keys()) | set(run2.metrics.keys())
        metric_comparison = {}

        for metric in all_metrics:
            val1 = run1.metrics.get(metric)
            val2 = run2.metrics.get(metric)

            if val1 is not None and val2 is not None:
                metric_comparison[metric] = {
                    "run1": val1,
                    "run2": val2,
                    "difference": val1 - val2,
                    "percent_change": (
                        ((val1 - val2) / val2) * 100 if val2 != 0 else None
                    ),
                }
            else:
                metric_comparison[metric] = {
                    "run1": val1,
                    "run2": val2,
                    "difference": None,
                    "percent_change": None,
                }

        return {
            "run1_id": str(run_id_1),
            "run2_id": str(run_id_2),
            "run1_name": run1.name,
            "run2_name": run2.name,
            "metrics": metric_comparison,
            "duration_comparison": {
                "run1_duration": run1.duration_seconds,
                "run2_duration": run2.duration_seconds,
            },
        }

    def get_info(self) -> dict[str, Any]:
        """Get comprehensive information about the experiment."""
        return {
            "id": str(self.id),
            "name": self.name,
            "description": self.description,
            "experiment_type": self.experiment_type.value,
            "objective": self.objective,
            "created_at": self.created_at.isoformat(),
            "created_by": self.created_by,
            "status": self.status.value,
            "run_count": self.run_count,
            "successful_runs": len(self.successful_runs),
            "failed_runs": len(self.failed_runs),
            "success_rate": self.success_rate,
            "baseline_run_id": (
                str(self.baseline_run_id) if self.baseline_run_id else None
            ),
            "best_run_id": str(self.best_run_id) if self.best_run_id else None,
            "metrics_to_optimize": self.metrics_to_optimize.copy(),
            "tags": self.tags.copy(),
            "dataset_ids": [str(d) for d in self.dataset_ids],
            "metadata": self.metadata.copy(),
        }

    def __str__(self) -> str:
        """Human-readable representation."""
        return (
            f"Experiment('{self.name}', {self.experiment_type.value}, "
            f"status={self.status.value}, runs={self.run_count})"
        )

    def __repr__(self) -> str:
        """Developer representation."""
        return (
            f"Experiment(id={self.id}, name='{self.name}', "
            f"type={self.experiment_type.value}, status={self.status.value})"
        )
