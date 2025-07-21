"""Application service for managing ML experiments and runs."""

from __future__ import annotations

from datetime import datetime
from typing import Any
from uuid import UUID

from monorepo.domain.entities import (
    Experiment,
    ExperimentRun,
    ExperimentStatus,
    ExperimentType,
)
from monorepo.domain.exceptions import (
    ExperimentNotFoundError,
    InvalidExperimentStateError,
)
# TODO: Create local protocol interfaces


class ExperimentManagementService:
    """Service for managing ML experiments and their runs."""

    def __init__(
        self,
        experiment_repository: ExperimentRepositoryProtocol,
        experiment_run_repository: ExperimentRunRepositoryProtocol,
    ):
        """Initialize the service.

        Args:
            experiment_repository: Repository for experiments
            experiment_run_repository: Repository for experiment runs
        """
        self.experiment_repository = experiment_repository
        self.experiment_run_repository = experiment_run_repository

    async def create_experiment(
        self,
        name: str,
        description: str,
        experiment_type: ExperimentType,
        objective: str,
        created_by: str,
        optimization_metrics: list[dict[str, str]] | None = None,
        tags: list[str] | None = None,
    ) -> Experiment:
        """Create a new experiment.

        Args:
            name: Name of the experiment
            description: Description of the experiment
            experiment_type: Type of experiment
            objective: Objective of the experiment
            created_by: User creating the experiment
            optimization_metrics: List of metrics to optimize
            tags: Tags for the experiment

        Returns:
            Created experiment
        """
        # Check if experiment name already exists
        existing_experiments = await self.experiment_repository.find_by_name(name)
        if existing_experiments:
            raise ValueError(f"Experiment with name '{name}' already exists")

        experiment = Experiment(
            name=name,
            description=description,
            experiment_type=experiment_type,
            objective=objective,
            created_by=created_by,
            tags=tags or [],
        )

        # Add optimization metrics
        if optimization_metrics:
            for metric in optimization_metrics:
                experiment.add_optimization_metric(
                    metric["name"], metric.get("direction", "maximize")
                )

        await self.experiment_repository.save(experiment)
        return experiment

    async def add_experiment_run(
        self,
        experiment_id: UUID,
        run_name: str,
        detector_id: UUID,
        dataset_id: UUID,
        parameters: dict[str, Any],
        created_by: str,
        description: str = "",
        tags: list[str] | None = None,
    ) -> ExperimentRun:
        """Add a new run to an experiment.

        Args:
            experiment_id: ID of the experiment
            run_name: Name of the run
            detector_id: ID of the detector used
            dataset_id: ID of the dataset used
            parameters: Run parameters
            created_by: User creating the run
            description: Description of the run
            tags: Tags for the run

        Returns:
            Created experiment run
        """
        # Verify experiment exists
        experiment = await self.experiment_repository.find_by_id(experiment_id)
        if not experiment:
            raise ExperimentNotFoundError(experiment_id=experiment_id)

        # Check if experiment is active
        if experiment.status != ExperimentStatus.ACTIVE:
            raise InvalidExperimentStateError(
                experiment_id=experiment_id,
                operation="add_run",
                reason=f"Cannot add run to {experiment.status.value} experiment",
            )

        # Create run
        run = ExperimentRun(
            name=run_name,
            detector_id=detector_id,
            dataset_id=dataset_id,
            parameters=parameters,
            created_by=created_by,
            description=description,
            tags=tags or [],
        )

        # Save run first
        await self.experiment_run_repository.save(run)

        # Add to experiment
        experiment.add_run(run)
        await self.experiment_repository.save(experiment)

        return run

    async def start_experiment_run(self, run_id: UUID) -> ExperimentRun:
        """Start an experiment run.

        Args:
            run_id: ID of the run to start

        Returns:
            Updated experiment run
        """
        run = await self.experiment_run_repository.find_by_id(run_id)
        if not run:
            raise ValueError(f"Run {run_id} not found")

        run.start()
        await self.experiment_run_repository.save(run)
        return run

    async def complete_experiment_run(
        self,
        run_id: UUID,
        metrics: dict[str, float],
        artifacts: dict[str, str] | None = None,
    ) -> ExperimentRun:
        """Complete an experiment run with results.

        Args:
            run_id: ID of the run to complete
            metrics: Performance metrics achieved
            artifacts: Artifacts produced by the run

        Returns:
            Updated experiment run
        """
        run = await self.experiment_run_repository.find_by_id(run_id)
        if not run:
            raise ValueError(f"Run {run_id} not found")

        run.complete(metrics)

        if artifacts:
            for name, path in artifacts.items():
                run.add_artifact(name, path)

        await self.experiment_run_repository.save(run)
        return run

    async def fail_experiment_run(
        self, run_id: UUID, error_message: str
    ) -> ExperimentRun:
        """Mark an experiment run as failed.

        Args:
            run_id: ID of the run that failed
            error_message: Error message describing the failure

        Returns:
            Updated experiment run
        """
        run = await self.experiment_run_repository.find_by_id(run_id)
        if not run:
            raise ValueError(f"Run {run_id} not found")

        run.fail(error_message)
        await self.experiment_run_repository.save(run)
        return run

    async def complete_experiment(self, experiment_id: UUID) -> Experiment:
        """Complete an experiment.

        Args:
            experiment_id: ID of the experiment to complete

        Returns:
            Updated experiment
        """
        experiment = await self.experiment_repository.find_by_id(experiment_id)
        if not experiment:
            raise ExperimentNotFoundError(experiment_id=experiment_id)

        experiment.complete_experiment()
        await self.experiment_repository.save(experiment)
        return experiment

    async def archive_experiment(
        self, experiment_id: UUID, archived_by: str, reason: str = ""
    ) -> Experiment:
        """Archive an experiment.

        Args:
            experiment_id: ID of the experiment to archive
            archived_by: User archiving the experiment
            reason: Reason for archiving

        Returns:
            Updated experiment
        """
        experiment = await self.experiment_repository.find_by_id(experiment_id)
        if not experiment:
            raise ExperimentNotFoundError(experiment_id=experiment_id)

        experiment.update_status(ExperimentStatus.ARCHIVED)
        experiment.update_metadata("archived_by", archived_by)
        experiment.update_metadata("archived_at", datetime.utcnow().isoformat())
        if reason:
            experiment.update_metadata("archive_reason", reason)

        await self.experiment_repository.save(experiment)
        return experiment

    async def get_experiment_with_runs(self, experiment_id: UUID) -> dict[str, Any]:
        """Get experiment with all its runs.

        Args:
            experiment_id: ID of the experiment

        Returns:
            Experiment information with runs
        """
        experiment = await self.experiment_repository.find_by_id(experiment_id)
        if not experiment:
            raise ExperimentNotFoundError(experiment_id=experiment_id)

        runs = await self.experiment_run_repository.find_by_experiment_id(experiment_id)

        return {
            "experiment": experiment.get_info(),
            "runs": [run.get_info() for run in runs],
            "run_count": len(runs),
            "success_rate": experiment.success_rate,
            "best_run": (
                experiment.get_best_run_info() if experiment.best_run_id else None
            ),
            "metric_summaries": self._get_metric_summaries(experiment, runs),
        }

    async def compare_experiment_runs(
        self, run_id_1: UUID, run_id_2: UUID
    ) -> dict[str, Any]:
        """Compare two experiment runs.

        Args:
            run_id_1: First run to compare
            run_id_2: Second run to compare

        Returns:
            Comparison results
        """
        run1 = await self.experiment_run_repository.find_by_id(run_id_1)
        run2 = await self.experiment_run_repository.find_by_id(run_id_2)

        if not run1 or not run2:
            raise ValueError("One or both runs not found")

        comparison = run1.compare_performance(run2)

        return {
            "run1": {
                "id": str(run1.id),
                "name": run1.name,
                "parameters": run1.parameters,
                "metrics": run1.metrics,
            },
            "run2": {
                "id": str(run2.id),
                "name": run2.name,
                "parameters": run2.parameters,
                "metrics": run2.metrics,
            },
            "metric_differences": comparison,
            "recommendation": self._get_run_recommendation(comparison),
        }

    async def get_experiment_leaderboard(
        self, experiment_id: UUID, metric_name: str, top_k: int = 10
    ) -> dict[str, Any]:
        """Get top performing runs for an experiment.

        Args:
            experiment_id: ID of the experiment
            metric_name: Metric to rank by
            top_k: Number of top runs to return

        Returns:
            Leaderboard data
        """
        experiment = await self.experiment_repository.find_by_id(experiment_id)
        if not experiment:
            raise ExperimentNotFoundError(experiment_id=experiment_id)

        runs = await self.experiment_run_repository.find_by_experiment_id(experiment_id)

        # Filter runs with the metric and sort
        valid_runs = [run for run in runs if metric_name in run.metrics]

        # Get optimization direction for the metric
        metric_direction = experiment.optimization_metrics.get(metric_name, "maximize")
        reverse = metric_direction == "maximize"

        sorted_runs = sorted(
            valid_runs, key=lambda r: r.metrics[metric_name], reverse=reverse
        )

        top_runs = sorted_runs[:top_k]

        return {
            "experiment_id": str(experiment_id),
            "experiment_name": experiment.name,
            "metric_name": metric_name,
            "optimization_direction": metric_direction,
            "total_runs": len(valid_runs),
            "leaderboard": [
                {
                    "rank": i + 1,
                    "run_id": str(run.id),
                    "run_name": run.name,
                    "metric_value": run.metrics[metric_name],
                    "parameters": run.parameters,
                    "created_at": run.created_at.isoformat(),
                }
                for i, run in enumerate(top_runs)
            ],
        }

    def _get_metric_summaries(
        self, experiment: Experiment, runs: list[ExperimentRun]
    ) -> dict[str, dict[str, float]]:
        """Get summary statistics for all metrics."""
        summaries = {}

        for metric_name in experiment.optimization_metrics:
            summary = experiment.get_metric_summary(metric_name)
            if summary:
                summaries[metric_name] = summary

        return summaries

    def _get_run_recommendation(self, comparison: dict[str, float]) -> str:
        """Get recommendation based on run comparison."""
        if not comparison:
            return "No metrics to compare"

        positive_diffs = sum(1 for diff in comparison.values() if diff > 0)
        negative_diffs = sum(1 for diff in comparison.values() if diff < 0)

        if positive_diffs > negative_diffs:
            return "Run 1 shows better overall performance"
        elif negative_diffs > positive_diffs:
            return "Run 2 shows better overall performance"
        else:
            return "Runs show similar performance, review specific metrics"

    async def get_active_experiments(self) -> list[Experiment]:
        """Get all active experiments.

        Returns:
            List of active experiments
        """
        all_experiments = await self.experiment_repository.find_by_status(
            ExperimentStatus.ACTIVE
        )
        return all_experiments

    async def get_experiment_analytics(self, experiment_id: UUID) -> dict[str, Any]:
        """Get analytics data for an experiment.

        Args:
            experiment_id: ID of the experiment

        Returns:
            Analytics data
        """
        experiment = await self.experiment_repository.find_by_id(experiment_id)
        if not experiment:
            raise ExperimentNotFoundError(experiment_id=experiment_id)

        runs = await self.experiment_run_repository.find_by_experiment_id(experiment_id)

        # Calculate analytics
        total_runs = len(runs)
        successful_runs = len([r for r in runs if r.is_successful])
        failed_runs = len([r for r in runs if r.is_failed])

        avg_duration = None
        if runs:
            durations = [
                r.duration_seconds for r in runs if r.duration_seconds is not None
            ]
            if durations:
                avg_duration = sum(durations) / len(durations)

        return {
            "experiment_id": str(experiment_id),
            "experiment_name": experiment.name,
            "total_runs": total_runs,
            "successful_runs": successful_runs,
            "failed_runs": failed_runs,
            "success_rate": experiment.success_rate,
            "average_duration_seconds": avg_duration,
            "optimization_metrics": list(experiment.optimization_metrics.keys()),
            "run_timeline": [
                {
                    "run_id": str(run.id),
                    "run_name": run.name,
                    "started_at": (
                        run.started_at.isoformat() if run.started_at else None
                    ),
                    "completed_at": (
                        run.completed_at.isoformat() if run.completed_at else None
                    ),
                    "status": run.status,
                    "key_metrics": {
                        k: v
                        for k, v in run.metrics.items()
                        if k in experiment.optimization_metrics
                    },
                }
                for run in sorted(runs, key=lambda r: r.created_at)
            ],
        }
