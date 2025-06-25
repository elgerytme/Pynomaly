"""Application service for managing ML pipelines and orchestration."""

from __future__ import annotations

from typing import Any
from uuid import UUID

from pynomaly.domain.entities import (
    Pipeline,
    PipelineRun,
    PipelineStatus,
    PipelineStep,
    PipelineType,
    StepType,
)
from pynomaly.domain.exceptions import InvalidPipelineStateError, PipelineNotFoundError
from pynomaly.shared.protocols import (
    PipelineRepositoryProtocol,
    PipelineRunRepositoryProtocol,
)


class PipelineOrchestrationService:
    """Service for managing ML pipelines and their orchestration."""

    def __init__(
        self,
        pipeline_repository: PipelineRepositoryProtocol,
        pipeline_run_repository: PipelineRunRepositoryProtocol,
    ):
        """Initialize the service.

        Args:
            pipeline_repository: Repository for pipelines
            pipeline_run_repository: Repository for pipeline runs
        """
        self.pipeline_repository = pipeline_repository
        self.pipeline_run_repository = pipeline_run_repository

    async def create_pipeline(
        self,
        name: str,
        description: str,
        pipeline_type: PipelineType,
        created_by: str,
        environment: str = "development",
        model_id: UUID | None = None,
        datasets: list[UUID] | None = None,
        tags: list[str] | None = None,
    ) -> Pipeline:
        """Create a new pipeline.

        Args:
            name: Name of the pipeline
            description: Description of the pipeline
            pipeline_type: Type of pipeline
            created_by: User creating the pipeline
            environment: Target environment
            model_id: Associated model ID
            datasets: Dataset IDs used by pipeline
            tags: Tags for the pipeline

        Returns:
            Created pipeline
        """
        # Check if pipeline name already exists in the environment
        existing_pipelines = (
            await self.pipeline_repository.find_by_name_and_environment(
                name, environment
            )
        )
        if existing_pipelines:
            raise ValueError(
                f"Pipeline with name '{name}' already exists in {environment}"
            )

        pipeline = Pipeline(
            name=name,
            description=description,
            pipeline_type=pipeline_type,
            created_by=created_by,
            environment=environment,
            model_id=model_id,
            datasets=datasets or [],
            tags=tags or [],
        )

        await self.pipeline_repository.save(pipeline)
        return pipeline

    async def add_pipeline_step(
        self,
        pipeline_id: UUID,
        step_name: str,
        step_type: StepType,
        description: str,
        configuration: dict[str, Any],
        order: int | None = None,
        dependencies: list[UUID] | None = None,
        timeout_seconds: int | None = None,
    ) -> PipelineStep:
        """Add a step to a pipeline.

        Args:
            pipeline_id: ID of the pipeline
            step_name: Name of the step
            step_type: Type of step
            description: Description of the step
            configuration: Step configuration
            order: Execution order (auto-assigned if None)
            dependencies: Step dependencies
            timeout_seconds: Timeout for step execution

        Returns:
            Created pipeline step
        """
        # Verify pipeline exists
        pipeline = await self.pipeline_repository.find_by_id(pipeline_id)
        if not pipeline:
            raise PipelineNotFoundError(pipeline_id=pipeline_id)

        # Check if pipeline is in draft state
        if pipeline.status != PipelineStatus.DRAFT:
            raise InvalidPipelineStateError(
                pipeline_id=pipeline_id,
                operation="add_step",
                reason=f"Cannot add step to {pipeline.status.value} pipeline",
            )

        # Create step
        step = PipelineStep(
            name=step_name,
            step_type=step_type,
            description=description,
            order=order or 0,
            configuration=configuration,
            dependencies=dependencies or [],
            timeout_seconds=timeout_seconds,
        )

        # Add to pipeline
        pipeline.add_step(step)
        await self.pipeline_repository.save(pipeline)

        return step

    async def update_pipeline_step(
        self,
        pipeline_id: UUID,
        step_id: UUID,
        configuration: dict[str, Any] | None = None,
        description: str | None = None,
        timeout_seconds: int | None = None,
        is_enabled: bool | None = None,
    ) -> PipelineStep:
        """Update a pipeline step.

        Args:
            pipeline_id: ID of the pipeline
            step_id: ID of the step to update
            configuration: New configuration
            description: New description
            timeout_seconds: New timeout
            is_enabled: Enable/disable step

        Returns:
            Updated pipeline step
        """
        pipeline = await self.pipeline_repository.find_by_id(pipeline_id)
        if not pipeline:
            raise PipelineNotFoundError(pipeline_id=pipeline_id)

        step = pipeline.get_step(step_id)
        if not step:
            raise ValueError(f"Step {step_id} not found in pipeline")

        # Update step properties
        if configuration is not None:
            step.update_configuration(configuration)

        if description is not None:
            step.description = description

        if timeout_seconds is not None:
            step.timeout_seconds = timeout_seconds

        if is_enabled is not None:
            step.is_enabled = is_enabled

        await self.pipeline_repository.save(pipeline)
        return step

    async def remove_pipeline_step(self, pipeline_id: UUID, step_id: UUID) -> bool:
        """Remove a step from a pipeline.

        Args:
            pipeline_id: ID of the pipeline
            step_id: ID of the step to remove

        Returns:
            True if step was removed, False if not found
        """
        pipeline = await self.pipeline_repository.find_by_id(pipeline_id)
        if not pipeline:
            raise PipelineNotFoundError(pipeline_id=pipeline_id)

        # Check if pipeline is in draft state
        if pipeline.status != PipelineStatus.DRAFT:
            raise InvalidPipelineStateError(
                pipeline_id=pipeline_id,
                operation="remove_step",
                reason=f"Cannot remove step from {pipeline.status.value} pipeline",
            )

        removed = pipeline.remove_step(step_id)
        if removed:
            await self.pipeline_repository.save(pipeline)

        return removed

    async def validate_and_activate_pipeline(self, pipeline_id: UUID) -> Pipeline:
        """Validate and activate a pipeline.

        Args:
            pipeline_id: ID of the pipeline to activate

        Returns:
            Activated pipeline
        """
        pipeline = await self.pipeline_repository.find_by_id(pipeline_id)
        if not pipeline:
            raise PipelineNotFoundError(pipeline_id=pipeline_id)

        # Validate pipeline
        is_valid, issues = pipeline.validate_pipeline()
        if not is_valid:
            raise ValueError(f"Pipeline validation failed: {'; '.join(issues)}")

        # Activate pipeline
        pipeline.activate()
        await self.pipeline_repository.save(pipeline)

        return pipeline

    async def pause_pipeline(self, pipeline_id: UUID) -> Pipeline:
        """Pause a pipeline.

        Args:
            pipeline_id: ID of the pipeline to pause

        Returns:
            Paused pipeline
        """
        pipeline = await self.pipeline_repository.find_by_id(pipeline_id)
        if not pipeline:
            raise PipelineNotFoundError(pipeline_id=pipeline_id)

        if pipeline.status != PipelineStatus.ACTIVE:
            raise InvalidPipelineStateError(
                pipeline_id=pipeline_id,
                operation="pause",
                reason=f"Cannot pause {pipeline.status.value} pipeline",
            )

        pipeline.pause()
        await self.pipeline_repository.save(pipeline)

        return pipeline

    async def set_pipeline_schedule(
        self, pipeline_id: UUID, cron_expression: str
    ) -> Pipeline:
        """Set a schedule for a pipeline.

        Args:
            pipeline_id: ID of the pipeline
            cron_expression: Cron expression for scheduling

        Returns:
            Updated pipeline
        """
        pipeline = await self.pipeline_repository.find_by_id(pipeline_id)
        if not pipeline:
            raise PipelineNotFoundError(pipeline_id=pipeline_id)

        pipeline.set_schedule(cron_expression)
        await self.pipeline_repository.save(pipeline)

        return pipeline

    async def trigger_pipeline_run(
        self,
        pipeline_id: UUID,
        triggered_by: str,
        trigger_reason: str = "",
        override_parameters: dict[str, Any] | None = None,
    ) -> PipelineRun:
        """Trigger a pipeline run.

        Args:
            pipeline_id: ID of the pipeline to run
            triggered_by: User or system triggering the run
            trigger_reason: Reason for triggering
            override_parameters: Parameters to override for this run

        Returns:
            Created pipeline run
        """
        pipeline = await self.pipeline_repository.find_by_id(pipeline_id)
        if not pipeline:
            raise PipelineNotFoundError(pipeline_id=pipeline_id)

        if pipeline.status not in [PipelineStatus.ACTIVE]:
            raise InvalidPipelineStateError(
                pipeline_id=pipeline_id,
                operation="trigger_run",
                reason=f"Cannot run {pipeline.status.value} pipeline",
            )

        # Create pipeline run
        run = PipelineRun(
            pipeline_id=pipeline_id,
            triggered_by=triggered_by,
            trigger_reason=trigger_reason,
        )

        # Apply parameter overrides
        if override_parameters:
            run.metadata["parameter_overrides"] = override_parameters

        # Start the run
        run.start(triggered_by, trigger_reason)

        await self.pipeline_run_repository.save(run)
        return run

    async def complete_pipeline_run(
        self,
        run_id: UUID,
        step_results: dict[str, dict[str, Any]] | None = None,
        artifacts: dict[str, str] | None = None,
    ) -> PipelineRun:
        """Complete a pipeline run.

        Args:
            run_id: ID of the run to complete
            step_results: Results from each step
            artifacts: Artifacts produced by the run

        Returns:
            Updated pipeline run
        """
        run = await self.pipeline_run_repository.find_by_id(run_id)
        if not run:
            raise ValueError(f"Pipeline run {run_id} not found")

        # Add step results
        if step_results:
            for step_id, result in step_results.items():
                run.add_step_result(step_id, result)

        # Add artifacts
        if artifacts:
            run.artifacts.update(artifacts)

        run.complete()
        await self.pipeline_run_repository.save(run)

        return run

    async def fail_pipeline_run(
        self, run_id: UUID, error_message: str, failed_step_id: str | None = None
    ) -> PipelineRun:
        """Mark a pipeline run as failed.

        Args:
            run_id: ID of the run that failed
            error_message: Error message describing the failure
            failed_step_id: ID of the step that failed

        Returns:
            Updated pipeline run
        """
        run = await self.pipeline_run_repository.find_by_id(run_id)
        if not run:
            raise ValueError(f"Pipeline run {run_id} not found")

        if failed_step_id:
            run.metadata["failed_step_id"] = failed_step_id

        run.fail(error_message)
        await self.pipeline_run_repository.save(run)

        return run

    async def get_pipeline_with_runs(
        self, pipeline_id: UUID, limit: int = 10
    ) -> dict[str, Any]:
        """Get pipeline with recent runs.

        Args:
            pipeline_id: ID of the pipeline
            limit: Maximum number of runs to include

        Returns:
            Pipeline information with runs
        """
        pipeline = await self.pipeline_repository.find_by_id(pipeline_id)
        if not pipeline:
            raise PipelineNotFoundError(pipeline_id=pipeline_id)

        runs = await self.pipeline_run_repository.find_by_pipeline_id(pipeline_id)

        # Sort by start time, most recent first
        runs.sort(key=lambda r: r.started_at or r.id, reverse=True)
        recent_runs = runs[:limit]

        return {
            "pipeline": pipeline.get_info(),
            "recent_runs": [run.get_info() for run in recent_runs],
            "total_runs": len(runs),
            "success_rate": self._calculate_success_rate(runs),
            "average_duration": self._calculate_average_duration(runs),
            "last_successful_run": self._get_last_successful_run(runs),
            "last_failed_run": self._get_last_failed_run(runs),
        }

    async def get_pipeline_analytics(self, pipeline_id: UUID) -> dict[str, Any]:
        """Get analytics data for a pipeline.

        Args:
            pipeline_id: ID of the pipeline

        Returns:
            Analytics data
        """
        pipeline = await self.pipeline_repository.find_by_id(pipeline_id)
        if not pipeline:
            raise PipelineNotFoundError(pipeline_id=pipeline_id)

        runs = await self.pipeline_run_repository.find_by_pipeline_id(pipeline_id)

        # Calculate analytics
        total_runs = len(runs)
        successful_runs = len([r for r in runs if r.is_completed])
        failed_runs = len([r for r in runs if r.is_failed])

        return {
            "pipeline_id": str(pipeline_id),
            "pipeline_name": pipeline.name,
            "total_runs": total_runs,
            "successful_runs": successful_runs,
            "failed_runs": failed_runs,
            "success_rate": self._calculate_success_rate(runs),
            "average_duration_seconds": self._calculate_average_duration(runs),
            "run_frequency": self._calculate_run_frequency(runs),
            "step_failure_analysis": self._analyze_step_failures(runs),
            "performance_trend": self._calculate_performance_trend(runs),
        }

    async def get_active_pipelines(
        self, environment: str | None = None
    ) -> list[Pipeline]:
        """Get all active pipelines.

        Args:
            environment: Filter by environment (optional)

        Returns:
            List of active pipelines
        """
        all_pipelines = await self.pipeline_repository.find_by_status(
            PipelineStatus.ACTIVE
        )

        if environment:
            return [p for p in all_pipelines if p.environment == environment]

        return all_pipelines

    async def get_scheduled_pipelines(self) -> list[Pipeline]:
        """Get all pipelines with schedules.

        Returns:
            List of scheduled pipelines
        """
        all_pipelines = await self.pipeline_repository.find_all()
        return [
            p
            for p in all_pipelines
            if p.is_scheduled and p.status == PipelineStatus.ACTIVE
        ]

    def _calculate_success_rate(self, runs: list[PipelineRun]) -> float:
        """Calculate success rate for pipeline runs."""
        if not runs:
            return 0.0

        successful = len([r for r in runs if r.is_completed])
        return successful / len(runs)

    def _calculate_average_duration(self, runs: list[PipelineRun]) -> float | None:
        """Calculate average duration for completed runs."""
        completed_runs = [r for r in runs if r.duration_seconds is not None]
        if not completed_runs:
            return None

        total_duration = sum(r.duration_seconds for r in completed_runs)
        return total_duration / len(completed_runs)

    def _calculate_run_frequency(self, runs: list[PipelineRun]) -> dict[str, float]:
        """Calculate run frequency statistics."""
        if len(runs) < 2:
            return {"daily_average": 0.0, "weekly_average": 0.0}

        # Sort by start time
        sorted_runs = sorted(
            [r for r in runs if r.started_at], key=lambda r: r.started_at
        )

        if len(sorted_runs) < 2:
            return {"daily_average": 0.0, "weekly_average": 0.0}

        # Calculate time span
        time_span = sorted_runs[-1].started_at - sorted_runs[0].started_at
        days = time_span.total_seconds() / (24 * 3600)

        if days == 0:
            return {"daily_average": len(runs), "weekly_average": len(runs) * 7}

        daily_avg = len(runs) / days
        weekly_avg = daily_avg * 7

        return {"daily_average": daily_avg, "weekly_average": weekly_avg}

    def _analyze_step_failures(self, runs: list[PipelineRun]) -> dict[str, Any]:
        """Analyze which steps fail most frequently."""
        failed_runs = [r for r in runs if r.is_failed]

        step_failures = {}
        total_failures = len(failed_runs)

        for run in failed_runs:
            failed_step = run.metadata.get("failed_step_id")
            if failed_step:
                step_failures[failed_step] = step_failures.get(failed_step, 0) + 1

        return {
            "total_failures": total_failures,
            "step_failure_counts": step_failures,
            "most_problematic_step": max(step_failures.items(), key=lambda x: x[1])[0]
            if step_failures
            else None,
        }

    def _calculate_performance_trend(self, runs: list[PipelineRun]) -> dict[str, Any]:
        """Calculate performance trend over time."""
        if len(runs) < 5:
            return {"trend": "insufficient_data", "recent_performance": "unknown"}

        # Sort by start time
        sorted_runs = sorted(
            [r for r in runs if r.started_at], key=lambda r: r.started_at
        )

        # Compare recent vs historical performance
        recent_count = min(10, len(sorted_runs) // 2)
        recent_runs = sorted_runs[-recent_count:]
        historical_runs = sorted_runs[:-recent_count]

        recent_success_rate = self._calculate_success_rate(recent_runs)
        historical_success_rate = self._calculate_success_rate(historical_runs)

        if recent_success_rate > historical_success_rate + 0.1:
            trend = "improving"
        elif recent_success_rate < historical_success_rate - 0.1:
            trend = "declining"
        else:
            trend = "stable"

        return {
            "trend": trend,
            "recent_success_rate": recent_success_rate,
            "historical_success_rate": historical_success_rate,
            "recent_performance": "good"
            if recent_success_rate > 0.8
            else "poor"
            if recent_success_rate < 0.5
            else "moderate",
        }

    def _get_last_successful_run(
        self, runs: list[PipelineRun]
    ) -> dict[str, Any] | None:
        """Get information about the last successful run."""
        successful_runs = [r for r in runs if r.is_completed and r.started_at]
        if not successful_runs:
            return None

        last_successful = max(successful_runs, key=lambda r: r.started_at)
        return {
            "run_id": str(last_successful.id),
            "started_at": last_successful.started_at.isoformat(),
            "duration_seconds": last_successful.duration_seconds,
        }

    def _get_last_failed_run(self, runs: list[PipelineRun]) -> dict[str, Any] | None:
        """Get information about the last failed run."""
        failed_runs = [r for r in runs if r.is_failed and r.started_at]
        if not failed_runs:
            return None

        last_failed = max(failed_runs, key=lambda r: r.started_at)
        return {
            "run_id": str(last_failed.id),
            "started_at": last_failed.started_at.isoformat(),
            "error_message": last_failed.error_message,
            "failed_step": last_failed.metadata.get("failed_step_id"),
        }
