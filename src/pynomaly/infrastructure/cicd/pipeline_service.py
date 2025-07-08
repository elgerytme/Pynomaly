"""CI/CD pipeline service for managing automated testing and deployment workflows."""

from __future__ import annotations

import asyncio
import json
import logging
import subprocess
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional, Set
from uuid import UUID, uuid4

from pynomaly.domain.models.cicd import (
    Deployment,
    DeploymentEnvironment,
    DeploymentStrategy,
    Pipeline,
    PipelineMetrics,
    PipelineStage,
    PipelineStatus,
    PipelineTemplate,
    TestResult,
    TestSuite,
    TestType,
    TriggerType,
)


class PipelineService:
    """Service for managing CI/CD pipeline execution and lifecycle."""

    def __init__(self, workspace_path: str = "/tmp/pynomaly-ci"):
        self.workspace_path = Path(workspace_path)
        self.workspace_path.mkdir(parents=True, exist_ok=True)
        self.logger = logging.getLogger(__name__)

        # Pipeline storage (would be replaced with persistent storage in production)
        self.pipelines: Dict[UUID, Pipeline] = {}
        self.pipeline_templates: Dict[UUID, PipelineTemplate] = {}
        self.pipeline_metrics: Dict[UUID, PipelineMetrics] = {}

        # Execution tracking
        self.running_pipelines: Set[UUID] = set()
        self.pipeline_queues: Dict[str, List[UUID]] = {
            "high": [],
            "normal": [],
            "low": [],
        }

        # Background tasks
        self.execution_tasks: Set[asyncio.Task] = set()
        self.is_running = False

        # Default templates
        self._initialize_default_templates()

        self.logger.info(f"Pipeline service initialized with workspace: {workspace_path}")

    def _initialize_default_templates(self) -> None:
        """Initialize default pipeline templates."""

        # Python package template
        python_template = PipelineTemplate(
            template_id=uuid4(),
            name="Python Package CI/CD",
            description="Standard CI/CD pipeline for Python packages",
            template_type="python_package",
            stages_config=[
                {
                    "name": "checkout",
                    "stage_type": "build",
                    "commands": ["git checkout $COMMIT_SHA"],
                    "timeout_minutes": 5,
                },
                {
                    "name": "setup",
                    "stage_type": "build",
                    "commands": [
                        "python -m pip install --upgrade pip",
                        "pip install poetry",
                        "poetry install",
                    ],
                    "depends_on": ["checkout"],
                    "timeout_minutes": 10,
                },
                {
                    "name": "lint",
                    "stage_type": "test",
                    "commands": [
                        "poetry run ruff check src/",
                        "poetry run mypy src/",
                    ],
                    "depends_on": ["setup"],
                    "timeout_minutes": 5,
                },
                {
                    "name": "test",
                    "stage_type": "test",
                    "commands": ["poetry run pytest tests/ -v --cov"],
                    "depends_on": ["setup"],
                    "timeout_minutes": 15,
                },
                {
                    "name": "build",
                    "stage_type": "build",
                    "commands": ["poetry build"],
                    "depends_on": ["lint", "test"],
                    "timeout_minutes": 5,
                },
                {
                    "name": "deploy_staging",
                    "stage_type": "deploy",
                    "commands": ["poetry publish --repository staging"],
                    "depends_on": ["build"],
                    "timeout_minutes": 10,
                    "environment": {"DEPLOYMENT_ENV": "staging"},
                },
            ],
            environment_variables={
                "PYTHON_VERSION": "3.11",
                "POETRY_VERSION": "1.6.1",
            },
            quality_gates={
                "test_coverage_threshold": 80,
                "lint_errors_allowed": 0,
                "security_scan_required": True,
            },
        )

        self.pipeline_templates[python_template.template_id] = python_template

        # ML model template
        ml_template = PipelineTemplate(
            template_id=uuid4(),
            name="ML Model CI/CD",
            description="CI/CD pipeline for machine learning models",
            template_type="ml_model",
            stages_config=[
                {
                    "name": "checkout",
                    "stage_type": "build",
                    "commands": ["git checkout $COMMIT_SHA"],
                    "timeout_minutes": 5,
                },
                {
                    "name": "setup",
                    "stage_type": "build",
                    "commands": [
                        "python -m pip install --upgrade pip",
                        "pip install -r requirements.txt",
                    ],
                    "depends_on": ["checkout"],
                    "timeout_minutes": 15,
                },
                {
                    "name": "data_validation",
                    "stage_type": "test",
                    "commands": ["python scripts/validate_data.py"],
                    "depends_on": ["setup"],
                    "timeout_minutes": 10,
                },
                {
                    "name": "model_training",
                    "stage_type": "build",
                    "commands": ["python scripts/train_model.py"],
                    "depends_on": ["data_validation"],
                    "timeout_minutes": 60,
                },
                {
                    "name": "model_evaluation",
                    "stage_type": "test",
                    "commands": ["python scripts/evaluate_model.py"],
                    "depends_on": ["model_training"],
                    "timeout_minutes": 20,
                },
                {
                    "name": "deploy_model",
                    "stage_type": "deploy",
                    "commands": ["python scripts/deploy_model.py"],
                    "depends_on": ["model_evaluation"],
                    "timeout_minutes": 15,
                },
            ],
            environment_variables={
                "MODEL_VERSION": "latest",
                "DATA_PATH": "/data",
            },
            quality_gates={
                "model_accuracy_threshold": 0.85,
                "performance_regression_threshold": 0.05,
            },
        )

        self.pipeline_templates[ml_template.template_id] = ml_template

    async def start_pipeline_executor(self) -> None:
        """Start background pipeline executor."""

        if self.is_running:
            return

        self.is_running = True

        # Start executor tasks
        tasks = [
            asyncio.create_task(self._pipeline_executor_loop()),
            asyncio.create_task(self._metrics_collector_loop()),
            asyncio.create_task(self._cleanup_loop()),
        ]

        self.execution_tasks.update(tasks)

        self.logger.info("Started pipeline executor")

    async def stop_pipeline_executor(self) -> None:
        """Stop background pipeline executor."""

        self.is_running = False

        # Cancel running pipelines
        for pipeline_id in list(self.running_pipelines):
            await self.cancel_pipeline(pipeline_id)

        # Cancel executor tasks
        for task in self.execution_tasks:
            task.cancel()

        await asyncio.gather(*self.execution_tasks, return_exceptions=True)
        self.execution_tasks.clear()

        self.logger.info("Stopped pipeline executor")

    async def create_pipeline_from_template(
        self,
        template_id: UUID,
        pipeline_name: str,
        repository_url: str,
        branch: str = "main",
        commit_sha: str = "",
        triggered_by: Optional[UUID] = None,
        trigger_type: TriggerType = TriggerType.MANUAL,
        environment_overrides: Optional[Dict[str, str]] = None,
    ) -> Pipeline:
        """Create pipeline from template."""

        if template_id not in self.pipeline_templates:
            raise ValueError(f"Template not found: {template_id}")

        template = self.pipeline_templates[template_id]

        pipeline = template.create_pipeline(
            pipeline_name=pipeline_name,
            repository_url=repository_url,
            branch=branch,
            commit_sha=commit_sha,
        )

        # Apply overrides
        if triggered_by:
            pipeline.triggered_by = triggered_by

        pipeline.trigger_type = trigger_type

        if environment_overrides:
            pipeline.environment_variables.update(environment_overrides)

        # Assign pipeline number
        pipeline.pipeline_number = len(self.pipelines) + 1

        self.pipelines[pipeline.pipeline_id] = pipeline

        self.logger.info(f"Created pipeline: {pipeline_name} from template {template.name}")
        return pipeline

    async def trigger_pipeline(
        self,
        pipeline_id: UUID,
        priority: str = "normal",
    ) -> bool:
        """Trigger pipeline execution."""

        if pipeline_id not in self.pipelines:
            return False

        pipeline = self.pipelines[pipeline_id]

        # Check if already running
        if pipeline_id in self.running_pipelines:
            self.logger.warning(f"Pipeline {pipeline_id} is already running")
            return False

        # Add to execution queue
        if priority not in self.pipeline_queues:
            priority = "normal"

        self.pipeline_queues[priority].append(pipeline_id)

        self.logger.info(f"Queued pipeline {pipeline.name} for execution (priority: {priority})")
        return True

    async def execute_pipeline(self, pipeline_id: UUID) -> bool:
        """Execute a pipeline."""

        if pipeline_id not in self.pipelines:
            return False

        pipeline = self.pipelines[pipeline_id]

        if pipeline_id in self.running_pipelines:
            return False

        self.running_pipelines.add(pipeline_id)

        try:
            # Start pipeline
            pipeline.start_pipeline()

            # Prepare workspace
            workspace = self.workspace_path / str(pipeline_id)
            workspace.mkdir(parents=True, exist_ok=True)

            # Clone repository
            await self._clone_repository(pipeline, workspace)

            # Execute stages
            success = await self._execute_stages(pipeline, workspace)

            # Complete pipeline
            final_status = PipelineStatus.SUCCESS if success else PipelineStatus.FAILED
            pipeline.complete_pipeline(final_status)

            self.logger.info(f"Pipeline {pipeline.name} completed with status: {final_status.value}")
            return success

        except Exception as e:
            pipeline.complete_pipeline(PipelineStatus.FAILED)
            self.logger.error(f"Pipeline {pipeline.name} failed with error: {e}")
            return False

        finally:
            self.running_pipelines.discard(pipeline_id)

    async def cancel_pipeline(self, pipeline_id: UUID) -> bool:
        """Cancel running pipeline."""

        if pipeline_id not in self.pipelines:
            return False

        pipeline = self.pipelines[pipeline_id]

        if pipeline_id not in self.running_pipelines:
            return False

        # Mark pipeline as cancelled
        pipeline.complete_pipeline(PipelineStatus.CANCELLED)

        # Cancel running stages
        for stage in pipeline.stages:
            if stage.status == PipelineStatus.RUNNING:
                stage.complete_execution(PipelineStatus.CANCELLED)

        self.running_pipelines.discard(pipeline_id)

        self.logger.info(f"Cancelled pipeline: {pipeline.name}")
        return True

    async def get_pipeline_status(self, pipeline_id: UUID) -> Optional[Dict[str, Any]]:
        """Get pipeline execution status."""

        if pipeline_id not in self.pipelines:
            return None

        pipeline = self.pipelines[pipeline_id]

        return {
            "pipeline": pipeline.get_pipeline_summary(),
            "is_running": pipeline_id in self.running_pipelines,
            "queue_position": self._get_queue_position(pipeline_id),
        }

    async def get_pipeline_logs(
        self,
        pipeline_id: UUID,
        stage_name: Optional[str] = None,
    ) -> Optional[Dict[str, Any]]:
        """Get pipeline execution logs."""

        if pipeline_id not in self.pipelines:
            return None

        pipeline = self.pipelines[pipeline_id]

        if stage_name:
            stage = pipeline.get_stage_by_name(stage_name)
            if not stage:
                return None

            return {
                "stage": stage_name,
                "output": stage.output,
                "error": stage.error_message,
                "exit_code": stage.exit_code,
            }

        # Return all stage logs
        logs = {}
        for stage in pipeline.stages:
            logs[stage.name] = {
                "output": stage.output,
                "error": stage.error_message,
                "exit_code": stage.exit_code,
                "status": stage.status.value,
            }

        return logs

    async def get_pipeline_metrics(self, pipeline_id: UUID) -> Optional[PipelineMetrics]:
        """Get pipeline metrics."""

        if pipeline_id not in self.pipeline_metrics:
            # Calculate metrics for this pipeline
            await self._calculate_pipeline_metrics(pipeline_id)

        return self.pipeline_metrics.get(pipeline_id)

    async def list_pipelines(
        self,
        status_filter: Optional[PipelineStatus] = None,
        limit: int = 50,
    ) -> List[Dict[str, Any]]:
        """List pipelines with optional filtering."""

        pipelines = list(self.pipelines.values())

        if status_filter:
            pipelines = [p for p in pipelines if p.status == status_filter]

        # Sort by creation time (newest first)
        pipelines.sort(key=lambda p: p.created_at, reverse=True)

        return [p.get_pipeline_summary() for p in pipelines[:limit]]

    async def _pipeline_executor_loop(self) -> None:
        """Background loop for executing queued pipelines."""

        while self.is_running:
            try:
                # Check for pipelines to execute
                pipeline_id = await self._get_next_pipeline()

                if pipeline_id:
                    # Execute pipeline in background
                    task = asyncio.create_task(self.execute_pipeline(pipeline_id))
                    self.execution_tasks.add(task)

                    # Clean up completed tasks
                    self.execution_tasks = {
                        task for task in self.execution_tasks
                        if not task.done()
                    }

            except Exception as e:
                self.logger.error(f"Pipeline executor error: {e}")

            await asyncio.sleep(5)  # Check every 5 seconds

    async def _get_next_pipeline(self) -> Optional[UUID]:
        """Get next pipeline from queue (priority order)."""

        # Check high priority first
        for priority in ["high", "normal", "low"]:
            if self.pipeline_queues[priority]:
                return self.pipeline_queues[priority].pop(0)

        return None

    def _get_queue_position(self, pipeline_id: UUID) -> Optional[int]:
        """Get pipeline position in queue."""

        for priority, queue in self.pipeline_queues.items():
            if pipeline_id in queue:
                return queue.index(pipeline_id) + 1

        return None

    async def _clone_repository(self, pipeline: Pipeline, workspace: Path) -> None:
        """Clone repository to workspace."""

        try:
            # Simple git clone (would use proper git library in production)
            cmd = [
                "git", "clone",
                "--depth", "1",
                "--branch", pipeline.branch,
                pipeline.repository_url,
                str(workspace / "repo")
            ]

            if pipeline.commit_sha:
                # Clone full repo if specific commit needed
                cmd = [
                    "git", "clone",
                    pipeline.repository_url,
                    str(workspace / "repo")
                ]

            process = await asyncio.create_subprocess_exec(
                *cmd,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )

            stdout, stderr = await process.communicate()

            if process.returncode != 0:
                raise Exception(f"Git clone failed: {stderr.decode()}")

            # Checkout specific commit if provided
            if pipeline.commit_sha:
                checkout_cmd = ["git", "checkout", pipeline.commit_sha]
                process = await asyncio.create_subprocess_exec(
                    *checkout_cmd,
                    cwd=workspace / "repo",
                    stdout=asyncio.subprocess.PIPE,
                    stderr=asyncio.subprocess.PIPE,
                )

                await process.communicate()

                if process.returncode != 0:
                    raise Exception(f"Git checkout failed for commit {pipeline.commit_sha}")

        except Exception as e:
            self.logger.error(f"Repository clone failed: {e}")
            raise

    async def _execute_stages(self, pipeline: Pipeline, workspace: Path) -> bool:
        """Execute pipeline stages."""

        # Build dependency graph
        completed_stages = set()

        while len(completed_stages) < len(pipeline.stages):
            # Find stages ready to execute
            ready_stages = []

            for stage in pipeline.stages:
                if stage.name in completed_stages:
                    continue

                # Check if all dependencies are completed
                dependencies_met = all(
                    dep in completed_stages
                    for dep in stage.depends_on
                )

                if dependencies_met:
                    ready_stages.append(stage)

            if not ready_stages:
                # No stages ready - check for circular dependencies
                remaining_stages = [
                    s.name for s in pipeline.stages
                    if s.name not in completed_stages
                ]
                self.logger.error(f"Circular dependency detected in stages: {remaining_stages}")
                return False

            # Execute ready stages in parallel
            stage_tasks = []
            for stage in ready_stages:
                task = asyncio.create_task(
                    self._execute_stage(stage, pipeline, workspace)
                )
                stage_tasks.append((stage, task))

            # Wait for all stages to complete
            for stage, task in stage_tasks:
                success = await task
                completed_stages.add(stage.name)

                if not success:
                    # Stage failed - pipeline fails
                    self.logger.error(f"Stage {stage.name} failed")
                    return False

        return True

    async def _execute_stage(
        self,
        stage: PipelineStage,
        pipeline: Pipeline,
        workspace: Path,
    ) -> bool:
        """Execute a single pipeline stage."""

        stage.start_execution()

        try:
            # Prepare environment
            env = {
                **pipeline.environment_variables,
                **stage.environment,
                "PIPELINE_ID": str(pipeline.pipeline_id),
                "PIPELINE_NAME": pipeline.name,
                "COMMIT_SHA": pipeline.commit_sha or "HEAD",
                "BRANCH": pipeline.branch,
            }

            # Set working directory
            work_dir = workspace / "repo"
            if stage.working_directory:
                work_dir = work_dir / stage.working_directory

            # Execute commands
            output_lines = []

            for command in stage.commands:
                # Replace environment variables in command
                formatted_command = self._substitute_env_vars(command, env)

                self.logger.debug(f"Executing: {formatted_command}")

                # Execute command
                process = await asyncio.create_subprocess_shell(
                    formatted_command,
                    cwd=work_dir,
                    env=env,
                    stdout=asyncio.subprocess.PIPE,
                    stderr=asyncio.subprocess.STDOUT,
                )

                # Stream output
                while True:
                    line = await process.stdout.readline()
                    if not line:
                        break

                    line_str = line.decode().strip()
                    output_lines.append(line_str)
                    self.logger.debug(f"Stage {stage.name}: {line_str}")

                await process.wait()

                if process.returncode != 0:
                    stage.complete_execution(
                        PipelineStatus.FAILED,
                        exit_code=process.returncode,
                        error_message=f"Command failed: {formatted_command}",
                    )
                    stage.output = "\n".join(output_lines)
                    return False

            # Stage completed successfully
            stage.complete_execution(PipelineStatus.SUCCESS, exit_code=0)
            stage.output = "\n".join(output_lines)

            return True

        except Exception as e:
            stage.complete_execution(
                PipelineStatus.FAILED,
                error_message=str(e),
            )
            return False

    def _substitute_env_vars(self, command: str, env: Dict[str, str]) -> str:
        """Substitute environment variables in command."""

        result = command
        for key, value in env.items():
            result = result.replace(f"${key}", value)
            result = result.replace(f"${{{key}}}", value)

        return result

    async def _metrics_collector_loop(self) -> None:
        """Background loop for collecting pipeline metrics."""

        while self.is_running:
            try:
                # Calculate metrics for all pipelines
                for pipeline_id in self.pipelines:
                    await self._calculate_pipeline_metrics(pipeline_id)

            except Exception as e:
                self.logger.error(f"Metrics collection error: {e}")

            await asyncio.sleep(300)  # Calculate every 5 minutes

    async def _calculate_pipeline_metrics(self, pipeline_id: UUID) -> None:
        """Calculate metrics for a specific pipeline."""

        if pipeline_id not in self.pipelines:
            return

        # Get related pipelines (same name, different executions)
        target_pipeline = self.pipelines[pipeline_id]
        related_pipelines = [
            p for p in self.pipelines.values()
            if p.name == target_pipeline.name
        ]

        if pipeline_id not in self.pipeline_metrics:
            metrics = PipelineMetrics(
                metrics_id=uuid4(),
                pipeline_id=pipeline_id,
            )
            self.pipeline_metrics[pipeline_id] = metrics
        else:
            metrics = self.pipeline_metrics[pipeline_id]

        # Calculate metrics from related executions
        metrics.calculate_metrics(related_pipelines)

    async def _cleanup_loop(self) -> None:
        """Background loop for cleaning up old pipeline data."""

        while self.is_running:
            try:
                # Clean up old workspaces
                await self._cleanup_workspaces()

                # Clean up old pipeline data
                await self._cleanup_old_pipelines()

            except Exception as e:
                self.logger.error(f"Cleanup error: {e}")

            await asyncio.sleep(3600)  # Clean up every hour

    async def _cleanup_workspaces(self) -> None:
        """Clean up old pipeline workspaces."""

        cutoff_time = datetime.utcnow() - timedelta(days=7)

        for workspace_dir in self.workspace_path.iterdir():
            if workspace_dir.is_dir():
                try:
                    # Check if workspace is for old pipeline
                    pipeline_id = UUID(workspace_dir.name)

                    if pipeline_id in self.pipelines:
                        pipeline = self.pipelines[pipeline_id]
                        if pipeline.created_at < cutoff_time and pipeline_id not in self.running_pipelines:
                            # Remove workspace
                            import shutil
                            shutil.rmtree(workspace_dir)
                            self.logger.debug(f"Cleaned up workspace: {workspace_dir}")

                except (ValueError, Exception):
                    # Invalid UUID or other error - skip
                    continue

    async def _cleanup_old_pipelines(self) -> None:
        """Clean up old pipeline records."""

        cutoff_time = datetime.utcnow() - timedelta(days=90)

        pipelines_to_remove = []
        for pipeline_id, pipeline in self.pipelines.items():
            if (pipeline.created_at < cutoff_time and
                pipeline_id not in self.running_pipelines):
                pipelines_to_remove.append(pipeline_id)

        for pipeline_id in pipelines_to_remove:
            del self.pipelines[pipeline_id]
            if pipeline_id in self.pipeline_metrics:
                del self.pipeline_metrics[pipeline_id]

        if pipelines_to_remove:
            self.logger.info(f"Cleaned up {len(pipelines_to_remove)} old pipeline records")
