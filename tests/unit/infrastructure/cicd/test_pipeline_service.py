"""Unit tests for CI/CD pipeline service."""

import asyncio
import pytest
from datetime import datetime, timedelta
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch
from uuid import uuid4

from pynomaly.domain.models.cicd import (
    DeploymentEnvironment,
    DeploymentStrategy,
    Pipeline,
    PipelineStatus,
    PipelineTemplate,
    TriggerType,
)
from pynomaly.infrastructure.cicd.pipeline_service import PipelineService


@pytest.fixture
def pipeline_service():
    """Create pipeline service for testing."""
    return PipelineService(workspace_path="/tmp/test-pynomaly-ci")


@pytest.fixture
def sample_template():
    """Create sample pipeline template."""
    return PipelineTemplate(
        template_id=uuid4(),
        name="Test Template",
        description="Test template for unit tests",
        template_type="test",
        stages_config=[
            {
                "name": "test",
                "stage_type": "test",
                "commands": ["echo 'testing'"],
                "timeout_minutes": 5,
            }
        ],
    )


class TestPipelineService:
    """Test cases for PipelineService."""

    def test_initialization(self, pipeline_service):
        """Test service initialization."""
        assert pipeline_service.workspace_path.exists()
        assert isinstance(pipeline_service.pipelines, dict)
        assert isinstance(pipeline_service.pipeline_templates, dict)
        assert len(pipeline_service.pipeline_templates) >= 2  # Default templates
        assert not pipeline_service.is_running

    def test_default_templates_created(self, pipeline_service):
        """Test that default templates are created."""
        template_types = [
            t.template_type for t in pipeline_service.pipeline_templates.values()
        ]
        assert "python_package" in template_types
        assert "ml_model" in template_types

    async def test_create_pipeline_from_template(
        self, pipeline_service, sample_template
    ):
        """Test creating pipeline from template."""
        # Add template to service
        pipeline_service.pipeline_templates[sample_template.template_id] = (
            sample_template
        )

        # Create pipeline
        pipeline = await pipeline_service.create_pipeline_from_template(
            template_id=sample_template.template_id,
            pipeline_name="Test Pipeline",
            repository_url="https://github.com/test/repo.git",
            branch="main",
            commit_sha="abc123",
        )

        assert pipeline.name == "Test Pipeline"
        assert pipeline.repository_url == "https://github.com/test/repo.git"
        assert pipeline.branch == "main"
        assert pipeline.commit_sha == "abc123"
        assert pipeline.status == PipelineStatus.PENDING
        assert len(pipeline.stages) == 1
        assert pipeline.stages[0].name == "test"
        assert pipeline.pipeline_id in pipeline_service.pipelines

    async def test_create_pipeline_from_nonexistent_template(self, pipeline_service):
        """Test creating pipeline from non-existent template raises error."""
        with pytest.raises(ValueError, match="Template not found"):
            await pipeline_service.create_pipeline_from_template(
                template_id=uuid4(),
                pipeline_name="Test Pipeline",
                repository_url="https://github.com/test/repo.git",
            )

    async def test_trigger_pipeline(self, pipeline_service, sample_template):
        """Test triggering pipeline execution."""
        # Create pipeline
        pipeline_service.pipeline_templates[sample_template.template_id] = (
            sample_template
        )
        pipeline = await pipeline_service.create_pipeline_from_template(
            template_id=sample_template.template_id,
            pipeline_name="Test Pipeline",
            repository_url="https://github.com/test/repo.git",
        )

        # Trigger pipeline
        success = await pipeline_service.trigger_pipeline(pipeline.pipeline_id)
        assert success
        assert pipeline.pipeline_id in pipeline_service.pipeline_queues["normal"]

    async def test_trigger_nonexistent_pipeline(self, pipeline_service):
        """Test triggering non-existent pipeline returns False."""
        success = await pipeline_service.trigger_pipeline(uuid4())
        assert not success

    async def test_trigger_already_running_pipeline(
        self, pipeline_service, sample_template
    ):
        """Test triggering already running pipeline returns False."""
        # Create pipeline
        pipeline_service.pipeline_templates[sample_template.template_id] = (
            sample_template
        )
        pipeline = await pipeline_service.create_pipeline_from_template(
            template_id=sample_template.template_id,
            pipeline_name="Test Pipeline",
            repository_url="https://github.com/test/repo.git",
        )

        # Mark as running
        pipeline_service.running_pipelines.add(pipeline.pipeline_id)

        # Try to trigger
        success = await pipeline_service.trigger_pipeline(pipeline.pipeline_id)
        assert not success

    @patch(
        "pynomaly.infrastructure.cicd.pipeline_service.asyncio.create_subprocess_exec"
    )
    async def test_execute_pipeline_success(
        self, mock_subprocess, pipeline_service, sample_template
    ):
        """Test successful pipeline execution."""
        # Mock subprocess
        mock_process = AsyncMock()
        mock_process.returncode = 0
        mock_process.communicate.return_value = (b"output", b"")
        mock_subprocess.return_value = mock_process

        # Create pipeline
        pipeline_service.pipeline_templates[sample_template.template_id] = (
            sample_template
        )
        pipeline = await pipeline_service.create_pipeline_from_template(
            template_id=sample_template.template_id,
            pipeline_name="Test Pipeline",
            repository_url="https://github.com/test/repo.git",
        )

        # Execute pipeline
        success = await pipeline_service.execute_pipeline(pipeline.pipeline_id)

        assert success
        assert pipeline.status == PipelineStatus.SUCCESS
        assert pipeline.pipeline_id not in pipeline_service.running_pipelines

    @patch(
        "pynomaly.infrastructure.cicd.pipeline_service.asyncio.create_subprocess_exec"
    )
    async def test_execute_pipeline_failure(
        self, mock_subprocess, pipeline_service, sample_template
    ):
        """Test failed pipeline execution."""
        # Mock subprocess failure
        mock_process = AsyncMock()
        mock_process.returncode = 1
        mock_process.communicate.return_value = (b"output", b"error")
        mock_subprocess.return_value = mock_process

        # Create pipeline
        pipeline_service.pipeline_templates[sample_template.template_id] = (
            sample_template
        )
        pipeline = await pipeline_service.create_pipeline_from_template(
            template_id=sample_template.template_id,
            pipeline_name="Test Pipeline",
            repository_url="https://github.com/test/repo.git",
        )

        # Execute pipeline
        success = await pipeline_service.execute_pipeline(pipeline.pipeline_id)

        assert not success
        assert pipeline.status == PipelineStatus.FAILED
        assert pipeline.pipeline_id not in pipeline_service.running_pipelines

    async def test_cancel_pipeline(self, pipeline_service, sample_template):
        """Test canceling running pipeline."""
        # Create pipeline
        pipeline_service.pipeline_templates[sample_template.template_id] = (
            sample_template
        )
        pipeline = await pipeline_service.create_pipeline_from_template(
            template_id=sample_template.template_id,
            pipeline_name="Test Pipeline",
            repository_url="https://github.com/test/repo.git",
        )

        # Mark as running
        pipeline_service.running_pipelines.add(pipeline.pipeline_id)

        # Cancel pipeline
        success = await pipeline_service.cancel_pipeline(pipeline.pipeline_id)

        assert success
        assert pipeline.status == PipelineStatus.CANCELLED
        assert pipeline.pipeline_id not in pipeline_service.running_pipelines

    async def test_get_pipeline_status(self, pipeline_service, sample_template):
        """Test getting pipeline status."""
        # Create pipeline
        pipeline_service.pipeline_templates[sample_template.template_id] = (
            sample_template
        )
        pipeline = await pipeline_service.create_pipeline_from_template(
            template_id=sample_template.template_id,
            pipeline_name="Test Pipeline",
            repository_url="https://github.com/test/repo.git",
        )

        # Get status
        status = await pipeline_service.get_pipeline_status(pipeline.pipeline_id)

        assert status is not None
        assert "pipeline" in status
        assert "is_running" in status
        assert status["pipeline"]["name"] == "Test Pipeline"
        assert not status["is_running"]

    async def test_get_nonexistent_pipeline_status(self, pipeline_service):
        """Test getting status of non-existent pipeline."""
        status = await pipeline_service.get_pipeline_status(uuid4())
        assert status is None

    async def test_list_pipelines(self, pipeline_service, sample_template):
        """Test listing pipelines."""
        # Create multiple pipelines
        pipeline_service.pipeline_templates[sample_template.template_id] = (
            sample_template
        )

        pipelines = []
        for i in range(3):
            pipeline = await pipeline_service.create_pipeline_from_template(
                template_id=sample_template.template_id,
                pipeline_name=f"Test Pipeline {i}",
                repository_url="https://github.com/test/repo.git",
            )
            pipelines.append(pipeline)

        # List pipelines
        pipeline_list = await pipeline_service.list_pipelines()

        assert len(pipeline_list) == 3
        assert all("name" in p for p in pipeline_list)
        assert all("status" in p for p in pipeline_list)

    async def test_list_pipelines_with_filter(self, pipeline_service, sample_template):
        """Test listing pipelines with status filter."""
        # Create pipeline
        pipeline_service.pipeline_templates[sample_template.template_id] = (
            sample_template
        )
        pipeline = await pipeline_service.create_pipeline_from_template(
            template_id=sample_template.template_id,
            pipeline_name="Test Pipeline",
            repository_url="https://github.com/test/repo.git",
        )

        # Mark as success
        pipeline.complete_pipeline(PipelineStatus.SUCCESS)

        # List with filter
        success_pipelines = await pipeline_service.list_pipelines(
            status_filter=PipelineStatus.SUCCESS
        )
        pending_pipelines = await pipeline_service.list_pipelines(
            status_filter=PipelineStatus.PENDING
        )

        assert len(success_pipelines) == 1
        assert len(pending_pipelines) == 0

    def test_substitute_env_vars(self, pipeline_service):
        """Test environment variable substitution."""
        env = {"VAR1": "value1", "VAR2": "value2"}

        # Test $VAR format
        result = pipeline_service._substitute_env_vars("echo $VAR1 $VAR2", env)
        assert result == "echo value1 value2"

        # Test ${VAR} format
        result = pipeline_service._substitute_env_vars("echo ${VAR1} ${VAR2}", env)
        assert result == "echo value1 value2"

        # Test mixed format
        result = pipeline_service._substitute_env_vars("echo $VAR1 ${VAR2}", env)
        assert result == "echo value1 value2"

    async def test_get_next_pipeline_priority(self, pipeline_service):
        """Test getting next pipeline respects priority."""
        # Add pipelines to different priority queues
        high_id = uuid4()
        normal_id = uuid4()
        low_id = uuid4()

        pipeline_service.pipeline_queues["high"].append(high_id)
        pipeline_service.pipeline_queues["normal"].append(normal_id)
        pipeline_service.pipeline_queues["low"].append(low_id)

        # High priority should come first
        next_id = await pipeline_service._get_next_pipeline()
        assert next_id == high_id

        # Normal priority should come next
        next_id = await pipeline_service._get_next_pipeline()
        assert next_id == normal_id

        # Low priority should come last
        next_id = await pipeline_service._get_next_pipeline()
        assert next_id == low_id

        # No more pipelines
        next_id = await pipeline_service._get_next_pipeline()
        assert next_id is None

    async def test_get_queue_position(self, pipeline_service):
        """Test getting pipeline position in queue."""
        # Add pipelines to queue
        pipeline_ids = [uuid4() for _ in range(3)]

        for pipeline_id in pipeline_ids:
            pipeline_service.pipeline_queues["normal"].append(pipeline_id)

        # Check positions
        assert pipeline_service._get_queue_position(pipeline_ids[0]) == 1
        assert pipeline_service._get_queue_position(pipeline_ids[1]) == 2
        assert pipeline_service._get_queue_position(pipeline_ids[2]) == 3
        assert pipeline_service._get_queue_position(uuid4()) is None

    async def test_start_stop_executor(self, pipeline_service):
        """Test starting and stopping pipeline executor."""
        # Start executor
        await pipeline_service.start_pipeline_executor()
        assert pipeline_service.is_running
        assert len(pipeline_service.execution_tasks) > 0

        # Stop executor
        await pipeline_service.stop_pipeline_executor()
        assert not pipeline_service.is_running
        assert len(pipeline_service.execution_tasks) == 0

    @patch(
        "pynomaly.infrastructure.cicd.pipeline_service.asyncio.create_subprocess_shell"
    )
    async def test_execute_stage_success(
        self, mock_subprocess, pipeline_service, sample_template
    ):
        """Test successful stage execution."""
        # Mock subprocess
        mock_process = AsyncMock()
        mock_process.returncode = 0
        mock_process.stdout.readline.side_effect = [b"test output\n", b""]
        mock_process.wait.return_value = None
        mock_subprocess.return_value = mock_process

        # Create pipeline and stage
        pipeline_service.pipeline_templates[sample_template.template_id] = (
            sample_template
        )
        pipeline = await pipeline_service.create_pipeline_from_template(
            template_id=sample_template.template_id,
            pipeline_name="Test Pipeline",
            repository_url="https://github.com/test/repo.git",
        )

        stage = pipeline.stages[0]
        workspace = Path("/tmp/test")

        # Execute stage
        success = await pipeline_service._execute_stage(stage, pipeline, workspace)

        assert success
        assert stage.status == PipelineStatus.SUCCESS
        assert stage.exit_code == 0
        assert "test output" in stage.output

    @patch(
        "pynomaly.infrastructure.cicd.pipeline_service.asyncio.create_subprocess_shell"
    )
    async def test_execute_stage_failure(
        self, mock_subprocess, pipeline_service, sample_template
    ):
        """Test failed stage execution."""
        # Mock subprocess failure
        mock_process = AsyncMock()
        mock_process.returncode = 1
        mock_process.stdout.readline.side_effect = [b"error output\n", b""]
        mock_process.wait.return_value = None
        mock_subprocess.return_value = mock_process

        # Create pipeline and stage
        pipeline_service.pipeline_templates[sample_template.template_id] = (
            sample_template
        )
        pipeline = await pipeline_service.create_pipeline_from_template(
            template_id=sample_template.template_id,
            pipeline_name="Test Pipeline",
            repository_url="https://github.com/test/repo.git",
        )

        stage = pipeline.stages[0]
        workspace = Path("/tmp/test")

        # Execute stage
        success = await pipeline_service._execute_stage(stage, pipeline, workspace)

        assert not success
        assert stage.status == PipelineStatus.FAILED
        assert stage.exit_code == 1
        assert "error output" in stage.output

    async def test_calculate_pipeline_metrics(self, pipeline_service, sample_template):
        """Test calculating pipeline metrics."""
        # Create pipeline
        pipeline_service.pipeline_templates[sample_template.template_id] = (
            sample_template
        )
        pipeline = await pipeline_service.create_pipeline_from_template(
            template_id=sample_template.template_id,
            pipeline_name="Test Pipeline",
            repository_url="https://github.com/test/repo.git",
        )

        # Complete pipeline
        pipeline.complete_pipeline(PipelineStatus.SUCCESS)

        # Calculate metrics
        await pipeline_service._calculate_pipeline_metrics(pipeline.pipeline_id)

        assert pipeline.pipeline_id in pipeline_service.pipeline_metrics
        metrics = pipeline_service.pipeline_metrics[pipeline.pipeline_id]
        assert metrics.sample_size >= 1

    async def test_get_pipeline_logs(self, pipeline_service, sample_template):
        """Test getting pipeline logs."""
        # Create pipeline
        pipeline_service.pipeline_templates[sample_template.template_id] = (
            sample_template
        )
        pipeline = await pipeline_service.create_pipeline_from_template(
            template_id=sample_template.template_id,
            pipeline_name="Test Pipeline",
            repository_url="https://github.com/test/repo.git",
        )

        # Add some output to stage
        stage = pipeline.stages[0]
        stage.output = "test output"
        stage.error_message = "test error"
        stage.exit_code = 0

        # Get all logs
        logs = await pipeline_service.get_pipeline_logs(pipeline.pipeline_id)
        assert logs is not None
        assert "test" in logs
        assert logs["test"]["output"] == "test output"
        assert logs["test"]["error"] == "test error"
        assert logs["test"]["exit_code"] == 0

        # Get specific stage logs
        stage_logs = await pipeline_service.get_pipeline_logs(
            pipeline.pipeline_id, "test"
        )
        assert stage_logs is not None
        assert stage_logs["stage"] == "test"
        assert stage_logs["output"] == "test output"

    async def test_get_logs_nonexistent_pipeline(self, pipeline_service):
        """Test getting logs for non-existent pipeline."""
        logs = await pipeline_service.get_pipeline_logs(uuid4())
        assert logs is None

    async def test_get_logs_nonexistent_stage(self, pipeline_service, sample_template):
        """Test getting logs for non-existent stage."""
        # Create pipeline
        pipeline_service.pipeline_templates[sample_template.template_id] = (
            sample_template
        )
        pipeline = await pipeline_service.create_pipeline_from_template(
            template_id=sample_template.template_id,
            pipeline_name="Test Pipeline",
            repository_url="https://github.com/test/repo.git",
        )

        # Get logs for non-existent stage
        logs = await pipeline_service.get_pipeline_logs(
            pipeline.pipeline_id, "nonexistent"
        )
        assert logs is None


@pytest.mark.asyncio
class TestPipelineServiceIntegration:
    """Integration tests for PipelineService."""

    async def test_full_pipeline_workflow(self, pipeline_service):
        """Test complete pipeline workflow."""
        # Get python template
        python_template = None
        for template in pipeline_service.pipeline_templates.values():
            if template.template_type == "python_package":
                python_template = template
                break

        assert python_template is not None

        # Create pipeline
        pipeline = await pipeline_service.create_pipeline_from_template(
            template_id=python_template.template_id,
            pipeline_name="Integration Test Pipeline",
            repository_url="https://github.com/test/repo.git",
            branch="main",
            triggered_by=uuid4(),
        )

        # Verify pipeline creation
        assert pipeline.name == "Integration Test Pipeline"
        assert len(pipeline.stages) > 0
        assert pipeline.status == PipelineStatus.PENDING

        # Trigger pipeline
        success = await pipeline_service.trigger_pipeline(pipeline.pipeline_id)
        assert success

        # Check queue
        assert pipeline.pipeline_id in pipeline_service.pipeline_queues["normal"]

        # Get status
        status = await pipeline_service.get_pipeline_status(pipeline.pipeline_id)
        assert status is not None
        assert not status["is_running"]
        assert status["queue_position"] == 1

        # Cancel pipeline
        pipeline_service.running_pipelines.add(pipeline.pipeline_id)  # Simulate running
        success = await pipeline_service.cancel_pipeline(pipeline.pipeline_id)
        assert success
        assert pipeline.status == PipelineStatus.CANCELLED

    async def test_environment_variable_handling(self, pipeline_service):
        """Test environment variable handling in pipeline execution."""
        # Create custom template with environment variables
        template = PipelineTemplate(
            template_id=uuid4(),
            name="Env Test Template",
            description="Template for testing environment variables",
            template_type="env_test",
            stages_config=[
                {
                    "name": "env_stage",
                    "stage_type": "test",
                    "commands": ["echo $TEST_VAR", "echo ${ANOTHER_VAR}"],
                    "environment": {"STAGE_VAR": "stage_value"},
                    "timeout_minutes": 5,
                }
            ],
            environment_variables={
                "TEST_VAR": "test_value",
                "ANOTHER_VAR": "another_value",
            },
        )

        pipeline_service.pipeline_templates[template.template_id] = template

        # Create pipeline
        pipeline = await pipeline_service.create_pipeline_from_template(
            template_id=template.template_id,
            pipeline_name="Env Test Pipeline",
            repository_url="https://github.com/test/repo.git",
        )

        # Check environment variables are applied
        assert pipeline.environment_variables["TEST_VAR"] == "test_value"
        assert pipeline.environment_variables["ANOTHER_VAR"] == "another_value"

        # Check stage environment
        stage = pipeline.stages[0]
        assert stage.environment["STAGE_VAR"] == "stage_value"

        # Test variable substitution
        test_command = "echo $TEST_VAR ${ANOTHER_VAR}"
        substituted = pipeline_service._substitute_env_vars(
            test_command, pipeline.environment_variables
        )
        assert substituted == "echo test_value another_value"
