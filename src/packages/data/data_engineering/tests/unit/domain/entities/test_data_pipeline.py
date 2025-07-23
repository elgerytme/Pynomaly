"""Comprehensive unit tests for DataPipeline domain entity."""

import pytest
from datetime import datetime, timedelta
from uuid import UUID, uuid4
from unittest.mock import patch

from data_engineering.domain.entities.data_pipeline import (
    DataPipeline, PipelineStep, PipelineStatus, StepStatus
)


class TestPipelineStatus:
    """Test cases for PipelineStatus enum."""

    def test_pipeline_status_values(self):
        """Test all PipelineStatus enum values."""
        assert PipelineStatus.CREATED.value == "created"
        assert PipelineStatus.QUEUED.value == "queued"
        assert PipelineStatus.RUNNING.value == "running"
        assert PipelineStatus.COMPLETED.value == "completed"
        assert PipelineStatus.FAILED.value == "failed"
        assert PipelineStatus.CANCELLED.value == "cancelled"
        assert PipelineStatus.PAUSED.value == "paused"


class TestStepStatus:
    """Test cases for StepStatus enum."""

    def test_step_status_values(self):
        """Test all StepStatus enum values."""
        assert StepStatus.PENDING.value == "pending"
        assert StepStatus.RUNNING.value == "running" 
        assert StepStatus.COMPLETED.value == "completed"
        assert StepStatus.FAILED.value == "failed"
        assert StepStatus.SKIPPED.value == "skipped"


class TestPipelineStep:
    """Test cases for PipelineStep entity."""

    def test_initialization_defaults(self):
        """Test step initialization with defaults."""
        step = PipelineStep(name="test_step", step_type="extract")
        
        assert isinstance(step.id, UUID)
        assert step.name == "test_step"
        assert step.step_type == "extract"
        assert step.config == {}
        assert step.dependencies == []
        assert step.status == StepStatus.PENDING
        assert step.started_at is None
        assert step.completed_at is None
        assert step.error_message is None
        assert step.output_data == {}

    def test_initialization_with_data(self):
        """Test step initialization with provided data."""
        step_id = uuid4()
        dependencies = [uuid4(), uuid4()]
        config = {"param1": "value1", "param2": 42}
        started_at = datetime(2024, 1, 15, 10, 0, 0)
        
        step = PipelineStep(
            id=step_id,
            name="transform_step",
            step_type="transform",
            config=config,
            dependencies=dependencies,
            status=StepStatus.RUNNING,
            started_at=started_at
        )
        
        assert step.id == step_id
        assert step.name == "transform_step"
        assert step.step_type == "transform"
        assert step.config == config
        assert step.dependencies == dependencies
        assert step.status == StepStatus.RUNNING
        assert step.started_at == started_at

    def test_post_init_validation_empty_name(self):
        """Test validation fails for empty name."""
        with pytest.raises(ValueError, match="Step name cannot be empty"):
            PipelineStep(name="", step_type="extract")

    def test_post_init_validation_empty_step_type(self):
        """Test validation fails for empty step type."""
        with pytest.raises(ValueError, match="Step type cannot be empty"):
            PipelineStep(name="test_step", step_type="")

    def test_duration_property_none(self):
        """Test duration property when times are None."""
        step = PipelineStep(name="test", step_type="extract")
        assert step.duration is None

    def test_duration_property_calculated(self):
        """Test duration property calculation."""
        step = PipelineStep(name="test", step_type="extract")
        step.started_at = datetime(2024, 1, 15, 10, 0, 0)
        step.completed_at = datetime(2024, 1, 15, 10, 5, 30)
        
        assert step.duration == 330.0  # 5 minutes 30 seconds

    def test_is_completed_property(self):
        """Test is_completed property."""
        step = PipelineStep(name="test", step_type="extract")
        
        assert step.is_completed is False
        
        step.status = StepStatus.COMPLETED
        assert step.is_completed is True
        
        step.status = StepStatus.FAILED
        assert step.is_completed is True
        
        step.status = StepStatus.SKIPPED
        assert step.is_completed is True
        
        step.status = StepStatus.RUNNING
        assert step.is_completed is False

    def test_start_success(self):
        """Test starting step successfully."""
        step = PipelineStep(name="test", step_type="extract")
        
        with patch('data_engineering.domain.entities.data_pipeline.datetime') as mock_datetime:
            mock_now = datetime(2024, 1, 15, 12, 0, 0)
            mock_datetime.utcnow.return_value = mock_now
            
            step.start()
            
            assert step.status == StepStatus.RUNNING
            assert step.started_at == mock_now

    def test_start_from_invalid_status(self):
        """Test starting step from invalid status fails."""
        step = PipelineStep(name="test", step_type="extract")
        step.status = StepStatus.RUNNING
        
        with pytest.raises(ValueError, match="Cannot start step in running status"):
            step.start()

    def test_complete_success(self):
        """Test completing step successfully."""
        step = PipelineStep(name="test", step_type="extract")
        step.status = StepStatus.RUNNING
        
        with patch('data_engineering.domain.entities.data_pipeline.datetime') as mock_datetime:
            mock_now = datetime(2024, 1, 15, 13, 0, 0)
            mock_datetime.utcnow.return_value = mock_now
            
            output_data = {"records_processed": 1000}
            step.complete(output_data)
            
            assert step.status == StepStatus.COMPLETED
            assert step.completed_at == mock_now
            assert step.output_data["records_processed"] == 1000

    def test_complete_without_output_data(self):
        """Test completing step without output data."""
        step = PipelineStep(name="test", step_type="extract")
        step.status = StepStatus.RUNNING
        
        step.complete()
        
        assert step.status == StepStatus.COMPLETED
        assert step.output_data == {}

    def test_complete_from_invalid_status(self):
        """Test completing step from invalid status fails."""
        step = PipelineStep(name="test", step_type="extract")
        
        with pytest.raises(ValueError, match="Cannot complete step in pending status"):
            step.complete()

    def test_fail_step(self):
        """Test failing a step."""
        step = PipelineStep(name="test", step_type="extract")
        step.status = StepStatus.RUNNING
        
        with patch('data_engineering.domain.entities.data_pipeline.datetime') as mock_datetime:
            mock_now = datetime(2024, 1, 15, 14, 0, 0)
            mock_datetime.utcnow.return_value = mock_now
            
            error_message = "Database connection failed"
            step.fail(error_message)
            
            assert step.status == StepStatus.FAILED
            assert step.completed_at == mock_now
            assert step.error_message == error_message

    def test_skip_step(self):
        """Test skipping a step."""
        step = PipelineStep(name="test", step_type="extract")
        
        with patch('data_engineering.domain.entities.data_pipeline.datetime') as mock_datetime:
            mock_now = datetime(2024, 1, 15, 15, 0, 0)
            mock_datetime.utcnow.return_value = mock_now
            
            reason = "Source data not available"
            step.skip(reason)
            
            assert step.status == StepStatus.SKIPPED
            assert step.completed_at == mock_now
            assert step.error_message == f"Skipped: {reason}"


class TestDataPipeline:
    """Test cases for DataPipeline entity."""

    def test_initialization_defaults(self):
        """Test pipeline initialization with defaults."""
        pipeline = DataPipeline(name="test_pipeline")
        
        assert isinstance(pipeline.id, UUID)
        assert pipeline.name == "test_pipeline"
        assert pipeline.description == ""
        assert pipeline.version == "1.0.0"
        assert pipeline.status == PipelineStatus.CREATED
        assert pipeline.steps == []
        assert pipeline.schedule_cron is None
        assert pipeline.max_retry_attempts == 3
        assert pipeline.timeout_seconds is None
        assert isinstance(pipeline.created_at, datetime)
        assert isinstance(pipeline.updated_at, datetime)
        assert pipeline.created_by == ""
        assert pipeline.started_at is None
        assert pipeline.completed_at is None
        assert pipeline.retry_count == 0
        assert pipeline.config == {}
        assert pipeline.environment_vars == {}
        assert pipeline.tags == []

    def test_initialization_with_data(self):
        """Test pipeline initialization with provided data."""
        pipeline_id = uuid4()
        created_at = datetime(2024, 1, 15, 10, 0, 0)
        config = {"batch_size": 1000}
        env_vars = {"ENV": "test"}
        tags = ["etl", "daily"]
        
        pipeline = DataPipeline(
            id=pipeline_id,
            name="complex_pipeline",
            description="Complex ETL pipeline",
            version="2.0.0",
            status=PipelineStatus.QUEUED,
            schedule_cron="0 2 * * *",
            max_retry_attempts=5,
            timeout_seconds=3600,
            created_at=created_at,
            created_by="engineer",
            config=config,
            environment_vars=env_vars,
            tags=tags
        )
        
        assert pipeline.id == pipeline_id
        assert pipeline.name == "complex_pipeline"
        assert pipeline.description == "Complex ETL pipeline"
        assert pipeline.version == "2.0.0"
        assert pipeline.status == PipelineStatus.QUEUED
        assert pipeline.schedule_cron == "0 2 * * *"
        assert pipeline.max_retry_attempts == 5
        assert pipeline.timeout_seconds == 3600
        assert pipeline.created_at == created_at
        assert pipeline.created_by == "engineer"
        assert pipeline.config == config
        assert pipeline.environment_vars == env_vars
        assert pipeline.tags == tags

    def test_post_init_validation_empty_name(self):
        """Test validation fails for empty name."""
        with pytest.raises(ValueError, match="Pipeline name cannot be empty"):
            DataPipeline(name="")

    def test_post_init_validation_name_too_long(self):
        """Test validation fails for name too long."""
        long_name = "a" * 101
        with pytest.raises(ValueError, match="Pipeline name cannot exceed 100 characters"):
            DataPipeline(name=long_name)

    def test_post_init_validation_negative_retry_attempts(self):
        """Test validation fails for negative retry attempts."""
        with pytest.raises(ValueError, match="Max retry attempts cannot be negative"):
            DataPipeline(name="test", max_retry_attempts=-1)

    def test_post_init_validation_invalid_timeout(self):
        """Test validation fails for invalid timeout."""
        with pytest.raises(ValueError, match="Timeout must be positive"):
            DataPipeline(name="test", timeout_seconds=0)

    def test_duration_property_none(self):
        """Test duration property when times are None."""
        pipeline = DataPipeline(name="test")
        assert pipeline.duration is None

    def test_duration_property_calculated(self):
        """Test duration property calculation."""
        pipeline = DataPipeline(name="test")
        pipeline.started_at = datetime(2024, 1, 15, 10, 0, 0)
        pipeline.completed_at = datetime(2024, 1, 15, 12, 30, 0)
        
        assert pipeline.duration == 9000.0  # 2.5 hours in seconds

    def test_is_running_property(self):
        """Test is_running property."""
        pipeline = DataPipeline(name="test")
        
        assert pipeline.is_running is False
        
        pipeline.status = PipelineStatus.RUNNING
        assert pipeline.is_running is True

    def test_is_completed_property(self):
        """Test is_completed property."""
        pipeline = DataPipeline(name="test")
        
        assert pipeline.is_completed is False
        
        pipeline.status = PipelineStatus.COMPLETED
        assert pipeline.is_completed is True
        
        pipeline.status = PipelineStatus.FAILED
        assert pipeline.is_completed is True
        
        pipeline.status = PipelineStatus.CANCELLED
        assert pipeline.is_completed is True
        
        pipeline.status = PipelineStatus.RUNNING
        assert pipeline.is_completed is False

    def test_success_rate_property_no_steps(self):
        """Test success_rate property with no steps."""
        pipeline = DataPipeline(name="test")
        assert pipeline.success_rate == 0.0

    def test_success_rate_property_with_steps(self, sample_pipeline_step):
        """Test success_rate property with steps."""
        pipeline = DataPipeline(name="test")
        
        step1 = PipelineStep(name="step1", step_type="extract")
        step1.status = StepStatus.COMPLETED
        
        step2 = PipelineStep(name="step2", step_type="transform")
        step2.status = StepStatus.FAILED
        
        step3 = PipelineStep(name="step3", step_type="load")
        step3.status = StepStatus.COMPLETED
        
        pipeline.steps = [step1, step2, step3]
        
        assert pipeline.success_rate == 2/3  # 2 out of 3 completed

    def test_add_step_success(self, sample_pipeline_step):
        """Test adding step successfully."""
        pipeline = DataPipeline(name="test")
        
        with patch('data_engineering.domain.entities.data_pipeline.datetime') as mock_datetime:
            mock_now = datetime(2024, 1, 15, 16, 0, 0)
            mock_datetime.utcnow.return_value = mock_now
            
            pipeline.add_step(sample_pipeline_step)
            
            assert len(pipeline.steps) == 1
            assert pipeline.steps[0] == sample_pipeline_step
            assert pipeline.updated_at == mock_now

    def test_add_step_duplicate_name(self, sample_pipeline_step):
        """Test adding step with duplicate name fails."""
        pipeline = DataPipeline(name="test")
        pipeline.add_step(sample_pipeline_step)
        
        duplicate_step = PipelineStep(name="extract_data", step_type="extract")
        
        with pytest.raises(ValueError, match="Step with name 'extract_data' already exists"):
            pipeline.add_step(duplicate_step)

    def test_remove_step_success(self, sample_pipeline_step):
        """Test removing step successfully."""
        pipeline = DataPipeline(name="test")
        pipeline.add_step(sample_pipeline_step)
        
        with patch('data_engineering.domain.entities.data_pipeline.datetime') as mock_datetime:
            mock_now = datetime(2024, 1, 15, 17, 0, 0)
            mock_datetime.utcnow.return_value = mock_now
            
            result = pipeline.remove_step(sample_pipeline_step.id)
            
            assert result is True
            assert len(pipeline.steps) == 0
            assert pipeline.updated_at == mock_now

    def test_remove_step_not_found(self):
        """Test removing non-existent step."""
        pipeline = DataPipeline(name="test")
        
        result = pipeline.remove_step(uuid4())
        
        assert result is False

    def test_remove_step_with_dependencies(self):
        """Test removing step with dependencies fails."""
        pipeline = DataPipeline(name="test")
        
        step1 = PipelineStep(name="step1", step_type="extract")
        step2 = PipelineStep(name="step2", step_type="transform", dependencies=[step1.id])
        
        pipeline.add_step(step1)
        pipeline.add_step(step2)
        
        with pytest.raises(ValueError, match="Cannot remove step: 1 steps depend on it"):
            pipeline.remove_step(step1.id)

    def test_get_step_by_id(self, sample_pipeline_step):
        """Test getting step by ID."""
        pipeline = DataPipeline(name="test")
        pipeline.add_step(sample_pipeline_step)
        
        found_step = pipeline.get_step(sample_pipeline_step.id)
        assert found_step == sample_pipeline_step
        
        not_found = pipeline.get_step(uuid4())
        assert not_found is None

    def test_get_step_by_name(self, sample_pipeline_step):
        """Test getting step by name."""
        pipeline = DataPipeline(name="test")
        pipeline.add_step(sample_pipeline_step)
        
        found_step = pipeline.get_step_by_name("extract_data")
        assert found_step == sample_pipeline_step
        
        not_found = pipeline.get_step_by_name("nonexistent")
        assert not_found is None

    def test_start_success(self):
        """Test starting pipeline successfully."""
        pipeline = DataPipeline(name="test")
        
        with patch('data_engineering.domain.entities.data_pipeline.datetime') as mock_datetime:
            mock_now = datetime(2024, 1, 15, 18, 0, 0)
            mock_datetime.utcnow.return_value = mock_now
            
            pipeline.start()
            
            assert pipeline.status == PipelineStatus.RUNNING
            assert pipeline.started_at == mock_now
            assert pipeline.updated_at == mock_now
            assert pipeline.last_run_at == mock_now

    def test_start_from_queued(self):
        """Test starting pipeline from queued status."""
        pipeline = DataPipeline(name="test")
        pipeline.status = PipelineStatus.QUEUED
        
        pipeline.start()
        
        assert pipeline.status == PipelineStatus.RUNNING

    def test_start_from_invalid_status(self):
        """Test starting pipeline from invalid status fails."""
        pipeline = DataPipeline(name="test")
        pipeline.status = PipelineStatus.RUNNING
        
        with pytest.raises(ValueError, match="Cannot start pipeline in running status"):
            pipeline.start()

    def test_complete_success(self):
        """Test completing pipeline successfully."""
        pipeline = DataPipeline(name="test")
        pipeline.status = PipelineStatus.RUNNING
        
        with patch('data_engineering.domain.entities.data_pipeline.datetime') as mock_datetime:
            mock_now = datetime(2024, 1, 15, 19, 0, 0)
            mock_datetime.utcnow.return_value = mock_now
            
            pipeline.complete()
            
            assert pipeline.status == PipelineStatus.COMPLETED
            assert pipeline.completed_at == mock_now
            assert pipeline.updated_at == mock_now

    def test_complete_from_invalid_status(self):
        """Test completing pipeline from invalid status fails."""
        pipeline = DataPipeline(name="test")
        
        with pytest.raises(ValueError, match="Cannot complete pipeline in created status"):
            pipeline.complete()

    def test_fail_pipeline(self):
        """Test failing pipeline."""
        pipeline = DataPipeline(name="test")
        pipeline.status = PipelineStatus.RUNNING
        
        with patch('data_engineering.domain.entities.data_pipeline.datetime') as mock_datetime:
            mock_now = datetime(2024, 1, 15, 20, 0, 0)
            mock_datetime.utcnow.return_value = mock_now
            
            error_message = "Database connection failed"
            pipeline.fail(error_message)
            
            assert pipeline.status == PipelineStatus.FAILED
            assert pipeline.completed_at == mock_now
            assert pipeline.updated_at == mock_now
            assert pipeline.config["last_error"] == error_message

    def test_fail_pipeline_without_message(self):
        """Test failing pipeline without error message."""
        pipeline = DataPipeline(name="test")
        pipeline.status = PipelineStatus.RUNNING
        
        pipeline.fail()
        
        assert pipeline.status == PipelineStatus.FAILED
        assert "last_error" not in pipeline.config

    def test_cancel_pipeline(self):
        """Test cancelling pipeline."""
        pipeline = DataPipeline(name="test")
        pipeline.status = PipelineStatus.RUNNING
        
        with patch('data_engineering.domain.entities.data_pipeline.datetime') as mock_datetime:
            mock_now = datetime(2024, 1, 15, 21, 0, 0)
            mock_datetime.utcnow.return_value = mock_now
            
            pipeline.cancel()
            
            assert pipeline.status == PipelineStatus.CANCELLED
            assert pipeline.completed_at == mock_now
            assert pipeline.updated_at == mock_now

    def test_cancel_from_valid_statuses(self):
        """Test cancelling from valid statuses."""
        pipeline = DataPipeline(name="test")
        
        # Test from queued
        pipeline.status = PipelineStatus.QUEUED
        pipeline.cancel()
        assert pipeline.status == PipelineStatus.CANCELLED
        
        # Test from paused
        pipeline.status = PipelineStatus.PAUSED
        pipeline.cancel()
        assert pipeline.status == PipelineStatus.CANCELLED

    def test_cancel_from_invalid_status(self):
        """Test cancelling from invalid status fails."""
        pipeline = DataPipeline(name="test")
        pipeline.status = PipelineStatus.COMPLETED
        
        with pytest.raises(ValueError, match="Cannot cancel pipeline in completed status"):
            pipeline.cancel()

    def test_pause_pipeline(self):
        """Test pausing pipeline."""
        pipeline = DataPipeline(name="test")
        pipeline.status = PipelineStatus.RUNNING
        
        with patch('data_engineering.domain.entities.data_pipeline.datetime') as mock_datetime:
            mock_now = datetime(2024, 1, 15, 22, 0, 0)
            mock_datetime.utcnow.return_value = mock_now
            
            pipeline.pause()
            
            assert pipeline.status == PipelineStatus.PAUSED
            assert pipeline.updated_at == mock_now

    def test_pause_from_invalid_status(self):
        """Test pausing from invalid status fails."""
        pipeline = DataPipeline(name="test")
        
        with pytest.raises(ValueError, match="Cannot pause pipeline in created status"):
            pipeline.pause()

    def test_resume_pipeline(self):
        """Test resuming pipeline."""
        pipeline = DataPipeline(name="test")
        pipeline.status = PipelineStatus.PAUSED
        
        with patch('data_engineering.domain.entities.data_pipeline.datetime') as mock_datetime:
            mock_now = datetime(2024, 1, 15, 23, 0, 0)
            mock_datetime.utcnow.return_value = mock_now
            
            pipeline.resume()
            
            assert pipeline.status == PipelineStatus.RUNNING
            assert pipeline.updated_at == mock_now

    def test_resume_from_invalid_status(self):
        """Test resuming from invalid status fails."""
        pipeline = DataPipeline(name="test")
        
        with pytest.raises(ValueError, match="Cannot resume pipeline in created status"):
            pipeline.resume()

    def test_can_retry_true(self):
        """Test can_retry returns True when conditions are met."""
        pipeline = DataPipeline(name="test")
        pipeline.status = PipelineStatus.FAILED
        pipeline.retry_count = 1
        pipeline.max_retry_attempts = 3
        
        assert pipeline.can_retry() is True

    def test_can_retry_false_max_attempts(self):
        """Test can_retry returns False when max attempts reached."""
        pipeline = DataPipeline(name="test")
        pipeline.status = PipelineStatus.FAILED
        pipeline.retry_count = 3
        pipeline.max_retry_attempts = 3
        
        assert pipeline.can_retry() is False

    def test_can_retry_false_not_failed(self):
        """Test can_retry returns False when not failed."""
        pipeline = DataPipeline(name="test")
        pipeline.status = PipelineStatus.COMPLETED
        pipeline.retry_count = 0
        pipeline.max_retry_attempts = 3
        
        assert pipeline.can_retry() is False

    def test_retry_success(self):
        """Test retrying pipeline successfully."""
        pipeline = DataPipeline(name="test")
        pipeline.status = PipelineStatus.FAILED
        pipeline.retry_count = 1
        pipeline.started_at = datetime(2024, 1, 15, 10, 0, 0)
        pipeline.completed_at = datetime(2024, 1, 15, 11, 0, 0)
        
        # Add failed step
        step = PipelineStep(name="failed_step", step_type="extract")
        step.status = StepStatus.FAILED
        step.started_at = datetime(2024, 1, 15, 10, 30, 0)
        step.completed_at = datetime(2024, 1, 15, 10, 45, 0)
        step.error_message = "Test error"
        pipeline.steps.append(step)
        
        with patch('data_engineering.domain.entities.data_pipeline.datetime') as mock_datetime:
            mock_now = datetime(2024, 1, 16, 10, 0, 0)
            mock_datetime.utcnow.return_value = mock_now
            
            pipeline.retry()
            
            assert pipeline.retry_count == 2
            assert pipeline.status == PipelineStatus.QUEUED
            assert pipeline.started_at is None
            assert pipeline.completed_at is None
            assert pipeline.updated_at == mock_now
            
            # Check step was reset
            assert step.status == StepStatus.PENDING
            assert step.started_at is None
            assert step.completed_at is None
            assert step.error_message is None

    def test_retry_cannot_retry(self):
        """Test retrying when cannot retry fails."""
        pipeline = DataPipeline(name="test")
        pipeline.status = PipelineStatus.COMPLETED
        
        with pytest.raises(ValueError, match="Pipeline cannot be retried"):
            pipeline.retry()

    def test_add_tag(self):
        """Test adding tags."""
        pipeline = DataPipeline(name="test")
        
        with patch('data_engineering.domain.entities.data_pipeline.datetime') as mock_datetime:
            mock_now = datetime(2024, 1, 16, 11, 0, 0)
            mock_datetime.utcnow.return_value = mock_now
            
            pipeline.add_tag("production")
            pipeline.add_tag("daily")
            
            assert "production" in pipeline.tags
            assert "daily" in pipeline.tags
            assert len(pipeline.tags) == 2
            assert pipeline.updated_at == mock_now

    def test_add_tag_duplicate(self):
        """Test adding duplicate tag."""
        pipeline = DataPipeline(name="test")
        pipeline.add_tag("production")
        pipeline.add_tag("production")  # Duplicate
        
        assert pipeline.tags.count("production") == 1

    def test_add_tag_empty(self):
        """Test adding empty tag does nothing."""
        pipeline = DataPipeline(name="test")
        original_updated_at = pipeline.updated_at
        
        pipeline.add_tag("")
        
        assert len(pipeline.tags) == 0
        assert pipeline.updated_at == original_updated_at

    def test_remove_tag(self):
        """Test removing tags."""
        pipeline = DataPipeline(name="test")
        pipeline.tags = ["production", "daily", "etl"]
        
        with patch('data_engineering.domain.entities.data_pipeline.datetime') as mock_datetime:
            mock_now = datetime(2024, 1, 16, 12, 0, 0)
            mock_datetime.utcnow.return_value = mock_now
            
            pipeline.remove_tag("daily")
            
            assert "daily" not in pipeline.tags
            assert len(pipeline.tags) == 2
            assert pipeline.updated_at == mock_now

    def test_remove_tag_nonexistent(self):
        """Test removing non-existent tag."""
        pipeline = DataPipeline(name="test")
        pipeline.tags = ["production"]
        original_updated_at = pipeline.updated_at
        
        pipeline.remove_tag("nonexistent")
        
        assert len(pipeline.tags) == 1
        assert pipeline.updated_at == original_updated_at

    def test_has_tag(self):
        """Test checking for tags."""
        pipeline = DataPipeline(name="test")
        pipeline.tags = ["production", "daily"]
        
        assert pipeline.has_tag("production") is True
        assert pipeline.has_tag("daily") is True
        assert pipeline.has_tag("nonexistent") is False

    def test_get_failed_steps(self):
        """Test getting failed steps."""
        pipeline = DataPipeline(name="test")
        
        step1 = PipelineStep(name="step1", step_type="extract")
        step1.status = StepStatus.COMPLETED
        
        step2 = PipelineStep(name="step2", step_type="transform")
        step2.status = StepStatus.FAILED
        
        step3 = PipelineStep(name="step3", step_type="load")
        step3.status = StepStatus.FAILED
        
        pipeline.steps = [step1, step2, step3]
        
        failed_steps = pipeline.get_failed_steps()
        
        assert len(failed_steps) == 2
        assert step2 in failed_steps
        assert step3 in failed_steps

    def test_get_completed_steps(self):
        """Test getting completed steps."""
        pipeline = DataPipeline(name="test")
        
        step1 = PipelineStep(name="step1", step_type="extract")
        step1.status = StepStatus.COMPLETED
        
        step2 = PipelineStep(name="step2", step_type="transform")
        step2.status = StepStatus.FAILED
        
        step3 = PipelineStep(name="step3", step_type="load")
        step3.status = StepStatus.COMPLETED
        
        pipeline.steps = [step1, step2, step3]
        
        completed_steps = pipeline.get_completed_steps()
        
        assert len(completed_steps) == 2
        assert step1 in completed_steps
        assert step3 in completed_steps

    def test_get_runnable_steps(self):
        """Test getting runnable steps."""
        pipeline = DataPipeline(name="test")
        
        step1 = PipelineStep(name="step1", step_type="extract")
        step1.status = StepStatus.PENDING
        
        step2 = PipelineStep(name="step2", step_type="transform", dependencies=[step1.id])
        step2.status = StepStatus.PENDING
        
        step3 = PipelineStep(name="step3", step_type="load")
        step3.status = StepStatus.PENDING
        
        pipeline.steps = [step1, step2, step3]
        
        # Initially, only step1 and step3 are runnable (no dependencies)
        runnable = pipeline.get_runnable_steps()
        assert len(runnable) == 2
        assert step1 in runnable
        assert step3 in runnable
        
        # After step1 completes, step2 becomes runnable
        step1.status = StepStatus.COMPLETED
        runnable = pipeline.get_runnable_steps()
        assert len(runnable) == 2
        assert step2 in runnable
        assert step3 in runnable

    def test_to_dict(self):
        """Test converting pipeline to dictionary."""
        pipeline = DataPipeline(
            name="test_pipeline",
            description="Test pipeline",
            version="2.0.0",
            created_by="engineer",
            config={"batch_size": 1000},
            environment_vars={"ENV": "test"},
            tags=["etl", "daily"]
        )
        
        step = PipelineStep(name="extract", step_type="extract")
        pipeline.add_step(step)
        
        result = pipeline.to_dict()
        
        assert result["id"] == str(pipeline.id)
        assert result["name"] == "test_pipeline"
        assert result["description"] == "Test pipeline"
        assert result["version"] == "2.0.0"
        assert result["status"] == "created"
        assert result["steps"] == [step.id]
        assert result["created_by"] == "engineer"
        assert result["config"] == {"batch_size": 1000}
        assert result["environment_vars"] == {"ENV": "test"}
        assert result["tags"] == ["etl", "daily"]
        assert result["success_rate"] == 0.0
        assert result["is_running"] is False
        assert result["is_completed"] is False

    def test_str_representation(self):
        """Test string representation."""
        pipeline = DataPipeline(name="test_pipeline")
        
        step = PipelineStep(name="extract", step_type="extract")
        pipeline.add_step(step)
        
        str_repr = str(pipeline)
        
        assert "DataPipeline('test_pipeline'" in str_repr
        assert "status=created" in str_repr
        assert "steps=1" in str_repr

    def test_repr_representation(self):
        """Test detailed representation."""
        pipeline = DataPipeline(name="test_pipeline")
        
        step = PipelineStep(name="extract", step_type="extract")
        pipeline.add_step(step)
        
        repr_str = repr(pipeline)
        
        assert f"id={pipeline.id}" in repr_str
        assert "name='test_pipeline'" in repr_str
        assert "status=created" in repr_str
        assert "steps=1" in repr_str