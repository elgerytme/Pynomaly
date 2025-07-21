import pytest
from uuid import UUID
from datetime import datetime

from data_transformation.domain.entities.transformation_pipeline import (
    TransformationPipeline, PipelineStatus
)
from data_transformation.domain.value_objects.pipeline_config import PipelineConfig, SourceType
from data_transformation.domain.value_objects.transformation_step import (
    TransformationStep, StepType, StepStatus
)


class TestTransformationPipeline:
    def test_create_pipeline_with_defaults(self, basic_pipeline_config):
        pipeline = TransformationPipeline(
            name="test_pipeline",
            config=basic_pipeline_config
        )
        
        assert pipeline.name == "test_pipeline"
        assert pipeline.config == basic_pipeline_config
        assert pipeline.status == PipelineStatus.CREATED
        assert isinstance(pipeline.id, UUID)
        assert len(pipeline.steps) == 0
        assert pipeline.created_at is not None
        assert pipeline.execution_time is None
        assert pipeline.metadata == {}

    def test_add_step(self, basic_pipeline_config):
        pipeline = TransformationPipeline(
            name="test_pipeline",
            config=basic_pipeline_config
        )
        
        step = TransformationStep(
            type=StepType.CLEANING,
            name="data_cleaning",
            config={"strategy": "auto"}
        )
        
        pipeline.add_step(step)
        assert len(pipeline.steps) == 1
        assert pipeline.steps[0] == step

    def test_start_execution(self, basic_pipeline_config):
        pipeline = TransformationPipeline(
            name="test_pipeline",
            config=basic_pipeline_config
        )
        
        pipeline.start_execution()
        assert pipeline.status == PipelineStatus.RUNNING
        assert pipeline.started_at is not None

    def test_complete_execution(self, basic_pipeline_config):
        pipeline = TransformationPipeline(
            name="test_pipeline",
            config=basic_pipeline_config
        )
        
        pipeline.start_execution()
        pipeline.complete_execution()
        
        assert pipeline.status == PipelineStatus.COMPLETED
        assert pipeline.completed_at is not None
        assert pipeline.execution_time is not None

    def test_fail_execution(self, basic_pipeline_config):
        pipeline = TransformationPipeline(
            name="test_pipeline",
            config=basic_pipeline_config
        )
        
        error_message = "Test error"
        pipeline.fail_execution(error_message)
        
        assert pipeline.status == PipelineStatus.FAILED
        assert pipeline.error_message == error_message

    def test_update_metadata(self, basic_pipeline_config):
        pipeline = TransformationPipeline(
            name="test_pipeline",
            config=basic_pipeline_config
        )
        
        pipeline.update_metadata({"key": "value"})
        assert pipeline.metadata["key"] == "value"

    def test_get_step_by_name(self, basic_pipeline_config):
        pipeline = TransformationPipeline(
            name="test_pipeline",
            config=basic_pipeline_config
        )
        
        step = TransformationStep(
            type=StepType.CLEANING,
            name="data_cleaning",
            config={"strategy": "auto"}
        )
        
        pipeline.add_step(step)
        found_step = pipeline.get_step_by_name("data_cleaning")
        assert found_step == step
        
        not_found = pipeline.get_step_by_name("nonexistent")
        assert not_found is None

    def test_get_steps_by_type(self, basic_pipeline_config):
        pipeline = TransformationPipeline(
            name="test_pipeline",
            config=basic_pipeline_config
        )
        
        cleaning_step = TransformationStep(
            type=StepType.CLEANING,
            name="data_cleaning",
            config={"strategy": "auto"}
        )
        
        scaling_step = TransformationStep(
            type=StepType.SCALING,
            name="data_scaling",
            config={"method": "standard"}
        )
        
        pipeline.add_step(cleaning_step)
        pipeline.add_step(scaling_step)
        
        cleaning_steps = pipeline.get_steps_by_type(StepType.CLEANING)
        assert len(cleaning_steps) == 1
        assert cleaning_steps[0] == cleaning_step

    def test_get_execution_summary(self, basic_pipeline_config):
        pipeline = TransformationPipeline(
            name="test_pipeline",
            config=basic_pipeline_config
        )
        
        step = TransformationStep(
            type=StepType.CLEANING,
            name="data_cleaning",
            config={"strategy": "auto"}
        )
        step.status = StepStatus.COMPLETED
        
        pipeline.add_step(step)
        pipeline.start_execution()
        pipeline.complete_execution()
        
        summary = pipeline.get_execution_summary()
        assert summary["pipeline_id"] == str(pipeline.id)
        assert summary["name"] == pipeline.name
        assert summary["status"] == PipelineStatus.COMPLETED.value
        assert summary["total_steps"] == 1
        assert summary["completed_steps"] == 1
        assert "execution_time" in summary

    def test_pipeline_status_enum(self):
        assert PipelineStatus.CREATED.value == "created"
        assert PipelineStatus.RUNNING.value == "running"
        assert PipelineStatus.COMPLETED.value == "completed"
        assert PipelineStatus.FAILED.value == "failed"