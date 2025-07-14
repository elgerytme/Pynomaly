import pytest
from uuid import UUID
from datetime import datetime

from data_transformation.domain.value_objects.pipeline_config import (
    PipelineConfig, SourceType, CleaningStrategy, ScalingMethod, 
    EncodingStrategy, OutputFormat
)
from data_transformation.domain.value_objects.transformation_step import (
    TransformationStep, StepType, StepStatus
)


class TestPipelineConfig:
    def test_create_basic_config(self):
        config = PipelineConfig(
            source_type=SourceType.CSV,
            cleaning_strategy=CleaningStrategy.AUTO,
            scaling_method=ScalingMethod.STANDARD,
            encoding_strategy=EncodingStrategy.ONE_HOT
        )
        
        assert config.source_type == SourceType.CSV
        assert config.cleaning_strategy == CleaningStrategy.AUTO
        assert config.scaling_method == ScalingMethod.STANDARD
        assert config.encoding_strategy == EncodingStrategy.ONE_HOT
        assert config.feature_engineering is True
        assert config.output_format == OutputFormat.PANDAS

    def test_create_advanced_config(self):
        config = PipelineConfig(
            source_type=SourceType.PARQUET,
            cleaning_strategy=CleaningStrategy.STATISTICAL,
            scaling_method=ScalingMethod.ROBUST,
            encoding_strategy=EncodingStrategy.TARGET,
            feature_engineering=False,
            output_format=OutputFormat.POLARS,
            parallel_processing=True,
            cache_enabled=True,
            validation_enabled=True
        )
        
        assert config.source_type == SourceType.PARQUET
        assert config.cleaning_strategy == CleaningStrategy.STATISTICAL
        assert config.scaling_method == ScalingMethod.ROBUST
        assert config.encoding_strategy == EncodingStrategy.TARGET
        assert config.feature_engineering is False
        assert config.output_format == OutputFormat.POLARS
        assert config.parallel_processing is True
        assert config.cache_enabled is True
        assert config.validation_enabled is True

    def test_config_validation(self):
        config = PipelineConfig(
            source_type=SourceType.CSV,
            cleaning_strategy=CleaningStrategy.AUTO,
            scaling_method=ScalingMethod.STANDARD,
            encoding_strategy=EncodingStrategy.ONE_HOT
        )
        
        validation_result = config.validate()
        assert validation_result["is_valid"] is True
        assert len(validation_result["errors"]) == 0

    def test_config_validation_with_issues(self):
        config = PipelineConfig(
            source_type=SourceType.CSV,
            cleaning_strategy=CleaningStrategy.AUTO,
            scaling_method=ScalingMethod.STANDARD,
            encoding_strategy=EncodingStrategy.ONE_HOT,
            parallel_processing=True,
            chunk_size=0  # Invalid chunk size
        )
        
        validation_result = config.validate()
        assert validation_result["is_valid"] is False
        assert len(validation_result["errors"]) > 0

    def test_config_to_dict(self):
        config = PipelineConfig(
            source_type=SourceType.CSV,
            cleaning_strategy=CleaningStrategy.AUTO,
            scaling_method=ScalingMethod.STANDARD,
            encoding_strategy=EncodingStrategy.ONE_HOT
        )
        
        config_dict = config.to_dict()
        assert config_dict["source_type"] == "csv"
        assert config_dict["cleaning_strategy"] == "auto"
        assert config_dict["scaling_method"] == "standard"
        assert config_dict["encoding_strategy"] == "one_hot"

    def test_config_from_dict(self):
        config_dict = {
            "source_type": "csv",
            "cleaning_strategy": "auto",
            "scaling_method": "standard",
            "encoding_strategy": "one_hot",
            "feature_engineering": True,
            "output_format": "pandas"
        }
        
        config = PipelineConfig.from_dict(config_dict)
        assert config.source_type == SourceType.CSV
        assert config.cleaning_strategy == CleaningStrategy.AUTO
        assert config.scaling_method == ScalingMethod.STANDARD
        assert config.encoding_strategy == EncodingStrategy.ONE_HOT

    def test_enum_values(self):
        assert SourceType.CSV.value == "csv"
        assert SourceType.JSON.value == "json"
        assert SourceType.PARQUET.value == "parquet"
        assert SourceType.EXCEL.value == "excel"
        
        assert CleaningStrategy.AUTO.value == "auto"
        assert CleaningStrategy.MANUAL.value == "manual"
        assert CleaningStrategy.STATISTICAL.value == "statistical"
        
        assert ScalingMethod.STANDARD.value == "standard"
        assert ScalingMethod.MINMAX.value == "minmax"
        assert ScalingMethod.ROBUST.value == "robust"
        
        assert EncodingStrategy.ONE_HOT.value == "one_hot"
        assert EncodingStrategy.LABEL.value == "label"
        assert EncodingStrategy.TARGET.value == "target"


class TestTransformationStep:
    def test_create_step_with_defaults(self):
        step = TransformationStep(
            type=StepType.CLEANING,
            name="data_cleaning",
            config={"strategy": "auto"}
        )
        
        assert step.type == StepType.CLEANING
        assert step.name == "data_cleaning"
        assert step.config == {"strategy": "auto"}
        assert step.status == StepStatus.PENDING
        assert isinstance(step.id, UUID)
        assert step.created_at is not None
        assert step.started_at is None
        assert step.completed_at is None
        assert step.error_message is None
        assert step.execution_time is None
        assert step.metadata == {}

    def test_start_step(self):
        step = TransformationStep(
            type=StepType.CLEANING,
            name="data_cleaning",
            config={"strategy": "auto"}
        )
        
        step.start()
        assert step.status == StepStatus.RUNNING
        assert step.started_at is not None

    def test_complete_step(self):
        step = TransformationStep(
            type=StepType.CLEANING,
            name="data_cleaning",
            config={"strategy": "auto"}
        )
        
        step.start()
        step.complete()
        
        assert step.status == StepStatus.COMPLETED
        assert step.completed_at is not None
        assert step.execution_time is not None

    def test_fail_step(self):
        step = TransformationStep(
            type=StepType.CLEANING,
            name="data_cleaning",
            config={"strategy": "auto"}
        )
        
        error_message = "Test error"
        step.fail(error_message)
        
        assert step.status == StepStatus.FAILED
        assert step.error_message == error_message

    def test_update_metadata(self):
        step = TransformationStep(
            type=StepType.CLEANING,
            name="data_cleaning",
            config={"strategy": "auto"}
        )
        
        step.update_metadata({"rows_processed": 1000})
        assert step.metadata["rows_processed"] == 1000

    def test_get_summary(self):
        step = TransformationStep(
            type=StepType.CLEANING,
            name="data_cleaning",
            config={"strategy": "auto"}
        )
        
        step.start()
        step.complete()
        step.update_metadata({"rows_processed": 1000})
        
        summary = step.get_summary()
        assert summary["id"] == str(step.id)
        assert summary["name"] == step.name
        assert summary["type"] == StepType.CLEANING.value
        assert summary["status"] == StepStatus.COMPLETED.value
        assert "execution_time" in summary
        assert summary["metadata"]["rows_processed"] == 1000

    def test_step_type_enum(self):
        assert StepType.LOADING.value == "loading"
        assert StepType.VALIDATION.value == "validation"
        assert StepType.CLEANING.value == "cleaning"
        assert StepType.TRANSFORMATION.value == "transformation"
        assert StepType.SCALING.value == "scaling"
        assert StepType.ENCODING.value == "encoding"
        assert StepType.FEATURE_ENGINEERING.value == "feature_engineering"
        assert StepType.EXPORT.value == "export"

    def test_step_status_enum(self):
        assert StepStatus.PENDING.value == "pending"
        assert StepStatus.RUNNING.value == "running"
        assert StepStatus.COMPLETED.value == "completed"
        assert StepStatus.FAILED.value == "failed"
        assert StepStatus.SKIPPED.value == "skipped"