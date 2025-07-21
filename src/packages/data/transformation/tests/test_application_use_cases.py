import pytest
import pandas as pd
import numpy as np
from pathlib import Path
import tempfile

from data_transformation.application.use_cases.data_pipeline import DataPipelineUseCase
from data_transformation.domain.value_objects.pipeline_config import (
    PipelineConfig, SourceType, CleaningStrategy, ScalingMethod, EncodingStrategy
)


class TestDataPipelineUseCase:
    def setup_method(self):
        self.config = PipelineConfig(
            source_type=SourceType.CSV,
            cleaning_strategy=CleaningStrategy.AUTO,
            scaling_method=ScalingMethod.STANDARD,
            encoding_strategy=EncodingStrategy.ONEHOT,
            feature_engineering=True
        )
        self.use_case = DataPipelineUseCase(self.config)

    def test_execute_with_dataframe(self, sample_dataframe):
        result = self.use_case.execute(sample_dataframe)
        
        assert result.success is True
        assert result.data is not None
        assert isinstance(result.data, pd.DataFrame)
        assert result.pipeline_id is not None
        assert result.execution_time is not None
        assert len(result.steps_executed) > 0

    def test_execute_with_csv_file(self, sample_csv_file):
        result = self.use_case.execute(sample_csv_file)
        
        assert result.success is True
        assert result.data is not None
        assert isinstance(result.data, pd.DataFrame)
        assert len(result.data) > 0

    def test_execute_with_custom_config(self, sample_dataframe):
        custom_config = PipelineConfig(
            source_type=SourceType.CSV,
            cleaning_strategy=CleaningStrategy.MANUAL,
            scaling_method=ScalingMethod.MINMAX,
            encoding_strategy=EncodingStrategy.LABEL,
            feature_engineering=False
        )
        
        result = self.use_case.execute(sample_dataframe, config=custom_config)
        
        assert result.success is True
        assert result.config == custom_config

    def test_execute_with_validation_enabled(self, sample_dataframe):
        config_with_validation = PipelineConfig(
            source_type=SourceType.CSV,
            cleaning_strategy=CleaningStrategy.AUTO,
            scaling_method=ScalingMethod.STANDARD,
            encoding_strategy=EncodingStrategy.ONEHOT,
            feature_engineering=True,
            validation_enabled=True
        )
        
        use_case = DataPipelineUseCase(config_with_validation)
        result = use_case.execute(sample_dataframe)
        
        assert result.success is True
        assert result.validation_results is not None

    def test_execute_with_caching_enabled(self, sample_dataframe):
        config_with_cache = PipelineConfig(
            source_type=SourceType.CSV,
            cleaning_strategy=CleaningStrategy.AUTO,
            scaling_method=ScalingMethod.STANDARD,
            encoding_strategy=EncodingStrategy.ONEHOT,
            feature_engineering=True,
            cache_enabled=True
        )
        
        use_case = DataPipelineUseCase(config_with_cache)
        result = use_case.execute(sample_dataframe)
        
        assert result.success is True

    def test_execute_with_parallel_processing(self, sample_dataframe):
        config_parallel = PipelineConfig(
            source_type=SourceType.CSV,
            cleaning_strategy=CleaningStrategy.AUTO,
            scaling_method=ScalingMethod.STANDARD,
            encoding_strategy=EncodingStrategy.ONEHOT,
            feature_engineering=True,
            parallel_processing=True
        )
        
        use_case = DataPipelineUseCase(config_parallel)
        result = use_case.execute(sample_dataframe)
        
        assert result.success is True

    def test_get_pipeline_status(self, sample_dataframe):
        result = self.use_case.execute(sample_dataframe)
        pipeline_id = result.pipeline_id
        
        status = self.use_case.get_pipeline_status(pipeline_id)
        assert status is not None
        assert "status" in status
        assert "steps" in status

    def test_get_execution_metrics(self, sample_dataframe):
        result = self.use_case.execute(sample_dataframe)
        pipeline_id = result.pipeline_id
        
        metrics = self.use_case.get_execution_metrics(pipeline_id)
        assert metrics is not None
        assert "execution_time" in metrics
        assert "memory_usage" in metrics
        assert "rows_processed" in metrics

    def test_list_available_steps(self):
        steps = self.use_case.list_available_steps()
        
        assert isinstance(steps, list)
        assert len(steps) > 0
        expected_steps = ["loading", "validation", "cleaning", "transformation", 
                         "scaling", "encoding", "feature_engineering", "export"]
        for step in expected_steps:
            assert step in steps

    def test_validate_pipeline_config(self):
        validation_result = self.use_case.validate_pipeline_config(self.config)
        
        assert validation_result["is_valid"] is True
        assert "errors" in validation_result

    def test_execute_error_handling_invalid_source(self):
        result = self.use_case.execute("/nonexistent/file.csv")
        
        assert result.success is False
        assert result.error_message is not None
        assert "Error loading data" in result.error_message

    def test_execute_with_statistical_cleaning(self, sample_dataframe):
        config = PipelineConfig(
            source_type=SourceType.CSV,
            cleaning_strategy=CleaningStrategy.STATISTICAL,
            scaling_method=ScalingMethod.ROBUST,
            encoding_strategy=EncodingStrategy.TARGET,
            feature_engineering=True
        )
        
        use_case = DataPipelineUseCase(config)
        result = use_case.execute(sample_dataframe)
        
        assert result.success is True

    def test_execute_steps_tracking(self, sample_dataframe):
        result = self.use_case.execute(sample_dataframe)
        
        assert len(result.steps_executed) > 0
        # Should have at least loading and export steps
        step_types = [step["type"] for step in result.steps_executed]
        assert "loading" in step_types
        assert "export" in step_types

    def test_execute_with_feature_engineering_disabled(self, sample_dataframe):
        config = PipelineConfig(
            source_type=SourceType.CSV,
            cleaning_strategy=CleaningStrategy.AUTO,
            scaling_method=ScalingMethod.STANDARD,
            encoding_strategy=EncodingStrategy.ONEHOT,
            feature_engineering=False
        )
        
        use_case = DataPipelineUseCase(config)
        result = use_case.execute(sample_dataframe)
        
        assert result.success is True
        # Should not have feature engineering step
        step_types = [step["type"] for step in result.steps_executed]
        assert "feature_engineering" not in step_types

    def test_pipeline_persistence(self, sample_dataframe):
        result = self.use_case.execute(sample_dataframe)
        pipeline_id = result.pipeline_id
        
        # Pipeline should be stored and retrievable
        status = self.use_case.get_pipeline_status(pipeline_id)
        assert status is not None

    def test_execute_with_json_source(self, sample_json_file):
        config = PipelineConfig(
            source_type=SourceType.JSON,
            cleaning_strategy=CleaningStrategy.AUTO,
            scaling_method=ScalingMethod.STANDARD,
            encoding_strategy=EncodingStrategy.ONEHOT
        )
        
        use_case = DataPipelineUseCase(config)
        result = use_case.execute(sample_json_file, config=config)
        
        assert result.success is True
        assert isinstance(result.data, pd.DataFrame)

    def test_execute_memory_efficient_mode(self, sample_dataframe):
        # Create larger sample for memory testing
        large_df = pd.concat([sample_dataframe] * 100, ignore_index=True)
        
        config = PipelineConfig(
            source_type=SourceType.CSV,
            cleaning_strategy=CleaningStrategy.AUTO,
            scaling_method=ScalingMethod.STANDARD,
            encoding_strategy=EncodingStrategy.ONEHOT,
            memory_efficient=True
        )
        
        use_case = DataPipelineUseCase(config)
        result = use_case.execute(large_df)
        
        assert result.success is True

    def test_execute_with_target_encoding(self, sample_dataframe):
        # Add target variable for target encoding
        target = np.random.random(len(sample_dataframe))
        
        config = PipelineConfig(
            source_type=SourceType.CSV,
            cleaning_strategy=CleaningStrategy.AUTO,
            scaling_method=ScalingMethod.STANDARD,
            encoding_strategy=EncodingStrategy.TARGET,
            feature_engineering=True
        )
        
        use_case = DataPipelineUseCase(config)
        result = use_case.execute(sample_dataframe, target=target)
        
        assert result.success is True