import pytest
import pandas as pd
import numpy as np
from pathlib import Path
import tempfile

from data_transformation import (
    DataPipelineUseCase, PipelineConfig, SourceType, CleaningStrategy, 
    ScalingMethod, EncodingStrategy
)


class TestIntegration:
    """Integration tests for the complete data transformation pipeline."""
    
    def test_end_to_end_csv_pipeline(self, sample_csv_file):
        """Test complete pipeline from CSV file to processed output."""
        config = PipelineConfig(
            source_type=SourceType.CSV,
            cleaning_strategy=CleaningStrategy.AUTO,
            scaling_method=ScalingMethod.STANDARD,
            encoding_strategy=EncodingStrategy.ONEHOT,
            feature_engineering=True,
            validation_enabled=True
        )
        
        use_case = DataPipelineUseCase(config)
        result = use_case.execute(sample_csv_file)
        
        # Verify successful execution
        assert result.success is True
        assert result.data is not None
        assert isinstance(result.data, pd.DataFrame)
        assert len(result.data) > 0
        
        # Verify pipeline tracking
        assert result.pipeline_id is not None
        assert result.execution_time > 0
        assert len(result.steps_executed) > 0
        
        # Verify validation was performed
        assert result.validation_results is not None
        
        # Verify specific transformations were applied
        step_types = [step.step_type for step in result.steps_executed]
        assert "loading" in step_types
        assert "cleaning" in step_types
        assert "scaling" in step_types
        assert "encoding" in step_types

    def test_end_to_end_dataframe_pipeline(self, sample_dataframe):
        """Test complete pipeline with DataFrame input."""
        config = PipelineConfig(
            source_type=SourceType.CSV,  # Will be overridden
            cleaning_strategy=CleaningStrategy.STATISTICAL,
            scaling_method=ScalingMethod.ROBUST,
            encoding_strategy=EncodingStrategy.LABEL,
            feature_engineering=True,
            parallel_processing=True
        )
        
        use_case = DataPipelineUseCase(config)
        result = use_case.execute(sample_dataframe)
        
        # Verify successful execution
        assert result.success is True
        assert result.data is not None
        
        # Verify data transformations
        original_shape = sample_dataframe.shape
        processed_shape = result.data.shape
        
        # Should have same or fewer rows (due to cleaning)
        assert processed_shape[0] <= original_shape[0]
        # May have different number of columns (due to encoding/feature engineering)
        # This is expected and okay

    def test_pipeline_with_custom_preprocessing(self, sample_dataframe):
        """Test pipeline with custom preprocessing options."""
        # Add some data quality issues
        dirty_df = sample_dataframe.copy()
        dirty_df.loc[len(dirty_df)] = dirty_df.iloc[0]  # Add duplicate
        dirty_df.loc[2, 'numeric_col'] = 1000  # Add outlier
        
        config = PipelineConfig(
            source_type=SourceType.CSV,
            cleaning_strategy=CleaningStrategy.MANUAL,
            scaling_method=ScalingMethod.MINMAX,
            encoding_strategy=EncodingStrategy.ONEHOT,
            feature_engineering=False,
            validation_enabled=True
        )
        
        use_case = DataPipelineUseCase(config)
        result = use_case.execute(dirty_df)
        
        assert result.success is True
        assert result.validation_results is not None

    def test_pipeline_error_handling(self):
        """Test pipeline error handling with invalid input."""
        config = PipelineConfig(
            source_type=SourceType.CSV,
            cleaning_strategy=CleaningStrategy.AUTO,
            scaling_method=ScalingMethod.STANDARD,
            encoding_strategy=EncodingStrategy.ONEHOT
        )
        
        use_case = DataPipelineUseCase(config)
        result = use_case.execute("/nonexistent/file.csv")
        
        # Should handle error gracefully
        assert result.success is False
        assert result.error_message is not None
        assert "Error loading data" in result.error_message

    def test_pipeline_with_target_encoding(self, sample_dataframe):
        """Test pipeline with target encoding strategy."""
        # Create target variable
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
        assert result.data is not None

    def test_memory_efficient_pipeline(self, sample_dataframe):
        """Test memory-efficient processing with larger dataset."""
        # Create larger dataset
        large_df = pd.concat([sample_dataframe] * 50, ignore_index=True)
        
        config = PipelineConfig(
            source_type=SourceType.CSV,
            cleaning_strategy=CleaningStrategy.AUTO,
            scaling_method=ScalingMethod.STANDARD,
            encoding_strategy=EncodingStrategy.ONEHOT,
            feature_engineering=True,
            memory_efficient=True,
            chunk_size=100
        )
        
        use_case = DataPipelineUseCase(config)
        result = use_case.execute(large_df)
        
        assert result.success is True
        assert result.metrics is not None

    def test_cached_pipeline_execution(self, sample_dataframe):
        """Test pipeline execution with caching enabled."""
        config = PipelineConfig(
            source_type=SourceType.CSV,
            cleaning_strategy=CleaningStrategy.AUTO,
            scaling_method=ScalingMethod.STANDARD,
            encoding_strategy=EncodingStrategy.ONEHOT,
            feature_engineering=True,
            cache_enabled=True
        )
        
        use_case = DataPipelineUseCase(config)
        
        # First execution
        result1 = use_case.execute(sample_dataframe)
        assert result1.success is True
        
        # Second execution (should use cache)
        result2 = use_case.execute(sample_dataframe)
        assert result2.success is True

    def test_json_to_processed_output(self, sample_json_file):
        """Test complete JSON processing pipeline."""
        config = PipelineConfig(
            source_type=SourceType.JSON,
            cleaning_strategy=CleaningStrategy.AUTO,
            scaling_method=ScalingMethod.ROBUST,
            encoding_strategy=EncodingStrategy.LABEL,
            feature_engineering=True
        )
        
        use_case = DataPipelineUseCase(config)
        result = use_case.execute(sample_json_file, config=config)
        
        assert result.success is True
        assert isinstance(result.data, pd.DataFrame)

    def test_excel_processing_pipeline(self, sample_dataframe):
        """Test Excel file processing pipeline."""
        # Create temporary Excel file
        with tempfile.NamedTemporaryFile(suffix='.xlsx', delete=False) as f:
            sample_dataframe.to_excel(f.name, index=False)
            
            config = PipelineConfig(
                source_type=SourceType.EXCEL,
                cleaning_strategy=CleaningStrategy.AUTO,
                scaling_method=ScalingMethod.STANDARD,
                encoding_strategy=EncodingStrategy.ONEHOT,
                feature_engineering=True
            )
            
            use_case = DataPipelineUseCase(config)
            result = use_case.execute(f.name, config=config)
            
            assert result.success is True
            assert isinstance(result.data, pd.DataFrame)
            
        Path(f.name).unlink(missing_ok=True)

    def test_parquet_processing_pipeline(self, sample_dataframe):
        """Test Parquet file processing pipeline."""
        # Create temporary Parquet file
        with tempfile.NamedTemporaryFile(suffix='.parquet', delete=False) as f:
            sample_dataframe.to_parquet(f.name, index=False)
            
            config = PipelineConfig(
                source_type=SourceType.PARQUET,
                cleaning_strategy=CleaningStrategy.STATISTICAL,
                scaling_method=ScalingMethod.MINMAX,
                encoding_strategy=EncodingStrategy.LABEL,
                feature_engineering=False
            )
            
            use_case = DataPipelineUseCase(config)
            result = use_case.execute(f.name, config=config)
            
            assert result.success is True
            assert isinstance(result.data, pd.DataFrame)
            
        Path(f.name).unlink(missing_ok=True)

    def test_pipeline_status_tracking(self, sample_dataframe):
        """Test pipeline status and metrics tracking."""
        config = PipelineConfig(
            source_type=SourceType.CSV,
            cleaning_strategy=CleaningStrategy.AUTO,
            scaling_method=ScalingMethod.STANDARD,
            encoding_strategy=EncodingStrategy.ONEHOT,
            feature_engineering=True
        )
        
        use_case = DataPipelineUseCase(config)
        result = use_case.execute(sample_dataframe)
        
        # Test status tracking
        pipeline_id = result.pipeline_id
        status = use_case.get_pipeline_status(pipeline_id)
        assert status is not None
        assert "status" in status
        
        # Test metrics tracking
        metrics = use_case.get_execution_metrics(pipeline_id)
        assert metrics is not None
        assert "execution_time" in metrics

    def test_configuration_validation_integration(self):
        """Test configuration validation in complete pipeline."""
        # Invalid configuration
        invalid_config = PipelineConfig(
            source_type=SourceType.CSV,
            cleaning_strategy=CleaningStrategy.AUTO,
            scaling_method=ScalingMethod.STANDARD,
            encoding_strategy=EncodingStrategy.ONEHOT,
            parallel_processing=True,
            chunk_size=0  # Invalid
        )
        
        use_case = DataPipelineUseCase(invalid_config)
        validation_result = use_case.validate_pipeline_config(invalid_config)
        
        assert validation_result["is_valid"] is False
        assert len(validation_result["errors"]) > 0