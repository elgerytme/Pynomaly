"""
Comprehensive Data Pipeline Testing Suite

Complete testing for data ingestion, transformation, validation, preprocessing,
feature engineering, and data flow through the entire pipeline.
"""

import pytest
import asyncio
import numpy as np
import pandas as pd
import tempfile
import os
import json
import csv
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock, AsyncMock
from typing import Dict, List, Any, Optional, Iterator
from datetime import datetime, timedelta
from io import StringIO, BytesIO
import gzip
import pickle

from pynomaly.infrastructure.data_pipeline import (
    DataPipeline, PipelineStage, DataIngestionStage,
    DataValidationStage, DataTransformationStage,
    FeatureEngineeringStage, DataQualityStage
)
from pynomaly.infrastructure.data_sources import (
    CSVDataSource, ParquetDataSource, DatabaseDataSource,
    StreamingDataSource, APIDataSource, FileSystemDataSource
)
from pynomaly.infrastructure.data_transformers import (
    StandardScaler, MinMaxScaler, RobustScaler,
    PCATransformer, ICATransformer, OutlierRemover,
    MissingValueImputer, CategoricalEncoder
)
from pynomaly.infrastructure.data_validators import (
    SchemaValidator, DataQualityValidator, ConsistencyValidator,
    CompletenessValidator, UniquenessValidator
)
from pynomaly.domain.entities import Dataset
from pynomaly.domain.exceptions import (
    DataPipelineError, DataValidationError, DataTransformationError,
    DataIngestionError, InsufficientDataError
)


class TestDataPipelineCore:
    """Test core data pipeline functionality."""
    
    @pytest.fixture
    def sample_pipeline_config(self):
        """Sample pipeline configuration."""
        return {
            "name": "test_pipeline",
            "stages": [
                {
                    "name": "ingestion",
                    "type": "DataIngestionStage",
                    "config": {"source_type": "csv"}
                },
                {
                    "name": "validation",
                    "type": "DataValidationStage",
                    "config": {"schema_file": "schema.json"}
                },
                {
                    "name": "transformation",
                    "type": "DataTransformationStage",
                    "config": {"scalers": ["standard", "minmax"]}
                }
            ],
            "error_handling": {
                "strategy": "continue_on_error",
                "max_errors": 10
            },
            "monitoring": {
                "enabled": True,
                "metrics": ["throughput", "error_rate", "latency"]
            }
        }
    
    def test_pipeline_creation(self, sample_pipeline_config):
        """Test data pipeline creation."""
        pipeline = DataPipeline(sample_pipeline_config)
        
        assert pipeline.name == "test_pipeline"
        assert len(pipeline.stages) == 3
        assert pipeline.error_handling["max_errors"] == 10
        
    def test_pipeline_stage_registration(self):
        """Test pipeline stage registration."""
        pipeline = DataPipeline({"name": "test", "stages": []})
        
        # Add stages
        ingestion_stage = DataIngestionStage("ingestion", {})
        validation_stage = DataValidationStage("validation", {})
        
        pipeline.add_stage(ingestion_stage)
        pipeline.add_stage(validation_stage)
        
        assert len(pipeline.stages) == 2
        assert pipeline.get_stage("ingestion") == ingestion_stage
        assert pipeline.get_stage("validation") == validation_stage
        
    def test_pipeline_execution_order(self, sample_pipeline_config):
        """Test pipeline stage execution order."""
        execution_order = []
        
        # Mock stages to track execution
        with patch('pynomaly.infrastructure.data_pipeline.DataIngestionStage') as mock_ingestion, \
             patch('pynomaly.infrastructure.data_pipeline.DataValidationStage') as mock_validation, \
             patch('pynomaly.infrastructure.data_pipeline.DataTransformationStage') as mock_transformation:
            
            def track_execution(stage_name):
                def execute(data):
                    execution_order.append(stage_name)
                    return data
                return execute
            
            mock_ingestion.return_value.execute = track_execution("ingestion")
            mock_validation.return_value.execute = track_execution("validation")
            mock_transformation.return_value.execute = track_execution("transformation")
            
            pipeline = DataPipeline(sample_pipeline_config)
            test_data = {"data": [[1, 2], [3, 4]]}
            
            pipeline.execute(test_data)
            
        assert execution_order == ["ingestion", "validation", "transformation"]
        
    def test_pipeline_error_handling_continue(self):
        """Test pipeline error handling with continue strategy."""
        config = {
            "name": "error_test",
            "stages": [],
            "error_handling": {
                "strategy": "continue_on_error",
                "max_errors": 2
            }
        }
        
        pipeline = DataPipeline(config)
        
        # Add failing stage
        failing_stage = Mock()
        failing_stage.execute.side_effect = DataValidationError("Test error")
        pipeline.add_stage(failing_stage)
        
        # Add successful stage
        success_stage = Mock()
        success_stage.execute.return_value = {"processed": True}
        pipeline.add_stage(success_stage)
        
        test_data = {"data": [[1, 2]]}
        result = pipeline.execute(test_data)
        
        # Should continue despite error
        assert success_stage.execute.called
        assert "errors" in result
        assert len(result["errors"]) == 1
        
    def test_pipeline_error_handling_fail_fast(self):
        """Test pipeline error handling with fail fast strategy."""
        config = {
            "name": "fail_fast_test",
            "stages": [],
            "error_handling": {
                "strategy": "fail_fast"
            }
        }
        
        pipeline = DataPipeline(config)
        
        # Add failing stage
        failing_stage = Mock()
        failing_stage.execute.side_effect = DataValidationError("Critical error")
        pipeline.add_stage(failing_stage)
        
        # Add stage that shouldn't be reached
        unreachable_stage = Mock()
        pipeline.add_stage(unreachable_stage)
        
        test_data = {"data": [[1, 2]]}
        
        with pytest.raises(DataPipelineError):
            pipeline.execute(test_data)
            
        # Second stage should not be called
        assert not unreachable_stage.execute.called
        
    def test_pipeline_monitoring_metrics(self, sample_pipeline_config):
        """Test pipeline monitoring and metrics collection."""
        pipeline = DataPipeline(sample_pipeline_config)
        
        # Mock stages for successful execution
        for stage in pipeline.stages:
            stage.execute = Mock(return_value={"processed": True})
            
        test_data = {"data": [[1, 2], [3, 4]]}
        result = pipeline.execute(test_data)
        
        assert "metrics" in result
        assert "execution_time" in result["metrics"]
        assert "throughput" in result["metrics"]
        assert "stages_executed" in result["metrics"]
        
    def test_pipeline_parallel_execution(self):
        """Test pipeline parallel stage execution."""
        config = {
            "name": "parallel_test",
            "stages": [],
            "execution": {
                "mode": "parallel",
                "max_workers": 2
            }
        }
        
        pipeline = DataPipeline(config)
        
        # Add independent stages that can run in parallel
        stage1 = Mock()
        stage1.execute.return_value = {"stage1": "done"}
        stage1.dependencies = []
        
        stage2 = Mock()
        stage2.execute.return_value = {"stage2": "done"}
        stage2.dependencies = []
        
        pipeline.add_stage(stage1)
        pipeline.add_stage(stage2)
        
        test_data = {"data": [[1, 2]]}
        result = pipeline.execute_parallel(test_data)
        
        assert stage1.execute.called
        assert stage2.execute.called
        assert "parallel_results" in result


class TestDataIngestionStage:
    """Test data ingestion stage functionality."""
    
    @pytest.fixture
    def sample_csv_data(self):
        """Sample CSV data for testing."""
        return """feature_1,feature_2,feature_3
1.0,2.0,3.0
4.0,5.0,6.0
7.0,8.0,9.0
10.0,11.0,12.0"""
    
    @pytest.fixture
    def temp_csv_file(self, sample_csv_data):
        """Create temporary CSV file."""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as f:
            f.write(sample_csv_data)
            temp_file = f.name
            
        yield temp_file
        os.unlink(temp_file)
        
    def test_csv_data_ingestion(self, temp_csv_file):
        """Test CSV data ingestion."""
        config = {
            "source_type": "csv",
            "file_path": temp_csv_file,
            "delimiter": ",",
            "header": True
        }
        
        stage = DataIngestionStage("csv_ingestion", config)
        result = stage.execute({})
        
        assert "data" in result
        assert len(result["data"]) == 4  # 4 rows
        assert len(result["data"][0]) == 3  # 3 columns
        assert "features" in result
        assert result["features"] == ["feature_1", "feature_2", "feature_3"]
        
    def test_csv_data_ingestion_no_header(self, temp_csv_file):
        """Test CSV data ingestion without header."""
        config = {
            "source_type": "csv",
            "file_path": temp_csv_file,
            "delimiter": ",",
            "header": False
        }
        
        stage = DataIngestionStage("csv_no_header", config)
        result = stage.execute({})
        
        assert "data" in result
        assert len(result["data"]) == 5  # 5 rows (including header row)
        assert "features" in result
        # Should generate default feature names
        assert all(f.startswith("feature_") for f in result["features"])
        
    def test_json_data_ingestion(self):
        """Test JSON data ingestion."""
        json_data = {
            "data": [[1, 2, 3], [4, 5, 6]],
            "features": ["x", "y", "z"]
        }
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            json.dump(json_data, f)
            temp_file = f.name
            
        try:
            config = {
                "source_type": "json",
                "file_path": temp_file
            }
            
            stage = DataIngestionStage("json_ingestion", config)
            result = stage.execute({})
            
            assert result["data"] == json_data["data"]
            assert result["features"] == json_data["features"]
            
        finally:
            os.unlink(temp_file)
            
    def test_database_data_ingestion(self):
        """Test database data ingestion with mocking."""
        config = {
            "source_type": "database",
            "connection_string": "sqlite:///:memory:",
            "query": "SELECT * FROM test_table"
        }
        
        # Mock database connection
        with patch('pandas.read_sql') as mock_read_sql:
            mock_df = pd.DataFrame({
                "col1": [1, 2, 3],
                "col2": [4, 5, 6]
            })
            mock_read_sql.return_value = mock_df
            
            stage = DataIngestionStage("db_ingestion", config)
            result = stage.execute({})
            
            assert "data" in result
            assert len(result["data"]) == 3
            assert "features" in result
            assert result["features"] == ["col1", "col2"]
            
    def test_streaming_data_ingestion(self):
        """Test streaming data ingestion."""
        config = {
            "source_type": "stream",
            "stream_url": "ws://localhost:8080/stream",
            "batch_size": 10,
            "timeout": 5
        }
        
        # Mock streaming data
        mock_data_batches = [
            [[1, 2], [3, 4]],
            [[5, 6], [7, 8]],
            [[9, 10], [11, 12]]
        ]
        
        with patch('pynomaly.infrastructure.data_sources.StreamingDataSource') as mock_source:
            mock_source_instance = mock_source.return_value
            mock_source_instance.read_batch.side_effect = mock_data_batches
            
            stage = DataIngestionStage("stream_ingestion", config)
            
            # Test batch reading
            for i, expected_batch in enumerate(mock_data_batches):
                result = stage.execute_batch({})
                assert result["data"] == expected_batch
                
    def test_api_data_ingestion(self):
        """Test API data ingestion."""
        config = {
            "source_type": "api",
            "endpoint": "https://api.example.com/data",
            "headers": {"Authorization": "Bearer token123"},
            "params": {"limit": 100}
        }
        
        # Mock API response
        mock_response_data = {
            "data": [[1, 2, 3], [4, 5, 6]],
            "features": ["a", "b", "c"],
            "metadata": {"total": 2}
        }
        
        with patch('requests.get') as mock_get:
            mock_response = Mock()
            mock_response.status_code = 200
            mock_response.json.return_value = mock_response_data
            mock_get.return_value = mock_response
            
            stage = DataIngestionStage("api_ingestion", config)
            result = stage.execute({})
            
            assert result["data"] == mock_response_data["data"]
            assert result["features"] == mock_response_data["features"]
            assert "metadata" in result
            
    def test_ingestion_error_handling(self):
        """Test ingestion error handling."""
        config = {
            "source_type": "csv",
            "file_path": "/nonexistent/file.csv"
        }
        
        stage = DataIngestionStage("error_ingestion", config)
        
        with pytest.raises(DataIngestionError):
            stage.execute({})
            
    def test_large_file_ingestion(self):
        """Test large file ingestion with chunking."""
        # Create large CSV data
        large_data = []
        for i in range(1000):
            large_data.append(f"{i},{i+1},{i+2}")
        
        large_csv = "col1,col2,col3\n" + "\n".join(large_data)
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as f:
            f.write(large_csv)
            temp_file = f.name
            
        try:
            config = {
                "source_type": "csv",
                "file_path": temp_file,
                "chunk_size": 100,
                "header": True
            }
            
            stage = DataIngestionStage("large_ingestion", config)
            
            # Test chunked reading
            total_rows = 0
            chunk_count = 0
            
            for chunk in stage.execute_chunked({}):
                chunk_count += 1
                total_rows += len(chunk["data"])
                assert len(chunk["data"]) <= 100  # Chunk size limit
                
            assert total_rows == 1000
            assert chunk_count == 10  # 1000 rows / 100 chunk_size
            
        finally:
            os.unlink(temp_file)


class TestDataValidationStage:
    """Test data validation stage functionality."""
    
    @pytest.fixture
    def sample_schema(self):
        """Sample data schema for validation."""
        return {
            "type": "object",
            "properties": {
                "data": {
                    "type": "array",
                    "items": {
                        "type": "array",
                        "minItems": 3,
                        "maxItems": 3,
                        "items": {"type": "number"}
                    }
                },
                "features": {
                    "type": "array",
                    "items": {"type": "string"},
                    "minItems": 3,
                    "maxItems": 3
                }
            },
            "required": ["data", "features"]
        }
    
    def test_schema_validation_valid(self, sample_schema):
        """Test schema validation with valid data."""
        config = {
            "schema": sample_schema,
            "strict_mode": True
        }
        
        stage = DataValidationStage("schema_validation", config)
        
        valid_data = {
            "data": [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]],
            "features": ["x", "y", "z"]
        }
        
        result = stage.execute(valid_data)
        
        assert result == valid_data  # Should pass through unchanged
        assert "validation_passed" in result
        assert result["validation_passed"] is True
        
    def test_schema_validation_invalid(self, sample_schema):
        """Test schema validation with invalid data."""
        config = {
            "schema": sample_schema,
            "strict_mode": True
        }
        
        stage = DataValidationStage("schema_validation", config)
        
        invalid_data = {
            "data": [[1.0, 2.0], [4.0, 5.0]],  # Missing third column
            "features": ["x", "y"]
        }
        
        with pytest.raises(DataValidationError):
            stage.execute(invalid_data)
            
    def test_data_quality_validation(self):
        """Test data quality validation."""
        config = {
            "quality_checks": {
                "min_completeness": 0.9,
                "max_duplicates_rate": 0.1,
                "detect_outliers": True
            }
        }
        
        stage = DataValidationStage("quality_validation", config)
        
        # Data with quality issues
        data_with_issues = {
            "data": [
                [1.0, 2.0, 3.0],
                [4.0, None, 6.0],  # Missing value
                [7.0, 8.0, 9.0],
                [1.0, 2.0, 3.0],  # Duplicate
                [100.0, 200.0, 300.0]  # Potential outlier
            ],
            "features": ["x", "y", "z"]
        }
        
        result = stage.execute(data_with_issues)
        
        assert "quality_report" in result
        assert "completeness" in result["quality_report"]
        assert "duplicates_rate" in result["quality_report"]
        assert "outliers_detected" in result["quality_report"]
        
    def test_consistency_validation(self):
        """Test data consistency validation."""
        config = {
            "consistency_checks": {
                "check_data_types": True,
                "check_value_ranges": True,
                "expected_ranges": {
                    "x": {"min": 0, "max": 100},
                    "y": {"min": 0, "max": 100}
                }
            }
        }
        
        stage = DataValidationStage("consistency_validation", config)
        
        # Data with consistency issues
        inconsistent_data = {
            "data": [
                [1.0, 2.0, 3.0],
                ["invalid", 5.0, 6.0],  # Wrong data type
                [7.0, 150.0, 9.0]  # Out of expected range
            ],
            "features": ["x", "y", "z"]
        }
        
        result = stage.execute(inconsistent_data)
        
        assert "consistency_report" in result
        assert "type_errors" in result["consistency_report"]
        assert "range_violations" in result["consistency_report"]
        
    def test_completeness_validation(self):
        """Test data completeness validation."""
        config = {
            "completeness_threshold": 0.8,
            "required_features": ["x", "y", "z"]
        }
        
        stage = DataValidationStage("completeness_validation", config)
        
        # Data with missing values
        incomplete_data = {
            "data": [
                [1.0, 2.0, 3.0],
                [4.0, None, 6.0],
                [7.0, 8.0, None],
                [10.0, 11.0, 12.0]
            ],
            "features": ["x", "y", "z"]
        }
        
        result = stage.execute(incomplete_data)
        
        assert "completeness_report" in result
        assert "overall_completeness" in result["completeness_report"]
        assert "feature_completeness" in result["completeness_report"]
        
    def test_uniqueness_validation(self):
        """Test data uniqueness validation."""
        config = {
            "uniqueness_checks": {
                "check_duplicates": True,
                "unique_columns": ["id"],
                "duplicate_threshold": 0.05
            }
        }
        
        stage = DataValidationStage("uniqueness_validation", config)
        
        # Data with duplicates
        data_with_duplicates = {
            "data": [
                [1, 10.0, 20.0],
                [2, 11.0, 21.0],
                [1, 10.0, 20.0],  # Duplicate
                [3, 12.0, 22.0]
            ],
            "features": ["id", "x", "y"]
        }
        
        result = stage.execute(data_with_duplicates)
        
        assert "uniqueness_report" in result
        assert "duplicate_count" in result["uniqueness_report"]
        assert "duplicate_rate" in result["uniqueness_report"]
        
    def test_validation_with_repair(self):
        """Test validation with automatic data repair."""
        config = {
            "auto_repair": True,
            "repair_strategies": {
                "missing_values": "impute_mean",
                "duplicates": "remove",
                "outliers": "clip"
            }
        }
        
        stage = DataValidationStage("validation_repair", config)
        
        # Data with issues
        problematic_data = {
            "data": [
                [1.0, 2.0, 3.0],
                [4.0, None, 6.0],  # Missing value
                [7.0, 8.0, 9.0],
                [1.0, 2.0, 3.0],  # Duplicate
                [100.0, 200.0, 300.0]  # Outlier
            ],
            "features": ["x", "y", "z"]
        }
        
        result = stage.execute(problematic_data)
        
        assert "repaired_data" in result
        assert "repair_report" in result
        assert len(result["repaired_data"]) < len(problematic_data["data"])  # Duplicates removed


class TestDataTransformationStage:
    """Test data transformation stage functionality."""
    
    @pytest.fixture
    def sample_numeric_data(self):
        """Sample numeric data for transformation."""
        return {
            "data": [
                [1.0, 10.0, 100.0],
                [2.0, 20.0, 200.0],
                [3.0, 30.0, 300.0],
                [4.0, 40.0, 400.0],
                [5.0, 50.0, 500.0]
            ],
            "features": ["small", "medium", "large"]
        }
    
    def test_standard_scaling_transformation(self, sample_numeric_data):
        """Test standard scaling transformation."""
        config = {
            "transformers": [
                {
                    "type": "StandardScaler",
                    "features": ["small", "medium", "large"]
                }
            ]
        }
        
        stage = DataTransformationStage("standard_scaling", config)
        result = stage.execute(sample_numeric_data)
        
        assert "transformed_data" in result
        assert "transformation_metadata" in result
        
        # Check that data is scaled (mean should be close to 0)
        transformed_data = np.array(result["transformed_data"])
        means = np.mean(transformed_data, axis=0)
        assert all(abs(mean) < 0.1 for mean in means)  # Close to zero
        
    def test_minmax_scaling_transformation(self, sample_numeric_data):
        """Test min-max scaling transformation."""
        config = {
            "transformers": [
                {
                    "type": "MinMaxScaler",
                    "features": ["small", "medium", "large"],
                    "feature_range": [0, 1]
                }
            ]
        }
        
        stage = DataTransformationStage("minmax_scaling", config)
        result = stage.execute(sample_numeric_data)
        
        # Check that data is scaled to [0, 1] range
        transformed_data = np.array(result["transformed_data"])
        assert np.all(transformed_data >= 0)
        assert np.all(transformed_data <= 1)
        
    def test_robust_scaling_transformation(self, sample_numeric_data):
        """Test robust scaling transformation."""
        # Add outliers to test robustness
        data_with_outliers = sample_numeric_data.copy()
        data_with_outliers["data"].append([1000.0, 2000.0, 3000.0])  # Outlier
        
        config = {
            "transformers": [
                {
                    "type": "RobustScaler",
                    "features": ["small", "medium", "large"]
                }
            ]
        }
        
        stage = DataTransformationStage("robust_scaling", config)
        result = stage.execute(data_with_outliers)
        
        assert "transformed_data" in result
        # Robust scaler should be less affected by outliers
        transformed_data = np.array(result["transformed_data"])
        assert transformed_data.shape[0] == 6  # All rows preserved
        
    def test_pca_transformation(self, sample_numeric_data):
        """Test PCA dimensionality reduction."""
        config = {
            "transformers": [
                {
                    "type": "PCATransformer",
                    "n_components": 2,
                    "features": ["small", "medium", "large"]
                }
            ]
        }
        
        stage = DataTransformationStage("pca_transform", config)
        result = stage.execute(sample_numeric_data)
        
        # Check dimensionality reduction
        transformed_data = result["transformed_data"]
        assert len(transformed_data[0]) == 2  # Reduced to 2 components
        assert "pca_explained_variance" in result["transformation_metadata"]
        
    def test_missing_value_imputation(self):
        """Test missing value imputation."""
        data_with_missing = {
            "data": [
                [1.0, 2.0, 3.0],
                [4.0, None, 6.0],
                [7.0, 8.0, None],
                [10.0, 11.0, 12.0]
            ],
            "features": ["x", "y", "z"]
        }
        
        config = {
            "transformers": [
                {
                    "type": "MissingValueImputer",
                    "strategy": "mean",
                    "features": ["x", "y", "z"]
                }
            ]
        }
        
        stage = DataTransformationStage("imputation", config)
        result = stage.execute(data_with_missing)
        
        # Check that missing values are imputed
        transformed_data = result["transformed_data"]
        for row in transformed_data:
            assert all(val is not None for val in row)
            
    def test_categorical_encoding(self):
        """Test categorical feature encoding."""
        categorical_data = {
            "data": [
                ["red", "small", 1.0],
                ["blue", "medium", 2.0],
                ["green", "large", 3.0],
                ["red", "small", 4.0]
            ],
            "features": ["color", "size", "value"]
        }
        
        config = {
            "transformers": [
                {
                    "type": "CategoricalEncoder",
                    "encoding_type": "one_hot",
                    "categorical_features": ["color", "size"]
                }
            ]
        }
        
        stage = DataTransformationStage("categorical_encoding", config)
        result = stage.execute(categorical_data)
        
        # Check that categorical features are encoded
        transformed_data = result["transformed_data"]
        # One-hot encoding should increase feature count
        assert len(result["features"]) > len(categorical_data["features"])
        
    def test_outlier_removal_transformation(self):
        """Test outlier removal transformation."""
        data_with_outliers = {
            "data": [
                [1.0, 2.0, 3.0],
                [2.0, 3.0, 4.0],
                [3.0, 4.0, 5.0],
                [100.0, 200.0, 300.0],  # Clear outlier
                [4.0, 5.0, 6.0]
            ],
            "features": ["x", "y", "z"]
        }
        
        config = {
            "transformers": [
                {
                    "type": "OutlierRemover",
                    "method": "iqr",
                    "threshold": 1.5
                }
            ]
        }
        
        stage = DataTransformationStage("outlier_removal", config)
        result = stage.execute(data_with_outliers)
        
        # Check that outliers are removed
        transformed_data = result["transformed_data"]
        assert len(transformed_data) < len(data_with_outliers["data"])
        
    def test_chained_transformations(self, sample_numeric_data):
        """Test chained transformations."""
        config = {
            "transformers": [
                {
                    "type": "StandardScaler",
                    "features": ["small", "medium", "large"]
                },
                {
                    "type": "PCATransformer",
                    "n_components": 2
                }
            ]
        }
        
        stage = DataTransformationStage("chained_transforms", config)
        result = stage.execute(sample_numeric_data)
        
        # Both transformations should be applied
        assert "transformed_data" in result
        assert len(result["transformed_data"][0]) == 2  # PCA reduced to 2 components
        assert len(result["transformation_metadata"]) == 2  # Two transformations
        
    def test_transformation_inverse(self, sample_numeric_data):
        """Test transformation inverse (if supported)."""
        config = {
            "transformers": [
                {
                    "type": "StandardScaler",
                    "features": ["small", "medium", "large"],
                    "save_for_inverse": True
                }
            ]
        }
        
        stage = DataTransformationStage("invertible_transform", config)
        result = stage.execute(sample_numeric_data)
        
        # Test inverse transformation
        if "transformer_objects" in result["transformation_metadata"]:
            inverse_data = stage.inverse_transform(
                result["transformed_data"],
                result["transformation_metadata"]
            )
            
            # Should be close to original data
            original_data = np.array(sample_numeric_data["data"])
            reconstructed_data = np.array(inverse_data)
            
            assert np.allclose(original_data, reconstructed_data, atol=1e-10)


class TestFeatureEngineeringStage:
    """Test feature engineering stage functionality."""
    
    @pytest.fixture
    def sample_time_series_data(self):
        """Sample time series data for feature engineering."""
        dates = pd.date_range('2023-01-01', periods=100, freq='D')
        return {
            "data": [
                [date.timestamp(), np.sin(i * 0.1), np.cos(i * 0.1)]
                for i, date in enumerate(dates)
            ],
            "features": ["timestamp", "sin_value", "cos_value"]
        }
    
    def test_polynomial_feature_generation(self):
        """Test polynomial feature generation."""
        simple_data = {
            "data": [[1, 2], [3, 4], [5, 6]],
            "features": ["x1", "x2"]
        }
        
        config = {
            "feature_generators": [
                {
                    "type": "PolynomialFeatures",
                    "degree": 2,
                    "include_bias": False,
                    "interaction_only": False
                }
            ]
        }
        
        stage = FeatureEngineeringStage("polynomial_features", config)
        result = stage.execute(simple_data)
        
        # Polynomial features should increase feature count
        # x1, x2, x1^2, x1*x2, x2^2 = 5 features
        assert len(result["features"]) > len(simple_data["features"])
        assert "feature_engineering_metadata" in result
        
    def test_time_series_feature_extraction(self, sample_time_series_data):
        """Test time series feature extraction."""
        config = {
            "feature_generators": [
                {
                    "type": "TimeSeriesFeatures",
                    "timestamp_column": "timestamp",
                    "extract_features": [
                        "hour", "day_of_week", "month", "quarter"
                    ]
                }
            ]
        }
        
        stage = FeatureEngineeringStage("time_features", config)
        result = stage.execute(sample_time_series_data)
        
        # Should have additional time-based features
        new_features = result["features"]
        assert "hour" in new_features
        assert "day_of_week" in new_features
        assert "month" in new_features
        assert "quarter" in new_features
        
    def test_statistical_feature_generation(self):
        """Test statistical feature generation."""
        windowed_data = {
            "data": [
                [1, 2, 3, 4, 5],
                [2, 3, 4, 5, 6],
                [3, 4, 5, 6, 7],
                [4, 5, 6, 7, 8]
            ],
            "features": ["t1", "t2", "t3", "t4", "t5"]
        }
        
        config = {
            "feature_generators": [
                {
                    "type": "StatisticalFeatures",
                    "statistics": ["mean", "std", "min", "max", "skew"],
                    "window_size": 5
                }
            ]
        }
        
        stage = FeatureEngineeringStage("statistical_features", config)
        result = stage.execute(windowed_data)
        
        # Should have statistical features added
        new_features = result["features"]
        stat_features = [f for f in new_features if any(stat in f for stat in ["mean", "std", "min", "max", "skew"])]
        assert len(stat_features) > 0
        
    def test_frequency_domain_features(self):
        """Test frequency domain feature extraction."""
        signal_data = {
            "data": [
                [np.sin(2 * np.pi * 0.1 * i) + 0.5 * np.sin(2 * np.pi * 0.3 * i)]
                for i in range(100)
            ],
            "features": ["signal"]
        }
        
        config = {
            "feature_generators": [
                {
                    "type": "FrequencyDomainFeatures",
                    "signal_columns": ["signal"],
                    "fft_features": ["dominant_frequency", "spectral_centroid", "spectral_bandwidth"]
                }
            ]
        }
        
        stage = FeatureEngineeringStage("frequency_features", config)
        result = stage.execute(signal_data)
        
        # Should have frequency domain features
        new_features = result["features"]
        freq_features = [f for f in new_features if any(feat in f for feat in ["frequency", "spectral"])]
        assert len(freq_features) > 0
        
    def test_interaction_feature_generation(self):
        """Test interaction feature generation."""
        multi_feature_data = {
            "data": [
                [1, 2, 3],
                [4, 5, 6],
                [7, 8, 9]
            ],
            "features": ["feature_a", "feature_b", "feature_c"]
        }
        
        config = {
            "feature_generators": [
                {
                    "type": "InteractionFeatures",
                    "feature_pairs": [
                        ["feature_a", "feature_b"],
                        ["feature_b", "feature_c"]
                    ],
                    "operations": ["multiply", "add", "subtract"]
                }
            ]
        }
        
        stage = FeatureEngineeringStage("interaction_features", config)
        result = stage.execute(multi_feature_data)
        
        # Should have interaction features
        new_features = result["features"]
        interaction_features = [f for f in new_features if any(op in f for op in ["multiply", "add", "subtract"])]
        assert len(interaction_features) > 0
        
    def test_domain_specific_features(self):
        """Test domain-specific feature generation."""
        anomaly_detection_data = {
            "data": [
                [1.0, 2.0, 3.0],
                [1.1, 2.1, 3.1],
                [0.9, 1.9, 2.9],
                [10.0, 20.0, 30.0]  # Potential anomaly
            ],
            "features": ["x", "y", "z"]
        }
        
        config = {
            "feature_generators": [
                {
                    "type": "AnomalyDetectionFeatures",
                    "features": [
                        "isolation_score",
                        "local_outlier_factor",
                        "distance_to_centroid",
                        "mahalanobis_distance"
                    ]
                }
            ]
        }
        
        stage = FeatureEngineeringStage("anomaly_features", config)
        result = stage.execute(anomaly_detection_data)
        
        # Should have anomaly detection features
        new_features = result["features"]
        anomaly_features = [f for f in new_features if any(feat in f for feat in ["isolation", "outlier", "distance"])]
        assert len(anomaly_features) > 0
        
    def test_feature_selection(self):
        """Test feature selection during engineering."""
        high_dim_data = {
            "data": [
                [i + j for j in range(20)]  # 20 features
                for i in range(100)
            ],
            "features": [f"feature_{i}" for i in range(20)]
        }
        
        config = {
            "feature_generators": [
                {
                    "type": "FeatureSelector",
                    "method": "mutual_info",
                    "k_best": 10,
                    "target_estimation": "auto"
                }
            ]
        }
        
        stage = FeatureEngineeringStage("feature_selection", config)
        result = stage.execute(high_dim_data)
        
        # Should reduce feature count
        assert len(result["features"]) <= 10
        assert "feature_selection_scores" in result["feature_engineering_metadata"]


class TestDataQualityStage:
    """Test data quality assessment and improvement stage."""
    
    def test_comprehensive_quality_assessment(self):
        """Test comprehensive data quality assessment."""
        problematic_data = {
            "data": [
                [1.0, 2.0, 3.0],
                [4.0, None, 6.0],  # Missing value
                [7.0, 8.0, 9.0],
                [1.0, 2.0, 3.0],  # Duplicate
                [100.0, 200.0, 300.0],  # Outlier
                ["invalid", 11.0, 12.0],  # Type inconsistency
                [13.0, 14.0, 15.0]
            ],
            "features": ["x", "y", "z"]
        }
        
        config = {
            "quality_assessments": [
                "completeness",
                "consistency",
                "uniqueness",
                "validity",
                "accuracy"
            ]
        }
        
        stage = DataQualityStage("quality_assessment", config)
        result = stage.execute(problematic_data)
        
        assert "quality_report" in result
        quality_report = result["quality_report"]
        
        # Check all quality dimensions
        assert "completeness_score" in quality_report
        assert "consistency_score" in quality_report
        assert "uniqueness_score" in quality_report
        assert "validity_score" in quality_report
        assert "overall_quality_score" in quality_report
        
        # Scores should be between 0 and 1
        for score_key in ["completeness_score", "consistency_score", "uniqueness_score", "validity_score"]:
            assert 0 <= quality_report[score_key] <= 1
            
    def test_data_profiling(self):
        """Test data profiling functionality."""
        diverse_data = {
            "data": [
                [1, 10.5, "category_a", True],
                [2, 20.3, "category_b", False],
                [3, 30.7, "category_a", True],
                [4, 40.1, "category_c", None],
                [5, 50.9, "category_b", False]
            ],
            "features": ["int_col", "float_col", "cat_col", "bool_col"]
        }
        
        config = {
            "profiling_options": {
                "include_correlations": True,
                "include_distributions": True,
                "include_patterns": True
            }
        }
        
        stage = DataQualityStage("data_profiling", config)
        result = stage.execute(diverse_data)
        
        assert "data_profile" in result
        profile = result["data_profile"]
        
        # Check profile components
        assert "feature_types" in profile
        assert "feature_statistics" in profile
        assert "correlations" in profile
        assert "distributions" in profile
        
    def test_anomaly_detection_in_quality_stage(self):
        """Test anomaly detection within quality stage."""
        data_with_anomalies = {
            "data": [
                [1.0, 2.0, 3.0],
                [1.1, 2.1, 3.1],
                [0.9, 1.9, 2.9],
                [1.2, 2.2, 3.2],
                [100.0, 200.0, 300.0],  # Clear anomaly
                [0.8, 1.8, 2.8]
            ],
            "features": ["x", "y", "z"]
        }
        
        config = {
            "anomaly_detection": {
                "enabled": True,
                "methods": ["isolation_forest", "local_outlier_factor"],
                "contamination": 0.1
            }
        }
        
        stage = DataQualityStage("anomaly_detection", config)
        result = stage.execute(data_with_anomalies)
        
        assert "anomaly_report" in result
        anomaly_report = result["anomaly_report"]
        
        assert "anomalies_detected" in anomaly_report
        assert "anomaly_scores" in anomaly_report
        assert "detection_methods" in anomaly_report
        
    def test_quality_improvement_recommendations(self):
        """Test quality improvement recommendations."""
        low_quality_data = {
            "data": [
                [1.0, None, 3.0],
                [None, 2.0, None],
                [1.0, None, 3.0],  # Duplicate with missing values
                ["invalid", 2.0, 3.0]  # Type error
            ],
            "features": ["x", "y", "z"]
        }
        
        config = {
            "generate_recommendations": True,
            "recommendation_types": [
                "missing_value_handling",
                "duplicate_removal",
                "data_type_correction",
                "outlier_treatment"
            ]
        }
        
        stage = DataQualityStage("quality_recommendations", config)
        result = stage.execute(low_quality_data)
        
        assert "quality_recommendations" in result
        recommendations = result["quality_recommendations"]
        
        assert isinstance(recommendations, list)
        assert len(recommendations) > 0
        
        # Check recommendation structure
        for rec in recommendations:
            assert "issue_type" in rec
            assert "severity" in rec
            assert "recommendation" in rec
            assert "confidence" in rec
            
    def test_quality_monitoring_alerts(self):
        """Test quality monitoring and alerting."""
        config = {
            "monitoring": {
                "enabled": True,
                "thresholds": {
                    "completeness": 0.9,
                    "consistency": 0.8,
                    "uniqueness": 0.95
                },
                "alert_on_threshold_breach": True
            }
        }
        
        # Data that breaches thresholds
        poor_quality_data = {
            "data": [
                [1.0, 2.0],
                [None, None],  # Poor completeness
                [1.0, 2.0],    # Poor uniqueness
                ["string", 4.0]  # Poor consistency
            ],
            "features": ["x", "y"]
        }
        
        stage = DataQualityStage("quality_monitoring", config)
        result = stage.execute(poor_quality_data)
        
        assert "quality_alerts" in result
        alerts = result["quality_alerts"]
        
        # Should have alerts for threshold breaches
        assert len(alerts) > 0
        
        for alert in alerts:
            assert "metric" in alert
            assert "threshold" in alert
            assert "actual_value" in alert
            assert "severity" in alert


class TestPipelineIntegration:
    """Test complete pipeline integration scenarios."""
    
    async def test_end_to_end_pipeline_execution(self):
        """Test complete end-to-end pipeline execution."""
        # Create sample data file
        sample_data = """x,y,z
1.0,2.0,3.0
4.0,5.0,6.0
7.0,8.0,9.0
10.0,11.0,12.0
"""
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as f:
            f.write(sample_data)
            temp_file = f.name
            
        try:
            # Define complete pipeline
            pipeline_config = {
                "name": "end_to_end_test",
                "stages": [
                    {
                        "name": "ingestion",
                        "type": "DataIngestionStage",
                        "config": {
                            "source_type": "csv",
                            "file_path": temp_file,
                            "header": True
                        }
                    },
                    {
                        "name": "validation",
                        "type": "DataValidationStage",
                        "config": {
                            "quality_checks": {
                                "min_completeness": 0.8
                            }
                        }
                    },
                    {
                        "name": "transformation",
                        "type": "DataTransformationStage",
                        "config": {
                            "transformers": [
                                {
                                    "type": "StandardScaler",
                                    "features": ["x", "y", "z"]
                                }
                            ]
                        }
                    },
                    {
                        "name": "feature_engineering",
                        "type": "FeatureEngineeringStage",
                        "config": {
                            "feature_generators": [
                                {
                                    "type": "PolynomialFeatures",
                                    "degree": 2,
                                    "include_bias": False
                                }
                            ]
                        }
                    },
                    {
                        "name": "quality_assessment",
                        "type": "DataQualityStage",
                        "config": {
                            "quality_assessments": ["completeness", "consistency"]
                        }
                    }
                ]
            }
            
            # Execute pipeline
            pipeline = DataPipeline(pipeline_config)
            result = await pipeline.execute_async({})
            
            # Verify pipeline execution
            assert "final_data" in result
            assert "pipeline_metrics" in result
            assert "stage_results" in result
            
            # Check that all stages executed
            assert len(result["stage_results"]) == 5
            
            # Verify data transformation occurred
            final_data = result["final_data"]
            assert "transformed_data" in final_data
            assert "features" in final_data
            
            # Features should be expanded due to polynomial features
            assert len(final_data["features"]) > 3
            
        finally:
            os.unlink(temp_file)
            
    def test_pipeline_with_error_recovery(self):
        """Test pipeline with error recovery."""
        config = {
            "name": "error_recovery_test",
            "stages": [],
            "error_handling": {
                "strategy": "retry_with_fallback",
                "max_retries": 2,
                "fallback_strategy": "skip_stage"
            }
        }
        
        pipeline = DataPipeline(config)
        
        # Add stage that fails initially but succeeds on retry
        retry_count = 0
        def failing_then_succeeding_execute(data):
            nonlocal retry_count
            retry_count += 1
            if retry_count < 3:
                raise DataTransformationError("Temporary failure")
            return {"retried_data": data}
            
        retry_stage = Mock()
        retry_stage.execute = failing_then_succeeding_execute
        pipeline.add_stage(retry_stage)
        
        test_data = {"data": [[1, 2]]}
        result = pipeline.execute(test_data)
        
        # Should succeed after retries
        assert "retried_data" in result
        assert retry_count == 3
        
    def test_pipeline_branching_and_merging(self):
        """Test pipeline with branching and merging logic."""
        config = {
            "name": "branching_test",
            "stages": [],
            "execution": {
                "mode": "branched",
                "branches": {
                    "numerical_branch": {
                        "condition": "has_numerical_features",
                        "stages": ["numerical_processing"]
                    },
                    "categorical_branch": {
                        "condition": "has_categorical_features",
                        "stages": ["categorical_processing"]
                    }
                },
                "merge_strategy": "concatenate"
            }
        }
        
        pipeline = DataPipeline(config)
        
        # Mock branch stages
        numerical_stage = Mock()
        numerical_stage.execute.return_value = {"numerical_features": ["num_1", "num_2"]}
        
        categorical_stage = Mock()
        categorical_stage.execute.return_value = {"categorical_features": ["cat_1"]}
        
        pipeline.add_stage(numerical_stage, branch="numerical_branch")
        pipeline.add_stage(categorical_stage, branch="categorical_branch")
        
        # Data with both types
        mixed_data = {
            "data": [[1.0, "category_a"], [2.0, "category_b"]],
            "features": ["numeric_col", "category_col"],
            "feature_types": ["numerical", "categorical"]
        }
        
        result = pipeline.execute_branched(mixed_data)
        
        # Both branches should execute and merge
        assert "numerical_features" in result
        assert "categorical_features" in result
        assert "branch_execution_log" in result
        
    def test_pipeline_caching_and_checkpoints(self):
        """Test pipeline caching and checkpoint functionality."""
        config = {
            "name": "caching_test",
            "stages": [],
            "caching": {
                "enabled": True,
                "cache_stages": ["expensive_transformation"],
                "checkpoint_frequency": 2
            }
        }
        
        pipeline = DataPipeline(config)
        
        # Add expensive stage that should be cached
        expensive_stage = Mock()
        expensive_stage.name = "expensive_transformation"
        expensive_stage.execute.return_value = {"expensive_result": "computed"}
        pipeline.add_stage(expensive_stage)
        
        test_data = {"data": [[1, 2]]}
        
        # First execution
        result1 = pipeline.execute(test_data)
        assert expensive_stage.execute.call_count == 1
        
        # Second execution with same data (should use cache)
        result2 = pipeline.execute(test_data)
        assert expensive_stage.execute.call_count == 1  # No additional call
        
        # Results should be identical
        assert result1 == result2
        
    def test_pipeline_monitoring_and_metrics(self):
        """Test pipeline monitoring and metrics collection."""
        config = {
            "name": "monitoring_test",
            "stages": [],
            "monitoring": {
                "enabled": True,
                "metrics": [
                    "execution_time",
                    "memory_usage",
                    "throughput",
                    "error_rate"
                ],
                "export_metrics": True
            }
        }
        
        pipeline = DataPipeline(config)
        
        # Add stages with different characteristics
        fast_stage = Mock()
        fast_stage.execute.return_value = {"fast_result": True}
        
        slow_stage = Mock()
        def slow_execute(data):
            import time
            time.sleep(0.1)  # Simulate slow operation
            return {"slow_result": True}
        slow_stage.execute = slow_execute
        
        pipeline.add_stage(fast_stage)
        pipeline.add_stage(slow_stage)
        
        test_data = {"data": [[1, 2], [3, 4], [5, 6]]}
        result = pipeline.execute(test_data)
        
        # Check monitoring data
        assert "monitoring_data" in result
        monitoring = result["monitoring_data"]
        
        assert "total_execution_time" in monitoring
        assert "stage_execution_times" in monitoring
        assert "throughput" in monitoring
        assert "memory_usage" in monitoring
        
        # Stage execution times should be recorded
        assert len(monitoring["stage_execution_times"]) == 2
        
    async def test_streaming_pipeline_execution(self):
        """Test streaming pipeline execution."""
        config = {
            "name": "streaming_test",
            "stages": [],
            "execution": {
                "mode": "streaming",
                "batch_size": 2,
                "buffer_size": 10
            }
        }
        
        pipeline = DataPipeline(config)
        
        # Add streaming-compatible stage
        streaming_stage = Mock()
        streaming_stage.execute_batch.return_value = {"processed_batch": True}
        pipeline.add_stage(streaming_stage)
        
        # Simulate streaming data
        async def data_generator():
            for i in range(10):
                yield {"data": [[i, i+1]], "timestamp": datetime.utcnow()}
                await asyncio.sleep(0.01)
                
        # Execute streaming pipeline
        results = []
        async for batch_result in pipeline.execute_streaming(data_generator()):
            results.append(batch_result)
            
        # Should process all batches
        assert len(results) > 0
        assert all("processed_batch" in result for result in results)


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
