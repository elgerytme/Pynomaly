"""Test cases for unified data service."""

import asyncio
from pathlib import Path
from unittest.mock import AsyncMock, Mock, patch

import numpy as np
import pandas as pd
import pytest

from pynomaly.application.services.unified_data_service import (
    DataSourceType,
    UnifiedDataService,
)
from pynomaly.domain.entities import Dataset
from pynomaly.domain.exceptions import DataValidationError
from pynomaly.infrastructure.data_processing.advanced_data_pipeline import (
    ProcessingConfig,
    ProcessingReport,
)


class TestDataSourceType:
    """Test cases for DataSourceType constants."""

    def test_data_source_types(self):
        """Test data source type constants."""
        assert DataSourceType.FILE == "file"
        assert DataSourceType.DATABASE == "database"
        assert DataSourceType.URL == "url"
        assert DataSourceType.DATAFRAME == "dataframe"
        assert DataSourceType.STREAM == "stream"


class TestUnifiedDataService:
    """Test cases for UnifiedDataService."""

    def test_init_default(self):
        """Test default initialization."""
        service = UnifiedDataService()

        assert service.data_loader_factory is not None
        assert service.database_loader is not None
        assert service.data_pipeline is not None
        assert service.max_workers == 4
        assert service.default_processing_config is not None
        assert service.smart_loader is not None
        assert service._processing_cache == {}
        assert service._dataset_registry == {}

    def test_init_custom_components(self):
        """Test initialization with custom components."""
        mock_factory = Mock()
        mock_db_loader = Mock()
        mock_pipeline = Mock()
        custom_config = ProcessingConfig(apply_scaling=False)

        service = UnifiedDataService(
            data_loader_factory=mock_factory,
            database_loader=mock_db_loader,
            data_pipeline=mock_pipeline,
            max_workers=8,
            default_processing_config=custom_config,
        )

        assert service.data_loader_factory is mock_factory
        assert service.database_loader is mock_db_loader
        assert service.data_pipeline is mock_pipeline
        assert service.max_workers == 8
        assert service.default_processing_config.apply_scaling is False

    @pytest.mark.asyncio
    async def test_load_and_process_dataframe(self):
        """Test loading and processing DataFrame."""
        df = pd.DataFrame(
            {
                "feature1": [1.0, 2.0, 3.0],
                "feature2": [4.0, 5.0, 6.0],
                "target": [0, 1, 0],
            }
        )

        service = UnifiedDataService()

        # Mock pipeline processing
        mock_processed_dataset = Dataset(
            name="test_processed",
            data=df.copy(),
            target_column="target",
            metadata={"processed": True},
        )

        with patch.object(
            service.data_pipeline,
            "process_dataset",
            return_value=mock_processed_dataset,
        ):
            result = await service.load_and_process(
                df, name="test_dataset", target_column="target"
            )

        assert isinstance(result, Dataset)
        assert result.target_column == "target"
        assert "test_dataset" in service._dataset_registry

    @pytest.mark.asyncio
    async def test_load_and_process_with_report(self):
        """Test loading and processing with report."""
        df = pd.DataFrame({"col1": [1, 2, 3], "col2": [4, 5, 6]})

        service = UnifiedDataService()

        mock_dataset = Dataset(name="test", data=df, metadata={})
        mock_report = ProcessingReport(
            original_shape=(3, 2),
            final_shape=(3, 2),
            processing_time=1.0,
            steps_performed=["test"],
        )

        with patch.object(
            service.data_pipeline,
            "process_dataset",
            return_value=(mock_dataset, mock_report),
        ):
            result, report = await service.load_and_process(df, return_report=True)

        assert isinstance(result, Dataset)
        assert isinstance(report, ProcessingReport)
        assert report.processing_time == 1.0

    @pytest.mark.asyncio
    async def test_load_and_process_with_caching(self):
        """Test caching functionality."""
        df = pd.DataFrame({"col1": [1, 2, 3]})

        service = UnifiedDataService()

        mock_dataset = Dataset(name="test", data=df, metadata={})
        mock_report = ProcessingReport(
            original_shape=(3, 1),
            final_shape=(3, 1),
            processing_time=1.0,
            steps_performed=["test"],
        )

        # First call should process
        with patch.object(
            service.data_pipeline,
            "process_dataset",
            return_value=(mock_dataset, mock_report),
        ) as mock_process:
            result1 = await service.load_and_process(df, cache_result=True)

            # Second call should use cache
            result2 = await service.load_and_process(df, cache_result=True)

        # Should only call process_dataset once
        assert mock_process.call_count == 1
        assert result1.name == result2.name

    @pytest.mark.asyncio
    async def test_load_and_process_auto_detect_target(self):
        """Test auto-detection of target column."""
        df = pd.DataFrame(
            {
                "feature1": [1.0, 2.0, 3.0],
                "feature2": [4.0, 5.0, 6.0],
                "target": [0, 1, 0],  # Should be auto-detected
            }
        )

        service = UnifiedDataService()

        mock_dataset = Dataset(
            name="test", data=df, target_column="target", metadata={}
        )

        with patch.object(
            service.data_pipeline, "process_dataset", return_value=mock_dataset
        ):
            result = await service.load_and_process(df, auto_detect_target=True)

        assert result.target_column == "target"

    @pytest.mark.asyncio
    async def test_load_and_process_file_source(self):
        """Test loading from file source."""
        service = UnifiedDataService()

        mock_dataset = Dataset(
            name="test_file",
            data=pd.DataFrame({"col1": [1, 2, 3]}),
            metadata={"source": "file"},
        )

        with (
            patch.object(service.smart_loader, "load", return_value=mock_dataset),
            patch.object(
                service.data_pipeline, "process_dataset", return_value=mock_dataset
            ),
        ):

            result = await service.load_and_process("test.csv")

        assert isinstance(result, Dataset)
        assert result.name == "test_file"

    @pytest.mark.asyncio
    async def test_load_and_process_database_source(self):
        """Test loading from database source."""
        service = UnifiedDataService()

        mock_dataset = Dataset(
            name="db_query",
            data=pd.DataFrame({"id": [1, 2, 3], "value": [10, 20, 30]}),
            metadata={"source": "database"},
        )

        with (
            patch.object(
                service.database_loader, "load_query", return_value=mock_dataset
            ),
            patch.object(
                service.data_pipeline, "process_dataset", return_value=mock_dataset
            ),
        ):

            result = await service.load_and_process(
                "postgresql://user:pass@host/db", query="SELECT * FROM test_table"
            )

        assert isinstance(result, Dataset)
        assert result.name == "db_query"

    @pytest.mark.asyncio
    async def test_load_and_process_url_source(self):
        """Test loading from URL source."""
        service = UnifiedDataService()

        mock_dataset = Dataset(
            name="url_data",
            data=pd.DataFrame({"col1": [1, 2, 3]}),
            metadata={"source": "url"},
        )

        with (
            patch.object(service.smart_loader, "load", return_value=mock_dataset),
            patch.object(
                service.data_pipeline, "process_dataset", return_value=mock_dataset
            ),
        ):

            result = await service.load_and_process("https://example.com/data.csv")

        assert isinstance(result, Dataset)
        assert result.name == "url_data"

    @pytest.mark.asyncio
    async def test_load_multiple_sources_parallel(self):
        """Test loading multiple sources in parallel."""
        service = UnifiedDataService()

        dataset1 = Dataset(
            name="dataset1", data=pd.DataFrame({"a": [1, 2]}), metadata={}
        )
        dataset2 = Dataset(
            name="dataset2", data=pd.DataFrame({"b": [3, 4]}), metadata={}
        )

        with patch.object(
            service, "load_and_process", side_effect=[dataset1, dataset2]
        ):
            result = await service.load_multiple_sources(
                ["file1.csv", "file2.csv"],
                names=["dataset1", "dataset2"],
                parallel=True,
            )

        assert len(result) == 2
        assert result[0].name == "dataset1"
        assert result[1].name == "dataset2"

    @pytest.mark.asyncio
    async def test_load_multiple_sources_sequential(self):
        """Test loading multiple sources sequentially."""
        service = UnifiedDataService()

        dataset1 = Dataset(
            name="dataset1", data=pd.DataFrame({"a": [1, 2]}), metadata={}
        )
        dataset2 = Dataset(
            name="dataset2", data=pd.DataFrame({"b": [3, 4]}), metadata={}
        )

        with patch.object(
            service, "load_and_process", side_effect=[dataset1, dataset2]
        ):
            result = await service.load_multiple_sources(
                ["file1.csv", "file2.csv"], parallel=False
            )

        assert len(result) == 2
        assert isinstance(result, list)

    @pytest.mark.asyncio
    async def test_load_multiple_sources_combine(self):
        """Test loading and combining multiple sources."""
        service = UnifiedDataService()

        dataset1 = Dataset(
            name="dataset1", data=pd.DataFrame({"a": [1, 2]}), metadata={}
        )
        dataset2 = Dataset(
            name="dataset2", data=pd.DataFrame({"a": [3, 4]}), metadata={}
        )

        with patch.object(
            service, "load_and_process", side_effect=[dataset1, dataset2]
        ):
            result = await service.load_multiple_sources(
                ["file1.csv", "file2.csv"], combine=True
            )

        assert isinstance(result, Dataset)
        assert result.name == "combined_dataset"
        assert len(result.data) == 4  # Combined rows

    @pytest.mark.asyncio
    async def test_load_multiple_sources_with_failures(self):
        """Test handling failures when loading multiple sources."""
        service = UnifiedDataService()

        dataset1 = Dataset(
            name="dataset1", data=pd.DataFrame({"a": [1, 2]}), metadata={}
        )

        with patch.object(
            service,
            "load_and_process",
            side_effect=[dataset1, Exception("Load failed")],
        ):
            result = await service.load_multiple_sources(
                ["file1.csv", "file2.csv"], parallel=True
            )

        assert len(result) == 1  # Only successful dataset
        assert result[0].name == "dataset1"

    @pytest.mark.asyncio
    async def test_load_multiple_sources_all_fail(self):
        """Test error when all sources fail to load."""
        service = UnifiedDataService()

        with patch.object(
            service, "load_and_process", side_effect=Exception("Load failed")
        ):
            with pytest.raises(
                DataValidationError, match="No sources could be loaded successfully"
            ):
                await service.load_multiple_sources(["file1.csv", "file2.csv"])

    def test_create_processing_config_default(self):
        """Test creating default processing configuration."""
        service = UnifiedDataService()

        config = service.create_processing_config()

        assert isinstance(config, ProcessingConfig)
        assert config.apply_scaling is True  # Default anomaly detection optimization

    def test_create_processing_config_with_characteristics(self):
        """Test creating config optimized for dataset characteristics."""
        service = UnifiedDataService()

        characteristics = {
            "n_samples": 150000,  # Large dataset
            "n_features": 1500,  # High dimensional
            "has_categorical": True,
        }

        config = service.create_processing_config(
            dataset_characteristics=characteristics, performance_preference="fast"
        )

        assert config.memory_efficient is True  # Large dataset optimization
        assert config.apply_feature_selection is True  # High dimensional optimization
        assert config.encode_categoricals is True  # Categorical handling
        assert config.parallel_processing is True  # Fast preference
        assert config.validate_data is False  # Fast preference

    def test_create_processing_config_classification_use_case(self):
        """Test creating config for classification use case."""
        service = UnifiedDataService()

        config = service.create_processing_config(
            use_case="classification", performance_preference="thorough"
        )

        assert config.apply_feature_selection is True
        assert config.scale_target is False  # Don't scale classification targets
        assert config.validate_data is True  # Thorough preference
        assert config.strict_validation is True  # Thorough preference

    def test_create_processing_config_custom_settings(self):
        """Test creating config with custom setting overrides."""
        service = UnifiedDataService()

        config = service.create_processing_config(
            apply_scaling=False,  # Custom override
            max_workers=16,
            custom_setting="value",  # Should be ignored if not valid
        )

        assert config.apply_scaling is False
        assert config.max_workers == 16

    def test_validate_dataset_quality_good(self):
        """Test quality validation for good dataset."""
        service = UnifiedDataService()

        df = pd.DataFrame(
            {
                "feature1": np.random.randn(1000),
                "feature2": np.random.randn(1000),
                "feature3": np.random.randn(1000),
            }
        )

        dataset = Dataset(name="test", data=df, metadata={})

        is_valid, quality_report = service.validate_dataset_quality(dataset)

        assert is_valid is True
        assert quality_report["overall_quality"] == "good"
        assert len(quality_report["issues"]) == 0
        assert "metrics" in quality_report
        assert quality_report["metrics"]["n_samples"] == 1000
        assert quality_report["metrics"]["n_features"] == 3

    def test_validate_dataset_quality_poor(self):
        """Test quality validation for poor dataset."""
        service = UnifiedDataService()

        # Create problematic dataset
        df = pd.DataFrame(
            {
                "mostly_missing": [1] + [np.nan] * 99,  # 99% missing
                "duplicate_col": [1] * 100,  # No variance
                "feature3": np.random.randn(100),
            }
        )

        dataset = Dataset(name="test", data=df, metadata={})

        requirements = {
            "min_samples": 200,  # More than we have
            "max_missing_percentage": 0.3,  # Less than we have
        }

        is_valid, quality_report = service.validate_dataset_quality(
            dataset, requirements
        )

        assert is_valid is False
        assert quality_report["overall_quality"] == "poor"
        assert len(quality_report["issues"]) > 0
        assert any(
            "Insufficient samples" in issue for issue in quality_report["issues"]
        )
        assert any(
            "Too many missing values" in issue for issue in quality_report["issues"]
        )

    def test_validate_dataset_quality_with_warnings(self):
        """Test quality validation with warnings but no errors."""
        service = UnifiedDataService()

        # Create dataset with issues but not severe enough for errors
        df = pd.DataFrame(
            {
                "feature1": np.random.randn(1000),
                "feature2": np.random.randn(1000),
                "correlated": np.random.randn(1000),  # Will create correlation
                "low_variance": [1.0] * 900 + [1.1] * 100,  # Low variance
            }
        )

        # Make features highly correlated
        df["correlated"] = df["feature1"] + 0.01 * np.random.randn(1000)

        dataset = Dataset(name="test", data=df, metadata={})

        is_valid, quality_report = service.validate_dataset_quality(dataset)

        assert is_valid is True  # No errors, just warnings
        assert quality_report["overall_quality"] == "fair"
        assert len(quality_report["warnings"]) > 0
        assert len(quality_report["recommendations"]) > 0

    def test_get_dataset_summary_existing(self):
        """Test getting summary for existing dataset."""
        service = UnifiedDataService()

        df = pd.DataFrame(
            {
                "numeric": [1.0, 2.0, 3.0],
                "categorical": ["A", "B", "C"],
                "datetime": pd.to_datetime(["2023-01-01", "2023-01-02", "2023-01-03"]),
                "target": [0, 1, 0],
            }
        )

        dataset = Dataset(
            name="test_dataset",
            data=df,
            target_column="target",
            metadata={"source": "test"},
        )

        service._dataset_registry["test_dataset"] = dataset

        summary = service.get_dataset_summary("test_dataset")

        assert summary is not None
        assert summary["name"] == "test_dataset"
        assert summary["shape"] == (3, 4)
        assert summary["target_column"] == "target"
        assert summary["has_target"] is True
        assert summary["columns"]["numeric"] == 1
        assert summary["columns"]["categorical"] == 1
        assert summary["columns"]["datetime"] == 1
        assert summary["quality"]["missing_values"] == 0
        assert summary["metadata"] == {"source": "test"}

    def test_get_dataset_summary_nonexistent(self):
        """Test getting summary for non-existent dataset."""
        service = UnifiedDataService()

        summary = service.get_dataset_summary("nonexistent")

        assert summary is None

    def test_list_registered_datasets(self):
        """Test listing registered datasets."""
        service = UnifiedDataService()

        # Initially empty
        assert service.list_registered_datasets() == []

        # Add some datasets
        dataset1 = Dataset(name="dataset1", data=pd.DataFrame(), metadata={})
        dataset2 = Dataset(name="dataset2", data=pd.DataFrame(), metadata={})

        service._dataset_registry["dataset1"] = dataset1
        service._dataset_registry["dataset2"] = dataset2

        names = service.list_registered_datasets()

        assert len(names) == 2
        assert "dataset1" in names
        assert "dataset2" in names

    def test_get_registered_dataset(self):
        """Test getting registered dataset by name."""
        service = UnifiedDataService()

        dataset = Dataset(name="test", data=pd.DataFrame(), metadata={})
        service._dataset_registry["test"] = dataset

        result = service.get_registered_dataset("test")
        assert result is dataset

        result = service.get_registered_dataset("nonexistent")
        assert result is None

    def test_clear_cache(self):
        """Test clearing all caches."""
        service = UnifiedDataService()

        # Add some cached data
        service._processing_cache["key1"] = ("data1", "report1")
        service._dataset_registry["dataset1"] = Mock()

        service.clear_cache()

        assert len(service._processing_cache) == 0
        assert len(service._dataset_registry) == 0

    def test_auto_detect_target_column_by_name(self):
        """Test auto-detection of target column by common names."""
        service = UnifiedDataService()

        df = pd.DataFrame(
            {
                "feature1": [1, 2, 3],
                "feature2": [4, 5, 6],
                "target": [0, 1, 0],  # Should be detected
            }
        )

        target_col = service._auto_detect_target_column(df)

        assert target_col == "target"

    def test_auto_detect_target_column_binary(self):
        """Test auto-detection of binary target column."""
        service = UnifiedDataService()

        df = pd.DataFrame(
            {
                "feature1": [1, 2, 3, 4],
                "feature2": [10, 20, 30, 40],
                "binary_col": [0, 1, 0, 1],  # Binary values
            }
        )

        target_col = service._auto_detect_target_column(df)

        assert target_col == "binary_col"

    def test_auto_detect_target_column_none_found(self):
        """Test when no target column can be auto-detected."""
        service = UnifiedDataService()

        df = pd.DataFrame(
            {
                "feature1": np.random.randn(10),
                "feature2": np.random.randn(10),
                "feature3": np.random.randn(10),
            }
        )

        target_col = service._auto_detect_target_column(df)

        assert target_col is None

    def test_combine_datasets_empty_list(self):
        """Test error when combining empty dataset list."""
        service = UnifiedDataService()

        with pytest.raises(ValueError, match="No datasets to combine"):
            service._combine_datasets([])

    def test_combine_datasets_single(self):
        """Test combining single dataset returns same dataset."""
        service = UnifiedDataService()

        dataset = Dataset(name="test", data=pd.DataFrame({"a": [1, 2]}), metadata={})

        result = service._combine_datasets([dataset])

        assert result is dataset

    def test_combine_datasets_multiple(self):
        """Test combining multiple datasets."""
        service = UnifiedDataService()

        dataset1 = Dataset(
            name="dataset1",
            data=pd.DataFrame({"a": [1, 2], "b": [3, 4]}),
            target_column="b",
            metadata={"source": "file1"},
        )

        dataset2 = Dataset(
            name="dataset2",
            data=pd.DataFrame({"a": [5, 6], "b": [7, 8]}),
            metadata={"source": "file2"},
        )

        result = service._combine_datasets([dataset1, dataset2])

        assert result.name == "combined_dataset"
        assert len(result.data) == 4
        assert list(result.data["a"]) == [1, 2, 5, 6]
        assert result.target_column == "b"  # From first dataset
        assert "combined_from" in result.metadata
        assert result.metadata["combined_from"] == ["dataset1", "dataset2"]
        assert "original_shapes" in result.metadata

    def test_generate_cache_key(self):
        """Test cache key generation."""
        service = UnifiedDataService()

        df = pd.DataFrame({"col1": [1, 2, 3], "col2": [4, 5, 6]})
        dataset = Dataset(name="test", data=df, metadata={})
        config = ProcessingConfig()

        key1 = service._generate_cache_key(dataset, config)
        key2 = service._generate_cache_key(dataset, config)

        # Same inputs should generate same key
        assert key1 == key2
        assert isinstance(key1, str)
        assert len(key1) > 0

        # Different config should generate different key
        config2 = ProcessingConfig(apply_scaling=False)
        key3 = service._generate_cache_key(dataset, config2)

        assert key1 != key3

    def test_is_database_source(self):
        """Test database source detection."""
        service = UnifiedDataService()

        assert service._is_database_source("postgresql://user:pass@host/db") is True
        assert service._is_database_source("mysql://user:pass@host/db") is True
        assert service._is_database_source("sqlite:///path/to/db.sqlite") is True
        assert service._is_database_source("mssql://user:pass@host/db") is True
        assert service._is_database_source("oracle://user:pass@host/db") is True
        assert service._is_database_source("snowflake://user:pass@host/db") is True

        assert service._is_database_source("file.csv") is False
        assert service._is_database_source("https://example.com/data.csv") is False
        assert service._is_database_source("table_name") is False

    def test_load_from_dataframe_empty(self):
        """Test error when loading empty DataFrame."""
        service = UnifiedDataService()

        with pytest.raises(DataValidationError, match="DataFrame is empty"):
            service._load_from_dataframe(pd.DataFrame(), "test")

    def test_load_from_dataframe_success(self):
        """Test successful DataFrame loading."""
        service = UnifiedDataService()

        df = pd.DataFrame({"col1": [1, 2, 3], "col2": [4, 5, 6]})

        result = service._load_from_dataframe(df, "test_dataset", target_column="col2")

        assert isinstance(result, Dataset)
        assert result.name == "test_dataset"
        assert result.target_column == "col2"
        assert result.metadata["source"] == "dataframe"
        assert result.metadata["original_shape"] == (3, 2)

    def test_load_from_database_with_query(self):
        """Test loading from database with query."""
        service = UnifiedDataService()

        mock_dataset = Dataset(
            name="query_result", data=pd.DataFrame({"id": [1, 2, 3]}), metadata={}
        )

        with patch.object(
            service.database_loader, "load_query", return_value=mock_dataset
        ):
            result = service._load_from_database(
                "postgresql://user:pass@host/db",
                "test_query",
                query="SELECT * FROM test_table",
            )

        assert result is mock_dataset

    def test_load_from_database_with_table(self):
        """Test loading from database with table name."""
        service = UnifiedDataService()

        mock_dataset = Dataset(
            name="table_data", data=pd.DataFrame({"id": [1, 2, 3]}), metadata={}
        )

        with patch.object(
            service.database_loader, "load_table", return_value=mock_dataset
        ):
            result = service._load_from_database(
                "postgresql://user:pass@host/db", "test_table", table_name="users"
            )

        assert result is mock_dataset

    def test_load_from_database_no_query_or_table(self):
        """Test error when no query or table specified for database."""
        service = UnifiedDataService()

        with pytest.raises(
            DataValidationError, match="Either 'query' or 'table_name' must be provided"
        ):
            service._load_from_database("postgresql://user:pass@host/db", "test")

    def test_load_from_url(self):
        """Test loading from URL."""
        service = UnifiedDataService()

        mock_dataset = Dataset(
            name="url_data", data=pd.DataFrame({"col1": [1, 2, 3]}), metadata={}
        )

        with patch.object(service.smart_loader, "load", return_value=mock_dataset):
            result = service._load_from_url("https://example.com/data.csv", "url_data")

        assert result is mock_dataset

    def test_load_from_file(self):
        """Test loading from file."""
        service = UnifiedDataService()

        mock_dataset = Dataset(
            name="file_data", data=pd.DataFrame({"col1": [1, 2, 3]}), metadata={}
        )

        with patch.object(service.smart_loader, "load", return_value=mock_dataset):
            result = service._load_from_file("test.csv", "file_data")

        assert result is mock_dataset
