"""Branch coverage tests for data loader factory infrastructure."""

import os
import tempfile
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock
from urllib.parse import urlparse

import pytest

from pynomaly.domain.entities import Dataset
from pynomaly.domain.exceptions import DataValidationError
from pynomaly.infrastructure.data_loaders.data_loader_factory import (
    DataLoaderFactory,
    SmartDataLoader,
)


class TestDataLoaderFactoryDetectionBranches:
    """Test data loader factory format detection branches."""

    def test_detect_loader_type_from_extension(self):
        """Test auto-detection from file extension."""
        factory = DataLoaderFactory()
        
        # Test various extensions
        assert factory._detect_loader_type("data.csv") == "csv"
        assert factory._detect_loader_type("data.parquet") == "parquet"
        assert factory._detect_loader_type("data.json") == "json"
        assert factory._detect_loader_type("data.xlsx") == "xlsx"

    def test_detect_loader_type_case_insensitive(self):
        """Test case-insensitive extension detection."""
        factory = DataLoaderFactory()
        
        assert factory._detect_loader_type("DATA.CSV") == "csv"
        assert factory._detect_loader_type("Data.Parquet") == "parquet"
        assert factory._detect_loader_type("file.JSON") == "json"

    def test_detect_loader_type_from_url_with_extension(self):
        """Test detection from URL with extension."""
        factory = DataLoaderFactory()
        
        url = "https://example.com/data/file.csv"
        assert factory._detect_loader_type(url) == "csv"
        
        url = "http://data.example.com/dataset.parquet"
        assert factory._detect_loader_type(url) == "parquet"

    def test_detect_loader_type_from_url_without_extension(self):
        """Test detection failure from URL without extension."""
        factory = DataLoaderFactory()
        
        url = "https://example.com/api/data"
        
        with pytest.raises(DataValidationError) as exc_info:
            factory._detect_loader_type(url)
        
        assert "Cannot detect file type from URL" in str(exc_info.value)
        assert "Specify loader_type explicitly" in str(exc_info.value)

    def test_detect_loader_type_no_extension(self):
        """Test detection failure for file without extension."""
        factory = DataLoaderFactory()
        
        with pytest.raises(DataValidationError) as exc_info:
            factory._detect_loader_type("filename_without_extension")
        
        assert "Cannot detect file type: no extension found" in str(exc_info.value)
        assert "Specify loader_type explicitly" in str(exc_info.value)

    def test_detect_loader_type_unsupported_extension(self):
        """Test detection failure for unsupported extension."""
        factory = DataLoaderFactory()
        
        with pytest.raises(DataValidationError) as exc_info:
            factory._detect_loader_type("data.unknown")
        
        assert "Unsupported file extension: .unknown" in str(exc_info.value)
        assert "supported_formats" in str(exc_info.value)

    def test_detect_loader_type_url_parsing_branches(self):
        """Test URL parsing branches."""
        factory = DataLoaderFactory()
        
        # Test URL with path that has extension
        url = "https://example.com/path/to/file.csv?param=value"
        assert factory._detect_loader_type(url) == "csv"
        
        # Test URL with complex path
        url = "https://example.com/api/v1/data/export.parquet#section"
        assert factory._detect_loader_type(url) == "parquet"


class TestDataLoaderFactoryCreationBranches:
    """Test data loader factory creation branches."""

    def test_create_loader_with_explicit_type(self):
        """Test creating loader with explicit type."""
        factory = DataLoaderFactory()
        
        # Mock loader class
        mock_loader_class = Mock()
        mock_loader_instance = Mock()
        mock_loader_class.return_value = mock_loader_instance
        
        factory._loaders["custom"] = mock_loader_class
        
        loader = factory.create_loader("test.unknown", loader_type="custom")
        
        assert loader is mock_loader_instance
        mock_loader_class.assert_called_once()

    def test_create_loader_with_auto_detection(self):
        """Test creating loader with auto-detection."""
        factory = DataLoaderFactory()
        
        # Mock CSV loader
        mock_csv_loader = Mock()
        mock_csv_instance = Mock()
        mock_csv_loader.return_value = mock_csv_instance
        
        factory._loaders["csv"] = mock_csv_loader
        
        loader = factory.create_loader("data.csv")
        
        assert loader is mock_csv_instance
        mock_csv_loader.assert_called_once()

    def test_create_loader_unsupported_type(self):
        """Test creating loader with unsupported type."""
        factory = DataLoaderFactory()
        
        with pytest.raises(DataValidationError) as exc_info:
            factory.create_loader("test.csv", loader_type="unsupported")
        
        assert "Unsupported data format: unsupported" in str(exc_info.value)
        assert "supported_formats" in str(exc_info.value)

    def test_create_loader_case_normalization(self):
        """Test loader type case normalization."""
        factory = DataLoaderFactory()
        
        # Mock loader
        mock_loader = Mock()
        mock_instance = Mock()
        mock_loader.return_value = mock_instance
        
        factory._loaders["csv"] = mock_loader
        
        # Test uppercase type
        loader = factory.create_loader("test.csv", loader_type="CSV")
        assert loader is mock_instance

    def test_create_loader_config_merging(self):
        """Test configuration merging branches."""
        factory = DataLoaderFactory()
        
        # Set up default config
        factory._default_configs["csv"] = {"delimiter": ",", "encoding": "utf-8"}
        
        # Mock loader class
        mock_loader_class = Mock()
        mock_loader_instance = Mock()
        mock_loader_class.return_value = mock_loader_instance
        
        factory._loaders["csv"] = mock_loader_class
        
        # Test with additional kwargs
        loader = factory.create_loader(
            "test.csv",
            loader_type="csv",
            delimiter=";",
            header=True
        )
        
        # Should merge default config with kwargs
        expected_config = {
            "delimiter": ";",  # User override
            "encoding": "utf-8",  # Default
            "header": True  # User addition
        }
        
        mock_loader_class.assert_called_once_with(**expected_config)

    def test_create_loader_no_default_config(self):
        """Test creating loader with no default config."""
        factory = DataLoaderFactory()
        
        # Mock loader without default config
        mock_loader_class = Mock()
        mock_loader_instance = Mock()
        mock_loader_class.return_value = mock_loader_instance
        
        factory._loaders["custom"] = mock_loader_class
        # Note: no default config for "custom"
        
        loader = factory.create_loader("test.custom", loader_type="custom", param1="value1")
        
        # Should use only user kwargs
        mock_loader_class.assert_called_once_with(param1="value1")


class TestDataLoaderFactoryValidationBranches:
    """Test data loader factory validation branches."""

    def test_validate_source_success(self):
        """Test successful source validation."""
        factory = DataLoaderFactory()
        
        # Mock loader with successful validation
        mock_loader = Mock()
        mock_loader.validate.return_value = True
        
        mock_loader_class = Mock(return_value=mock_loader)
        factory._loaders["csv"] = mock_loader_class
        
        result = factory.validate_source("test.csv")
        
        assert result is True
        mock_loader.validate.assert_called_once_with("test.csv")

    def test_validate_source_failure(self):
        """Test source validation failure."""
        factory = DataLoaderFactory()
        
        # Mock loader with failed validation
        mock_loader = Mock()
        mock_loader.validate.return_value = False
        
        mock_loader_class = Mock(return_value=mock_loader)
        factory._loaders["csv"] = mock_loader_class
        
        result = factory.validate_source("test.csv")
        
        assert result is False

    def test_validate_source_exception(self):
        """Test source validation with exception."""
        factory = DataLoaderFactory()
        
        # Mock loader that raises exception
        mock_loader = Mock()
        mock_loader.validate.side_effect = Exception("validation error")
        
        mock_loader_class = Mock(return_value=mock_loader)
        factory._loaders["csv"] = mock_loader_class
        
        result = factory.validate_source("test.csv")
        
        assert result is False

    def test_validate_source_with_explicit_type(self):
        """Test validation with explicit loader type."""
        factory = DataLoaderFactory()
        
        # Mock loader
        mock_loader = Mock()
        mock_loader.validate.return_value = True
        
        mock_loader_class = Mock(return_value=mock_loader)
        factory._loaders["json"] = mock_loader_class
        
        result = factory.validate_source("test.unknown", loader_type="json")
        
        assert result is True
        mock_loader.validate.assert_called_once_with("test.unknown")


class TestDataLoaderFactoryInfoBranches:
    """Test data loader factory info retrieval branches."""

    def test_get_loader_info_success(self):
        """Test successful loader info retrieval."""
        factory = DataLoaderFactory()
        
        # Mock loader class
        mock_loader_class = Mock()
        mock_loader_class.__name__ = "MockLoader"
        mock_loader_class.__doc__ = "Mock loader documentation"
        
        # Mock loader instance
        mock_loader_instance = Mock()
        mock_loader_instance.supported_formats = ["csv", "tsv"]
        mock_loader_class.return_value = mock_loader_instance
        
        factory._loaders["csv"] = mock_loader_class
        factory._default_configs["csv"] = {"delimiter": ","}
        
        info = factory.get_loader_info("csv")
        
        expected_info = {
            "class_name": "MockLoader",
            "supported_formats": ["csv", "tsv"],
            "default_config": {"delimiter": ","},
            "description": "Mock loader documentation"
        }
        
        assert info == expected_info

    def test_get_loader_info_unknown_type(self):
        """Test loader info for unknown type."""
        factory = DataLoaderFactory()
        
        with pytest.raises(ValueError) as exc_info:
            factory.get_loader_info("unknown")
        
        assert "Unknown loader type: unknown" in str(exc_info.value)

    def test_get_loader_info_case_normalization(self):
        """Test loader info with case normalization."""
        factory = DataLoaderFactory()
        
        # Mock loader
        mock_loader_class = Mock()
        mock_loader_class.__name__ = "CSVLoader"
        mock_loader_class.__doc__ = None  # Test None doc branch
        
        mock_loader_instance = Mock()
        mock_loader_instance.supported_formats = ["csv"]
        mock_loader_class.return_value = mock_loader_instance
        
        factory._loaders["csv"] = mock_loader_class
        
        info = factory.get_loader_info("CSV")  # Uppercase
        
        assert info["class_name"] == "CSVLoader"
        assert info["description"] == ""  # Empty string for None doc

    def test_get_loader_info_no_default_config(self):
        """Test loader info with no default config."""
        factory = DataLoaderFactory()
        
        # Mock loader without default config
        mock_loader_class = Mock()
        mock_loader_class.__name__ = "CustomLoader"
        mock_loader_class.__doc__ = "Custom loader"
        
        mock_loader_instance = Mock()
        mock_loader_instance.supported_formats = ["custom"]
        mock_loader_class.return_value = mock_loader_instance
        
        factory._loaders["custom"] = mock_loader_class
        # No default config for "custom"
        
        info = factory.get_loader_info("custom")
        
        assert info["default_config"] == {}


class TestDataLoaderFactoryRegistrationBranches:
    """Test data loader factory registration branches."""

    def test_register_loader_single_extension(self):
        """Test registering loader with single extension."""
        factory = DataLoaderFactory()
        
        mock_loader_class = Mock()
        mock_loader_class.__name__ = "CustomLoader"
        
        factory.register_loader(["custom"], mock_loader_class)
        
        assert "custom" in factory._loaders
        assert factory._loaders["custom"] is mock_loader_class

    def test_register_loader_multiple_extensions(self):
        """Test registering loader with multiple extensions."""
        factory = DataLoaderFactory()
        
        mock_loader_class = Mock()
        mock_loader_class.__name__ = "MultiLoader"
        
        factory.register_loader(["ext1", "ext2", "ext3"], mock_loader_class)
        
        for ext in ["ext1", "ext2", "ext3"]:
            assert ext in factory._loaders
            assert factory._loaders[ext] is mock_loader_class

    def test_register_loader_with_default_config(self):
        """Test registering loader with default config."""
        factory = DataLoaderFactory()
        
        mock_loader_class = Mock()
        mock_loader_class.__name__ = "ConfigLoader"
        
        default_config = {"param1": "value1", "param2": "value2"}
        
        factory.register_loader(["config"], mock_loader_class, default_config)
        
        assert "config" in factory._loaders
        assert factory._loaders["config"] is mock_loader_class
        assert factory._default_configs["config"] == default_config

    def test_register_loader_extension_normalization(self):
        """Test extension normalization during registration."""
        factory = DataLoaderFactory()
        
        mock_loader_class = Mock()
        mock_loader_class.__name__ = "NormLoader"
        
        # Test with dots and mixed case
        factory.register_loader([".EXT", "Ext2", ".ext3"], mock_loader_class)
        
        # Should normalize to lowercase without dots
        for ext in ["ext", "ext2", "ext3"]:
            assert ext in factory._loaders
            assert factory._loaders[ext] is mock_loader_class

    def test_register_loader_no_default_config(self):
        """Test registering loader without default config."""
        factory = DataLoaderFactory()
        
        mock_loader_class = Mock()
        mock_loader_class.__name__ = "NoConfigLoader"
        
        factory.register_loader(["noconfig"], mock_loader_class, None)
        
        assert "noconfig" in factory._loaders
        assert factory._loaders["noconfig"] is mock_loader_class
        assert "noconfig" not in factory._default_configs


class TestSmartDataLoaderBranches:
    """Test smart data loader branches."""

    def test_smart_loader_default_factory(self):
        """Test smart loader with default factory."""
        smart_loader = SmartDataLoader()
        
        assert smart_loader.factory is not None
        assert isinstance(smart_loader.factory, DataLoaderFactory)

    def test_smart_loader_custom_factory(self):
        """Test smart loader with custom factory."""
        custom_factory = Mock()
        smart_loader = SmartDataLoader(factory=custom_factory)
        
        assert smart_loader.factory is custom_factory

    def test_smart_loader_file_not_found(self):
        """Test smart loader with non-existent file."""
        smart_loader = SmartDataLoader()
        
        with pytest.raises(DataValidationError) as exc_info:
            smart_loader.load("nonexistent_file.csv")
        
        assert "Source file not found" in str(exc_info.value)

    def test_smart_loader_small_file_standard_loading(self):
        """Test standard loading for small files."""
        smart_loader = SmartDataLoader(memory_threshold_mb=1000.0)
        
        # Create small temporary file
        with tempfile.NamedTemporaryFile(suffix=".csv", delete=False) as tmp:
            tmp.write(b"col1,col2\n1,2\n3,4\n")
            tmp_path = tmp.name
        
        try:
            # Mock the factory and loader
            mock_factory = Mock()
            mock_loader = Mock()
            mock_dataset = Mock()
            
            mock_factory.create_loader.return_value = mock_loader
            mock_loader.load.return_value = mock_dataset
            
            smart_loader.factory = mock_factory
            
            result = smart_loader.load(tmp_path)
            
            assert result is mock_dataset
            mock_factory.create_loader.assert_called_once()
            mock_loader.load.assert_called_once()
            
        finally:
            os.unlink(tmp_path)

    def test_smart_loader_large_file_optimized_loading(self):
        """Test optimized loading for large files."""
        smart_loader = SmartDataLoader(
            auto_optimize=True,
            memory_threshold_mb=0.001  # Very small threshold
        )
        
        # Create temporary file that will be considered "large"
        with tempfile.NamedTemporaryFile(suffix=".csv", delete=False) as tmp:
            tmp.write(b"col1,col2\n" * 1000)  # Make it larger than threshold
            tmp_path = tmp.name
        
        try:
            # Mock the factory and loader
            mock_factory = Mock()
            mock_loader = Mock()
            mock_dataset = Mock()
            
            mock_factory.create_loader.return_value = mock_loader
            mock_loader.load.return_value = mock_dataset
            
            smart_loader.factory = mock_factory
            
            result = smart_loader.load(tmp_path)
            
            assert result is mock_dataset
            
        finally:
            os.unlink(tmp_path)

    def test_smart_loader_optimization_disabled(self):
        """Test smart loader with optimization disabled."""
        smart_loader = SmartDataLoader(auto_optimize=False)
        
        # Create temporary file
        with tempfile.NamedTemporaryFile(suffix=".csv", delete=False) as tmp:
            tmp.write(b"col1,col2\n" * 1000)
            tmp_path = tmp.name
        
        try:
            # Mock the factory and loader
            mock_factory = Mock()
            mock_loader = Mock()
            mock_dataset = Mock()
            
            mock_factory.create_loader.return_value = mock_loader
            mock_loader.load.return_value = mock_dataset
            
            smart_loader.factory = mock_factory
            
            result = smart_loader.load(tmp_path)
            
            assert result is mock_dataset
            
        finally:
            os.unlink(tmp_path)

    def test_smart_loader_large_file_with_batch_loading(self):
        """Test large file loading with batch loading support."""
        smart_loader = SmartDataLoader(
            auto_optimize=True,
            memory_threshold_mb=0.001
        )
        
        # Create temporary file
        with tempfile.NamedTemporaryFile(suffix=".csv", delete=False) as tmp:
            tmp.write(b"col1,col2\n" * 1000)
            tmp_path = tmp.name
        
        try:
            # Mock loader with batch loading support
            mock_factory = Mock()
            mock_loader = Mock()
            mock_dataset = Mock()
            
            # Add load_batch method to indicate batch loading support
            mock_loader.load_batch = Mock()
            mock_loader.load.return_value = mock_dataset
            
            mock_factory.create_loader.return_value = mock_loader
            smart_loader.factory = mock_factory
            
            result = smart_loader.load(tmp_path)
            
            assert result is mock_dataset
            
        finally:
            os.unlink(tmp_path)

    def test_smart_loader_optimization_fallback(self):
        """Test optimization fallback when optimized loading fails."""
        smart_loader = SmartDataLoader(
            auto_optimize=True,
            memory_threshold_mb=0.001
        )
        
        # Create temporary file
        with tempfile.NamedTemporaryFile(suffix=".csv", delete=False) as tmp:
            tmp.write(b"col1,col2\n" * 1000)
            tmp_path = tmp.name
        
        try:
            # Mock loader that fails with optimization then succeeds
            mock_factory = Mock()
            mock_loader = Mock()
            mock_dataset = Mock()
            
            mock_loader.load_batch = Mock()  # Has batch loading
            mock_loader.load.side_effect = [
                Exception("Optimized loading failed"),  # First call fails
                mock_dataset  # Second call succeeds
            ]
            
            mock_factory.create_loader.return_value = mock_loader
            smart_loader.factory = mock_factory
            
            result = smart_loader.load(tmp_path)
            
            assert result is mock_dataset
            assert mock_loader.load.call_count == 2  # Called twice (optimized + fallback)
            
        finally:
            os.unlink(tmp_path)


class TestSmartDataLoaderMultipleFilesBranches:
    """Test smart data loader multiple files branches."""

    def test_load_multiple_without_names(self):
        """Test loading multiple files without names."""
        smart_loader = SmartDataLoader()
        
        # Create temporary files
        files = []
        for i in range(2):
            with tempfile.NamedTemporaryFile(suffix=".csv", delete=False) as tmp:
                tmp.write(f"col1,col2\n{i},{i+1}\n".encode())
                files.append(tmp.name)
        
        try:
            # Mock the load method
            mock_datasets = [Mock(), Mock()]
            with patch.object(smart_loader, 'load', side_effect=mock_datasets):
                result = smart_loader.load_multiple(files)
            
            assert result == mock_datasets
            
        finally:
            for f in files:
                os.unlink(f)

    def test_load_multiple_with_names(self):
        """Test loading multiple files with names."""
        smart_loader = SmartDataLoader()
        
        # Create temporary files
        files = []
        for i in range(2):
            with tempfile.NamedTemporaryFile(suffix=".csv", delete=False) as tmp:
                tmp.write(f"col1,col2\n{i},{i+1}\n".encode())
                files.append(tmp.name)
        
        try:
            names = ["dataset1", "dataset2"]
            mock_datasets = [Mock(), Mock()]
            
            with patch.object(smart_loader, 'load', side_effect=mock_datasets):
                result = smart_loader.load_multiple(files, names=names)
            
            assert result == mock_datasets
            
        finally:
            for f in files:
                os.unlink(f)

    def test_load_multiple_with_partial_names(self):
        """Test loading multiple files with partial names."""
        smart_loader = SmartDataLoader()
        
        # Create temporary files
        files = []
        for i in range(3):
            with tempfile.NamedTemporaryFile(suffix=".csv", delete=False) as tmp:
                tmp.write(f"col1,col2\n{i},{i+1}\n".encode())
                files.append(tmp.name)
        
        try:
            names = ["dataset1"]  # Only one name for three files
            mock_datasets = [Mock(), Mock(), Mock()]
            
            with patch.object(smart_loader, 'load', side_effect=mock_datasets):
                result = smart_loader.load_multiple(files, names=names)
            
            assert result == mock_datasets
            
        finally:
            for f in files:
                os.unlink(f)

    def test_load_multiple_combine_single_dataset(self):
        """Test combining single dataset (edge case)."""
        smart_loader = SmartDataLoader()
        
        # Create temporary file
        with tempfile.NamedTemporaryFile(suffix=".csv", delete=False) as tmp:
            tmp.write(b"col1,col2\n1,2\n")
            tmp_path = tmp.name
        
        try:
            mock_dataset = Mock()
            
            with patch.object(smart_loader, 'load', return_value=mock_dataset):
                result = smart_loader.load_multiple([tmp_path], combine=True)
            
            assert result is mock_dataset
            
        finally:
            os.unlink(tmp_path)

    def test_load_multiple_combine_no_datasets(self):
        """Test combining with no datasets (error case)."""
        smart_loader = SmartDataLoader()
        
        with patch.object(smart_loader, '_combine_datasets') as mock_combine:
            mock_combine.side_effect = ValueError("No datasets to combine")
            
            with patch.object(smart_loader, 'load', return_value=Mock()):
                with pytest.raises(ValueError):
                    smart_loader.load_multiple([], combine=True)


class TestSmartDataLoaderEstimationBranches:
    """Test smart data loader estimation branches."""

    def test_estimate_load_time_file_not_found(self):
        """Test estimation with non-existent file."""
        smart_loader = SmartDataLoader()
        
        with pytest.raises(DataValidationError) as exc_info:
            smart_loader.estimate_load_time("nonexistent.csv")
        
        assert "Source file not found" in str(exc_info.value)

    def test_estimate_load_time_with_loader_estimation(self):
        """Test estimation with loader that supports size estimation."""
        smart_loader = SmartDataLoader()
        
        # Create temporary file
        with tempfile.NamedTemporaryFile(suffix=".csv", delete=False) as tmp:
            tmp.write(b"col1,col2\n1,2\n")
            tmp_path = tmp.name
        
        try:
            # Mock factory and loader with size estimation
            mock_factory = Mock()
            mock_loader = Mock()
            mock_loader.estimate_size.return_value = {
                "rows": 1000,
                "columns": 10,
                "memory_mb": 50
            }
            
            mock_factory.create_loader.return_value = mock_loader
            smart_loader.factory = mock_factory
            
            result = smart_loader.estimate_load_time(tmp_path)
            
            assert "rows" in result
            assert "columns" in result
            assert "memory_mb" in result
            assert "estimated_load_time_seconds" in result
            
        finally:
            os.unlink(tmp_path)

    def test_estimate_load_time_without_loader_estimation(self):
        """Test estimation with loader that doesn't support size estimation."""
        smart_loader = SmartDataLoader()
        
        # Create temporary file
        with tempfile.NamedTemporaryFile(suffix=".csv", delete=False) as tmp:
            tmp.write(b"col1,col2\n1,2\n")
            tmp_path = tmp.name
        
        try:
            # Mock factory and loader without size estimation
            mock_factory = Mock()
            mock_loader = Mock()
            # No estimate_size method
            
            mock_factory.create_loader.return_value = mock_loader
            smart_loader.factory = mock_factory
            
            result = smart_loader.estimate_load_time(tmp_path)
            
            assert "file_size_mb" in result
            assert "estimated_load_time_seconds" in result
            
        finally:
            os.unlink(tmp_path)

    def test_estimate_load_time_format_adjustments(self):
        """Test load time estimation format adjustments."""
        smart_loader = SmartDataLoader()
        
        # Test different file formats
        formats = [
            (".json", 2.0),   # JSON multiplier
            (".jsonl", 2.0),  # JSONL multiplier  
            (".xlsx", 3.0),   # Excel multiplier
            (".csv", 1.0),    # No multiplier
        ]
        
        for suffix, expected_multiplier in formats:
            with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
                tmp.write(b"test data" * 100)  # Make it measurable
                tmp_path = tmp.name
            
            try:
                # Mock factory
                mock_factory = Mock()
                mock_loader = Mock()
                mock_factory.create_loader.return_value = mock_loader
                smart_loader.factory = mock_factory
                
                result = smart_loader.estimate_load_time(tmp_path)
                
                assert "file_extension" in result
                assert result["file_extension"] == suffix
                assert "estimated_load_time_seconds" in result
                
            finally:
                os.unlink(tmp_path)

    def test_estimate_load_time_exception_handling(self):
        """Test estimation with exception during processing."""
        smart_loader = SmartDataLoader()
        
        # Create temporary file
        with tempfile.NamedTemporaryFile(suffix=".csv", delete=False) as tmp:
            tmp.write(b"col1,col2\n1,2\n")
            tmp_path = tmp.name
        
        try:
            # Mock factory that raises exception
            mock_factory = Mock()
            mock_factory.create_loader.side_effect = Exception("Factory error")
            smart_loader.factory = mock_factory
            
            result = smart_loader.estimate_load_time(tmp_path)
            
            assert "file_size_mb" in result
            assert "error" in result
            assert result["estimated_load_time_seconds"] == "unknown"
            
        finally:
            os.unlink(tmp_path)

    def test_estimate_load_time_memory_threshold_recommendation(self):
        """Test memory threshold recommendation branch."""
        smart_loader = SmartDataLoader(memory_threshold_mb=1.0)
        
        # Create file larger than threshold
        with tempfile.NamedTemporaryFile(suffix=".csv", delete=False) as tmp:
            tmp.write(b"col1,col2\n" * 10000)  # Large file
            tmp_path = tmp.name
        
        try:
            # Mock factory
            mock_factory = Mock()
            mock_loader = Mock()
            mock_factory.create_loader.return_value = mock_loader
            smart_loader.factory = mock_factory
            
            result = smart_loader.estimate_load_time(tmp_path)
            
            assert "recommended_batch_loading" in result
            
        finally:
            os.unlink(tmp_path)


class TestSmartDataLoaderCombinationBranches:
    """Test smart data loader dataset combination branches."""

    def test_combine_datasets_target_column_selection(self):
        """Test target column selection in dataset combination."""
        smart_loader = SmartDataLoader()
        
        # Mock datasets with different target columns
        mock_dataset1 = Mock()
        mock_dataset1.name = "dataset1"
        mock_dataset1.target_column = None
        mock_dataset1.data = Mock()
        mock_dataset1.metadata = {"source": "file1"}
        
        mock_dataset2 = Mock()
        mock_dataset2.name = "dataset2"
        mock_dataset2.target_column = "target_col"
        mock_dataset2.data = Mock()
        mock_dataset2.metadata = {"source": "file2"}
        
        mock_dataset3 = Mock()
        mock_dataset3.name = "dataset3"
        mock_dataset3.target_column = "another_target"
        mock_dataset3.data = Mock()
        mock_dataset3.metadata = {"source": "file3"}
        
        datasets = [mock_dataset1, mock_dataset2, mock_dataset3]
        
        with patch('pandas.concat') as mock_concat:
            mock_combined_data = Mock()
            mock_combined_data.shape = (100, 5)
            mock_concat.return_value = mock_combined_data
            
            with patch('pynomaly.infrastructure.data_loaders.data_loader_factory.Dataset') as mock_dataset_class:
                mock_combined_dataset = Mock()
                mock_dataset_class.return_value = mock_combined_dataset
                
                result = smart_loader._combine_datasets(datasets)
                
                # Should use first non-None target column
                mock_dataset_class.assert_called_once()
                call_args = mock_dataset_class.call_args
                assert call_args[1]["target_column"] == "target_col"

    def test_combine_datasets_no_target_column(self):
        """Test dataset combination when no target column exists."""
        smart_loader = SmartDataLoader()
        
        # Mock datasets with no target columns
        mock_dataset1 = Mock()
        mock_dataset1.name = "dataset1"
        mock_dataset1.target_column = None
        mock_dataset1.data = Mock()
        mock_dataset1.metadata = {"source": "file1"}
        
        mock_dataset2 = Mock()
        mock_dataset2.name = "dataset2"
        mock_dataset2.target_column = None
        mock_dataset2.data = Mock()
        mock_dataset2.metadata = {"source": "file2"}
        
        datasets = [mock_dataset1, mock_dataset2]
        
        with patch('pandas.concat') as mock_concat:
            mock_combined_data = Mock()
            mock_combined_data.shape = (100, 5)
            mock_concat.return_value = mock_combined_data
            
            with patch('pynomaly.infrastructure.data_loaders.data_loader_factory.Dataset') as mock_dataset_class:
                mock_combined_dataset = Mock()
                mock_dataset_class.return_value = mock_combined_dataset
                
                result = smart_loader._combine_datasets(datasets)
                
                # Should use None for target column
                mock_dataset_class.assert_called_once()
                call_args = mock_dataset_class.call_args
                assert call_args[1]["target_column"] is None

    def test_combine_datasets_metadata_construction(self):
        """Test metadata construction in dataset combination."""
        smart_loader = SmartDataLoader()
        
        # Mock datasets with metadata
        mock_dataset1 = Mock()
        mock_dataset1.name = "dataset1"
        mock_dataset1.target_column = None
        mock_dataset1.data = Mock()
        mock_dataset1.data.shape = (50, 3)
        mock_dataset1.metadata = {"source": "file1", "rows": 50}
        
        mock_dataset2 = Mock()
        mock_dataset2.name = "dataset2"
        mock_dataset2.target_column = None
        mock_dataset2.data = Mock()
        mock_dataset2.data.shape = (30, 3)
        mock_dataset2.metadata = {"source": "file2", "rows": 30}
        
        datasets = [mock_dataset1, mock_dataset2]
        
        with patch('pandas.concat') as mock_concat:
            mock_combined_data = Mock()
            mock_combined_data.shape = (80, 3)
            mock_concat.return_value = mock_combined_data
            
            with patch('pynomaly.infrastructure.data_loaders.data_loader_factory.Dataset') as mock_dataset_class:
                mock_combined_dataset = Mock()
                mock_dataset_class.return_value = mock_combined_dataset
                
                result = smart_loader._combine_datasets(datasets)
                
                # Check metadata construction
                mock_dataset_class.assert_called_once()
                call_args = mock_dataset_class.call_args
                metadata = call_args[1]["metadata"]
                
                assert "combined_from" in metadata
                assert "original_shapes" in metadata
                assert "combined_shape" in metadata
                assert "dataset_0_metadata" in metadata
                assert "dataset_1_metadata" in metadata
                
                assert metadata["combined_from"] == ["dataset1", "dataset2"]
                assert metadata["original_shapes"] == [(50, 3), (30, 3)]
                assert metadata["combined_shape"] == (80, 3)
