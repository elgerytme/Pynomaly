"""Test cases for data loader factory."""

from unittest.mock import Mock, patch

import pandas as pd
import pytest

from monorepo.domain.entities import Dataset
from monorepo.domain.exceptions import DataValidationError
from monorepo.infrastructure.data_loaders.data_loader_factory import (
    DataLoaderFactory,
    SmartDataLoader,
)


class TestDataLoaderFactory:
    """Test cases for DataLoaderFactory."""

    def test_init_default_loaders(self):
        """Test factory initializes with default loaders."""
        factory = DataLoaderFactory()

        supported_formats = factory.get_supported_formats()
        expected_formats = [
            "csv",
            "tsv",
            "txt",
            "parquet",
            "pq",
            "json",
            "jsonl",
            "ndjson",
            "xlsx",
            "xls",
        ]

        assert all(fmt in supported_formats for fmt in expected_formats)

    def test_detect_loader_type_csv(self):
        """Test detection of CSV file type."""
        factory = DataLoaderFactory()

        # Test various CSV extensions
        assert factory._detect_loader_type("test.csv") == "csv"
        assert factory._detect_loader_type("data.CSV") == "csv"
        assert factory._detect_loader_type("/path/to/file.csv") == "csv"

    def test_detect_loader_type_parquet(self):
        """Test detection of Parquet file type."""
        factory = DataLoaderFactory()

        assert factory._detect_loader_type("test.parquet") == "parquet"
        assert factory._detect_loader_type("data.pq") == "pq"

    def test_detect_loader_type_json(self):
        """Test detection of JSON file type."""
        factory = DataLoaderFactory()

        assert factory._detect_loader_type("test.json") == "json"
        assert factory._detect_loader_type("data.jsonl") == "jsonl"
        assert factory._detect_loader_type("stream.ndjson") == "ndjson"

    def test_detect_loader_type_excel(self):
        """Test detection of Excel file type."""
        factory = DataLoaderFactory()

        assert factory._detect_loader_type("test.xlsx") == "xlsx"
        assert factory._detect_loader_type("data.xls") == "xls"

    def test_detect_loader_type_url(self):
        """Test detection from URL."""
        factory = DataLoaderFactory()

        url = "https://example.com/data.csv"
        assert factory._detect_loader_type(url) == "csv"

        url = "http://example.com/path/file.parquet"
        assert factory._detect_loader_type(url) == "parquet"

    def test_detect_loader_type_no_extension(self):
        """Test error when no extension found."""
        factory = DataLoaderFactory()

        with pytest.raises(DataValidationError, match="Cannot detect file type"):
            factory._detect_loader_type("filename_without_extension")

    def test_detect_loader_type_unsupported(self):
        """Test error for unsupported extension."""
        factory = DataLoaderFactory()

        with pytest.raises(DataValidationError, match="Unsupported file extension"):
            factory._detect_loader_type("test.unsupported")

    def test_create_loader_explicit_type(self):
        """Test creating loader with explicit type."""
        factory = DataLoaderFactory()

        loader = factory.create_loader("test.csv", loader_type="csv")
        assert loader is not None
        assert hasattr(loader, "load")

    def test_create_loader_auto_detect(self):
        """Test creating loader with auto-detection."""
        factory = DataLoaderFactory()

        loader = factory.create_loader("test.csv")
        assert loader is not None
        assert hasattr(loader, "load")

    def test_create_loader_unsupported_type(self):
        """Test error for unsupported loader type."""
        factory = DataLoaderFactory()

        with pytest.raises(DataValidationError, match="Unsupported data format"):
            factory.create_loader("test.csv", loader_type="unsupported")

    def test_create_loader_with_config(self):
        """Test creating loader with custom configuration."""
        factory = DataLoaderFactory()

        loader = factory.create_loader("test.csv", delimiter=";", encoding="latin-1")
        assert loader is not None

    @patch("monorepo.infrastructure.data_loaders.data_loader_factory.CSVLoader")
    def test_load_data_success(self, mock_csv_loader):
        """Test successful data loading."""
        factory = DataLoaderFactory()

        # Mock loader instance
        mock_loader = Mock()
        mock_dataset = Dataset(
            name="test", data=pd.DataFrame({"col1": [1, 2, 3]}), metadata={}
        )
        mock_loader.load.return_value = mock_dataset
        mock_csv_loader.return_value = mock_loader

        result = factory.load_data("test.csv", name="test_dataset")

        assert result == mock_dataset
        mock_loader.load.assert_called_once()

    def test_validate_source_valid(self):
        """Test validation of valid source."""
        factory = DataLoaderFactory()

        with patch.object(factory, "create_loader") as mock_create:
            mock_loader = Mock()
            mock_loader.validate.return_value = True
            mock_create.return_value = mock_loader

            assert factory.validate_source("test.csv") is True

    def test_validate_source_invalid(self):
        """Test validation of invalid source."""
        factory = DataLoaderFactory()

        with patch.object(factory, "create_loader") as mock_create:
            mock_create.side_effect = Exception("Creation failed")

            assert factory.validate_source("test.csv") is False

    def test_get_loader_info(self):
        """Test getting loader information."""
        factory = DataLoaderFactory()

        info = factory.get_loader_info("csv")

        assert "class_name" in info
        assert "supported_formats" in info
        assert "default_config" in info
        assert "description" in info

    def test_get_loader_info_unknown(self):
        """Test error for unknown loader type."""
        factory = DataLoaderFactory()

        with pytest.raises(ValueError, match="Unknown loader type"):
            factory.get_loader_info("unknown")

    def test_register_loader(self):
        """Test registering custom loader."""
        factory = DataLoaderFactory()

        class CustomLoader:
            def __init__(self, **kwargs):
                self.config = kwargs

            def load(self, source, name=None, **kwargs):
                return Dataset(name="custom", data=pd.DataFrame(), metadata={})

            @property
            def supported_formats(self):
                return ["custom"]

        factory.register_loader(
            extensions=["custom"],
            loader_class=CustomLoader,
            default_config={"option": "value"},
        )

        assert "custom" in factory.get_supported_formats()
        loader = factory.create_loader("test.custom")
        assert isinstance(loader, CustomLoader)


class TestSmartDataLoader:
    """Test cases for SmartDataLoader."""

    def test_init_default(self):
        """Test default initialization."""
        loader = SmartDataLoader()

        assert loader.factory is not None
        assert loader.auto_optimize is True
        assert loader.memory_threshold_mb == 1000.0

    def test_init_custom_factory(self):
        """Test initialization with custom factory."""
        custom_factory = DataLoaderFactory()
        loader = SmartDataLoader(factory=custom_factory)

        assert loader.factory is custom_factory

    @patch("pathlib.Path.exists")
    @patch("pathlib.Path.stat")
    def test_load_small_file(self, mock_stat, mock_exists):
        """Test loading small file (standard strategy)."""
        mock_exists.return_value = True
        mock_stat.return_value = Mock(st_size=500 * 1024 * 1024)  # 500 MB

        loader = SmartDataLoader(memory_threshold_mb=1000.0)

        with patch.object(loader.factory, "create_loader") as mock_create:
            mock_data_loader = Mock()
            mock_dataset = Dataset(
                name="test", data=pd.DataFrame({"col1": [1, 2, 3]}), metadata={}
            )
            mock_data_loader.load.return_value = mock_dataset
            mock_create.return_value = mock_data_loader

            result = loader.load("test.csv")

            assert result == mock_dataset
            mock_data_loader.load.assert_called_once()

    @patch("pathlib.Path.exists")
    @patch("pathlib.Path.stat")
    def test_load_large_file(self, mock_stat, mock_exists):
        """Test loading large file (optimized strategy)."""
        mock_exists.return_value = True
        mock_stat.return_value = Mock(st_size=1500 * 1024 * 1024)  # 1500 MB

        loader = SmartDataLoader(memory_threshold_mb=1000.0)

        with patch.object(loader.factory, "create_loader") as mock_create:
            mock_data_loader = Mock()
            mock_dataset = Dataset(
                name="test", data=pd.DataFrame({"col1": [1, 2, 3]}), metadata={}
            )
            mock_data_loader.load.return_value = mock_dataset
            mock_create.return_value = mock_data_loader

            result = loader.load("test.csv")

            assert result == mock_dataset
            # Should be called with optimized kwargs
            mock_data_loader.load.assert_called()

    @patch("pathlib.Path.exists")
    def test_load_file_not_found(self, mock_exists):
        """Test error when file doesn't exist."""
        mock_exists.return_value = False

        loader = SmartDataLoader()

        with pytest.raises(DataValidationError, match="Source file not found"):
            loader.load("nonexistent.csv")

    @patch("pathlib.Path.exists")
    @patch("pathlib.Path.stat")
    def test_load_multiple_sources(self, mock_stat, mock_exists):
        """Test loading multiple sources."""
        mock_exists.return_value = True
        mock_stat.return_value = Mock(st_size=100 * 1024 * 1024)  # 100 MB

        loader = SmartDataLoader()

        with patch.object(loader, "load") as mock_load:
            dataset1 = Dataset(
                name="dataset1", data=pd.DataFrame({"a": [1, 2]}), metadata={}
            )
            dataset2 = Dataset(
                name="dataset2", data=pd.DataFrame({"b": [3, 4]}), metadata={}
            )
            mock_load.side_effect = [dataset1, dataset2]

            result = loader.load_multiple(["file1.csv", "file2.csv"])

            assert len(result) == 2
            assert result[0] == dataset1
            assert result[1] == dataset2

    @patch("pathlib.Path.exists")
    @patch("pathlib.Path.stat")
    def test_load_multiple_sources_combine(self, mock_stat, mock_exists):
        """Test loading and combining multiple sources."""
        mock_exists.return_value = True
        mock_stat.return_value = Mock(st_size=100 * 1024 * 1024)  # 100 MB

        loader = SmartDataLoader()

        with patch.object(loader, "load") as mock_load:
            dataset1 = Dataset(
                name="dataset1", data=pd.DataFrame({"a": [1, 2]}), metadata={}
            )
            dataset2 = Dataset(
                name="dataset2", data=pd.DataFrame({"a": [3, 4]}), metadata={}
            )
            mock_load.side_effect = [dataset1, dataset2]

            result = loader.load_multiple(["file1.csv", "file2.csv"], combine=True)

            assert isinstance(result, Dataset)
            assert result.name == "combined_dataset"
            assert len(result.data) == 4  # Combined rows

    @patch("pathlib.Path.exists")
    @patch("pathlib.Path.stat")
    def test_estimate_load_time(self, mock_stat, mock_exists):
        """Test load time estimation."""
        mock_exists.return_value = True
        mock_stat.return_value = Mock(st_size=100 * 1024 * 1024)  # 100 MB

        loader = SmartDataLoader()

        with patch.object(loader.factory, "create_loader") as mock_create:
            mock_data_loader = Mock()
            mock_data_loader.estimate_size.return_value = {
                "file_size_mb": 100,
                "estimated_rows": 1000000,
            }
            mock_create.return_value = mock_data_loader

            result = loader.estimate_load_time("test.csv")

            assert "estimated_load_time_seconds" in result
            assert "recommended_batch_loading" in result
            assert "file_extension" in result

    @patch("pathlib.Path.exists")
    @patch("pathlib.Path.stat")
    def test_estimate_load_time_json_adjustment(self, mock_stat, mock_exists):
        """Test load time estimation with JSON format adjustment."""
        mock_exists.return_value = True
        mock_stat.return_value = Mock(st_size=100 * 1024 * 1024)  # 100 MB

        loader = SmartDataLoader()

        with patch.object(loader.factory, "create_loader") as mock_create:
            mock_data_loader = Mock()
            mock_data_loader.estimate_size.return_value = {"file_size_mb": 100}
            mock_create.return_value = mock_data_loader

            result = loader.estimate_load_time("test.json")

            # JSON should take longer (2x multiplier)
            assert result["estimated_load_time_seconds"] == 4.0  # (100/50)*2

    @patch("pathlib.Path.exists")
    @patch("pathlib.Path.stat")
    def test_estimate_load_time_excel_adjustment(self, mock_stat, mock_exists):
        """Test load time estimation with Excel format adjustment."""
        mock_exists.return_value = True
        mock_stat.return_value = Mock(st_size=100 * 1024 * 1024)  # 100 MB

        loader = SmartDataLoader()

        with patch.object(loader.factory, "create_loader") as mock_create:
            mock_data_loader = Mock()
            mock_data_loader.estimate_size.return_value = {"file_size_mb": 100}
            mock_create.return_value = mock_data_loader

            result = loader.estimate_load_time("test.xlsx")

            # Excel should take longer (3x multiplier)
            assert result["estimated_load_time_seconds"] == 6.0  # (100/50)*3

    @patch("pathlib.Path.exists")
    def test_estimate_load_time_file_not_found(self, mock_exists):
        """Test error when estimating load time for non-existent file."""
        mock_exists.return_value = False

        loader = SmartDataLoader()

        with pytest.raises(DataValidationError, match="Source file not found"):
            loader.estimate_load_time("nonexistent.csv")

    def test_combine_datasets_empty(self):
        """Test error when combining empty dataset list."""
        loader = SmartDataLoader()

        with pytest.raises(ValueError, match="No datasets to combine"):
            loader._combine_datasets([])

    def test_combine_datasets_single(self):
        """Test combining single dataset returns same dataset."""
        loader = SmartDataLoader()

        dataset = Dataset(name="test", data=pd.DataFrame({"a": [1, 2]}), metadata={})
        result = loader._combine_datasets([dataset])

        assert result is dataset

    def test_combine_datasets_multiple(self):
        """Test combining multiple datasets."""
        loader = SmartDataLoader()

        dataset1 = Dataset(
            name="dataset1",
            data=pd.DataFrame({"a": [1, 2], "b": [3, 4]}),
            metadata={"source": "file1"},
        )
        dataset2 = Dataset(
            name="dataset2",
            data=pd.DataFrame({"a": [5, 6], "b": [7, 8]}),
            metadata={"source": "file2"},
        )

        result = loader._combine_datasets([dataset1, dataset2])

        assert result.name == "combined_dataset"
        assert len(result.data) == 4
        assert list(result.data["a"]) == [1, 2, 5, 6]
        assert "combined_from" in result.metadata
        assert result.metadata["combined_from"] == ["dataset1", "dataset2"]

    def test_combine_datasets_with_target_column(self):
        """Test combining datasets preserves target column."""
        loader = SmartDataLoader()

        dataset1 = Dataset(
            name="dataset1",
            data=pd.DataFrame({"a": [1, 2], "target": [0, 1]}),
            target_column="target",
            metadata={},
        )
        dataset2 = Dataset(
            name="dataset2",
            data=pd.DataFrame({"a": [3, 4], "target": [1, 0]}),
            metadata={},
        )

        result = loader._combine_datasets([dataset1, dataset2])

        assert result.target_column == "target"
        assert "target" in result.data.columns
