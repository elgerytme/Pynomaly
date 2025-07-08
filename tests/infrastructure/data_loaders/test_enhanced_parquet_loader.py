"""Test cases for enhanced Parquet data loader."""

from pathlib import Path
from unittest.mock import Mock, patch

import pandas as pd
import pytest
from pynomaly.domain.entities import Dataset
from pynomaly.domain.exceptions import DataValidationError
from pynomaly.infrastructure.data_loaders.enhanced_parquet_loader import (
    EnhancedParquetLoader,
)


class TestEnhancedParquetLoader:
    """Test cases for EnhancedParquetLoader."""

    def test_init_default(self):
        """Test default initialization."""
        loader = EnhancedParquetLoader()

        assert loader.engine == "pyarrow"
        assert loader.use_memory_map is True
        assert loader.columns is None
        assert loader.filters is None
        assert loader.use_pandas_metadata is True
        assert loader.validate_schema is True

    def test_init_custom_config(self):
        """Test initialization with custom configuration."""
        loader = EnhancedParquetLoader(
            engine="fastparquet",
            use_memory_map=False,
            columns=["col1", "col2"],
            validate_schema=False,
        )

        assert loader.engine == "fastparquet"
        assert loader.use_memory_map is False
        assert loader.columns == ["col1", "col2"]
        assert loader.validate_schema is False

    def test_init_pyarrow_missing(self):
        """Test error when PyArrow is missing."""
        with patch(
            "importlib.import_module",
            side_effect=ImportError("No module named 'pyarrow'"),
        ):
            with pytest.raises(ImportError, match="PyArrow is required"):
                EnhancedParquetLoader(engine="pyarrow")

    def test_init_fastparquet_missing(self):
        """Test error when fastparquet is missing."""
        with patch(
            "importlib.import_module",
            side_effect=ImportError("No module named 'fastparquet'"),
        ):
            with pytest.raises(ImportError, match="fastparquet is required"):
                EnhancedParquetLoader(engine="fastparquet")

    def test_supported_formats(self):
        """Test supported file formats."""
        loader = EnhancedParquetLoader()

        expected_formats = ["parquet", "pq"]
        assert loader.supported_formats == expected_formats

    @patch("pandas.read_parquet")
    @patch("pathlib.Path.exists")
    def test_load_success(self, mock_exists, mock_read_parquet):
        """Test successful Parquet file loading."""
        mock_exists.return_value = True

        # Mock data
        mock_df = pd.DataFrame(
            {
                "feature1": [1.0, 2.0, 3.0],
                "feature2": [4.0, 5.0, 6.0],
                "target": [0, 1, 0],
            }
        )
        mock_read_parquet.return_value = mock_df

        loader = EnhancedParquetLoader()

        # Mock validation and metadata extraction
        with (
            patch.object(loader, "validate", return_value=True),
            patch.object(loader, "_extract_metadata", return_value={}),
            patch.object(loader, "_get_file_size_mb", return_value=10.5),
        ):
            result = loader.load(
                "test.parquet", name="test_dataset", target_column="target"
            )

        assert isinstance(result, Dataset)
        assert result.name == "test_dataset"
        assert result.target_column == "target"
        assert len(result.data) == 3
        assert "loader" in result.metadata
        assert result.metadata["loader"] == "EnhancedParquetLoader"

    @patch("pandas.read_parquet")
    @patch("pathlib.Path.exists")
    def test_load_empty_file(self, mock_exists, mock_read_parquet):
        """Test error when Parquet file is empty."""
        mock_exists.return_value = True
        mock_read_parquet.return_value = pd.DataFrame()  # Empty DataFrame

        loader = EnhancedParquetLoader()

        with patch.object(loader, "validate", return_value=True):
            with pytest.raises(DataValidationError, match="Parquet file is empty"):
                loader.load("test.parquet")

    @patch("pandas.read_parquet")
    @patch("pathlib.Path.exists")
    def test_load_target_column_not_found(self, mock_exists, mock_read_parquet):
        """Test error when target column not found."""
        mock_exists.return_value = True
        mock_df = pd.DataFrame({"col1": [1, 2, 3], "col2": [4, 5, 6]})
        mock_read_parquet.return_value = mock_df

        loader = EnhancedParquetLoader()

        with (
            patch.object(loader, "validate", return_value=True),
            patch.object(loader, "_extract_metadata", return_value={}),
        ):
            with pytest.raises(
                DataValidationError, match="Target column 'missing' not found"
            ):
                loader.load("test.parquet", target_column="missing")

    def test_load_invalid_source(self):
        """Test error with invalid source."""
        loader = EnhancedParquetLoader()

        with patch.object(loader, "validate", return_value=False):
            with pytest.raises(DataValidationError, match="Invalid Parquet source"):
                loader.load("invalid.parquet")

    @patch("pandas.read_parquet")
    @patch("pathlib.Path.exists")
    def test_load_with_custom_options(self, mock_exists, mock_read_parquet):
        """Test loading with custom read options."""
        mock_exists.return_value = True
        mock_df = pd.DataFrame({"col1": [1, 2, 3]})
        mock_read_parquet.return_value = mock_df

        loader = EnhancedParquetLoader(columns=["col1"])

        with (
            patch.object(loader, "validate", return_value=True),
            patch.object(loader, "_extract_metadata", return_value={}),
            patch.object(loader, "_get_file_size_mb", return_value=1.0),
        ):
            result = loader.load("test.parquet", memory_map=False, filters=[])

        # Verify pandas.read_parquet was called with correct arguments
        mock_read_parquet.assert_called_once()
        call_kwargs = mock_read_parquet.call_args[1]
        assert call_kwargs["engine"] == "pyarrow"
        assert call_kwargs["columns"] == ["col1"]
        assert call_kwargs["memory_map"] is False

    @patch("pathlib.Path.exists")
    @patch("pathlib.Path.is_file")
    @patch("pathlib.Path.suffix")
    def test_validate_file_exists(self, mock_suffix, mock_is_file, mock_exists):
        """Test validation of existing Parquet file."""
        mock_exists.return_value = True
        mock_is_file.return_value = True
        mock_suffix.lower.return_value = ".parquet"

        loader = EnhancedParquetLoader()

        with patch("pyarrow.parquet.read_metadata") as mock_read_metadata:
            mock_read_metadata.return_value = Mock()  # Valid metadata

            assert loader.validate("test.parquet") is True

    @patch("pathlib.Path.exists")
    def test_validate_file_not_exists(self, mock_exists):
        """Test validation fails when file doesn't exist."""
        mock_exists.return_value = False

        loader = EnhancedParquetLoader()

        assert loader.validate("nonexistent.parquet") is False

    @patch("pathlib.Path.exists")
    @patch("pathlib.Path.is_dir")
    @patch("pathlib.Path.glob")
    def test_validate_directory_with_parquet_files(
        self, mock_glob, mock_is_dir, mock_exists
    ):
        """Test validation of directory containing Parquet files."""
        mock_exists.return_value = True
        mock_is_dir.return_value = True
        mock_glob.side_effect = [
            [Path("file1.parquet"), Path("file2.parquet")],  # *.parquet
            [],  # *.pq
        ]

        loader = EnhancedParquetLoader()

        assert loader.validate("parquet_directory/") is True

    @patch("pathlib.Path.exists")
    @patch("pathlib.Path.is_dir")
    @patch("pathlib.Path.glob")
    def test_validate_directory_no_parquet_files(
        self, mock_glob, mock_is_dir, mock_exists
    ):
        """Test validation fails for directory without Parquet files."""
        mock_exists.return_value = True
        mock_is_dir.return_value = True
        mock_glob.side_effect = [[], []]  # No .parquet or .pq files

        loader = EnhancedParquetLoader()

        assert loader.validate("empty_directory/") is False

    @patch("pathlib.Path.exists")
    @patch("pathlib.Path.is_file")
    @patch("pathlib.Path.suffix")
    def test_validate_wrong_extension(self, mock_suffix, mock_is_file, mock_exists):
        """Test validation fails for wrong file extension."""
        mock_exists.return_value = True
        mock_is_file.return_value = True
        mock_suffix.lower.return_value = ".csv"

        loader = EnhancedParquetLoader()

        assert loader.validate("test.csv") is False

    @patch("pathlib.Path.exists")
    @patch("pathlib.Path.is_file")
    @patch("pathlib.Path.suffix")
    def test_validate_corrupt_parquet(self, mock_suffix, mock_is_file, mock_exists):
        """Test validation fails for corrupt Parquet file."""
        mock_exists.return_value = True
        mock_is_file.return_value = True
        mock_suffix.lower.return_value = ".parquet"

        loader = EnhancedParquetLoader()

        with patch(
            "pyarrow.parquet.read_metadata", side_effect=Exception("Corrupt file")
        ):
            assert loader.validate("corrupt.parquet") is False

    @patch("pyarrow.parquet.ParquetFile")
    def test_load_batch_pyarrow(self, mock_parquet_file):
        """Test batch loading with PyArrow engine."""
        # Mock ParquetFile
        mock_file = Mock()
        mock_file.num_row_groups = 4

        # Mock table read from row groups
        mock_table1 = Mock()
        mock_df1 = pd.DataFrame({"col1": [1, 2], "col2": [3, 4]})
        mock_table1.to_pandas.return_value = mock_df1

        mock_table2 = Mock()
        mock_df2 = pd.DataFrame({"col1": [5, 6], "col2": [7, 8]})
        mock_table2.to_pandas.return_value = mock_df2

        mock_file.read_row_groups.side_effect = [mock_table1, mock_table2]
        mock_parquet_file.return_value = mock_file

        loader = EnhancedParquetLoader()

        with patch.object(loader, "validate", return_value=True):
            batches = list(loader.load_batch("test.parquet", batch_size=1000))

        assert len(batches) == 2
        assert isinstance(batches[0], Dataset)
        assert batches[0].name == "test_batch_0"
        assert len(batches[0].data) == 2
        assert batches[1].name == "test_batch_2"

    @patch("pandas.read_parquet")
    def test_load_batch_fallback(self, mock_read_parquet):
        """Test batch loading fallback for non-PyArrow engines."""
        # Mock full DataFrame
        mock_df = pd.DataFrame({"col1": list(range(10)), "col2": list(range(10, 20))})
        mock_read_parquet.return_value = mock_df

        loader = EnhancedParquetLoader(engine="fastparquet")

        with patch.object(loader, "validate", return_value=True):
            batches = list(loader.load_batch("test.parquet", batch_size=4))

        assert len(batches) == 3  # 10 rows / 4 batch_size = 3 batches
        assert len(batches[0].data) == 4
        assert len(batches[1].data) == 4
        assert len(batches[2].data) == 2  # Remaining rows

    def test_load_batch_invalid_file(self):
        """Test error when batch loading invalid file."""
        loader = EnhancedParquetLoader()

        with patch.object(loader, "validate", return_value=False):
            with pytest.raises(DataValidationError, match="Invalid Parquet file"):
                list(loader.load_batch("invalid.parquet", batch_size=1000))

    @patch("pyarrow.parquet.ParquetFile")
    def test_estimate_size_pyarrow(self, mock_parquet_file):
        """Test size estimation with PyArrow engine."""
        # Mock ParquetFile and metadata
        mock_file = Mock()
        mock_metadata = Mock()
        mock_metadata.num_rows = 1000
        mock_metadata.num_columns = 5
        mock_metadata.num_row_groups = 4
        mock_metadata.serialized_size = 50 * 1024 * 1024  # 50 MB
        mock_metadata.format_version = "2.0"

        mock_file.metadata = mock_metadata

        # Mock schema
        mock_schema = Mock()
        mock_field1 = Mock()
        mock_field1.name = "col1"
        mock_field1.type = "int64"
        mock_field2 = Mock()
        mock_field2.name = "col2"
        mock_field2.type = "float64"

        mock_schema.__len__ = lambda self: 2
        mock_schema.field.side_effect = [mock_field1, mock_field2]
        mock_file.schema_arrow = mock_schema

        mock_parquet_file.return_value = mock_file

        loader = EnhancedParquetLoader()

        with (
            patch.object(loader, "validate", return_value=True),
            patch.object(loader, "_get_file_size_mb", return_value=100.0),
        ):
            result = loader.estimate_size("test.parquet")

        assert result["total_rows"] == 1000
        assert result["num_columns"] == 5
        assert result["num_row_groups"] == 4
        assert result["compressed_size_mb"] == 50.0
        assert result["parquet_version"] == "2.0"
        assert "column_types" in result

    def test_estimate_size_fallback(self):
        """Test size estimation fallback for non-PyArrow engines."""
        loader = EnhancedParquetLoader(engine="fastparquet")

        with (
            patch.object(loader, "validate", return_value=True),
            patch.object(loader, "_get_file_size_mb", return_value=100.0),
        ):
            result = loader.estimate_size("test.parquet")

        assert result["file_size_mb"] == 100.0
        assert result["estimated_rows"] == "unknown"
        assert result["engine"] == "fastparquet"

    def test_estimate_size_invalid_file(self):
        """Test size estimation for invalid file."""
        loader = EnhancedParquetLoader()

        with patch.object(loader, "validate", return_value=False):
            with pytest.raises(DataValidationError, match="Invalid Parquet file"):
                loader.estimate_size("invalid.parquet")

    @patch("pyarrow.parquet.ParquetFile")
    def test_get_schema_info_pyarrow(self, mock_parquet_file):
        """Test schema information extraction with PyArrow."""
        # Mock ParquetFile and schema
        mock_file = Mock()
        mock_schema = Mock()

        # Mock fields
        mock_field1 = Mock()
        mock_field1.name = "col1"
        mock_field1.type = "int64"
        mock_field1.nullable = True
        mock_field1.metadata = {"key": "value"}

        mock_field2 = Mock()
        mock_field2.name = "col2"
        mock_field2.type = "float64"
        mock_field2.nullable = False
        mock_field2.metadata = None

        mock_schema.__len__ = lambda self: 2
        mock_schema.field.side_effect = [mock_field1, mock_field2]
        mock_schema.metadata = {"schema_key": "schema_value"}
        mock_schema.pandas_metadata = {"pandas_version": "1.0"}

        mock_file.schema_arrow = mock_schema
        mock_parquet_file.return_value = mock_file

        loader = EnhancedParquetLoader()

        with patch.object(loader, "validate", return_value=True):
            result = loader.get_schema_info("test.parquet")

        assert result["num_columns"] == 2
        assert len(result["columns"]) == 2
        assert result["columns"][0]["name"] == "col1"
        assert result["columns"][0]["type"] == "int64"
        assert result["columns"][0]["nullable"] is True
        assert result["schema_metadata"] == {"schema_key": "schema_value"}
        assert result["pandas_metadata"] == {"pandas_version": "1.0"}

    @patch("pandas.read_parquet")
    def test_get_schema_info_fallback(self, mock_read_parquet):
        """Test schema information fallback for non-PyArrow engines."""
        mock_df = pd.DataFrame({"col1": [1, 2, 3], "col2": [1.0, 2.0, 3.0]})
        mock_read_parquet.return_value = mock_df

        loader = EnhancedParquetLoader(engine="fastparquet")

        with patch.object(loader, "validate", return_value=True):
            result = loader.get_schema_info("test.parquet")

        assert result["num_columns"] == 2
        assert len(result["columns"]) == 2
        assert result["columns"][0]["name"] == "col1"
        assert result["columns"][0]["type"] == "int64"
        assert result["columns"][0]["nullable"] is True  # Default assumption

    def test_get_schema_info_invalid_file(self):
        """Test schema information for invalid file."""
        loader = EnhancedParquetLoader()

        with patch.object(loader, "validate", return_value=False):
            with pytest.raises(DataValidationError, match="Invalid Parquet file"):
                loader.get_schema_info("invalid.parquet")

    @patch("pyarrow.parquet.ParquetFile")
    def test_extract_metadata_success(self, mock_parquet_file):
        """Test successful metadata extraction."""
        # Mock ParquetFile and metadata
        mock_file = Mock()
        mock_metadata = Mock()
        mock_metadata.num_rows = 1000
        mock_metadata.num_columns = 5
        mock_metadata.num_row_groups = 4
        mock_metadata.format_version = "2.0"
        mock_metadata.created_by = "pyarrow version 10.0.0"
        mock_metadata.serialized_size = 50000000

        mock_file.metadata = mock_metadata

        # Mock schema
        mock_schema = Mock()
        mock_schema.names = ["col1", "col2", "col3"]
        mock_field = Mock()
        mock_field.type = "int64"
        mock_schema.field.return_value = mock_field
        mock_schema.__len__ = lambda self: 3
        mock_file.schema_arrow = mock_schema

        mock_parquet_file.return_value = mock_file

        loader = EnhancedParquetLoader()
        result = loader._extract_metadata(Path("test.parquet"))

        assert result["num_rows"] == 1000
        assert result["num_columns"] == 5
        assert result["num_row_groups"] == 4
        assert result["format_version"] == "2.0"
        assert result["created_by"] == "pyarrow version 10.0.0"
        assert result["serialized_size"] == 50000000
        assert "schema" in result
        assert result["schema"]["names"] == ["col1", "col2", "col3"]

    def test_extract_metadata_error(self):
        """Test metadata extraction with error."""
        loader = EnhancedParquetLoader()

        with patch("pyarrow.parquet.ParquetFile", side_effect=Exception("Read error")):
            result = loader._extract_metadata(Path("test.parquet"))

        assert "extraction_error" in result
        assert "Read error" in result["extraction_error"]

    @patch("pathlib.Path.stat")
    def test_get_file_size_mb_file(self, mock_stat):
        """Test file size calculation for single file."""
        mock_stat.return_value = Mock(st_size=10 * 1024 * 1024)  # 10 MB

        loader = EnhancedParquetLoader()

        with (
            patch("pathlib.Path.is_file", return_value=True),
            patch("pathlib.Path.is_dir", return_value=False),
        ):
            size = loader._get_file_size_mb(Path("test.parquet"))

        assert size == 10.0

    @patch("pathlib.Path.rglob")
    def test_get_file_size_mb_directory(self, mock_rglob):
        """Test file size calculation for directory."""
        # Mock parquet files in directory
        file1 = Mock()
        file1.stat.return_value.st_size = 5 * 1024 * 1024  # 5 MB
        file2 = Mock()
        file2.stat.return_value.st_size = 3 * 1024 * 1024  # 3 MB

        mock_rglob.side_effect = [[file1, file2], []]  # *.parquet files  # *.pq files

        loader = EnhancedParquetLoader()

        with (
            patch("pathlib.Path.is_file", return_value=False),
            patch("pathlib.Path.is_dir", return_value=True),
        ):
            size = loader._get_file_size_mb(Path("parquet_dir/"))

        assert size == 8.0  # 5 + 3 MB

    def test_get_file_size_mb_invalid_path(self):
        """Test file size calculation for invalid path."""
        loader = EnhancedParquetLoader()

        with (
            patch("pathlib.Path.is_file", return_value=False),
            patch("pathlib.Path.is_dir", return_value=False),
        ):
            size = loader._get_file_size_mb(Path("nonexistent"))

        assert size == 0.0
