"""
Tests for the import protocol implementation.

This module tests the ImportProtocol to ensure proper contract enforcement
and runtime behavior checking for all import implementations.
"""

from pathlib import Path
from typing import Any
from uuid import uuid4

import numpy as np
import pandas as pd
import pytest

from monorepo.domain.entities import Dataset
from monorepo.shared.protocols.import_protocol import ImportProtocol


class MockDataset:
    """Mock Dataset for testing."""

    def __init__(self, data: pd.DataFrame, name: str = "test_dataset"):
        self.id = uuid4()
        self.name = name
        self.data = data
        self.metadata = {
            "source": "test",
            "rows": len(data),
            "columns": len(data.columns) if not data.empty else 0,
        }


class MockImporter(ImportProtocol):
    """Mock implementation of ImportProtocol for testing."""

    def __init__(self):
        self.supported_formats = [".csv", ".xlsx", ".json", ".parquet"]
        self.import_count = 0
        self.last_import_path = None
        self.last_import_options = None
        self.should_fail_validation = False
        self.should_fail_import = False
        self.mock_data = None

    def import_dataset(
        self, file_path: str | Path, options: dict[str, Any] = None
    ) -> Dataset:
        """Mock import implementation."""
        if self.should_fail_import:
            raise ValueError("Import failed as requested")

        self.import_count += 1
        self.last_import_path = str(file_path)
        self.last_import_options = options

        # Create mock data based on file extension
        if self.mock_data is not None:
            data = self.mock_data
        else:
            # Default mock data
            data = pd.DataFrame(
                {
                    "feature_1": [1.0, 2.0, 3.0, 4.0, 5.0],
                    "feature_2": [0.5, 1.5, 2.5, 3.5, 4.5],
                    "feature_3": [10, 20, 30, 40, 50],
                }
            )

        # Apply options if provided
        if options:
            if "nrows" in options:
                data = data.head(options["nrows"])
            if "columns" in options:
                available_cols = [
                    col for col in options["columns"] if col in data.columns
                ]
                if available_cols:
                    data = data[available_cols]

        return MockDataset(data, f"imported_from_{Path(file_path).stem}")

    def get_supported_formats(self) -> list[str]:
        """Return supported formats."""
        return self.supported_formats

    def validate_file(self, file_path: str | Path) -> bool:
        """Validate file path."""
        if self.should_fail_validation:
            return False

        path = Path(file_path)

        # Check extension
        if path.suffix not in self.supported_formats:
            return False

        # Mock file existence check (always pass for testing)
        return True


class TestImportProtocol:
    """Test suite for ImportProtocol contract enforcement."""

    @pytest.fixture
    def mock_importer(self):
        """Create mock importer for testing."""
        return MockImporter()

    @pytest.fixture
    def sample_csv_data(self):
        """Create sample CSV data."""
        return pd.DataFrame(
            {
                "timestamp": pd.date_range("2025-01-01", periods=100, freq="H"),
                "sensor_1": np.random.randn(100),
                "sensor_2": np.random.randn(100),
                "sensor_3": np.random.randn(100),
                "label": np.random.choice([0, 1], 100),
            }
        )

    @pytest.fixture
    def import_options(self):
        """Create sample import options."""
        return {
            "encoding": "utf-8",
            "delimiter": ",",
            "header": 0,
            "nrows": 50,
            "columns": ["sensor_1", "sensor_2", "sensor_3"],
        }

    def test_protocol_compliance(self, mock_importer):
        """Test that mock importer implements the protocol correctly."""
        assert isinstance(mock_importer, ImportProtocol)

        # Check that all required methods exist
        assert hasattr(mock_importer, "import_dataset")
        assert hasattr(mock_importer, "get_supported_formats")
        assert hasattr(mock_importer, "validate_file")

        # Check method signatures
        assert callable(mock_importer.import_dataset)
        assert callable(mock_importer.get_supported_formats)
        assert callable(mock_importer.validate_file)

    def test_import_dataset_basic_functionality(self, mock_importer, import_options):
        """Test basic import dataset functionality."""
        file_path = "/data/test_dataset.csv"

        # Test successful import
        dataset = mock_importer.import_dataset(file_path, import_options)

        # Verify return type and structure
        assert hasattr(dataset, "data")
        assert hasattr(dataset, "name")
        assert hasattr(dataset, "metadata")
        assert isinstance(dataset.data, pd.DataFrame)
        assert not dataset.data.empty

        # Verify importer state
        assert mock_importer.import_count == 1
        assert mock_importer.last_import_path == file_path
        assert mock_importer.last_import_options == import_options

    def test_import_dataset_different_formats(self, mock_importer):
        """Test import with different file formats."""
        formats = [".csv", ".xlsx", ".json", ".parquet"]

        for format_ext in formats:
            file_path = f"/data/test_dataset{format_ext}"
            dataset = mock_importer.import_dataset(file_path)

            assert dataset.name == "imported_from_test_dataset"
            assert isinstance(dataset.data, pd.DataFrame)

    def test_import_dataset_without_options(self, mock_importer):
        """Test import without providing options."""
        file_path = "/data/test_dataset.csv"

        # Should work with None options
        dataset = mock_importer.import_dataset(file_path, None)

        assert isinstance(dataset.data, pd.DataFrame)
        assert mock_importer.last_import_options is None

    def test_import_dataset_with_path_object(self, mock_importer):
        """Test import with Path object instead of string."""
        file_path = Path("/data/test_dataset.csv")

        dataset = mock_importer.import_dataset(file_path)

        assert isinstance(dataset.data, pd.DataFrame)
        assert mock_importer.last_import_path == str(file_path)

    def test_import_dataset_error_handling(self, mock_importer):
        """Test import error handling."""
        mock_importer.should_fail_import = True

        with pytest.raises(ValueError, match="Import failed as requested"):
            mock_importer.import_dataset("/data/fail.csv")

    def test_import_dataset_with_row_limit(self, mock_importer, sample_csv_data):
        """Test import with row limit option."""
        mock_importer.mock_data = sample_csv_data

        options = {"nrows": 10}
        dataset = mock_importer.import_dataset("/data/limited.csv", options)

        assert len(dataset.data) == 10
        assert mock_importer.last_import_options == options

    def test_import_dataset_with_column_selection(self, mock_importer, sample_csv_data):
        """Test import with column selection."""
        mock_importer.mock_data = sample_csv_data

        options = {"columns": ["sensor_1", "sensor_2"]}
        dataset = mock_importer.import_dataset("/data/selected_cols.csv", options)

        assert list(dataset.data.columns) == ["sensor_1", "sensor_2"]

    def test_import_dataset_with_invalid_columns(self, mock_importer, sample_csv_data):
        """Test import with invalid column names."""
        mock_importer.mock_data = sample_csv_data

        options = {"columns": ["nonexistent_col", "sensor_1"]}
        dataset = mock_importer.import_dataset("/data/invalid_cols.csv", options)

        # Should only include valid columns
        assert list(dataset.data.columns) == ["sensor_1"]

    def test_get_supported_formats(self, mock_importer):
        """Test supported formats retrieval."""
        formats = mock_importer.get_supported_formats()

        assert isinstance(formats, list)
        assert len(formats) > 0
        assert all(isinstance(fmt, str) for fmt in formats)
        assert all(fmt.startswith(".") for fmt in formats)

        # Check expected formats
        expected_formats = [".csv", ".xlsx", ".json", ".parquet"]
        for fmt in expected_formats:
            assert fmt in formats

    def test_validate_file_success(self, mock_importer):
        """Test successful file validation."""
        valid_paths = [
            "/data/test.csv",
            Path("/data/test.xlsx"),
            "/input/data.json",
            Path("/files/dataset.parquet"),
        ]

        for path in valid_paths:
            assert mock_importer.validate_file(path) is True

    def test_validate_file_failure(self, mock_importer):
        """Test file validation failure."""
        invalid_paths = [
            "/data/test.txt",  # Unsupported format
            "/data/test.pdf",  # Unsupported format
            "/data/test",  # No extension
            Path("/data/test.doc"),  # Unsupported format
        ]

        for path in invalid_paths:
            assert mock_importer.validate_file(path) is False

    def test_validate_file_forced_failure(self, mock_importer):
        """Test file validation when forced to fail."""
        mock_importer.should_fail_validation = True

        # Even valid formats should fail
        assert mock_importer.validate_file("/data/test.csv") is False
        assert mock_importer.validate_file(Path("/data/test.xlsx")) is False

    def test_import_empty_file_handling(self, mock_importer):
        """Test import with empty data."""
        # Set empty mock data
        mock_importer.mock_data = pd.DataFrame()

        dataset = mock_importer.import_dataset("/data/empty.csv")

        assert isinstance(dataset.data, pd.DataFrame)
        assert dataset.data.empty
        assert dataset.metadata["rows"] == 0
        assert dataset.metadata["columns"] == 0

    def test_multiple_imports(self, mock_importer):
        """Test multiple consecutive imports."""
        file_paths = [
            "/data/dataset1.csv",
            "/data/dataset2.xlsx",
            "/data/dataset3.json",
        ]

        for i, path in enumerate(file_paths, 1):
            dataset = mock_importer.import_dataset(path)

            assert isinstance(dataset.data, pd.DataFrame)
            assert mock_importer.import_count == i
            assert mock_importer.last_import_path == path

    def test_import_with_complex_options(self, mock_importer, sample_csv_data):
        """Test import with complex options."""
        mock_importer.mock_data = sample_csv_data

        complex_options = {
            "encoding": "utf-16",
            "delimiter": ";",
            "header": 0,
            "nrows": 25,
            "columns": ["timestamp", "sensor_1", "sensor_2"],
            "parse_dates": ["timestamp"],
            "index_col": 0,
        }

        dataset = mock_importer.import_dataset("/data/complex.csv", complex_options)

        # Verify options were applied
        assert len(dataset.data) == 25
        assert "timestamp" in dataset.data.columns
        assert "sensor_1" in dataset.data.columns
        assert "sensor_2" in dataset.data.columns
        assert mock_importer.last_import_options == complex_options

    def test_import_large_dataset(self, mock_importer):
        """Test import with large dataset."""
        # Create large mock dataset
        large_data = pd.DataFrame(
            {f"feature_{i}": np.random.randn(10000) for i in range(20)}
        )
        mock_importer.mock_data = large_data

        dataset = mock_importer.import_dataset("/data/large_dataset.csv")

        assert len(dataset.data) == 10000
        assert len(dataset.data.columns) == 20
        assert dataset.metadata["rows"] == 10000
        assert dataset.metadata["columns"] == 20


class IncompleteImporter:
    """Incomplete implementation missing required methods."""

    def import_dataset(self, file_path, options=None):
        return MockDataset(pd.DataFrame(), "test")

    # Missing get_supported_formats and validate_file


class TestImportProtocolEnforcement:
    """Test protocol enforcement and runtime checking."""

    def test_incomplete_implementation_detection(self):
        """Test that incomplete implementations are detected."""
        incomplete = IncompleteImporter()

        # Should not be considered a valid implementation
        assert not isinstance(incomplete, ImportProtocol)

    def test_protocol_method_signatures(self):
        """Test that protocol defines correct method signatures."""
        # This test ensures the protocol is properly defined
        methods = ImportProtocol.__abstractmethods__

        expected_methods = {"import_dataset", "get_supported_formats", "validate_file"}
        assert methods == expected_methods

    def test_protocol_runtime_checking(self):
        """Test runtime protocol checking works correctly."""
        mock_importer = MockImporter()

        # Should pass runtime check
        assert isinstance(mock_importer, ImportProtocol)

        # Protocol methods should be callable
        assert callable(mock_importer.import_dataset)
        assert callable(mock_importer.get_supported_formats)
        assert callable(mock_importer.validate_file)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
