"""Tests for infrastructure data loader implementations."""

from __future__ import annotations

import tempfile
from pathlib import Path

import pandas as pd
import pytest
from pynomaly.domain.entities import Dataset
from pynomaly.domain.exceptions import DataValidationError
from pynomaly.infrastructure.data_loaders import CSVLoader, ParquetLoader


class TestCSVLoader:
    """Test CSVLoader implementation."""

    def test_init_default_parameters(self):
        """Test initialization with default parameters."""
        loader = CSVLoader()

        assert loader.delimiter == ","
        assert loader.encoding == "utf-8"
        assert loader.parse_dates is True
        assert loader.low_memory is False

    def test_init_custom_parameters(self):
        """Test initialization with custom parameters."""
        loader = CSVLoader(
            delimiter=";", encoding="latin-1", parse_dates=False, low_memory=True
        )

        assert loader.delimiter == ";"
        assert loader.encoding == "latin-1"
        assert loader.parse_dates is False
        assert loader.low_memory is True

    def test_supported_formats(self):
        """Test supported file formats."""
        loader = CSVLoader()
        formats = loader.supported_formats

        assert "csv" in formats
        assert "tsv" in formats
        assert "txt" in formats
        assert len(formats) >= 3

    def test_load_simple_csv(self):
        """Test loading a simple CSV file."""
        # Create sample CSV data
        csv_data = """feature1,feature2,feature3
1.0,2.0,3.0
4.0,5.0,6.0
7.0,8.0,9.0"""

        # Create temporary file
        with tempfile.NamedTemporaryFile(mode="w", suffix=".csv", delete=False) as f:
            f.write(csv_data)
            temp_path = f.name

        try:
            loader = CSVLoader()
            dataset = loader.load(temp_path, name="test_csv")

            # Verify dataset properties
            assert dataset.name == "test_csv"
            assert dataset.n_samples == 3
            assert dataset.n_features == 3
            assert list(dataset.data.columns) == ["feature1", "feature2", "feature3"]

            # Verify data content
            expected_data = pd.DataFrame(
                {
                    "feature1": [1.0, 4.0, 7.0],
                    "feature2": [2.0, 5.0, 8.0],
                    "feature3": [3.0, 6.0, 9.0],
                }
            )
            pd.testing.assert_frame_equal(dataset.data, expected_data)

        finally:
            # Clean up
            Path(temp_path).unlink()

    def test_load_csv_with_custom_delimiter(self):
        """Test loading CSV with custom delimiter."""
        # Create sample TSV data
        tsv_data = """feature1\tfeature2\tfeature3
1.0\t2.0\t3.0
4.0\t5.0\t6.0"""

        with tempfile.NamedTemporaryFile(mode="w", suffix=".tsv", delete=False) as f:
            f.write(tsv_data)
            temp_path = f.name

        try:
            loader = CSVLoader(delimiter="\t")
            dataset = loader.load(temp_path, name="test_tsv")

            assert dataset.n_samples == 2
            assert dataset.n_features == 3
            assert list(dataset.data.columns) == ["feature1", "feature2", "feature3"]

        finally:
            Path(temp_path).unlink()

    def test_load_csv_with_missing_values(self):
        """Test loading CSV with missing values."""
        csv_data = """feature1,feature2,feature3
1.0,2.0,3.0
4.0,,6.0
7.0,8.0,"""

        with tempfile.NamedTemporaryFile(mode="w", suffix=".csv", delete=False) as f:
            f.write(csv_data)
            temp_path = f.name

        try:
            loader = CSVLoader()
            dataset = loader.load(temp_path, name="test_missing")

            assert dataset.n_samples == 3
            assert dataset.n_features == 3

            # Check that missing values are handled
            assert pd.isna(dataset.data.iloc[1, 1])  # Row 1, column 1
            assert pd.isna(dataset.data.iloc[2, 2])  # Row 2, column 2

        finally:
            Path(temp_path).unlink()

    def test_load_nonexistent_file(self):
        """Test loading a non-existent file raises appropriate error."""
        loader = CSVLoader()

        with pytest.raises((FileNotFoundError, DataValidationError)):
            loader.load("nonexistent_file.csv")

    def test_load_empty_file(self):
        """Test loading an empty file."""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".csv", delete=False) as f:
            # Write empty content
            f.write("")
            temp_path = f.name

        try:
            loader = CSVLoader()

            with pytest.raises((pd.errors.EmptyDataError, DataValidationError)):
                loader.load(temp_path)

        finally:
            Path(temp_path).unlink()

    def test_load_with_target_column(self):
        """Test loading CSV with target column specification."""
        csv_data = """feature1,feature2,target
1.0,2.0,0
4.0,5.0,1
7.0,8.0,0"""

        with tempfile.NamedTemporaryFile(mode="w", suffix=".csv", delete=False) as f:
            f.write(csv_data)
            temp_path = f.name

        try:
            loader = CSVLoader()
            dataset = loader.load(temp_path, name="test_target", target_column="target")

            assert dataset.n_samples == 3
            assert dataset.n_features == 2  # Excluding target
            assert dataset.has_target is True
            assert "target" in dataset.data.columns

            # Verify features don't include target
            features = dataset.get_numeric_features()
            assert "target" not in features
            assert "feature1" in features
            assert "feature2" in features

        finally:
            Path(temp_path).unlink()

    def test_load_batch_processing(self):
        """Test batch loading functionality."""
        csv_data = """feature1,feature2
""" + "\n".join(
            [f"{i},{i + 1}" for i in range(100)]
        )

        with tempfile.NamedTemporaryFile(mode="w", suffix=".csv", delete=False) as f:
            f.write(csv_data)
            temp_path = f.name

        try:
            loader = CSVLoader()

            # Test batch loading
            batches = list(loader.load_batch(temp_path, batch_size=25))

            assert len(batches) == 4  # 100 rows / 25 batch size

            for i, dataset in enumerate(batches):
                assert dataset.n_samples == 25
                assert dataset.n_features == 2
                assert dataset.name == f"batch_{i}"

        finally:
            Path(temp_path).unlink()


class TestParquetLoader:
    """Test ParquetLoader implementation."""

    def test_init_default_parameters(self):
        """Test initialization with default parameters."""
        loader = ParquetLoader()

        assert loader.engine == "auto"
        assert loader.use_pandas_metadata is True

    def test_init_custom_parameters(self):
        """Test initialization with custom parameters."""
        loader = ParquetLoader(engine="pyarrow", use_pandas_metadata=False)

        assert loader.engine == "pyarrow"
        assert loader.use_pandas_metadata is False

    def test_supported_formats(self):
        """Test supported file formats."""
        loader = ParquetLoader()
        formats = loader.supported_formats

        assert "parquet" in formats
        assert len(formats) >= 1

    @pytest.mark.skipif(
        True,  # Skip by default as pyarrow might not be available
        reason="pyarrow dependency may not be available",
    )
    def test_load_parquet_file(self):
        """Test loading a Parquet file."""
        # Create sample data
        data = pd.DataFrame(
            {
                "feature1": [1.0, 4.0, 7.0],
                "feature2": [2.0, 5.0, 8.0],
                "feature3": [3.0, 6.0, 9.0],
            }
        )

        with tempfile.NamedTemporaryFile(suffix=".parquet", delete=False) as f:
            temp_path = f.name

        try:
            # Write parquet file
            data.to_parquet(temp_path)

            loader = ParquetLoader()
            dataset = loader.load(temp_path, name="test_parquet")

            assert dataset.name == "test_parquet"
            assert dataset.n_samples == 3
            assert dataset.n_features == 3

            # Verify data content
            pd.testing.assert_frame_equal(dataset.data, data)

        finally:
            Path(temp_path).unlink()

    def test_load_nonexistent_parquet_file(self):
        """Test loading a non-existent Parquet file."""
        loader = ParquetLoader()

        with pytest.raises((FileNotFoundError, DataValidationError)):
            loader.load("nonexistent_file.parquet")


class TestDataLoaderErrorHandling:
    """Test error handling across data loaders."""

    def test_csv_loader_invalid_encoding(self):
        """Test CSV loader with invalid encoding."""
        # Create file with special characters
        csv_data = "feature1,feature2\nα,β\nγ,δ"

        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".csv", encoding="utf-8", delete=False
        ) as f:
            f.write(csv_data)
            temp_path = f.name

        try:
            # Try to load with wrong encoding
            loader = CSVLoader(encoding="ascii")

            # Should handle encoding error gracefully
            with pytest.raises((UnicodeDecodeError, DataValidationError)):
                loader.load(temp_path)

        finally:
            Path(temp_path).unlink()

    def test_csv_loader_malformed_data(self):
        """Test CSV loader with malformed data."""
        csv_data = """feature1,feature2,feature3
1.0,2.0,3.0
4.0,5.0
7.0,8.0,9.0,10.0"""  # Inconsistent columns

        with tempfile.NamedTemporaryFile(mode="w", suffix=".csv", delete=False) as f:
            f.write(csv_data)
            temp_path = f.name

        try:
            loader = CSVLoader()

            # Should handle malformed data gracefully
            dataset = loader.load(temp_path, name="malformed")

            # pandas should handle this, might have NaN values
            assert dataset.n_samples > 0

        finally:
            Path(temp_path).unlink()


class TestDataLoaderIntegration:
    """Test data loader integration with Dataset entity."""

    def test_dataset_creation_from_csv(self):
        """Test that Dataset is properly created from CSV data."""
        csv_data = """timestamp,sensor1,sensor2,anomaly_label
2024-01-01,1.0,2.0,0
2024-01-02,1.5,2.5,0
2024-01-03,10.0,20.0,1"""

        with tempfile.NamedTemporaryFile(mode="w", suffix=".csv", delete=False) as f:
            f.write(csv_data)
            temp_path = f.name

        try:
            loader = CSVLoader()
            dataset = loader.load(
                temp_path, name="sensor_data", target_column="anomaly_label"
            )

            # Verify Dataset properties
            assert isinstance(dataset, Dataset)
            assert dataset.name == "sensor_data"
            assert dataset.has_target is True
            assert dataset.n_samples == 3

            # Verify numeric features
            numeric_features = dataset.get_numeric_features()
            assert "sensor1" in numeric_features
            assert "sensor2" in numeric_features
            assert "anomaly_label" not in numeric_features  # Target excluded

            # Verify data types
            assert dataset.data["sensor1"].dtype.kind in ["f", "i"]  # float or int
            assert dataset.data["sensor2"].dtype.kind in ["f", "i"]

        finally:
            Path(temp_path).unlink()

    def test_dataset_metadata_preservation(self):
        """Test that dataset metadata is properly preserved."""
        csv_data = """id,feature1,feature2
1,1.0,2.0
2,4.0,5.0
3,7.0,8.0"""

        with tempfile.NamedTemporaryFile(mode="w", suffix=".csv", delete=False) as f:
            f.write(csv_data)
            temp_path = f.name

        try:
            loader = CSVLoader()
            dataset = loader.load(temp_path, name="metadata_test")

            # Verify metadata is set
            assert hasattr(dataset, "metadata")
            assert dataset.metadata.get("loader_type") == "csv"
            assert "source_file" in dataset.metadata

            # Verify shape calculations
            assert dataset.shape == (3, 3)

        finally:
            Path(temp_path).unlink()
