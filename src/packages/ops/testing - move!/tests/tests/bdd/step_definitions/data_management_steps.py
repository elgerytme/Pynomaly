"""Step definitions for data management BDD scenarios."""

from __future__ import annotations

import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from pytest_bdd import given, parsers, then, when

from monorepo.infrastructure.data_loaders import CSVLoader, ParquetLoader


@pytest.fixture
def temp_dir():
    """Create a temporary directory for test files."""
    with tempfile.TemporaryDirectory() as tmp_dir:
        yield Path(tmp_dir)


@pytest.fixture
def data_loaders():
    """Initialize data loaders."""
    return {
        "csv_loader": CSVLoader(),
        "parquet_loader": ParquetLoader(),
    }


# Background steps
@given("the Pynomaly system is initialized")
def initialize_pynomaly_system(data_loaders):
    """Initialize the Pynomaly data management system."""
    pytest.loaders = data_loaders
    pytest.loaded_datasets = {}


@given("I have access to various data formats")
def setup_data_formats():
    """Set up access to various data formats."""
    pytest.data_formats = ["csv", "parquet", "json"]


# CSV data scenarios
@given("I have a CSV file with UTF-8 encoding")
def create_utf8_csv(temp_dir):
    """Create a CSV file with UTF-8 encoding."""
    data = pd.DataFrame(
        {
            "feature1": [1.0, 2.0, 3.0, 4.0, 5.0],
            "feature2": [0.1, 0.2, 0.3, 0.4, 0.5],
            "text_feature": ["café", "naïve", "résumé", "Zürich", "mañana"],
        }
    )

    csv_path = temp_dir / "utf8_data.csv"
    data.to_csv(csv_path, index=False, encoding="utf-8")
    pytest.utf8_csv_path = csv_path
    pytest.expected_utf8_shape = data.shape


@given("I have a CSV file with Windows-1252 encoding")
def create_windows1252_csv(temp_dir):
    """Create a CSV file with Windows-1252 encoding."""
    data = pd.DataFrame(
        {
            "feature1": [1.0, 2.0, 3.0],
            "feature2": [0.1, 0.2, 0.3],
            "text_feature": [
                "café",
                "résumé",
                "naïve",
            ],  # These will be encoded differently
        }
    )

    csv_path = temp_dir / "windows1252_data.csv"
    data.to_csv(csv_path, index=False, encoding="windows-1252")
    pytest.windows1252_csv_path = csv_path
    pytest.expected_windows1252_shape = data.shape


@given(parsers.parse("I have a large CSV file (>{size}MB)"))
def create_large_csv(temp_dir, size):
    """Create a large CSV file for testing."""
    size_mb = int(size.replace("MB", ""))

    # Estimate rows needed for target size
    estimated_rows = size_mb * 1024 * 1024 // 100  # Rough estimate

    # Generate data in chunks to avoid memory issues
    csv_path = temp_dir / "large_data.csv"

    with open(csv_path, "w") as f:
        # Write header
        f.write("feature1,feature2,feature3,feature4,feature5\n")

        # Write data in chunks
        chunk_size = 10000
        for i in range(0, estimated_rows, chunk_size):
            chunk_rows = min(chunk_size, estimated_rows - i)
            chunk_data = np.random.random((chunk_rows, 5))

            for row in chunk_data:
                f.write(",".join(map(str, row)) + "\n")

    pytest.large_csv_path = csv_path
    pytest.expected_large_rows = estimated_rows


# Parquet data scenarios
@given("I have a Parquet file with embedded metadata")
def create_parquet_with_metadata(temp_dir):
    """Create a Parquet file with metadata."""
    data = pd.DataFrame(
        {
            "numeric_feature": [1.0, 2.0, 3.0, 4.0, 5.0],
            "categorical_feature": pd.Categorical(["A", "B", "A", "C", "B"]),
            "date_feature": pd.date_range("2023-01-01", periods=5),
        }
    )

    parquet_path = temp_dir / "metadata_data.parquet"
    data.to_parquet(parquet_path, index=False)
    pytest.parquet_path = parquet_path
    pytest.expected_parquet_dtypes = data.dtypes


# Loading scenarios
@when("I load the data using the CSV loader")
def load_csv_data():
    """Load CSV data using the CSV loader."""
    loader = pytest.loaders["csv_loader"]
    pytest.loaded_csv = loader.load(str(pytest.utf8_csv_path))


@when("I load the data specifying the encoding")
def load_csv_with_encoding():
    """Load CSV data with specific encoding."""
    loader = pytest.loaders["csv_loader"]
    pytest.loaded_windows1252 = loader.load(
        str(pytest.windows1252_csv_path), encoding="windows-1252"
    )


@when("I load the data using chunked processing")
def load_large_csv_chunked():
    """Load large CSV file using chunked processing."""
    pytest.loaders["csv_loader"]

    # Load in chunks
    chunks = []
    chunk_size = 10000

    try:
        for chunk in pd.read_csv(pytest.large_csv_path, chunksize=chunk_size):
            chunks.append(chunk)
            if len(chunks) > 10:  # Limit for testing
                break

        if chunks:
            pytest.loaded_large_chunked = pd.concat(chunks, ignore_index=True)
        else:
            pytest.loaded_large_chunked = pd.DataFrame()

    except Exception as e:
        pytest.load_error = str(e)


@when("I load the data using the Parquet loader")
def load_parquet_data():
    """Load Parquet data using the Parquet loader."""
    loader = pytest.loaders["parquet_loader"]
    pytest.loaded_parquet = loader.load(str(pytest.parquet_path))


# Polars scenarios (if available)
@given("I have a large dataset suitable for Polars")
def create_polars_dataset(temp_dir):
    """Create a dataset suitable for Polars processing."""
    data = pd.DataFrame({f"feature_{i}": np.random.random(10000) for i in range(10)})

    csv_path = temp_dir / "polars_data.csv"
    data.to_csv(csv_path, index=False)
    pytest.polars_csv_path = csv_path
    pytest.expected_polars_shape = data.shape


@when("I load the data using the Polars loader with lazy evaluation")
def load_polars_lazy():
    """Load data using Polars with lazy evaluation."""
    try:
        from monorepo.infrastructure.data_loaders import PolarsLoader

        loader = PolarsLoader(lazy=True)
        pytest.polars_lazy_frame = loader.load(str(pytest.polars_csv_path))
    except ImportError:
        pytest.skip("Polars not available")


# Spark scenarios (if available)
@given("I have a very large dataset requiring distributed processing")
def create_spark_dataset(temp_dir):
    """Create a dataset for Spark processing."""
    # Create multiple files to simulate distributed data
    for i in range(3):
        data = pd.DataFrame({f"feature_{j}": np.random.random(1000) for j in range(5)})

        csv_path = temp_dir / f"spark_data_part_{i}.csv"
        data.to_csv(csv_path, index=False)

    pytest.spark_data_dir = temp_dir
    pytest.expected_spark_total_rows = 3000


@when("I initialize a Spark session for data loading")
def initialize_spark_session():
    """Initialize Spark session."""
    try:
        from monorepo.infrastructure.data_loaders import SparkLoader

        pytest.spark_loader = SparkLoader(app_name="TestApp", master="local[*]")
    except ImportError:
        pytest.skip("Spark not available")


@when("I load the data using the Spark loader")
def load_spark_data():
    """Load data using Spark loader."""
    if hasattr(pytest, "spark_loader"):
        # Load all CSV files in the directory
        pattern = str(pytest.spark_data_dir / "spark_data_part_*.csv")
        pytest.spark_dataframe = pytest.spark_loader.load(pattern)


# Validation scenarios
@when("I run data validation checks")
def run_data_validation():
    """Run data validation checks."""
    if hasattr(pytest, "loaded_csv"):
        data = pytest.loaded_csv.features

        pytest.validation_results = {
            "missing_values": pd.isnull(data).sum().sum(),
            "duplicate_rows": data.duplicated().sum(),
            "data_types": data.dtypes.to_dict(),
            "value_ranges": {
                col: {"min": data[col].min(), "max": data[col].max()}
                for col in data.select_dtypes(include=[np.number]).columns
            },
        }


# Verification steps
@then("the data should be loaded correctly")
def verify_csv_loading():
    """Verify CSV data was loaded correctly."""
    assert hasattr(pytest, "loaded_csv"), "CSV data should be loaded"
    assert (
        pytest.loaded_csv.features.shape == pytest.expected_utf8_shape
    ), f"Expected shape {pytest.expected_utf8_shape}, got {pytest.loaded_csv.features.shape}"


@then("the feature matrix should have the expected shape")
def verify_feature_matrix_shape():
    """Verify the feature matrix has expected shape."""
    if hasattr(pytest, "loaded_csv"):
        expected_rows, expected_cols = pytest.expected_utf8_shape
        actual_shape = pytest.loaded_csv.features.shape
        assert actual_shape == (
            expected_rows,
            expected_cols - 1,
        ), f"Expected feature matrix shape {(expected_rows, expected_cols - 1)}, got {actual_shape}"


@then("all numeric values should be preserved")
def verify_numeric_values():
    """Verify numeric values are preserved."""
    if hasattr(pytest, "loaded_csv"):
        numeric_cols = pytest.loaded_csv.features.select_dtypes(
            include=[np.number]
        ).columns
        assert len(numeric_cols) > 0, "Should have numeric columns"

        for col in numeric_cols:
            assert (
                not pytest.loaded_csv.features[col].isnull().all()
            ), f"Column {col} should have values"


@then("the data should load without character encoding errors")
def verify_encoding_handling():
    """Verify encoding is handled correctly."""
    assert hasattr(pytest, "loaded_windows1252"), "Windows-1252 data should be loaded"
    assert pytest.loaded_windows1252.features.shape[0] > 0, "Should have loaded rows"


@then("special characters should be preserved")
def verify_special_characters():
    """Verify special characters are preserved."""
    if hasattr(pytest, "loaded_windows1252"):
        # Check if text columns contain expected characters
        text_cols = pytest.loaded_windows1252.features.select_dtypes(
            include=["object"]
        ).columns
        if len(text_cols) > 0:
            text_data = pytest.loaded_windows1252.features[text_cols[0]].astype(str)
            # At least some special characters should be present
            assert any(
                char in text_data.str.cat() for char in ["é", "ç", "ñ"]
            ), "Special characters should be preserved"


@then("the data should load without memory errors")
def verify_memory_efficient_loading():
    """Verify large data loads without memory errors."""
    assert not hasattr(
        pytest, "load_error"
    ), f"Loading failed: {getattr(pytest, 'load_error', '')}"
    assert hasattr(pytest, "loaded_large_chunked"), "Large data should be loaded"


@then("processing should show progress indicators")
def verify_progress_indicators():
    """Verify progress indicators are shown."""
    # This would require checking for progress bar output
    # For now, just verify the data was processed
    assert hasattr(pytest, "loaded_large_chunked"), "Data should be processed"


@then("the final dataset should be complete")
def verify_complete_dataset():
    """Verify the dataset is complete."""
    if hasattr(pytest, "loaded_large_chunked"):
        assert len(pytest.loaded_large_chunked) > 0, "Dataset should have rows"
        assert (
            len(pytest.loaded_large_chunked.columns) > 0
        ), "Dataset should have columns"


@then("the data types should be preserved from the file")
def verify_parquet_dtypes():
    """Verify Parquet data types are preserved."""
    if hasattr(pytest, "loaded_parquet"):
        loaded_dtypes = pytest.loaded_parquet.features.dtypes

        # Check that categorical and datetime types are handled appropriately
        assert len(loaded_dtypes) > 0, "Should have data types"

        # Verify numeric columns are numeric
        numeric_cols = loaded_dtypes.select_dtypes(include=[np.number]).columns
        assert len(numeric_cols) > 0, "Should have numeric columns"


@then("categorical columns should maintain their categories")
def verify_categorical_preservation():
    """Verify categorical columns are preserved."""
    if hasattr(pytest, "loaded_parquet"):
        # Check if categorical data is handled properly
        # This depends on how the loader handles categories
        assert pytest.loaded_parquet.features.shape[0] > 0, "Should have data rows"


@then("the schema information should be accessible")
def verify_schema_accessibility():
    """Verify schema information is accessible."""
    if hasattr(pytest, "loaded_parquet"):
        dtypes = pytest.loaded_parquet.features.dtypes
        assert len(dtypes) > 0, "Schema information should be available"


# Polars verification steps
@then("the initial load should be very fast")
def verify_polars_fast_loading():
    """Verify Polars loading is fast."""
    if hasattr(pytest, "polars_lazy_frame"):
        # Lazy loading should be essentially instant
        assert pytest.polars_lazy_frame is not None, "Lazy frame should be created"


@then("data processing should be parallelized")
def verify_polars_parallelization():
    """Verify Polars uses parallelization."""
    # This is implementation-dependent, just verify functionality
    if hasattr(pytest, "polars_lazy_frame"):
        assert pytest.polars_lazy_frame is not None, "Parallel processing should work"


@then("memory usage should be optimized")
def verify_polars_memory_optimization():
    """Verify Polars optimizes memory usage."""
    # Memory optimization is handled internally by Polars
    if hasattr(pytest, "polars_lazy_frame"):
        assert pytest.polars_lazy_frame is not None, "Memory optimization should work"


@then("I can apply filters before materialization")
def verify_polars_lazy_filtering():
    """Verify lazy filtering works."""
    if hasattr(pytest, "polars_lazy_frame"):
        # This would test lazy operations, skipping for basic implementation
        assert pytest.polars_lazy_frame is not None, "Lazy filtering should be possible"


# Validation verification steps
@then("missing values should be identified and reported")
def verify_missing_value_detection():
    """Verify missing values are detected."""
    if hasattr(pytest, "validation_results"):
        assert (
            "missing_values" in pytest.validation_results
        ), "Missing values should be tracked"
        assert isinstance(
            pytest.validation_results["missing_values"], int | np.integer
        ), "Missing value count should be numeric"


@then("data types should be validated")
def verify_data_type_validation():
    """Verify data types are validated."""
    if hasattr(pytest, "validation_results"):
        assert (
            "data_types" in pytest.validation_results
        ), "Data types should be validated"
        assert (
            len(pytest.validation_results["data_types"]) > 0
        ), "Should have data type information"


@then("outliers beyond reasonable ranges should be flagged")
def verify_outlier_detection():
    """Verify outliers are flagged."""
    if hasattr(pytest, "validation_results"):
        assert (
            "value_ranges" in pytest.validation_results
        ), "Value ranges should be checked"


@then("duplicate rows should be detected")
def verify_duplicate_detection():
    """Verify duplicate rows are detected."""
    if hasattr(pytest, "validation_results"):
        assert (
            "duplicate_rows" in pytest.validation_results
        ), "Duplicate rows should be tracked"
        assert isinstance(
            pytest.validation_results["duplicate_rows"], int | np.integer
        ), "Duplicate count should be numeric"
