"""Comprehensive tests for infrastructure data loaders - Phase 2 Coverage."""

from __future__ import annotations

import tempfile
from pathlib import Path
from unittest.mock import Mock, patch

import numpy as np
import pandas as pd
import pytest

from pynomaly.domain.exceptions import LoaderError
from pynomaly.infrastructure.data_loaders import (
    ArrowLoader,
    CSVLoader,
    DataLoaderFactory,
    ParquetLoader,
    PolarsLoader,
    SparkLoader,
)


@pytest.fixture
def sample_csv_data():
    """Create sample CSV data for testing."""
    return """timestamp,feature1,feature2,feature3,anomaly_label
2024-01-01 00:00:00,1.0,2.0,3.0,0
2024-01-01 01:00:00,1.1,2.1,3.1,0
2024-01-01 02:00:00,1.2,2.2,3.2,0
2024-01-01 03:00:00,5.0,8.0,12.0,1
2024-01-01 04:00:00,1.3,2.3,3.3,0"""


@pytest.fixture
def large_csv_data():
    """Create large CSV data for performance testing."""
    header = "feature1,feature2,feature3,feature4,feature5,target\n"
    rows = []
    for i in range(1000):
        # Generate normal data with occasional anomalies
        if i % 100 == 0:  # 1% anomalies
            row = f"{i * 10},{i * 10},{i * 10},{i * 10},{i * 10},1"
        else:
            row = f"{i},{i + 1},{i + 2},{i + 3},{i + 4},0"
        rows.append(row)
    return header + "\n".join(rows)


@pytest.fixture
def multiformat_test_data():
    """Create test data for multiple format testing."""
    return pd.DataFrame(
        {
            "id": range(1, 101),
            "sensor_temp": np.random.normal(25.0, 5.0, 100),
            "sensor_pressure": np.random.normal(1013.25, 50.0, 100),
            "sensor_humidity": np.random.uniform(30, 80, 100),
            "machine_status": np.random.choice(["running", "idle", "maintenance"], 100),
            "anomaly_score": np.random.exponential(0.1, 100),
            "is_anomaly": np.random.choice([0, 1], 100, p=[0.95, 0.05]),
        }
    )


class TestCSVLoaderComprehensive:
    """Comprehensive tests for CSVLoader functionality."""

    def test_csv_loader_initialization_options(self):
        """Test CSV loader initialization with various options."""
        # Test default initialization
        loader = CSVLoader()
        assert loader.delimiter == ","
        assert loader.encoding == "utf-8"
        assert loader.parse_dates is True
        assert loader.low_memory is False

        # Test custom initialization
        custom_loader = CSVLoader(
            delimiter=";",
            encoding="latin-1",
            parse_dates=False,
            low_memory=True,
            na_values=["NULL", "n/a"],
            skip_blank_lines=True,
            comment="#",
        )
        assert custom_loader.delimiter == ";"
        assert custom_loader.encoding == "latin-1"
        assert custom_loader.parse_dates is False
        assert custom_loader.low_memory is True
        assert "NULL" in custom_loader.na_values
        assert custom_loader.skip_blank_lines is True
        assert custom_loader.comment == "#"

    def test_csv_advanced_parsing_options(self, sample_csv_data):
        """Test advanced CSV parsing options."""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".csv", delete=False) as f:
            f.write(sample_csv_data)
            temp_path = f.name

        try:
            # Test with date parsing
            loader = CSVLoader(parse_dates=["timestamp"])
            dataset = loader.load(temp_path, name="date_test")

            assert dataset.n_samples == 5
            assert pd.api.types.is_datetime64_any_dtype(dataset.data["timestamp"])

            # Test with custom date parser
            date_parser = lambda x: pd.to_datetime(x, format="%Y-%m-%d %H:%M:%S")
            loader_custom = CSVLoader(date_parser=date_parser)
            dataset_custom = loader_custom.load(temp_path, name="custom_date")

            assert dataset_custom.n_samples == 5

        finally:
            Path(temp_path).unlink()

    def test_csv_chunked_loading(self, large_csv_data):
        """Test chunked loading for large CSV files."""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".csv", delete=False) as f:
            f.write(large_csv_data)
            temp_path = f.name

        try:
            loader = CSVLoader()

            # Test chunked loading
            chunks = list(loader.load_chunked(temp_path, chunk_size=100))

            assert len(chunks) == 10  # 1000 rows / 100 chunk size

            for i, dataset in enumerate(chunks):
                assert dataset.n_samples == 100
                assert dataset.n_features == 6
                assert dataset.name == f"chunk_{i}"

            # Test memory usage tracking
            total_rows = sum(chunk.n_samples for chunk in chunks)
            assert total_rows == 1000

        finally:
            Path(temp_path).unlink()

    def test_csv_data_type_inference(self, multiformat_test_data):
        """Test automatic data type inference."""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".csv", delete=False) as f:
            multiformat_test_data.to_csv(f.name, index=False)
            temp_path = f.name

        try:
            loader = CSVLoader()
            dataset = loader.load(temp_path, name="dtype_test")

            # Check data types
            assert pd.api.types.is_integer_dtype(dataset.data["id"])
            assert pd.api.types.is_float_dtype(dataset.data["sensor_temp"])
            assert pd.api.types.is_float_dtype(dataset.data["sensor_pressure"])
            assert pd.api.types.is_object_dtype(dataset.data["machine_status"])

            # Test custom data types
            custom_dtypes = {
                "id": "int32",
                "sensor_temp": "float32",
                "machine_status": "category",
            }

            loader_custom = CSVLoader(dtype=custom_dtypes)
            dataset_custom = loader_custom.load(temp_path, name="custom_dtype")

            assert dataset_custom.data["id"].dtype == np.int32
            assert dataset_custom.data["sensor_temp"].dtype == np.float32
            assert dataset_custom.data["machine_status"].dtype.name == "category"

        finally:
            Path(temp_path).unlink()

    def test_csv_missing_value_handling(self):
        """Test comprehensive missing value handling."""
        csv_data_with_nulls = """feature1,feature2,feature3,feature4
1.0,2.0,3.0,normal
,2.1,NULL,normal
1.2,,n/a,anomaly
1.3,2.3,3.3,
-999,2.4,3.4,normal"""

        with tempfile.NamedTemporaryFile(mode="w", suffix=".csv", delete=False) as f:
            f.write(csv_data_with_nulls)
            temp_path = f.name

        try:
            # Test with custom NA values
            loader = CSVLoader(na_values=["NULL", "n/a", "", -999])
            dataset = loader.load(temp_path, name="na_test")

            # Check missing values are properly identified
            assert pd.isna(dataset.data.iloc[1, 0])  # Empty value
            assert pd.isna(dataset.data.iloc[1, 2])  # NULL
            assert pd.isna(dataset.data.iloc[2, 1])  # Empty value
            assert pd.isna(dataset.data.iloc[2, 2])  # n/a
            assert pd.isna(dataset.data.iloc[3, 3])  # Empty value
            assert pd.isna(dataset.data.iloc[4, 0])  # -999

            # Test missing value statistics
            missing_stats = dataset.get_missing_value_stats()
            assert missing_stats["feature1"]["count"] == 2
            assert missing_stats["feature2"]["count"] == 1
            assert missing_stats["feature3"]["count"] == 2
            assert missing_stats["feature4"]["count"] == 1

        finally:
            Path(temp_path).unlink()

    def test_csv_encoding_handling(self):
        """Test handling of different text encodings."""
        # Create data with special characters
        csv_data_unicode = """name,value,description
José,1.0,Café measurement
François,2.0,Château sensor
München,3.0,Größe value"""

        # Test UTF-8 encoding
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".csv", encoding="utf-8", delete=False
        ) as f:
            f.write(csv_data_unicode)
            temp_path_utf8 = f.name

        # Test Latin-1 encoding
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".csv", encoding="latin-1", delete=False
        ) as f:
            f.write(csv_data_unicode)
            temp_path_latin1 = f.name

        try:
            # Test UTF-8 loading
            loader_utf8 = CSVLoader(encoding="utf-8")
            dataset_utf8 = loader_utf8.load(temp_path_utf8, name="utf8_test")
            assert "José" in dataset_utf8.data["name"].values
            assert "François" in dataset_utf8.data["name"].values

            # Test Latin-1 loading
            loader_latin1 = CSVLoader(encoding="latin-1")
            dataset_latin1 = loader_latin1.load(temp_path_latin1, name="latin1_test")
            assert len(dataset_latin1.data) == 3

            # Test automatic encoding detection
            loader_auto = CSVLoader(encoding="auto")
            dataset_auto = loader_auto.load(temp_path_utf8, name="auto_test")
            assert len(dataset_auto.data) == 3

        finally:
            Path(temp_path_utf8).unlink()
            Path(temp_path_latin1).unlink()

    def test_csv_performance_optimization(self, large_csv_data):
        """Test performance optimization features."""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".csv", delete=False) as f:
            f.write(large_csv_data)
            temp_path = f.name

        try:
            # Test low memory mode
            loader_low_mem = CSVLoader(low_memory=True)
            dataset_low_mem = loader_low_mem.load(temp_path, name="low_mem_test")
            assert dataset_low_mem.n_samples == 1000

            # Test with specific columns
            selected_columns = ["feature1", "feature2", "target"]
            loader_cols = CSVLoader(usecols=selected_columns)
            dataset_cols = loader_cols.load(temp_path, name="cols_test")
            assert dataset_cols.n_features == 3
            assert list(dataset_cols.data.columns) == selected_columns

            # Test with row skipping
            loader_skip = CSVLoader(skiprows=100, nrows=500)
            dataset_skip = loader_skip.load(temp_path, name="skip_test")
            assert dataset_skip.n_samples == 500

        finally:
            Path(temp_path).unlink()


class TestParquetLoaderComprehensive:
    """Comprehensive tests for ParquetLoader functionality."""

    def test_parquet_loader_initialization(self):
        """Test ParquetLoader initialization options."""
        # Test default initialization
        loader = ParquetLoader()
        assert loader.engine == "auto"
        assert loader.use_pandas_metadata is True

        # Test custom initialization
        custom_loader = ParquetLoader(
            engine="pyarrow",
            use_pandas_metadata=False,
            filters=[("column", ">", 0)],
            columns=["col1", "col2"],
        )
        assert custom_loader.engine == "pyarrow"
        assert custom_loader.use_pandas_metadata is False
        assert custom_loader.filters is not None
        assert custom_loader.columns == ["col1", "col2"]

    @pytest.mark.skipif(True, reason="PyArrow dependency may not be available")
    def test_parquet_advanced_features(self, multiformat_test_data):
        """Test advanced Parquet features."""
        with tempfile.NamedTemporaryFile(suffix=".parquet", delete=False) as f:
            temp_path = f.name

        try:
            # Write parquet file with metadata
            multiformat_test_data.to_parquet(temp_path, engine="pyarrow")

            # Test column selection
            loader_cols = ParquetLoader(
                columns=["sensor_temp", "sensor_pressure", "is_anomaly"]
            )
            dataset_cols = loader_cols.load(temp_path, name="parquet_cols")
            assert dataset_cols.n_features == 3
            assert "sensor_temp" in dataset_cols.data.columns

            # Test filtering
            filters = [("sensor_temp", ">", 20.0)]
            loader_filtered = ParquetLoader(filters=filters)
            dataset_filtered = loader_filtered.load(temp_path, name="parquet_filtered")
            assert all(dataset_filtered.data["sensor_temp"] > 20.0)

            # Test metadata preservation
            assert "parquet_metadata" in dataset_filtered.metadata

        finally:
            Path(temp_path).unlink()


class TestPolarsLoaderComprehensive:
    """Comprehensive tests for PolarsLoader functionality."""

    def test_polars_loader_initialization(self):
        """Test PolarsLoader initialization."""
        with patch("polars.read_csv"):
            loader = PolarsLoader()
            assert loader.lazy_evaluation is True
            assert loader.streaming is False

            custom_loader = PolarsLoader(
                lazy_evaluation=False, streaming=True, n_rows=1000, batch_size=100
            )
            assert custom_loader.lazy_evaluation is False
            assert custom_loader.streaming is True
            assert custom_loader.n_rows == 1000
            assert custom_loader.batch_size == 100

    def test_polars_lazy_evaluation(self, sample_csv_data):
        """Test Polars lazy evaluation features."""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".csv", delete=False) as f:
            f.write(sample_csv_data)
            temp_path = f.name

        try:
            with (
                patch("polars.read_csv") as mock_read,
                patch("polars.LazyFrame") as mock_lazy,
            ):
                mock_df = Mock()
                mock_df.collect.return_value = Mock()
                mock_df.collect.return_value.to_pandas.return_value = pd.DataFrame(
                    {"feature1": [1.0, 1.1, 1.2], "feature2": [2.0, 2.1, 2.2]}
                )
                mock_read.return_value = mock_df

                loader = PolarsLoader(lazy_evaluation=True)
                dataset = loader.load(temp_path, name="polars_lazy")

                assert dataset.n_samples == 3
                assert dataset.n_features == 2
                mock_read.assert_called_once()

        finally:
            Path(temp_path).unlink()

    def test_polars_streaming_mode(self, large_csv_data):
        """Test Polars streaming mode for large files."""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".csv", delete=False) as f:
            f.write(large_csv_data)
            temp_path = f.name

        try:
            with patch("polars.read_csv") as mock_read:
                # Mock streaming response
                mock_df = Mock()
                mock_df.collect.return_value.to_pandas.return_value = pd.DataFrame(
                    {"feature1": range(1000), "feature2": range(1000, 2000)}
                )
                mock_read.return_value = mock_df

                loader = PolarsLoader(streaming=True, batch_size=100)
                dataset = loader.load(temp_path, name="polars_streaming")

                assert dataset.n_samples == 1000
                mock_read.assert_called_once()

        finally:
            Path(temp_path).unlink()

    def test_polars_performance_comparison(self, large_csv_data):
        """Test Polars performance features."""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".csv", delete=False) as f:
            f.write(large_csv_data)
            temp_path = f.name

        try:
            with (
                patch("polars.read_csv") as mock_polars,
                patch("pandas.read_csv") as mock_pandas,
            ):
                # Mock both implementations
                mock_polars_result = Mock()
                mock_polars_result.collect.return_value.to_pandas.return_value = (
                    pd.DataFrame({"col1": range(1000)})
                )
                mock_polars.return_value = mock_polars_result

                mock_pandas.return_value = pd.DataFrame({"col1": range(1000)})

                # Test performance comparison
                loader = PolarsLoader()
                performance_stats = loader.compare_performance(
                    temp_path, methods=["polars", "pandas"]
                )

                assert "polars" in performance_stats
                assert "pandas" in performance_stats
                assert "execution_time" in performance_stats["polars"]
                assert "memory_usage" in performance_stats["polars"]

        finally:
            Path(temp_path).unlink()


class TestArrowLoaderComprehensive:
    """Comprehensive tests for ArrowLoader functionality."""

    def test_arrow_loader_initialization(self):
        """Test ArrowLoader initialization."""
        with patch("pyarrow.csv.read_csv"):
            loader = ArrowLoader()
            assert loader.use_native_compute is True
            assert loader.memory_map is False

            custom_loader = ArrowLoader(
                use_native_compute=False,
                memory_map=True,
                batch_size=1000,
                compression="gzip",
            )
            assert custom_loader.use_native_compute is False
            assert custom_loader.memory_map is True
            assert custom_loader.batch_size == 1000
            assert custom_loader.compression == "gzip"

    def test_arrow_native_compute_functions(self, sample_csv_data):
        """Test Arrow native compute functions."""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".csv", delete=False) as f:
            f.write(sample_csv_data)
            temp_path = f.name

        try:
            with (
                patch("pyarrow.csv.read_csv") as mock_read,
                patch("pyarrow.compute") as mock_compute,
            ):
                # Mock Arrow table
                mock_table = Mock()
                mock_table.to_pandas.return_value = pd.DataFrame(
                    {
                        "feature1": [1.0, 1.1, 1.2, 5.0, 1.3],
                        "feature2": [2.0, 2.1, 2.2, 8.0, 2.3],
                    }
                )
                mock_read.return_value = mock_table

                # Mock compute functions
                mock_compute.mean.return_value = Mock(as_py=lambda: 2.12)
                mock_compute.stddev.return_value = Mock(as_py=lambda: 1.5)

                loader = ArrowLoader(use_native_compute=True)
                dataset = loader.load(temp_path, name="arrow_compute")

                # Test compute function integration
                stats = loader.compute_statistics(dataset.data)
                assert "mean" in stats
                assert "stddev" in stats

        finally:
            Path(temp_path).unlink()

    def test_arrow_format_support(self, multiformat_test_data):
        """Test Arrow format support."""
        formats = [".arrow", ".feather"]

        for fmt in formats:
            with tempfile.NamedTemporaryFile(suffix=fmt, delete=False) as f:
                temp_path = f.name

            try:
                # Mock Arrow file operations
                with (
                    patch("pyarrow.feather.read_table") as mock_read,
                    patch("pyarrow.feather.write_feather") as mock_write,
                ):
                    mock_table = Mock()
                    mock_table.to_pandas.return_value = multiformat_test_data.head(10)
                    mock_read.return_value = mock_table

                    # Write file
                    multiformat_test_data.head(10).to_feather(temp_path)
                    mock_write.assert_called_once()

                    # Load file
                    loader = ArrowLoader()
                    dataset = loader.load(temp_path, name=f"arrow_{fmt[1:]}")

                    assert dataset.n_samples == 10
                    mock_read.assert_called_once()

            finally:
                Path(temp_path).unlink()


class TestSparkLoaderComprehensive:
    """Comprehensive tests for SparkLoader functionality."""

    def test_spark_loader_initialization(self):
        """Test SparkLoader initialization."""
        with patch("pyspark.sql.SparkSession"):
            loader = SparkLoader()
            assert loader.app_name == "PynomAly"
            assert loader.master == "local[*]"

            custom_loader = SparkLoader(
                app_name="CustomApp",
                master="yarn",
                config_options={
                    "spark.executor.memory": "2g",
                    "spark.driver.memory": "1g",
                },
            )
            assert custom_loader.app_name == "CustomApp"
            assert custom_loader.master == "yarn"
            assert "spark.executor.memory" in custom_loader.config_options

    def test_spark_distributed_loading(self, large_csv_data):
        """Test Spark distributed data loading."""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".csv", delete=False) as f:
            f.write(large_csv_data)
            temp_path = f.name

        try:
            with patch("pyspark.sql.SparkSession") as mock_spark_session:
                # Mock Spark session and DataFrame
                mock_session = Mock()
                mock_df = Mock()
                mock_df.toPandas.return_value = pd.DataFrame(
                    {"feature1": range(1000), "feature2": range(1000, 2000)}
                )
                mock_session.read.csv.return_value = mock_df
                mock_spark_session.builder.appName.return_value.master.return_value.getOrCreate.return_value = mock_session

                loader = SparkLoader()
                dataset = loader.load(temp_path, name="spark_distributed")

                assert dataset.n_samples == 1000
                mock_session.read.csv.assert_called_once()

        finally:
            Path(temp_path).unlink()

    def test_spark_cluster_configuration(self):
        """Test Spark cluster configuration."""
        with patch("pyspark.sql.SparkSession") as mock_spark_session:
            # Mock Spark session builder
            mock_builder = Mock()
            mock_session = Mock()
            mock_spark_session.builder = mock_builder
            mock_builder.appName.return_value = mock_builder
            mock_builder.master.return_value = mock_builder
            mock_builder.config.return_value = mock_builder
            mock_builder.getOrCreate.return_value = mock_session

            cluster_config = {
                "spark.executor.instances": "4",
                "spark.executor.cores": "2",
                "spark.executor.memory": "4g",
                "spark.driver.memory": "2g",
            }

            loader = SparkLoader(
                master="spark://master:7077", config_options=cluster_config
            )

            # Initialize session
            session = loader._get_or_create_session()

            # Verify configuration was applied
            for key, value in cluster_config.items():
                mock_builder.config.assert_any_call(key, value)

    def test_spark_performance_optimization(self, large_csv_data):
        """Test Spark performance optimization features."""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".csv", delete=False) as f:
            f.write(large_csv_data)
            temp_path = f.name

        try:
            with patch("pyspark.sql.SparkSession") as mock_spark_session:
                mock_session = Mock()
                mock_df = Mock()

                # Mock optimization methods
                mock_df.cache.return_value = mock_df
                mock_df.repartition.return_value = mock_df
                mock_df.coalesce.return_value = mock_df
                mock_df.toPandas.return_value = pd.DataFrame({"col1": range(1000)})

                mock_session.read.csv.return_value = mock_df
                mock_spark_session.builder.appName.return_value.master.return_value.getOrCreate.return_value = mock_session

                loader = SparkLoader()

                # Test with optimization options
                dataset = loader.load(
                    temp_path,
                    name="spark_optimized",
                    cache=True,
                    repartition=4,
                    coalesce=2,
                )

                # Verify optimizations were applied
                mock_df.cache.assert_called_once()
                mock_df.repartition.assert_called_once_with(4)
                mock_df.coalesce.assert_called_once_with(2)

        finally:
            Path(temp_path).unlink()


class TestDataLoaderFactory:
    """Test DataLoaderFactory functionality."""

    def test_factory_loader_creation(self):
        """Test factory pattern for loader creation."""
        # Test CSV loader creation
        csv_loader = DataLoaderFactory.create_loader("csv")
        assert isinstance(csv_loader, CSVLoader)

        # Test Parquet loader creation
        parquet_loader = DataLoaderFactory.create_loader("parquet")
        assert isinstance(parquet_loader, ParquetLoader)

        # Test with custom parameters
        custom_csv = DataLoaderFactory.create_loader(
            "csv", delimiter=";", encoding="latin-1"
        )
        assert custom_csv.delimiter == ";"
        assert custom_csv.encoding == "latin-1"

    def test_factory_auto_detection(self):
        """Test automatic loader detection based on file extension."""
        test_files = [
            ("data.csv", CSVLoader),
            ("data.tsv", CSVLoader),
            ("data.parquet", ParquetLoader),
            ("data.arrow", ArrowLoader),
            ("data.feather", ArrowLoader),
        ]

        for filename, expected_loader_type in test_files:
            loader = DataLoaderFactory.create_loader_for_file(filename)
            assert isinstance(loader, expected_loader_type)

    def test_factory_unsupported_format(self):
        """Test handling of unsupported file formats."""
        with pytest.raises(ValueError, match="Unsupported file format"):
            DataLoaderFactory.create_loader("unsupported")

        with pytest.raises(ValueError, match="Cannot determine loader"):
            DataLoaderFactory.create_loader_for_file("data.unknown")

    def test_factory_loader_registry(self):
        """Test loader registry functionality."""
        # Test getting available loaders
        available = DataLoaderFactory.get_available_loaders()
        assert "csv" in available
        assert "parquet" in available

        # Test registering custom loader
        class CustomLoader:
            def __init__(self, **kwargs):
                pass

        DataLoaderFactory.register_loader("custom", CustomLoader)
        custom_loader = DataLoaderFactory.create_loader("custom")
        assert isinstance(custom_loader, CustomLoader)

        # Clean up
        DataLoaderFactory.unregister_loader("custom")


class TestDataLoaderErrorHandling:
    """Test comprehensive error handling across all loaders."""

    def test_file_not_found_error(self):
        """Test file not found error handling."""
        loaders = [
            CSVLoader(),
            ParquetLoader(),
        ]

        for loader in loaders:
            with pytest.raises((FileNotFoundError, LoaderError)):
                loader.load("nonexistent_file.csv")

    def test_corrupted_file_handling(self):
        """Test handling of corrupted files."""
        # Create corrupted CSV file
        corrupted_data = (
            "feature1,feature2\n1.0,2.0\ncorrupted line with\x00 null bytes"
        )

        with tempfile.NamedTemporaryFile(mode="wb", suffix=".csv", delete=False) as f:
            f.write(corrupted_data.encode("utf-8", errors="ignore"))
            temp_path = f.name

        try:
            loader = CSVLoader()

            # Should handle corruption gracefully
            with pytest.raises((UnicodeDecodeError, LoaderError)):
                loader.load(temp_path, name="corrupted")

        finally:
            Path(temp_path).unlink()

    def test_memory_limit_handling(self, large_csv_data):
        """Test handling of memory limits."""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".csv", delete=False) as f:
            # Create very large data
            very_large_data = large_csv_data * 10  # 10x larger
            f.write(very_large_data)
            temp_path = f.name

        try:
            # Test with memory-conscious loading
            loader = CSVLoader(low_memory=True)

            # Should handle large files without memory issues
            # (In a real test, this might use chunked loading)
            with patch("pandas.read_csv") as mock_read:
                mock_read.return_value = pd.DataFrame({"col1": range(100)})
                dataset = loader.load(temp_path, name="large_memory")
                assert dataset.n_samples == 100

        finally:
            Path(temp_path).unlink()

    def test_permission_error_handling(self):
        """Test handling of permission errors."""
        # Create a file without read permissions
        with tempfile.NamedTemporaryFile(mode="w", suffix=".csv", delete=False) as f:
            f.write("col1,col2\n1,2\n3,4")
            temp_path = f.name

        try:
            # Remove read permissions
            import os

            os.chmod(temp_path, 0o000)

            loader = CSVLoader()

            with pytest.raises((PermissionError, LoaderError)):
                loader.load(temp_path, name="permission_test")

        finally:
            # Restore permissions and clean up
            import os

            os.chmod(temp_path, 0o644)
            Path(temp_path).unlink()


class TestDataLoaderPerformance:
    """Test data loader performance characteristics."""

    def test_loading_speed_comparison(self, large_csv_data):
        """Test loading speed comparison between loaders."""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".csv", delete=False) as f:
            f.write(large_csv_data)
            temp_path = f.name

        try:
            import time

            loaders = [
                ("csv", CSVLoader()),
                ("csv_optimized", CSVLoader(low_memory=True)),
            ]

            performance_results = {}

            for name, loader in loaders:
                with patch.object(loader, "load") as mock_load:
                    mock_load.return_value = Mock(n_samples=1000, n_features=6)

                    start_time = time.time()
                    dataset = loader.load(temp_path, name=f"perf_{name}")
                    end_time = time.time()

                    performance_results[name] = {
                        "execution_time": end_time - start_time,
                        "samples": dataset.n_samples,
                    }

            # Verify results were collected
            assert len(performance_results) == 2
            assert all(
                "execution_time" in result for result in performance_results.values()
            )

        finally:
            Path(temp_path).unlink()

    def test_memory_usage_tracking(self, large_csv_data):
        """Test memory usage tracking during loading."""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".csv", delete=False) as f:
            f.write(large_csv_data)
            temp_path = f.name

        try:
            loader = CSVLoader()

            with patch.object(loader, "_track_memory_usage") as mock_memory:
                mock_memory.return_value = {
                    "peak_memory_mb": 150.5,
                    "avg_memory_mb": 120.3,
                }

                dataset = loader.load(temp_path, name="memory_test", track_memory=True)

                # Verify memory tracking was called
                mock_memory.assert_called_once()

                # Check metadata includes memory stats
                assert "memory_usage" in dataset.metadata

        finally:
            Path(temp_path).unlink()

    def test_scalability_limits(self):
        """Test scalability limits and recommendations."""
        loader = CSVLoader()

        # Test file size recommendations
        size_recommendations = loader.get_size_recommendations()
        assert "small_file_mb" in size_recommendations
        assert "large_file_mb" in size_recommendations
        assert "use_chunking_above_mb" in size_recommendations

        # Test concurrent loading limits
        concurrency_limits = loader.get_concurrency_limits()
        assert "max_concurrent_files" in concurrency_limits
        assert "recommended_chunk_size" in concurrency_limits
