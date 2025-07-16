"""Tests for memory-efficient data processing infrastructure."""

import tempfile
from pathlib import Path
from unittest.mock import Mock, patch

import numpy as np
import pandas as pd
import pytest

from pynomaly.domain.entities import Dataset
from pynomaly.infrastructure.data_processing import (
    DataChunk,
    DataValidator,
    LargeDatasetAnalyzer,
    MemoryOptimizedDataLoader,
    StreamingDataProcessor,
    ValidationCategory,
    ValidationSeverity,
    get_memory_usage,
    monitor_memory_usage,
    validate_file_format,
)


class TestMemoryOptimizedDataLoader:
    """Test memory-optimized data loading."""

    def test_chunk_dataframe(self):
        """Test DataFrame chunking."""
        # Create test DataFrame
        df = pd.DataFrame(
            {"A": range(1000), "B": np.random.randn(1000), "C": ["test"] * 1000}
        )

        loader = MemoryOptimizedDataLoader(chunk_size=100)
        chunks = list(loader._chunk_dataframe(df, "test_source"))

        assert len(chunks) == 10  # 1000 / 100
        assert all(isinstance(chunk, DataChunk) for chunk in chunks)
        assert chunks[0].chunk_id == 0
        assert chunks[-1].chunk_id == 9
        assert len(chunks[0].data) == 100
        assert len(chunks[-1].data) == 100

    def test_csv_loading(self):
        """Test CSV file loading in chunks."""
        # Create temporary CSV file
        df = pd.DataFrame({"value": range(500), "category": ["A", "B"] * 250})

        with tempfile.NamedTemporaryFile(mode="w", suffix=".csv", delete=False) as f:
            df.to_csv(f.name, index=False)
            temp_path = Path(f.name)

        try:
            loader = MemoryOptimizedDataLoader(chunk_size=100)
            chunks = list(loader.load_csv(temp_path))

            assert len(chunks) == 5  # 500 / 100
            assert all(chunk.metadata["memory_optimized"] for chunk in chunks)

            # Verify data integrity
            reconstructed = pd.concat(
                [chunk.data for chunk in chunks], ignore_index=True
            )
            assert len(reconstructed) == 500
            assert list(reconstructed.columns) == ["value", "category"]

        finally:
            temp_path.unlink()

    def test_dataset_loading(self):
        """Test Dataset entity loading."""
        # Test with in-memory dataset
        df = pd.DataFrame({"values": range(100)})
        dataset = Dataset(id="test", name="Test Dataset", data=df)

        loader = MemoryOptimizedDataLoader(chunk_size=25)
        chunks = list(loader.load_dataset(dataset))

        assert len(chunks) == 4  # 100 / 25
        assert all(len(chunk.data) == 25 for chunk in chunks)

    def test_datatype_optimization(self):
        """Test automatic datatype optimization."""
        # Create DataFrame with inefficient types
        df = pd.DataFrame(
            {
                "small_int": pd.Series([1, 2, 3, 4, 5], dtype="int64"),
                "large_int": pd.Series([1000000] * 5, dtype="int64"),
                "float_data": pd.Series([1.0, 2.0, 3.0], dtype="float64"),
                "category_data": pd.Series(["A", "B", "A", "B", "A"], dtype="object"),
            }
        )

        loader = MemoryOptimizedDataLoader()
        optimized_df = loader._optimize_datatypes(df)

        # Check that small integers were downsized
        assert optimized_df["small_int"].dtype in ["int8", "int16", "int32"]
        # Category data should be categorical if low cardinality
        assert optimized_df["category_data"].dtype.name == "category"

    def test_memory_monitoring(self):
        """Test memory usage monitoring."""
        with monitor_memory_usage("test_operation") as start_metrics:
            # Do some memory-intensive operation
            pd.DataFrame(np.random.randn(1000, 10))

            assert start_metrics.total_mb > 0
            assert start_metrics.process_mb > 0


class TestStreamingDataProcessor:
    """Test streaming data processor."""

    def test_basic_processing(self):
        """Test basic streaming processing."""
        # Create test dataset
        df = pd.DataFrame({"values": range(100)})
        dataset = Dataset(id="test", name="Test Dataset", data=df)

        # Mock processor
        mock_processor = Mock()
        mock_processor.process_chunk.side_effect = lambda chunk: chunk
        mock_processor.finalize.return_value = "processing_complete"

        processor = StreamingDataProcessor()
        result = processor.process_dataset(dataset, mock_processor)

        assert result == "processing_complete"
        assert mock_processor.process_chunk.called
        assert mock_processor.finalize.called

    def test_progress_callback(self):
        """Test progress callback functionality."""
        df = pd.DataFrame({"values": range(50)})
        dataset = Dataset(id="test", name="Test Dataset", data=df)

        # Mock processor
        mock_processor = Mock()
        mock_processor.process_chunk.side_effect = lambda chunk: chunk
        mock_processor.finalize.return_value = "done"

        # Track progress calls
        progress_calls = []

        def progress_callback(**kwargs):
            progress_calls.append(kwargs)

        processor = StreamingDataProcessor()
        processor.loader.chunk_size = 10  # Small chunks for testing

        processor.process_dataset(dataset, mock_processor, progress_callback)

        assert len(progress_calls) == 5  # 50 / 10
        assert all("chunk_id" in call for call in progress_calls)
        assert all("rows_processed" in call for call in progress_calls)


class TestLargeDatasetAnalyzer:
    """Test large dataset analyzer."""

    def test_dataset_analysis(self):
        """Test comprehensive dataset analysis."""
        # Create test dataset with various data types
        df = pd.DataFrame(
            {
                "numeric": range(100),
                "float_data": np.random.randn(100),
                "categorical": ["A", "B", "C"] * 33 + ["A"],
                "text": ["text_" + str(i) for i in range(100)],
                "missing_data": [1.0] * 50 + [np.nan] * 50,
            }
        )

        dataset = Dataset(id="test", name="Test Dataset", data=df)
        analyzer = LargeDatasetAnalyzer()

        stats = analyzer.analyze_dataset(dataset)

        assert stats["total_rows"] == 100
        assert stats["total_columns"] == 5
        assert "numeric" in stats["numeric_columns"]
        assert "float_data" in stats["numeric_columns"]
        assert "categorical" in stats["categorical_columns"]

        # Check missing values analysis
        assert stats["missing_values"]["missing_data"]["count"] == 50
        assert stats["missing_values"]["missing_data"]["percentage"] == 50.0

        # Check column info
        assert "min" in stats["column_info"]["numeric"]
        assert "max" in stats["column_info"]["numeric"]


class TestDataValidator:
    """Test data validation infrastructure."""

    def test_basic_validation(self):
        """Test basic dataset validation."""
        # Create test dataset
        df = pd.DataFrame(
            {
                "good_data": range(100),
                "has_missing": [1.0] * 80 + [np.nan] * 20,
                "constant": [42] * 100,
                "outliers": list(range(95)) + [1000, 2000, 3000, 4000, 5000],
            }
        )

        dataset = Dataset(id="test", name="Test Dataset", data=df)
        validator = DataValidator()

        result = validator.validate_dataset(dataset)

        assert isinstance(result.is_valid, bool)
        assert len(result.issues) > 0

        # Should find constant column
        constant_issues = result.get_issues_by_category(ValidationCategory.RANGE)
        assert any("constant" in issue.message.lower() for issue in constant_issues)

        # Should find missing values
        missing_issues = result.get_issues_by_category(ValidationCategory.COMPLETENESS)
        assert any("missing" in issue.message.lower() for issue in missing_issues)

    def test_empty_dataset_validation(self):
        """Test validation of empty dataset."""
        df = pd.DataFrame()
        dataset = Dataset(id="test", name="Empty Dataset", data=df)
        validator = DataValidator()

        result = validator.validate_dataset(dataset)

        assert not result.is_valid
        assert result.has_critical_issues

        critical_issues = result.get_issues_by_severity(ValidationSeverity.CRITICAL)
        assert len(critical_issues) > 0
        assert any("empty" in issue.message.lower() for issue in critical_issues)

    def test_duplicate_validation(self):
        """Test duplicate data validation."""
        # Create dataset with duplicates
        df = pd.DataFrame(
            {
                "A": [1, 2, 3, 1, 2],  # Some duplicates
                "B": [1, 2, 3, 1, 2],  # Exact same duplicates
            }
        )

        dataset = Dataset(id="test", name="Test Dataset", data=df)
        validator = DataValidator()

        result = validator.validate_dataset(dataset)

        consistency_issues = result.get_issues_by_category(
            ValidationCategory.CONSISTENCY
        )
        assert any("duplicate" in issue.message.lower() for issue in consistency_issues)

    def test_custom_validation_rules(self):
        """Test custom validation rules."""
        df = pd.DataFrame({"values": range(10)})
        dataset = Dataset(id="test", name="Test Dataset", data=df)

        # Custom rule that always finds an issue
        def custom_rule(df, threshold=5):
            from src.pynomaly.infrastructure.data_processing.data_validator import (
                ValidationCategory,
                ValidationIssue,
                ValidationSeverity,
            )

            return [
                ValidationIssue(
                    category=ValidationCategory.QUALITY,
                    severity=ValidationSeverity.WARNING,
                    message=f"Custom rule triggered with threshold {threshold}",
                )
            ]

        custom_rules = {
            "test_rule": {"function": custom_rule, "parameters": {"threshold": 10}}
        }

        validator = DataValidator()
        result = validator.validate_dataset(dataset, custom_rules=custom_rules)

        custom_issues = [
            issue for issue in result.issues if "Custom rule" in issue.message
        ]
        assert len(custom_issues) == 1
        assert "threshold 10" in custom_issues[0].message


class TestFileValidation:
    """Test file format validation."""

    def test_file_exists_validation(self):
        """Test validation of file existence."""
        # Test non-existent file
        result = validate_file_format("/non/existent/file.csv")
        assert not result.is_valid
        assert result.has_critical_issues

        critical_issues = result.get_issues_by_severity(ValidationSeverity.CRITICAL)
        assert any("does not exist" in issue.message for issue in critical_issues)

    def test_file_extension_validation(self):
        """Test file extension validation."""
        # Create temporary file with unsupported extension
        with tempfile.NamedTemporaryFile(suffix=".xyz", delete=False) as f:
            f.write(b"test data")
            temp_path = Path(f.name)

        try:
            result = validate_file_format(temp_path)

            # Should warn about unsupported extension
            warning_issues = result.get_issues_by_severity(ValidationSeverity.WARNING)
            assert any(
                "unsupported" in issue.message.lower() for issue in warning_issues
            )

        finally:
            temp_path.unlink()

    def test_large_file_validation(self):
        """Test large file size validation."""
        # This test would be slow with actual large files, so we'll mock it
        with patch("pathlib.Path.stat") as mock_stat:
            # Mock a 2GB file
            mock_stat.return_value.st_size = 2 * 1024 * 1024 * 1024

            with patch("pathlib.Path.exists", return_value=True):
                result = validate_file_format("large_file.csv")

                # Should warn about large file size
                warning_issues = result.get_issues_by_severity(
                    ValidationSeverity.WARNING
                )
                assert any(
                    "large file" in issue.message.lower() for issue in warning_issues
                )


class TestMemoryMetrics:
    """Test memory usage metrics."""

    def test_memory_metrics_collection(self):
        """Test memory metrics collection."""
        metrics = get_memory_usage()

        assert metrics.total_mb > 0
        assert metrics.available_mb > 0
        assert metrics.process_mb > 0
        assert 0 <= metrics.percent_used <= 100

    def test_memory_metrics_dict_conversion(self):
        """Test metrics dictionary conversion."""
        metrics = get_memory_usage()
        metrics_dict = metrics.to_dict()

        required_keys = [
            "total_mb",
            "available_mb",
            "used_mb",
            "percent_used",
            "process_mb",
            "timestamp",
        ]
        assert all(key in metrics_dict for key in required_keys)
        assert isinstance(metrics_dict["timestamp"], str)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
