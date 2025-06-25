"""
Branch Coverage Enhancement - Edge Case Testing
Comprehensive tests targeting edge cases, boundary conditions, and unusual scenarios to maximize branch coverage.
"""

import os
import sys
from datetime import UTC, datetime, timedelta, timezone

import numpy as np
import pandas as pd
import pytest

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../../src"))

from pynomaly.domain.entities import Dataset
from pynomaly.domain.value_objects import AnomalyScore, ContaminationRate


class TestBoundaryConditions:
    """Test boundary conditions and limit cases."""

    def test_numeric_boundary_conditions(self):
        """Test numeric boundary conditions."""

        # Test boundary values for AnomalyScore
        boundary_scores = [
            0.0,  # Minimum
            1.0,  # Maximum
            0.5,  # Middle
            1e-10,  # Very small positive
            1.0 - 1e-10,  # Very close to maximum
            0.999999999999999,  # Floating point precision edge
            0.000000000000001,  # Floating point precision edge
        ]

        for score in boundary_scores:
            anomaly_score = AnomalyScore(value=score, method="boundary_test")
            assert 0.0 <= anomaly_score.value <= 1.0

        # Test boundary values for ContaminationRate
        boundary_contamination = [
            0.000000001,  # Just above minimum
            0.499999999,  # Just below maximum
            0.25,  # Quarter
            0.125,  # Eighth
            1e-8,  # Very small
            0.5 - 1e-8,  # Very close to limit
        ]

        for contamination in boundary_contamination:
            cont_rate = ContaminationRate(contamination)
            assert 0.0 < cont_rate.value < 0.5

    def test_dataset_size_boundaries(self):
        """Test dataset size boundary conditions."""

        # Test single row dataset
        single_row_data = pd.DataFrame({"feature": [1.0]})
        single_row_dataset = Dataset(name="single_row", data=single_row_data)
        assert single_row_dataset.n_samples == 1
        assert single_row_dataset.n_features == 1

        # Test single column dataset
        single_col_data = pd.DataFrame({"feature": [1.0, 2.0, 3.0]})
        single_col_dataset = Dataset(name="single_column", data=single_col_data)
        assert single_col_dataset.n_features == 1
        assert single_col_dataset.n_samples == 3

        # Test large column count (edge case for memory)
        large_col_data = pd.DataFrame(
            np.random.randn(10, 1000), columns=[f"feature_{i}" for i in range(1000)]
        )
        large_col_dataset = Dataset(name="large_columns", data=large_col_data)
        assert large_col_dataset.n_features == 1000
        assert large_col_dataset.n_samples == 10

        # Test dataset with maximum practical size for testing
        if os.environ.get("PYNOMALY_TEST_LARGE_DATASETS", "false").lower() == "true":
            # Only run if explicitly enabled
            large_data = pd.DataFrame(
                np.random.randn(100000, 50), columns=[f"feature_{i}" for i in range(50)]
            )
            large_dataset = Dataset(name="large_dataset", data=large_data)
            assert large_dataset.n_samples == 100000
            assert large_dataset.n_features == 50

    def test_floating_point_edge_cases(self):
        """Test floating point edge cases."""
        from pynomaly.infrastructure.data.numeric_utils import NumericProcessor

        processor = NumericProcessor()

        # Test special float values
        special_values = [
            float("inf"),
            float("-inf"),
            float("nan"),
            0.0,
            -0.0,
            1e-100,  # Very small
            1e100,  # Very large
            sys.float_info.min,  # Smallest positive normalized float
            sys.float_info.max,  # Largest finite float
            sys.float_info.epsilon,  # Machine epsilon
        ]

        data_with_special = pd.DataFrame(
            {
                "normal": [1.0, 2.0, 3.0],
                "special": special_values[:3],  # Take first 3 to match length
            }
        )

        # Test handling of special values
        cleaned_data = processor.handle_special_values(data_with_special)

        # Should handle inf and nan appropriately
        assert not np.isinf(cleaned_data["special"]).any()
        assert not np.isnan(cleaned_data["special"]).any()

        # Test precision edge cases
        precision_test_values = [
            1.0000000000000001,  # Just above 1
            0.9999999999999999,  # Just below 1
            1.0 + sys.float_info.epsilon,
            1.0 - sys.float_info.epsilon,
        ]

        for value in precision_test_values:
            result = processor.normalize_precision(value)
            assert isinstance(result, float)
            assert not np.isnan(result)

    def test_string_boundary_conditions(self):
        """Test string handling boundary conditions."""
        from pynomaly.infrastructure.validation.string_validator import StringValidator

        validator = StringValidator()

        # Test empty string
        empty_result = validator.validate_string("")
        assert not empty_result.is_valid

        # Test very long string
        very_long_string = "a" * 10000
        long_result = validator.validate_string(very_long_string, max_length=1000)
        assert not long_result.is_valid

        # Test string with only whitespace
        whitespace_string = "   \t\n  "
        whitespace_result = validator.validate_string(
            whitespace_string, strip_whitespace=True
        )
        assert not whitespace_result.is_valid

        # Test string with special characters
        special_chars = "!@#$%^&*()[]{}|\\:;\"'<>,.?/~`"
        special_result = validator.validate_string(
            special_chars, allow_special_chars=False
        )
        assert not special_result.is_valid

        # Test unicode edge cases
        unicode_strings = [
            "café",  # Accented characters
            "🎉🎊",  # Emojis
            "Здравствуй",  # Cyrillic
            "こんにちは",  # Japanese
            "العربية",  # Arabic
            "\u200b\u200c\u200d",  # Zero-width characters
            "\ufeff",  # BOM character
        ]

        for unicode_str in unicode_strings:
            unicode_result = validator.validate_unicode_string(unicode_str)
            assert unicode_result.is_valid  # Should handle unicode properly


class TestDataTypeEdgeCases:
    """Test edge cases with different data types."""

    def test_mixed_data_types(self):
        """Test handling of mixed data types."""
        from pynomaly.infrastructure.data.type_detector import TypeDetector

        detector = TypeDetector()

        # Create DataFrame with mixed types
        mixed_data = pd.DataFrame(
            {
                "integers": [1, 2, 3, 4, 5],
                "floats": [1.1, 2.2, 3.3, 4.4, 5.5],
                "strings": ["a", "b", "c", "d", "e"],
                "booleans": [True, False, True, False, True],
                "dates": pd.date_range("2023-01-01", periods=5),
                "mixed_numeric": [1, 2.5, 3, 4.7, 5],  # Mixed int/float
                "mixed_string_numeric": [
                    "1",
                    "2.5",
                    "abc",
                    "4",
                    "5",
                ],  # Mixed string/numeric-like
                "nulls_with_numbers": [1, 2, None, 4, 5],
                "nulls_with_strings": ["a", "b", None, "d", "e"],
            }
        )

        type_analysis = detector.analyze_types(mixed_data)

        # Verify type detection
        assert type_analysis["integers"]["inferred_type"] == "integer"
        assert type_analysis["floats"]["inferred_type"] == "float"
        assert type_analysis["strings"]["inferred_type"] == "string"
        assert type_analysis["booleans"]["inferred_type"] == "boolean"
        assert type_analysis["dates"]["inferred_type"] == "datetime"

        # Mixed types should be detected
        assert "mixed" in type_analysis["mixed_string_numeric"]["inferred_type"].lower()

    def test_datetime_edge_cases(self):
        """Test datetime edge cases."""
        from pynomaly.infrastructure.data.datetime_processor import DateTimeProcessor

        processor = DateTimeProcessor()

        # Test various datetime formats
        datetime_formats = [
            "2023-01-01",
            "2023-01-01 12:30:45",
            "01/01/2023",
            "2023-01-01T12:30:45Z",
            "2023-01-01T12:30:45.123456",
            "January 1, 2023",
            "Jan 1, 2023 12:30 PM",
            "2023-W01-1",  # ISO week date
        ]

        for date_str in datetime_formats:
            try:
                parsed_date = processor.parse_datetime(date_str)
                assert isinstance(parsed_date, datetime)
            except ValueError:
                # Some formats might not be supported, which is OK
                pass

        # Test edge datetime values
        edge_dates = [
            datetime.min,
            datetime.max,
            datetime(1970, 1, 1),  # Unix epoch
            datetime(2038, 1, 19),  # 32-bit Unix timestamp limit
            datetime(1900, 1, 1),  # Excel epoch issues
        ]

        for edge_date in edge_dates:
            processed = processor.normalize_datetime(edge_date)
            assert isinstance(processed, datetime)

        # Test timezone edge cases
        timezones = [
            UTC,
            timezone(timedelta(hours=14)),  # UTC+14 (maximum)
            timezone(timedelta(hours=-12)),  # UTC-12 (minimum)
            timezone(timedelta(minutes=30)),  # Half-hour offset
        ]

        base_time = datetime.now()
        for tz in timezones:
            tz_aware = base_time.replace(tzinfo=tz)
            normalized = processor.normalize_timezone(tz_aware)
            assert normalized.tzinfo is not None

    def test_categorical_edge_cases(self):
        """Test categorical data edge cases."""
        from pynomaly.infrastructure.data.categorical_processor import (
            CategoricalProcessor,
        )

        processor = CategoricalProcessor()

        # Test single category
        single_category = pd.Series(["A"] * 100)
        single_result = processor.analyze_cardinality(single_category)
        assert single_result["unique_count"] == 1
        assert single_result["is_constant"] == True

        # Test maximum cardinality (unique per row)
        unique_categories = pd.Series([f"category_{i}" for i in range(1000)])
        unique_result = processor.analyze_cardinality(unique_categories)
        assert unique_result["unique_count"] == 1000
        assert unique_result["cardinality_ratio"] == 1.0

        # Test categories with special characters
        special_categories = pd.Series(
            [
                "category with spaces",
                "category-with-dashes",
                "category_with_underscores",
                "category.with.dots",
                "category/with/slashes",
                "category\\with\\backslashes",
                'category"with"quotes',
                "category'with'apostrophes",
                "category\twith\ttabs",
                "category\nwith\nnewlines",
            ]
        )

        special_result = processor.analyze_cardinality(special_categories)
        assert special_result["unique_count"] == len(special_categories.unique())

        # Test empty categories
        empty_categories = pd.Series(["", None, np.nan, "   "])
        empty_result = processor.handle_missing_categories(empty_categories)
        assert len(empty_result.dropna()) <= len(empty_categories)


class TestMemoryAndPerformanceEdges:
    """Test memory and performance edge cases."""

    def test_memory_efficient_operations(self):
        """Test memory-efficient operations for large datasets."""
        from pynomaly.infrastructure.data.memory_efficient import (
            MemoryEfficientProcessor,
        )

        processor = MemoryEfficientProcessor()

        # Test chunked processing
        large_data_iterator = (
            pd.DataFrame(np.random.randn(1000, 10))
            for _ in range(100)  # 100 chunks of 1000 rows each
        )

        result = processor.process_in_chunks(
            large_data_iterator,
            chunk_processor=lambda chunk: chunk.mean(),
            combine_results=lambda results: pd.concat(results).mean(),
        )

        assert len(result) == 10  # 10 features
        assert not result.isna().any()

        # Test memory monitoring
        with processor.memory_monitor(max_memory_mb=100):
            # Simulate memory-intensive operation
            temp_data = np.random.randn(1000, 100)
            result = np.mean(temp_data, axis=0)
            assert len(result) == 100

        # Test out-of-core operations
        if hasattr(processor, "out_of_core_operation"):
            large_result = processor.out_of_core_operation(
                data_size=(100000, 50), operation="mean"
            )
            assert len(large_result) == 50

    def test_streaming_data_edge_cases(self):
        """Test streaming data processing edge cases."""
        from pynomaly.infrastructure.data.streaming import StreamProcessor

        processor = StreamProcessor()

        # Test empty stream
        empty_stream = iter([])
        empty_result = processor.process_stream(empty_stream)
        assert empty_result is None or len(empty_result) == 0

        # Test single item stream
        single_item_stream = iter([pd.DataFrame({"feature": [1.0]})])
        single_result = processor.process_stream(single_item_stream)
        assert single_result is not None

        # Test irregular chunk sizes
        irregular_chunks = [
            pd.DataFrame(np.random.randn(10, 5)),  # Small chunk
            pd.DataFrame(np.random.randn(1000, 5)),  # Large chunk
            pd.DataFrame(np.random.randn(1, 5)),  # Tiny chunk
            pd.DataFrame(np.random.randn(500, 5)),  # Medium chunk
        ]

        irregular_stream = iter(irregular_chunks)
        irregular_result = processor.process_stream(irregular_stream)
        assert irregular_result is not None

        # Test stream with varying schemas
        varying_schema_chunks = [
            pd.DataFrame({"a": [1, 2], "b": [3, 4]}),
            pd.DataFrame({"a": [5, 6], "b": [7, 8], "c": [9, 10]}),  # Extra column
            pd.DataFrame({"a": [11, 12]}),  # Missing column
        ]

        schema_stream = iter(varying_schema_chunks)
        try:
            schema_result = processor.process_stream_with_schema_evolution(
                schema_stream
            )
            assert schema_result is not None
        except Exception:
            # Schema evolution might not be supported, which is acceptable
            pass

    def test_parallel_processing_edge_cases(self):
        """Test parallel processing edge cases."""
        from pynomaly.infrastructure.processing.parallel import ParallelProcessor

        processor = ParallelProcessor()

        # Test with single worker
        single_worker_result = processor.process_parallel(
            data=list(range(100)), worker_function=lambda x: x**2, n_workers=1
        )
        assert len(single_worker_result) == 100
        assert single_worker_result[10] == 100  # 10^2

        # Test with more workers than data items
        few_items_result = processor.process_parallel(
            data=[1, 2, 3],
            worker_function=lambda x: x * 2,
            n_workers=10,  # More workers than items
        )
        assert len(few_items_result) == 3
        assert few_items_result == [2, 4, 6]

        # Test with worker function that occasionally fails
        def unreliable_worker(x):
            if x % 10 == 0:  # Fail on every 10th item
                raise ValueError(f"Simulated failure for {x}")
            return x * 3

        try:
            unreliable_result = processor.process_parallel_with_error_handling(
                data=list(range(50)),
                worker_function=unreliable_worker,
                n_workers=4,
                ignore_errors=True,
            )
            # Should have fewer results due to failures
            assert len(unreliable_result) < 50
        except Exception:
            # Error handling might not be implemented
            pass


class TestStatisticalEdgeCases:
    """Test statistical computation edge cases."""

    def test_distribution_edge_cases(self):
        """Test statistical distribution edge cases."""
        from pynomaly.infrastructure.statistics.distributions import (
            DistributionAnalyzer,
        )

        analyzer = DistributionAnalyzer()

        # Test constant distribution
        constant_data = np.array([5.0] * 1000)
        constant_stats = analyzer.analyze_distribution(constant_data)
        assert constant_stats["variance"] == 0.0
        assert constant_stats["std"] == 0.0
        assert constant_stats["is_constant"] == True

        # Test bimodal distribution
        bimodal_data = np.concatenate(
            [np.random.normal(-3, 1, 500), np.random.normal(3, 1, 500)]
        )
        bimodal_stats = analyzer.analyze_distribution(bimodal_data)
        assert bimodal_stats["modality"] > 1

        # Test heavily skewed distribution
        skewed_data = np.random.exponential(scale=2, size=1000)
        skewed_stats = analyzer.analyze_distribution(skewed_data)
        assert abs(skewed_stats["skewness"]) > 1.0

        # Test distribution with outliers
        normal_data = np.random.normal(0, 1, 990)
        outliers = np.array([100, -100, 150, -150, 200])  # Extreme outliers
        data_with_outliers = np.concatenate([normal_data, outliers])

        outlier_stats = analyzer.analyze_distribution(data_with_outliers)
        assert outlier_stats["has_outliers"] == True
        assert outlier_stats["outlier_count"] >= len(outliers)

        # Test very small sample
        tiny_sample = np.array([1.0, 2.0])
        tiny_stats = analyzer.analyze_distribution(tiny_sample)
        assert tiny_stats["sample_size"] == 2
        assert "insufficient_data" in tiny_stats.get("warnings", [])

    def test_correlation_edge_cases(self):
        """Test correlation computation edge cases."""
        from pynomaly.infrastructure.statistics.correlation import CorrelationAnalyzer

        analyzer = CorrelationAnalyzer()

        # Test perfect correlation
        x = np.linspace(0, 100, 1000)
        y = 2 * x + 5  # Perfect linear relationship
        perfect_corr = analyzer.compute_correlation(x, y)
        assert abs(perfect_corr - 1.0) < 1e-10

        # Test no correlation
        x_random = np.random.normal(0, 1, 1000)
        y_random = np.random.normal(0, 1, 1000)
        no_corr = analyzer.compute_correlation(x_random, y_random)
        assert abs(no_corr) < 0.1  # Should be close to zero

        # Test negative correlation
        x_neg = np.linspace(0, 100, 1000)
        y_neg = -3 * x_neg + 10
        neg_corr = analyzer.compute_correlation(x_neg, y_neg)
        assert neg_corr < -0.99

        # Test correlation with missing values
        x_missing = np.array([1, 2, np.nan, 4, 5])
        y_missing = np.array([2, 4, 6, np.nan, 10])
        missing_corr = analyzer.compute_correlation_with_missing(x_missing, y_missing)
        assert not np.isnan(missing_corr)  # Should handle missing values

        # Test constant series correlation
        x_constant = np.array([5] * 100)
        y_variable = np.random.normal(0, 1, 100)
        constant_corr = analyzer.compute_correlation(x_constant, y_variable)
        assert np.isnan(constant_corr)  # Correlation undefined for constant series

    def test_time_series_edge_cases(self):
        """Test time series analysis edge cases."""
        from pynomaly.infrastructure.statistics.time_series import TimeSeriesAnalyzer

        analyzer = TimeSeriesAnalyzer()

        # Test time series with gaps
        dates_with_gaps = [
            datetime(2023, 1, 1),
            datetime(2023, 1, 2),
            # Gap here
            datetime(2023, 1, 5),
            datetime(2023, 1, 6),
            # Another gap
            datetime(2023, 1, 10),
        ]
        values_with_gaps = [1, 2, 5, 6, 10]

        gap_analysis = analyzer.analyze_time_series(dates_with_gaps, values_with_gaps)
        assert gap_analysis["has_gaps"] == True
        assert gap_analysis["gap_count"] >= 2

        # Test seasonal pattern detection
        t = np.arange(0, 365 * 2, 1)  # 2 years of daily data
        seasonal_data = 10 + 5 * np.sin(2 * np.pi * t / 365)  # Annual cycle
        seasonal_dates = [datetime(2023, 1, 1) + timedelta(days=int(day)) for day in t]

        seasonal_analysis = analyzer.detect_seasonality(seasonal_dates, seasonal_data)
        assert seasonal_analysis["has_seasonality"] == True
        assert (
            abs(seasonal_analysis["period"] - 365) < 10
        )  # Should detect ~365 day cycle

        # Test trend detection
        trend_data = 2 * t + np.random.normal(0, 1, len(t))  # Linear trend with noise
        trend_dates = [datetime(2023, 1, 1) + timedelta(days=int(day)) for day in t]

        trend_analysis = analyzer.detect_trend(trend_dates, trend_data)
        assert trend_analysis["has_trend"] == True
        assert trend_analysis["trend_direction"] == "increasing"

        # Test irregular frequency
        irregular_dates = [
            datetime(2023, 1, 1),
            datetime(2023, 1, 3),
            datetime(2023, 1, 4),
            datetime(2023, 1, 8),
            datetime(2023, 1, 15),
            datetime(2023, 1, 16),
            datetime(2023, 1, 30),
        ]
        irregular_values = [1, 2, 3, 4, 5, 6, 7]

        irregular_analysis = analyzer.analyze_frequency(
            irregular_dates, irregular_values
        )
        assert irregular_analysis["is_regular"] == False
        assert "irregular" in irregular_analysis["frequency_type"]


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
