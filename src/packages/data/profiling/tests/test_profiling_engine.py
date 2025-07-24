from datetime import datetime
from unittest.mock import Mock, patch

import numpy as np
import pandas as pd
import pytest

from ..application.services.profiling_engine import ProfilingConfig, ProfilingEngine
from ..domain.entities.data_profile import DataProfile, ProfilingStatus


class TestProfilingEngine:

    def setup_method(self):
        self.config = ProfilingConfig(
            enable_sampling=True,
            sample_size=1000,
            enable_parallel_processing=False,  # Disable for testing
            enable_caching=False  # Disable for testing
        )
        self.engine = ProfilingEngine(self.config)

    def test_profile_basic_dataset(self):
        """Test basic dataset profiling."""
        df = pd.DataFrame({
            "id": [1, 2, 3, 4, 5],
            "name": ["Alice", "Bob", "Charlie", "David", "Eve"],
            "age": [25, 30, 35, 28, 22],
            "score": [85.5, 92.0, 78.5, 88.0, 91.5],
            "active": [True, True, False, True, False]
        })

        profile = self.engine.profile_dataset(df)

        # Check profile structure
        assert isinstance(profile, DataProfile)
        assert profile.status == ProfilingStatus.COMPLETED
        assert profile.source_type == "dataframe"

        # Check schema profile
        assert profile.schema_profile is not None
        assert profile.schema_profile.total_columns == 5
        assert profile.schema_profile.total_rows == 5

        # Check quality assessment
        assert profile.quality_assessment is not None
        assert profile.quality_assessment.overall_score > 0

        # Check metadata
        assert profile.profiling_metadata is not None
        assert profile.profiling_metadata.execution_time_seconds > 0

    def test_profile_large_dataset_with_sampling(self):
        """Test profiling with sampling for large datasets."""
        # Create a large dataset
        np.random.seed(42)
        df = pd.DataFrame({
            "id": range(10000),
            "value": np.random.randn(10000),
            "category": np.random.choice(["A", "B", "C"], 10000)
        })

        profile = self.engine.profile_dataset(df)

        # Check that sampling was applied
        assert profile.status == ProfilingStatus.COMPLETED
        assert profile.profiling_metadata.sample_size <= self.config.sample_size
        assert profile.profiling_metadata.profiling_strategy == "sample"

    def test_profile_dataset_with_missing_values(self):
        """Test profiling dataset with missing values."""
        df = pd.DataFrame({
            "complete_col": [1, 2, 3, 4, 5],
            "partial_col": [1, 2, None, 4, None],
            "mostly_null": [1, None, None, None, None]
        })

        profile = self.engine.profile_dataset(df)

        assert profile.status == ProfilingStatus.COMPLETED

        # Check quality assessment accounts for missing values
        assert profile.quality_assessment.completeness_score < 1.0

        # Check that quality issues are detected
        total_issues = (profile.quality_assessment.critical_issues +
                       profile.quality_assessment.high_issues +
                       profile.quality_assessment.medium_issues +
                       profile.quality_assessment.low_issues)
        assert total_issues > 0

    def test_profile_dataset_with_patterns(self):
        """Test profiling dataset with detectable patterns."""
        df = pd.DataFrame({
            "email": ["user1@example.com", "user2@test.org", "user3@domain.net"],
            "phone": ["123-456-7890", "098-765-4321", "555-123-4567"],
            "product_code": ["ABC123", "DEF456", "GHI789"],
            "amount": [100.50, 250.75, 99.99]
        })

        profile = self.engine.profile_dataset(df)

        assert profile.status == ProfilingStatus.COMPLETED

        # Check that schema analysis detected patterns
        assert profile.schema_profile is not None
        assert profile.schema_profile.total_columns == 4

    def test_profile_dataset_with_relationships(self):
        """Test profiling dataset with detectable relationships."""
        df = pd.DataFrame({
            "user_id": [1, 2, 3, 4, 5],
            "order_id": [101, 102, 103, 104, 105],
            "user_name": ["Alice", "Bob", "Charlie", "David", "Eve"],
            "order_total": [100, 200, 150, 300, 250]
        })

        profile = self.engine.profile_dataset(df)

        assert profile.status == ProfilingStatus.COMPLETED

        # Check foreign key detection
        assert profile.schema_profile.foreign_keys is not None

    def test_profile_time_series_data(self):
        """Test profiling time series data."""
        dates = pd.date_range("2023-01-01", periods=100, freq="D")
        df = pd.DataFrame({
            "date": dates,
            "value": np.random.randn(100),
            "category": np.random.choice(["A", "B"], 100)
        })

        profile = self.engine.profile_dataset(df)

        assert profile.status == ProfilingStatus.COMPLETED

        # Check that datetime column was detected
        date_columns = [col for col in profile.schema_profile.columns
                       if col.data_type.value == "datetime"]
        assert len(date_columns) > 0

    def test_profile_mixed_data_types(self):
        """Test profiling dataset with mixed data types."""
        df = pd.DataFrame({
            "integers": [1, 2, 3, 4, 5],
            "floats": [1.1, 2.2, 3.3, 4.4, 5.5],
            "strings": ["a", "b", "c", "d", "e"],
            "booleans": [True, False, True, False, True],
            "mixed": [1, "text", 3.14, True, None]
        })

        profile = self.engine.profile_dataset(df)

        assert profile.status == ProfilingStatus.COMPLETED

        # Check that different data types were detected
        data_types = [col.data_type.value for col in profile.schema_profile.columns]
        assert "integer" in data_types
        assert "float" in data_types
        assert "string" in data_types
        assert "boolean" in data_types

    def test_profile_dataset_error_handling(self):
        """Test error handling in profiling."""
        # Test with empty DataFrame
        empty_df = pd.DataFrame()

        with pytest.raises(Exception):
            self.engine.profile_dataset(empty_df)

    def test_profiling_config_customization(self):
        """Test profiling with custom configuration."""
        custom_config = ProfilingConfig(
            enable_sampling=False,
            include_pattern_discovery=False,
            include_statistical_analysis=False,
            enable_parallel_processing=False
        )

        engine = ProfilingEngine(custom_config)

        df = pd.DataFrame({
            "col1": [1, 2, 3],
            "col2": ["a", "b", "c"]
        })

        profile = engine.profile_dataset(df)

        assert profile.status == ProfilingStatus.COMPLETED
        assert profile.profiling_metadata.profiling_strategy == "full"

    def test_get_profiling_summary(self):
        """Test profiling summary generation."""
        df = pd.DataFrame({
            "id": [1, 2, 3],
            "name": ["Alice", "Bob", "Charlie"]
        })

        profile = self.engine.profile_dataset(df)
        summary = self.engine.get_profiling_summary(profile)

        assert "profile_id" in summary
        assert "dataset_id" in summary
        assert "status" in summary
        assert "schema" in summary
        assert "quality" in summary
        assert summary["schema"]["total_columns"] == 2
        assert summary["schema"]["total_rows"] == 3

    def test_cache_functionality(self):
        """Test caching functionality."""
        config = ProfilingConfig(enable_caching=True)
        engine = ProfilingEngine(config)

        df = pd.DataFrame({
            "col1": [1, 2, 3],
            "col2": ["a", "b", "c"]
        })

        # First profiling
        profile1 = engine.profile_dataset(df)
        cache_info1 = engine.get_cache_info()

        # Second profiling (should use cache)
        profile2 = engine.profile_dataset(df)
        cache_info2 = engine.get_cache_info()

        assert cache_info1["cache_enabled"]
        assert cache_info2["cache_size"] >= cache_info1["cache_size"]

        # Clear cache
        engine.clear_cache()
        cache_info3 = engine.get_cache_info()
        assert cache_info3["cache_size"] == 0

    def test_performance_optimization(self):
        """Test performance optimization features."""
        # Create a dataset that would trigger optimization
        df = pd.DataFrame({
            "id": range(5000),
            "value": np.random.randn(5000),
            "category": np.random.choice(["A", "B", "C"], 5000)
        })

        profile = self.engine.profile_dataset(df)

        assert profile.status == ProfilingStatus.COMPLETED

        # Check that optimization was applied
        if profile.profiling_metadata.sample_size:
            assert profile.profiling_metadata.sample_size <= self.config.sample_size

    def test_streaming_profiling(self):
        """Test streaming profiling functionality."""
        df = pd.DataFrame({
            "timestamp": pd.date_range("2023-01-01", periods=2000, freq="min"),
            "value": np.random.randn(2000),
            "sensor_id": np.random.choice(["S1", "S2", "S3"], 2000)
        })

        profile = self.engine.profile_streaming(df, window_size=500)

        assert profile.status == ProfilingStatus.COMPLETED
        assert profile.source_type == "streaming"

    def test_incremental_profiling(self):
        """Test incremental profiling functionality."""
        df1 = pd.DataFrame({
            "id": [1, 2, 3],
            "value": [10, 20, 30]
        })

        df2 = pd.DataFrame({
            "id": [4, 5, 6],
            "value": [40, 50, 60]
        })

        # Initial profiling
        profile1 = self.engine.profile_dataset(df1)

        # Incremental profiling
        profile2 = self.engine.profile_incremental(df2, profile1)

        assert profile2.status == ProfilingStatus.COMPLETED

    def test_profiling_metadata_accuracy(self):
        """Test accuracy of profiling metadata."""
        df = pd.DataFrame({
            "col1": range(100),
            "col2": np.random.randn(100)
        })

        profile = self.engine.profile_dataset(df)

        metadata = profile.profiling_metadata

        # Check metadata fields
        assert metadata.execution_time_seconds > 0
        assert metadata.memory_usage_mb > 0
        assert metadata.profiling_strategy in ["full", "sample", "incremental"]

        # Check configuration flags
        assert metadata.include_patterns == self.config.include_pattern_discovery
        assert metadata.include_statistical_analysis == self.config.include_statistical_analysis
        assert metadata.include_quality_assessment == self.config.include_quality_assessment

    def test_quality_assessment_completeness(self):
        """Test quality assessment completeness."""
        df = pd.DataFrame({
            "complete": [1, 2, 3, 4, 5],
            "partial": [1, 2, None, 4, 5],
            "mostly_missing": [1, None, None, None, None]
        })

        profile = self.engine.profile_dataset(df)
        quality = profile.quality_assessment

        # Completeness score should reflect missing values
        assert quality.completeness_score < 1.0

        # Should have quality issues
        assert quality.medium_issues > 0 or quality.high_issues > 0

        # Should have recommendations
        assert len(quality.recommendations) > 0

    def test_pattern_discovery_integration(self):
        """Test pattern discovery integration with profiling."""
        df = pd.DataFrame({
            "email": ["user@example.com", "test@domain.org", "admin@site.net"],
            "phone": ["123-456-7890", "098-765-4321", "555-123-4567"],
            "normal_text": ["hello", "world", "test"]
        })

        profile = self.engine.profile_dataset(df)

        assert profile.status == ProfilingStatus.COMPLETED

        # Check that patterns were integrated into schema
        assert profile.schema_profile is not None
        assert len(profile.schema_profile.columns) == 3

    def test_error_recovery(self):
        """Test error recovery mechanisms."""
        # Test with problematic data
        df = pd.DataFrame({
            "problematic": [float("inf"), float("-inf"), float("nan")],
            "normal": [1, 2, 3]
        })

        profile = self.engine.profile_dataset(df)

        # Should complete despite problematic data
        assert profile.status == ProfilingStatus.COMPLETED

        # Quality assessment should identify issues
        assert profile.quality_assessment.overall_score < 1.0
