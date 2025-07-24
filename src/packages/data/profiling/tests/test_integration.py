import os
import tempfile
from datetime import datetime, timedelta

import numpy as np
import pandas as pd
import pytest

from ..application.services.pattern_discovery_service import PatternDiscoveryService
from ..application.services.profiling_engine import ProfilingConfig, ProfilingEngine
from ..application.services.quality_assessment_service import QualityAssessmentService
from ..application.services.schema_analysis_service import SchemaAnalysisService
from ..application.services.statistical_profiling_service import (
    StatisticalProfilingService,
)
from ..domain.entities.data_profile import DataProfile, ProfilingStatus
from ..infrastructure.adapters.file_adapter import FileAdapter


class TestDataProfilingIntegration:
    """Integration tests for the complete data profiling package."""

    def setup_method(self):
        self.config = ProfilingConfig(
            enable_sampling=True,
            sample_size=1000,
            enable_parallel_processing=True,
            max_workers=2,
            enable_caching=True
        )
        self.engine = ProfilingEngine(self.config)

    def test_complete_profiling_workflow(self):
        """Test complete profiling workflow from data loading to reporting."""
        # Create comprehensive test dataset
        np.random.seed(42)
        n_rows = 5000

        df = pd.DataFrame({
            "user_id": range(1, n_rows + 1),
            "email": [f"user{i}@example.com" for i in range(1, n_rows + 1)],
            "age": np.random.randint(18, 80, n_rows),
            "salary": np.random.normal(50000, 15000, n_rows),
            "department": np.random.choice(["Engineering", "Sales", "Marketing", "HR"], n_rows),
            "hire_date": pd.date_range("2020-01-01", periods=n_rows, freq="D")[:n_rows],
            "active": np.random.choice([True, False], n_rows, p=[0.8, 0.2]),
            "score": np.random.uniform(0, 100, n_rows),
            "phone": [f"555-{np.random.randint(100, 999):03d}-{np.random.randint(1000, 9999):04d}" for _ in range(n_rows)],
            "product_code": [f"PROD{np.random.randint(100, 999):03d}" for _ in range(n_rows)]
        })

        # Add some quality issues
        df.loc[df.sample(frac=0.1).index, "email"] = None  # 10% missing emails
        df.loc[df.sample(frac=0.05).index, "age"] = -1     # 5% invalid ages
        df.loc[df.sample(frac=0.02).index, "salary"] = df.loc[df.sample(frac=0.02).index, "salary"] * 10  # Outliers

        # Run complete profiling
        profile = self.engine.profile_dataset(df, dataset_id="test_dataset")

        # Verify profiling completed successfully
        assert profile.status == ProfilingStatus.COMPLETED
        assert profile.dataset_id is not None
        assert profile.profile_id is not None

        # Verify schema analysis
        schema = profile.schema_profile
        assert schema is not None
        assert schema.total_columns == 10
        assert schema.total_rows == n_rows
        assert len(schema.columns) == 10

        # Verify data types were inferred correctly
        column_types = {col.column_name: col.data_type.value for col in schema.columns}
        assert column_types["user_id"] == "integer"
        assert column_types["email"] == "string"
        assert column_types["age"] == "integer"
        assert column_types["salary"] == "float"
        assert column_types["hire_date"] == "datetime"
        assert column_types["active"] == "boolean"

        # Verify quality assessment
        quality = profile.quality_assessment
        assert quality is not None
        assert quality.overall_score > 0
        assert quality.completeness_score < 1.0  # Due to missing emails

        # Verify quality issues were detected
        total_issues = (quality.critical_issues + quality.high_issues +
                       quality.medium_issues + quality.low_issues)
        assert total_issues > 0

        # Verify recommendations were generated
        assert len(quality.recommendations) > 0

        # Verify metadata
        metadata = profile.profiling_metadata
        assert metadata is not None
        assert metadata.execution_time_seconds > 0
        assert metadata.memory_usage_mb > 0
        assert metadata.sample_size <= self.config.sample_size

        # Test profiling summary
        summary = self.engine.get_profiling_summary(profile)
        assert summary["schema"]["total_columns"] == 10
        assert summary["quality"]["overall_score"] > 0

    def test_file_adapter_integration(self):
        """Test integration with file adapter."""
        # Create test data
        df = pd.DataFrame({
            "id": [1, 2, 3, 4, 5],
            "name": ["Alice", "Bob", "Charlie", "David", "Eve"],
            "value": [10.5, 20.3, 30.1, 40.7, 50.2]
        })

        # Save to temporary CSV file
        with tempfile.NamedTemporaryFile(mode="w", suffix=".csv", delete=False) as f:
            df.to_csv(f.name, index=False)
            temp_file = f.name

        try:
            # Load using file adapter
            adapter = FileAdapter(temp_file)
            loaded_df = adapter.load_data()

            # Profile the loaded data
            profile = self.engine.profile_dataset(loaded_df, source_type="file")

            assert profile.status == ProfilingStatus.COMPLETED
            assert profile.source_type == "file"
            assert profile.schema_profile.total_columns == 3
            assert profile.schema_profile.total_rows == 5

        finally:
            # Clean up
            os.unlink(temp_file)

    def test_advanced_pattern_discovery(self):
        """Test advanced pattern discovery capabilities."""
        df = pd.DataFrame({
            "emails": ["john.doe@company.com", "jane.smith@org.net", "bob@test.co.uk"],
            "phones": ["(555) 123-4567", "555-987-6543", "+1-555-111-2222"],
            "credit_cards": ["4532-1234-5678-9012", "5555-4444-3333-2222", "4111-1111-1111-1111"],
            "product_codes": ["ABC-123", "DEF-456", "GHI-789"],
            "currencies": ["$123.45", "$67.89", "$999.99"],
            "dates": ["2023-01-15", "2023-02-20", "2023-03-25"],
            "urls": ["https://example.com", "http://test.org", "https://domain.co.uk"]
        })

        profile = self.engine.profile_dataset(df)

        assert profile.status == ProfilingStatus.COMPLETED

        # The pattern discovery should identify various patterns
        # This would be verified through the integrated patterns in schema
        assert profile.schema_profile is not None
        assert len(profile.schema_profile.columns) == 7

    def test_statistical_analysis_integration(self):
        """Test statistical analysis integration."""
        # Create data with interesting statistical properties
        np.random.seed(42)
        df = pd.DataFrame({
            "normal_dist": np.random.normal(100, 15, 1000),
            "skewed_dist": np.random.exponential(2, 1000),
            "uniform_dist": np.random.uniform(0, 100, 1000),
            "correlated_var": np.random.normal(100, 15, 1000)
        })

        # Make variables correlated
        df["correlated_var"] = df["normal_dist"] * 0.8 + np.random.normal(0, 5, 1000)

        # Add outliers
        df.loc[df.sample(20).index, "normal_dist"] = 1000

        profile = self.engine.profile_dataset(df)

        assert profile.status == ProfilingStatus.COMPLETED

        # Check that statistical analysis was performed
        for column in profile.schema_profile.columns:
            if column.statistical_summary is not None:
                assert column.statistical_summary.min_value is not None
                assert column.statistical_summary.max_value is not None
                assert column.statistical_summary.mean is not None
                assert column.statistical_summary.std_dev is not None

    def test_quality_assessment_integration(self):
        """Test quality assessment integration."""
        df = pd.DataFrame({
            "complete_column": [1, 2, 3, 4, 5] * 200,
            "partial_column": [1, 2, None, 4, 5] * 200,
            "mostly_null": [1] + [None] * 999,
            "duplicated_values": [1, 1, 1, 1, 1] * 200,
            "outlier_column": [1, 2, 3, 4, 1000] * 200,
            "inconsistent_format": ["2023-01-01", "01/02/2023", "2023-03-03", "04/05/2023", "2023-05-05"] * 200
        })

        profile = self.engine.profile_dataset(df)

        assert profile.status == ProfilingStatus.COMPLETED

        quality = profile.quality_assessment
        assert quality is not None

        # Should detect completeness issues
        assert quality.completeness_score < 1.0

        # Should detect quality issues
        assert quality.medium_issues > 0 or quality.high_issues > 0

        # Should provide recommendations
        assert len(quality.recommendations) > 0

    def test_performance_with_large_dataset(self):
        """Test performance with large dataset."""
        # Create large dataset
        np.random.seed(42)
        n_rows = 50000

        df = pd.DataFrame({
            "id": range(n_rows),
            "value1": np.random.randn(n_rows),
            "value2": np.random.randn(n_rows),
            "category": np.random.choice(["A", "B", "C", "D"], n_rows),
            "timestamp": pd.date_range("2023-01-01", periods=n_rows, freq="min")
        })

        # Profile with sampling
        profile = self.engine.profile_dataset(df)

        assert profile.status == ProfilingStatus.COMPLETED
        assert profile.profiling_metadata.execution_time_seconds < 60  # Should complete in under 1 minute

        # Check that sampling was applied
        if profile.profiling_metadata.sample_size:
            assert profile.profiling_metadata.sample_size <= self.config.sample_size

    def test_error_handling_integration(self):
        """Test error handling across the system."""
        # Test with problematic data
        df = pd.DataFrame({
            "inf_values": [1, 2, float("inf"), 4, 5],
            "nan_values": [1, 2, float("nan"), 4, 5],
            "mixed_types": [1, "text", 3.14, True, None],
            "empty_strings": ["", "valid", "", "data", ""]
        })

        profile = self.engine.profile_dataset(df)

        # Should complete despite problematic data
        assert profile.status == ProfilingStatus.COMPLETED

        # Quality assessment should identify issues
        assert profile.quality_assessment.overall_score < 1.0

    def test_caching_performance(self):
        """Test caching performance benefits."""
        df = pd.DataFrame({
            "col1": range(1000),
            "col2": np.random.randn(1000),
            "col3": np.random.choice(["A", "B", "C"], 1000)
        })

        # First profiling (populate cache)
        start_time = datetime.now()
        profile1 = self.engine.profile_dataset(df)
        first_duration = (datetime.now() - start_time).total_seconds()

        # Second profiling (should use cache)
        start_time = datetime.now()
        profile2 = self.engine.profile_dataset(df)
        second_duration = (datetime.now() - start_time).total_seconds()

        assert profile1.status == ProfilingStatus.COMPLETED
        assert profile2.status == ProfilingStatus.COMPLETED

        # Cache should improve performance (though this might be minimal for small datasets)
        cache_info = self.engine.get_cache_info()
        assert cache_info["cache_enabled"]
        assert cache_info["cache_size"] > 0

    def test_concurrent_profiling(self):
        """Test concurrent profiling operations."""
        import threading

        # Create different datasets
        datasets = []
        for i in range(3):
            df = pd.DataFrame({
                "id": range(i * 100, (i + 1) * 100),
                "value": np.random.randn(100),
                "category": np.random.choice(["X", "Y", "Z"], 100)
            })
            datasets.append(df)

        profiles = []
        errors = []

        def profile_dataset(df):
            try:
                profile = self.engine.profile_dataset(df)
                profiles.append(profile)
            except Exception as e:
                errors.append(e)

        # Start concurrent profiling
        threads = []
        for df in datasets:
            thread = threading.Thread(target=profile_dataset, args=(df,))
            threads.append(thread)
            thread.start()

        # Wait for all threads to complete
        for thread in threads:
            thread.join()

        # Check results
        assert len(errors) == 0
        assert len(profiles) == 3

        for profile in profiles:
            assert profile.status == ProfilingStatus.COMPLETED

    def test_memory_management(self):
        """Test memory management with large datasets."""
        import gc
        import os

        import psutil

        # Get initial memory usage
        process = psutil.Process(os.getpid())
        initial_memory = process.memory_info().rss / 1024 / 1024  # MB

        # Create and profile large dataset
        df = pd.DataFrame({
            "col1": range(10000),
            "col2": np.random.randn(10000),
            "col3": np.random.choice(["A", "B", "C"], 10000),
            "col4": [f"text_{i}" for i in range(10000)]
        })

        profile = self.engine.profile_dataset(df)

        # Force garbage collection
        del df
        gc.collect()

        # Check memory usage after cleanup
        final_memory = process.memory_info().rss / 1024 / 1024  # MB
        memory_increase = final_memory - initial_memory

        assert profile.status == ProfilingStatus.COMPLETED
        assert memory_increase < 500  # Should not increase memory by more than 500MB

    def test_comprehensive_reporting(self):
        """Test comprehensive reporting functionality."""
        df = pd.DataFrame({
            "user_id": [1, 2, 3, 4, 5],
            "email": ["user1@test.com", "user2@test.com", None, "invalid-email", "user5@test.com"],
            "age": [25, 30, -5, 150, 35],  # Invalid ages
            "salary": [50000, 60000, 70000, 80000, 999999],  # One outlier
            "department": ["IT", "Sales", "IT", "HR", "Sales"]
        })

        profile = self.engine.profile_dataset(df)

        assert profile.status == ProfilingStatus.COMPLETED

        # Generate comprehensive report
        summary = self.engine.get_profiling_summary(profile)

        # Verify report structure
        assert "profile_id" in summary
        assert "dataset_id" in summary
        assert "schema" in summary
        assert "quality" in summary

        # Verify schema information
        schema_info = summary["schema"]
        assert schema_info["total_columns"] == 5
        assert schema_info["total_rows"] == 5

        # Verify quality information
        quality_info = summary["quality"]
        assert quality_info["overall_score"] >= 0
        assert quality_info["overall_score"] <= 1
        assert quality_info["total_issues"] > 0  # Should detect issues

        # Test quality assessment report
        quality_service = QualityAssessmentService()
        quality_report = quality_service.generate_quality_report(
            profile.quality_assessment,
            profile.schema_profile
        )

        assert "overall_quality" in quality_report
        assert "dimension_scores" in quality_report
        assert "issue_summary" in quality_report
        assert "recommendations" in quality_report
        assert "column_quality" in quality_report
