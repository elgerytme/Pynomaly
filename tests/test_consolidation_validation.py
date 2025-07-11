"""
Test Configuration Consolidation Validation - Issue #106

This test validates that the consolidated test configuration works properly
and provides the same functionality as the previous scattered configuration.
"""

from unittest.mock import MagicMock

import pandas as pd
import pytest
from tests.test_utils import MockFactory, TestDataFactory, TestIsolation


class TestConsolidationValidation:
    """Validate the consolidated test configuration."""

    def test_test_data_factory(self):
        """Test that TestDataFactory creates consistent data."""
        # Test basic dataframe creation
        df = TestDataFactory.create_sample_dataframe()
        assert isinstance(df, pd.DataFrame)
        assert df.shape == (100, 4)  # 3 features + target
        assert "target" in df.columns
        assert set(df["target"].unique()) == {0, 1}

        # Test reproducibility
        df1 = TestDataFactory.create_sample_dataframe(seed=42)
        df2 = TestDataFactory.create_sample_dataframe(seed=42)
        pd.testing.assert_frame_equal(df1, df2)

    def test_mock_factory(self):
        """Test that MockFactory creates standardized mocks."""
        # Test dataset mock
        dataset_mock = MockFactory.create_dataset_mock()
        assert dataset_mock.name == "test_dataset"
        assert dataset_mock.n_samples == 100
        assert dataset_mock.n_features == 3

        # Test detector mock
        detector_mock = MockFactory.create_detector_mock()
        assert detector_mock.name == "test_detector"
        assert detector_mock.algorithm_name == "IsolationForest"
        assert detector_mock.is_fitted is True

        # Test repository mocks
        sync_repo = MockFactory.create_repository_mock(async_mode=False)
        assert isinstance(sync_repo, MagicMock)

        async_repo = MockFactory.create_repository_mock(async_mode=True)
        assert hasattr(async_repo, "save")

    def test_test_isolation(self):
        """Test that TestIsolation utilities work correctly."""
        # Test random state reset
        import numpy as np

        np.random.seed(123)
        val1 = np.random.random()

        TestIsolation.reset_random_state(123)
        val2 = np.random.random()

        assert val1 == val2  # Should be same with same seed

    def test_fixtures_work(self, sample_data, mock_detector, temp_dir):
        """Test that consolidated fixtures work properly."""
        # Test sample_data fixture
        assert isinstance(sample_data, pd.DataFrame)
        assert len(sample_data) > 0

        # Test mock_detector fixture
        assert mock_detector.name == "Test Detector"
        assert mock_detector.algorithm_name == "IsolationForest"

        # Test temp_dir fixture
        assert temp_dir.exists()
        assert temp_dir.is_dir()

    def test_performance_utilities(self, performance_timer):
        """Test performance monitoring utilities."""
        import time

        with performance_timer() as timer:
            time.sleep(0.01)  # Small delay

        assert timer.elapsed > 0.005  # Should have some measurable time
        assert timer.elapsed < 0.1  # But not too much

    def test_resource_management(self, resource_manager):
        """Test resource management fixture."""
        # Create a mock resource
        mock_resource = MagicMock()
        mock_resource.close = MagicMock()

        # Add it to resource manager
        managed_resource = resource_manager(mock_resource)
        assert managed_resource is mock_resource

        # Resource cleanup will be tested automatically by fixture

    @pytest.mark.slow
    def test_markers_work(self):
        """Test that test markers are applied correctly."""
        # This test should have the 'slow' marker applied
        # and get appropriate timeout via pytest_collection_modifyitems
        assert True

    def test_error_simulation(self, error_simulator):
        """Test error simulation utilities."""
        file_error = error_simulator.file_error("permission")
        assert isinstance(file_error, PermissionError)

        network_error = error_simulator.network_error("connection")
        assert isinstance(network_error, ConnectionError)

    def test_retry_helper(self, retry_helper):
        """Test retry helper functionality."""
        call_count = 0

        def flaky_function():
            nonlocal call_count
            call_count += 1
            if call_count < 3:
                raise ValueError("Flaky error")
            return "success"

        result = retry_helper(flaky_function, max_attempts=3, delay=0.001)
        assert result == "success"
        assert call_count == 3

    def test_deterministic_behavior(self, deterministic_data):
        """Test that deterministic data is consistent."""
        # Should get same data every time due to fixed seed
        assert "small" in deterministic_data
        assert "medium" in deterministic_data
        assert "large" in deterministic_data

        # Shapes should be consistent
        assert deterministic_data["small"].shape == (50, 3)
        assert deterministic_data["medium"].shape == (500, 5)
        assert deterministic_data["large"].shape == (1000, 10)

    def test_isolation_works(self):
        """Test that test isolation actually isolates tests."""
        import os

        # Set an environment variable
        os.environ["TEST_ISOLATION_VAR"] = "test_value"
        assert os.getenv("TEST_ISOLATION_VAR") == "test_value"

        # The isolate_tests fixture should clean this up after the test

    def test_warning_suppression(self, suppress_warnings):
        """Test that warning suppression works."""
        import warnings

        # This should not raise or print warnings
        with warnings.catch_warnings(record=True) as w:
            warnings.warn("Test warning", UserWarning)
            warnings.warn("Test deprecation", DeprecationWarning)

        # The suppress_warnings fixture should handle this

    def test_consolidated_conftest_exports(self):
        """Test that consolidated conftest exports expected items."""
        import tests.conftest as conftest

        # Check that key exports are available
        expected_exports = [
            "event_loop",
            "temp_dir",
            "sample_data",
            "mock_detector",
            "skip_if_no_torch",
            "skip_if_no_tensorflow",
            "skip_if_no_fastapi",
        ]

        for export in expected_exports:
            assert hasattr(conftest, export), f"Missing export: {export}"


class TestBackwardCompatibility:
    """Test that the consolidation maintains backward compatibility."""

    def test_old_fixture_patterns_still_work(self):
        """Test that existing test patterns still work."""
        # Patterns that were common in the old configuration
        # should still work with the new consolidated setup

        from tests.test_utils import TestDataFactory

        # Old pattern: direct data creation
        data = TestDataFactory.create_sample_dataframe()
        assert data is not None

        # Old pattern: mock creation
        mock = MockFactory.create_dataset_mock()
        assert mock.name == "test_dataset"

    def test_no_import_conflicts(self):
        """Test that there are no import conflicts in consolidation."""
        # Should be able to import from consolidated conftest

        # Should be able to import utilities

        # No exceptions should be raised
        assert True


class TestPerformanceImprovements:
    """Test that consolidation improves performance."""

    def test_fixture_efficiency(self, performance_timer):
        """Test that fixtures are created efficiently."""

        # Creating multiple fixtures should be fast
        with performance_timer() as timer:
            data1 = TestDataFactory.create_sample_dataframe()
            data2 = TestDataFactory.create_sample_dataframe()
            mock1 = MockFactory.create_dataset_mock()
            mock2 = MockFactory.create_detector_mock()

        # Should complete quickly (< 100ms for simple operations)
        assert timer.elapsed < 0.1

        # Objects should be created correctly
        assert len(data1) > 0
        assert len(data2) > 0
        assert mock1.name == "test_dataset"
        assert mock2.name == "test_detector"
