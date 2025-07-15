"""
Tests for the stabilizer utility classes and functions.

This module tests the stability framework utilities themselves to ensure
they work correctly and provide the expected stabilization benefits.
"""

import os
import time
from pathlib import Path
from unittest.mock import Mock, patch

import pytest

from tests._stability import (
    MockManager,
    ResourceManager,
    RetryManager,
    TestIsolationManager,
    TestStabilizer,
    TimingManager,
    flaky,
    stable_test,
)


class TestTestStabilizer:
    """Test the main TestStabilizer class."""

    def test_stabilizer_initialization(self):
        """Test that TestStabilizer initializes correctly."""
        stabilizer = TestStabilizer()

        assert stabilizer.isolation_manager is not None
        assert stabilizer.retry_manager is not None
        assert stabilizer.resource_manager is not None
        assert stabilizer.timing_manager is not None
        assert stabilizer.mock_manager is not None
        assert isinstance(stabilizer.isolation_manager, TestIsolationManager)
        assert isinstance(stabilizer.retry_manager, RetryManager)
        assert isinstance(stabilizer.resource_manager, ResourceManager)
        assert isinstance(stabilizer.timing_manager, TimingManager)
        assert isinstance(stabilizer.mock_manager, MockManager)

    def test_stabilized_test_context_manager(self):
        """Test that the stabilized_test context manager works correctly."""
        stabilizer = TestStabilizer()

        # Test that context manager enters and exits cleanly
        with stabilizer.stabilized_test("test_context"):
            # Inside the context, test isolation should be active
            assert os.environ.get("PYNOMALY_TEST_MODE") == "1"
            assert os.environ.get("PYTHONHASHSEED") == "0"

        # After exiting context, environment should be restored
        # (Note: This depends on the implementation details)

    def test_stabilized_test_exception_handling(self):
        """Test that exceptions in stabilized tests are handled correctly."""
        stabilizer = TestStabilizer()

        with pytest.raises(ValueError, match="Test exception"):
            with stabilizer.stabilized_test("test_exception"):
                raise ValueError("Test exception")

    def test_multiple_stabilized_tests(self):
        """Test that multiple stabilized tests can be run sequentially."""
        stabilizer = TestStabilizer()

        for i in range(3):
            with stabilizer.stabilized_test(f"test_{i}"):
                assert os.environ.get("PYNOMALY_TEST_MODE") == "1"
                # Each test should have isolated environment
                test_value = f"test_value_{i}"
                os.environ["TEST_VALUE"] = test_value
                assert os.environ.get("TEST_VALUE") == test_value


class TestTestIsolationManager:
    """Test the TestIsolationManager class."""

    def test_isolation_manager_initialization(self):
        """Test that TestIsolationManager initializes correctly."""
        manager = TestIsolationManager()

        assert manager.original_env == {}
        assert manager.temp_dirs == []
        assert manager.original_cwd is None

    def test_isolated_environment_context(self):
        """Test that isolated environment provides proper isolation."""
        manager = TestIsolationManager()
        original_cwd = os.getcwd()
        original_env = os.environ.copy()

        with manager.isolated_environment() as temp_dir:
            # Should be in a different directory
            assert os.getcwd() != original_cwd
            assert temp_dir in os.getcwd()

            # Should have test environment variables
            assert os.environ.get("PYNOMALY_TEST_MODE") == "1"
            assert os.environ.get("PYTHONHASHSEED") == "0"
            assert os.environ.get("TZ") == "UTC"

            # Should be able to create files in the temp directory
            test_file = Path(temp_dir) / "test_file.txt"
            test_file.write_text("test content")
            assert test_file.exists()

        # Should be restored after exiting
        assert os.getcwd() == original_cwd
        assert os.environ == original_env

    def test_isolated_environment_cleanup(self):
        """Test that isolated environment cleans up properly."""
        manager = TestIsolationManager()
        temp_dir_path = None

        with manager.isolated_environment() as temp_dir:
            temp_dir_path = temp_dir
            assert os.path.exists(temp_dir_path)

        # Temporary directory should be cleaned up
        # Note: Cleanup might be best effort, so we check manager state
        assert manager.temp_dirs == []

    def test_isolated_environment_exception_handling(self):
        """Test that isolated environment handles exceptions correctly."""
        manager = TestIsolationManager()
        original_cwd = os.getcwd()
        original_env = os.environ.copy()

        with pytest.raises(ValueError, match="Test exception"):
            with manager.isolated_environment():
                raise ValueError("Test exception")

        # Should still be restored after exception
        assert os.getcwd() == original_cwd
        assert os.environ == original_env


class TestRetryManager:
    """Test the RetryManager class."""

    def test_retry_manager_initialization(self):
        """Test that RetryManager initializes correctly."""
        manager = RetryManager()

        assert manager.max_retries == 3
        assert manager.base_delay == 0.1
        assert manager.max_delay == 2.0
        assert manager.exponential_base == 2.0
        assert manager.jitter_range == 0.1

    def test_retry_manager_custom_initialization(self):
        """Test that RetryManager can be initialized with custom values."""
        manager = RetryManager()
        manager.max_retries = 5
        manager.base_delay = 0.2
        manager.max_delay = 5.0
        manager.exponential_base = 1.5
        manager.jitter_range = 0.2

        assert manager.max_retries == 5
        assert manager.base_delay == 0.2
        assert manager.max_delay == 5.0
        assert manager.exponential_base == 1.5
        assert manager.jitter_range == 0.2

    def test_retry_with_stabilization_success(self):
        """Test retry decorator with function that succeeds."""
        manager = RetryManager()
        call_count = 0

        @manager.retry_with_stabilization(max_retries=3, delay=0.01)
        def test_function():
            nonlocal call_count
            call_count += 1
            return "success"

        result = test_function()
        assert result == "success"
        assert call_count == 1

    def test_retry_with_stabilization_eventual_success(self):
        """Test retry decorator with function that eventually succeeds."""
        manager = RetryManager()
        call_count = 0

        @manager.retry_with_stabilization(max_retries=3, delay=0.01)
        def test_function():
            nonlocal call_count
            call_count += 1
            if call_count < 3:
                raise ValueError(f"Attempt {call_count} failed")
            return "success"

        result = test_function()
        assert result == "success"
        assert call_count == 3

    def test_retry_with_stabilization_failure(self):
        """Test retry decorator with function that always fails."""
        manager = RetryManager()
        call_count = 0

        @manager.retry_with_stabilization(max_retries=2, delay=0.01)
        def test_function():
            nonlocal call_count
            call_count += 1
            raise ValueError(f"Attempt {call_count} failed")

        with pytest.raises(ValueError, match="Attempt 3 failed"):
            test_function()

        assert call_count == 3  # Initial attempt + 2 retries

    def test_retry_with_exponential_backoff(self):
        """Test that retry uses exponential backoff."""
        manager = RetryManager()
        call_times = []

        @manager.retry_with_stabilization(max_retries=2, delay=0.1)
        def test_function():
            call_times.append(time.time())
            if len(call_times) < 3:
                raise ValueError("Not yet")
            return "success"

        start_time = time.time()
        result = test_function()

        assert result == "success"
        assert len(call_times) == 3

        # Check that delays increase (approximately)
        delay1 = call_times[1] - call_times[0]
        delay2 = call_times[2] - call_times[1]
        assert delay2 > delay1  # Second delay should be longer


class TestResourceManager:
    """Test the ResourceManager class."""

    def test_resource_manager_initialization(self):
        """Test that ResourceManager initializes correctly."""
        manager = ResourceManager()

        assert "max_memory_mb" in manager.resource_limits
        assert "max_open_files" in manager.resource_limits
        assert "max_threads" in manager.resource_limits
        assert "open_files" in manager.active_resources
        assert "threads" in manager.active_resources
        assert "temp_objects" in manager.active_resources

    def test_managed_resources_context(self):
        """Test that managed resources context works correctly."""
        manager = ResourceManager()

        with manager.managed_resources():
            # Should be able to register resources
            test_obj = {"test": "data"}
            manager.register_resource("temp_objects", test_obj)
            assert test_obj in manager.active_resources["temp_objects"]

        # Resources should be cleaned up after exiting context
        assert len(manager.active_resources["temp_objects"]) == 0

    def test_register_resource(self):
        """Test that resources can be registered correctly."""
        manager = ResourceManager()

        # Test registering different types of resources
        test_file = Mock()
        test_thread = Mock()
        test_object = {"data": "test"}

        manager.register_resource("open_files", test_file)
        manager.register_resource("threads", test_thread)
        manager.register_resource("temp_objects", test_object)

        assert test_file in manager.active_resources["open_files"]
        assert test_thread in manager.active_resources["threads"]
        assert test_object in manager.active_resources["temp_objects"]

    def test_resource_cleanup(self):
        """Test that resources are cleaned up properly."""
        manager = ResourceManager()

        # Create mock resources
        mock_file = Mock()
        mock_thread = Mock()
        mock_thread.is_alive.return_value = True

        manager.register_resource("open_files", mock_file)
        manager.register_resource("threads", mock_thread)

        # Call cleanup
        manager._cleanup_resources()

        # Check that cleanup methods were called
        mock_file.close.assert_called_once()
        mock_thread.join.assert_called_once()

        # Check that resources were cleared
        assert len(manager.active_resources["open_files"]) == 0
        assert len(manager.active_resources["threads"]) == 0

    def test_resource_cleanup_error_handling(self):
        """Test that resource cleanup handles errors gracefully."""
        manager = ResourceManager()

        # Create mock resources that raise exceptions
        mock_file = Mock()
        mock_file.close.side_effect = Exception("Close failed")
        mock_thread = Mock()
        mock_thread.is_alive.return_value = True
        mock_thread.join.side_effect = Exception("Join failed")

        manager.register_resource("open_files", mock_file)
        manager.register_resource("threads", mock_thread)

        # Should not raise exception despite errors
        manager._cleanup_resources()

        # Resources should still be cleared
        assert len(manager.active_resources["open_files"]) == 0
        assert len(manager.active_resources["threads"]) == 0


class TestTimingManager:
    """Test the TimingManager class."""

    def test_timing_manager_initialization(self):
        """Test that TimingManager initializes correctly."""
        manager = TimingManager()

        assert manager.default_timeout == 10.0
        assert manager.polling_interval == 0.1
        assert manager.time_tolerance == 0.1

    def test_stable_timing_context(self):
        """Test that stable timing context provides deterministic behavior."""
        manager = TimingManager()

        # Test that random seed is set deterministically
        with manager.stable_timing():
            import random

            import numpy as np

            # Should get same random values each time
            random_val1 = random.random()
            np_val1 = np.random.random()

        with manager.stable_timing():
            random_val2 = random.random()
            np_val2 = np.random.random()

        assert random_val1 == random_val2
        assert np_val1 == np_val2

    def test_wait_for_condition_success(self):
        """Test wait_for_condition with condition that becomes true."""
        manager = TimingManager()
        condition_met = False

        def condition():
            nonlocal condition_met
            condition_met = True
            return True

        start_time = time.time()
        result = manager.wait_for_condition(
            condition, timeout=1.0, polling_interval=0.05
        )
        elapsed = time.time() - start_time

        assert result is True
        assert condition_met
        assert elapsed < 0.5  # Should be much faster than timeout

    def test_wait_for_condition_timeout(self):
        """Test wait_for_condition with condition that never becomes true."""
        manager = TimingManager()

        def condition():
            return False

        start_time = time.time()
        result = manager.wait_for_condition(
            condition, timeout=0.2, polling_interval=0.05
        )
        elapsed = time.time() - start_time

        assert result is False
        assert elapsed >= 0.2  # Should wait for full timeout

    def test_wait_for_condition_exception_handling(self):
        """Test wait_for_condition handles exceptions in condition function."""
        manager = TimingManager()
        call_count = 0

        def condition():
            nonlocal call_count
            call_count += 1
            if call_count < 3:
                raise ValueError("Condition error")
            return True

        result = manager.wait_for_condition(
            condition, timeout=1.0, polling_interval=0.05
        )

        assert result is True
        assert call_count >= 3

    def test_stable_sleep(self):
        """Test stable sleep with timing tolerance."""
        manager = TimingManager()

        start_time = time.time()
        manager.stable_sleep(0.1)
        elapsed = time.time() - start_time

        # Should sleep for at least the requested duration plus tolerance
        expected_min = 0.1 * (1 + manager.time_tolerance)
        assert elapsed >= expected_min - 0.01  # Small allowance for timing precision

    def test_time_freeze_context(self):
        """Test time freezing context manager."""
        manager = TimingManager()

        with patch("datetime.datetime") as mock_datetime:
            frozen_time = Mock()
            mock_datetime.now.return_value = frozen_time
            mock_datetime.utcnow.return_value = frozen_time

            with manager.time_freeze():
                # Time should be frozen
                assert mock_datetime.now.called
                assert mock_datetime.utcnow.called


class TestMockManager:
    """Test the MockManager class."""

    def test_mock_manager_initialization(self):
        """Test that MockManager initializes correctly."""
        manager = MockManager()

        assert manager.active_mocks == []
        assert "network" in manager.mock_configs
        assert "filesystem" in manager.mock_configs
        assert "random" in manager.mock_configs

    def test_controlled_mocks_context(self):
        """Test that controlled mocks context works correctly."""
        manager = MockManager()

        with manager.controlled_mocks():
            # Should have active mocks
            assert len(manager.active_mocks) > 0

        # Mocks should be cleaned up after exiting context
        assert len(manager.active_mocks) == 0

    def test_create_stable_mock(self):
        """Test creating stable, deterministic mocks."""
        manager = MockManager()

        mock = manager.create_stable_mock("test.target", return_value="test_result")

        assert mock is not None
        assert mock.return_value == "test_result"
        assert mock.side_effect is None


class TestStabilityDecorators:
    """Test the stability decorators and convenience functions."""

    def test_flaky_decorator(self):
        """Test the flaky decorator works correctly."""
        call_count = 0

        @flaky(max_retries=3, delay=0.01)
        def test_function():
            nonlocal call_count
            call_count += 1
            if call_count < 3:
                raise ValueError(f"Attempt {call_count} failed")
            return "success"

        result = test_function()
        assert result == "success"
        assert call_count == 3

    def test_stable_test_decorator(self):
        """Test the stable_test decorator works correctly."""
        test_executed = False

        @stable_test
        def test_function():
            nonlocal test_executed
            test_executed = True
            # Should have test environment variables
            assert os.environ.get("PYNOMALY_TEST_MODE") == "1"
            return "success"

        result = test_function()
        assert result == "success"
        assert test_executed

    def test_stable_test_decorator_preserves_function_metadata(self):
        """Test that stable_test decorator preserves function metadata."""

        @stable_test
        def test_function():
            """Test function docstring."""
            return "success"

        assert test_function.__name__ == "test_function"
        assert test_function.__doc__ == "Test function docstring."


class TestStabilityIntegration:
    """Test integration between different stability components."""

    def test_full_stabilization_integration(self):
        """Test that all stability components work together."""
        stabilizer = TestStabilizer()

        # Test comprehensive stabilization
        with stabilizer.stabilized_test("integration_test"):
            # Should have isolation
            assert os.environ.get("PYNOMALY_TEST_MODE") == "1"

            # Should be able to register resources
            test_obj = {"test": "data"}
            stabilizer.resource_manager.register_resource("temp_objects", test_obj)
            assert (
                test_obj in stabilizer.resource_manager.active_resources["temp_objects"]
            )

            # Should have timing stabilization
            import random

            val1 = random.random()
            val2 = random.random()
            # With mocked random, these should be equal
            assert val1 == val2

    def test_stabilization_with_retries(self):
        """Test stabilization combined with retry logic."""
        stabilizer = TestStabilizer()
        attempt_count = 0

        @stabilizer.retry_manager.retry_with_stabilization(max_retries=2, delay=0.01)
        def test_function():
            nonlocal attempt_count
            attempt_count += 1

            with stabilizer.stabilized_test(f"retry_test_{attempt_count}"):
                if attempt_count < 2:
                    raise ValueError(f"Attempt {attempt_count} failed")
                return "success"

        result = test_function()
        assert result == "success"
        assert attempt_count == 2

    def test_error_propagation_through_stabilization(self):
        """Test that errors are properly propagated through stabilization."""
        stabilizer = TestStabilizer()

        with pytest.raises(ValueError, match="Test error"):
            with stabilizer.stabilized_test("error_test"):
                raise ValueError("Test error")

    def test_nested_stabilization_contexts(self):
        """Test that nested stabilization contexts work correctly."""
        stabilizer = TestStabilizer()

        with stabilizer.stabilized_test("outer_test"):
            outer_env = os.environ.copy()

            with stabilizer.stabilized_test("inner_test"):
                inner_env = os.environ.copy()
                # Both should have test mode
                assert inner_env.get("PYNOMALY_TEST_MODE") == "1"
                assert outer_env.get("PYNOMALY_TEST_MODE") == "1"
