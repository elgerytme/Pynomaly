"""
Pytest fixtures for test stability framework.

This module provides pytest fixtures that make the stability testing tools
easily available to all tests in the project.
"""

import pytest

from .test_flaky_test_elimination import (
    MockManager,
    ResourceManager,
    RetryManager,
    TestIsolationManager,
    TestStabilizer,
    TimingManager,
)


@pytest.fixture
def test_stabilizer():
    """
    Pytest fixture providing a TestStabilizer instance.

    This fixture provides a fully configured TestStabilizer that can be used
    in tests for comprehensive stabilization.

    Example:
        def test_example(test_stabilizer):
            with test_stabilizer.stabilized_test("my_test"):
                # Test code here
                pass
    """
    return TestStabilizer()


@pytest.fixture
def isolation_manager():
    """
    Pytest fixture providing a TestIsolationManager instance.

    Use this when you only need environment isolation without full stabilization.
    """
    return TestIsolationManager()


@pytest.fixture
def retry_manager():
    """
    Pytest fixture providing a RetryManager instance.

    Use this when you only need retry logic without full stabilization.
    """
    return RetryManager()


@pytest.fixture
def resource_manager():
    """
    Pytest fixture providing a ResourceManager instance.

    Use this when you only need resource management without full stabilization.
    """
    return ResourceManager()


@pytest.fixture
def timing_manager():
    """
    Pytest fixture providing a TimingManager instance.

    Use this when you only need timing stabilization without full stabilization.
    """
    return TimingManager()


@pytest.fixture
def mock_manager():
    """
    Pytest fixture providing a MockManager instance.

    Use this when you only need mock management without full stabilization.
    """
    return MockManager()


@pytest.fixture
def stabilized_test_context(test_stabilizer):
    """
    Pytest fixture providing a pre-configured stabilized test context.

    This fixture automatically sets up a stabilized test environment
    for the duration of the test.

    Example:
        def test_example(stabilized_test_context):
            # Test runs in stabilized environment automatically
            pass
    """
    test_name = f"stabilized_test_{id(test_stabilizer)}"
    with test_stabilizer.stabilized_test(test_name):
        yield test_stabilizer


# Pytest hooks for stability testing
def pytest_configure(config):
    """Configure pytest with stability test markers."""
    config.addinivalue_line(
        "markers", "flaky: mark test as flaky (will be retried on failure)"
    )
    config.addinivalue_line(
        "markers", "stable: mark test as requiring full stabilization"
    )
    config.addinivalue_line(
        "markers", "isolation: mark test as requiring environment isolation"
    )
    config.addinivalue_line("markers", "retry: mark test as requiring retry logic")
    config.addinivalue_line(
        "markers", "timing: mark test as requiring timing stabilization"
    )
    config.addinivalue_line(
        "markers", "resource: mark test as requiring resource management"
    )


def pytest_runtest_setup(item):
    """
    Apply stability measures based on test markers.

    This hook automatically applies the appropriate stability measures
    based on the markers applied to individual tests.
    """
    # Check for stability markers and apply appropriate measures
    if item.get_closest_marker("flaky"):
        # Apply retry logic for flaky tests
        marker = item.get_closest_marker("flaky")
        max_retries = marker.kwargs.get("max_retries", 3)
        delay = marker.kwargs.get("delay", 0.1)

        # Wrap the test function with retry logic
        original_function = item.function
        retry_manager = RetryManager()
        item.function = retry_manager.retry_with_stabilization(
            max_retries=max_retries, delay=delay
        )(original_function)

    if item.get_closest_marker("stable"):
        # Apply full stabilization for stable tests
        original_function = item.function

        def stabilized_wrapper(*args, **kwargs):
            stabilizer = TestStabilizer()
            test_name = item.name
            with stabilizer.stabilized_test(test_name):
                return original_function(*args, **kwargs)

        item.function = stabilized_wrapper


def pytest_collection_modifyitems(config, items):
    """
    Modify test collection to apply stability measures.

    This hook processes all collected tests and applies stability measures
    based on their markers and naming patterns.
    """
    for item in items:
        # Auto-apply flaky marker to tests with "flaky" in the name
        if "flaky" in item.name.lower() and not item.get_closest_marker("flaky"):
            item.add_marker(pytest.mark.flaky(max_retries=3))

        # Auto-apply stable marker to tests with "stable" in the name
        if "stable" in item.name.lower() and not item.get_closest_marker("stable"):
            item.add_marker(pytest.mark.stable)

        # Auto-apply stability marker to tests in the _stability directory
        if "_stability" in str(item.fspath):
            if not item.get_closest_marker("stable"):
                item.add_marker(pytest.mark.stable)
