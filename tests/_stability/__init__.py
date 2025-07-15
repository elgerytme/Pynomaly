"""
Test Stability Foundation for Pynomaly

This module provides comprehensive tools for eliminating flaky tests through:
- Test isolation and environment management
- Intelligent retry mechanisms with exponential backoff
- Resource management and cleanup
- Timing stabilization and deterministic behavior
- Mock management for external dependencies

Key Components:
- TestStabilizer: Main stabilization framework
- TestIsolationManager: Environment isolation
- RetryManager: Intelligent retry logic
- ResourceManager: Resource cleanup
- TimingManager: Timing stabilization
- MockManager: Mock management

Usage:
    from tests._stability import TestStabilizer, flaky, stable_test

    # Use as context manager
    with TestStabilizer().stabilized_test("my_test"):
        # Your test code here
        pass

    # Use as decorator
    @flaky(max_retries=3)
    def test_something():
        pass

    # Use as pytest fixture
    @stable_test
    def test_with_stability(test_stabilizer):
        # test_stabilizer is automatically provided
        pass
"""

from .test_flaky_test_elimination import (
    MockManager,
    ResourceManager,
    RetryManager,
    TestIsolationManager,
    TestStabilizer,
    TimingManager,
)

__all__ = [
    "TestStabilizer",
    "TestIsolationManager",
    "RetryManager",
    "ResourceManager",
    "TimingManager",
    "MockManager",
    "flaky",
    "stable_test",
]


# Convenience decorators and fixtures
def flaky(max_retries: int = 3, delay: float = 0.1):
    """
    Decorator to mark tests as flaky and apply automatic retry logic.

    Args:
        max_retries: Maximum number of retry attempts
        delay: Base delay between retries (with exponential backoff)

    Example:
        @flaky(max_retries=5, delay=0.2)
        def test_potentially_flaky():
            # Test code that might be flaky
            pass
    """

    def decorator(func):
        retry_manager = RetryManager()
        return retry_manager.retry_with_stabilization(
            max_retries=max_retries, delay=delay
        )(func)

    return decorator


def stable_test(func):
    """
    Decorator to apply full test stabilization to a test function.

    This decorator:
    - Provides test isolation
    - Manages resources automatically
    - Applies timing stabilization
    - Sets up controlled mocks

    Example:
        @stable_test
        def test_with_stability():
            # This test will be fully stabilized
            pass
    """

    def wrapper(*args, **kwargs):
        stabilizer = TestStabilizer()
        test_name = func.__name__

        with stabilizer.stabilized_test(test_name):
            return func(*args, **kwargs)

    wrapper.__name__ = func.__name__
    wrapper.__doc__ = func.__doc__
    return wrapper
