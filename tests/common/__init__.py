"""Common test utilities package.

This package provides shared utilities, fixtures, and helper functions
to reduce code duplication across test files.
"""

from .utils import (
    # Main utility classes
    TestDataGenerator,
    MockFactory,
    TemporaryStorageManager,
    AsyncTestHelper,
    TestAssertions,
    ConfigurationHelper,
    # Global instances
    test_data_generator,
    mock_factory,
    storage_manager,
    async_helper,
    assertions,
    config_helper,
    # Common fixtures
    temp_storage,
    sample_data,
    time_series_data,
    mixed_data,
    mock_repository,
    mock_detector,
    mock_dataset,
    test_config,
)

__all__ = [
    # Classes
    "TestDataGenerator",
    "MockFactory",
    "TemporaryStorageManager",
    "AsyncTestHelper",
    "TestAssertions",
    "ConfigurationHelper",
    # Global instances
    "test_data_generator",
    "mock_factory",
    "storage_manager",
    "async_helper",
    "assertions",
    "config_helper",
    # Fixtures
    "temp_storage",
    "sample_data",
    "time_series_data",
    "mixed_data",
    "mock_repository",
    "mock_detector",
    "mock_dataset",
    "test_config",
]
