"""
Integration Testing Framework

This package provides the core framework for integration testing,
including base classes, test environment management, and data generation.
"""

from .integration_test_base import (
    CrossLayerIntegrationTest,
    IntegrationTestBase,
    IntegrationTestEnvironment,
    IntegrationTestRunner,
    ServiceIntegrationTest,
)
from .test_data_manager import IntegrationTestDataManager

__all__ = [
    "IntegrationTestEnvironment",
    "IntegrationTestBase",
    "ServiceIntegrationTest",
    "CrossLayerIntegrationTest",
    "IntegrationTestRunner",
    "IntegrationTestDataManager",
]
