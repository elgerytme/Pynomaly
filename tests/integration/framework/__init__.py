"""
Integration Testing Framework

This package provides the core framework for integration testing,
including base classes, test environment management, and data generation.
"""

from .integration_test_base import (
    IntegrationTestEnvironment,
    IntegrationTestBase,
    ServiceIntegrationTest,
    CrossLayerIntegrationTest,
    IntegrationTestRunner,
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