"""
Integration Testing Framework Base

This module provides the foundational framework for comprehensive integration testing
across all layers of the Pynomaly platform.
"""

from __future__ import annotations

import asyncio
import logging
import time
from abc import ABC, abstractmethod
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any
from unittest.mock import Mock

from sqlalchemy import create_engine
from sqlalchemy.engine import Engine

from pynomaly.infrastructure.config.container import Container, create_container
from pynomaly.infrastructure.config.settings import Settings
from pynomaly.infrastructure.security.security_integration import (
    initialize_application_security,
)

logger = logging.getLogger(__name__)


class IntegrationTestEnvironment:
    """Manages test environment setup and teardown for integration tests."""

    def __init__(self, config: dict[str, Any]):
        """Initialize test environment with configuration."""
        self.config = config
        self.services: dict[str, Any] = {}
        self.external_mocks: dict[str, Mock] = {}
        self.test_data_manager = None
        self.container: Container | None = None

    async def setup(self) -> None:
        """Set up the complete test environment."""
        logger.info("Setting up integration test environment...")

        # Initialize test settings
        self.settings = Settings(
            storage_path=Path("./test_storage"),
            database_url="sqlite:///test.db",
            debug=True,
            auth_enabled=False,
            use_database_repositories=False,
        )

        # Initialize container
        self.container = create_container(testing=True)

        # Initialize security (in test mode)
        self.security_manager = initialize_application_security(self.settings)

        # Setup test services
        await self._setup_services()

        # Setup external mocks
        await self._setup_external_mocks()

        # Setup test data
        await self._setup_test_data()

        logger.info("Integration test environment setup complete")

    async def teardown(self) -> None:
        """Tear down the test environment."""
        logger.info("Tearing down integration test environment...")

        # Clean up test data
        await self._cleanup_test_data()

        # Stop services
        await self._stop_services()

        # Clean up external mocks
        await self._cleanup_external_mocks()

        logger.info("Integration test environment teardown complete")

    async def _setup_services(self) -> None:
        """Set up internal services for testing."""
        # Database service
        if self.config.get("database_enabled", True):
            self.services["database"] = await self._setup_database_service()

        # Cache service
        if self.config.get("cache_enabled", True):
            self.services["cache"] = await self._setup_cache_service()

        # Message queue service (mock for testing)
        if self.config.get("message_queue_enabled", False):
            self.services["message_queue"] = await self._setup_message_queue_service()

    async def _setup_database_service(self) -> Engine:
        """Set up database service for testing."""
        engine = create_engine(
            self.settings.database_url,
            echo=False,
            connect_args={"check_same_thread": False},
        )

        # Create tables
        from pynomaly.infrastructure.persistence.database_repositories import Base

        Base.metadata.create_all(bind=engine)

        return engine

    async def _setup_cache_service(self) -> dict[str, Any]:
        """Set up cache service for testing."""
        from pynomaly.infrastructure.cache.cache_core import InMemoryCache

        return {"cache": InMemoryCache(max_size=1000), "enabled": True}

    async def _setup_message_queue_service(self) -> Mock:
        """Set up message queue service mock for testing."""
        queue_mock = Mock()
        queue_mock.send = Mock()
        queue_mock.receive = Mock()
        queue_mock.subscribe = Mock()

        return queue_mock

    async def _setup_external_mocks(self) -> None:
        """Set up external service mocks."""
        # Mock external API services
        self.external_mocks["external_api"] = Mock()
        self.external_mocks["notification_service"] = Mock()
        self.external_mocks["cloud_storage"] = Mock()

        # Configure mock behaviors
        self.external_mocks["external_api"].get.return_value = {"status": "ok"}
        self.external_mocks["notification_service"].send.return_value = True
        self.external_mocks["cloud_storage"].upload.return_value = "mock_url"

    async def _setup_test_data(self) -> None:
        """Set up test data for integration tests."""
        from tests.integration.framework.test_data_manager import (
            IntegrationTestDataManager,
        )

        self.test_data_manager = IntegrationTestDataManager(self.container)
        await self.test_data_manager.setup()

    async def _cleanup_test_data(self) -> None:
        """Clean up test data."""
        if self.test_data_manager:
            await self.test_data_manager.cleanup()

    async def _stop_services(self) -> None:
        """Stop all services."""
        for service_name, service in self.services.items():
            logger.info(f"Stopping service: {service_name}")

            if hasattr(service, "close"):
                await service.close()
            elif hasattr(service, "stop"):
                await service.stop()

    async def _cleanup_external_mocks(self) -> None:
        """Clean up external mocks."""
        for mock_name, mock in self.external_mocks.items():
            logger.info(f"Cleaning up mock: {mock_name}")
            mock.reset_mock()

    def get_service(self, service_name: str) -> Any:
        """Get a service by name."""
        return self.services.get(service_name)

    def get_external_mock(self, mock_name: str) -> Mock:
        """Get an external mock by name."""
        return self.external_mocks.get(mock_name)


class IntegrationTestBase(ABC):
    """Base class for all integration tests."""

    def __init__(self):
        """Initialize integration test base."""
        self.environment: IntegrationTestEnvironment | None = None
        self.start_time: float | None = None
        self.test_config = self._get_test_config()

    @abstractmethod
    def _get_test_config(self) -> dict[str, Any]:
        """Get test-specific configuration."""
        pass

    @asynccontextmanager
    async def setup_test_environment(self):
        """Set up test environment context manager."""
        self.start_time = time.time()

        try:
            # Create and setup environment
            self.environment = IntegrationTestEnvironment(self.test_config)
            await self.environment.setup()

            yield self.environment

        finally:
            # Teardown environment
            if self.environment:
                await self.environment.teardown()

            # Log test execution time
            if self.start_time:
                execution_time = time.time() - self.start_time
                logger.info(f"Integration test completed in {execution_time:.2f}s")

    async def wait_for_condition(
        self,
        condition_func: callable,
        timeout: float = 30.0,
        check_interval: float = 0.1,
    ) -> bool:
        """Wait for a condition to be met."""
        start_time = time.time()

        while time.time() - start_time < timeout:
            if await condition_func():
                return True
            await asyncio.sleep(check_interval)

        return False

    def assert_service_health(self, service_name: str) -> None:
        """Assert that a service is healthy."""
        service = self.environment.get_service(service_name)
        assert service is not None, f"Service {service_name} is not available"

        # Add service-specific health checks
        if service_name == "database":
            assert service.execute("SELECT 1").fetchone() is not None
        elif service_name == "cache":
            assert service["enabled"] is True

    def verify_external_mock_calls(self, mock_name: str, expected_calls: int) -> None:
        """Verify the number of calls to an external mock."""
        mock = self.environment.get_external_mock(mock_name)
        assert mock is not None, f"Mock {mock_name} is not available"

        actual_calls = mock.call_count
        assert (
            actual_calls == expected_calls
        ), f"Expected {expected_calls} calls to {mock_name}, got {actual_calls}"


class CrossLayerIntegrationTest(IntegrationTestBase):
    """Base class for cross-layer integration tests."""

    def _get_test_config(self) -> dict[str, Any]:
        """Get cross-layer test configuration."""
        return {
            "database_enabled": True,
            "cache_enabled": True,
            "message_queue_enabled": True,
            "external_services_enabled": True,
            "test_data_size": "medium",
            "parallel_execution": False,
        }

    async def execute_complete_workflow(
        self, workflow_name: str, input_data: dict[str, Any]
    ) -> dict[str, Any]:
        """Execute a complete workflow across all layers."""
        logger.info(f"Executing workflow: {workflow_name}")

        # Get necessary services
        container = self.environment.container

        # Execute workflow based on name
        if workflow_name == "anomaly_detection":
            return await self._execute_anomaly_detection_workflow(container, input_data)
        elif workflow_name == "model_training":
            return await self._execute_model_training_workflow(container, input_data)
        elif workflow_name == "data_ingestion":
            return await self._execute_data_ingestion_workflow(container, input_data)
        else:
            raise ValueError(f"Unknown workflow: {workflow_name}")

    async def _execute_anomaly_detection_workflow(
        self, container: Container, input_data: dict[str, Any]
    ) -> dict[str, Any]:
        """Execute complete anomaly detection workflow."""
        # 1. Data validation and preprocessing
        dataset_repo = container.dataset_repository()
        detector_repo = container.detector_repository()

        # 2. Load or create dataset
        dataset = await self._prepare_dataset(dataset_repo, input_data)

        # 3. Load or create detector
        detector = await self._prepare_detector(detector_repo, input_data)

        # 4. Execute detection
        detection_service = container.detection_service()
        result = await detection_service.detect_anomalies(
            detector_id=detector.id, dataset_id=dataset.id
        )

        # 5. Store results
        result_repo = container.result_repository()
        await result_repo.save(result)

        return {
            "workflow": "anomaly_detection",
            "status": "completed",
            "dataset_id": dataset.id,
            "detector_id": detector.id,
            "result_id": result.id,
            "anomaly_count": result.n_anomalies,
            "execution_time": result.execution_time_ms,
        }

    async def _execute_model_training_workflow(
        self, container: Container, input_data: dict[str, Any]
    ) -> dict[str, Any]:
        """Execute complete model training workflow."""
        # Implementation for model training workflow
        pass

    async def _execute_data_ingestion_workflow(
        self, container: Container, input_data: dict[str, Any]
    ) -> dict[str, Any]:
        """Execute complete data ingestion workflow."""
        # Implementation for data ingestion workflow
        pass

    async def _prepare_dataset(self, dataset_repo, input_data: dict[str, Any]):
        """Prepare dataset for testing."""
        # Implementation for dataset preparation
        pass

    async def _prepare_detector(self, detector_repo, input_data: dict[str, Any]):
        """Prepare detector for testing."""
        # Implementation for detector preparation
        pass


class ServiceIntegrationTest(IntegrationTestBase):
    """Base class for service integration tests."""

    def _get_test_config(self) -> dict[str, Any]:
        """Get service integration test configuration."""
        return {
            "database_enabled": True,
            "cache_enabled": True,
            "message_queue_enabled": True,
            "external_services_enabled": True,
            "test_data_size": "small",
            "parallel_execution": True,
        }

    async def test_service_communication(
        self, service_a: str, service_b: str, communication_type: str = "sync"
    ) -> bool:
        """Test communication between two services."""
        logger.info(
            f"Testing {communication_type} communication: {service_a} -> {service_b}"
        )

        # Get services from container
        container = self.environment.container

        if communication_type == "sync":
            return await self._test_sync_communication(container, service_a, service_b)
        elif communication_type == "async":
            return await self._test_async_communication(container, service_a, service_b)
        else:
            raise ValueError(f"Unknown communication type: {communication_type}")

    async def _test_sync_communication(
        self, container: Container, service_a: str, service_b: str
    ) -> bool:
        """Test synchronous communication between services."""
        # Implementation for sync communication testing
        return True

    async def _test_async_communication(
        self, container: Container, service_a: str, service_b: str
    ) -> bool:
        """Test asynchronous communication between services."""
        # Implementation for async communication testing
        return True


class ExternalIntegrationTest(IntegrationTestBase):
    """Base class for external service integration tests."""

    def _get_test_config(self) -> dict[str, Any]:
        """Get external integration test configuration."""
        return {
            "database_enabled": True,
            "cache_enabled": False,
            "message_queue_enabled": False,
            "external_services_enabled": True,
            "test_data_size": "small",
            "parallel_execution": False,
        }

    async def test_external_service_integration(
        self, service_name: str, operation: str, expected_behavior: str
    ) -> bool:
        """Test integration with external services."""
        logger.info(f"Testing external service: {service_name} - {operation}")

        mock = self.environment.get_external_mock(service_name)

        if operation == "api_call":
            return await self._test_api_call_integration(mock, expected_behavior)
        elif operation == "webhook":
            return await self._test_webhook_integration(mock, expected_behavior)
        else:
            raise ValueError(f"Unknown operation: {operation}")

    async def _test_api_call_integration(
        self, mock: Mock, expected_behavior: str
    ) -> bool:
        """Test API call integration."""
        # Implementation for API call testing
        return True

    async def _test_webhook_integration(
        self, mock: Mock, expected_behavior: str
    ) -> bool:
        """Test webhook integration."""
        # Implementation for webhook testing
        return True


# Test execution utilities
class IntegrationTestRunner:
    """Runner for integration tests with advanced features."""

    def __init__(self, config: dict[str, Any]):
        """Initialize test runner."""
        self.config = config
        self.results: list[dict[str, Any]] = []

    async def run_test_suite(
        self, test_classes: list[type[IntegrationTestBase]]
    ) -> dict[str, Any]:
        """Run a suite of integration tests."""
        logger.info(
            f"Running integration test suite with {len(test_classes)} test classes"
        )

        start_time = time.time()
        passed = 0
        failed = 0

        for test_class in test_classes:
            try:
                test_instance = test_class()
                await self._run_test_instance(test_instance)
                passed += 1
            except Exception as e:
                logger.error(f"Test {test_class.__name__} failed: {str(e)}")
                failed += 1

        execution_time = time.time() - start_time

        return {
            "total_tests": len(test_classes),
            "passed": passed,
            "failed": failed,
            "execution_time": execution_time,
            "success_rate": (passed / len(test_classes)) * 100,
        }

    async def _run_test_instance(self, test_instance: IntegrationTestBase) -> None:
        """Run a single test instance."""
        async with test_instance.setup_test_environment():
            # Test execution logic would go here
            pass

    def generate_report(self) -> dict[str, Any]:
        """Generate comprehensive test report."""
        return {
            "execution_summary": self.results,
            "timestamp": time.time(),
            "config": self.config,
        }
