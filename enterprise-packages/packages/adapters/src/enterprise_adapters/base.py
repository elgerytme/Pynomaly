"""Base adapter infrastructure for enterprise applications.

This module provides the foundation for all adapter implementations,
including factory patterns, registry management, and common utilities.
"""

from __future__ import annotations

import logging
from abc import abstractmethod
from contextlib import asynccontextmanager
from typing import Any, TypeVar

from enterprise_core import (
    ConfigurationError,
    HealthStatus,
    InfrastructureError,
    ServiceAdapter,
)
from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)


class AdapterConfiguration(BaseModel):
    """Configuration model for adapters."""

    adapter_type: str = Field(..., description="Type of adapter")
    connection_string: str | None = Field(None, description="Connection string")
    host: str | None = Field(None, description="Host address")
    port: int | None = Field(None, description="Port number")
    username: str | None = Field(None, description="Username for authentication")
    password: str | None = Field(None, description="Password for authentication")
    database: str | None = Field(None, description="Database name")
    timeout: int = Field(default=30, description="Connection timeout in seconds")
    pool_size: int = Field(default=10, description="Connection pool size")
    max_overflow: int = Field(default=20, description="Maximum pool overflow")
    retry_attempts: int = Field(default=3, description="Number of retry attempts")
    retry_delay: float = Field(
        default=1.0, description="Delay between retries in seconds"
    )
    ssl_enabled: bool = Field(default=False, description="Enable SSL/TLS")
    ssl_verify: bool = Field(default=True, description="Verify SSL certificates")
    metadata: dict[str, Any] = Field(
        default_factory=dict, description="Additional metadata"
    )


AdapterType = TypeVar("AdapterType", bound="BaseAdapter")


class BaseAdapter(ServiceAdapter):
    """Base class for all adapters.

    Provides common functionality for connection management, health checks,
    retry logic, and graceful degradation.
    """

    def __init__(self, config: AdapterConfiguration) -> None:
        super().__init__(config.model_dump())
        self.config = config
        self._connection: Any | None = None
        self._is_connected = False
        self._retry_count = 0

    @property
    def adapter_type(self) -> str:
        """Get the adapter type."""
        return self.config.adapter_type

    @property
    def is_connected(self) -> bool:
        """Check if the adapter is connected."""
        return self._is_connected

    @abstractmethod
    async def _create_connection(self) -> Any:
        """Create a connection to the external service."""
        pass

    @abstractmethod
    async def _close_connection(self) -> None:
        """Close the connection to the external service."""
        pass

    @abstractmethod
    async def _test_connection(self) -> bool:
        """Test the connection to verify it's working."""
        pass

    async def connect(self) -> None:
        """Establish connection to the external service."""
        if self._is_connected:
            return

        try:
            self._connection = await self._create_connection()
            self._is_connected = True
            self._retry_count = 0
            self._is_healthy = True
            logger.info(f"Connected to {self.adapter_type}")

        except Exception as e:
            self._is_healthy = False
            logger.error(f"Failed to connect to {self.adapter_type}: {e}")
            raise InfrastructureError(
                f"Failed to connect to {self.adapter_type}",
                error_code="CONNECTION_FAILED",
                details={"adapter_type": self.adapter_type, "error": str(e)},
                cause=e,
            ) from e

    async def disconnect(self) -> None:
        """Disconnect from the external service."""
        if not self._is_connected:
            return

        try:
            await self._close_connection()
            self._is_connected = False
            self._connection = None
            logger.info(f"Disconnected from {self.adapter_type}")

        except Exception as e:
            logger.warning(f"Error disconnecting from {self.adapter_type}: {e}")

    async def reconnect(self) -> None:
        """Reconnect to the external service."""
        await self.disconnect()
        await self.connect()

    async def initialize(self) -> None:
        """Initialize the adapter."""
        await self.connect()

    async def cleanup(self) -> None:
        """Clean up adapter resources."""
        await self.disconnect()

    async def health_check(self) -> HealthStatus:
        """Perform a health check on the adapter."""
        try:
            if not self._is_connected:
                return HealthStatus(
                    status="unhealthy",
                    message=f"{self.adapter_type} is not connected",
                    details={"adapter_type": self.adapter_type},
                )

            # Test the connection
            if await self._test_connection():
                return HealthStatus(
                    status="healthy",
                    message=f"{self.adapter_type} is healthy",
                    details={"adapter_type": self.adapter_type},
                )
            else:
                return HealthStatus(
                    status="degraded",
                    message=f"{self.adapter_type} connection test failed",
                    details={"adapter_type": self.adapter_type},
                )

        except Exception as e:
            return HealthStatus(
                status="unhealthy",
                message=f"{self.adapter_type} health check failed: {e}",
                details={
                    "adapter_type": self.adapter_type,
                    "error": str(e),
                },
            )

    @asynccontextmanager
    async def with_retry(self):
        """Context manager for operations with retry logic."""
        from tenacity import (
            retry,
            retry_if_exception_type,
            stop_after_attempt,
            wait_exponential,
        )

        @retry(
            stop=stop_after_attempt(self.config.retry_attempts),
            wait=wait_exponential(multiplier=self.config.retry_delay),
            retry=retry_if_exception_type(InfrastructureError),
        )
        async def _execute_with_retry():
            if not self._is_connected:
                await self.connect()
            return self

        try:
            adapter = await _execute_with_retry()
            yield adapter
        except Exception:
            self._is_healthy = False
            raise


class AdapterFactory:
    """Factory for creating adapter instances."""

    _adapters: dict[str, type[BaseAdapter]] = {}

    @classmethod
    def register(cls, adapter_type: str, adapter_class: type[BaseAdapter]) -> None:
        """Register an adapter class with the factory."""
        cls._adapters[adapter_type] = adapter_class
        logger.debug(f"Registered adapter: {adapter_type}")

    @classmethod
    def create(cls, config: AdapterConfiguration) -> BaseAdapter:
        """Create an adapter instance from configuration."""
        adapter_type = config.adapter_type

        if adapter_type not in cls._adapters:
            raise ConfigurationError(
                f"Unknown adapter type: {adapter_type}",
                error_code="UNKNOWN_ADAPTER_TYPE",
                details={
                    "adapter_type": adapter_type,
                    "available_types": list(cls._adapters.keys()),
                },
            )

        adapter_class = cls._adapters[adapter_type]
        return adapter_class(config)

    @classmethod
    def list_adapters(cls) -> list[str]:
        """List all registered adapter types."""
        return list(cls._adapters.keys())


class AdapterRegistry:
    """Registry for managing adapter instances."""

    def __init__(self) -> None:
        self._adapters: dict[str, BaseAdapter] = {}
        self._configurations: dict[str, AdapterConfiguration] = {}

    def register(self, name: str, adapter: BaseAdapter) -> None:
        """Register an adapter instance."""
        self._adapters[name] = adapter
        self._configurations[name] = adapter.config
        logger.debug(f"Registered adapter instance: {name}")

    def get(self, name: str) -> BaseAdapter | None:
        """Get an adapter instance by name."""
        return self._adapters.get(name)

    def get_required(self, name: str) -> BaseAdapter:
        """Get a required adapter instance, raising an error if not found."""
        adapter = self.get(name)
        if adapter is None:
            raise ConfigurationError(
                f"Required adapter '{name}' not found",
                error_code="ADAPTER_NOT_FOUND",
                details={
                    "adapter_name": name,
                    "available_adapters": list(self._adapters.keys()),
                },
            )
        return adapter

    def remove(self, name: str) -> None:
        """Remove an adapter instance."""
        if name in self._adapters:
            adapter = self._adapters[name]
            # Clean up the adapter
            if hasattr(adapter, "cleanup"):
                import asyncio

                asyncio.create_task(adapter.cleanup())

            del self._adapters[name]
            del self._configurations[name]
            logger.debug(f"Removed adapter instance: {name}")

    def list_adapters(self) -> list[str]:
        """List all registered adapter names."""
        return list(self._adapters.keys())

    async def initialize_all(self) -> None:
        """Initialize all registered adapters."""
        for name, adapter in self._adapters.items():
            try:
                await adapter.initialize()
                logger.info(f"Initialized adapter: {name}")
            except Exception as e:
                logger.error(f"Failed to initialize adapter {name}: {e}")
                raise

    async def cleanup_all(self) -> None:
        """Clean up all registered adapters."""
        for name, adapter in self._adapters.items():
            try:
                await adapter.cleanup()
                logger.info(f"Cleaned up adapter: {name}")
            except Exception as e:
                logger.warning(f"Error cleaning up adapter {name}: {e}")

    async def health_check_all(self) -> dict[str, HealthStatus]:
        """Perform health checks on all adapters."""
        results = {}
        for name, adapter in self._adapters.items():
            try:
                results[name] = await adapter.health_check()
            except Exception as e:
                results[name] = HealthStatus(
                    status="unhealthy",
                    message=f"Health check failed: {e}",
                    details={"error": str(e)},
                )
        return results


def adapter(adapter_type: str) -> callable:
    """Decorator to register an adapter class with the factory."""

    def decorator(cls: type[BaseAdapter]) -> type[BaseAdapter]:
        AdapterFactory.register(adapter_type, cls)
        return cls

    return decorator


# Global registry instance
adapter_registry = AdapterRegistry()
