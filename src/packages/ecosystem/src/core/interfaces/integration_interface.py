"""
Core integration interface defining the contract for all ecosystem integrations.

This module provides the fundamental interface that all platform integrations
must implement, ensuring consistent behavior and interoperability.
"""

import asyncio
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum, auto
from typing import Dict, List, Optional, Any, Union, Callable
from uuid import UUID, uuid4

import structlog

logger = structlog.get_logger(__name__)


class IntegrationStatus(Enum):
    """Status of an integration connection."""
    DISCONNECTED = "disconnected"
    CONNECTING = "connecting"
    CONNECTED = "connected"
    ERROR = "error"
    SUSPENDED = "suspended"
    MAINTENANCE = "maintenance"


class ConnectionHealth(Enum):
    """Health status of an integration connection."""
    HEALTHY = "healthy"
    DEGRADED = "degraded"
    UNHEALTHY = "unhealthy"
    UNKNOWN = "unknown"


class AuthenticationMethod(Enum):
    """Authentication method for integration."""
    API_KEY = "api_key"
    OAUTH2 = "oauth2"
    SERVICE_ACCOUNT = "service_account"
    CERTIFICATE = "certificate"
    BASIC_AUTH = "basic_auth"
    CUSTOM = "custom"


class DataFlowCapability(Enum):
    """Data flow capabilities supported by integration."""
    READ_ONLY = "read_only"
    WRITE_ONLY = "write_only"
    BIDIRECTIONAL = "bidirectional"
    STREAMING = "streaming"
    BATCH = "batch"
    REAL_TIME = "real_time"


@dataclass
class IntegrationConfig:
    """Configuration for an ecosystem integration."""
    
    # Core identification
    name: str
    platform: str
    version: str = "1.0.0"
    description: Optional[str] = None
    
    # Connection details
    endpoint: Optional[str] = None
    credentials: Dict[str, Any] = field(default_factory=dict)
    auth_method: AuthenticationMethod = AuthenticationMethod.API_KEY
    
    # Capabilities
    data_capabilities: List[DataFlowCapability] = field(default_factory=list)
    supported_formats: List[str] = field(default_factory=list)
    
    # Configuration
    timeout_seconds: int = 30
    retry_attempts: int = 3
    retry_delay_seconds: int = 5
    
    # Metadata
    tags: Dict[str, str] = field(default_factory=dict)
    created_at: datetime = field(default_factory=datetime.utcnow)
    updated_at: datetime = field(default_factory=datetime.utcnow)
    
    def __post_init__(self):
        """Post-initialization validation."""
        if not self.name:
            raise ValueError("Integration name is required")
        if not self.platform:
            raise ValueError("Platform identifier is required")


@dataclass
class ConnectionMetrics:
    """Metrics for integration connection."""
    
    total_requests: int = 0
    successful_requests: int = 0
    failed_requests: int = 0
    average_response_time_ms: float = 0.0
    last_request_at: Optional[datetime] = None
    uptime_percentage: float = 100.0
    
    @property
    def success_rate(self) -> float:
        """Calculate success rate percentage."""
        if self.total_requests == 0:
            return 100.0
        return (self.successful_requests / self.total_requests) * 100.0
    
    @property
    def error_rate(self) -> float:
        """Calculate error rate percentage."""
        return 100.0 - self.success_rate


class IntegrationInterface(ABC):
    """
    Abstract base class for all ecosystem integrations.
    
    This interface defines the core contract that all integrations must
    implement to ensure consistent behavior across the platform.
    """
    
    def __init__(self, config: IntegrationConfig):
        """Initialize integration with configuration."""
        self.config = config
        self.id = uuid4()
        self.status = IntegrationStatus.DISCONNECTED
        self.health = ConnectionHealth.UNKNOWN
        self.metrics = ConnectionMetrics()
        self.logger = logger.bind(
            integration=config.name,
            platform=config.platform,
            integration_id=str(self.id)
        )
        
        # Event callbacks
        self._status_callbacks: List[Callable[[IntegrationStatus], None]] = []
        self._health_callbacks: List[Callable[[ConnectionHealth], None]] = []
        
        self.logger.info("Integration initialized", config=config.name)
    
    # Core connection management
    
    @abstractmethod
    async def connect(self) -> bool:
        """
        Establish connection to the external platform.
        
        Returns:
            bool: True if connection successful, False otherwise
        """
        pass
    
    @abstractmethod
    async def disconnect(self) -> bool:
        """
        Close connection to the external platform.
        
        Returns:
            bool: True if disconnection successful, False otherwise
        """
        pass
    
    @abstractmethod
    async def test_connection(self) -> ConnectionHealth:
        """
        Test the connection health.
        
        Returns:
            ConnectionHealth: Current health status
        """
        pass
    
    @abstractmethod
    async def get_capabilities(self) -> Dict[str, Any]:
        """
        Get integration capabilities and supported features.
        
        Returns:
            Dict[str, Any]: Capability information
        """
        pass
    
    # Data operations
    
    @abstractmethod
    async def send_data(
        self,
        data: Any,
        destination: str,
        format_type: Optional[str] = None,
        options: Optional[Dict[str, Any]] = None
    ) -> bool:
        """
        Send data to the external platform.
        
        Args:
            data: Data to send
            destination: Target destination identifier
            format_type: Data format (optional)
            options: Additional send options
            
        Returns:
            bool: True if send successful, False otherwise
        """
        pass
    
    @abstractmethod
    async def receive_data(
        self,
        source: str,
        format_type: Optional[str] = None,
        options: Optional[Dict[str, Any]] = None
    ) -> Optional[Any]:
        """
        Receive data from the external platform.
        
        Args:
            source: Source identifier
            format_type: Expected data format (optional)
            options: Additional receive options
            
        Returns:
            Any: Received data or None if no data available
        """
        pass
    
    # Event handling
    
    @abstractmethod
    async def subscribe_to_events(
        self,
        event_types: List[str],
        callback: Callable[[str, Dict[str, Any]], None]
    ) -> str:
        """
        Subscribe to events from the external platform.
        
        Args:
            event_types: List of event types to subscribe to
            callback: Callback function for handling events
            
        Returns:
            str: Subscription ID
        """
        pass
    
    @abstractmethod
    async def unsubscribe_from_events(self, subscription_id: str) -> bool:
        """
        Unsubscribe from events.
        
        Args:
            subscription_id: Subscription ID to cancel
            
        Returns:
            bool: True if unsubscription successful, False otherwise
        """
        pass
    
    # Monitoring and metrics
    
    async def get_metrics(self) -> ConnectionMetrics:
        """Get connection metrics."""
        return self.metrics
    
    async def get_status(self) -> IntegrationStatus:
        """Get current integration status."""
        return self.status
    
    async def get_health(self) -> ConnectionHealth:
        """Get current connection health."""
        return self.health
    
    # Configuration management
    
    async def update_config(self, config: IntegrationConfig) -> bool:
        """
        Update integration configuration.
        
        Args:
            config: New configuration
            
        Returns:
            bool: True if update successful, False otherwise
        """
        try:
            old_config = self.config
            self.config = config
            self.config.updated_at = datetime.utcnow()
            
            self.logger.info(
                "Configuration updated",
                old_name=old_config.name,
                new_name=config.name
            )
            return True
            
        except Exception as e:
            self.logger.error("Failed to update configuration", error=str(e))
            return False
    
    async def validate_config(self, config: Optional[IntegrationConfig] = None) -> bool:
        """
        Validate integration configuration.
        
        Args:
            config: Configuration to validate (uses current if None)
            
        Returns:
            bool: True if configuration is valid, False otherwise
        """
        try:
            config_to_validate = config or self.config
            
            # Basic validation
            if not config_to_validate.name:
                self.logger.error("Configuration validation failed: name is required")
                return False
            
            if not config_to_validate.platform:
                self.logger.error("Configuration validation failed: platform is required")
                return False
            
            # Platform-specific validation should be implemented in subclasses
            return await self._validate_platform_config(config_to_validate)
            
        except Exception as e:
            self.logger.error("Configuration validation failed", error=str(e))
            return False
    
    # Event subscription management
    
    def add_status_callback(self, callback: Callable[[IntegrationStatus], None]) -> None:
        """Add callback for status changes."""
        self._status_callbacks.append(callback)
    
    def add_health_callback(self, callback: Callable[[ConnectionHealth], None]) -> None:
        """Add callback for health changes."""
        self._health_callbacks.append(callback)
    
    def remove_status_callback(self, callback: Callable[[IntegrationStatus], None]) -> None:
        """Remove status change callback."""
        if callback in self._status_callbacks:
            self._status_callbacks.remove(callback)
    
    def remove_health_callback(self, callback: Callable[[ConnectionHealth], None]) -> None:
        """Remove health change callback."""
        if callback in self._health_callbacks:
            self._health_callbacks.remove(callback)
    
    # Protected helper methods
    
    async def _set_status(self, status: IntegrationStatus) -> None:
        """Set integration status and notify callbacks."""
        if self.status != status:
            old_status = self.status
            self.status = status
            
            self.logger.info(
                "Integration status changed",
                old_status=old_status.value,
                new_status=status.value
            )
            
            # Notify callbacks
            for callback in self._status_callbacks:
                try:
                    callback(status)
                except Exception as e:
                    self.logger.error("Status callback failed", error=str(e))
    
    async def _set_health(self, health: ConnectionHealth) -> None:
        """Set connection health and notify callbacks."""
        if self.health != health:
            old_health = self.health
            self.health = health
            
            self.logger.info(
                "Connection health changed",
                old_health=old_health.value,
                new_health=health.value
            )
            
            # Notify callbacks
            for callback in self._health_callbacks:
                try:
                    callback(health)
                except Exception as e:
                    self.logger.error("Health callback failed", error=str(e))
    
    async def _update_metrics(
        self,
        request_successful: bool,
        response_time_ms: float
    ) -> None:
        """Update connection metrics."""
        self.metrics.total_requests += 1
        self.metrics.last_request_at = datetime.utcnow()
        
        if request_successful:
            self.metrics.successful_requests += 1
        else:
            self.metrics.failed_requests += 1
        
        # Update average response time using exponential moving average
        if self.metrics.average_response_time_ms == 0:
            self.metrics.average_response_time_ms = response_time_ms
        else:
            alpha = 0.1  # Smoothing factor
            self.metrics.average_response_time_ms = (
                alpha * response_time_ms + 
                (1 - alpha) * self.metrics.average_response_time_ms
            )
    
    @abstractmethod
    async def _validate_platform_config(self, config: IntegrationConfig) -> bool:
        """
        Platform-specific configuration validation.
        
        Args:
            config: Configuration to validate
            
        Returns:
            bool: True if valid, False otherwise
        """
        pass
    
    # Context manager support
    
    async def __aenter__(self):
        """Async context manager entry."""
        await self.connect()
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit."""
        await self.disconnect()
    
    def __repr__(self) -> str:
        """String representation."""
        return (
            f"Integration(name={self.config.name}, "
            f"platform={self.config.platform}, "
            f"status={self.status.value}, "
            f"health={self.health.value})"
        )