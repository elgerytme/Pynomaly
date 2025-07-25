"""Domain interfaces for service discovery operations."""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from datetime import datetime
from typing import Any, Dict, List, Optional, Callable
from enum import Enum


class ServiceStatus(Enum):
    """Service status states."""
    HEALTHY = "healthy"
    UNHEALTHY = "unhealthy"
    DEGRADED = "degraded"
    STARTING = "starting"
    STOPPING = "stopping"
    UNKNOWN = "unknown"


class ServiceType(Enum):
    """Service type definitions."""
    EXPERIMENT_TRACKING = "experiment_tracking"
    MODEL_REGISTRY = "model_registry"
    MONITORING = "monitoring"
    DATABASE = "database"
    STORAGE = "storage"
    DEPLOYMENT = "deployment"
    AUTHENTICATION = "authentication"
    NOTIFICATION = "notification"


@dataclass
class ServiceEndpoint:
    """Service endpoint information."""
    protocol: str
    host: str
    port: int
    path: str = ""
    
    @property
    def url(self) -> str:
        """Get the full URL for this endpoint."""
        base = f"{self.protocol}://{self.host}:{self.port}"
        return f"{base}{self.path}" if self.path else base


@dataclass
class ServiceMetadata:
    """Service metadata and configuration."""
    service_id: str
    service_name: str
    service_type: ServiceType
    version: str
    endpoints: List[ServiceEndpoint]
    status: ServiceStatus
    health_check_url: Optional[str] = None
    tags: Optional[Dict[str, str]] = None
    capabilities: Optional[List[str]] = None
    dependencies: Optional[List[str]] = None
    metadata: Optional[Dict[str, Any]] = None
    registered_at: Optional[datetime] = None
    last_heartbeat: Optional[datetime] = None


@dataclass
class ServiceQuery:
    """Query for finding services."""
    service_type: Optional[ServiceType] = None
    service_name: Optional[str] = None
    tags: Optional[Dict[str, str]] = None
    status: Optional[ServiceStatus] = None
    capabilities: Optional[List[str]] = None
    version_pattern: Optional[str] = None


@dataclass
class HealthCheckResult:
    """Result of a health check."""
    service_id: str
    status: ServiceStatus
    response_time_ms: float
    details: Optional[Dict[str, Any]] = None
    checked_at: Optional[datetime] = None
    error_message: Optional[str] = None


class ServiceRegistryPort(ABC):
    """Port for service registry operations."""
    
    @abstractmethod
    async def register_service(self, service: ServiceMetadata) -> bool:
        """Register a service in the registry.
        
        Args:
            service: Service metadata to register
            
        Returns:
            True if registration successful
        """
        pass
    
    @abstractmethod
    async def deregister_service(self, service_id: str) -> bool:
        """Deregister a service from the registry.
        
        Args:
            service_id: ID of the service to deregister
            
        Returns:
            True if deregistration successful
        """
        pass
    
    @abstractmethod
    async def get_service(self, service_id: str) -> Optional[ServiceMetadata]:
        """Get service metadata by ID.
        
        Args:
            service_id: ID of the service
            
        Returns:
            Service metadata or None if not found
        """
        pass
    
    @abstractmethod
    async def find_services(self, query: ServiceQuery) -> List[ServiceMetadata]:
        """Find services matching the query.
        
        Args:
            query: Service query criteria
            
        Returns:
            List of matching services
        """
        pass
    
    @abstractmethod
    async def list_all_services(self) -> List[ServiceMetadata]:
        """List all registered services.
        
        Returns:
            List of all registered services
        """
        pass
    
    @abstractmethod
    async def update_service_status(self, service_id: str, status: ServiceStatus) -> bool:
        """Update service status.
        
        Args:
            service_id: ID of the service
            status: New status
            
        Returns:
            True if update successful
        """
        pass
    
    @abstractmethod
    async def heartbeat(self, service_id: str, metadata: Optional[Dict[str, Any]] = None) -> bool:
        """Send heartbeat for a service.
        
        Args:
            service_id: ID of the service
            metadata: Optional metadata update
            
        Returns:
            True if heartbeat successful
        """
        pass


class ServiceDiscoveryPort(ABC):
    """Port for service discovery operations."""
    
    @abstractmethod
    async def discover_service(
        self, 
        service_type: ServiceType, 
        requirements: Optional[Dict[str, Any]] = None
    ) -> Optional[ServiceMetadata]:
        """Discover a service of the specified type.
        
        Args:
            service_type: Type of service to discover
            requirements: Optional requirements (version, capabilities, etc.)
            
        Returns:
            Service metadata or None if not found
        """
        pass
    
    @abstractmethod
    async def discover_services(
        self, 
        service_type: ServiceType, 
        requirements: Optional[Dict[str, Any]] = None
    ) -> List[ServiceMetadata]:
        """Discover all services of the specified type.
        
        Args:
            service_type: Type of service to discover
            requirements: Optional requirements
            
        Returns:
            List of matching services
        """
        pass
    
    @abstractmethod
    async def get_service_endpoint(
        self, 
        service_type: ServiceType, 
        endpoint_name: str = "default"
    ) -> Optional[ServiceEndpoint]:
        """Get endpoint for a service type.
        
        Args:
            service_type: Type of service
            endpoint_name: Name of the endpoint
            
        Returns:
            Service endpoint or None if not found
        """
        pass
    
    @abstractmethod
    async def watch_services(
        self, 
        service_type: ServiceType, 
        callback: Callable[[List[ServiceMetadata]], None]
    ) -> str:
        """Watch for changes in services of a specific type.
        
        Args:
            service_type: Type of service to watch
            callback: Function to call when services change
            
        Returns:
            Watch ID for cancelling the watch
        """
        pass
    
    @abstractmethod
    async def cancel_watch(self, watch_id: str) -> bool:
        """Cancel a service watch.
        
        Args:
            watch_id: ID of the watch to cancel
            
        Returns:
            True if watch cancelled successfully
        """
        pass


class HealthCheckPort(ABC):
    """Port for health check operations."""
    
    @abstractmethod
    async def perform_health_check(self, service: ServiceMetadata) -> HealthCheckResult:
        """Perform health check on a service.
        
        Args:
            service: Service to check
            
        Returns:
            Health check result
        """
        pass
    
    @abstractmethod
    async def perform_bulk_health_check(
        self, 
        services: List[ServiceMetadata]
    ) -> List[HealthCheckResult]:
        """Perform health checks on multiple services.
        
        Args:
            services: List of services to check
            
        Returns:
            List of health check results
        """
        pass
    
    @abstractmethod
    async def register_health_check(
        self, 
        service_id: str, 
        check_url: str, 
        interval_seconds: int = 30
    ) -> bool:
        """Register a recurring health check.
        
        Args:
            service_id: ID of the service
            check_url: URL to check
            interval_seconds: Check interval
            
        Returns:
            True if registration successful
        """
        pass
    
    @abstractmethod
    async def unregister_health_check(self, service_id: str) -> bool:
        """Unregister a health check.
        
        Args:
            service_id: ID of the service
            
        Returns:
            True if unregistration successful
        """
        pass


class LoadBalancerPort(ABC):
    """Port for load balancing operations."""
    
    @abstractmethod
    async def get_balanced_service(
        self, 
        service_type: ServiceType, 
        strategy: str = "round_robin"
    ) -> Optional[ServiceMetadata]:
        """Get a service using load balancing.
        
        Args:
            service_type: Type of service
            strategy: Load balancing strategy (round_robin, least_connections, random)
            
        Returns:
            Selected service or None if none available
        """
        pass
    
    @abstractmethod
    async def report_service_metrics(
        self, 
        service_id: str, 
        metrics: Dict[str, float]
    ) -> bool:
        """Report metrics for load balancing decisions.
        
        Args:
            service_id: ID of the service
            metrics: Metrics (response_time, error_rate, etc.)
            
        Returns:
            True if metrics reported successfully
        """
        pass
    
    @abstractmethod
    async def mark_service_unhealthy(self, service_id: str) -> bool:
        """Mark a service as unhealthy for load balancing.
        
        Args:
            service_id: ID of the service
            
        Returns:
            True if marked successfully
        """
        pass
    
    @abstractmethod
    async def mark_service_healthy(self, service_id: str) -> bool:
        """Mark a service as healthy for load balancing.
        
        Args:
            service_id: ID of the service
            
        Returns:
            True if marked successfully
        """
        pass