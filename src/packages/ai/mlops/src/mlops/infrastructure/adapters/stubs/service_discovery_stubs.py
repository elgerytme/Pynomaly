"""Stub implementations for service discovery operations."""

from typing import Any, Dict, List, Optional, Callable
from datetime import datetime

from mlops.domain.interfaces.service_discovery_operations import (
    ServiceRegistryPort,
    ServiceDiscoveryPort,
    HealthCheckPort,
    LoadBalancerPort,
    ServiceMetadata,
    ServiceQuery,
    ServiceStatus,
    ServiceType,
    ServiceEndpoint,
    HealthCheckResult
)


class ServiceRegistryStub(ServiceRegistryPort):
    """Stub implementation for service registry operations."""
    
    async def register_service(self, service: ServiceMetadata) -> bool:
        """Register a service in the registry."""
        return True
    
    async def deregister_service(self, service_id: str) -> bool:
        """Deregister a service from the registry."""
        return True
    
    async def get_service(self, service_id: str) -> Optional[ServiceMetadata]:
        """Get service metadata by ID."""
        return ServiceMetadata(
            service_id=service_id,
            service_name="stub_service",
            service_type=ServiceType.EXPERIMENT_TRACKING,
            version="1.0.0",
            endpoints=[ServiceEndpoint("http", "localhost", 8080)],
            status=ServiceStatus.HEALTHY
        )
    
    async def find_services(self, query: ServiceQuery) -> List[ServiceMetadata]:
        """Find services matching the query."""
        return []
    
    async def list_all_services(self) -> List[ServiceMetadata]:
        """List all registered services."""
        return []
    
    async def update_service_status(self, service_id: str, status: ServiceStatus) -> bool:
        """Update service status."""
        return True
    
    async def heartbeat(self, service_id: str, metadata: Optional[Dict[str, Any]] = None) -> bool:
        """Send heartbeat for a service."""
        return True


class ServiceDiscoveryStub(ServiceDiscoveryPort):
    """Stub implementation for service discovery operations."""
    
    async def discover_service(
        self, 
        service_type: ServiceType, 
        requirements: Optional[Dict[str, Any]] = None
    ) -> Optional[ServiceMetadata]:
        """Discover a service of the specified type."""
        return ServiceMetadata(
            service_id=f"stub_{service_type.value}",
            service_name=f"stub_{service_type.value}_service",
            service_type=service_type,
            version="1.0.0",
            endpoints=[ServiceEndpoint("http", "localhost", 8080)],
            status=ServiceStatus.HEALTHY
        )
    
    async def discover_services(
        self, 
        service_type: ServiceType, 
        requirements: Optional[Dict[str, Any]] = None
    ) -> List[ServiceMetadata]:
        """Discover all services of the specified type."""
        service = await self.discover_service(service_type, requirements)
        return [service] if service else []
    
    async def get_service_endpoint(
        self, 
        service_type: ServiceType, 
        endpoint_name: str = "default"
    ) -> Optional[ServiceEndpoint]:
        """Get endpoint for a service type."""
        return ServiceEndpoint("http", "localhost", 8080)
    
    async def watch_services(
        self, 
        service_type: ServiceType, 
        callback: Callable[[List[ServiceMetadata]], None]
    ) -> str:
        """Watch for changes in services of a specific type."""
        return f"watch_{service_type.value}"
    
    async def cancel_watch(self, watch_id: str) -> bool:
        """Cancel a service watch."""
        return True


class ServiceHealthCheckStub(HealthCheckPort):
    """Stub implementation for health check operations."""
    
    async def perform_health_check(self, service: ServiceMetadata) -> HealthCheckResult:
        """Perform health check on a service."""
        return HealthCheckResult(
            service_id=service.service_id,
            status=ServiceStatus.HEALTHY,
            response_time_ms=50.0,
            details={"check_type": "stub"},
            checked_at=datetime.now()
        )
    
    async def perform_bulk_health_check(
        self, 
        services: List[ServiceMetadata]
    ) -> List[HealthCheckResult]:
        """Perform health checks on multiple services."""
        return [await self.perform_health_check(service) for service in services]
    
    async def register_health_check(
        self, 
        service_id: str, 
        check_url: str, 
        interval_seconds: int = 30
    ) -> bool:
        """Register a recurring health check."""
        return True
    
    async def unregister_health_check(self, service_id: str) -> bool:
        """Unregister a health check."""
        return True


class LoadBalancerStub(LoadBalancerPort):
    """Stub implementation for load balancer operations."""
    
    async def get_balanced_service(
        self, 
        service_type: ServiceType, 
        strategy: str = "round_robin"
    ) -> Optional[ServiceMetadata]:
        """Get a service using load balancing."""
        return ServiceMetadata(
            service_id=f"balanced_{service_type.value}",
            service_name=f"balanced_{service_type.value}_service",
            service_type=service_type,
            version="1.0.0",
            endpoints=[ServiceEndpoint("http", "localhost", 8080)],
            status=ServiceStatus.HEALTHY
        )
    
    async def report_service_metrics(
        self, 
        service_id: str, 
        metrics: Dict[str, float]
    ) -> bool:
        """Report metrics for load balancing decisions."""
        return True
    
    async def mark_service_unhealthy(self, service_id: str) -> bool:
        """Mark a service as unhealthy for load balancing."""
        return True
    
    async def mark_service_healthy(self, service_id: str) -> bool:
        """Mark a service as healthy for load balancing."""
        return True