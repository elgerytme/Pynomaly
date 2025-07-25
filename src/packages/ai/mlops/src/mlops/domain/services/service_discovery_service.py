"""Service discovery domain service with dependency injection."""

from typing import Any, Dict, List, Optional, Callable
from datetime import datetime, timedelta

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


class ServiceDiscoveryService:
    """Domain service for service discovery operations using dependency injection."""
    
    def __init__(
        self,
        service_registry_port: ServiceRegistryPort,
        service_discovery_port: ServiceDiscoveryPort,
        health_check_port: HealthCheckPort,
        load_balancer_port: LoadBalancerPort
    ):
        self.service_registry_port = service_registry_port
        self.service_discovery_port = service_discovery_port
        self.health_check_port = health_check_port
        self.load_balancer_port = load_balancer_port
    
    async def register_mlops_service(
        self,
        service_name: str,
        service_type: ServiceType,
        endpoints: List[ServiceEndpoint],
        version: str = "1.0.0",
        capabilities: Optional[List[str]] = None,
        tags: Optional[Dict[str, str]] = None,
        health_check_url: Optional[str] = None
    ) -> str:
        """Register an MLOps service in the registry.
        
        Args:
            service_name: Name of the service
            service_type: Type of the service
            endpoints: List of service endpoints
            version: Service version
            capabilities: Service capabilities
            tags: Service tags
            health_check_url: Health check URL
            
        Returns:
            Service ID
        """
        service_id = f"{service_type.value}_{service_name}_{int(datetime.now().timestamp())}"
        
        service = ServiceMetadata(
            service_id=service_id,
            service_name=service_name,
            service_type=service_type,
            version=version,
            endpoints=endpoints,
            status=ServiceStatus.HEALTHY,  # Start as healthy for testing
            health_check_url=health_check_url,
            capabilities=capabilities or [],
            tags=tags or {},
            dependencies=[],
            metadata={}
        )
        
        success = await self.service_registry_port.register_service(service)
        
        if success and health_check_url:
            # Register health check
            await self.health_check_port.register_health_check(
                service_id, health_check_url, interval_seconds=30
            )
        
        return service_id if success else ""
    
    async def discover_mlops_service(
        self,
        service_type: ServiceType,
        capabilities: Optional[List[str]] = None,
        tags: Optional[Dict[str, str]] = None,
        version_pattern: Optional[str] = None
    ) -> Optional[ServiceMetadata]:
        """Discover an MLOps service.
        
        Args:
            service_type: Type of service to discover
            capabilities: Required capabilities
            tags: Required tags
            version_pattern: Version pattern to match
            
        Returns:
            Service metadata or None if not found
        """
        requirements = {}
        if capabilities:
            requirements["capabilities"] = capabilities
        if tags:
            requirements["tags"] = tags
        if version_pattern:
            requirements["version"] = version_pattern
        
        return await self.service_discovery_port.discover_service(service_type, requirements)
    
    async def get_healthy_service(
        self,
        service_type: ServiceType,
        load_balancing_strategy: str = "round_robin"
    ) -> Optional[ServiceMetadata]:
        """Get a healthy service using load balancing.
        
        Args:
            service_type: Type of service
            load_balancing_strategy: Load balancing strategy
            
        Returns:
            Healthy service metadata or None
        """
        return await self.load_balancer_port.get_balanced_service(
            service_type, load_balancing_strategy
        )
    
    async def check_service_health(self, service_id: str) -> Optional[HealthCheckResult]:
        """Check health of a specific service.
        
        Args:
            service_id: ID of the service to check
            
        Returns:
            Health check result or None if service not found
        """
        service = await self.service_registry_port.get_service(service_id)
        if not service:
            return None
        
        return await self.health_check_port.perform_health_check(service)
    
    async def monitor_service_ecosystem(self) -> Dict[str, Any]:
        """Monitor the health of the entire service ecosystem.
        
        Returns:
            Ecosystem health report
        """
        all_services = await self.service_registry_port.list_all_services()
        
        if not all_services:
            return {
                "total_services": 0,
                "healthy_services": 0,
                "unhealthy_services": 0,
                "ecosystem_health": "no_services",
                "services_by_type": {},
                "health_details": []
            }
        
        health_results = await self.health_check_port.perform_bulk_health_check(all_services)
        
        # Analyze results
        healthy_count = sum(1 for result in health_results if result.status == ServiceStatus.HEALTHY)
        unhealthy_count = len(health_results) - healthy_count
        
        # Group by service type
        services_by_type = {}
        for service in all_services:
            service_type = service.service_type.value
            if service_type not in services_by_type:
                services_by_type[service_type] = {"total": 0, "healthy": 0}
            services_by_type[service_type]["total"] += 1
        
        # Update health status for each type
        for result in health_results:
            service = next(s for s in all_services if s.service_id == result.service_id)
            service_type = service.service_type.value
            if result.status == ServiceStatus.HEALTHY:
                services_by_type[service_type]["healthy"] += 1
        
        # Determine overall ecosystem health
        health_ratio = healthy_count / len(all_services) if all_services else 0
        if health_ratio >= 0.9:
            ecosystem_health = "excellent"
        elif health_ratio >= 0.7:
            ecosystem_health = "good"
        elif health_ratio >= 0.5:
            ecosystem_health = "degraded"
        else:
            ecosystem_health = "critical"
        
        return {
            "total_services": len(all_services),
            "healthy_services": healthy_count,
            "unhealthy_services": unhealthy_count,
            "ecosystem_health": ecosystem_health,
            "health_ratio": health_ratio,
            "services_by_type": services_by_type,
            "health_details": [
                {
                    "service_id": result.service_id,
                    "service_name": next(s.service_name for s in all_services if s.service_id == result.service_id),
                    "service_type": next(s.service_type.value for s in all_services if s.service_id == result.service_id),
                    "status": result.status.value,
                    "response_time_ms": result.response_time_ms,
                    "error_message": result.error_message
                }
                for result in health_results
            ]
        }
    
    async def setup_service_monitoring(
        self,
        service_id: str,
        monitor_interval_seconds: int = 30,
        alert_threshold_ms: int = 5000
    ) -> bool:
        """Set up monitoring for a service.
        
        Args:
            service_id: ID of the service to monitor
            monitor_interval_seconds: Monitoring interval
            alert_threshold_ms: Alert threshold for response time
            
        Returns:
            True if monitoring setup successful
        """
        service = await self.service_registry_port.get_service(service_id)
        if not service or not service.health_check_url:
            return False
        
        # Register health check
        return await self.health_check_port.register_health_check(
            service_id, service.health_check_url, monitor_interval_seconds
        )
    
    async def handle_service_failure(self, service_id: str) -> Dict[str, Any]:
        """Handle service failure by updating status and load balancer.
        
        Args:
            service_id: ID of the failed service
            
        Returns:
            Failure handling result
        """
        # Update service status
        status_updated = await self.service_registry_port.update_service_status(
            service_id, ServiceStatus.UNHEALTHY
        )
        
        # Mark as unhealthy in load balancer
        lb_updated = await self.load_balancer_port.mark_service_unhealthy(service_id)
        
        # Get service details for response
        service = await self.service_registry_port.get_service(service_id)
        
        return {
            "service_id": service_id,
            "service_name": service.service_name if service else "unknown",
            "status_updated": status_updated,
            "load_balancer_updated": lb_updated,
            "failure_handled_at": datetime.now().isoformat(),
            "actions_taken": [
                "Updated service status to UNHEALTHY",
                "Removed service from load balancer pool"
            ]
        }
    
    async def handle_service_recovery(self, service_id: str) -> Dict[str, Any]:
        """Handle service recovery by updating status and load balancer.
        
        Args:
            service_id: ID of the recovered service
            
        Returns:
            Recovery handling result
        """
        # Verify service is actually healthy
        health_result = await self.check_service_health(service_id)
        if not health_result or health_result.status != ServiceStatus.HEALTHY:
            return {
                "service_id": service_id,
                "recovery_verified": False,
                "reason": "Service still unhealthy after recovery attempt"
            }
        
        # Update service status
        status_updated = await self.service_registry_port.update_service_status(
            service_id, ServiceStatus.HEALTHY
        )
        
        # Mark as healthy in load balancer
        lb_updated = await self.load_balancer_port.mark_service_healthy(service_id)
        
        # Get service details
        service = await self.service_registry_port.get_service(service_id)
        
        return {
            "service_id": service_id,
            "service_name": service.service_name if service else "unknown",
            "recovery_verified": True,
            "status_updated": status_updated,
            "load_balancer_updated": lb_updated,
            "recovery_handled_at": datetime.now().isoformat(),
            "actions_taken": [
                "Verified service health",
                "Updated service status to HEALTHY",
                "Added service back to load balancer pool"
            ]
        }
    
    async def get_service_dependencies(self, service_id: str) -> Dict[str, Any]:
        """Get service dependencies and their health status.
        
        Args:
            service_id: ID of the service
            
        Returns:
            Service dependencies report
        """
        service = await self.service_registry_port.get_service(service_id)
        if not service:
            return {"error": "Service not found"}
        
        dependencies = []
        if service.dependencies:
            for dep_id in service.dependencies:
                dep_service = await self.service_registry_port.get_service(dep_id)
                if dep_service:
                    health_result = await self.check_service_health(dep_id)
                    dependencies.append({
                        "service_id": dep_id,
                        "service_name": dep_service.service_name,
                        "service_type": dep_service.service_type.value,
                        "status": health_result.status.value if health_result else "unknown",
                        "response_time_ms": health_result.response_time_ms if health_result else 0
                    })
        
        return {
            "service_id": service_id,
            "service_name": service.service_name,
            "dependencies_count": len(dependencies),
            "healthy_dependencies": sum(1 for dep in dependencies if dep["status"] == "healthy"),
            "unhealthy_dependencies": sum(1 for dep in dependencies if dep["status"] != "healthy"),
            "dependencies": dependencies
        }
    
    async def deregister_mlops_service(self, service_id: str) -> bool:
        """Deregister an MLOps service.
        
        Args:
            service_id: ID of the service to deregister
            
        Returns:
            True if deregistration successful
        """
        # Remove from health check monitoring
        await self.health_check_port.unregister_health_check(service_id)
        
        # Mark as unhealthy in load balancer
        await self.load_balancer_port.mark_service_unhealthy(service_id)
        
        # Remove from registry
        return await self.service_registry_port.deregister_service(service_id)