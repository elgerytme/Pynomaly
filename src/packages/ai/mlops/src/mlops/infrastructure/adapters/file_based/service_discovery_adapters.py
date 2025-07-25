"""File-based implementations for service discovery operations."""

import json
import asyncio
import hashlib
import time

try:
    import aiohttp
    HAS_AIOHTTP = True
except ImportError:
    HAS_AIOHTTP = False
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional, Callable
from dataclasses import dataclass, asdict

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


class FileBasedServiceRegistry(ServiceRegistryPort):
    """File-based service registry implementation."""
    
    def __init__(self, registry_dir: str):
        self.registry_dir = Path(registry_dir)
        self.registry_dir.mkdir(parents=True, exist_ok=True)
        self.services_file = self.registry_dir / "services.json"
        self._initialize_registry()
    
    def _initialize_registry(self):
        """Initialize the registry file if it doesn't exist."""
        if not self.services_file.exists():
            self.services_file.write_text(json.dumps({}))
    
    async def _load_services(self) -> Dict[str, Dict[str, Any]]:
        """Load services from file."""
        try:
            return json.loads(self.services_file.read_text())
        except (json.JSONDecodeError, FileNotFoundError):
            return {}
    
    async def _save_services(self, services: Dict[str, Dict[str, Any]]):
        """Save services to file."""
        self.services_file.write_text(json.dumps(services, indent=2, default=str))
    
    def _service_to_dict(self, service: ServiceMetadata) -> Dict[str, Any]:
        """Convert service metadata to dictionary."""
        service_dict = asdict(service)
        # Convert enums to strings
        service_dict["service_type"] = service.service_type.value
        service_dict["status"] = service.status.value
        # Convert datetime objects to ISO strings
        if service.registered_at:
            service_dict["registered_at"] = service.registered_at.isoformat()
        if service.last_heartbeat:
            service_dict["last_heartbeat"] = service.last_heartbeat.isoformat()
        return service_dict
    
    def _dict_to_service(self, service_dict: Dict[str, Any]) -> ServiceMetadata:
        """Convert dictionary to service metadata."""
        # Convert string enums back to enum objects
        service_dict["service_type"] = ServiceType(service_dict["service_type"])
        service_dict["status"] = ServiceStatus(service_dict["status"])
        
        # Convert ISO strings back to datetime objects
        if service_dict.get("registered_at"):
            service_dict["registered_at"] = datetime.fromisoformat(service_dict["registered_at"])
        if service_dict.get("last_heartbeat"):
            service_dict["last_heartbeat"] = datetime.fromisoformat(service_dict["last_heartbeat"])
        
        # Convert endpoints
        endpoints = []
        for ep_dict in service_dict.get("endpoints", []):
            endpoints.append(ServiceEndpoint(**ep_dict))
        service_dict["endpoints"] = endpoints
        
        return ServiceMetadata(**service_dict)
    
    async def register_service(self, service: ServiceMetadata) -> bool:
        """Register a service in the registry."""
        try:
            services = await self._load_services()
            service.registered_at = datetime.now()
            service.last_heartbeat = datetime.now()
            services[service.service_id] = self._service_to_dict(service)
            await self._save_services(services)
            return True
        except Exception:
            return False
    
    async def deregister_service(self, service_id: str) -> bool:
        """Deregister a service from the registry."""
        try:
            services = await self._load_services()
            if service_id in services:
                del services[service_id]
                await self._save_services(services)
            return True
        except Exception:
            return False
    
    async def get_service(self, service_id: str) -> Optional[ServiceMetadata]:
        """Get service metadata by ID."""
        try:
            services = await self._load_services()
            service_dict = services.get(service_id)
            if service_dict:
                return self._dict_to_service(service_dict)
            return None
        except Exception:
            return None
    
    async def find_services(self, query: ServiceQuery) -> List[ServiceMetadata]:
        """Find services matching the query."""
        try:
            services = await self._load_services()
            results = []
            
            for service_dict in services.values():
                service = self._dict_to_service(service_dict)
                
                # Apply filters
                if query.service_type and service.service_type != query.service_type:
                    continue
                if query.service_name and query.service_name not in service.service_name:
                    continue
                if query.status and service.status != query.status:
                    continue
                if query.tags:
                    if not service.tags:
                        continue
                    if not all(service.tags.get(k) == v for k, v in query.tags.items()):
                        continue
                if query.capabilities:
                    if not service.capabilities:
                        continue
                    if not all(cap in service.capabilities for cap in query.capabilities):
                        continue
                
                results.append(service)
            
            return results
        except Exception:
            return []
    
    async def list_all_services(self) -> List[ServiceMetadata]:
        """List all registered services."""
        try:
            services = await self._load_services()
            return [self._dict_to_service(service_dict) for service_dict in services.values()]
        except Exception:
            return []
    
    async def update_service_status(self, service_id: str, status: ServiceStatus) -> bool:
        """Update service status."""
        try:
            services = await self._load_services()
            if service_id in services:
                services[service_id]["status"] = status.value
                await self._save_services(services)
                return True
            return False
        except Exception:
            return False
    
    async def heartbeat(self, service_id: str, metadata: Optional[Dict[str, Any]] = None) -> bool:
        """Send heartbeat for a service."""
        try:
            services = await self._load_services()
            if service_id in services:
                services[service_id]["last_heartbeat"] = datetime.now().isoformat()
                if metadata:
                    services[service_id]["metadata"].update(metadata)
                await self._save_services(services)
                return True
            return False
        except Exception:
            return False


class FileBasedServiceDiscovery(ServiceDiscoveryPort):
    """File-based service discovery implementation."""
    
    def __init__(self, service_registry: FileBasedServiceRegistry):
        self.service_registry = service_registry
        self._watches: Dict[str, Dict[str, Any]] = {}
        self._watch_tasks: Dict[str, asyncio.Task] = {}
    
    async def discover_service(
        self, 
        service_type: ServiceType, 
        requirements: Optional[Dict[str, Any]] = None
    ) -> Optional[ServiceMetadata]:
        """Discover a service of the specified type."""
        services = await self.discover_services(service_type, requirements)
        return services[0] if services else None
    
    async def discover_services(
        self, 
        service_type: ServiceType, 
        requirements: Optional[Dict[str, Any]] = None
    ) -> List[ServiceMetadata]:
        """Discover all services of the specified type."""
        query = ServiceQuery(
            service_type=service_type,
            status=ServiceStatus.HEALTHY
        )
        
        if requirements:
            if "version" in requirements:
                query.version_pattern = requirements["version"]
            if "capabilities" in requirements:
                query.capabilities = requirements["capabilities"]
            if "tags" in requirements:
                query.tags = requirements["tags"]
        
        return await self.service_registry.find_services(query)
    
    async def get_service_endpoint(
        self, 
        service_type: ServiceType, 
        endpoint_name: str = "default"
    ) -> Optional[ServiceEndpoint]:
        """Get endpoint for a service type."""
        service = await self.discover_service(service_type)
        if service and service.endpoints:
            # Return first endpoint if no specific name requested
            if endpoint_name == "default":
                return service.endpoints[0]
            # Look for named endpoint
            for endpoint in service.endpoints:
                if endpoint.path.endswith(endpoint_name):
                    return endpoint
        return None
    
    async def watch_services(
        self, 
        service_type: ServiceType, 
        callback: Callable[[List[ServiceMetadata]], None]
    ) -> str:
        """Watch for changes in services of a specific type."""
        watch_id = f"watch_{service_type.value}_{int(time.time())}"
        
        self._watches[watch_id] = {
            "service_type": service_type,
            "callback": callback,
            "last_services": []
        }
        
        # Start watch task
        task = asyncio.create_task(self._watch_loop(watch_id))
        self._watch_tasks[watch_id] = task
        
        return watch_id
    
    async def _watch_loop(self, watch_id: str):
        """Watch loop for service changes."""
        watch_info = self._watches[watch_id]
        service_type = watch_info["service_type"]
        callback = watch_info["callback"]
        
        while watch_id in self._watches:
            try:
                current_services = await self.discover_services(service_type)
                last_services = watch_info["last_services"]
                
                # Check if services changed
                if self._services_changed(last_services, current_services):
                    watch_info["last_services"] = current_services
                    callback(current_services)
                
                await asyncio.sleep(5)  # Check every 5 seconds
            except Exception:
                # Continue watching even if there's an error
                await asyncio.sleep(5)
    
    def _services_changed(
        self, 
        old_services: List[ServiceMetadata], 
        new_services: List[ServiceMetadata]
    ) -> bool:
        """Check if services list changed."""
        if len(old_services) != len(new_services):
            return True
        
        old_ids = {s.service_id for s in old_services}
        new_ids = {s.service_id for s in new_services}
        
        return old_ids != new_ids
    
    async def cancel_watch(self, watch_id: str) -> bool:
        """Cancel a service watch."""
        if watch_id in self._watches:
            del self._watches[watch_id]
        
        if watch_id in self._watch_tasks:
            task = self._watch_tasks[watch_id]
            task.cancel()
            del self._watch_tasks[watch_id]
            return True
        
        return False


class FileBasedHealthCheck(HealthCheckPort):
    """File-based health check implementation."""
    
    def __init__(self, health_dir: str):
        self.health_dir = Path(health_dir)
        self.health_dir.mkdir(parents=True, exist_ok=True)
        self._registered_checks: Dict[str, Dict[str, Any]] = {}
        self._check_tasks: Dict[str, asyncio.Task] = {}
    
    async def perform_health_check(self, service: ServiceMetadata) -> HealthCheckResult:
        """Perform health check on a service."""
        start_time = time.time()
        
        try:
            if service.health_check_url and HAS_AIOHTTP:
                async with aiohttp.ClientSession() as session:
                    async with session.get(service.health_check_url, timeout=aiohttp.ClientTimeout(total=10)) as response:
                        response_time = (time.time() - start_time) * 1000
                        
                        if response.status == 200:
                            status = ServiceStatus.HEALTHY
                            details = {"http_status": response.status}
                        else:
                            status = ServiceStatus.UNHEALTHY
                            details = {"http_status": response.status, "response_text": await response.text()}
                        
                        return HealthCheckResult(
                            service_id=service.service_id,
                            status=status,
                            response_time_ms=response_time,
                            details=details,
                            checked_at=datetime.now()
                        )
            elif service.endpoints and HAS_AIOHTTP:
                # Simple ping check to first endpoint
                endpoint = service.endpoints[0]
                async with aiohttp.ClientSession() as session:
                    health_url = f"{endpoint.protocol}://{endpoint.host}:{endpoint.port}/health"
                    async with session.get(health_url, timeout=aiohttp.ClientTimeout(total=10)) as response:
                        response_time = (time.time() - start_time) * 1000
                        
                        status = ServiceStatus.HEALTHY if response.status == 200 else ServiceStatus.UNHEALTHY
                        return HealthCheckResult(
                            service_id=service.service_id,
                            status=status,
                            response_time_ms=response_time,
                            details={"http_status": response.status},
                            checked_at=datetime.now()
                        )
                
        except Exception as e:
            response_time = (time.time() - start_time) * 1000
            return HealthCheckResult(
                service_id=service.service_id,
                status=ServiceStatus.UNHEALTHY,
                response_time_ms=response_time,
                details={"error": str(e)},
                checked_at=datetime.now(),
                error_message=str(e)
            )
        
        # Fallback: assume healthy if no check URL
        return HealthCheckResult(
            service_id=service.service_id,
            status=ServiceStatus.HEALTHY,
            response_time_ms=0,
            details={"check_type": "no_health_endpoint"},
            checked_at=datetime.now()
        )
    
    async def perform_bulk_health_check(
        self, 
        services: List[ServiceMetadata]
    ) -> List[HealthCheckResult]:
        """Perform health checks on multiple services."""
        tasks = [self.perform_health_check(service) for service in services]
        return await asyncio.gather(*tasks, return_exceptions=False)
    
    async def register_health_check(
        self, 
        service_id: str, 
        check_url: str, 
        interval_seconds: int = 30
    ) -> bool:
        """Register a recurring health check."""
        try:
            self._registered_checks[service_id] = {
                "check_url": check_url,
                "interval_seconds": interval_seconds
            }
            
            # Start recurring health check task
            task = asyncio.create_task(self._recurring_health_check(service_id))
            self._check_tasks[service_id] = task
            
            return True
        except Exception:
            return False
    
    async def _recurring_health_check(self, service_id: str):
        """Run recurring health check for a service."""
        check_info = self._registered_checks[service_id]
        interval = check_info["interval_seconds"]
        
        while service_id in self._registered_checks:
            try:
                # Create dummy service metadata for check
                service = ServiceMetadata(
                    service_id=service_id,
                    service_name=service_id,
                    service_type=ServiceType.MONITORING,  # Dummy type
                    version="1.0.0",
                    endpoints=[],
                    status=ServiceStatus.UNKNOWN,
                    health_check_url=check_info["check_url"]
                )
                
                result = await self.perform_health_check(service)
                
                # Save result to file
                result_file = self.health_dir / f"{service_id}_health.json"
                result_data = {
                    "service_id": result.service_id,
                    "status": result.status.value,
                    "response_time_ms": result.response_time_ms,
                    "details": result.details,
                    "checked_at": result.checked_at.isoformat() if result.checked_at else None,
                    "error_message": result.error_message
                }
                result_file.write_text(json.dumps(result_data, indent=2))
                
                await asyncio.sleep(interval)
            except Exception:
                await asyncio.sleep(interval)
    
    async def unregister_health_check(self, service_id: str) -> bool:
        """Unregister a health check."""
        if service_id in self._registered_checks:
            del self._registered_checks[service_id]
        
        if service_id in self._check_tasks:
            task = self._check_tasks[service_id]
            task.cancel()
            del self._check_tasks[service_id]
            return True
        
        return False


class FileBasedLoadBalancer(LoadBalancerPort):
    """File-based load balancer implementation."""
    
    def __init__(self, service_discovery: FileBasedServiceDiscovery, lb_dir: str):
        self.service_discovery = service_discovery
        self.lb_dir = Path(lb_dir)
        self.lb_dir.mkdir(parents=True, exist_ok=True)
        self._round_robin_state: Dict[str, int] = {}
        self._service_metrics: Dict[str, Dict[str, float]] = {}
        self._unhealthy_services: set = set()
    
    async def get_balanced_service(
        self, 
        service_type: ServiceType, 
        strategy: str = "round_robin"
    ) -> Optional[ServiceMetadata]:
        """Get a service using load balancing."""
        services = await self.service_discovery.discover_services(service_type)
        
        # Filter out unhealthy services
        healthy_services = [s for s in services if s.service_id not in self._unhealthy_services]
        
        if not healthy_services:
            return None
        
        if strategy == "round_robin":
            return self._round_robin_selection(service_type, healthy_services)
        elif strategy == "least_connections":
            return self._least_connections_selection(healthy_services)
        elif strategy == "random":
            import random
            return random.choice(healthy_services)
        else:
            # Default to round robin
            return self._round_robin_selection(service_type, healthy_services)
    
    def _round_robin_selection(
        self, 
        service_type: ServiceType, 
        services: List[ServiceMetadata]
    ) -> ServiceMetadata:
        """Round robin service selection."""
        type_key = service_type.value
        current_index = self._round_robin_state.get(type_key, 0)
        selected_service = services[current_index % len(services)]
        self._round_robin_state[type_key] = (current_index + 1) % len(services)
        return selected_service
    
    def _least_connections_selection(self, services: List[ServiceMetadata]) -> ServiceMetadata:
        """Least connections service selection."""
        best_service = services[0]
        best_connections = self._service_metrics.get(best_service.service_id, {}).get("active_connections", 0)
        
        for service in services[1:]:
            connections = self._service_metrics.get(service.service_id, {}).get("active_connections", 0)
            if connections < best_connections:
                best_service = service
                best_connections = connections
        
        return best_service
    
    async def report_service_metrics(
        self, 
        service_id: str, 
        metrics: Dict[str, float]
    ) -> bool:
        """Report metrics for load balancing decisions."""
        try:
            self._service_metrics[service_id] = metrics
            
            # Save metrics to file
            metrics_file = self.lb_dir / f"{service_id}_metrics.json"
            metrics_data = {
                "service_id": service_id,
                "metrics": metrics,
                "reported_at": datetime.now().isoformat()
            }
            metrics_file.write_text(json.dumps(metrics_data, indent=2))
            
            return True
        except Exception:
            return False
    
    async def mark_service_unhealthy(self, service_id: str) -> bool:
        """Mark a service as unhealthy for load balancing."""
        try:
            self._unhealthy_services.add(service_id)
            
            # Save unhealthy services list
            unhealthy_file = self.lb_dir / "unhealthy_services.json"
            unhealthy_data = {
                "unhealthy_services": list(self._unhealthy_services),
                "updated_at": datetime.now().isoformat()
            }
            unhealthy_file.write_text(json.dumps(unhealthy_data, indent=2))
            
            return True
        except Exception:
            return False
    
    async def mark_service_healthy(self, service_id: str) -> bool:
        """Mark a service as healthy for load balancing."""
        try:
            self._unhealthy_services.discard(service_id)
            
            # Save unhealthy services list
            unhealthy_file = self.lb_dir / "unhealthy_services.json"
            unhealthy_data = {
                "unhealthy_services": list(self._unhealthy_services),
                "updated_at": datetime.now().isoformat()
            }
            unhealthy_file.write_text(json.dumps(unhealthy_data, indent=2))
            
            return True
        except Exception:
            return False