"""
Monitoring Orchestrator

Central orchestration system for coordinating all monitoring and observability
components across the MLOps platform, providing unified configuration and
lifecycle management.
"""

import asyncio
import json
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Union
from dataclasses import dataclass, field
from enum import Enum
import yaml

import structlog

from mlops.infrastructure.monitoring.advanced_observability_platform import AdvancedObservabilityPlatform
from mlops.infrastructure.monitoring.model_drift_detector import ModelDriftDetector
from mlops.infrastructure.monitoring.pipeline_monitor import PipelineMonitor


class MonitoringService(Enum):
    """Available monitoring services."""
    OBSERVABILITY_PLATFORM = "observability_platform"
    DRIFT_DETECTOR = "drift_detector"
    PIPELINE_MONITOR = "pipeline_monitor"
    REAL_TIME_ANALYTICS = "real_time_analytics"
    BUSINESS_METRICS = "business_metrics"
    SECURITY_MONITOR = "security_monitor"


@dataclass
class ServiceConfig:
    """Configuration for a monitoring service."""
    name: str
    enabled: bool = True
    config: Dict[str, Any] = field(default_factory=dict)
    dependencies: List[str] = field(default_factory=list)
    health_check_interval: int = 60
    restart_on_failure: bool = True
    max_restart_attempts: int = 3


@dataclass
class MonitoringStack:
    """Complete monitoring stack configuration."""
    name: str
    description: str
    services: Dict[str, ServiceConfig] = field(default_factory=dict)
    global_config: Dict[str, Any] = field(default_factory=dict)
    deployment_environment: str = "development"
    created_at: datetime = field(default_factory=datetime.utcnow)


class MonitoringOrchestrator:
    """
    Central orchestrator for all monitoring and observability services
    in the MLOps platform.
    """
    
    def __init__(self, stack_config: Union[Dict[str, Any], str] = None):
        self.logger = structlog.get_logger(__name__)
        
        # Load configuration
        if isinstance(stack_config, str):
            self.stack = self._load_stack_from_file(stack_config)
        elif isinstance(stack_config, dict):
            self.stack = self._load_stack_from_dict(stack_config)
        else:
            self.stack = self._create_default_stack()
        
        # Service instances
        self.services: Dict[str, Any] = {}
        self.service_tasks: Dict[str, asyncio.Task] = {}
        self.health_check_tasks: Dict[str, asyncio.Task] = {}
        
        # Orchestrator state
        self.is_running = False
        self.startup_complete = False
        self.shutdown_initiated = False
        
        # Service dependencies graph
        self.dependency_graph = self._build_dependency_graph()
        
        # Monitoring metadata
        self.start_time: Optional[datetime] = None
        self.service_stats: Dict[str, Dict[str, Any]] = {}
    
    def _load_stack_from_file(self, config_file: str) -> MonitoringStack:
        """Load monitoring stack configuration from file."""
        try:
            with open(config_file, 'r') as f:
                if config_file.endswith('.yaml') or config_file.endswith('.yml'):
                    config_data = yaml.safe_load(f)
                else:
                    config_data = json.load(f)
            
            return self._load_stack_from_dict(config_data)
            
        except Exception as e:
            self.logger.error(f"Failed to load configuration from {config_file}: {e}")
            return self._create_default_stack()
    
    def _load_stack_from_dict(self, config_data: Dict[str, Any]) -> MonitoringStack:
        """Load monitoring stack configuration from dictionary."""
        services = {}
        
        for service_name, service_config in config_data.get("services", {}).items():
            services[service_name] = ServiceConfig(
                name=service_name,
                enabled=service_config.get("enabled", True),
                config=service_config.get("config", {}),
                dependencies=service_config.get("dependencies", []),
                health_check_interval=service_config.get("health_check_interval", 60),
                restart_on_failure=service_config.get("restart_on_failure", True),
                max_restart_attempts=service_config.get("max_restart_attempts", 3)
            )
        
        return MonitoringStack(
            name=config_data.get("name", "default_monitoring_stack"),
            description=config_data.get("description", "Default monitoring stack"),
            services=services,
            global_config=config_data.get("global_config", {}),
            deployment_environment=config_data.get("environment", "development")
        )
    
    def _create_default_stack(self) -> MonitoringStack:
        """Create default monitoring stack configuration."""
        services = {
            "observability_platform": ServiceConfig(
                name="observability_platform",
                enabled=True,
                config={
                    "enable_ai_insights": True,
                    "insights_interval_hours": 1,
                    "metrics_retention_days": 30
                },
                dependencies=[],
                health_check_interval=60
            ),
            "drift_detector": ServiceConfig(
                name="drift_detector",
                enabled=True,
                config={
                    "reference_window_days": 30,
                    "monitoring_window_hours": 24,
                    "drift_threshold": 0.05
                },
                dependencies=["observability_platform"],
                health_check_interval=120
            ),
            "pipeline_monitor": ServiceConfig(
                name="pipeline_monitor",
                enabled=True,
                config={
                    "enable_metrics": True,
                    "enable_performance_monitoring": True,
                    "enable_alerting": True,
                    "stage_timeout_minutes": 120
                },
                dependencies=["observability_platform"],
                health_check_interval=60
            ),
            "real_time_analytics": ServiceConfig(
                name="real_time_analytics",
                enabled=True,
                config={
                    "processing_interval_seconds": 30,
                    "buffer_size": 1000,
                    "enable_streaming": True
                },
                dependencies=["observability_platform", "pipeline_monitor"],
                health_check_interval=30
            )
        }
        
        return MonitoringStack(
            name="default_mlops_monitoring",
            description="Default MLOps monitoring and observability stack",
            services=services,
            global_config={
                "log_level": "INFO",
                "metrics_export_port": 8080,
                "dashboard_port": 3000,
                "enable_security_monitoring": True
            },
            deployment_environment="development"
        )
    
    def _build_dependency_graph(self) -> Dict[str, List[str]]:
        """Build service dependency graph."""
        graph = {}
        
        for service_name, service_config in self.stack.services.items():
            graph[service_name] = service_config.dependencies
        
        return graph
    
    def _get_startup_order(self) -> List[str]:
        """Get service startup order based on dependencies."""
        visited = set()
        order = []
        
        def dfs(service_name: str):
            if service_name in visited:
                return
            
            visited.add(service_name)
            
            # Start dependencies first
            for dependency in self.dependency_graph.get(service_name, []):
                if dependency in self.stack.services:
                    dfs(dependency)
            
            order.append(service_name)
        
        # Process all enabled services
        for service_name, service_config in self.stack.services.items():
            if service_config.enabled:
                dfs(service_name)
        
        return order
    
    async def start_stack(self) -> None:
        """Start the complete monitoring stack."""
        if self.is_running:
            self.logger.warning("Monitoring stack is already running")
            return
        
        try:
            self.start_time = datetime.utcnow()
            self.is_running = True
            
            # Get startup order
            startup_order = self._get_startup_order()
            
            self.logger.info(
                "Starting monitoring stack",
                stack_name=self.stack.name,
                environment=self.stack.deployment_environment,
                startup_order=startup_order
            )
            
            # Start services in dependency order
            for service_name in startup_order:
                if service_name in self.stack.services:
                    await self._start_service(service_name)
            
            # Start health monitoring for all services
            await self._start_health_monitoring()
            
            self.startup_complete = True
            
            self.logger.info(
                "Monitoring stack startup completed",
                services_started=len(self.services),
                total_startup_time=(datetime.utcnow() - self.start_time).total_seconds()
            )
            
        except Exception as e:
            self.logger.error(f"Failed to start monitoring stack: {e}")
            await self.stop_stack()
            raise
    
    async def _start_service(self, service_name: str) -> None:
        """Start a specific monitoring service."""
        service_config = self.stack.services[service_name]
        
        if not service_config.enabled:
            self.logger.info(f"Service {service_name} is disabled, skipping")
            return
        
        try:
            service_start_time = datetime.utcnow()
            
            # Create service instance based on type
            if service_name == "observability_platform":
                service_instance = AdvancedObservabilityPlatform(service_config.config)
                await service_instance.initialize_platform()
                
            elif service_name == "drift_detector":
                service_instance = ModelDriftDetector(**service_config.config)
                
            elif service_name == "pipeline_monitor":
                service_instance = PipelineMonitor()
                await service_instance.start_monitoring()
                
            elif service_name == "real_time_analytics":
                service_instance = await self._create_real_time_analytics_service(service_config.config)
                
            else:
                self.logger.warning(f"Unknown service type: {service_name}")
                return
            
            # Store service instance
            self.services[service_name] = service_instance
            
            # Initialize service statistics
            self.service_stats[service_name] = {
                "started_at": service_start_time,
                "startup_time_seconds": (datetime.utcnow() - service_start_time).total_seconds(),
                "restart_count": 0,
                "last_health_check": None,
                "health_status": "starting",
                "error_count": 0
            }
            
            self.logger.info(
                f"Service {service_name} started successfully",
                startup_time=self.service_stats[service_name]["startup_time_seconds"]
            )
            
        except Exception as e:
            self.logger.error(f"Failed to start service {service_name}: {e}")
            raise
    
    async def _create_real_time_analytics_service(self, config: Dict[str, Any]) -> Any:
        """Create real-time analytics service."""
        # This would create a real-time analytics service
        # For now, return a mock service
        
        class RealTimeAnalyticsService:
            def __init__(self, config):
                self.config = config
                self.is_running = False
            
            async def start(self):
                self.is_running = True
            
            async def stop(self):
                self.is_running = False
            
            async def health_check(self):
                return {"status": "healthy", "uptime": "100%"}
        
        service = RealTimeAnalyticsService(config)
        await service.start()
        return service
    
    async def _start_health_monitoring(self) -> None:
        """Start health monitoring for all services."""
        for service_name in self.services:
            task = asyncio.create_task(self._health_monitor_loop(service_name))
            self.health_check_tasks[service_name] = task
    
    async def _health_monitor_loop(self, service_name: str) -> None:
        """Health monitoring loop for a service."""
        service_config = self.stack.services[service_name]
        
        while self.is_running and not self.shutdown_initiated:
            try:
                # Perform health check
                health_status = await self._perform_health_check(service_name)
                
                # Update service statistics
                self.service_stats[service_name]["last_health_check"] = datetime.utcnow()
                self.service_stats[service_name]["health_status"] = health_status["status"]
                
                # Handle unhealthy services
                if health_status["status"] != "healthy":
                    await self._handle_unhealthy_service(service_name, health_status)
                
                # Wait before next check
                await asyncio.sleep(service_config.health_check_interval)
                
            except Exception as e:
                self.logger.error(
                    f"Health check failed for service {service_name}: {e}"
                )
                self.service_stats[service_name]["error_count"] += 1
                await asyncio.sleep(30)  # Wait before retry
    
    async def _perform_health_check(self, service_name: str) -> Dict[str, Any]:
        """Perform health check for a service."""
        service = self.services.get(service_name)
        
        if not service:
            return {"status": "not_found", "message": "Service not found"}
        
        try:
            # Different health check methods based on service type
            if hasattr(service, 'get_platform_health_status'):
                # Advanced observability platform
                health_data = await service.get_platform_health_status()
                return {
                    "status": "healthy" if health_data["overall_health_score"] > 0.7 else "degraded",
                    "details": health_data
                }
            
            elif hasattr(service, 'get_monitoring_status'):
                # Drift detector
                status = await service.get_monitoring_status()
                return {
                    "status": "healthy",
                    "details": status
                }
            
            elif hasattr(service, 'export_metrics_summary'):
                # Pipeline monitor
                summary = await service.export_metrics_summary()
                return {
                    "status": "healthy" if summary["monitoring_enabled"] else "degraded",
                    "details": summary
                }
            
            elif hasattr(service, 'health_check'):
                # Custom health check
                return await service.health_check()
            
            else:
                # Basic availability check
                return {"status": "healthy", "message": "Service is responsive"}
                
        except Exception as e:
            return {
                "status": "unhealthy",
                "message": f"Health check failed: {e}"
            }
    
    async def _handle_unhealthy_service(self, service_name: str, health_status: Dict[str, Any]) -> None:
        """Handle an unhealthy service."""
        service_config = self.stack.services[service_name]
        service_stats = self.service_stats[service_name]
        
        if not service_config.restart_on_failure:
            self.logger.warning(
                f"Service {service_name} is unhealthy but restart is disabled",
                health_status=health_status
            )
            return
        
        if service_stats["restart_count"] >= service_config.max_restart_attempts:
            self.logger.error(
                f"Service {service_name} exceeded max restart attempts",
                restart_count=service_stats["restart_count"],
                max_attempts=service_config.max_restart_attempts
            )
            return
        
        try:
            self.logger.warning(
                f"Restarting unhealthy service {service_name}",
                restart_count=service_stats["restart_count"]
            )
            
            # Stop the service
            await self._stop_service(service_name)
            
            # Wait a bit before restart
            await asyncio.sleep(5)
            
            # Restart the service
            await self._start_service(service_name)
            
            service_stats["restart_count"] += 1
            
            self.logger.info(f"Service {service_name} restarted successfully")
            
        except Exception as e:
            self.logger.error(f"Failed to restart service {service_name}: {e}")
            service_stats["error_count"] += 1
    
    async def stop_stack(self) -> None:
        """Stop the complete monitoring stack."""
        if not self.is_running:
            self.logger.warning("Monitoring stack is not running")
            return
        
        try:
            self.shutdown_initiated = True
            
            self.logger.info("Stopping monitoring stack")
            
            # Stop health monitoring tasks
            for task_name, task in self.health_check_tasks.items():
                if not task.done():
                    task.cancel()
            
            await asyncio.gather(*self.health_check_tasks.values(), return_exceptions=True)
            self.health_check_tasks.clear()
            
            # Stop services in reverse dependency order
            shutdown_order = list(reversed(self._get_startup_order()))
            
            for service_name in shutdown_order:
                if service_name in self.services:
                    await self._stop_service(service_name)
            
            self.is_running = False
            self.startup_complete = False
            
            self.logger.info("Monitoring stack stopped successfully")
            
        except Exception as e:
            self.logger.error(f"Error stopping monitoring stack: {e}")
            raise
    
    async def _stop_service(self, service_name: str) -> None:
        """Stop a specific service."""
        service = self.services.get(service_name)
        
        if not service:
            return
        
        try:
            if hasattr(service, 'shutdown'):
                await service.shutdown()
            elif hasattr(service, 'stop'):
                await service.stop()
            elif hasattr(service, 'stop_monitoring'):
                await service.stop_monitoring()
            
            self.logger.info(f"Service {service_name} stopped")
            
        except Exception as e:
            self.logger.error(f"Error stopping service {service_name}: {e}")
        
        finally:
            # Remove from services dict
            if service_name in self.services:
                del self.services[service_name]
    
    async def restart_service(self, service_name: str) -> bool:
        """Restart a specific service."""
        if service_name not in self.stack.services:
            self.logger.error(f"Service {service_name} not found in stack configuration")
            return False
        
        try:
            self.logger.info(f"Restarting service {service_name}")
            
            # Stop the service
            await self._stop_service(service_name)
            
            # Wait a bit
            await asyncio.sleep(2)
            
            # Start the service
            await self._start_service(service_name)
            
            # Restart health monitoring
            if service_name in self.health_check_tasks:
                self.health_check_tasks[service_name].cancel()
            
            task = asyncio.create_task(self._health_monitor_loop(service_name))
            self.health_check_tasks[service_name] = task
            
            self.logger.info(f"Service {service_name} restarted successfully")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to restart service {service_name}: {e}")
            return False
    
    async def get_stack_status(self) -> Dict[str, Any]:
        """Get comprehensive stack status."""
        uptime = (datetime.utcnow() - self.start_time).total_seconds() if self.start_time else 0
        
        service_statuses = {}
        for service_name in self.stack.services:
            if service_name in self.service_stats:
                stats = self.service_stats[service_name]
                service_statuses[service_name] = {
                    "enabled": self.stack.services[service_name].enabled,
                    "running": service_name in self.services,
                    "health_status": stats["health_status"],
                    "restart_count": stats["restart_count"],
                    "error_count": stats["error_count"],
                    "last_health_check": stats["last_health_check"].isoformat() if stats["last_health_check"] else None,
                    "uptime_seconds": (datetime.utcnow() - stats["started_at"]).total_seconds() if stats.get("started_at") else 0
                }
            else:
                service_statuses[service_name] = {
                    "enabled": self.stack.services[service_name].enabled,
                    "running": False,
                    "health_status": "not_started"
                }
        
        healthy_services = sum(1 for status in service_statuses.values() if status.get("health_status") == "healthy")
        total_enabled_services = sum(1 for service_config in self.stack.services.values() if service_config.enabled)
        
        return {
            "stack_name": self.stack.name,
            "environment": self.stack.deployment_environment,
            "is_running": self.is_running,
            "startup_complete": self.startup_complete,
            "uptime_seconds": uptime,
            "healthy_services": healthy_services,
            "total_enabled_services": total_enabled_services,
            "health_ratio": healthy_services / total_enabled_services if total_enabled_services > 0 else 0,
            "services": service_statuses,
            "last_updated": datetime.utcnow().isoformat()
        }
    
    async def get_service_metrics(self, service_name: str) -> Dict[str, Any]:
        """Get detailed metrics for a specific service."""
        if service_name not in self.services:
            return {"error": "Service not found or not running"}
        
        service = self.services[service_name]
        
        try:
            # Get service-specific metrics
            if hasattr(service, 'get_platform_health_status'):
                return await service.get_platform_health_status()
            elif hasattr(service, 'get_monitoring_status'):
                return await service.get_monitoring_status()
            elif hasattr(service, 'export_metrics_summary'):
                return await service.export_metrics_summary()
            else:
                return {"message": "No detailed metrics available for this service"}
                
        except Exception as e:
            return {"error": f"Failed to get metrics: {e}"}
    
    def get_service_instance(self, service_name: str) -> Optional[Any]:
        """Get a service instance for direct interaction."""
        return self.services.get(service_name)
    
    async def export_stack_configuration(self) -> Dict[str, Any]:
        """Export the current stack configuration."""
        return {
            "name": self.stack.name,
            "description": self.stack.description,
            "environment": self.stack.deployment_environment,
            "global_config": self.stack.global_config,
            "services": {
                name: {
                    "enabled": config.enabled,
                    "config": config.config,
                    "dependencies": config.dependencies,
                    "health_check_interval": config.health_check_interval,
                    "restart_on_failure": config.restart_on_failure,
                    "max_restart_attempts": config.max_restart_attempts
                }
                for name, config in self.stack.services.items()
            },
            "created_at": self.stack.created_at.isoformat()
        }