"""
Example: Configuration composition patterns for different deployment scenarios.

This demonstrates how to compose packages for different environments:
1. Basic open-source deployment
2. Enterprise deployment with auth and monitoring
3. Custom deployment mixing specific features
"""

from typing import Dict, Any, Optional
from abc import ABC, abstractmethod

# Import stable interfaces
from interfaces.dto import HealthCheckRequest, HealthCheckResult
from interfaces.patterns import Service, Repository, HealthCheck, ConfigurationProvider

# Import shared infrastructure
from shared import (
    DIContainer, get_container, configure_container,
    register_service, register_repository
)


class BaseConfiguration(ABC):
    """Base configuration class for all deployment types."""
    
    def __init__(self):
        self.container = DIContainer()
        self._services_started = False
    
    @abstractmethod
    def configure_services(self) -> None:
        """Configure services in the DI container."""
        pass
    
    async def start_services(self) -> Dict[str, Any]:
        """Start all configured services."""
        if not self._services_started:
            self.configure_services()
            self._services_started = True
        
        # Return service registry
        return {
            "container": self.container,
            "health_check": self.container.resolve(HealthCheck) if self.container.is_registered(HealthCheck) else None
        }
    
    async def stop_services(self) -> None:
        """Stop all services."""
        self._services_started = False


# Mock implementations for demonstration
class MockDataQualityService(Service):
    """Mock data quality service."""
    
    async def execute(self, request) -> Any:
        return {"status": "completed", "score": 0.95}
    
    async def validate_request(self, request) -> bool:
        return True
    
    def get_service_info(self) -> Dict[str, Any]:
        return {"name": "MockDataQualityService", "domain": "data"}


class MockAnomalyDetectionService(Service):
    """Mock anomaly detection service."""
    
    async def execute(self, request) -> Any:
        return {"status": "completed", "anomalies": 3}
    
    async def validate_request(self, request) -> bool:
        return True
    
    def get_service_info(self) -> Dict[str, Any]:
        return {"name": "MockAnomalyDetectionService", "domain": "ai"}


class MockAuthService:
    """Mock enterprise authentication service."""
    
    def __init__(self, saml_enabled: bool = False):
        self.saml_enabled = saml_enabled
    
    async def authenticate(self, token: str) -> Dict[str, Any]:
        return {"user_id": "test_user", "roles": ["analyst"]}
    
    def get_info(self) -> Dict[str, Any]:
        return {"name": "AuthService", "saml_enabled": self.saml_enabled}


class MockMonitoringService:
    """Mock monitoring service."""
    
    def __init__(self, provider: str = "prometheus"):
        self.provider = provider
    
    def record_metric(self, name: str, value: float) -> None:
        print(f"Recording metric {name}: {value}")
    
    def get_info(self) -> Dict[str, Any]:
        return {"name": "MonitoringService", "provider": self.provider}


class MockHealthCheck(HealthCheck):
    """Mock health check service."""
    
    async def check_health(self) -> Dict[str, Any]:
        return {
            "status": "healthy",
            "timestamp": "2024-01-01T00:00:00Z",
            "checks": {
                "database": "healthy",
                "cache": "healthy"
            }
        }
    
    def get_component_name(self) -> str:
        return "system"
    
    async def check_dependencies(self) -> Dict[str, Dict[str, Any]]:
        return {
            "database": {"status": "healthy", "response_time": 10},
            "cache": {"status": "healthy", "response_time": 5}
        }


# Configuration compositions
class BasicOpenSourceConfiguration(BaseConfiguration):
    """
    Basic open-source configuration with core domain services only.
    
    Features:
    - Data quality service
    - Anomaly detection service
    - Basic health checks
    - No enterprise features
    """
    
    def configure_services(self) -> None:
        """Configure basic services."""
        # Register domain services
        register_service(self.container, Service, MockDataQualityService)
        self.container.register_singleton("data_quality", MockDataQualityService)
        self.container.register_singleton("anomaly_detection", MockAnomalyDetectionService)
        
        # Register health check
        self.container.register_singleton(HealthCheck, MockHealthCheck)
        
        print("✓ Configured basic open-source services")


class EnterpriseConfiguration(BaseConfiguration):
    """
    Enterprise configuration with full feature set.
    
    Features:
    - All basic services
    - Enterprise authentication (SAML)
    - Advanced monitoring (DataDog)
    - Multi-tenancy support
    - Audit logging
    """
    
    def __init__(self, auth_config: Optional[Dict[str, Any]] = None,
                 monitoring_config: Optional[Dict[str, Any]] = None):
        super().__init__()
        self.auth_config = auth_config or {"saml_enabled": True}
        self.monitoring_config = monitoring_config or {"provider": "datadog"}
    
    def configure_services(self) -> None:
        """Configure enterprise services."""
        # Configure basic services first
        basic_config = BasicOpenSourceConfiguration()
        basic_config.container = self.container
        basic_config.configure_services()
        
        # Add enterprise services
        auth_service = MockAuthService(saml_enabled=self.auth_config.get("saml_enabled", False))
        self.container.register_singleton("auth", lambda: auth_service)
        
        monitoring_service = MockMonitoringService(provider=self.monitoring_config.get("provider", "prometheus"))
        self.container.register_singleton("monitoring", lambda: monitoring_service)
        
        # Register enterprise-specific health checks
        self.container.register_singleton("enterprise_health", MockHealthCheck)
        
        print("✓ Configured enterprise services with auth and monitoring")


class CustomConfiguration(BaseConfiguration):
    """
    Custom configuration that mixes and matches features.
    
    This example shows how to create a configuration that:
    - Uses basic domain services
    - Adds monitoring but not auth
    - Uses specific integrations
    """
    
    def __init__(self, features: Dict[str, bool]):
        super().__init__()
        self.features = features
    
    def configure_services(self) -> None:
        """Configure services based on feature flags."""
        # Always include basic services
        register_service(self.container, Service, MockDataQualityService)
        self.container.register_singleton("data_quality", MockDataQualityService)
        self.container.register_singleton("anomaly_detection", MockAnomalyDetectionService)
        
        # Conditionally add features
        if self.features.get("monitoring", False):
            monitoring_service = MockMonitoringService(provider="prometheus")
            self.container.register_singleton("monitoring", lambda: monitoring_service)
            print("✓ Added monitoring service")
        
        if self.features.get("auth", False):
            auth_service = MockAuthService(saml_enabled=False)
            self.container.register_singleton("auth", lambda: auth_service)
            print("✓ Added authentication service")
        
        if self.features.get("health_checks", True):
            self.container.register_singleton(HealthCheck, MockHealthCheck)
            print("✓ Added health checks")
        
        print(f"✓ Configured custom deployment with features: {list(self.features.keys())}")


class ConfigurationFactory:
    """Factory for creating different configuration types."""
    
    @staticmethod
    def create_basic() -> BasicOpenSourceConfiguration:
        """Create basic open-source configuration."""
        return BasicOpenSourceConfiguration()
    
    @staticmethod
    def create_enterprise(auth_config: Optional[Dict[str, Any]] = None,
                         monitoring_config: Optional[Dict[str, Any]] = None) -> EnterpriseConfiguration:
        """Create enterprise configuration."""
        return EnterpriseConfiguration(auth_config, monitoring_config)
    
    @staticmethod
    def create_custom(features: Dict[str, bool]) -> CustomConfiguration:
        """Create custom configuration."""
        return CustomConfiguration(features)
    
    @staticmethod
    def create_from_environment() -> BaseConfiguration:
        """Create configuration based on environment variables."""
        import os
        
        deployment_type = os.getenv("DEPLOYMENT_TYPE", "basic")
        
        if deployment_type == "enterprise":
            return ConfigurationFactory.create_enterprise()
        elif deployment_type == "custom":
            # Parse feature flags from environment
            features = {
                "monitoring": os.getenv("ENABLE_MONITORING", "false").lower() == "true",
                "auth": os.getenv("ENABLE_AUTH", "false").lower() == "true",
                "health_checks": os.getenv("ENABLE_HEALTH_CHECKS", "true").lower() == "true"
            }
            return ConfigurationFactory.create_custom(features)
        else:
            return ConfigurationFactory.create_basic()


async def demonstrate_configurations():
    """Demonstrate different configuration patterns."""
    print("=== Package Composition Examples ===\n")
    
    # 1. Basic open-source deployment
    print("1. Basic Open-Source Configuration:")
    basic_config = ConfigurationFactory.create_basic()
    basic_services = await basic_config.start_services()
    
    health_check = basic_services.get("health_check")
    if health_check:
        health_result = await health_check.check_health()
        print(f"   Health check: {health_result['status']}")
    
    await basic_config.stop_services()
    print()
    
    # 2. Enterprise deployment
    print("2. Enterprise Configuration:")
    enterprise_config = ConfigurationFactory.create_enterprise(
        auth_config={"saml_enabled": True, "enable_rbac": True},
        monitoring_config={"provider": "datadog", "api_key": "***"}
    )
    enterprise_services = await enterprise_config.start_services()
    
    # Test enterprise services
    container = enterprise_services["container"]
    auth_service = container.resolve("auth")
    monitoring_service = container.resolve("monitoring")
    
    print(f"   Auth service: {auth_service.get_info()}")
    print(f"   Monitoring: {monitoring_service.get_info()}")
    
    await enterprise_config.stop_services()
    print()
    
    # 3. Custom deployment
    print("3. Custom Configuration:")
    custom_config = ConfigurationFactory.create_custom({
        "monitoring": True,
        "auth": False,
        "health_checks": True,
        "caching": False
    })
    custom_services = await custom_config.start_services()
    
    container = custom_services["container"]
    if container.is_registered("monitoring"):
        monitoring = container.resolve("monitoring")
        print(f"   Monitoring enabled: {monitoring.get_info()}")
    
    await custom_config.stop_services()
    print()
    
    # 4. Environment-based configuration
    print("4. Environment-Based Configuration:")
    import os
    os.environ["DEPLOYMENT_TYPE"] = "custom"
    os.environ["ENABLE_MONITORING"] = "true"
    os.environ["ENABLE_AUTH"] = "false"
    
    env_config = ConfigurationFactory.create_from_environment()
    env_services = await env_config.start_services()
    print(f"   Created configuration based on environment variables")
    
    await env_config.stop_services()


if __name__ == "__main__":
    import asyncio
    asyncio.run(demonstrate_configurations())