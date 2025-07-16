"""
Integration configuration entity for managing cross-package settings.
"""

from dataclasses import dataclass, field
from datetime import datetime
from typing import Dict, Any, List, Optional
from uuid import UUID, uuid4

from core.domain.abstractions.base_entity import BaseEntity


@dataclass
class PackageConfig:
    """Configuration for a specific package."""
    package_name: str
    enabled: bool = True
    config: Dict[str, Any] = field(default_factory=dict)
    dependencies: List[str] = field(default_factory=list)
    health_check_url: Optional[str] = None
    timeout_seconds: int = 300
    retry_count: int = 3
    circuit_breaker_threshold: int = 5
    cache_ttl_seconds: int = 3600


@dataclass
class PerformanceConfig:
    """Performance optimization configuration."""
    max_concurrent_operations: int = 10
    memory_limit_mb: int = 8192
    cache_size_mb: int = 1024
    connection_pool_size: int = 20
    query_timeout_seconds: int = 30
    batch_size: int = 1000
    enable_parallel_processing: bool = True
    enable_caching: bool = True
    enable_compression: bool = True
    enable_connection_pooling: bool = True


@dataclass
class MonitoringConfig:
    """Monitoring and observability configuration."""
    metrics_enabled: bool = True
    tracing_enabled: bool = True
    logging_level: str = "INFO"
    metrics_collection_interval: int = 60
    health_check_interval: int = 30
    alert_thresholds: Dict[str, float] = field(default_factory=dict)
    notification_channels: List[str] = field(default_factory=list)
    dashboard_enabled: bool = True
    export_metrics: bool = True


@dataclass
class SecurityConfig:
    """Security configuration for integration."""
    authentication_enabled: bool = True
    authorization_enabled: bool = True
    encryption_enabled: bool = True
    audit_logging_enabled: bool = True
    rate_limiting_enabled: bool = True
    max_requests_per_minute: int = 1000
    token_expiry_minutes: int = 60
    allowed_origins: List[str] = field(default_factory=list)
    ssl_verification: bool = True


@dataclass
class IntegrationConfig(BaseEntity):
    """Configuration entity for integration settings."""
    
    name: str
    description: str
    packages: Dict[str, PackageConfig] = field(default_factory=dict)
    performance: PerformanceConfig = field(default_factory=PerformanceConfig)
    monitoring: MonitoringConfig = field(default_factory=MonitoringConfig)
    security: SecurityConfig = field(default_factory=SecurityConfig)
    environment: str = "development"
    created_at: datetime = field(default_factory=datetime.utcnow)
    updated_at: datetime = field(default_factory=datetime.utcnow)
    created_by: Optional[str] = None
    tags: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def __post_init__(self):
        """Initialize configuration after creation."""
        if not self.id:
            self.id = uuid4()
            
        # Initialize default alert thresholds if not provided
        if not self.monitoring.alert_thresholds:
            self.monitoring.alert_thresholds = {
                "cpu_usage": 80.0,
                "memory_usage": 85.0,
                "error_rate": 5.0,
                "response_time": 5000.0,
                "disk_usage": 90.0
            }
            
    def add_package(self, package_config: PackageConfig) -> None:
        """Add a package configuration."""
        self.packages[package_config.package_name] = package_config
        self.updated_at = datetime.utcnow()
        
    def get_package_config(self, package_name: str) -> Optional[PackageConfig]:
        """Get configuration for a specific package."""
        return self.packages.get(package_name)
        
    def is_package_enabled(self, package_name: str) -> bool:
        """Check if a package is enabled."""
        config = self.get_package_config(package_name)
        return config.enabled if config else False
        
    def get_enabled_packages(self) -> List[str]:
        """Get list of enabled package names."""
        return [name for name, config in self.packages.items() if config.enabled]
        
    def update_package_config(self, package_name: str, updates: Dict[str, Any]) -> None:
        """Update configuration for a specific package."""
        if package_name in self.packages:
            config = self.packages[package_name]
            for key, value in updates.items():
                if hasattr(config, key):
                    setattr(config, key, value)
                elif key in config.config:
                    config.config[key] = value
                    
            self.updated_at = datetime.utcnow()
            
    def validate_configuration(self) -> List[str]:
        """Validate the configuration and return any issues."""
        issues = []
        
        # Check for circular dependencies
        for package_name, config in self.packages.items():
            if self._has_circular_dependency(package_name, config.dependencies):
                issues.append(f"Circular dependency detected for package: {package_name}")
                
        # Check performance limits
        if self.performance.max_concurrent_operations <= 0:
            issues.append("max_concurrent_operations must be greater than 0")
            
        if self.performance.memory_limit_mb <= 0:
            issues.append("memory_limit_mb must be greater than 0")
            
        # Check monitoring configuration
        if self.monitoring.metrics_collection_interval <= 0:
            issues.append("metrics_collection_interval must be greater than 0")
            
        # Check security configuration
        if self.security.max_requests_per_minute <= 0:
            issues.append("max_requests_per_minute must be greater than 0")
            
        return issues
        
    def _has_circular_dependency(self, package_name: str, dependencies: List[str], 
                                visited: Optional[List[str]] = None) -> bool:
        """Check for circular dependencies recursively."""
        if visited is None:
            visited = []
            
        if package_name in visited:
            return True
            
        visited.append(package_name)
        
        for dep_name in dependencies:
            if dep_name in self.packages:
                dep_config = self.packages[dep_name]
                if self._has_circular_dependency(dep_name, dep_config.dependencies, visited.copy()):
                    return True
                    
        return False
        
    def get_package_load_order(self) -> List[str]:
        """Get the order in which packages should be loaded based on dependencies."""
        load_order = []
        remaining = set(self.packages.keys())
        
        while remaining:
            # Find packages with no unresolved dependencies
            ready_packages = []
            for package_name in remaining:
                config = self.packages[package_name]
                if all(dep in load_order for dep in config.dependencies):
                    ready_packages.append(package_name)
                    
            if not ready_packages:
                # If no packages are ready, there might be circular dependencies
                # Add remaining packages anyway to avoid infinite loop
                ready_packages = list(remaining)
                
            for package_name in ready_packages:
                load_order.append(package_name)
                remaining.remove(package_name)
                
        return load_order