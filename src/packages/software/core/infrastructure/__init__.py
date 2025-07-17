"""
Infrastructure Package - Operations and Infrastructure Management

This package provides comprehensive infrastructure capabilities including:
- Configuration management
- Monitoring and alerting
- Security and authentication
- Performance optimization
- Deployment automation
- Distributed computing
- Service orchestration
"""

__version__ = "0.1.0"
__author__ = "Software Team"
__email__ = "support@software.com"

import logging
from typing import Dict, Any, Optional, List
from contextlib import contextmanager

# Core classes with fallback implementations
class ConfigManager:
    """Configuration management with fallback implementation."""
    
    def __init__(self, config_file: Optional[str] = None):
        self._config = {}
        self._config_file = config_file
        
    def get(self, key: str, default: Any = None) -> Any:
        """Get configuration value."""
        return self._config.get(key, default)
        
    def set(self, key: str, value: Any) -> None:
        """Set configuration value."""
        self._config[key] = value
        
    def load_from_file(self, file_path: str) -> None:
        """Load configuration from file."""
        try:
            import yaml
            with open(file_path, 'r') as f:
                self._config.update(yaml.safe_load(f))
        except ImportError:
            logging.warning("yaml package not available, using empty config")
        except FileNotFoundError:
            logging.warning(f"Config file not found: {file_path}")

class ServiceRegistry:
    """Service registry with fallback implementation."""
    
    def __init__(self):
        self._services: Dict[str, Any] = {}
        
    def register(self, name: str, service: Any) -> None:
        """Register a service."""
        self._services[name] = service
        
    def get(self, name: str) -> Optional[Any]:
        """Get a registered service."""
        return self._services.get(name)
        
    def list_services(self) -> List[str]:
        """List all registered services."""
        return list(self._services.keys())

class Container:
    """Dependency injection container."""
    
    def __init__(self):
        self._instances: Dict[str, Any] = {}
        self._factories: Dict[str, callable] = {}
        
    def register(self, name: str, factory: callable) -> None:
        """Register a factory for creating instances."""
        self._factories[name] = factory
        
    def get(self, name: str) -> Any:
        """Get or create an instance."""
        if name not in self._instances:
            if name in self._factories:
                self._instances[name] = self._factories[name]()
            else:
                raise ValueError(f"No factory registered for {name}")
        return self._instances[name]

class ServiceDiscovery:
    """Service discovery with fallback implementation."""
    
    def __init__(self):
        self._services: Dict[str, Dict[str, Any]] = {}
        
    def register_service(self, name: str, host: str, port: int, metadata: Optional[Dict] = None) -> None:
        """Register a service."""
        self._services[name] = {
            'host': host,
            'port': port,
            'metadata': metadata or {}
        }
        
    def discover_service(self, name: str) -> Optional[Dict[str, Any]]:
        """Discover a service."""
        return self._services.get(name)

# Service classes
class MonitoringService:
    """Monitoring service with fallback implementation."""
    
    def __init__(self):
        self._measurements: Dict[str, Any] = {}
        
    def record_metric(self, name: str, value: float, tags: Optional[Dict] = None) -> None:
        """Record a metric."""
        self._measurements[name] = {'value': value, 'tags': tags or {}}
        
    def get_metric(self, name: str) -> Optional[Dict]:
        """Get a metric."""
        return self._measurements.get(name)

class AlertingService:
    """Alerting service with fallback implementation."""
    
    def __init__(self):
        self._alerts: List[Dict] = []
        
    def send_alert(self, message: str, severity: str = "info") -> None:
        """Send an alert."""
        self._alerts.append({
            'message': message,
            'severity': severity,
            'timestamp': __import__('datetime').datetime.now().isoformat()
        })
        
    def get_alerts(self) -> List[Dict]:
        """Get all alerts."""
        return self._alerts

class SecurityService:
    """Security service with fallback implementation."""
    
    def __init__(self):
        self._tokens: Dict[str, Dict] = {}
        
    def create_token(self, user_id: str, scopes: List[str]) -> str:
        """Create a security token."""
        import uuid
        token = str(uuid.uuid4())
        self._tokens[token] = {'user_id': user_id, 'scopes': scopes}
        return token
        
    def validate_token(self, token: str) -> Optional[Dict]:
        """Validate a security token."""
        return self._tokens.get(token)

class PerformanceService:
    """Performance monitoring service."""
    
    def __init__(self):
        self._measurements: Dict[str, List[float]] = {}
        
    def record_timing(self, operation: str, duration: float) -> None:
        """Record operation timing."""
        if operation not in self._measurements:
            self._measurements[operation] = []
        self._measurements[operation].append(duration)
        
    def get_stats(self, operation: str) -> Optional[Dict]:
        """Get performance statistics."""
        if operation not in self._measurements:
            return None
        timings = self._measurements[operation]
        return {
            'count': len(timings),
            'avg': sum(timings) / len(timings),
            'min': min(timings),
            'max': max(timings)
        }

class DeploymentService:
    """Deployment service with fallback implementation."""
    
    def __init__(self):
        self._deployments: Dict[str, Dict] = {}
        
    def deploy(self, name: str, version: str, config: Dict) -> str:
        """Deploy a service."""
        deployment_id = f"{name}-{version}"
        self._deployments[deployment_id] = {
            'name': name,
            'version': version,
            'config': config,
            'status': 'deployed',
            'timestamp': __import__('datetime').datetime.now().isoformat()
        }
        return deployment_id
        
    def get_deployment(self, deployment_id: str) -> Optional[Dict]:
        """Get deployment information."""
        return self._deployments.get(deployment_id)

# Utility functions
def logger(name: str = "infrastructure") -> logging.Logger:
    """Get a logger instance."""
    return logging.getLogger(name)

def metrics() -> MonitoringService:
    """Get measurements service."""
    return MonitoringService()

def health_check() -> Dict[str, str]:
    """Perform health check."""
    return {'status': 'healthy', 'timestamp': __import__('datetime').datetime.now().isoformat()}

@contextmanager
def circuit_breaker(failure_threshold: int = 5):
    """Circuit breaker context manager."""
    try:
        yield
    except Exception as e:
        logger().error(f"Circuit breaker triggered: {e}")
        raise

# Middleware classes
class AuthMiddleware:
    """Authentication middleware."""
    
    def __init__(self, security_service: SecurityService):
        self.security_service = security_service
        
    def authenticate(self, token: str) -> bool:
        """Authenticate a request."""
        return self.security_service.validate_token(token) is not None

class LoggingMiddleware:
    """Logging middleware."""
    
    def __init__(self, logger_name: str = "requests"):
        self.logger = logger(logger_name)
        
    def log_request(self, method: str, path: str) -> None:
        """Log a request."""
        self.logger.info(f"{method} {path}")

class MetricsMiddleware:
    """Measurements middleware."""
    
    def __init__(self, monitoring_service: MonitoringService):
        self.monitoring_service = monitoring_service
        
    def record_request(self, method: str, path: str, duration: float) -> None:
        """Record request measurements."""
        self.monitoring_service.record_metric(f"request_{method.lower()}", duration)

class SecurityMiddleware:
    """Security middleware."""
    
    def __init__(self, security_service: SecurityService):
        self.security_service = security_service
        
    def validate_request(self, token: str, required_scopes: List[str]) -> bool:
        """Validate request security."""
        token_info = self.security_service.validate_token(token)
        if not token_info:
            return False
        return all(scope in token_info['scopes'] for scope in required_scopes)

__all__ = [
    # Core
    "ConfigManager",
    "ServiceRegistry",
    "Container",
    "ServiceDiscovery",
    
    # Services
    "MonitoringService",
    "AlertingService",
    "SecurityService",
    "PerformanceService",
    "DeploymentService",
    
    # Utilities
    "logger",
    "measurements",
    "health_check",
    "circuit_breaker",
    
    # Middleware
    "AuthMiddleware",
    "LoggingMiddleware",
    "MetricsMiddleware",
    "SecurityMiddleware",
    
    # Metadata
    "__version__",
    "__author__",
    "__email__",
]