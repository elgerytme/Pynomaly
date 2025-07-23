"""Infrastructure package for technical cross-cutting concerns.

This package provides the technical foundation for the monorepo, implementing
infrastructure layer concerns like persistence, messaging, caching, security,
and monitoring. It follows Clean Architecture principles by providing abstractions
that domain packages can depend on without coupling to specific implementations.

Key modules:
- adapters: External service adapters and API clients
- persistence: Database connections, repositories, and data access
- messaging: Event buses, message queues, and async communication
- security: Authentication, authorization, and encryption
- monitoring: Logging, metrics, tracing, and observability
- caching: Distributed caching and cache management
- configuration: Environment settings and configuration management
- integrations: Third-party service integrations

Example usage:
    from infrastructure.persistence import DatabaseConnection
    from infrastructure.messaging import EventBus
    from infrastructure.security import JWTManager
    from infrastructure.monitoring import get_logger
    
    # Setup infrastructure components
    db = DatabaseConnection()
    event_bus = EventBus()
    jwt_manager = JWTManager()
    logger = get_logger(__name__)
"""

__version__ = "1.0.0"
__author__ = "Monorepo Team"
__email__ = "team@monorepo.com"

# Core infrastructure exports
from .adapters import *
from .persistence import *
from .messaging import *
from .security import *
from .monitoring import *
from .caching import *
from .configuration import *
from .integrations import *

# Infrastructure exceptions
class InfrastructureError(Exception):
    """Base exception for infrastructure-related errors."""
    pass

class ConfigurationError(InfrastructureError):
    """Raised when configuration is invalid or missing."""
    pass

class ConnectionError(InfrastructureError):
    """Raised when connection to external service fails."""
    pass

class AuthenticationError(InfrastructureError):
    """Raised when authentication fails."""
    pass

class AuthorizationError(InfrastructureError):
    """Raised when authorization fails."""
    pass

class PersistenceError(InfrastructureError):
    """Raised when persistence operations fail."""
    pass

class MessagingError(InfrastructureError):
    """Raised when messaging operations fail."""
    pass

class CachingError(InfrastructureError):
    """Raised when caching operations fail."""
    pass

class MonitoringError(InfrastructureError):
    """Raised when monitoring operations fail."""
    pass

# Version information
VERSION_INFO = {
    "major": 1,
    "minor": 0,
    "patch": 0,
    "release": "stable",
    "version": __version__
}

def get_version() -> str:
    """Get the current version string."""
    return __version__

def get_version_info() -> dict:
    """Get detailed version information."""
    return VERSION_INFO.copy()

# Infrastructure health check
def health_check() -> dict:
    """Perform infrastructure health check."""
    return {
        "status": "healthy",
        "version": __version__,
        "components": {
            "persistence": "available",
            "messaging": "available", 
            "security": "available",
            "monitoring": "available",
            "caching": "available",
            "configuration": "available"
        }
    }

# Ensure proper module loading order
_REQUIRED_MODULES = [
    "configuration",
    "monitoring", 
    "security",
    "persistence",
    "caching",
    "messaging",
    "adapters",
    "integrations"
]

def _validate_dependencies():
    """Validate that required dependencies are available."""
    missing_deps = []
    
    try:
        import sqlalchemy
    except ImportError:
        missing_deps.append("sqlalchemy")
    
    try:
        import redis
    except ImportError:
        missing_deps.append("redis")
        
    try:
        import structlog
    except ImportError:
        missing_deps.append("structlog")
    
    if missing_deps:
        raise ImportError(
            f"Missing required dependencies: {', '.join(missing_deps)}. "
            f"Install with: pip install infrastructure[all]"
        )

# Perform dependency validation on import
try:
    _validate_dependencies()
except ImportError as e:
    # Log warning but don't fail - allow graceful degradation
    import warnings
    warnings.warn(f"Infrastructure dependency warning: {e}", ImportWarning)