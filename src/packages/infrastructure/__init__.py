"""
Pynomaly Infrastructure Package

This package provides technical infrastructure components including:
- Database adapters and repositories
- Caching implementations
- Monitoring and observability
- Security and authentication
- Algorithm adapters
- External service integrations

Dependencies: Core package
"""

from .infrastructure.adapters import PyODAdapter, PyTorchAdapter, SklearnAdapter, TensorFlowAdapter
from .infrastructure.cache import CacheManager, InMemoryCache, RedisCache
from .infrastructure.monitoring import HealthService, PerformanceMonitor
from .infrastructure.persistence import DatabaseManager

__version__ = "0.1.1"
__all__ = [
    # Adapters
    "PyODAdapter",
    "SklearnAdapter",
    "PyTorchAdapter",
    "TensorFlowAdapter",
    # Cache
    "CacheManager",
    "RedisCache",
    "InMemoryCache",
    # Monitoring
    "HealthService",
    "PerformanceMonitor",
    # Persistence
    "DatabaseManager",
]
