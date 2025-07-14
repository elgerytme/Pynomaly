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

from .adapters import PyODAdapter, PyTorchAdapter, SKLearnAdapter, TensorFlowAdapter
from .cache import CacheManager, InMemoryCacheAdapter, RedisCacheAdapter
from .monitoring import HealthService, MetricsService, PerformanceMonitor
from .persistence import DatabaseManager, Repository, RepositoryFactory

__version__ = "0.1.1"
__all__ = [
    # Adapters
    "PyODAdapter",
    "SKLearnAdapter",
    "PyTorchAdapter",
    "TensorFlowAdapter",
    # Cache
    "CacheManager",
    "RedisCacheAdapter",
    "InMemoryCacheAdapter",
    # Monitoring
    "MetricsService",
    "HealthService",
    "PerformanceMonitor",
    # Persistence
    "DatabaseManager",
    "Repository",
    "RepositoryFactory",
]
