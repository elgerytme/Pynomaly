"""Advanced multi-layer caching system for anomaly detection platform.

This module provides comprehensive caching strategies including:
- Multi-layer hybrid caching (Memory + Redis + Disk)
- Domain-specific cache managers
- Intelligent cache eviction strategies
- Performance monitoring and statistics
- Cache decorators and utilities

Key Components:
- AdvancedCacheManager: Core cache management with multiple storage backends
- CachedDetectionService: Cached wrapper for anomaly detection operations
- CacheConfiguration: Flexible configuration system with profiles
- Multiple cache stores: Memory, Redis, Disk, and Hybrid implementations

Usage Examples:
    # Basic cache manager
    from anomaly_detection.infrastructure.caching import get_cache_manager
    cache = get_cache_manager()
    await cache.set("key", {"data": "value"})
    result = await cache.get("key")

    # Cached detection service
    from anomaly_detection.infrastructure.caching import CachedDetectionService, CacheProfile
    service = CachedDetectionService(cache_profile=CacheProfile.PRODUCTION)
    result = await service.detect_anomalies(data, algorithm="isolation_forest")

    # Domain-specific caching
    from anomaly_detection.infrastructure.caching import get_domain_cache_managers
    managers = get_domain_cache_managers()
    await managers.model_cache_manager.set("model_key", trained_model)
"""

from .advanced_cache_strategies import (
    # Core cache interfaces and implementations
    ICacheStore,
    MemoryCacheStore,
    RedisCacheStore,
    DiskCacheStore,
    HybridCacheStore,
    
    # Cache manager
    AdvancedCacheManager,
    get_cache_manager,
    
    # Cache entry and metadata
    CacheEntry,
    CacheStrategy,
    CacheLayer,
    
    # Decorators
    cache_detection_result,
    cache_model,
)

from .cache_config import (
    # Configuration classes
    CacheConfiguration,
    CacheConfigurationFactory,
    CacheManagerFactory,
    
    # Domain managers
    DomainCacheManagers,
    
    # Cache profiles
    CacheProfile,
    
    # Global functions
    get_cache_config,
    get_domain_cache_managers,
    initialize_cache_system,
    
    # Utility functions
    cache_model_prediction,
    get_cached_model_prediction,
    cache_preprocessed_data,
    get_cached_preprocessed_data,
)

from .cached_detection_service import (
    # Cached service wrapper
    CachedDetectionService,
)

# Version information
__version__ = "1.0.0"
__author__ = "Anomaly Detection Platform Team"
__description__ = "Advanced multi-layer caching system for anomaly detection"

# Default exports for easy access
__all__ = [
    # Core cache components
    "ICacheStore",
    "MemoryCacheStore", 
    "RedisCacheStore",
    "DiskCacheStore",
    "HybridCacheStore",
    "AdvancedCacheManager",
    "get_cache_manager",
    
    # Cache configuration
    "CacheConfiguration",
    "CacheConfigurationFactory", 
    "CacheManagerFactory",
    "DomainCacheManagers",
    "CacheProfile",
    "get_cache_config",
    "get_domain_cache_managers",
    "initialize_cache_system",
    
    # Cached services
    "CachedDetectionService",
    
    # Enums and data classes
    "CacheEntry",
    "CacheStrategy",
    "CacheLayer",
    
    # Decorators
    "cache_detection_result",
    "cache_model",
    
    # Utility functions
    "cache_model_prediction",
    "get_cached_model_prediction", 
    "cache_preprocessed_data",
    "get_cached_preprocessed_data",
]

# Module-level convenience functions
async def quick_cache_example():
    """Quick example of cache usage."""
    # Initialize cache system
    managers = initialize_cache_system(CacheProfile.DEVELOPMENT)
    
    # Cache some data
    await managers.detection_cache_manager.set("example_key", {"example": "data"})
    
    # Retrieve cached data
    result = await managers.detection_cache_manager.get("example_key")
    
    return result


def get_cache_info():
    """Get information about available cache strategies and profiles."""
    return {
        "strategies": [strategy.value for strategy in CacheStrategy],
        "layers": [layer.value for layer in CacheLayer],
        "profiles": [profile.value for profile in CacheProfile],
        "version": __version__,
        "description": __description__
    }


# Configuration validation
def validate_cache_configuration(config: CacheConfiguration) -> bool:
    """Validate cache configuration settings."""
    try:
        # Check memory cache settings
        if config.memory_cache_enabled and config.memory_cache_max_size <= 0:
            return False
        
        # Check disk cache settings
        if config.disk_cache_enabled and config.disk_cache_max_size_mb <= 0:
            return False
        
        # Check TTL settings
        if config.default_ttl_seconds <= 0:
            return False
        
        # Check Redis URL format (basic validation)
        if config.redis_cache_enabled and not config.redis_url.startswith(("redis://", "rediss://")):
            return False
        
        return True
        
    except Exception:
        return False


# Performance monitoring utilities
class CachePerformanceMonitor:
    """Monitor cache performance across the system."""
    
    def __init__(self):
        self.start_time = None
        self.metrics = {}
    
    async def collect_metrics(self) -> dict:
        """Collect performance metrics from all cache managers."""
        managers = get_domain_cache_managers()
        
        # Get combined statistics
        stats = await managers.get_combined_stats()
        
        # Calculate performance metrics
        performance_summary = {
            "total_cache_stores": len(stats),
            "overall_hit_rate": 0.0,
            "total_requests": 0,
            "total_hits": 0,
        }
        
        # Aggregate metrics across all cache stores
        for store_name, store_stats in stats.items():
            if store_stats and isinstance(store_stats, dict):
                total_requests = store_stats.get('total_requests', 0)
                hits = store_stats.get('hits', 0)
                
                performance_summary['total_requests'] += total_requests
                performance_summary['total_hits'] += hits
        
        # Calculate overall hit rate
        if performance_summary['total_requests'] > 0:
            performance_summary['overall_hit_rate'] = (
                performance_summary['total_hits'] / performance_summary['total_requests'] * 100
            )
        
        return {
            "summary": performance_summary,
            "detailed": stats
        }


# Export performance monitor
__all__.append("CachePerformanceMonitor")

# Module initialization
import logging
logger = logging.getLogger(__name__)
logger.info(f"Anomaly Detection Caching System v{__version__} initialized")