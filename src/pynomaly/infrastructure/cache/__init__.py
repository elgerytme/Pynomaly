"""Comprehensive caching system for Pynomaly."""

from .redis_cache import (
    RedisCache,
    CacheKeys,
    CachedRepository,
    DetectorCacheDecorator,
    init_cache,
    get_cache,
)

from .intelligent_cache import (
    IntelligentCacheManager,
    CacheStrategy,
    CompressionType,
    SerializationFormat,
    CacheEntry,
    CacheStats,
    AccessPattern,
    get_intelligent_cache_manager,
    close_intelligent_cache_manager,
)

from .decorators import (
    CacheConfig,
    CacheKeyGenerator,
    CacheDecorator,
    AsyncCacheDecorator,
    SyncCacheDecorator,
    CacheInvalidator,
    set_cache_manager,
    get_cache_manager,
    get_cache_invalidator,
    cached,
    cache_result,
    cache_expensive,
    cache_model_prediction,
    cache_database_query,
    cache_context,
    invalidate_cache,
    invalidate_cache_pattern,
    warm_cache,
    get_cache_stats,
)

from .cache_integration import (
    CacheConfiguration,
    CacheHealthMonitor,
    CacheIntegrationManager,
    get_cache_integration_manager,
    close_cache_integration_manager,
    cache_system_context,
    get_cache_health,
    get_cache_statistics,
    perform_cache_maintenance,
    warm_cache_with_critical_data,
)

__all__ = [
    # Redis cache
    "RedisCache",
    "CacheKeys",
    "CachedRepository",
    "DetectorCacheDecorator",
    "init_cache",
    "get_cache",
    
    # Intelligent cache
    "IntelligentCacheManager",
    "CacheStrategy",
    "CompressionType",
    "SerializationFormat",
    "CacheEntry",
    "CacheStats",
    "AccessPattern",
    "get_intelligent_cache_manager",
    "close_intelligent_cache_manager",
    
    # Decorators
    "CacheConfig",
    "CacheKeyGenerator",
    "CacheDecorator",
    "AsyncCacheDecorator",
    "SyncCacheDecorator",
    "CacheInvalidator",
    "set_cache_manager",
    "get_cache_manager",
    "get_cache_invalidator",
    "cached",
    "cache_result",
    "cache_expensive",
    "cache_model_prediction",
    "cache_database_query",
    "cache_context",
    "invalidate_cache",
    "invalidate_cache_pattern",
    "warm_cache",
    "get_cache_stats",
    
    # Integration
    "CacheConfiguration",
    "CacheHealthMonitor",
    "CacheIntegrationManager",
    "get_cache_integration_manager",
    "close_cache_integration_manager",
    "cache_system_context",
    "get_cache_health",
    "get_cache_statistics",
    "perform_cache_maintenance",
    "warm_cache_with_critical_data",
]