"""Caching infrastructure for distributed caching and performance optimization.

This module provides Redis-based caching, cache-aside patterns, and cache
invalidation strategies for improved application performance.

Example usage:
    from infrastructure.caching import CacheManager, cache_key
    
    cache = CacheManager()
    await cache.set("user:123", user_data, ttl=3600)
    
    @cache_key("user:{user_id}")
    async def get_user(user_id: str):
        return await user_repository.find_by_id(user_id)
"""

from .cache_manager import CacheManager
from .redis_cache import RedisCache
from .decorators import cache_key, cache_result
from .invalidation import CacheInvalidator

__all__ = [
    "CacheManager",
    "RedisCache",
    "cache_key",
    "cache_result", 
    "CacheInvalidator"
]