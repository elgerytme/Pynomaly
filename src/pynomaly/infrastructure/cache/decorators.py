"""Advanced caching decorators for function and method caching."""

from __future__ import annotations

import asyncio
import functools
import hashlib
import inspect
import json
import time
from typing import Any, Callable, Dict, Optional, TypeVar, Union, List, Tuple
from dataclasses import dataclass
import logging
from contextlib import asynccontextmanager

from pynomaly.shared.error_handling import (
    PynamolyError,
    InfrastructureError,
    ErrorCodes,
    create_infrastructure_error,
)
from .intelligent_cache import IntelligentCacheManager, CacheStrategy
from .redis_cache import RedisCache

logger = logging.getLogger(__name__)

F = TypeVar('F', bound=Callable[..., Any])


@dataclass
class CacheConfig:
    """Configuration for cache decorators."""
    ttl: Optional[int] = None
    key_prefix: str = ""
    strategy: CacheStrategy = CacheStrategy.WRITE_THROUGH
    ignore_args: List[str] = None
    ignore_kwargs: List[str] = None
    serialize_complex_args: bool = True
    cache_none_values: bool = False
    error_fallback: bool = True
    conditional_cache: Optional[Callable] = None


class CacheKeyGenerator:
    """Generates cache keys for functions and methods."""
    
    @staticmethod
    def generate_key(
        func: Callable,
        args: Tuple,
        kwargs: Dict[str, Any],
        prefix: str = "",
        ignore_args: List[str] = None,
        ignore_kwargs: List[str] = None,
        serialize_complex_args: bool = True,
    ) -> str:
        """Generate cache key for function call.
        
        Args:
            func: Function being cached
            args: Positional arguments
            kwargs: Keyword arguments
            prefix: Key prefix
            ignore_args: Arguments to ignore in key generation
            ignore_kwargs: Keyword arguments to ignore
            serialize_complex_args: Whether to serialize complex arguments
            
        Returns:
            Cache key string
        """
        # Get function signature
        sig = inspect.signature(func)
        bound_args = sig.bind(*args, **kwargs)
        bound_args.apply_defaults()
        
        # Build key components
        key_parts = [
            prefix if prefix else "",
            f"{func.__module__}.{func.__qualname__}",
        ]
        
        # Process arguments
        ignore_args = ignore_args or []
        ignore_kwargs = ignore_kwargs or []
        
        # Add positional arguments
        for i, (param_name, value) in enumerate(bound_args.arguments.items()):
            if param_name in ignore_args or param_name in ignore_kwargs:
                continue
                
            serialized_value = CacheKeyGenerator._serialize_value(
                value, serialize_complex_args
            )
            key_parts.append(f"{param_name}={serialized_value}")
        
        # Create final key
        key = ":".join(key_parts)
        
        # Hash if too long
        if len(key) > 200:
            key_hash = hashlib.md5(key.encode()).hexdigest()
            return f"{prefix}:hashed:{key_hash}"
        
        return key
    
    @staticmethod
    def _serialize_value(value: Any, serialize_complex: bool) -> str:
        """Serialize value for key generation."""
        if value is None:
            return "None"
        elif isinstance(value, (str, int, float, bool)):
            return str(value)
        elif isinstance(value, (list, tuple)):
            if serialize_complex:
                return f"[{','.join(CacheKeyGenerator._serialize_value(v, serialize_complex) for v in value)}]"
            else:
                return f"list_len_{len(value)}"
        elif isinstance(value, dict):
            if serialize_complex:
                sorted_items = sorted(value.items())
                return f"{{{','.join(f'{k}:{CacheKeyGenerator._serialize_value(v, serialize_complex)}' for k, v in sorted_items)}}}"
            else:
                return f"dict_len_{len(value)}"
        elif hasattr(value, '__dict__'):
            # Object with attributes
            if serialize_complex:
                return f"obj_{value.__class__.__name__}_{id(value)}"
            else:
                return f"obj_{value.__class__.__name__}"
        else:
            # Other types
            return f"type_{type(value).__name__}"


class CacheDecorator:
    """Base class for cache decorators."""
    
    def __init__(
        self,
        cache_manager: IntelligentCacheManager,
        config: CacheConfig,
    ):
        """Initialize cache decorator.
        
        Args:
            cache_manager: Cache manager instance
            config: Cache configuration
        """
        self.cache_manager = cache_manager
        self.config = config
        self.key_generator = CacheKeyGenerator()
    
    def should_cache(self, result: Any) -> bool:
        """Check if result should be cached."""
        # Don't cache None values unless explicitly configured
        if result is None and not self.config.cache_none_values:
            return False
        
        # Check conditional cache function
        if self.config.conditional_cache:
            return self.config.conditional_cache(result)
        
        return True
    
    def generate_cache_key(
        self,
        func: Callable,
        args: Tuple,
        kwargs: Dict[str, Any],
    ) -> str:
        """Generate cache key for function call."""
        return self.key_generator.generate_key(
            func=func,
            args=args,
            kwargs=kwargs,
            prefix=self.config.key_prefix,
            ignore_args=self.config.ignore_args,
            ignore_kwargs=self.config.ignore_kwargs,
            serialize_complex_args=self.config.serialize_complex_args,
        )


class AsyncCacheDecorator(CacheDecorator):
    """Async function cache decorator."""
    
    def __call__(self, func: Callable) -> Callable:
        """Decorate async function with caching."""
        @functools.wraps(func)
        async def wrapper(*args, **kwargs):
            # Generate cache key
            cache_key = self.generate_cache_key(func, args, kwargs)
            
            # Try to get from cache
            try:
                cached_result = await self.cache_manager.get(cache_key)
                if cached_result is not None:
                    logger.debug(f"Cache hit for key: {cache_key}")
                    return cached_result
            except Exception as e:
                if not self.config.error_fallback:
                    raise
                logger.warning(f"Cache get failed for key {cache_key}: {e}")
            
            # Execute function
            try:
                result = await func(*args, **kwargs)
                
                # Cache result if appropriate
                if self.should_cache(result):
                    try:
                        await self.cache_manager.set(
                            cache_key,
                            result,
                            self.config.ttl,
                            self.config.strategy,
                        )
                    except Exception as e:
                        if not self.config.error_fallback:
                            raise
                        logger.warning(f"Cache set failed for key {cache_key}: {e}")
                
                return result
                
            except Exception as e:
                logger.error(f"Function execution failed: {e}")
                raise
        
        return wrapper


class SyncCacheDecorator(CacheDecorator):
    """Sync function cache decorator."""
    
    def __call__(self, func: Callable) -> Callable:
        """Decorate sync function with caching."""
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            # Generate cache key
            cache_key = self.generate_cache_key(func, args, kwargs)
            
            # Try to get from cache (sync version)
            try:
                cached_result = self.cache_manager.redis_cache.get(cache_key)
                if cached_result is not None:
                    logger.debug(f"Cache hit for key: {cache_key}")
                    return cached_result
            except Exception as e:
                if not self.config.error_fallback:
                    raise
                logger.warning(f"Cache get failed for key {cache_key}: {e}")
            
            # Execute function
            try:
                result = func(*args, **kwargs)
                
                # Cache result if appropriate
                if self.should_cache(result):
                    try:
                        self.cache_manager.redis_cache.set(
                            cache_key,
                            result,
                            self.config.ttl,
                        )
                    except Exception as e:
                        if not self.config.error_fallback:
                            raise
                        logger.warning(f"Cache set failed for key {cache_key}: {e}")
                
                return result
                
            except Exception as e:
                logger.error(f"Function execution failed: {e}")
                raise
        
        return wrapper


class CacheInvalidator:
    """Handles cache invalidation for cached functions."""
    
    def __init__(self, cache_manager: IntelligentCacheManager):
        """Initialize cache invalidator.
        
        Args:
            cache_manager: Cache manager instance
        """
        self.cache_manager = cache_manager
    
    async def invalidate_function(
        self,
        func: Callable,
        args: Tuple = (),
        kwargs: Dict[str, Any] = None,
        prefix: str = "",
    ) -> bool:
        """Invalidate cache for specific function call.
        
        Args:
            func: Function to invalidate
            args: Function arguments
            kwargs: Function keyword arguments
            prefix: Cache key prefix
            
        Returns:
            Success status
        """
        kwargs = kwargs or {}
        
        # Generate cache key
        key_generator = CacheKeyGenerator()
        cache_key = key_generator.generate_key(
            func=func,
            args=args,
            kwargs=kwargs,
            prefix=prefix,
        )
        
        return await self.cache_manager.delete(cache_key)
    
    async def invalidate_pattern(self, pattern: str) -> int:
        """Invalidate all cache entries matching pattern.
        
        Args:
            pattern: Cache key pattern
            
        Returns:
            Number of keys invalidated
        """
        return await self.cache_manager.delete_pattern(pattern)
    
    async def invalidate_prefix(self, prefix: str) -> int:
        """Invalidate all cache entries with prefix.
        
        Args:
            prefix: Cache key prefix
            
        Returns:
            Number of keys invalidated
        """
        pattern = f"{prefix}*"
        return await self.cache_manager.delete_pattern(pattern)


# Global cache manager and invalidator
_cache_manager: Optional[IntelligentCacheManager] = None
_cache_invalidator: Optional[CacheInvalidator] = None


def set_cache_manager(manager: IntelligentCacheManager) -> None:
    """Set global cache manager."""
    global _cache_manager, _cache_invalidator
    _cache_manager = manager
    _cache_invalidator = CacheInvalidator(manager)


def get_cache_manager() -> Optional[IntelligentCacheManager]:
    """Get global cache manager."""
    return _cache_manager


def get_cache_invalidator() -> Optional[CacheInvalidator]:
    """Get global cache invalidator."""
    return _cache_invalidator


# Decorator factory functions
def cached(
    ttl: Optional[int] = None,
    key_prefix: str = "",
    strategy: CacheStrategy = CacheStrategy.WRITE_THROUGH,
    ignore_args: List[str] = None,
    ignore_kwargs: List[str] = None,
    serialize_complex_args: bool = True,
    cache_none_values: bool = False,
    error_fallback: bool = True,
    conditional_cache: Optional[Callable] = None,
) -> Callable[[F], F]:
    """Decorator for caching function results.
    
    Args:
        ttl: Cache TTL in seconds
        key_prefix: Cache key prefix
        strategy: Cache strategy
        ignore_args: Arguments to ignore in key generation
        ignore_kwargs: Keyword arguments to ignore
        serialize_complex_args: Whether to serialize complex arguments
        cache_none_values: Whether to cache None values
        error_fallback: Whether to continue on cache errors
        conditional_cache: Function to determine if result should be cached
        
    Returns:
        Decorator function
    """
    def decorator(func: F) -> F:
        if _cache_manager is None:
            logger.warning("No cache manager configured, caching disabled")
            return func
        
        config = CacheConfig(
            ttl=ttl,
            key_prefix=key_prefix,
            strategy=strategy,
            ignore_args=ignore_args,
            ignore_kwargs=ignore_kwargs,
            serialize_complex_args=serialize_complex_args,
            cache_none_values=cache_none_values,
            error_fallback=error_fallback,
            conditional_cache=conditional_cache,
        )
        
        # Choose decorator based on function type
        if asyncio.iscoroutinefunction(func):
            return AsyncCacheDecorator(_cache_manager, config)(func)
        else:
            return SyncCacheDecorator(_cache_manager, config)(func)
    
    return decorator


def cache_result(
    ttl: int = 3600,
    key_prefix: str = "",
) -> Callable[[F], F]:
    """Simple decorator for caching function results.
    
    Args:
        ttl: Cache TTL in seconds (default 1 hour)
        key_prefix: Cache key prefix
        
    Returns:
        Decorator function
    """
    return cached(
        ttl=ttl,
        key_prefix=key_prefix,
        error_fallback=True,
    )


def cache_expensive(
    ttl: int = 7200,
    key_prefix: str = "expensive",
) -> Callable[[F], F]:
    """Decorator for caching expensive computations.
    
    Args:
        ttl: Cache TTL in seconds (default 2 hours)
        key_prefix: Cache key prefix
        
    Returns:
        Decorator function
    """
    return cached(
        ttl=ttl,
        key_prefix=key_prefix,
        strategy=CacheStrategy.WRITE_THROUGH,
        serialize_complex_args=False,  # For performance
        error_fallback=True,
    )


def cache_model_prediction(
    ttl: int = 1800,
    key_prefix: str = "prediction",
) -> Callable[[F], F]:
    """Decorator for caching model predictions.
    
    Args:
        ttl: Cache TTL in seconds (default 30 minutes)
        key_prefix: Cache key prefix
        
    Returns:
        Decorator function
    """
    return cached(
        ttl=ttl,
        key_prefix=key_prefix,
        strategy=CacheStrategy.WRITE_BEHIND,  # Fast response
        cache_none_values=False,
        error_fallback=True,
    )


def cache_database_query(
    ttl: int = 900,
    key_prefix: str = "db_query",
) -> Callable[[F], F]:
    """Decorator for caching database queries.
    
    Args:
        ttl: Cache TTL in seconds (default 15 minutes)
        key_prefix: Cache key prefix
        
    Returns:
        Decorator function
    """
    return cached(
        ttl=ttl,
        key_prefix=key_prefix,
        strategy=CacheStrategy.CACHE_ASIDE,
        serialize_complex_args=True,
        error_fallback=True,
    )


# Context manager for cache management
@asynccontextmanager
async def cache_context(
    cache_manager: IntelligentCacheManager,
    auto_invalidate: bool = True,
    invalidation_patterns: List[str] = None,
):
    """Context manager for cache operations.
    
    Args:
        cache_manager: Cache manager instance
        auto_invalidate: Whether to auto-invalidate on exit
        invalidation_patterns: Patterns to invalidate on exit
    """
    # Set global cache manager
    set_cache_manager(cache_manager)
    
    try:
        yield cache_manager
    finally:
        # Auto-invalidate if requested
        if auto_invalidate and invalidation_patterns:
            invalidator = get_cache_invalidator()
            if invalidator:
                for pattern in invalidation_patterns:
                    await invalidator.invalidate_pattern(pattern)


# Utility functions
async def invalidate_cache(
    func: Callable,
    args: Tuple = (),
    kwargs: Dict[str, Any] = None,
    prefix: str = "",
) -> bool:
    """Invalidate cache for specific function call.
    
    Args:
        func: Function to invalidate
        args: Function arguments
        kwargs: Function keyword arguments
        prefix: Cache key prefix
        
    Returns:
        Success status
    """
    invalidator = get_cache_invalidator()
    if invalidator:
        return await invalidator.invalidate_function(func, args, kwargs, prefix)
    return False


async def invalidate_cache_pattern(pattern: str) -> int:
    """Invalidate all cache entries matching pattern.
    
    Args:
        pattern: Cache key pattern
        
    Returns:
        Number of keys invalidated
    """
    invalidator = get_cache_invalidator()
    if invalidator:
        return await invalidator.invalidate_pattern(pattern)
    return 0


async def warm_cache(keys_and_loaders: List[Tuple[str, Callable]]) -> int:
    """Warm cache with preloaded data.
    
    Args:
        keys_and_loaders: List of (key, loader_function) tuples
        
    Returns:
        Number of keys warmed
    """
    manager = get_cache_manager()
    if manager:
        return await manager.warm_cache(keys_and_loaders)
    return 0


async def get_cache_stats() -> Dict[str, Any]:
    """Get cache statistics.
    
    Returns:
        Cache statistics
    """
    manager = get_cache_manager()
    if manager:
        return await manager.get_stats()
    return {}