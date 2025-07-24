"""Advanced multi-layer caching strategies for anomaly detection platform."""

import asyncio
import hashlib
import json
import pickle
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, asdict
from datetime import datetime, timedelta
from enum import Enum
from typing import Any, Dict, List, Optional, Union, Callable, TypeVar, Generic
import logging
import zlib
from pathlib import Path

import numpy as np
import redis.asyncio as redis
from sklearn.base import BaseEstimator

logger = logging.getLogger(__name__)

T = TypeVar('T')


class CacheStrategy(str, Enum):
    """Cache eviction strategies."""
    LRU = "lru"  # Least Recently Used
    LFU = "lfu"  # Least Frequently Used
    TTL = "ttl"  # Time To Live
    FIFO = "fifo"  # First In First Out
    ADAPTIVE = "adaptive"  # Adaptive based on access patterns


class CacheLayer(str, Enum):
    """Cache layer types."""
    MEMORY = "memory"  # In-memory cache
    REDIS = "redis"  # Redis distributed cache
    DISK = "disk"  # Disk-based cache
    HYBRID = "hybrid"  # Multi-layer hybrid cache


@dataclass
class CacheEntry:
    """Cache entry with metadata."""
    key: str
    value: Any
    created_at: datetime
    last_accessed: datetime
    access_count: int = 0
    size_bytes: int = 0
    ttl_seconds: Optional[int] = None
    metadata: Dict[str, Any] = None
    
    def __post_init__(self):
        if self.metadata is None:
            self.metadata = {}
    
    @property
    def is_expired(self) -> bool:
        """Check if entry is expired."""
        if self.ttl_seconds is None:
            return False
        return (datetime.utcnow() - self.created_at).total_seconds() > self.ttl_seconds
    
    @property
    def age_seconds(self) -> float:
        """Get age of cache entry in seconds."""
        return (datetime.utcnow() - self.created_at).total_seconds()
    
    def touch(self):
        """Update access metadata."""
        self.last_accessed = datetime.utcnow()
        self.access_count += 1


class ICacheStore(ABC, Generic[T]):
    """Interface for cache storage implementations."""
    
    @abstractmethod
    async def get(self, key: str) -> Optional[T]:
        """Get value from cache."""
        pass
    
    @abstractmethod
    async def set(self, key: str, value: T, ttl_seconds: Optional[int] = None) -> bool:
        """Set value in cache."""
        pass
    
    @abstractmethod
    async def delete(self, key: str) -> bool:
        """Delete value from cache."""
        pass
    
    @abstractmethod
    async def exists(self, key: str) -> bool:
        """Check if key exists in cache."""
        pass
    
    @abstractmethod
    async def clear(self) -> bool:
        """Clear all cache entries."""
        pass
    
    @abstractmethod
    async def size(self) -> int:
        """Get number of entries in cache."""
        pass


class MemoryCacheStore(ICacheStore[T]):
    """In-memory cache store with configurable eviction strategies."""
    
    def __init__(self, max_size: int = 1000, strategy: CacheStrategy = CacheStrategy.LRU):
        self.max_size = max_size
        self.strategy = strategy
        self.cache: Dict[str, CacheEntry] = {}
        self._lock = asyncio.Lock()
    
    async def get(self, key: str) -> Optional[T]:
        """Get value from memory cache."""
        async with self._lock:
            if key not in self.cache:
                return None
            
            entry = self.cache[key]
            
            # Check expiration
            if entry.is_expired:
                del self.cache[key]
                return None
            
            # Update access metadata
            entry.touch()
            return entry.value
    
    async def set(self, key: str, value: T, ttl_seconds: Optional[int] = None) -> bool:
        """Set value in memory cache."""
        async with self._lock:
            try:
                # Calculate size
                size_bytes = len(pickle.dumps(value))
                
                # Create entry
                entry = CacheEntry(
                    key=key,
                    value=value,
                    created_at=datetime.utcnow(),
                    last_accessed=datetime.utcnow(),
                    size_bytes=size_bytes,
                    ttl_seconds=ttl_seconds
                )
                
                # Evict if necessary
                if len(self.cache) >= self.max_size and key not in self.cache:
                    await self._evict()
                
                self.cache[key] = entry
                return True
                
            except Exception as e:
                logger.error(f"Failed to set cache entry {key}: {e}")
                return False
    
    async def delete(self, key: str) -> bool:
        """Delete value from memory cache."""
        async with self._lock:
            if key in self.cache:
                del self.cache[key]
                return True
            return False
    
    async def exists(self, key: str) -> bool:
        """Check if key exists in memory cache."""
        async with self._lock:
            return key in self.cache and not self.cache[key].is_expired
    
    async def clear(self) -> bool:
        """Clear all entries from memory cache."""
        async with self._lock:
            self.cache.clear()
            return True
    
    async def size(self) -> int:
        """Get number of entries in memory cache."""
        async with self._lock:
            return len(self.cache)
    
    async def _evict(self):
        """Evict entries based on strategy."""
        if not self.cache:
            return
        
        if self.strategy == CacheStrategy.LRU:
            # Remove least recently used
            lru_key = min(self.cache.keys(), key=lambda k: self.cache[k].last_accessed)
            del self.cache[lru_key]
        
        elif self.strategy == CacheStrategy.LFU:
            # Remove least frequently used
            lfu_key = min(self.cache.keys(), key=lambda k: self.cache[k].access_count)
            del self.cache[lfu_key]
        
        elif self.strategy == CacheStrategy.FIFO:
            # Remove first in (oldest)
            fifo_key = min(self.cache.keys(), key=lambda k: self.cache[k].created_at)
            del self.cache[fifo_key]
        
        elif self.strategy == CacheStrategy.TTL:
            # Remove expired entries first, then oldest
            expired_keys = [k for k, v in self.cache.items() if v.is_expired]
            if expired_keys:
                del self.cache[expired_keys[0]]
            else:
                oldest_key = min(self.cache.keys(), key=lambda k: self.cache[k].created_at)
                del self.cache[oldest_key]


class RedisCacheStore(ICacheStore[T]):
    """Redis-based distributed cache store."""
    
    def __init__(self, redis_url: str = "redis://localhost:6379", prefix: str = "anomaly_detection"):
        self.redis_url = redis_url
        self.prefix = prefix
        self.redis_client: Optional[redis.Redis] = None
    
    async def _ensure_connection(self):
        """Ensure Redis connection is established."""
        if self.redis_client is None:
            self.redis_client = redis.from_url(self.redis_url)
    
    def _make_key(self, key: str) -> str:
        """Create prefixed key."""
        return f"{self.prefix}:{key}"
    
    async def get(self, key: str) -> Optional[T]:
        """Get value from Redis cache."""
        await self._ensure_connection()
        try:
            prefixed_key = self._make_key(key)
            data = await self.redis_client.get(prefixed_key)
            
            if data is None:
                return None
            
            # Decompress and deserialize
            decompressed = zlib.decompress(data)
            return pickle.loads(decompressed)
            
        except Exception as e:
            logger.error(f"Failed to get from Redis cache {key}: {e}")
            return None
    
    async def set(self, key: str, value: T, ttl_seconds: Optional[int] = None) -> bool:
        """Set value in Redis cache."""
        await self._ensure_connection()
        try:
            prefixed_key = self._make_key(key)
            
            # Serialize and compress
            serialized = pickle.dumps(value)
            compressed = zlib.compress(serialized)
            
            if ttl_seconds:
                await self.redis_client.setex(prefixed_key, ttl_seconds, compressed)
            else:
                await self.redis_client.set(prefixed_key, compressed)
                
            return True
            
        except Exception as e:
            logger.error(f"Failed to set Redis cache {key}: {e}")
            return False
    
    async def delete(self, key: str) -> bool:
        """Delete value from Redis cache."""
        await self._ensure_connection()
        try:
            prefixed_key = self._make_key(key)
            deleted = await self.redis_client.delete(prefixed_key)
            return deleted > 0
            
        except Exception as e:
            logger.error(f"Failed to delete from Redis cache {key}: {e}")
            return False
    
    async def exists(self, key: str) -> bool:
        """Check if key exists in Redis cache."""
        await self._ensure_connection()
        try:
            prefixed_key = self._make_key(key)
            return await self.redis_client.exists(prefixed_key) > 0
            
        except Exception as e:
            logger.error(f"Failed to check Redis cache existence {key}: {e}")
            return False
    
    async def clear(self) -> bool:
        """Clear all entries from Redis cache."""
        await self._ensure_connection()
        try:
            pattern = f"{self.prefix}:*"
            keys = []
            async for key in self.redis_client.scan_iter(match=pattern):
                keys.append(key)
            
            if keys:
                await self.redis_client.delete(*keys)
            
            return True
            
        except Exception as e:
            logger.error(f"Failed to clear Redis cache: {e}")
            return False
    
    async def size(self) -> int:
        """Get number of entries in Redis cache."""
        await self._ensure_connection()
        try:
            pattern = f"{self.prefix}:*"
            count = 0
            async for _ in self.redis_client.scan_iter(match=pattern):
                count += 1
            return count
            
        except Exception as e:
            logger.error(f"Failed to get Redis cache size: {e}")
            return 0


class DiskCacheStore(ICacheStore[T]):
    """Disk-based cache store for large objects."""
    
    def __init__(self, cache_dir: str = "/tmp/anomaly_detection_cache", max_size_mb: int = 1000):
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.max_size_mb = max_size_mb
        self._lock = asyncio.Lock()
    
    def _get_file_path(self, key: str) -> Path:
        """Get file path for cache key."""
        key_hash = hashlib.sha256(key.encode()).hexdigest()
        return self.cache_dir / f"{key_hash}.cache"
    
    def _get_metadata_path(self, key: str) -> Path:
        """Get metadata file path for cache key."""
        key_hash = hashlib.sha256(key.encode()).hexdigest()
        return self.cache_dir / f"{key_hash}.meta"
    
    async def get(self, key: str) -> Optional[T]:
        """Get value from disk cache."""
        async with self._lock:
            file_path = self._get_file_path(key)
            meta_path = self._get_metadata_path(key)
            
            if not file_path.exists() or not meta_path.exists():
                return None
            
            try:
                # Check metadata for expiration
                with open(meta_path, 'r') as f:
                    metadata = json.load(f)
                
                if metadata.get('ttl_seconds'):
                    created_at = datetime.fromisoformat(metadata['created_at'])
                    if (datetime.utcnow() - created_at).total_seconds() > metadata['ttl_seconds']:
                        # Expired, remove files
                        file_path.unlink(missing_ok=True)
                        meta_path.unlink(missing_ok=True)
                        return None
                
                # Load and decompress data
                with open(file_path, 'rb') as f:
                    compressed_data = f.read()
                
                decompressed = zlib.decompress(compressed_data)
                value = pickle.loads(decompressed)
                
                # Update access metadata
                metadata['last_accessed'] = datetime.utcnow().isoformat()
                metadata['access_count'] = metadata.get('access_count', 0) + 1
                
                with open(meta_path, 'w') as f:
                    json.dump(metadata, f)
                
                return value
                
            except Exception as e:
                logger.error(f"Failed to get from disk cache {key}: {e}")
                return None
    
    async def set(self, key: str, value: T, ttl_seconds: Optional[int] = None) -> bool:
        """Set value in disk cache."""
        async with self._lock:
            try:
                file_path = self._get_file_path(key)
                meta_path = self._get_metadata_path(key)
                
                # Serialize and compress
                serialized = pickle.dumps(value)
                compressed = zlib.compress(serialized)
                
                # Check disk space before writing
                await self._cleanup_if_needed(len(compressed))
                
                # Write data file
                with open(file_path, 'wb') as f:
                    f.write(compressed)
                
                # Write metadata file
                metadata = {
                    'key': key,
                    'created_at': datetime.utcnow().isoformat(),
                    'last_accessed': datetime.utcnow().isoformat(),
                    'access_count': 1,
                    'size_bytes': len(compressed),
                    'ttl_seconds': ttl_seconds
                }
                
                with open(meta_path, 'w') as f:
                    json.dump(metadata, f)
                
                return True
                
            except Exception as e:
                logger.error(f"Failed to set disk cache {key}: {e}")
                return False
    
    async def delete(self, key: str) -> bool:
        """Delete value from disk cache."""
        async with self._lock:
            file_path = self._get_file_path(key)
            meta_path = self._get_metadata_path(key)
            
            deleted = False
            if file_path.exists():
                file_path.unlink()
                deleted = True
            
            if meta_path.exists():
                meta_path.unlink()
                deleted = True
            
            return deleted
    
    async def exists(self, key: str) -> bool:
        """Check if key exists in disk cache."""
        async with self._lock:
            file_path = self._get_file_path(key)
            meta_path = self._get_metadata_path(key)
            return file_path.exists() and meta_path.exists()
    
    async def clear(self) -> bool:
        """Clear all entries from disk cache."""
        async with self._lock:
            try:
                for file_path in self.cache_dir.glob("*.cache"):
                    file_path.unlink()
                for file_path in self.cache_dir.glob("*.meta"):
                    file_path.unlink()
                return True
            except Exception as e:
                logger.error(f"Failed to clear disk cache: {e}")
                return False
    
    async def size(self) -> int:
        """Get number of entries in disk cache."""
        async with self._lock:
            return len(list(self.cache_dir.glob("*.cache")))
    
    async def _cleanup_if_needed(self, new_size_bytes: int):
        """Cleanup old entries if cache size limit would be exceeded."""
        # Get current cache size
        total_size = sum(f.stat().st_size for f in self.cache_dir.glob("*.cache"))
        max_size_bytes = self.max_size_mb * 1024 * 1024
        
        if total_size + new_size_bytes > max_size_bytes:
            # Get all cache files with metadata
            cache_files = []
            for cache_file in self.cache_dir.glob("*.cache"):
                meta_file = cache_file.with_suffix(".meta")
                if meta_file.exists():
                    try:
                        with open(meta_file, 'r') as f:
                            metadata = json.load(f)
                        cache_files.append((cache_file, meta_file, metadata))
                    except:
                        continue
            
            # Sort by last accessed time (LRU strategy)
            cache_files.sort(key=lambda x: x[2].get('last_accessed', ''))
            
            # Remove oldest files until under limit
            for cache_file, meta_file, metadata in cache_files:
                if total_size <= max_size_bytes:
                    break
                
                file_size = cache_file.stat().st_size
                cache_file.unlink(missing_ok=True)
                meta_file.unlink(missing_ok=True)
                total_size -= file_size


class HybridCacheStore(ICacheStore[T]):
    """Multi-layer hybrid cache combining memory, Redis, and disk storage."""
    
    def __init__(
        self,
        memory_cache: Optional[MemoryCacheStore] = None,
        redis_cache: Optional[RedisCacheStore] = None,
        disk_cache: Optional[DiskCacheStore] = None
    ):
        self.memory_cache = memory_cache or MemoryCacheStore(max_size=500)
        self.redis_cache = redis_cache
        self.disk_cache = disk_cache
        
        # Cache layers in order of preference (fastest to slowest)
        self.layers = []
        if self.memory_cache:
            self.layers.append(("memory", self.memory_cache))
        if self.redis_cache:
            self.layers.append(("redis", self.redis_cache))
        if self.disk_cache:
            self.layers.append(("disk", self.disk_cache))
    
    async def get(self, key: str) -> Optional[T]:
        """Get value from hybrid cache, checking layers in order."""
        for layer_name, cache_store in self.layers:
            try:
                value = await cache_store.get(key)
                if value is not None:
                    # Promote to faster layers
                    await self._promote_to_faster_layers(key, value, layer_name)
                    return value
            except Exception as e:
                logger.warning(f"Failed to get from {layer_name} cache: {e}")
                continue
        
        return None
    
    async def set(self, key: str, value: T, ttl_seconds: Optional[int] = None) -> bool:
        """Set value in hybrid cache across all layers."""
        success = False
        
        for layer_name, cache_store in self.layers:
            try:
                if await cache_store.set(key, value, ttl_seconds):
                    success = True
            except Exception as e:
                logger.warning(f"Failed to set in {layer_name} cache: {e}")
        
        return success
    
    async def delete(self, key: str) -> bool:
        """Delete value from all cache layers."""
        success = False
        
        for layer_name, cache_store in self.layers:
            try:
                if await cache_store.delete(key):
                    success = True
            except Exception as e:
                logger.warning(f"Failed to delete from {layer_name} cache: {e}")
        
        return success
    
    async def exists(self, key: str) -> bool:
        """Check if key exists in any cache layer."""
        for layer_name, cache_store in self.layers:
            try:
                if await cache_store.exists(key):
                    return True
            except Exception as e:
                logger.warning(f"Failed to check existence in {layer_name} cache: {e}")
        
        return False
    
    async def clear(self) -> bool:
        """Clear all cache layers."""
        success = False
        
        for layer_name, cache_store in self.layers:
            try:
                if await cache_store.clear():
                    success = True
            except Exception as e:
                logger.warning(f"Failed to clear {layer_name} cache: {e}")
        
        return success
    
    async def size(self) -> int:
        """Get total number of unique entries across all layers."""
        all_keys = set()
        
        for layer_name, cache_store in self.layers:
            try:
                # This is a simplified implementation
                # In practice, you'd need to enumerate keys per cache store
                size = await cache_store.size()
                logger.info(f"{layer_name} cache size: {size}")
            except Exception as e:
                logger.warning(f"Failed to get size from {layer_name} cache: {e}")
        
        # Return size of memory cache as primary indicator
        if self.memory_cache:
            return await self.memory_cache.size()
        
        return 0
    
    async def _promote_to_faster_layers(self, key: str, value: T, found_layer: str):
        """Promote cache entry to faster layers."""
        promote = False
        
        for layer_name, cache_store in self.layers:
            if layer_name == found_layer:
                break
            
            # Only promote to memory cache to avoid overwhelming Redis
            if layer_name == "memory":
                try:
                    await cache_store.set(key, value)
                except Exception as e:
                    logger.warning(f"Failed to promote to {layer_name} cache: {e}")


class AdvancedCacheManager:
    """Advanced cache manager with intelligent caching strategies."""
    
    def __init__(
        self,
        cache_store: Optional[ICacheStore] = None,
        enable_stats: bool = True
    ):
        self.cache_store = cache_store or HybridCacheStore()
        self.enable_stats = enable_stats
        
        # Cache statistics
        self.stats = {
            'hits': 0,
            'misses': 0,
            'sets': 0,
            'deletes': 0,
            'errors': 0
        } if enable_stats else None
        
        # Cache key generators for different object types
        self.key_generators = {
            'data_array': self._generate_data_key,
            'model': self._generate_model_key,
            'result': self._generate_result_key,
            'generic': self._generate_generic_key
        }
    
    async def get_or_compute(
        self,
        key: str,
        compute_func: Callable[[], T],
        ttl_seconds: Optional[int] = None,
        cache_type: str = 'generic'
    ) -> T:
        """Get value from cache or compute if not found."""
        cache_key = self._generate_cache_key(key, cache_type)
        
        # Try to get from cache
        try:
            cached_value = await self.cache_store.get(cache_key)
            if cached_value is not None:
                if self.enable_stats:
                    self.stats['hits'] += 1
                return cached_value
        except Exception as e:
            logger.error(f"Cache get error: {e}")
            if self.enable_stats:
                self.stats['errors'] += 1
        
        # Cache miss - compute value
        if self.enable_stats:
            self.stats['misses'] += 1
        
        try:
            # Handle both sync and async compute functions
            if asyncio.iscoroutinefunction(compute_func):
                computed_value = await compute_func()
            else:
                computed_value = compute_func()
            
            # Store in cache
            await self.set(cache_key, computed_value, ttl_seconds)
            
            return computed_value
            
        except Exception as e:
            logger.error(f"Compute function error: {e}")
            raise
    
    async def set(self, key: str, value: T, ttl_seconds: Optional[int] = None) -> bool:
        """Set value in cache."""
        try:
            success = await self.cache_store.set(key, value, ttl_seconds)
            if self.enable_stats:
                self.stats['sets'] += 1
            return success
        except Exception as e:
            logger.error(f"Cache set error: {e}")
            if self.enable_stats:
                self.stats['errors'] += 1
            return False
    
    async def get(self, key: str) -> Optional[T]:
        """Get value from cache."""
        try:
            value = await self.cache_store.get(key)
            if self.enable_stats:
                if value is not None:
                    self.stats['hits'] += 1
                else:
                    self.stats['misses'] += 1
            return value
        except Exception as e:
            logger.error(f"Cache get error: {e}")
            if self.enable_stats:
                self.stats['errors'] += 1
            return None
    
    async def delete(self, key: str) -> bool:
        """Delete value from cache."""
        try:
            success = await self.cache_store.delete(key)
            if self.enable_stats:
                self.stats['deletes'] += 1
            return success
        except Exception as e:
            logger.error(f"Cache delete error: {e}")
            if self.enable_stats:
                self.stats['errors'] += 1
            return False
    
    async def clear(self) -> bool:
        """Clear all cache entries."""
        return await self.cache_store.clear()
    
    def get_stats(self) -> Optional[Dict[str, Any]]:
        """Get cache statistics."""
        if not self.enable_stats or not self.stats:
            return None
        
        total_requests = self.stats['hits'] + self.stats['misses']
        hit_rate = (self.stats['hits'] / total_requests * 100) if total_requests > 0 else 0
        
        return {
            **self.stats,
            'hit_rate_percent': round(hit_rate, 2),
            'total_requests': total_requests
        }
    
    def _generate_cache_key(self, key: str, cache_type: str) -> str:
        """Generate cache key using appropriate generator."""
        generator = self.key_generators.get(cache_type, self.key_generators['generic'])
        return generator(key)
    
    def _generate_data_key(self, data_identifier: str) -> str:
        """Generate cache key for data arrays."""
        return f"data:{data_identifier}"
    
    def _generate_model_key(self, model_identifier: str) -> str:
        """Generate cache key for ML models."""
        return f"model:{model_identifier}"
    
    def _generate_result_key(self, result_identifier: str) -> str:
        """Generate cache key for computation results."""
        return f"result:{result_identifier}"
    
    def _generate_generic_key(self, key: str) -> str:
        """Generate generic cache key."""
        return f"generic:{key}"


# Specialized cache decorators for anomaly detection
def cache_detection_result(ttl_seconds: int = 3600):
    """Decorator to cache anomaly detection results."""
    def decorator(func):
        async def wrapper(*args, **kwargs):
            # Generate cache key from function arguments
            cache_key_data = {
                'func_name': func.__name__,
                'args': str(args),
                'kwargs': sorted(kwargs.items())
            }
            cache_key = hashlib.sha256(str(cache_key_data).encode()).hexdigest()
            
            # Get cache manager (would be injected in real implementation)
            cache_manager = AdvancedCacheManager()
            
            # Try to get from cache
            cached_result = await cache_manager.get(f"detection_result:{cache_key}")
            if cached_result is not None:
                return cached_result
            
            # Compute result
            if asyncio.iscoroutinefunction(func):
                result = await func(*args, **kwargs)
            else:
                result = func(*args, **kwargs)
            
            # Cache result
            await cache_manager.set(f"detection_result:{cache_key}", result, ttl_seconds)
            
            return result
        
        return wrapper
    return decorator


def cache_model(ttl_seconds: int = 7200):
    """Decorator to cache trained models."""
    def decorator(func):
        async def wrapper(*args, **kwargs):
            # Generate cache key for model
            model_params = kwargs.get('parameters', {})
            algorithm = kwargs.get('algorithm', 'unknown')
            
            cache_key_data = {
                'algorithm': algorithm,
                'parameters': sorted(model_params.items()) if isinstance(model_params, dict) else str(model_params)
            }
            cache_key = hashlib.sha256(str(cache_key_data).encode()).hexdigest()
            
            cache_manager = AdvancedCacheManager()
            
            # Try to get from cache
            cached_model = await cache_manager.get(f"trained_model:{cache_key}")
            if cached_model is not None:
                return cached_model
            
            # Train model
            if asyncio.iscoroutinefunction(func):
                model = await func(*args, **kwargs)
            else:
                model = func(*args, **kwargs)
            
            # Cache model
            await cache_manager.set(f"trained_model:{cache_key}", model, ttl_seconds)
            
            return model
        
        return wrapper
    return decorator


# Global cache manager instance
_global_cache_manager: Optional[AdvancedCacheManager] = None

def get_cache_manager() -> AdvancedCacheManager:
    """Get global cache manager instance."""
    global _global_cache_manager
    if _global_cache_manager is None:
        _global_cache_manager = AdvancedCacheManager()
    return _global_cache_manager


if __name__ == "__main__":
    # Example usage and testing
    async def demo():
        print("🚀 Advanced Caching Strategies Demo")
        print("=" * 50)
        
        # Create hybrid cache manager
        memory_cache = MemoryCacheStore(max_size=100, strategy=CacheStrategy.LRU)
        disk_cache = DiskCacheStore(max_size_mb=10)
        hybrid_cache = HybridCacheStore(memory_cache=memory_cache, disk_cache=disk_cache)
        
        cache_manager = AdvancedCacheManager(cache_store=hybrid_cache)
        
        # Test caching with compute function
        print("\n1️⃣ Testing cache-or-compute pattern...")
        
        async def expensive_computation():
            print("   Computing expensive result...")
            await asyncio.sleep(1)  # Simulate expensive computation
            return {"result": "expensive_data", "computed_at": time.time()}
        
        # First call - should compute
        start_time = time.time()
        result1 = await cache_manager.get_or_compute(
            "expensive_key",
            expensive_computation,
            ttl_seconds=300
        )
        time1 = time.time() - start_time
        print(f"   First call took {time1:.2f}s: {result1['result']}")
        
        # Second call - should use cache
        start_time = time.time()
        result2 = await cache_manager.get_or_compute(
            "expensive_key",
            expensive_computation,
            ttl_seconds=300
        )
        time2 = time.time() - start_time
        print(f"   Second call took {time2:.3f}s: {result2['result']}")
        
        # Test cache statistics
        print("\n2️⃣ Cache Statistics:")
        stats = cache_manager.get_stats()
        if stats:
            for key, value in stats.items():
                print(f"   {key}: {value}")
        
        # Test different cache layers
        print("\n3️⃣ Testing cache layers...")
        
        # Test memory cache
        await cache_manager.set("memory_test", {"layer": "memory", "data": list(range(10))})
        memory_result = await cache_manager.get("memory_test")
        print(f"   Memory cache: {memory_result['layer'] if memory_result else 'None'}")
        
        # Test disk cache  
        large_data = {"layer": "disk", "data": list(range(1000))}
        await cache_manager.set("disk_test", large_data)
        disk_result = await cache_manager.get("disk_test")
        print(f"   Disk cache: {disk_result['layer'] if disk_result else 'None'}")
        
        # Test cache size
        cache_size = await cache_manager.cache_store.size()
        print(f"   Total cache entries: {cache_size}")
        
        print("\n✅ Advanced caching demo completed!")
    
    # Run demo
    asyncio.run(demo())