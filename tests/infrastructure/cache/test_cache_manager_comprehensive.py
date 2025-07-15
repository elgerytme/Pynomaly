"""
Comprehensive tests for cache manager infrastructure.

This module provides extensive testing for cache management including
multiple backends, compression, eviction policies, write strategies,
and performance optimization.
"""

import asyncio
import pickle
import threading
import time
from datetime import datetime, timedelta
from unittest.mock import AsyncMock, Mock, patch

import pytest

from pynomaly.infrastructure.cache.cache_manager import (
    CacheConfig,
    CacheConnectionError,
    CacheEntry,
    CacheKeyError,
    CacheManager,
    CacheSerializationError,
    CacheStats,
    CompressionType,
    EvictionPolicy,
    InMemoryCacheBackend,
    RedisCacheBackend,
)


class TestCacheConfig:
    """Test cache configuration."""

    def test_cache_config_creation(self):
        """Test cache configuration creation."""
        config = CacheConfig(
            backend_type="redis",
            host="localhost",
            port=6379,
            db=0,
            password="secret",
            max_connections=10,
            default_ttl=3600,
            compression=CompressionType.GZIP,
            eviction_policy=EvictionPolicy.LRU,
            max_memory_mb=512,
        )

        assert config.backend_type == "redis"
        assert config.host == "localhost"
        assert config.port == 6379
        assert config.db == 0
        assert config.password == "secret"
        assert config.max_connections == 10
        assert config.default_ttl == 3600
        assert config.compression == CompressionType.GZIP
        assert config.eviction_policy == EvictionPolicy.LRU
        assert config.max_memory_mb == 512

    def test_cache_config_defaults(self):
        """Test default cache configuration values."""
        config = CacheConfig()

        assert config.backend_type == "memory"
        assert config.host == "localhost"
        assert config.port == 6379
        assert config.db == 0
        assert config.password is None
        assert config.max_connections == 10
        assert config.default_ttl == 3600
        assert config.compression == CompressionType.NONE
        assert config.eviction_policy == EvictionPolicy.LRU
        assert config.max_memory_mb == 128

    def test_cache_config_validation(self):
        """Test cache configuration validation."""
        # Valid configuration
        config = CacheConfig(max_memory_mb=256)
        assert config.max_memory_mb == 256

        # Invalid memory size
        with pytest.raises(ValueError):
            CacheConfig(max_memory_mb=0)

        # Invalid TTL
        with pytest.raises(ValueError):
            CacheConfig(default_ttl=-1)

        # Invalid port
        with pytest.raises(ValueError):
            CacheConfig(port=0)

    def test_cache_config_serialization(self):
        """Test cache configuration serialization."""
        config = CacheConfig(
            backend_type="redis",
            host="cache.example.com",
            port=6380,
            max_memory_mb=1024,
        )

        config_dict = config.to_dict()

        assert config_dict["backend_type"] == "redis"
        assert config_dict["host"] == "cache.example.com"
        assert config_dict["port"] == 6380
        assert config_dict["max_memory_mb"] == 1024

        # Test from_dict
        restored_config = CacheConfig.from_dict(config_dict)

        assert restored_config.backend_type == config.backend_type
        assert restored_config.host == config.host
        assert restored_config.port == config.port
        assert restored_config.max_memory_mb == config.max_memory_mb


class TestCacheEntry:
    """Test cache entry functionality."""

    def test_cache_entry_creation(self):
        """Test cache entry creation."""
        data = {"key": "value", "number": 42}
        entry = CacheEntry(
            key="test_key",
            value=data,
            ttl=3600,
            created_at=datetime.now(),
            access_count=0,
            compressed=False,
        )

        assert entry.key == "test_key"
        assert entry.value == data
        assert entry.ttl == 3600
        assert entry.access_count == 0
        assert entry.compressed is False
        assert entry.created_at is not None

    def test_cache_entry_expiration(self):
        """Test cache entry expiration logic."""
        # Non-expiring entry
        entry = CacheEntry(
            key="non_expiring", value="data", ttl=None, created_at=datetime.now()
        )

        assert entry.is_expired() is False

        # Expired entry
        past_time = datetime.now() - timedelta(seconds=3600)
        expired_entry = CacheEntry(
            key="expired",
            value="data",
            ttl=1800,  # 30 minutes
            created_at=past_time,
        )

        assert expired_entry.is_expired() is True

        # Non-expired entry
        recent_time = datetime.now() - timedelta(seconds=60)
        valid_entry = CacheEntry(
            key="valid", value="data", ttl=3600, created_at=recent_time
        )

        assert valid_entry.is_expired() is False

    def test_cache_entry_size_calculation(self):
        """Test cache entry size calculation."""
        small_entry = CacheEntry("small", "data", 3600, datetime.now())
        large_data = "x" * 10000
        large_entry = CacheEntry("large", large_data, 3600, datetime.now())

        small_size = small_entry.get_size()
        large_size = large_entry.get_size()

        assert large_size > small_size
        assert small_size > 0
        assert large_size > 10000

    def test_cache_entry_access_tracking(self):
        """Test cache entry access tracking."""
        entry = CacheEntry("tracked", "data", 3600, datetime.now())

        assert entry.access_count == 0
        assert entry.last_accessed is None

        entry.record_access()

        assert entry.access_count == 1
        assert entry.last_accessed is not None

        # Multiple accesses
        for _ in range(5):
            entry.record_access()

        assert entry.access_count == 6

    def test_cache_entry_compression(self):
        """Test cache entry compression functionality."""
        large_data = "x" * 1000
        entry = CacheEntry("compressible", large_data, 3600, datetime.now())

        # Compress entry
        entry.compress()

        assert entry.compressed is True
        assert isinstance(entry.value, bytes)

        # Decompress entry
        entry.decompress()

        assert entry.compressed is False
        assert entry.value == large_data

    def test_cache_entry_serialization(self):
        """Test cache entry serialization."""
        data = {"complex": "data", "list": [1, 2, 3], "nested": {"key": "value"}}
        entry = CacheEntry("serializable", data, 3600, datetime.now())

        # Serialize
        serialized = entry.serialize()
        assert isinstance(serialized, bytes)

        # Deserialize
        deserialized_entry = CacheEntry.deserialize(serialized)

        assert deserialized_entry.key == entry.key
        assert deserialized_entry.value == entry.value
        assert deserialized_entry.ttl == entry.ttl


class TestInMemoryCacheBackend:
    """Test in-memory cache backend."""

    @pytest.fixture
    def cache_config(self):
        """Create cache configuration."""
        return CacheConfig(
            backend_type="memory", max_memory_mb=64, eviction_policy=EvictionPolicy.LRU
        )

    @pytest.fixture
    def memory_cache(self, cache_config):
        """Create in-memory cache backend."""
        return InMemoryCacheBackend(cache_config)

    def test_memory_cache_initialization(self, memory_cache, cache_config):
        """Test memory cache initialization."""
        assert memory_cache.config == cache_config
        assert len(memory_cache._cache) == 0
        assert memory_cache._current_size == 0

    def test_memory_cache_set_and_get(self, memory_cache):
        """Test basic set and get operations."""
        key = "test_key"
        value = {"data": "test_value", "number": 42}

        # Set value
        memory_cache.set(key, value, ttl=3600)

        # Get value
        retrieved_value = memory_cache.get(key)

        assert retrieved_value == value

    def test_memory_cache_get_nonexistent(self, memory_cache):
        """Test getting non-existent key."""
        result = memory_cache.get("nonexistent_key")
        assert result is None

    def test_memory_cache_delete(self, memory_cache):
        """Test delete operation."""
        key = "deletable_key"
        value = "deletable_value"

        # Set and verify
        memory_cache.set(key, value)
        assert memory_cache.get(key) == value

        # Delete and verify
        result = memory_cache.delete(key)
        assert result is True
        assert memory_cache.get(key) is None

    def test_memory_cache_delete_nonexistent(self, memory_cache):
        """Test deleting non-existent key."""
        result = memory_cache.delete("nonexistent")
        assert result is False

    def test_memory_cache_exists(self, memory_cache):
        """Test key existence check."""
        key = "existence_key"

        # Key doesn't exist initially
        assert memory_cache.exists(key) is False

        # Set key
        memory_cache.set(key, "value")

        # Key exists now
        assert memory_cache.exists(key) is True

    def test_memory_cache_clear(self, memory_cache):
        """Test cache clear operation."""
        # Set multiple keys
        for i in range(5):
            memory_cache.set(f"key_{i}", f"value_{i}")

        assert len(memory_cache._cache) == 5

        # Clear cache
        memory_cache.clear()

        assert len(memory_cache._cache) == 0
        assert memory_cache._current_size == 0

    def test_memory_cache_ttl_expiration(self, memory_cache):
        """Test TTL expiration."""
        key = "expiring_key"
        value = "expiring_value"

        # Set with short TTL
        memory_cache.set(key, value, ttl=1)

        # Should exist immediately
        assert memory_cache.get(key) == value

        # Wait for expiration
        time.sleep(1.1)

        # Should be expired and return None
        assert memory_cache.get(key) is None

    def test_memory_cache_lru_eviction(self, memory_cache):
        """Test LRU eviction policy."""
        # Fill cache to near capacity
        for i in range(100):
            memory_cache.set(f"key_{i}", "x" * 1000)  # 1KB each

        # Access some keys to make them recently used
        for i in range(10):
            memory_cache.get(f"key_{i}")

        # Add more data to trigger eviction
        for i in range(100, 150):
            memory_cache.set(f"key_{i}", "x" * 1000)

        # Recently accessed keys should still exist
        for i in range(10):
            assert memory_cache.exists(f"key_{i}") is True

        # Some older keys should be evicted
        evicted_count = 0
        for i in range(50, 100):
            if not memory_cache.exists(f"key_{i}"):
                evicted_count += 1

        assert evicted_count > 0

    def test_memory_cache_size_tracking(self, memory_cache):
        """Test memory size tracking."""
        initial_size = memory_cache._current_size

        # Add data
        large_value = "x" * 5000  # 5KB
        memory_cache.set("large_key", large_value)

        # Size should increase
        assert memory_cache._current_size > initial_size

        # Delete data
        memory_cache.delete("large_key")

        # Size should decrease
        assert memory_cache._current_size == initial_size

    def test_memory_cache_stats(self, memory_cache):
        """Test cache statistics."""
        # Generate some activity
        for i in range(10):
            memory_cache.set(f"key_{i}", f"value_{i}")

        # Some hits
        for i in range(5):
            memory_cache.get(f"key_{i}")

        # Some misses
        for i in range(10, 15):
            memory_cache.get(f"key_{i}")

        stats = memory_cache.get_stats()

        assert isinstance(stats, CacheStats)
        assert stats.hits >= 5
        assert stats.misses >= 5
        assert stats.sets >= 10
        assert stats.total_keys == 10

    def test_memory_cache_thread_safety(self, memory_cache):
        """Test memory cache thread safety."""

        def worker(worker_id):
            for i in range(100):
                key = f"worker_{worker_id}_key_{i}"
                value = f"worker_{worker_id}_value_{i}"
                memory_cache.set(key, value)

                retrieved = memory_cache.get(key)
                assert retrieved == value

        # Start multiple worker threads
        threads = []
        for worker_id in range(5):
            thread = threading.Thread(target=worker, args=(worker_id,))
            threads.append(thread)
            thread.start()

        # Wait for completion
        for thread in threads:
            thread.join()

        # Verify final state
        stats = memory_cache.get_stats()
        assert stats.total_keys == 500  # 5 workers * 100 keys each

    def test_memory_cache_compression(self, memory_cache):
        """Test memory cache with compression."""
        # Configure compression
        memory_cache.config.compression = CompressionType.GZIP

        large_data = "x" * 10000  # 10KB of repeated data (compresses well)
        key = "compressed_key"

        # Set data (should be compressed)
        memory_cache.set(key, large_data)

        # Get data (should be decompressed)
        retrieved = memory_cache.get(key)

        assert retrieved == large_data

        # Check that stored data is actually compressed
        entry = memory_cache._cache[key]
        assert entry.compressed is True


class TestRedisCacheBackend:
    """Test Redis cache backend."""

    @pytest.fixture
    def redis_config(self):
        """Create Redis cache configuration."""
        return CacheConfig(
            backend_type="redis",
            host="localhost",
            port=6379,
            db=1,  # Use different DB for testing
            max_connections=5,
        )

    @pytest.fixture
    def mock_redis(self):
        """Create mock Redis client."""
        mock_redis = Mock()
        mock_redis.get.return_value = None
        mock_redis.set.return_value = True
        mock_redis.delete.return_value = 1
        mock_redis.exists.return_value = False
        mock_redis.flushdb.return_value = True
        mock_redis.info.return_value = {"used_memory": 1024, "connected_clients": 1}
        return mock_redis

    @pytest.fixture
    def redis_cache(self, redis_config, mock_redis):
        """Create Redis cache backend with mock."""
        with patch("redis.Redis", return_value=mock_redis):
            cache = RedisCacheBackend(redis_config)
            cache._redis = mock_redis
            return cache

    def test_redis_cache_initialization(self, redis_cache, redis_config):
        """Test Redis cache initialization."""
        assert redis_cache.config == redis_config
        assert redis_cache._redis is not None

    def test_redis_cache_connection_failure(self, redis_config):
        """Test Redis connection failure handling."""
        with patch("redis.Redis") as mock_redis_class:
            mock_redis_instance = Mock()
            mock_redis_instance.ping.side_effect = ConnectionError("Redis unavailable")
            mock_redis_class.return_value = mock_redis_instance

            with pytest.raises(CacheConnectionError):
                RedisCacheBackend(redis_config)

    def test_redis_cache_set_and_get(self, redis_cache, mock_redis):
        """Test Redis set and get operations."""
        key = "redis_test_key"
        value = {"data": "redis_test_value"}

        # Mock serialized value
        serialized_value = pickle.dumps(value)
        mock_redis.get.return_value = serialized_value

        # Set value
        redis_cache.set(key, value, ttl=3600)

        # Verify Redis set was called
        mock_redis.setex.assert_called_once()

        # Get value
        retrieved_value = redis_cache.get(key)

        # Verify Redis get was called
        mock_redis.get.assert_called_with(key)
        assert retrieved_value == value

    def test_redis_cache_get_nonexistent(self, redis_cache, mock_redis):
        """Test Redis get for non-existent key."""
        mock_redis.get.return_value = None

        result = redis_cache.get("nonexistent")

        assert result is None
        mock_redis.get.assert_called_with("nonexistent")

    def test_redis_cache_delete(self, redis_cache, mock_redis):
        """Test Redis delete operation."""
        key = "deletable_redis_key"

        mock_redis.delete.return_value = 1  # 1 key deleted

        result = redis_cache.delete(key)

        assert result is True
        mock_redis.delete.assert_called_with(key)

    def test_redis_cache_delete_nonexistent(self, redis_cache, mock_redis):
        """Test Redis delete for non-existent key."""
        mock_redis.delete.return_value = 0  # 0 keys deleted

        result = redis_cache.delete("nonexistent")

        assert result is False

    def test_redis_cache_exists(self, redis_cache, mock_redis):
        """Test Redis key existence check."""
        key = "existence_redis_key"

        mock_redis.exists.return_value = True

        result = redis_cache.exists(key)

        assert result is True
        mock_redis.exists.assert_called_with(key)

    def test_redis_cache_clear(self, redis_cache, mock_redis):
        """Test Redis cache clear operation."""
        redis_cache.clear()

        mock_redis.flushdb.assert_called_once()

    def test_redis_cache_pipeline_operations(self, redis_cache, mock_redis):
        """Test Redis pipeline operations."""
        # Mock pipeline
        mock_pipeline = Mock()
        mock_redis.pipeline.return_value = mock_pipeline
        mock_pipeline.execute.return_value = [True, True, True]

        operations = [
            ("set", "key1", "value1"),
            ("set", "key2", "value2"),
            ("set", "key3", "value3"),
        ]

        redis_cache.execute_pipeline(operations)

        mock_redis.pipeline.assert_called_once()
        mock_pipeline.execute.assert_called_once()

    def test_redis_cache_stats(self, redis_cache, mock_redis):
        """Test Redis cache statistics."""
        mock_redis.info.return_value = {
            "used_memory": 2048,
            "connected_clients": 5,
            "total_commands_processed": 1000,
        }

        stats = redis_cache.get_stats()

        assert isinstance(stats, CacheStats)
        assert stats.memory_usage == 2048
        mock_redis.info.assert_called()

    def test_redis_cache_connection_pooling(self, redis_cache, mock_redis):
        """Test Redis connection pooling."""
        # Connection pool should be configured
        assert redis_cache.config.max_connections == 5

        # Multiple operations should use the pool
        for i in range(10):
            redis_cache.get(f"key_{i}")

        # Verify calls were made (connection pooling is transparent)
        assert mock_redis.get.call_count == 10

    def test_redis_cache_serialization_error(self, redis_cache, mock_redis):
        """Test Redis serialization error handling."""
        # Create an object that can't be pickled
        unpicklable_object = lambda x: x

        with pytest.raises(CacheSerializationError):
            redis_cache.set("unpicklable", unpicklable_object)

    def test_redis_cache_deserialization_error(self, redis_cache, mock_redis):
        """Test Redis deserialization error handling."""
        # Mock corrupted data
        mock_redis.get.return_value = b"corrupted_data"

        with pytest.raises(CacheSerializationError):
            redis_cache.get("corrupted_key")

    @pytest.mark.asyncio
    async def test_redis_cache_async_operations(self, redis_cache, mock_redis):
        """Test Redis async operations."""
        # Mock async Redis client
        mock_async_redis = AsyncMock()
        mock_async_redis.get.return_value = pickle.dumps("async_value")
        mock_async_redis.setex.return_value = True

        redis_cache._async_redis = mock_async_redis

        # Async set
        await redis_cache.aset("async_key", "async_value", ttl=3600)
        mock_async_redis.setex.assert_called_once()

        # Async get
        result = await redis_cache.aget("async_key")
        assert result == "async_value"
        mock_async_redis.get.assert_called_with("async_key")


class TestCacheManager:
    """Test cache manager functionality."""

    @pytest.fixture
    def memory_config(self):
        """Create memory cache configuration."""
        return CacheConfig(backend_type="memory", max_memory_mb=64)

    @pytest.fixture
    def cache_manager(self, memory_config):
        """Create cache manager."""
        return CacheManager(memory_config)

    def test_cache_manager_initialization(self, cache_manager, memory_config):
        """Test cache manager initialization."""
        assert cache_manager.config == memory_config
        assert cache_manager.backend is not None
        assert isinstance(cache_manager.backend, InMemoryCacheBackend)

    def test_cache_manager_backend_selection(self):
        """Test cache backend selection."""
        # Memory backend
        memory_config = CacheConfig(backend_type="memory")
        memory_manager = CacheManager(memory_config)
        assert isinstance(memory_manager.backend, InMemoryCacheBackend)

        # Redis backend (with mock)
        with patch("redis.Redis"):
            redis_config = CacheConfig(backend_type="redis")
            redis_manager = CacheManager(redis_config)
            assert isinstance(redis_manager.backend, RedisCacheBackend)

    def test_cache_manager_invalid_backend(self):
        """Test cache manager with invalid backend."""
        invalid_config = CacheConfig(backend_type="invalid")

        with pytest.raises(ValueError):
            CacheManager(invalid_config)

    def test_cache_manager_basic_operations(self, cache_manager):
        """Test basic cache operations through manager."""
        key = "manager_test_key"
        value = {"manager": "test_value", "complex": {"nested": "data"}}

        # Set value
        cache_manager.set(key, value, ttl=3600)

        # Get value
        retrieved = cache_manager.get(key)
        assert retrieved == value

        # Check existence
        assert cache_manager.exists(key) is True

        # Delete value
        assert cache_manager.delete(key) is True
        assert cache_manager.exists(key) is False

    def test_cache_manager_batch_operations(self, cache_manager):
        """Test batch operations."""
        # Batch set
        data = {
            "batch_key1": "batch_value1",
            "batch_key2": "batch_value2",
            "batch_key3": "batch_value3",
        }

        cache_manager.set_many(data, ttl=3600)

        # Batch get
        keys = list(data.keys())
        results = cache_manager.get_many(keys)

        assert len(results) == 3
        for key, value in data.items():
            assert results[key] == value

        # Batch delete
        deleted_count = cache_manager.delete_many(keys)
        assert deleted_count == 3

    def test_cache_manager_decorator(self, cache_manager):
        """Test cache decorator functionality."""
        call_count = 0

        @cache_manager.cached(ttl=3600)
        def expensive_function(x, y):
            nonlocal call_count
            call_count += 1
            return x + y

        # First call should execute function
        result1 = expensive_function(1, 2)
        assert result1 == 3
        assert call_count == 1

        # Second call should use cache
        result2 = expensive_function(1, 2)
        assert result2 == 3
        assert call_count == 1  # Still 1, cached result used

        # Different arguments should execute function again
        result3 = expensive_function(2, 3)
        assert result3 == 5
        assert call_count == 2

    def test_cache_manager_async_decorator(self, cache_manager):
        """Test async cache decorator functionality."""
        call_count = 0

        @cache_manager.async_cached(ttl=3600)
        async def async_expensive_function(x, y):
            nonlocal call_count
            call_count += 1
            await asyncio.sleep(0.01)  # Simulate async work
            return x * y

        async def test_async():
            # First call should execute function
            result1 = await async_expensive_function(3, 4)
            assert result1 == 12
            assert call_count == 1

            # Second call should use cache
            result2 = await async_expensive_function(3, 4)
            assert result2 == 12
            assert call_count == 1  # Still 1, cached result used

        asyncio.run(test_async())

    def test_cache_manager_invalidation_patterns(self, cache_manager):
        """Test cache invalidation patterns."""
        # Set up test data
        cache_manager.set("user:123:profile", {"name": "John"})
        cache_manager.set("user:123:settings", {"theme": "dark"})
        cache_manager.set("user:456:profile", {"name": "Jane"})
        cache_manager.set("product:789:details", {"name": "Widget"})

        # Invalidate by pattern
        invalidated = cache_manager.invalidate_pattern("user:123:*")

        assert invalidated >= 2  # Should invalidate user:123 keys
        assert cache_manager.exists("user:123:profile") is False
        assert cache_manager.exists("user:123:settings") is False
        assert cache_manager.exists("user:456:profile") is True  # Different user
        assert cache_manager.exists("product:789:details") is True  # Different pattern

    def test_cache_manager_tags(self, cache_manager):
        """Test cache tagging functionality."""
        # Set values with tags
        cache_manager.set("user:123", {"name": "John"}, tags=["user", "profile"])
        cache_manager.set("user:456", {"name": "Jane"}, tags=["user", "profile"])
        cache_manager.set("product:789", {"name": "Widget"}, tags=["product"])

        # Invalidate by tag
        invalidated = cache_manager.invalidate_tag("user")

        assert invalidated >= 2
        assert cache_manager.exists("user:123") is False
        assert cache_manager.exists("user:456") is False
        assert cache_manager.exists("product:789") is True  # Different tag

    def test_cache_manager_write_through_strategy(self, cache_manager):
        """Test write-through caching strategy."""
        # Mock database
        database = {}

        def db_write(key, value):
            database[key] = value

        def db_read(key):
            return database.get(key)

        # Configure write-through
        cache_manager.configure_write_strategy("write_through", db_write, db_read)

        key = "write_through_key"
        value = {"data": "write_through_value"}

        # Set with write-through
        cache_manager.set_with_strategy(key, value)

        # Should be in both cache and database
        assert cache_manager.get(key) == value
        assert database[key] == value

    def test_cache_manager_write_behind_strategy(self, cache_manager):
        """Test write-behind caching strategy."""
        # Mock database
        database = {}
        write_queue = []

        def db_write(key, value):
            database[key] = value
            write_queue.append((key, value))

        # Configure write-behind
        cache_manager.configure_write_strategy("write_behind", db_write, None)

        key = "write_behind_key"
        value = {"data": "write_behind_value"}

        # Set with write-behind
        cache_manager.set_with_strategy(key, value)

        # Should be in cache immediately
        assert cache_manager.get(key) == value

        # Database write should happen asynchronously
        # (This would require a background worker in real implementation)

    def test_cache_manager_statistics(self, cache_manager):
        """Test cache statistics collection."""
        # Generate some activity
        for i in range(10):
            cache_manager.set(f"stats_key_{i}", f"value_{i}")

        # Some hits
        for i in range(5):
            cache_manager.get(f"stats_key_{i}")

        # Some misses
        for i in range(10, 15):
            cache_manager.get(f"stats_key_{i}")

        stats = cache_manager.get_stats()

        assert isinstance(stats, CacheStats)
        assert stats.hits >= 5
        assert stats.misses >= 5
        assert stats.sets >= 10
        assert stats.hit_rate > 0
        assert stats.miss_rate > 0

    def test_cache_manager_health_check(self, cache_manager):
        """Test cache health check."""
        health = cache_manager.health_check()

        assert "status" in health
        assert "backend_type" in health
        assert "memory_usage" in health
        assert "connection_status" in health

        # Should be healthy for in-memory cache
        assert health["status"] == "healthy"

    def test_cache_manager_monitoring(self, cache_manager):
        """Test cache monitoring functionality."""
        # Generate activity
        for i in range(20):
            cache_manager.set(f"monitor_key_{i}", f"value_{i}")
            if i % 2 == 0:
                cache_manager.get(f"monitor_key_{i}")

        monitoring_data = cache_manager.get_monitoring_data()

        assert "performance_metrics" in monitoring_data
        assert "error_rates" in monitoring_data
        assert "capacity_metrics" in monitoring_data
        assert "operation_latencies" in monitoring_data

    def test_cache_manager_performance_under_load(self, cache_manager):
        """Test cache manager performance under load."""
        import time

        # Measure performance of many operations
        start_time = time.time()

        # Perform many operations
        for i in range(1000):
            key = f"perf_key_{i}"
            value = f"perf_value_{i}"

            cache_manager.set(key, value)
            retrieved = cache_manager.get(key)
            assert retrieved == value

        end_time = time.time()
        duration = end_time - start_time

        # Should complete reasonably quickly
        assert duration < 5.0  # Less than 5 seconds for 1000 ops

        # Verify all data is accessible
        for i in range(0, 1000, 100):  # Sample every 100th key
            key = f"perf_key_{i}"
            assert cache_manager.exists(key) is True

    def test_cache_manager_concurrent_access(self, cache_manager):
        """Test concurrent access to cache manager."""

        def worker(worker_id):
            for i in range(50):
                key = f"concurrent_{worker_id}_{i}"
                value = f"value_{worker_id}_{i}"

                cache_manager.set(key, value)
                retrieved = cache_manager.get(key)
                assert retrieved == value

        # Start multiple workers
        threads = []
        for worker_id in range(10):
            thread = threading.Thread(target=worker, args=(worker_id,))
            threads.append(thread)
            thread.start()

        # Wait for completion
        for thread in threads:
            thread.join()

        # Verify final state
        stats = cache_manager.get_stats()
        assert stats.total_keys == 500  # 10 workers * 50 keys
        assert stats.hits >= 500  # At least as many hits as sets

    def test_cache_manager_memory_pressure_handling(self, cache_manager):
        """Test cache behavior under memory pressure."""
        # Fill cache to capacity
        large_value = "x" * 10000  # 10KB per entry

        for i in range(100):  # Try to fill beyond capacity
            cache_manager.set(f"pressure_key_{i}", large_value)

        # Cache should handle memory pressure gracefully
        stats = cache_manager.get_stats()

        # Should not exceed configured memory limit significantly
        memory_usage_mb = stats.memory_usage / (1024 * 1024)
        assert (
            memory_usage_mb <= cache_manager.config.max_memory_mb * 1.2
        )  # 20% tolerance

    def test_cache_manager_error_handling(self, cache_manager):
        """Test cache manager error handling."""
        # Test with various error conditions

        # Invalid key type
        with pytest.raises((TypeError, CacheKeyError)):
            cache_manager.set(None, "value")

        # Invalid value for serialization
        with pytest.raises((TypeError, CacheSerializationError)):
            cache_manager.set("key", lambda x: x)  # Function can't be serialized

        # Cache should remain functional after errors
        cache_manager.set("recovery_key", "recovery_value")
        assert cache_manager.get("recovery_key") == "recovery_value"

    def test_cache_manager_configuration_changes(self, cache_manager):
        """Test dynamic configuration changes."""
        original_ttl = cache_manager.config.default_ttl

        # Change default TTL
        cache_manager.update_config(default_ttl=7200)

        assert cache_manager.config.default_ttl == 7200
        assert cache_manager.config.default_ttl != original_ttl

        # Test with new TTL
        cache_manager.set("config_test_key", "config_test_value")

        # Verify it uses new TTL (would need to check actual expiration time)
        assert cache_manager.exists("config_test_key") is True

    def test_cache_manager_backup_and_restore(self, cache_manager):
        """Test cache backup and restore functionality."""
        # Set up test data
        test_data = {
            "backup_key1": "backup_value1",
            "backup_key2": {"complex": "data"},
            "backup_key3": [1, 2, 3, 4, 5],
        }

        for key, value in test_data.items():
            cache_manager.set(key, value)

        # Create backup
        backup_data = cache_manager.create_backup()

        assert isinstance(backup_data, (dict, bytes))

        # Clear cache
        cache_manager.clear()
        assert cache_manager.get_stats().total_keys == 0

        # Restore from backup
        cache_manager.restore_backup(backup_data)

        # Verify restored data
        for key, expected_value in test_data.items():
            assert cache_manager.get(key) == expected_value
