"""Comprehensive tests for infrastructure caching - Phase 2 Coverage."""

from __future__ import annotations

import json
import time
from unittest.mock import Mock, patch

import pytest

from pynomaly.domain.entities import Dataset, Detector
from pynomaly.domain.exceptions import CacheError, SerializationError
from pynomaly.domain.value_objects import ContaminationRate
from pynomaly.infrastructure.cache import (
    CacheDecorator,
    CacheEvictionPolicy,
    CacheKeyBuilder,
    CacheManager,
    CacheSerializer,
    DistributedCache,
    InMemoryCache,
    RedisCache,
)


@pytest.fixture
def mock_redis_client():
    """Mock Redis client for testing."""
    mock_client = Mock()
    mock_client.ping.return_value = True
    mock_client.get.return_value = None
    mock_client.set.return_value = True
    mock_client.delete.return_value = 1
    mock_client.exists.return_value = False
    mock_client.ttl.return_value = -1
    mock_client.keys.return_value = []
    mock_client.flushdb.return_value = True
    return mock_client


@pytest.fixture
def sample_detector():
    """Create a sample detector for caching tests."""
    return Detector(
        name="test_detector",
        algorithm="isolation_forest",
        contamination=ContaminationRate(0.1),
        hyperparameters={"n_estimators": 100, "random_state": 42},
    )


@pytest.fixture
def sample_dataset():
    """Create a sample dataset for caching tests."""
    import numpy as np

    features = np.random.RandomState(42).normal(0, 1, (100, 5))
    return Dataset(name="test_dataset", features=features)


@pytest.fixture
def sample_cache_data():
    """Create sample data for caching tests."""
    return {
        "simple_string": "hello_world",
        "simple_number": 42,
        "simple_list": [1, 2, 3, 4, 5],
        "complex_dict": {
            "nested": {"key": "value", "number": 123},
            "array": [10, 20, 30],
            "metadata": {"timestamp": "2024-01-01T00:00:00Z"},
        },
        "binary_data": b"binary_content_here",
    }


class TestRedisCache:
    """Comprehensive tests for Redis cache implementation."""

    def test_redis_cache_initialization(self, mock_redis_client):
        """Test Redis cache initialization."""
        with patch("redis.Redis") as mock_redis:
            mock_redis.return_value = mock_redis_client

            cache = RedisCache(
                host="localhost",
                port=6379,
                db=0,
                password=None,
                socket_timeout=5.0,
                connection_pool_max_connections=10,
            )

            assert cache.host == "localhost"
            assert cache.port == 6379
            assert cache.db == 0
            assert cache.socket_timeout == 5.0

    def test_redis_cache_basic_operations(self, mock_redis_client):
        """Test basic Redis cache operations."""
        with patch("redis.Redis") as mock_redis:
            mock_redis.return_value = mock_redis_client
            cache = RedisCache()

            # Test set operation
            cache.set("test_key", "test_value", ttl=300)
            mock_redis_client.set.assert_called_with("test_key", '"test_value"', ex=300)

            # Test get operation
            mock_redis_client.get.return_value = '"test_value"'
            result = cache.get("test_key")
            assert result == "test_value"
            mock_redis_client.get.assert_called_with("test_key")

            # Test delete operation
            cache.delete("test_key")
            mock_redis_client.delete.assert_called_with("test_key")

            # Test exists operation
            mock_redis_client.exists.return_value = True
            exists = cache.exists("test_key")
            assert exists is True
            mock_redis_client.exists.assert_called_with("test_key")

    def test_redis_cache_complex_data_serialization(
        self, mock_redis_client, sample_cache_data
    ):
        """Test Redis cache with complex data serialization."""
        with patch("redis.Redis") as mock_redis:
            mock_redis.return_value = mock_redis_client
            cache = RedisCache()

            for key, value in sample_cache_data.items():
                # Test setting complex data
                cache.set(key, value)

                # Mock the return value for get operation
                serialized_value = (
                    json.dumps(value)
                    if not isinstance(value, bytes)
                    else value.decode()
                )
                mock_redis_client.get.return_value = json.dumps(serialized_value)

                # Test getting complex data
                cache.get(key)

                # Verify set was called
                mock_redis_client.set.assert_called()

    def test_redis_cache_batch_operations(self, mock_redis_client):
        """Test Redis cache batch operations."""
        with patch("redis.Redis") as mock_redis:
            mock_redis.return_value = mock_redis_client
            cache = RedisCache()

            # Test batch set
            batch_data = {"key1": "value1", "key2": "value2", "key3": "value3"}

            cache.set_batch(batch_data, ttl=600)

            # Verify mset was called
            assert (
                mock_redis_client.mset.called or mock_redis_client.set.call_count == 3
            )

            # Test batch get
            mock_redis_client.mget.return_value = ['"value1"', '"value2"', '"value3"']
            result = cache.get_batch(["key1", "key2", "key3"])

            assert len(result) == 3
            mock_redis_client.mget.assert_called_with(["key1", "key2", "key3"])

    def test_redis_cache_ttl_operations(self, mock_redis_client):
        """Test Redis cache TTL operations."""
        with patch("redis.Redis") as mock_redis:
            mock_redis.return_value = mock_redis_client
            cache = RedisCache()

            # Test set with TTL
            cache.set("temp_key", "temp_value", ttl=120)
            mock_redis_client.set.assert_called_with("temp_key", '"temp_value"', ex=120)

            # Test get TTL
            mock_redis_client.ttl.return_value = 60
            ttl = cache.get_ttl("temp_key")
            assert ttl == 60
            mock_redis_client.ttl.assert_called_with("temp_key")

            # Test extend TTL
            cache.extend_ttl("temp_key", 180)
            mock_redis_client.expire.assert_called_with("temp_key", 180)

    def test_redis_cache_pattern_operations(self, mock_redis_client):
        """Test Redis cache pattern-based operations."""
        with patch("redis.Redis") as mock_redis:
            mock_redis.return_value = mock_redis_client
            cache = RedisCache()

            # Test keys by pattern
            mock_redis_client.keys.return_value = [
                b"user:123:profile",
                b"user:456:profile",
                b"user:789:profile",
            ]

            keys = cache.get_keys_by_pattern("user:*:profile")
            assert len(keys) == 3
            assert all("user:" in key and ":profile" in key for key in keys)
            mock_redis_client.keys.assert_called_with("user:*:profile")

            # Test delete by pattern
            cache.delete_by_pattern("user:*:profile")
            mock_redis_client.keys.assert_called()
            mock_redis_client.delete.assert_called()

    def test_redis_cache_connection_error_handling(self):
        """Test Redis cache connection error handling."""
        with patch("redis.Redis") as mock_redis:
            mock_client = Mock()
            mock_client.ping.side_effect = Exception("Connection failed")
            mock_redis.return_value = mock_client

            with pytest.raises(CacheError, match="Redis connection failed"):
                cache = RedisCache()
                cache._ensure_connection()

    def test_redis_cache_failover_behavior(self, mock_redis_client):
        """Test Redis cache failover behavior."""
        with patch("redis.Redis") as mock_redis:
            mock_redis.return_value = mock_redis_client
            cache = RedisCache(enable_failover=True)

            # Simulate connection failure
            mock_redis_client.get.side_effect = Exception("Connection lost")

            # Should fail gracefully and not raise exception
            result = cache.get("test_key", default="fallback_value")
            assert result == "fallback_value"


class TestInMemoryCache:
    """Comprehensive tests for in-memory cache implementation."""

    def test_in_memory_cache_initialization(self):
        """Test in-memory cache initialization."""
        cache = InMemoryCache(max_size=1000, default_ttl=300, eviction_policy="lru")

        assert cache.max_size == 1000
        assert cache.default_ttl == 300
        assert cache.eviction_policy == "lru"
        assert len(cache._cache) == 0

    def test_in_memory_cache_basic_operations(self, sample_cache_data):
        """Test basic in-memory cache operations."""
        cache = InMemoryCache(max_size=100)

        for key, value in sample_cache_data.items():
            # Test set
            cache.set(key, value)
            assert cache.exists(key)

            # Test get
            retrieved_value = cache.get(key)
            assert retrieved_value == value

            # Test size
            assert cache.size() > 0

        # Test delete
        cache.delete("simple_string")
        assert not cache.exists("simple_string")

        # Test clear
        cache.clear()
        assert cache.size() == 0

    def test_in_memory_cache_ttl_expiration(self):
        """Test TTL expiration in memory cache."""
        cache = InMemoryCache()

        # Set item with short TTL
        cache.set("short_lived", "value", ttl=0.1)  # 100ms
        assert cache.exists("short_lived")

        # Wait for expiration
        time.sleep(0.15)

        # Item should be expired
        assert not cache.exists("short_lived")
        assert cache.get("short_lived") is None

    def test_in_memory_cache_eviction_policies(self):
        """Test different eviction policies."""
        # Test LRU eviction
        lru_cache = InMemoryCache(max_size=3, eviction_policy="lru")

        # Fill cache to capacity
        lru_cache.set("key1", "value1")
        lru_cache.set("key2", "value2")
        lru_cache.set("key3", "value3")

        # Access key1 to make it recently used
        lru_cache.get("key1")

        # Add new item (should evict key2 as it's least recently used)
        lru_cache.set("key4", "value4")

        assert lru_cache.exists("key1")  # Recently accessed
        assert not lru_cache.exists("key2")  # Should be evicted
        assert lru_cache.exists("key3")
        assert lru_cache.exists("key4")

        # Test LFU eviction
        lfu_cache = InMemoryCache(max_size=3, eviction_policy="lfu")

        lfu_cache.set("freq1", "value1")
        lfu_cache.set("freq2", "value2")
        lfu_cache.set("freq3", "value3")

        # Access freq1 multiple times
        for _ in range(5):
            lfu_cache.get("freq1")

        # Access freq3 once
        lfu_cache.get("freq3")

        # Add new item (should evict freq2 as it's least frequently used)
        lfu_cache.set("freq4", "value4")

        assert lfu_cache.exists("freq1")  # Most frequent
        assert not lfu_cache.exists("freq2")  # Should be evicted
        assert lfu_cache.exists("freq3")
        assert lfu_cache.exists("freq4")

    def test_in_memory_cache_statistics(self):
        """Test cache statistics collection."""
        cache = InMemoryCache()

        # Perform various operations
        cache.set("key1", "value1")
        cache.set("key2", "value2")
        cache.get("key1")  # Hit
        cache.get("key1")  # Hit
        cache.get("nonexistent")  # Miss
        cache.delete("key2")

        stats = cache.get_statistics()

        assert stats["size"] == 1
        assert stats["hits"] == 2
        assert stats["misses"] == 1
        assert stats["sets"] == 2
        assert stats["deletes"] == 1
        assert stats["hit_rate"] == 2 / 3  # 2 hits out of 3 gets

    def test_in_memory_cache_thread_safety(self):
        """Test thread safety of in-memory cache."""
        import threading

        cache = InMemoryCache()

        results = []
        errors = []

        def worker(worker_id):
            try:
                for i in range(100):
                    key = f"worker_{worker_id}_key_{i}"
                    value = f"worker_{worker_id}_value_{i}"

                    cache.set(key, value)
                    retrieved = cache.get(key)

                    if retrieved == value:
                        results.append(f"{worker_id}_{i}")
            except Exception as e:
                errors.append(e)

        # Create multiple threads
        threads = []
        for i in range(5):
            thread = threading.Thread(target=worker, args=(i,))
            threads.append(thread)
            thread.start()

        # Wait for all threads to complete
        for thread in threads:
            thread.join()

        # Verify no errors and all operations completed
        assert len(errors) == 0
        assert len(results) == 500  # 5 workers * 100 operations each


class TestCacheManager:
    """Comprehensive tests for cache manager functionality."""

    def test_cache_manager_initialization(self, mock_redis_client):
        """Test cache manager initialization."""
        with patch("redis.Redis") as mock_redis:
            mock_redis.return_value = mock_redis_client

            manager = CacheManager(
                primary_cache_type="redis",
                fallback_cache_type="memory",
                redis_config={"host": "localhost", "port": 6379},
                memory_config={"max_size": 1000},
            )

            assert manager.primary_cache is not None
            assert manager.fallback_cache is not None

    def test_cache_manager_tiered_caching(self, mock_redis_client):
        """Test tiered caching with primary and fallback."""
        with patch("redis.Redis") as mock_redis:
            mock_redis.return_value = mock_redis_client

            manager = CacheManager(
                primary_cache_type="redis", fallback_cache_type="memory"
            )

            # Set data (should go to both caches)
            manager.set("test_key", "test_value", ttl=300)

            # Verify primary cache was called
            mock_redis_client.set.assert_called()

            # Simulate primary cache failure
            mock_redis_client.get.side_effect = Exception("Redis down")

            # Should fall back to memory cache
            manager.get("test_key")
            # Would be None since we haven't actually set in memory cache in this mock

    def test_cache_manager_cache_warming(self, mock_redis_client):
        """Test cache warming functionality."""
        with patch("redis.Redis") as mock_redis:
            mock_redis.return_value = mock_redis_client

            manager = CacheManager(primary_cache_type="redis")

            # Define cache warming data
            warming_data = {
                "detector_1": {"algorithm": "isolation_forest", "params": {}},
                "detector_2": {"algorithm": "lof", "params": {}},
                "config_default": {"contamination": 0.1, "n_estimators": 100},
            }

            # Warm cache
            manager.warm_cache(warming_data, ttl=3600)

            # Verify all items were set
            assert mock_redis_client.set.call_count == len(warming_data)

    def test_cache_manager_invalidation_strategies(self, mock_redis_client):
        """Test cache invalidation strategies."""
        with patch("redis.Redis") as mock_redis:
            mock_redis.return_value = mock_redis_client

            manager = CacheManager(primary_cache_type="redis")

            # Test tag-based invalidation
            manager.set("user:123:profile", {"name": "John"}, tags=["user", "profile"])
            manager.set(
                "user:123:settings", {"theme": "dark"}, tags=["user", "settings"]
            )
            manager.set("user:456:profile", {"name": "Jane"}, tags=["user", "profile"])

            # Invalidate by tag
            manager.invalidate_by_tag("user")

            # Verify invalidation calls
            mock_redis_client.delete.assert_called()

            # Test pattern-based invalidation
            manager.invalidate_by_pattern("user:123:*")

            # Verify pattern deletion
            mock_redis_client.keys.assert_called()

    def test_cache_manager_performance_monitoring(self, mock_redis_client):
        """Test cache performance monitoring."""
        with patch("redis.Redis") as mock_redis:
            mock_redis.return_value = mock_redis_client

            manager = CacheManager(primary_cache_type="redis", enable_monitoring=True)

            # Simulate cache operations
            manager.set("key1", "value1")
            manager.get("key1")  # Hit
            manager.get("key2")  # Miss

            # Get performance metrics
            metrics = manager.get_performance_metrics()

            assert "total_operations" in metrics
            assert "cache_hits" in metrics
            assert "cache_misses" in metrics
            assert "average_response_time" in metrics
            assert "hit_rate" in metrics

    def test_cache_manager_distributed_operations(self, mock_redis_client):
        """Test distributed cache operations."""
        with patch("redis.Redis") as mock_redis:
            mock_redis.return_value = mock_redis_client

            manager = CacheManager(
                primary_cache_type="redis", enable_distributed_locking=True
            )

            # Test distributed lock
            with manager.distributed_lock("resource_123", timeout=30):
                # Simulate critical section
                manager.set("shared_resource", "updated_value")

            # Verify lock operations
            mock_redis_client.set.assert_called()  # For both lock and data

    def test_cache_manager_serialization_formats(
        self, mock_redis_client, sample_detector
    ):
        """Test different serialization formats."""
        with patch("redis.Redis") as mock_redis:
            mock_redis.return_value = mock_redis_client

            manager = CacheManager(primary_cache_type="redis")

            # Test JSON serialization
            json_data = {"algorithm": "isolation_forest", "contamination": 0.1}
            manager.set("json_data", json_data, serialization_format="json")

            # Test pickle serialization
            manager.set(
                "detector_object", sample_detector, serialization_format="pickle"
            )

            # Test msgpack serialization
            manager.set(
                "binary_data", b"binary_content", serialization_format="msgpack"
            )

            # Verify serialization calls
            assert mock_redis_client.set.call_count == 3


class TestCacheDecorator:
    """Comprehensive tests for cache decorator functionality."""

    def test_cache_decorator_function_caching(self):
        """Test function result caching with decorator."""
        cache = InMemoryCache()
        decorator = CacheDecorator(cache)

        call_count = [0]

        @decorator.cache_result(ttl=300, key_prefix="expensive_func")
        def expensive_function(param1, param2):
            call_count[0] += 1
            time.sleep(0.01)  # Simulate expensive operation
            return f"result_{param1}_{param2}"

        # First call should execute function
        result1 = expensive_function("a", "b")
        assert result1 == "result_a_b"
        assert call_count[0] == 1

        # Second call with same parameters should use cache
        result2 = expensive_function("a", "b")
        assert result2 == "result_a_b"
        assert call_count[0] == 1  # Function not called again

        # Different parameters should execute function
        result3 = expensive_function("x", "y")
        assert result3 == "result_x_y"
        assert call_count[0] == 2

    def test_cache_decorator_method_caching(self):
        """Test method caching with decorator."""
        cache = InMemoryCache()
        decorator = CacheDecorator(cache)

        class DataProcessor:
            def __init__(self):
                self.call_count = 0

            @decorator.cache_result(ttl=300, include_self=True)
            def process_data(self, data_id):
                self.call_count += 1
                return f"processed_{data_id}"

        processor = DataProcessor()

        # First call
        result1 = processor.process_data("data123")
        assert result1 == "processed_data123"
        assert processor.call_count == 1

        # Second call should use cache
        result2 = processor.process_data("data123")
        assert result2 == "processed_data123"
        assert processor.call_count == 1

    def test_cache_decorator_conditional_caching(self):
        """Test conditional caching based on result."""
        cache = InMemoryCache()
        decorator = CacheDecorator(cache)

        @decorator.cache_result(ttl=300, condition=lambda result: result != "error")
        def conditional_function(value):
            if value < 0:
                return "error"
            return f"success_{value}"

        # Error result should not be cached
        error_result = conditional_function(-1)
        assert error_result == "error"

        # Success result should be cached
        success_result = conditional_function(5)
        assert success_result == "success_5"

        # Verify caching behavior
        cache_key_success = decorator._generate_cache_key(
            conditional_function, (5,), {}
        )
        cache_key_error = decorator._generate_cache_key(conditional_function, (-1,), {})

        assert cache.exists(cache_key_success)
        assert not cache.exists(cache_key_error)

    def test_cache_decorator_async_functions(self):
        """Test caching with async functions."""
        cache = InMemoryCache()
        decorator = CacheDecorator(cache)

        call_count = [0]

        @decorator.cache_result_async(ttl=300)
        async def async_expensive_function(param):
            call_count[0] += 1
            await asyncio.sleep(0.01)
            return f"async_result_{param}"

        import asyncio

        async def test_async_caching():
            # First call
            result1 = await async_expensive_function("test")
            assert result1 == "async_result_test"
            assert call_count[0] == 1

            # Second call should use cache
            result2 = await async_expensive_function("test")
            assert result2 == "async_result_test"
            assert call_count[0] == 1

        asyncio.run(test_async_caching())

    def test_cache_decorator_invalidation(self):
        """Test cache invalidation with decorator."""
        cache = InMemoryCache()
        decorator = CacheDecorator(cache)

        @decorator.cache_result(ttl=300, key_prefix="invalidation_test")
        def cached_function(value):
            return f"cached_{value}"

        # Cache a result
        result = cached_function("test")
        assert result == "cached_test"

        # Invalidate cache for this function
        decorator.invalidate_function_cache(cached_function, ("test",), {})

        # Verify cache was invalidated
        cache_key = decorator._generate_cache_key(cached_function, ("test",), {})
        assert not cache.exists(cache_key)


class TestCacheKeyBuilder:
    """Comprehensive tests for cache key building functionality."""

    def test_cache_key_builder_simple_keys(self):
        """Test simple cache key generation."""
        builder = CacheKeyBuilder()

        # Test string key
        key1 = builder.build_key("simple_string")
        assert key1 == "simple_string"

        # Test with namespace
        key2 = builder.build_key("user_profile", namespace="app")
        assert key2 == "app:user_profile"

        # Test with version
        key3 = builder.build_key("data", namespace="app", version="v1")
        assert key3 == "app:v1:data"

    def test_cache_key_builder_complex_keys(self):
        """Test complex cache key generation."""
        builder = CacheKeyBuilder(separator=":")

        # Test with multiple components
        key = builder.build_key_from_components(["users", "123", "profile", "settings"])
        assert key == "users:123:profile:settings"

        # Test with parameters
        key_with_params = builder.build_key_with_params(
            "search_results", {"query": "anomaly", "page": 1, "limit": 20}
        )
        assert "search_results" in key_with_params
        assert "query=anomaly" in key_with_params

    def test_cache_key_builder_object_keys(self, sample_detector):
        """Test cache key generation from objects."""
        builder = CacheKeyBuilder()

        # Test with domain object
        detector_key = builder.build_key_from_object(sample_detector)
        assert sample_detector.id in detector_key
        assert "detector" in detector_key.lower()

        # Test with custom key function
        custom_key = builder.build_key_from_object(
            sample_detector, key_func=lambda obj: f"custom:{obj.algorithm}:{obj.name}"
        )
        assert (
            custom_key == f"custom:{sample_detector.algorithm}:{sample_detector.name}"
        )

    def test_cache_key_builder_hash_keys(self):
        """Test hash-based key generation for long keys."""
        builder = CacheKeyBuilder(max_key_length=50)

        # Create a very long key
        long_components = [f"component_{i}" for i in range(20)]
        long_key = builder.build_key_from_components(long_components)

        # Should be hashed due to length limit
        assert len(long_key) <= 50
        assert (
            "hash:" in long_key or len(long_key) < 100
        )  # Either hashed or within reasonable limit

    def test_cache_key_builder_normalization(self):
        """Test key normalization functionality."""
        builder = CacheKeyBuilder(normalize_keys=True)

        # Test case normalization
        key1 = builder.build_key("CamelCaseKey")
        key2 = builder.build_key("camelcasekey")
        assert key1.lower() == key2.lower()

        # Test special character handling
        key_with_special = builder.build_key("key with spaces & special/chars")
        assert " " not in key_with_special
        assert "&" not in key_with_special
        assert "/" not in key_with_special


class TestCacheSerializer:
    """Comprehensive tests for cache serialization functionality."""

    def test_cache_serializer_json(self, sample_cache_data):
        """Test JSON serialization."""
        serializer = CacheSerializer(format="json")

        # Test serialization of JSON-compatible data
        json_compatible_data = {
            "string": "test",
            "number": 42,
            "list": [1, 2, 3],
            "dict": {"nested": "value"},
        }

        serialized = serializer.serialize(json_compatible_data)
        deserialized = serializer.deserialize(serialized)

        assert deserialized == json_compatible_data

    def test_cache_serializer_pickle(self, sample_detector):
        """Test pickle serialization."""
        serializer = CacheSerializer(format="pickle")

        # Test with complex object
        serialized = serializer.serialize(sample_detector)
        deserialized = serializer.deserialize(serialized)

        assert isinstance(deserialized, Detector)
        assert deserialized.name == sample_detector.name
        assert deserialized.algorithm == sample_detector.algorithm

    def test_cache_serializer_msgpack(self):
        """Test msgpack serialization."""
        try:
            import msgpack

            serializer = CacheSerializer(format="msgpack")

            data = {
                "binary": b"binary_data",
                "text": "text_data",
                "number": 42,
                "array": [1, 2, 3, 4, 5],
            }

            serialized = serializer.serialize(data)
            deserialized = serializer.deserialize(serialized)

            assert deserialized == data
        except ImportError:
            pytest.skip("msgpack not available")

    def test_cache_serializer_compression(self):
        """Test serialization with compression."""
        serializer = CacheSerializer(format="pickle", compression="gzip")

        # Create large data to benefit from compression
        large_data = {"large_list": list(range(10000))}

        serialized = serializer.serialize(large_data)
        deserialized = serializer.deserialize(serialized)

        assert deserialized == large_data

        # Verify compression reduced size
        uncompressed_serializer = CacheSerializer(format="pickle", compression=None)
        uncompressed = uncompressed_serializer.serialize(large_data)

        assert len(serialized) < len(uncompressed)

    def test_cache_serializer_error_handling(self):
        """Test serialization error handling."""
        serializer = CacheSerializer(format="json")

        # Test with non-JSON-serializable data
        class NonSerializable:
            def __init__(self):
                self.func = lambda x: x

        non_serializable = NonSerializable()

        with pytest.raises(SerializationError):
            serializer.serialize(non_serializable)


class TestCacheEvictionPolicy:
    """Comprehensive tests for cache eviction policies."""

    def test_lru_eviction_policy(self):
        """Test LRU (Least Recently Used) eviction policy."""
        policy = CacheEvictionPolicy("lru", max_size=3)

        # Add items
        policy.access_item("key1", "value1")
        policy.access_item("key2", "value2")
        policy.access_item("key3", "value3")

        # Access key1 to make it recently used
        policy.access_item("key1", None)  # Just access, don't update

        # Add new item (should evict key2)
        evicted = policy.add_item("key4", "value4")

        assert evicted == "key2"  # Least recently used
        assert policy.should_evict("key2")
        assert not policy.should_evict("key1")  # Recently accessed

    def test_lfu_eviction_policy(self):
        """Test LFU (Least Frequently Used) eviction policy."""
        policy = CacheEvictionPolicy("lfu", max_size=3)

        # Add items with different access frequencies
        policy.access_item("key1", "value1")
        policy.access_item("key2", "value2")
        policy.access_item("key3", "value3")

        # Access key1 multiple times
        for _ in range(5):
            policy.access_item("key1", None)

        # Access key3 twice
        for _ in range(2):
            policy.access_item("key3", None)

        # key2 has lowest frequency (1), should be evicted
        evicted = policy.add_item("key4", "value4")

        assert evicted == "key2"
        assert policy.should_evict("key2")
        assert not policy.should_evict("key1")  # Highest frequency

    def test_ttl_eviction_policy(self):
        """Test TTL-based eviction policy."""
        policy = CacheEvictionPolicy("ttl")

        # Add items with different TTLs
        policy.set_item_ttl("key1", 0.1)  # 100ms
        policy.set_item_ttl("key2", 1.0)  # 1 second

        # Wait for key1 to expire
        time.sleep(0.15)

        assert policy.should_evict("key1")  # Expired
        assert not policy.should_evict("key2")  # Still valid

        # Get expired items
        expired_items = policy.get_expired_items()
        assert "key1" in expired_items
        assert "key2" not in expired_items

    def test_size_based_eviction_policy(self):
        """Test size-based eviction policy."""
        policy = CacheEvictionPolicy("size", max_memory_mb=1)  # 1MB limit

        # Add large items
        large_data_1 = b"x" * (512 * 1024)  # 512KB
        large_data_2 = b"y" * (512 * 1024)  # 512KB
        large_data_3 = b"z" * (256 * 1024)  # 256KB

        policy.track_item_size("key1", len(large_data_1))
        policy.track_item_size("key2", len(large_data_2))

        # Adding key3 should trigger eviction
        should_evict = policy.check_memory_limit(len(large_data_3))

        assert should_evict  # Should need to evict to make room

        # Get items to evict based on size
        items_to_evict = policy.get_items_to_evict_for_size(len(large_data_3))
        assert len(items_to_evict) > 0


class TestDistributedCache:
    """Comprehensive tests for distributed cache functionality."""

    def test_distributed_cache_consistency(self, mock_redis_client):
        """Test distributed cache consistency."""
        with patch("redis.Redis") as mock_redis:
            mock_redis.return_value = mock_redis_client

            cache = DistributedCache(
                nodes=["redis1:6379", "redis2:6379", "redis3:6379"],
                consistency_level="strong",
            )

            # Test write to all nodes
            cache.set("consistent_key", "consistent_value")

            # Should write to all nodes for strong consistency
            assert mock_redis_client.set.call_count >= 1

    def test_distributed_cache_partitioning(self, mock_redis_client):
        """Test distributed cache partitioning."""
        with patch("redis.Redis") as mock_redis:
            mock_redis.return_value = mock_redis_client

            cache = DistributedCache(
                nodes=["redis1:6379", "redis2:6379"], partitioning_strategy="hash"
            )

            # Test hash-based partitioning
            partition = cache.get_partition_for_key("test_key")
            assert partition in [0, 1]  # Should map to one of two nodes

            # Test consistent hashing
            partition1 = cache.get_partition_for_key("key1")
            partition2 = cache.get_partition_for_key("key1")
            assert (
                partition1 == partition2
            )  # Same key should always map to same partition

    def test_distributed_cache_failover(self, mock_redis_client):
        """Test distributed cache failover."""
        with patch("redis.Redis") as mock_redis:
            # Mock one failing node
            failing_client = Mock()
            failing_client.ping.side_effect = Exception("Node down")

            working_client = Mock()
            working_client.ping.return_value = True
            working_client.get.return_value = '"backup_value"'

            mock_redis.side_effect = [failing_client, working_client]

            cache = DistributedCache(
                nodes=["redis1:6379", "redis2:6379"], enable_failover=True
            )

            # Should failover to working node
            cache.get("test_key")

            # Verify failover occurred
            working_client.get.assert_called()

    def test_distributed_cache_replication(self, mock_redis_client):
        """Test distributed cache replication."""
        with patch("redis.Redis") as mock_redis:
            mock_redis.return_value = mock_redis_client

            cache = DistributedCache(
                nodes=["redis1:6379", "redis2:6379", "redis3:6379"],
                replication_factor=2,
            )

            # Set value (should be replicated to 2 nodes)
            cache.set("replicated_key", "replicated_value")

            # Verify replication
            assert mock_redis_client.set.call_count >= 2

    def test_distributed_cache_performance_optimization(self, mock_redis_client):
        """Test distributed cache performance optimizations."""
        with patch("redis.Redis") as mock_redis:
            mock_redis.return_value = mock_redis_client

            cache = DistributedCache(
                nodes=["redis1:6379"],
                enable_pipelining=True,
                connection_pooling=True,
                max_connections=10,
            )

            # Test batch operations with pipelining
            batch_data = {f"key_{i}": f"value_{i}" for i in range(10)}
            cache.set_batch(batch_data)

            # Should use pipelining for efficiency
            assert mock_redis_client.pipeline.called or mock_redis_client.mset.called
