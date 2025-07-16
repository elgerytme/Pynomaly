"""Tests for cache decorators module."""

from datetime import datetime, timedelta
from unittest.mock import AsyncMock, MagicMock

import pytest

from pynomaly.infrastructure.cache.decorators import (
    AsyncCacheDecorator,
    CacheConfiguration,
    CacheDecorator,
    CacheInvalidator,
    CacheStats,
    SyncCacheDecorator,
    cache_async,
    cache_key_generator,
    cache_sync,
    invalidate_cache,
)


class TestCacheDecorator:
    """Test CacheDecorator base class."""

    def test_cache_decorator_initialization(self):
        """Test CacheDecorator initialization."""
        config = CacheConfiguration(
            ttl=3600,
            key_prefix="test",
            enabled=True
        )

        decorator = CacheDecorator(config)

        assert decorator.config == config
        assert decorator.stats is not None
        assert decorator.stats.hits == 0
        assert decorator.stats.misses == 0

    def test_generate_cache_key(self):
        """Test cache key generation."""
        config = CacheConfiguration(key_prefix="test")
        decorator = CacheDecorator(config)

        # Test with simple arguments
        key = decorator.generate_cache_key("func_name", (1, 2), {"key": "value"})
        assert key.startswith("test:")
        assert "func_name" in key

        # Test with complex arguments
        key = decorator.generate_cache_key("func_name", ([1, 2, 3], {"nested": "dict"}), {"key": "value"})
        assert key.startswith("test:")
        assert "func_name" in key

    def test_generate_cache_key_with_custom_generator(self):
        """Test cache key generation with custom generator."""
        config = CacheConfiguration(key_prefix="test")
        decorator = CacheDecorator(config)

        def custom_generator(func_name, args, kwargs):
            return f"custom:{func_name}:{len(args)}"

        key = decorator.generate_cache_key("func_name", (1, 2), {"key": "value"}, custom_generator)
        assert key == "custom:func_name:2"

    def test_is_cache_enabled(self):
        """Test cache enabled check."""
        config = CacheConfiguration(enabled=True)
        decorator = CacheDecorator(config)
        assert decorator.is_cache_enabled() is True

        config = CacheConfiguration(enabled=False)
        decorator = CacheDecorator(config)
        assert decorator.is_cache_enabled() is False

    def test_should_cache_result(self):
        """Test should cache result logic."""
        config = CacheConfiguration()
        decorator = CacheDecorator(config)

        # Should cache non-None values
        assert decorator.should_cache_result("value") is True
        assert decorator.should_cache_result(42) is True
        assert decorator.should_cache_result([1, 2, 3]) is True

        # Should not cache None by default
        assert decorator.should_cache_result(None) is False

        # Test with cache_none enabled
        config = CacheConfiguration(cache_none=True)
        decorator = CacheDecorator(config)
        assert decorator.should_cache_result(None) is True

    def test_serialize_result(self):
        """Test result serialization."""
        config = CacheConfiguration(serialization_format="json")
        decorator = CacheDecorator(config)

        # Test basic serialization
        result = decorator.serialize_result({"key": "value"})
        assert isinstance(result, str)

        # Test with pickle format
        config = CacheConfiguration(serialization_format="pickle")
        decorator = CacheDecorator(config)
        result = decorator.serialize_result({"key": "value"})
        assert isinstance(result, bytes)

    def test_deserialize_result(self):
        """Test result deserialization."""
        config = CacheConfiguration(serialization_format="json")
        decorator = CacheDecorator(config)

        # Test JSON deserialization
        original = {"key": "value"}
        serialized = decorator.serialize_result(original)
        deserialized = decorator.deserialize_result(serialized)
        assert deserialized == original

    def test_update_stats(self):
        """Test statistics updating."""
        config = CacheConfiguration()
        decorator = CacheDecorator(config)

        initial_hits = decorator.stats.hits
        initial_misses = decorator.stats.misses

        decorator.update_stats(hit=True)
        assert decorator.stats.hits == initial_hits + 1
        assert decorator.stats.misses == initial_misses

        decorator.update_stats(hit=False)
        assert decorator.stats.hits == initial_hits + 1
        assert decorator.stats.misses == initial_misses + 1

    def test_get_stats(self):
        """Test getting statistics."""
        config = CacheConfiguration()
        decorator = CacheDecorator(config)

        decorator.stats.hits = 10
        decorator.stats.misses = 5

        stats = decorator.get_stats()
        assert stats.hits == 10
        assert stats.misses == 5
        assert stats.hit_rate == 0.667
        assert stats.total_requests == 15

    def test_reset_stats(self):
        """Test resetting statistics."""
        config = CacheConfiguration()
        decorator = CacheDecorator(config)

        decorator.stats.hits = 10
        decorator.stats.misses = 5

        decorator.reset_stats()
        assert decorator.stats.hits == 0
        assert decorator.stats.misses == 0


class TestAsyncCacheDecorator:
    """Test AsyncCacheDecorator class."""

    def test_async_cache_decorator_initialization(self):
        """Test AsyncCacheDecorator initialization."""
        config = CacheConfiguration()
        mock_cache = AsyncMock()

        decorator = AsyncCacheDecorator(config, mock_cache)

        assert decorator.config == config
        assert decorator.cache == mock_cache

    @pytest.mark.asyncio
    async def test_async_cache_decorator_cache_hit(self):
        """Test async cache decorator with cache hit."""
        config = CacheConfiguration()
        mock_cache = AsyncMock()
        mock_cache.get.return_value = '{"result": "cached_value"}'

        decorator = AsyncCacheDecorator(config, mock_cache)

        @decorator
        async def test_func(x, y):
            return {"result": f"computed_{x}_{y}"}

        result = await test_func(1, 2)

        assert result == {"result": "cached_value"}
        mock_cache.get.assert_called_once()
        assert decorator.stats.hits == 1
        assert decorator.stats.misses == 0

    @pytest.mark.asyncio
    async def test_async_cache_decorator_cache_miss(self):
        """Test async cache decorator with cache miss."""
        config = CacheConfiguration()
        mock_cache = AsyncMock()
        mock_cache.get.return_value = None
        mock_cache.set.return_value = True

        decorator = AsyncCacheDecorator(config, mock_cache)

        @decorator
        async def test_func(x, y):
            return {"result": f"computed_{x}_{y}"}

        result = await test_func(1, 2)

        assert result == {"result": "computed_1_2"}
        mock_cache.get.assert_called_once()
        mock_cache.set.assert_called_once()
        assert decorator.stats.hits == 0
        assert decorator.stats.misses == 1

    @pytest.mark.asyncio
    async def test_async_cache_decorator_disabled(self):
        """Test async cache decorator when disabled."""
        config = CacheConfiguration(enabled=False)
        mock_cache = AsyncMock()

        decorator = AsyncCacheDecorator(config, mock_cache)

        @decorator
        async def test_func(x, y):
            return {"result": f"computed_{x}_{y}"}

        result = await test_func(1, 2)

        assert result == {"result": "computed_1_2"}
        mock_cache.get.assert_not_called()
        mock_cache.set.assert_not_called()

    @pytest.mark.asyncio
    async def test_async_cache_decorator_with_custom_key_generator(self):
        """Test async cache decorator with custom key generator."""
        config = CacheConfiguration()
        mock_cache = AsyncMock()
        mock_cache.get.return_value = None
        mock_cache.set.return_value = True

        decorator = AsyncCacheDecorator(config, mock_cache)

        def custom_key_generator(func_name, args, kwargs):
            return f"custom:{func_name}:{args[0]}"

        @decorator(key_generator=custom_key_generator)
        async def test_func(x, y):
            return {"result": f"computed_{x}_{y}"}

        await test_func(1, 2)

        # Check that custom key was used
        mock_cache.set.assert_called_once()
        call_args = mock_cache.set.call_args
        assert call_args[0][0] == "custom:test_func:1"

    @pytest.mark.asyncio
    async def test_async_cache_decorator_error_handling(self):
        """Test async cache decorator error handling."""
        config = CacheConfiguration()
        mock_cache = AsyncMock()
        mock_cache.get.side_effect = Exception("Cache error")

        decorator = AsyncCacheDecorator(config, mock_cache)

        @decorator
        async def test_func(x, y):
            return {"result": f"computed_{x}_{y}"}

        # Should still work even if cache fails
        result = await test_func(1, 2)
        assert result == {"result": "computed_1_2"}

    @pytest.mark.asyncio
    async def test_async_cache_decorator_with_ttl(self):
        """Test async cache decorator with TTL."""
        config = CacheConfiguration(ttl=3600)
        mock_cache = AsyncMock()
        mock_cache.get.return_value = None
        mock_cache.set.return_value = True

        decorator = AsyncCacheDecorator(config, mock_cache)

        @decorator
        async def test_func(x, y):
            return {"result": f"computed_{x}_{y}"}

        await test_func(1, 2)

        # Check that TTL was passed to cache.set
        mock_cache.set.assert_called_once()
        call_args = mock_cache.set.call_args
        assert call_args[1]["ttl"] == 3600


class TestSyncCacheDecorator:
    """Test SyncCacheDecorator class."""

    def test_sync_cache_decorator_initialization(self):
        """Test SyncCacheDecorator initialization."""
        config = CacheConfiguration()
        mock_cache = MagicMock()

        decorator = SyncCacheDecorator(config, mock_cache)

        assert decorator.config == config
        assert decorator.cache == mock_cache

    def test_sync_cache_decorator_cache_hit(self):
        """Test sync cache decorator with cache hit."""
        config = CacheConfiguration()
        mock_cache = MagicMock()
        mock_cache.get.return_value = '{"result": "cached_value"}'

        decorator = SyncCacheDecorator(config, mock_cache)

        @decorator
        def test_func(x, y):
            return {"result": f"computed_{x}_{y}"}

        result = test_func(1, 2)

        assert result == {"result": "cached_value"}
        mock_cache.get.assert_called_once()
        assert decorator.stats.hits == 1
        assert decorator.stats.misses == 0

    def test_sync_cache_decorator_cache_miss(self):
        """Test sync cache decorator with cache miss."""
        config = CacheConfiguration()
        mock_cache = MagicMock()
        mock_cache.get.return_value = None
        mock_cache.set.return_value = True

        decorator = SyncCacheDecorator(config, mock_cache)

        @decorator
        def test_func(x, y):
            return {"result": f"computed_{x}_{y}"}

        result = test_func(1, 2)

        assert result == {"result": "computed_1_2"}
        mock_cache.get.assert_called_once()
        mock_cache.set.assert_called_once()
        assert decorator.stats.hits == 0
        assert decorator.stats.misses == 1

    def test_sync_cache_decorator_disabled(self):
        """Test sync cache decorator when disabled."""
        config = CacheConfiguration(enabled=False)
        mock_cache = MagicMock()

        decorator = SyncCacheDecorator(config, mock_cache)

        @decorator
        def test_func(x, y):
            return {"result": f"computed_{x}_{y}"}

        result = test_func(1, 2)

        assert result == {"result": "computed_1_2"}
        mock_cache.get.assert_not_called()
        mock_cache.set.assert_not_called()

    def test_sync_cache_decorator_error_handling(self):
        """Test sync cache decorator error handling."""
        config = CacheConfiguration()
        mock_cache = MagicMock()
        mock_cache.get.side_effect = Exception("Cache error")

        decorator = SyncCacheDecorator(config, mock_cache)

        @decorator
        def test_func(x, y):
            return {"result": f"computed_{x}_{y}"}

        # Should still work even if cache fails
        result = test_func(1, 2)
        assert result == {"result": "computed_1_2"}


class TestCacheInvalidator:
    """Test CacheInvalidator class."""

    def test_cache_invalidator_initialization(self):
        """Test CacheInvalidator initialization."""
        mock_cache = MagicMock()
        invalidator = CacheInvalidator(mock_cache)

        assert invalidator.cache == mock_cache
        assert invalidator.invalidation_count == 0

    def test_invalidate_by_key(self):
        """Test invalidating cache by key."""
        mock_cache = MagicMock()
        mock_cache.delete.return_value = True

        invalidator = CacheInvalidator(mock_cache)
        result = invalidator.invalidate_by_key("test_key")

        assert result is True
        mock_cache.delete.assert_called_once_with("test_key")
        assert invalidator.invalidation_count == 1

    def test_invalidate_by_pattern(self):
        """Test invalidating cache by pattern."""
        mock_cache = MagicMock()
        mock_cache.keys.return_value = ["test:1", "test:2", "other:1"]
        mock_cache.delete.return_value = True

        invalidator = CacheInvalidator(mock_cache)
        result = invalidator.invalidate_by_pattern("test:*")

        assert result == 2  # Should invalidate 2 keys
        mock_cache.keys.assert_called_once_with("test:*")
        assert mock_cache.delete.call_count == 2

    def test_invalidate_by_tags(self):
        """Test invalidating cache by tags."""
        mock_cache = MagicMock()
        mock_cache.get_keys_by_tags.return_value = ["tagged:1", "tagged:2"]
        mock_cache.delete.return_value = True

        invalidator = CacheInvalidator(mock_cache)
        result = invalidator.invalidate_by_tags(["tag1", "tag2"])

        assert result == 2
        mock_cache.get_keys_by_tags.assert_called_once_with(["tag1", "tag2"])
        assert mock_cache.delete.call_count == 2

    def test_invalidate_all(self):
        """Test invalidating all cache entries."""
        mock_cache = MagicMock()
        mock_cache.clear.return_value = True

        invalidator = CacheInvalidator(mock_cache)
        result = invalidator.invalidate_all()

        assert result is True
        mock_cache.clear.assert_called_once()
        assert invalidator.invalidation_count == 1

    def test_schedule_invalidation(self):
        """Test scheduling cache invalidation."""
        mock_cache = MagicMock()
        invalidator = CacheInvalidator(mock_cache)

        future_time = datetime.utcnow() + timedelta(seconds=10)
        invalidator.schedule_invalidation("test_key", future_time)

        assert len(invalidator.scheduled_invalidations) == 1
        assert invalidator.scheduled_invalidations[0]["key"] == "test_key"
        assert invalidator.scheduled_invalidations[0]["time"] == future_time

    def test_process_scheduled_invalidations(self):
        """Test processing scheduled invalidations."""
        mock_cache = MagicMock()
        mock_cache.delete.return_value = True

        invalidator = CacheInvalidator(mock_cache)

        # Add past invalidation
        past_time = datetime.utcnow() - timedelta(seconds=10)
        invalidator.schedule_invalidation("past_key", past_time)

        # Add future invalidation
        future_time = datetime.utcnow() + timedelta(seconds=10)
        invalidator.schedule_invalidation("future_key", future_time)

        processed = invalidator.process_scheduled_invalidations()

        assert processed == 1
        mock_cache.delete.assert_called_once_with("past_key")
        assert len(invalidator.scheduled_invalidations) == 1  # Future one remains

    def test_get_invalidation_stats(self):
        """Test getting invalidation statistics."""
        mock_cache = MagicMock()
        invalidator = CacheInvalidator(mock_cache)

        invalidator.invalidation_count = 5
        invalidator.schedule_invalidation("key1", datetime.utcnow() + timedelta(seconds=10))
        invalidator.schedule_invalidation("key2", datetime.utcnow() + timedelta(seconds=20))

        stats = invalidator.get_invalidation_stats()

        assert stats["total_invalidations"] == 5
        assert stats["scheduled_invalidations"] == 2


class TestCacheHelperFunctions:
    """Test cache helper functions."""

    @pytest.mark.asyncio
    async def test_cache_async_decorator(self):
        """Test cache_async decorator function."""
        mock_cache = AsyncMock()
        mock_cache.get.return_value = None
        mock_cache.set.return_value = True

        config = CacheConfiguration()

        @cache_async(config, mock_cache)
        async def test_func(x, y):
            return f"result_{x}_{y}"

        result = await test_func(1, 2)

        assert result == "result_1_2"
        mock_cache.get.assert_called_once()
        mock_cache.set.assert_called_once()

    def test_cache_sync_decorator(self):
        """Test cache_sync decorator function."""
        mock_cache = MagicMock()
        mock_cache.get.return_value = None
        mock_cache.set.return_value = True

        config = CacheConfiguration()

        @cache_sync(config, mock_cache)
        def test_func(x, y):
            return f"result_{x}_{y}"

        result = test_func(1, 2)

        assert result == "result_1_2"
        mock_cache.get.assert_called_once()
        mock_cache.set.assert_called_once()

    def test_invalidate_cache_decorator(self):
        """Test invalidate_cache decorator function."""
        mock_cache = MagicMock()
        mock_cache.delete.return_value = True

        @invalidate_cache(mock_cache, "test_key")
        def test_func():
            return "result"

        result = test_func()

        assert result == "result"
        mock_cache.delete.assert_called_once_with("test_key")

    def test_cache_key_generator(self):
        """Test cache_key_generator function."""
        key = cache_key_generator("test_func", (1, 2), {"key": "value"})

        assert isinstance(key, str)
        assert "test_func" in key
        assert len(key) > 0

    def test_cache_key_generator_with_prefix(self):
        """Test cache_key_generator with prefix."""
        key = cache_key_generator("test_func", (1, 2), {"key": "value"}, prefix="myapp")

        assert key.startswith("myapp:")
        assert "test_func" in key


class TestCacheStats:
    """Test CacheStats class."""

    def test_cache_stats_initialization(self):
        """Test CacheStats initialization."""
        stats = CacheStats()

        assert stats.hits == 0
        assert stats.misses == 0
        assert stats.total_requests == 0
        assert stats.hit_rate == 0.0

    def test_cache_stats_with_values(self):
        """Test CacheStats with values."""
        stats = CacheStats(hits=10, misses=5)

        assert stats.hits == 10
        assert stats.misses == 5
        assert stats.total_requests == 15
        assert stats.hit_rate == 0.667

    def test_cache_stats_zero_requests(self):
        """Test CacheStats with zero requests."""
        stats = CacheStats(hits=0, misses=0)

        assert stats.total_requests == 0
        assert stats.hit_rate == 0.0

    def test_cache_stats_to_dict(self):
        """Test CacheStats to_dict method."""
        stats = CacheStats(hits=10, misses=5)

        result = stats.to_dict()

        assert result["hits"] == 10
        assert result["misses"] == 5
        assert result["total_requests"] == 15
        assert result["hit_rate"] == 0.667

    def test_cache_stats_from_dict(self):
        """Test CacheStats from_dict method."""
        data = {
            "hits": 10,
            "misses": 5,
            "total_requests": 15,
            "hit_rate": 0.667
        }

        stats = CacheStats.from_dict(data)

        assert stats.hits == 10
        assert stats.misses == 5
        assert stats.total_requests == 15
        assert stats.hit_rate == 0.667


class TestCacheConfiguration:
    """Test CacheConfiguration class."""

    def test_cache_configuration_defaults(self):
        """Test CacheConfiguration default values."""
        config = CacheConfiguration()

        assert config.enabled is True
        assert config.ttl == 3600
        assert config.key_prefix == "pynomaly"
        assert config.serialization_format == "json"
        assert config.compression_enabled is False
        assert config.cache_none is False
        assert config.max_key_length == 1000
        assert config.max_value_size == 1048576  # 1MB

    def test_cache_configuration_custom_values(self):
        """Test CacheConfiguration with custom values."""
        config = CacheConfiguration(
            enabled=False,
            ttl=7200,
            key_prefix="custom",
            serialization_format="pickle",
            compression_enabled=True,
            cache_none=True,
            max_key_length=500,
            max_value_size=524288  # 512KB
        )

        assert config.enabled is False
        assert config.ttl == 7200
        assert config.key_prefix == "custom"
        assert config.serialization_format == "pickle"
        assert config.compression_enabled is True
        assert config.cache_none is True
        assert config.max_key_length == 500
        assert config.max_value_size == 524288

    def test_cache_configuration_validation(self):
        """Test CacheConfiguration validation."""
        # Test invalid TTL
        with pytest.raises(ValueError, match="TTL must be positive"):
            CacheConfiguration(ttl=0)

        # Test invalid serialization format
        with pytest.raises(ValueError, match="Serialization format must be"):
            CacheConfiguration(serialization_format="invalid")

        # Test invalid max_key_length
        with pytest.raises(ValueError, match="Max key length must be positive"):
            CacheConfiguration(max_key_length=0)

        # Test invalid max_value_size
        with pytest.raises(ValueError, match="Max value size must be positive"):
            CacheConfiguration(max_value_size=0)
