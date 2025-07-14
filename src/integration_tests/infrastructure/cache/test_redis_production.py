"""Tests for enhanced Redis production caching implementation (Issue #99)."""

import asyncio
from datetime import datetime, timedelta
from unittest.mock import MagicMock, patch

import pytest

from pynomaly.infrastructure.cache.redis_production import (
    CacheMetrics,
    CacheWarmingConfig,
    ProductionRedisCache,
)


class TestProductionRedisCache:
    """Test production Redis cache functionality."""

    @pytest.fixture
    def mock_settings(self):
        """Mock application settings."""
        settings = MagicMock()
        settings.redis_url = "redis://localhost:6379/0"
        settings.cache_namespace = "test"
        return settings

    @pytest.fixture
    def mock_redis(self):
        """Mock Redis client."""
        redis_mock = MagicMock()
        redis_mock.ping.return_value = True
        redis_mock.get.return_value = None
        redis_mock.set.return_value = True
        redis_mock.setex.return_value = True
        redis_mock.delete.return_value = 1
        redis_mock.keys.return_value = []
        redis_mock.smembers.return_value = set()
        redis_mock.info.return_value = {
            "redis_version": "7.0.0",
            "used_memory": 1024000,
            "connected_clients": 5,
            "total_commands_processed": 1000,
            "instantaneous_ops_per_sec": 10,
            "keyspace_hits": 800,
            "keyspace_misses": 200,
            "evicted_keys": 0,
        }
        return redis_mock

    @pytest.fixture
    def production_cache(self, mock_settings, mock_redis):
        """Create production cache instance with mocked Redis."""
        with patch(
            "pynomaly.infrastructure.cache.redis_production.redis.Redis",
            return_value=mock_redis,
        ):
            cache = ProductionRedisCache(
                settings=mock_settings,
                enable_monitoring=True,
                enable_cache_warming=True,
                enable_circuit_breaker=True,
            )
            cache.redis = mock_redis
            return cache

    async def test_cache_initialization(self, mock_settings):
        """Test cache initialization with different configurations."""
        # Test standalone mode
        with patch(
            "pynomaly.infrastructure.cache.redis_production.redis.Redis"
        ) as mock_redis_class:
            mock_redis = MagicMock()
            mock_redis.ping.return_value = True
            mock_redis_class.return_value = mock_redis

            cache = ProductionRedisCache(mock_settings)
            assert cache.enable_monitoring is True
            assert cache.enable_cache_warming is True
            assert cache.enable_circuit_breaker is True
            assert not cache.circuit_breaker_open

    async def test_sentinel_initialization(self, mock_settings):
        """Test Redis Sentinel initialization."""
        sentinel_hosts = ["sentinel1:26379", "sentinel2:26379"]

        with patch(
            "pynomaly.infrastructure.cache.redis_production.redis.sentinel.Sentinel"
        ) as mock_sentinel_class:
            mock_sentinel = MagicMock()
            mock_master = MagicMock()
            mock_master.ping.return_value = True
            mock_sentinel.master_for.return_value = mock_master
            mock_sentinel_class.return_value = mock_sentinel

            cache = ProductionRedisCache(
                settings=mock_settings, sentinel_hosts=sentinel_hosts
            )

            assert cache.sentinel_hosts == sentinel_hosts
            mock_sentinel_class.assert_called_once()

    async def test_get_operation(self, production_cache, mock_redis):
        """Test cache get operation with circuit breaker."""
        # Test successful get
        mock_redis.get.return_value = b'{"test": "value"}'

        result = await production_cache.get("api:test_key")
        assert result == {"test": "value"}
        mock_redis.get.assert_called_with("test:api:test_key")

    async def test_get_operation_miss(self, production_cache, mock_redis):
        """Test cache miss scenario."""
        mock_redis.get.return_value = None

        result = await production_cache.get("missing_key", default="default_value")
        assert result == "default_value"
        assert production_cache.metrics.misses == 1

    async def test_set_operation(self, production_cache, mock_redis):
        """Test cache set operation."""
        mock_redis.setex.return_value = True

        result = await production_cache.set(
            "api:test_key", {"test": "value"}, ttl=3600, tags={"api", "test"}
        )

        assert result is True
        mock_redis.setex.assert_called_once()

    async def test_delete_operation(self, production_cache, mock_redis):
        """Test cache delete operation."""
        mock_redis.delete.return_value = 1

        result = await production_cache.delete("test_key")
        assert result is True
        mock_redis.delete.assert_called_with("test:test_key")

    async def test_tag_invalidation(self, production_cache, mock_redis):
        """Test tag-based cache invalidation."""
        mock_redis.smembers.return_value = {b"key1", b"key2", b"key3"}
        mock_redis.pipeline.return_value.__enter__.return_value.execute.return_value = [
            1,
            1,
            1,
            1,
        ]

        result = await production_cache.invalidate_by_tag("test_tag")
        assert result == 3

    async def test_circuit_breaker_open(self, production_cache):
        """Test circuit breaker functionality."""
        # Simulate failures to open circuit breaker
        production_cache.circuit_breaker_failures = 5
        production_cache._open_circuit_breaker()

        assert production_cache.circuit_breaker_open is True

        # Test that operations fail when circuit breaker is open
        with pytest.raises(Exception):
            await production_cache.get("test_key")

    async def test_circuit_breaker_timeout(self, production_cache):
        """Test circuit breaker timeout and recovery."""
        # Open circuit breaker
        production_cache._open_circuit_breaker()
        assert production_cache.circuit_breaker_open is True

        # Simulate timeout passage
        production_cache.circuit_breaker_last_failure = datetime.utcnow() - timedelta(
            seconds=120
        )

        # Circuit breaker should close
        is_open = production_cache._is_circuit_breaker_open()
        assert is_open is False

    async def test_cache_warming(self, production_cache, mock_redis):
        """Test cache warming functionality."""
        warming_data = {"key1": "value1", "key2": "value2", "key3": "value3"}

        mock_redis.pipeline.return_value.__enter__.return_value.execute.return_value = [
            True,
            True,
            True,
        ]

        await production_cache.warm_cache(warming_data)
        assert production_cache.metrics.cache_warming_time > 0

    async def test_serialization_json(self, production_cache):
        """Test JSON serialization for API keys."""
        data = {"test": "value", "number": 42}

        # Test JSON serialization (for API keys)
        serialized = production_cache._serialize(data, "api:test")
        deserialized = production_cache._deserialize(serialized, "api:test")

        assert deserialized == data

    async def test_serialization_pickle(self, production_cache):
        """Test pickle serialization for complex objects."""
        data = {"complex": [1, 2, 3], "nested": {"key": "value"}}

        # Test pickle serialization (for non-API keys)
        serialized = production_cache._serialize(data, "model:test")
        deserialized = production_cache._deserialize(serialized, "model:test")

        assert deserialized == data

    async def test_health_check(self, production_cache, mock_redis):
        """Test Redis health check functionality."""
        mock_redis.info.return_value = {
            "used_memory": 1024000,
            "maxmemory": 2048000,
            "used_memory_human": "1M",
        }

        health_status = await production_cache.health_check()

        assert health_status["status"] == "healthy"
        assert "connectivity" in health_status["checks"]
        assert "memory" in health_status["checks"]
        assert "circuit_breaker" in health_status["checks"]

    async def test_health_check_failure(self, production_cache, mock_redis):
        """Test health check failure scenario."""
        mock_redis.ping.side_effect = Exception("Connection failed")

        health_status = await production_cache.health_check()

        assert health_status["status"] == "unhealthy"
        assert "error" in health_status

    async def test_cache_statistics(self, production_cache, mock_redis):
        """Test cache statistics collection."""
        # Setup mock metrics
        production_cache.metrics.hits = 800
        production_cache.metrics.misses = 200
        production_cache.metrics.evictions = 5
        production_cache.metrics.avg_response_time = 0.001

        stats = await production_cache.get_cache_statistics()

        assert "cache_metrics" in stats
        assert "redis_info" in stats
        assert "circuit_breaker" in stats
        assert "configuration" in stats

        assert stats["cache_metrics"]["hits"] == 800
        assert stats["cache_metrics"]["misses"] == 200
        assert stats["cache_metrics"]["hit_rate"] == 0.8

    async def test_namespaced_keys(self, production_cache):
        """Test key namespacing functionality."""
        key = "test_key"
        namespaced = production_cache._get_namespaced_key(key)
        assert namespaced == "test:test_key"

    async def test_concurrent_operations(self, production_cache, mock_redis):
        """Test concurrent cache operations."""
        mock_redis.get.return_value = b'{"value": "test"}'
        mock_redis.set.return_value = True

        # Create multiple concurrent operations
        tasks = []
        for i in range(10):
            tasks.append(production_cache.get(f"key_{i}"))
            tasks.append(production_cache.set(f"key_{i}", f"value_{i}"))

        results = await asyncio.gather(*tasks, return_exceptions=True)

        # Check that no exceptions occurred
        exceptions = [r for r in results if isinstance(r, Exception)]
        assert len(exceptions) == 0

    async def test_memory_optimization(self, production_cache):
        """Test memory optimization features."""
        # Test that metrics are reasonable
        assert production_cache.metrics.memory_usage >= 0
        assert production_cache.metrics.connection_count >= 0

    async def test_error_handling(self, production_cache, mock_redis):
        """Test error handling in cache operations."""
        # Simulate Redis error
        mock_redis.get.side_effect = Exception("Redis error")

        # Should not raise exception, should return default
        result = await production_cache.get("test_key", default="fallback")
        assert result == "fallback"

        # Should increment failure count
        assert production_cache.metrics.misses > 0

    async def test_cleanup(self, production_cache):
        """Test cache cleanup and resource management."""
        # Add some warming tasks
        production_cache._warming_tasks.add(asyncio.create_task(asyncio.sleep(0.1)))

        # Test cleanup
        await production_cache.close()

        # Verify tasks are cleaned up
        assert len([t for t in production_cache._warming_tasks if not t.done()]) == 0


class TestCacheMetrics:
    """Test cache metrics functionality."""

    def test_cache_metrics_initialization(self):
        """Test metrics initialization."""
        metrics = CacheMetrics()
        assert metrics.hits == 0
        assert metrics.misses == 0
        assert metrics.evictions == 0
        assert metrics.memory_usage == 0
        assert metrics.connection_count == 0
        assert metrics.avg_response_time == 0.0

    def test_cache_metrics_update(self):
        """Test metrics updates."""
        metrics = CacheMetrics()
        metrics.hits = 100
        metrics.misses = 20
        metrics.avg_response_time = 0.005

        assert metrics.hits == 100
        assert metrics.misses == 20
        assert metrics.avg_response_time == 0.005


class TestCacheWarmingConfig:
    """Test cache warming configuration."""

    def test_warming_config_defaults(self):
        """Test default warming configuration."""
        config = CacheWarmingConfig()
        assert config.enabled is True
        assert config.warmup_on_startup is True
        assert config.background_warming is True
        assert config.warmup_batch_size == 100
        assert config.warmup_delay_seconds == 0.1

    def test_warming_config_customization(self):
        """Test custom warming configuration."""
        config = CacheWarmingConfig(
            enabled=False,
            warmup_batch_size=50,
            warmup_delay_seconds=0.05,
            critical_keys=["key1", "key2"],
        )

        assert config.enabled is False
        assert config.warmup_batch_size == 50
        assert config.warmup_delay_seconds == 0.05
        assert config.critical_keys == ["key1", "key2"]


@pytest.mark.integration
class TestRedisIntegration:
    """Integration tests for Redis cache (requires running Redis)."""

    @pytest.fixture
    def real_settings(self):
        """Real settings for integration tests."""
        settings = MagicMock()
        settings.redis_url = "redis://localhost:6379/15"  # Use test database
        settings.cache_namespace = "test_integration"
        return settings

    @pytest.mark.skipif(
        condition=True,  # Skip by default, enable for real Redis testing
        reason="Requires running Redis instance",
    )
    async def test_real_redis_operations(self, real_settings):
        """Test operations against real Redis instance."""
        cache = ProductionRedisCache(
            settings=real_settings, enable_monitoring=True, enable_cache_warming=True
        )

        try:
            # Test basic operations
            await cache.set("test_key", {"value": "test"}, ttl=60)
            result = await cache.get("test_key")
            assert result == {"value": "test"}

            # Test deletion
            deleted = await cache.delete("test_key")
            assert deleted is True

            # Test cache miss
            result = await cache.get("test_key")
            assert result is None

        finally:
            await cache.close()
