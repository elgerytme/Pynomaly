"""
Branch coverage tests for caching infrastructure.
Focuses on edge cases, error paths, and conditional logic branches in caching services.
"""

import asyncio
import json
import pickle
import time
from unittest.mock import Mock, patch, AsyncMock, MagicMock
from typing import Any, Dict, List

import pytest

try:
    import redis
    REDIS_AVAILABLE = True
except ImportError:
    REDIS_AVAILABLE = False


@pytest.fixture
def mock_redis():
    """Create mock Redis client."""
    mock_client = Mock()
    mock_client.ping.return_value = True
    mock_client.get.return_value = None
    mock_client.set.return_value = True
    mock_client.delete.return_value = 1
    mock_client.exists.return_value = False
    mock_client.ttl.return_value = -1
    mock_client.flushall.return_value = True
    return mock_client


@pytest.fixture
def cache_service(mock_redis):
    """Create cache service with mocked dependencies."""
    from pynomaly.infrastructure.caching.advanced_cache_service import AdvancedCacheService
    
    config = {
        'redis_url': 'redis://localhost:6379',
        'memory_cache_size': 1000,
        'default_ttl': 3600,
        'compression_enabled': True,
        'serialization_format': 'pickle',
    }
    
    service = AdvancedCacheService(config)
    service.redis_client = mock_redis
    return service


class TestAdvancedCacheServiceBranchCoverage:
    """Test AdvancedCacheService with focus on branch coverage and edge cases."""

    def test_cache_initialization_edge_cases(self):
        """Test cache service initialization with various configurations."""
        from pynomaly.infrastructure.caching.advanced_cache_service import AdvancedCacheService
        
        # Test with minimal config
        minimal_config = {'redis_url': 'redis://localhost:6379'}
        service = AdvancedCacheService(minimal_config)
        assert service.config['memory_cache_size'] == 1000  # Default
        
        # Test with custom config
        custom_config = {
            'redis_url': 'redis://localhost:6379',
            'memory_cache_size': 5000,
            'default_ttl': 7200,
            'compression_enabled': False,
            'serialization_format': 'json',
        }
        service = AdvancedCacheService(custom_config)
        assert service.config['memory_cache_size'] == 5000
        assert service.config['compression_enabled'] is False

    def test_redis_connection_failure_fallback(self, cache_service):
        """Test fallback behavior when Redis is unavailable."""
        # Mock Redis connection failure
        cache_service.redis_client.ping.side_effect = Exception("Connection failed")
        
        # Should fallback to memory cache
        with patch.object(cache_service, '_is_redis_available') as mock_check:
            mock_check.return_value = False
            result = cache_service.set("key1", "value1")
            # Should still succeed using memory cache
            assert result is True

    def test_serialization_format_fallbacks(self, cache_service):
        """Test serialization format handling and fallbacks."""
        test_data = {"complex": "data", "with": ["nested", "structures"]}
        
        # Test pickle serialization (default)
        cache_service.config['serialization_format'] = 'pickle'
        serialized = cache_service._serialize_data(test_data)
        deserialized = cache_service._deserialize_data(serialized)
        assert deserialized == test_data
        
        # Test JSON serialization
        cache_service.config['serialization_format'] = 'json'
        serialized = cache_service._serialize_data(test_data)
        deserialized = cache_service._deserialize_data(serialized)
        assert deserialized == test_data
        
        # Test unsupported format (should fallback to pickle)
        cache_service.config['serialization_format'] = 'unsupported'
        serialized = cache_service._serialize_data(test_data)
        assert serialized is not None  # Should fallback successfully

    def test_compression_edge_cases(self, cache_service):
        """Test compression handling with edge cases."""
        # Test with compression enabled
        cache_service.config['compression_enabled'] = True
        
        # Small data (should not compress)
        small_data = "small"
        compressed = cache_service._compress_data(small_data.encode())
        assert len(compressed) <= len(small_data.encode()) + 50  # Allow overhead
        
        # Large data (should compress)
        large_data = "x" * 10000
        compressed = cache_service._compress_data(large_data.encode())
        assert len(compressed) < len(large_data.encode())
        
        # Test decompression
        decompressed = cache_service._decompress_data(compressed)
        assert decompressed.decode() == large_data
        
        # Test compression disabled
        cache_service.config['compression_enabled'] = False
        not_compressed = cache_service._compress_data(large_data.encode())
        assert not_compressed == large_data.encode()

    def test_memory_cache_eviction_policies(self, cache_service):
        """Test memory cache eviction policies."""
        # Set small cache size to trigger eviction
        cache_service.config['memory_cache_size'] = 3
        cache_service.memory_cache.clear()
        
        # Fill cache to capacity
        cache_service.memory_cache["key1"] = {"data": "value1", "timestamp": time.time()}
        cache_service.memory_cache["key2"] = {"data": "value2", "timestamp": time.time()}
        cache_service.memory_cache["key3"] = {"data": "value3", "timestamp": time.time()}
        
        # Add one more item (should trigger eviction)
        cache_service._store_in_memory_cache("key4", "value4")
        
        # Should have evicted oldest item
        assert len(cache_service.memory_cache) <= 3
        assert "key4" in cache_service.memory_cache

    def test_ttl_handling_edge_cases(self, cache_service, mock_redis):
        """Test TTL (Time To Live) handling edge cases."""
        # Test setting with custom TTL
        cache_service.set("key1", "value1", ttl=300)
        mock_redis.setex.assert_called()
        
        # Test setting with no TTL (uses default)
        cache_service.set("key2", "value2")
        # Should use default TTL
        
        # Test TTL expiry check
        mock_redis.ttl.return_value = 0  # Expired
        result = cache_service.get("key1")
        # Should return None for expired key
        
        # Test negative TTL (no expiry)
        mock_redis.ttl.return_value = -1
        mock_redis.get.return_value = b"serialized_value"
        result = cache_service.get("key1")
        # Should return value for non-expiring key

    def test_batch_operations_error_handling(self, cache_service, mock_redis):
        """Test batch operations with error scenarios."""
        # Test batch get with some Redis failures
        keys = ["key1", "key2", "key3"]
        mock_redis.mget.side_effect = Exception("Redis error")
        
        # Should fallback to individual gets or memory cache
        with patch.object(cache_service, 'get') as mock_get:
            mock_get.side_effect = [None, "value2", None]
            results = cache_service.batch_get(keys)
            assert len(results) == 3
            assert results["key2"] == "value2"

    def test_cache_warming_edge_cases(self, cache_service):
        """Test cache warming with various edge cases."""
        # Mock data provider
        def mock_data_provider(keys):
            return {key: f"value_{key}" for key in keys if not key.startswith("fail")}
        
        # Test warming with some failures
        keys_to_warm = ["key1", "key2", "fail_key3"]
        
        with patch.object(cache_service, 'set') as mock_set:
            cache_service.warm_cache(keys_to_warm, mock_data_provider)
            # Should have called set for successful keys only
            assert mock_set.call_count == 2

    def test_memory_pressure_handling(self, cache_service):
        """Test handling of memory pressure situations."""
        # Mock memory usage check
        with patch('psutil.virtual_memory') as mock_memory:
            # High memory usage scenario
            mock_memory.return_value.percent = 95
            
            # Should trigger cache cleanup
            with patch.object(cache_service, '_cleanup_memory_cache') as mock_cleanup:
                cache_service._check_memory_pressure()
                mock_cleanup.assert_called_once()
            
            # Low memory usage scenario
            mock_memory.return_value.percent = 50
            
            # Should not trigger cleanup
            with patch.object(cache_service, '_cleanup_memory_cache') as mock_cleanup:
                cache_service._check_memory_pressure()
                mock_cleanup.assert_not_called()

    def test_cache_statistics_edge_cases(self, cache_service):
        """Test cache statistics collection with edge cases."""
        # Reset statistics
        cache_service._reset_statistics()
        
        # Test hit/miss tracking
        cache_service._record_cache_hit("memory")
        cache_service._record_cache_hit("redis")
        cache_service._record_cache_miss()
        
        stats = cache_service.get_statistics()
        assert stats["hits"] == 2
        assert stats["misses"] == 1
        assert stats["hit_rate"] == 2/3
        
        # Test with no operations (avoid division by zero)
        cache_service._reset_statistics()
        stats = cache_service.get_statistics()
        assert stats["hit_rate"] == 0.0

    def test_concurrent_access_scenarios(self, cache_service):
        """Test concurrent access scenarios."""
        import threading
        
        # Test concurrent sets
        def concurrent_set(key_suffix):
            for i in range(10):
                cache_service.set(f"concurrent_key_{key_suffix}_{i}", f"value_{i}")
        
        threads = []
        for i in range(5):
            thread = threading.Thread(target=concurrent_set, args=(i,))
            threads.append(thread)
            thread.start()
        
        for thread in threads:
            thread.join()
        
        # Should have handled concurrent access without errors
        assert len(cache_service.memory_cache) > 0

    def test_cache_key_validation_edge_cases(self, cache_service):
        """Test cache key validation with edge cases."""
        # Test normal keys
        assert cache_service._validate_key("normal_key") is True
        assert cache_service._validate_key("key123") is True
        
        # Test edge cases
        assert cache_service._validate_key("") is False
        assert cache_service._validate_key(None) is False
        assert cache_service._validate_key("a" * 251) is False  # Too long
        
        # Test keys with special characters
        assert cache_service._validate_key("key:with:colons") is True
        assert cache_service._validate_key("key with spaces") is False

    def test_serialization_error_handling(self, cache_service):
        """Test serialization error handling."""
        # Test unserializable object
        class UnserializableClass:
            def __init__(self):
                self.circular_ref = self
        
        unserializable = UnserializableClass()
        
        # Should handle serialization error gracefully
        result = cache_service.set("bad_key", unserializable)
        assert result is False  # Should fail gracefully

    def test_redis_cluster_fallback(self, cache_service):
        """Test Redis cluster vs single instance handling."""
        # Mock Redis cluster mode
        with patch.object(cache_service, '_is_redis_cluster') as mock_cluster:
            mock_cluster.return_value = True
            
            # Should use cluster-specific operations
            with patch.object(cache_service.redis_client, 'execute_command') as mock_exec:
                cache_service.flush_all()
                # Should use cluster-aware flush
                mock_exec.assert_called()

    def test_cache_layering_fallback_strategy(self, cache_service, mock_redis):
        """Test multi-level cache fallback strategies."""
        # Test L1 (memory) -> L2 (Redis) -> L3 (fallback) strategy
        
        # L1 miss, L2 hit
        cache_service.memory_cache.clear()
        mock_redis.get.return_value = cache_service._serialize_data("redis_value")
        
        result = cache_service.get("test_key")
        assert result == "redis_value"
        
        # L1 miss, L2 miss, fallback
        mock_redis.get.return_value = None
        
        def fallback_provider(key):
            return f"fallback_value_{key}"
        
        result = cache_service.get_with_fallback("test_key", fallback_provider)
        assert result == "fallback_value_test_key"

    @pytest.mark.asyncio
    async def test_async_cache_operations(self, cache_service):
        """Test asynchronous cache operations."""
        # Mock async Redis operations
        async_redis = AsyncMock()
        cache_service.async_redis_client = async_redis
        
        # Test async get
        async_redis.get.return_value = cache_service._serialize_data("async_value")
        result = await cache_service.async_get("async_key")
        assert result == "async_value"
        
        # Test async set
        async_redis.set.return_value = True
        result = await cache_service.async_set("async_key", "async_value")
        assert result is True
        
        # Test async error handling
        async_redis.get.side_effect = Exception("Async Redis error")
        result = await cache_service.async_get("error_key")
        # Should fallback to memory cache
        assert result is None  # Not in memory cache

    def test_cache_monitoring_and_alerts(self, cache_service):
        """Test cache monitoring and alerting functionality."""
        # Test performance monitoring
        with patch('time.time') as mock_time:
            mock_time.side_effect = [0, 0.1]  # 100ms operation
            
            cache_service._monitor_performance("get", "test_key")
            
            # Should record performance metrics
            assert "performance_metrics" in cache_service.__dict__

    def test_cache_configuration_reload(self, cache_service):
        """Test dynamic cache configuration reloading."""
        # Change configuration
        new_config = {
            'memory_cache_size': 2000,
            'compression_enabled': False,
            'default_ttl': 1800,
        }
        
        cache_service.reload_configuration(new_config)
        
        # Should update configuration
        assert cache_service.config['memory_cache_size'] == 2000
        assert cache_service.config['compression_enabled'] is False

    def test_cache_health_check_edge_cases(self, cache_service, mock_redis):
        """Test cache health check with various scenarios."""
        # Healthy scenario
        mock_redis.ping.return_value = True
        health = cache_service.health_check()
        assert health['redis']['status'] == 'healthy'
        
        # Redis unhealthy scenario
        mock_redis.ping.side_effect = Exception("Connection error")
        health = cache_service.health_check()
        assert health['redis']['status'] == 'unhealthy'
        
        # Memory cache health (always available)
        assert health['memory']['status'] == 'healthy'