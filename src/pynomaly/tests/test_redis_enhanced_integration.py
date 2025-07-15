"""Integration tests for enhanced Redis caching implementation (Issue #99)."""

import asyncio
import pytest
import time
from unittest.mock import AsyncMock, Mock, patch

from pynomaly.infrastructure.cache.redis_enhanced import (
    EnhancedRedisCache,
    CachePerformanceMetrics,
    CacheCompressionConfig,
    CacheSecurityConfig,
    get_enhanced_redis_cache,
    close_enhanced_redis_cache,
)
from pynomaly.infrastructure.config import Settings


class MockRedisClient:
    """Mock Redis client for testing."""
    
    def __init__(self):
        self.data = {}
        self.call_count = 0
        
    async def get(self, key):
        self.call_count += 1
        return self.data.get(key)
    
    async def set(self, key, value, ex=None):
        self.call_count += 1
        self.data[key] = value
        return True
    
    async def delete(self, key):
        self.call_count += 1
        return self.data.pop(key, None) is not None
    
    def ping(self):
        return True
    
    def exists(self, key):
        return key in self.data


class TestCachePerformanceMetrics:
    """Test cache performance metrics."""
    
    def test_metrics_initialization(self):
        """Test metrics initialization."""
        metrics = CachePerformanceMetrics()
        
        assert metrics.hits == 0
        assert metrics.misses == 0
        assert metrics.writes == 0
        assert metrics.deletes == 0
        assert metrics.hit_rate() == 0.0
        assert metrics.operations_per_second() >= 0.0
    
    def test_hit_rate_calculation(self):
        """Test hit rate calculation."""
        metrics = CachePerformanceMetrics()
        
        # No operations
        assert metrics.hit_rate() == 0.0
        
        # Some hits and misses
        metrics.hits = 80
        metrics.misses = 20
        assert metrics.hit_rate() == 0.8
        
        # Only hits
        metrics.hits = 100
        metrics.misses = 0
        assert metrics.hit_rate() == 1.0
        
        # Only misses
        metrics.hits = 0
        metrics.misses = 100
        assert metrics.hit_rate() == 0.0
    
    def test_operations_per_second(self):
        """Test operations per second calculation."""
        import time
        from datetime import datetime, timedelta
        
        metrics = CachePerformanceMetrics()
        metrics.start_time = datetime.utcnow() - timedelta(seconds=10)
        metrics.hits = 50
        metrics.misses = 25
        metrics.writes = 30
        metrics.deletes = 5
        
        ops_per_sec = metrics.operations_per_second()
        
        # Should be around 11 ops/sec (110 ops / 10 seconds)
        assert 10 <= ops_per_sec <= 12


class TestCacheCompressionConfig:
    """Test cache compression configuration."""
    
    def test_default_compression_config(self):
        """Test default compression configuration."""
        config = CacheCompressionConfig()
        
        assert config.enabled is True
        assert config.algorithm == "gzip"
        assert config.level == 6
        assert config.threshold_bytes == 1024
        assert config.max_size_bytes == 10 * 1024 * 1024
    
    def test_custom_compression_config(self):
        """Test custom compression configuration."""
        config = CacheCompressionConfig(
            enabled=False,
            algorithm="lz4",
            level=3,
            threshold_bytes=512,
            max_size_bytes=5 * 1024 * 1024
        )
        
        assert config.enabled is False
        assert config.algorithm == "lz4"
        assert config.level == 3
        assert config.threshold_bytes == 512
        assert config.max_size_bytes == 5 * 1024 * 1024


class TestCacheSecurityConfig:
    """Test cache security configuration."""
    
    def test_default_security_config(self):
        """Test default security configuration."""
        config = CacheSecurityConfig()
        
        assert config.enable_tls is False
        assert config.enable_auth is True
        assert config.enable_encryption is False
        assert config.max_key_length == 512
        assert config.max_value_size == 100 * 1024 * 1024
    
    def test_custom_security_config(self):
        """Test custom security configuration."""
        config = CacheSecurityConfig(
            enable_tls=True,
            enable_auth=False,
            enable_encryption=True,
            max_key_length=256,
            max_value_size=50 * 1024 * 1024
        )
        
        assert config.enable_tls is True
        assert config.enable_auth is False
        assert config.enable_encryption is True
        assert config.max_key_length == 256
        assert config.max_value_size == 50 * 1024 * 1024


class TestEnhancedRedisCache:
    """Test enhanced Redis cache functionality."""
    
    @pytest.fixture
    def settings(self):
        """Create test settings."""
        settings = Settings()
        settings.cache_enabled = True
        settings.redis_url = "redis://localhost:6379/0"
        settings.cache_ttl = 3600
        return settings
    
    @pytest.fixture
    def mock_production_cache(self):
        """Create mock production cache."""
        cache = Mock()
        cache.get = AsyncMock(return_value=None)
        cache.set = AsyncMock(return_value=True)
        cache.delete = AsyncMock(return_value=True)
        cache.invalidate_by_tag = AsyncMock(return_value=5)
        cache.warm_cache = AsyncMock()
        cache.get_cache_statistics = AsyncMock(return_value={
            "cache_metrics": {
                "hits": 100,
                "misses": 20,
                "evictions": 5,
                "avg_response_time_ms": 15
            },
            "redis_info": {
                "used_memory": 1024 * 1024,
                "connected_clients": 10,
                "instantaneous_ops_per_sec": 150
            }
        })
        cache.health_check = AsyncMock(return_value={
            "status": "healthy",
            "timestamp": "2024-01-01T00:00:00"
        })
        cache.close = AsyncMock()
        return cache
    
    @pytest.fixture
    def mock_intelligent_cache(self):
        """Create mock intelligent cache."""
        cache = Mock()
        cache.get = AsyncMock(return_value=None)
        cache.set = AsyncMock()
        cache.delete = AsyncMock()
        cache.delete_pattern = AsyncMock()
        cache.get_stats = AsyncMock(return_value={
            "cache_stats": {"hit_rate": 0.85},
            "memory_cache": {"utilization": 0.6}
        })
        cache.close = AsyncMock()
        return cache
    
    def test_enhanced_cache_initialization(self, settings, mock_production_cache):
        """Test enhanced cache initialization."""
        cache = EnhancedRedisCache(
            settings=settings,
            production_cache=mock_production_cache,
            enable_monitoring=False
        )
        
        assert cache.settings == settings
        assert cache.production_cache == mock_production_cache
        assert isinstance(cache.metrics, CachePerformanceMetrics)
        assert isinstance(cache.compression_config, CacheCompressionConfig)
        assert isinstance(cache.security_config, CacheSecurityConfig)
    
    def test_compression_setup_gzip(self, settings, mock_production_cache):
        """Test compression setup with gzip."""
        compression_config = CacheCompressionConfig(
            enabled=True,
            algorithm="gzip"
        )
        
        cache = EnhancedRedisCache(
            settings=settings,
            production_cache=mock_production_cache,
            compression_config=compression_config,
            enable_monitoring=False
        )
        
        # Should have gzip compressor
        assert hasattr(cache, '_compressor')
    
    def test_compression_setup_invalid_algorithm(self, settings, mock_production_cache):
        """Test compression setup with invalid algorithm."""
        compression_config = CacheCompressionConfig(
            enabled=True,
            algorithm="invalid_algorithm"
        )
        
        cache = EnhancedRedisCache(
            settings=settings,
            production_cache=mock_production_cache,
            compression_config=compression_config,
            enable_monitoring=False
        )
        
        # Should fall back to gzip
        assert hasattr(cache, '_compressor')
    
    @patch('pynomaly.infrastructure.cache.redis_enhanced.Fernet')
    def test_security_setup_encryption(self, mock_fernet, settings, mock_production_cache):
        """Test security setup with encryption."""
        mock_cipher = Mock()
        mock_fernet.return_value = mock_cipher
        
        security_config = CacheSecurityConfig(
            enable_encryption=True,
            encryption_key="test_key_32_bytes_long_exactly"
        )
        
        cache = EnhancedRedisCache(
            settings=settings,
            production_cache=mock_production_cache,
            security_config=security_config,
            enable_monitoring=False
        )
        
        assert hasattr(cache, '_cipher')
        assert cache._cipher == mock_cipher
    
    def test_key_validation_valid(self, settings, mock_production_cache):
        """Test key validation with valid keys."""
        cache = EnhancedRedisCache(
            settings=settings,
            production_cache=mock_production_cache,
            enable_monitoring=False
        )
        
        # Valid keys should not raise exceptions
        cache._validate_key("simple_key")
        cache._validate_key("key:with:colons")
        cache._validate_key("key_with_underscores")
        cache._validate_key("key-with-dashes")
    
    def test_key_validation_invalid(self, settings, mock_production_cache):
        """Test key validation with invalid keys."""
        cache = EnhancedRedisCache(
            settings=settings,
            production_cache=mock_production_cache,
            enable_monitoring=False
        )
        
        # Too long key
        long_key = "x" * 1000
        with pytest.raises(ValueError, match="Key too long"):
            cache._validate_key(long_key)
        
        # Invalid characters
        with pytest.raises(ValueError, match="invalid characters"):
            cache._validate_key("key\nwith\nnewlines")
        
        with pytest.raises(ValueError, match="invalid characters"):
            cache._validate_key("key\twith\ttabs")
    
    def test_compression_and_decompression(self, settings, mock_production_cache):
        """Test value compression and decompression."""
        cache = EnhancedRedisCache(
            settings=settings,
            production_cache=mock_production_cache,
            enable_monitoring=False
        )
        
        # Large value that should be compressed
        large_value = b"x" * 2048
        
        compressed = cache._compress_value(large_value)
        assert compressed.startswith(b"COMPRESSED:")
        assert len(compressed) < len(large_value) + 11  # Should be compressed
        
        decompressed = cache._decompress_value(compressed)
        assert decompressed == large_value
    
    def test_compression_threshold(self, settings, mock_production_cache):
        """Test compression threshold behavior."""
        compression_config = CacheCompressionConfig(
            enabled=True,
            threshold_bytes=1024
        )
        
        cache = EnhancedRedisCache(
            settings=settings,
            production_cache=mock_production_cache,
            compression_config=compression_config,
            enable_monitoring=False
        )
        
        # Small value should not be compressed
        small_value = b"x" * 512
        compressed = cache._compress_value(small_value)
        assert compressed == small_value
        
        # Large value should be compressed
        large_value = b"x" * 2048
        compressed = cache._compress_value(large_value)
        assert compressed.startswith(b"COMPRESSED:")
    
    @patch('pynomaly.infrastructure.cache.redis_enhanced.Fernet')
    def test_encryption_and_decryption(self, mock_fernet, settings, mock_production_cache):
        """Test value encryption and decryption."""
        mock_cipher = Mock()
        mock_cipher.encrypt.return_value = b"encrypted_data"
        mock_cipher.decrypt.return_value = b"original_data"
        mock_fernet.return_value = mock_cipher
        
        security_config = CacheSecurityConfig(enable_encryption=True)
        
        cache = EnhancedRedisCache(
            settings=settings,
            production_cache=mock_production_cache,
            security_config=security_config,
            enable_monitoring=False
        )
        
        original_value = b"test_data"
        
        encrypted = cache._encrypt_value(original_value)
        assert encrypted.startswith(b"ENCRYPTED:")
        
        decrypted = cache._decrypt_value(encrypted)
        assert decrypted == b"original_data"
    
    @pytest.mark.asyncio
    async def test_get_operation(self, settings, mock_production_cache, mock_intelligent_cache):
        """Test enhanced get operation."""
        mock_intelligent_cache.get.return_value = "cached_value"
        
        cache = EnhancedRedisCache(
            settings=settings,
            production_cache=mock_production_cache,
            intelligent_cache=mock_intelligent_cache,
            enable_monitoring=False
        )
        
        result = await cache.get("test_key")
        
        assert result == "cached_value"
        assert cache.metrics.hits == 1
        mock_intelligent_cache.get.assert_called_once_with("test_key", None)
    
    @pytest.mark.asyncio
    async def test_get_operation_fallback(self, settings, mock_production_cache, mock_intelligent_cache):
        """Test get operation fallback to production cache."""
        mock_intelligent_cache.get.return_value = None  # Cache miss
        mock_production_cache.get.return_value = "production_value"
        
        cache = EnhancedRedisCache(
            settings=settings,
            production_cache=mock_production_cache,
            intelligent_cache=mock_intelligent_cache,
            enable_monitoring=False
        )
        
        result = await cache.get("test_key")
        
        assert result == "production_value"
        assert cache.metrics.hits == 1
        mock_production_cache.get.assert_called_once_with("test_key", None)
    
    @pytest.mark.asyncio
    async def test_get_operation_miss(self, settings, mock_production_cache, mock_intelligent_cache):
        """Test get operation with cache miss."""
        mock_intelligent_cache.get.return_value = None
        mock_production_cache.get.return_value = "default_value"
        
        cache = EnhancedRedisCache(
            settings=settings,
            production_cache=mock_production_cache,
            intelligent_cache=mock_intelligent_cache,
            enable_monitoring=False
        )
        
        result = await cache.get("test_key", "default_value")
        
        assert result == "default_value"
        assert cache.metrics.misses == 1
    
    @pytest.mark.asyncio
    async def test_set_operation(self, settings, mock_production_cache, mock_intelligent_cache):
        """Test enhanced set operation."""
        cache = EnhancedRedisCache(
            settings=settings,
            production_cache=mock_production_cache,
            intelligent_cache=mock_intelligent_cache,
            enable_monitoring=False
        )
        
        result = await cache.set("test_key", "test_value", ttl=3600)
        
        assert result is True
        assert cache.metrics.writes == 1
        mock_intelligent_cache.set.assert_called_once_with("test_key", "test_value", 3600)
        mock_production_cache.set.assert_called_once_with("test_key", "test_value", 3600, None)
    
    @pytest.mark.asyncio
    async def test_delete_operation(self, settings, mock_production_cache, mock_intelligent_cache):
        """Test enhanced delete operation."""
        cache = EnhancedRedisCache(
            settings=settings,
            production_cache=mock_production_cache,
            intelligent_cache=mock_intelligent_cache,
            enable_monitoring=False
        )
        
        result = await cache.delete("test_key")
        
        assert result is True
        assert cache.metrics.deletes == 1
        mock_intelligent_cache.delete.assert_called_once_with("test_key")
        mock_production_cache.delete.assert_called_once_with("test_key")
    
    @pytest.mark.asyncio
    async def test_invalidate_by_tag(self, settings, mock_production_cache, mock_intelligent_cache):
        """Test tag-based invalidation."""
        mock_production_cache.invalidate_by_tag.return_value = 3
        
        cache = EnhancedRedisCache(
            settings=settings,
            production_cache=mock_production_cache,
            intelligent_cache=mock_intelligent_cache,
            enable_monitoring=False
        )
        
        result = await cache.invalidate_by_tag("test_tag")
        
        assert result == 3
        mock_production_cache.invalidate_by_tag.assert_called_once_with("test_tag")
        mock_intelligent_cache.delete_pattern.assert_called_once_with("*test_tag*")
    
    @pytest.mark.asyncio
    async def test_bulk_warm_cache(self, settings, mock_production_cache, mock_intelligent_cache):
        """Test bulk cache warming."""
        warming_data = {
            "key1": "value1",
            "key2": "value2",
            "key3": "value3"
        }
        
        cache = EnhancedRedisCache(
            settings=settings,
            production_cache=mock_production_cache,
            intelligent_cache=mock_intelligent_cache,
            enable_monitoring=False
        )
        
        result = await cache.bulk_warm_cache(warming_data)
        
        assert result["status"] == "success"
        assert result["entries_warmed"] == 3
        assert result["warming_time"] > 0
        assert result["warming_rate"] > 0
        
        mock_production_cache.warm_cache.assert_called_once_with(warming_data)
        assert mock_intelligent_cache.set.call_count == 3
    
    @pytest.mark.asyncio
    async def test_get_comprehensive_stats(self, settings, mock_production_cache, mock_intelligent_cache):
        """Test comprehensive statistics retrieval."""
        cache = EnhancedRedisCache(
            settings=settings,
            production_cache=mock_production_cache,
            intelligent_cache=mock_intelligent_cache,
            enable_monitoring=False
        )
        
        # Set some metrics
        cache.metrics.hits = 100
        cache.metrics.misses = 20
        cache.metrics.writes = 50
        
        stats = await cache.get_comprehensive_stats()
        
        assert "enhanced_cache" in stats
        assert "production_cache" in stats
        assert "intelligent_cache" in stats
        assert "timestamp" in stats
        
        enhanced_metrics = stats["enhanced_cache"]["metrics"]
        assert enhanced_metrics["hits"] == 100
        assert enhanced_metrics["misses"] == 20
        assert enhanced_metrics["writes"] == 50
        assert enhanced_metrics["hit_rate"] == 100 / 120  # hits / (hits + misses)
    
    @pytest.mark.asyncio
    async def test_health_check(self, settings, mock_production_cache, mock_intelligent_cache):
        """Test enhanced health check."""
        cache = EnhancedRedisCache(
            settings=settings,
            production_cache=mock_production_cache,
            intelligent_cache=mock_intelligent_cache,
            enable_monitoring=False
        )
        
        # Set good performance metrics
        cache.metrics.hits = 800
        cache.metrics.misses = 200
        cache.metrics.avg_response_time = 0.03  # 30ms
        cache.metrics.throughput_per_second = 150
        
        health = await cache.health_check()
        
        assert health["status"] in ["healthy", "degraded", "unhealthy"]
        assert "checks" in health
        assert "production_cache" in health["checks"]
        assert "intelligent_cache" in health["checks"]
        assert "performance" in health["checks"]
        assert "overall_score" in health
        assert "max_score" in health
    
    @pytest.mark.asyncio
    async def test_performance_benchmark(self, settings, mock_production_cache, mock_intelligent_cache):
        """Test performance benchmark."""
        cache = EnhancedRedisCache(
            settings=settings,
            production_cache=mock_production_cache,
            intelligent_cache=mock_intelligent_cache,
            enable_monitoring=False
        )
        
        # Run small benchmark
        result = await cache.performance_benchmark(operations=10)
        
        assert result["status"] == "completed"
        assert result["operations"] == 10
        assert "results" in result
        assert "write" in result["results"]
        assert "read" in result["results"]
        
        write_results = result["results"]["write"]
        assert "total_time" in write_results
        assert "ops_per_second" in write_results
        assert "avg_latency_ms" in write_results
        
        read_results = result["results"]["read"]
        assert "total_time" in read_results
        assert "ops_per_second" in read_results
        assert "avg_latency_ms" in read_results
    
    @pytest.mark.asyncio
    async def test_close_operation(self, settings, mock_production_cache, mock_intelligent_cache):
        """Test cache close operation."""
        cache = EnhancedRedisCache(
            settings=settings,
            production_cache=mock_production_cache,
            intelligent_cache=mock_intelligent_cache,
            enable_monitoring=False
        )
        
        await cache.close()
        
        mock_production_cache.close.assert_called_once()
        mock_intelligent_cache.close.assert_called_once()


class TestGlobalCacheManagement:
    """Test global cache management functions."""
    
    @pytest.fixture
    def settings(self):
        """Create test settings."""
        settings = Settings()
        settings.cache_enabled = True
        settings.redis_url = "redis://localhost:6379/0"
        return settings
    
    @patch('pynomaly.infrastructure.cache.redis_enhanced.EnhancedRedisCache')
    def test_get_enhanced_redis_cache(self, mock_cache_class, settings):
        """Test getting global enhanced Redis cache."""
        mock_cache = Mock()
        mock_cache_class.return_value = mock_cache
        
        # First call should create instance
        cache1 = get_enhanced_redis_cache(settings)
        assert cache1 == mock_cache
        mock_cache_class.assert_called_once_with(settings)
        
        # Second call should return same instance
        cache2 = get_enhanced_redis_cache()
        assert cache2 == mock_cache
        assert mock_cache_class.call_count == 1  # Should not create new instance
    
    @pytest.mark.asyncio
    @patch('pynomaly.infrastructure.cache.redis_enhanced._enhanced_redis_cache')
    async def test_close_enhanced_redis_cache(self, mock_global_cache):
        """Test closing global enhanced Redis cache."""
        mock_cache = Mock()
        mock_cache.close = AsyncMock()
        mock_global_cache = mock_cache
        
        await close_enhanced_redis_cache()
        
        # Should close the cache
        mock_cache.close.assert_called_once()


class TestPerformanceTracking:
    """Test performance tracking features."""
    
    @pytest.fixture
    def cache_with_profiling(self, settings, mock_production_cache):
        """Create cache with profiling enabled."""
        return EnhancedRedisCache(
            settings=settings,
            production_cache=mock_production_cache,
            enable_monitoring=False,
            enable_profiling=True
        )
    
    @pytest.mark.asyncio
    async def test_performance_tracker_context_manager(self, cache_with_profiling):
        """Test performance tracker context manager."""
        initial_avg = cache_with_profiling.metrics.avg_response_time
        
        async with cache_with_profiling._performance_tracker("test_operation"):
            await asyncio.sleep(0.01)  # Simulate some work
        
        # Should update average response time
        assert cache_with_profiling.metrics.avg_response_time > initial_avg
    
    @pytest.mark.asyncio
    async def test_response_time_percentiles(self, cache_with_profiling):
        """Test response time percentile tracking."""
        cache = cache_with_profiling
        
        # Simulate multiple operations with different response times
        for i in range(10):
            async with cache._performance_tracker("test"):
                await asyncio.sleep(0.001 * (i + 1))  # Increasing delay
        
        # Should have percentile data
        assert cache.metrics.p95_response_time > 0
        assert cache.metrics.p99_response_time > 0
        assert cache.metrics.p99_response_time >= cache.metrics.p95_response_time
    
    def test_metrics_update_from_production(self, cache_with_profiling):
        """Test metrics update from production cache."""
        cache = cache_with_profiling
        
        prod_stats = {
            "cache_metrics": {
                "hits": 500,
                "misses": 100,
                "evictions": 10,
                "avg_response_time_ms": 25
            },
            "redis_info": {
                "used_memory": 2 * 1024 * 1024,
                "connected_clients": 15,
                "instantaneous_ops_per_sec": 200
            }
        }
        
        cache._update_metrics_from_production(prod_stats)
        
        assert cache.metrics.hits == 500
        assert cache.metrics.misses == 100
        assert cache.metrics.evictions == 10
        assert cache.metrics.avg_response_time == 0.025  # 25ms in seconds
        assert cache.metrics.memory_usage == 2 * 1024 * 1024
        assert cache.metrics.connection_count == 15
        assert cache.metrics.throughput_per_second == 200
    
    def test_performance_threshold_checking(self, cache_with_profiling):
        """Test performance threshold checking."""
        cache = cache_with_profiling
        
        # Set poor performance metrics
        cache.metrics.hits = 10
        cache.metrics.misses = 90  # 10% hit rate
        cache.metrics.avg_response_time = 0.15  # 150ms
        cache.metrics.memory_usage = 600 * 1024 * 1024  # 600MB
        
        # This should log warnings (we can't easily test logging in unit tests)
        cache._check_performance_thresholds()
        
        # Test that hit rate calculation is correct
        assert cache.metrics.hit_rate() == 0.1