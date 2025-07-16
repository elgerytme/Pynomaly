"""Comprehensive tests for enhanced Redis caching implementation."""

import asyncio
import json
import pytest
import time
from datetime import datetime, timedelta
from unittest.mock import AsyncMock, MagicMock, Mock, patch

from src.pynomaly.infrastructure.cache.enhanced_redis_cache import (
    CacheConfiguration,
    CacheEvent,
    CacheMetrics,
    CircuitBreaker,
    EnhancedRedisCache,
    EvictionPolicy,
    InvalidationStrategy,
    MetricsCollector,
    cache_decorator,
    get_cache_instance,
)
from src.pynomaly.infrastructure.cache.cache_monitoring import (
    Alert,
    AlertSeverity,
    AlertType,
    CacheMonitor,
    MonitoringThresholds,
)


class TestCacheConfiguration:
    """Test cache configuration."""
    
    def test_default_configuration(self):
        """Test default configuration values."""
        config = CacheConfiguration()
        
        assert config.redis_url == "redis://localhost:6379/0"
        assert config.default_ttl == 3600
        assert config.enable_compression is True
        assert config.enable_monitoring is True
        assert config.eviction_policy == EvictionPolicy.LRU
        assert config.invalidation_strategy == InvalidationStrategy.TTL_BASED
    
    def test_custom_configuration(self):
        """Test custom configuration values."""
        config = CacheConfiguration(
            redis_url="redis://custom:6380/1",
            default_ttl=7200,
            enable_compression=False,
            eviction_policy=EvictionPolicy.LFU,
            invalidation_strategy=InvalidationStrategy.IMMEDIATE
        )
        
        assert config.redis_url == "redis://custom:6380/1"
        assert config.default_ttl == 7200
        assert config.enable_compression is False
        assert config.eviction_policy == EvictionPolicy.LFU
        assert config.invalidation_strategy == InvalidationStrategy.IMMEDIATE


class TestCacheMetrics:
    """Test cache metrics calculation."""
    
    def test_hit_rate_calculation(self):
        """Test hit rate calculation."""
        metrics = CacheMetrics(hits=80, misses=20)
        assert metrics.hit_rate == 0.8
        assert abs(metrics.miss_rate - 0.2) < 0.0001
    
    def test_hit_rate_no_requests(self):
        """Test hit rate when no requests."""
        metrics = CacheMetrics()
        assert metrics.hit_rate == 0.0
        assert metrics.miss_rate == 1.0
    
    def test_error_rate_calculation(self):
        """Test error rate calculation."""
        metrics = CacheMetrics(
            total_requests=100,
            connection_errors=2,
            timeout_errors=1,
            serialization_errors=1
        )
        assert metrics.error_rate == 0.04


class TestCircuitBreaker:
    """Test circuit breaker functionality."""
    
    def test_circuit_breaker_closed_state(self):
        """Test circuit breaker in closed state."""
        breaker = CircuitBreaker(failure_threshold=3)
        
        # Should allow function calls
        result = breaker.call(lambda: "success")
        assert result == "success"
        assert breaker.state == "CLOSED"
    
    def test_circuit_breaker_opens_on_failures(self):
        """Test circuit breaker opens on repeated failures."""
        breaker = CircuitBreaker(failure_threshold=2)
        
        # First failure
        with pytest.raises(ValueError):
            breaker.call(lambda: exec('raise ValueError("test")'))
        assert breaker.state == "CLOSED"
        
        # Second failure - should open circuit
        with pytest.raises(ValueError):
            breaker.call(lambda: exec('raise ValueError("test")'))
        assert breaker.state == "OPEN"
    
    def test_circuit_breaker_blocks_when_open(self):
        """Test circuit breaker blocks calls when open."""
        breaker = CircuitBreaker(failure_threshold=1, recovery_timeout=1)
        
        # Trigger failure to open circuit
        with pytest.raises(ValueError):
            breaker.call(lambda: exec('raise ValueError("test")'))
        
        # Should block subsequent calls
        with pytest.raises(Exception) as exc_info:
            breaker.call(lambda: "should_not_execute")
        assert "Circuit breaker is OPEN" in str(exc_info.value)
    
    def test_circuit_breaker_half_open_recovery(self):
        """Test circuit breaker recovery through half-open state."""
        breaker = CircuitBreaker(failure_threshold=1, recovery_timeout=0.1)
        
        # Open the circuit
        with pytest.raises(ValueError):
            breaker.call(lambda: exec('raise ValueError("test")'))
        
        # Wait for recovery timeout
        time.sleep(0.2)
        
        # Should attempt reset on next call
        result = breaker.call(lambda: "recovered")
        assert result == "recovered"
        assert breaker.state == "CLOSED"


class TestMetricsCollector:
    """Test metrics collection."""
    
    @pytest.mark.asyncio
    async def test_metrics_collection(self):
        """Test basic metrics collection."""
        collector = MetricsCollector()
        
        # Simulate cache hit event
        hit_event = CacheEvent(
            event_type="cache_get",
            key="test_key",
            operation="get",
            duration_ms=10.5,
            success=True,
            metadata={"hit": True}
        )
        
        await collector.on_event(hit_event)
        
        metrics = collector.get_metrics()
        assert metrics.hits == 1
        assert metrics.misses == 0
        assert metrics.total_requests == 1
        assert metrics.average_response_time == 10.5
    
    @pytest.mark.asyncio
    async def test_response_time_tracking(self):
        """Test response time tracking."""
        collector = MetricsCollector()
        
        # Add multiple events with different response times
        for duration in [10, 20, 30]:
            event = CacheEvent(
                event_type="cache_get",
                key="test",
                operation="get",
                duration_ms=duration,
                success=True,
                metadata={"hit": True}
            )
            await collector.on_event(event)
        
        metrics = collector.get_metrics()
        assert metrics.average_response_time == 20.0  # (10+20+30)/3
        assert metrics.max_response_time == 30.0
        assert metrics.min_response_time == 10.0


@pytest.fixture
def mock_redis():
    """Mock Redis client for testing."""
    redis_mock = Mock()
    redis_mock.get.return_value = None
    redis_mock.set.return_value = True
    redis_mock.setex.return_value = True
    redis_mock.delete.return_value = 1
    redis_mock.exists.return_value = True
    redis_mock.flushdb.return_value = True
    redis_mock.info.return_value = {
        'redis_version': '6.2.0',
        'used_memory_human': '1.5M',
        'connected_clients': 5,
        'keyspace_hits': 100,
        'keyspace_misses': 20
    }
    redis_mock.ping.return_value = True
    return redis_mock


class TestEnhancedRedisCache:
    """Test enhanced Redis cache functionality."""
    
    @pytest.mark.asyncio
    async def test_cache_get_miss(self, mock_redis):
        """Test cache get with miss."""
        config = CacheConfiguration()
        
        with patch('redis.from_url', return_value=mock_redis):
            cache = EnhancedRedisCache(config)
            
            result = await cache.get("nonexistent_key", default="default_value")
            assert result == "default_value"
            mock_redis.get.assert_called_once_with("nonexistent_key")
    
    @pytest.mark.asyncio
    async def test_cache_set_and_get(self, mock_redis):
        """Test cache set and get operations."""
        import pickle
        import zlib
        
        config = CacheConfiguration(enable_compression=True, compression_threshold=10)
        test_data = {"key": "value", "number": 42}
        serialized_data = pickle.dumps(test_data)
        compressed_data = zlib.compress(serialized_data)
        
        # Mock Redis to return compressed data
        mock_redis.get.return_value = compressed_data
        
        with patch('redis.from_url', return_value=mock_redis):
            cache = EnhancedRedisCache(config)
            
            # Test set operation
            success = await cache.set("test_key", test_data, ttl=3600)
            assert success is True
            mock_redis.setex.assert_called_once()
            
            # Test get operation
            result = await cache.get("test_key")
            assert result == test_data
            mock_redis.get.assert_called_with("test_key")
    
    @pytest.mark.asyncio
    async def test_cache_delete(self, mock_redis):
        """Test cache delete operation."""
        config = CacheConfiguration()
        
        with patch('redis.from_url', return_value=mock_redis):
            cache = EnhancedRedisCache(config)
            
            success = await cache.delete("test_key")
            assert success is True
            mock_redis.delete.assert_called_once_with("test_key")
    
    @pytest.mark.asyncio
    async def test_cache_exists(self, mock_redis):
        """Test cache exists operation."""
        config = CacheConfiguration()
        
        with patch('redis.from_url', return_value=mock_redis):
            cache = EnhancedRedisCache(config)
            
            exists = await cache.exists("test_key")
            assert exists is True
            mock_redis.exists.assert_called_once_with("test_key")
    
    @pytest.mark.asyncio
    async def test_cache_clear(self, mock_redis):
        """Test cache clear operation."""
        config = CacheConfiguration()
        
        with patch('redis.from_url', return_value=mock_redis):
            cache = EnhancedRedisCache(config)
            
            success = await cache.clear()
            assert success is True
            mock_redis.flushdb.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_cache_clear_with_pattern(self, mock_redis):
        """Test cache clear with pattern."""
        config = CacheConfiguration()
        mock_redis.keys.return_value = [b'pattern:key1', b'pattern:key2']
        
        with patch('redis.from_url', return_value=mock_redis):
            cache = EnhancedRedisCache(config)
            
            success = await cache.clear(pattern="pattern:*")
            assert success is True
            mock_redis.keys.assert_called_once_with("pattern:*")
            mock_redis.delete.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_circuit_breaker_integration(self, mock_redis):
        """Test circuit breaker integration with cache operations."""
        config = CacheConfiguration()
        
        # Mock Redis to raise connection error
        mock_redis.get.side_effect = Exception("Connection failed")
        
        with patch('redis.from_url', return_value=mock_redis):
            cache = EnhancedRedisCache(config)
            
            # First few calls should fail and eventually open circuit
            for _ in range(6):  # Default threshold is 5
                result = await cache.get("test_key")
                assert result is None  # Should return default on error
            
            # Circuit should be open now
            assert cache.circuit_breaker.state == "OPEN"
    
    @pytest.mark.asyncio
    async def test_cache_info(self, mock_redis):
        """Test cache info retrieval."""
        config = CacheConfiguration()
        
        with patch('redis.from_url', return_value=mock_redis):
            cache = EnhancedRedisCache(config)
            
            info = await cache.get_info()
            
            assert "redis_info" in info
            assert "circuit_breaker" in info
            assert "metrics" in info
            assert "configuration" in info
            
            assert info["redis_info"]["redis_version"] == "6.2.0"
            assert info["configuration"]["default_ttl"] == 3600
    
    @pytest.mark.asyncio
    async def test_health_check(self, mock_redis):
        """Test cache health check."""
        config = CacheConfiguration()
        
        with patch('redis.from_url', return_value=mock_redis):
            cache = EnhancedRedisCache(config)
            
            health = await cache.health_check()
            
            assert "status" in health
            assert "response_time_ms" in health
            assert "operations" in health
            assert "timestamp" in health
            
            # Should be healthy if all operations succeed
            if all(health["operations"].values()):
                assert health["status"] == "healthy"


class TestCacheDecorator:
    """Test cache decorator functionality."""
    
    @pytest.mark.asyncio
    async def test_cache_decorator_caching(self, mock_redis):
        """Test that decorator caches function results."""
        config = CacheConfiguration()
        call_count = 0
        
        with patch('redis.from_url', return_value=mock_redis):
            cache_instance = EnhancedRedisCache(config)
            
            # Mock cache to return None first (miss), then cached result
            cache_instance.get = AsyncMock(side_effect=[None, "cached_result"])
            cache_instance.set = AsyncMock(return_value=True)
            
            with patch('src.pynomaly.infrastructure.cache.enhanced_redis_cache.get_cache_instance', return_value=cache_instance):
                
                @cache_decorator(ttl=3600)
                async def expensive_function(x, y):
                    nonlocal call_count
                    call_count += 1
                    return f"result_{x}_{y}"
                
                # First call should execute function and cache result
                result1 = await expensive_function(1, 2)
                assert result1 == "result_1_2"
                assert call_count == 1
                
                # Second call should return cached result
                result2 = await expensive_function(1, 2)
                assert result2 == "cached_result"
                assert call_count == 1  # Function not called again
    
    @pytest.mark.asyncio
    async def test_cache_decorator_custom_key(self, mock_redis):
        """Test cache decorator with custom key function."""
        config = CacheConfiguration()
        
        def custom_key_func(x, y):
            return f"custom:{x}:{y}"
        
        with patch('redis.from_url', return_value=mock_redis):
            cache_instance = EnhancedRedisCache(config)
            cache_instance.get = AsyncMock(return_value=None)
            cache_instance.set = AsyncMock(return_value=True)
            
            with patch('src.pynomaly.infrastructure.cache.enhanced_redis_cache.get_cache_instance', return_value=cache_instance):
                
                @cache_decorator(key_func=custom_key_func, ttl=1800)
                async def test_function(x, y):
                    return f"result_{x}_{y}"
                
                await test_function(1, 2)
                
                # Verify custom key was used
                cache_instance.get.assert_called_with("custom:1:2")
                cache_instance.set.assert_called_with("custom:1:2", "result_1_2", ttl=1800, tags=None)


class TestCacheMonitoring:
    """Test cache monitoring functionality."""
    
    @pytest.mark.asyncio
    async def test_alert_creation(self):
        """Test alert creation and management."""
        from src.pynomaly.infrastructure.cache.cache_monitoring import AlertManager
        
        alert_manager = AlertManager()
        
        alert = await alert_manager.create_alert(
            AlertType.HIGH_MISS_RATE,
            AlertSeverity.WARNING,
            "Cache hit rate is low",
            {"hit_rate": 0.6}
        )
        
        assert alert.alert_type == AlertType.HIGH_MISS_RATE
        assert alert.severity == AlertSeverity.WARNING
        assert alert.message == "Cache hit rate is low"
        assert not alert.resolved
        
        # Check active alerts
        active_alerts = alert_manager.get_active_alerts()
        assert len(active_alerts) == 1
        assert active_alerts[0] == alert
    
    @pytest.mark.asyncio
    async def test_alert_resolution(self):
        """Test alert resolution."""
        from src.pynomaly.infrastructure.cache.cache_monitoring import AlertManager
        
        alert_manager = AlertManager()
        
        alert = await alert_manager.create_alert(
            AlertType.SLOW_RESPONSE,
            AlertSeverity.WARNING,
            "Response time is high"
        )
        
        alert_key = list(alert_manager.active_alerts.keys())[0]
        await alert_manager.resolve_alert(alert_key, "test_user")
        
        assert alert.resolved is True
        assert alert.resolved_at is not None
        assert "Resolved by test_user" in alert.acknowledgments
    
    def test_monitoring_thresholds(self):
        """Test monitoring thresholds configuration."""
        thresholds = MonitoringThresholds(
            max_response_time_ms=50.0,
            min_hit_rate=0.9,
            max_error_rate=0.01
        )
        
        assert thresholds.max_response_time_ms == 50.0
        assert thresholds.min_hit_rate == 0.9
        assert thresholds.max_error_rate == 0.01
    
    @pytest.mark.asyncio
    async def test_cache_monitor_health_analysis(self, mock_redis):
        """Test cache monitor health analysis."""
        config = CacheConfiguration()
        
        with patch('redis.from_url', return_value=mock_redis):
            cache = EnhancedRedisCache(config)
            monitor = CacheMonitor(cache)
            
            # Mock cache info
            cache.get_info = AsyncMock(return_value={
                'metrics': {
                    'hit_rate': 0.85,
                    'average_response_time': 25.0,
                    'error_rate': 0.02,
                    'memory_usage_mb': 500
                },
                'redis_info': {'memory_usage': '500M'},
                'circuit_breaker': {'state': 'CLOSED'},
                'configuration': {'default_ttl': 3600}
            })
            
            cache.health_check = AsyncMock(return_value={
                'status': 'healthy',
                'response_time_ms': 15.0
            })
            
            dashboard_data = await monitor.get_dashboard_data()
            
            assert "health" in dashboard_data
            assert "metrics" in dashboard_data
            assert "trends" in dashboard_data
            assert "alerts" in dashboard_data
            
            health = dashboard_data["health"]
            assert "score" in health
            assert "status" in health
            assert "recommendations" in health
    
    @pytest.mark.asyncio
    async def test_metrics_aggregator(self):
        """Test metrics aggregation over time."""
        from src.pynomaly.infrastructure.cache.cache_monitoring import MetricsAggregator
        
        aggregator = MetricsAggregator(window_size_minutes=60)
        
        # Add sample metrics
        metrics1 = {"hit_rate": 0.8, "response_time": 20.0}
        metrics2 = {"hit_rate": 0.85, "response_time": 25.0}
        metrics3 = {"hit_rate": 0.9, "response_time": 15.0}
        
        await aggregator.add_metrics(metrics1)
        await aggregator.add_metrics(metrics2)
        await aggregator.add_metrics(metrics3)
        
        # Test average calculation
        avg_hit_rate = aggregator.get_average_metric("hit_rate")
        assert abs(avg_hit_rate - 0.85) < 0.01  # (0.8 + 0.85 + 0.9) / 3
        
        avg_response_time = aggregator.get_average_metric("response_time")
        assert abs(avg_response_time - 20.0) < 0.1  # (20 + 25 + 15) / 3


class TestIntegration:
    """Integration tests for the entire caching system."""
    
    @pytest.mark.asyncio
    async def test_full_cache_lifecycle(self, mock_redis):
        """Test complete cache lifecycle with monitoring."""
        import pickle
        
        config = CacheConfiguration(
            enable_monitoring=True,
            enable_compression=True
        )
        
        # Setup mock Redis responses
        stored_data = {}
        
        def mock_set(key, value):
            stored_data[key] = value
            return True
        
        def mock_setex(key, ttl, value):
            stored_data[key] = value
            return True
        
        def mock_get(key):
            return stored_data.get(key)
        
        def mock_delete(key):
            if key in stored_data:
                del stored_data[key]
                return 1
            return 0
        
        def mock_exists(key):
            return 1 if key in stored_data else 0
        
        mock_redis.set.side_effect = mock_set
        mock_redis.setex.side_effect = mock_setex
        mock_redis.get.side_effect = mock_get
        mock_redis.delete.side_effect = mock_delete
        mock_redis.exists.side_effect = mock_exists
        
        with patch('redis.from_url', return_value=mock_redis):
            # Create cache and monitor
            cache = EnhancedRedisCache(config)
            monitor = CacheMonitor(cache)
            
            # Test data operations
            test_data = {"message": "Hello, World!", "timestamp": time.time()}
            
            # Set data
            success = await cache.set("test_key", test_data, ttl=3600)
            assert success is True
            
            # Get data
            retrieved_data = await cache.get("test_key")
            assert retrieved_data == test_data
            
            # Check existence
            exists = await cache.exists("test_key")
            assert exists is True
            
            # Get cache info
            info = await cache.get_info()
            assert "metrics" in info
            assert info["metrics"]["hits"] >= 1
            
            # Health check
            health = await cache.health_check()
            assert health["status"] == "healthy"
            
            # Monitor dashboard
            dashboard_data = await monitor.get_dashboard_data()
            assert dashboard_data["health"]["status"] in ["excellent", "good", "warning", "critical"]
            
            # Delete data
            success = await cache.delete("test_key")
            assert success is True
            
            # Verify deletion
            result = await cache.get("test_key")
            assert result is None
            
            # Clean up
            await cache.close()
    
    @pytest.mark.asyncio
    async def test_error_handling_and_resilience(self, mock_redis):
        """Test error handling and system resilience."""
        config = CacheConfiguration()
        
        # Test connection failures
        mock_redis.get.side_effect = Exception("Connection failed")
        mock_redis.set.side_effect = Exception("Connection failed")
        
        with patch('redis.from_url', return_value=mock_redis):
            cache = EnhancedRedisCache(config)
            
            # Operations should handle errors gracefully
            result = await cache.get("test_key", default="fallback")
            assert result == "fallback"
            
            success = await cache.set("test_key", "test_value")
            assert success is False
            
            # Circuit breaker should open after repeated failures
            for _ in range(6):
                await cache.get("test_key")
            
            assert cache.circuit_breaker.state == "OPEN"
            
            await cache.close()


if __name__ == "__main__":
    # Run tests
    pytest.main([__file__, "-v"])