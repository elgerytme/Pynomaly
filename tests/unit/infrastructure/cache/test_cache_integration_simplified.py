"""Simplified tests for cache integration module."""

import asyncio
import pytest
from unittest.mock import AsyncMock, MagicMock, patch
from datetime import datetime, timedelta

from pynomaly.infrastructure.cache.cache_integration import (
    CacheConfiguration,
    CacheHealthMonitor,
    CacheIntegrationManager,
    get_cache_integration_manager,
    get_cache_health,
    get_cache_statistics,
    perform_cache_maintenance,
    warm_cache_with_critical_data,
)
from pynomaly.infrastructure.config.settings import Settings


class TestCacheConfiguration:
    """Test CacheConfiguration class."""

    def test_cache_configuration_initialization(self):
        """Test CacheConfiguration initialization with default values."""
        config = CacheConfiguration()
        
        assert config.enabled is True
        assert config.redis_enabled is True
        assert config.redis_url is None
        assert config.default_ttl == 3600
        assert config.max_memory_size == 100 * 1024 * 1024  # 100MB
        assert config.compression_threshold == 1024
        assert config.prefetch_enabled is True
        assert config.adaptive_ttl is True
        assert config.health_check_interval == 60

    def test_cache_configuration_custom_values(self):
        """Test CacheConfiguration with custom values."""
        config = CacheConfiguration(
            enabled=False,
            redis_enabled=False,
            redis_url="redis://localhost:6379",
            default_ttl=7200,
            max_memory_size=200 * 1024 * 1024,
            compression_threshold=2048,
            prefetch_enabled=False,
            adaptive_ttl=False,
            health_check_interval=30
        )
        
        assert config.enabled is False
        assert config.redis_enabled is False
        assert config.redis_url == "redis://localhost:6379"
        assert config.default_ttl == 7200
        assert config.max_memory_size == 200 * 1024 * 1024
        assert config.compression_threshold == 2048
        assert config.prefetch_enabled is False
        assert config.adaptive_ttl is False
        assert config.health_check_interval == 30

    def test_cache_configuration_from_settings(self):
        """Test CacheConfiguration from Settings."""
        settings = Settings()
        settings.cache_enabled = True
        settings.redis_url = "redis://localhost:6379"
        settings.cache_ttl = 7200
        
        config = CacheConfiguration.from_settings(settings)
        
        assert config.enabled is True
        assert config.redis_enabled is True
        assert config.redis_url == "redis://localhost:6379"
        assert config.default_ttl == 7200

    def test_cache_configuration_from_environment(self):
        """Test CacheConfiguration from environment variables."""
        with patch.dict('os.environ', {
            'PYNOMALY_CACHE_ENABLED': 'false',
            'PYNOMALY_REDIS_ENABLED': 'false',
            'PYNOMALY_REDIS_URL': 'redis://test:6379',
            'PYNOMALY_CACHE_TTL': '1800',
            'PYNOMALY_CACHE_MEMORY_SIZE': '50000000',
            'PYNOMALY_CACHE_COMPRESSION_THRESHOLD': '512',
            'PYNOMALY_CACHE_PREFETCH': 'false',
            'PYNOMALY_CACHE_ADAPTIVE_TTL': 'false',
            'PYNOMALY_CACHE_HEALTH_CHECK_INTERVAL': '30'
        }):
            config = CacheConfiguration.from_environment()
            
            assert config.enabled is False
            assert config.redis_enabled is False
            assert config.redis_url == "redis://test:6379"
            assert config.default_ttl == 1800
            assert config.max_memory_size == 50000000
            assert config.compression_threshold == 512
            assert config.prefetch_enabled is False
            assert config.adaptive_ttl is False
            assert config.health_check_interval == 30


class TestCacheHealthMonitor:
    """Test CacheHealthMonitor class."""

    def test_cache_health_monitor_initialization(self):
        """Test CacheHealthMonitor initialization."""
        mock_cache_manager = MagicMock()
        monitor = CacheHealthMonitor(mock_cache_manager, check_interval=30)
        
        assert monitor.cache_manager == mock_cache_manager
        assert monitor.check_interval == 30
        assert monitor.monitoring_task is None
        assert monitor.health_history == []
        assert monitor.max_history == 100

    def test_cache_health_monitor_start_monitoring(self):
        """Test starting health monitoring."""
        mock_cache_manager = MagicMock()
        monitor = CacheHealthMonitor(mock_cache_manager)
        
        with patch('asyncio.create_task') as mock_create_task:
            mock_task = MagicMock()
            mock_create_task.return_value = mock_task
            
            monitor.start_monitoring()
            
            assert monitor.monitoring_task == mock_task
            mock_create_task.assert_called_once()

    @pytest.mark.asyncio
    async def test_cache_health_monitor_stop_monitoring(self):
        """Test stopping health monitoring."""
        mock_cache_manager = MagicMock()
        monitor = CacheHealthMonitor(mock_cache_manager)
        
        # Mock a running task
        mock_task = AsyncMock()
        mock_task.cancel.return_value = None
        mock_task.cancelled.return_value = True
        monitor.monitoring_task = mock_task
        
        await monitor.stop_monitoring()
        
        assert monitor.monitoring_task is None
        mock_task.cancel.assert_called_once()

    @pytest.mark.asyncio
    async def test_cache_health_monitor_check_health(self):
        """Test health check functionality."""
        mock_cache_manager = AsyncMock()
        mock_cache_manager.get_stats.return_value = {
            "cache_stats": {
                "hit_rate": 0.85,
                "avg_access_time": 0.005
            },
            "memory_cache": {
                "utilization": 0.7
            }
        }
        mock_cache_manager.redis_cache.exists.return_value = True
        
        monitor = CacheHealthMonitor(mock_cache_manager)
        
        health_status = await monitor.check_health()
        
        assert health_status["overall_health"] == "healthy"
        assert health_status["health_score"] > 0
        assert health_status["max_score"] == 8
        assert "metrics" in health_status
        assert "timestamp" in health_status

    @pytest.mark.asyncio
    async def test_cache_health_monitor_check_health_degraded(self):
        """Test health check with degraded performance."""
        mock_cache_manager = AsyncMock()
        mock_cache_manager.get_stats.return_value = {
            "cache_stats": {
                "hit_rate": 0.5,  # Low hit rate
                "avg_access_time": 0.08  # Slow access time
            },
            "memory_cache": {
                "utilization": 0.95  # High memory utilization
            }
        }
        mock_cache_manager.redis_cache.exists.return_value = True
        
        monitor = CacheHealthMonitor(mock_cache_manager)
        
        health_status = await monitor.check_health()
        
        assert health_status["overall_health"] == "degraded"
        assert len(health_status["issues"]) > 0

    @pytest.mark.asyncio
    async def test_cache_health_monitor_check_health_unhealthy(self):
        """Test health check with unhealthy status."""
        mock_cache_manager = AsyncMock()
        mock_cache_manager.get_stats.return_value = {
            "cache_stats": {
                "hit_rate": 0.2,  # Very low hit rate
                "avg_access_time": 0.1  # Very slow access time
            },
            "memory_cache": {
                "utilization": 0.95  # High memory utilization
            }
        }
        mock_cache_manager.redis_cache.exists.side_effect = Exception("Redis failed")
        
        monitor = CacheHealthMonitor(mock_cache_manager)
        
        health_status = await monitor.check_health()
        
        assert health_status["overall_health"] == "unhealthy"
        assert len(health_status["issues"]) > 0

    @pytest.mark.asyncio
    async def test_cache_health_monitor_check_health_exception(self):
        """Test health check with exception."""
        mock_cache_manager = AsyncMock()
        mock_cache_manager.get_stats.side_effect = Exception("Stats failed")
        
        monitor = CacheHealthMonitor(mock_cache_manager)
        
        health_status = await monitor.check_health()
        
        assert health_status["overall_health"] == "unhealthy"
        assert health_status["health_score"] == 0
        assert len(health_status["issues"]) > 0

    def test_cache_health_monitor_get_health_history(self):
        """Test getting health history."""
        mock_cache_manager = MagicMock()
        monitor = CacheHealthMonitor(mock_cache_manager)
        
        # Add some history
        monitor.health_history = [
            {"overall_health": "healthy", "timestamp": 1},
            {"overall_health": "degraded", "timestamp": 2},
            {"overall_health": "healthy", "timestamp": 3}
        ]
        
        history = monitor.get_health_history(limit=2)
        
        assert len(history) == 2
        assert history[0]["timestamp"] == 2
        assert history[1]["timestamp"] == 3

    def test_cache_health_monitor_get_health_summary(self):
        """Test getting health summary."""
        mock_cache_manager = MagicMock()
        monitor = CacheHealthMonitor(mock_cache_manager)
        
        # Add some history
        monitor.health_history = [
            {"overall_health": "healthy"},
            {"overall_health": "healthy"},
            {"overall_health": "degraded"},
            {"overall_health": "unhealthy"},
            {"overall_health": "healthy"}
        ]
        
        summary = monitor.get_health_summary()
        
        assert summary["total_checks"] == 5
        assert summary["recent_health_distribution"]["healthy"] == 3
        assert summary["recent_health_distribution"]["degraded"] == 1
        assert summary["recent_health_distribution"]["unhealthy"] == 1
        assert summary["current_status"] == "healthy"

    def test_cache_health_monitor_get_health_summary_no_data(self):
        """Test getting health summary with no data."""
        mock_cache_manager = MagicMock()
        monitor = CacheHealthMonitor(mock_cache_manager)
        
        summary = monitor.get_health_summary()
        
        assert summary["status"] == "no_data"


class TestCacheIntegrationManager:
    """Test CacheIntegrationManager class."""

    @patch('pynomaly.infrastructure.cache.cache_integration.init_cache')
    @patch('pynomaly.infrastructure.cache.cache_integration.get_intelligent_cache_manager')
    def test_cache_integration_manager_initialization(self, mock_get_intelligent, mock_init_cache):
        """Test CacheIntegrationManager initialization."""
        config = CacheConfiguration(
            enabled=True,
            redis_enabled=True,
            redis_url="redis://localhost:6379"
        )
        
        mock_redis_cache = MagicMock()
        mock_init_cache.return_value = mock_redis_cache
        
        mock_intelligent_cache = MagicMock()
        mock_get_intelligent.return_value = mock_intelligent_cache
        
        manager = CacheIntegrationManager(config)
        
        assert manager.config == config
        assert manager.redis_cache == mock_redis_cache
        assert manager.intelligent_cache == mock_intelligent_cache
        assert manager.health_monitor is not None

    @patch('pynomaly.infrastructure.cache.cache_integration.init_cache')
    def test_cache_integration_manager_disabled(self, mock_init_cache):
        """Test CacheIntegrationManager with caching disabled."""
        config = CacheConfiguration(enabled=False)
        
        manager = CacheIntegrationManager(config)
        
        assert manager.config == config
        assert manager.redis_cache is None
        assert manager.intelligent_cache is None
        assert manager.health_monitor is None
        mock_init_cache.assert_not_called()

    @patch('pynomaly.infrastructure.cache.cache_integration.init_cache')
    @patch('pynomaly.infrastructure.cache.cache_integration.get_intelligent_cache_manager')
    @pytest.mark.asyncio
    async def test_cache_integration_manager_get_comprehensive_stats(self, mock_get_intelligent, mock_init_cache):
        """Test getting comprehensive stats."""
        config = CacheConfiguration(
            enabled=True,
            redis_enabled=True,
            redis_url="redis://localhost:6379"
        )
        
        mock_redis_cache = MagicMock()
        mock_init_cache.return_value = mock_redis_cache
        
        mock_intelligent_cache = AsyncMock()
        mock_intelligent_cache.get_stats.return_value = {"cache_stats": {"hit_rate": 0.8}}
        mock_get_intelligent.return_value = mock_intelligent_cache
        
        manager = CacheIntegrationManager(config)
        
        stats = await manager.get_comprehensive_stats()
        
        assert "configuration" in stats
        assert "redis_cache" in stats
        assert "intelligent_cache" in stats
        assert "health_monitoring" in stats
        assert stats["configuration"]["enabled"] is True
        assert stats["redis_cache"]["available"] is True
        assert stats["intelligent_cache"]["available"] is True

    @patch('pynomaly.infrastructure.cache.cache_integration.init_cache')
    @patch('pynomaly.infrastructure.cache.cache_integration.get_intelligent_cache_manager')
    @pytest.mark.asyncio
    async def test_cache_integration_manager_perform_maintenance(self, mock_get_intelligent, mock_init_cache):
        """Test performing maintenance."""
        config = CacheConfiguration(
            enabled=True,
            redis_enabled=True,
            redis_url="redis://localhost:6379"
        )
        
        mock_redis_cache = MagicMock()
        mock_init_cache.return_value = mock_redis_cache
        
        mock_intelligent_cache = AsyncMock()
        mock_intelligent_cache.get_stats.return_value = {
            "cache_stats": {"hit_rate": 0.3},  # Low hit rate
            "memory_cache": {"utilization": 0.95}  # High memory utilization
        }
        mock_get_intelligent.return_value = mock_intelligent_cache
        
        manager = CacheIntegrationManager(config)
        
        results = await manager.perform_maintenance()
        
        assert "timestamp" in results
        assert "operations" in results
        assert "errors" in results
        assert len(results["operations"]) > 0

    @patch('pynomaly.infrastructure.cache.cache_integration.init_cache')
    @patch('pynomaly.infrastructure.cache.cache_integration.get_intelligent_cache_manager')
    @pytest.mark.asyncio
    async def test_cache_integration_manager_warm_critical_cache(self, mock_get_intelligent, mock_init_cache):
        """Test warming critical cache."""
        config = CacheConfiguration(
            enabled=True,
            redis_enabled=True,
            redis_url="redis://localhost:6379"
        )
        
        mock_redis_cache = MagicMock()
        mock_init_cache.return_value = mock_redis_cache
        
        mock_intelligent_cache = AsyncMock()
        mock_intelligent_cache.warm_cache.return_value = 5
        mock_get_intelligent.return_value = mock_intelligent_cache
        
        manager = CacheIntegrationManager(config)
        
        result = await manager.warm_critical_cache()
        
        assert result == 5
        mock_intelligent_cache.warm_cache.assert_called_once()

    @patch('pynomaly.infrastructure.cache.cache_integration.init_cache')
    @patch('pynomaly.infrastructure.cache.cache_integration.get_intelligent_cache_manager')
    @pytest.mark.asyncio
    async def test_cache_integration_manager_cleanup_cache(self, mock_get_intelligent, mock_init_cache):
        """Test cleaning up cache."""
        config = CacheConfiguration(
            enabled=True,
            redis_enabled=True,
            redis_url="redis://localhost:6379"
        )
        
        mock_redis_cache = MagicMock()
        mock_init_cache.return_value = mock_redis_cache
        
        mock_intelligent_cache = AsyncMock()
        mock_intelligent_cache.delete_pattern.side_effect = [2, 1, 0]  # Different patterns
        mock_get_intelligent.return_value = mock_intelligent_cache
        
        manager = CacheIntegrationManager(config)
        
        result = await manager.cleanup_cache()
        
        assert result == 3  # Sum of deleted items
        assert mock_intelligent_cache.delete_pattern.call_count == 3

    @patch('pynomaly.infrastructure.cache.cache_integration.init_cache')
    @patch('pynomaly.infrastructure.cache.cache_integration.get_intelligent_cache_manager')
    @pytest.mark.asyncio
    async def test_cache_integration_manager_emergency_reset(self, mock_get_intelligent, mock_init_cache):
        """Test emergency cache reset."""
        config = CacheConfiguration(
            enabled=True,
            redis_enabled=True,
            redis_url="redis://localhost:6379"
        )
        
        mock_redis_cache = MagicMock()
        mock_init_cache.return_value = mock_redis_cache
        
        mock_intelligent_cache = AsyncMock()
        mock_get_intelligent.return_value = mock_intelligent_cache
        
        manager = CacheIntegrationManager(config)
        
        results = await manager.emergency_cache_reset()
        
        assert results["status"] == "success"
        assert len(results["actions"]) > 0
        mock_redis_cache.clear.assert_called_once()
        mock_intelligent_cache.close.assert_called_once()

    @patch('pynomaly.infrastructure.cache.cache_integration.init_cache')
    @patch('pynomaly.infrastructure.cache.cache_integration.get_intelligent_cache_manager')
    @pytest.mark.asyncio
    async def test_cache_integration_manager_close(self, mock_get_intelligent, mock_init_cache):
        """Test closing cache integration manager."""
        config = CacheConfiguration(
            enabled=True,
            redis_enabled=True,
            redis_url="redis://localhost:6379"
        )
        
        mock_redis_cache = MagicMock()
        mock_init_cache.return_value = mock_redis_cache
        
        mock_intelligent_cache = AsyncMock()
        mock_get_intelligent.return_value = mock_intelligent_cache
        
        manager = CacheIntegrationManager(config)
        
        await manager.close()
        
        manager.health_monitor.stop_monitoring.assert_called_once()
        mock_intelligent_cache.close.assert_called_once()
        mock_redis_cache.close.assert_called_once()


class TestCacheIntegrationHelpers:
    """Test cache integration helper functions."""

    @patch('pynomaly.infrastructure.cache.cache_integration.get_cache_integration_manager')
    def test_get_cache_integration_manager(self, mock_get_manager):
        """Test getting cache integration manager."""
        mock_manager = MagicMock()
        mock_get_manager.return_value = mock_manager
        
        result = get_cache_integration_manager()
        
        assert result == mock_manager
        mock_get_manager.assert_called_once()

    @patch('pynomaly.infrastructure.cache.cache_integration.get_cache_integration_manager')
    @pytest.mark.asyncio
    async def test_get_cache_health(self, mock_get_manager):
        """Test getting cache health."""
        mock_manager = MagicMock()
        mock_health_monitor = AsyncMock()
        mock_health_monitor.check_health.return_value = {"overall_health": "healthy"}
        mock_manager.health_monitor = mock_health_monitor
        mock_get_manager.return_value = mock_manager
        
        result = await get_cache_health()
        
        assert result["overall_health"] == "healthy"
        mock_health_monitor.check_health.assert_called_once()

    @patch('pynomaly.infrastructure.cache.cache_integration.get_cache_integration_manager')
    @pytest.mark.asyncio
    async def test_get_cache_health_no_monitoring(self, mock_get_manager):
        """Test getting cache health with no monitoring."""
        mock_manager = MagicMock()
        mock_manager.health_monitor = None
        mock_get_manager.return_value = mock_manager
        
        result = await get_cache_health()
        
        assert result["status"] == "no_monitoring"

    @patch('pynomaly.infrastructure.cache.cache_integration.get_cache_integration_manager')
    @pytest.mark.asyncio
    async def test_get_cache_statistics(self, mock_get_manager):
        """Test getting cache statistics."""
        mock_manager = AsyncMock()
        mock_manager.get_comprehensive_stats.return_value = {"cache_stats": {"hit_rate": 0.8}}
        mock_get_manager.return_value = mock_manager
        
        result = await get_cache_statistics()
        
        assert result["cache_stats"]["hit_rate"] == 0.8
        mock_manager.get_comprehensive_stats.assert_called_once()

    @patch('pynomaly.infrastructure.cache.cache_integration.get_cache_integration_manager')
    @pytest.mark.asyncio
    async def test_perform_cache_maintenance(self, mock_get_manager):
        """Test performing cache maintenance."""
        mock_manager = AsyncMock()
        mock_manager.perform_maintenance.return_value = {"operations": ["test_op"]}
        mock_get_manager.return_value = mock_manager
        
        result = await perform_cache_maintenance()
        
        assert result["operations"] == ["test_op"]
        mock_manager.perform_maintenance.assert_called_once()

    @patch('pynomaly.infrastructure.cache.cache_integration.get_cache_integration_manager')
    @pytest.mark.asyncio
    async def test_warm_cache_with_critical_data(self, mock_get_manager):
        """Test warming cache with critical data."""
        mock_manager = AsyncMock()
        mock_manager.warm_critical_cache.return_value = 10
        mock_get_manager.return_value = mock_manager
        
        result = await warm_cache_with_critical_data()
        
        assert result == 10
        mock_manager.warm_critical_cache.assert_called_once()