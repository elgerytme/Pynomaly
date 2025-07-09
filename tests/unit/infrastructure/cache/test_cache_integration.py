"""Tests for cache integration module."""

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
from pynomaly.shared.exceptions import CacheError, ConfigurationError


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
            ttl=7200,
            max_size=20000,
            redis_url="redis://localhost:6379",
            use_redis=True,
            key_prefix="custom_prefix",
            compression_enabled=True,
            serialization_format="pickle",
            health_check_interval=30,
            maintenance_interval=600,
            backup_enabled=True,
            recovery_enabled=True
        )
        
        assert config.enabled is False
        assert config.ttl == 7200
        assert config.max_size == 20000
        assert config.redis_url == "redis://localhost:6379"
        assert config.use_redis is True
        assert config.key_prefix == "custom_prefix"
        assert config.compression_enabled is True
        assert config.serialization_format == "pickle"
        assert config.health_check_interval == 30
        assert config.maintenance_interval == 600
        assert config.backup_enabled is True
        assert config.recovery_enabled is True

    def test_cache_configuration_validation(self):
        """Test CacheConfiguration validation."""
        # Test invalid TTL
        with pytest.raises(ValueError, match="TTL must be positive"):
            CacheConfiguration(ttl=0)
        
        # Test invalid max_size
        with pytest.raises(ValueError, match="Max size must be positive"):
            CacheConfiguration(max_size=0)
        
        # Test invalid health_check_interval
        with pytest.raises(ValueError, match="Health check interval must be positive"):
            CacheConfiguration(health_check_interval=0)
        
        # Test invalid maintenance_interval
        with pytest.raises(ValueError, match="Maintenance interval must be positive"):
            CacheConfiguration(maintenance_interval=0)

    def test_cache_configuration_serialization(self):
        """Test CacheConfiguration serialization methods."""
        config = CacheConfiguration(
            ttl=7200,
            max_size=20000,
            redis_url="redis://localhost:6379",
            use_redis=True
        )
        
        # Test to_dict
        config_dict = config.to_dict()
        assert config_dict["ttl"] == 7200
        assert config_dict["max_size"] == 20000
        assert config_dict["redis_url"] == "redis://localhost:6379"
        assert config_dict["use_redis"] is True
        
        # Test from_dict
        new_config = CacheConfiguration.from_dict(config_dict)
        assert new_config.ttl == 7200
        assert new_config.max_size == 20000
        assert new_config.redis_url == "redis://localhost:6379"
        assert new_config.use_redis is True


class TestCacheHealthMonitor:
    """Test CacheHealthMonitor class."""

    def test_cache_health_monitor_initialization(self):
        """Test CacheHealthMonitor initialization."""
        config = CacheConfiguration()
        monitor = CacheHealthMonitor(config)
        
        assert monitor.config == config
        assert monitor.is_healthy is True
        assert monitor.last_check is None
        assert monitor.error_count == 0
        assert monitor.consecutive_errors == 0
        assert monitor.max_consecutive_errors == 5
        assert monitor.performance_history == []
        assert monitor.alert_threshold == 3

    @pytest.mark.asyncio
    async def test_check_health_success(self):
        """Test successful health check."""
        config = CacheConfiguration()
        monitor = CacheHealthMonitor(config)
        
        mock_cache = AsyncMock()
        mock_cache.ping.return_value = True
        mock_cache.get_stats.return_value = CacheStatistics(
            hits=100,
            misses=20,
            hit_rate=0.833,
            memory_usage=1024,
            key_count=50,
            evictions=5
        )
        
        status = await monitor.check_health(mock_cache)
        
        assert status.is_healthy is True
        assert status.response_time > 0
        assert status.memory_usage == 1024
        assert status.hit_rate == 0.833
        assert status.error_count == 0
        assert monitor.consecutive_errors == 0
        assert monitor.is_healthy is True

    @pytest.mark.asyncio
    async def test_check_health_failure(self):
        """Test health check failure."""
        config = CacheConfiguration()
        monitor = CacheHealthMonitor(config)
        
        mock_cache = AsyncMock()
        mock_cache.ping.side_effect = Exception("Connection error")
        
        status = await monitor.check_health(mock_cache)
        
        assert status.is_healthy is False
        assert status.error_message == "Connection error"
        assert monitor.consecutive_errors == 1
        assert monitor.error_count == 1

    @pytest.mark.asyncio
    async def test_check_health_consecutive_errors(self):
        """Test health check with consecutive errors."""
        config = CacheConfiguration()
        monitor = CacheHealthMonitor(config)
        
        mock_cache = AsyncMock()
        mock_cache.ping.side_effect = Exception("Connection error")
        
        # Simulate consecutive errors
        for i in range(6):
            await monitor.check_health(mock_cache)
        
        assert monitor.consecutive_errors == 6
        assert monitor.is_healthy is False

    def test_get_performance_metrics(self):
        """Test getting performance metrics."""
        config = CacheConfiguration()
        monitor = CacheHealthMonitor(config)
        
        # Add some performance history
        now = datetime.utcnow()
        monitor.performance_history = [
            CacheHealthStatus(
                is_healthy=True,
                response_time=10.5,
                memory_usage=1024,
                hit_rate=0.8,
                timestamp=now - timedelta(minutes=5)
            ),
            CacheHealthStatus(
                is_healthy=True,
                response_time=12.0,
                memory_usage=1536,
                hit_rate=0.75,
                timestamp=now
            )
        ]
        
        metrics = monitor.get_performance_metrics()
        
        assert metrics["average_response_time"] == 11.25
        assert metrics["average_memory_usage"] == 1280
        assert metrics["average_hit_rate"] == 0.775
        assert metrics["uptime_percentage"] == 100.0
        assert metrics["total_checks"] == 2

    def test_get_performance_metrics_empty_history(self):
        """Test getting performance metrics with empty history."""
        config = CacheConfiguration()
        monitor = CacheHealthMonitor(config)
        
        metrics = monitor.get_performance_metrics()
        
        assert metrics["average_response_time"] == 0.0
        assert metrics["average_memory_usage"] == 0.0
        assert metrics["average_hit_rate"] == 0.0
        assert metrics["uptime_percentage"] == 0.0
        assert metrics["total_checks"] == 0

    def test_should_alert(self):
        """Test alert threshold logic."""
        config = CacheConfiguration()
        monitor = CacheHealthMonitor(config)
        
        # No alert initially
        assert monitor.should_alert() is False
        
        # Set consecutive errors to trigger alert
        monitor.consecutive_errors = 3
        assert monitor.should_alert() is True
        
        # Set error count to trigger alert
        monitor.consecutive_errors = 1
        monitor.error_count = 10
        assert monitor.should_alert() is True

    def test_reset_stats(self):
        """Test resetting statistics."""
        config = CacheConfiguration()
        monitor = CacheHealthMonitor(config)
        
        # Set some values
        monitor.error_count = 5
        monitor.consecutive_errors = 3
        monitor.is_healthy = False
        monitor.performance_history = [MagicMock()]
        
        monitor.reset_stats()
        
        assert monitor.error_count == 0
        assert monitor.consecutive_errors == 0
        assert monitor.is_healthy is True
        assert monitor.performance_history == []


class TestCacheIntegrationManager:
    """Test CacheIntegrationManager class."""

    def test_cache_integration_manager_initialization(self):
        """Test CacheIntegrationManager initialization."""
        config = CacheConfiguration()
        manager = CacheIntegrationManager(config)
        
        assert manager.config == config
        assert manager.cache is None
        assert manager.health_monitor is not None
        assert manager.maintenance_tasks == []
        assert manager.is_running is False

    @pytest.mark.asyncio
    async def test_initialize_redis_cache(self):
        """Test initializing Redis cache."""
        config = CacheConfiguration(use_redis=True, redis_url="redis://localhost:6379")
        manager = CacheIntegrationManager(config)
        
        with patch("pynomaly.infrastructure.cache.cache_integration.redis.Redis") as mock_redis:
            mock_redis_instance = AsyncMock()
            mock_redis.return_value = mock_redis_instance
            mock_redis_instance.ping.return_value = True
            
            await manager.initialize()
            
            assert manager.cache is not None
            assert manager.is_running is True
            mock_redis_instance.ping.assert_called_once()

    @pytest.mark.asyncio
    async def test_initialize_memory_cache(self):
        """Test initializing memory cache."""
        config = CacheConfiguration(use_redis=False)
        manager = CacheIntegrationManager(config)
        
        await manager.initialize()
        
        assert manager.cache is not None
        assert manager.is_running is True

    @pytest.mark.asyncio
    async def test_initialize_disabled_cache(self):
        """Test initializing disabled cache."""
        config = CacheConfiguration(enabled=False)
        manager = CacheIntegrationManager(config)
        
        await manager.initialize()
        
        assert manager.cache is None
        assert manager.is_running is False

    @pytest.mark.asyncio
    async def test_shutdown(self):
        """Test shutting down cache manager."""
        config = CacheConfiguration()
        manager = CacheIntegrationManager(config)
        
        await manager.initialize()
        assert manager.is_running is True
        
        await manager.shutdown()
        assert manager.is_running is False
        assert manager.cache is None

    @pytest.mark.asyncio
    async def test_run_maintenance(self):
        """Test running maintenance tasks."""
        config = CacheConfiguration()
        manager = CacheIntegrationManager(config)
        
        # Add a mock maintenance task
        mock_task = AsyncMock()
        mock_task.run.return_value = True
        manager.maintenance_tasks = [mock_task]
        
        await manager.initialize()
        await manager.run_maintenance()
        
        mock_task.run.assert_called_once()

    @pytest.mark.asyncio
    async def test_backup_cache(self):
        """Test backing up cache."""
        config = CacheConfiguration(backup_enabled=True)
        manager = CacheIntegrationManager(config)
        
        mock_cache = AsyncMock()
        mock_cache.keys.return_value = ["key1", "key2"]
        mock_cache.get.side_effect = ["value1", "value2"]
        manager.cache = mock_cache
        
        backup_config = CacheBackupConfig(
            enabled=True,
            location="/tmp/backup",
            retention_days=7
        )
        
        with patch("builtins.open", MagicMock()):
            with patch("json.dump", MagicMock()) as mock_dump:
                result = await manager.backup_cache(backup_config)
                
                assert result is True
                mock_dump.assert_called_once()

    @pytest.mark.asyncio
    async def test_restore_cache(self):
        """Test restoring cache."""
        config = CacheConfiguration(recovery_enabled=True)
        manager = CacheIntegrationManager(config)
        
        mock_cache = AsyncMock()
        manager.cache = mock_cache
        
        recovery_config = CacheRecoveryConfig(
            enabled=True,
            backup_location="/tmp/backup",
            auto_recovery=True
        )
        
        backup_data = {
            "key1": "value1",
            "key2": "value2"
        }
        
        with patch("builtins.open", MagicMock()):
            with patch("json.load", return_value=backup_data) as mock_load:
                result = await manager.restore_cache(recovery_config)
                
                assert result is True
                mock_load.assert_called_once()
                assert mock_cache.set.call_count == 2

    @pytest.mark.asyncio
    async def test_get_cache_statistics(self):
        """Test getting cache statistics."""
        config = CacheConfiguration()
        manager = CacheIntegrationManager(config)
        
        mock_cache = AsyncMock()
        mock_cache.get_stats.return_value = CacheStatistics(
            hits=100,
            misses=20,
            hit_rate=0.833,
            memory_usage=1024,
            key_count=50,
            evictions=5
        )
        manager.cache = mock_cache
        
        stats = await manager.get_cache_statistics()
        
        assert stats.hits == 100
        assert stats.misses == 20
        assert stats.hit_rate == 0.833
        assert stats.memory_usage == 1024
        assert stats.key_count == 50
        assert stats.evictions == 5

    @pytest.mark.asyncio
    async def test_clear_cache(self):
        """Test clearing cache."""
        config = CacheConfiguration()
        manager = CacheIntegrationManager(config)
        
        mock_cache = AsyncMock()
        mock_cache.clear.return_value = True
        manager.cache = mock_cache
        
        result = await manager.clear_cache()
        
        assert result is True
        mock_cache.clear.assert_called_once()

    @pytest.mark.asyncio
    async def test_get_cache_keys(self):
        """Test getting cache keys."""
        config = CacheConfiguration()
        manager = CacheIntegrationManager(config)
        
        mock_cache = AsyncMock()
        mock_cache.keys.return_value = ["key1", "key2", "key3"]
        manager.cache = mock_cache
        
        keys = await manager.get_cache_keys()
        
        assert keys == ["key1", "key2", "key3"]
        mock_cache.keys.assert_called_once()

    @pytest.mark.asyncio
    async def test_get_cache_keys_with_pattern(self):
        """Test getting cache keys with pattern."""
        config = CacheConfiguration()
        manager = CacheIntegrationManager(config)
        
        mock_cache = AsyncMock()
        mock_cache.keys.return_value = ["user:1", "user:2"]
        manager.cache = mock_cache
        
        keys = await manager.get_cache_keys("user:*")
        
        assert keys == ["user:1", "user:2"]
        mock_cache.keys.assert_called_once_with("user:*")

    @pytest.mark.asyncio
    async def test_error_handling(self):
        """Test error handling in cache operations."""
        config = CacheConfiguration()
        manager = CacheIntegrationManager(config)
        
        mock_cache = AsyncMock()
        mock_cache.get_stats.side_effect = Exception("Redis error")
        manager.cache = mock_cache
        
        with pytest.raises(CacheError):
            await manager.get_cache_statistics()


class TestCacheMaintenanceTask:
    """Test CacheMaintenanceTask class."""

    def test_cache_maintenance_task_initialization(self):
        """Test CacheMaintenanceTask initialization."""
        task = CacheMaintenanceTask(
            name="test_task",
            interval=60,
            description="Test task"
        )
        
        assert task.name == "test_task"
        assert task.interval == 60
        assert task.description == "Test task"
        assert task.last_run is None
        assert task.enabled is True
        assert task.error_count == 0
        assert task.success_count == 0

    def test_should_run_never_run(self):
        """Test should_run when task has never run."""
        task = CacheMaintenanceTask(name="test", interval=60)
        
        assert task.should_run() is True

    def test_should_run_recently_run(self):
        """Test should_run when task was recently run."""
        task = CacheMaintenanceTask(name="test", interval=60)
        task.last_run = datetime.utcnow()
        
        assert task.should_run() is False

    def test_should_run_ready_to_run(self):
        """Test should_run when task is ready to run."""
        task = CacheMaintenanceTask(name="test", interval=60)
        task.last_run = datetime.utcnow() - timedelta(seconds=70)
        
        assert task.should_run() is True

    def test_should_run_disabled(self):
        """Test should_run when task is disabled."""
        task = CacheMaintenanceTask(name="test", interval=60, enabled=False)
        
        assert task.should_run() is False

    @pytest.mark.asyncio
    async def test_run_abstract_method(self):
        """Test that run method is abstract."""
        task = CacheMaintenanceTask(name="test", interval=60)
        
        with pytest.raises(NotImplementedError):
            await task.run()

    def test_record_success(self):
        """Test recording successful task execution."""
        task = CacheMaintenanceTask(name="test", interval=60)
        initial_count = task.success_count
        
        task.record_success()
        
        assert task.success_count == initial_count + 1
        assert task.last_run is not None

    def test_record_error(self):
        """Test recording task execution error."""
        task = CacheMaintenanceTask(name="test", interval=60)
        initial_count = task.error_count
        
        task.record_error("Test error")
        
        assert task.error_count == initial_count + 1
        assert task.last_error == "Test error"

    def test_get_statistics(self):
        """Test getting task statistics."""
        task = CacheMaintenanceTask(name="test", interval=60)
        task.success_count = 10
        task.error_count = 2
        
        stats = task.get_statistics()
        
        assert stats["name"] == "test"
        assert stats["success_count"] == 10
        assert stats["error_count"] == 2
        assert stats["total_runs"] == 12
        assert stats["success_rate"] == 0.833
        assert stats["enabled"] is True


class TestCacheBackupConfig:
    """Test CacheBackupConfig class."""

    def test_cache_backup_config_initialization(self):
        """Test CacheBackupConfig initialization."""
        config = CacheBackupConfig(
            enabled=True,
            location="/tmp/backup",
            retention_days=7
        )
        
        assert config.enabled is True
        assert config.location == "/tmp/backup"
        assert config.retention_days == 7
        assert config.compression_enabled is False
        assert config.schedule_interval == 3600
        assert config.max_backup_size == 1073741824  # 1GB


class TestCacheRecoveryConfig:
    """Test CacheRecoveryConfig class."""

    def test_cache_recovery_config_initialization(self):
        """Test CacheRecoveryConfig initialization."""
        config = CacheRecoveryConfig(
            enabled=True,
            backup_location="/tmp/backup",
            auto_recovery=True
        )
        
        assert config.enabled is True
        assert config.backup_location == "/tmp/backup"
        assert config.auto_recovery is True
        assert config.recovery_timeout == 300
        assert config.verify_integrity is True
        assert config.fallback_enabled is True