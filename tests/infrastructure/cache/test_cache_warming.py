"""Tests for cache warming functionality (Issue #99)."""

import asyncio
import time
from datetime import datetime
from unittest.mock import AsyncMock

import pytest

from pynomaly.infrastructure.cache.cache_warming import (
    CacheWarmingService,
    WarmingMetrics,
    WarmingStrategy,
)


async def wait_for_condition(condition_func, timeout=5.0, poll_interval=0.01):
    """Wait for a condition to be true with polling instead of fixed delays."""
    start_time = time.time()
    while time.time() - start_time < timeout:
        if await condition_func() if asyncio.iscoroutinefunction(condition_func) else condition_func():
            return True
        await asyncio.sleep(poll_interval)
    return False


class TestWarmingStrategy:
    """Test warming strategy configuration."""

    def test_warming_strategy_defaults(self):
        """Test default strategy configuration."""
        strategy = WarmingStrategy(name="test_strategy")

        assert strategy.name == "test_strategy"
        assert strategy.priority == 1
        assert strategy.enabled is True
        assert strategy.warm_on_startup is True
        assert strategy.warm_on_demand is True
        assert strategy.batch_size == 100
        assert strategy.delay_between_batches == 0.1
        assert strategy.ttl_seconds == 3600
        assert len(strategy.tags) == 0

    def test_warming_strategy_custom(self):
        """Test custom strategy configuration."""
        strategy = WarmingStrategy(
            name="custom_strategy",
            priority=5,
            enabled=False,
            warm_on_startup=False,
            batch_size=50,
            delay_between_batches=0.2,
            ttl_seconds=1800,
            tags={"custom", "test"},
        )

        assert strategy.name == "custom_strategy"
        assert strategy.priority == 5
        assert strategy.enabled is False
        assert strategy.warm_on_startup is False
        assert strategy.batch_size == 50
        assert strategy.delay_between_batches == 0.2
        assert strategy.ttl_seconds == 1800
        assert strategy.tags == {"custom", "test"}


class TestWarmingMetrics:
    """Test warming metrics functionality."""

    def test_warming_metrics_initialization(self):
        """Test metrics initialization."""
        started_at = datetime.utcnow()
        metrics = WarmingMetrics(strategy_name="test_strategy", started_at=started_at)

        assert metrics.strategy_name == "test_strategy"
        assert metrics.started_at == started_at
        assert metrics.completed_at is None
        assert metrics.items_warmed == 0
        assert metrics.items_failed == 0
        assert metrics.total_time_seconds == 0.0

    def test_success_rate_calculation(self):
        """Test success rate calculation."""
        metrics = WarmingMetrics(
            strategy_name="test",
            started_at=datetime.utcnow(),
            items_warmed=80,
            items_failed=20,
        )

        assert metrics.success_rate == 80.0

    def test_success_rate_no_items(self):
        """Test success rate with no items."""
        metrics = WarmingMetrics(strategy_name="test", started_at=datetime.utcnow())

        assert metrics.success_rate == 0.0

    def test_items_per_second_calculation(self):
        """Test items per second calculation."""
        metrics = WarmingMetrics(
            strategy_name="test",
            started_at=datetime.utcnow(),
            items_warmed=100,
            total_time_seconds=10.0,
        )

        assert metrics.items_per_second == 10.0

    def test_items_per_second_no_time(self):
        """Test items per second with zero time."""
        metrics = WarmingMetrics(
            strategy_name="test",
            started_at=datetime.utcnow(),
            items_warmed=100,
            total_time_seconds=0.0,
        )

        assert metrics.items_per_second == 0.0


class TestCacheWarmingService:
    """Test cache warming service functionality."""

    @pytest.fixture
    def mock_cache(self):
        """Mock cache backend."""
        cache = AsyncMock()
        cache.set.return_value = True
        return cache

    @pytest.fixture
    def warming_service(self, mock_cache):
        """Create warming service with mock cache."""
        return CacheWarmingService(
            cache_backend=mock_cache,
            enable_background_warming=False,  # Disable for testing
        )

    def test_service_initialization(self, mock_cache):
        """Test service initialization."""
        service = CacheWarmingService(mock_cache)

        assert service.cache == mock_cache
        assert service.enable_background_warming is True
        assert len(service.strategies) > 0  # Default strategies registered
        assert "critical_app_data" in service.strategies
        assert "detector_models" in service.strategies

    def test_register_strategy(self, warming_service):
        """Test strategy registration."""
        strategy = WarmingStrategy(name="test_strategy", priority=3, tags={"test"})

        warming_service.register_strategy(strategy)

        assert "test_strategy" in warming_service.strategies
        assert warming_service.strategies["test_strategy"] == strategy
        assert "test_strategy" in warming_service.metrics

    def test_remove_strategy(self, warming_service):
        """Test strategy removal."""
        strategy = WarmingStrategy(name="removable_strategy")
        warming_service.register_strategy(strategy)

        # Verify strategy exists
        assert "removable_strategy" in warming_service.strategies

        # Remove strategy
        result = warming_service.remove_strategy("removable_strategy")
        assert result is True
        assert "removable_strategy" not in warming_service.strategies

    def test_remove_nonexistent_strategy(self, warming_service):
        """Test removing non-existent strategy."""
        result = warming_service.remove_strategy("nonexistent")
        assert result is False

    async def test_execute_strategy_success(self, warming_service, mock_cache):
        """Test successful strategy execution."""

        # Create strategy with data generator
        async def mock_generator():
            return {"key1": "value1", "key2": "value2", "key3": "value3"}

        strategy = WarmingStrategy(
            name="test_strategy",
            batch_size=2,
            delay_between_batches=0.001,  # Minimal delay for testing
            data_generator=mock_generator,
        )

        warming_service.register_strategy(strategy)

        # Execute strategy
        metrics = await warming_service._execute_strategy(strategy)

        assert metrics.strategy_name == "test_strategy"
        assert metrics.items_warmed == 3
        assert metrics.items_failed == 0
        assert metrics.success_rate == 100.0
        assert metrics.completed_at is not None
        assert metrics.total_time_seconds > 0

    async def test_execute_strategy_with_failures(self, warming_service, mock_cache):
        """Test strategy execution with failures."""

        # Mock cache to fail on specific keys
        async def mock_set(key, value, ttl=None, tags=None):
            if key == "key2":
                raise Exception("Cache error")
            return True

        mock_cache.set.side_effect = mock_set

        # Create strategy with data generator
        async def mock_generator():
            return {
                "key1": "value1",
                "key2": "value2",  # This will fail
                "key3": "value3",
            }

        strategy = WarmingStrategy(
            name="failing_strategy", data_generator=mock_generator
        )

        warming_service.register_strategy(strategy)

        # Execute strategy
        metrics = await warming_service._execute_strategy(strategy)

        assert metrics.strategy_name == "failing_strategy"
        assert metrics.items_warmed == 2
        assert metrics.items_failed == 1
        assert metrics.success_rate == pytest.approx(66.67, rel=1e-2)
        assert len(metrics.errors) == 1

    async def test_execute_strategy_no_data(self, warming_service):
        """Test strategy execution with no data."""

        # Create strategy with empty data generator
        async def empty_generator():
            return {}

        strategy = WarmingStrategy(
            name="empty_strategy", data_generator=empty_generator
        )

        warming_service.register_strategy(strategy)

        # Execute strategy
        metrics = await warming_service._execute_strategy(strategy)

        assert metrics.strategy_name == "empty_strategy"
        assert metrics.items_warmed == 0
        assert metrics.items_failed == 0
        assert metrics.completed_at is not None

    async def test_warm_startup_strategies(self, warming_service, mock_cache):
        """Test warming startup strategies."""
        # Register strategies with different startup settings
        startup_strategy = WarmingStrategy(
            name="startup_strategy",
            warm_on_startup=True,
            priority=1,
            data_generator=lambda: {"key1": "value1"},
        )

        non_startup_strategy = WarmingStrategy(
            name="non_startup_strategy",
            warm_on_startup=False,
            priority=2,
            data_generator=lambda: {"key2": "value2"},
        )

        warming_service.register_strategy(startup_strategy)
        warming_service.register_strategy(non_startup_strategy)

        # Execute startup warming
        results = await warming_service.warm_on_startup()

        # Should include startup strategy but not non-startup
        assert "startup_strategy" in results
        assert "non_startup_strategy" not in results

    async def test_warm_specific_strategy(self, warming_service, mock_cache):
        """Test warming a specific strategy."""
        strategy = WarmingStrategy(
            name="specific_strategy",
            data_generator=lambda: {"specific_key": "specific_value"},
        )

        warming_service.register_strategy(strategy)

        # Warm specific strategy
        metrics = await warming_service.warm_strategy("specific_strategy")

        assert metrics is not None
        assert metrics.strategy_name == "specific_strategy"
        assert metrics.items_warmed >= 0

    async def test_warm_nonexistent_strategy(self, warming_service):
        """Test warming non-existent strategy."""
        metrics = await warming_service.warm_strategy("nonexistent")
        assert metrics is None

    async def test_warm_disabled_strategy(self, warming_service):
        """Test warming disabled strategy."""
        strategy = WarmingStrategy(
            name="disabled_strategy",
            enabled=False,
            data_generator=lambda: {"key": "value"},
        )

        warming_service.register_strategy(strategy)

        # Should return None for disabled strategy
        metrics = await warming_service.warm_strategy("disabled_strategy")
        assert metrics is None

    async def test_warm_all_strategies(self, warming_service, mock_cache):
        """Test warming all enabled strategies."""
        # Register multiple strategies
        strategy1 = WarmingStrategy(
            name="strategy1", priority=1, data_generator=lambda: {"key1": "value1"}
        )

        strategy2 = WarmingStrategy(
            name="strategy2",
            priority=2,
            enabled=True,
            data_generator=lambda: {"key2": "value2"},
        )

        strategy3 = WarmingStrategy(
            name="strategy3",
            priority=3,
            enabled=False,  # Disabled
            data_generator=lambda: {"key3": "value3"},
        )

        warming_service.register_strategy(strategy1)
        warming_service.register_strategy(strategy2)
        warming_service.register_strategy(strategy3)

        # Warm all strategies
        results = await warming_service.warm_all_strategies()

        # Should include enabled strategies but not disabled
        assert "strategy1" in results
        assert "strategy2" in results
        assert "strategy3" not in results

    def test_get_strategy_metrics(self, warming_service):
        """Test getting strategy metrics."""
        strategy_name = "test_strategy"
        strategy = WarmingStrategy(name=strategy_name)
        warming_service.register_strategy(strategy)

        # Initially no metrics
        metrics = warming_service.get_strategy_metrics(strategy_name)
        assert metrics == []

        # Add some metrics
        test_metrics = WarmingMetrics(
            strategy_name=strategy_name, started_at=datetime.utcnow(), items_warmed=10
        )
        warming_service.metrics[strategy_name].append(test_metrics)

        # Should return metrics
        metrics = warming_service.get_strategy_metrics(strategy_name)
        assert len(metrics) == 1
        assert metrics[0] == test_metrics

    def test_get_all_metrics(self, warming_service):
        """Test getting all metrics."""
        all_metrics = warming_service.get_all_metrics()
        assert isinstance(all_metrics, dict)

        # Should include default strategies
        assert len(all_metrics) > 0

    def test_get_strategy_summary(self, warming_service):
        """Test getting strategy summary."""
        summary = warming_service.get_strategy_summary()
        assert isinstance(summary, dict)

        # Should include default strategies
        assert len(summary) > 0

        # Check structure of summary
        for strategy_name, info in summary.items():
            assert "enabled" in info
            assert "priority" in info
            assert "warm_on_startup" in info
            assert "batch_size" in info
            assert "ttl_seconds" in info
            assert "tags" in info
            assert "executions_count" in info
            assert "last_execution" in info

    async def test_default_data_generators(self, warming_service):
        """Test default data generators."""
        # Test critical data generator
        critical_data = await warming_service._generate_critical_data()
        assert isinstance(critical_data, dict)
        assert len(critical_data) > 0
        assert any("app:" in key for key in critical_data.keys())

        # Test detector data generator
        detector_data = await warming_service._generate_detector_data()
        assert isinstance(detector_data, dict)
        assert len(detector_data) > 0
        assert any("detector:" in key for key in detector_data.keys())

        # Test dataset data generator
        dataset_data = await warming_service._generate_dataset_data()
        assert isinstance(dataset_data, dict)
        assert len(dataset_data) > 0
        assert any("dataset:" in key for key in dataset_data.keys())

        # Test API cache data generator
        api_data = await warming_service._generate_api_cache_data()
        assert isinstance(api_data, dict)
        assert len(api_data) > 0
        assert any("api:" in key for key in api_data.keys())

        # Test session data generator
        session_data = await warming_service._generate_session_data()
        assert isinstance(session_data, dict)
        assert len(session_data) > 0
        assert any("session:" in key for key in session_data.keys())

    async def test_metrics_retention(self, warming_service, mock_cache):
        """Test that metrics are retained with limits."""
        strategy = WarmingStrategy(
            name="retention_test", data_generator=lambda: {"key": "value"}
        )

        warming_service.register_strategy(strategy)

        # Execute strategy multiple times
        for i in range(15):  # More than the retention limit of 10
            await warming_service._execute_strategy(strategy)

        # Should only keep last 10 metrics
        metrics = warming_service.get_strategy_metrics("retention_test")
        assert len(metrics) == 10

    async def test_background_warming_disabled(self, mock_cache):
        """Test background warming when disabled."""
        service = CacheWarmingService(
            cache_backend=mock_cache, enable_background_warming=False
        )

        await service.start_background_warming()
        assert len(service.warming_tasks) == 0

    async def test_shutdown(self, warming_service):
        """Test service shutdown."""
        # Add some mock tasks
        task1 = asyncio.create_task(asyncio.sleep(1.0))  # Longer sleep to ensure proper cancellation
        task2 = asyncio.create_task(asyncio.sleep(1.0))
        warming_service.warming_tasks.add(task1)
        warming_service.warming_tasks.add(task2)

        # Shutdown service
        await warming_service.shutdown()

        # Wait for tasks to be properly cancelled using polling
        async def tasks_cancelled():
            return (task1.cancelled() or task1.done()) and (task2.cancelled() or task2.done())
        
        cancelled = await wait_for_condition(tasks_cancelled, timeout=2.0)
        assert cancelled, "Tasks were not properly cancelled within timeout"
        assert warming_service._shutdown_event.is_set()


@pytest.mark.integration
class TestCacheWarmingIntegration:
    """Integration tests for cache warming."""

    @pytest.fixture
    def integration_cache(self):
        """Mock cache for integration testing."""
        cache = AsyncMock()
        cache.set.return_value = True
        return cache

    async def test_full_warming_cycle(self, integration_cache):
        """Test complete warming cycle."""
        service = CacheWarmingService(
            cache_backend=integration_cache, enable_background_warming=False
        )

        # Execute startup warming
        startup_results = await service.warm_on_startup()
        assert len(startup_results) > 0

        # Execute all strategies
        all_results = await service.warm_all_strategies()
        assert len(all_results) > 0

        # Get summary
        summary = service.get_strategy_summary()
        assert len(summary) > 0

        # Shutdown
        await service.shutdown()

    async def test_performance_characteristics(self, integration_cache):
        """Test performance characteristics of warming."""
        service = CacheWarmingService(
            cache_backend=integration_cache, enable_background_warming=False
        )

        # Create large dataset strategy
        async def large_data_generator():
            return {f"key_{i}": f"value_{i}" for i in range(1000)}

        large_strategy = WarmingStrategy(
            name="large_dataset",
            batch_size=100,
            delay_between_batches=0.0001,  # Ultra-minimal delay for performance test
            data_generator=large_data_generator,
        )

        service.register_strategy(large_strategy)

        # Execute and measure performance
        start_time = datetime.utcnow()
        metrics = await service._execute_strategy(large_strategy)
        end_time = datetime.utcnow()

        # Verify performance characteristics
        assert metrics.items_warmed == 1000
        assert metrics.total_time_seconds > 0
        assert metrics.items_per_second > 0

        # Should complete within reasonable time
        total_time = (end_time - start_time).total_seconds()
        assert total_time < 10.0  # Should complete in under 10 seconds
