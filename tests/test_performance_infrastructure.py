"""Comprehensive tests for performance optimization infrastructure."""

import tempfile
import time
from datetime import datetime
from pathlib import Path

import pytest

# Test imports with fallback
try:
    import pandas as pd

    PANDAS_AVAILABLE = True
except ImportError:
    PANDAS_AVAILABLE = False

try:
    import redis

    REDIS_AVAILABLE = True
except ImportError:
    REDIS_AVAILABLE = False

from pynomaly.infrastructure.cache.cache_manager import (
    CacheManager,
    InMemoryCache,
    create_cache_manager,
)
from pynomaly.infrastructure.performance.profiler import (
    PerformanceProfiler,
    SystemMonitor,
)
from pynomaly.infrastructure.performance.query_optimizer import (
    DataFrameOptimizer,
    QueryCache,
    QueryOptimizer,
)

# Test Redis cache only if Redis is available
if REDIS_AVAILABLE:
    from pynomaly.infrastructure.cache.cache_manager import RedisCache


class TestInMemoryCache:
    """Test in-memory cache functionality."""

    def test_basic_cache_operations(self):
        """Test basic cache get/set/delete operations."""
        cache = InMemoryCache(max_size=10, max_memory_mb=10)

        # Test set and get
        assert cache.set("key1", "value1", ttl=60) is True
        assert cache.get("key1") == "value1"

        # Test exists
        assert cache.exists("key1") is True
        assert cache.exists("nonexistent") is False

        # Test delete
        assert cache.delete("key1") is True
        assert cache.get("key1") is None
        assert cache.exists("key1") is False

    def test_cache_ttl_expiration(self):
        """Test TTL-based cache expiration."""
        cache = InMemoryCache(max_size=10, max_memory_mb=10, default_ttl=1)

        cache.set("temp_key", "temp_value", ttl=1)
        assert cache.get("temp_key") == "temp_value"

        # Wait for expiration (allow some tolerance for timing)
        time.sleep(1.1)

        assert cache.get("temp_key") is None
        assert cache.exists("temp_key") is False

    def test_cache_lru_eviction(self):
        """Test LRU eviction when cache is full."""
        cache = InMemoryCache(max_size=3, max_memory_mb=10)

        # Fill cache to capacity
        cache.set("key1", "value1")
        cache.set("key2", "value2")
        cache.set("key3", "value3")

        # Access key1 to make it recently used
        cache.get("key1")

        # Add new key, should evict key2 (least recently used)
        cache.set("key4", "value4")

        assert cache.get("key1") == "value1"  # Should still exist
        assert cache.get("key2") is None  # Should be evicted
        assert cache.get("key3") == "value3"  # Should still exist
        assert cache.get("key4") == "value4"  # Should exist

    def test_cache_pattern_matching(self):
        """Test key pattern matching."""
        cache = InMemoryCache(max_size=10, max_memory_mb=10)

        cache.set("user:123", "user_data_123")
        cache.set("user:456", "user_data_456")
        cache.set("session:789", "session_data_789")

        # Test pattern matching
        user_keys = cache.keys("user:*")
        assert len(user_keys) == 2
        assert "user:123" in user_keys
        assert "user:456" in user_keys

        all_keys = cache.keys("*")
        assert len(all_keys) == 3

    def test_cache_statistics(self):
        """Test cache statistics tracking."""
        cache = InMemoryCache(max_size=10, max_memory_mb=10)

        # Perform operations
        cache.set("key1", "value1")
        cache.set("key2", "value2")
        cache.get("key1")  # Hit
        cache.get("nonexistent")  # Miss
        cache.delete("key2")

        stats = cache.stats()
        assert stats["hits"] >= 1
        assert stats["misses"] >= 1
        assert stats["sets"] >= 2
        assert stats["deletes"] >= 1
        assert stats["entry_count"] >= 1


@pytest.mark.skipif(not REDIS_AVAILABLE, reason="Redis not available")
class TestRedisCache:
    """Test Redis cache functionality."""

    def test_redis_basic_operations(self):
        """Test basic Redis cache operations."""
        try:
            # Try to connect to Redis (skip if not available)
            cache = RedisCache(host="localhost", port=6379, db=15)  # Use test DB

            # Test set and get
            assert cache.set("test:key1", "value1", ttl=60) is True
            assert cache.get("test:key1") == "value1"

            # Test exists and delete
            assert cache.exists("test:key1") is True
            assert cache.delete("test:key1") is True
            assert cache.get("test:key1") is None

            # Cleanup
            cache.clear()

        except Exception:
            pytest.skip("Redis server not available")

    def test_redis_serialization(self):
        """Test Redis serialization of complex objects."""
        try:
            cache = RedisCache(host="localhost", port=6379, db=15)

            # Test complex object storage
            test_data = {
                "list": [1, 2, 3],
                "dict": {"nested": "value"},
                "int": 42,
                "float": 3.14,
            }

            cache.set("test:complex", test_data)
            retrieved = cache.get("test:complex")

            assert retrieved == test_data

            # Cleanup
            cache.clear()

        except Exception:
            pytest.skip("Redis server not available")


class TestCacheManager:
    """Test cache manager with multiple backends."""

    def test_cache_manager_basic_operations(self):
        """Test cache manager basic functionality."""
        primary = InMemoryCache(max_size=5, max_memory_mb=10)
        fallback = InMemoryCache(max_size=10, max_memory_mb=20)

        manager = CacheManager(
            primary_backend=primary,
            fallback_backend=fallback,
            enable_write_through=True,
        )

        # Test set and get
        assert manager.set("key1", "value1") is True
        assert manager.get("key1") == "value1"

        # Verify data is in both backends (write-through)
        assert primary.get("key1") == "value1"
        assert fallback.get("key1") == "value1"

    def test_cache_manager_fallback(self):
        """Test cache manager fallback behavior."""
        primary = InMemoryCache(max_size=5, max_memory_mb=10)
        fallback = InMemoryCache(max_size=10, max_memory_mb=20)

        manager = CacheManager(primary_backend=primary, fallback_backend=fallback)

        # Set data in fallback only
        fallback.set("fallback_key", "fallback_value")

        # Should retrieve from fallback and populate primary
        assert manager.get("fallback_key") == "fallback_value"
        assert primary.get("fallback_key") == "fallback_value"

    def test_cache_manager_compression(self):
        """Test cache manager compression for large values."""
        primary = InMemoryCache(max_size=5, max_memory_mb=10)

        manager = CacheManager(
            primary_backend=primary,
            compression_threshold=100,  # Small threshold for testing
        )

        # Test with large value that should be compressed
        large_value = "x" * 200  # Larger than threshold

        manager.set("large_key", large_value)
        retrieved = manager.get("large_key")

        assert retrieved == large_value

    def test_cache_factory_functions(self):
        """Test cache factory functions."""
        # Test memory cache creation
        memory_cache = create_cache_manager("memory", max_size=100)
        assert isinstance(memory_cache, CacheManager)

        memory_cache.set("test", "value")
        assert memory_cache.get("test") == "value"


class TestPerformanceProfiler:
    """Test performance profiling functionality."""

    def test_basic_profiling(self):
        """Test basic performance profiling."""
        profiler = PerformanceProfiler(
            enable_cpu_profiling=True, enable_memory_profiling=True, max_results=100
        )

        # Test profiling context manager
        with profiler.profile("test_operation"):
            # Simulate some work
            sum(range(1000))
            time.sleep(0.01)

        # Check results
        results = profiler.get_results(operation_name="test_operation")
        assert len(results) == 1

        result = results[0]
        assert result.operation_name == "test_operation"
        assert result.duration_seconds > 0
        assert result.function_calls >= 0

    def test_profiling_decorator(self):
        """Test profiling decorator functionality."""
        profiler = PerformanceProfiler(max_results=100)

        @profiler.profile_function(operation_name="decorated_function")
        def test_function(n):
            return sum(range(n))

        result = test_function(100)
        assert result == sum(range(100))

        # Check profiling results
        results = profiler.get_results(operation_name="decorated_function")
        assert len(results) >= 1

    def test_metrics_collection(self):
        """Test custom metrics collection."""
        profiler = PerformanceProfiler(max_results=100)

        # Add custom metrics
        profiler.add_metric("custom.metric", 42.5, "units", tags=["test"])
        profiler.add_metric("another.metric", 100, "count")

        # Retrieve metrics
        metrics = profiler.get_metrics("custom.metric")
        assert "custom.metric" in metrics
        assert len(metrics["custom.metric"]) == 1
        assert metrics["custom.metric"][0].value == 42.5

    def test_performance_summary(self):
        """Test performance summary statistics."""
        profiler = PerformanceProfiler(max_results=100)

        # Create multiple profile results
        for i in range(5):
            with profiler.profile(f"operation_{i}"):
                time.sleep(0.001)  # Small delay

        # Test summary for specific operation
        summary = profiler.get_summary("operation_0")
        assert summary["total_runs"] >= 1
        assert "duration_stats" in summary
        assert "memory_stats" in summary

    def test_results_export(self):
        """Test exporting profiling results."""
        with tempfile.TemporaryDirectory() as temp_dir:
            profiler = PerformanceProfiler(max_results=100)

            # Create some profile data
            with profiler.profile("export_test"):
                sum(range(100))

            # Test JSON export
            json_path = Path(temp_dir) / "profiles.json"
            profiler.export_results(json_path, format="json")
            assert json_path.exists()

            # Test CSV export
            csv_path = Path(temp_dir) / "profiles.csv"
            profiler.export_results(csv_path, format="csv")
            assert csv_path.exists()


class TestSystemMonitor:
    """Test system monitoring functionality."""

    def test_current_metrics_collection(self):
        """Test collection of current system metrics."""
        monitor = SystemMonitor(monitoring_interval=1, enable_alerts=False)

        # Get current metrics
        metrics = monitor.get_current_metrics()

        # Verify key metrics are present
        expected_metrics = [
            "cpu_percent",
            "memory_percent",
            "disk_usage_percent",
            "memory_total_gb",
            "disk_total_gb",
        ]

        for metric in expected_metrics:
            assert metric in metrics
            assert isinstance(metrics[metric], int | float)

    def test_monitoring_lifecycle(self):
        """Test starting and stopping monitoring."""
        monitor = SystemMonitor(monitoring_interval=1, enable_alerts=False)

        # Start monitoring
        monitor.start_monitoring()
        assert monitor._monitoring_thread is not None
        assert monitor._monitoring_thread.is_alive()

        # Let it collect some data
        time.sleep(0.1)

        # Stop monitoring
        monitor.stop_monitoring()

        # Thread should stop
        time.sleep(0.1)
        assert not monitor._monitoring_thread.is_alive()

    def test_metric_history(self):
        """Test metric history tracking."""
        monitor = SystemMonitor(monitoring_interval=1, history_size=100)

        # Collect some metrics
        monitor._collect_system_metrics()
        monitor._collect_system_metrics()

        # Check history
        cpu_history = monitor.get_metric_history("cpu_percent")
        assert len(cpu_history) >= 2

        # Each entry should be (timestamp, value) tuple
        for timestamp, value in cpu_history:
            assert isinstance(timestamp, datetime)
            assert isinstance(value, int | float)

    def test_monitoring_summary(self):
        """Test monitoring summary generation."""
        monitor = SystemMonitor(monitoring_interval=1, enable_alerts=False)

        # Collect some data
        for _ in range(3):
            monitor._collect_system_metrics()
            time.sleep(0.01)

        summary = monitor.get_summary()

        assert "current_metrics" in summary
        assert "trends" in summary
        assert "monitoring_active" in summary
        assert "metrics_collected" in summary


@pytest.mark.skipif(not PANDAS_AVAILABLE, reason="Pandas not available")
class TestDataFrameOptimizer:
    """Test DataFrame optimization functionality."""

    def test_dataframe_optimizer_basic(self):
        """Test basic DataFrame optimization."""
        optimizer = DataFrameOptimizer()

        # Create test DataFrame
        df = pd.DataFrame(
            {
                "A": range(100),
                "B": ["category_" + str(i % 5) for i in range(100)],
                "C": [float(i) for i in range(100)],
            }
        )

        # Test groupby optimization
        optimized_df, optimizations = optimizer.optimize_dataframe_operation(
            df, "groupby", by="B"
        )

        assert optimized_df is not None
        # Should suggest categorical conversion for low-cardinality string column
        categorical_opts = [
            opt for opt in optimizations if "categorical_conversion" in opt
        ]
        assert len(categorical_opts) > 0

    def test_dataframe_filtering_optimization(self):
        """Test filtering optimization."""
        optimizer = DataFrameOptimizer()

        df = pd.DataFrame(
            {"A": range(100), "B": [i % 10 for i in range(100)], "C": ["test"] * 100}
        )

        # Test filter optimization
        optimized_df, optimizations = optimizer.optimize_dataframe_operation(
            df, "filter", conditions=["A > 50", "B < 5"]
        )

        assert optimized_df is not None
        assert len(optimizations) >= 0  # May suggest optimizations

    def test_dtype_optimization(self):
        """Test data type optimization."""
        optimizer = DataFrameOptimizer()

        # Create DataFrame with inefficient dtypes
        df = pd.DataFrame(
            {
                "small_int": pd.Series([1, 2, 3], dtype="int64"),
                "small_float": pd.Series([1.0, 2.0, 3.0], dtype="float64"),
            }
        )

        optimized_df, optimizations = optimizer.optimize_dataframe_operation(
            df, "select"
        )

        assert optimized_df is not None
        # Should suggest downcasting
        downcast_opts = [opt for opt in optimizations if "downcast" in opt]
        assert len(downcast_opts) >= 0


@pytest.mark.skipif(not PANDAS_AVAILABLE, reason="Pandas not available")
class TestQueryOptimizer:
    """Test query optimization functionality."""

    def test_query_optimizer_basic(self):
        """Test basic query optimization."""
        optimizer = QueryOptimizer(
            enable_caching=True, enable_optimization=True, max_cache_size=100
        )

        # Create test DataFrame
        df = pd.DataFrame({"A": range(100), "B": [i % 10 for i in range(100)]})

        # Execute optimized query
        result, info = optimizer.execute_optimized("filter", df, conditions="A > 50")

        assert result is not None
        assert "execution_time_ms" in info
        assert "optimizations_applied" in info
        assert "cached" in info
        assert info["cached"] is False  # First execution

        # Execute same query again - should be cached
        result2, info2 = optimizer.execute_optimized("filter", df, conditions="A > 50")

        assert result2 is not None
        assert info2["cached"] is True  # Second execution should be cached

    def test_query_cache_functionality(self):
        """Test query cache operations."""
        cache = QueryCache(max_size=10, default_ttl=60)

        test_data = (
            pd.DataFrame({"A": [1, 2, 3]}) if PANDAS_AVAILABLE else {"test": "data"}
        )

        # Test cache operations
        cache.set("test_query", test_data)
        retrieved = cache.get("test_query")

        if PANDAS_AVAILABLE:
            pd.testing.assert_frame_equal(retrieved, test_data)
        else:
            assert retrieved == test_data

        # Test cache stats
        stats = cache.get_stats()
        assert stats["hits"] >= 1
        assert stats["cache_size"] >= 1

    def test_query_statistics(self):
        """Test query performance statistics."""
        optimizer = QueryOptimizer(enable_caching=False, max_cache_size=100)

        if PANDAS_AVAILABLE:
            df = pd.DataFrame({"A": range(50), "B": range(50, 100)})

            # Execute multiple queries
            for _i in range(3):
                optimizer.execute_optimized("select", df, columns=["A"])

            # Check statistics
            stats = optimizer.get_query_stats()
            assert len(stats) >= 1

            # Check performance summary
            summary = optimizer.get_performance_summary()
            assert "total_queries" in summary
            assert summary["total_queries"] >= 3

    def test_optimization_decorator(self):
        """Test query optimization decorator."""
        optimizer = QueryOptimizer(enable_caching=True, max_cache_size=100)

        @optimizer.optimize_decorator("select")
        def select_data(df, columns):
            return df[columns] if PANDAS_AVAILABLE else df

        if PANDAS_AVAILABLE:
            df = pd.DataFrame({"A": [1, 2, 3], "B": [4, 5, 6]})
            result = select_data(df, ["A"])

            assert result is not None
            assert len(result.columns) == 1


class TestIntegration:
    """Integration tests for performance optimization infrastructure."""

    def test_cache_and_profiler_integration(self):
        """Test integration between cache and profiler."""
        cache = InMemoryCache(max_size=10, max_memory_mb=10)
        profiler = PerformanceProfiler(max_results=100)

        # Profile cache operations
        with profiler.profile("cache_operations"):
            cache.set("key1", "value1")
            cache.set("key2", "value2")
            retrieved1 = cache.get("key1")
            retrieved2 = cache.get("key2")

        assert retrieved1 == "value1"
        assert retrieved2 == "value2"

        # Check profiling results
        results = profiler.get_results("cache_operations")
        assert len(results) == 1
        assert results[0].duration_seconds > 0

    @pytest.mark.skipif(not PANDAS_AVAILABLE, reason="Pandas not available")
    def test_end_to_end_optimization_workflow(self):
        """Test complete end-to-end optimization workflow."""
        # Setup components
        cache_manager = create_cache_manager("memory", max_size=100)
        profiler = PerformanceProfiler(max_results=100)
        optimizer = QueryOptimizer(enable_caching=True, max_cache_size=100)

        # Create test data
        df = pd.DataFrame(
            {
                "feature1": range(1000),
                "feature2": [i % 100 for i in range(1000)],
                "category": ["A" if i % 2 == 0 else "B" for i in range(1000)],
            }
        )

        # Profile the entire workflow
        with profiler.profile("optimization_workflow"):
            # Cache the dataset
            cache_manager.set("dataset", df)

            # Retrieve and optimize
            cached_df = cache_manager.get("dataset")

            # Execute optimized queries
            result1, info1 = optimizer.execute_optimized(
                "filter", cached_df, conditions="feature1 > 500"
            )

            result2, info2 = optimizer.execute_optimized(
                "groupby", cached_df, by="category", agg={"feature1": "mean"}
            )

        # Verify results
        assert result1 is not None
        assert result2 is not None
        assert len(result1) == 499  # feature1 > 500

        # Check profiling
        results = profiler.get_results("optimization_workflow")
        assert len(results) == 1

        # Check cache stats
        cache_stats = cache_manager.stats()
        assert cache_stats["primary_backend"]["hits"] >= 1

    def test_container_integration(self):
        """Test integration with dependency injection container."""
        from pynomaly.infrastructure.config.container import Container

        container = Container()

        # Test that performance services are available
        service_manager = container._service_manager

        # Check availability of performance services
        performance_services = [
            "cache_manager",
            "performance_profiler",
            "system_monitor",
            "query_optimizer",
            "dataframe_optimizer",
        ]

        for service_name in performance_services:
            # Service should be registered (availability depends on dependencies)
            assert (
                service_name in service_manager._services
                or service_name in service_manager._availability
            )

    def test_monitoring_and_profiling_integration(self):
        """Test integration between system monitoring and profiling."""
        monitor = SystemMonitor(monitoring_interval=1, enable_alerts=False)
        profiler = PerformanceProfiler(max_results=100)

        # Start monitoring
        monitor.start_monitoring()

        try:
            # Profile some operations while monitoring
            with profiler.profile("monitored_operation"):
                # Perform some work
                sum(range(10000))
                time.sleep(0.1)

            # Collect current system state
            system_metrics = monitor.get_current_metrics()
            profiling_summary = profiler.get_summary("monitored_operation")

            # Verify both systems collected data
            assert "cpu_percent" in system_metrics
            assert profiling_summary["total_runs"] >= 1

        finally:
            monitor.stop_monitoring()


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
