#!/usr/bin/env python3
"""
Tests for the Performance Optimization Engine
"""

import time

import numpy as np
import pandas as pd
import pytest

from monorepo.infrastructure.performance.optimization_engine import (
    CacheKey,
    IntelligentCache,
    MemoryOptimizer,
    OptimizationConfig,
    ParallelExecutor,
    PerformanceOptimizationEngine,
    create_optimization_engine,
)


class TestCacheKey:
    """Test cache key generation."""

    def test_simple_key_generation(self):
        """Test simple key generation."""
        key = CacheKey.generate_key("test_func", (1, 2, 3), {"a": 1, "b": 2})
        assert isinstance(key, str)
        assert len(key) == 64  # SHA256 hash length

    def test_dataframe_key_generation(self):
        """Test key generation with DataFrame."""
        df = pd.DataFrame({"a": [1, 2, 3], "b": [4, 5, 6]})
        key = CacheKey.generate_key("test_func", (df,), {})
        assert isinstance(key, str)
        assert len(key) == 64

    def test_numpy_array_key_generation(self):
        """Test key generation with numpy array."""
        arr = np.array([1, 2, 3, 4, 5])
        key = CacheKey.generate_key("test_func", (arr,), {})
        assert isinstance(key, str)
        assert len(key) == 64

    def test_consistent_key_generation(self):
        """Test that same inputs generate same key."""
        key1 = CacheKey.generate_key("test_func", (1, 2, 3), {"a": 1})
        key2 = CacheKey.generate_key("test_func", (1, 2, 3), {"a": 1})
        assert key1 == key2

    def test_different_inputs_different_keys(self):
        """Test that different inputs generate different keys."""
        key1 = CacheKey.generate_key("test_func", (1, 2, 3), {"a": 1})
        key2 = CacheKey.generate_key("test_func", (1, 2, 3), {"a": 2})
        assert key1 != key2


class TestIntelligentCache:
    """Test intelligent cache implementation."""

    def test_cache_initialization(self):
        """Test cache initialization."""
        config = OptimizationConfig(cache_size_mb=1)
        cache = IntelligentCache(config)
        assert cache.config == config
        assert cache._cache_size == 0

    def test_cache_set_and_get(self):
        """Test basic cache operations."""
        config = OptimizationConfig(cache_size_mb=1)
        cache = IntelligentCache(config)

        # Set value
        result = cache.set("test_key", "test_value")
        assert result is True

        # Get value
        value = cache.get("test_key")
        assert value == "test_value"

    def test_cache_ttl_expiration(self):
        """Test TTL expiration."""
        config = OptimizationConfig(cache_size_mb=1, cache_ttl_seconds=1)
        cache = IntelligentCache(config)

        # Set value
        cache.set("test_key", "test_value")

        # Should be available immediately
        assert cache.get("test_key") == "test_value"

        # Wait for TTL expiration
        time.sleep(1.1)

        # Should be expired
        assert cache.get("test_key") is None

    def test_cache_eviction_lru(self):
        """Test LRU eviction."""
        config = OptimizationConfig(cache_size_mb=1, cache_strategy="lru")
        cache = IntelligentCache(config)

        # Fill cache beyond capacity with larger items
        for i in range(20):
            cache.set(f"key_{i}", f"value_{i}", size_hint=100000)  # 100KB each

        # First keys should be evicted due to size constraints
        first_key_exists = cache.get("key_0") is not None
        last_key_exists = cache.get("key_19") is not None

        # At least some eviction should have occurred
        assert not (
            first_key_exists and last_key_exists
        ), "Cache should have evicted some items"

    def test_cache_clear(self):
        """Test cache clearing."""
        config = OptimizationConfig(cache_size_mb=1)
        cache = IntelligentCache(config)

        cache.set("test_key", "test_value")
        assert cache.get("test_key") == "test_value"

        cache.clear()
        assert cache.get("test_key") is None
        assert cache._cache_size == 0

    def test_cache_stats(self):
        """Test cache statistics."""
        config = OptimizationConfig(cache_size_mb=1)
        cache = IntelligentCache(config)

        cache.set("test_key", "test_value")
        stats = cache.get_stats()

        assert isinstance(stats, dict)
        assert "total_entries" in stats
        assert "total_size_mb" in stats
        assert stats["total_entries"] == 1


class TestMemoryOptimizer:
    """Test memory optimization utilities."""

    def test_memory_optimizer_initialization(self):
        """Test memory optimizer initialization."""
        config = OptimizationConfig()
        optimizer = MemoryOptimizer(config)
        assert optimizer.config == config

    def test_object_pooling(self):
        """Test object pooling."""
        config = OptimizationConfig(object_pooling=True)
        optimizer = MemoryOptimizer(config)

        # Get object from pool
        obj = optimizer.get_pooled_object(list)
        assert isinstance(obj, list)

        # Return to pool
        optimizer.return_to_pool(obj)

    def test_memory_pressure_check(self):
        """Test memory pressure detection."""
        config = OptimizationConfig(memory_threshold_mb=1)
        optimizer = MemoryOptimizer(config)

        # Should return boolean
        pressure = optimizer.check_memory_pressure()
        assert isinstance(pressure, bool)

    def test_dataframe_optimization(self):
        """Test DataFrame memory optimization."""
        config = OptimizationConfig(enable_memory_optimization=True)
        optimizer = MemoryOptimizer(config)

        # Create DataFrame with inefficient types
        df = pd.DataFrame(
            {
                "small_int": [1, 2, 3, 4, 5],
                "float_col": [1.1, 2.2, 3.3, 4.4, 5.5],
                "category": ["A", "B", "A", "B", "A"],
            }
        )

        optimized_df = optimizer.optimize_dataframe(df)

        # Check that optimization was applied
        assert isinstance(optimized_df, pd.DataFrame)
        assert len(optimized_df) == len(df)

    def test_gc_trigger(self):
        """Test garbage collection trigger."""
        config = OptimizationConfig(gc_threshold=1)
        optimizer = MemoryOptimizer(config)

        # Should not raise exception
        optimizer.trigger_gc_if_needed()
        optimizer.trigger_gc_if_needed()  # Should trigger GC


class TestParallelExecutor:
    """Test parallel execution engine."""

    def test_parallel_executor_initialization(self):
        """Test parallel executor initialization."""
        config = OptimizationConfig()
        executor = ParallelExecutor(config)
        assert executor.config == config

    @pytest.mark.asyncio
    async def test_parallel_execution(self):
        """Test parallel execution."""
        config = OptimizationConfig(max_workers=2)
        executor = ParallelExecutor(config)

        def square(x):
            return x * x

        items = [1, 2, 3, 4, 5]
        results = await executor.execute_parallel(square, items)

        assert results == [1, 4, 9, 16, 25]

    @pytest.mark.asyncio
    async def test_concurrent_execution(self):
        """Test concurrent task execution."""
        config = OptimizationConfig(max_workers=2)
        executor = ParallelExecutor(config)

        def task1():
            return "task1"

        def task2():
            return "task2"

        tasks = [task1, task2]
        results = await executor.execute_concurrent(tasks)

        assert "task1" in results
        assert "task2" in results

    def test_executor_cleanup(self):
        """Test executor cleanup."""
        config = OptimizationConfig()
        executor = ParallelExecutor(config)

        # Should not raise exception
        executor.cleanup()


class TestPerformanceOptimizationEngine:
    """Test the main performance optimization engine."""

    def test_engine_initialization(self):
        """Test engine initialization."""
        config = OptimizationConfig()
        engine = PerformanceOptimizationEngine(config)

        assert engine.config == config
        assert engine.cache is not None
        assert engine.batch_processor is not None
        assert engine.parallel_executor is not None
        assert engine.memory_optimizer is not None

    def test_cached_decorator_sync(self):
        """Test cached decorator for sync functions."""
        config = OptimizationConfig(enable_caching=True)
        engine = PerformanceOptimizationEngine(config)

        call_count = 0

        @engine.cached()
        def expensive_function(x):
            nonlocal call_count
            call_count += 1
            return x * 2

        # First call
        result1 = expensive_function(5)
        assert result1 == 10
        assert call_count == 1

        # Second call - should use cache
        result2 = expensive_function(5)
        assert result2 == 10
        assert call_count == 1  # Should not increase

    @pytest.mark.asyncio
    async def test_cached_decorator_async(self):
        """Test cached decorator for async functions."""
        config = OptimizationConfig(enable_caching=True)
        engine = PerformanceOptimizationEngine(config)

        call_count = 0

        @engine.cached()
        async def async_expensive_function(x):
            nonlocal call_count
            call_count += 1
            return x * 2

        # First call
        result1 = await async_expensive_function(5)
        assert result1 == 10
        assert call_count == 1

        # Second call - should use cache
        result2 = await async_expensive_function(5)
        assert result2 == 10
        assert call_count == 1  # Should not increase

    def test_memory_optimized_decorator(self):
        """Test memory optimized decorator."""
        config = OptimizationConfig(enable_memory_optimization=True)
        engine = PerformanceOptimizationEngine(config)

        @engine.memory_optimized()
        def process_dataframe(df):
            return df.copy()

        # Create test DataFrame
        df = pd.DataFrame({"a": [1, 2, 3], "b": [4, 5, 6]})
        result = process_dataframe(df)

        assert isinstance(result, pd.DataFrame)
        assert len(result) == len(df)

    @pytest.mark.asyncio
    async def test_parallel_decorator(self):
        """Test parallel decorator."""
        config = OptimizationConfig(enable_parallel_processing=True)
        engine = PerformanceOptimizationEngine(config)

        @engine.parallel()
        def process_item(item):  # Changed to sync function
            return item * 2

        items = [1, 2, 3, 4, 5]
        results = await process_item(items)

        assert results == [2, 4, 6, 8, 10]

    def test_optimization_stats(self):
        """Test optimization statistics."""
        config = OptimizationConfig()
        engine = PerformanceOptimizationEngine(config)

        stats = engine.get_optimization_stats()

        assert isinstance(stats, dict)
        assert "cache_stats" in stats
        assert "optimization_counters" in stats
        assert "memory_status" in stats
        assert "parallel_execution" in stats

    def test_engine_cleanup(self):
        """Test engine cleanup."""
        config = OptimizationConfig()
        engine = PerformanceOptimizationEngine(config)

        # Should not raise exception
        engine.cleanup()


class TestOptimizationEngineFactory:
    """Test optimization engine factory function."""

    def test_create_default_engine(self):
        """Test creating engine with defaults."""
        engine = create_optimization_engine()

        assert isinstance(engine, PerformanceOptimizationEngine)
        assert engine.config.cache_size_mb == 512
        assert engine.config.enable_caching is True

    def test_create_custom_engine(self):
        """Test creating engine with custom settings."""
        engine = create_optimization_engine(
            cache_size_mb=256, max_workers=4, enable_all_optimizations=False
        )

        assert isinstance(engine, PerformanceOptimizationEngine)
        assert engine.config.cache_size_mb == 256
        assert engine.config.max_workers == 4
        assert engine.config.enable_caching is False


class TestIntegrationScenarios:
    """Test integration scenarios."""

    @pytest.mark.asyncio
    async def test_combined_optimizations(self):
        """Test combining multiple optimizations."""
        config = OptimizationConfig(
            enable_caching=True,
            enable_memory_optimization=True,
            enable_parallel_processing=True,
            cache_size_mb=1,
        )
        engine = PerformanceOptimizationEngine(config)

        call_count = 0

        @engine.cached()
        @engine.memory_optimized()
        async def optimized_function(data):
            nonlocal call_count
            call_count += 1
            if isinstance(data, pd.DataFrame):
                return data.sum().sum()
            return sum(data)

        # Test with list
        result1 = await optimized_function([1, 2, 3, 4, 5])
        assert result1 == 15
        assert call_count == 1

        # Test with cache hit
        result2 = await optimized_function([1, 2, 3, 4, 5])
        assert result2 == 15
        assert call_count == 1  # Should use cache

        # Test with DataFrame
        df = pd.DataFrame({"a": [1, 2, 3], "b": [4, 5, 6]})
        result3 = await optimized_function(df)
        assert result3 == 21
        assert call_count == 2

    def test_performance_monitoring(self):
        """Test performance monitoring integration."""
        config = OptimizationConfig()
        engine = PerformanceOptimizationEngine(config)

        # Test that stats are collected
        stats = engine.get_optimization_stats()
        assert "cache_stats" in stats

        # Test cache hit/miss tracking
        engine.optimization_stats["cache_hits"] = 5
        engine.optimization_stats["cache_misses"] = 2

        stats = engine.get_optimization_stats()
        assert stats["optimization_counters"]["cache_hits"] == 5
        assert stats["optimization_counters"]["cache_misses"] == 2

    @pytest.mark.asyncio
    async def test_error_handling(self):
        """Test error handling in optimizations."""
        config = OptimizationConfig()
        engine = PerformanceOptimizationEngine(config)

        @engine.cached()
        async def failing_function(x):
            if x < 0:
                raise ValueError("Negative value")
            return x * 2

        # Test successful call
        result = await failing_function(5)
        assert result == 10

        # Test error handling
        with pytest.raises(ValueError):
            await failing_function(-1)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
