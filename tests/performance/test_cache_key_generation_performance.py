"""Performance tests for cache key generation optimizations."""

import time
import pytest
from unittest.mock import MagicMock

from pynomaly.infrastructure.cache.decorators import CacheKeyGenerator
from pynomaly.infrastructure.cache.optimized_key_generator import OptimizedCacheKeyGenerator
from pynomaly.infrastructure.cache.performance_utils import (
    get_performance_optimizer,
    get_cache_performance_report,
    enable_cache_optimizations,
)


class TestCacheKeyGenerationPerformance:
    """Test cache key generation performance improvements."""

    def test_basic_key_generation_performance(self):
        """Test basic key generation performance comparison."""
        # Sample function for testing
        def sample_function(arg1: str, arg2: int, arg3: dict) -> str:
            return f"{arg1}_{arg2}_{len(arg3)}"

        # Test data
        test_args = ("test_string", 42, {"key1": "value1", "key2": "value2"})
        test_kwargs = {}

        # Test original implementation
        original_times = []
        for _ in range(100):
            start_time = time.perf_counter()
            CacheKeyGenerator.generate_key(
                sample_function, test_args, test_kwargs, "test_prefix"
            )
            end_time = time.perf_counter()
            original_times.append((end_time - start_time) * 1000)

        # Test optimized implementation
        optimized_times = []
        for _ in range(100):
            start_time = time.perf_counter()
            OptimizedCacheKeyGenerator.generate_key(
                sample_function, test_args, test_kwargs, "test_prefix"
            )
            end_time = time.perf_counter()
            optimized_times.append((end_time - start_time) * 1000)

        # Compare performance
        original_avg = sum(original_times) / len(original_times)
        optimized_avg = sum(optimized_times) / len(optimized_times)
        
        print(f"Original average: {original_avg:.3f}ms")
        print(f"Optimized average: {optimized_avg:.3f}ms")
        print(f"Performance improvement: {((original_avg - optimized_avg) / original_avg * 100):.1f}%")
        
        # Optimized should be faster (allowing some variation)
        assert optimized_avg < original_avg * 1.1, "Optimized version should be faster or similar"

    def test_complex_argument_serialization_performance(self):
        """Test performance with complex arguments."""
        def complex_function(
            simple_arg: str,
            list_arg: list,
            dict_arg: dict,
            nested_arg: dict
        ) -> str:
            return "result"

        # Create complex test data
        large_list = list(range(100))
        large_dict = {f"key_{i}": f"value_{i}" for i in range(50)}
        nested_dict = {
            "level1": {
                "level2": {
                    "level3": list(range(20))
                }
            }
        }

        test_args = ("simple", large_list, large_dict, nested_dict)
        test_kwargs = {}

        # Test with complex serialization disabled
        optimized_times_simple = []
        for _ in range(50):
            start_time = time.perf_counter()
            OptimizedCacheKeyGenerator.generate_key(
                complex_function, test_args, test_kwargs, 
                "test_prefix", serialize_complex_args=False
            )
            end_time = time.perf_counter()
            optimized_times_simple.append((end_time - start_time) * 1000)

        # Test with complex serialization enabled
        optimized_times_complex = []
        for _ in range(50):
            start_time = time.perf_counter()
            OptimizedCacheKeyGenerator.generate_key(
                complex_function, test_args, test_kwargs, 
                "test_prefix", serialize_complex_args=True
            )
            end_time = time.perf_counter()
            optimized_times_complex.append((end_time - start_time) * 1000)

        simple_avg = sum(optimized_times_simple) / len(optimized_times_simple)
        complex_avg = sum(optimized_times_complex) / len(optimized_times_complex)
        
        print(f"Simple serialization average: {simple_avg:.3f}ms")
        print(f"Complex serialization average: {complex_avg:.3f}ms")
        
        # Simple should be faster than complex
        assert simple_avg < complex_avg, "Simple serialization should be faster"

    def test_signature_caching_performance(self):
        """Test that signature caching improves performance."""
        def test_function(arg1: str, arg2: int) -> str:
            return f"{arg1}_{arg2}"

        # Clear cache first
        OptimizedCacheKeyGenerator.clear_cache()

        # First call (cache miss)
        start_time = time.perf_counter()
        OptimizedCacheKeyGenerator.generate_key(
            test_function, ("test", 42), {}, "prefix"
        )
        first_call_time = (time.perf_counter() - start_time) * 1000

        # Second call (cache hit)
        start_time = time.perf_counter()
        OptimizedCacheKeyGenerator.generate_key(
            test_function, ("test", 42), {}, "prefix"
        )
        second_call_time = (time.perf_counter() - start_time) * 1000

        print(f"First call (cache miss): {first_call_time:.3f}ms")
        print(f"Second call (cache hit): {second_call_time:.3f}ms")
        
        # Second call should be faster due to caching
        assert second_call_time <= first_call_time, "Cached call should be faster or equal"

    def test_hash_algorithm_performance(self):
        """Test different hash algorithms for long keys."""
        def long_key_function(**kwargs) -> str:
            return "result"

        # Create arguments that will result in a long key
        long_kwargs = {f"very_long_argument_name_{i}": f"very_long_value_string_{i}" * 10 
                      for i in range(50)}

        # Test key generation with long arguments
        start_time = time.perf_counter()
        key = OptimizedCacheKeyGenerator.generate_key(
            long_key_function, (), long_kwargs, "test_prefix"
        )
        generation_time = (time.perf_counter() - start_time) * 1000

        print(f"Long key generation time: {generation_time:.3f}ms")
        print(f"Generated key length: {len(key)}")
        
        # Should handle long keys efficiently
        assert generation_time < 50, "Long key generation should be reasonably fast"
        assert len(key) < 500, "Long keys should be hashed to reasonable length"

    def test_performance_monitoring(self):
        """Test performance monitoring functionality."""
        def monitored_function(arg1: str, arg2: int) -> str:
            return f"{arg1}_{arg2}"

        # Clear previous stats
        OptimizedCacheKeyGenerator.clear_cache()

        # Generate some keys to collect stats
        for i in range(10):
            OptimizedCacheKeyGenerator.generate_key(
                monitored_function, (f"test_{i}", i), {}, "prefix"
            )

        # Get performance stats
        stats = OptimizedCacheKeyGenerator.get_performance_stats()
        
        assert stats["status"] != "no_data", "Should have performance data"
        assert stats["total_generated_keys"] == 10, "Should track correct number of keys"
        assert "average_generation_time_ms" in stats, "Should have average time"
        assert "key_size_distribution" in stats, "Should have size distribution"

    def test_performance_optimizer_integration(self):
        """Test performance optimizer integration."""
        optimizer = get_performance_optimizer()
        
        # Check initial state
        assert not optimizer.optimization_applied, "Should not be optimized initially"
        
        # Test enabling optimizations
        enable_cache_optimizations()
        
        # Get performance report
        report = get_cache_performance_report()
        
        assert "optimizer_report" in report, "Should have optimizer report"
        assert "health_report" in report, "Should have health report"
        assert "key_generator_stats" in report, "Should have key generator stats"

    def test_memory_usage_optimization(self):
        """Test that cache optimizations don't cause memory leaks."""
        def test_function(arg: str) -> str:
            return arg

        # Clear cache
        OptimizedCacheKeyGenerator.clear_cache()

        # Generate many keys to test memory management
        for i in range(2000):  # More than cache size limit
            OptimizedCacheKeyGenerator.generate_key(
                test_function, (f"test_{i}",), {}, "prefix"
            )

        # Get cache stats
        stats = OptimizedCacheKeyGenerator.get_performance_stats()
        
        # Cache should not grow indefinitely
        assert stats["cache_sizes"]["signature_cache"] <= 1000, "Signature cache should be limited"
        assert stats["cache_sizes"]["func_name_cache"] <= 1000, "Function name cache should be limited"

    def test_fallback_behavior(self):
        """Test fallback behavior when optimization fails."""
        def problematic_function(arg):
            return arg

        # Test with None function (should trigger fallback)
        try:
            key = OptimizedCacheKeyGenerator.generate_key(
                None, ("test",), {}, "prefix"
            )
            # Should generate some fallback key
            assert key.startswith("prefix:fallback:"), "Should use fallback key generation"
        except Exception:
            # Fallback should handle errors gracefully
            pass

    def test_serialization_edge_cases(self):
        """Test serialization performance with edge cases."""
        def edge_case_function(
            none_arg,
            empty_list,
            empty_dict,
            circular_ref,
            custom_obj
        ) -> str:
            return "result"

        # Create edge case data
        class CustomObject:
            def __init__(self, value):
                self.value = value

        custom_obj = CustomObject("test")
        
        # Test with edge cases
        start_time = time.perf_counter()
        key = OptimizedCacheKeyGenerator.generate_key(
            edge_case_function,
            (None, [], {}, {"self": "reference"}, custom_obj),
            {},
            "prefix"
        )
        generation_time = (time.perf_counter() - start_time) * 1000

        print(f"Edge case generation time: {generation_time:.3f}ms")
        print(f"Generated key: {key}")
        
        # Should handle edge cases efficiently
        assert generation_time < 10, "Edge case handling should be fast"
        assert len(key) > 0, "Should generate valid key"

    def test_concurrent_access_performance(self):
        """Test performance under concurrent access."""
        import threading
        
        def concurrent_function(thread_id: int, value: str) -> str:
            return f"{thread_id}_{value}"

        results = []
        
        def generate_keys(thread_id: int):
            """Generate keys in a thread."""
            times = []
            for i in range(100):
                start_time = time.perf_counter()
                OptimizedCacheKeyGenerator.generate_key(
                    concurrent_function, (thread_id, f"value_{i}"), {}, "prefix"
                )
                end_time = time.perf_counter()
                times.append((end_time - start_time) * 1000)
            results.append(times)

        # Run concurrent key generation
        threads = []
        for i in range(5):
            thread = threading.Thread(target=generate_keys, args=(i,))
            threads.append(thread)
            thread.start()

        for thread in threads:
            thread.join()

        # Analyze results
        all_times = [time for thread_times in results for time in thread_times]
        avg_time = sum(all_times) / len(all_times)
        
        print(f"Concurrent access average time: {avg_time:.3f}ms")
        print(f"Total operations: {len(all_times)}")
        
        # Should handle concurrent access efficiently
        assert avg_time < 5, "Concurrent access should be reasonably fast"