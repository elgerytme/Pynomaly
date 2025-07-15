"""Performance and load testing for enhanced Redis cache implementation."""

import asyncio
import pytest
import time
from concurrent.futures import ThreadPoolExecutor
from statistics import mean, stdev
from typing import List, Dict, Any
from unittest.mock import Mock, patch

from src.pynomaly.infrastructure.cache.enhanced_redis_cache import (
    CacheConfiguration,
    EnhancedRedisCache,
    InvalidationStrategy,
    EvictionPolicy,
)


class PerformanceTestRunner:
    """Performance test runner with metrics collection."""
    
    def __init__(self):
        self.results = []
        self.metrics = {}
    
    def record_operation(self, operation: str, duration: float, success: bool):
        """Record operation metrics."""
        self.results.append({
            'operation': operation,
            'duration': duration,
            'success': success,
            'timestamp': time.time()
        })
    
    def calculate_metrics(self) -> Dict[str, Any]:
        """Calculate performance metrics."""
        if not self.results:
            return {}
        
        # Group by operation type
        operations = {}
        for result in self.results:
            op = result['operation']
            if op not in operations:
                operations[op] = []
            operations[op].append(result)
        
        metrics = {}
        for op, results in operations.items():
            durations = [r['duration'] for r in results]
            success_count = sum(1 for r in results if r['success'])
            
            metrics[op] = {
                'total_operations': len(results),
                'success_rate': success_count / len(results),
                'avg_duration_ms': mean(durations) * 1000,
                'min_duration_ms': min(durations) * 1000,
                'max_duration_ms': max(durations) * 1000,
                'std_deviation_ms': stdev(durations) * 1000 if len(durations) > 1 else 0,
                'operations_per_second': len(results) / (max(r['timestamp'] for r in results) - min(r['timestamp'] for r in results)) if len(results) > 1 else 0
            }
        
        return metrics
    
    def print_report(self):
        """Print performance report."""
        metrics = self.calculate_metrics()
        
        print("\n" + "="*80)
        print("CACHE PERFORMANCE TEST REPORT")
        print("="*80)
        
        for operation, stats in metrics.items():
            print(f"\n{operation.upper()} OPERATIONS:")
            print(f"  Total Operations: {stats['total_operations']:,}")
            print(f"  Success Rate: {stats['success_rate']:.2%}")
            print(f"  Average Duration: {stats['avg_duration_ms']:.2f}ms")
            print(f"  Min Duration: {stats['min_duration_ms']:.2f}ms")
            print(f"  Max Duration: {stats['max_duration_ms']:.2f}ms")
            print(f"  Std Deviation: {stats['std_deviation_ms']:.2f}ms")
            print(f"  Operations/Second: {stats['operations_per_second']:.2f}")
        
        print("\n" + "="*80)


@pytest.fixture
def mock_redis_performance():
    """Mock Redis client optimized for performance testing."""
    redis_mock = Mock()
    
    # Simulate realistic Redis response times
    def mock_get(key):
        time.sleep(0.001)  # 1ms simulated latency
        return None
    
    def mock_set(key, value):
        time.sleep(0.0015)  # 1.5ms simulated latency
        return True
    
    def mock_setex(key, ttl, value):
        time.sleep(0.0015)  # 1.5ms simulated latency
        return True
    
    def mock_delete(key):
        time.sleep(0.0012)  # 1.2ms simulated latency
        return 1
    
    def mock_exists(key):
        time.sleep(0.0008)  # 0.8ms simulated latency
        return True
    
    redis_mock.get.side_effect = mock_get
    redis_mock.set.side_effect = mock_set
    redis_mock.setex.side_effect = mock_setex
    redis_mock.delete.side_effect = mock_delete
    redis_mock.exists.side_effect = mock_exists
    redis_mock.ping.return_value = True
    redis_mock.flushdb.return_value = True
    redis_mock.info.return_value = {
        'redis_version': '6.2.0',
        'used_memory_human': '10M',
        'connected_clients': 100,
        'keyspace_hits': 1000,
        'keyspace_misses': 200
    }
    
    return redis_mock


class TestCachePerformance:
    """Cache performance tests."""
    
    @pytest.mark.asyncio
    async def test_sequential_operations_performance(self, mock_redis_performance):
        """Test performance of sequential cache operations."""
        config = CacheConfiguration(
            enable_compression=True,
            enable_monitoring=True
        )
        
        runner = PerformanceTestRunner()
        
        with patch('redis.from_url', return_value=mock_redis_performance):
            cache = EnhancedRedisCache(config)
            
            # Test sequential SET operations
            num_operations = 1000
            test_data = {"key": "value", "number": 42}
            
            print(f"\nTesting {num_operations} sequential SET operations...")
            
            for i in range(num_operations):
                start_time = time.time()
                success = await cache.set(f"key_{i}", test_data, ttl=3600)
                duration = time.time() - start_time
                runner.record_operation("set", duration, success)
            
            # Test sequential GET operations
            print(f"Testing {num_operations} sequential GET operations...")
            
            for i in range(num_operations):
                start_time = time.time()
                result = await cache.get(f"key_{i}")
                duration = time.time() - start_time
                runner.record_operation("get", duration, result is not None)
            
            # Test sequential DELETE operations
            print(f"Testing {num_operations} sequential DELETE operations...")
            
            for i in range(num_operations):
                start_time = time.time()
                success = await cache.delete(f"key_{i}")
                duration = time.time() - start_time
                runner.record_operation("delete", duration, success)
            
            await cache.close()
        
        runner.print_report()
        
        # Verify performance benchmarks
        metrics = runner.calculate_metrics()
        
        # SET operations should be fast
        assert metrics['set']['avg_duration_ms'] < 10.0  # Less than 10ms average
        assert metrics['set']['success_rate'] > 0.99  # 99%+ success rate
        
        # GET operations should be faster
        assert metrics['get']['avg_duration_ms'] < 5.0  # Less than 5ms average
        
        # DELETE operations should be fast
        assert metrics['delete']['avg_duration_ms'] < 8.0  # Less than 8ms average
    
    @pytest.mark.asyncio
    async def test_concurrent_operations_performance(self, mock_redis_performance):
        """Test performance under concurrent load."""
        config = CacheConfiguration(
            enable_compression=True,
            enable_monitoring=True,
            connection_pool_size=50
        )
        
        runner = PerformanceTestRunner()
        
        with patch('redis.from_url', return_value=mock_redis_performance):
            cache = EnhancedRedisCache(config)
            
            async def perform_mixed_operations(worker_id: int, operations_per_worker: int):
                """Perform mixed cache operations for a single worker."""
                for i in range(operations_per_worker):
                    key = f"worker_{worker_id}_key_{i}"
                    test_data = {"worker": worker_id, "operation": i}
                    
                    # SET operation
                    start_time = time.time()
                    success = await cache.set(key, test_data, ttl=3600)
                    duration = time.time() - start_time
                    runner.record_operation("concurrent_set", duration, success)
                    
                    # GET operation
                    start_time = time.time()
                    result = await cache.get(key)
                    duration = time.time() - start_time
                    runner.record_operation("concurrent_get", duration, result is not None)
                    
                    # Simulate some processing time
                    await asyncio.sleep(0.001)
            
            # Run concurrent workers
            num_workers = 50
            operations_per_worker = 100
            total_operations = num_workers * operations_per_worker
            
            print(f"\nTesting {total_operations} concurrent operations with {num_workers} workers...")
            
            start_time = time.time()
            
            # Create and run concurrent tasks
            tasks = []
            for worker_id in range(num_workers):
                task = asyncio.create_task(perform_mixed_operations(worker_id, operations_per_worker))
                tasks.append(task)
            
            await asyncio.gather(*tasks)
            
            total_duration = time.time() - start_time
            
            await cache.close()
        
        runner.print_report()
        
        print(f"\nCONCURRENCY METRICS:")
        print(f"Total Time: {total_duration:.2f}s")
        print(f"Total Operations: {total_operations * 2:,}")  # SET + GET
        print(f"Overall Throughput: {(total_operations * 2) / total_duration:.2f} ops/sec")
        
        # Verify concurrent performance
        metrics = runner.calculate_metrics()
        
        assert metrics['concurrent_set']['success_rate'] > 0.98  # 98%+ success rate
        assert metrics['concurrent_get']['success_rate'] > 0.98  # 98%+ success rate
        
        # Throughput should be reasonable
        overall_throughput = (total_operations * 2) / total_duration
        assert overall_throughput > 1000  # At least 1000 ops/sec
    
    @pytest.mark.asyncio
    async def test_large_payload_performance(self, mock_redis_performance):
        """Test performance with large data payloads."""
        config = CacheConfiguration(
            enable_compression=True,
            compression_threshold=1024
        )
        
        runner = PerformanceTestRunner()
        
        # Create payloads of different sizes
        payloads = {
            "small": {"data": "x" * 100},  # 100 bytes
            "medium": {"data": "x" * 10000},  # 10KB
            "large": {"data": "x" * 100000},  # 100KB
            "xlarge": {"data": "x" * 1000000}  # 1MB
        }
        
        with patch('redis.from_url', return_value=mock_redis_performance):
            cache = EnhancedRedisCache(config)
            
            for size_name, payload in payloads.items():
                print(f"\nTesting {size_name} payload performance...")
                
                # Test SET operations with different payload sizes
                for i in range(50):  # Fewer operations for large payloads
                    start_time = time.time()
                    success = await cache.set(f"{size_name}_key_{i}", payload, ttl=3600)
                    duration = time.time() - start_time
                    runner.record_operation(f"set_{size_name}", duration, success)
                
                # Test GET operations
                for i in range(50):
                    start_time = time.time()
                    result = await cache.get(f"{size_name}_key_{i}")
                    duration = time.time() - start_time
                    runner.record_operation(f"get_{size_name}", duration, result is not None)
            
            await cache.close()
        
        runner.print_report()
        
        # Verify that performance degrades gracefully with payload size
        metrics = runner.calculate_metrics()
        
        # Small payloads should be fastest
        assert metrics['set_small']['avg_duration_ms'] < metrics['set_large']['avg_duration_ms']
        assert metrics['get_small']['avg_duration_ms'] < metrics['get_large']['avg_duration_ms']
        
        # All operations should still succeed
        for metric_name, metric_data in metrics.items():
            assert metric_data['success_rate'] > 0.95  # 95%+ success rate
    
    @pytest.mark.asyncio
    async def test_cache_warming_performance(self, mock_redis_performance):
        """Test cache warming performance."""
        config = CacheConfiguration(
            enable_cache_warming=True,
            enable_monitoring=True
        )
        
        runner = PerformanceTestRunner()
        
        with patch('redis.from_url', return_value=mock_redis_performance):
            cache = EnhancedRedisCache(config)
            
            # Setup cache warming patterns
            async def sample_loader():
                """Sample data loader for warming."""
                await asyncio.sleep(0.01)  # Simulate loading time
                return {"warmed": True, "timestamp": time.time()}
            
            # Add warming patterns
            for i in range(100):
                await cache.cache_warmer.add_warming_pattern(
                    f"pattern_{i}",
                    sample_loader,
                    priority=i % 3
                )
            
            # Test warming performance
            keys_to_warm = [f"pattern_{i}" for i in range(100)]
            
            start_time = time.time()
            await cache.cache_warmer.warm_cache(keys_to_warm)
            warming_duration = time.time() - start_time
            
            runner.record_operation("cache_warming", warming_duration, True)
            
            await cache.close()
        
        runner.print_report()
        
        print(f"\nCACHE WARMING METRICS:")
        print(f"Keys Warmed: 100")
        print(f"Total Warming Time: {warming_duration:.2f}s")
        print(f"Average Time per Key: {(warming_duration / 100) * 1000:.2f}ms")
        
        # Verify warming performance
        assert warming_duration < 5.0  # Should complete within 5 seconds
    
    @pytest.mark.asyncio
    async def test_memory_usage_performance(self, mock_redis_performance):
        """Test memory usage and performance correlation."""
        config = CacheConfiguration(
            enable_monitoring=True,
            max_memory_cache_size=1000
        )
        
        runner = PerformanceTestRunner()
        
        with patch('redis.from_url', return_value=mock_redis_performance):
            cache = EnhancedRedisCache(config)
            
            # Gradually increase memory usage and measure performance
            for batch in range(10):
                batch_size = 100
                print(f"\nTesting batch {batch + 1} (total keys: {(batch + 1) * batch_size})")
                
                # Add data to increase memory usage
                for i in range(batch_size):
                    key = f"memory_test_{batch}_{i}"
                    # Vary data size
                    data_size = 1000 + (batch * 100)  # Increasing data size
                    test_data = {"data": "x" * data_size, "batch": batch}
                    
                    start_time = time.time()
                    success = await cache.set(key, test_data, ttl=3600)
                    duration = time.time() - start_time
                    runner.record_operation(f"memory_set_batch_{batch}", duration, success)
                
                # Test read performance at current memory level
                for i in range(batch_size // 2):  # Sample reads
                    key = f"memory_test_{batch}_{i}"
                    start_time = time.time()
                    result = await cache.get(key)
                    duration = time.time() - start_time
                    runner.record_operation(f"memory_get_batch_{batch}", duration, result is not None)
                
                # Get current cache info
                info = await cache.get_info()
                memory_info = info.get('redis_info', {})
                print(f"Memory Usage: {memory_info.get('memory_usage', 'N/A')}")
            
            await cache.close()
        
        runner.print_report()
        
        # Analyze memory impact on performance
        metrics = runner.calculate_metrics()
        
        # Performance should not degrade significantly with memory usage
        set_metrics = {k: v for k, v in metrics.items() if k.startswith('memory_set')}
        get_metrics = {k: v for k, v in metrics.items() if k.startswith('memory_get')}
        
        # Check that performance remains reasonable
        for metric_name, metric_data in set_metrics.items():
            assert metric_data['avg_duration_ms'] < 20.0  # Less than 20ms
            assert metric_data['success_rate'] > 0.95  # 95%+ success rate
    
    @pytest.mark.asyncio
    async def test_invalidation_performance(self, mock_redis_performance):
        """Test cache invalidation performance."""
        config = CacheConfiguration(
            invalidation_strategy=InvalidationStrategy.IMMEDIATE
        )
        
        runner = PerformanceTestRunner()
        
        with patch('redis.from_url', return_value=mock_redis_performance):
            cache = EnhancedRedisCache(config)
            
            # Set up data for invalidation testing
            num_keys = 1000
            test_data = {"value": "test_data"}
            
            # Populate cache
            for i in range(num_keys):
                await cache.set(f"invalidation_key_{i}", test_data, ttl=3600)
            
            # Test pattern-based invalidation
            start_time = time.time()
            await cache.invalidate_by_pattern("invalidation_key_*")
            pattern_invalidation_duration = time.time() - start_time
            
            runner.record_operation("pattern_invalidation", pattern_invalidation_duration, True)
            
            # Test tag-based invalidation
            # First, set data with tags
            for i in range(num_keys // 2):
                await cache.set(f"tagged_key_{i}", test_data, ttl=3600, tags={"test_tag"})
            
            start_time = time.time()
            await cache.invalidate_by_tag("test_tag")
            tag_invalidation_duration = time.time() - start_time
            
            runner.record_operation("tag_invalidation", tag_invalidation_duration, True)
            
            await cache.close()
        
        runner.print_report()
        
        print(f"\nINVALIDATION METRICS:")
        print(f"Pattern Invalidation ({num_keys} keys): {pattern_invalidation_duration:.3f}s")
        print(f"Tag Invalidation ({num_keys // 2} keys): {tag_invalidation_duration:.3f}s")
        
        # Verify invalidation performance
        assert pattern_invalidation_duration < 1.0  # Should complete within 1 second
        assert tag_invalidation_duration < 0.5  # Should complete within 0.5 seconds
    
    @pytest.mark.asyncio
    async def test_circuit_breaker_performance_impact(self, mock_redis_performance):
        """Test circuit breaker performance impact."""
        config = CacheConfiguration(
            enable_circuit_breaker=True
        )
        
        runner = PerformanceTestRunner()
        
        with patch('redis.from_url', return_value=mock_redis_performance):
            cache = EnhancedRedisCache(config)
            
            # Test normal operations with circuit breaker
            for i in range(500):
                start_time = time.time()
                success = await cache.set(f"cb_key_{i}", {"data": "test"}, ttl=3600)
                duration = time.time() - start_time
                runner.record_operation("cb_normal_set", duration, success)
            
            # Simulate failures to trigger circuit breaker
            original_get = mock_redis_performance.get
            mock_redis_performance.get.side_effect = Exception("Simulated failure")
            
            # These should fail and eventually open the circuit
            for i in range(10):
                start_time = time.time()
                result = await cache.get(f"cb_key_{i}")
                duration = time.time() - start_time
                runner.record_operation("cb_failure_get", duration, result is not None)
            
            # Test operations when circuit is open
            for i in range(100):
                start_time = time.time()
                result = await cache.get(f"cb_key_{i}")
                duration = time.time() - start_time
                runner.record_operation("cb_open_get", duration, result is not None)
            
            await cache.close()
        
        runner.print_report()
        
        # Verify circuit breaker performance
        metrics = runner.calculate_metrics()
        
        # Normal operations should be fast
        assert metrics['cb_normal_set']['avg_duration_ms'] < 10.0
        
        # Open circuit operations should be very fast (no Redis call)
        assert metrics['cb_open_get']['avg_duration_ms'] < 1.0  # Much faster when circuit is open


class TestCacheStressTest:
    """Stress tests for cache system."""
    
    @pytest.mark.asyncio
    async def test_high_throughput_stress(self, mock_redis_performance):
        """Stress test with high throughput."""
        config = CacheConfiguration(
            connection_pool_size=100,
            enable_compression=True,
            enable_monitoring=True
        )
        
        with patch('redis.from_url', return_value=mock_redis_performance):
            cache = EnhancedRedisCache(config)
            
            async def stress_worker(worker_id: int, duration_seconds: int):
                """Stress test worker."""
                end_time = time.time() + duration_seconds
                operations = 0
                
                while time.time() < end_time:
                    key = f"stress_{worker_id}_{operations}"
                    data = {"worker": worker_id, "op": operations}
                    
                    # Random operation
                    import random
                    operation = random.choice(["set", "get", "delete", "exists"])
                    
                    if operation == "set":
                        await cache.set(key, data, ttl=3600)
                    elif operation == "get":
                        await cache.get(key)
                    elif operation == "delete":
                        await cache.delete(key)
                    elif operation == "exists":
                        await cache.exists(key)
                    
                    operations += 1
                    
                    # Small delay to prevent overwhelming
                    await asyncio.sleep(0.001)
                
                return operations
            
            # Run stress test
            num_workers = 20
            duration = 10  # seconds
            
            print(f"\nRunning stress test with {num_workers} workers for {duration} seconds...")
            
            start_time = time.time()
            
            tasks = []
            for worker_id in range(num_workers):
                task = asyncio.create_task(stress_worker(worker_id, duration))
                tasks.append(task)
            
            results = await asyncio.gather(*tasks)
            
            actual_duration = time.time() - start_time
            total_operations = sum(results)
            
            await cache.close()
            
            print(f"\nSTRESS TEST RESULTS:")
            print(f"Duration: {actual_duration:.2f}s")
            print(f"Total Operations: {total_operations:,}")
            print(f"Throughput: {total_operations / actual_duration:.2f} ops/sec")
            print(f"Average per Worker: {total_operations / num_workers:.0f} ops")
            
            # Verify stress test results
            assert total_operations > 10000  # Should handle significant load
            assert total_operations / actual_duration > 500  # Reasonable throughput
    
    @pytest.mark.asyncio
    async def test_memory_pressure_handling(self, mock_redis_performance):
        """Test handling of memory pressure scenarios."""
        config = CacheConfiguration(
            max_memory_cache_size=100,  # Small cache for testing
            eviction_policy=EvictionPolicy.LRU
        )
        
        with patch('redis.from_url', return_value=mock_redis_performance):
            cache = EnhancedRedisCache(config)
            
            # Fill cache beyond capacity
            large_data = {"data": "x" * 10000}  # 10KB per item
            
            success_count = 0
            for i in range(500):  # Try to store 500 items
                success = await cache.set(f"memory_pressure_{i}", large_data, ttl=3600)
                if success:
                    success_count += 1
            
            # Test that cache still functions under pressure
            info = await cache.get_info()
            health = await cache.health_check()
            
            await cache.close()
            
            print(f"\nMEMORY PRESSURE TEST:")
            print(f"Items Successfully Stored: {success_count}/500")
            print(f"Cache Health: {health['status']}")
            
            # Cache should remain functional
            assert success_count > 0  # Should store some items
            assert health['status'] == 'healthy'  # Should remain healthy


if __name__ == "__main__":
    # Run performance tests
    pytest.main([__file__, "-v", "-s"])  # -s to see print output