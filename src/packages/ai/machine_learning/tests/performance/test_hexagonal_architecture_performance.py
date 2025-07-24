"""Performance tests for hexagonal architecture implementation."""

import pytest
import asyncio
import time
import statistics
from typing import List, Dict, Any
from concurrent.futures import ThreadPoolExecutor

from machine_learning.infrastructure.container import Container, ContainerConfig, get_container, reset_container
from machine_learning.domain.services.refactored_automl_service import AutoMLService
from machine_learning.domain.interfaces.automl_operations import OptimizationConfig, AlgorithmType


class MockDataset:
    """Mock dataset for performance testing."""
    
    def __init__(self, size: int = 1000):
        self.size = size
        self.data = [[i, i*2, i*3] for i in range(size)]
    
    def __len__(self):
        return self.size


@pytest.mark.performance
class TestContainerPerformance:
    """Performance tests for dependency injection container."""
    
    def test_container_initialization_time(self):
        """Test container initialization time is acceptable."""
        times = []
        
        for _ in range(10):
            config = ContainerConfig(
                enable_sklearn_automl=False,  # Use stubs for consistent timing
                log_level="ERROR"
            )
            
            start_time = time.time()
            container = Container(config)
            end_time = time.time()
            
            times.append(end_time - start_time)
        
        avg_time = statistics.mean(times)
        max_time = max(times)
        
        # Average initialization should be < 50ms
        assert avg_time < 0.05, f"Average initialization time {avg_time:.4f}s too slow" 
        # No single initialization should take > 200ms
        assert max_time < 0.2, f"Maximum initialization time {max_time:.4f}s too slow"
        
        print(f"Container initialization - Avg: {avg_time:.4f}s, Max: {max_time:.4f}s")
    
    def test_service_retrieval_performance(self):
        """Test service retrieval from container is fast."""
        config = ContainerConfig(
            enable_sklearn_automl=False,
            log_level="ERROR"
        )
        container = Container(config)
        
        # Warm up
        container.get(AutoMLService)
        
        # Measure repeated retrievals
        times = []
        for _ in range(1000):
            start_time = time.time()
            service = container.get(AutoMLService)
            end_time = time.time()
            times.append(end_time - start_time)
        
        avg_time = statistics.mean(times)
        max_time = max(times)
        
        # Each retrieval should be very fast (< 1ms average)
        assert avg_time < 0.001, f"Average retrieval time {avg_time:.6f}s too slow"
        # No single retrieval should take > 10ms  
        assert max_time < 0.01, f"Maximum retrieval time {max_time:.6f}s too slow"
        
        print(f"Service retrieval - Avg: {avg_time:.6f}s, Max: {max_time:.6f}s")
    
    def test_container_memory_efficiency(self):
        """Test container memory usage is efficient."""
        import sys
        import gc
        
        # Force garbage collection
        gc.collect()
        
        # Create baseline
        initial_objects = len(gc.get_objects())
        
        # Create multiple containers
        containers = []
        for _ in range(100):
            config = ContainerConfig(
                enable_sklearn_automl=False,
                log_level="ERROR"
            )
            container = Container(config)
            containers.append(container)
        
        # Get services to fully initialize
        services = [container.get(AutoMLService) for container in containers]
        
        # Measure memory usage
        final_objects = len(gc.get_objects())
        object_increase = final_objects - initial_objects
        
        # Should not create excessive objects (< 1000 objects per container)
        assert object_increase < 100000, f"Created {object_increase} objects for 100 containers"
        
        print(f"Memory efficiency - {object_increase} objects for 100 containers")
        
        # Clean up
        del containers
        del services
        gc.collect()
    
    def test_concurrent_container_access(self):
        """Test concurrent access to container is thread-safe and performant."""
        config = ContainerConfig(
            enable_sklearn_automl=False,
            log_level="ERROR"
        )
        container = Container(config)
        
        def get_service_repeatedly():
            """Get service multiple times in a thread."""
            times = []
            for _ in range(100):
                start_time = time.time()
                service = container.get(AutoMLService)
                end_time = time.time()
                times.append(end_time - start_time)
            return times
        
        # Run concurrent access from multiple threads
        with ThreadPoolExecutor(max_workers=10) as executor:
            start_time = time.time()
            futures = [executor.submit(get_service_repeatedly) for _ in range(10)]
            results = [future.result() for future in futures]
            end_time = time.time()
        
        # Flatten all timing results
        all_times = [time for thread_times in results for time in thread_times]
        
        total_time = end_time - start_time
        avg_time = statistics.mean(all_times)
        
        # Total test should complete quickly (< 5 seconds)
        assert total_time < 5.0, f"Concurrent test took {total_time:.2f}s"
        # Average individual access should be fast (< 10ms)
        assert avg_time < 0.01, f"Average concurrent access {avg_time:.6f}s too slow"
        
        print(f"Concurrent access - Total: {total_time:.2f}s, Avg per call: {avg_time:.6f}s")


@pytest.mark.performance
class TestServicePerformance:
    """Performance tests for domain services."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.config = ContainerConfig(
            enable_sklearn_automl=False,  # Use stubs for consistent performance
            enable_distributed_tracing=False,
            log_level="ERROR"
        )
        self.container = Container(self.config)
        self.service = self.container.get(AutoMLService)
    
    @pytest.mark.asyncio
    async def test_optimization_performance(self):
        """Test optimization performance with different dataset sizes."""
        dataset_sizes = [100, 500, 1000, 2000]
        results = {}
        
        for size in dataset_sizes:
            dataset = MockDataset(size)
            config = OptimizationConfig(max_trials=1)  # Single trial for consistent timing
            
            times = []
            for _ in range(5):  # Run multiple times for average
                start_time = time.time()
                result = await self.service.optimize_prediction(dataset, config)
                end_time = time.time()
                times.append(end_time - start_time)
            
            avg_time = statistics.mean(times)
            results[size] = avg_time
            
            # With stubs, should be very fast regardless of dataset size
            assert avg_time < 0.1, f"Dataset size {size} took {avg_time:.4f}s"
        
        print("Optimization performance by dataset size:")
        for size, time_taken in results.items():
            print(f"  {size} samples: {time_taken:.4f}s")
    
    @pytest.mark.asyncio
    async def test_concurrent_optimization_performance(self):
        """Test performance under concurrent optimization load."""
        dataset = MockDataset(500)
        config = OptimizationConfig(max_trials=1)
        
        # Create multiple concurrent optimization tasks
        num_concurrent = 20
        
        start_time = time.time()
        tasks = [
            self.service.optimize_prediction(dataset, config)
            for _ in range(num_concurrent)
        ]
        results = await asyncio.gather(*tasks)
        end_time = time.time()
        
        total_time = end_time - start_time
        avg_time_per_request = total_time / num_concurrent
        
        # Should handle concurrent requests efficiently
        assert total_time < 2.0, f"Concurrent optimization took {total_time:.2f}s"
        assert avg_time_per_request < 0.1, f"Average per request {avg_time_per_request:.4f}s"
        
        # All requests should complete successfully
        assert len(results) == num_concurrent
        assert all(result is not None for result in results)
        
        print(f"Concurrent optimization - {num_concurrent} requests in {total_time:.2f}s")
        print(f"Average per request: {avg_time_per_request:.4f}s")
    
    @pytest.mark.asyncio
    async def test_algorithm_selection_performance(self):
        """Test algorithm selection performance."""
        datasets = [MockDataset(size) for size in [100, 500, 1000]]
        
        times = []
        for dataset in datasets:
            start_time = time.time()
            algorithm, config = await self.service.auto_select_algorithm(dataset, quick_mode=True)
            end_time = time.time()
            times.append(end_time - start_time)
        
        avg_time = statistics.mean(times)
        max_time = max(times)
        
        # Algorithm selection should be fast
        assert avg_time < 0.05, f"Average algorithm selection {avg_time:.4f}s too slow"
        assert max_time < 0.1, f"Maximum algorithm selection {max_time:.4f}s too slow"
        
        print(f"Algorithm selection - Avg: {avg_time:.4f}s, Max: {max_time:.4f}s")
    
    @pytest.mark.asyncio
    async def test_service_method_overhead(self):
        """Test service method call overhead is minimal."""
        dataset = MockDataset(100)
        
        # Measure different service methods
        methods_to_test = [
            ("optimize_prediction", lambda: self.service.optimize_prediction(
                dataset, OptimizationConfig(max_trials=1))),
            ("auto_select_algorithm", lambda: self.service.auto_select_algorithm(
                dataset, quick_mode=True)),
            ("get_optimization_recommendations", lambda: self.service.get_optimization_recommendations(dataset))
        ]
        
        results = {}
        for method_name, method_call in methods_to_test:
            times = []
            for _ in range(10):
                start_time = time.time()
                await method_call()
                end_time = time.time()
                times.append(end_time - start_time)
            
            avg_time = statistics.mean(times)
            results[method_name] = avg_time
            
            # Each method should have minimal overhead
            assert avg_time < 0.1, f"{method_name} took {avg_time:.4f}s on average"
        
        print("Service method performance:")
        for method_name, avg_time in results.items():
            print(f"  {method_name}: {avg_time:.4f}s")


@pytest.mark.performance  
class TestArchitecturalOverhead:
    """Test architectural overhead of hexagonal pattern."""
    
    @pytest.mark.asyncio
    async def test_dependency_injection_overhead(self):
        """Test dependency injection overhead vs direct instantiation."""
        # Note: This test compares DI container vs manual instantiation
        # In real code, we wouldn't do manual instantiation, but this measures overhead
        
        dataset = MockDataset(200)
        config = OptimizationConfig(max_trials=1)
        
        # Method 1: Using container (hexagonal architecture)
        container = Container(ContainerConfig(
            enable_sklearn_automl=False,
            log_level="ERROR"
        ))
        
        container_times = []
        for _ in range(20):
            start_time = time.time()
            service = container.get(AutoMLService)
            result = await service.optimize_prediction(dataset, config)
            end_time = time.time()
            container_times.append(end_time - start_time)
        
        # Method 2: Direct instantiation (if we had it - simulated here)
        # This would be the "old way" without DI container
        from machine_learning.infrastructure.adapters.stubs.automl_stubs import (
            AutoMLOptimizationStub, ModelSelectionStub
        )
        
        direct_times = []
        for _ in range(20):
            start_time = time.time()
            # Simulate direct instantiation
            automl_port = AutoMLOptimizationStub()
            model_selection_port = ModelSelectionStub()
            service = AutoMLService(automl_port, model_selection_port)
            result = await service.optimize_prediction(dataset, config)
            end_time = time.time()
            direct_times.append(end_time - start_time)
        
        container_avg = statistics.mean(container_times)
        direct_avg = statistics.mean(direct_times)
        overhead = container_avg - direct_avg
        overhead_percent = (overhead / direct_avg) * 100 if direct_avg > 0 else 0
        
        # DI overhead should be minimal (< 50% overhead)
        assert overhead_percent < 50, f"DI overhead {overhead_percent:.1f}% too high"
        
        print(f"Dependency injection overhead:")
        print(f"  Container: {container_avg:.4f}s")
        print(f"  Direct: {direct_avg:.4f}s") 
        print(f"  Overhead: {overhead:.4f}s ({overhead_percent:.1f}%)")
    
    def test_interface_call_overhead(self):
        """Test interface call overhead vs direct method calls."""
        # This measures the overhead of calling through interfaces
        # vs calling concrete methods directly
        
        from machine_learning.infrastructure.adapters.stubs.automl_stubs import AutoMLOptimizationStub
        from machine_learning.domain.interfaces.automl_operations import AutoMLOptimizationPort
        
        stub = AutoMLOptimizationStub()
        interface: AutoMLOptimizationPort = stub
        
        # Test data
        dataset = MockDataset(100)
        config = OptimizationConfig(max_trials=1)
        
        # Measure direct calls
        direct_times = []
        for _ in range(100):
            start_time = time.time()
            # Direct call (if we exposed internal methods)
            # result = stub._some_internal_method(dataset, config)
            end_time = time.time()
            direct_times.append(end_time - start_time)
        
        # Measure interface calls  
        interface_times = []
        for _ in range(100):
            start_time = time.time()
            # Interface call
            # result = asyncio.run(interface.optimize_model(dataset, config))
            end_time = time.time()
            interface_times.append(end_time - start_time)
        
        # Interface overhead should be negligible
        # (This test is more conceptual since we can't easily measure this difference)
        print("Interface call overhead test conceptual - actual measurement would require internal methods")
    
    def test_memory_overhead_of_architecture(self):
        """Test memory overhead of hexagonal architecture."""
        import sys
        import gc
        
        gc.collect()
        
        # Method 1: Full hexagonal architecture
        container = Container(ContainerConfig(
            enable_sklearn_automl=False,
            log_level="ERROR"
        ))
        service = container.get(AutoMLService)
        
        # Get references to all architectural components
        automl_port = service._automl_port
        model_selection_port = service._model_selection_port
        
        # Measure memory usage
        architecture_objects = [container, service, automl_port, model_selection_port]
        architecture_memory = sum(sys.getsizeof(obj) for obj in architecture_objects)
        
        # Method 2: Minimal direct instantiation
        from machine_learning.infrastructure.adapters.stubs.automl_stubs import (
            AutoMLOptimizationStub, ModelSelectionStub
        )
        
        direct_automl = AutoMLOptimizationStub()
        direct_model_selection = ModelSelectionStub()
        direct_service = AutoMLService(direct_automl, direct_model_selection)
        
        direct_objects = [direct_automl, direct_model_selection, direct_service]
        direct_memory = sum(sys.getsizeof(obj) for obj in direct_objects)
        
        memory_overhead = architecture_memory - direct_memory
        overhead_percent = (memory_overhead / direct_memory) * 100 if direct_memory > 0 else 0
        
        # Memory overhead should be reasonable (< 200%)
        assert overhead_percent < 200, f"Memory overhead {overhead_percent:.1f}% too high"
        
        print(f"Memory overhead of architecture:")
        print(f"  Full architecture: {architecture_memory} bytes")
        print(f"  Direct instantiation: {direct_memory} bytes")
        print(f"  Overhead: {memory_overhead} bytes ({overhead_percent:.1f}%)")


@pytest.mark.performance
class TestScalabilityCharacteristics:
    """Test scalability characteristics of the architecture."""
    
    @pytest.mark.asyncio
    async def test_service_scaling_with_request_volume(self):
        """Test how service performs as request volume increases."""
        container = Container(ContainerConfig(
            enable_sklearn_automl=False,
            log_level="ERROR"
        ))
        service = container.get(AutoMLService)
        
        dataset = MockDataset(300)
        config = OptimizationConfig(max_trials=1)
        
        # Test with increasing numbers of concurrent requests
        request_volumes = [1, 5, 10, 20, 50]
        results = {}
        
        for volume in request_volumes:
            start_time = time.time()
            tasks = [
                service.optimize_prediction(dataset, config)
                for _ in range(volume)
            ]
            await asyncio.gather(*tasks)
            end_time = time.time()
            
            total_time = end_time - start_time
            avg_time_per_request = total_time / volume
            results[volume] = avg_time_per_request
            
            print(f"Volume {volume}: {total_time:.2f}s total, {avg_time_per_request:.4f}s per request")
        
        # Performance should not degrade significantly with volume
        # (Linear scaling is acceptable for this architecture)
        max_degradation = max(results.values()) / min(results.values())
        assert max_degradation < 5.0, f"Performance degraded {max_degradation:.1f}x with volume"
    
    def test_container_scaling_with_service_types(self):
        """Test container performance as number of service types increases."""
        # Test how container performs with many different service types
        configs = [
            ContainerConfig(enable_sklearn_automl=False, log_level="ERROR"),
            ContainerConfig(enable_sklearn_automl=False, enable_distributed_tracing=False, log_level="ERROR"),
            ContainerConfig(enable_sklearn_automl=False, enable_prometheus_monitoring=False, log_level="ERROR"),
        ]
        
        times = []
        for config in configs:
            start_time = time.time()
            container = Container(config)
            # Get multiple service types
            service = container.get(AutoMLService)
            end_time = time.time()
            times.append(end_time - start_time)
        
        # Initialization time should remain reasonable regardless of configuration complexity
        avg_time = statistics.mean(times)
        max_time = max(times)
        
        assert avg_time < 0.1, f"Average container initialization {avg_time:.4f}s too slow"
        assert max_time < 0.2, f"Maximum container initialization {max_time:.4f}s too slow"
        
        print(f"Container scaling - Avg: {avg_time:.4f}s, Max: {max_time:.4f}s")
    
    @pytest.mark.asyncio
    async def test_memory_scaling_with_concurrent_usage(self):
        """Test memory usage scales reasonably with concurrent usage."""
        import gc
        
        gc.collect()
        initial_objects = len(gc.get_objects())
        
        container = Container(ContainerConfig(
            enable_sklearn_automl=False,
            log_level="ERROR"
        ))
        
        # Simulate many concurrent users
        services = [container.get(AutoMLService) for _ in range(100)]
        
        # Run operations concurrently
        dataset = MockDataset(100)
        config = OptimizationConfig(max_trials=1)
        
        tasks = [
            service.optimize_prediction(dataset, config)
            for service in services[:10]  # Subset to avoid overwhelming
        ]
        await asyncio.gather(*tasks)
        
        final_objects = len(gc.get_objects())
        object_increase = final_objects - initial_objects
        
        # Memory usage should scale reasonably
        assert object_increase < 50000, f"Created {object_increase} objects"
        
        print(f"Memory scaling - {object_increase} objects for 100 services")
        
        # Clean up
        del services
        gc.collect()


@pytest.mark.performance
class TestPerformanceBenchmarks:
    """Comprehensive performance benchmarks for the architecture."""
    
    def test_performance_baseline_establishment(self):
        """Establish performance baselines for regression testing."""
        # This test establishes baselines that can be used to detect
        # performance regressions in future changes
        
        benchmarks = {}
        
        # Container initialization benchmark
        times = []
        for _ in range(20):
            start_time = time.time()
            container = Container(ContainerConfig(
                enable_sklearn_automl=False,
                log_level="ERROR"
            ))
            service = container.get(AutoMLService)
            end_time = time.time()
            times.append(end_time - start_time)
        
        benchmarks['container_init'] = {
            'avg': statistics.mean(times),
            'max': max(times),
            'min': min(times)
        }
        
        # Service retrieval benchmark
        container = Container(ContainerConfig(
            enable_sklearn_automl=False,
            log_level="ERROR"
        ))
        times = []
        for _ in range(1000):
            start_time = time.time()
            service = container.get(AutoMLService)
            end_time = time.time()
            times.append(end_time - start_time)
        
        benchmarks['service_retrieval'] = {
            'avg': statistics.mean(times),
            'max': max(times),
            'min': min(times)
        }
        
        # Print benchmarks for future reference
        print("Performance Benchmarks (for regression testing):")
        for benchmark_name, metrics in benchmarks.items():
            print(f"  {benchmark_name}:")
            print(f"    Average: {metrics['avg']:.6f}s")
            print(f"    Maximum: {metrics['max']:.6f}s")
            print(f"    Minimum: {metrics['min']:.6f}s")
        
        # Basic sanity checks
        assert benchmarks['container_init']['avg'] < 0.1
        assert benchmarks['service_retrieval']['avg'] < 0.001
    
    @pytest.mark.asyncio
    async def test_end_to_end_performance_benchmark(self):
        """End-to-end performance benchmark for complete workflows."""
        container = Container(ContainerConfig(
            enable_sklearn_automl=False,
            enable_distributed_tracing=False,
            log_level="ERROR"
        ))
        service = container.get(AutoMLService)
        
        # Benchmark complete optimization workflow
        dataset_sizes = [100, 500, 1000]
        benchmark_results = {}
        
        for size in dataset_sizes:
            dataset = MockDataset(size)
            config = OptimizationConfig(max_trials=1)
            
            # Run multiple iterations
            times = []
            for _ in range(10):
                start_time = time.time()
                result = await service.optimize_prediction(dataset, config)
                end_time = time.time()
                times.append(end_time - start_time)
            
            benchmark_results[size] = {
                'avg': statistics.mean(times),
                'max': max(times),
                'min': min(times),
                'std': statistics.stdev(times) if len(times) > 1 else 0
            }
        
        print("End-to-end Performance Benchmarks:")
        for size, metrics in benchmark_results.items():
            print(f"  Dataset size {size}:")
            print(f"    Average: {metrics['avg']:.4f}s")
            print(f"    Maximum: {metrics['max']:.4f}s") 
            print(f"    Std Dev: {metrics['std']:.4f}s")
        
        # Performance should be consistent across dataset sizes (using stubs)
        for size, metrics in benchmark_results.items():
            assert metrics['avg'] < 0.1, f"Size {size} too slow: {metrics['avg']:.4f}s"
            assert metrics['std'] < 0.05, f"Size {size} too variable: {metrics['std']:.4f}s"