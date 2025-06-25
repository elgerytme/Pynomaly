"""Comprehensive performance portability tests.

This module tests performance characteristics across different platforms,
hardware configurations, and deployment scenarios to ensure consistent
and acceptable performance regardless of the target environment.
"""

import pytest
import time
import psutil
import platform
import threading
import multiprocessing as mp
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, Any, List, Tuple, Optional
from unittest.mock import Mock, patch
import tempfile
import json


class TestCPUPerformancePortability:
    """Test CPU performance across different architectures and configurations."""
    
    def test_single_core_performance(self):
        """Test performance on single-core systems."""
        # Simulate single-core constraint
        original_cpu_count = mp.cpu_count()
        
        # Test CPU-intensive operations with single thread
        def cpu_intensive_task(n: int) -> float:
            start_time = time.time()
            total = 0
            for i in range(n):
                total += i ** 0.5
            end_time = time.time()
            return end_time - start_time
        
        # Test with progressively larger workloads
        workloads = [10000, 50000, 100000]
        execution_times = []
        
        for workload in workloads:
            exec_time = cpu_intensive_task(workload)
            execution_times.append(exec_time)
            
            # Performance should be reasonable (not hanging)
            assert exec_time < 10.0, f"Single-core task took too long: {exec_time}s"
        
        # Performance should scale roughly linearly with workload
        time_ratios = [execution_times[i+1] / execution_times[i] for i in range(len(execution_times)-1)]
        for ratio in time_ratios:
            assert 1.0 <= ratio <= 10.0, f"Performance scaling issue: ratio {ratio}"
    
    def test_multi_core_performance_scaling(self):
        """Test performance scaling across multiple CPU cores."""
        cpu_count = mp.cpu_count()
        
        def parallel_computation(data_chunk):
            """CPU-intensive computation on data chunk."""
            return np.sum(np.sqrt(data_chunk))
        
        # Generate test data
        test_data_size = 100000
        test_data = np.random.random(test_data_size)
        
        # Test sequential processing
        start_time = time.time()
        sequential_result = parallel_computation(test_data)
        sequential_time = time.time() - start_time
        
        # Test parallel processing with different thread counts
        thread_counts = [1, min(2, cpu_count), min(4, cpu_count), cpu_count]
        parallel_times = []
        
        for thread_count in thread_counts:
            if thread_count <= cpu_count:
                # Split data into chunks
                chunk_size = len(test_data) // thread_count
                chunks = [test_data[i:i+chunk_size] for i in range(0, len(test_data), chunk_size)]
                
                # Process in parallel
                start_time = time.time()
                with ThreadPoolExecutor(max_workers=thread_count) as executor:
                    results = list(executor.map(parallel_computation, chunks))
                parallel_time = time.time() - start_time
                parallel_times.append(parallel_time)
                
                # Verify result consistency
                parallel_result = sum(results)
                assert abs(parallel_result - sequential_result) / sequential_result < 0.01
        
        # Test that parallel processing shows some benefit (up to GIL limitations)
        if len(parallel_times) > 1:
            # Even with GIL, some speedup might be observed for I/O-bound parts
            best_parallel_time = min(parallel_times)
            speedup_ratio = sequential_time / best_parallel_time
            assert speedup_ratio >= 0.8, f"Parallel processing too slow: {speedup_ratio}"
    
    def test_cpu_architecture_compatibility(self):
        """Test compatibility across different CPU architectures."""
        machine_type = platform.machine().lower()
        processor = platform.processor().lower()
        
        # Test architecture-specific optimizations
        if "x86" in machine_type or "amd64" in machine_type:
            # x86/x64 architecture
            # Test vectorized operations (should benefit from SIMD)
            large_array = np.random.random(100000)
            
            start_time = time.time()
            vectorized_result = np.sum(large_array ** 2)
            vectorized_time = time.time() - start_time
            
            start_time = time.time()
            loop_result = sum(x ** 2 for x in large_array)
            loop_time = time.time() - start_time
            
            # Vectorized operations should be faster
            assert abs(vectorized_result - loop_result) < 1e-10
            speedup = loop_time / vectorized_time
            assert speedup >= 2.0, f"Vectorization not effective: {speedup}x speedup"
            
        elif "arm" in machine_type or "aarch64" in machine_type:
            # ARM architecture
            # Test ARM-specific optimizations
            test_data = np.random.random(10000)
            
            # Test basic numerical operations
            operations = [
                np.mean(test_data),
                np.std(test_data),
                np.sum(test_data),
                np.max(test_data),
                np.min(test_data)
            ]
            
            # All operations should complete efficiently
            for result in operations:
                assert isinstance(result, (int, float, np.number))
                assert not np.isnan(result)
                assert not np.isinf(result)
        
        # Test generic CPU performance characteristics
        cpu_info = {
            "architecture": machine_type,
            "processor": processor,
            "cpu_count": mp.cpu_count(),
            "cpu_freq": None
        }
        
        # Get CPU frequency if available
        try:
            import psutil
            cpu_freq = psutil.cpu_freq()
            if cpu_freq:
                cpu_info["cpu_freq"] = cpu_freq.current
        except (ImportError, AttributeError):
            pass
        
        # Verify reasonable CPU configuration
        assert cpu_info["cpu_count"] >= 1
        assert cpu_info["cpu_count"] <= 256  # Reasonable upper bound
    
    def test_cpu_thermal_throttling_resilience(self):
        """Test resilience to CPU thermal throttling."""
        # Simulate sustained CPU load
        def sustained_cpu_load(duration_seconds: float) -> List[float]:
            """Run sustained CPU load and measure performance."""
            start_time = time.time()
            performance_samples = []
            
            while time.time() - start_time < duration_seconds:
                # CPU-intensive task
                iteration_start = time.time()
                _ = sum(i ** 0.5 for i in range(100000))
                iteration_time = time.time() - iteration_start
                performance_samples.append(1.0 / iteration_time)  # Operations per second
                
                # Brief pause to allow temperature readings
                time.sleep(0.1)
            
            return performance_samples
        
        # Run sustained load for a short period
        load_duration = 5.0  # seconds
        performance_data = sustained_cpu_load(load_duration)
        
        # Analyze performance consistency
        if len(performance_data) > 2:
            avg_performance = np.mean(performance_data)
            std_performance = np.std(performance_data)
            
            # Performance should be relatively stable
            coefficient_of_variation = std_performance / avg_performance
            assert coefficient_of_variation < 0.5, f"High performance variation: {coefficient_of_variation}"
            
            # No severe performance drops (thermal throttling indicator)
            min_performance = np.min(performance_data)
            max_performance = np.max(performance_data)
            performance_ratio = min_performance / max_performance
            assert performance_ratio > 0.3, f"Severe performance drop detected: {performance_ratio}"


class TestMemoryPerformancePortability:
    """Test memory performance across different memory configurations."""
    
    def test_memory_allocation_patterns(self):
        """Test memory allocation performance patterns."""
        try:
            import psutil
            initial_memory = psutil.virtual_memory()
        except ImportError:
            pytest.skip("psutil not available for memory monitoring")
        
        # Test small frequent allocations
        small_allocations = []
        start_time = time.time()
        
        for _ in range(1000):
            small_array = np.random.random(100)
            small_allocations.append(small_array)
        
        small_alloc_time = time.time() - start_time
        
        # Test large single allocation
        start_time = time.time()
        large_array = np.random.random(100000)
        large_alloc_time = time.time() - start_time
        
        # Small allocations should not be excessively slow
        assert small_alloc_time < 5.0, f"Small allocations too slow: {small_alloc_time}s"
        assert large_alloc_time < 2.0, f"Large allocation too slow: {large_alloc_time}s"
        
        # Clean up
        del small_allocations
        del large_array
        
        # Force garbage collection
        import gc
        gc.collect()
    
    def test_memory_access_patterns(self):
        """Test memory access pattern performance."""
        # Create test array
        test_size = 100000
        test_array = np.random.random(test_size)
        
        # Test sequential access
        start_time = time.time()
        sequential_sum = 0
        for i in range(len(test_array)):
            sequential_sum += test_array[i]
        sequential_time = time.time() - start_time
        
        # Test random access
        random_indices = np.random.randint(0, test_size, test_size // 10)
        start_time = time.time()
        random_sum = 0
        for idx in random_indices:
            random_sum += test_array[idx]
        random_time = time.time() - start_time
        
        # Test vectorized access
        start_time = time.time()
        vectorized_sum = np.sum(test_array)
        vectorized_time = time.time() - start_time
        
        # Vectorized should be fastest
        assert vectorized_time < sequential_time, "Vectorized access not fastest"
        assert sequential_time < random_time * 2, "Sequential access unexpectedly slow"
        
        # Verify correctness
        assert abs(sequential_sum - vectorized_sum) < 1e-10
    
    def test_memory_pressure_handling(self):
        """Test handling of memory pressure scenarios."""
        try:
            import psutil
            available_memory = psutil.virtual_memory().available
            
            # Use a fraction of available memory to avoid system issues
            safe_memory_usage = min(available_memory // 4, 500 * 1024 * 1024)  # 500MB max
            
        except ImportError:
            # Fallback to conservative memory usage
            safe_memory_usage = 100 * 1024 * 1024  # 100MB
        
        # Calculate array size for target memory usage
        bytes_per_float64 = 8
        array_size = safe_memory_usage // bytes_per_float64
        
        try:
            # Allocate large array
            start_time = time.time()
            large_data = np.random.random(array_size)
            allocation_time = time.time() - start_time
            
            # Test operations on large array
            start_time = time.time()
            mean_value = np.mean(large_data)
            operation_time = time.time() - start_time
            
            # Verify allocation and operation completed
            assert allocation_time < 30.0, f"Large allocation too slow: {allocation_time}s"
            assert operation_time < 10.0, f"Large array operation too slow: {operation_time}s"
            assert isinstance(mean_value, (float, np.floating))
            
            # Clean up
            del large_data
            import gc
            gc.collect()
            
        except MemoryError:
            # System doesn't have enough memory - this is acceptable
            pytest.skip("Insufficient memory for large allocation test")
    
    def test_memory_fragmentation_resilience(self):
        """Test resilience to memory fragmentation."""
        # Simulate fragmentation by creating and deleting arrays
        arrays = []
        
        # Create fragmentation pattern
        for i in range(100):
            size = np.random.randint(1000, 10000)
            array = np.random.random(size)
            arrays.append(array)
            
            # Randomly delete some arrays to create fragmentation
            if i > 10 and np.random.random() < 0.3:
                del_idx = np.random.randint(0, len(arrays))
                del arrays[del_idx]
        
        # Test allocation after fragmentation
        start_time = time.time()
        post_frag_array = np.random.random(50000)
        post_frag_time = time.time() - start_time
        
        # Should still allocate reasonably quickly
        assert post_frag_time < 5.0, f"Post-fragmentation allocation slow: {post_frag_time}s"
        
        # Clean up
        del arrays
        del post_frag_array
        import gc
        gc.collect()


class TestStoragePerformancePortability:
    """Test storage performance across different storage systems."""
    
    def test_file_io_performance(self):
        """Test file I/O performance across storage types."""
        # Test different file sizes
        file_sizes = [1024, 10240, 102400]  # 1KB, 10KB, 100KB
        
        for file_size in file_sizes:
            # Generate test data
            test_data = np.random.bytes(file_size)
            
            with tempfile.NamedTemporaryFile(delete=False) as temp_file:
                # Test write performance
                start_time = time.time()
                temp_file.write(test_data)
                temp_file.flush()
                write_time = time.time() - start_time
                
                temp_file_path = temp_file.name
            
            # Test read performance
            start_time = time.time()
            with open(temp_file_path, 'rb') as f:
                read_data = f.read()
            read_time = time.time() - start_time
            
            # Verify data integrity
            assert read_data == test_data
            
            # Performance should be reasonable
            write_throughput = file_size / write_time / 1024  # KB/s
            read_throughput = file_size / read_time / 1024    # KB/s
            
            assert write_throughput > 50, f"Write too slow: {write_throughput} KB/s"
            assert read_throughput > 100, f"Read too slow: {read_throughput} KB/s"
            
            # Clean up
            Path(temp_file_path).unlink()
    
    def test_dataframe_io_performance(self):
        """Test DataFrame I/O performance across formats."""
        # Create test DataFrame
        test_df = pd.DataFrame({
            'int_col': np.random.randint(0, 1000, 10000),
            'float_col': np.random.random(10000),
            'str_col': [f"string_{i}" for i in range(10000)],
            'datetime_col': pd.date_range('2023-01-01', periods=10000, freq='1min')
        })
        
        # Test different formats
        formats = ['csv', 'parquet', 'json']
        
        for fmt in formats:
            with tempfile.NamedTemporaryFile(suffix=f'.{fmt}', delete=False) as temp_file:
                temp_path = temp_file.name
            
            try:
                # Test write performance
                start_time = time.time()
                if fmt == 'csv':
                    test_df.to_csv(temp_path, index=False)
                elif fmt == 'parquet':
                    test_df.to_parquet(temp_path, index=False)
                elif fmt == 'json':
                    test_df.to_json(temp_path, orient='records')
                write_time = time.time() - start_time
                
                # Test read performance
                start_time = time.time()
                if fmt == 'csv':
                    loaded_df = pd.read_csv(temp_path)
                elif fmt == 'parquet':
                    loaded_df = pd.read_parquet(temp_path)
                elif fmt == 'json':
                    loaded_df = pd.read_json(temp_path, orient='records')
                read_time = time.time() - start_time
                
                # Verify basic structure
                assert len(loaded_df) == len(test_df)
                assert len(loaded_df.columns) >= 3  # Some type conversion may occur
                
                # Performance should be reasonable
                assert write_time < 10.0, f"{fmt} write too slow: {write_time}s"
                assert read_time < 10.0, f"{fmt} read too slow: {read_time}s"
                
            except ImportError:
                # Format not supported (e.g., parquet without pyarrow)
                continue
            
            finally:
                # Clean up
                if Path(temp_path).exists():
                    Path(temp_path).unlink()
    
    def test_database_io_performance(self):
        """Test database I/O performance simulation."""
        # Simulate database operations with SQLite
        import sqlite3
        
        with tempfile.NamedTemporaryFile(suffix='.db', delete=False) as temp_db:
            db_path = temp_db.name
        
        try:
            # Create test data
            test_data = pd.DataFrame({
                'id': range(1000),
                'value': np.random.random(1000),
                'category': np.random.choice(['A', 'B', 'C'], 1000)
            })
            
            # Test database write performance
            start_time = time.time()
            conn = sqlite3.connect(db_path)
            test_data.to_sql('test_table', conn, index=False, if_exists='replace')
            conn.close()
            write_time = time.time() - start_time
            
            # Test database read performance
            start_time = time.time()
            conn = sqlite3.connect(db_path)
            loaded_data = pd.read_sql('SELECT * FROM test_table', conn)
            conn.close()
            read_time = time.time() - start_time
            
            # Verify data integrity
            assert len(loaded_data) == len(test_data)
            assert list(loaded_data.columns) == list(test_data.columns)
            
            # Performance should be reasonable
            assert write_time < 5.0, f"Database write too slow: {write_time}s"
            assert read_time < 3.0, f"Database read too slow: {read_time}s"
            
        finally:
            # Clean up
            if Path(db_path).exists():
                Path(db_path).unlink()


class TestNetworkPerformancePortability:
    """Test network performance characteristics."""
    
    def test_localhost_communication_performance(self):
        """Test localhost communication performance."""
        import socket
        import threading
        
        # Simple echo server for testing
        def echo_server(port):
            try:
                server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
                server_socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
                server_socket.bind(('localhost', port))
                server_socket.listen(1)
                server_socket.settimeout(5)  # 5 second timeout
                
                conn, addr = server_socket.accept()
                data = conn.recv(1024)
                conn.send(data)  # Echo back
                conn.close()
                server_socket.close()
                
            except (socket.error, socket.timeout):
                # Server error - acceptable for testing
                pass
        
        # Find available port
        test_port = 0
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            s.bind(('localhost', 0))
            test_port = s.getsockname()[1]
        
        # Start server in background
        server_thread = threading.Thread(target=echo_server, args=(test_port,))
        server_thread.daemon = True
        server_thread.start()
        
        time.sleep(0.1)  # Give server time to start
        
        try:
            # Test client communication
            start_time = time.time()
            client_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            client_socket.settimeout(2)
            client_socket.connect(('localhost', test_port))
            
            test_message = b"Hello, localhost!"
            client_socket.send(test_message)
            response = client_socket.recv(1024)
            client_socket.close()
            
            communication_time = time.time() - start_time
            
            # Verify echo functionality
            assert response == test_message
            
            # Localhost communication should be fast
            assert communication_time < 1.0, f"Localhost communication slow: {communication_time}s"
            
        except (socket.error, socket.timeout):
            pytest.skip("Localhost communication not available")
        
        # Wait for server thread to complete
        server_thread.join(timeout=1)
    
    def test_serialization_performance(self):
        """Test data serialization performance for network transmission."""
        # Create test data
        test_data = {
            'results': [
                {
                    'id': i,
                    'score': float(np.random.random()),
                    'features': list(np.random.random(10)),
                    'metadata': {'timestamp': f"2023-10-{i:02d}T10:00:00Z"}
                }
                for i in range(1, 101)
            ]
        }
        
        # Test JSON serialization
        start_time = time.time()
        json_data = json.dumps(test_data)
        json_serialize_time = time.time() - start_time
        
        start_time = time.time()
        loaded_json_data = json.loads(json_data)
        json_deserialize_time = time.time() - start_time
        
        # Test pickle serialization
        import pickle
        
        start_time = time.time()
        pickle_data = pickle.dumps(test_data)
        pickle_serialize_time = time.time() - start_time
        
        start_time = time.time()
        loaded_pickle_data = pickle.loads(pickle_data)
        pickle_deserialize_time = time.time() - start_time
        
        # Verify data integrity
        assert loaded_json_data == test_data
        assert loaded_pickle_data == test_data
        
        # Performance should be reasonable
        assert json_serialize_time < 1.0, f"JSON serialization slow: {json_serialize_time}s"
        assert json_deserialize_time < 1.0, f"JSON deserialization slow: {json_deserialize_time}s"
        assert pickle_serialize_time < 1.0, f"Pickle serialization slow: {pickle_serialize_time}s"
        assert pickle_deserialize_time < 1.0, f"Pickle deserialization slow: {pickle_deserialize_time}s"
        
        # Compare sizes
        json_size = len(json_data.encode('utf-8'))
        pickle_size = len(pickle_data)
        
        # Both should be reasonable for network transmission
        assert json_size < 100000, f"JSON payload too large: {json_size} bytes"
        assert pickle_size < 100000, f"Pickle payload too large: {pickle_size} bytes"


class TestConcurrencyPerformancePortability:
    """Test concurrency performance across different scenarios."""
    
    def test_threading_performance_characteristics(self):
        """Test threading performance characteristics."""
        def io_bound_task(duration: float) -> float:
            """Simulate I/O bound task."""
            start_time = time.time()
            time.sleep(duration)
            return time.time() - start_time
        
        # Test sequential execution
        task_duration = 0.1
        num_tasks = 5
        
        start_time = time.time()
        sequential_results = [io_bound_task(task_duration) for _ in range(num_tasks)]
        sequential_time = time.time() - start_time
        
        # Test threaded execution
        start_time = time.time()
        with ThreadPoolExecutor(max_workers=num_tasks) as executor:
            threaded_results = list(executor.map(io_bound_task, [task_duration] * num_tasks))
        threaded_time = time.time() - start_time
        
        # Verify results
        assert len(sequential_results) == num_tasks
        assert len(threaded_results) == num_tasks
        
        # Threading should provide speedup for I/O bound tasks
        speedup = sequential_time / threaded_time
        assert speedup > 2.0, f"Threading speedup insufficient: {speedup}x"
        
        # Individual task times should be similar
        avg_sequential = np.mean(sequential_results)
        avg_threaded = np.mean(threaded_results)
        assert abs(avg_sequential - avg_threaded) < 0.05
    
    def test_multiprocessing_performance_characteristics(self):
        """Test multiprocessing performance characteristics."""
        def cpu_bound_task(n: int) -> float:
            """CPU-intensive task for multiprocessing test."""
            start_time = time.time()
            total = sum(i ** 0.5 for i in range(n))
            return time.time() - start_time
        
        task_size = 100000
        num_tasks = min(4, mp.cpu_count())
        
        # Test sequential execution
        start_time = time.time()
        sequential_results = [cpu_bound_task(task_size) for _ in range(num_tasks)]
        sequential_time = time.time() - start_time
        
        # Test multiprocessing execution
        start_time = time.time()
        with ProcessPoolExecutor(max_workers=num_tasks) as executor:
            parallel_results = list(executor.map(cpu_bound_task, [task_size] * num_tasks))
        parallel_time = time.time() - start_time
        
        # Verify results
        assert len(sequential_results) == num_tasks
        assert len(parallel_results) == num_tasks
        
        # Multiprocessing may provide speedup for CPU-bound tasks
        speedup = sequential_time / parallel_time
        # Note: Speedup depends on system capabilities and overhead
        assert speedup > 0.5, f"Multiprocessing severely degraded performance: {speedup}x"
        
        # Don't expect linear speedup due to overhead and system constraints
        max_expected_speedup = num_tasks + 1  # Allow some overhead
        assert speedup < max_expected_speedup, f"Unrealistic speedup reported: {speedup}x"
    
    def test_async_performance_characteristics(self):
        """Test async/await performance characteristics."""
        import asyncio
        
        async def async_io_task(duration: float) -> float:
            """Simulate async I/O task."""
            start_time = time.time()
            await asyncio.sleep(duration)
            return time.time() - start_time
        
        async def run_sequential_async(duration: float, num_tasks: int) -> Tuple[List[float], float]:
            """Run async tasks sequentially."""
            start_time = time.time()
            results = []
            for _ in range(num_tasks):
                result = await async_io_task(duration)
                results.append(result)
            total_time = time.time() - start_time
            return results, total_time
        
        async def run_concurrent_async(duration: float, num_tasks: int) -> Tuple[List[float], float]:
            """Run async tasks concurrently."""
            start_time = time.time()
            tasks = [async_io_task(duration) for _ in range(num_tasks)]
            results = await asyncio.gather(*tasks)
            total_time = time.time() - start_time
            return results, total_time
        
        # Test parameters
        task_duration = 0.1
        num_tasks = 5
        
        # Run tests
        sequential_results, sequential_time = asyncio.run(
            run_sequential_async(task_duration, num_tasks)
        )
        
        concurrent_results, concurrent_time = asyncio.run(
            run_concurrent_async(task_duration, num_tasks)
        )
        
        # Verify results
        assert len(sequential_results) == num_tasks
        assert len(concurrent_results) == num_tasks
        
        # Concurrent execution should be much faster
        speedup = sequential_time / concurrent_time
        assert speedup > 3.0, f"Async concurrency speedup insufficient: {speedup}x"
        
        # Individual task durations should be similar
        avg_sequential = np.mean(sequential_results)
        avg_concurrent = np.mean(concurrent_results)
        assert abs(avg_sequential - avg_concurrent) < 0.05


class TestScalabilityPortability:
    """Test scalability characteristics across different system configurations."""
    
    def test_data_size_scalability(self):
        """Test performance scalability with increasing data sizes."""
        # Test with exponentially increasing data sizes
        base_size = 1000
        size_multipliers = [1, 5, 10, 50]
        
        processing_times = []
        
        for multiplier in size_multipliers:
            data_size = base_size * multiplier
            test_data = np.random.random(data_size)
            
            # Test processing time
            start_time = time.time()
            result = np.mean(test_data)
            processing_time = time.time() - start_time
            processing_times.append(processing_time)
            
            # Basic sanity check
            assert isinstance(result, (float, np.floating))
            assert 0.0 <= result <= 1.0
        
        # Analyze scalability
        if len(processing_times) > 1:
            # Calculate time ratios
            time_ratios = [
                processing_times[i] / processing_times[0] 
                for i in range(1, len(processing_times))
            ]
            
            size_ratios = [
                size_multipliers[i] / size_multipliers[0] 
                for i in range(1, len(size_multipliers))
            ]
            
            # Performance should scale reasonably (not exponentially worse)
            for i, (time_ratio, size_ratio) in enumerate(zip(time_ratios, size_ratios)):
                # Allow for some overhead, but not quadratic scaling
                max_acceptable_ratio = size_ratio * 2
                assert time_ratio <= max_acceptable_ratio, \
                    f"Poor scalability at {size_multipliers[i+1]}x size: {time_ratio}x time"
    
    def test_concurrent_user_scalability(self):
        """Test scalability with multiple concurrent operations."""
        def simulate_user_operation():
            """Simulate a user operation."""
            # Generate small dataset
            data = np.random.random(1000)
            
            # Perform computation
            result = {
                'mean': float(np.mean(data)),
                'std': float(np.std(data)),
                'max': float(np.max(data)),
                'min': float(np.min(data))
            }
            
            return result
        
        # Test with increasing numbers of concurrent users
        user_counts = [1, 5, 10, 20]
        
        for user_count in user_counts:
            start_time = time.time()
            
            # Run concurrent operations
            with ThreadPoolExecutor(max_workers=user_count) as executor:
                futures = [executor.submit(simulate_user_operation) for _ in range(user_count)]
                results = [future.result() for future in futures]
            
            total_time = time.time() - start_time
            
            # Verify all operations completed
            assert len(results) == user_count
            
            # All results should be valid
            for result in results:
                assert isinstance(result, dict)
                assert 'mean' in result
                assert 'std' in result
                assert 0.0 <= result['mean'] <= 1.0
                assert result['std'] >= 0.0
            
            # Performance should degrade gracefully
            avg_time_per_user = total_time / user_count
            assert avg_time_per_user < 5.0, \
                f"Poor per-user performance with {user_count} users: {avg_time_per_user}s"
    
    def test_memory_usage_scalability(self):
        """Test memory usage scalability."""
        try:
            import psutil
            initial_memory = psutil.Process().memory_info().rss
        except ImportError:
            pytest.skip("psutil not available for memory monitoring")
        
        # Test memory usage with increasing data structures
        data_structures = []
        memory_measurements = []
        
        for i in range(10):
            # Add data structure
            data = pd.DataFrame({
                'values': np.random.random(1000),
                'categories': np.random.choice(['A', 'B', 'C'], 1000)
            })
            data_structures.append(data)
            
            # Measure memory usage
            current_memory = psutil.Process().memory_info().rss
            memory_increase = current_memory - initial_memory
            memory_measurements.append(memory_increase)
        
        # Analyze memory growth
        if len(memory_measurements) > 2:
            # Memory should grow roughly linearly
            first_half_avg = np.mean(memory_measurements[:5])
            second_half_avg = np.mean(memory_measurements[5:])
            
            # Memory growth should be reasonable
            growth_ratio = second_half_avg / first_half_avg if first_half_avg > 0 else 1
            assert growth_ratio < 5.0, f"Excessive memory growth: {growth_ratio}x"
            
            # Total memory usage should be reasonable
            max_memory_mb = max(memory_measurements) / (1024 * 1024)
            assert max_memory_mb < 500, f"Excessive memory usage: {max_memory_mb} MB"
        
        # Clean up
        del data_structures
        import gc
        gc.collect()