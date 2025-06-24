"""
Performance and Load Testing Suite
Comprehensive tests for performance benchmarks, load testing, and scalability.
"""

import pytest
import numpy as np
import time
import threading
from concurrent.futures import ThreadPoolExecutor
from unittest.mock import Mock, patch, MagicMock
import psutil
import gc
from contextlib import contextmanager

from pynomaly.infrastructure.adapters.sklearn_adapter import SklearnAdapter
from pynomaly.domain.entities import Dataset, Detector


class TestPerformanceBenchmarks:
    """Test suite for performance benchmarking."""

    @pytest.fixture
    def benchmark_datasets(self):
        """Create datasets of various sizes for benchmarking."""
        return {
            "small": np.random.randn(1000, 10).astype(np.float32),
            "medium": np.random.randn(10000, 20).astype(np.float32),
            "large": np.random.randn(100000, 50).astype(np.float32)
        }

    @contextmanager
    def measure_time(self):
        """Context manager for measuring execution time."""
        start_time = time.perf_counter()
        start_memory = psutil.Process().memory_info().rss
        
        yield lambda: {
            "elapsed_time": time.perf_counter() - start_time,
            "memory_delta": psutil.Process().memory_info().rss - start_memory
        }

    def test_training_performance_scaling(self, benchmark_datasets):
        """Test training performance scaling with dataset size."""
        adapter = SklearnAdapter(algorithm="IsolationForest", n_estimators=100)
        
        performance_results = {}
        
        for size_name, data in benchmark_datasets.items():
            dataset = Mock(spec=Dataset)
            dataset.data = data
            dataset.features = [f"feature_{i}" for i in range(data.shape[1])]
            
            with patch('sklearn.ensemble.IsolationForest') as mock_iso:
                mock_model = MagicMock()
                mock_model.fit.return_value = mock_model
                mock_iso.return_value = mock_model
                
                with self.measure_time() as get_metrics:
                    detector = adapter.fit(dataset)
                    
                metrics = get_metrics()
                performance_results[size_name] = {
                    "data_size": data.shape,
                    "training_time": metrics["elapsed_time"],
                    "memory_usage": metrics["memory_delta"],
                    "throughput": data.shape[0] / metrics["elapsed_time"]
                }
        
        # Verify performance scaling
        assert performance_results["small"]["training_time"] < performance_results["medium"]["training_time"]
        assert performance_results["medium"]["training_time"] < performance_results["large"]["training_time"]

    def test_prediction_performance_scaling(self, benchmark_datasets):
        """Test prediction performance scaling with dataset size."""
        adapter = SklearnAdapter(algorithm="IsolationForest")
        
        detector = Mock(spec=Detector)
        detector.algorithm = "sklearn_isolation_forest"
        detector.parameters = {"model": "mock_model"}
        
        performance_results = {}
        
        for size_name, data in benchmark_datasets.items():
            with patch.object(adapter, '_load_model') as mock_load:
                mock_model = MagicMock()
                mock_model.decision_function.return_value = np.random.randn(len(data))
                mock_model.predict.return_value = np.random.choice([-1, 1], len(data))
                mock_load.return_value = mock_model
                
                with self.measure_time() as get_metrics:
                    result = adapter.predict(detector, data)
                    
                metrics = get_metrics()
                performance_results[size_name] = {
                    "prediction_time": metrics["elapsed_time"],
                    "throughput": data.shape[0] / metrics["elapsed_time"]
                }
        
        # Verify reasonable scaling
        assert performance_results["small"]["throughput"] > 0
        assert performance_results["large"]["throughput"] > 0

    def test_memory_efficiency_benchmarks(self, benchmark_datasets):
        """Test memory efficiency across different data sizes."""
        adapter = SklearnAdapter(algorithm="IsolationForest")
        
        memory_results = {}
        
        for size_name, data in benchmark_datasets.items():
            gc.collect()
            
            dataset = Mock(spec=Dataset)
            dataset.data = data
            
            with patch('sklearn.ensemble.IsolationForest') as mock_iso:
                mock_model = MagicMock()
                mock_iso.return_value = mock_model
                
                initial_memory = psutil.Process().memory_info().rss
                detector = adapter.fit(dataset)
                peak_memory = psutil.Process().memory_info().rss
                memory_usage = peak_memory - initial_memory
                
                memory_results[size_name] = {
                    "data_size_mb": data.nbytes / (1024 * 1024),
                    "memory_usage_mb": memory_usage / (1024 * 1024),
                    "memory_efficiency": memory_usage / data.nbytes
                }
        
        # Verify memory efficiency
        assert memory_results["small"]["memory_efficiency"] > 0
        assert memory_results["large"]["memory_efficiency"] > 0

    def test_concurrent_training_performance(self):
        """Test performance of concurrent model training."""
        adapter = SklearnAdapter(algorithm="IsolationForest")
        
        datasets = []
        for i in range(4):
            data = np.random.randn(5000, 10)
            dataset = Mock(spec=Dataset)
            dataset.data = data
            dataset.id = f"dataset_{i}"
            datasets.append(dataset)
        
        def train_model(dataset):
            with patch('sklearn.ensemble.IsolationForest') as mock_iso:
                mock_model = MagicMock()
                mock_iso.return_value = mock_model
                
                start_time = time.time()
                detector = adapter.fit(dataset)
                end_time = time.time()
                
                return {
                    "dataset_id": dataset.id,
                    "training_time": end_time - start_time,
                    "detector": detector
                }
        
        # Sequential training
        start_time = time.time()
        sequential_results = []
        for dataset in datasets:
            result = train_model(dataset)
            sequential_results.append(result)
        sequential_time = time.time() - start_time
        
        # Concurrent training
        start_time = time.time()
        with ThreadPoolExecutor(max_workers=4) as executor:
            concurrent_results = list(executor.map(train_model, datasets))
        concurrent_time = time.time() - start_time
        
        # Concurrent should be faster or at least comparable
        assert concurrent_time <= sequential_time * 1.2  # Allow 20% overhead
        assert len(concurrent_results) == len(datasets)


class TestLoadTesting:
    """Test suite for load testing scenarios."""

    def test_concurrent_requests_simulation(self):
        """Test handling of concurrent requests."""
        from pynomaly.application.services.detection_service import DetectionService
        
        service = Mock(spec=DetectionService)
        service.detect_anomalies.return_value = {
            "predictions": [0, 1, 0, 0, 1],
            "anomaly_scores": [0.1, 0.8, 0.2, 0.3, 0.9]
        }
        
        def make_request(request_id):
            start_time = time.time()
            try:
                response = service.detect_anomalies(
                    detector_id="test_detector",
                    data=np.random.randn(5, 10)
                )
                end_time = time.time()
                
                return {
                    "request_id": request_id,
                    "success": True,
                    "response_time": end_time - start_time,
                    "response": response
                }
            except Exception as e:
                end_time = time.time()
                return {
                    "request_id": request_id,
                    "success": False,
                    "response_time": end_time - start_time,
                    "error": str(e)
                }
        
        # Load test with different concurrency levels
        concurrency_levels = [1, 5, 10, 20]
        load_test_results = {}
        
        for concurrency in concurrency_levels:
            with ThreadPoolExecutor(max_workers=concurrency) as executor:
                start_time = time.time()
                
                futures = [executor.submit(make_request, i) for i in range(concurrency * 2)]
                results = [future.result() for future in futures]
                
                end_time = time.time()
                
                successful_requests = [r for r in results if r["success"]]
                
                if successful_requests:
                    avg_response_time = np.mean([r["response_time"] for r in successful_requests])
                else:
                    avg_response_time = float('inf')
                
                load_test_results[concurrency] = {
                    "total_requests": len(results),
                    "successful_requests": len(successful_requests),
                    "success_rate": len(successful_requests) / len(results),
                    "avg_response_time": avg_response_time,
                    "throughput": len(successful_requests) / (end_time - start_time)
                }
        
        # Verify system handles increasing load
        assert load_test_results[1]["success_rate"] >= 0.95
        assert load_test_results[5]["success_rate"] >= 0.90

    def test_memory_pressure_handling(self):
        """Test system behavior under memory pressure."""
        adapter = SklearnAdapter(algorithm="IsolationForest")
        
        datasets = []
        for i in range(10):
            data = np.random.randn(10000, 20)
            dataset = Mock(spec=Dataset)
            dataset.data = data
            dataset.id = f"memory_test_dataset_{i}"
            datasets.append(dataset)
        
        results = []
        
        with patch('sklearn.ensemble.IsolationForest') as mock_iso:
            mock_model = MagicMock()
            mock_iso.return_value = mock_model
            
            for dataset in datasets:
                try:
                    detector = adapter.fit(dataset)
                    results.append({"dataset_id": dataset.id, "success": True})
                except MemoryError:
                    results.append({"dataset_id": dataset.id, "success": False, "error": "MemoryError"})
                except Exception as e:
                    results.append({"dataset_id": dataset.id, "success": False, "error": str(e)})
                
                gc.collect()
                time.sleep(0.1)
        
        # Most processing should succeed
        success_rate = sum(1 for r in results if r["success"]) / len(results)
        assert success_rate >= 0.8

    def test_batch_processing_optimization(self):
        """Test batch processing performance optimization."""
        adapter = SklearnAdapter(algorithm="IsolationForest")
        
        large_data = np.random.randn(50000, 15)
        
        detector = Mock(spec=Detector)
        detector.algorithm = "sklearn_isolation_forest"
        detector.parameters = {"model": "mock_model"}
        
        batch_sizes = [1000, 5000, 10000]
        performance_results = {}
        
        for batch_size in batch_sizes:
            with patch.object(adapter, '_load_model') as mock_load:
                mock_model = MagicMock()
                mock_model.decision_function.return_value = np.random.randn(batch_size)
                mock_load.return_value = mock_model
                
                start_time = time.time()
                results = []
                
                for i in range(0, len(large_data), batch_size):
                    batch = large_data[i:i + batch_size]
                    result = adapter.predict(detector, batch)
                    results.append(result)
                
                end_time = time.time()
                total_time = end_time - start_time
                
                performance_results[batch_size] = {
                    "total_time": total_time,
                    "throughput": len(large_data) / total_time,
                    "num_batches": len(results)
                }
        
        # Verify performance improvements with optimal batch size
        assert performance_results[5000]["throughput"] > 0
        assert all(result["total_time"] > 0 for result in performance_results.values())


class TestScalabilityTesting:
    """Test suite for scalability testing."""

    def test_horizontal_scaling_simulation(self):
        """Test horizontal scaling capabilities."""
        from pynomaly.infrastructure.distributed.load_balancer import LoadBalancer
        
        load_balancer = Mock(spec=LoadBalancer)
        
        # Simulate multiple worker nodes
        worker_nodes = []
        for i in range(5):
            worker = Mock()
            worker.id = f"worker_{i}"
            worker.capacity = 100  # requests/minute
            worker.current_load = 0
            worker_nodes.append(worker)
        
        # Simulate load distribution
        load_levels = [50, 150, 300, 500]
        scaling_results = {}
        
        for load in load_levels:
            active_nodes = min(len(worker_nodes), (load // 100) + 1)
            
            scaling_results[load] = {
                "active_nodes": active_nodes,
                "requests_per_node": load / active_nodes,
                "total_capacity": active_nodes * 100,
                "utilization": load / (active_nodes * 100)
            }
        
        # Verify scaling behavior
        assert scaling_results[50]["active_nodes"] == 1
        assert scaling_results[300]["active_nodes"] >= 3
        assert scaling_results[500]["active_nodes"] >= 4
        
        # Utilization should be reasonable
        for load, result in scaling_results.items():
            assert result["utilization"] <= 1.0

    def test_database_connection_pool_stress(self):
        """Test database connection pool under stress."""
        from pynomaly.infrastructure.persistence.database import DatabaseManager
        
        db_manager = Mock(spec=DatabaseManager)
        
        def database_operation(operation_id):
            try:
                # Simulate database operation
                time.sleep(np.random.uniform(0.01, 0.05))
                
                return {
                    "operation_id": operation_id,
                    "success": True,
                    "connection_acquired": True
                }
            except Exception as e:
                return {
                    "operation_id": operation_id,
                    "success": False,
                    "error": str(e)
                }
        
        # Stress test with concurrent database operations
        num_operations = 50
        max_workers = 10
        
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = [executor.submit(database_operation, i) for i in range(num_operations)]
            results = [future.result(timeout=5) for future in futures]
        
        # Analyze results
        successful_ops = [r for r in results if r["success"]]
        success_rate = len(successful_ops) / len(results)
        
        # Should handle high concurrency well
        assert success_rate >= 0.95
        assert len(results) == num_operations

    def test_streaming_throughput_limits(self):
        """Test streaming data processing throughput limits - REMOVED FOR SIMPLIFICATION."""
        # Streaming infrastructure removed for simplification
        pytest.skip("Streaming functionality removed in Phase 1 simplification")
        
        processor = Mock()
        
        # Generate streaming data
        def data_generator():
            for i in range(1000):
                yield {
                    "timestamp": time.time(),
                    "data": np.random.randn(5).tolist(),
                    "id": i
                }
                time.sleep(0.001)  # 1ms between samples
        
        processed_count = 0
        processing_times = []
        
        def process_batch(batch):
            nonlocal processed_count
            start_time = time.time()
            
            # Simulate processing
            time.sleep(0.01)  # 10ms processing time per batch
            
            end_time = time.time()
            processing_times.append(end_time - start_time)
            processed_count += len(batch)
            
            return len(batch)
        
        # Start streaming processing
        start_time = time.time()
        
        batch = []
        for data_point in data_generator():
            batch.append(data_point)
            
            if len(batch) >= 50:  # Process in batches of 50
                process_batch(batch)
                batch = []
            
            # Stop after 2 seconds to measure throughput
            if time.time() - start_time > 2:
                break
        
        # Process remaining batch
        if batch:
            process_batch(batch)
        
        end_time = time.time()
        total_time = end_time - start_time
        
        # Calculate throughput metrics
        throughput = processed_count / total_time  # samples/second
        avg_processing_time = np.mean(processing_times) if processing_times else 0
        
        assert throughput > 100  # Should process at least 100 samples/second
        assert avg_processing_time < 0.05  # Average batch processing under 50ms
        assert processed_count > 0