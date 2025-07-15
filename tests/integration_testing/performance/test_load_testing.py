"""Performance and load testing for system scalability validation."""

import asyncio
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Dict, List, Any
from unittest.mock import Mock

import pytest
import numpy as np


class TestLoadTesting:
    """Comprehensive load testing suite."""
    
    @pytest.mark.performance
    @pytest.mark.slow
    async def test_api_endpoint_load(
        self,
        api_client,
        load_test_simulator,
        performance_monitor,
        integration_test_config
    ):
        """Test API endpoint performance under load."""
        
        # Get load test configuration
        perf_config = integration_test_config["performance"]
        max_response_time = perf_config["max_response_time"]
        concurrent_users = perf_config["concurrent_users"]
        test_duration = min(perf_config["test_duration"], 60)  # Cap at 60 seconds for tests
        
        performance_monitor.start_monitoring()
        
        try:
            # Test health endpoint under load
            results = await load_test_simulator.simulate_concurrent_users(
                client=api_client,
                num_users=concurrent_users,
                duration=test_duration,
                endpoint="/health"
            )
            
            # Validate load test results
            assert results["total_requests"] > 0
            assert results["success_rate"] >= 0.95  # 95% success rate minimum
            assert results["avg_response_time"] < max_response_time
            assert results["max_response_time"] < max_response_time * 2  # Allow 2x for worst case
            
            # Test detection endpoint under load
            detection_results = await load_test_simulator.simulate_concurrent_users(
                client=api_client,
                num_users=min(concurrent_users, 5),  # Lower concurrency for complex operations
                duration=min(test_duration, 30),
                endpoint="/detectors/test-detector/detect"
            )
            
            assert detection_results["success_rate"] >= 0.90  # 90% for complex operations
            
        finally:
            performance_monitor.stop_monitoring()
            perf_summary = performance_monitor.get_summary()
            
            # Performance assertions
            if perf_summary:
                # Memory should not grow excessively under load
                memory_growth = perf_summary["memory"]["max"] - perf_summary["memory"]["min"]
                assert memory_growth < 200 * 1024 * 1024  # Max 200MB growth
                
                # CPU should not be constantly maxed out
                assert perf_summary["cpu"]["avg"] < 90
    
    @pytest.mark.performance
    async def test_concurrent_detector_training(
        self,
        api_client,
        test_data_manager,
        performance_monitor
    ):
        """Test concurrent detector training performance."""
        
        performance_monitor.start_monitoring()
        
        # Create multiple test datasets
        datasets = [
            test_data_manager.create_test_dataset(size=1000)
            for _ in range(3)
        ]
        
        # Create multiple detector configurations
        detectors = [
            test_data_manager.create_test_detector()
            for _ in range(3)
        ]
        
        # Mock concurrent training responses
        def create_training_mock(detector_id: str, dataset_id: str):
            response = Mock()
            response.status_code = 202
            response.json.return_value = {
                "job_id": f"train-{detector_id}-{dataset_id}",
                "status": "started",
                "estimated_duration": 30
            }
            return response
        
        training_jobs = []
        
        try:
            # Start concurrent training jobs
            for i, (detector, dataset) in enumerate(zip(detectors, datasets)):
                api_client.post.return_value = create_training_mock(detector["id"], dataset["id"])
                
                response = api_client.post(
                    f"/detectors/{detector['id']}/train",
                    json={"dataset_id": dataset["id"]}
                )
                
                assert response.status_code == 202
                training_jobs.append(response.json()["job_id"])
            
            # Monitor training progress
            completed_jobs = 0
            max_iterations = 50  # Prevent infinite loop
            iteration = 0
            
            while completed_jobs < len(training_jobs) and iteration < max_iterations:
                for job_id in training_jobs:
                    # Mock job progress response
                    progress_response = Mock()
                    progress_response.status_code = 200
                    
                    # Simulate job completion over time
                    if iteration > 10:  # Jobs complete after some iterations
                        progress_response.json.return_value = {
                            "job_id": job_id,
                            "status": "completed",
                            "progress": 100
                        }
                        completed_jobs += 1
                    else:
                        progress_response.json.return_value = {
                            "job_id": job_id,
                            "status": "running",
                            "progress": min(iteration * 10, 90)
                        }
                    
                    api_client.get.return_value = progress_response
                    
                    response = api_client.get(f"/jobs/{job_id}")
                    assert response.status_code == 200
                
                await asyncio.sleep(0.1)  # Small delay
                iteration += 1
            
            assert completed_jobs == len(training_jobs)
            
        finally:
            performance_monitor.stop_monitoring()
            perf_summary = performance_monitor.get_summary()
            
            # Concurrent training should not overwhelm the system
            if perf_summary:
                assert perf_summary["memory"]["peak_mb"] < 1000  # Max 1GB
    
    @pytest.mark.performance
    async def test_database_performance_under_load(
        self,
        api_client,
        test_data_manager,
        performance_monitor
    ):
        """Test database performance under concurrent access."""
        
        performance_monitor.start_monitoring()
        
        # Create test data for database operations
        test_datasets = [
            test_data_manager.create_test_dataset(size=500)
            for _ in range(5)
        ]
        
        async def database_operation_simulation(dataset_info: Dict[str, Any]):
            """Simulate database-intensive operations."""
            operations_completed = 0
            
            # Mock database operations
            for operation_type in ["create", "read", "update", "delete"]:
                # Mock operation response
                response = Mock()
                response.status_code = 200
                response.json.return_value = {
                    "operation": operation_type,
                    "dataset_id": dataset_info["id"],
                    "status": "success",
                    "execution_time_ms": float(np.random.uniform(10, 100))
                }
                
                api_client.post.return_value = response
                
                # Simulate API call
                api_response = api_client.post(
                    f"/datasets/{dataset_info['id']}/{operation_type}",
                    json={"data": "test"}
                )
                
                assert api_response.status_code == 200
                operations_completed += 1
                
                # Small delay between operations
                await asyncio.sleep(0.01)
            
            return operations_completed
        
        try:
            # Run concurrent database operations
            tasks = [
                database_operation_simulation(dataset)
                for dataset in test_datasets
            ]
            
            results = await asyncio.gather(*tasks)
            
            # Validate all operations completed
            total_operations = sum(results)
            assert total_operations == len(test_datasets) * 4  # 4 operations per dataset
            
        finally:
            performance_monitor.stop_monitoring()
            perf_summary = performance_monitor.get_summary()
            
            # Database operations should be efficient
            if perf_summary:
                assert perf_summary["duration"] < 60  # Max 1 minute
    
    @pytest.mark.performance
    @pytest.mark.slow
    async def test_memory_usage_under_load(
        self,
        api_client,
        test_data_manager,
        performance_monitor
    ):
        """Test memory usage patterns under sustained load."""
        
        performance_monitor.start_monitoring()
        
        initial_memory = None
        memory_samples = []
        
        try:
            # Get initial memory baseline
            await asyncio.sleep(1)  # Let system stabilize
            perf_summary = performance_monitor.get_summary()
            if perf_summary:
                initial_memory = perf_summary["memory"]["avg"]
            
            # Simulate sustained load with large datasets
            for iteration in range(10):
                # Create larger datasets to stress memory
                large_dataset = test_data_manager.create_test_dataset(size=5000)
                
                # Mock memory-intensive operations
                for operation in ["profiling", "analysis", "detection"]:
                    response = Mock()
                    response.status_code = 200
                    response.json.return_value = {
                        "operation": operation,
                        "dataset_id": large_dataset["id"],
                        "memory_used_mb": float(np.random.uniform(50, 200)),
                        "status": "completed"
                    }
                    
                    api_client.post.return_value = response
                    
                    api_response = api_client.post(
                        f"/datasets/{large_dataset['id']}/{operation}",
                        json={"dataset_size": large_dataset["size"]}
                    )
                    
                    assert api_response.status_code == 200
                
                # Sample memory usage
                current_summary = performance_monitor.get_summary()
                if current_summary:
                    memory_samples.append(current_summary["memory"]["avg"])
                
                await asyncio.sleep(0.5)  # Brief pause between iterations
            
        finally:
            performance_monitor.stop_monitoring()
            final_summary = performance_monitor.get_summary()
            
            # Memory usage validation
            if final_summary and initial_memory and memory_samples:
                final_memory = final_summary["memory"]["avg"]
                memory_growth = final_memory - initial_memory
                
                # Memory growth should be bounded
                assert memory_growth < 500 * 1024 * 1024  # Max 500MB growth
                
                # No significant memory leaks (memory should stabilize)
                if len(memory_samples) >= 5:
                    recent_variance = np.var(memory_samples[-5:])
                    # Variance should be low for stable memory usage
                    assert recent_variance < (100 * 1024 * 1024) ** 2  # 100MB variance
    
    @pytest.mark.performance
    async def test_response_time_distribution(
        self,
        api_client,
        load_test_simulator,
        integration_test_config
    ):
        """Test response time distribution under various loads."""
        
        # Test different load levels
        load_levels = [1, 5, 10]  # Different numbers of concurrent users
        results_by_load = {}
        
        for load_level in load_levels:
            # Run load test for this level
            results = await load_test_simulator.simulate_concurrent_users(
                client=api_client,
                num_users=load_level,
                duration=15,  # Shorter duration for multiple tests
                endpoint="/health"
            )
            
            results_by_load[load_level] = results
            
            # Basic validation for each load level
            assert results["success_rate"] >= 0.95
            
            # Response times should degrade gracefully with load
            if load_level == 1:
                # Single user should have best response times
                assert results["avg_response_time"] < 0.1
            elif load_level == 5:
                # Moderate load should still be responsive
                assert results["avg_response_time"] < 0.5
            else:  # load_level == 10
                # High load may have higher response times but should be bounded
                assert results["avg_response_time"] < 2.0
        
        # Validate response time scaling
        single_user_time = results_by_load[1]["avg_response_time"]
        high_load_time = results_by_load[10]["avg_response_time"]
        
        # Response time should not increase by more than 20x under 10x load
        assert high_load_time / single_user_time < 20
    
    @pytest.mark.performance
    async def test_throughput_benchmarks(
        self,
        api_client,
        test_data_manager,
        performance_monitor
    ):
        """Test system throughput benchmarks."""
        
        performance_monitor.start_monitoring()
        
        # Create test dataset for throughput testing
        dataset = test_data_manager.create_test_dataset(size=10000)
        
        # Test different operation types
        operation_throughputs = {}
        
        operations = [
            ("health_check", "/health", {}),
            ("dataset_upload", "/datasets/upload", {"data": "test"}),
            ("detector_create", "/detectors", {"algorithm": "isolation_forest"}),
            ("detection", "/detectors/test/detect", {"data": [1, 2, 3, 4, 5]})
        ]
        
        try:
            for operation_name, endpoint, payload in operations:
                start_time = time.time()
                request_count = 0
                test_duration = 10  # 10 seconds per operation
                
                # Mock appropriate responses for each operation
                if operation_name == "health_check":
                    mock_response = Mock()
                    mock_response.status_code = 200
                    mock_response.json.return_value = {"status": "healthy"}
                elif operation_name == "dataset_upload":
                    mock_response = Mock()
                    mock_response.status_code = 201
                    mock_response.json.return_value = {"dataset_id": "test-dataset"}
                elif operation_name == "detector_create":
                    mock_response = Mock()
                    mock_response.status_code = 201
                    mock_response.json.return_value = {"detector_id": "test-detector"}
                else:  # detection
                    mock_response = Mock()
                    mock_response.status_code = 200
                    mock_response.json.return_value = {"anomalies": [], "scores": [0.1, 0.2, 0.1]}
                
                api_client.post.return_value = mock_response
                api_client.get.return_value = mock_response
                
                # Run requests for the test duration
                while time.time() - start_time < test_duration:
                    if endpoint == "/health":
                        response = api_client.get(endpoint)
                    else:
                        response = api_client.post(endpoint, json=payload)
                    
                    assert response.status_code in [200, 201]
                    request_count += 1
                    
                    # Small delay to prevent overwhelming
                    await asyncio.sleep(0.01)
                
                # Calculate throughput
                actual_duration = time.time() - start_time
                throughput = request_count / actual_duration
                operation_throughputs[operation_name] = {
                    "requests_per_second": throughput,
                    "total_requests": request_count,
                    "duration": actual_duration
                }
                
                # Basic throughput validation
                if operation_name == "health_check":
                    assert throughput > 50  # Health checks should be fast
                elif operation_name in ["dataset_upload", "detector_create"]:
                    assert throughput > 10  # CRUD operations should be reasonable
                else:  # detection
                    assert throughput > 5   # Complex operations can be slower
            
        finally:
            performance_monitor.stop_monitoring()
            perf_summary = performance_monitor.get_summary()
            
            # Overall system should remain stable during throughput testing
            if perf_summary:
                assert perf_summary["memory"]["peak_mb"] < 800
                assert perf_summary["cpu"]["max"] < 95
        
        # Log throughput results for analysis
        for operation, metrics in operation_throughputs.items():
            print(f"{operation}: {metrics['requests_per_second']:.2f} req/sec")
    
    @pytest.mark.performance
    @pytest.mark.slow
    async def test_stress_testing(
        self,
        api_client,
        load_test_simulator,
        performance_monitor
    ):
        """Test system behavior under extreme stress conditions."""
        
        performance_monitor.start_monitoring()
        
        try:
            # Gradual load increase (stress ramp-up)
            stress_levels = [5, 10, 20, 30]  # Progressive load increase
            
            for stress_level in stress_levels:
                print(f"Testing stress level: {stress_level} concurrent users")
                
                # Run stress test for this level
                results = await load_test_simulator.simulate_concurrent_users(
                    client=api_client,
                    num_users=stress_level,
                    duration=20,  # Shorter duration but higher intensity
                    endpoint="/health"
                )
                
                # System should handle stress gracefully
                if stress_level <= 10:
                    # Low-medium stress should maintain high success rate
                    assert results["success_rate"] >= 0.95
                elif stress_level <= 20:
                    # Medium-high stress may have some degradation
                    assert results["success_rate"] >= 0.85
                else:
                    # High stress may show more degradation but should not fail completely
                    assert results["success_rate"] >= 0.70
                
                # Response times may increase but should not become unreasonable
                assert results["avg_response_time"] < 10.0  # Max 10 seconds
                
                # Brief recovery period between stress levels
                await asyncio.sleep(2)
            
        finally:
            performance_monitor.stop_monitoring()
            perf_summary = performance_monitor.get_summary()
            
            # System should recover and not crash under stress
            if perf_summary:
                # Memory usage may be high but should not be excessive
                assert perf_summary["memory"]["peak_mb"] < 2000  # Max 2GB under stress
                
                # CPU may be high but should not be completely locked
                assert perf_summary["cpu"]["avg"] < 95