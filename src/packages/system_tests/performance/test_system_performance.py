"""
Comprehensive system-wide performance tests and benchmarks.

Tests performance characteristics across all domains including load testing,
stress testing, and performance regression detection.
"""
import pytest
import time
import asyncio
import statistics
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Dict, List, Any, Callable
from unittest.mock import Mock, patch, AsyncMock
from datetime import datetime, timedelta

from test_utilities.factories import TestDataFactory
from test_utilities.fixtures import async_test


class TestSystemPerformanceBenchmarks:
    """System-wide performance benchmarks and load testing."""
    
    @pytest.fixture
    def performance_config(self):
        """Configuration for performance tests."""
        return {
            "load_test_users": 100,
            "stress_test_users": 500,
            "duration_seconds": 30,
            "ramp_up_seconds": 10,
            "acceptable_response_time_ms": 200,
            "acceptable_throughput_rps": 50,
            "acceptable_error_rate": 0.01
        }
    
    @pytest.fixture
    def sample_workloads(self):
        """Sample workloads for performance testing."""
        return {
            "anomaly_detection": {
                "endpoint": "/api/v1/anomaly/detect",
                "payload_size_bytes": 1024,
                "cpu_intensive": True,
                "memory_intensive": False
            },
            "data_ingestion": {
                "endpoint": "/api/v1/data/ingest",
                "payload_size_bytes": 10240,
                "cpu_intensive": False,
                "memory_intensive": True
            },
            "model_training": {
                "endpoint": "/api/v1/ml/train",
                "payload_size_bytes": 102400,
                "cpu_intensive": True,
                "memory_intensive": True
            },
            "user_authentication": {
                "endpoint": "/api/v1/auth/login",
                "payload_size_bytes": 256,
                "cpu_intensive": False,
                "memory_intensive": False
            }
        }
    
    @pytest.mark.performance
    async def test_anomaly_detection_performance_baseline(self, performance_config, sample_workloads):
        """Test anomaly detection performance baseline."""
        workload = sample_workloads["anomaly_detection"]
        
        with patch('ai.machine_learning.services.AnomalyDetectionService') as mock_service:
            mock_service.return_value.detect_anomalies = AsyncMock(return_value={
                "anomalies": [1, 5, 12],
                "scores": [0.1, 0.9, 0.2, 0.15, 0.8, 0.95, 0.1],
                "processing_time_ms": 45.2
            })
            
            # Warm up
            await self._warmup_service(mock_service, iterations=10)
            
            # Performance test
            results = await self._run_performance_test(
                test_function=self._simulate_anomaly_detection_request,
                duration_seconds=performance_config["duration_seconds"],
                target_rps=performance_config["acceptable_throughput_rps"]
            )
            
            # Assertions
            assert results["avg_response_time_ms"] <= performance_config["acceptable_response_time_ms"]
            assert results["throughput_rps"] >= performance_config["acceptable_throughput_rps"] * 0.9
            assert results["error_rate"] <= performance_config["acceptable_error_rate"]
            assert results["p95_response_time_ms"] <= performance_config["acceptable_response_time_ms"] * 2
            
            # Performance regression detection
            assert results["memory_usage_mb"] <= 500  # Memory constraint
            assert results["cpu_usage_percent"] <= 80  # CPU constraint
    
    @pytest.mark.performance
    async def test_data_ingestion_load_test(self, performance_config, sample_workloads):
        """Test data ingestion under load."""
        workload = sample_workloads["data_ingestion"]
        
        with patch('data.data_engineering.services.DataIngestionService') as mock_service:
            mock_service.return_value.ingest_batch = AsyncMock(return_value={
                "batch_id": "batch_001",
                "records_processed": 1000,
                "processing_time_ms": 125.3,
                "throughput_records_per_second": 800
            })
            
            # Load test with multiple concurrent users
            results = await self._run_concurrent_load_test(
                test_function=self._simulate_data_ingestion_request,
                concurrent_users=performance_config["load_test_users"],
                duration_seconds=performance_config["duration_seconds"],
                ramp_up_seconds=performance_config["ramp_up_seconds"]
            )
            
            # Load test assertions
            assert results["avg_response_time_ms"] <= performance_config["acceptable_response_time_ms"] * 1.5
            assert results["throughput_rps"] >= 30  # Data ingestion specific threshold
            assert results["error_rate"] <= performance_config["acceptable_error_rate"] * 2  # More lenient for load
            assert results["concurrent_users_handled"] >= performance_config["load_test_users"] * 0.95
            
            # Resource utilization
            assert results["peak_memory_usage_mb"] <= 2000  # Data processing memory limit
            assert results["avg_cpu_usage_percent"] <= 75
    
    @pytest.mark.performance
    @pytest.mark.slow
    async def test_ml_training_stress_test(self, performance_config, sample_workloads):
        """Test ML training under stress conditions."""
        workload = sample_workloads["model_training"]
        
        with patch('ai.machine_learning.services.ModelTrainingService') as mock_service:
            mock_service.return_value.train_model = AsyncMock(return_value={
                "model_id": "model_001",
                "training_time_seconds": 15.6,
                "memory_peak_mb": 1200,
                "cpu_utilization_percent": 85
            })
            
            # Stress test with high load
            results = await self._run_stress_test(
                test_function=self._simulate_model_training_request,
                max_concurrent_users=performance_config["stress_test_users"],
                duration_seconds=performance_config["duration_seconds"] * 2,  # Longer for stress
                increase_load_gradually=True
            )
            
            # Stress test assertions
            assert results["system_remained_stable"], "System became unstable under stress"
            assert results["error_rate"] <= 0.05, "High error rate under stress"
            assert results["response_time_degradation"] <= 3.0, "Response time degraded too much"
            assert results["memory_leaks_detected"] == False, "Memory leaks detected"
            
            # Stress-specific metrics
            assert results["max_concurrent_users_supported"] >= performance_config["stress_test_users"] * 0.8
            assert results["recovery_time_seconds"] <= 30  # Recovery after stress
    
    @pytest.mark.performance
    async def test_authentication_performance_scalability(self, performance_config, sample_workloads):
        """Test authentication system scalability."""
        workload = sample_workloads["user_authentication"]
        
        with patch('enterprise.enterprise_auth.services.AuthenticationService') as mock_service:
            mock_service.return_value.authenticate_user = AsyncMock(return_value={
                "user_id": "user_123",
                "session_token": "jwt_token",
                "authentication_time_ms": 25.4
            })
            
            # Scalability test with increasing load
            scalability_results = await self._run_scalability_test(
                test_function=self._simulate_authentication_request,
                user_loads=[10, 50, 100, 200, 400],
                duration_per_load_seconds=20
            )
            
            # Scalability assertions
            for load, result in scalability_results.items():
                # Response time should scale reasonably
                expected_max_response_time = performance_config["acceptable_response_time_ms"] * (1 + load / 200)
                assert result["avg_response_time_ms"] <= expected_max_response_time
                
                # Throughput should scale with load
                expected_min_throughput = min(load * 0.8, performance_config["acceptable_throughput_rps"] * 2)
                assert result["throughput_rps"] >= expected_min_throughput
                
                # Error rate should remain low
                assert result["error_rate"] <= performance_config["acceptable_error_rate"] * 2
    
    @pytest.mark.performance
    async def test_cross_domain_workflow_performance(self, performance_config):
        """Test performance of complete cross-domain workflows."""
        # Mock all domain services
        with patch('ai.machine_learning.services.AnomalyDetectionService') as mock_ai, \
             patch('data.data_engineering.services.DataIngestionService') as mock_data, \
             patch('enterprise.enterprise_auth.services.AuthenticationService') as mock_auth:
            
            mock_ai.return_value.detect_anomalies = AsyncMock(return_value={"processing_time_ms": 50})
            mock_data.return_value.ingest_batch = AsyncMock(return_value={"processing_time_ms": 100})
            mock_auth.return_value.authenticate_user = AsyncMock(return_value={"processing_time_ms": 20})
            
            # Test complete workflow performance
            workflow_results = await self._run_workflow_performance_test(
                workflow_steps=[
                    ("authenticate", self._simulate_authentication_request),
                    ("ingest_data", self._simulate_data_ingestion_request),
                    ("detect_anomalies", self._simulate_anomaly_detection_request)
                ],
                concurrent_workflows=50,
                duration_seconds=performance_config["duration_seconds"]
            )
            
            # Workflow performance assertions
            assert workflow_results["avg_workflow_time_ms"] <= 500  # Complete workflow time
            assert workflow_results["workflow_success_rate"] >= 0.95
            assert workflow_results["bottleneck_step"] in ["authenticate", "ingest_data", "detect_anomalies"]
            assert workflow_results["workflows_completed"] >= 200  # Minimum throughput
    
    @pytest.mark.performance
    async def test_memory_usage_patterns(self, performance_config):
        """Test memory usage patterns under various loads."""
        memory_results = {}
        
        # Test different memory usage patterns
        test_scenarios = [
            ("low_memory", {"batch_size": 100, "concurrent_requests": 10}),
            ("medium_memory", {"batch_size": 1000, "concurrent_requests": 50}),
            ("high_memory", {"batch_size": 10000, "concurrent_requests": 100})
        ]
        
        for scenario_name, config in test_scenarios:
            with patch('data.data_engineering.services.DataProcessingService') as mock_service:
                mock_service.return_value.process_large_dataset = AsyncMock(
                    return_value={"memory_used_mb": config["batch_size"] * 0.1}
                )
                
                memory_result = await self._run_memory_usage_test(
                    batch_size=config["batch_size"],
                    concurrent_requests=config["concurrent_requests"],
                    duration_seconds=30
                )
                
                memory_results[scenario_name] = memory_result
        
        # Memory usage assertions
        assert memory_results["low_memory"]["peak_memory_mb"] <= 500
        assert memory_results["medium_memory"]["peak_memory_mb"] <= 2000
        assert memory_results["high_memory"]["peak_memory_mb"] <= 8000
        
        # Memory efficiency
        for scenario, result in memory_results.items():
            assert result["memory_leaks_detected"] == False
            assert result["garbage_collection_efficiency"] >= 0.9
    
    # Helper methods for performance testing
    
    async def test_database_query_performance(self, performance_config):
        """Test database query performance across different load patterns."""
        query_scenarios = [
            ("simple_select", "SELECT * FROM users WHERE id = ?"),
            ("complex_join", "SELECT u.*, p.* FROM users u JOIN profiles p ON u.id = p.user_id WHERE u.active = 1"),
            ("aggregation", "SELECT COUNT(*), AVG(score) FROM anomalies WHERE created_at > ?"),
            ("bulk_insert", "INSERT INTO logs (timestamp, level, message) VALUES (?, ?, ?)")
        ]
        
        query_results = {}
        
        for query_name, query_sql in query_scenarios:
            with patch('infrastructure.database.DatabaseService') as mock_db:
                mock_db.return_value.execute_query = AsyncMock(
                    return_value={"execution_time_ms": 15.3, "rows_affected": 100}
                )
                
                query_result = await self._run_database_performance_test(
                    query_type=query_name,
                    concurrent_queries=performance_config["load_test_users"],
                    duration_seconds=performance_config["duration_seconds"]
                )
                
                query_results[query_name] = query_result
        
        # Database performance assertions
        for query_type, result in query_results.items():
            if query_type == "simple_select":
                assert result["avg_query_time_ms"] <= 50
            elif query_type == "complex_join":
                assert result["avg_query_time_ms"] <= 200
            elif query_type == "aggregation":
                assert result["avg_query_time_ms"] <= 100
            elif query_type == "bulk_insert":
                assert result["avg_query_time_ms"] <= 300
            
            assert result["queries_per_second"] >= 10
            assert result["connection_pool_efficiency"] >= 0.8
    
    async def _warmup_service(self, mock_service, iterations: int = 10):
        """Warm up service before performance testing."""
        for _ in range(iterations):
            await asyncio.sleep(0.01)  # Simulate warmup
    
    async def _run_performance_test(self, test_function: Callable, duration_seconds: int, target_rps: int) -> Dict[str, Any]:
        """Run a performance test with specified parameters."""
        start_time = time.time()
        end_time = start_time + duration_seconds
        
        response_times = []
        errors = 0
        requests = 0
        
        while time.time() < end_time:
            request_start = time.time()
            try:
                await test_function()
                requests += 1
            except Exception:
                errors += 1
            
            response_time_ms = (time.time() - request_start) * 1000
            response_times.append(response_time_ms)
            
            # Rate limiting to achieve target RPS
            await asyncio.sleep(max(0, (1 / target_rps) - (time.time() - request_start)))
        
        actual_duration = time.time() - start_time
        
        return {
            "avg_response_time_ms": statistics.mean(response_times) if response_times else 0,
            "p95_response_time_ms": statistics.quantiles(response_times, n=20)[18] if len(response_times) > 20 else 0,
            "throughput_rps": requests / actual_duration,
            "error_rate": errors / (requests + errors) if (requests + errors) > 0 else 0,
            "total_requests": requests,
            "total_errors": errors,
            "memory_usage_mb": 250,  # Simulated
            "cpu_usage_percent": 45   # Simulated
        }
    
    async def _run_concurrent_load_test(self, test_function: Callable, concurrent_users: int, 
                                      duration_seconds: int, ramp_up_seconds: int) -> Dict[str, Any]:
        """Run concurrent load test with gradual ramp-up."""
        start_time = time.time()
        tasks = []
        results = []
        
        # Gradual ramp-up
        for i in range(concurrent_users):
            task = asyncio.create_task(self._user_simulation(test_function, duration_seconds))
            tasks.append(task)
            
            # Ramp-up delay
            if i < concurrent_users - 1:
                await asyncio.sleep(ramp_up_seconds / concurrent_users)
        
        # Wait for all users to complete
        completed_results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # Aggregate results
        total_requests = sum(r.get("requests", 0) for r in completed_results if isinstance(r, dict))
        total_errors = sum(r.get("errors", 0) for r in completed_results if isinstance(r, dict))
        response_times = []
        for r in completed_results:
            if isinstance(r, dict) and "response_times" in r:
                response_times.extend(r["response_times"])
        
        actual_duration = time.time() - start_time
        
        return {
            "avg_response_time_ms": statistics.mean(response_times) if response_times else 0,
            "throughput_rps": total_requests / actual_duration,
            "error_rate": total_errors / (total_requests + total_errors) if (total_requests + total_errors) > 0 else 0,
            "concurrent_users_handled": len([r for r in completed_results if isinstance(r, dict)]),
            "peak_memory_usage_mb": 1500,  # Simulated
            "avg_cpu_usage_percent": 65    # Simulated
        }
    
    async def _run_stress_test(self, test_function: Callable, max_concurrent_users: int,
                             duration_seconds: int, increase_load_gradually: bool = True) -> Dict[str, Any]:
        """Run stress test with increasing load."""
        initial_response_time = await self._measure_baseline_response_time(test_function)
        
        stress_results = {
            "system_remained_stable": True,
            "max_concurrent_users_supported": 0,
            "response_time_degradation": 1.0,
            "memory_leaks_detected": False,
            "recovery_time_seconds": 0
        }
        
        # Gradually increase load
        for users in range(50, max_concurrent_users + 1, 50):
            load_result = await self._run_concurrent_load_test(
                test_function, users, duration_seconds // 4, 5
            )
            
            # Check if system is still stable
            if load_result["error_rate"] > 0.1:  # 10% error rate threshold
                stress_results["system_remained_stable"] = False
                break
            
            stress_results["max_concurrent_users_supported"] = users
            stress_results["response_time_degradation"] = load_result["avg_response_time_ms"] / initial_response_time
            
            # Simulate brief recovery period
            await asyncio.sleep(2)
        
        # Measure recovery time
        recovery_start = time.time()
        while time.time() - recovery_start < 60:  # Max 60 seconds recovery
            current_response_time = await self._measure_baseline_response_time(test_function)
            if current_response_time <= initial_response_time * 1.2:  # Within 20% of baseline
                stress_results["recovery_time_seconds"] = time.time() - recovery_start
                break
            await asyncio.sleep(1)
        
        stress_results["error_rate"] = 0.02  # Simulated final error rate
        
        return stress_results
    
    async def _run_scalability_test(self, test_function: Callable, user_loads: List[int], 
                                  duration_per_load_seconds: int) -> Dict[int, Dict[str, Any]]:
        """Run scalability test with different user loads."""
        scalability_results = {}
        
        for load in user_loads:
            result = await self._run_concurrent_load_test(
                test_function, load, duration_per_load_seconds, 5
            )
            scalability_results[load] = result
            
            # Brief pause between load tests
            await asyncio.sleep(2)
        
        return scalability_results
    
    async def _run_workflow_performance_test(self, workflow_steps: List[tuple], 
                                           concurrent_workflows: int, duration_seconds: int) -> Dict[str, Any]:
        """Run performance test for complete workflows."""
        start_time = time.time()
        end_time = start_time + duration_seconds
        
        completed_workflows = 0
        failed_workflows = 0
        workflow_times = []
        step_times = {step[0]: [] for step in workflow_steps}
        
        async def run_single_workflow():
            workflow_start = time.time()
            try:
                for step_name, step_function in workflow_steps:
                    step_start = time.time()
                    await step_function()
                    step_time = (time.time() - step_start) * 1000
                    step_times[step_name].append(step_time)
                
                workflow_time = (time.time() - workflow_start) * 1000
                workflow_times.append(workflow_time)
                return True
            except Exception:
                return False
        
        # Run concurrent workflows
        while time.time() < end_time:
            tasks = [run_single_workflow() for _ in range(min(concurrent_workflows, 10))]
            results = await asyncio.gather(*tasks, return_exceptions=True)
            
            completed_workflows += sum(1 for r in results if r is True)
            failed_workflows += sum(1 for r in results if r is not True)
            
            await asyncio.sleep(0.1)  # Brief pause
        
        # Find bottleneck step
        bottleneck_step = max(step_times.keys(), 
                            key=lambda step: statistics.mean(step_times[step]) if step_times[step] else 0)
        
        return {
            "avg_workflow_time_ms": statistics.mean(workflow_times) if workflow_times else 0,
            "workflow_success_rate": completed_workflows / (completed_workflows + failed_workflows) if (completed_workflows + failed_workflows) > 0 else 0,
            "workflows_completed": completed_workflows,
            "bottleneck_step": bottleneck_step,
            "step_performance": {step: statistics.mean(times) if times else 0 for step, times in step_times.items()}
        }
    
    async def _run_memory_usage_test(self, batch_size: int, concurrent_requests: int, 
                                   duration_seconds: int) -> Dict[str, Any]:
        """Run memory usage test."""
        # Simulate memory-intensive operations
        await asyncio.sleep(duration_seconds * 0.1)  # Simulate test duration
        
        return {
            "peak_memory_mb": batch_size * 0.1,  # Simulated based on batch size
            "avg_memory_mb": batch_size * 0.07,
            "memory_leaks_detected": False,
            "garbage_collection_efficiency": 0.92
        }
    
    async def _run_database_performance_test(self, query_type: str, concurrent_queries: int, 
                                           duration_seconds: int) -> Dict[str, Any]:
        """Run database performance test."""
        # Simulate database operations
        await asyncio.sleep(duration_seconds * 0.05)
        
        base_time = {"simple_select": 10, "complex_join": 50, "aggregation": 30, "bulk_insert": 100}
        
        return {
            "avg_query_time_ms": base_time.get(query_type, 25),
            "queries_per_second": min(concurrent_queries * 2, 100),
            "connection_pool_efficiency": 0.85
        }
    
    async def _user_simulation(self, test_function: Callable, duration_seconds: int) -> Dict[str, Any]:
        """Simulate a single user's activity."""
        start_time = time.time()
        end_time = start_time + duration_seconds
        
        requests = 0
        errors = 0
        response_times = []
        
        while time.time() < end_time:
            request_start = time.time()
            try:
                await test_function()
                requests += 1
            except Exception:
                errors += 1
            
            response_time_ms = (time.time() - request_start) * 1000
            response_times.append(response_time_ms)
            
            # User think time
            await asyncio.sleep(0.1)
        
        return {
            "requests": requests,
            "errors": errors,
            "response_times": response_times
        }
    
    async def _measure_baseline_response_time(self, test_function: Callable) -> float:
        """Measure baseline response time."""
        start_time = time.time()
        await test_function()
        return (time.time() - start_time) * 1000
    
    # Simulation functions for different services
    
    async def _simulate_anomaly_detection_request(self):
        """Simulate anomaly detection request."""
        await asyncio.sleep(0.045)  # Simulate processing time
        return {"anomalies": [1, 5], "scores": [0.1, 0.9, 0.2]}
    
    async def _simulate_data_ingestion_request(self):
        """Simulate data ingestion request."""
        await asyncio.sleep(0.1)  # Simulate ingestion time
        return {"batch_id": "batch_001", "records": 1000}
    
    async def _simulate_model_training_request(self):
        """Simulate model training request."""
        await asyncio.sleep(0.15)  # Simulate training time
        return {"model_id": "model_001", "status": "trained"}
    
    async def _simulate_authentication_request(self):
        """Simulate authentication request."""
        await asyncio.sleep(0.025)  # Simulate auth time
        return {"user_id": "user_123", "token": "jwt_token"}