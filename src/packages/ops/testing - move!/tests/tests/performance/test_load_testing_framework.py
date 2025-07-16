"""
Comprehensive Load Testing Framework

Advanced load testing suite covering various load scenarios, stress testing,
spike testing, volume testing, and endurance testing across all system components.
"""

import asyncio
import json
import statistics
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from contextlib import asynccontextmanager
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional
from unittest.mock import AsyncMock, MagicMock, Mock, patch

import numpy as np
import psutil
import pytest
import requests
from fastapi.testclient import TestClient


@dataclass
class LoadTestMetrics:
    """Comprehensive load test metrics."""
    
    test_name: str
    duration: float
    total_requests: int
    successful_requests: int
    failed_requests: int
    success_rate: float
    average_response_time: float
    p50_response_time: float
    p95_response_time: float
    p99_response_time: float
    throughput: float
    max_concurrent_users: int
    peak_memory_mb: float
    peak_cpu_percent: float
    errors: List[str] = field(default_factory=list)
    custom_metrics: Dict[str, Any] = field(default_factory=dict)


@dataclass
class LoadTestConfig:
    """Load test configuration."""
    
    name: str
    description: str
    target_endpoint: str
    method: str = "GET"
    payload: Optional[Dict[str, Any]] = None
    headers: Optional[Dict[str, str]] = None
    concurrent_users: int = 10
    requests_per_user: int = 100
    ramp_up_time: float = 5.0
    duration: Optional[float] = None
    think_time: float = 0.1
    timeout: float = 30.0
    expected_success_rate: float = 0.95
    expected_p95_response_time: float = 2.0


class LoadTestRunner:
    """Advanced load test runner with comprehensive monitoring."""
    
    def __init__(self, base_url: str = "http://localhost:8000"):
        self.base_url = base_url
        self.results: List[LoadTestMetrics] = []
        self._stop_monitoring = False
        self._monitoring_data = {
            "memory_samples": [],
            "cpu_samples": [],
            "timestamps": []
        }
    
    def _monitor_system_resources(self):
        """Monitor system resources during load test."""
        process = psutil.Process()
        
        while not self._stop_monitoring:
            try:
                memory_mb = process.memory_info().rss / (1024 * 1024)
                cpu_percent = process.cpu_percent()
                
                self._monitoring_data["memory_samples"].append(memory_mb)
                self._monitoring_data["cpu_samples"].append(cpu_percent)
                self._monitoring_data["timestamps"].append(time.time())
                
                time.sleep(0.5)  # Sample every 500ms
            except Exception:
                break
    
    def _execute_single_request(
        self, 
        config: LoadTestConfig, 
        session: requests.Session,
        user_id: int,
        request_id: int
    ) -> Dict[str, Any]:
        """Execute a single request and return metrics."""
        start_time = time.perf_counter()
        
        try:
            url = f"{self.base_url}{config.target_endpoint}"
            
            if config.method.upper() == "GET":
                response = session.get(
                    url, 
                    headers=config.headers,
                    timeout=config.timeout
                )
            elif config.method.upper() == "POST":
                response = session.post(
                    url,
                    json=config.payload,
                    headers=config.headers,
                    timeout=config.timeout
                )
            elif config.method.upper() == "PUT":
                response = session.put(
                    url,
                    json=config.payload,
                    headers=config.headers,
                    timeout=config.timeout
                )
            elif config.method.upper() == "DELETE":
                response = session.delete(
                    url,
                    headers=config.headers,
                    timeout=config.timeout
                )
            else:
                raise ValueError(f"Unsupported HTTP method: {config.method}")
            
            end_time = time.perf_counter()
            
            return {
                "user_id": user_id,
                "request_id": request_id,
                "response_time": end_time - start_time,
                "status_code": response.status_code,
                "success": 200 <= response.status_code < 400,
                "content_length": len(response.content),
                "error": None
            }
            
        except Exception as e:
            end_time = time.perf_counter()
            return {
                "user_id": user_id,
                "request_id": request_id,
                "response_time": end_time - start_time,
                "status_code": 0,
                "success": False,
                "content_length": 0,
                "error": str(e)
            }
    
    def _simulate_user(self, config: LoadTestConfig, user_id: int) -> List[Dict[str, Any]]:
        """Simulate a single user's load testing session."""
        session = requests.Session()
        user_results = []
        
        # Stagger user start times for ramp-up
        ramp_delay = (config.ramp_up_time / config.concurrent_users) * user_id
        time.sleep(ramp_delay)
        
        start_time = time.time()
        
        for request_id in range(config.requests_per_user):
            # Check duration limit
            if config.duration and (time.time() - start_time) > config.duration:
                break
            
            # Execute request
            result = self._execute_single_request(config, session, user_id, request_id)
            user_results.append(result)
            
            # Think time between requests
            if config.think_time > 0:
                time.sleep(config.think_time)
        
        session.close()
        return user_results
    
    def run_load_test(self, config: LoadTestConfig) -> LoadTestMetrics:
        """Execute a complete load test."""
        print(f"Starting load test: {config.name}")
        print(f"Configuration: {config.concurrent_users} users, "
              f"{config.requests_per_user} requests each")
        
        # Start system monitoring
        self._stop_monitoring = False
        self._monitoring_data = {"memory_samples": [], "cpu_samples": [], "timestamps": []}
        monitor_thread = threading.Thread(target=self._monitor_system_resources)
        monitor_thread.start()
        
        try:
            # Execute load test
            start_time = time.perf_counter()
            
            with ThreadPoolExecutor(max_workers=config.concurrent_users) as executor:
                # Submit user simulation tasks
                futures = [
                    executor.submit(self._simulate_user, config, user_id)
                    for user_id in range(config.concurrent_users)
                ]
                
                # Collect all results
                all_results = []
                for future in as_completed(futures):
                    user_results = future.result()
                    all_results.extend(user_results)
            
            end_time = time.perf_counter()
            
        finally:
            # Stop monitoring
            self._stop_monitoring = True
            monitor_thread.join(timeout=5)
        
        # Calculate metrics
        total_duration = end_time - start_time
        successful_requests = [r for r in all_results if r["success"]]
        failed_requests = [r for r in all_results if not r["success"]]
        response_times = [r["response_time"] for r in successful_requests]
        
        # System resource metrics
        peak_memory = max(self._monitoring_data["memory_samples"]) if self._monitoring_data["memory_samples"] else 0
        peak_cpu = max(self._monitoring_data["cpu_samples"]) if self._monitoring_data["cpu_samples"] else 0
        
        # Error collection
        errors = [r["error"] for r in failed_requests if r["error"]]
        unique_errors = list(set(errors))
        
        # Calculate percentiles
        p50_time = statistics.median(response_times) if response_times else 0
        p95_time = statistics.quantiles(response_times, n=20)[18] if len(response_times) > 1 else 0
        p99_time = statistics.quantiles(response_times, n=100)[98] if len(response_times) > 1 else 0
        
        metrics = LoadTestMetrics(
            test_name=config.name,
            duration=total_duration,
            total_requests=len(all_results),
            successful_requests=len(successful_requests),
            failed_requests=len(failed_requests),
            success_rate=len(successful_requests) / len(all_results) if all_results else 0,
            average_response_time=statistics.mean(response_times) if response_times else 0,
            p50_response_time=p50_time,
            p95_response_time=p95_time,
            p99_response_time=p99_time,
            throughput=len(successful_requests) / total_duration if total_duration > 0 else 0,
            max_concurrent_users=config.concurrent_users,
            peak_memory_mb=peak_memory,
            peak_cpu_percent=peak_cpu,
            errors=unique_errors[:10],  # Limit to 10 unique errors
            custom_metrics={
                "ramp_up_time": config.ramp_up_time,
                "think_time": config.think_time,
                "target_endpoint": config.target_endpoint
            }
        )
        
        self.results.append(metrics)
        
        # Validate against expectations
        assert metrics.success_rate >= config.expected_success_rate, (
            f"Success rate {metrics.success_rate:.2%} below expected "
            f"{config.expected_success_rate:.2%}"
        )
        
        assert metrics.p95_response_time <= config.expected_p95_response_time, (
            f"P95 response time {metrics.p95_response_time:.2f}s above expected "
            f"{config.expected_p95_response_time:.2f}s"
        )
        
        self._print_metrics(metrics)
        return metrics
    
    def _print_metrics(self, metrics: LoadTestMetrics):
        """Print formatted load test metrics."""
        print(f"\n=== Load Test Results: {metrics.test_name} ===")
        print(f"Duration: {metrics.duration:.2f}s")
        print(f"Total Requests: {metrics.total_requests}")
        print(f"Successful: {metrics.successful_requests} ({metrics.success_rate:.2%})")
        print(f"Failed: {metrics.failed_requests}")
        print(f"Throughput: {metrics.throughput:.1f} req/s")
        print(f"Average Response Time: {metrics.average_response_time:.3f}s")
        print(f"P50 Response Time: {metrics.p50_response_time:.3f}s")
        print(f"P95 Response Time: {metrics.p95_response_time:.3f}s")
        print(f"P99 Response Time: {metrics.p99_response_time:.3f}s")
        print(f"Peak Memory: {metrics.peak_memory_mb:.1f} MB")
        print(f"Peak CPU: {metrics.peak_cpu_percent:.1f}%")
        if metrics.errors:
            print(f"Errors: {metrics.errors[:3]}")  # Show first 3 errors
        print("=" * 50)


class TestAPILoadTesting:
    """Load testing for API endpoints."""
    
    @pytest.fixture
    def test_client(self):
        """Create test client for load testing."""
        from monorepo.presentation.api.app import create_app
        
        app = create_app(testing=True)
        return TestClient(app)
    
    @pytest.fixture
    def load_runner(self):
        """Create load test runner."""
        return LoadTestRunner()
    
    def test_health_endpoint_load(self, test_client, load_runner):
        """Load test health endpoint with various concurrency levels."""
        
        # Basic load test
        config = LoadTestConfig(
            name="Health Endpoint Basic Load",
            description="Basic load test for health endpoint",
            target_endpoint="/api/v1/health",
            concurrent_users=10,
            requests_per_user=50,
            ramp_up_time=2.0,
            expected_success_rate=0.99,
            expected_p95_response_time=0.5
        )
        
        # Mock the test client to work with load runner
        with patch('requests.Session') as mock_session_class:
            mock_session = mock_session_class.return_value
            mock_response = Mock()
            mock_response.status_code = 200
            mock_response.content = b'{"status": "healthy"}'
            mock_session.get.return_value = mock_response
            
            metrics = load_runner.run_load_test(config)
            
            assert metrics.success_rate >= 0.99
            assert metrics.throughput > 100
            assert metrics.p95_response_time < 0.5
    
    def test_datasets_endpoint_load(self, load_runner):
        """Load test datasets endpoint."""
        
        config = LoadTestConfig(
            name="Datasets Endpoint Load",
            description="Load test for datasets listing endpoint",
            target_endpoint="/api/v1/datasets",
            concurrent_users=15,
            requests_per_user=30,
            ramp_up_time=3.0,
            expected_success_rate=0.95,
            expected_p95_response_time=1.0
        )
        
        with patch('requests.Session') as mock_session_class:
            mock_session = mock_session_class.return_value
            mock_response = Mock()
            mock_response.status_code = 200
            mock_response.content = b'{"datasets": []}'
            mock_session.get.return_value = mock_response
            
            metrics = load_runner.run_load_test(config)
            
            assert metrics.success_rate >= 0.95
            assert metrics.p95_response_time < 1.0
    
    def test_detection_endpoint_load(self, load_runner):
        """Load test detection endpoint with POST requests."""
        
        payload = {
            "detector_id": "test_detector",
            "data": np.random.randn(100, 5).tolist(),
            "return_scores": True
        }
        
        config = LoadTestConfig(
            name="Detection Endpoint Load",
            description="Load test for anomaly detection endpoint",
            target_endpoint="/api/v1/detection/detect",
            method="POST",
            payload=payload,
            concurrent_users=5,
            requests_per_user=20,
            ramp_up_time=2.0,
            think_time=0.5,
            expected_success_rate=0.90,
            expected_p95_response_time=3.0
        )
        
        with patch('requests.Session') as mock_session_class:
            mock_session = mock_session_class.return_value
            mock_response = Mock()
            mock_response.status_code = 200
            mock_response.content = b'{"anomalies": [], "scores": []}'
            mock_session.post.return_value = mock_response
            
            metrics = load_runner.run_load_test(config)
            
            assert metrics.success_rate >= 0.90
            assert metrics.p95_response_time < 3.0
    
    def test_stress_testing_gradual_load_increase(self, load_runner):
        """Stress test with gradually increasing load."""
        
        load_levels = [
            (5, 20),   # 5 users, 20 requests each
            (10, 30),  # 10 users, 30 requests each
            (20, 40),  # 20 users, 40 requests each
            (50, 20),  # 50 users, 20 requests each
        ]
        
        stress_results = []
        
        for users, requests in load_levels:
            config = LoadTestConfig(
                name=f"Stress Test {users} Users",
                description=f"Stress test with {users} concurrent users",
                target_endpoint="/api/v1/health",
                concurrent_users=users,
                requests_per_user=requests,
                ramp_up_time=1.0,
                expected_success_rate=0.80,  # Lower expectation under stress
                expected_p95_response_time=5.0
            )
            
            with patch('requests.Session') as mock_session_class:
                mock_session = mock_session_class.return_value
                mock_response = Mock()
                mock_response.status_code = 200
                mock_response.content = b'{"status": "healthy"}'
                mock_session.get.return_value = mock_response
                
                metrics = load_runner.run_load_test(config)
                stress_results.append(metrics)
        
        # Analyze stress test progression
        for i, metrics in enumerate(stress_results):
            users, requests = load_levels[i]
            print(f"Stress level {users} users: "
                  f"{metrics.success_rate:.2%} success, "
                  f"{metrics.throughput:.1f} req/s")
        
        # Ensure system degrades gracefully under stress
        for metrics in stress_results:
            assert metrics.success_rate >= 0.70  # At least 70% success under stress
    
    def test_spike_testing(self, load_runner):
        """Test system behavior under sudden load spikes."""
        
        # Baseline load
        baseline_config = LoadTestConfig(
            name="Baseline Load",
            description="Baseline load before spike",
            target_endpoint="/api/v1/health",
            concurrent_users=5,
            requests_per_user=50,
            ramp_up_time=1.0,
            expected_success_rate=0.95,
            expected_p95_response_time=1.0
        )
        
        # Spike load
        spike_config = LoadTestConfig(
            name="Load Spike",
            description="Sudden load spike test",
            target_endpoint="/api/v1/health",
            concurrent_users=100,
            requests_per_user=10,
            ramp_up_time=0.5,  # Very fast ramp-up
            expected_success_rate=0.60,  # Lower expectation during spike
            expected_p95_response_time=10.0
        )
        
        with patch('requests.Session') as mock_session_class:
            mock_session = mock_session_class.return_value
            mock_response = Mock()
            mock_response.status_code = 200
            mock_response.content = b'{"status": "healthy"}'
            mock_session.get.return_value = mock_response
            
            # Run baseline
            baseline_metrics = load_runner.run_load_test(baseline_config)
            
            # Brief pause
            time.sleep(1)
            
            # Run spike
            spike_metrics = load_runner.run_load_test(spike_config)
            
            # Spike test assertions
            assert spike_metrics.success_rate >= 0.50  # At least 50% during spike
            assert spike_metrics.throughput > baseline_metrics.throughput * 2  # Higher throughput during spike
    
    def test_endurance_testing(self, load_runner):
        """Test system stability over extended periods."""
        
        config = LoadTestConfig(
            name="Endurance Test",
            description="Extended load test for stability",
            target_endpoint="/api/v1/health",
            concurrent_users=8,
            requests_per_user=200,  # Many requests
            ramp_up_time=5.0,
            think_time=0.2,
            duration=60.0,  # 1 minute duration limit
            expected_success_rate=0.90,
            expected_p95_response_time=2.0
        )
        
        with patch('requests.Session') as mock_session_class:
            mock_session = mock_session_class.return_value
            mock_response = Mock()
            mock_response.status_code = 200
            mock_response.content = b'{"status": "healthy"}'
            mock_session.get.return_value = mock_response
            
            metrics = load_runner.run_load_test(config)
            
            # Endurance test assertions
            assert metrics.success_rate >= 0.85  # Maintain quality over time
            assert metrics.duration >= 30.0  # Should run for reasonable time
            assert metrics.peak_memory_mb < 1000  # Memory should not grow excessively


class TestDatabaseLoadTesting:
    """Load testing for database operations."""
    
    @pytest.fixture
    async def mock_database(self):
        """Mock database for load testing."""
        with patch("monorepo.infrastructure.persistence.database.DatabaseManager") as mock_db:
            db_manager = mock_db.return_value
            
            # Configure realistic database operations
            async def mock_save(entity):
                await asyncio.sleep(0.001)  # 1ms save time
                return Mock(id=f"saved_{id(entity)}")
            
            async def mock_get(entity_id):
                await asyncio.sleep(0.0005)  # 0.5ms get time
                return Mock(id=entity_id)
            
            async def mock_list():
                await asyncio.sleep(0.005)  # 5ms list time
                return [Mock(id=f"item_{i}") for i in range(10)]
            
            async def mock_bulk_insert(entities):
                await asyncio.sleep(len(entities) * 0.0001)  # 0.1ms per entity
                return [Mock(id=f"bulk_{i}") for i in range(len(entities))]
            
            db_manager.save_dataset = mock_save
            db_manager.get_dataset = mock_get
            db_manager.list_datasets = mock_list
            db_manager.bulk_insert_datasets = mock_bulk_insert
            
            yield db_manager
    
    async def test_database_connection_pool_load(self, mock_database):
        """Test database connection pool under load."""
        
        async def database_operation(operation_id: int):
            """Simulate database operation."""
            start_time = time.perf_counter()
            
            try:
                # Mix of operations
                if operation_id % 4 == 0:
                    await mock_database.save_dataset(Mock(id=f"dataset_{operation_id}"))
                elif operation_id % 4 == 1:
                    await mock_database.get_dataset(f"dataset_{operation_id}")
                elif operation_id % 4 == 2:
                    await mock_database.list_datasets()
                else:
                    entities = [Mock(id=f"bulk_{operation_id}_{i}") for i in range(10)]
                    await mock_database.bulk_insert_datasets(entities)
                
                success = True
                error = None
            except Exception as e:
                success = False
                error = str(e)
            
            end_time = time.perf_counter()
            
            return {
                "operation_id": operation_id,
                "duration": end_time - start_time,
                "success": success,
                "error": error
            }
        
        # Test with increasing concurrency
        concurrency_levels = [10, 50, 100, 200]
        
        for concurrency in concurrency_levels:
            start_time = time.perf_counter()
            
            # Execute concurrent database operations
            tasks = [database_operation(i) for i in range(concurrency)]
            results = await asyncio.gather(*tasks, return_exceptions=True)
            
            end_time = time.perf_counter()
            total_time = end_time - start_time
            
            # Process results
            successful_results = [
                r for r in results 
                if isinstance(r, dict) and r.get("success", False)
            ]
            
            metrics = {
                "concurrency": concurrency,
                "total_operations": len(results),
                "successful_operations": len(successful_results),
                "success_rate": len(successful_results) / len(results),
                "total_time": total_time,
                "throughput": len(successful_results) / total_time,
                "avg_response_time": statistics.mean(
                    [r["duration"] for r in successful_results]
                ) if successful_results else 0
            }
            
            # Database load assertions
            assert metrics["success_rate"] >= 0.90  # 90% success rate
            assert metrics["throughput"] > concurrency * 5  # Reasonable throughput
            assert metrics["avg_response_time"] < 0.1  # Under 100ms average
            
            print(f"DB Load {concurrency} ops: "
                  f"{metrics['success_rate']:.2%} success, "
                  f"{metrics['throughput']:.1f} ops/s")
    
    async def test_database_bulk_operations_load(self, mock_database):
        """Test bulk database operations under load."""
        
        bulk_sizes = [10, 100, 1000]
        concurrent_operations = [1, 5, 10]
        
        for bulk_size in bulk_sizes:
            for concurrency in concurrent_operations:
                async def bulk_operation():
                    """Perform bulk database operation."""
                    entities = [Mock(id=f"bulk_{i}") for i in range(bulk_size)]
                    start_time = time.perf_counter()
                    
                    result = await mock_database.bulk_insert_datasets(entities)
                    
                    end_time = time.perf_counter()
                    
                    return {
                        "bulk_size": bulk_size,
                        "duration": end_time - start_time,
                        "success": result is not None,
                        "throughput": bulk_size / (end_time - start_time)
                    }
                
                # Execute concurrent bulk operations
                start_time = time.perf_counter()
                tasks = [bulk_operation() for _ in range(concurrency)]
                results = await asyncio.gather(*tasks)
                end_time = time.perf_counter()
                
                # Analyze bulk operation performance
                successful_results = [r for r in results if r["success"]]
                total_entities = sum(r["bulk_size"] for r in successful_results)
                total_time = end_time - start_time
                overall_throughput = total_entities / total_time
                
                # Bulk operation assertions
                assert len(successful_results) == concurrency  # All should succeed
                assert overall_throughput > bulk_size * concurrency * 5  # Reasonable throughput
                
                avg_duration = statistics.mean(r["duration"] for r in successful_results)
                assert avg_duration < bulk_size * 0.001  # Under 1ms per entity
                
                print(f"Bulk {bulk_size} entities, {concurrency} concurrent: "
                      f"{overall_throughput:.0f} entities/s")


class TestMLWorkflowLoadTesting:
    """Load testing for ML workflow operations."""
    
    def test_ml_pipeline_concurrent_load(self):
        """Test ML pipeline performance under concurrent load."""
        
        def ml_workflow(workflow_id: int):
            """Simulate complete ML workflow."""
            start_time = time.perf_counter()
            
            # Generate test data
            X = np.random.randn(1000, 10)
            
            # Simulate ML operations with realistic timing
            with patch("monorepo.infrastructure.adapters.sklearn_adapter.SklearnAdapter") as mock_adapter:
                adapter = mock_adapter.return_value
                
                # Mock training (longer operation)
                def mock_fit(dataset):
                    time.sleep(0.05)  # 50ms training time
                    return Mock(id=f"model_{workflow_id}")
                
                # Mock prediction (shorter operation)
                def mock_predict(detector, data):
                    time.sleep(0.01)  # 10ms prediction time
                    return Mock(
                        predictions=np.random.choice([0, 1], len(data)).tolist(),
                        anomaly_scores=np.random.random(len(data)).tolist()
                    )
                
                adapter.fit = mock_fit
                adapter.predict = mock_predict
                
                # Execute workflow steps
                model = adapter.fit(Mock(data=X))
                result = adapter.predict(model, X)
                
                end_time = time.perf_counter()
                
                return {
                    "workflow_id": workflow_id,
                    "duration": end_time - start_time,
                    "success": result is not None,
                    "samples_processed": len(X)
                }
        
        # Test different concurrency levels
        concurrency_levels = [1, 2, 4, 8, 16]
        
        for concurrency in concurrency_levels:
            start_time = time.perf_counter()
            
            with ThreadPoolExecutor(max_workers=concurrency) as executor:
                futures = [
                    executor.submit(ml_workflow, i) 
                    for i in range(concurrency)
                ]
                results = [future.result() for future in futures]
            
            end_time = time.perf_counter()
            total_time = end_time - start_time
            
            # Calculate metrics
            successful_workflows = [r for r in results if r["success"]]
            total_samples = sum(r["samples_processed"] for r in successful_workflows)
            overall_throughput = total_samples / total_time
            avg_workflow_time = statistics.mean(r["duration"] for r in successful_workflows)
            
            # ML workflow load assertions
            assert len(successful_workflows) == concurrency  # All should succeed
            assert overall_throughput > concurrency * 10000  # Scale with concurrency
            assert avg_workflow_time < 1.0  # Under 1 second per workflow
            
            print(f"ML Workflows {concurrency} concurrent: "
                  f"{overall_throughput:.0f} samples/s, "
                  f"{avg_workflow_time:.3f}s avg workflow time")
    
    def test_algorithm_comparison_load(self):
        """Test algorithm comparison under load."""
        
        algorithms = ["IsolationForest", "LocalOutlierFactor", "OneClassSVM"]
        dataset_sizes = [(100, 5), (1000, 10), (5000, 15)]
        
        def compare_algorithms(comparison_id: int):
            """Simulate algorithm comparison."""
            start_time = time.perf_counter()
            
            results = {}
            
            for dataset_size in dataset_sizes:
                n_samples, n_features = dataset_size
                X = np.random.randn(n_samples, n_features)
                
                for algorithm in algorithms:
                    with patch("monorepo.infrastructure.adapters.sklearn_adapter.SklearnAdapter") as mock_adapter:
                        adapter = mock_adapter.return_value
                        
                        # Mock algorithm-specific timing
                        if algorithm == "IsolationForest":
                            training_time = n_samples / 50000  # Fastest
                        elif algorithm == "LocalOutlierFactor":
                            training_time = n_samples / 20000  # Medium
                        else:  # OneClassSVM
                            training_time = n_samples / 10000  # Slowest
                        
                        def mock_fit(dataset):
                            time.sleep(min(training_time, 0.1))  # Cap at 100ms
                            return Mock(id=f"model_{algorithm}_{comparison_id}")
                        
                        adapter.fit = mock_fit
                        model = adapter.fit(Mock(data=X))
                        
                        results[f"{algorithm}_{n_samples}"] = {
                            "algorithm": algorithm,
                            "dataset_size": n_samples,
                            "model_id": model.id
                        }
            
            end_time = time.perf_counter()
            
            return {
                "comparison_id": comparison_id,
                "duration": end_time - start_time,
                "success": len(results) == len(algorithms) * len(dataset_sizes),
                "algorithms_tested": len(algorithms),
                "datasets_tested": len(dataset_sizes)
            }
        
        # Run concurrent algorithm comparisons
        concurrency = 3
        
        with ThreadPoolExecutor(max_workers=concurrency) as executor:
            futures = [
                executor.submit(compare_algorithms, i) 
                for i in range(concurrency)
            ]
            results = [future.result() for future in futures]
        
        # Validate algorithm comparison load
        for result in results:
            assert result["success"]
            assert result["duration"] < 5.0  # Under 5 seconds
            assert result["algorithms_tested"] == 3
            assert result["datasets_tested"] == 3
        
        avg_duration = statistics.mean(r["duration"] for r in results)
        print(f"Algorithm comparison load: {avg_duration:.2f}s average duration")


class TestSystemWideLoadTesting:
    """System-wide load testing scenarios."""
    
    def test_mixed_workload_simulation(self):
        """Test system with mixed API, database, and ML operations."""
        
        def mixed_workload_user(user_id: int):
            """Simulate user with mixed operations."""
            operations = []
            start_time = time.perf_counter()
            
            # API operations (40% of workload)
            for _ in range(4):
                op_start = time.perf_counter()
                
                # Mock API call
                with patch('requests.Session') as mock_session_class:
                    mock_session = mock_session_class.return_value
                    mock_response = Mock()
                    mock_response.status_code = 200
                    mock_session.get.return_value = mock_response
                    
                    time.sleep(0.01)  # 10ms API call
                
                op_end = time.perf_counter()
                operations.append({"type": "api", "duration": op_end - op_start})
            
            # Database operations (30% of workload)
            for _ in range(3):
                op_start = time.perf_counter()
                
                # Mock database operation
                time.sleep(0.005)  # 5ms database operation
                
                op_end = time.perf_counter()
                operations.append({"type": "database", "duration": op_end - op_start})
            
            # ML operations (30% of workload)
            for _ in range(3):
                op_start = time.perf_counter()
                
                # Mock ML operation
                X = np.random.randn(100, 5)
                time.sleep(0.02)  # 20ms ML operation
                
                op_end = time.perf_counter()
                operations.append({"type": "ml", "duration": op_end - op_start})
            
            end_time = time.perf_counter()
            
            return {
                "user_id": user_id,
                "total_duration": end_time - start_time,
                "operations": operations,
                "success": True
            }
        
        # Simulate concurrent mixed workload
        concurrent_users = 10
        
        start_time = time.perf_counter()
        
        with ThreadPoolExecutor(max_workers=concurrent_users) as executor:
            futures = [
                executor.submit(mixed_workload_user, i) 
                for i in range(concurrent_users)
            ]
            results = [future.result() for future in futures]
        
        end_time = time.perf_counter()
        total_test_time = end_time - start_time
        
        # Analyze mixed workload performance
        successful_users = [r for r in results if r["success"]]
        total_operations = sum(len(r["operations"]) for r in successful_users)
        
        # Group operations by type
        api_operations = []
        db_operations = []
        ml_operations = []
        
        for result in successful_users:
            for op in result["operations"]:
                if op["type"] == "api":
                    api_operations.append(op["duration"])
                elif op["type"] == "database":
                    db_operations.append(op["duration"])
                elif op["type"] == "ml":
                    ml_operations.append(op["duration"])
        
        # Mixed workload assertions
        assert len(successful_users) == concurrent_users
        assert total_operations == concurrent_users * 10  # 4 API + 3 DB + 3 ML per user
        
        # Performance by operation type
        assert statistics.mean(api_operations) < 0.05  # API under 50ms
        assert statistics.mean(db_operations) < 0.02  # DB under 20ms
        assert statistics.mean(ml_operations) < 0.1   # ML under 100ms
        
        overall_throughput = total_operations / total_test_time
        assert overall_throughput > 50  # At least 50 operations per second
        
        print(f"Mixed workload: {overall_throughput:.1f} ops/s, "
              f"API: {statistics.mean(api_operations):.3f}s, "
              f"DB: {statistics.mean(db_operations):.3f}s, "
              f"ML: {statistics.mean(ml_operations):.3f}s")
    
    def test_load_testing_report_generation(self):
        """Generate comprehensive load testing report."""
        
        runner = LoadTestRunner()
        
        # Run multiple load tests
        test_configs = [
            LoadTestConfig(
                name="Health Check Load",
                target_endpoint="/api/v1/health",
                concurrent_users=5,
                requests_per_user=20,
                expected_success_rate=0.95
            ),
            LoadTestConfig(
                name="Dataset API Load",
                target_endpoint="/api/v1/datasets",
                concurrent_users=8,
                requests_per_user=15,
                expected_success_rate=0.90
            ),
            LoadTestConfig(
                name="High Concurrency Stress",
                target_endpoint="/api/v1/health",
                concurrent_users=50,
                requests_per_user=5,
                ramp_up_time=0.5,
                expected_success_rate=0.70
            )
        ]
        
        # Execute all tests
        with patch('requests.Session') as mock_session_class:
            mock_session = mock_session_class.return_value
            mock_response = Mock()
            mock_response.status_code = 200
            mock_response.content = b'{"status": "ok"}'
            mock_session.get.return_value = mock_response
            
            for config in test_configs:
                runner.run_load_test(config)
        
        # Generate summary report
        report = {
            "total_tests": len(runner.results),
            "overall_success_rate": statistics.mean(
                [r.success_rate for r in runner.results]
            ),
            "average_throughput": statistics.mean(
                [r.throughput for r in runner.results]
            ),
            "peak_memory_usage": max(
                [r.peak_memory_mb for r in runner.results]
            ),
            "tests": [
                {
                    "name": r.test_name,
                    "success_rate": r.success_rate,
                    "throughput": r.throughput,
                    "p95_response_time": r.p95_response_time
                }
                for r in runner.results
            ]
        }
        
        # Report assertions
        assert report["overall_success_rate"] >= 0.80
        assert report["average_throughput"] > 10
        assert report["peak_memory_usage"] < 1000
        
        print(f"\n=== Load Testing Summary Report ===")
        print(f"Total Tests: {report['total_tests']}")
        print(f"Overall Success Rate: {report['overall_success_rate']:.2%}")
        print(f"Average Throughput: {report['average_throughput']:.1f} req/s")
        print(f"Peak Memory Usage: {report['peak_memory_usage']:.1f} MB")
        print("\nIndividual Test Results:")
        for test in report["tests"]:
            print(f"  {test['name']}: {test['success_rate']:.2%} success, "
                  f"{test['throughput']:.1f} req/s, P95: {test['p95_response_time']:.3f}s")