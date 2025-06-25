"""Integration tests for performance, scalability, and load testing."""

import asyncio
import time
import pytest
from httpx import AsyncClient

from tests.integration.conftest import IntegrationTestHelper


class TestPerformanceIntegration:
    """Test performance, scalability, and load scenarios."""

    @pytest.mark.asyncio
    async def test_api_performance_under_load(
        self,
        async_test_client: AsyncClient,
        integration_helper: IntegrationTestHelper,
        sample_dataset_csv: str,
        disable_auth
    ):
        """Test API performance under concurrent load."""
        
        # Setup
        dataset = await integration_helper.upload_dataset(
            sample_dataset_csv,
            "performance_test_dataset"
        )
        
        detector = await integration_helper.create_detector(
            dataset["id"],
            "isolation_forest"
        )
        
        await integration_helper.train_detector(detector["id"])
        
        # Test data for predictions
        test_data_points = [
            {"feature1": i * 0.1, "feature2": i * 0.2, "feature3": i * 0.05}
            for i in range(100)
        ]
        
        # Concurrent prediction load test
        async def make_prediction_batch(batch_id: int, batch_size: int):
            """Make a batch of predictions and measure performance."""
            start_time = time.time()
            results = []
            
            for i in range(batch_size):
                data_index = (batch_id * batch_size + i) % len(test_data_points)
                test_data = {"data": [test_data_points[data_index]]}
                
                try:
                    response = await async_test_client.post(
                        f"/api/detection/predict/{detector['id']}",
                        json=test_data
                    )
                    response.raise_for_status()
                    result = response.json()["data"]
                    results.append({
                        "success": True,
                        "response_time": time.time() - start_time,
                        "anomaly_score": result[0]["anomaly_score"]
                    })
                except Exception as e:
                    results.append({
                        "success": False,
                        "error": str(e),
                        "response_time": time.time() - start_time
                    })
            
            end_time = time.time()
            return {
                "batch_id": batch_id,
                "batch_size": batch_size,
                "total_time": end_time - start_time,
                "avg_response_time": (end_time - start_time) / batch_size,
                "results": results,
                "success_rate": sum(1 for r in results if r["success"]) / len(results)
            }
        
        # Run concurrent load test
        num_concurrent_batches = 10
        batch_size = 5
        
        start_load_test = time.time()
        batch_tasks = [
            make_prediction_batch(i, batch_size)
            for i in range(num_concurrent_batches)
        ]
        
        batch_results = await asyncio.gather(*batch_tasks, return_exceptions=True)
        end_load_test = time.time()
        
        # Analyze performance results
        successful_batches = [r for r in batch_results if not isinstance(r, Exception)]
        assert len(successful_batches) >= num_concurrent_batches * 0.9  # 90% success rate
        
        # Calculate performance metrics
        total_requests = num_concurrent_batches * batch_size
        total_time = end_load_test - start_load_test
        overall_throughput = total_requests / total_time
        
        avg_response_times = [r["avg_response_time"] for r in successful_batches]
        avg_response_time = sum(avg_response_times) / len(avg_response_times)
        
        success_rates = [r["success_rate"] for r in successful_batches]
        overall_success_rate = sum(success_rates) / len(success_rates)
        
        # Performance assertions
        assert overall_throughput > 10  # At least 10 requests per second
        assert avg_response_time < 2.0  # Average response time under 2 seconds
        assert overall_success_rate > 0.95  # 95% success rate
        
        print(f"Load test results:")
        print(f"  Total requests: {total_requests}")
        print(f"  Total time: {total_time:.2f}s")
        print(f"  Throughput: {overall_throughput:.2f} req/s")
        print(f"  Avg response time: {avg_response_time:.3f}s")
        print(f"  Success rate: {overall_success_rate:.2%}")

    @pytest.mark.asyncio
    async def test_streaming_performance_scalability(
        self,
        async_test_client: AsyncClient,
        integration_helper: IntegrationTestHelper,
        sample_dataset_csv: str,
        disable_auth
    ):
        """Test streaming performance scalability with multiple sessions."""
        
        # Setup
        dataset = await integration_helper.upload_dataset(
            sample_dataset_csv,
            "streaming_scale_dataset"
        )
        
        detector = await integration_helper.create_detector(
            dataset["id"],
            "isolation_forest"
        )
        
        await integration_helper.train_detector(detector["id"])
        
        # Create multiple streaming sessions for scale testing
        num_sessions = 5
        sessions = []
        
        for i in range(num_sessions):
            session_config = {
                "name": f"scale_test_session_{i}",
                "detector_id": detector["id"],
                "data_source": {
                    "source_type": "mock",
                    "connection_config": {"mock_data_rate": 20}
                },
                "configuration": {
                    "processing_mode": "real_time",
                    "batch_size": 1,
                    "max_throughput": 100
                }
            }
            
            response = await async_test_client.post(
                "/api/streaming/sessions?created_by=test_user",
                json=session_config
            )
            response.raise_for_status()
            session = response.json()["data"]
            sessions.append(session)
            integration_helper.created_resources["sessions"].append(session["id"])
        
        # Start all sessions
        for session in sessions:
            response = await async_test_client.post(f"/api/streaming/sessions/{session['id']}/start")
            response.raise_for_status()
        
        # Concurrent data processing across sessions
        async def process_data_load(session_id: str, num_messages: int):
            """Process data load for a specific session."""
            start_time = time.time()
            processed_count = 0
            error_count = 0
            
            for i in range(num_messages):
                test_data = {
                    "data": {
                        "timestamp": f"2024-12-25T10:{i:02d}:00Z",
                        "feature1": i * 0.1,
                        "feature2": i * 0.2,
                        "feature3": i * 0.05
                    }
                }
                
                try:
                    response = await async_test_client.post(
                        f"/api/streaming/sessions/{session_id}/process",
                        json=test_data
                    )
                    response.raise_for_status()
                    processed_count += 1
                except Exception as e:
                    error_count += 1
            
            end_time = time.time()
            
            # Get final metrics
            response = await async_test_client.get(f"/api/streaming/sessions/{session_id}/metrics")
            if response.status_code == 200:
                metrics = response.json()["data"]
            else:
                metrics = {}
            
            return {
                "session_id": session_id,
                "processed_count": processed_count,
                "error_count": error_count,
                "total_time": end_time - start_time,
                "throughput": processed_count / (end_time - start_time),
                "error_rate": error_count / (processed_count + error_count) if (processed_count + error_count) > 0 else 0,
                "final_metrics": metrics
            }
        
        # Run concurrent processing across all sessions
        messages_per_session = 20
        start_scale_test = time.time()
        
        processing_tasks = [
            process_data_load(session["id"], messages_per_session)
            for session in sessions
        ]
        
        session_results = await asyncio.gather(*processing_tasks, return_exceptions=True)
        end_scale_test = time.time()
        
        # Analyze scalability results
        successful_sessions = [r for r in session_results if not isinstance(r, Exception)]
        assert len(successful_sessions) >= num_sessions * 0.8  # 80% of sessions successful
        
        # Calculate aggregate metrics
        total_processed = sum(r["processed_count"] for r in successful_sessions)
        total_errors = sum(r["error_count"] for r in successful_sessions)
        total_scale_time = end_scale_test - start_scale_test
        
        aggregate_throughput = total_processed / total_scale_time
        aggregate_error_rate = total_errors / (total_processed + total_errors) if (total_processed + total_errors) > 0 else 0
        
        # Scalability assertions
        assert aggregate_throughput > 50  # At least 50 messages/second aggregate
        assert aggregate_error_rate < 0.1  # Less than 10% error rate
        assert total_processed >= messages_per_session * num_sessions * 0.8  # 80% message processing
        
        print(f"Streaming scale test results:")
        print(f"  Sessions: {num_sessions}")
        print(f"  Messages per session: {messages_per_session}")
        print(f"  Total processed: {total_processed}")
        print(f"  Total errors: {total_errors}")
        print(f"  Aggregate throughput: {aggregate_throughput:.2f} msg/s")
        print(f"  Error rate: {aggregate_error_rate:.2%}")
        
        # Stop all sessions
        for session in sessions:
            response = await async_test_client.post(f"/api/streaming/sessions/{session['id']}/stop")
            response.raise_for_status()

    @pytest.mark.asyncio
    async def test_database_performance_under_load(
        self,
        async_test_client: AsyncClient,
        integration_helper: IntegrationTestHelper,
        sample_dataset_csv: str,
        disable_auth
    ):
        """Test database performance under concurrent load."""
        
        # Test concurrent dataset operations
        async def concurrent_dataset_operations(operation_id: int):
            """Perform concurrent dataset operations."""
            start_time = time.time()
            operations = []
            
            try:
                # Upload dataset
                dataset = await integration_helper.upload_dataset(
                    sample_dataset_csv,
                    f"concurrent_dataset_{operation_id}"
                )
                operations.append({"operation": "upload", "success": True, "time": time.time() - start_time})
                
                # Get dataset stats
                response = await async_test_client.get(f"/api/datasets/{dataset['id']}/stats")
                response.raise_for_status()
                operations.append({"operation": "stats", "success": True, "time": time.time() - start_time})
                
                # Update dataset metadata
                update_data = {
                    "description": f"Updated description for dataset {operation_id}",
                    "tags": [f"concurrent_test_{operation_id}", "performance"]
                }
                response = await async_test_client.put(f"/api/datasets/{dataset['id']}", json=update_data)
                response.raise_for_status()
                operations.append({"operation": "update", "success": True, "time": time.time() - start_time})
                
                # List datasets
                response = await async_test_client.get("/api/datasets")
                response.raise_for_status()
                operations.append({"operation": "list", "success": True, "time": time.time() - start_time})
                
                return {
                    "operation_id": operation_id,
                    "total_time": time.time() - start_time,
                    "operations": operations,
                    "dataset_id": dataset["id"],
                    "success": True
                }
                
            except Exception as e:
                return {
                    "operation_id": operation_id,
                    "total_time": time.time() - start_time,
                    "operations": operations,
                    "error": str(e),
                    "success": False
                }
        
        # Run concurrent database operations
        num_concurrent_ops = 8
        start_db_test = time.time()
        
        db_tasks = [
            concurrent_dataset_operations(i)
            for i in range(num_concurrent_ops)
        ]
        
        db_results = await asyncio.gather(*db_tasks, return_exceptions=True)
        end_db_test = time.time()
        
        # Analyze database performance
        successful_ops = [r for r in db_results if not isinstance(r, Exception) and r["success"]]
        assert len(successful_ops) >= num_concurrent_ops * 0.8  # 80% success rate
        
        total_db_time = end_db_test - start_db_test
        avg_operation_time = sum(r["total_time"] for r in successful_ops) / len(successful_ops)
        
        # Count operation types
        operation_counts = {}
        operation_times = {}
        
        for result in successful_ops:
            for op in result["operations"]:
                op_type = op["operation"]
                operation_counts[op_type] = operation_counts.get(op_type, 0) + 1
                if op_type not in operation_times:
                    operation_times[op_type] = []
                operation_times[op_type].append(op["time"])
        
        # Database performance assertions
        assert avg_operation_time < 10.0  # Average operation time under 10 seconds
        assert total_db_time < 30.0  # Total test time under 30 seconds
        
        print(f"Database performance test results:")
        print(f"  Concurrent operations: {num_concurrent_ops}")
        print(f"  Successful operations: {len(successful_ops)}")
        print(f"  Total test time: {total_db_time:.2f}s")
        print(f"  Average operation time: {avg_operation_time:.2f}s")
        
        for op_type, times in operation_times.items():
            avg_time = sum(times) / len(times)
            print(f"  {op_type} operations: {len(times)}, avg time: {avg_time:.2f}s")
        
        # Cleanup created datasets
        for result in successful_ops:
            if "dataset_id" in result:
                try:
                    await async_test_client.delete(f"/api/datasets/{result['dataset_id']}")
                except:
                    pass  # Ignore cleanup errors

    @pytest.mark.asyncio
    async def test_memory_and_resource_usage(
        self,
        async_test_client: AsyncClient,
        integration_helper: IntegrationTestHelper,
        sample_dataset_csv: str,
        disable_auth
    ):
        """Test memory usage and resource consumption patterns."""
        
        # Setup for resource monitoring
        dataset = await integration_helper.upload_dataset(
            sample_dataset_csv,
            "resource_test_dataset"
        )
        
        detector = await integration_helper.create_detector(
            dataset["id"],
            "isolation_forest"
        )
        
        await integration_helper.train_detector(detector["id"])
        
        # Test 1: Large batch prediction memory usage
        large_batch_size = 500
        large_batch_data = {
            "data": [
                {"feature1": i * 0.001, "feature2": i * 0.002, "feature3": i * 0.0005}
                for i in range(large_batch_size)
            ]
        }
        
        start_memory_test = time.time()
        response = await async_test_client.post(
            f"/api/detection/predict/{detector['id']}",
            json=large_batch_data
        )
        response.raise_for_status()
        memory_test_time = time.time() - start_memory_test
        
        predictions = response.json()["data"]
        assert len(predictions) == large_batch_size
        assert memory_test_time < 30.0  # Should complete within 30 seconds
        
        # Test 2: Sustained prediction load (memory leak detection)
        sustained_batches = 20
        batch_size = 50
        
        memory_baseline_time = time.time()
        sustained_times = []
        
        for i in range(sustained_batches):
            batch_data = {
                "data": [
                    {"feature1": j * 0.01 + i, "feature2": j * 0.02 + i, "feature3": j * 0.005 + i}
                    for j in range(batch_size)
                ]
            }
            
            batch_start = time.time()
            response = await async_test_client.post(
                f"/api/detection/predict/{detector['id']}",
                json=batch_data
            )
            response.raise_for_status()
            batch_time = time.time() - batch_start
            sustained_times.append(batch_time)
        
        total_sustained_time = time.time() - memory_baseline_time
        
        # Analyze performance degradation (memory leak indicator)
        first_half_avg = sum(sustained_times[:sustained_batches//2]) / (sustained_batches//2)
        second_half_avg = sum(sustained_times[sustained_batches//2:]) / (sustained_batches//2)
        
        performance_degradation = (second_half_avg - first_half_avg) / first_half_avg
        
        # Resource usage assertions
        assert performance_degradation < 0.5  # Less than 50% performance degradation
        assert max(sustained_times) < 5.0  # No single batch should take more than 5 seconds
        assert total_sustained_time < 60.0  # Total sustained test under 60 seconds
        
        print(f"Memory and resource usage test results:")
        print(f"  Large batch ({large_batch_size} items): {memory_test_time:.2f}s")
        print(f"  Sustained batches: {sustained_batches}")
        print(f"  First half avg time: {first_half_avg:.3f}s")
        print(f"  Second half avg time: {second_half_avg:.3f}s")
        print(f"  Performance degradation: {performance_degradation:.2%}")
        
        # Test 3: Concurrent resource usage
        async def resource_intensive_task(task_id: int):
            """Resource-intensive task for concurrent testing."""
            start_time = time.time()
            
            # Multiple operations to stress resources
            operations = [
                # Model validation
                async_test_client.post(f"/api/detection/validate/{detector['id']}"),
                # Dataset stats
                async_test_client.get(f"/api/datasets/{dataset['id']}/stats"),
                # Predictions
                async_test_client.post(
                    f"/api/detection/predict/{detector['id']}",
                    json={"data": [{"feature1": task_id * 0.1, "feature2": task_id * 0.2, "feature3": task_id * 0.05}]}
                )
            ]
            
            responses = await asyncio.gather(*operations, return_exceptions=True)
            successful_ops = sum(1 for r in responses if not isinstance(r, Exception) and r.status_code == 200)
            
            return {
                "task_id": task_id,
                "time": time.time() - start_time,
                "successful_operations": successful_ops,
                "total_operations": len(operations)
            }
        
        # Run concurrent resource-intensive tasks
        num_concurrent_tasks = 6
        concurrent_start = time.time()
        
        resource_tasks = [
            resource_intensive_task(i)
            for i in range(num_concurrent_tasks)
        ]
        
        task_results = await asyncio.gather(*resource_tasks, return_exceptions=True)
        concurrent_total_time = time.time() - concurrent_start
        
        successful_tasks = [r for r in task_results if not isinstance(r, Exception)]
        avg_task_time = sum(r["time"] for r in successful_tasks) / len(successful_tasks)
        total_successful_ops = sum(r["successful_operations"] for r in successful_tasks)
        
        # Concurrent resource assertions
        assert len(successful_tasks) >= num_concurrent_tasks * 0.8  # 80% task success
        assert avg_task_time < 10.0  # Average task time under 10 seconds
        assert total_successful_ops >= num_concurrent_tasks * 2  # At least 2 ops per task on average
        
        print(f"Concurrent resource usage:")
        print(f"  Concurrent tasks: {num_concurrent_tasks}")
        print(f"  Successful tasks: {len(successful_tasks)}")
        print(f"  Total time: {concurrent_total_time:.2f}s")
        print(f"  Average task time: {avg_task_time:.2f}s")
        print(f"  Total successful operations: {total_successful_ops}")

    @pytest.mark.asyncio
    async def test_api_rate_limiting_and_throttling(
        self,
        async_test_client: AsyncClient,
        integration_helper: IntegrationTestHelper,
        sample_dataset_csv: str,
        disable_auth
    ):
        """Test API rate limiting and throttling behavior."""
        
        # Setup
        dataset = await integration_helper.upload_dataset(
            sample_dataset_csv,
            "rate_limit_dataset"
        )
        
        detector = await integration_helper.create_detector(
            dataset["id"],
            "isolation_forest"
        )
        
        await integration_helper.train_detector(detector["id"])
        
        # Test rapid-fire requests to trigger rate limiting
        async def rapid_requests(request_count: int, delay: float = 0.01):
            """Make rapid consecutive requests."""
            results = []
            start_time = time.time()
            
            for i in range(request_count):
                test_data = {"data": [{"feature1": i * 0.1, "feature2": i * 0.2, "feature3": i * 0.05}]}
                
                try:
                    response = await async_test_client.post(
                        f"/api/detection/predict/{detector['id']}",
                        json=test_data
                    )
                    
                    results.append({
                        "request_id": i,
                        "status_code": response.status_code,
                        "success": response.status_code == 200,
                        "rate_limited": response.status_code == 429,
                        "response_time": time.time() - start_time
                    })
                    
                    if delay > 0:
                        await asyncio.sleep(delay)
                        
                except Exception as e:
                    results.append({
                        "request_id": i,
                        "status_code": None,
                        "success": False,
                        "rate_limited": False,
                        "error": str(e),
                        "response_time": time.time() - start_time
                    })
            
            total_time = time.time() - start_time
            return {
                "results": results,
                "total_time": total_time,
                "request_rate": request_count / total_time,
                "success_count": sum(1 for r in results if r["success"]),
                "rate_limited_count": sum(1 for r in results if r.get("rate_limited", False))
            }
        
        # Test 1: Rapid requests without delay
        rapid_result = await rapid_requests(50, delay=0.01)
        
        # Test 2: Burst followed by normal rate
        burst_result = await rapid_requests(20, delay=0.001)  # Very rapid burst
        await asyncio.sleep(1)  # Cool down period
        normal_result = await rapid_requests(10, delay=0.5)   # Normal rate
        
        # Analyze rate limiting behavior
        total_requests = rapid_result["success_count"] + burst_result["success_count"] + normal_result["success_count"]
        total_rate_limited = rapid_result["rate_limited_count"] + burst_result["rate_limited_count"] + normal_result["rate_limited_count"]
        
        # Rate limiting assertions
        # Note: These may pass even without rate limiting implemented
        assert total_requests > 0  # At least some requests should succeed
        assert rapid_result["request_rate"] > 10  # Should achieve reasonable request rate
        
        # If rate limiting is implemented, we might see 429 responses
        if total_rate_limited > 0:
            assert burst_result["rate_limited_count"] >= rapid_result["rate_limited_count"]  # Burst should trigger more limiting
            assert normal_result["rate_limited_count"] == 0  # Normal rate should not be limited
        
        print(f"Rate limiting test results:")
        print(f"  Rapid requests: {rapid_result['success_count']}/{len(rapid_result['results'])} success, rate: {rapid_result['request_rate']:.2f} req/s")
        print(f"  Burst requests: {burst_result['success_count']}/{len(burst_result['results'])} success, rate: {burst_result['request_rate']:.2f} req/s")
        print(f"  Normal requests: {normal_result['success_count']}/{len(normal_result['results'])} success, rate: {normal_result['request_rate']:.2f} req/s")
        print(f"  Total rate limited: {total_rate_limited}")
        
        # Test 3: Sustained load with concurrent clients
        async def sustained_client_load(client_id: int, duration_seconds: int):
            """Simulate sustained load from a client."""
            start_time = time.time()
            request_count = 0
            success_count = 0
            
            while time.time() - start_time < duration_seconds:
                test_data = {"data": [{"feature1": client_id * 0.1, "feature2": request_count * 0.01, "feature3": 0.1}]}
                
                try:
                    response = await async_test_client.post(
                        f"/api/detection/predict/{detector['id']}",
                        json=test_data
                    )
                    request_count += 1
                    if response.status_code == 200:
                        success_count += 1
                    
                    await asyncio.sleep(0.1)  # 10 requests per second per client
                    
                except Exception:
                    request_count += 1
            
            actual_duration = time.time() - start_time
            return {
                "client_id": client_id,
                "requests": request_count,
                "successes": success_count,
                "duration": actual_duration,
                "success_rate": success_count / request_count if request_count > 0 else 0,
                "request_rate": request_count / actual_duration
            }
        
        # Run sustained load test with multiple clients
        num_clients = 3
        test_duration = 5  # 5 seconds
        
        sustained_start = time.time()
        client_tasks = [
            sustained_client_load(i, test_duration)
            for i in range(num_clients)
        ]
        
        client_results = await asyncio.gather(*client_tasks, return_exceptions=True)
        sustained_total_time = time.time() - sustained_start
        
        successful_clients = [r for r in client_results if not isinstance(r, Exception)]
        
        total_sustained_requests = sum(r["requests"] for r in successful_clients)
        total_sustained_successes = sum(r["successes"] for r in successful_clients)
        aggregate_success_rate = total_sustained_successes / total_sustained_requests if total_sustained_requests > 0 else 0
        aggregate_request_rate = total_sustained_requests / sustained_total_time
        
        # Sustained load assertions
        assert len(successful_clients) == num_clients  # All clients should complete
        assert aggregate_success_rate > 0.8  # 80% success rate under sustained load
        assert aggregate_request_rate > 15  # Aggregate rate should be reasonable
        
        print(f"Sustained load test results:")
        print(f"  Clients: {num_clients}")
        print(f"  Duration: {sustained_total_time:.2f}s")
        print(f"  Total requests: {total_sustained_requests}")
        print(f"  Total successes: {total_sustained_successes}")
        print(f"  Success rate: {aggregate_success_rate:.2%}")
        print(f"  Aggregate request rate: {aggregate_request_rate:.2f} req/s")