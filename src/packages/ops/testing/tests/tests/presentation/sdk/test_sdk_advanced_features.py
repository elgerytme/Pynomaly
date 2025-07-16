"""
Advanced SDK Features Testing
============================

This module provides comprehensive testing for advanced SDK features including
streaming, concurrent operations, caching, and performance optimization.
"""

import asyncio
import hashlib
import json
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from unittest.mock import AsyncMock, Mock

import numpy as np
import pytest


class TestSDKStreamingFeatures:
    """Test suite for SDK streaming features."""

    @pytest.fixture
    def mock_streaming_client(self):
        """Create mock streaming client."""
        client = AsyncMock()
        client.base_url = "https://api.pynomaly.com"
        client.headers = {"X-API-Key": "test-key"}
        return client

    @pytest.fixture
    def streaming_data_generator(self):
        """Create streaming data generator."""

        def generate_data():
            for i in range(100):
                yield {
                    "sample_id": i,
                    "timestamp": datetime.now().isoformat(),
                    "features": [np.random.random() for _ in range(5)],
                    "metadata": {"batch": i // 10},
                }

        return generate_data

    @pytest.mark.asyncio
    async def test_real_time_streaming_detection(
        self, mock_streaming_client, streaming_data_generator
    ):
        """Test real-time streaming detection."""

        # Mock streaming detection response
        async def mock_stream_detect(detector_id, data_stream):
            async for data_point in data_stream:
                # Simulate processing delay
                await asyncio.sleep(0.001)

                # Generate mock prediction
                prediction = {
                    "sample_id": data_point["sample_id"],
                    "prediction": np.random.choice([0, 1], p=[0.9, 0.1]),
                    "score": np.random.random(),
                    "timestamp": datetime.now().isoformat(),
                    "processing_time": 0.002,
                }
                yield prediction

        mock_streaming_client.stream_detection = mock_stream_detect

        # Create async data stream
        async def async_data_generator():
            for data in streaming_data_generator():
                yield data
                await asyncio.sleep(0.001)  # Simulate real-time arrival

        # Process streaming data
        results = []
        async for result in mock_streaming_client.stream_detection(
            "detector-123", async_data_generator()
        ):
            results.append(result)
            if len(results) >= 10:  # Test first 10 samples
                break

        assert len(results) == 10
        assert all("sample_id" in r for r in results)
        assert all("prediction" in r for r in results)
        assert all("score" in r for r in results)

    @pytest.mark.asyncio
    async def test_streaming_with_buffering(self, mock_streaming_client):
        """Test streaming with buffering mechanism."""

        # Mock buffered streaming
        class StreamBuffer:
            def __init__(self, buffer_size=10):
                self.buffer_size = buffer_size
                self.buffer = []
                self.processed_count = 0

            async def add_sample(self, sample):
                self.buffer.append(sample)
                if len(self.buffer) >= self.buffer_size:
                    await self.flush_buffer()

            async def flush_buffer(self):
                if self.buffer:
                    # Process buffer batch
                    batch_results = []
                    for sample in self.buffer:
                        result = {
                            "sample_id": sample["id"],
                            "prediction": 0,
                            "score": 0.1,
                            "batch_id": self.processed_count,
                        }
                        batch_results.append(result)

                    self.processed_count += 1
                    self.buffer.clear()
                    return batch_results
                return []

        buffer = StreamBuffer(buffer_size=5)

        # Add samples to buffer
        for i in range(12):
            sample = {"id": i, "data": [1, 2, 3]}
            results = await buffer.add_sample(sample)

        # Flush remaining buffer
        final_results = await buffer.flush_buffer()

        assert buffer.processed_count >= 2  # At least 2 batches processed
        assert len(buffer.buffer) == 2  # 2 samples remaining

    @pytest.mark.asyncio
    async def test_streaming_error_recovery(self, mock_streaming_client):
        """Test streaming error recovery and reconnection."""
        connection_attempts = 0
        max_retries = 3

        async def mock_unreliable_stream(detector_id, data_stream):
            nonlocal connection_attempts
            connection_attempts += 1

            if connection_attempts <= 2:
                # Simulate connection failure
                await asyncio.sleep(0.1)
                raise ConnectionError(
                    f"Connection failed (attempt {connection_attempts})"
                )

            # Successful connection on 3rd attempt
            count = 0
            async for data in data_stream:
                yield {"sample_id": count, "prediction": 0, "score": 0.1}
                count += 1
                if count >= 5:
                    break

        async def data_generator():
            for i in range(10):
                yield {"id": i, "data": [i, i + 1, i + 2]}

        # Mock retry logic
        async def stream_with_retry(detector_id, data_stream, max_retries=3):
            for attempt in range(max_retries + 1):
                try:
                    async for result in mock_unreliable_stream(
                        detector_id, data_stream
                    ):
                        yield result
                    break
                except ConnectionError as e:
                    if attempt == max_retries:
                        raise e
                    await asyncio.sleep(0.1 * (2**attempt))  # Exponential backoff

        # Test streaming with retry
        results = []
        async for result in stream_with_retry("detector-123", data_generator()):
            results.append(result)

        assert len(results) == 5
        assert connection_attempts == 3  # Failed twice, succeeded on third

    @pytest.mark.asyncio
    async def test_streaming_backpressure_handling(self, mock_streaming_client):
        """Test streaming backpressure handling."""

        # Mock streaming with backpressure
        class BackpressureHandler:
            def __init__(self, max_queue_size=10):
                self.queue = asyncio.Queue(maxsize=max_queue_size)
                self.dropped_samples = 0

            async def add_sample(self, sample):
                try:
                    self.queue.put_nowait(sample)
                except asyncio.QueueFull:
                    self.dropped_samples += 1
                    # Drop oldest sample and add new one
                    try:
                        self.queue.get_nowait()
                        self.queue.put_nowait(sample)
                    except asyncio.QueueEmpty:
                        pass

            async def process_samples(self):
                while True:
                    try:
                        sample = await asyncio.wait_for(self.queue.get(), timeout=0.1)
                        # Simulate processing time
                        await asyncio.sleep(0.05)
                        yield {"processed": sample["id"], "timestamp": time.time()}
                    except TimeoutError:
                        break

        handler = BackpressureHandler(max_queue_size=5)

        # Rapid sample addition (faster than processing)
        for i in range(15):
            await handler.add_sample({"id": i, "data": [i]})

        # Process samples
        processed = []
        async for result in handler.process_samples():
            processed.append(result)

        assert len(processed) <= 5  # Queue size limit
        assert handler.dropped_samples > 0  # Some samples were dropped


class TestSDKConcurrentOperations:
    """Test suite for SDK concurrent operations."""

    @pytest.fixture
    def mock_concurrent_client(self):
        """Create mock concurrent client."""
        client = AsyncMock()
        client.max_concurrent = 5
        client._semaphore = asyncio.Semaphore(5)
        return client

    @pytest.mark.asyncio
    async def test_concurrent_batch_processing(self, mock_concurrent_client):
        """Test concurrent batch processing."""

        # Mock batch processing method
        async def mock_process_batch(batch_id, data):
            # Simulate processing time
            await asyncio.sleep(0.1)
            return {
                "batch_id": batch_id,
                "processed_samples": len(data),
                "anomalies_detected": sum(1 for x in data if x > 0.8),
                "processing_time": 0.1,
            }

        mock_concurrent_client.process_batch = mock_process_batch

        # Create multiple batches
        batches = [
            {"batch_id": i, "data": np.random.random(100).tolist()} for i in range(10)
        ]

        # Process batches concurrently
        async def process_with_semaphore(batch):
            async with mock_concurrent_client._semaphore:
                return await mock_concurrent_client.process_batch(
                    batch["batch_id"], batch["data"]
                )

        start_time = time.time()
        tasks = [process_with_semaphore(batch) for batch in batches]
        results = await asyncio.gather(*tasks)
        end_time = time.time()

        assert len(results) == 10
        assert all("batch_id" in r for r in results)
        assert all("processed_samples" in r for r in results)

        # Should be faster than sequential processing
        sequential_time = 10 * 0.1  # 10 batches * 0.1s each
        concurrent_time = end_time - start_time
        assert concurrent_time < sequential_time

    @pytest.mark.asyncio
    async def test_parallel_detector_training(self, mock_concurrent_client):
        """Test parallel detector training."""
        # Mock training configurations
        training_configs = [
            {"algorithm": "isolation_forest", "n_estimators": 100},
            {"algorithm": "local_outlier_factor", "n_neighbors": 20},
            {"algorithm": "one_class_svm", "gamma": "scale"},
            {"algorithm": "elliptic_envelope", "contamination": 0.1},
        ]

        # Mock training method
        async def mock_train_detector(config):
            # Simulate training time based on algorithm
            training_times = {
                "isolation_forest": 0.2,
                "local_outlier_factor": 0.15,
                "one_class_svm": 0.3,
                "elliptic_envelope": 0.1,
            }

            await asyncio.sleep(training_times.get(config["algorithm"], 0.2))

            return {
                "detector_id": f"detector-{config['algorithm']}",
                "algorithm": config["algorithm"],
                "status": "trained",
                "training_time": training_times.get(config["algorithm"], 0.2),
                "performance": {"accuracy": np.random.uniform(0.8, 0.95)},
            }

        mock_concurrent_client.train_detector = mock_train_detector

        # Train detectors in parallel
        start_time = time.time()
        tasks = [
            mock_concurrent_client.train_detector(config) for config in training_configs
        ]
        results = await asyncio.gather(*tasks)
        end_time = time.time()

        assert len(results) == 4
        assert all(r["status"] == "trained" for r in results)

        # Verify parallel execution
        total_sequential_time = sum(0.2, 0.15, 0.3, 0.1)  # Sum of training times
        parallel_time = end_time - start_time
        assert parallel_time < total_sequential_time

    @pytest.mark.asyncio
    async def test_concurrent_detection_with_rate_limiting(
        self, mock_concurrent_client
    ):
        """Test concurrent detection with rate limiting."""

        # Mock rate limiter
        class RateLimiter:
            def __init__(self, rate_per_second=10):
                self.rate_per_second = rate_per_second
                self.tokens = rate_per_second
                self.last_update = time.time()
                self.lock = asyncio.Lock()

            async def acquire(self):
                async with self.lock:
                    now = time.time()
                    elapsed = now - self.last_update
                    self.tokens = min(
                        self.rate_per_second,
                        self.tokens + elapsed * self.rate_per_second,
                    )
                    self.last_update = now

                    if self.tokens >= 1:
                        self.tokens -= 1
                        return True
                    else:
                        # Wait until token is available
                        wait_time = (1 - self.tokens) / self.rate_per_second
                        await asyncio.sleep(wait_time)
                        self.tokens = 0
                        return True

        rate_limiter = RateLimiter(rate_per_second=5)

        # Mock detection method with rate limiting
        async def mock_detect_with_rate_limit(data):
            await rate_limiter.acquire()
            # Simulate detection
            await asyncio.sleep(0.01)
            return {
                "prediction": np.random.choice([0, 1]),
                "score": np.random.random(),
                "timestamp": time.time(),
            }

        mock_concurrent_client.detect = mock_detect_with_rate_limit

        # Perform concurrent detections
        detection_data = [{"sample": i} for i in range(20)]

        start_time = time.time()
        tasks = [mock_concurrent_client.detect(data) for data in detection_data]
        results = await asyncio.gather(*tasks)
        end_time = time.time()

        assert len(results) == 20

        # Verify rate limiting (should take at least 20/5 = 4 seconds)
        execution_time = end_time - start_time
        expected_min_time = 20 / 5  # 20 requests at 5 requests/second
        assert execution_time >= expected_min_time * 0.8  # Allow some tolerance

    def test_thread_safe_client_operations(self, mock_sync_client):
        """Test thread-safe client operations."""
        # Mock thread-safe client
        mock_sync_client._lock = threading.Lock()
        mock_sync_client.request_count = 0

        def mock_thread_safe_detect(data):
            with mock_sync_client._lock:
                mock_sync_client.request_count += 1
                time.sleep(0.01)  # Simulate processing
                return {
                    "request_id": mock_sync_client.request_count,
                    "prediction": 0,
                    "score": 0.1,
                }

        mock_sync_client.detect = mock_thread_safe_detect

        # Concurrent thread execution
        def worker(thread_id):
            results = []
            for i in range(5):
                result = mock_sync_client.detect({"thread": thread_id, "sample": i})
                results.append(result)
            return results

        # Execute with multiple threads
        with ThreadPoolExecutor(max_workers=4) as executor:
            futures = [executor.submit(worker, i) for i in range(4)]
            all_results = []
            for future in as_completed(futures):
                thread_results = future.result()
                all_results.extend(thread_results)

        assert len(all_results) == 20  # 4 threads * 5 requests each
        assert mock_sync_client.request_count == 20

        # Verify unique request IDs (no race conditions)
        request_ids = [r["request_id"] for r in all_results]
        assert len(set(request_ids)) == 20  # All IDs should be unique


class TestSDKCachingAndOptimization:
    """Test suite for SDK caching and optimization features."""

    @pytest.fixture
    def mock_caching_client(self):
        """Create mock client with caching."""
        client = Mock()
        client._cache = {}
        client.cache_enabled = True
        client.cache_ttl = 300  # 5 minutes
        return client

    def test_response_caching(self, mock_caching_client):
        """Test response caching functionality."""

        # Mock caching implementation
        def cache_key(method, url, params=None):
            key_data = f"{method}:{url}:{json.dumps(params or {}, sort_keys=True)}"
            return hashlib.md5(key_data.encode()).hexdigest()

        def cached_request(method, url, params=None):
            key = cache_key(method, url, params)
            current_time = time.time()

            # Check cache
            if key in mock_caching_client._cache:
                cached_item = mock_caching_client._cache[key]
                if (
                    current_time - cached_item["timestamp"]
                    < mock_caching_client.cache_ttl
                ):
                    cached_item["cache_hit"] = True
                    return cached_item["response"]

            # Generate response
            response = {
                "id": "detector-123",
                "status": "trained",
                "timestamp": current_time,
            }

            # Cache response
            mock_caching_client._cache[key] = {
                "response": response,
                "timestamp": current_time,
                "cache_hit": False,
            }

            response["cache_hit"] = False
            return response

        mock_caching_client.cached_request = cached_request

        # First request (cache miss)
        result1 = mock_caching_client.cached_request("GET", "/detectors/123")
        assert result1["cache_hit"] is False

        # Second request (cache hit)
        result2 = mock_caching_client.cached_request("GET", "/detectors/123")
        assert result2["cache_hit"] is True

        # Different request (cache miss)
        result3 = mock_caching_client.cached_request("GET", "/detectors/456")
        assert result3["cache_hit"] is False

    def test_cache_invalidation(self, mock_caching_client):
        """Test cache invalidation."""

        # Mock cache with TTL
        def add_to_cache(key, value, ttl=None):
            mock_caching_client._cache[key] = {
                "value": value,
                "timestamp": time.time(),
                "ttl": ttl or mock_caching_client.cache_ttl,
            }

        def get_from_cache(key):
            if key not in mock_caching_client._cache:
                return None

            item = mock_caching_client._cache[key]
            if time.time() - item["timestamp"] > item["ttl"]:
                del mock_caching_client._cache[key]
                return None

            return item["value"]

        def invalidate_cache_pattern(pattern):
            keys_to_remove = [
                k for k in mock_caching_client._cache.keys() if pattern in k
            ]
            for key in keys_to_remove:
                del mock_caching_client._cache[key]

        mock_caching_client.add_to_cache = add_to_cache
        mock_caching_client.get_from_cache = get_from_cache
        mock_caching_client.invalidate_cache_pattern = invalidate_cache_pattern

        # Add items to cache
        mock_caching_client.add_to_cache("detector-123", {"id": "detector-123"})
        mock_caching_client.add_to_cache("detector-456", {"id": "detector-456"})
        mock_caching_client.add_to_cache("dataset-789", {"id": "dataset-789"})

        assert len(mock_caching_client._cache) == 3

        # Invalidate detector cache
        mock_caching_client.invalidate_cache_pattern("detector")
        assert len(mock_caching_client._cache) == 1
        assert "dataset-789" in mock_caching_client._cache

    def test_request_deduplication(self, mock_caching_client):
        """Test request deduplication."""
        # Mock in-flight request tracking
        mock_caching_client._in_flight = {}

        async def deduplicated_request(key, request_func):
            if key in mock_caching_client._in_flight:
                # Wait for existing request to complete
                return await mock_caching_client._in_flight[key]

            # Create new request
            future = asyncio.Future()
            mock_caching_client._in_flight[key] = future

            try:
                result = await request_func()
                future.set_result(result)
                return result
            except Exception as e:
                future.set_exception(e)
                raise
            finally:
                del mock_caching_client._in_flight[key]

        mock_caching_client.deduplicated_request = deduplicated_request

        # Mock request function
        request_count = 0

        async def mock_request():
            nonlocal request_count
            request_count += 1
            await asyncio.sleep(0.1)  # Simulate network delay
            return {"request_id": request_count, "result": "success"}

        # Test concurrent identical requests
        async def test_deduplication():
            tasks = []
            for _ in range(5):
                task = mock_caching_client.deduplicated_request(
                    "same-key", mock_request
                )
                tasks.append(task)

            results = await asyncio.gather(*tasks)
            return results

        # Run test
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        try:
            results = loop.run_until_complete(test_deduplication())

            # All requests should return the same result
            assert len(results) == 5
            assert all(r["request_id"] == 1 for r in results)  # Only one actual request
            assert request_count == 1  # Verify deduplication worked
        finally:
            loop.close()

    def test_adaptive_timeout(self, mock_caching_client):
        """Test adaptive timeout based on historical performance."""

        # Mock adaptive timeout implementation
        class AdaptiveTimeout:
            def __init__(self, initial_timeout=30):
                self.timeouts = []
                self.initial_timeout = initial_timeout
                self.min_timeout = 5
                self.max_timeout = 120

            def add_response_time(self, response_time):
                self.timeouts.append(response_time)
                # Keep only last 10 measurements
                if len(self.timeouts) > 10:
                    self.timeouts.pop(0)

            def get_adaptive_timeout(self):
                if not self.timeouts:
                    return self.initial_timeout

                # Calculate percentile-based timeout
                avg_time = sum(self.timeouts) / len(self.timeouts)
                p95_time = sorted(self.timeouts)[int(len(self.timeouts) * 0.95)]

                # Set timeout to 3x P95 with bounds
                adaptive_timeout = max(
                    self.min_timeout, min(self.max_timeout, p95_time * 3)
                )
                return adaptive_timeout

        timeout_manager = AdaptiveTimeout()

        # Simulate response times
        response_times = [0.5, 0.8, 1.2, 0.6, 2.1, 0.9, 1.5, 0.7, 3.2, 1.1]
        for rt in response_times:
            timeout_manager.add_response_time(rt)

        adaptive_timeout = timeout_manager.get_adaptive_timeout()

        # Timeout should be reasonable based on historical data
        assert 5 <= adaptive_timeout <= 120
        assert adaptive_timeout > max(
            response_times
        )  # Should handle worst-case scenarios

    def test_connection_pooling_optimization(self, mock_caching_client):
        """Test connection pooling optimization."""

        # Mock connection pool
        class ConnectionPool:
            def __init__(self, max_connections=10, max_keepalive=5):
                self.max_connections = max_connections
                self.max_keepalive = max_keepalive
                self.active_connections = 0
                self.idle_connections = []
                self.connection_stats = {"created": 0, "reused": 0, "closed": 0}

            def get_connection(self):
                if self.idle_connections:
                    connection = self.idle_connections.pop()
                    self.connection_stats["reused"] += 1
                    return connection

                if self.active_connections < self.max_connections:
                    connection = {
                        "id": self.connection_stats["created"] + 1,
                        "created_at": time.time(),
                    }
                    self.active_connections += 1
                    self.connection_stats["created"] += 1
                    return connection

                # No available connections
                return None

            def return_connection(self, connection):
                if len(self.idle_connections) < self.max_keepalive:
                    self.idle_connections.append(connection)
                else:
                    self.active_connections -= 1
                    self.connection_stats["closed"] += 1

        pool = ConnectionPool(max_connections=5, max_keepalive=3)

        # Simulate connection usage
        connections = []

        # Get 5 connections
        for i in range(5):
            conn = pool.get_connection()
            assert conn is not None
            connections.append(conn)

        # Try to get 6th connection (should fail)
        conn = pool.get_connection()
        assert conn is None

        # Return connections
        for conn in connections[:3]:
            pool.return_connection(conn)

        # Get connections again (should reuse idle ones)
        new_connections = []
        for i in range(3):
            conn = pool.get_connection()
            assert conn is not None
            new_connections.append(conn)

        # Verify connection reuse
        assert pool.connection_stats["created"] == 5
        assert pool.connection_stats["reused"] == 3
        assert len(pool.idle_connections) == 0
