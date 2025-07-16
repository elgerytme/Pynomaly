"""
Comprehensive tests for circuit breaker resilience infrastructure.

This module provides extensive testing for circuit breaker patterns,
including state management, failure tracking, recovery mechanisms,
and integration with databases, APIs, and Redis.
"""

import asyncio
import threading
import time

import pytest

from monorepo.infrastructure.resilience.circuit_breaker import (
    APICircuitBreaker,
    CircuitBreaker,
    CircuitBreakerConfig,
    CircuitBreakerManager,
    CircuitBreakerOpenError,
    CircuitBreakerState,
    CircuitBreakerTimeoutError,
    DatabaseCircuitBreaker,
    RedisCircuitBreaker,
)


class TestCircuitBreakerState:
    """Test circuit breaker state enumeration."""

    def test_circuit_breaker_states(self):
        """Test circuit breaker state values."""
        assert CircuitBreakerState.CLOSED.value == "closed"
        assert CircuitBreakerState.OPEN.value == "open"
        assert CircuitBreakerState.HALF_OPEN.value == "half_open"

    def test_state_transitions(self):
        """Test valid state transitions."""
        # Valid transitions
        closed = CircuitBreakerState.CLOSED
        open = CircuitBreakerState.OPEN
        half_open = CircuitBreakerState.HALF_OPEN

        assert closed != open
        assert open != half_open
        assert half_open != closed


class TestCircuitBreakerConfig:
    """Test circuit breaker configuration."""

    def test_config_creation(self):
        """Test circuit breaker configuration creation."""
        config = CircuitBreakerConfig(
            failure_threshold=5,
            recovery_timeout=30,
            timeout=10,
            expected_exception=Exception,
        )

        assert config.failure_threshold == 5
        assert config.recovery_timeout == 30
        assert config.timeout == 10
        assert config.expected_exception == Exception

    def test_config_defaults(self):
        """Test default configuration values."""
        config = CircuitBreakerConfig()

        assert config.failure_threshold == 5
        assert config.recovery_timeout == 60
        assert config.timeout == 30
        assert config.expected_exception == Exception

    def test_config_validation(self):
        """Test configuration validation."""
        # Valid configuration
        config = CircuitBreakerConfig(failure_threshold=3)
        assert config.failure_threshold == 3

        # Invalid threshold
        with pytest.raises(ValueError):
            CircuitBreakerConfig(failure_threshold=0)

        # Invalid timeout
        with pytest.raises(ValueError):
            CircuitBreakerConfig(timeout=-1)


class TestCircuitBreaker:
    """Test basic circuit breaker functionality."""

    @pytest.fixture
    def circuit_breaker(self):
        """Create circuit breaker for testing."""
        config = CircuitBreakerConfig(
            failure_threshold=3,
            recovery_timeout=1,  # Short timeout for testing
            timeout=5,
        )
        return CircuitBreaker("test_breaker", config)

    def test_circuit_breaker_initialization(self, circuit_breaker):
        """Test circuit breaker initialization."""
        assert circuit_breaker.name == "test_breaker"
        assert circuit_breaker.state == CircuitBreakerState.CLOSED
        assert circuit_breaker.failure_count == 0
        assert circuit_breaker.last_failure_time is None

    def test_successful_call_closed_state(self, circuit_breaker):
        """Test successful call in closed state."""

        def successful_function():
            return "success"

        result = circuit_breaker.call(successful_function)

        assert result == "success"
        assert circuit_breaker.state == CircuitBreakerState.CLOSED
        assert circuit_breaker.failure_count == 0

    def test_failed_call_closed_state(self, circuit_breaker):
        """Test failed call in closed state."""

        def failing_function():
            raise Exception("Test failure")

        # First failure
        with pytest.raises(Exception):
            circuit_breaker.call(failing_function)

        assert circuit_breaker.state == CircuitBreakerState.CLOSED
        assert circuit_breaker.failure_count == 1

    def test_circuit_breaker_opens_after_threshold(self, circuit_breaker):
        """Test circuit breaker opens after failure threshold."""

        def failing_function():
            raise Exception("Test failure")

        # Reach failure threshold
        for i in range(circuit_breaker.config.failure_threshold):
            with pytest.raises(Exception):
                circuit_breaker.call(failing_function)

        # Circuit should now be open
        assert circuit_breaker.state == CircuitBreakerState.OPEN
        assert circuit_breaker.failure_count == circuit_breaker.config.failure_threshold

    def test_circuit_breaker_rejects_calls_when_open(self, circuit_breaker):
        """Test circuit breaker rejects calls when open."""

        def failing_function():
            raise Exception("Test failure")

        # Open the circuit
        for _ in range(circuit_breaker.config.failure_threshold):
            with pytest.raises(Exception):
                circuit_breaker.call(failing_function)

        # Further calls should be rejected
        with pytest.raises(CircuitBreakerOpenError):
            circuit_breaker.call(failing_function)

    def test_circuit_breaker_half_open_transition(self, circuit_breaker):
        """Test transition to half-open state."""

        def failing_function():
            raise Exception("Test failure")

        # Open the circuit
        for _ in range(circuit_breaker.config.failure_threshold):
            with pytest.raises(Exception):
                circuit_breaker.call(failing_function)

        assert circuit_breaker.state == CircuitBreakerState.OPEN

        # Wait for recovery timeout
        time.sleep(circuit_breaker.config.recovery_timeout + 0.1)

        # Next call should transition to half-open
        with pytest.raises(Exception):
            circuit_breaker.call(failing_function)

        assert circuit_breaker.state == CircuitBreakerState.HALF_OPEN

    def test_circuit_breaker_closes_after_successful_half_open(self, circuit_breaker):
        """Test circuit breaker closes after successful call in half-open state."""

        def failing_function():
            raise Exception("Test failure")

        def successful_function():
            return "success"

        # Open the circuit
        for _ in range(circuit_breaker.config.failure_threshold):
            with pytest.raises(Exception):
                circuit_breaker.call(failing_function)

        # Wait for recovery timeout
        time.sleep(circuit_breaker.config.recovery_timeout + 0.1)

        # Transition to half-open with failed call
        with pytest.raises(Exception):
            circuit_breaker.call(failing_function)

        assert circuit_breaker.state == CircuitBreakerState.HALF_OPEN

        # Successful call should close the circuit
        result = circuit_breaker.call(successful_function)

        assert result == "success"
        assert circuit_breaker.state == CircuitBreakerState.CLOSED
        assert circuit_breaker.failure_count == 0

    def test_circuit_breaker_reopens_on_half_open_failure(self, circuit_breaker):
        """Test circuit breaker reopens on failure in half-open state."""

        def failing_function():
            raise Exception("Test failure")

        # Open the circuit
        for _ in range(circuit_breaker.config.failure_threshold):
            with pytest.raises(Exception):
                circuit_breaker.call(failing_function)

        # Wait for recovery timeout
        time.sleep(circuit_breaker.config.recovery_timeout + 0.1)

        # Transition to half-open
        with pytest.raises(Exception):
            circuit_breaker.call(failing_function)

        assert circuit_breaker.state == CircuitBreakerState.HALF_OPEN

        # Another failure should reopen the circuit
        with pytest.raises(Exception):
            circuit_breaker.call(failing_function)

        assert circuit_breaker.state == CircuitBreakerState.OPEN

    def test_circuit_breaker_timeout_handling(self, circuit_breaker):
        """Test circuit breaker timeout handling."""

        def slow_function():
            time.sleep(circuit_breaker.config.timeout + 1)
            return "slow"

        with pytest.raises(CircuitBreakerTimeoutError):
            circuit_breaker.call(slow_function)

        assert circuit_breaker.failure_count == 1

    def test_circuit_breaker_with_args_and_kwargs(self, circuit_breaker):
        """Test circuit breaker with function arguments."""

        def function_with_args(a, b, c=None):
            return f"{a}-{b}-{c}"

        result = circuit_breaker.call(function_with_args, 1, 2, c=3)

        assert result == "1-2-3"

    def test_circuit_breaker_reset(self, circuit_breaker):
        """Test circuit breaker reset functionality."""

        def failing_function():
            raise Exception("Test failure")

        # Cause some failures
        for _ in range(2):
            with pytest.raises(Exception):
                circuit_breaker.call(failing_function)

        assert circuit_breaker.failure_count == 2

        # Reset the circuit breaker
        circuit_breaker.reset()

        assert circuit_breaker.state == CircuitBreakerState.CLOSED
        assert circuit_breaker.failure_count == 0
        assert circuit_breaker.last_failure_time is None

    def test_circuit_breaker_get_stats(self, circuit_breaker):
        """Test circuit breaker statistics."""

        def failing_function():
            raise Exception("Test failure")

        def successful_function():
            return "success"

        # Make some calls
        circuit_breaker.call(successful_function)
        try:
            circuit_breaker.call(failing_function)
        except Exception:
            pass

        stats = circuit_breaker.get_stats()

        assert stats["name"] == "test_breaker"
        assert stats["state"] == CircuitBreakerState.CLOSED.value
        assert stats["failure_count"] == 1
        assert stats["total_calls"] == 2
        assert stats["success_calls"] == 1
        assert stats["failure_calls"] == 1

    def test_circuit_breaker_thread_safety(self, circuit_breaker):
        """Test circuit breaker thread safety."""

        def successful_function():
            time.sleep(0.01)  # Small delay
            return "success"

        results = []
        errors = []

        def worker():
            try:
                for _ in range(10):
                    result = circuit_breaker.call(successful_function)
                    results.append(result)
            except Exception as e:
                errors.append(e)

        # Start multiple threads
        threads = []
        for _ in range(5):
            thread = threading.Thread(target=worker)
            threads.append(thread)
            thread.start()

        # Wait for completion
        for thread in threads:
            thread.join()

        # Check results
        assert len(errors) == 0
        assert len(results) == 50
        assert all(result == "success" for result in results)


class TestAsyncCircuitBreaker:
    """Test asynchronous circuit breaker functionality."""

    @pytest.fixture
    def circuit_breaker(self):
        """Create circuit breaker for testing."""
        config = CircuitBreakerConfig(
            failure_threshold=3, recovery_timeout=1, timeout=5
        )
        return CircuitBreaker("async_test_breaker", config)

    @pytest.mark.asyncio
    async def test_async_successful_call(self, circuit_breaker):
        """Test async successful call."""

        async def async_successful_function():
            await asyncio.sleep(0.1)
            return "async_success"

        result = await circuit_breaker.acall(async_successful_function)

        assert result == "async_success"
        assert circuit_breaker.state == CircuitBreakerState.CLOSED

    @pytest.mark.asyncio
    async def test_async_failed_call(self, circuit_breaker):
        """Test async failed call."""

        async def async_failing_function():
            await asyncio.sleep(0.1)
            raise Exception("Async test failure")

        with pytest.raises(Exception):
            await circuit_breaker.acall(async_failing_function)

        assert circuit_breaker.failure_count == 1

    @pytest.mark.asyncio
    async def test_async_circuit_opens(self, circuit_breaker):
        """Test async circuit breaker opens after threshold."""

        async def async_failing_function():
            raise Exception("Async test failure")

        # Reach failure threshold
        for _ in range(circuit_breaker.config.failure_threshold):
            with pytest.raises(Exception):
                await circuit_breaker.acall(async_failing_function)

        assert circuit_breaker.state == CircuitBreakerState.OPEN

    @pytest.mark.asyncio
    async def test_async_timeout_handling(self, circuit_breaker):
        """Test async timeout handling."""

        async def slow_async_function():
            await asyncio.sleep(circuit_breaker.config.timeout + 1)
            return "slow"

        with pytest.raises(CircuitBreakerTimeoutError):
            await circuit_breaker.acall(slow_async_function)

    @pytest.mark.asyncio
    async def test_async_concurrent_calls(self, circuit_breaker):
        """Test concurrent async calls."""

        async def async_function(delay):
            await asyncio.sleep(delay)
            return f"result_{delay}"

        # Make concurrent calls
        tasks = [
            circuit_breaker.acall(async_function, 0.1),
            circuit_breaker.acall(async_function, 0.2),
            circuit_breaker.acall(async_function, 0.3),
        ]

        results = await asyncio.gather(*tasks)

        assert "result_0.1" in results
        assert "result_0.2" in results
        assert "result_0.3" in results


class TestDatabaseCircuitBreaker:
    """Test database-specific circuit breaker."""

    @pytest.fixture
    def db_circuit_breaker(self):
        """Create database circuit breaker."""
        config = CircuitBreakerConfig(
            failure_threshold=2,
            recovery_timeout=1,
            timeout=5,
            expected_exception=(ConnectionError, TimeoutError),
        )
        return DatabaseCircuitBreaker("test_db", config)

    def test_database_circuit_breaker_initialization(self, db_circuit_breaker):
        """Test database circuit breaker initialization."""
        assert db_circuit_breaker.name == "test_db"
        assert db_circuit_breaker.database_name == "test_db"
        assert ConnectionError in db_circuit_breaker.config.expected_exception
        assert TimeoutError in db_circuit_breaker.config.expected_exception

    def test_database_connection_failure(self, db_circuit_breaker):
        """Test database connection failure handling."""

        def failing_db_call():
            raise ConnectionError("Database connection failed")

        # Should handle database-specific exceptions
        with pytest.raises(ConnectionError):
            db_circuit_breaker.call(failing_db_call)

        assert db_circuit_breaker.failure_count == 1

    def test_database_timeout_failure(self, db_circuit_breaker):
        """Test database timeout handling."""

        def timeout_db_call():
            raise TimeoutError("Database query timeout")

        with pytest.raises(TimeoutError):
            db_circuit_breaker.call(timeout_db_call)

        assert db_circuit_breaker.failure_count == 1

    def test_database_successful_query(self, db_circuit_breaker):
        """Test successful database query."""

        def successful_db_call():
            return [{"id": 1, "name": "test"}]

        result = db_circuit_breaker.call(successful_db_call)

        assert result == [{"id": 1, "name": "test"}]
        assert db_circuit_breaker.failure_count == 0

    def test_database_circuit_breaker_with_transaction(self, db_circuit_breaker):
        """Test database circuit breaker with transaction simulation."""

        def transaction_simulation():
            # Simulate database transaction
            return {"transaction_id": "tx_123", "status": "committed"}

        result = db_circuit_breaker.call(transaction_simulation)

        assert result["transaction_id"] == "tx_123"
        assert result["status"] == "committed"

    def test_database_health_check(self, db_circuit_breaker):
        """Test database health check functionality."""

        def health_check():
            return {"status": "healthy", "connection_pool": "available"}

        # Health check should work even when circuit is closed
        result = db_circuit_breaker.health_check(health_check)

        assert result["status"] == "healthy"

    def test_database_stats_include_db_info(self, db_circuit_breaker):
        """Test database stats include database-specific information."""
        stats = db_circuit_breaker.get_stats()

        assert stats["database_name"] == "test_db"
        assert "connection_pool_status" in stats
        assert "last_health_check" in stats


class TestAPICircuitBreaker:
    """Test API-specific circuit breaker."""

    @pytest.fixture
    def api_circuit_breaker(self):
        """Create API circuit breaker."""
        config = CircuitBreakerConfig(
            failure_threshold=3,
            recovery_timeout=2,
            timeout=10,
            expected_exception=(ConnectionError, TimeoutError, ValueError),
        )
        return APICircuitBreaker(
            "external_api", config, base_url="https://api.example.com"
        )

    def test_api_circuit_breaker_initialization(self, api_circuit_breaker):
        """Test API circuit breaker initialization."""
        assert api_circuit_breaker.name == "external_api"
        assert api_circuit_breaker.base_url == "https://api.example.com"
        assert api_circuit_breaker.request_count == 0

    def test_api_http_error_handling(self, api_circuit_breaker):
        """Test API HTTP error handling."""

        def failing_api_call():
            # Simulate HTTP 500 error
            raise ConnectionError("HTTP 500: Internal Server Error")

        with pytest.raises(ConnectionError):
            api_circuit_breaker.call(failing_api_call)

        assert api_circuit_breaker.failure_count == 1
        assert api_circuit_breaker.request_count == 1

    def test_api_timeout_handling(self, api_circuit_breaker):
        """Test API timeout handling."""

        def timeout_api_call():
            raise TimeoutError("Request timeout")

        with pytest.raises(TimeoutError):
            api_circuit_breaker.call(timeout_api_call)

        assert api_circuit_breaker.failure_count == 1

    def test_api_successful_request(self, api_circuit_breaker):
        """Test successful API request."""

        def successful_api_call():
            return {"status": "success", "data": [1, 2, 3]}

        result = api_circuit_breaker.call(successful_api_call)

        assert result["status"] == "success"
        assert api_circuit_breaker.request_count == 1
        assert api_circuit_breaker.failure_count == 0

    def test_api_rate_limiting_detection(self, api_circuit_breaker):
        """Test API rate limiting detection."""

        def rate_limited_call():
            # Simulate rate limiting
            raise ValueError("HTTP 429: Too Many Requests")

        with pytest.raises(ValueError):
            api_circuit_breaker.call(rate_limited_call)

        # Rate limiting should count as failure
        assert api_circuit_breaker.failure_count == 1

    def test_api_stats_include_request_info(self, api_circuit_breaker):
        """Test API stats include request-specific information."""

        def api_call():
            return {"data": "test"}

        api_circuit_breaker.call(api_call)

        stats = api_circuit_breaker.get_stats()

        assert stats["base_url"] == "https://api.example.com"
        assert stats["total_requests"] == 1
        assert stats["success_rate"] >= 0
        assert "average_response_time" in stats

    def test_api_circuit_breaker_with_retries(self, api_circuit_breaker):
        """Test API circuit breaker with retry logic."""
        call_count = 0

        def intermittent_api_call():
            nonlocal call_count
            call_count += 1
            if call_count < 3:
                raise ConnectionError("Temporary failure")
            return {"success": True}

        # First two calls should fail
        with pytest.raises(ConnectionError):
            api_circuit_breaker.call(intermittent_api_call)

        with pytest.raises(ConnectionError):
            api_circuit_breaker.call(intermittent_api_call)

        # Third call should succeed
        result = api_circuit_breaker.call(intermittent_api_call)
        assert result["success"] is True


class TestRedisCircuitBreaker:
    """Test Redis-specific circuit breaker."""

    @pytest.fixture
    def redis_circuit_breaker(self):
        """Create Redis circuit breaker."""
        config = CircuitBreakerConfig(
            failure_threshold=3,
            recovery_timeout=1,
            timeout=5,
            expected_exception=(ConnectionError, TimeoutError),
        )
        return RedisCircuitBreaker(
            "redis_cache", config, redis_url="redis://localhost:6379"
        )

    def test_redis_circuit_breaker_initialization(self, redis_circuit_breaker):
        """Test Redis circuit breaker initialization."""
        assert redis_circuit_breaker.name == "redis_cache"
        assert redis_circuit_breaker.redis_url == "redis://localhost:6379"
        assert redis_circuit_breaker.operation_count == 0

    def test_redis_connection_failure(self, redis_circuit_breaker):
        """Test Redis connection failure handling."""

        def failing_redis_call():
            raise ConnectionError("Redis connection refused")

        with pytest.raises(ConnectionError):
            redis_circuit_breaker.call(failing_redis_call)

        assert redis_circuit_breaker.failure_count == 1
        assert redis_circuit_breaker.operation_count == 1

    def test_redis_successful_operation(self, redis_circuit_breaker):
        """Test successful Redis operation."""

        def successful_redis_call():
            return {"key": "value", "cached": True}

        result = redis_circuit_breaker.call(successful_redis_call)

        assert result["key"] == "value"
        assert result["cached"] is True
        assert redis_circuit_breaker.operation_count == 1

    def test_redis_timeout_handling(self, redis_circuit_breaker):
        """Test Redis timeout handling."""

        def timeout_redis_call():
            raise TimeoutError("Redis operation timeout")

        with pytest.raises(TimeoutError):
            redis_circuit_breaker.call(timeout_redis_call)

        assert redis_circuit_breaker.failure_count == 1

    def test_redis_pipeline_operation(self, redis_circuit_breaker):
        """Test Redis pipeline operation simulation."""

        def pipeline_operation():
            # Simulate Redis pipeline
            return [
                {"operation": "SET", "key": "key1", "result": "OK"},
                {"operation": "GET", "key": "key1", "result": "value1"},
                {"operation": "INCR", "key": "counter", "result": 1},
            ]

        result = redis_circuit_breaker.call(pipeline_operation)

        assert len(result) == 3
        assert result[0]["operation"] == "SET"
        assert result[1]["operation"] == "GET"
        assert result[2]["operation"] == "INCR"

    def test_redis_stats_include_cache_info(self, redis_circuit_breaker):
        """Test Redis stats include cache-specific information."""

        def cache_operation():
            return {"hit": True}

        redis_circuit_breaker.call(cache_operation)

        stats = redis_circuit_breaker.get_stats()

        assert stats["redis_url"] == "redis://localhost:6379"
        assert stats["total_operations"] == 1
        assert "cache_hit_rate" in stats
        assert "average_operation_time" in stats

    def test_redis_health_check(self, redis_circuit_breaker):
        """Test Redis health check functionality."""

        def redis_ping():
            return {"ping": "PONG", "connected": True}

        result = redis_circuit_breaker.health_check(redis_ping)

        assert result["ping"] == "PONG"
        assert result["connected"] is True


class TestCircuitBreakerManager:
    """Test circuit breaker manager functionality."""

    @pytest.fixture
    def manager(self):
        """Create circuit breaker manager."""
        return CircuitBreakerManager()

    def test_manager_initialization(self, manager):
        """Test manager initialization."""
        assert len(manager.circuit_breakers) == 0
        assert manager.default_config is not None

    def test_create_circuit_breaker(self, manager):
        """Test creating circuit breaker through manager."""
        config = CircuitBreakerConfig(failure_threshold=3)
        breaker = manager.create_circuit_breaker("test_service", config)

        assert breaker.name == "test_service"
        assert breaker.config.failure_threshold == 3
        assert "test_service" in manager.circuit_breakers

    def test_get_circuit_breaker(self, manager):
        """Test getting circuit breaker from manager."""
        config = CircuitBreakerConfig(failure_threshold=3)
        original_breaker = manager.create_circuit_breaker("test_service", config)

        retrieved_breaker = manager.get_circuit_breaker("test_service")

        assert retrieved_breaker is original_breaker

    def test_get_nonexistent_circuit_breaker(self, manager):
        """Test getting non-existent circuit breaker."""
        breaker = manager.get_circuit_breaker("nonexistent")

        assert breaker is None

    def test_get_or_create_circuit_breaker(self, manager):
        """Test get or create circuit breaker functionality."""
        # First call should create
        breaker1 = manager.get_or_create_circuit_breaker("auto_service")
        assert breaker1.name == "auto_service"
        assert "auto_service" in manager.circuit_breakers

        # Second call should return existing
        breaker2 = manager.get_or_create_circuit_breaker("auto_service")
        assert breaker2 is breaker1

    def test_remove_circuit_breaker(self, manager):
        """Test removing circuit breaker from manager."""
        config = CircuitBreakerConfig()
        manager.create_circuit_breaker("removable", config)

        assert "removable" in manager.circuit_breakers

        result = manager.remove_circuit_breaker("removable")

        assert result is True
        assert "removable" not in manager.circuit_breakers

    def test_remove_nonexistent_circuit_breaker(self, manager):
        """Test removing non-existent circuit breaker."""
        result = manager.remove_circuit_breaker("nonexistent")

        assert result is False

    def test_list_circuit_breakers(self, manager):
        """Test listing all circuit breakers."""
        # Create multiple breakers
        for i in range(3):
            manager.create_circuit_breaker(f"service_{i}")

        breakers = manager.list_circuit_breakers()

        assert len(breakers) == 3
        assert "service_0" in breakers
        assert "service_1" in breakers
        assert "service_2" in breakers

    def test_get_all_stats(self, manager):
        """Test getting stats for all circuit breakers."""
        # Create and use some breakers
        breaker1 = manager.create_circuit_breaker("service_1")
        breaker2 = manager.create_circuit_breaker("service_2")

        # Make some calls
        breaker1.call(lambda: "success")
        try:
            breaker2.call(lambda: exec('raise Exception("fail")'))
        except Exception:
            pass

        all_stats = manager.get_all_stats()

        assert len(all_stats) == 2
        assert "service_1" in all_stats
        assert "service_2" in all_stats
        assert all_stats["service_1"]["success_calls"] == 1
        assert all_stats["service_2"]["failure_calls"] == 1

    def test_reset_all_circuit_breakers(self, manager):
        """Test resetting all circuit breakers."""
        # Create breakers and cause failures
        breaker1 = manager.create_circuit_breaker("service_1")
        breaker2 = manager.create_circuit_breaker("service_2")

        # Cause failures
        for breaker in [breaker1, breaker2]:
            try:
                breaker.call(lambda: exec('raise Exception("fail")'))
            except Exception:
                pass

        assert breaker1.failure_count > 0
        assert breaker2.failure_count > 0

        # Reset all
        manager.reset_all()

        assert breaker1.failure_count == 0
        assert breaker2.failure_count == 0
        assert breaker1.state == CircuitBreakerState.CLOSED
        assert breaker2.state == CircuitBreakerState.CLOSED

    def test_configure_all_circuit_breakers(self, manager):
        """Test configuring all circuit breakers."""
        # Create breakers with default config
        breaker1 = manager.create_circuit_breaker("service_1")
        breaker2 = manager.create_circuit_breaker("service_2")

        original_threshold = breaker1.config.failure_threshold

        # Update configuration for all
        new_config = CircuitBreakerConfig(failure_threshold=10)
        manager.configure_all(new_config)

        assert breaker1.config.failure_threshold == 10
        assert breaker2.config.failure_threshold == 10
        assert breaker1.config.failure_threshold != original_threshold

    def test_manager_health_check(self, manager):
        """Test manager health check functionality."""
        # Create breakers
        breaker1 = manager.create_circuit_breaker("healthy_service")
        breaker2 = manager.create_circuit_breaker("unhealthy_service")

        # Make unhealthy service fail
        for _ in range(breaker2.config.failure_threshold):
            try:
                breaker2.call(lambda: exec('raise Exception("fail")'))
            except Exception:
                pass

        health_status = manager.get_health_status()

        assert "healthy_service" in health_status
        assert "unhealthy_service" in health_status
        assert health_status["healthy_service"]["state"] == "closed"
        assert health_status["unhealthy_service"]["state"] == "open"
        assert health_status["healthy_service"]["healthy"] is True
        assert health_status["unhealthy_service"]["healthy"] is False

    def test_manager_monitoring_integration(self, manager):
        """Test manager monitoring integration."""
        # Create breakers and generate activity
        for i in range(3):
            breaker = manager.create_circuit_breaker(f"monitored_service_{i}")

            # Generate mixed success/failure
            for j in range(5):
                try:
                    if j % 2 == 0:
                        breaker.call(lambda: "success")
                    else:
                        breaker.call(lambda: exec('raise Exception("fail")'))
                except Exception:
                    pass

        # Get monitoring data
        monitoring_data = manager.get_monitoring_data()

        assert "total_circuit_breakers" in monitoring_data
        assert "healthy_breakers" in monitoring_data
        assert "unhealthy_breakers" in monitoring_data
        assert "total_calls" in monitoring_data
        assert "total_failures" in monitoring_data
        assert monitoring_data["total_circuit_breakers"] == 3


class TestCircuitBreakerDecorator:
    """Test circuit breaker decorator functionality."""

    @pytest.fixture
    def circuit_breaker(self):
        """Create circuit breaker for decorator testing."""
        config = CircuitBreakerConfig(failure_threshold=2, recovery_timeout=1)
        return CircuitBreaker("decorator_test", config)

    def test_decorator_successful_function(self, circuit_breaker):
        """Test decorator with successful function."""

        @circuit_breaker.decorator
        def successful_function(x, y):
            return x + y

        result = successful_function(2, 3)

        assert result == 5
        assert circuit_breaker.failure_count == 0

    def test_decorator_failing_function(self, circuit_breaker):
        """Test decorator with failing function."""

        @circuit_breaker.decorator
        def failing_function():
            raise Exception("Decorator test failure")

        with pytest.raises(Exception):
            failing_function()

        assert circuit_breaker.failure_count == 1

    def test_decorator_circuit_opens(self, circuit_breaker):
        """Test decorator causes circuit to open."""

        @circuit_breaker.decorator
        def failing_function():
            raise Exception("Decorator test failure")

        # Reach failure threshold
        for _ in range(circuit_breaker.config.failure_threshold):
            with pytest.raises(Exception):
                failing_function()

        # Circuit should be open, next call should be rejected
        with pytest.raises(CircuitBreakerOpenError):
            failing_function()

    def test_async_decorator(self, circuit_breaker):
        """Test async decorator functionality."""

        @circuit_breaker.async_decorator
        async def async_function(delay):
            await asyncio.sleep(delay)
            return f"result_{delay}"

        async def test_async():
            result = await async_function(0.1)
            assert result == "result_0.1"
            assert circuit_breaker.failure_count == 0

        asyncio.run(test_async())


class TestCircuitBreakerIntegration:
    """Test circuit breaker integration scenarios."""

    def test_database_and_cache_integration(self):
        """Test integration between database and cache circuit breakers."""
        # Create manager
        manager = CircuitBreakerManager()

        # Create specialized circuit breakers
        db_config = CircuitBreakerConfig(failure_threshold=2, recovery_timeout=1)
        cache_config = CircuitBreakerConfig(failure_threshold=3, recovery_timeout=2)

        db_breaker = manager.create_circuit_breaker("database", db_config)
        cache_breaker = manager.create_circuit_breaker("cache", cache_config)

        def get_data_with_cache(key):
            # Try cache first
            try:
                return cache_breaker.call(lambda: f"cached_{key}")
            except CircuitBreakerOpenError:
                # Cache is down, go to database
                return db_breaker.call(lambda: f"db_{key}")

        # Test normal operation
        result = get_data_with_cache("test")
        assert result == "cached_test"

        # Open cache circuit
        for _ in range(cache_config.failure_threshold):
            try:
                cache_breaker.call(lambda: exec('raise Exception("cache fail")'))
            except Exception:
                pass

        # Should fallback to database
        result = get_data_with_cache("test")
        assert result == "db_test"

    def test_cascading_failures(self):
        """Test cascading failure scenarios."""
        manager = CircuitBreakerManager()

        # Create chain of dependencies
        service_a = manager.create_circuit_breaker(
            "service_a", CircuitBreakerConfig(failure_threshold=2)
        )
        service_b = manager.create_circuit_breaker(
            "service_b", CircuitBreakerConfig(failure_threshold=2)
        )
        service_c = manager.create_circuit_breaker(
            "service_c", CircuitBreakerConfig(failure_threshold=2)
        )

        def call_service_chain():
            # Service A calls Service B calls Service C
            result_c = service_c.call(lambda: "service_c_result")
            result_b = service_b.call(lambda: f"service_b_calls_{result_c}")
            result_a = service_a.call(lambda: f"service_a_calls_{result_b}")
            return result_a

        # Normal operation
        result = call_service_chain()
        assert "service_c_result" in result

        # Break service C
        for _ in range(service_c.config.failure_threshold):
            try:
                service_c.call(lambda: exec('raise Exception("service_c fail")'))
            except Exception:
                pass

        # Chain should break when service C is down
        with pytest.raises(CircuitBreakerOpenError):
            call_service_chain()

    def test_circuit_breaker_with_retry_mechanism(self):
        """Test circuit breaker with retry mechanism."""
        config = CircuitBreakerConfig(failure_threshold=3, recovery_timeout=1)
        breaker = CircuitBreaker("retry_test", config)

        call_count = 0

        def flaky_service():
            nonlocal call_count
            call_count += 1
            if call_count <= 2:
                raise Exception("Temporary failure")
            return "success"

        # Implement retry logic with circuit breaker
        def call_with_retry(max_retries=3):
            for attempt in range(max_retries):
                try:
                    return breaker.call(flaky_service)
                except CircuitBreakerOpenError:
                    raise  # Don't retry if circuit is open
                except Exception:
                    if attempt == max_retries - 1:
                        raise
                    time.sleep(0.1)

        # Should succeed after retries
        result = call_with_retry()
        assert result == "success"

    def test_performance_under_load(self):
        """Test circuit breaker performance under load."""
        config = CircuitBreakerConfig(failure_threshold=10, recovery_timeout=1)
        breaker = CircuitBreaker("load_test", config)

        def fast_operation():
            return "fast_result"

        # Measure performance
        start_time = time.time()

        # Make many calls
        results = []
        for _ in range(1000):
            result = breaker.call(fast_operation)
            results.append(result)

        end_time = time.time()
        duration = end_time - start_time

        # Should complete quickly
        assert duration < 1.0  # Less than 1 second for 1000 calls
        assert len(results) == 1000
        assert all(result == "fast_result" for result in results)

    def test_memory_usage_stability(self):
        """Test circuit breaker memory usage stability."""
        config = CircuitBreakerConfig(failure_threshold=5)
        breaker = CircuitBreaker("memory_test", config)

        # Make many calls to check for memory leaks
        for i in range(10000):
            try:
                if i % 10 == 0:
                    # Occasional failure
                    breaker.call(lambda: exec('raise Exception("fail")'))
                else:
                    # Mostly success
                    breaker.call(lambda: "success")
            except Exception:
                pass

        # Check that stats are reasonable (not accumulating indefinitely)
        stats = breaker.get_stats()
        assert stats["total_calls"] == 10000
        assert stats["failure_calls"] == 1000
        assert stats["success_calls"] == 9000

        # Memory should be stable (this is more of a smoke test)
        import sys

        memory_usage = sys.getsizeof(breaker)
        assert memory_usage < 10000  # Should not grow excessively
