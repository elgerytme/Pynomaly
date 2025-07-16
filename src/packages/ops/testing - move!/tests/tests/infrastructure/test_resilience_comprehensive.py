"""Comprehensive tests for infrastructure resilience patterns - Phase 2 Coverage."""

from __future__ import annotations

import asyncio
import time
from unittest.mock import patch

import pytest

from monorepo.domain.exceptions import (
    BulkheadRejectionError,
    CircuitBreakerOpenError,
    RateLimitExceededError,
    RetryExhaustedError,
    TimeoutError,
)
from monorepo.infrastructure.resilience import (
    BulkheadIsolation,
    CircuitBreaker,
    CircuitBreakerState,
    RateLimiter,
    ResilientService,
    RetryPolicy,
    TimeoutManager,
)


@pytest.fixture
def sample_circuit_breaker():
    """Create a sample circuit breaker for testing."""
    return CircuitBreaker(
        name="test_breaker",
        failure_threshold=3,
        recovery_timeout=5.0,
        success_threshold=2,
    )


@pytest.fixture
def sample_retry_policy():
    """Create a sample retry policy for testing."""
    return RetryPolicy(
        max_attempts=3,
        base_delay=0.1,
        max_delay=1.0,
        backoff_multiplier=2.0,
        jitter=True,
    )


@pytest.fixture
def sample_timeout_manager():
    """Create a sample timeout manager for testing."""
    return TimeoutManager(
        default_timeout=1.0,
        operation_timeouts={"database": 2.0, "api_call": 5.0, "file_operation": 10.0},
    )


class TestCircuitBreaker:
    """Comprehensive tests for CircuitBreaker functionality."""

    def test_circuit_breaker_initialization(self):
        """Test circuit breaker initialization."""
        cb = CircuitBreaker(
            name="test", failure_threshold=5, recovery_timeout=10.0, success_threshold=3
        )

        assert cb.name == "test"
        assert cb.failure_threshold == 5
        assert cb.recovery_timeout == 10.0
        assert cb.success_threshold == 3
        assert cb.state == CircuitBreakerState.CLOSED
        assert cb.failure_count == 0
        assert cb.success_count == 0

    def test_circuit_breaker_success_flow(self, sample_circuit_breaker):
        """Test circuit breaker with successful operations."""

        def successful_operation():
            return "success"

        # Execute successful operations
        for _i in range(5):
            result = sample_circuit_breaker.call(successful_operation)
            assert result == "success"
            assert sample_circuit_breaker.state == CircuitBreakerState.CLOSED
            assert sample_circuit_breaker.failure_count == 0

    def test_circuit_breaker_failure_threshold(self, sample_circuit_breaker):
        """Test circuit breaker failure threshold behavior."""

        def failing_operation():
            raise ValueError("Operation failed")

        # Execute operations that fail
        for _i in range(sample_circuit_breaker.failure_threshold - 1):
            with pytest.raises(ValueError):
                sample_circuit_breaker.call(failing_operation)
            assert sample_circuit_breaker.state == CircuitBreakerState.CLOSED

        # One more failure should open the circuit
        with pytest.raises(ValueError):
            sample_circuit_breaker.call(failing_operation)

        assert sample_circuit_breaker.state == CircuitBreakerState.OPEN
        assert (
            sample_circuit_breaker.failure_count
            == sample_circuit_breaker.failure_threshold
        )

    def test_circuit_breaker_open_state_rejection(self, sample_circuit_breaker):
        """Test circuit breaker rejection in open state."""

        def failing_operation():
            raise ValueError("Operation failed")

        # Force circuit breaker to open state
        for _i in range(sample_circuit_breaker.failure_threshold):
            with pytest.raises(ValueError):
                sample_circuit_breaker.call(failing_operation)

        assert sample_circuit_breaker.state == CircuitBreakerState.OPEN

        # Further calls should be rejected immediately
        with pytest.raises(CircuitBreakerOpenError):
            sample_circuit_breaker.call(lambda: "should not execute")

    def test_circuit_breaker_half_open_transition(self, sample_circuit_breaker):
        """Test circuit breaker half-open state transition."""

        def failing_operation():
            raise ValueError("Operation failed")

        # Force to open state
        for _i in range(sample_circuit_breaker.failure_threshold):
            with pytest.raises(ValueError):
                sample_circuit_breaker.call(failing_operation)

        assert sample_circuit_breaker.state == CircuitBreakerState.OPEN

        # Mock time passage for recovery timeout
        with patch("time.time") as mock_time:
            mock_time.return_value = (
                time.time() + sample_circuit_breaker.recovery_timeout + 1
            )

            # Force state check
            sample_circuit_breaker._check_state_transition()
            assert sample_circuit_breaker.state == CircuitBreakerState.HALF_OPEN

    def test_circuit_breaker_recovery_flow(self, sample_circuit_breaker):
        """Test circuit breaker recovery flow."""

        def failing_operation():
            raise ValueError("Operation failed")

        def successful_operation():
            return "success"

        # Force to open state
        for _i in range(sample_circuit_breaker.failure_threshold):
            with pytest.raises(ValueError):
                sample_circuit_breaker.call(failing_operation)

        # Transition to half-open
        sample_circuit_breaker.state = CircuitBreakerState.HALF_OPEN
        sample_circuit_breaker.failure_count = 0
        sample_circuit_breaker.success_count = 0

        # Execute successful operations to close circuit
        for _i in range(sample_circuit_breaker.success_threshold):
            result = sample_circuit_breaker.call(successful_operation)
            assert result == "success"

        assert sample_circuit_breaker.state == CircuitBreakerState.CLOSED
        assert sample_circuit_breaker.success_count == 0  # Reset after closing

    @pytest.mark.asyncio
    async def test_circuit_breaker_async_operations(self, sample_circuit_breaker):
        """Test circuit breaker with async operations."""

        async def async_successful_operation():
            await asyncio.sleep(0.01)
            return "async_success"

        async def async_failing_operation():
            await asyncio.sleep(0.01)
            raise ValueError("Async operation failed")

        # Test successful async operation
        result = await sample_circuit_breaker.call_async(async_successful_operation)
        assert result == "async_success"

        # Test failing async operations
        for _i in range(sample_circuit_breaker.failure_threshold):
            with pytest.raises(ValueError):
                await sample_circuit_breaker.call_async(async_failing_operation)

        assert sample_circuit_breaker.state == CircuitBreakerState.OPEN

    def test_circuit_breaker_statistics(self, sample_circuit_breaker):
        """Test circuit breaker statistics collection."""

        def successful_operation():
            return "success"

        def failing_operation():
            raise ValueError("Operation failed")

        # Execute mixed operations
        sample_circuit_breaker.call(successful_operation)
        sample_circuit_breaker.call(successful_operation)

        try:
            sample_circuit_breaker.call(failing_operation)
        except ValueError:
            pass

        # Get statistics
        stats = sample_circuit_breaker.get_statistics()

        assert stats["total_calls"] == 3
        assert stats["successful_calls"] == 2
        assert stats["failed_calls"] == 1
        assert stats["success_rate"] == 2 / 3
        assert stats["current_state"] == CircuitBreakerState.CLOSED.value
        assert "last_failure_time" in stats
        assert "last_success_time" in stats

    def test_circuit_breaker_configuration_validation(self):
        """Test circuit breaker configuration validation."""
        # Test valid configuration
        cb = CircuitBreaker("test", failure_threshold=3, recovery_timeout=5.0)
        assert cb.failure_threshold == 3

        # Test invalid configurations
        with pytest.raises(ValueError, match="Failure threshold must be positive"):
            CircuitBreaker("test", failure_threshold=0)

        with pytest.raises(ValueError, match="Recovery timeout must be positive"):
            CircuitBreaker("test", recovery_timeout=-1.0)

        with pytest.raises(ValueError, match="Success threshold must be positive"):
            CircuitBreaker("test", success_threshold=0)


class TestRetryPolicy:
    """Comprehensive tests for RetryPolicy functionality."""

    def test_retry_policy_initialization(self):
        """Test retry policy initialization."""
        policy = RetryPolicy(
            max_attempts=5,
            base_delay=0.5,
            max_delay=10.0,
            backoff_multiplier=2.0,
            jitter=True,
        )

        assert policy.max_attempts == 5
        assert policy.base_delay == 0.5
        assert policy.max_delay == 10.0
        assert policy.backoff_multiplier == 2.0
        assert policy.jitter is True

    def test_retry_policy_successful_operation(self, sample_retry_policy):
        """Test retry policy with successful operation."""
        call_count = [0]

        def successful_operation():
            call_count[0] += 1
            return f"success_{call_count[0]}"

        result = sample_retry_policy.execute(successful_operation)
        assert result == "success_1"
        assert call_count[0] == 1  # Should not retry on success

    def test_retry_policy_eventual_success(self, sample_retry_policy):
        """Test retry policy with eventual success."""
        call_count = [0]

        def eventually_successful_operation():
            call_count[0] += 1
            if call_count[0] < 3:
                raise ValueError(f"Attempt {call_count[0]} failed")
            return f"success_after_{call_count[0]}_attempts"

        result = sample_retry_policy.execute(eventually_successful_operation)
        assert result == "success_after_3_attempts"
        assert call_count[0] == 3

    def test_retry_policy_max_attempts_exceeded(self, sample_retry_policy):
        """Test retry policy when max attempts are exceeded."""
        call_count = [0]

        def always_failing_operation():
            call_count[0] += 1
            raise ValueError(f"Attempt {call_count[0]} failed")

        with pytest.raises(RetryExhaustedError) as exc_info:
            sample_retry_policy.execute(always_failing_operation)

        assert call_count[0] == sample_retry_policy.max_attempts
        assert "Max retry attempts" in str(exc_info.value)
        assert isinstance(exc_info.value.last_exception, ValueError)

    def test_retry_policy_exponential_backoff(self):
        """Test exponential backoff calculation."""
        policy = RetryPolicy(
            max_attempts=4,
            base_delay=0.1,
            backoff_multiplier=2.0,
            jitter=False,  # Disable jitter for predictable testing
        )

        delays = []

        def mock_sleep(duration):
            delays.append(duration)
            # Don't actually sleep in tests

        with patch("time.sleep", mock_sleep):
            call_count = [0]

            def failing_operation():
                call_count[0] += 1
                raise ValueError("Always fails")

            try:
                policy.execute(failing_operation)
            except RetryExhaustedError:
                pass

        # Check exponential backoff: 0.1, 0.2, 0.4
        expected_delays = [0.1, 0.2, 0.4]
        assert len(delays) == 3  # 3 retries for 4 attempts
        for i, expected in enumerate(expected_delays):
            assert abs(delays[i] - expected) < 0.01

    def test_retry_policy_max_delay_cap(self):
        """Test retry policy max delay cap."""
        policy = RetryPolicy(
            max_attempts=5,
            base_delay=1.0,
            max_delay=2.0,
            backoff_multiplier=3.0,
            jitter=False,
        )

        delays = []

        def mock_sleep(duration):
            delays.append(duration)

        with patch("time.sleep", mock_sleep):
            call_count = [0]

            def failing_operation():
                call_count[0] += 1
                raise ValueError("Always fails")

            try:
                policy.execute(failing_operation)
            except RetryExhaustedError:
                pass

        # Delays should be capped at max_delay
        # Expected: 1.0, 2.0 (capped), 2.0 (capped), 2.0 (capped)
        assert all(delay <= policy.max_delay for delay in delays)
        assert delays[1] == policy.max_delay  # Second delay should be capped

    def test_retry_policy_jitter(self):
        """Test retry policy jitter functionality."""
        policy = RetryPolicy(max_attempts=3, base_delay=1.0, jitter=True)

        delays = []

        def mock_sleep(duration):
            delays.append(duration)

        with patch("time.sleep", mock_sleep):
            call_count = [0]

            def failing_operation():
                call_count[0] += 1
                raise ValueError("Always fails")

            try:
                policy.execute(failing_operation)
            except RetryExhaustedError:
                pass

        # With jitter, delays should vary from base delay
        if len(delays) > 0:
            # Jitter should make delays different from exact base delay
            assert any(abs(delay - policy.base_delay) > 0.01 for delay in delays)

    @pytest.mark.asyncio
    async def test_retry_policy_async_operations(self, sample_retry_policy):
        """Test retry policy with async operations."""
        call_count = [0]

        async def async_eventually_successful():
            call_count[0] += 1
            await asyncio.sleep(0.01)
            if call_count[0] < 2:
                raise ValueError(f"Async attempt {call_count[0]} failed")
            return f"async_success_{call_count[0]}"

        result = await sample_retry_policy.execute_async(async_eventually_successful)
        assert result == "async_success_2"
        assert call_count[0] == 2

    def test_retry_policy_conditional_retry(self):
        """Test conditional retry based on exception type."""
        policy = RetryPolicy(
            max_attempts=3,
            retryable_exceptions=[ValueError, ConnectionError],
            non_retryable_exceptions=[TypeError],
        )

        call_count = [0]

        # Test retryable exception
        def retriable_failure():
            call_count[0] += 1
            raise ValueError("Retryable error")

        try:
            policy.execute(retriable_failure)
        except RetryExhaustedError:
            pass

        assert call_count[0] == 3  # Should retry

        # Test non-retryable exception
        call_count[0] = 0

        def non_retriable_failure():
            call_count[0] += 1
            raise TypeError("Non-retryable error")

        with pytest.raises(TypeError):
            policy.execute(non_retriable_failure)

        assert call_count[0] == 1  # Should not retry


class TestTimeoutManager:
    """Comprehensive tests for TimeoutManager functionality."""

    def test_timeout_manager_initialization(self):
        """Test timeout manager initialization."""
        tm = TimeoutManager(
            default_timeout=5.0, operation_timeouts={"database": 10.0, "api": 30.0}
        )

        assert tm.default_timeout == 5.0
        assert tm.operation_timeouts["database"] == 10.0
        assert tm.operation_timeouts["api"] == 30.0

    def test_timeout_manager_successful_operation(self, sample_timeout_manager):
        """Test timeout manager with successful operation."""

        def quick_operation():
            time.sleep(0.1)  # Quick operation
            return "completed"

        result = sample_timeout_manager.execute(quick_operation, timeout=1.0)
        assert result == "completed"

    def test_timeout_manager_timeout_exceeded(self, sample_timeout_manager):
        """Test timeout manager when timeout is exceeded."""

        def slow_operation():
            time.sleep(2.0)  # Slow operation
            return "should not complete"

        with pytest.raises(TimeoutError):
            sample_timeout_manager.execute(slow_operation, timeout=0.5)

    def test_timeout_manager_operation_specific_timeout(self, sample_timeout_manager):
        """Test operation-specific timeout configuration."""

        def database_operation():
            time.sleep(1.5)  # Takes 1.5 seconds
            return "database_result"

        # Should succeed with database-specific timeout (2.0s)
        result = sample_timeout_manager.execute_with_operation_timeout(
            database_operation, operation_type="database"
        )
        assert result == "database_result"

        # Should timeout with default timeout (1.0s)
        with pytest.raises(TimeoutError):
            sample_timeout_manager.execute(database_operation, timeout=1.0)

    @pytest.mark.asyncio
    async def test_timeout_manager_async_operations(self, sample_timeout_manager):
        """Test timeout manager with async operations."""

        async def async_quick_operation():
            await asyncio.sleep(0.1)
            return "async_completed"

        async def async_slow_operation():
            await asyncio.sleep(2.0)
            return "should_not_complete"

        # Test successful async operation
        result = await sample_timeout_manager.execute_async(
            async_quick_operation(), timeout=1.0
        )
        assert result == "async_completed"

        # Test async timeout
        with pytest.raises(TimeoutError):
            await sample_timeout_manager.execute_async(
                async_slow_operation(), timeout=0.5
            )

    def test_timeout_manager_context_manager(self, sample_timeout_manager):
        """Test timeout manager as context manager."""
        # Test successful operation
        with sample_timeout_manager.timeout_context(1.0):
            time.sleep(0.1)
            result = "completed"

        assert result == "completed"

        # Test timeout in context
        with pytest.raises(TimeoutError):
            with sample_timeout_manager.timeout_context(0.5):
                time.sleep(1.0)

    def test_timeout_manager_nested_timeouts(self, sample_timeout_manager):
        """Test nested timeout contexts."""

        def nested_operation():
            with sample_timeout_manager.timeout_context(0.5):  # Inner timeout
                time.sleep(0.3)
                return "inner_completed"

        # Outer timeout should not interfere if inner completes
        with sample_timeout_manager.timeout_context(1.0):  # Outer timeout
            result = nested_operation()

        assert result == "inner_completed"

        # Test inner timeout triggering first
        def slow_nested_operation():
            with sample_timeout_manager.timeout_context(0.2):  # Short inner timeout
                time.sleep(0.5)
                return "should_not_complete"

        with pytest.raises(TimeoutError):
            with sample_timeout_manager.timeout_context(1.0):  # Longer outer timeout
                slow_nested_operation()

    def test_timeout_manager_statistics(self, sample_timeout_manager):
        """Test timeout manager statistics collection."""
        # Execute some operations
        sample_timeout_manager.execute(lambda: time.sleep(0.1), timeout=1.0)
        sample_timeout_manager.execute(lambda: "quick", timeout=1.0)

        try:
            sample_timeout_manager.execute(lambda: time.sleep(2.0), timeout=0.5)
        except TimeoutError:
            pass

        stats = sample_timeout_manager.get_statistics()

        assert stats["total_operations"] == 3
        assert stats["successful_operations"] == 2
        assert stats["timed_out_operations"] == 1
        assert stats["average_execution_time"] > 0
        assert "timeout_rate" in stats


class TestResilientService:
    """Comprehensive tests for ResilientService integration."""

    def test_resilient_service_initialization(self):
        """Test resilient service initialization."""
        service = ResilientService(
            name="test_service",
            circuit_breaker_config={"failure_threshold": 5, "recovery_timeout": 10.0},
            retry_policy_config={"max_attempts": 3, "base_delay": 0.1},
            timeout_config={"default_timeout": 5.0},
        )

        assert service.name == "test_service"
        assert service.circuit_breaker is not None
        assert service.retry_policy is not None
        assert service.timeout_manager is not None

    def test_resilient_service_combined_patterns(self):
        """Test resilient service with combined resilience patterns."""
        service = ResilientService(
            name="combined_test",
            circuit_breaker_config={"failure_threshold": 2},
            retry_policy_config={"max_attempts": 2},
            timeout_config={"default_timeout": 1.0},
        )

        call_count = [0]

        def flaky_operation():
            call_count[0] += 1
            if call_count[0] == 1:
                raise ValueError("First attempt fails")
            time.sleep(0.1)  # Quick operation
            return f"success_on_attempt_{call_count[0]}"

        # Should succeed on retry
        result = service.execute(flaky_operation)
        assert result == "success_on_attempt_2"
        assert call_count[0] == 2

    def test_resilient_service_circuit_breaker_integration(self):
        """Test resilient service circuit breaker integration."""
        service = ResilientService(
            name="cb_test",
            circuit_breaker_config={"failure_threshold": 2},
            retry_policy_config={"max_attempts": 1},  # No retries for clear testing
        )

        def always_failing():
            raise ValueError("Always fails")

        # Execute failing operations to open circuit
        for _i in range(2):
            with pytest.raises(ValueError):
                service.execute(always_failing)

        # Circuit should be open, next call should be rejected
        with pytest.raises(CircuitBreakerOpenError):
            service.execute(always_failing)

    @pytest.mark.asyncio
    async def test_resilient_service_async_operations(self):
        """Test resilient service with async operations."""
        service = ResilientService(
            name="async_test",
            circuit_breaker_config={"failure_threshold": 3},
            retry_policy_config={"max_attempts": 2},
            timeout_config={"default_timeout": 1.0},
        )

        call_count = [0]

        async def async_flaky_operation():
            call_count[0] += 1
            if call_count[0] == 1:
                raise ValueError("First async attempt fails")
            await asyncio.sleep(0.1)
            return f"async_success_{call_count[0]}"

        result = await service.execute_async(async_flaky_operation)
        assert result == "async_success_2"
        assert call_count[0] == 2

    def test_resilient_service_statistics(self):
        """Test resilient service statistics aggregation."""
        service = ResilientService(
            name="stats_test",
            circuit_breaker_config={"failure_threshold": 5},
            retry_policy_config={"max_attempts": 2},
        )

        # Execute mixed operations
        service.execute(lambda: "success_1")
        service.execute(lambda: "success_2")

        try:
            service.execute(lambda: 1 / 0)  # Causes ZeroDivisionError
        except ZeroDivisionError:
            pass

        stats = service.get_comprehensive_statistics()

        assert "circuit_breaker" in stats
        assert "retry_policy" in stats
        assert "timeout_manager" in stats
        assert "overall" in stats

        overall_stats = stats["overall"]
        assert overall_stats["total_operations"] == 3
        assert overall_stats["successful_operations"] == 2
        assert overall_stats["failed_operations"] == 1


class TestBulkheadIsolation:
    """Comprehensive tests for BulkheadIsolation functionality."""

    def test_bulkhead_initialization(self):
        """Test bulkhead isolation initialization."""
        bulkhead = BulkheadIsolation(
            name="test_bulkhead", max_concurrent_calls=3, max_queue_size=5
        )

        assert bulkhead.name == "test_bulkhead"
        assert bulkhead.max_concurrent_calls == 3
        assert bulkhead.max_queue_size == 5
        assert bulkhead.current_calls == 0
        assert bulkhead.queued_calls == 0

    def test_bulkhead_concurrent_execution(self):
        """Test bulkhead concurrent execution limits."""
        bulkhead = BulkheadIsolation(
            name="concurrent_test", max_concurrent_calls=2, max_queue_size=1
        )

        execution_order = []

        def slow_operation(op_id):
            execution_order.append(f"start_{op_id}")
            time.sleep(0.2)
            execution_order.append(f"end_{op_id}")
            return f"result_{op_id}"

        # Execute operations concurrently
        import threading

        results = {}
        threads = []

        for i in range(2):  # Within limit
            thread = threading.Thread(
                target=lambda i=i: results.update(
                    {i: bulkhead.execute(lambda: slow_operation(i))}
                )
            )
            threads.append(thread)
            thread.start()

        # Wait for completion
        for thread in threads:
            thread.join()

        assert len(results) == 2
        assert all(f"result_{i}" in results.values() for i in range(2))

    def test_bulkhead_rejection_when_full(self):
        """Test bulkhead rejection when capacity is exceeded."""
        bulkhead = BulkheadIsolation(
            name="rejection_test", max_concurrent_calls=1, max_queue_size=1
        )

        def blocking_operation():
            time.sleep(1.0)
            return "completed"

        # Start first operation (will block)
        import threading

        thread1 = threading.Thread(target=lambda: bulkhead.execute(blocking_operation))
        thread1.start()

        time.sleep(0.1)  # Ensure first operation is running

        # Second operation should be queued
        thread2 = threading.Thread(target=lambda: bulkhead.execute(blocking_operation))
        thread2.start()

        time.sleep(0.1)  # Ensure second operation is queued

        # Third operation should be rejected
        with pytest.raises(BulkheadRejectionError):
            bulkhead.execute(lambda: "should_be_rejected")

        # Cleanup
        thread1.join()
        thread2.join()

    @pytest.mark.asyncio
    async def test_bulkhead_async_operations(self):
        """Test bulkhead with async operations."""
        bulkhead = BulkheadIsolation(
            name="async_bulkhead", max_concurrent_calls=2, max_queue_size=2
        )

        async def async_operation(op_id):
            await asyncio.sleep(0.1)
            return f"async_result_{op_id}"

        # Execute multiple async operations
        tasks = [bulkhead.execute_async(async_operation(i)) for i in range(3)]

        results = await asyncio.gather(*tasks)
        assert len(results) == 3
        assert all("async_result_" in result for result in results)

    def test_bulkhead_statistics(self):
        """Test bulkhead statistics collection."""
        bulkhead = BulkheadIsolation(
            name="stats_bulkhead", max_concurrent_calls=2, max_queue_size=2
        )

        # Execute some operations
        bulkhead.execute(lambda: time.sleep(0.1) or "result1")
        bulkhead.execute(lambda: "result2")

        try:
            # This should be rejected if bulkhead is configured strictly
            bulkhead.execute(lambda: "result3")
        except BulkheadRejectionError:
            pass

        stats = bulkhead.get_statistics()

        assert "total_requests" in stats
        assert "successful_executions" in stats
        assert "rejected_requests" in stats
        assert "current_concurrent_calls" in stats
        assert "max_concurrent_calls_reached" in stats


class TestRateLimiter:
    """Comprehensive tests for RateLimiter functionality."""

    def test_rate_limiter_initialization(self):
        """Test rate limiter initialization."""
        limiter = RateLimiter(
            name="test_limiter",
            max_requests=10,
            time_window=60.0,
            algorithm="token_bucket",
        )

        assert limiter.name == "test_limiter"
        assert limiter.max_requests == 10
        assert limiter.time_window == 60.0
        assert limiter.algorithm == "token_bucket"

    def test_rate_limiter_token_bucket(self):
        """Test token bucket rate limiting algorithm."""
        limiter = RateLimiter(
            name="token_test", max_requests=3, time_window=1.0, algorithm="token_bucket"
        )

        # Should allow initial requests up to limit
        for _i in range(3):
            assert limiter.is_allowed() is True

        # Should deny further requests
        assert limiter.is_allowed() is False

        # Should allow requests after time window
        time.sleep(1.1)
        assert limiter.is_allowed() is True

    def test_rate_limiter_sliding_window(self):
        """Test sliding window rate limiting algorithm."""
        limiter = RateLimiter(
            name="sliding_test",
            max_requests=2,
            time_window=1.0,
            algorithm="sliding_window",
        )

        # Make requests
        assert limiter.is_allowed() is True
        assert limiter.is_allowed() is True
        assert limiter.is_allowed() is False  # Exceeds limit

        # Wait half the window and try again
        time.sleep(0.6)
        assert limiter.is_allowed() is False  # Still within window

        # Wait for full window reset
        time.sleep(0.6)
        assert limiter.is_allowed() is True  # Window reset

    def test_rate_limiter_per_client(self):
        """Test per-client rate limiting."""
        limiter = RateLimiter(
            name="client_test", max_requests=2, time_window=1.0, per_client=True
        )

        # Client A should be allowed
        assert limiter.is_allowed(client_id="client_a") is True
        assert limiter.is_allowed(client_id="client_a") is True
        assert limiter.is_allowed(client_id="client_a") is False

        # Client B should have separate limit
        assert limiter.is_allowed(client_id="client_b") is True
        assert limiter.is_allowed(client_id="client_b") is True
        assert limiter.is_allowed(client_id="client_b") is False

    def test_rate_limiter_execution_wrapper(self):
        """Test rate limiter as execution wrapper."""
        limiter = RateLimiter(name="wrapper_test", max_requests=2, time_window=1.0)

        def test_operation():
            return "operation_result"

        # Should allow initial executions
        result1 = limiter.execute(test_operation)
        assert result1 == "operation_result"

        result2 = limiter.execute(test_operation)
        assert result2 == "operation_result"

        # Should deny further executions
        with pytest.raises(RateLimitExceededError):
            limiter.execute(test_operation)

    @pytest.mark.asyncio
    async def test_rate_limiter_async_operations(self):
        """Test rate limiter with async operations."""
        limiter = RateLimiter(name="async_limiter", max_requests=2, time_window=1.0)

        async def async_operation():
            await asyncio.sleep(0.01)
            return "async_result"

        # Should allow async executions within limit
        result1 = await limiter.execute_async(async_operation)
        assert result1 == "async_result"

        result2 = await limiter.execute_async(async_operation)
        assert result2 == "async_result"

        # Should deny when limit exceeded
        with pytest.raises(RateLimitExceededError):
            await limiter.execute_async(async_operation)

    def test_rate_limiter_statistics(self):
        """Test rate limiter statistics collection."""
        limiter = RateLimiter(name="stats_limiter", max_requests=2, time_window=1.0)

        # Make requests
        limiter.is_allowed()
        limiter.is_allowed()
        limiter.is_allowed()  # This should be denied

        stats = limiter.get_statistics()

        assert stats["total_requests"] == 3
        assert stats["allowed_requests"] == 2
        assert stats["denied_requests"] == 1
        assert stats["current_rate"] >= 0
        assert "reset_time" in stats


class TestResilienceIntegration:
    """Test integration of multiple resilience patterns."""

    def test_full_resilience_stack(self):
        """Test full resilience stack integration."""
        # Create a service with all resilience patterns
        service = ResilientService(
            name="full_stack",
            circuit_breaker_config={"failure_threshold": 3, "recovery_timeout": 1.0},
            retry_policy_config={"max_attempts": 2, "base_delay": 0.1},
            timeout_config={"default_timeout": 1.0},
            bulkhead_config={"max_concurrent_calls": 2, "max_queue_size": 1},
            rate_limiter_config={"max_requests": 5, "time_window": 1.0},
        )

        call_count = [0]

        def monitored_operation():
            call_count[0] += 1
            if call_count[0] == 1:
                raise ValueError("First attempt fails")
            time.sleep(0.1)
            return f"success_attempt_{call_count[0]}"

        # Should succeed with retry
        result = service.execute(monitored_operation)
        assert result == "success_attempt_2"
        assert call_count[0] == 2

    def test_resilience_pattern_conflicts(self):
        """Test handling of resilience pattern conflicts."""
        # Create service with potentially conflicting configurations
        service = ResilientService(
            name="conflict_test",
            circuit_breaker_config={"failure_threshold": 1},  # Very sensitive
            retry_policy_config={"max_attempts": 5},  # Many retries
            timeout_config={"default_timeout": 0.1},  # Very short timeout
        )

        def slow_operation():
            time.sleep(0.2)  # Slower than timeout
            return "should_not_complete"

        # Should timeout before retries can help
        with pytest.raises(TimeoutError):
            service.execute(slow_operation)

    def test_resilience_metrics_aggregation(self):
        """Test aggregation of metrics across resilience patterns."""
        service = ResilientService(
            name="metrics_test",
            circuit_breaker_config={"failure_threshold": 5},
            retry_policy_config={"max_attempts": 2},
            timeout_config={"default_timeout": 1.0},
        )

        # Execute various operations
        service.execute(lambda: "success_1")
        service.execute(lambda: "success_2")

        try:
            service.execute(lambda: 1 / 0)
        except ZeroDivisionError:
            pass

        try:
            service.execute(lambda: time.sleep(2.0))
        except TimeoutError:
            pass

        # Get comprehensive metrics
        metrics = service.get_comprehensive_statistics()

        # Verify all component metrics are included
        assert "circuit_breaker" in metrics
        assert "retry_policy" in metrics
        assert "timeout_manager" in metrics
        assert "overall" in metrics

        # Verify overall metrics make sense
        overall = metrics["overall"]
        assert overall["total_operations"] == 4
        assert overall["successful_operations"] == 2
        assert overall["failed_operations"] == 2
        assert "average_response_time" in overall
        assert "error_rate" in overall

    def test_resilience_configuration_validation(self):
        """Test validation of resilience configuration."""
        # Test valid configuration
        valid_config = {
            "circuit_breaker_config": {
                "failure_threshold": 5,
                "recovery_timeout": 10.0,
            },
            "retry_policy_config": {"max_attempts": 3, "base_delay": 0.1},
        }

        service = ResilientService(name="valid", **valid_config)
        assert service.validate_configuration() is True

        # Test invalid configuration
        invalid_config = {
            "circuit_breaker_config": {
                "failure_threshold": 0,  # Invalid
                "recovery_timeout": -1.0,  # Invalid
            }
        }

        with pytest.raises(ValueError):
            ResilientService(name="invalid", **invalid_config)
