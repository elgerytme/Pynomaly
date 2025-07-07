"""Branch coverage tests for circuit breaker resilience infrastructure."""

import asyncio
import time
from unittest.mock import Mock, patch

import pytest

from pynomaly.infrastructure.resilience.circuit_breaker import (
    CircuitBreaker,
    CircuitBreakerError,
    CircuitBreakerRegistry,
    CircuitState,
    circuit_breaker,
    database_circuit_breaker,
    external_api_circuit_breaker,
    redis_circuit_breaker,
)


class TestCircuitBreakerStateBranches:
    """Test circuit breaker state transition branches."""

    def test_closed_state_allows_calls(self):
        """Test that closed state allows all calls."""
        cb = CircuitBreaker(failure_threshold=2)
        
        # Mock successful function
        mock_func = Mock(return_value="success")
        
        # Should allow call in closed state
        result = cb.call(mock_func)
        assert result == "success"
        assert cb.state == CircuitState.CLOSED
        assert cb.failure_count == 0

    def test_open_state_blocks_calls(self):
        """Test that open state blocks all calls."""
        cb = CircuitBreaker(failure_threshold=1, recovery_timeout=1.0)
        
        # Force circuit to open
        mock_func = Mock(side_effect=Exception("test error"))
        
        with pytest.raises(Exception):
            cb.call(mock_func)
        
        # Circuit should be open now
        assert cb.state == CircuitState.OPEN
        
        # Next call should be blocked
        with pytest.raises(CircuitBreakerError):
            cb.call(mock_func)

    def test_half_open_state_allows_single_call(self):
        """Test that half-open state allows testing calls."""
        cb = CircuitBreaker(failure_threshold=1, recovery_timeout=0.1)
        
        # Open the circuit
        mock_func = Mock(side_effect=Exception("test error"))
        with pytest.raises(Exception):
            cb.call(mock_func)
        
        # Wait for recovery timeout
        time.sleep(0.2)
        
        # Should transition to half-open and allow call
        mock_func.reset_mock()
        mock_func.side_effect = None
        mock_func.return_value = "success"
        
        result = cb.call(mock_func)
        assert result == "success"
        assert cb.state == CircuitState.CLOSED  # Should recover

    def test_half_open_failure_reopens_circuit(self):
        """Test that failure in half-open state reopens circuit."""
        cb = CircuitBreaker(failure_threshold=1, recovery_timeout=0.1)
        
        # Open the circuit
        mock_func = Mock(side_effect=Exception("test error"))
        with pytest.raises(Exception):
            cb.call(mock_func)
        
        # Wait for recovery timeout
        time.sleep(0.2)
        
        # Fail in half-open state
        with pytest.raises(Exception):
            cb.call(mock_func)
        
        # Should reopen circuit
        assert cb.state == CircuitState.OPEN

    def test_closed_state_failure_count_reset(self):
        """Test that closed state resets failure count on success."""
        cb = CircuitBreaker(failure_threshold=3)
        
        # Cause some failures (but not enough to open)
        mock_func = Mock(side_effect=Exception("test error"))
        
        with pytest.raises(Exception):
            cb.call(mock_func)
        with pytest.raises(Exception):
            cb.call(mock_func)
        
        assert cb.failure_count == 2
        assert cb.state == CircuitState.CLOSED
        
        # Success should reset failure count
        mock_func.side_effect = None
        mock_func.return_value = "success"
        
        result = cb.call(mock_func)
        assert result == "success"
        assert cb.failure_count == 0


class TestCircuitBreakerTimeoutBranches:
    """Test circuit breaker timeout-related branches."""

    def test_recovery_timeout_not_reached(self):
        """Test calls blocked when recovery timeout not reached."""
        cb = CircuitBreaker(failure_threshold=1, recovery_timeout=10.0)
        
        # Open the circuit
        mock_func = Mock(side_effect=Exception("test error"))
        with pytest.raises(Exception):
            cb.call(mock_func)
        
        # Should be blocked immediately
        with pytest.raises(CircuitBreakerError) as exc_info:
            cb.call(mock_func)
        
        assert "Next attempt in" in str(exc_info.value)
        assert cb.state == CircuitState.OPEN

    def test_recovery_timeout_reached(self):
        """Test transition to half-open when timeout reached."""
        cb = CircuitBreaker(failure_threshold=1, recovery_timeout=0.1)
        
        # Open the circuit
        mock_func = Mock(side_effect=Exception("test error"))
        with pytest.raises(Exception):
            cb.call(mock_func)
        
        # Wait for recovery timeout
        time.sleep(0.2)
        
        # Should transition to half-open
        mock_func.reset_mock()
        mock_func.return_value = "success"
        
        result = cb.call(mock_func)
        assert result == "success"
        assert cb.state == CircuitState.CLOSED

    def test_next_attempt_time_none_branch(self):
        """Test branch when next_attempt_time is None."""
        cb = CircuitBreaker(failure_threshold=1, recovery_timeout=1.0)
        
        # Manually set state to OPEN without setting next_attempt_time
        cb._state = CircuitState.OPEN
        cb._next_attempt_time = None
        
        # Should block call
        mock_func = Mock(return_value="success")
        with pytest.raises(CircuitBreakerError):
            cb.call(mock_func)


class TestCircuitBreakerExceptionHandling:
    """Test exception handling branches."""

    def test_expected_exception_handling(self):
        """Test that expected exceptions are handled correctly."""
        cb = CircuitBreaker(
            failure_threshold=1,
            expected_exception=ValueError
        )
        
        # ValueError should be caught and recorded
        mock_func = Mock(side_effect=ValueError("test error"))
        
        with pytest.raises(ValueError):
            cb.call(mock_func)
        
        assert cb.failure_count == 1
        assert cb.state == CircuitState.OPEN

    def test_unexpected_exception_not_counted(self):
        """Test that unexpected exceptions are not counted as failures."""
        cb = CircuitBreaker(
            failure_threshold=1,
            expected_exception=ValueError
        )
        
        # TypeError should not be counted as failure
        mock_func = Mock(side_effect=TypeError("test error"))
        
        with pytest.raises(TypeError):
            cb.call(mock_func)
        
        assert cb.failure_count == 0
        assert cb.state == CircuitState.CLOSED

    def test_exception_tuple_handling(self):
        """Test handling of exception tuple."""
        cb = CircuitBreaker(
            failure_threshold=1,
            expected_exception=(ValueError, TypeError)
        )
        
        # Both exceptions should be counted
        mock_func = Mock(side_effect=ValueError("test error"))
        
        with pytest.raises(ValueError):
            cb.call(mock_func)
        
        assert cb.failure_count == 1
        
        # Reset and try with TypeError
        cb.reset()
        mock_func.side_effect = TypeError("test error")
        
        with pytest.raises(TypeError):
            cb.call(mock_func)
        
        assert cb.failure_count == 1


class TestCircuitBreakerAsyncBranches:
    """Test async circuit breaker branches."""

    @pytest.mark.asyncio
    async def test_async_coroutine_function_branch(self):
        """Test async function handling branch."""
        cb = CircuitBreaker(failure_threshold=2)
        
        async def async_func():
            return "async_success"
        
        result = await cb.acall(async_func)
        assert result == "async_success"

    @pytest.mark.asyncio
    async def test_async_sync_function_branch(self):
        """Test sync function in async call branch."""
        cb = CircuitBreaker(failure_threshold=2)
        
        def sync_func():
            return "sync_success"
        
        result = await cb.acall(sync_func)
        assert result == "sync_success"

    @pytest.mark.asyncio
    async def test_async_decorator_branches(self):
        """Test decorator branches for async and sync functions."""
        cb = CircuitBreaker(failure_threshold=2)
        
        # Test async function decoration
        @cb
        async def async_func():
            return "async_decorated"
        
        result = await async_func()
        assert result == "async_decorated"
        
        # Test sync function decoration
        @cb
        def sync_func():
            return "sync_decorated"
        
        result = sync_func()
        assert result == "sync_decorated"

    @pytest.mark.asyncio
    async def test_async_circuit_open_blocking(self):
        """Test async call blocking when circuit is open."""
        cb = CircuitBreaker(failure_threshold=1, recovery_timeout=1.0)
        
        # Open circuit
        async def failing_func():
            raise Exception("test error")
        
        with pytest.raises(Exception):
            await cb.acall(failing_func)
        
        # Should block next call
        with pytest.raises(CircuitBreakerError):
            await cb.acall(failing_func)


class TestCircuitBreakerRegistryBranches:
    """Test circuit breaker registry branches."""

    def test_registry_get_existing_breaker(self):
        """Test getting existing breaker from registry."""
        registry = CircuitBreakerRegistry()
        breaker = CircuitBreaker(failure_threshold=2)
        
        registry.register("test_breaker", breaker)
        
        retrieved = registry.get("test_breaker")
        assert retrieved is breaker
        assert retrieved.name == "test_breaker"

    def test_registry_get_nonexistent_breaker(self):
        """Test getting nonexistent breaker returns None."""
        registry = CircuitBreakerRegistry()
        
        result = registry.get("nonexistent")
        assert result is None

    def test_registry_create_and_register(self):
        """Test creating and registering new breaker."""
        registry = CircuitBreakerRegistry()
        
        breaker = registry.create_and_register(
            "new_breaker",
            failure_threshold=3,
            recovery_timeout=30.0
        )
        
        assert breaker.name == "new_breaker"
        assert breaker.failure_threshold == 3
        assert breaker.recovery_timeout == 30.0
        
        # Should be retrievable
        retrieved = registry.get("new_breaker")
        assert retrieved is breaker

    def test_registry_stats_multiple_breakers(self):
        """Test getting stats for multiple breakers."""
        registry = CircuitBreakerRegistry()
        
        breaker1 = registry.create_and_register("breaker1", failure_threshold=2)
        breaker2 = registry.create_and_register("breaker2", failure_threshold=3)
        
        stats = registry.get_stats()
        
        assert "breaker1" in stats
        assert "breaker2" in stats
        assert stats["breaker1"]["failure_threshold"] == 2
        assert stats["breaker2"]["failure_threshold"] == 3

    def test_registry_reset_all(self):
        """Test resetting all breakers in registry."""
        registry = CircuitBreakerRegistry()
        
        breaker = registry.create_and_register("test", failure_threshold=1)
        
        # Force failure
        mock_func = Mock(side_effect=Exception("test error"))
        with pytest.raises(Exception):
            breaker.call(mock_func)
        
        assert breaker.state == CircuitState.OPEN
        
        # Reset all
        registry.reset_all()
        
        assert breaker.state == CircuitState.CLOSED
        assert breaker.failure_count == 0


class TestCircuitBreakerDecoratorBranches:
    """Test circuit breaker decorator branches."""

    def test_decorator_creates_new_breaker(self):
        """Test decorator creates new breaker when none exists."""
        @circuit_breaker("new_decorator_breaker", failure_threshold=2)
        def test_func():
            return "success"
        
        result = test_func()
        assert result == "success"
        
        # Should be registered
        from pynomaly.infrastructure.resilience.circuit_breaker import circuit_breaker_registry
        breaker = circuit_breaker_registry.get("new_decorator_breaker")
        assert breaker is not None
        assert breaker.failure_threshold == 2

    def test_decorator_reuses_existing_breaker(self):
        """Test decorator reuses existing breaker."""
        from pynomaly.infrastructure.resilience.circuit_breaker import circuit_breaker_registry
        
        # Create breaker first
        existing_breaker = circuit_breaker_registry.create_and_register(
            "existing_breaker",
            failure_threshold=5
        )
        
        @circuit_breaker("existing_breaker", failure_threshold=2)
        def test_func():
            return "success"
        
        result = test_func()
        assert result == "success"
        
        # Should use existing breaker (not create new one)
        breaker = circuit_breaker_registry.get("existing_breaker")
        assert breaker is existing_breaker
        assert breaker.failure_threshold == 5  # Original threshold preserved


class TestSpecializedCircuitBreakerBranches:
    """Test specialized circuit breaker factory function branches."""

    def test_database_circuit_breaker_with_imports(self):
        """Test database circuit breaker with available imports."""
        with patch.dict('sys.modules', {'psycopg2': Mock(), 'sqlalchemy.exc': Mock()}):
            cb = database_circuit_breaker("test_db")
            
            assert cb.name == "test_db"
            assert cb.failure_threshold == 3
            assert cb.recovery_timeout == 30.0

    def test_database_circuit_breaker_without_imports(self):
        """Test database circuit breaker without imports (ImportError branch)."""
        with patch.dict('sys.modules', {}):
            # Mock ImportError
            with patch('builtins.__import__', side_effect=ImportError):
                cb = database_circuit_breaker("test_db_no_imports")
                
                assert cb.name == "test_db_no_imports"
                assert cb.failure_threshold == 3

    def test_external_api_circuit_breaker_with_imports(self):
        """Test external API circuit breaker with available imports."""
        with patch.dict('sys.modules', {'httpx': Mock(), 'requests': Mock()}):
            cb = external_api_circuit_breaker("test_api")
            
            assert cb.name == "test_api"
            assert cb.failure_threshold == 5
            assert cb.recovery_timeout == 60.0

    def test_external_api_circuit_breaker_without_imports(self):
        """Test external API circuit breaker without imports."""
        with patch.dict('sys.modules', {}):
            with patch('builtins.__import__', side_effect=ImportError):
                cb = external_api_circuit_breaker("test_api_no_imports")
                
                assert cb.name == "test_api_no_imports"
                assert cb.failure_threshold == 5

    def test_redis_circuit_breaker_with_imports(self):
        """Test Redis circuit breaker with available imports."""
        with patch.dict('sys.modules', {'redis': Mock()}):
            cb = redis_circuit_breaker("test_redis")
            
            assert cb.name == "test_redis"
            assert cb.failure_threshold == 3
            assert cb.recovery_timeout == 15.0

    def test_redis_circuit_breaker_without_imports(self):
        """Test Redis circuit breaker without imports."""
        with patch.dict('sys.modules', {}):
            with patch('builtins.__import__', side_effect=ImportError):
                cb = redis_circuit_breaker("test_redis_no_imports")
                
                assert cb.name == "test_redis_no_imports"
                assert cb.failure_threshold == 3


class TestCircuitBreakerStatsBranches:
    """Test circuit breaker statistics branches."""

    def test_stats_with_zero_total_calls(self):
        """Test failure rate calculation with zero total calls."""
        cb = CircuitBreaker(failure_threshold=2)
        
        stats = cb.stats
        
        assert stats["total_calls"] == 0
        assert stats["failure_rate"] == 0.0  # Should use max(total_calls, 1)

    def test_stats_with_multiple_calls(self):
        """Test failure rate calculation with multiple calls."""
        cb = CircuitBreaker(failure_threshold=5)
        
        mock_func = Mock(return_value="success")
        
        # Make successful calls
        cb.call(mock_func)
        cb.call(mock_func)
        
        # Make failed calls
        mock_func.side_effect = Exception("test error")
        
        with pytest.raises(Exception):
            cb.call(mock_func)
        
        stats = cb.stats
        
        assert stats["total_calls"] == 3
        assert stats["successful_calls"] == 2
        assert stats["failed_calls"] == 1
        assert stats["failure_rate"] == 1/3

    def test_stats_time_fields(self):
        """Test stats time field branches."""
        cb = CircuitBreaker(failure_threshold=1, recovery_timeout=1.0)
        
        # Initially no failure time
        stats = cb.stats
        assert stats["last_failure_time"] is None
        assert stats["next_attempt_time"] is None
        
        # After failure
        mock_func = Mock(side_effect=Exception("test error"))
        with pytest.raises(Exception):
            cb.call(mock_func)
        
        stats = cb.stats
        assert stats["last_failure_time"] is not None
        assert stats["next_attempt_time"] is not None

    def test_blocked_calls_counter(self):
        """Test blocked calls counter branch."""
        cb = CircuitBreaker(failure_threshold=1, recovery_timeout=1.0)
        
        # Open circuit
        mock_func = Mock(side_effect=Exception("test error"))
        with pytest.raises(Exception):
            cb.call(mock_func)
        
        # Try multiple blocked calls
        for _ in range(3):
            with pytest.raises(CircuitBreakerError):
                cb.call(mock_func)
        
        stats = cb.stats
        assert stats["blocked_calls"] == 3


class TestCircuitBreakerEdgeCases:
    """Test edge cases and error conditions."""

    def test_name_default_assignment(self):
        """Test default name assignment branch."""
        cb = CircuitBreaker()
        assert cb.name == "unnamed"
        
        cb_with_name = CircuitBreaker(name="custom")
        assert cb_with_name.name == "custom"

    def test_failure_threshold_boundary(self):
        """Test failure threshold boundary conditions."""
        cb = CircuitBreaker(failure_threshold=1)
        
        mock_func = Mock(side_effect=Exception("test error"))
        
        # First failure should open circuit
        with pytest.raises(Exception):
            cb.call(mock_func)
        
        assert cb.state == CircuitState.OPEN
        assert cb.failure_count == 1

    def test_recovery_timeout_zero(self):
        """Test zero recovery timeout edge case."""
        cb = CircuitBreaker(failure_threshold=1, recovery_timeout=0.0)
        
        # Open circuit
        mock_func = Mock(side_effect=Exception("test error"))
        with pytest.raises(Exception):
            cb.call(mock_func)
        
        # Should immediately allow retry
        mock_func.reset_mock()
        mock_func.return_value = "success"
        
        result = cb.call(mock_func)
        assert result == "success"
        assert cb.state == CircuitState.CLOSED

    def test_coroutine_function_detection(self):
        """Test asyncio.iscoroutinefunction branches."""
        cb = CircuitBreaker(failure_threshold=2)
        
        async def async_func():
            return "async"
        
        def sync_func():
            return "sync"
        
        # Test decorator detection
        async_decorated = cb(async_func)
        sync_decorated = cb(sync_func)
        
        assert asyncio.iscoroutinefunction(async_decorated)
        assert not asyncio.iscoroutinefunction(sync_decorated)

    def test_circuit_breaker_error_message_formatting(self):
        """Test CircuitBreakerError message formatting."""
        cb = CircuitBreaker(failure_threshold=1, recovery_timeout=5.0)
        
        # Open circuit
        mock_func = Mock(side_effect=Exception("test error"))
        with pytest.raises(Exception):
            cb.call(mock_func)
        
        # Check error message format
        with pytest.raises(CircuitBreakerError) as exc_info:
            cb.call(mock_func)
        
        error_msg = str(exc_info.value)
        assert "Circuit breaker" in error_msg
        assert "is OPEN" in error_msg
        assert "Next attempt in" in error_msg
        assert "seconds" in error_msg