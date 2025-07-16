"""Comprehensive tests for error handling infrastructure."""

import asyncio
import logging
import time
from unittest.mock import Mock, patch

import pytest

from monorepo.domain.exceptions import (
    AuthenticationError,
    AuthorizationError,
    ConfigurationError,
    DatasetError,
    DetectorError,
    InfrastructureError,
    PynamolyError,
    ValidationError,
)
from monorepo.infrastructure.error_handling.error_handler import (
    ErrorHandler,
    create_default_error_handler,
)
from monorepo.infrastructure.error_handling.recovery_strategies import (
    CircuitBreakerStrategy,
    FallbackStrategy,
    RecoveryStrategy,
    RecoveryStrategyRegistry,
    RetryStrategy,
    create_default_recovery_registry,
)


class TestErrorHandler:
    """Test error handler functionality."""

    def test_error_handler_initialization(self):
        """Test error handler initialization with default settings."""
        handler = ErrorHandler()

        assert handler.logger is not None
        assert handler.enable_recovery is True
        assert handler.enable_reporting is True

    def test_error_handler_initialization_custom_settings(self):
        """Test error handler initialization with custom settings."""
        mock_logger = Mock(spec=logging.Logger)

        handler = ErrorHandler(
            logger=mock_logger,
            enable_recovery=False,
            enable_reporting=False,
        )

        assert handler.logger is mock_logger
        assert handler.enable_recovery is False
        assert handler.enable_reporting is False

    def test_handle_error_basic(self):
        """Test basic error handling."""
        handler = ErrorHandler()
        error = ValidationError("Invalid input", field="username", value="")

        response = handler.handle_error(error)

        assert response["error"] is True
        assert response["error_id"] is not None
        assert response["timestamp"] is not None
        assert response["type"] == "ValidationError"
        assert response["message"] == "Invalid input"
        assert response["error_code"] == "VALIDATION_ERROR"
        assert response["category"] == "client_error"
        assert "recovery_suggestions" in response

    def test_handle_error_with_context(self):
        """Test error handling with context."""
        handler = ErrorHandler()
        error = DatasetError("Dataset not found")
        context = {"dataset_id": "123", "operation": "load_dataset"}

        response = handler.handle_error(
            error,
            context=context,
            user_id="user456",
            operation="data_processing",
        )

        assert response["error"] is True
        assert response["type"] == "DatasetError"
        assert response["error_code"] == "DOMAIN_ERROR"
        assert response["category"] == "business_error"

    def test_handle_error_with_details(self):
        """Test error handling with error details."""
        handler = ErrorHandler()
        error = ValidationError(
            "Multiple validation errors",
            field="data",
            value=None,
            details={
                "field_errors": ["name is required", "email is invalid"],
                "validation_context": "user_creation",
            },
        )

        response = handler.handle_error(error)

        assert response["error"] is True
        assert response["details"] == {
            "field_errors": ["name is required", "email is invalid"],
            "validation_context": "user_creation",
        }

    def test_handle_error_logging(self):
        """Test that errors are logged appropriately."""
        mock_logger = Mock(spec=logging.Logger)
        handler = ErrorHandler(logger=mock_logger)

        # Test warning-level error
        validation_error = ValidationError("Invalid input")
        handler.handle_error(validation_error)

        mock_logger.log.assert_called_once()
        call_args = mock_logger.log.call_args
        assert call_args[0][0] == logging.WARNING  # Log level
        assert "ValidationError" in call_args[0][1]  # Log message

    def test_handle_error_logging_with_stack_trace(self):
        """Test that severe errors are logged with stack trace."""
        mock_logger = Mock(spec=logging.Logger)
        handler = ErrorHandler(logger=mock_logger)

        # Test error-level error
        config_error = ConfigurationError("Invalid configuration")
        handler.handle_error(config_error)

        mock_logger.log.assert_called_once()
        call_args = mock_logger.log.call_args
        assert call_args[0][0] == logging.ERROR  # Log level
        assert call_args[1]["exc_info"] is True  # Stack trace enabled

    def test_handle_error_reporting(self):
        """Test error reporting functionality."""
        mock_logger = Mock(spec=logging.Logger)
        handler = ErrorHandler(logger=mock_logger, enable_reporting=True)

        error = InfrastructureError("Database connection failed")
        handler.handle_error(error)

        # Check that debug log was called for reporting
        debug_calls = [
            call
            for call in mock_logger.debug.call_args_list
            if "Error reported" in str(call)
        ]
        assert len(debug_calls) > 0

    def test_handle_error_reporting_disabled(self):
        """Test that reporting can be disabled."""
        mock_logger = Mock(spec=logging.Logger)
        handler = ErrorHandler(logger=mock_logger, enable_reporting=False)

        error = InfrastructureError("Database connection failed")
        handler.handle_error(error)

        # Check that no debug log was called for reporting
        debug_calls = [
            call
            for call in mock_logger.debug.call_args_list
            if "Error reported" in str(call)
        ]
        assert len(debug_calls) == 0

    def test_handle_error_reporting_failure(self):
        """Test handling of error reporting failures."""
        mock_logger = Mock(spec=logging.Logger)

        # Make debug method raise an exception
        mock_logger.debug.side_effect = Exception("Reporting service unavailable")

        handler = ErrorHandler(logger=mock_logger, enable_reporting=True)

        error = InfrastructureError("Database connection failed")

        # Should not raise exception even if reporting fails
        response = handler.handle_error(error)

        assert response["error"] is True
        assert response["type"] == "InfrastructureError"

        # Check that warning was logged about reporting failure
        mock_logger.warning.assert_called_once()
        warning_call = mock_logger.warning.call_args[0][0]
        assert "Failed to report error" in warning_call

    def test_get_error_log_level(self):
        """Test error log level determination."""
        handler = ErrorHandler()

        # Test different error types
        assert handler._get_error_log_level(ValidationError("test")) == logging.WARNING
        assert (
            handler._get_error_log_level(AuthenticationError("test")) == logging.WARNING
        )
        assert (
            handler._get_error_log_level(AuthorizationError("test")) == logging.WARNING
        )
        assert handler._get_error_log_level(ConfigurationError("test")) == logging.ERROR
        assert handler._get_error_log_level(DatasetError("test")) == logging.WARNING
        assert handler._get_error_log_level(DetectorError("test")) == logging.WARNING
        assert (
            handler._get_error_log_level(InfrastructureError("test")) == logging.ERROR
        )
        assert handler._get_error_log_level(PynamolyError("test")) == logging.WARNING
        assert handler._get_error_log_level(Exception("test")) == logging.ERROR

    def test_get_recovery_suggestions(self):
        """Test recovery suggestions for different error types."""
        handler = ErrorHandler()

        # Test validation error suggestions
        validation_error = ValidationError("Invalid input")
        suggestions = handler._get_recovery_suggestions(validation_error)
        assert "Check input parameters" in suggestions[0]
        assert "API documentation" in suggestions[1]

        # Test authentication error suggestions
        auth_error = AuthenticationError("Invalid credentials")
        suggestions = handler._get_recovery_suggestions(auth_error)
        assert "authentication credentials" in suggestions[0]
        assert "logging in again" in suggestions[1]

        # Test authorization error suggestions
        authz_error = AuthorizationError("Insufficient permissions")
        suggestions = handler._get_recovery_suggestions(authz_error)
        assert "administrator" in suggestions[0]

        # Test dataset error suggestions
        dataset_error = DatasetError("Invalid dataset")
        suggestions = handler._get_recovery_suggestions(dataset_error)
        assert "dataset format" in suggestions[0]
        assert "missing or invalid data" in suggestions[1]

        # Test detector error suggestions
        detector_error = DetectorError("Detector not trained")
        suggestions = handler._get_recovery_suggestions(detector_error)
        assert "detector configuration" in suggestions[0]
        assert "properly trained" in suggestions[1]

        # Test infrastructure error suggestions
        infra_error = InfrastructureError("Service unavailable")
        suggestions = handler._get_recovery_suggestions(infra_error)
        assert "Try again" in suggestions[0]
        assert "contact support" in suggestions[1]

        # Test unknown error suggestions
        unknown_error = Exception("Unknown error")
        suggestions = handler._get_recovery_suggestions(unknown_error)
        assert "Try again or contact support" in suggestions[0]

    def test_handle_validation_error_helper(self):
        """Test validation error helper method."""
        handler = ErrorHandler()

        response = handler.handle_validation_error(
            field="username",
            value="",
            message="Username is required",
            context={"form": "registration"},
        )

        assert response["error"] is True
        assert response["type"] == "ValidationError"
        assert response["message"] == "Username is required"
        assert response["error_code"] == "VALIDATION_ERROR"

    def test_handle_not_found_error_helper(self):
        """Test not found error helper method."""
        handler = ErrorHandler()

        response = handler.handle_not_found_error(
            resource_type="User",
            resource_id="123",
            context={"operation": "get_user"},
        )

        assert response["error"] is True
        assert response["type"] == "EntityNotFoundError"
        assert "User with ID '123' not found" in response["message"]

    def test_handle_unexpected_error_helper(self):
        """Test unexpected error helper method."""
        handler = ErrorHandler()

        original_error = KeyError("missing_key")
        response = handler.handle_unexpected_error(
            error=original_error,
            context={"operation": "data_processing"},
        )

        assert response["error"] is True
        assert response["type"] == "PynamolyError"
        assert "Unexpected error" in response["message"]
        assert response["details"]["original_error_type"] == "KeyError"

    def test_error_response_format(self):
        """Test error response format consistency."""
        handler = ErrorHandler()

        error = ValidationError("Test error")
        response = handler.handle_error(error)

        # Check required fields
        required_fields = [
            "error",
            "error_id",
            "timestamp",
            "type",
            "message",
            "error_code",
            "category",
            "recovery_suggestions",
        ]
        for field in required_fields:
            assert field in response

        # Check field types
        assert isinstance(response["error"], bool)
        assert isinstance(response["error_id"], str)
        assert isinstance(response["timestamp"], str)
        assert isinstance(response["type"], str)
        assert isinstance(response["message"], str)
        assert isinstance(response["error_code"], str)
        assert isinstance(response["category"], str)
        assert isinstance(response["recovery_suggestions"], list)

    def test_create_default_error_handler(self):
        """Test default error handler creation."""
        handler = create_default_error_handler()

        assert isinstance(handler, ErrorHandler)
        assert handler.enable_recovery is True
        assert handler.enable_reporting is True
        assert handler.logger.name == "monorepo.errors"


class TestRetryStrategy:
    """Test retry strategy functionality."""

    def test_retry_strategy_initialization(self):
        """Test retry strategy initialization."""
        strategy = RetryStrategy(
            max_retries=5,
            base_delay=2.0,
            max_delay=30.0,
            exponential_base=1.5,
            jitter=False,
        )

        assert strategy.max_retries == 5
        assert strategy.base_delay == 2.0
        assert strategy.max_delay == 30.0
        assert strategy.exponential_base == 1.5
        assert strategy.jitter is False

    def test_retry_strategy_can_recover(self):
        """Test retry strategy recovery capability check."""
        strategy = RetryStrategy()

        # Test recoverable errors
        assert (
            asyncio.run(strategy.can_recover(InfrastructureError("test"), {})) is True
        )
        assert asyncio.run(strategy.can_recover(ConnectionError("test"), {})) is True
        assert asyncio.run(strategy.can_recover(TimeoutError("test"), {})) is True

        # Test error message patterns
        assert (
            asyncio.run(strategy.can_recover(Exception("Connection failed"), {}))
            is True
        )
        assert (
            asyncio.run(strategy.can_recover(Exception("Request timeout"), {})) is True
        )
        assert (
            asyncio.run(strategy.can_recover(Exception("Service unavailable"), {}))
            is True
        )
        assert (
            asyncio.run(strategy.can_recover(Exception("Rate limit exceeded"), {}))
            is True
        )

        # Test non-recoverable errors
        assert asyncio.run(strategy.can_recover(ValidationError("test"), {})) is False
        assert (
            asyncio.run(strategy.can_recover(AuthenticationError("test"), {})) is False
        )

    def test_retry_strategy_successful_recovery(self):
        """Test successful retry recovery."""
        strategy = RetryStrategy(max_retries=3, base_delay=0.01)

        call_count = 0

        def failing_operation():
            nonlocal call_count
            call_count += 1
            if call_count < 3:
                raise ConnectionError("Connection failed")
            return "success"

        result = asyncio.run(
            strategy.recover(failing_operation, ConnectionError("test"), {})
        )

        assert result == "success"
        assert call_count == 3

    def test_retry_strategy_async_operation(self):
        """Test retry strategy with async operations."""
        strategy = RetryStrategy(max_retries=2, base_delay=0.01)

        call_count = 0

        async def async_failing_operation():
            nonlocal call_count
            call_count += 1
            if call_count < 2:
                raise ConnectionError("Connection failed")
            return "async_success"

        result = asyncio.run(
            strategy.recover(async_failing_operation, ConnectionError("test"), {})
        )

        assert result == "async_success"
        assert call_count == 2

    def test_retry_strategy_max_retries_exceeded(self):
        """Test retry strategy when max retries are exceeded."""
        strategy = RetryStrategy(max_retries=2, base_delay=0.01)

        def always_failing_operation():
            raise ConnectionError("Connection failed")

        with pytest.raises(
            PynamolyError, match="Operation failed after 2 retry attempts"
        ):
            asyncio.run(
                strategy.recover(always_failing_operation, ConnectionError("test"), {})
            )

    def test_retry_strategy_exponential_backoff(self):
        """Test exponential backoff calculation."""
        strategy = RetryStrategy(
            max_retries=3, base_delay=1.0, exponential_base=2.0, jitter=False
        )

        call_times = []

        def failing_operation():
            call_times.append(time.time())
            raise ConnectionError("Connection failed")

        start_time = time.time()

        with pytest.raises(PynamolyError):
            asyncio.run(
                strategy.recover(failing_operation, ConnectionError("test"), {})
            )

        # Check that delays increase exponentially
        assert len(call_times) == 3

        # First call should be immediate
        assert call_times[0] - start_time < 0.1

        # Second call should be after ~1s delay
        assert 0.9 < call_times[1] - call_times[0] < 1.1

        # Third call should be after ~2s delay
        assert 1.9 < call_times[2] - call_times[1] < 2.1

    def test_retry_strategy_max_delay_limit(self):
        """Test that delays don't exceed max_delay."""
        strategy = RetryStrategy(
            max_retries=5,
            base_delay=1.0,
            max_delay=2.0,
            exponential_base=10.0,
            jitter=False,
        )

        call_times = []

        def failing_operation():
            call_times.append(time.time())
            raise ConnectionError("Connection failed")

        with pytest.raises(PynamolyError):
            asyncio.run(
                strategy.recover(failing_operation, ConnectionError("test"), {})
            )

        # Check that delays don't exceed max_delay
        for i in range(1, len(call_times)):
            delay = call_times[i] - call_times[i - 1]
            assert delay <= 2.1  # Allow small tolerance

    def test_retry_strategy_with_jitter(self):
        """Test retry strategy with jitter enabled."""
        strategy = RetryStrategy(max_retries=2, base_delay=1.0, jitter=True)

        call_times = []

        def failing_operation():
            call_times.append(time.time())
            raise ConnectionError("Connection failed")

        with pytest.raises(PynamolyError):
            asyncio.run(
                strategy.recover(failing_operation, ConnectionError("test"), {})
            )

        # Check that jitter affects delays
        if len(call_times) > 1:
            delay = call_times[1] - call_times[0]
            # With jitter, delay should be between 0.5 and 1.5 seconds
            assert 0.4 < delay < 1.6


class TestCircuitBreakerStrategy:
    """Test circuit breaker strategy functionality."""

    def test_circuit_breaker_initialization(self):
        """Test circuit breaker initialization."""
        strategy = CircuitBreakerStrategy(
            failure_threshold=3,
            recovery_timeout=30.0,
            expected_exception=ConnectionError,
        )

        assert strategy.failure_threshold == 3
        assert strategy.recovery_timeout == 30.0
        assert strategy.expected_exception == ConnectionError
        assert strategy.failure_count == 0
        assert strategy.state == "closed"

    def test_circuit_breaker_can_recover(self):
        """Test circuit breaker recovery capability check."""
        strategy = CircuitBreakerStrategy(expected_exception=ConnectionError)

        # Test expected exception
        assert asyncio.run(strategy.can_recover(ConnectionError("test"), {})) is True

        # Test unexpected exception
        assert asyncio.run(strategy.can_recover(ValidationError("test"), {})) is False

    def test_circuit_breaker_closed_state(self):
        """Test circuit breaker in closed state."""
        strategy = CircuitBreakerStrategy(failure_threshold=3)

        def successful_operation():
            return "success"

        # Should work normally in closed state
        result = asyncio.run(
            strategy.recover(successful_operation, Exception("test"), {})
        )
        assert result == "success"
        assert strategy.state == "closed"

    def test_circuit_breaker_opening(self):
        """Test circuit breaker opening after failures."""
        strategy = CircuitBreakerStrategy(failure_threshold=2)

        def failing_operation():
            raise ConnectionError("Connection failed")

        # First failure
        with pytest.raises(ConnectionError):
            asyncio.run(
                strategy.recover(failing_operation, ConnectionError("test"), {})
            )

        assert strategy.failure_count == 1
        assert strategy.state == "closed"

        # Second failure - should open circuit
        with pytest.raises(ConnectionError):
            asyncio.run(
                strategy.recover(failing_operation, ConnectionError("test"), {})
            )

        assert strategy.failure_count == 2
        assert strategy.state == "open"

    def test_circuit_breaker_open_state(self):
        """Test circuit breaker in open state."""
        strategy = CircuitBreakerStrategy(failure_threshold=1, recovery_timeout=60.0)

        # Force circuit open
        strategy.failure_count = 1
        strategy.state = "open"
        strategy.last_failure_time = time.time()

        def operation():
            return "success"

        # Should reject calls when circuit is open
        with pytest.raises(PynamolyError, match="Circuit breaker is open"):
            asyncio.run(strategy.recover(operation, Exception("test"), {}))

    def test_circuit_breaker_half_open_state(self):
        """Test circuit breaker in half-open state."""
        strategy = CircuitBreakerStrategy(failure_threshold=1, recovery_timeout=0.1)

        # Force circuit open
        strategy.failure_count = 1
        strategy.state = "open"
        strategy.last_failure_time = time.time() - 0.2  # Past recovery timeout

        def successful_operation():
            return "success"

        # Should try operation in half-open state and reset on success
        result = asyncio.run(
            strategy.recover(successful_operation, Exception("test"), {})
        )

        assert result == "success"
        assert strategy.state == "closed"
        assert strategy.failure_count == 0

    def test_circuit_breaker_half_open_failure(self):
        """Test circuit breaker failure in half-open state."""
        strategy = CircuitBreakerStrategy(failure_threshold=1, recovery_timeout=0.1)

        # Force circuit open
        strategy.failure_count = 1
        strategy.state = "open"
        strategy.last_failure_time = time.time() - 0.2  # Past recovery timeout

        def failing_operation():
            raise ConnectionError("Still failing")

        # Should go back to open state on failure
        with pytest.raises(ConnectionError):
            asyncio.run(strategy.recover(failing_operation, Exception("test"), {}))

        assert strategy.state == "open"

    def test_circuit_breaker_async_operation(self):
        """Test circuit breaker with async operations."""
        strategy = CircuitBreakerStrategy(failure_threshold=3)

        async def async_operation():
            return "async_success"

        result = asyncio.run(strategy.recover(async_operation, Exception("test"), {}))

        assert result == "async_success"
        assert strategy.state == "closed"


class TestFallbackStrategy:
    """Test fallback strategy functionality."""

    def test_fallback_strategy_initialization(self):
        """Test fallback strategy initialization."""

        def fallback_func():
            return "fallback_result"

        strategy = FallbackStrategy(fallback_operation=fallback_func)

        assert strategy.fallback_operation is fallback_func

    def test_fallback_strategy_can_recover(self):
        """Test fallback strategy recovery capability check."""

        def fallback_func():
            return "fallback_result"

        strategy = FallbackStrategy(fallback_operation=fallback_func)

        # Should be able to recover when fallback is available
        assert asyncio.run(strategy.can_recover(Exception("test"), {})) is True

        # Should not be able to recover when no fallback
        strategy_no_fallback = FallbackStrategy(fallback_operation=None)
        assert (
            asyncio.run(strategy_no_fallback.can_recover(Exception("test"), {}))
            is False
        )

    def test_fallback_strategy_successful_recovery(self):
        """Test successful fallback recovery."""

        def fallback_func():
            return "fallback_result"

        strategy = FallbackStrategy(fallback_operation=fallback_func)

        def failing_operation():
            raise Exception("Primary failed")

        result = asyncio.run(strategy.recover(failing_operation, Exception("test"), {}))

        assert result == "fallback_result"

    def test_fallback_strategy_async_fallback(self):
        """Test fallback strategy with async fallback operation."""

        async def async_fallback():
            return "async_fallback_result"

        strategy = FallbackStrategy(fallback_operation=async_fallback)

        def failing_operation():
            raise Exception("Primary failed")

        result = asyncio.run(strategy.recover(failing_operation, Exception("test"), {}))

        assert result == "async_fallback_result"

    def test_fallback_strategy_no_fallback(self):
        """Test fallback strategy when no fallback is available."""
        strategy = FallbackStrategy(fallback_operation=None)

        def failing_operation():
            raise Exception("Primary failed")

        with pytest.raises(PynamolyError, match="No fallback operation available"):
            asyncio.run(strategy.recover(failing_operation, Exception("test"), {}))

    def test_fallback_strategy_fallback_failure(self):
        """Test fallback strategy when fallback also fails."""

        def failing_fallback():
            raise Exception("Fallback also failed")

        strategy = FallbackStrategy(fallback_operation=failing_fallback)

        def failing_operation():
            raise Exception("Primary failed")

        with pytest.raises(
            PynamolyError, match="Both primary and fallback operations failed"
        ):
            asyncio.run(strategy.recover(failing_operation, Exception("test"), {}))


class TestRecoveryStrategyRegistry:
    """Test recovery strategy registry functionality."""

    def test_registry_initialization(self):
        """Test registry initialization."""
        registry = RecoveryStrategyRegistry()

        assert len(registry.strategies) == 0

    def test_registry_register_strategy(self):
        """Test registering recovery strategies."""
        registry = RecoveryStrategyRegistry()
        strategy = RetryStrategy(max_retries=3)

        registry.register_strategy(strategy)

        assert len(registry.strategies) == 1
        assert registry.strategies[0] is strategy

    def test_registry_attempt_recovery_success(self):
        """Test successful recovery using registry."""
        registry = RecoveryStrategyRegistry()

        # Register a strategy that can recover from ConnectionError
        strategy = RetryStrategy(max_retries=2, base_delay=0.01)
        registry.register_strategy(strategy)

        call_count = 0

        def failing_operation():
            nonlocal call_count
            call_count += 1
            if call_count < 2:
                raise ConnectionError("Connection failed")
            return "success"

        result = asyncio.run(
            registry.attempt_recovery(
                failing_operation,
                ConnectionError("test"),
                {"operation": "test"},
            )
        )

        assert result == "success"
        assert call_count == 2

    def test_registry_attempt_recovery_multiple_strategies(self):
        """Test recovery with multiple strategies."""
        registry = RecoveryStrategyRegistry()

        # Register strategies in order
        # First strategy can't recover from ValidationError
        retry_strategy = RetryStrategy(max_retries=2, base_delay=0.01)
        registry.register_strategy(retry_strategy)

        # Second strategy can recover
        fallback_strategy = FallbackStrategy(
            fallback_operation=lambda: "fallback_result"
        )
        registry.register_strategy(fallback_strategy)

        def failing_operation():
            raise ValidationError("Invalid input")

        result = asyncio.run(
            registry.attempt_recovery(
                failing_operation,
                ValidationError("test"),
                {"operation": "test"},
            )
        )

        assert result == "fallback_result"

    def test_registry_attempt_recovery_all_strategies_fail(self):
        """Test recovery when all strategies fail."""
        registry = RecoveryStrategyRegistry()

        # Register a strategy that can't recover from this error
        strategy = RetryStrategy(max_retries=1, base_delay=0.01)
        registry.register_strategy(strategy)

        def failing_operation():
            raise ValidationError("Invalid input")

        with pytest.raises(PynamolyError, match="All recovery strategies failed"):
            asyncio.run(
                registry.attempt_recovery(
                    failing_operation,
                    ValidationError("test"),
                    {"operation": "test"},
                )
            )

    def test_registry_recovery_strategy_failure(self):
        """Test registry handling of strategy failures."""
        registry = RecoveryStrategyRegistry()

        # Create a mock strategy that fails during recovery
        mock_strategy = Mock(spec=RecoveryStrategy)
        mock_strategy.can_recover.return_value = asyncio.Future()
        mock_strategy.can_recover.return_value.set_result(True)
        mock_strategy.recover.return_value = asyncio.Future()
        mock_strategy.recover.return_value.set_exception(Exception("Strategy failed"))

        registry.register_strategy(mock_strategy)

        # Add a working fallback strategy
        fallback_strategy = FallbackStrategy(
            fallback_operation=lambda: "fallback_result"
        )
        registry.register_strategy(fallback_strategy)

        def failing_operation():
            raise Exception("Primary failed")

        # Should use fallback strategy after first strategy fails
        result = asyncio.run(
            registry.attempt_recovery(
                failing_operation,
                Exception("test"),
                {"operation": "test"},
            )
        )

        assert result == "fallback_result"

    def test_create_default_recovery_registry(self):
        """Test default recovery registry creation."""
        registry = create_default_recovery_registry()

        assert isinstance(registry, RecoveryStrategyRegistry)
        assert len(registry.strategies) == 2
        assert isinstance(registry.strategies[0], RetryStrategy)
        assert isinstance(registry.strategies[1], CircuitBreakerStrategy)


class TestRecoveryStrategyIntegration:
    """Test recovery strategy integration scenarios."""

    def test_end_to_end_recovery_flow(self):
        """Test complete end-to-end recovery flow."""
        registry = create_default_recovery_registry()

        # Test recovery from temporary infrastructure failure
        call_count = 0

        def unstable_operation():
            nonlocal call_count
            call_count += 1
            if call_count < 3:
                raise InfrastructureError("Service temporarily unavailable")
            return "operation_success"

        result = asyncio.run(
            registry.attempt_recovery(
                unstable_operation,
                InfrastructureError("test"),
                {"operation": "database_query"},
            )
        )

        assert result == "operation_success"
        assert call_count == 3

    def test_recovery_with_circuit_breaker(self):
        """Test recovery with circuit breaker behavior."""
        registry = RecoveryStrategyRegistry()

        # Add circuit breaker for infrastructure errors
        circuit_breaker = CircuitBreakerStrategy(
            failure_threshold=2,
            recovery_timeout=0.1,
            expected_exception=InfrastructureError,
        )
        registry.register_strategy(circuit_breaker)

        def failing_operation():
            raise InfrastructureError("Service down")

        # First failure
        with pytest.raises(InfrastructureError):
            asyncio.run(
                registry.attempt_recovery(
                    failing_operation,
                    InfrastructureError("test"),
                    {},
                )
            )

        # Second failure - should open circuit
        with pytest.raises(InfrastructureError):
            asyncio.run(
                registry.attempt_recovery(
                    failing_operation,
                    InfrastructureError("test"),
                    {},
                )
            )

        # Third call should be rejected due to open circuit
        with pytest.raises(PynamolyError, match="Circuit breaker is open"):
            asyncio.run(
                registry.attempt_recovery(
                    failing_operation,
                    InfrastructureError("test"),
                    {},
                )
            )

    def test_recovery_with_fallback_chain(self):
        """Test recovery with fallback chain."""
        registry = RecoveryStrategyRegistry()

        # Add retry strategy first
        retry_strategy = RetryStrategy(max_retries=2, base_delay=0.01)
        registry.register_strategy(retry_strategy)

        # Add fallback strategy second
        fallback_strategy = FallbackStrategy(
            fallback_operation=lambda: "fallback_result"
        )
        registry.register_strategy(fallback_strategy)

        def always_failing_operation():
            raise ValidationError("Always fails")

        # Retry strategy can't recover from ValidationError, so fallback should be used
        result = asyncio.run(
            registry.attempt_recovery(
                always_failing_operation,
                ValidationError("test"),
                {},
            )
        )

        assert result == "fallback_result"

    def test_recovery_performance_under_load(self):
        """Test recovery performance under load."""
        registry = RecoveryStrategyRegistry()

        # Add fast retry strategy
        retry_strategy = RetryStrategy(max_retries=3, base_delay=0.001)
        registry.register_strategy(retry_strategy)

        results = []

        async def load_test():
            call_count = 0

            def unstable_operation():
                nonlocal call_count
                call_count += 1
                if call_count < 2:
                    raise ConnectionError("Temporary failure")
                return f"success_{call_count}"

            result = await registry.attempt_recovery(
                unstable_operation,
                ConnectionError("test"),
                {},
            )
            results.append(result)

        # Run multiple concurrent recovery attempts
        start_time = time.time()
        asyncio.run(asyncio.gather(*[load_test() for _ in range(10)]))
        end_time = time.time()

        # Should complete within reasonable time
        assert end_time - start_time < 2.0
        assert len(results) == 10
        assert all("success_" in result for result in results)

    def test_recovery_error_handling_robustness(self):
        """Test recovery robustness under various error conditions."""
        registry = RecoveryStrategyRegistry()

        # Add strategies
        retry_strategy = RetryStrategy(max_retries=1, base_delay=0.01)
        registry.register_strategy(retry_strategy)

        fallback_strategy = FallbackStrategy(
            fallback_operation=lambda: "fallback_result"
        )
        registry.register_strategy(fallback_strategy)

        # Test with None operation
        with pytest.raises(Exception):
            asyncio.run(
                registry.attempt_recovery(
                    None,
                    Exception("test"),
                    {},
                )
            )

        # Test with operation that returns None
        def none_operation():
            return None

        result = asyncio.run(
            registry.attempt_recovery(
                none_operation,
                Exception("test"),
                {},
            )
        )

        assert result is None

        # Test with complex error context
        def complex_operation():
            raise Exception("Complex error")

        complex_context = {
            "user_id": "user123",
            "operation": "data_processing",
            "nested_data": {"key": "value", "list": [1, 2, 3]},
        }

        result = asyncio.run(
            registry.attempt_recovery(
                complex_operation,
                Exception("test"),
                complex_context,
            )
        )

        assert result == "fallback_result"

    def test_recovery_with_custom_strategy(self):
        """Test recovery with custom recovery strategy."""

        class CustomRecoveryStrategy(RecoveryStrategy):
            async def can_recover(self, error: Exception, context: dict) -> bool:
                return isinstance(error, ValueError)

            async def recover(self, operation, error: Exception, context: dict):
                # Custom recovery logic
                return "custom_recovery_result"

        registry = RecoveryStrategyRegistry()
        registry.register_strategy(CustomRecoveryStrategy())

        def failing_operation():
            raise ValueError("Custom error")

        result = asyncio.run(
            registry.attempt_recovery(
                failing_operation,
                ValueError("test"),
                {},
            )
        )

        assert result == "custom_recovery_result"

    def test_recovery_logging(self):
        """Test that recovery attempts are properly logged."""
        registry = RecoveryStrategyRegistry()

        # Mock the logger
        with patch.object(registry, "logger") as mock_logger:
            retry_strategy = RetryStrategy(max_retries=2, base_delay=0.01)
            registry.register_strategy(retry_strategy)

            call_count = 0

            def failing_operation():
                nonlocal call_count
                call_count += 1
                if call_count < 2:
                    raise ConnectionError("Connection failed")
                return "success"

            result = asyncio.run(
                registry.attempt_recovery(
                    failing_operation,
                    ConnectionError("test"),
                    {},
                )
            )

            assert result == "success"

            # Check that recovery attempt was logged
            mock_logger.info.assert_called_once()
            log_message = mock_logger.info.call_args[0][0]
            assert "Attempting recovery with RetryStrategy" in log_message
