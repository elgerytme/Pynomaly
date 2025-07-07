"""
Enhanced Error Handling Testing Suite
Comprehensive tests for error handling, recovery, and resilience.
"""

import time
from unittest.mock import patch

import numpy as np
import pytest

from pynomaly.domain.exceptions import (
    AdapterError,
    DataError,
    DetectorError,
    ModelError,
    NetworkError,
    ValidationError,
)
from pynomaly.infrastructure.error_handling.circuit_breaker import CircuitBreaker
from pynomaly.infrastructure.error_handling.error_handler import ErrorHandler
from pynomaly.infrastructure.error_handling.fallback_strategy import FallbackStrategy
from pynomaly.infrastructure.error_handling.retry_policy import RetryPolicy


class TestErrorHandler:
    """Test suite for centralized error handling."""

    @pytest.fixture
    def error_handler(self):
        """Create error handler instance."""
        return ErrorHandler(
            enable_logging=True, enable_metrics=True, enable_notifications=True
        )

    def test_error_classification(self, error_handler):
        """Test error classification by type and severity."""
        errors = [
            ValueError("Invalid parameter"),
            FileNotFoundError("Model file not found"),
            MemoryError("Out of memory"),
            KeyboardInterrupt("User interrupted"),
            RuntimeError("CUDA error"),
        ]

        for error in errors:
            classification = error_handler.classify_error(error)

            assert "type" in classification
            assert "severity" in classification
            assert "category" in classification
            assert classification["severity"] in ["low", "medium", "high", "critical"]

    def test_error_context_capture(self, error_handler):
        """Test error context capture and enrichment."""
        try:
            # Simulate error with context
            data = np.random.randn(100, 5)
            model_id = "test_model_123"

            # Force an error
            raise DetectorError("Model prediction failed")

        except DetectorError as e:
            context = error_handler.capture_context(
                error=e, data_shape=data.shape, model_id=model_id, operation="predict"
            )

            assert context["error_type"] == "DetectorError"
            assert context["data_shape"] == (100, 5)
            assert context["model_id"] == "test_model_123"
            assert context["operation"] == "predict"
            assert "timestamp" in context
            assert "stack_trace" in context

    def test_error_aggregation_and_patterns(self, error_handler):
        """Test error aggregation and pattern detection."""
        # Simulate multiple similar errors
        errors = [
            DetectorError("Model prediction failed for dataset A"),
            DetectorError("Model prediction failed for dataset B"),
            DetectorError("Model prediction failed for dataset C"),
            AdapterError("PyTorch adapter failed"),
            AdapterError("TensorFlow adapter failed"),
        ]

        for error in errors:
            error_handler.record_error(error)

        patterns = error_handler.analyze_error_patterns(time_window=3600)  # 1 hour

        assert len(patterns) >= 2  # Should detect at least 2 patterns
        assert any(pattern["error_type"] == "DetectorError" for pattern in patterns)
        assert any(pattern["count"] >= 3 for pattern in patterns)

    def test_error_notification_system(self, error_handler):
        """Test error notification and alerting."""
        with (
            patch(
                "pynomaly.infrastructure.notifications.email_notifier.send_email"
            ) as mock_email,
            patch(
                "pynomaly.infrastructure.notifications.slack_notifier.send_message"
            ) as mock_slack,
        ):
            critical_error = ModelError("Critical model corruption detected")

            error_handler.handle_error(
                error=critical_error, severity="critical", notify=True
            )

            # Should send notifications for critical errors
            mock_email.assert_called()
            mock_slack.assert_called()

    def test_error_recovery_strategies(self, error_handler):
        """Test automatic error recovery strategies."""

        def failing_operation():
            raise MemoryError("Out of memory")

        def fallback_operation():
            return "fallback_result"

        with patch.object(error_handler, "clear_memory_cache") as mock_clear:
            result = error_handler.handle_with_recovery(
                operation=failing_operation,
                fallback=fallback_operation,
                recovery_strategies=["clear_memory", "reduce_batch_size"],
            )

            assert result == "fallback_result"
            mock_clear.assert_called()

    def test_error_rate_limiting(self, error_handler):
        """Test error rate limiting to prevent spam."""
        error_handler.enable_rate_limiting = True
        error_handler.max_errors_per_minute = 5

        # Generate many errors quickly
        for i in range(10):
            error_handler.record_error(ValidationError(f"Validation error {i}"))

        # Should limit error processing after threshold
        assert error_handler.get_error_count(last_minutes=1) <= 5

    def test_structured_error_logging(self, error_handler):
        """Test structured error logging."""
        with patch("logging.Logger.error") as mock_log:
            error = DataError("Invalid data format detected")
            context = {
                "user_id": "user123",
                "session_id": "session456",
                "request_id": "req789",
            }

            error_handler.log_structured_error(error, context)

            # Verify structured logging was called
            mock_log.assert_called()
            call_args = mock_log.call_args[1]
            assert "extra" in call_args
            assert call_args["extra"]["user_id"] == "user123"


class TestCircuitBreaker:
    """Test suite for circuit breaker pattern implementation."""

    @pytest.fixture
    def circuit_breaker(self):
        """Create circuit breaker instance."""
        return CircuitBreaker(
            failure_threshold=3, timeout=60, expected_exception=Exception
        )

    def test_circuit_breaker_closed_state(self, circuit_breaker):
        """Test circuit breaker in closed state (normal operation)."""

        def successful_operation():
            return "success"

        # Should work normally when closed
        result = circuit_breaker.call(successful_operation)

        assert result == "success"
        assert circuit_breaker.state == "closed"

    def test_circuit_breaker_open_state(self, circuit_breaker):
        """Test circuit breaker opening after failures."""

        def failing_operation():
            raise RuntimeError("Operation failed")

        # Cause enough failures to open the circuit
        for _ in range(3):
            with pytest.raises(RuntimeError):
                circuit_breaker.call(failing_operation)

        # Circuit should now be open
        assert circuit_breaker.state == "open"

        # Further calls should be rejected immediately
        with pytest.raises(Exception):  # CircuitBreakerOpenError
            circuit_breaker.call(failing_operation)

    def test_circuit_breaker_half_open_state(self, circuit_breaker):
        """Test circuit breaker half-open state (recovery testing)."""

        def failing_then_succeeding_operation():
            if hasattr(failing_then_succeeding_operation, "call_count"):
                failing_then_succeeding_operation.call_count += 1
            else:
                failing_then_succeeding_operation.call_count = 1

            if failing_then_succeeding_operation.call_count <= 3:
                raise RuntimeError("Still failing")
            return "recovered"

        # Open the circuit
        for _ in range(3):
            with pytest.raises(RuntimeError):
                circuit_breaker.call(failing_then_succeeding_operation)

        # Simulate timeout passage
        circuit_breaker._last_failure_time = time.time() - 61

        # Next call should put it in half-open state
        result = circuit_breaker.call(failing_then_succeeding_operation)

        assert result == "recovered"
        assert circuit_breaker.state == "closed"  # Should close after success

    def test_circuit_breaker_with_fallback(self, circuit_breaker):
        """Test circuit breaker with fallback mechanism."""

        def failing_primary():
            raise NetworkError("Network unavailable")

        def fallback_operation():
            return "fallback_result"

        circuit_breaker.set_fallback(fallback_operation)

        # After failures, should use fallback
        for _ in range(3):
            try:
                circuit_breaker.call(failing_primary)
            except NetworkError:
                pass

        # Circuit is now open, should use fallback
        result = circuit_breaker.call_with_fallback(failing_primary)
        assert result == "fallback_result"

    def test_circuit_breaker_metrics(self, circuit_breaker):
        """Test circuit breaker metrics collection."""

        def mixed_operation():
            if hasattr(mixed_operation, "call_count"):
                mixed_operation.call_count += 1
            else:
                mixed_operation.call_count = 1

            if mixed_operation.call_count % 2 == 0:
                raise RuntimeError("Intermittent failure")
            return "success"

        # Make several calls
        for _ in range(6):
            try:
                circuit_breaker.call(mixed_operation)
            except RuntimeError:
                pass

        metrics = circuit_breaker.get_metrics()

        assert metrics["total_calls"] == 6
        assert metrics["successful_calls"] == 3
        assert metrics["failed_calls"] == 3
        assert abs(metrics["success_rate"] - 0.5) < 0.1


class TestRetryPolicy:
    """Test suite for retry policy implementation."""

    @pytest.fixture
    def retry_policy(self):
        """Create retry policy instance."""
        return RetryPolicy(
            max_attempts=3,
            base_delay=1.0,
            max_delay=10.0,
            backoff_strategy="exponential",
            jitter=True,
        )

    def test_exponential_backoff_retry(self, retry_policy):
        """Test exponential backoff retry strategy."""
        attempt_times = []

        def failing_operation():
            attempt_times.append(time.time())
            if len(attempt_times) < 3:
                raise NetworkError("Network timeout")
            return "success"

        with patch("time.sleep") as mock_sleep:
            result = retry_policy.execute(failing_operation)

            assert result == "success"
            assert len(attempt_times) == 3
            assert mock_sleep.call_count == 2  # 2 retries, so 2 sleeps

    def test_linear_backoff_retry(self, retry_policy):
        """Test linear backoff retry strategy."""
        retry_policy.backoff_strategy = "linear"

        def failing_operation():
            raise ValueError("Persistent error")

        with patch("time.sleep") as mock_sleep:
            with pytest.raises(ValueError):
                retry_policy.execute(failing_operation)

            # Should have made 3 attempts (original + 2 retries)
            assert mock_sleep.call_count == 2

    def test_conditional_retry(self, retry_policy):
        """Test conditional retry based on exception type."""
        retry_policy.retryable_exceptions = [NetworkError, TimeoutError]

        def network_error_operation():
            raise NetworkError("Retryable network error")

        def validation_error_operation():
            raise ValidationError("Non-retryable validation error")

        with patch("time.sleep"):
            # Should retry network errors
            with pytest.raises(NetworkError):
                retry_policy.execute(network_error_operation)

            # Should not retry validation errors
            with pytest.raises(ValidationError):
                retry_policy.execute(validation_error_operation)

    def test_retry_with_jitter(self, retry_policy):
        """Test retry with jitter to avoid thundering herd."""
        retry_policy.jitter = True

        def failing_operation():
            raise TimeoutError("Timeout")

        delays = []

        def mock_sleep(delay):
            delays.append(delay)

        with patch("time.sleep", side_effect=mock_sleep):
            with pytest.raises(TimeoutError):
                retry_policy.execute(failing_operation)

        # Delays should have some randomness (jitter)
        assert len(delays) == 2
        assert all(delay > 0 for delay in delays)

    def test_retry_callback_hooks(self, retry_policy):
        """Test retry callback hooks."""
        on_retry_calls = []
        on_failure_calls = []

        def on_retry_callback(attempt, exception):
            on_retry_calls.append((attempt, exception))

        def on_final_failure_callback(exception):
            on_failure_calls.append(exception)

        retry_policy.on_retry = on_retry_callback
        retry_policy.on_final_failure = on_final_failure_callback

        def failing_operation():
            raise RuntimeError("Persistent failure")

        with patch("time.sleep"):
            with pytest.raises(RuntimeError):
                retry_policy.execute(failing_operation)

        assert len(on_retry_calls) == 2  # 2 retries
        assert len(on_failure_calls) == 1  # 1 final failure


class TestFallbackStrategy:
    """Test suite for fallback strategy implementation."""

    @pytest.fixture
    def fallback_strategy(self):
        """Create fallback strategy instance."""
        return FallbackStrategy()

    def test_model_fallback_strategy(self, fallback_strategy):
        """Test fallback to alternative models."""

        def primary_model_predict(data):
            raise ModelError("Primary model failed")

        def secondary_model_predict(data):
            return np.random.random(len(data))

        def simple_model_predict(data):
            return np.zeros(len(data))  # Simple baseline

        fallback_strategy.add_fallback("secondary_model", secondary_model_predict)
        fallback_strategy.add_fallback("simple_baseline", simple_model_predict)

        data = np.random.randn(100, 5)

        result = fallback_strategy.execute_with_fallback(
            primary_operation=lambda: primary_model_predict(data),
            context={"data": data},
        )

        assert result is not None
        assert len(result) == len(data)

    def test_data_source_fallback(self, fallback_strategy):
        """Test fallback to alternative data sources."""

        def primary_data_source():
            raise NetworkError("Primary database unavailable")

        def cache_data_source():
            return {"cached_data": [1, 2, 3, 4, 5]}

        def default_data_source():
            return {"default_data": [0, 0, 0, 0, 0]}

        fallback_strategy.add_fallback("cache", cache_data_source)
        fallback_strategy.add_fallback("default", default_data_source)

        result = fallback_strategy.execute_with_fallback(
            primary_operation=primary_data_source
        )

        assert "cached_data" in result
        assert result["cached_data"] == [1, 2, 3, 4, 5]

    def test_computation_fallback(self, fallback_strategy):
        """Test fallback to simpler computation methods."""

        def gpu_computation(data):
            raise RuntimeError("CUDA out of memory")

        def cpu_computation(data):
            return np.mean(data, axis=1)

        def approximation_computation(data):
            # Very simple approximation
            return np.random.random(data.shape[0])

        fallback_strategy.add_fallback("cpu", cpu_computation)
        fallback_strategy.add_fallback("approximation", approximation_computation)

        data = np.random.randn(1000, 50)

        result = fallback_strategy.execute_with_fallback(
            primary_operation=lambda: gpu_computation(data), context={"data": data}
        )

        assert result is not None
        assert len(result) == len(data)

    def test_cascading_fallbacks(self, fallback_strategy):
        """Test cascading through multiple fallback levels."""

        def level_1_operation():
            raise NetworkError("Level 1 failed")

        def level_2_operation():
            raise TimeoutError("Level 2 failed")

        def level_3_operation():
            return "final_fallback_result"

        fallback_strategy.add_fallback("level_2", level_2_operation)
        fallback_strategy.add_fallback("level_3", level_3_operation)

        result = fallback_strategy.execute_with_fallback(
            primary_operation=level_1_operation
        )

        assert result == "final_fallback_result"

    def test_fallback_metrics_tracking(self, fallback_strategy):
        """Test fallback usage metrics tracking."""

        def primary_fails():
            raise RuntimeError("Primary failed")

        def fallback_succeeds():
            return "fallback_success"

        fallback_strategy.add_fallback("backup", fallback_succeeds)

        # Execute multiple times
        for _ in range(5):
            fallback_strategy.execute_with_fallback(primary_operation=primary_fails)

        metrics = fallback_strategy.get_metrics()

        assert metrics["total_executions"] == 5
        assert metrics["primary_failures"] == 5
        assert metrics["fallback_usage"]["backup"] == 5


class TestErrorRecovery:
    """Test suite for error recovery mechanisms."""

    def test_memory_error_recovery(self):
        """Test recovery from memory errors."""
        from pynomaly.infrastructure.error_handling.recovery import MemoryRecovery

        recovery = MemoryRecovery()

        def memory_intensive_operation():
            # Simulate memory allocation
            if not hasattr(memory_intensive_operation, "attempt"):
                memory_intensive_operation.attempt = 0

            memory_intensive_operation.attempt += 1

            if memory_intensive_operation.attempt == 1:
                raise MemoryError("Out of memory")

            return "success_after_recovery"

        with (
            patch("gc.collect") as mock_gc,
            patch("torch.cuda.empty_cache") as mock_empty_cache,
        ):
            result = recovery.recover_from_memory_error(memory_intensive_operation)

            assert result == "success_after_recovery"
            mock_gc.assert_called()
            mock_empty_cache.assert_called()

    def test_model_corruption_recovery(self):
        """Test recovery from model corruption."""
        from pynomaly.infrastructure.error_handling.recovery import ModelRecovery

        recovery = ModelRecovery()

        def load_corrupted_model():
            raise ModelError("Model file corrupted")

        def load_backup_model():
            return "backup_model_loaded"

        with patch.object(recovery, "load_backup_model", side_effect=load_backup_model):
            result = recovery.recover_from_corruption(
                primary_loader=load_corrupted_model, model_id="test_model"
            )

            assert result == "backup_model_loaded"

    def test_network_error_recovery(self):
        """Test recovery from network errors."""
        from pynomaly.infrastructure.error_handling.recovery import NetworkRecovery

        recovery = NetworkRecovery()

        def unreliable_network_call():
            if not hasattr(unreliable_network_call, "attempts"):
                unreliable_network_call.attempts = 0

            unreliable_network_call.attempts += 1

            if unreliable_network_call.attempts <= 2:
                raise NetworkError("Connection failed")

            return "network_call_successful"

        result = recovery.recover_from_network_error(
            operation=unreliable_network_call, max_retries=3, backoff_factor=1.5
        )

        assert result == "network_call_successful"

    def test_concurrent_error_handling(self):
        """Test error handling in concurrent operations."""
        from pynomaly.infrastructure.error_handling.concurrent import (
            ConcurrentErrorHandler,
        )

        handler = ConcurrentErrorHandler(max_workers=4)

        def mixed_operations(task_id):
            if task_id % 3 == 0:
                raise RuntimeError(f"Task {task_id} failed")
            return f"Task {task_id} completed"

        tasks = list(range(10))

        results = handler.execute_concurrent(
            operation=mixed_operations, tasks=tasks, continue_on_error=True
        )

        # Should have results for successful tasks
        successful_results = [r for r in results if not isinstance(r, Exception)]
        failed_results = [r for r in results if isinstance(r, Exception)]

        assert len(successful_results) > 0
        assert len(failed_results) > 0
        assert len(results) == len(tasks)


class TestErrorHandlingIntegration:
    """Integration tests for comprehensive error handling."""

    def test_end_to_end_error_handling_pipeline(self):
        """Test complete error handling pipeline."""
        from pynomaly.infrastructure.adapters.pytorch_adapter import PyTorchAdapter

        PyTorchAdapter()
        error_handler = ErrorHandler()
        circuit_breaker = CircuitBreaker(failure_threshold=2, timeout=60)
        retry_policy = RetryPolicy(max_attempts=3)

        def unreliable_training_operation():
            if not hasattr(unreliable_training_operation, "attempts"):
                unreliable_training_operation.attempts = 0

            unreliable_training_operation.attempts += 1

            if unreliable_training_operation.attempts <= 2:
                raise RuntimeError("Training failed")

            return "training_successful"

        # Integrate all error handling mechanisms
        result = error_handler.handle_with_recovery(
            operation=lambda: circuit_breaker.call(
                lambda: retry_policy.execute(unreliable_training_operation)
            ),
            fallback=lambda: "fallback_training_result",
        )

        assert result in ["training_successful", "fallback_training_result"]

    def test_error_handling_with_monitoring(self):
        """Test error handling integration with monitoring."""
        from pynomaly.infrastructure.monitoring.error_monitor import ErrorMonitor

        monitor = ErrorMonitor()
        error_handler = ErrorHandler()

        errors_to_simulate = [
            DetectorError("Detection failed"),
            ModelError("Model loading failed"),
            DataError("Data validation failed"),
            NetworkError("API call failed"),
        ]

        with patch.object(monitor, "record_error") as mock_record:
            for error in errors_to_simulate:
                error_handler.handle_error(error, notify_monitor=True)

            # Should record all errors in monitoring
            assert mock_record.call_count == len(errors_to_simulate)

    def test_distributed_error_handling(self):
        """Test error handling in distributed environments."""
        from pynomaly.infrastructure.error_handling.distributed import (
            DistributedErrorHandler,
        )

        handler = DistributedErrorHandler(node_id="node_1")

        # Simulate distributed operation failure
        def distributed_operation():
            raise NetworkError("Node communication failed")

        with patch.object(handler, "notify_other_nodes") as mock_notify:
            result = handler.handle_distributed_error(
                operation=distributed_operation, fallback_nodes=["node_2", "node_3"]
            )

            # Should notify other nodes and attempt fallback
            mock_notify.assert_called()
            assert result is not None
