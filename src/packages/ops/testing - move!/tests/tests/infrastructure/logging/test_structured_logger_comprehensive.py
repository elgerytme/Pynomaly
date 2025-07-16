"""Comprehensive tests for structured logging infrastructure."""

import tempfile
import threading
import time
from pathlib import Path
from unittest.mock import patch

from monorepo.infrastructure.logging.structured_logger import (
    LogContext,
    LogLevel,
    PerformanceLogger,
    StructuredLogger,
    configure_logging,
    get_logger,
    performance_logger,
)


class TestLogContext:
    """Test log context functionality."""

    def test_log_context_creation(self):
        """Test basic log context creation."""
        context = LogContext()

        # Check default values
        assert context.service_name == "monorepo"
        assert context.service_version == "1.0.0"
        assert context.environment == "development"
        assert context.correlation_id is not None
        assert context.instance_id is not None
        assert len(context.instance_id) == 8

    def test_log_context_custom_values(self):
        """Test log context with custom values."""
        context = LogContext(
            service_name="test-service",
            service_version="2.0.0",
            environment="production",
            user_id="user123",
            session_id="session456",
            detector_id="detector789",
        )

        assert context.service_name == "test-service"
        assert context.service_version == "2.0.0"
        assert context.environment == "production"
        assert context.user_id == "user123"
        assert context.session_id == "session456"
        assert context.detector_id == "detector789"

    def test_log_context_to_dict(self):
        """Test log context to dictionary conversion."""
        context = LogContext(
            user_id="user123",
            session_id=None,  # Should be excluded
            detector_id="detector456",
            custom_fields={"key1": "value1", "key2": "value2"},
        )

        context_dict = context.to_dict()

        # Check included fields
        assert context_dict["user_id"] == "user123"
        assert context_dict["detector_id"] == "detector456"
        assert context_dict["key1"] == "value1"
        assert context_dict["key2"] == "value2"

        # Check excluded fields
        assert "session_id" not in context_dict
        assert "custom_fields" not in context_dict

    def test_log_context_add_custom(self):
        """Test adding custom fields to log context."""
        context = LogContext()

        # Add custom field
        new_context = context.add_custom("experiment_id", "exp123")

        # Original context should be unchanged
        assert "experiment_id" not in context.custom_fields

        # New context should have the custom field
        assert new_context.custom_fields["experiment_id"] == "exp123"

        # Check dictionary representation
        context_dict = new_context.to_dict()
        assert context_dict["experiment_id"] == "exp123"

    def test_log_context_with_operation(self):
        """Test adding operation to log context."""
        context = LogContext()

        new_context = context.with_operation("data_processing")

        assert context.operation is None
        assert new_context.operation == "data_processing"

    def test_log_context_with_performance(self):
        """Test adding performance metrics to log context."""
        context = LogContext()

        new_context = context.with_performance(
            duration_ms=150.5,
            memory_usage_mb=256.0,
            cpu_usage_percent=75.5,
        )

        assert context.duration_ms is None
        assert new_context.duration_ms == 150.5
        assert new_context.memory_usage_mb == 256.0
        assert new_context.cpu_usage_percent == 75.5

    def test_log_context_with_error(self):
        """Test adding error information to log context."""
        context = LogContext()

        try:
            raise ValueError("Test error")
        except ValueError as e:
            new_context = context.with_error(e, "ERR_001")

        assert context.error_type is None
        assert new_context.error_type == "ValueError"
        assert new_context.error_code == "ERR_001"
        assert new_context.stack_trace is not None
        assert "ValueError" in new_context.stack_trace


class TestStructuredLogger:
    """Test structured logger functionality."""

    def test_logger_initialization(self):
        """Test logger initialization with default settings."""
        logger = StructuredLogger("test-logger")

        assert logger.name == "test-logger"
        assert logger.level == LogLevel.INFO
        assert logger.enable_console is True
        assert logger.enable_json is True
        assert logger.sanitize_sensitive_data is True
        assert logger.metrics["logs_written"] == 0

    def test_logger_initialization_custom_settings(self):
        """Test logger initialization with custom settings."""
        with tempfile.TemporaryDirectory() as temp_dir:
            output_path = Path(temp_dir) / "test.log"

            logger = StructuredLogger(
                "test-logger",
                level=LogLevel.WARNING,
                output_path=output_path,
                enable_console=False,
                enable_json=False,
                max_file_size_mb=50,
                backup_count=3,
                sanitize_sensitive_data=False,
            )

            assert logger.name == "test-logger"
            assert logger.level == LogLevel.WARNING
            assert logger.output_path == output_path
            assert logger.enable_console is False
            assert logger.enable_json is False
            assert logger.max_file_size_mb == 50
            assert logger.backup_count == 3
            assert logger.sanitize_sensitive_data is False

    def test_logger_context_management(self):
        """Test logger context management."""
        logger = StructuredLogger("test-logger")

        # Initially no context
        assert logger.get_current_context() is None

        # Set context
        context = LogContext(user_id="user123")
        logger.set_context(context)

        # Check context is set
        current_context = logger.get_current_context()
        assert current_context.user_id == "user123"

        # Clear context
        logger.clear_context()
        assert logger.get_current_context() is None

    def test_logger_basic_logging(self):
        """Test basic logging functionality."""
        logger = StructuredLogger("test-logger")

        # Test different log levels
        logger.debug("Debug message")
        logger.info("Info message")
        logger.warning("Warning message")
        logger.error("Error message")
        logger.critical("Critical message")

        # Check metrics
        metrics = logger.get_metrics()
        assert metrics["logs_written"] >= 4  # Info and above
        assert metrics["errors_logged"] == 2  # Error and critical
        assert metrics["warnings_logged"] == 1

    def test_logger_with_context(self):
        """Test logging with context."""
        logger = StructuredLogger("test-logger")

        # Set context
        context = LogContext(
            user_id="user123",
            detector_id="detector456",
            operation="test_operation",
        )
        logger.set_context(context)

        # Mock the internal logger to capture the log data
        with patch.object(logger, "_logger") as mock_logger:
            logger.info("Test message with context")

            # Check that the logger was called with context data
            mock_logger.info.assert_called_once()
            call_args = mock_logger.info.call_args
            assert "user_id" in str(call_args) or "user123" in str(call_args)

    def test_logger_error_with_exception(self):
        """Test error logging with exception."""
        logger = StructuredLogger("test-logger")

        try:
            raise ValueError("Test error")
        except ValueError as e:
            logger.error("Error occurred", error=e)

        # Check metrics
        metrics = logger.get_metrics()
        assert metrics["errors_logged"] == 1

    def test_logger_performance_logging(self):
        """Test performance logging."""
        logger = StructuredLogger("test-logger")

        logger.performance("database_query", 123.45, rows_processed=1000)

        # Check metrics
        metrics = logger.get_metrics()
        assert metrics["performance_logs"] == 1
        assert metrics["logs_written"] == 1

    def test_logger_audit_logging(self):
        """Test audit logging."""
        logger = StructuredLogger("test-logger")

        logger.audit(
            "user_login",
            user_id="user123",
            resource="dashboard",
            ip_address="192.168.1.1",
        )

        # Check that log was written
        metrics = logger.get_metrics()
        assert metrics["logs_written"] == 1

    def test_logger_security_logging(self):
        """Test security logging."""
        logger = StructuredLogger("test-logger")

        logger.security(
            "failed_login_attempt",
            severity="high",
            user_id="user123",
            ip_address="192.168.1.1",
        )

        # Check that log was written
        metrics = logger.get_metrics()
        assert metrics["logs_written"] == 1

    def test_logger_business_logging(self):
        """Test business logging."""
        logger = StructuredLogger("test-logger")

        logger.business(
            "anomaly_detected",
            metric_value=0.85,
            detector_id="detector123",
            dataset_id="dataset456",
        )

        # Check that log was written
        metrics = logger.get_metrics()
        assert metrics["logs_written"] == 1

    def test_logger_sensitive_data_sanitization(self):
        """Test sensitive data sanitization."""
        logger = StructuredLogger("test-logger", sanitize_sensitive_data=True)

        # Mock the internal logger to capture sanitized data
        with patch.object(logger, "_logger") as mock_logger:
            logger.info(
                "Login attempt",
                username="user123",
                password="secret123",
                api_key="key456",
                normal_field="normal_value",
            )

            # Check that sanitization occurred
            call_args = mock_logger.info.call_args
            call_str = str(call_args)
            assert "***REDACTED***" in call_str
            assert "secret123" not in call_str
            assert "key456" not in call_str
            assert "normal_value" in call_str

        # Check sanitization metrics
        metrics = logger.get_metrics()
        assert metrics["sanitized_fields"] >= 2  # password and api_key

    def test_logger_sanitization_disabled(self):
        """Test logging with sanitization disabled."""
        logger = StructuredLogger("test-logger", sanitize_sensitive_data=False)

        # Mock the internal logger to capture non-sanitized data
        with patch.object(logger, "_logger") as mock_logger:
            logger.info(
                "Login attempt",
                username="user123",
                password="secret123",
            )

            # Check that no sanitization occurred
            call_args = mock_logger.info.call_args
            call_str = str(call_args)
            assert "***REDACTED***" not in call_str

        # Check no sanitization metrics
        metrics = logger.get_metrics()
        assert metrics["sanitized_fields"] == 0

    def test_logger_nested_dict_sanitization(self):
        """Test sanitization of nested dictionaries."""
        logger = StructuredLogger("test-logger", sanitize_sensitive_data=True)

        # Mock the internal logger to capture sanitized data
        with patch.object(logger, "_logger") as mock_logger:
            logger.info(
                "User data",
                user_data={
                    "id": "user123",
                    "password": "secret123",
                    "profile": {
                        "name": "John Doe",
                        "secret": "hidden_value",
                    },
                },
            )

            # Check that nested sanitization occurred
            call_args = mock_logger.info.call_args
            call_str = str(call_args)
            assert "***REDACTED***" in call_str
            assert "secret123" not in call_str
            assert "hidden_value" not in call_str
            assert "John Doe" in call_str

    def test_logger_file_output(self):
        """Test logger file output."""
        with tempfile.TemporaryDirectory() as temp_dir:
            output_path = Path(temp_dir) / "test.log"

            logger = StructuredLogger(
                "test-logger",
                output_path=output_path,
                enable_console=False,
                enable_json=True,
            )

            # Log some messages
            logger.info("Test message 1")
            logger.error("Test error message")

            # Check file was created and has content
            assert output_path.exists()
            content = output_path.read_text()
            assert "Test message 1" in content
            assert "Test error message" in content

    def test_logger_log_level_filtering(self):
        """Test log level filtering."""
        logger = StructuredLogger("test-logger", level=LogLevel.WARNING)

        # Mock the internal logger to check what gets logged
        with patch.object(logger, "_logger") as mock_logger:
            logger.debug("Debug message")  # Should be filtered out
            logger.info("Info message")  # Should be filtered out
            logger.warning("Warning message")  # Should be logged
            logger.error("Error message")  # Should be logged

            # Check only warning and error were logged
            assert mock_logger.debug.call_count == 0
            assert mock_logger.info.call_count == 0
            assert mock_logger.warning.call_count == 1
            assert mock_logger.error.call_count == 1

    def test_logger_metrics(self):
        """Test logger metrics collection."""
        logger = StructuredLogger("test-logger")

        # Log various types of messages
        logger.info("Info message")
        logger.warning("Warning message")
        logger.error("Error message")
        logger.performance("test_operation", 100.0)
        logger.audit("test_action", user_id="user123")

        # Check metrics
        metrics = logger.get_metrics()
        assert metrics["logger_name"] == "test-logger"
        assert metrics["logs_written"] == 5
        assert metrics["errors_logged"] == 1
        assert metrics["warnings_logged"] == 1
        assert metrics["performance_logs"] == 1
        assert metrics["level"] == "INFO"

    def test_logger_concurrent_access(self):
        """Test logger thread safety."""
        logger = StructuredLogger("test-logger")
        results = []
        errors = []

        def log_worker(worker_id):
            try:
                context = LogContext(user_id=f"user_{worker_id}")
                logger.set_context(context)

                for i in range(10):
                    logger.info(f"Worker {worker_id} - Message {i}")
                    time.sleep(0.001)  # Small delay to encourage concurrency

                results.append(worker_id)
            except Exception as e:
                errors.append(e)

        # Start multiple threads
        threads = []
        for i in range(5):
            thread = threading.Thread(target=log_worker, args=(i,))
            threads.append(thread)
            thread.start()

        # Wait for all threads to complete
        for thread in threads:
            thread.join()

        # Check results
        assert len(errors) == 0
        assert len(results) == 5
        assert logger.get_metrics()["logs_written"] == 50


class TestPerformanceLogger:
    """Test performance logger functionality."""

    def test_performance_logger_basic(self):
        """Test basic performance logging."""
        logger = StructuredLogger("test-logger")

        with patch.object(logger, "performance") as mock_performance:
            with PerformanceLogger(logger, "test_operation"):
                time.sleep(0.1)  # Simulate work

            # Check that performance was logged
            mock_performance.assert_called_once()
            call_args = mock_performance.call_args
            assert call_args[0][0] == "test_operation"
            assert call_args[0][1] >= 100  # Should be at least 100ms

    def test_performance_logger_with_context(self):
        """Test performance logging with context."""
        logger = StructuredLogger("test-logger")
        context = LogContext(user_id="user123")

        with patch.object(logger, "set_context") as mock_set_context:
            with patch.object(logger, "clear_context") as mock_clear_context:
                with PerformanceLogger(logger, "test_operation", context=context):
                    time.sleep(0.01)

                # Check context was set and cleared
                mock_set_context.assert_called_once_with(context)
                mock_clear_context.assert_called_once()

    def test_performance_logger_with_exception(self):
        """Test performance logging when exception occurs."""
        logger = StructuredLogger("test-logger")

        with patch.object(logger, "error") as mock_error:
            try:
                with PerformanceLogger(logger, "test_operation"):
                    raise ValueError("Test error")
            except ValueError:
                pass

            # Check that error was logged
            mock_error.assert_called_once()
            call_args = mock_error.call_args
            assert "Operation failed: test_operation" in call_args[0][0]

    def test_performance_logger_minimum_duration(self):
        """Test performance logging with minimum duration filter."""
        logger = StructuredLogger("test-logger")

        with patch.object(logger, "performance") as mock_performance:
            with PerformanceLogger(logger, "test_operation", min_duration_ms=100):
                time.sleep(0.01)  # Less than minimum

            # Should not log because duration was too short
            mock_performance.assert_not_called()

    def test_performance_logger_decorator(self):
        """Test performance logging decorator."""
        logger = StructuredLogger("test-logger")

        @performance_logger("test_function", logger=logger)
        def test_function():
            time.sleep(0.01)
            return "result"

        with patch.object(logger, "performance") as mock_performance:
            result = test_function()

            # Check result and performance logging
            assert result == "result"
            mock_performance.assert_called_once()

    def test_performance_logger_decorator_with_exception(self):
        """Test performance logging decorator with exception."""
        logger = StructuredLogger("test-logger")

        @performance_logger("test_function", logger=logger)
        def test_function():
            raise ValueError("Test error")

        with patch.object(logger, "error") as mock_error:
            try:
                test_function()
            except ValueError:
                pass

            # Check that error was logged
            mock_error.assert_called_once()

    def test_performance_logger_decorator_auto_logger(self):
        """Test performance logging decorator with automatic logger creation."""

        @performance_logger("test_function")  # No logger provided
        def test_function():
            return "result"

        # Should not raise an exception
        result = test_function()
        assert result == "result"


class TestGlobalLoggerManagement:
    """Test global logger management functions."""

    def test_get_logger_singleton(self):
        """Test that get_logger returns the same instance."""
        logger1 = get_logger("test-logger")
        logger2 = get_logger("test-logger")

        assert logger1 is logger2
        assert logger1.name == "test-logger"

    def test_get_logger_different_names(self):
        """Test that get_logger returns different instances for different names."""
        logger1 = get_logger("logger1")
        logger2 = get_logger("logger2")

        assert logger1 is not logger2
        assert logger1.name == "logger1"
        assert logger2.name == "logger2"

    def test_get_logger_with_custom_settings(self):
        """Test get_logger with custom settings."""
        logger = get_logger(
            "test-logger",
            level=LogLevel.DEBUG,
            enable_console=False,
            enable_json=False,
        )

        assert logger.level == LogLevel.DEBUG
        assert logger.enable_console is False
        assert logger.enable_json is False

    def test_configure_logging(self):
        """Test global logging configuration."""
        with tempfile.TemporaryDirectory() as temp_dir:
            output_dir = Path(temp_dir)

            logger = configure_logging(
                level=LogLevel.WARNING,
                output_dir=output_dir,
                enable_console=False,
                enable_json=True,
                service_name="test-service",
                service_version="2.0.0",
                environment="production",
            )

            # Check logger configuration
            assert logger.level == LogLevel.WARNING
            assert logger.enable_console is False
            assert logger.enable_json is True
            assert logger.output_path == output_dir / "monorepo.log"

            # Check that context was set
            context = StructuredLogger.get_current_context()
            assert context.service_name == "test-service"
            assert context.service_version == "2.0.0"
            assert context.environment == "production"


class TestLoggerIntegration:
    """Test logger integration scenarios."""

    def test_end_to_end_logging_workflow(self):
        """Test complete end-to-end logging workflow."""
        with tempfile.TemporaryDirectory() as temp_dir:
            output_path = Path(temp_dir) / "test.log"

            # Configure logger
            logger = StructuredLogger(
                "integration-test",
                level=LogLevel.INFO,
                output_path=output_path,
                enable_console=False,
                enable_json=True,
            )

            # Set context
            context = LogContext(
                user_id="user123",
                detector_id="detector456",
                session_id="session789",
            )
            logger.set_context(context)

            # Log various types of messages
            logger.info("Starting anomaly detection", detector_id="detector456")
            logger.performance("data_preprocessing", 150.5, rows_processed=1000)
            logger.warning("High memory usage", memory_usage_mb=512.0)
            logger.audit("model_training", user_id="user123", action="start")
            logger.security("unusual_access_pattern", severity="medium")
            logger.business("anomaly_detected", metric_value=0.85)

            try:
                raise ValueError("Processing error")
            except ValueError as e:
                logger.error("Processing failed", error=e)

            # Check file output
            assert output_path.exists()
            content = output_path.read_text()

            # Verify content contains expected elements
            assert "Starting anomaly detection" in content
            assert "data_preprocessing" in content
            assert "High memory usage" in content
            assert "model_training" in content
            assert "unusual_access_pattern" in content
            assert "anomaly_detected" in content
            assert "Processing failed" in content
            assert "user123" in content
            assert "detector456" in content

    def test_logger_with_context_inheritance(self):
        """Test logger context inheritance across operations."""
        logger = StructuredLogger("test-logger")

        # Set base context
        base_context = LogContext(
            user_id="user123",
            session_id="session456",
        )
        logger.set_context(base_context)

        # Create derived context for specific operation
        operation_context = base_context.with_operation("anomaly_detection")
        logger.set_context(operation_context)

        # Mock the internal logger to verify context
        with patch.object(logger, "_logger") as mock_logger:
            logger.info("Processing data")

            # Check that context was included
            call_args = mock_logger.info.call_args
            call_str = str(call_args)
            assert "user123" in call_str
            assert "session456" in call_str
            assert "anomaly_detection" in call_str

    def test_logger_error_handling_robustness(self):
        """Test logger robustness with various error conditions."""
        logger = StructuredLogger("test-logger")

        # Test with None values
        logger.info("Message with None", value=None)

        # Test with complex objects
        class ComplexObject:
            def __str__(self):
                return "ComplexObject"

        logger.info("Message with complex object", obj=ComplexObject())

        # Test with circular references
        circular_dict = {"key": "value"}
        circular_dict["self"] = circular_dict

        # Should not crash
        logger.info("Message with circular reference", data=circular_dict)

        # Test with very large strings
        large_string = "a" * 1000
        logger.info("Message with large string", data=large_string)

        # Logger should still be functional
        metrics = logger.get_metrics()
        assert metrics["logs_written"] >= 4

    def test_logger_performance_under_load(self):
        """Test logger performance under high load."""
        logger = StructuredLogger("test-logger")

        start_time = time.time()

        # Log many messages quickly
        for i in range(1000):
            logger.info(f"Load test message {i}", iteration=i)

        end_time = time.time()
        duration = end_time - start_time

        # Should complete within reasonable time (less than 2 seconds)
        assert duration < 2.0

        # Check metrics
        metrics = logger.get_metrics()
        assert metrics["logs_written"] == 1000

    def test_logger_memory_usage(self):
        """Test logger memory usage doesn't grow unbounded."""
        logger = StructuredLogger("test-logger")

        # Log many messages with different contexts
        for i in range(100):
            context = LogContext(user_id=f"user_{i}")
            logger.set_context(context)
            logger.info(f"Message {i}", data=f"data_{i}")

        # Clear context to allow garbage collection
        logger.clear_context()

        # Logger should still be functional
        logger.info("Final message")

        metrics = logger.get_metrics()
        assert metrics["logs_written"] == 101
