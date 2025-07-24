"""Unit tests for infrastructure logging components."""

import pytest
import json
import logging
import time
import asyncio
from unittest.mock import Mock, patch, MagicMock, call
from datetime import datetime
from typing import Dict, Any

from anomaly_detection.infrastructure.logging.structured_logger import (
    StructuredLogger, get_logger, configure_logging
)
from anomaly_detection.infrastructure.logging.log_decorator import (
    log_decorator, timing_decorator, async_log_decorator
)
from anomaly_detection.infrastructure.logging.error_handler import (
    ErrorHandler, ErrorCategory, BaseApplicationError,
    ValidationError, DataProcessingError, ModelOperationError,
    AlgorithmError, PersistenceError, ConfigurationError,
    SystemError, ExternalServiceError
)
from anomaly_detection.infrastructure.config.settings import LoggingSettings


class TestStructuredLogger:
    """Test cases for StructuredLogger class."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.settings = LoggingSettings(
            level="INFO",
            format="json",
            enable_structured_logging=True,
            enable_request_tracking=True,
            enable_performance_logging=True
        )
        self.logger = StructuredLogger("test_logger", self.settings)
    
    def test_logger_initialization(self):
        """Test logger initialization with settings."""
        assert self.logger.name == "test_logger"
        assert self.logger.settings == self.settings
        assert self.logger._context == {}
    
    @patch('structlog.get_logger')
    def test_info_logging(self, mock_get_logger):
        """Test info level logging."""
        mock_structlog = Mock()
        mock_get_logger.return_value = Mock()
        mock_get_logger.return_value.info = mock_structlog
        
        self.logger.info("Test message", key="value")
        
        mock_structlog.assert_called_once_with("Test message", key="value")
    
    @patch('structlog.get_logger')
    def test_error_logging(self, mock_get_logger):
        """Test error level logging."""
        mock_structlog = Mock()
        mock_get_logger.return_value = Mock()
        mock_get_logger.return_value.error = mock_structlog
        
        self.logger.error("Error message", error_code="E001")
        
        mock_structlog.assert_called_once_with("Error message", error_code="E001")
    
    @patch('structlog.get_logger')
    def test_warning_logging(self, mock_get_logger):
        """Test warning level logging."""
        mock_structlog = Mock()
        mock_get_logger.return_value = Mock()
        mock_get_logger.return_value.warning = mock_structlog
        
        self.logger.warning("Warning message", component="test")
        
        mock_structlog.assert_called_once_with("Warning message", component="test")
    
    @patch('structlog.get_logger')
    def test_debug_logging(self, mock_get_logger):
        """Test debug level logging."""
        mock_structlog = Mock()
        mock_get_logger.return_value = Mock()
        mock_get_logger.return_value.debug = mock_structlog
        
        self.logger.debug("Debug message", details="extra info")
        
        mock_structlog.assert_called_once_with("Debug message", details="extra info")
    
    def test_set_context(self):
        """Test setting logging context."""
        context_data = {"request_id": "req_123", "user_id": "user_456"}
        
        self.logger.set_context(context_data)
        
        assert self.logger._context == context_data
    
    def test_clear_context(self):
        """Test clearing logging context."""
        self.logger._context = {"key": "value"}
        
        self.logger.clear_context()
        
        assert self.logger._context == {}
    
    def test_get_context(self):
        """Test getting current logging context."""
        context_data = {"request_id": "req_123"}
        self.logger._context = context_data
        
        result = self.logger.get_context()
        
        assert result == context_data
    
    @patch('structlog.get_logger')
    def test_log_operation_start(self, mock_get_logger):
        """Test logging operation start."""
        mock_structlog = Mock()
        mock_get_logger.return_value = Mock()
        mock_get_logger.return_value.info = mock_structlog
        
        self.logger.log_operation_start("test_operation", param1="value1")
        
        mock_structlog.assert_called_once()
        call_args = mock_structlog.call_args[1]
        assert "Starting operation: test_operation" in mock_structlog.call_args[0]
        assert call_args["operation"] == "test_operation"
        assert call_args["param1"] == "value1"
    
    @patch('structlog.get_logger')
    @patch('time.time')
    def test_log_operation_end(self, mock_time, mock_get_logger):
        """Test logging operation end with timing."""
        mock_time.side_effect = [1000.0, 1002.5]  # 2.5 second duration
        mock_structlog = Mock()
        mock_get_logger.return_value = Mock()
        mock_get_logger.return_value.info = mock_structlog
        
        # Start operation
        self.logger.log_operation_start("test_operation")
        
        # End operation
        self.logger.log_operation_end("test_operation", result="success")
        
        # Check that end was logged with duration
        assert mock_structlog.call_count == 2
        end_call = mock_structlog.call_args_list[1]
        assert "Completed operation: test_operation" in end_call[0]
        assert end_call[1]["duration_ms"] == 2500.0
        assert end_call[1]["result"] == "success"
    
    @patch('structlog.get_logger')
    def test_log_metric(self, mock_get_logger):
        """Test metric logging."""
        mock_structlog = Mock()
        mock_get_logger.return_value = Mock()
        mock_get_logger.return_value.info = mock_structlog
        
        self.logger.log_metric("accuracy", 0.95, labels={"model": "test_model"})
        
        mock_structlog.assert_called_once()
        call_args = mock_structlog.call_args[1]
        assert call_args["metric_name"] == "accuracy"
        assert call_args["metric_value"] == 0.95
        assert call_args["labels"] == {"model": "test_model"}
    
    @patch('structlog.get_logger')
    def test_log_performance_with_slow_operation(self, mock_get_logger):
        """Test performance logging with slow operation detection."""
        # Configure slow threshold
        self.settings.slow_detection_threshold_ms = 1000.0
        mock_structlog = Mock()
        mock_get_logger.return_value = Mock()
        mock_get_logger.return_value.warning = mock_structlog
        
        # Log slow operation
        self.logger.log_performance("detection", 1500.0, {"samples": 1000})
        
        mock_structlog.assert_called_once()
        call_args = mock_structlog.call_args[1]
        assert call_args["operation"] == "detection"
        assert call_args["duration_ms"] == 1500.0
        assert call_args["is_slow"] is True
    
    @patch('structlog.get_logger')
    def test_log_data_quality(self, mock_get_logger):
        """Test data quality logging."""
        mock_structlog = Mock()
        mock_get_logger.return_value = Mock()
        mock_get_logger.return_value.info = mock_structlog
        
        quality_metrics = {
            "missing_values": 5,
            "outliers": 12,
            "total_samples": 1000
        }
        
        self.logger.log_data_quality(quality_metrics)
        
        mock_structlog.assert_called_once_with(
            "Data quality metrics",
            **quality_metrics
        )
    
    @patch('structlog.get_logger')
    def test_log_model_performance(self, mock_get_logger):
        """Test model performance logging."""
        mock_structlog = Mock()
        mock_get_logger.return_value = Mock()
        mock_get_logger.return_value.info = mock_structlog
        
        performance_metrics = {
            "precision": 0.85,
            "recall": 0.78,
            "f1_score": 0.81
        }
        
        self.logger.log_model_performance("test_model", performance_metrics)
        
        mock_structlog.assert_called_once()
        call_args = mock_structlog.call_args[1]
        assert call_args["model_id"] == "test_model"
        assert call_args["precision"] == 0.85
        assert call_args["recall"] == 0.78
        assert call_args["f1_score"] == 0.81
    
    def test_sensitive_data_sanitization(self):
        """Test that sensitive data is sanitized from logs."""
        sensitive_data = {
            "password": "secret123",
            "api_key": "key_abc123",
            "token": "token_xyz789",
            "secret": "secret_value",
            "normal_field": "normal_value"
        }
        
        sanitized = self.logger._sanitize_data(sensitive_data)
        
        assert sanitized["password"] == "[REDACTED]"
        assert sanitized["api_key"] == "[REDACTED]"
        assert sanitized["token"] == "[REDACTED]"
        assert sanitized["secret"] == "[REDACTED]"
        assert sanitized["normal_field"] == "normal_value"
    
    def test_context_manager_functionality(self):
        """Test logger context manager."""
        initial_context = {"key1": "value1"}
        additional_context = {"key2": "value2"}
        
        self.logger.set_context(initial_context)
        
        with self.logger.context(**additional_context):
            # Context should be merged
            context = self.logger.get_context()
            assert context["key1"] == "value1"
            assert context["key2"] == "value2"
        
        # Context should be restored after exiting
        context = self.logger.get_context()
        assert context == initial_context


class TestLoggerFactory:
    """Test cases for logger factory functions."""
    
    @patch('anomaly_detection.infrastructure.logging.structured_logger.configure_logging')
    @patch('anomaly_detection.infrastructure.config.settings.get_settings')
    def test_get_logger(self, mock_get_settings, mock_configure):
        """Test getting a logger instance."""
        mock_settings = Mock()
        mock_settings.logging = LoggingSettings()
        mock_get_settings.return_value = mock_settings
        
        logger = get_logger("test_module")
        
        assert isinstance(logger, StructuredLogger)
        assert logger.name == "test_module"
        mock_configure.assert_called_once()
    
    @patch('structlog.configure')
    @patch('logging.basicConfig')
    def test_configure_logging_json_format(self, mock_basic_config, mock_structlog_configure):
        """Test logging configuration with JSON format."""
        settings = LoggingSettings(
            level="DEBUG",
            format="json",
            enable_structured_logging=True
        )
        
        configure_logging(settings)
        
        mock_basic_config.assert_called_once()
        mock_structlog_configure.assert_called_once()
        
        # Check structlog configuration
        config_kwargs = mock_structlog_configure.call_args[1]
        assert "processors" in config_kwargs
        assert "logger_factory" in config_kwargs
    
    @patch('structlog.configure')
    @patch('logging.basicConfig')
    def test_configure_logging_console_format(self, mock_basic_config, mock_structlog_configure):
        """Test logging configuration with console format."""
        settings = LoggingSettings(
            level="INFO",
            format="console",
            enable_structured_logging=True
        )
        
        configure_logging(settings)
        
        mock_basic_config.assert_called_once()
        mock_structlog_configure.assert_called_once()
    
    @patch('logging.FileHandler')
    @patch('structlog.configure')
    @patch('logging.basicConfig')
    def test_configure_logging_with_file(self, mock_basic_config, mock_structlog_configure, mock_file_handler):
        """Test logging configuration with file output."""
        settings = LoggingSettings(
            level="INFO",
            file_enabled=True,
            file_path="/var/log/app.log"
        )
        
        configure_logging(settings)
        
        mock_file_handler.assert_called_once_with("/var/log/app.log")


class TestLogDecorator:
    """Test cases for log decorator functionality."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.mock_logger = Mock(spec=StructuredLogger)
    
    @patch('anomaly_detection.infrastructure.logging.log_decorator.get_logger')
    def test_log_decorator_success(self, mock_get_logger):
        """Test log decorator with successful function execution."""
        mock_get_logger.return_value = self.mock_logger
        
        @log_decorator
        def test_function(x, y):
            return x + y
        
        result = test_function(2, 3)
        
        assert result == 5
        self.mock_logger.log_operation_start.assert_called_once()
        self.mock_logger.log_operation_end.assert_called_once()
    
    @patch('anomaly_detection.infrastructure.logging.log_decorator.get_logger')
    def test_log_decorator_with_exception(self, mock_get_logger):
        """Test log decorator with function that raises exception."""
        mock_get_logger.return_value = self.mock_logger
        
        @log_decorator
        def failing_function():
            raise ValueError("Test error")
        
        with pytest.raises(ValueError, match="Test error"):
            failing_function()
        
        self.mock_logger.log_operation_start.assert_called_once()
        self.mock_logger.error.assert_called_once()
    
    @patch('anomaly_detection.infrastructure.logging.log_decorator.get_logger')
    @patch('time.time')
    def test_timing_decorator(self, mock_time, mock_get_logger):
        """Test timing decorator functionality."""
        mock_time.side_effect = [1000.0, 1002.0]  # 2 second duration
        mock_get_logger.return_value = self.mock_logger
        
        @timing_decorator
        def timed_function():
            return "result"
        
        result = timed_function()
        
        assert result == "result"
        self.mock_logger.log_performance.assert_called_once()
        
        # Check performance log call
        perf_call = self.mock_logger.log_performance.call_args
        assert perf_call[0][0] == "timed_function"  # Operation name
        assert perf_call[0][1] == 2000.0  # Duration in ms
    
    @patch('anomaly_detection.infrastructure.logging.log_decorator.get_logger')
    def test_timing_decorator_with_slow_threshold(self, mock_get_logger):
        """Test timing decorator with slow operation detection."""
        # Configure logger with slow threshold
        self.mock_logger.settings = LoggingSettings(slow_detection_threshold_ms=500.0)
        mock_get_logger.return_value = self.mock_logger
        
        @timing_decorator
        def slow_function():
            time.sleep(0.001)  # Small sleep to ensure some duration
            return "result"
        
        slow_function()
        
        self.mock_logger.log_performance.assert_called_once()
    
    @patch('anomaly_detection.infrastructure.logging.log_decorator.get_logger')
    def test_async_log_decorator(self, mock_get_logger):
        """Test async log decorator functionality."""
        mock_get_logger.return_value = self.mock_logger
        
        @async_log_decorator
        async def async_function(value):
            await asyncio.sleep(0.001)
            return value * 2
        
        # Run async function
        result = asyncio.run(async_function(5))
        
        assert result == 10
        self.mock_logger.log_operation_start.assert_called_once()
        self.mock_logger.log_operation_end.assert_called_once()
    
    @patch('anomaly_detection.infrastructure.logging.log_decorator.get_logger')
    def test_decorator_argument_sanitization(self, mock_get_logger):
        """Test that decorator sanitizes sensitive arguments."""
        mock_get_logger.return_value = self.mock_logger
        
        @log_decorator
        def function_with_sensitive_args(username, password, api_key):
            return f"User: {username}"
        
        function_with_sensitive_args("testuser", "secret123", "key_abc")
        
        # Check that start operation was called with sanitized args
        start_call = self.mock_logger.log_operation_start.call_args
        assert start_call[1]["password"] == "[REDACTED]"
        assert start_call[1]["api_key"] == "[REDACTED]"
        assert start_call[1]["username"] == "testuser"  # Not sensitive
    
    @patch('anomaly_detection.infrastructure.logging.log_decorator.get_logger')
    def test_decorator_with_custom_logger_name(self, mock_get_logger):
        """Test decorator with custom logger name."""
        mock_get_logger.return_value = self.mock_logger
        
        @log_decorator(logger_name="custom_logger")
        def test_function():
            return "result"
        
        test_function()
        
        mock_get_logger.assert_called_with("custom_logger")
    
    def test_sensitive_data_identification(self):
        """Test identification of sensitive data fields."""
        from anomaly_detection.infrastructure.logging.log_decorator import _is_sensitive_field
        
        assert _is_sensitive_field("password") is True
        assert _is_sensitive_field("api_key") is True
        assert _is_sensitive_field("secret") is True
        assert _is_sensitive_field("token") is True
        assert _is_sensitive_field("auth") is True
        assert _is_sensitive_field("credential") is True
        
        assert _is_sensitive_field("username") is False
        assert _is_sensitive_field("email") is False
        assert _is_sensitive_field("normal_field") is False


class TestErrorHandler:
    """Test cases for ErrorHandler class."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.mock_logger = Mock(spec=StructuredLogger)
        self.error_handler = ErrorHandler(self.mock_logger)
    
    def test_error_handler_initialization(self):
        """Test error handler initialization."""
        assert self.error_handler.logger == self.mock_logger
    
    def test_handle_validation_error(self):
        """Test handling validation errors."""
        error = ValidationError("Invalid input format", field="email")
        
        result = self.error_handler.handle_error(error, {"request_id": "req_123"})
        
        assert result["error_category"] == ErrorCategory.INPUT_VALIDATION.value
        assert result["error_type"] == "ValidationError"
        assert result["message"] == "Invalid input format"
        assert result["field"] == "email"
        
        self.mock_logger.error.assert_called_once()
    
    def test_handle_data_processing_error(self):
        """Test handling data processing errors."""
        error = DataProcessingError("Failed to process dataset", dataset_size=1000)
        
        result = self.error_handler.handle_error(error)
        
        assert result["error_category"] == ErrorCategory.DATA_PROCESSING.value
        assert result["error_type"] == "DataProcessingError"
        assert result["dataset_size"] == 1000
    
    def test_handle_model_operation_error(self):
        """Test handling model operation errors."""
        error = ModelOperationError("Model training failed", model_id="model_123")
        
        result = self.error_handler.handle_error(error)
        
        assert result["error_category"] == ErrorCategory.MODEL_OPERATION.value
        assert result["model_id"] == "model_123"
    
    def test_handle_algorithm_error(self):
        """Test handling algorithm errors."""
        error = AlgorithmError("Invalid algorithm parameters", algorithm="isolation_forest")
        
        result = self.error_handler.handle_error(error)
        
        assert result["error_category"] == ErrorCategory.ALGORITHM_ERROR.value
        assert result["algorithm"] == "isolation_forest"
    
    def test_handle_persistence_error(self):
        """Test handling persistence errors."""
        error = PersistenceError("Database connection failed", operation="save")
        
        result = self.error_handler.handle_error(error)
        
        assert result["error_category"] == ErrorCategory.PERSISTENCE.value
        assert result["operation"] == "save"
    
    def test_handle_configuration_error(self):
        """Test handling configuration errors."""
        error = ConfigurationError("Missing required setting", setting="database_url")
        
        result = self.error_handler.handle_error(error)
        
        assert result["error_category"] == ErrorCategory.CONFIGURATION.value
        assert result["setting"] == "database_url"
    
    def test_handle_system_error(self):
        """Test handling system errors."""
        error = SystemError("Out of memory", available_memory="2GB")
        
        result = self.error_handler.handle_error(error)
        
        assert result["error_category"] == ErrorCategory.SYSTEM.value
        assert result["available_memory"] == "2GB"
    
    def test_handle_external_service_error(self):
        """Test handling external service errors."""
        error = ExternalServiceError("API rate limit exceeded", service="data_api")
        
        result = self.error_handler.handle_error(error)
        
        assert result["error_category"] == ErrorCategory.EXTERNAL.value
        assert result["service"] == "data_api"
    
    def test_handle_generic_exception(self):
        """Test handling generic Python exceptions."""
        error = ValueError("Generic error message")
        
        result = self.error_handler.handle_error(error)
        
        assert result["error_category"] == ErrorCategory.SYSTEM.value
        assert result["error_type"] == "ValueError"
        assert result["message"] == "Generic error message"
    
    def test_error_classification(self):
        """Test automatic error classification."""
        # Test different error types
        validation_error = ValidationError("Validation failed")
        data_error = DataProcessingError("Data processing failed")
        
        val_result = self.error_handler.classify_error(validation_error)
        data_result = self.error_handler.classify_error(data_error)
        
        assert val_result == ErrorCategory.INPUT_VALIDATION
        assert data_result == ErrorCategory.DATA_PROCESSING
    
    def test_error_context_logging(self):
        """Test that error context is properly logged."""
        error = ValidationError("Test error")
        context = {
            "request_id": "req_123",
            "user_id": "user_456",
            "operation": "validate_input"
        }
        
        self.error_handler.handle_error(error, context)
        
        # Check that error was logged with context
        self.mock_logger.error.assert_called_once()
        log_call = self.mock_logger.error.call_args
        assert log_call[1]["request_id"] == "req_123"
        assert log_call[1]["user_id"] == "user_456"
        assert log_call[1]["operation"] == "validate_input"
    
    def test_api_error_response_generation(self):
        """Test generation of API error responses."""
        error = ValidationError("Invalid email format", field="email")
        
        response = self.error_handler.create_api_error_response(error, 400)
        
        assert response["status_code"] == 400
        assert response["error"]["type"] == "ValidationError"
        assert response["error"]["message"] == "Invalid email format"
        assert response["error"]["category"] == ErrorCategory.INPUT_VALIDATION.value
        assert "timestamp" in response
        assert "error_id" in response
    
    def test_error_recovery_suggestions(self):
        """Test error recovery suggestion generation."""
        validation_error = ValidationError("Invalid input")
        persistence_error = PersistenceError("Database error")
        
        val_suggestion = self.error_handler.get_recovery_suggestion(validation_error)
        persist_suggestion = self.error_handler.get_recovery_suggestion(persistence_error)
        
        assert "validate" in val_suggestion.lower()
        assert "retry" in persist_suggestion.lower() or "connection" in persist_suggestion.lower()
    
    def test_error_metrics_collection(self):
        """Test that error metrics are collected."""
        error = ValidationError("Test error")
        
        self.error_handler.handle_error(error)
        
        # Check that metrics were logged
        self.mock_logger.log_metric.assert_called_once()
        metric_call = self.mock_logger.log_metric.call_args
        assert metric_call[0][0] == "error_count"
        assert metric_call[0][1] == 1
        assert metric_call[1]["labels"]["error_category"] == ErrorCategory.INPUT_VALIDATION.value


class TestCustomExceptions:
    """Test cases for custom exception classes."""
    
    def test_base_application_error(self):
        """Test base application error."""
        error = BaseApplicationError("Base error", category=ErrorCategory.SYSTEM)
        
        assert str(error) == "Base error"
        assert error.category == ErrorCategory.SYSTEM
        assert error.details == {}
    
    def test_validation_error_with_details(self):
        """Test validation error with additional details."""
        error = ValidationError(
            "Validation failed",
            field="email",
            expected_format="user@domain.com"
        )
        
        assert error.category == ErrorCategory.INPUT_VALIDATION
        assert error.details["field"] == "email"
        assert error.details["expected_format"] == "user@domain.com"
    
    def test_data_processing_error_inheritance(self):
        """Test that custom errors inherit properly."""
        error = DataProcessingError("Processing failed")
        
        assert isinstance(error, BaseApplicationError)
        assert error.category == ErrorCategory.DATA_PROCESSING
    
    def test_model_operation_error_with_context(self):
        """Test model operation error with context."""
        error = ModelOperationError(
            "Training failed",
            model_id="model_123",
            algorithm="isolation_forest",
            samples=1000
        )
        
        assert error.details["model_id"] == "model_123"
        assert error.details["algorithm"] == "isolation_forest"
        assert error.details["samples"] == 1000


class TestLoggingIntegration:
    """Test integration between logging components."""
    
    @patch('anomaly_detection.infrastructure.logging.structured_logger.get_logger')
    def test_decorator_error_handler_integration(self, mock_get_logger):
        """Test integration between decorators and error handling."""
        mock_logger = Mock(spec=StructuredLogger)
        mock_get_logger.return_value = mock_logger
        
        @log_decorator
        def function_that_raises():
            raise ValidationError("Invalid input", field="email")
        
        with pytest.raises(ValidationError):
            function_that_raises()
        
        # Check that error was properly logged
        mock_logger.error.assert_called_once()
        error_call = mock_logger.error.call_args
        assert "ValidationError" in str(error_call)
    
    def test_structured_logger_context_preservation(self):
        """Test that logging context is preserved across operations."""
        settings = LoggingSettings()
        logger = StructuredLogger("test", settings)
        
        # Set initial context
        logger.set_context({"request_id": "req_123"})
        
        # Use context manager
        with logger.context(operation="test_op"):
            context = logger.get_context()
            assert context["request_id"] == "req_123"
            assert context["operation"] == "test_op"
        
        # Context should be restored
        context = logger.get_context()
        assert context == {"request_id": "req_123"}
    
    @patch('structlog.get_logger')
    def test_performance_logging_integration(self, mock_structlog):
        """Test integration of performance logging with structured logging."""
        settings = LoggingSettings(
            enable_performance_logging=True,
            slow_detection_threshold_ms=100.0
        )
        logger = StructuredLogger("test", settings)
        mock_struct_logger = Mock()
        mock_structlog.return_value = mock_struct_logger
        
        # Log performance that exceeds threshold
        logger.log_performance("slow_operation", 150.0, {"param": "value"})
        
        # Should log as warning due to slow threshold
        mock_struct_logger.warning.assert_called_once()


class TestLoggingConfiguration:
    """Test logging configuration scenarios."""
    
    def test_logging_settings_defaults(self):
        """Test default logging settings."""
        settings = LoggingSettings()
        
        assert settings.level == "INFO"
        assert settings.format == "json"
        assert settings.enable_structured_logging is True
        assert settings.enable_request_tracking is True
        assert settings.enable_performance_logging is True
        assert settings.sanitize_sensitive_data is True
    
    def test_logging_settings_from_environment(self):
        """Test loading logging settings from environment variables."""
        with patch.dict('os.environ', {
            'LOG_LEVEL': 'DEBUG',
            'LOG_FORMAT': 'console',
            'LOG_STRUCTURED': 'false',
            'LOG_PERFORMANCE': 'false'
        }):
            settings = LoggingSettings.from_env()
            
            assert settings.level == "DEBUG"
            assert settings.format == "console"
            assert settings.enable_structured_logging is False
            assert settings.enable_performance_logging is False
    
    def test_performance_threshold_configuration(self):
        """Test performance threshold configuration."""
        settings = LoggingSettings(
            slow_query_threshold_ms=500.0,
            slow_detection_threshold_ms=2000.0,
            slow_model_training_threshold_ms=10000.0
        )
        
        assert settings.slow_query_threshold_ms == 500.0
        assert settings.slow_detection_threshold_ms == 2000.0
        assert settings.slow_model_training_threshold_ms == 10000.0
    
    @patch('logging.basicConfig')
    @patch('structlog.configure')
    def test_configure_logging_with_different_levels(self, mock_structlog, mock_basic):
        """Test logging configuration with different log levels."""
        for level in ["DEBUG", "INFO", "WARNING", "ERROR"]:
            settings = LoggingSettings(level=level)
            configure_logging(settings)
            
            # Check that basic config was called with correct level
            mock_basic.assert_called()
            basic_call = mock_basic.call_args[1]
            assert basic_call["level"] == getattr(logging, level)