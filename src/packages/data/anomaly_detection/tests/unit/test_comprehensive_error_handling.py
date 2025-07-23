"""Tests for comprehensive error handling system."""

import pytest
import asyncio
from unittest.mock import Mock, patch, AsyncMock
from datetime import datetime
import numpy as np
from fastapi import Request, HTTPException, status
from fastapi.responses import JSONResponse

from anomaly_detection.infrastructure.middleware.comprehensive_error_middleware import (
    ComprehensiveErrorMiddleware,
    RateLimiter,
    SecurityValidator,
    RequestValidator
)
from anomaly_detection.infrastructure.validation.comprehensive_validators import (
    ComprehensiveValidator,
    DataValidator,
    ParameterValidator,
    EnsembleValidator,
    ValidationResult
)
from anomaly_detection.infrastructure.api.response_utilities import (
    ResponseBuilder,
    ErrorResponseBuilder,
    DetailedError,
    ValidationError,
    PaginationMetadata
)
from anomaly_detection.infrastructure.logging.error_handler import (
    ErrorHandler,
    InputValidationError,
    AlgorithmError
)


class TestRateLimiter:
    """Test rate limiting functionality."""
    
    @pytest.mark.asyncio
    async def test_rate_limiter_allows_initial_requests(self):
        """Test that rate limiter allows initial requests."""
        limiter = RateLimiter(max_requests=5, window_seconds=60)
        
        allowed, info = await limiter.is_allowed("test_client")
        
        assert allowed is True
        assert info["limit"] == 5
        assert info["remaining"] == 4
        assert info["current_count"] == 0
    
    @pytest.mark.asyncio
    async def test_rate_limiter_blocks_excess_requests(self):
        """Test that rate limiter blocks requests exceeding limit."""
        limiter = RateLimiter(max_requests=2, window_seconds=60)
        
        # Make allowed requests
        await limiter.is_allowed("test_client")
        await limiter.is_allowed("test_client")
        
        # This should be blocked
        allowed, info = await limiter.is_allowed("test_client")
        
        assert allowed is False
        assert info["remaining"] == 0
        assert info["current_count"] == 2
    
    @pytest.mark.asyncio
    async def test_rate_limiter_separate_clients(self):
        """Test that rate limiter tracks clients separately."""
        limiter = RateLimiter(max_requests=1, window_seconds=60)
        
        # Client 1 uses their limit
        allowed1, _ = await limiter.is_allowed("client1")
        blocked1, _ = await limiter.is_allowed("client1")
        
        # Client 2 should still be allowed
        allowed2, _ = await limiter.is_allowed("client2")
        
        assert allowed1 is True
        assert blocked1 is False
        assert allowed2 is True


class TestSecurityValidator:
    """Test security validation functionality."""
    
    def setup_method(self):
        """Set up test environment."""
        self.validator = SecurityValidator()
    
    @pytest.mark.asyncio
    async def test_blocked_ip_validation(self):
        """Test IP blocking functionality."""
        request = Mock(spec=Request)
        request.headers = {}
        request.client = Mock()
        request.client.host = "192.168.1.100"
        
        # Block the IP
        self.validator.block_ip("192.168.1.100")
        
        result = await self.validator.validate_request(request)
        
        assert result is not None
        assert result["error"] == "blocked_ip"
        assert "192.168.1.100" in result["message"]
    
    @pytest.mark.asyncio
    async def test_payload_size_validation(self):
        """Test payload size validation."""
        request = Mock(spec=Request)
        request.headers = {"content-length": str(self.validator.max_payload_size + 1)}
        request.client = Mock()
        request.client.host = "192.168.1.1"
        
        result = await self.validator.validate_request(request)
        
        assert result is not None
        assert result["error"] == "payload_too_large"
    
    @pytest.mark.asyncio
    async def test_suspicious_url_validation(self):
        """Test suspicious URL pattern detection."""
        request = Mock(spec=Request)
        request.headers = {}
        request.client = Mock()
        request.client.host = "192.168.1.1"
        request.url = Mock()
        request.url.__str__ = Mock(return_value="http://example.com/api?query=SELECT * FROM users")
        
        result = await self.validator.validate_request(request)
        
        assert result is not None
        assert result["error"] == "suspicious_url"
    
    @pytest.mark.asyncio
    async def test_valid_request_passes(self):
        """Test that valid requests pass validation."""
        request = Mock(spec=Request)
        request.headers = {"content-length": "100"}
        request.client = Mock()
        request.client.host = "192.168.1.1"
        request.url = Mock()
        request.url.__str__ = Mock(return_value="http://example.com/api/detect")
        
        result = await self.validator.validate_request(request)
        
        assert result is None


class TestDataValidator:
    """Test data validation functionality."""
    
    def setup_method(self):
        """Set up test environment."""
        self.validator = DataValidator()
    
    def test_valid_data_passes_validation(self):
        """Test that valid data passes validation."""
        data = np.random.rand(100, 5).astype(np.float64)
        
        result = self.validator.validate_detection_data(data, "iforest", 0.1)
        
        assert result.is_valid is True
        assert len(result.errors) == 0
    
    def test_insufficient_samples_validation(self):
        """Test validation with insufficient samples."""
        data = np.random.rand(5, 3).astype(np.float64)  # Less than minimum
        
        result = self.validator.validate_detection_data(data, "iforest", 0.1)
        
        assert result.is_valid is False
        assert any("Too few samples" in error for error in result.errors)
    
    def test_nan_values_validation(self):
        """Test validation with NaN values."""
        data = np.random.rand(50, 3).astype(np.float64)
        data[0, 0] = np.nan
        
        result = self.validator.validate_detection_data(data, "iforest", 0.1)
        
        assert result.is_valid is False
        assert any("NaN values" in error for error in result.errors)
    
    def test_infinite_values_validation(self):
        """Test validation with infinite values."""
        data = np.random.rand(50, 3).astype(np.float64)
        data[0, 0] = np.inf
        
        result = self.validator.validate_detection_data(data, "iforest", 0.1)
        
        assert result.is_valid is False
        assert any("infinite values" in error for error in result.errors)
    
    def test_constant_features_warning(self):
        """Test warning for constant features."""
        data = np.random.rand(50, 3).astype(np.float64)
        data[:, 1] = 1.0  # Constant feature
        
        result = self.validator.validate_detection_data(data, "iforest", 0.1)
        
        assert result.is_valid is True
        assert any("no variance" in warning for warning in result.warnings)
    
    def test_algorithm_specific_warnings(self):
        """Test algorithm-specific warnings."""
        data = np.random.rand(15, 3).astype(np.float64)  # Small dataset
        
        result = self.validator.validate_detection_data(data, "lof", 0.1)
        
        assert result.is_valid is True
        assert any("LOF algorithm may not work well" in warning for warning in result.warnings)


class TestParameterValidator:
    """Test parameter validation functionality."""
    
    def setup_method(self):
        """Set up test environment."""
        self.validator = ParameterValidator()
    
    def test_valid_parameters_pass(self):
        """Test that valid parameters pass validation."""
        params = {
            "n_estimators": 100,
            "contamination": 0.1,
            "random_state": 42
        }
        
        result = self.validator.validate_parameters("iforest", params)
        
        assert result.is_valid is True
        assert len(result.errors) == 0
    
    def test_invalid_parameter_type(self):
        """Test validation with invalid parameter type."""
        params = {
            "n_estimators": "invalid",  # Should be int
            "contamination": 0.1
        }
        
        result = self.validator.validate_parameters("iforest", params)
        
        assert result.is_valid is False
        assert any("must be of type int" in error for error in result.errors)
    
    def test_parameter_out_of_range(self):
        """Test validation with parameter out of range."""
        params = {
            "n_estimators": 5,  # Below minimum
            "contamination": 0.1
        }
        
        result = self.validator.validate_parameters("iforest", params)
        
        assert result.is_valid is False
        assert any("must be >= 10" in error for error in result.errors)
    
    def test_unknown_parameter_warning(self):
        """Test warning for unknown parameters."""
        params = {
            "n_estimators": 100,
            "unknown_param": "value"
        }
        
        result = self.validator.validate_parameters("iforest", params)
        
        assert result.is_valid is True
        assert any("Unknown parameter" in warning for warning in result.warnings)
    
    def test_unsupported_algorithm(self):
        """Test error for unsupported algorithm."""
        params = {"param": "value"}
        
        result = self.validator.validate_parameters("unknown_algorithm", params)
        
        assert result.is_valid is False
        assert any("Unsupported algorithm" in error for error in result.errors)


class TestEnsembleValidator:
    """Test ensemble validation functionality."""
    
    def setup_method(self):
        """Set up test environment."""
        self.validator = EnsembleValidator()
    
    def test_valid_ensemble_request(self):
        """Test valid ensemble request validation."""
        algorithms = ["iforest", "lof", "ocsvm"]
        method = "majority"
        
        result = self.validator.validate_ensemble_request(algorithms, method)
        
        assert result.is_valid is True
        assert len(result.errors) == 0
    
    def test_insufficient_algorithms(self):
        """Test validation with insufficient algorithms."""
        algorithms = ["iforest"]  # Need at least 2
        method = "majority"
        
        result = self.validator.validate_ensemble_request(algorithms, method)
        
        assert result.is_valid is False
        assert any("at least 2 algorithms" in error for error in result.errors)
    
    def test_unsupported_method(self):
        """Test validation with unsupported method."""
        algorithms = ["iforest", "lof"]
        method = "unknown_method"
        
        result = self.validator.validate_ensemble_request(algorithms, method)
        
        assert result.is_valid is False
        assert any("Unsupported ensemble method" in error for error in result.errors)
    
    def test_weighted_average_without_weights(self):
        """Test weighted average method without weights."""
        algorithms = ["iforest", "lof"]
        method = "weighted_average"
        
        result = self.validator.validate_ensemble_request(algorithms, method)
        
        assert result.is_valid is False
        assert any("Weights required" in error for error in result.errors)
    
    def test_weighted_average_with_invalid_weights(self):
        """Test weighted average with invalid weights."""
        algorithms = ["iforest", "lof"]
        method = "weighted_average"
        weights = [0.3, 0.8]  # Sum > 1
        
        result = self.validator.validate_ensemble_request(algorithms, method, weights)
        
        assert result.is_valid is True  # Should be valid but with warning
        assert any("normalizing to 1.0" in warning for warning in result.warnings)
    
    def test_duplicate_algorithms_warning(self):
        """Test warning for duplicate algorithms."""
        algorithms = ["iforest", "lof", "iforest"]
        method = "majority"
        
        result = self.validator.validate_ensemble_request(algorithms, method)
        
        assert result.is_valid is True
        assert any("Duplicate algorithms" in warning for warning in result.warnings)


class TestResponseBuilder:
    """Test response builder functionality."""
    
    def setup_method(self):
        """Set up test environment."""
        self.builder = ResponseBuilder(request_id="test-123")
    
    def test_success_response(self):
        """Test creating success response."""
        data = {"result": "success"}
        message = "Operation completed"
        
        response = self.builder.success(data=data, message=message)
        
        assert response.success is True
        assert response.status == "success"
        assert response.message == message
        assert response.data == data
        assert response.request_id == "test-123"
        assert response.processing_time_ms is not None
    
    def test_error_response_with_string(self):
        """Test creating error response with string error."""
        error_msg = "Something went wrong"
        
        response = self.builder.error(error_msg)
        
        assert response.success is False
        assert response.status == "error"
        assert error_msg in response.errors
        assert response.request_id == "test-123"
    
    def test_error_response_with_exception(self):
        """Test creating error response with exception."""
        exception = ValueError("Invalid input")
        
        response = self.builder.error(exception)
        
        assert response.success is False
        assert response.status == "error"
        assert "Invalid input" in response.errors[0]
        assert response.request_id == "test-123"
    
    def test_validation_error_response(self):
        """Test creating validation error response."""
        validation_errors = [
            ValidationError(field="data", value="invalid", message="Data is required"),
            ValidationError(field="algorithm", value="unknown", message="Unknown algorithm")
        ]
        
        response = self.builder.validation_error(validation_errors)
        
        assert response.success is False
        assert response.status == "error"
        assert len(response.errors) == 2
        assert "validation_errors" in response.metadata
        assert response.metadata["error_count"] == 2
    
    def test_partial_success_response(self):
        """Test creating partial success response."""
        data = {"processed": 8, "failed": 2}
        warnings = ["2 items could not be processed"]
        
        response = self.builder.partial_success(data, warnings)
        
        assert response.success is True
        assert response.status == "partial"
        assert response.data == data
        assert response.warnings == warnings
    
    def test_paginated_success_response(self):
        """Test creating paginated success response."""
        items = [{"id": 1}, {"id": 2}, {"id": 3}]
        pagination = PaginationMetadata.create(page=1, per_page=10, total_items=25)
        
        response = self.builder.paginated_success(items, pagination)
        
        assert response.success is True
        assert response.status == "success"
        assert response.data["items"] == items
        assert response.data["pagination"]["total_items"] == 25
        assert response.data["pagination"]["has_next"] is True


class TestErrorResponseBuilder:
    """Test error response builder functionality."""
    
    def setup_method(self):
        """Set up test environment."""
        self.builder = ErrorResponseBuilder(request_id="test-123")
    
    def test_bad_request_error(self):
        """Test creating bad request error."""
        message = "Invalid input data"
        details = {"field": "data", "issue": "empty"}
        
        error = self.builder.bad_request(message, details)
        
        assert error.type == "BadRequestError"
        assert error.category == "input_validation"
        assert error.message == message
        assert error.code == "BAD_REQUEST"
        assert error.recoverable is True
        assert error.details == details
    
    def test_not_found_error(self):
        """Test creating not found error."""
        resource = "model"
        resource_id = "model-123"
        
        error = self.builder.not_found(resource, resource_id)
        
        assert error.type == "NotFoundError"
        assert error.category == "resource_not_found"
        assert "Model not found" in error.message
        assert "model-123" in error.message
        assert error.code == "NOT_FOUND"
        assert error.recoverable is True
    
    def test_rate_limit_exceeded_error(self):
        """Test creating rate limit exceeded error."""
        limit = 100
        window_seconds = 60
        retry_after = 30
        
        error = self.builder.rate_limit_exceeded(limit, window_seconds, retry_after)
        
        assert error.type == "RateLimitExceededError"
        assert error.category == "rate_limit"
        assert "100 requests per 60 seconds" in error.message
        assert error.code == "RATE_LIMIT_EXCEEDED"
        assert error.recoverable is True
        assert error.details["retry_after"] == retry_after
    
    def test_internal_server_error(self):
        """Test creating internal server error."""
        message = "Database connection failed"
        error_id = "err-456"
        
        error = self.builder.internal_server_error(message, error_id)
        
        assert error.type == "InternalServerError"
        assert error.category == "system"
        assert error.message == message
        assert error.code == "INTERNAL_SERVER_ERROR"
        assert error.recoverable is False
        assert error.details["error_id"] == error_id
    
    def test_timeout_error(self):
        """Test creating timeout error."""
        operation = "model_training"
        timeout_seconds = 300.0
        
        error = self.builder.timeout_error(operation, timeout_seconds)
        
        assert error.type == "TimeoutError"
        assert error.category == "timeout"
        assert "model_training operation timed out" in error.message
        assert error.code == "TIMEOUT"
        assert error.recoverable is True
        assert error.details["timeout_seconds"] == timeout_seconds


class TestPaginationMetadata:
    """Test pagination metadata functionality."""
    
    def test_pagination_creation(self):
        """Test creating pagination metadata."""
        pagination = PaginationMetadata.create(page=2, per_page=10, total_items=25)
        
        assert pagination.page == 2
        assert pagination.per_page == 10
        assert pagination.total_items == 25
        assert pagination.total_pages == 3
        assert pagination.has_next is True
        assert pagination.has_previous is True
    
    def test_pagination_first_page(self):
        """Test pagination for first page."""
        pagination = PaginationMetadata.create(page=1, per_page=10, total_items=25)
        
        assert pagination.has_previous is False
        assert pagination.has_next is True
    
    def test_pagination_last_page(self):
        """Test pagination for last page."""
        pagination = PaginationMetadata.create(page=3, per_page=10, total_items=25)
        
        assert pagination.has_next is False
        assert pagination.has_previous is True
    
    def test_pagination_single_page(self):
        """Test pagination for single page."""
        pagination = PaginationMetadata.create(page=1, per_page=10, total_items=5)
        
        assert pagination.total_pages == 1
        assert pagination.has_next is False
        assert pagination.has_previous is False


@pytest.mark.asyncio
class TestComprehensiveErrorMiddleware:
    """Test comprehensive error middleware functionality."""
    
    def setup_method(self):
        """Set up test environment."""
        self.app = Mock()
        self.middleware = ComprehensiveErrorMiddleware(
            self.app,
            enable_detailed_errors=True
        )
    
    async def test_successful_request_processing(self):
        """Test successful request processing."""
        request = Mock(spec=Request)
        request.method = "GET"
        request.url = Mock()
        request.url.path = "/api/test"
        request.state = Mock()
        
        # Mock the call_next function
        response = Mock()
        response.status_code = 200
        response.headers = {}
        
        async def call_next(req):
            return response
        
        result = await self.middleware.dispatch(request, call_next)
        
        assert result == response
        assert "X-Request-ID" in response.headers
    
    async def test_middleware_statistics(self):
        """Test middleware statistics collection."""
        stats = self.middleware.get_middleware_stats()
        
        assert "total_requests" in stats
        assert "total_errors" in stats
        assert "blocked_requests" in stats
        assert "error_rate" in stats
        assert "uptime_seconds" in stats
        assert isinstance(stats["start_time"], str)


class TestComprehensiveValidator:
    """Test comprehensive validator integration."""
    
    def setup_method(self):
        """Set up test environment."""
        self.validator = ComprehensiveValidator()
    
    def test_comprehensive_detection_validation(self):
        """Test comprehensive detection request validation."""
        data = np.random.rand(100, 5).astype(np.float64)
        algorithm = "iforest"
        contamination = 0.1
        parameters = {"n_estimators": 100, "random_state": 42}
        
        result = self.validator.validate_detection_request(
            data, algorithm, contamination, parameters
        )
        
        assert result.is_valid is True
        assert len(result.errors) == 0
    
    def test_comprehensive_ensemble_validation(self):
        """Test comprehensive ensemble request validation."""
        data = np.random.rand(100, 5).astype(np.float64)
        algorithms = ["iforest", "lof"]
        method = "majority"
        contamination = 0.1
        
        result = self.validator.validate_ensemble_request(
            data, algorithms, method, contamination
        )
        
        assert result.is_valid is True
        assert len(result.errors) == 0
    
    def test_comprehensive_validation_with_errors(self):
        """Test comprehensive validation with multiple error sources."""
        data = np.random.rand(5, 3).astype(np.float64)  # Too few samples
        algorithm = "unknown_algorithm"  # Invalid algorithm
        contamination = 0.8  # High contamination
        parameters = {"invalid_param": "value"}  # Invalid parameter
        
        result = self.validator.validate_detection_request(
            data, algorithm, contamination, parameters
        )
        
        assert result.is_valid is False
        assert len(result.errors) > 0
        assert len(result.warnings) > 0