"""Comprehensive API response utilities with enhanced error handling."""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Union
from datetime import datetime
from pydantic import BaseModel, Field
from enum import Enum
import structlog

logger = structlog.get_logger(__name__)


class ResponseStatus(Enum):
    """API response status enumeration."""
    SUCCESS = "success"
    ERROR = "error"
    WARNING = "warning"
    PARTIAL = "partial"


class APIResponse(BaseModel):
    """Standardized API response model."""
    
    status: ResponseStatus = Field(..., description="Response status")
    success: bool = Field(..., description="Whether operation was successful")
    message: Optional[str] = Field(None, description="Human-readable message")
    data: Optional[Dict[str, Any]] = Field(None, description="Response data")
    errors: List[str] = Field(default_factory=list, description="List of errors")
    warnings: List[str] = Field(default_factory=list, description="List of warnings")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Response metadata")
    timestamp: str = Field(default_factory=lambda: datetime.utcnow().isoformat(), description="Response timestamp")
    request_id: Optional[str] = Field(None, description="Request identifier")
    processing_time_ms: Optional[float] = Field(None, description="Processing time in milliseconds")
    
    class Config:
        """Pydantic configuration."""
        use_enum_values = True


class PaginationMetadata(BaseModel):
    """Pagination metadata for list responses."""
    
    page: int = Field(..., ge=1, description="Current page number")
    per_page: int = Field(..., ge=1, le=1000, description="Items per page")
    total_items: int = Field(..., ge=0, description="Total number of items")
    total_pages: int = Field(..., ge=0, description="Total number of pages")
    has_next: bool = Field(..., description="Whether there is a next page")
    has_previous: bool = Field(..., description="Whether there is a previous page")
    
    @classmethod
    def create(cls, page: int, per_page: int, total_items: int) -> 'PaginationMetadata':
        """Create pagination metadata from parameters."""
        total_pages = (total_items + per_page - 1) // per_page
        return cls(
            page=page,
            per_page=per_page,
            total_items=total_items,
            total_pages=total_pages,
            has_next=page < total_pages,
            has_previous=page > 1
        )


class ValidationError(BaseModel):
    """Detailed validation error information."""
    
    field: str = Field(..., description="Field that failed validation")
    value: Any = Field(..., description="Invalid value")
    message: str = Field(..., description="Validation error message")
    code: Optional[str] = Field(None, description="Error code")


class DetailedError(BaseModel):
    """Detailed error information."""
    
    type: str = Field(..., description="Error type")
    category: str = Field(..., description="Error category")
    message: str = Field(..., description="Error message")
    code: Optional[str] = Field(None, description="Error code")
    recoverable: bool = Field(..., description="Whether error is recoverable")
    details: Dict[str, Any] = Field(default_factory=dict, description="Additional error details")
    validation_errors: List[ValidationError] = Field(default_factory=list, description="Validation errors")
    suggestion: Optional[str] = Field(None, description="Suggestion for fixing the error")


class ResponseBuilder:
    """Builder class for creating standardized API responses."""
    
    def __init__(self, request_id: Optional[str] = None):
        """Initialize response builder.
        
        Args:
            request_id: Optional request identifier
        """
        self.request_id = request_id
        self.start_time = datetime.utcnow()
    
    def success(
        self, 
        data: Optional[Dict[str, Any]] = None,
        message: Optional[str] = None,
        warnings: Optional[List[str]] = None,
        metadata: Optional[Dict[str, Any]] = None
    ) -> APIResponse:
        """Create successful response.
        
        Args:
            data: Response data
            message: Success message
            warnings: Optional warnings
            metadata: Additional metadata
            
        Returns:
            APIResponse with success status
        """
        processing_time = (datetime.utcnow() - self.start_time).total_seconds() * 1000
        
        return APIResponse(
            status=ResponseStatus.SUCCESS,
            success=True,
            message=message,
            data=data or {},
            warnings=warnings or [],
            metadata=metadata or {},
            request_id=self.request_id,
            processing_time_ms=processing_time
        )
    
    def error(
        self,
        error: Union[str, Exception, DetailedError],
        message: Optional[str] = None,
        data: Optional[Dict[str, Any]] = None,
        metadata: Optional[Dict[str, Any]] = None
    ) -> APIResponse:
        """Create error response.
        
        Args:
            error: Error information
            message: Optional override message
            data: Optional partial data
            metadata: Additional metadata
            
        Returns:
            APIResponse with error status
        """
        processing_time = (datetime.utcnow() - self.start_time).total_seconds() * 1000
        
        if isinstance(error, str):
            errors = [error]
            error_message = message or error
        elif isinstance(error, Exception):
            errors = [str(error)]
            error_message = message or f"Operation failed: {str(error)}"
        elif isinstance(error, DetailedError):
            errors = [error.message]
            error_message = message or error.message
            if metadata is None:
                metadata = {}
            metadata.update({
                "error_type": error.type,
                "error_category": error.category,
                "error_code": error.code,
                "recoverable": error.recoverable,
                "error_details": error.details
            })
        else:
            errors = ["Unknown error occurred"]
            error_message = message or "Unknown error occurred"
        
        return APIResponse(
            status=ResponseStatus.ERROR,
            success=False,
            message=error_message,
            data=data,
            errors=errors,
            metadata=metadata or {},
            request_id=self.request_id,
            processing_time_ms=processing_time
        )
    
    def validation_error(
        self,
        validation_errors: List[ValidationError],
        message: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None
    ) -> APIResponse:
        """Create validation error response.
        
        Args:
            validation_errors: List of validation errors
            message: Optional override message
            metadata: Additional metadata
            
        Returns:
            APIResponse with validation error status
        """
        processing_time = (datetime.utcnow() - self.start_time).total_seconds() * 1000
        
        error_messages = [f"{err.field}: {err.message}" for err in validation_errors]
        
        return APIResponse(
            status=ResponseStatus.ERROR,
            success=False,
            message=message or "Input validation failed",
            errors=error_messages,
            metadata={
                **(metadata or {}),
                "validation_errors": [err.dict() for err in validation_errors],
                "error_count": len(validation_errors)
            },
            request_id=self.request_id,
            processing_time_ms=processing_time
        )
    
    def partial_success(
        self,
        data: Dict[str, Any],
        warnings: List[str],
        message: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None
    ) -> APIResponse:
        """Create partial success response.
        
        Args:
            data: Partial response data
            warnings: List of warnings
            message: Optional message
            metadata: Additional metadata
            
        Returns:
            APIResponse with partial status
        """
        processing_time = (datetime.utcnow() - self.start_time).total_seconds() * 1000
        
        return APIResponse(
            status=ResponseStatus.PARTIAL,
            success=True,
            message=message or "Operation completed with warnings",
            data=data,
            warnings=warnings,
            metadata=metadata or {},
            request_id=self.request_id,
            processing_time_ms=processing_time
        )
    
    def paginated_success(
        self,
        items: List[Any],
        pagination: PaginationMetadata,
        message: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None
    ) -> APIResponse:
        """Create paginated success response.
        
        Args:
            items: List of items for current page
            pagination: Pagination metadata
            message: Optional message
            metadata: Additional metadata
            
        Returns:
            APIResponse with paginated data
        """
        processing_time = (datetime.utcnow() - self.start_time).total_seconds() * 1000
        
        return APIResponse(
            status=ResponseStatus.SUCCESS,
            success=True,
            message=message,
            data={
                "items": items,
                "pagination": pagination.dict()
            },
            metadata=metadata or {},
            request_id=self.request_id,
            processing_time_ms=processing_time
        )


class ErrorResponseBuilder:
    """Specialized builder for error responses."""
    
    def __init__(self, request_id: Optional[str] = None):
        """Initialize error response builder.
        
        Args:
            request_id: Optional request identifier
        """
        self.request_id = request_id
    
    def bad_request(
        self,
        message: str = "Bad request",
        details: Optional[Dict[str, Any]] = None,
        suggestions: Optional[List[str]] = None
    ) -> DetailedError:
        """Create bad request error.
        
        Args:
            message: Error message
            details: Additional error details
            suggestions: Suggestions for fixing the error
            
        Returns:
            DetailedError for bad request
        """
        return DetailedError(
            type="BadRequestError",
            category="input_validation",
            message=message,
            code="BAD_REQUEST",
            recoverable=True,
            details=details or {},
            suggestion="; ".join(suggestions) if suggestions else None
        )
    
    def not_found(
        self,
        resource: str,
        resource_id: Optional[str] = None
    ) -> DetailedError:
        """Create not found error.
        
        Args:
            resource: Resource type that was not found
            resource_id: Optional resource identifier
            
        Returns:
            DetailedError for not found
        """
        message = f"{resource.title()} not found"
        if resource_id:
            message += f" with id '{resource_id}'"
        
        return DetailedError(
            type="NotFoundError",
            category="resource_not_found",
            message=message,
            code="NOT_FOUND",
            recoverable=True,
            details={"resource": resource, "resource_id": resource_id},
            suggestion=f"Check that the {resource} exists and the identifier is correct"
        )
    
    def unauthorized(
        self,
        message: str = "Authentication required",
        details: Optional[Dict[str, Any]] = None
    ) -> DetailedError:
        """Create unauthorized error.
        
        Args:
            message: Error message
            details: Additional error details
            
        Returns:
            DetailedError for unauthorized access
        """
        return DetailedError(
            type="UnauthorizedError",
            category="authentication",
            message=message,
            code="UNAUTHORIZED",
            recoverable=True,
            details=details or {},
            suggestion="Provide valid authentication credentials"
        )
    
    def forbidden(
        self,
        message: str = "Access forbidden",
        required_permission: Optional[str] = None
    ) -> DetailedError:
        """Create forbidden error.
        
        Args:
            message: Error message
            required_permission: Permission required for access
            
        Returns:
            DetailedError for forbidden access
        """
        details = {}
        if required_permission:
            details["required_permission"] = required_permission
        
        return DetailedError(
            type="ForbiddenError",
            category="authorization",
            message=message,
            code="FORBIDDEN",
            recoverable=True,
            details=details,
            suggestion=f"Ensure you have the required permission: {required_permission}" if required_permission else None
        )
    
    def rate_limit_exceeded(
        self,
        limit: int,
        window_seconds: int,
        retry_after: int
    ) -> DetailedError:
        """Create rate limit exceeded error.
        
        Args:
            limit: Request limit
            window_seconds: Time window in seconds
            retry_after: Seconds to wait before retrying
            
        Returns:
            DetailedError for rate limit exceeded
        """
        return DetailedError(
            type="RateLimitExceededError",
            category="rate_limit",
            message=f"Rate limit exceeded: {limit} requests per {window_seconds} seconds",
            code="RATE_LIMIT_EXCEEDED",
            recoverable=True,
            details={
                "limit": limit,
                "window_seconds": window_seconds,
                "retry_after": retry_after
            },
            suggestion=f"Wait {retry_after} seconds before making another request"
        )
    
    def internal_server_error(
        self,
        message: str = "Internal server error",
        error_id: Optional[str] = None,
        details: Optional[Dict[str, Any]] = None
    ) -> DetailedError:
        """Create internal server error.
        
        Args:
            message: Error message
            error_id: Optional error identifier for tracking
            details: Additional error details
            
        Returns:
            DetailedError for internal server error
        """
        error_details = details or {}
        if error_id:
            error_details["error_id"] = error_id
        
        return DetailedError(
            type="InternalServerError",
            category="system",
            message=message,
            code="INTERNAL_SERVER_ERROR",
            recoverable=False,
            details=error_details,
            suggestion="Try again later or contact support if the problem persists"
        )
    
    def service_unavailable(
        self,
        service: str,
        retry_after: Optional[int] = None
    ) -> DetailedError:
        """Create service unavailable error.
        
        Args:
            service: Service that is unavailable
            retry_after: Optional seconds to wait before retrying
            
        Returns:
            DetailedError for service unavailable
        """
        details = {"service": service}
        if retry_after:
            details["retry_after"] = retry_after
        
        suggestion = f"The {service} service is temporarily unavailable"
        if retry_after:
            suggestion += f". Try again in {retry_after} seconds"
        
        return DetailedError(
            type="ServiceUnavailableError",
            category="service_unavailable",
            message=f"{service} service is unavailable",
            code="SERVICE_UNAVAILABLE",
            recoverable=True,
            details=details,
            suggestion=suggestion
        )
    
    def timeout_error(
        self,
        operation: str,
        timeout_seconds: float
    ) -> DetailedError:
        """Create timeout error.
        
        Args:
            operation: Operation that timed out
            timeout_seconds: Timeout duration in seconds
            
        Returns:
            DetailedError for timeout
        """
        return DetailedError(
            type="TimeoutError",
            category="timeout",
            message=f"{operation} operation timed out after {timeout_seconds} seconds",
            code="TIMEOUT",
            recoverable=True,
            details={
                "operation": operation,
                "timeout_seconds": timeout_seconds
            },
            suggestion="Try reducing the dataset size or increasing the timeout limit"
        )


class HealthCheckResponse(BaseModel):
    """Health check response model."""
    
    status: str = Field(..., description="Overall health status")
    service: str = Field(..., description="Service name")
    version: str = Field(..., description="Service version")
    timestamp: str = Field(..., description="Check timestamp")
    checks: Dict[str, Dict[str, Any]] = Field(default_factory=dict, description="Individual health checks")
    uptime_seconds: Optional[float] = Field(None, description="Service uptime")
    environment: Optional[str] = Field(None, description="Environment name")


class MetricsResponse(BaseModel):
    """Metrics response model."""
    
    metrics: Dict[str, Any] = Field(..., description="System metrics")
    timestamp: str = Field(..., description="Metrics timestamp")
    period_seconds: Optional[int] = Field(None, description="Metrics collection period")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Additional metadata")


def create_response_builder(request_id: Optional[str] = None) -> ResponseBuilder:
    """Factory function to create response builder.
    
    Args:
        request_id: Optional request identifier
        
    Returns:
        ResponseBuilder instance
    """
    return ResponseBuilder(request_id)


def create_error_builder(request_id: Optional[str] = None) -> ErrorResponseBuilder:
    """Factory function to create error response builder.
    
    Args:
        request_id: Optional request identifier
        
    Returns:
        ErrorResponseBuilder instance
    """
    return ErrorResponseBuilder(request_id)