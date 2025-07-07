"""Common response models for API documentation."""

from __future__ import annotations

from datetime import datetime
from typing import Any, Generic, TypeVar

from pydantic import BaseModel, Field

T = TypeVar("T")


class BaseResponse(BaseModel):
    """Base response model with common metadata."""

    timestamp: datetime = Field(
        default_factory=datetime.utcnow, description="Response timestamp in UTC"
    )
    request_id: str | None = Field(
        None, description="Unique request identifier for tracking"
    )


class SuccessResponse(BaseResponse, Generic[T]):
    """Generic success response wrapper."""

    success: bool = Field(True, description="Indicates successful operation")
    data: T = Field(..., description="Response data")
    message: str | None = Field(None, description="Optional success message")

    class Config:
        json_schema_extra = {
            "example": {
                "success": True,
                "data": {"example": "data"},
                "message": "Operation completed successfully",
                "timestamp": "2024-12-25T10:30:00Z",
                "request_id": "req_12345",
            }
        }


class ErrorResponse(BaseResponse):
    """Standard error response model."""

    success: bool = Field(False, description="Indicates failed operation")
    error: str = Field(..., description="Error message")
    error_code: str | None = Field(None, description="Specific error code")
    details: dict[str, Any] | None = Field(None, description="Additional error details")

    class Config:
        json_schema_extra = {
            "example": {
                "success": False,
                "error": "Validation failed",
                "error_code": "VALIDATION_ERROR",
                "details": {"field": "username", "message": "Username is required"},
                "timestamp": "2024-12-25T10:30:00Z",
                "request_id": "req_12345",
            }
        }


class ValidationErrorResponse(ErrorResponse):
    """Validation error response with field details."""

    validation_errors: list[dict[str, Any]] = Field(
        ..., description="List of validation errors"
    )

    class Config:
        json_schema_extra = {
            "example": {
                "success": False,
                "error": "Validation failed",
                "error_code": "VALIDATION_ERROR",
                "validation_errors": [
                    {
                        "field": "email",
                        "message": "Invalid email format",
                        "value": "invalid-email",
                    },
                    {
                        "field": "age",
                        "message": "Must be a positive integer",
                        "value": -5,
                    },
                ],
                "timestamp": "2024-12-25T10:30:00Z",
                "request_id": "req_12345",
            }
        }


class PaginationMeta(BaseModel):
    """Pagination metadata."""

    page: int = Field(..., description="Current page number (1-based)")
    page_size: int = Field(..., description="Number of items per page")
    total_items: int = Field(..., description="Total number of items")
    total_pages: int = Field(..., description="Total number of pages")
    has_next: bool = Field(..., description="Whether there is a next page")
    has_previous: bool = Field(..., description="Whether there is a previous page")

    class Config:
        json_schema_extra = {
            "example": {
                "page": 2,
                "page_size": 20,
                "total_items": 157,
                "total_pages": 8,
                "has_next": True,
                "has_previous": True,
            }
        }


class PaginationResponse(BaseResponse, Generic[T]):
    """Paginated response wrapper."""

    success: bool = Field(True, description="Indicates successful operation")
    data: list[T] = Field(..., description="List of items for current page")
    pagination: PaginationMeta = Field(..., description="Pagination metadata")

    class Config:
        json_schema_extra = {
            "example": {
                "success": True,
                "data": [{"id": 1, "name": "Item 1"}, {"id": 2, "name": "Item 2"}],
                "pagination": {
                    "page": 1,
                    "page_size": 20,
                    "total_items": 157,
                    "total_pages": 8,
                    "has_next": True,
                    "has_previous": False,
                },
                "timestamp": "2024-12-25T10:30:00Z",
                "request_id": "req_12345",
            }
        }


class HealthResponse(BaseResponse):
    """Health check response model."""

    status: str = Field(..., description="Overall health status")
    version: str = Field(..., description="Application version")
    environment: str = Field(..., description="Environment name")
    uptime: float = Field(..., description="Uptime in seconds")
    services: dict[str, str] = Field(..., description="Individual service statuses")

    class Config:
        json_schema_extra = {
            "example": {
                "status": "healthy",
                "version": "1.0.0",
                "environment": "production",
                "uptime": 86400.0,
                "services": {
                    "database": "healthy",
                    "cache": "healthy",
                    "storage": "healthy",
                },
                "timestamp": "2024-12-25T10:30:00Z",
                "request_id": "health_12345",
            }
        }


class TaskResponse(BaseResponse):
    """Asynchronous task response model."""

    task_id: str = Field(..., description="Unique task identifier")
    status: str = Field(..., description="Task status")
    progress: float | None = Field(None, description="Task progress (0.0 to 1.0)")
    estimated_completion: datetime | None = Field(
        None, description="Estimated completion time"
    )
    result_url: str | None = Field(None, description="URL to retrieve task result")

    class Config:
        json_schema_extra = {
            "example": {
                "task_id": "task_abc123",
                "status": "running",
                "progress": 0.65,
                "estimated_completion": "2024-12-25T10:35:00Z",
                "result_url": "/tasks/task_abc123/result",
                "timestamp": "2024-12-25T10:30:00Z",
                "request_id": "req_12345",
            }
        }


class MetricsResponse(BaseResponse):
    """Metrics and statistics response model."""

    metrics: dict[str, Any] = Field(..., description="Metrics data")
    period: str = Field(..., description="Time period for metrics")
    aggregation: str = Field(..., description="Aggregation method used")

    class Config:
        json_schema_extra = {
            "example": {
                "metrics": {
                    "requests_total": 15420,
                    "avg_response_time": 0.245,
                    "error_rate": 0.02,
                    "active_users": 47,
                },
                "period": "1h",
                "aggregation": "average",
                "timestamp": "2024-12-25T10:30:00Z",
                "request_id": "req_12345",
            }
        }


# Common HTTP status code responses for OpenAPI documentation
class HTTPResponses:
    """Common HTTP response definitions for OpenAPI docs."""

    @staticmethod
    def success_200(description: str = "Successful operation") -> dict[str, Any]:
        """200 OK response."""
        return {
            "description": description,
            "content": {
                "application/json": {"schema": SuccessResponse[Any].model_json_schema()}
            },
        }

    @staticmethod
    def created_201(
        description: str = "Resource created successfully",
    ) -> dict[str, Any]:
        """201 Created response."""
        return {
            "description": description,
            "content": {
                "application/json": {"schema": SuccessResponse[Any].model_json_schema()}
            },
        }

    @staticmethod
    def no_content_204(
        description: str = "Operation completed, no content",
    ) -> dict[str, Any]:
        """204 No Content response."""
        return {"description": description}

    @staticmethod
    def bad_request_400(description: str = "Bad request") -> dict[str, Any]:
        """400 Bad Request response."""
        return {
            "description": description,
            "content": {
                "application/json": {
                    "schema": ValidationErrorResponse.model_json_schema()
                }
            },
        }

    @staticmethod
    def unauthorized_401(
        description: str = "Authentication required",
    ) -> dict[str, Any]:
        """401 Unauthorized response."""
        return {
            "description": description,
            "content": {
                "application/json": {"schema": ErrorResponse.model_json_schema()}
            },
        }

    @staticmethod
    def forbidden_403(description: str = "Insufficient permissions") -> dict[str, Any]:
        """403 Forbidden response."""
        return {
            "description": description,
            "content": {
                "application/json": {"schema": ErrorResponse.model_json_schema()}
            },
        }

    @staticmethod
    def not_found_404(description: str = "Resource not found") -> dict[str, Any]:
        """404 Not Found response."""
        return {
            "description": description,
            "content": {
                "application/json": {"schema": ErrorResponse.model_json_schema()}
            },
        }

    @staticmethod
    def conflict_409(description: str = "Resource conflict") -> dict[str, Any]:
        """409 Conflict response."""
        return {
            "description": description,
            "content": {
                "application/json": {"schema": ErrorResponse.model_json_schema()}
            },
        }

    @staticmethod
    def rate_limit_429(description: str = "Rate limit exceeded") -> dict[str, Any]:
        """429 Too Many Requests response."""
        return {
            "description": description,
            "content": {
                "application/json": {"schema": ErrorResponse.model_json_schema()}
            },
        }

    @staticmethod
    def server_error_500(description: str = "Internal server error") -> dict[str, Any]:
        """500 Internal Server Error response."""
        return {
            "description": description,
            "content": {
                "application/json": {"schema": ErrorResponse.model_json_schema()}
            },
        }
