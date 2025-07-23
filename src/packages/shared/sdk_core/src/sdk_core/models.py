"""Common response models for SDK clients."""

from datetime import datetime
from typing import Any, Dict, Generic, List, Optional, TypeVar, Union
from pydantic import BaseModel, Field


T = TypeVar("T")


class BaseResponse(BaseModel):
    """Base response model for all API responses."""
    
    success: bool = Field(description="Indicates if the request was successful")
    message: Optional[str] = Field(None, description="Optional message from the server")
    timestamp: datetime = Field(default_factory=datetime.utcnow, description="Response timestamp")
    request_id: Optional[str] = Field(None, description="Unique request identifier")


class ErrorDetail(BaseModel):
    """Details about an error."""
    
    field: Optional[str] = Field(None, description="Field that caused the error")
    code: str = Field(description="Error code")
    message: str = Field(description="Human-readable error message")


class ErrorResponse(BaseResponse):
    """Error response model."""
    
    success: bool = Field(default=False, description="Always false for error responses")
    error_code: str = Field(description="Machine-readable error code")
    error_message: str = Field(description="Human-readable error message")
    details: List[ErrorDetail] = Field(default_factory=list, description="Detailed error information")
    
    
class DataResponse(BaseResponse, Generic[T]):
    """Response model that contains data."""
    
    data: T = Field(description="Response data")


class PaginationInfo(BaseModel):
    """Pagination information."""
    
    page: int = Field(ge=1, description="Current page number")
    page_size: int = Field(ge=1, le=1000, description="Number of items per page")
    total_items: int = Field(ge=0, description="Total number of items")
    total_pages: int = Field(ge=0, description="Total number of pages")
    has_next: bool = Field(description="Whether there is a next page")
    has_previous: bool = Field(description="Whether there is a previous page")


class PaginatedResponse(BaseResponse, Generic[T]):
    """Paginated response model."""
    
    data: List[T] = Field(description="List of items for current page")
    pagination: PaginationInfo = Field(description="Pagination information")


class HealthResponse(BaseResponse):
    """Health check response."""
    
    status: str = Field(description="Health status (healthy, degraded, unhealthy)")
    version: str = Field(description="API version")
    uptime: float = Field(description="Uptime in seconds")
    checks: Dict[str, Dict[str, Any]] = Field(
        default_factory=dict,
        description="Individual health checks"
    )


class MetricsResponse(BaseResponse):
    """Metrics response."""
    
    metrics: Dict[str, Union[int, float, str]] = Field(
        description="System metrics"
    )


class BatchResponse(BaseResponse, Generic[T]):
    """Response for batch operations."""
    
    results: List[T] = Field(description="Results for each item in the batch")
    successful_count: int = Field(ge=0, description="Number of successful operations")
    failed_count: int = Field(ge=0, description="Number of failed operations")
    
    
class AsyncOperationResponse(BaseResponse):
    """Response for asynchronous operations."""
    
    operation_id: str = Field(description="Unique operation identifier")
    status: str = Field(description="Operation status (pending, running, completed, failed)")
    estimated_completion: Optional[datetime] = Field(
        None, 
        description="Estimated completion time"
    )
    progress: float = Field(
        ge=0.0, 
        le=1.0, 
        description="Operation progress (0.0 to 1.0)"
    )


# Common field definitions for reuse
class ModelInfo(BaseModel):
    """Common model information."""
    
    id: str = Field(description="Unique model identifier")
    name: str = Field(description="Model name")
    version: str = Field(description="Model version")
    algorithm: str = Field(description="Algorithm used")
    status: str = Field(description="Model status")
    created_at: datetime = Field(description="Creation timestamp")
    updated_at: Optional[datetime] = Field(None, description="Last update timestamp")
    
    
class UserInfo(BaseModel):
    """User information."""
    
    id: str = Field(description="User ID")
    username: str = Field(description="Username")
    email: Optional[str] = Field(None, description="Email address")
    roles: List[str] = Field(default_factory=list, description="User roles")
    permissions: List[str] = Field(default_factory=list, description="User permissions")
    
    
class ApiKeyInfo(BaseModel):
    """API key information."""
    
    id: str = Field(description="API key ID")
    name: str = Field(description="API key name")
    prefix: str = Field(description="API key prefix (first 8 characters)")
    scopes: List[str] = Field(default_factory=list, description="API key scopes")
    expires_at: Optional[datetime] = Field(None, description="Expiration timestamp")
    last_used_at: Optional[datetime] = Field(None, description="Last used timestamp")