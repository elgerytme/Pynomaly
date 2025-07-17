"""Value objects for API interfaces."""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any
from uuid import UUID, uuid4
from enum import Enum


class HTTPMethod(Enum):
    """HTTP method enumeration."""
    GET = "GET"
    POST = "POST"
    PUT = "PUT"
    DELETE = "DELETE"
    PATCH = "PATCH"
    HEAD = "HEAD"
    OPTIONS = "OPTIONS"


class StatusCode(Enum):
    """HTTP status code enumeration."""
    OK = 200
    CREATED = 201
    ACCEPTED = 202
    NO_CONTENT = 204
    BAD_REQUEST = 400
    UNAUTHORIZED = 401
    FORBIDDEN = 403
    NOT_FOUND = 404
    METHOD_NOT_ALLOWED = 405
    CONFLICT = 409
    INTERNAL_SERVER_ERROR = 500
    NOT_IMPLEMENTED = 501
    BAD_GATEWAY = 502
    SERVICE_UNAVAILABLE = 503


@dataclass(frozen=True)
class APIId:
    """Unique identifier for APIs."""
    value: UUID = field(default_factory=uuid4)
    
    def __str__(self) -> str:
        return str(self.value)


@dataclass(frozen=True)
class EndpointId:
    """Unique identifier for endpoints."""
    value: UUID = field(default_factory=uuid4)
    
    def __str__(self) -> str:
        return str(self.value)


@dataclass(frozen=True)
class APIConfiguration:
    """Configuration for APIs."""
    name: str
    version: str
    base_url: str
    authentication: Dict[str, Any] = field(default_factory=dict)
    rate_limiting: Dict[str, Any] = field(default_factory=dict)
    cors_settings: Dict[str, Any] = field(default_factory=dict)
    timeout_seconds: int = 30
    retry_attempts: int = 3
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert configuration to dictionary."""
        return {
            "name": self.name,
            "version": self.version,
            "base_url": self.base_url,
            "authentication": self.authentication,
            "rate_limiting": self.rate_limiting,
            "cors_settings": self.cors_settings,
            "timeout_seconds": self.timeout_seconds,
            "retry_attempts": self.retry_attempts,
        }


@dataclass(frozen=True)
class EndpointConfiguration:
    """Configuration for endpoints."""
    path: str
    method: HTTPMethod
    summary: str
    description: Optional[str] = None
    parameters: List[Dict[str, Any]] = field(default_factory=list)
    request_body: Optional[Dict[str, Any]] = None
    responses: Dict[int, Dict[str, Any]] = field(default_factory=dict)
    security: List[Dict[str, Any]] = field(default_factory=list)
    tags: List[str] = field(default_factory=list)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert configuration to dictionary."""
        return {
            "path": self.path,
            "method": self.method.value,
            "summary": self.summary,
            "description": self.description,
            "parameters": self.parameters,
            "request_body": self.request_body,
            "responses": self.responses,
            "security": self.security,
            "tags": self.tags,
        }


@dataclass(frozen=True)
class APIMetrics:
    """Metrics for APIs."""
    total_requests: int = 0
    successful_requests: int = 0
    failed_requests: int = 0
    average_response_time: float = 0.0
    error_rate: float = 0.0
    requests_per_second: float = 0.0
    uptime: float = 0.0
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert metrics to dictionary."""
        return {
            "total_requests": self.total_requests,
            "successful_requests": self.successful_requests,
            "failed_requests": self.failed_requests,
            "average_response_time": self.average_response_time,
            "error_rate": self.error_rate,
            "requests_per_second": self.requests_per_second,
            "uptime": self.uptime,
        }