"""API gateway and endpoint management for Pynomaly."""

from __future__ import annotations

import asyncio
import json
import logging
import uuid
from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any

from pynomaly.infrastructure.security.rate_limiting import RateLimiter, RateLimitScope
from pynomaly.shared.error_handling import (
    ErrorCodes,
    create_infrastructure_error,
)

logger = logging.getLogger(__name__)


class HTTPMethod(Enum):
    """HTTP methods for API endpoints."""

    GET = "GET"
    POST = "POST"
    PUT = "PUT"
    DELETE = "DELETE"
    PATCH = "PATCH"
    HEAD = "HEAD"
    OPTIONS = "OPTIONS"


class EndpointStatus(Enum):
    """API endpoint status."""

    ACTIVE = "active"
    INACTIVE = "inactive"
    DEPRECATED = "deprecated"
    MAINTENANCE = "maintenance"


class ResponseFormat(Enum):
    """API response formats."""

    JSON = "json"
    XML = "xml"
    PLAIN_TEXT = "plain_text"
    BINARY = "binary"


@dataclass
class APIRequest:
    """API request data structure."""

    request_id: str
    method: HTTPMethod
    path: str
    headers: dict[str, str] = field(default_factory=dict)
    query_params: dict[str, str] = field(default_factory=dict)
    body: Any | None = None
    timestamp: datetime = field(default_factory=datetime.now)
    client_ip: str = ""
    user_agent: str = ""
    auth_token: str | None = None

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary representation."""
        return {
            "request_id": self.request_id,
            "method": self.method.value,
            "path": self.path,
            "headers": self.headers,
            "query_params": self.query_params,
            "body": self.body,
            "timestamp": self.timestamp.isoformat(),
            "client_ip": self.client_ip,
            "user_agent": self.user_agent,
            "auth_token": "***" if self.auth_token else None,
        }


@dataclass
class APIResponse:
    """API response data structure."""

    request_id: str
    status_code: int
    headers: dict[str, str] = field(default_factory=dict)
    body: Any | None = None
    format: ResponseFormat = ResponseFormat.JSON
    timestamp: datetime = field(default_factory=datetime.now)
    execution_time_ms: float = 0.0

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary representation."""
        return {
            "request_id": self.request_id,
            "status_code": self.status_code,
            "headers": self.headers,
            "body": self.body,
            "format": self.format.value,
            "timestamp": self.timestamp.isoformat(),
            "execution_time_ms": self.execution_time_ms,
        }


@dataclass
class EndpointMetadata:
    """API endpoint metadata."""

    path: str
    method: HTTPMethod
    handler: Callable[[APIRequest], Awaitable[APIResponse]]
    name: str = ""
    description: str = ""
    version: str = "1.0.0"
    status: EndpointStatus = EndpointStatus.ACTIVE
    rate_limit: int | None = None
    auth_required: bool = False
    tags: list[str] = field(default_factory=list)
    created_at: datetime = field(default_factory=datetime.now)
    deprecated_at: datetime | None = None

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary representation."""
        return {
            "path": self.path,
            "method": self.method.value,
            "name": self.name,
            "description": self.description,
            "version": self.version,
            "status": self.status.value,
            "rate_limit": self.rate_limit,
            "auth_required": self.auth_required,
            "tags": self.tags,
            "created_at": self.created_at.isoformat(),
            "deprecated_at": self.deprecated_at.isoformat()
            if self.deprecated_at
            else None,
        }


class RequestProcessor:
    """Request processing and validation."""

    def __init__(self):
        """Initialize request processor."""
        self.processing_stats = {
            "total_requests": 0,
            "successful_requests": 0,
            "failed_requests": 0,
            "validation_errors": 0,
            "processing_times": [],
        }

    async def process_request(self, request: APIRequest) -> APIRequest:
        """Process and validate incoming request.

        Args:
            request: Raw API request

        Returns:
            Processed and validated request

        Raises:
            PynamolyError: If request validation fails
        """
        try:
            start_time = datetime.now()

            # Basic validation
            self._validate_request(request)

            # Sanitize inputs
            sanitized_request = self._sanitize_request(request)

            # Log request
            logger.debug(
                f"Processing request {request.request_id}: {request.method.value} {request.path}"
            )

            # Update stats
            processing_time = (datetime.now() - start_time).total_seconds() * 1000
            self.processing_stats["total_requests"] += 1
            self.processing_stats["successful_requests"] += 1
            self.processing_stats["processing_times"].append(processing_time)

            # Keep only last 1000 processing times
            if len(self.processing_stats["processing_times"]) > 1000:
                self.processing_stats["processing_times"] = self.processing_stats[
                    "processing_times"
                ][-1000:]

            return sanitized_request

        except Exception as e:
            self.processing_stats["total_requests"] += 1
            self.processing_stats["failed_requests"] += 1

            logger.error(f"Request processing failed for {request.request_id}: {e}")
            raise create_infrastructure_error(
                error_code=ErrorCodes.INF_PROCESSING_ERROR,
                message=f"Request processing failed: {str(e)}",
                cause=e,
            )

    def _validate_request(self, request: APIRequest) -> None:
        """Validate request structure and content."""
        if not request.path:
            self.processing_stats["validation_errors"] += 1
            raise ValueError("Request path cannot be empty")

        if not request.path.startswith("/"):
            self.processing_stats["validation_errors"] += 1
            raise ValueError("Request path must start with '/'")

        # Validate method
        if not isinstance(request.method, HTTPMethod):
            self.processing_stats["validation_errors"] += 1
            raise ValueError("Invalid HTTP method")

        # Validate content length for POST/PUT requests
        if request.method in [HTTPMethod.POST, HTTPMethod.PUT]:
            if request.body is not None:
                body_str = (
                    json.dumps(request.body)
                    if isinstance(request.body, dict)
                    else str(request.body)
                )
                if len(body_str) > 10_000_000:  # 10MB limit
                    self.processing_stats["validation_errors"] += 1
                    raise ValueError("Request body too large")

    def _sanitize_request(self, request: APIRequest) -> APIRequest:
        """Sanitize request data."""
        # Create a copy to avoid modifying original
        sanitized = APIRequest(
            request_id=request.request_id,
            method=request.method,
            path=request.path.strip(),
            headers={k.lower(): v.strip() for k, v in request.headers.items()},
            query_params={
                k.strip(): v.strip() for k, v in request.query_params.items()
            },
            body=request.body,
            timestamp=request.timestamp,
            client_ip=request.client_ip.strip(),
            user_agent=request.user_agent.strip(),
            auth_token=request.auth_token,
        )

        return sanitized

    async def get_processing_stats(self) -> dict[str, Any]:
        """Get request processing statistics."""
        processing_times = self.processing_stats["processing_times"]

        return {
            **self.processing_stats,
            "average_processing_time_ms": (
                sum(processing_times) / len(processing_times) if processing_times else 0
            ),
            "success_rate": (
                self.processing_stats["successful_requests"]
                / self.processing_stats["total_requests"]
                if self.processing_stats["total_requests"] > 0
                else 0
            ),
            "validation_error_rate": (
                self.processing_stats["validation_errors"]
                / self.processing_stats["total_requests"]
                if self.processing_stats["total_requests"] > 0
                else 0
            ),
        }


class ResponseProcessor:
    """Response processing and formatting."""

    def __init__(self):
        """Initialize response processor."""
        self.response_stats = {
            "total_responses": 0,
            "successful_responses": 0,
            "error_responses": 0,
            "status_code_counts": {},
        }

    async def process_response(self, response: APIResponse) -> APIResponse:
        """Process and format response.

        Args:
            response: Raw API response

        Returns:
            Processed and formatted response
        """
        try:
            # Add standard headers
            response.headers.update(
                {
                    "Content-Type": self._get_content_type(response.format),
                    "X-Request-ID": response.request_id,
                    "X-Response-Time": f"{response.execution_time_ms:.2f}ms",
                    "X-Timestamp": response.timestamp.isoformat(),
                }
            )

            # Format body based on response format
            if response.format == ResponseFormat.JSON and response.body is not None:
                if not isinstance(response.body, str):
                    response.body = json.dumps(response.body, indent=2)

            # Update stats
            self.response_stats["total_responses"] += 1

            if 200 <= response.status_code < 300:
                self.response_stats["successful_responses"] += 1
            else:
                self.response_stats["error_responses"] += 1

            # Track status code counts
            status_code = str(response.status_code)
            self.response_stats["status_code_counts"][status_code] = (
                self.response_stats["status_code_counts"].get(status_code, 0) + 1
            )

            logger.debug(
                f"Processing response {response.request_id}: {response.status_code}"
            )

            return response

        except Exception as e:
            logger.error(f"Response processing failed for {response.request_id}: {e}")

            # Create error response
            error_response = APIResponse(
                request_id=response.request_id,
                status_code=500,
                body={"error": "Internal server error", "message": str(e)},
                format=ResponseFormat.JSON,
                timestamp=datetime.now(),
            )

            return await self.process_response(error_response)

    def _get_content_type(self, format: ResponseFormat) -> str:
        """Get content type for response format."""
        content_types = {
            ResponseFormat.JSON: "application/json",
            ResponseFormat.XML: "application/xml",
            ResponseFormat.PLAIN_TEXT: "text/plain",
            ResponseFormat.BINARY: "application/octet-stream",
        }
        return content_types.get(format, "application/json")

    async def get_response_stats(self) -> dict[str, Any]:
        """Get response processing statistics."""
        return {
            **self.response_stats,
            "success_rate": (
                self.response_stats["successful_responses"]
                / self.response_stats["total_responses"]
                if self.response_stats["total_responses"] > 0
                else 0
            ),
            "error_rate": (
                self.response_stats["error_responses"]
                / self.response_stats["total_responses"]
                if self.response_stats["total_responses"] > 0
                else 0
            ),
        }


class EndpointManager:
    """Endpoint registration and management."""

    def __init__(self):
        """Initialize endpoint manager."""
        self.endpoints: dict[str, dict[HTTPMethod, EndpointMetadata]] = {}
        self.endpoint_stats: dict[str, dict[str, Any]] = {}

    def register_endpoint(self, metadata: EndpointMetadata) -> bool:
        """Register API endpoint.

        Args:
            metadata: Endpoint metadata

        Returns:
            True if registered successfully
        """
        try:
            path = metadata.path
            method = metadata.method

            if path not in self.endpoints:
                self.endpoints[path] = {}

            if method in self.endpoints[path]:
                logger.warning(
                    f"Endpoint {method.value} {path} already exists, overwriting"
                )

            self.endpoints[path][method] = metadata

            # Initialize stats
            endpoint_key = f"{method.value}:{path}"
            self.endpoint_stats[endpoint_key] = {
                "request_count": 0,
                "error_count": 0,
                "total_execution_time_ms": 0.0,
                "average_execution_time_ms": 0.0,
                "last_called": None,
            }

            logger.info(f"Registered endpoint: {method.value} {path}")
            return True

        except Exception as e:
            logger.error(
                f"Failed to register endpoint {metadata.method.value} {metadata.path}: {e}"
            )
            return False

    def get_endpoint(self, path: str, method: HTTPMethod) -> EndpointMetadata | None:
        """Get endpoint metadata.

        Args:
            path: Endpoint path
            method: HTTP method

        Returns:
            Endpoint metadata or None if not found
        """
        return self.endpoints.get(path, {}).get(method)

    def list_endpoints(
        self, status_filter: EndpointStatus | None = None, tag_filter: str | None = None
    ) -> list[EndpointMetadata]:
        """List registered endpoints.

        Args:
            status_filter: Filter by endpoint status
            tag_filter: Filter by tag

        Returns:
            List of endpoint metadata
        """
        endpoints = []

        for path_endpoints in self.endpoints.values():
            for metadata in path_endpoints.values():
                if status_filter and metadata.status != status_filter:
                    continue

                if tag_filter and tag_filter not in metadata.tags:
                    continue

                endpoints.append(metadata)

        # Sort by path and method
        endpoints.sort(key=lambda e: (e.path, e.method.value))

        return endpoints

    def deprecate_endpoint(self, path: str, method: HTTPMethod) -> bool:
        """Deprecate an endpoint.

        Args:
            path: Endpoint path
            method: HTTP method

        Returns:
            True if deprecated successfully
        """
        metadata = self.get_endpoint(path, method)
        if not metadata:
            return False

        metadata.status = EndpointStatus.DEPRECATED
        metadata.deprecated_at = datetime.now()

        logger.info(f"Deprecated endpoint: {method.value} {path}")
        return True

    def update_endpoint_stats(
        self,
        path: str,
        method: HTTPMethod,
        execution_time_ms: float,
        success: bool = True,
    ) -> None:
        """Update endpoint statistics.

        Args:
            path: Endpoint path
            method: HTTP method
            execution_time_ms: Request execution time
            success: Whether request was successful
        """
        endpoint_key = f"{method.value}:{path}"

        if endpoint_key not in self.endpoint_stats:
            return

        stats = self.endpoint_stats[endpoint_key]
        stats["request_count"] += 1
        stats["total_execution_time_ms"] += execution_time_ms
        stats["average_execution_time_ms"] = (
            stats["total_execution_time_ms"] / stats["request_count"]
        )
        stats["last_called"] = datetime.now().isoformat()

        if not success:
            stats["error_count"] += 1

    async def get_endpoint_stats(self, path: str | None = None) -> dict[str, Any]:
        """Get endpoint statistics.

        Args:
            path: Specific endpoint path (all if None)

        Returns:
            Endpoint statistics
        """
        if path:
            filtered_stats = {
                k: v for k, v in self.endpoint_stats.items() if k.endswith(f":{path}")
            }
            return filtered_stats

        return self.endpoint_stats


class APIVersioning:
    """API versioning management."""

    def __init__(self):
        """Initialize API versioning."""
        self.versions: dict[str, dict[str, Any]] = {}
        self.default_version = "v1"

    def register_version(
        self,
        version: str,
        description: str = "",
        deprecated: bool = False,
        sunset_date: datetime | None = None,
    ) -> bool:
        """Register API version.

        Args:
            version: Version identifier
            description: Version description
            deprecated: Whether version is deprecated
            sunset_date: When version will be removed

        Returns:
            True if registered successfully
        """
        try:
            self.versions[version] = {
                "description": description,
                "deprecated": deprecated,
                "sunset_date": sunset_date.isoformat() if sunset_date else None,
                "created_at": datetime.now().isoformat(),
                "endpoints": [],
            }

            logger.info(f"Registered API version: {version}")
            return True

        except Exception as e:
            logger.error(f"Failed to register API version {version}: {e}")
            return False

    def get_version_from_path(self, path: str) -> str:
        """Extract version from request path.

        Args:
            path: Request path

        Returns:
            API version
        """
        # Support paths like /v1/endpoint or /api/v1/endpoint
        parts = path.strip("/").split("/")

        for part in parts:
            if part.startswith("v") and part[1:].isdigit():
                return part

        return self.default_version

    def is_version_supported(self, version: str) -> bool:
        """Check if version is supported.

        Args:
            version: Version to check

        Returns:
            True if version is supported
        """
        return version in self.versions

    def is_version_deprecated(self, version: str) -> bool:
        """Check if version is deprecated.

        Args:
            version: Version to check

        Returns:
            True if version is deprecated
        """
        version_info = self.versions.get(version, {})
        return version_info.get("deprecated", False)

    def get_version_info(self, version: str) -> dict[str, Any]:
        """Get version information.

        Args:
            version: Version identifier

        Returns:
            Version information
        """
        return self.versions.get(version, {})


class APIGateway:
    """Main API gateway coordinating all components."""

    def __init__(self):
        """Initialize API gateway."""
        self.request_processor = RequestProcessor()
        self.response_processor = ResponseProcessor()
        self.endpoint_manager = EndpointManager()
        self.versioning = APIVersioning()
        self.rate_limiter = RateLimiter()
        self.gateway_stats = {
            "total_requests": 0,
            "total_responses": 0,
            "rate_limited_requests": 0,
            "start_time": datetime.now(),
        }

    async def handle_request(self, request: APIRequest) -> APIResponse:
        """Handle incoming API request.

        Args:
            request: API request to handle

        Returns:
            API response
        """
        start_time = datetime.now()

        try:
            # Update gateway stats
            self.gateway_stats["total_requests"] += 1

            # Process request
            processed_request = await self.request_processor.process_request(request)

            # Check rate limiting
            if not await self._check_rate_limit(processed_request):
                self.gateway_stats["rate_limited_requests"] += 1
                return await self._create_rate_limit_response(processed_request)

            # Get API version
            version = self.versioning.get_version_from_path(processed_request.path)

            # Check version support
            if not self.versioning.is_version_supported(version):
                return await self._create_error_response(
                    processed_request, 400, f"Unsupported API version: {version}"
                )

            # Check version deprecation
            if self.versioning.is_version_deprecated(version):
                logger.warning(
                    f"Using deprecated API version {version} for request {request.request_id}"
                )

            # Find endpoint
            endpoint = self.endpoint_manager.get_endpoint(
                processed_request.path, processed_request.method
            )

            if not endpoint:
                return await self._create_error_response(
                    processed_request,
                    404,
                    f"Endpoint not found: {processed_request.method.value} {processed_request.path}",
                )

            # Check endpoint status
            if endpoint.status == EndpointStatus.INACTIVE:
                return await self._create_error_response(
                    processed_request, 503, "Endpoint is temporarily unavailable"
                )

            if endpoint.status == EndpointStatus.MAINTENANCE:
                return await self._create_error_response(
                    processed_request, 503, "Endpoint is under maintenance"
                )

            # Execute endpoint handler
            response = await endpoint.handler(processed_request)

            # Calculate execution time
            execution_time = (datetime.now() - start_time).total_seconds() * 1000
            response.execution_time_ms = execution_time

            # Update endpoint stats
            self.endpoint_manager.update_endpoint_stats(
                processed_request.path,
                processed_request.method,
                execution_time,
                200 <= response.status_code < 300,
            )

            # Process response
            processed_response = await self.response_processor.process_response(
                response
            )

            # Update gateway stats
            self.gateway_stats["total_responses"] += 1

            return processed_response

        except Exception as e:
            logger.error(f"Request handling failed for {request.request_id}: {e}")

            # Create error response
            execution_time = (datetime.now() - start_time).total_seconds() * 1000
            error_response = APIResponse(
                request_id=request.request_id,
                status_code=500,
                body={"error": "Internal server error", "message": str(e)},
                format=ResponseFormat.JSON,
                timestamp=datetime.now(),
                execution_time_ms=execution_time,
            )

            self.gateway_stats["total_responses"] += 1

            return await self.response_processor.process_response(error_response)

    async def _check_rate_limit(self, request: APIRequest) -> bool:
        """Check rate limiting for request."""
        try:
            await self.rate_limiter.check_limit(
                identifier=request.client_ip,
                scope=RateLimitScope.GLOBAL,
                operation="api_request",
            )
            return True
        except Exception:
            return False

    async def _create_rate_limit_response(self, request: APIRequest) -> APIResponse:
        """Create rate limit exceeded response."""
        response = APIResponse(
            request_id=request.request_id,
            status_code=429,
            body={"error": "Rate limit exceeded", "message": "Too many requests"},
            format=ResponseFormat.JSON,
            timestamp=datetime.now(),
        )

        return await self.response_processor.process_response(response)

    async def _create_error_response(
        self, request: APIRequest, status_code: int, message: str
    ) -> APIResponse:
        """Create error response."""
        response = APIResponse(
            request_id=request.request_id,
            status_code=status_code,
            body={"error": "Request failed", "message": message},
            format=ResponseFormat.JSON,
            timestamp=datetime.now(),
        )

        return await self.response_processor.process_response(response)

    async def register_anomaly_detection_endpoints(self) -> None:
        """Register standard anomaly detection endpoints."""

        # Health check endpoint
        async def health_check(request: APIRequest) -> APIResponse:
            """Health check endpoint."""
            return APIResponse(
                request_id=request.request_id,
                status_code=200,
                body={
                    "status": "healthy",
                    "timestamp": datetime.now().isoformat(),
                    "version": "1.0.0",
                },
                format=ResponseFormat.JSON,
            )

        health_metadata = EndpointMetadata(
            path="/health",
            method=HTTPMethod.GET,
            handler=health_check,
            name="Health Check",
            description="System health check endpoint",
            version="1.0.0",
            tags=["system", "monitoring"],
        )

        # Detection endpoint
        async def detect_anomalies(request: APIRequest) -> APIResponse:
            """Anomaly detection endpoint."""
            if not request.body:
                return APIResponse(
                    request_id=request.request_id,
                    status_code=400,
                    body={"error": "Missing request body"},
                    format=ResponseFormat.JSON,
                )

            # Simulate detection process
            await asyncio.sleep(0.1)

            return APIResponse(
                request_id=request.request_id,
                status_code=200,
                body={
                    "detection_id": str(uuid.uuid4()),
                    "anomalies_detected": 3,
                    "timestamp": datetime.now().isoformat(),
                    "processing_time_ms": 100.0,
                },
                format=ResponseFormat.JSON,
            )

        detection_metadata = EndpointMetadata(
            path="/v1/detect",
            method=HTTPMethod.POST,
            handler=detect_anomalies,
            name="Anomaly Detection",
            description="Detect anomalies in provided data",
            version="1.0.0",
            rate_limit=100,  # 100 requests per minute
            tags=["detection", "core"],
        )

        # Register endpoints
        self.endpoint_manager.register_endpoint(health_metadata)
        self.endpoint_manager.register_endpoint(detection_metadata)

        # Register API version
        self.versioning.register_version("v1", "Initial API version")

    async def get_gateway_status(self) -> dict[str, Any]:
        """Get comprehensive gateway status.

        Returns:
            Gateway status information
        """
        uptime = (datetime.now() - self.gateway_stats["start_time"]).total_seconds()

        return {
            "gateway_stats": {
                **self.gateway_stats,
                "uptime_seconds": uptime,
                "requests_per_second": self.gateway_stats["total_requests"] / uptime
                if uptime > 0
                else 0,
            },
            "request_processor_stats": await self.request_processor.get_processing_stats(),
            "response_processor_stats": await self.response_processor.get_response_stats(),
            "endpoint_stats": await self.endpoint_manager.get_endpoint_stats(),
            "registered_endpoints": len(
                [
                    endpoint
                    for path_endpoints in self.endpoint_manager.endpoints.values()
                    for endpoint in path_endpoints.values()
                ]
            ),
            "api_versions": list(self.versioning.versions.keys()),
            "timestamp": datetime.now().isoformat(),
        }


# Global API gateway
_api_gateway: APIGateway | None = None


def get_api_gateway() -> APIGateway:
    """Get global API gateway.

    Returns:
        API gateway instance
    """
    global _api_gateway

    if _api_gateway is None:
        _api_gateway = APIGateway()

    return _api_gateway
