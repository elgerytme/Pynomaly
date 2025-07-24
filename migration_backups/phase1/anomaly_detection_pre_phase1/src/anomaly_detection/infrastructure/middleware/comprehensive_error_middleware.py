"""Comprehensive error handling middleware for FastAPI applications."""

from __future__ import annotations

import time
import uuid
from typing import Any, Dict, Optional, List
from collections import defaultdict, deque
from datetime import datetime, timedelta
import asyncio
import structlog

from fastapi import Request, Response, HTTPException, status
from fastapi.responses import JSONResponse
from starlette.middleware.base import BaseHTTPMiddleware

from ..logging.error_handler import (
    ErrorHandler, 
    AnomalyDetectionError, 
    InputValidationError,
    default_error_handler
)

logger = structlog.get_logger(__name__)


class RateLimiter:
    """Rate limiter for API endpoints."""
    
    def __init__(self, max_requests: int = 100, window_seconds: int = 60):
        """Initialize rate limiter.
        
        Args:
            max_requests: Maximum requests per window
            window_seconds: Time window in seconds
        """
        self.max_requests = max_requests
        self.window_seconds = window_seconds
        self._requests: Dict[str, deque] = defaultdict(lambda: deque())
        self._lock = asyncio.Lock()
    
    async def is_allowed(self, client_id: str) -> tuple[bool, Dict[str, Any]]:
        """Check if request is allowed for client.
        
        Args:
            client_id: Client identifier (IP address or API key)
            
        Returns:
            Tuple of (allowed, rate_limit_info)
        """
        async with self._lock:
            now = time.time()
            client_requests = self._requests[client_id]
            
            # Remove old requests outside window
            while client_requests and client_requests[0] < now - self.window_seconds:
                client_requests.popleft()
            
            current_count = len(client_requests)
            allowed = current_count < self.max_requests
            
            if allowed:
                client_requests.append(now)
            
            # Calculate reset time
            reset_time = now + self.window_seconds if client_requests else now
            remaining = max(0, self.max_requests - current_count - (1 if allowed else 0))
            
            rate_limit_info = {
                "limit": self.max_requests,
                "remaining": remaining,
                "reset": int(reset_time),
                "window_seconds": self.window_seconds,
                "current_count": current_count
            }
            
            return allowed, rate_limit_info


class SecurityValidator:
    """Security validation for API requests."""
    
    def __init__(self):
        """Initialize security validator."""
        self.blocked_ips: set[str] = set()
        self.suspicious_patterns = [
            # SQL injection patterns
            r"(?i)(union|select|insert|update|delete|drop|create|alter)\s+",
            # XSS patterns
            r"(?i)<script[^>]*>",
            r"(?i)javascript:",
            # Path traversal
            r"\.\./",
            r"\.\.\\",
            # Command injection
            r"(?i)(exec|eval|system|shell_exec)",
        ]
        self.max_payload_size = 10 * 1024 * 1024  # 10MB
        self.max_header_size = 8192  # 8KB
    
    async def validate_request(self, request: Request) -> Optional[Dict[str, Any]]:
        """Validate request for security issues.
        
        Args:
            request: FastAPI request object
            
        Returns:
            None if valid, error dict if invalid
        """
        client_ip = self._get_client_ip(request)
        
        # Check blocked IPs
        if client_ip in self.blocked_ips:
            return {
                "error": "blocked_ip",
                "message": f"IP address {client_ip} is blocked",
                "client_ip": client_ip
            }
        
        # Check payload size
        content_length = request.headers.get("content-length")
        if content_length:
            try:
                size = int(content_length)
                if size > self.max_payload_size:
                    return {
                        "error": "payload_too_large",
                        "message": f"Payload size {size} exceeds limit {self.max_payload_size}",
                        "size": size,
                        "limit": self.max_payload_size
                    }
            except ValueError:
                return {
                    "error": "invalid_content_length",
                    "message": "Invalid Content-Length header",
                    "content_length": content_length
                }
        
        # Check header sizes
        total_header_size = sum(len(k) + len(v) for k, v in request.headers.items())
        if total_header_size > self.max_header_size:
            return {
                "error": "headers_too_large",
                "message": f"Headers size {total_header_size} exceeds limit {self.max_header_size}",
                "size": total_header_size,
                "limit": self.max_header_size
            }
        
        # Check for suspicious patterns in URL
        url_str = str(request.url)
        for pattern in self.suspicious_patterns:
            import re
            if re.search(pattern, url_str):
                return {
                    "error": "suspicious_url",
                    "message": "URL contains suspicious patterns",
                    "pattern": pattern,
                    "client_ip": client_ip
                }
        
        return None
    
    def _get_client_ip(self, request: Request) -> str:
        """Get client IP address from request."""
        # Check for forwarded headers
        forwarded_for = request.headers.get("x-forwarded-for")
        if forwarded_for:
            return forwarded_for.split(",")[0].strip()
        
        real_ip = request.headers.get("x-real-ip")
        if real_ip:
            return real_ip
        
        # Fallback to direct client
        if hasattr(request, "client") and request.client:
            return request.client.host
        
        return "unknown"
    
    def block_ip(self, ip_address: str) -> None:
        """Block an IP address."""
        self.blocked_ips.add(ip_address)
        logger.warning("IP address blocked", ip=ip_address)
    
    def unblock_ip(self, ip_address: str) -> None:
        """Unblock an IP address."""
        self.blocked_ips.discard(ip_address)
        logger.info("IP address unblocked", ip=ip_address)


class RequestValidator:
    """Enhanced request validation."""
    
    def __init__(self):
        """Initialize request validator."""
        self.required_content_types = {
            "POST": ["application/json", "multipart/form-data"],
            "PUT": ["application/json", "multipart/form-data"],
            "PATCH": ["application/json"]
        }
        self.max_query_params = 50
        self.max_path_length = 2048
    
    async def validate_request(self, request: Request) -> Optional[Dict[str, Any]]:
        """Validate request structure and content.
        
        Args:
            request: FastAPI request object
            
        Returns:
            None if valid, error dict if invalid
        """
        # Check path length
        if len(str(request.url.path)) > self.max_path_length:
            return {
                "error": "path_too_long",
                "message": f"URL path exceeds maximum length {self.max_path_length}",
                "path_length": len(str(request.url.path)),
                "limit": self.max_path_length
            }
        
        # Check query parameter count
        if len(request.query_params) > self.max_query_params:
            return {
                "error": "too_many_query_params",
                "message": f"Too many query parameters: {len(request.query_params)} > {self.max_query_params}",
                "count": len(request.query_params),
                "limit": self.max_query_params
            }
        
        # Check content type for methods that require body
        method = request.method.upper()
        if method in self.required_content_types:
            content_type = request.headers.get("content-type", "").lower()
            allowed_types = self.required_content_types[method]
            
            if not any(allowed_type in content_type for allowed_type in allowed_types):
                return {
                    "error": "invalid_content_type",
                    "message": f"Invalid content type for {method} request",
                    "content_type": content_type,
                    "allowed_types": allowed_types
                }
        
        return None


class ComprehensiveErrorMiddleware(BaseHTTPMiddleware):
    """Comprehensive error handling middleware with rate limiting and security."""
    
    def __init__(
        self, 
        app,
        error_handler: Optional[ErrorHandler] = None,
        rate_limiter: Optional[RateLimiter] = None,
        security_validator: Optional[SecurityValidator] = None,
        request_validator: Optional[RequestValidator] = None,
        enable_detailed_errors: bool = False
    ):
        """Initialize comprehensive error middleware.
        
        Args:
            app: FastAPI application
            error_handler: Custom error handler
            rate_limiter: Rate limiter instance
            security_validator: Security validator instance
            request_validator: Request validator instance
            enable_detailed_errors: Whether to include detailed error information
        """
        super().__init__(app)
        self.error_handler = error_handler or default_error_handler
        self.rate_limiter = rate_limiter or RateLimiter(max_requests=100, window_seconds=60)
        self.security_validator = security_validator or SecurityValidator()
        self.request_validator = request_validator or RequestValidator()
        self.enable_detailed_errors = enable_detailed_errors
        
        # Request tracking
        self._request_count = 0
        self._error_count = 0
        self._blocked_requests = 0
        self._start_time = datetime.utcnow()
    
    async def dispatch(self, request: Request, call_next) -> Response:
        """Process request through comprehensive middleware chain."""
        request_id = str(uuid.uuid4())
        start_time = time.time()
        client_ip = self.security_validator._get_client_ip(request)
        
        # Set request context
        request.state.request_id = request_id
        request.state.start_time = start_time
        request.state.client_ip = client_ip
        
        try:
            self._request_count += 1
            
            # Security validation
            security_error = await self.security_validator.validate_request(request)
            if security_error:
                self._blocked_requests += 1
                return await self._create_security_error_response(security_error, request_id)
            
            # Request structure validation
            validation_error = await self.request_validator.validate_request(request)
            if validation_error:
                return await self._create_validation_error_response(validation_error, request_id)
            
            # Rate limiting
            allowed, rate_info = await self.rate_limiter.is_allowed(client_ip)
            if not allowed:
                return await self._create_rate_limit_response(rate_info, request_id)
            
            # Process request
            response = await call_next(request)
            
            # Add rate limit headers
            self._add_rate_limit_headers(response, rate_info)
            
            # Add request ID header
            response.headers["X-Request-ID"] = request_id
            
            # Log successful request
            processing_time = (time.time() - start_time) * 1000
            logger.info(
                "Request processed successfully",
                request_id=request_id,
                method=request.method,
                path=str(request.url.path),
                status_code=response.status_code,
                processing_time_ms=processing_time,
                client_ip=client_ip
            )
            
            return response
            
        except AnomalyDetectionError as e:
            self._error_count += 1
            return await self._handle_custom_error(e, request_id, request)
            
        except HTTPException as e:
            self._error_count += 1
            return await self._handle_http_exception(e, request_id, request)
            
        except Exception as e:
            self._error_count += 1
            return await self._handle_unexpected_error(e, request_id, request)
    
    async def _create_security_error_response(
        self, 
        security_error: Dict[str, Any], 
        request_id: str
    ) -> JSONResponse:
        """Create response for security violations."""
        logger.warning(
            "Security violation detected",
            request_id=request_id,
            security_error=security_error
        )
        
        return JSONResponse(
            status_code=status.HTTP_403_FORBIDDEN,
            content={
                "success": False,
                "error": {
                    "type": "SecurityViolation",
                    "category": "security",
                    "message": "Request blocked due to security policy violation",
                    "recoverable": False,
                    "details": security_error if self.enable_detailed_errors else {}
                },
                "request_id": request_id,
                "timestamp": datetime.utcnow().isoformat()
            },
            headers={"X-Request-ID": request_id}
        )
    
    async def _create_validation_error_response(
        self, 
        validation_error: Dict[str, Any], 
        request_id: str
    ) -> JSONResponse:
        """Create response for validation errors."""
        logger.warning(
            "Request validation failed",
            request_id=request_id,
            validation_error=validation_error
        )
        
        return JSONResponse(
            status_code=status.HTTP_400_BAD_REQUEST,
            content={
                "success": False,
                "error": {
                    "type": "ValidationError",
                    "category": "input_validation",
                    "message": validation_error.get("message", "Request validation failed"),
                    "recoverable": True,
                    "details": validation_error if self.enable_detailed_errors else {}
                },
                "request_id": request_id,
                "timestamp": datetime.utcnow().isoformat()
            },
            headers={"X-Request-ID": request_id}
        )
    
    async def _create_rate_limit_response(
        self, 
        rate_info: Dict[str, Any], 
        request_id: str
    ) -> JSONResponse:
        """Create response for rate limit exceeded."""
        logger.warning(
            "Rate limit exceeded",
            request_id=request_id,
            rate_info=rate_info
        )
        
        headers = {
            "X-Request-ID": request_id,
            "X-RateLimit-Limit": str(rate_info["limit"]),
            "X-RateLimit-Remaining": str(rate_info["remaining"]),
            "X-RateLimit-Reset": str(rate_info["reset"]),
            "Retry-After": str(rate_info["window_seconds"])
        }
        
        return JSONResponse(
            status_code=status.HTTP_429_TOO_MANY_REQUESTS,
            content={
                "success": False,
                "error": {
                    "type": "RateLimitExceeded",
                    "category": "rate_limit",
                    "message": f"Rate limit exceeded: {rate_info['current_count']}/{rate_info['limit']} requests per {rate_info['window_seconds']}s",
                    "recoverable": True,
                    "details": {
                        "retry_after_seconds": rate_info["window_seconds"],
                        "limit": rate_info["limit"],
                        "window_seconds": rate_info["window_seconds"]
                    }
                },
                "request_id": request_id,
                "timestamp": datetime.utcnow().isoformat()
            },
            headers=headers
        )
    
    async def _handle_custom_error(
        self, 
        error: AnomalyDetectionError, 
        request_id: str, 
        request: Request
    ) -> JSONResponse:
        """Handle custom anomaly detection errors."""
        processing_time = (time.time() - request.state.start_time) * 1000
        
        logger.error(
            "Custom error in request processing",
            request_id=request_id,
            error_type=type(error).__name__,
            error_category=error.category.value,
            error_message=str(error),
            method=request.method,
            path=str(request.url.path),
            processing_time_ms=processing_time,
            client_ip=request.state.client_ip,
            recoverable=error.recoverable
        )
        
        status_code = status.HTTP_400_BAD_REQUEST if error.recoverable else status.HTTP_500_INTERNAL_SERVER_ERROR
        
        error_response = self.error_handler.create_error_response(error)
        error_response.update({
            "request_id": request_id,
            "timestamp": datetime.utcnow().isoformat(),
            "processing_time_ms": processing_time
        })
        
        # Add detailed error info if enabled
        if self.enable_detailed_errors and error.original_error:
            error_response["error"]["original_error"] = {
                "type": type(error.original_error).__name__,
                "message": str(error.original_error)
            }
        
        return JSONResponse(
            status_code=status_code,
            content=error_response,
            headers={"X-Request-ID": request_id}
        )
    
    async def _handle_http_exception(
        self, 
        error: HTTPException, 
        request_id: str, 
        request: Request
    ) -> JSONResponse:
        """Handle FastAPI HTTP exceptions."""
        processing_time = (time.time() - request.state.start_time) * 1000
        
        logger.warning(
            "HTTP exception in request processing",
            request_id=request_id,
            status_code=error.status_code,
            detail=error.detail,
            method=request.method,
            path=str(request.url.path),
            processing_time_ms=processing_time,
            client_ip=request.state.client_ip
        )
        
        return JSONResponse(
            status_code=error.status_code,
            content={
                "success": False,
                "error": {
                    "type": "HTTPException",
                    "category": "http_error",
                    "message": error.detail,
                    "recoverable": error.status_code < 500,
                    "details": {"status_code": error.status_code}
                },
                "request_id": request_id,
                "timestamp": datetime.utcnow().isoformat(),
                "processing_time_ms": processing_time
            },
            headers={"X-Request-ID": request_id}
        )
    
    async def _handle_unexpected_error(
        self, 
        error: Exception, 
        request_id: str, 
        request: Request
    ) -> JSONResponse:
        """Handle unexpected errors."""
        processing_time = (time.time() - request.state.start_time) * 1000
        
        # Use error handler to classify and log
        classified_error = self.error_handler.handle_error(
            error=error,
            context={
                "request_id": request_id,
                "method": request.method,
                "path": str(request.url.path),
                "client_ip": request.state.client_ip,
                "processing_time_ms": processing_time
            },
            operation="api_request",
            reraise=False
        )
        
        error_response = self.error_handler.create_error_response(classified_error)
        error_response.update({
            "request_id": request_id,
            "timestamp": datetime.utcnow().isoformat(),
            "processing_time_ms": processing_time
        })
        
        # Add detailed error info if enabled
        if self.enable_detailed_errors:
            error_response["error"]["original_error"] = {
                "type": type(error).__name__,
                "message": str(error)
            }
        
        return JSONResponse(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            content=error_response,
            headers={"X-Request-ID": request_id}
        )
    
    def _add_rate_limit_headers(self, response: Response, rate_info: Dict[str, Any]) -> None:
        """Add rate limiting headers to response."""
        response.headers["X-RateLimit-Limit"] = str(rate_info["limit"])
        response.headers["X-RateLimit-Remaining"] = str(rate_info["remaining"])
        response.headers["X-RateLimit-Reset"] = str(rate_info["reset"])
    
    def get_middleware_stats(self) -> Dict[str, Any]:
        """Get middleware statistics."""
        uptime = (datetime.utcnow() - self._start_time).total_seconds()
        
        return {
            "total_requests": self._request_count,
            "total_errors": self._error_count,
            "blocked_requests": self._blocked_requests,
            "error_rate": self._error_count / max(1, self._request_count),
            "blocked_rate": self._blocked_requests / max(1, self._request_count),
            "uptime_seconds": uptime,
            "requests_per_second": self._request_count / max(1, uptime),
            "start_time": self._start_time.isoformat()
        }