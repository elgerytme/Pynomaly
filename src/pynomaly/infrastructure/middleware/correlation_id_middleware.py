"""Correlation ID middleware for request tracking."""

from __future__ import annotations

from uuid import uuid4

from fastapi import Request, Response
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.types import ASGIApp


class CorrelationIdMiddleware(BaseHTTPMiddleware):
    """Middleware that adds correlation IDs to requests and responses."""

    def __init__(self, app: ASGIApp, header_name: str = "X-Correlation-ID") -> None:
        """Initialize the correlation ID middleware.
        
        Args:
            app: The ASGI application
            header_name: Name of the correlation ID header
        """
        super().__init__(app)
        self.header_name = header_name

    async def dispatch(self, request: Request, call_next) -> Response:
        """Process request and add correlation ID to response."""
        # Get or generate correlation ID
        correlation_id = request.headers.get(self.header_name, str(uuid4()))
        
        # Store in request state for use in exception handlers
        request.state.correlation_id = correlation_id
        
        # Process request
        response = await call_next(request)
        
        # Add correlation ID to response headers
        response.headers[self.header_name] = correlation_id
        
        return response


async def add_correlation_id(request: Request, call_next):
    """Simple middleware function to add correlation IDs."""
    correlation_id = request.headers.get('X-Correlation-ID', str(uuid4()))
    request.state.correlation_id = correlation_id
    response = await call_next(request)
    response.headers['X-Correlation-ID'] = correlation_id
    return response
