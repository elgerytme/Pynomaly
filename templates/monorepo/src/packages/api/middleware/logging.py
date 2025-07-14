"""Logging middleware for request/response logging."""

import time
from typing import Callable

from fastapi import Request, Response
from starlette.middleware.base import BaseHTTPMiddleware


class LoggingMiddleware(BaseHTTPMiddleware):
    """Middleware for logging HTTP requests and responses."""
    
    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        """Process request and response."""
        start_time = time.time()
        
        # Log request
        print(f"Request: {request.method} {request.url}")
        
        # Process request
        response = await call_next(request)
        
        # Calculate processing time
        process_time = time.time() - start_time
        
        # Log response
        print(f"Response: {response.status_code} - {process_time:.3f}s")
        
        # Add processing time header
        response.headers["X-Process-Time"] = str(process_time)
        
        return response