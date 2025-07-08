"""FastAPI middleware for comprehensive error handling."""

from __future__ import annotations

import logging
import time
from typing import Callable

from fastapi import HTTPException, Request, Response
from fastapi.responses import JSONResponse
from starlette.middleware.base import BaseHTTPMiddleware

from pynomaly.domain.exceptions import (
    AuthenticationError,
    AuthorizationError,
    PynamolyError,
    ValidationError,
)

from .error_handler import ErrorHandler


class ErrorHandlingMiddleware(BaseHTTPMiddleware):
    """Middleware for handling errors across the entire FastAPI application."""

    def __init__(
        self,
        app,
        error_handler: ErrorHandler | None = None,
        include_debug_info: bool = False,
    ) -> None:
        """Initialize error handling middleware.

        Args:
            app: FastAPI application instance
            error_handler: Error handler instance
            include_debug_info: Whether to include debug information in responses
        """
        super().__init__(app)
        self.error_handler = error_handler or ErrorHandler()
        self.include_debug_info = include_debug_info
        self.logger = logging.getLogger(__name__)

    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        """Handle request and catch any errors."""
        start_time = time.time()

        try:
            # Extract request context
            context = await self._extract_request_context(request)

            # Process request
            response = await call_next(request)

            # Log successful requests
            duration = time.time() - start_time
            self.logger.info(
                f"Request completed: {request.method} {request.url.path} "
                f"-> {response.status_code} ({duration:.3f}s)"
            )

            return response

        except HTTPException as http_error:
            # Let FastAPI handle HTTP exceptions normally
            duration = time.time() - start_time
            self.logger.warning(
                f"HTTP error: {request.method} {request.url.path} "
                f"-> {http_error.status_code} ({duration:.3f}s): {http_error.detail}"
            )
            
            # Send system health alert for HTTP errors
            try:
                from pynomaly.infrastructure.monitoring.dual_metrics_service import get_dual_metrics_service
                dual_metrics_service = get_dual_metrics_service()
                if dual_metrics_service:
                    await dual_metrics_service.record_error(
                        error_type="HTTPException",
                        component="API",
                        severity="warning" if http_error.status_code < 500 else "error",
                        exception=http_error,
                        send_alert=http_error.status_code >= 500
                    )
            except Exception as e:
                self.logger.error(f"Failed to record HTTP error metrics: {e}")
            
            raise

        except Exception as error:
            # Handle all other errors
            duration = time.time() - start_time
            context = await self._extract_request_context(request)
            context["request_duration"] = duration

            # Handle the error
            error_response = self.error_handler.handle_error(
                error=error,
                context=context,
                operation=f"{request.method} {request.url.path}",
            )

            # Determine HTTP status code
            status_code = self._get_http_status_code(error)
            
            # Send system health alert for general exceptions
            try:
                from pynomaly.infrastructure.monitoring.dual_metrics_service import get_dual_metrics_service
                dual_metrics_service = get_dual_metrics_service()
                if dual_metrics_service:
                    await dual_metrics_service.record_error(
                        error_type=type(error).__name__,
                        component="API",
                        severity="critical" if status_code >= 500 else "error",
                        exception=error,
                        send_alert=True
                    )
            except Exception as e:
                self.logger.error(f"Failed to record general error metrics: {e}")

            # Add debug information if enabled
            if self.include_debug_info:
                error_response["debug"] = {
                    "request_id": context.get("request_id"),
                    "duration": duration,
                    "path": str(request.url.path),
                    "method": request.method,
                }

            return JSONResponse(
                status_code=status_code,
                content=error_response,
                headers={"X-Error-ID": error_response["error_id"]},
            )

    async def _extract_request_context(self, request: Request) -> dict:
        """Extract context information from the request."""
        context = {
            "method": request.method,
            "path": request.url.path,
            "query_params": dict(request.query_params),
            "headers": dict(request.headers),
            "client_ip": request.client.host if request.client else None,
        }

        # Extract user information if available
        if hasattr(request.state, "user"):
            context["user_id"] = getattr(request.state.user, "id", None)

        # Generate request ID if not present
        if "x-request-id" in request.headers:
            context["request_id"] = request.headers["x-request-id"]
        else:
            import uuid

            context["request_id"] = str(uuid.uuid4())

        return context

    def _get_http_status_code(self, error: Exception) -> int:
        """Determine appropriate HTTP status code for the error."""
        if isinstance(error, ValidationError):
            return 400  # Bad Request
        elif isinstance(error, AuthenticationError):
            return 401  # Unauthorized
        elif isinstance(error, AuthorizationError):
            return 403  # Forbidden
        elif isinstance(error, PynamolyError):
            # Check if it's a not found error
            if "not found" in str(error).lower():
                return 404  # Not Found
            elif "conflict" in str(error).lower():
                return 409  # Conflict
            else:
                return 400  # Bad Request (business logic error)
        else:
            return 500  # Internal Server Error


def create_error_middleware(
    error_handler: ErrorHandler | None = None,
    debug: bool = False,
) -> type[ErrorHandlingMiddleware]:
    """Create error handling middleware class with configuration.

    Args:
        error_handler: Error handler instance to use
        debug: Whether to include debug information

    Returns:
        Configured middleware class
    """

    class ConfiguredErrorMiddleware(ErrorHandlingMiddleware):
        def __init__(self, app):
            super().__init__(
                app=app,
                error_handler=error_handler,
                include_debug_info=debug,
            )

    return ConfiguredErrorMiddleware
