"""
Error handlers for specific web UI scenarios
Provides specialized error handling for common web UI operations
"""

from collections.abc import Callable
from functools import wraps
from typing import Any

import httpx
from fastapi import Request
from fastapi.responses import HTMLResponse
from jinja2 import TemplateError, TemplateNotFound
from redis.exceptions import RedisError
from sqlalchemy.exc import SQLAlchemyError

from .error_handling import (
    ErrorCode,
    ErrorLevel,
    WebUIError,
    get_web_ui_logger,
    log_web_ui_error,
)


def handle_template_errors(func: Callable) -> Callable:
    """Decorator to handle template rendering errors"""
    @wraps(func)
    async def wrapper(*args, **kwargs):
        try:
            return await func(*args, **kwargs)
        except TemplateNotFound as e:
            error = WebUIError(
                message=f"Template not found: {e.name}",
                error_code=ErrorCode.TEMPLATE_RENDER_ERROR,
                error_level=ErrorLevel.ERROR,
                details={"template_name": e.name},
                user_message="The requested page could not be loaded.",
                suggestion="Please check the URL and try again."
            )

            # Get request from args if available
            request = None
            for arg in args:
                if isinstance(arg, Request):
                    request = arg
                    break

            log_web_ui_error(error, request)
            raise error

        except TemplateError as e:
            error = WebUIError(
                message=f"Template rendering error: {str(e)}",
                error_code=ErrorCode.TEMPLATE_RENDER_ERROR,
                error_level=ErrorLevel.ERROR,
                details={"template_error": str(e)},
                user_message="There was an error loading the page.",
                suggestion="Please refresh the page or try again later."
            )

            # Get request from args if available
            request = None
            for arg in args:
                if isinstance(arg, Request):
                    request = arg
                    break

            log_web_ui_error(error, request)
            raise error

        except Exception as e:
            error = WebUIError(
                message=f"Unexpected template error: {str(e)}",
                error_code=ErrorCode.TEMPLATE_RENDER_ERROR,
                error_level=ErrorLevel.CRITICAL,
                details={"exception_type": type(e).__name__, "exception_message": str(e)},
                user_message="An unexpected error occurred while loading the page.",
                suggestion="Please try refreshing the page."
            )

            # Get request from args if available
            request = None
            for arg in args:
                if isinstance(arg, Request):
                    request = arg
                    break

            log_web_ui_error(error, request)
            raise error

    return wrapper


def handle_database_errors(func: Callable) -> Callable:
    """Decorator to handle database errors"""
    @wraps(func)
    async def wrapper(*args, **kwargs):
        try:
            return await func(*args, **kwargs)
        except SQLAlchemyError as e:
            error_code = ErrorCode.DATABASE_CONNECTION_ERROR
            error_level = ErrorLevel.ERROR

            # Determine specific error type
            if "connection" in str(e).lower():
                error_code = ErrorCode.DATABASE_CONNECTION_ERROR
                user_message = "Database connection error. Please try again."
            else:
                error_code = ErrorCode.DATABASE_QUERY_ERROR
                user_message = "Database query error. Please check your input."

            error = WebUIError(
                message=f"Database error: {str(e)}",
                error_code=error_code,
                error_level=error_level,
                details={"database_error": str(e)},
                user_message=user_message,
                suggestion="Please try again later or contact support if the problem persists."
            )

            # Get request from args if available
            request = None
            for arg in args:
                if isinstance(arg, Request):
                    request = arg
                    break

            log_web_ui_error(error, request)
            raise error

        except Exception as e:
            if "database" in str(e).lower() or "sql" in str(e).lower():
                error = WebUIError(
                    message=f"Database error: {str(e)}",
                    error_code=ErrorCode.DATABASE_CONNECTION_ERROR,
                    error_level=ErrorLevel.ERROR,
                    details={"database_error": str(e)},
                    user_message="Database temporarily unavailable.",
                    suggestion="Please try again in a few moments."
                )

                # Get request from args if available
                request = None
                for arg in args:
                    if isinstance(arg, Request):
                        request = arg
                        break

                log_web_ui_error(error, request)
                raise error
            else:
                raise

    return wrapper


def handle_cache_errors(func: Callable) -> Callable:
    """Decorator to handle cache errors"""
    @wraps(func)
    async def wrapper(*args, **kwargs):
        try:
            return await func(*args, **kwargs)
        except RedisError as e:
            error = WebUIError(
                message=f"Cache error: {str(e)}",
                error_code=ErrorCode.CACHE_ERROR,
                error_level=ErrorLevel.WARNING,
                details={"cache_error": str(e)},
                user_message="Cache temporarily unavailable. Some features may be slower.",
                suggestion="Please continue using the application normally."
            )

            # Get request from args if available
            request = None
            for arg in args:
                if isinstance(arg, Request):
                    request = arg
                    break

            log_web_ui_error(error, request)

            # For cache errors, we might want to continue without cache
            # rather than failing completely
            return await func(*args, **kwargs)

        except Exception as e:
            if "redis" in str(e).lower() or "cache" in str(e).lower():
                error = WebUIError(
                    message=f"Cache error: {str(e)}",
                    error_code=ErrorCode.CACHE_ERROR,
                    error_level=ErrorLevel.WARNING,
                    details={"cache_error": str(e)},
                    user_message="Cache temporarily unavailable.",
                    suggestion="Please continue using the application normally."
                )

                # Get request from args if available
                request = None
                for arg in args:
                    if isinstance(arg, Request):
                        request = arg
                        break

                log_web_ui_error(error, request)

                # Continue without cache
                return await func(*args, **kwargs)
            else:
                raise

    return wrapper


def handle_external_service_errors(func: Callable) -> Callable:
    """Decorator to handle external service errors"""
    @wraps(func)
    async def wrapper(*args, **kwargs):
        try:
            return await func(*args, **kwargs)
        except httpx.TimeoutException as e:
            error = WebUIError(
                message=f"External service timeout: {str(e)}",
                error_code=ErrorCode.API_TIMEOUT_ERROR,
                error_level=ErrorLevel.ERROR,
                details={"timeout_error": str(e)},
                user_message="The request took too long to process.",
                suggestion="Please try again or check your internet connection."
            )

            # Get request from args if available
            request = None
            for arg in args:
                if isinstance(arg, Request):
                    request = arg
                    break

            log_web_ui_error(error, request)
            raise error

        except httpx.HTTPStatusError as e:
            error = WebUIError(
                message=f"External service error: {e.response.status_code} {e.response.reason_phrase}",
                error_code=ErrorCode.EXTERNAL_SERVICE_ERROR,
                error_level=ErrorLevel.ERROR,
                details={
                    "status_code": e.response.status_code,
                    "response_text": e.response.text,
                    "url": str(e.response.url)
                },
                user_message="External service temporarily unavailable.",
                suggestion="Please try again later."
            )

            # Get request from args if available
            request = None
            for arg in args:
                if isinstance(arg, Request):
                    request = arg
                    break

            log_web_ui_error(error, request)
            raise error

        except httpx.ConnectError as e:
            error = WebUIError(
                message=f"External service connection error: {str(e)}",
                error_code=ErrorCode.EXTERNAL_SERVICE_ERROR,
                error_level=ErrorLevel.ERROR,
                details={"connection_error": str(e)},
                user_message="Cannot connect to external service.",
                suggestion="Please check your internet connection and try again."
            )

            # Get request from args if available
            request = None
            for arg in args:
                if isinstance(arg, Request):
                    request = arg
                    break

            log_web_ui_error(error, request)
            raise error

        except Exception as e:
            if "http" in str(e).lower() or "request" in str(e).lower():
                error = WebUIError(
                    message=f"External service error: {str(e)}",
                    error_code=ErrorCode.EXTERNAL_SERVICE_ERROR,
                    error_level=ErrorLevel.ERROR,
                    details={"service_error": str(e)},
                    user_message="External service error.",
                    suggestion="Please try again later."
                )

                # Get request from args if available
                request = None
                for arg in args:
                    if isinstance(arg, Request):
                        request = arg
                        break

                log_web_ui_error(error, request)
                raise error
            else:
                raise

    return wrapper


def handle_csrf_errors(func: Callable) -> Callable:
    """Decorator to handle CSRF token errors"""
    @wraps(func)
    async def wrapper(*args, **kwargs):
        try:
            return await func(*args, **kwargs)
        except Exception as e:
            if "csrf" in str(e).lower() or "token" in str(e).lower():
                error = WebUIError(
                    message=f"CSRF token error: {str(e)}",
                    error_code=ErrorCode.CSRF_TOKEN_ERROR,
                    error_level=ErrorLevel.ERROR,
                    details={"csrf_error": str(e)},
                    user_message="Security token expired or invalid.",
                    suggestion="Please refresh the page and try again."
                )

                # Get request from args if available
                request = None
                for arg in args:
                    if isinstance(arg, Request):
                        request = arg
                        break

                log_web_ui_error(error, request)
                raise error
            else:
                raise

    return wrapper


def handle_performance_errors(func: Callable) -> Callable:
    """Decorator to handle performance-related errors"""
    @wraps(func)
    async def wrapper(*args, **kwargs):
        try:
            return await func(*args, **kwargs)
        except TimeoutError as e:
            error = WebUIError(
                message=f"Operation timeout: {str(e)}",
                error_code=ErrorCode.API_TIMEOUT_ERROR,
                error_level=ErrorLevel.ERROR,
                details={"timeout_error": str(e)},
                user_message="The operation took too long to complete.",
                suggestion="Please try again or try with a smaller dataset."
            )

            # Get request from args if available
            request = None
            for arg in args:
                if isinstance(arg, Request):
                    request = arg
                    break

            log_web_ui_error(error, request)
            raise error

        except MemoryError as e:
            error = WebUIError(
                message=f"Memory error: {str(e)}",
                error_code=ErrorCode.MEMORY_LIMIT_ERROR,
                error_level=ErrorLevel.ERROR,
                details={"memory_error": str(e)},
                user_message="The operation requires too much memory.",
                suggestion="Please try with a smaller dataset or contact support."
            )

            # Get request from args if available
            request = None
            for arg in args:
                if isinstance(arg, Request):
                    request = arg
                    break

            log_web_ui_error(error, request)
            raise error

        except Exception as e:
            if "memory" in str(e).lower() or "timeout" in str(e).lower():
                error = WebUIError(
                    message=f"Performance error: {str(e)}",
                    error_code=ErrorCode.PERFORMANCE_THRESHOLD_ERROR,
                    error_level=ErrorLevel.ERROR,
                    details={"performance_error": str(e)},
                    user_message="The operation exceeded performance limits.",
                    suggestion="Please try with optimized parameters."
                )

                # Get request from args if available
                request = None
                for arg in args:
                    if isinstance(arg, Request):
                        request = arg
                        break

                log_web_ui_error(error, request)
                raise error
            else:
                raise

    return wrapper


def handle_security_errors(func: Callable) -> Callable:
    """Decorator to handle security-related errors"""
    @wraps(func)
    async def wrapper(*args, **kwargs):
        try:
            return await func(*args, **kwargs)
        except Exception as e:
            error_message = str(e).lower()

            if "xss" in error_message or "script" in error_message:
                error = WebUIError(
                    message=f"XSS attempt detected: {str(e)}",
                    error_code=ErrorCode.XSS_ATTEMPT_ERROR,
                    error_level=ErrorLevel.CRITICAL,
                    details={"xss_error": str(e)},
                    user_message="Security violation detected.",
                    suggestion="Please ensure your input is safe and try again."
                )

                # Get request from args if available
                request = None
                for arg in args:
                    if isinstance(arg, Request):
                        request = arg
                        break

                log_web_ui_error(error, request)
                raise error

            elif "sql" in error_message or "injection" in error_message:
                error = WebUIError(
                    message=f"SQL injection attempt detected: {str(e)}",
                    error_code=ErrorCode.SQL_INJECTION_ERROR,
                    error_level=ErrorLevel.CRITICAL,
                    details={"sql_error": str(e)},
                    user_message="Security violation detected.",
                    suggestion="Please ensure your input is safe and try again."
                )

                # Get request from args if available
                request = None
                for arg in args:
                    if isinstance(arg, Request):
                        request = arg
                        break

                log_web_ui_error(error, request)
                raise error

            elif "security" in error_message or "unauthorized" in error_message:
                error = WebUIError(
                    message=f"Security error: {str(e)}",
                    error_code=ErrorCode.SECURITY_VIOLATION_ERROR,
                    error_level=ErrorLevel.ERROR,
                    details={"security_error": str(e)},
                    user_message="Security violation detected.",
                    suggestion="Please ensure you have proper permissions."
                )

                # Get request from args if available
                request = None
                for arg in args:
                    if isinstance(arg, Request):
                        request = arg
                        break

                log_web_ui_error(error, request)
                raise error
            else:
                raise

    return wrapper


def handle_all_web_ui_errors(func: Callable) -> Callable:
    """Comprehensive decorator that combines all error handlers"""
    @handle_template_errors
    @handle_database_errors
    @handle_cache_errors
    @handle_external_service_errors
    @handle_csrf_errors
    @handle_performance_errors
    @handle_security_errors
    @wraps(func)
    async def wrapper(*args, **kwargs):
        return await func(*args, **kwargs)

    return wrapper


class ErrorRecoveryManager:
    """Manages error recovery strategies"""

    def __init__(self):
        self.logger = get_web_ui_logger()
        self.recovery_strategies = {
            ErrorCode.TEMPLATE_RENDER_ERROR: self._recover_template_error,
            ErrorCode.DATABASE_CONNECTION_ERROR: self._recover_database_error,
            ErrorCode.CACHE_ERROR: self._recover_cache_error,
            ErrorCode.EXTERNAL_SERVICE_ERROR: self._recover_external_service_error,
            ErrorCode.CSRF_TOKEN_ERROR: self._recover_csrf_error,
            ErrorCode.API_TIMEOUT_ERROR: self._recover_timeout_error,
        }

    async def attempt_recovery(self, error: WebUIError, request: Request) -> Any | None:
        """Attempt to recover from an error"""
        strategy = self.recovery_strategies.get(error.error_code)

        if strategy:
            try:
                return await strategy(error, request)
            except Exception as e:
                self.logger.logger.error(
                    "Error recovery failed",
                    error_id=error.error_id,
                    recovery_error=str(e)
                )

        return None

    async def _recover_template_error(self, error: WebUIError, request: Request) -> HTMLResponse | None:
        """Recover from template rendering errors"""
        # Return a basic error page
        html_content = """
        <!DOCTYPE html>
        <html>
        <head><title>Page Error</title></head>
        <body>
            <h1>Page Temporarily Unavailable</h1>
            <p>We're sorry, but the page you requested is temporarily unavailable.</p>
            <p><a href="/">Return to home</a></p>
        </body>
        </html>
        """
        return HTMLResponse(content=html_content, status_code=500)

    async def _recover_database_error(self, error: WebUIError, request: Request) -> Any | None:
        """Recover from database errors"""
        # Could implement fallback to cached data or simplified view
        return None

    async def _recover_cache_error(self, error: WebUIError, request: Request) -> Any | None:
        """Recover from cache errors"""
        # Continue without cache
        return {"cache_disabled": True}

    async def _recover_external_service_error(self, error: WebUIError, request: Request) -> Any | None:
        """Recover from external service errors"""
        # Could implement fallback data or degraded functionality
        return None

    async def _recover_csrf_error(self, error: WebUIError, request: Request) -> Any | None:
        """Recover from CSRF errors"""
        # Generate new CSRF token
        from .csrf import generate_csrf_token
        return {"new_csrf_token": generate_csrf_token()}

    async def _recover_timeout_error(self, error: WebUIError, request: Request) -> Any | None:
        """Recover from timeout errors"""
        # Could implement retry with shorter timeout
        return None
