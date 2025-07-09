"""
Production-ready error handling and logging for Web UI
Provides centralized error management, logging, and user-friendly error responses
"""

import logging
import traceback
import sys
from datetime import datetime
from typing import Dict, Any, Optional, List
from pathlib import Path
import json
import uuid
from enum import Enum

from fastapi import Request, HTTPException
from fastapi.responses import HTMLResponse, JSONResponse
from starlette.exceptions import HTTPException as StarletteHTTPException
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.responses import Response
import structlog

from ...infrastructure.config.settings import get_settings


class ErrorLevel(Enum):
    """Error severity levels"""
    CRITICAL = "critical"
    ERROR = "error"
    WARNING = "warning"
    INFO = "info"
    DEBUG = "debug"


class ErrorCode(Enum):
    """Standardized error codes for web UI"""
    # General errors
    INTERNAL_SERVER_ERROR = "INTERNAL_SERVER_ERROR"
    VALIDATION_ERROR = "VALIDATION_ERROR"
    AUTHENTICATION_ERROR = "AUTHENTICATION_ERROR"
    AUTHORIZATION_ERROR = "AUTHORIZATION_ERROR"
    NOT_FOUND_ERROR = "NOT_FOUND_ERROR"
    
    # Web UI specific errors
    TEMPLATE_RENDER_ERROR = "TEMPLATE_RENDER_ERROR"
    STATIC_FILE_ERROR = "STATIC_FILE_ERROR"
    CSRF_TOKEN_ERROR = "CSRF_TOKEN_ERROR"
    SESSION_ERROR = "SESSION_ERROR"
    
    # API errors
    API_ENDPOINT_ERROR = "API_ENDPOINT_ERROR"
    API_RATE_LIMIT_ERROR = "API_RATE_LIMIT_ERROR"
    API_TIMEOUT_ERROR = "API_TIMEOUT_ERROR"
    
    # Performance errors
    PERFORMANCE_THRESHOLD_ERROR = "PERFORMANCE_THRESHOLD_ERROR"
    MEMORY_LIMIT_ERROR = "MEMORY_LIMIT_ERROR"
    
    # Security errors
    SECURITY_VIOLATION_ERROR = "SECURITY_VIOLATION_ERROR"
    XSS_ATTEMPT_ERROR = "XSS_ATTEMPT_ERROR"
    SQL_INJECTION_ERROR = "SQL_INJECTION_ERROR"
    
    # Database errors
    DATABASE_CONNECTION_ERROR = "DATABASE_CONNECTION_ERROR"
    DATABASE_QUERY_ERROR = "DATABASE_QUERY_ERROR"
    
    # External service errors
    EXTERNAL_SERVICE_ERROR = "EXTERNAL_SERVICE_ERROR"
    CACHE_ERROR = "CACHE_ERROR"


class WebUIError(Exception):
    """Base exception for web UI errors"""
    
    def __init__(
        self,
        message: str,
        error_code: ErrorCode,
        error_level: ErrorLevel = ErrorLevel.ERROR,
        details: Optional[Dict[str, Any]] = None,
        user_message: Optional[str] = None,
        suggestion: Optional[str] = None,
        error_id: Optional[str] = None
    ):
        self.message = message
        self.error_code = error_code
        self.error_level = error_level
        self.details = details or {}
        self.user_message = user_message or self._get_default_user_message()
        self.suggestion = suggestion
        self.error_id = error_id or str(uuid.uuid4())
        self.timestamp = datetime.utcnow()
        
        super().__init__(self.message)
    
    def _get_default_user_message(self) -> str:
        """Get default user-friendly message based on error code"""
        user_messages = {
            ErrorCode.INTERNAL_SERVER_ERROR: "An unexpected error occurred. Please try again later.",
            ErrorCode.VALIDATION_ERROR: "Please check your input and try again.",
            ErrorCode.AUTHENTICATION_ERROR: "Please sign in to access this resource.",
            ErrorCode.AUTHORIZATION_ERROR: "You don't have permission to access this resource.",
            ErrorCode.NOT_FOUND_ERROR: "The requested resource was not found.",
            ErrorCode.TEMPLATE_RENDER_ERROR: "There was an error loading the page. Please refresh.",
            ErrorCode.STATIC_FILE_ERROR: "Some resources failed to load. Please refresh the page.",
            ErrorCode.CSRF_TOKEN_ERROR: "Security token expired. Please refresh the page.",
            ErrorCode.SESSION_ERROR: "Your session has expired. Please sign in again.",
            ErrorCode.API_RATE_LIMIT_ERROR: "Too many requests. Please wait a moment and try again.",
            ErrorCode.API_TIMEOUT_ERROR: "The request took too long. Please try again.",
            ErrorCode.SECURITY_VIOLATION_ERROR: "A security violation was detected.",
            ErrorCode.DATABASE_CONNECTION_ERROR: "Database temporarily unavailable. Please try again.",
            ErrorCode.EXTERNAL_SERVICE_ERROR: "External service temporarily unavailable.",
        }
        return user_messages.get(self.error_code, "An error occurred. Please try again.")
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert error to dictionary for logging/serialization"""
        return {
            "error_id": self.error_id,
            "error_code": self.error_code.value,
            "error_level": self.error_level.value,
            "message": self.message,
            "user_message": self.user_message,
            "suggestion": self.suggestion,
            "details": self.details,
            "timestamp": self.timestamp.isoformat(),
            "traceback": traceback.format_exc() if sys.exc_info()[0] else None
        }


class WebUILogger:
    """Production-ready logger for web UI with structured logging"""
    
    def __init__(self):
        self.settings = get_settings()
        self.logger = self._setup_logger()
        self.error_buffer: List[Dict[str, Any]] = []
        self.max_buffer_size = 1000
        
    def _setup_logger(self) -> structlog.stdlib.BoundLogger:
        """Setup structured logger with appropriate configuration"""
        # Configure structlog
        structlog.configure(
            processors=[
                structlog.stdlib.filter_by_level,
                structlog.stdlib.add_logger_name,
                structlog.stdlib.add_log_level,
                structlog.stdlib.PositionalArgumentsFormatter(),
                structlog.processors.TimeStamper(fmt="iso"),
                structlog.processors.StackInfoRenderer(),
                structlog.processors.format_exc_info,
                structlog.processors.UnicodeDecoder(),
                structlog.processors.JSONRenderer()
            ],
            context_class=dict,
            logger_factory=structlog.stdlib.LoggerFactory(),
            wrapper_class=structlog.stdlib.BoundLogger,
            cache_logger_on_first_use=True,
        )
        
        # Setup logging configuration
        logging.basicConfig(
            level=getattr(logging, self.settings.monitoring.log_level),
            format="%(message)s",
            handlers=[
                logging.StreamHandler(sys.stdout),
                logging.FileHandler(self.settings.log_path / "web_ui.log")
            ]
        )
        
        return structlog.get_logger("pynomaly.web_ui")
    
    def log_error(
        self,
        error: WebUIError,
        request: Optional[Request] = None,
        user_id: Optional[str] = None,
        additional_context: Optional[Dict[str, Any]] = None
    ):
        """Log error with comprehensive context"""
        context = {
            "error_id": error.error_id,
            "error_code": error.error_code.value,
            "error_level": error.error_level.value,
            "message": error.message,
            "user_message": error.user_message,
            "details": error.details,
            "timestamp": error.timestamp.isoformat()
        }
        
        # Add request context if available
        if request:
            context.update({
                "request_id": getattr(request.state, 'request_id', None),
                "method": request.method,
                "url": str(request.url),
                "user_agent": request.headers.get("user-agent"),
                "ip_address": request.client.host if request.client else None,
                "referer": request.headers.get("referer")
            })
        
        # Add user context
        if user_id:
            context["user_id"] = user_id
        
        # Add additional context
        if additional_context:
            context.update(additional_context)
        
        # Log based on error level
        if error.error_level == ErrorLevel.CRITICAL:
            self.logger.critical("Critical web UI error", **context)
        elif error.error_level == ErrorLevel.ERROR:
            self.logger.error("Web UI error", **context)
        elif error.error_level == ErrorLevel.WARNING:
            self.logger.warning("Web UI warning", **context)
        elif error.error_level == ErrorLevel.INFO:
            self.logger.info("Web UI info", **context)
        else:
            self.logger.debug("Web UI debug", **context)
        
        # Buffer error for analysis
        self._buffer_error(error.to_dict())
    
    def _buffer_error(self, error_dict: Dict[str, Any]):
        """Buffer error for batch processing and analysis"""
        self.error_buffer.append(error_dict)
        
        # Maintain buffer size
        if len(self.error_buffer) > self.max_buffer_size:
            self.error_buffer.pop(0)
    
    def log_performance_issue(
        self,
        metric_name: str,
        value: float,
        threshold: float,
        request: Optional[Request] = None
    ):
        """Log performance issue"""
        error = WebUIError(
            message=f"Performance threshold exceeded: {metric_name}={value} > {threshold}",
            error_code=ErrorCode.PERFORMANCE_THRESHOLD_ERROR,
            error_level=ErrorLevel.WARNING,
            details={
                "metric_name": metric_name,
                "value": value,
                "threshold": threshold
            },
            user_message="The page is loading slowly. Please be patient.",
            suggestion="Consider optimizing the page or checking your internet connection."
        )
        
        self.log_error(error, request)
    
    def log_security_event(
        self,
        event_type: str,
        severity: str,
        details: Dict[str, Any],
        request: Optional[Request] = None
    ):
        """Log security event"""
        error_level = ErrorLevel.CRITICAL if severity == "critical" else ErrorLevel.ERROR
        error = WebUIError(
            message=f"Security event: {event_type}",
            error_code=ErrorCode.SECURITY_VIOLATION_ERROR,
            error_level=error_level,
            details=details,
            user_message="A security issue was detected and has been logged.",
            suggestion="Please ensure you're using the application securely."
        )
        
        self.log_error(error, request)
    
    def get_error_stats(self) -> Dict[str, Any]:
        """Get error statistics for monitoring"""
        if not self.error_buffer:
            return {"total_errors": 0, "error_types": {}, "error_levels": {}}
        
        error_types = {}
        error_levels = {}
        
        for error in self.error_buffer:
            error_code = error.get("error_code")
            error_level = error.get("error_level")
            
            error_types[error_code] = error_types.get(error_code, 0) + 1
            error_levels[error_level] = error_levels.get(error_level, 0) + 1
        
        return {
            "total_errors": len(self.error_buffer),
            "error_types": error_types,
            "error_levels": error_levels,
            "recent_errors": self.error_buffer[-10:] if len(self.error_buffer) > 10 else self.error_buffer
        }


class ErrorHandlingMiddleware(BaseHTTPMiddleware):
    """Middleware for centralized error handling"""
    
    def __init__(self, app, logger: WebUILogger):
        super().__init__(app)
        self.logger = logger
        self.settings = get_settings()
    
    async def dispatch(self, request: Request, call_next) -> Response:
        """Handle request with comprehensive error handling"""
        # Generate request ID for tracking
        request_id = str(uuid.uuid4())
        request.state.request_id = request_id
        
        try:
            # Process request
            response = await call_next(request)
            
            # Log successful requests in debug mode
            if self.settings.app.debug:
                self.logger.logger.debug(
                    "Request processed successfully",
                    request_id=request_id,
                    method=request.method,
                    url=str(request.url),
                    status_code=response.status_code
                )
            
            return response
            
        except WebUIError as e:
            # Handle custom web UI errors
            self.logger.log_error(e, request)
            return await self._create_error_response(e, request)
            
        except HTTPException as e:
            # Handle FastAPI HTTP exceptions
            web_error = WebUIError(
                message=f"HTTP {e.status_code}: {e.detail}",
                error_code=self._get_error_code_from_status(e.status_code),
                error_level=ErrorLevel.ERROR,
                details={"status_code": e.status_code, "detail": e.detail}
            )
            
            self.logger.log_error(web_error, request)
            return await self._create_error_response(web_error, request)
            
        except StarletteHTTPException as e:
            # Handle Starlette HTTP exceptions
            web_error = WebUIError(
                message=f"HTTP {e.status_code}: {e.detail}",
                error_code=self._get_error_code_from_status(e.status_code),
                error_level=ErrorLevel.ERROR,
                details={"status_code": e.status_code, "detail": e.detail}
            )
            
            self.logger.log_error(web_error, request)
            return await self._create_error_response(web_error, request)
            
        except Exception as e:
            # Handle unexpected errors
            web_error = WebUIError(
                message=f"Unexpected error: {str(e)}",
                error_code=ErrorCode.INTERNAL_SERVER_ERROR,
                error_level=ErrorLevel.CRITICAL,
                details={"exception_type": type(e).__name__, "exception_message": str(e)}
            )
            
            self.logger.log_error(web_error, request)
            return await self._create_error_response(web_error, request)
    
    def _get_error_code_from_status(self, status_code: int) -> ErrorCode:
        """Map HTTP status codes to error codes"""
        status_map = {
            400: ErrorCode.VALIDATION_ERROR,
            401: ErrorCode.AUTHENTICATION_ERROR,
            403: ErrorCode.AUTHORIZATION_ERROR,
            404: ErrorCode.NOT_FOUND_ERROR,
            429: ErrorCode.API_RATE_LIMIT_ERROR,
            500: ErrorCode.INTERNAL_SERVER_ERROR,
            502: ErrorCode.EXTERNAL_SERVICE_ERROR,
            503: ErrorCode.EXTERNAL_SERVICE_ERROR,
            504: ErrorCode.API_TIMEOUT_ERROR
        }
        return status_map.get(status_code, ErrorCode.INTERNAL_SERVER_ERROR)
    
    async def _create_error_response(self, error: WebUIError, request: Request) -> Response:
        """Create appropriate error response based on request type"""
        # Determine if request expects JSON or HTML
        accept_header = request.headers.get("accept", "")
        is_api_request = (
            request.url.path.startswith("/api/") or
            "application/json" in accept_header or
            request.headers.get("content-type") == "application/json"
        )
        
        if is_api_request:
            return await self._create_json_error_response(error, request)
        else:
            return await self._create_html_error_response(error, request)
    
    async def _create_json_error_response(self, error: WebUIError, request: Request) -> JSONResponse:
        """Create JSON error response for API requests"""
        error_data = {
            "error": {
                "error_id": error.error_id,
                "error_code": error.error_code.value,
                "message": error.user_message,
                "suggestion": error.suggestion,
                "timestamp": error.timestamp.isoformat()
            }
        }
        
        # Add details in debug mode
        if self.settings.app.debug:
            error_data["error"]["details"] = error.details
            error_data["error"]["technical_message"] = error.message
        
        # Determine status code
        status_code = self._get_status_code_from_error_code(error.error_code)
        
        return JSONResponse(
            status_code=status_code,
            content=error_data,
            headers={"X-Error-ID": error.error_id}
        )
    
    async def _create_html_error_response(self, error: WebUIError, request: Request) -> HTMLResponse:
        """Create HTML error response for web requests"""
        status_code = self._get_status_code_from_error_code(error.error_code)
        
        # Create user-friendly HTML error page
        html_content = self._generate_error_html(error, request, status_code)
        
        return HTMLResponse(
            content=html_content,
            status_code=status_code,
            headers={"X-Error-ID": error.error_id}
        )
    
    def _get_status_code_from_error_code(self, error_code: ErrorCode) -> int:
        """Map error codes to HTTP status codes"""
        status_map = {
            ErrorCode.VALIDATION_ERROR: 400,
            ErrorCode.AUTHENTICATION_ERROR: 401,
            ErrorCode.AUTHORIZATION_ERROR: 403,
            ErrorCode.NOT_FOUND_ERROR: 404,
            ErrorCode.API_RATE_LIMIT_ERROR: 429,
            ErrorCode.INTERNAL_SERVER_ERROR: 500,
            ErrorCode.TEMPLATE_RENDER_ERROR: 500,
            ErrorCode.STATIC_FILE_ERROR: 500,
            ErrorCode.CSRF_TOKEN_ERROR: 403,
            ErrorCode.SESSION_ERROR: 401,
            ErrorCode.API_ENDPOINT_ERROR: 500,
            ErrorCode.API_TIMEOUT_ERROR: 504,
            ErrorCode.PERFORMANCE_THRESHOLD_ERROR: 503,
            ErrorCode.MEMORY_LIMIT_ERROR: 503,
            ErrorCode.SECURITY_VIOLATION_ERROR: 403,
            ErrorCode.XSS_ATTEMPT_ERROR: 400,
            ErrorCode.SQL_INJECTION_ERROR: 400,
            ErrorCode.DATABASE_CONNECTION_ERROR: 503,
            ErrorCode.DATABASE_QUERY_ERROR: 500,
            ErrorCode.EXTERNAL_SERVICE_ERROR: 503,
            ErrorCode.CACHE_ERROR: 500
        }
        return status_map.get(error_code, 500)
    
    def _generate_error_html(self, error: WebUIError, request: Request, status_code: int) -> str:
        """Generate user-friendly HTML error page"""
        return f"""
        <!DOCTYPE html>
        <html lang="en">
        <head>
            <meta charset="UTF-8">
            <meta name="viewport" content="width=device-width, initial-scale=1.0">
            <title>Error {status_code} - Pynomaly</title>
            <script src="https://cdn.tailwindcss.com"></script>
            <style>
                .error-animation {{
                    animation: fadeIn 0.5s ease-in-out;
                }}
                @keyframes fadeIn {{
                    from {{ opacity: 0; transform: translateY(20px); }}
                    to {{ opacity: 1; transform: translateY(0); }}
                }}
            </style>
        </head>
        <body class="bg-gray-50 flex items-center justify-center min-h-screen">
            <div class="error-animation max-w-md w-full mx-auto p-6">
                <div class="bg-white rounded-lg shadow-lg p-8 text-center">
                    <div class="mb-6">
                        <div class="text-6xl text-red-500 mb-4">⚠️</div>
                        <h1 class="text-3xl font-bold text-gray-800 mb-2">Error {status_code}</h1>
                        <p class="text-gray-600 mb-4">{error.user_message}</p>
                        {f'<p class="text-sm text-blue-600">{error.suggestion}</p>' if error.suggestion else ''}
                    </div>
                    
                    <div class="space-y-4">
                        <button onclick="history.back()" 
                                class="w-full bg-blue-600 text-white px-4 py-2 rounded hover:bg-blue-700 transition-colors">
                            Go Back
                        </button>
                        <a href="/" 
                           class="w-full bg-gray-600 text-white px-4 py-2 rounded hover:bg-gray-700 transition-colors inline-block">
                            Return to Home
                        </a>
                        <button onclick="window.location.reload()" 
                                class="w-full bg-green-600 text-white px-4 py-2 rounded hover:bg-green-700 transition-colors">
                            Retry
                        </button>
                    </div>
                    
                    <div class="mt-6 text-xs text-gray-500">
                        <p>Error ID: {error.error_id}</p>
                        <p>Time: {error.timestamp.strftime('%Y-%m-%d %H:%M:%S UTC')}</p>
                    </div>
                </div>
            </div>
        </body>
        </html>
        """


# Global logger instance
_web_ui_logger: Optional[WebUILogger] = None


def get_web_ui_logger() -> WebUILogger:
    """Get global web UI logger instance"""
    global _web_ui_logger
    if _web_ui_logger is None:
        _web_ui_logger = WebUILogger()
    return _web_ui_logger


def log_web_ui_error(
    error: WebUIError,
    request: Optional[Request] = None,
    user_id: Optional[str] = None,
    additional_context: Optional[Dict[str, Any]] = None
):
    """Convenience function to log web UI errors"""
    logger = get_web_ui_logger()
    logger.log_error(error, request, user_id, additional_context)


def create_web_ui_error(
    message: str,
    error_code: ErrorCode,
    error_level: ErrorLevel = ErrorLevel.ERROR,
    details: Optional[Dict[str, Any]] = None,
    user_message: Optional[str] = None,
    suggestion: Optional[str] = None
) -> WebUIError:
    """Convenience function to create web UI errors"""
    return WebUIError(
        message=message,
        error_code=error_code,
        error_level=error_level,
        details=details,
        user_message=user_message,
        suggestion=suggestion
    )