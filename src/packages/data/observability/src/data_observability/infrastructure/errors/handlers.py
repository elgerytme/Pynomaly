"""Error handlers for FastAPI application."""

import logging
import traceback
from typing import Any, Dict, Union

from fastapi import FastAPI, HTTPException, Request, status
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from sqlalchemy.exc import SQLAlchemyError
from starlette.exceptions import HTTPException as StarletteHTTPException

from .exceptions import (
    AssetNotFoundError,
    AuthenticationError,
    AuthorizationError,
    ConfigurationError,
    DataObservabilityError,
    DatabaseError,
    ExternalSystemError,
    LineageError,
    PipelineError,
    QualityError,
    RateLimitError,
    ServiceError,
    ValidationError,
)

logger = logging.getLogger(__name__)


def log_error(
    error: Exception,
    request: Request,
    extra_context: Dict[str, Any] = None
) -> None:
    """Log error with context information."""
    context = {
        "method": request.method,
        "url": str(request.url),
        "headers": dict(request.headers),
        "error_type": type(error).__name__,
        "error_message": str(error),
    }
    
    if extra_context:
        context.update(extra_context)
    
    # Add user context if available
    if hasattr(request.state, "user"):
        context["user"] = request.state.user.get("username")
    
    # Log stack trace for unexpected errors
    if not isinstance(error, (DataObservabilityError, HTTPException, RequestValidationError)):
        context["traceback"] = traceback.format_exc()
        logger.error("Unexpected error occurred", extra=context)
    else:
        logger.warning("Application error occurred", extra=context)


def create_error_response(
    error: Exception,
    status_code: int,
    include_details: bool = True
) -> JSONResponse:
    """Create standardized error response."""
    if isinstance(error, DataObservabilityError):
        response_data = error.to_dict()
    else:
        response_data = {
            "error": type(error).__name__,
            "message": str(error),
        }
    
    # Add timestamp
    from datetime import datetime
    response_data["timestamp"] = datetime.utcnow().isoformat()
    
    # Remove sensitive details in production if needed
    if not include_details and "details" in response_data:
        response_data.pop("details", None)
    
    return JSONResponse(
        status_code=status_code,
        content=response_data
    )


async def data_observability_exception_handler(
    request: Request,
    exc: DataObservabilityError
) -> JSONResponse:
    """Handle custom data observability exceptions."""
    log_error(exc, request)
    
    # Map specific exceptions to HTTP status codes
    status_code_mapping = {
        AssetNotFoundError: status.HTTP_404_NOT_FOUND,
        ValidationError: status.HTTP_400_BAD_REQUEST,
        AuthenticationError: status.HTTP_401_UNAUTHORIZED,
        AuthorizationError: status.HTTP_403_FORBIDDEN,
        ConfigurationError: status.HTTP_500_INTERNAL_SERVER_ERROR,
        DatabaseError: status.HTTP_500_INTERNAL_SERVER_ERROR,
        ServiceError: status.HTTP_500_INTERNAL_SERVER_ERROR,
        LineageError: status.HTTP_500_INTERNAL_SERVER_ERROR,
        QualityError: status.HTTP_500_INTERNAL_SERVER_ERROR,
        PipelineError: status.HTTP_500_INTERNAL_SERVER_ERROR,
        ExternalSystemError: status.HTTP_502_BAD_GATEWAY,
        RateLimitError: status.HTTP_429_TOO_MANY_REQUESTS,
    }
    
    status_code = status_code_mapping.get(type(exc), status.HTTP_500_INTERNAL_SERVER_ERROR)
    
    return create_error_response(exc, status_code)


async def http_exception_handler(
    request: Request,
    exc: HTTPException
) -> JSONResponse:
    """Handle HTTP exceptions."""
    log_error(exc, request)
    
    response_data = {
        "error": "HTTPException",
        "message": exc.detail,
        "timestamp": __import__("datetime").datetime.utcnow().isoformat()
    }
    
    return JSONResponse(
        status_code=exc.status_code,
        content=response_data
    )


async def validation_exception_handler(
    request: Request,
    exc: RequestValidationError
) -> JSONResponse:
    """Handle request validation errors."""
    log_error(exc, request)
    
    # Format validation errors
    errors = []
    for error in exc.errors():
        field_path = " -> ".join(str(x) for x in error["loc"])
        errors.append({
            "field": field_path,
            "message": error["msg"],
            "type": error["type"],
        })
    
    response_data = {
        "error": "VALIDATION_ERROR",
        "message": "Request validation failed",
        "details": {"validation_errors": errors},
        "timestamp": __import__("datetime").datetime.utcnow().isoformat()
    }
    
    return JSONResponse(
        status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
        content=response_data
    )


async def sqlalchemy_exception_handler(
    request: Request,
    exc: SQLAlchemyError
) -> JSONResponse:
    """Handle SQLAlchemy database errors."""
    # Convert to our custom DatabaseError
    database_error = DatabaseError(
        message="Database operation failed",
        cause=exc,
        details={"database_error": str(exc)}
    )
    
    log_error(database_error, request)
    
    return create_error_response(
        database_error,
        status.HTTP_500_INTERNAL_SERVER_ERROR
    )


async def generic_exception_handler(
    request: Request,
    exc: Exception
) -> JSONResponse:
    """Handle unexpected exceptions."""
    log_error(exc, request)
    
    # Don't expose internal error details in production
    response_data = {
        "error": "INTERNAL_SERVER_ERROR",
        "message": "An unexpected error occurred",
        "timestamp": __import__("datetime").datetime.utcnow().isoformat()
    }
    
    # Add debug information in development
    try:
        from ..config.settings import settings
        if settings.is_development():
            response_data["debug"] = {
                "error_type": type(exc).__name__,
                "error_message": str(exc),
                "traceback": traceback.format_exc().split("\n")
            }
    except Exception:
        # If we can't get settings, assume production
        pass
    
    return JSONResponse(
        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        content=response_data
    )


def setup_error_handlers(app: FastAPI) -> None:
    """Setup error handlers for the FastAPI application."""
    
    # Custom exception handlers
    app.add_exception_handler(
        DataObservabilityError,
        data_observability_exception_handler
    )
    
    # Built-in exception handlers
    app.add_exception_handler(
        HTTPException,
        http_exception_handler
    )
    
    app.add_exception_handler(
        StarletteHTTPException,
        http_exception_handler
    )
    
    app.add_exception_handler(
        RequestValidationError,
        validation_exception_handler
    )
    
    # Database exception handler
    app.add_exception_handler(
        SQLAlchemyError,
        sqlalchemy_exception_handler
    )
    
    # Catch-all exception handler
    app.add_exception_handler(
        Exception,
        generic_exception_handler
    )
    
    logger.info("Error handlers configured successfully")