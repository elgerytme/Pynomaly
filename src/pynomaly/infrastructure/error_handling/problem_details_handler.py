"""RFC 7807 Problem Details error handlers for FastAPI."""

from __future__ import annotations

import logging
from typing import Any, Dict, Optional, Union

from fastapi import FastAPI, HTTPException, Request, status
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from pydantic import ValidationError as PydanticValidationError
from starlette.exceptions import HTTPException as StarletteHTTPException

logger = logging.getLogger(__name__)


class ProblemDetailsResponse(JSONResponse):
    """JSON response that conforms to RFC 7807 Problem Details."""

    def __init__(
        self,
        status_code: int,
        title: str,
        detail: Optional[str] = None,
        type_: str = "about:blank",
        instance: Optional[str] = None,
        extensions: Optional[Dict[str, Any]] = None,
        correlation_id: Optional[str] = None,
    ) -> None:
        """Initialize problem details response.
        
        Args:
            status_code: HTTP status code
            title: Short, human-readable summary of the problem
            detail: Human-readable explanation specific to this occurrence
            type_: URI reference that identifies the problem type
            instance: URI reference that identifies the specific occurrence
            extensions: Additional problem-specific information
            correlation_id: Correlation ID for request tracking
        """
        content = {
            "type": type_,
            "title": title,
            "status": status_code,
        }
        
        if detail:
            content["detail"] = detail
        if instance:
            content["instance"] = instance
        if extensions:
            content.update(extensions)
        
        headers = {}
        if correlation_id:
            headers["X-Correlation-ID"] = correlation_id
            
        super().__init__(content=content, status_code=status_code, headers=headers)


def add_exception_handlers(app: FastAPI) -> None:
    """Add RFC 7807 Problem Details exception handlers to FastAPI app."""

    @app.exception_handler(StarletteHTTPException)
    async def starlette_http_exception_handler(
        request: Request, exc: StarletteHTTPException
    ) -> ProblemDetailsResponse:
        """Handle Starlette HTTP exceptions."""
        return ProblemDetailsResponse(
            status_code=exc.status_code,
            title=_get_status_title(exc.status_code),
            detail=exc.detail,
            instance=str(request.url),
            correlation_id=getattr(request.state, "correlation_id", None),
        )

    @app.exception_handler(HTTPException)
    async def fastapi_http_exception_handler(
        request: Request, exc: HTTPException
    ) -> ProblemDetailsResponse:
        """Handle FastAPI HTTP exceptions."""
        return ProblemDetailsResponse(
            status_code=exc.status_code,
            title=_get_status_title(exc.status_code),
            detail=exc.detail,
            instance=str(request.url),
            correlation_id=getattr(request.state, "correlation_id", None),
        )

    @app.exception_handler(RequestValidationError)
    async def request_validation_exception_handler(
        request: Request, exc: RequestValidationError
    ) -> ProblemDetailsResponse:
        """Handle FastAPI request validation errors."""
        # Format validation errors for better readability
        errors = []
        for error in exc.errors():
            loc = " -> ".join(str(x) for x in error["loc"])
            errors.append(f"{loc}: {error['msg']}")
        
        return ProblemDetailsResponse(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            title="Validation Error",
            detail=f"Input validation failed: {'; '.join(errors)}",
            type_="https://tools.ietf.org/html/rfc7231#section-6.5.1",
            instance=str(request.url),
            extensions={"validation_errors": exc.errors()},
            correlation_id=getattr(request.state, "correlation_id", None),
        )

    @app.exception_handler(PydanticValidationError)
    async def validation_exception_handler(
        request: Request, exc: PydanticValidationError
    ) -> ProblemDetailsResponse:
        """Handle Pydantic validation errors."""
        # Format validation errors for better readability
        errors = []
        for error in exc.errors():
            loc = " -> ".join(str(x) for x in error["loc"])
            errors.append(f"{loc}: {error['msg']}")
        
        return ProblemDetailsResponse(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            title="Validation Error",
            detail=f"Input validation failed: {'; '.join(errors)}",
            type_="https://tools.ietf.org/html/rfc7231#section-6.5.1",
            instance=str(request.url),
            extensions={"validation_errors": exc.errors()},
            correlation_id=getattr(request.state, "correlation_id", None),
        )

    @app.exception_handler(Exception)
    async def general_exception_handler(
        request: Request, exc: Exception
    ) -> ProblemDetailsResponse:
        """Handle all other exceptions."""
        logger.error(
            f"Unhandled exception: {type(exc).__name__}: {str(exc)}",
            exc_info=True,
            extra={"correlation_id": getattr(request.state, "correlation_id", None)},
        )
        
        return ProblemDetailsResponse(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            title="Internal Server Error",
            detail="An unexpected error occurred. Please try again later.",
            instance=str(request.url),
            correlation_id=getattr(request.state, "correlation_id", None),
        )


def _get_status_title(status_code: int) -> str:
    """Get a human-readable title for HTTP status codes."""
    status_titles = {
        400: "Bad Request",
        401: "Unauthorized",
        403: "Forbidden",
        404: "Not Found",
        405: "Method Not Allowed",
        406: "Not Acceptable",
        409: "Conflict",
        410: "Gone",
        422: "Unprocessable Entity",
        429: "Too Many Requests",
        500: "Internal Server Error",
        501: "Not Implemented",
        502: "Bad Gateway",
        503: "Service Unavailable",
        504: "Gateway Timeout",
    }
    return status_titles.get(status_code, "HTTP Error")


def create_problem_details_response(
    request: Request,
    status_code: int,
    title: str,
    detail: Optional[str] = None,
    type_: str = "about:blank",
    extensions: Optional[Dict[str, Any]] = None,
) -> ProblemDetailsResponse:
    """Create a problem details response.
    
    Args:
        request: The FastAPI request object
        status_code: HTTP status code
        title: Short, human-readable summary of the problem
        detail: Human-readable explanation specific to this occurrence
        type_: URI reference that identifies the problem type
        extensions: Additional problem-specific information
        
    Returns:
        ProblemDetailsResponse instance
    """
    return ProblemDetailsResponse(
        status_code=status_code,
        title=title,
        detail=detail,
        type_=type_,
        instance=str(request.url),
        extensions=extensions,
        correlation_id=getattr(request.state, "correlation_id", None),
    )
