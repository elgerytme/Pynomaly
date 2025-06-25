"""API documentation configuration and utilities."""

from .openapi_config import (
    OpenAPIConfig,
    configure_openapi_docs,
    get_custom_redoc_html,
    get_custom_swagger_ui_html,
)
from .response_models import (
    ErrorResponse,
    HealthResponse,
    PaginationResponse,
    SuccessResponse,
)
from .schema_examples import SchemaExamples

__all__ = [
    "OpenAPIConfig",
    "configure_openapi_docs",
    "get_custom_redoc_html",
    "get_custom_swagger_ui_html",
    "ErrorResponse",
    "HealthResponse",
    "PaginationResponse",
    "SuccessResponse",
    "SchemaExamples",
]
