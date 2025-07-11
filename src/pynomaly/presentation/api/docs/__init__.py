"""API documentation configuration and utilities."""

from .common_responses import COMMON_RESPONSES, ENDPOINT_METADATA, configure_api_docs
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
    "COMMON_RESPONSES",
    "ENDPOINT_METADATA",
    "configure_api_docs",
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
