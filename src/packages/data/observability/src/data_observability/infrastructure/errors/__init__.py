"""Error handling module for data observability."""

from .exceptions import (
    DataObservabilityError,
    AssetNotFoundError,
    ValidationError,
    ConfigurationError,
    DatabaseError,
    ServiceError,
    AuthenticationError,
    AuthorizationError,
)
from .handlers import (
    setup_error_handlers,
    create_error_response,
    log_error,
)

__all__ = [
    "DataObservabilityError",
    "AssetNotFoundError",
    "ValidationError",
    "ConfigurationError",
    "DatabaseError",
    "ServiceError",
    "AuthenticationError",
    "AuthorizationError",
    "setup_error_handlers",
    "create_error_response",
    "log_error",
]