"""Core SDK package for platform client libraries."""

from sdk_core.auth import JWTAuth, TokenAuth
from sdk_core.client import BaseClient, SyncClient
from sdk_core.config import ClientConfig, Environment
from sdk_core.exceptions import (
    SDKError,
    AuthenticationError,
    RateLimitError,
    ValidationError,
    ServerError,
)
from sdk_core.models import (
    BaseResponse,
    ErrorResponse,
    PaginatedResponse,
    HealthResponse,
)

__version__ = "0.1.0"
__all__ = [
    # Authentication
    "JWTAuth",
    "TokenAuth",
    # Clients
    "BaseClient", 
    "SyncClient",
    # Configuration
    "ClientConfig",
    "Environment",
    # Exceptions
    "SDKError",
    "AuthenticationError", 
    "RateLimitError",
    "ValidationError",
    "ServerError",
    # Models
    "BaseResponse",
    "ErrorResponse",
    "PaginatedResponse", 
    "HealthResponse",
]