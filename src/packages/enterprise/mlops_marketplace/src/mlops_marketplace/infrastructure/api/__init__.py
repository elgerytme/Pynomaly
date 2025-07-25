"""
API infrastructure for the MLOps Marketplace.

Contains API gateway, client implementations, middleware, and
API-related infrastructure components.
"""

from mlops_marketplace.infrastructure.api.gateway import APIGateway
from mlops_marketplace.infrastructure.api.client import MarketplaceAPIClient
from mlops_marketplace.infrastructure.api.middleware import (
    AuthenticationMiddleware,
    RateLimitingMiddleware,
    LoggingMiddleware,
    CORSMiddleware,
)
from mlops_marketplace.infrastructure.api.rate_limiter import RateLimiter
from mlops_marketplace.infrastructure.api.auth import (
    JWTAuthHandler,
    APIKeyAuthHandler,
    OAuth2Handler,
)

__all__ = [
    "APIGateway",
    "MarketplaceAPIClient",
    "AuthenticationMiddleware",
    "RateLimitingMiddleware",
    "LoggingMiddleware",
    "CORSMiddleware",
    "RateLimiter",
    "JWTAuthHandler",
    "APIKeyAuthHandler",
    "OAuth2Handler",
]