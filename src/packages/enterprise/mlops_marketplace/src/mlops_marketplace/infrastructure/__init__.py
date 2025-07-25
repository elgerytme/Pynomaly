"""
Infrastructure layer for the MLOps Marketplace.

Contains adapters, external service integrations, persistence implementations,
and infrastructure concerns like API gateways, message queues, and caching.
"""

from mlops_marketplace.infrastructure.api import (
    MarketplaceAPIClient,
    APIGateway,
    RateLimiter,
    AuthenticationMiddleware,
)

from mlops_marketplace.infrastructure.sdk import (
    MarketplaceSDK,
    PythonSDK,
    JavaScriptSDK,
)

from mlops_marketplace.infrastructure.persistence import (
    PostgreSQLSolutionRepository,
    PostgreSQLUserRepository,
    PostgreSQLSubscriptionRepository,
    RedisCache,
    ElasticsearchIndex,
)

from mlops_marketplace.infrastructure.external import (
    StripePaymentGateway,
    SendGridNotificationService,
    ElasticsearchSearchEngine,
    MLflowModelRegistry,
    KubernetesDeploymentService,
)

from mlops_marketplace.infrastructure.monitoring import (
    PrometheusMetrics,
    StructlogLogger,
    OpenTelemetryTracer,
)

from mlops_marketplace.infrastructure.security import (
    JWTAuthenticationService,
    OAuth2Integration,
    APIKeyManager,
    SecurityScanner,
)

__all__ = [
    # API and Gateway
    "MarketplaceAPIClient",
    "APIGateway",
    "RateLimiter",
    "AuthenticationMiddleware",
    
    # SDK
    "MarketplaceSDK",
    "PythonSDK",
    "JavaScriptSDK",
    
    # Persistence
    "PostgreSQLSolutionRepository",
    "PostgreSQLUserRepository",
    "PostgreSQLSubscriptionRepository",
    "RedisCache",
    "ElasticsearchIndex",
    
    # External Services
    "StripePaymentGateway",
    "SendGridNotificationService",
    "ElasticsearchSearchEngine",
    "MLflowModelRegistry",
    "KubernetesDeploymentService",
    
    # Monitoring
    "PrometheusMetrics",
    "StructlogLogger",
    "OpenTelemetryTracer",
    
    # Security
    "JWTAuthenticationService",
    "OAuth2Integration",
    "APIKeyManager",
    "SecurityScanner",
]