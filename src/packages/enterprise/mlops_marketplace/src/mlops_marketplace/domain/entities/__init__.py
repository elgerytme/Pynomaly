"""
Domain entities for the MLOps Marketplace.

Contains the core business entities that represent the fundamental concepts
and data structures of the marketplace domain.
"""

from mlops_marketplace.domain.entities.solution import (
    Solution,
    SolutionVersion,
    SolutionCategory,
    SolutionMetadata,
    SolutionDependency,
)

from mlops_marketplace.domain.entities.provider import (
    SolutionProvider,
    ProviderProfile,
    ProviderMetrics,
    ProviderTier,
)

from mlops_marketplace.domain.entities.user import (
    MarketplaceUser,
    UserProfile,
    UserPreferences,
    UserRole,
)

from mlops_marketplace.domain.entities.commerce import (
    Subscription,
    Transaction,
    PaymentMethod,
    Invoice,
    Discount,
)

from mlops_marketplace.domain.entities.quality import (
    Review,
    Certification,
    QualityReport,
    SecurityScan,
    PerformanceReport,
)

from mlops_marketplace.domain.entities.deployment import (
    Deployment,
    DeploymentConfig,
    DeploymentStatus,
    ApiKey,
)

from mlops_marketplace.domain.entities.analytics import (
    UsageMetrics,
    PopularityMetrics,
    RevenueMetrics,
    PerformanceAnalytics,
)

__all__ = [
    # Solution entities
    "Solution",
    "SolutionVersion",
    "SolutionCategory",
    "SolutionMetadata",
    "SolutionDependency",
    
    # Provider entities
    "SolutionProvider",
    "ProviderProfile",
    "ProviderMetrics",
    "ProviderTier",
    
    # User entities
    "MarketplaceUser",
    "UserProfile",
    "UserPreferences",
    "UserRole",
    
    # Commerce entities
    "Subscription",
    "Transaction",
    "PaymentMethod",
    "Invoice",
    "Discount",
    
    # Quality entities
    "Review",
    "Certification",
    "QualityReport",
    "SecurityScan",
    "PerformanceReport",
    
    # Deployment entities
    "Deployment",
    "DeploymentConfig",
    "DeploymentStatus",
    "ApiKey",
    
    # Analytics entities
    "UsageMetrics",
    "PopularityMetrics",
    "RevenueMetrics",
    "PerformanceAnalytics",
]