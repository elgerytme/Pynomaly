"""
Enterprise MLOps Marketplace and Ecosystem Platform.

A comprehensive marketplace for ML solutions, models, and tools with enterprise-grade
features including solution catalog, developer portal, quality assurance, and
monetization capabilities.
"""

__version__ = "1.0.0"
__author__ = "AI Platform Team"
__email__ = "ai-platform@company.com"

from mlops_marketplace.domain.entities import (
    Solution,
    SolutionVersion,
    SolutionCategory,
    SolutionProvider,
    MarketplaceUser,
    Subscription,
    Transaction,
    Review,
    Certification,
)

from mlops_marketplace.domain.value_objects import (
    SolutionId,
    ProviderId,
    UserId,
    Price,
    Rating,
    Version,
)

from mlops_marketplace.application.services import (
    SolutionCatalogService,
    DeveloperPortalService,
    QualityAssuranceService,
    MonetizationService,
    MarketplaceAnalyticsService,
)

from mlops_marketplace.infrastructure.api import MarketplaceAPIClient
from mlops_marketplace.infrastructure.sdk import MarketplaceSDK

__all__ = [
    # Core domain entities
    "Solution",
    "SolutionVersion",
    "SolutionCategory",
    "SolutionProvider",
    "MarketplaceUser",
    "Subscription",
    "Transaction",
    "Review",
    "Certification",
    
    # Value objects
    "SolutionId",
    "ProviderId",
    "UserId",
    "Price",
    "Rating",
    "Version",
    
    # Application services
    "SolutionCatalogService",
    "DeveloperPortalService",
    "QualityAssuranceService",
    "MonetizationService",
    "MarketplaceAnalyticsService",
    
    # Infrastructure
    "MarketplaceAPIClient",
    "MarketplaceSDK",
]

# Package metadata
__title__ = "MLOps Marketplace"
__description__ = "Enterprise MLOps Marketplace and Ecosystem Platform"
__url__ = "https://github.com/company/monorepo"
__license__ = "MIT"
__copyright__ = "Copyright 2024 AI Platform Team"