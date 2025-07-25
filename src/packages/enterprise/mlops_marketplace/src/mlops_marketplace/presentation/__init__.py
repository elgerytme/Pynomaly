"""
Presentation layer for the MLOps Marketplace.

Contains API endpoints, web interfaces, CLI commands, and other
presentation layer components that handle user interactions.
"""

from mlops_marketplace.presentation.api import (
    MarketplaceAPI,
    SolutionsRouter,
    DeploymentsRouter,
    SubscriptionsRouter,
    ReviewsRouter,
    AnalyticsRouter,
    AdminRouter,
)

from mlops_marketplace.presentation.web import (
    MarketplaceWebApp,
    DashboardPages,
    DeveloperPortalPages,
)

from mlops_marketplace.presentation.cli import (
    MarketplaceCLI,
    SolutionCommands,
    DeploymentCommands,
    AnalyticsCommands,
)

__all__ = [
    # API
    "MarketplaceAPI",
    "SolutionsRouter",
    "DeploymentsRouter",
    "SubscriptionsRouter",
    "ReviewsRouter",
    "AnalyticsRouter",
    "AdminRouter",
    
    # Web
    "MarketplaceWebApp",
    "DashboardPages",
    "DeveloperPortalPages",
    
    # CLI
    "MarketplaceCLI",
    "SolutionCommands",
    "DeploymentCommands",
    "AnalyticsCommands",
]