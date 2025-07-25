"""
Domain layer for the MLOps Marketplace.

Contains the core business logic, entities, value objects, and domain services
that define the marketplace's business rules and operations.
"""

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
    QualityReport,
    ApiKey,
    Deployment,
)

from mlops_marketplace.domain.value_objects import (
    SolutionId,
    ProviderId,
    UserId,
    Price,
    Rating,
    Version,
    TechnicalSpecification,
    ComplianceRequirement,
    PerformanceMetric,
)

from mlops_marketplace.domain.services import (
    SolutionValidationService,
    PricingCalculationService,
    RecommendationService,
    QualityAssessmentService,
    CompatibilityCheckService,
)

from mlops_marketplace.domain.repositories import (
    SolutionRepository,
    ProviderRepository,
    UserRepository,
    SubscriptionRepository,
    TransactionRepository,
    ReviewRepository,
    CertificationRepository,
)

from mlops_marketplace.domain.interfaces import (
    PaymentGateway,
    NotificationService,
    SearchEngine,
    SecurityScanner,
    ModelRegistry,
    QualityAnalyzer,
)

__all__ = [
    # Entities
    "Solution",
    "SolutionVersion",
    "SolutionCategory",
    "SolutionProvider",
    "MarketplaceUser",
    "Subscription",
    "Transaction",
    "Review",
    "Certification",
    "QualityReport",
    "ApiKey",
    "Deployment",
    
    # Value objects
    "SolutionId",
    "ProviderId",
    "UserId",
    "Price",
    "Rating",
    "Version",
    "TechnicalSpecification",
    "ComplianceRequirement",
    "PerformanceMetric",
    
    # Domain services
    "SolutionValidationService",
    "PricingCalculationService",
    "RecommendationService",
    "QualityAssessmentService",
    "CompatibilityCheckService",
    
    # Repositories
    "SolutionRepository",
    "ProviderRepository",
    "UserRepository",
    "SubscriptionRepository",
    "TransactionRepository",
    "ReviewRepository",
    "CertificationRepository",
    
    # Interfaces
    "PaymentGateway",
    "NotificationService",
    "SearchEngine",
    "SecurityScanner",
    "ModelRegistry",
    "QualityAnalyzer",
]