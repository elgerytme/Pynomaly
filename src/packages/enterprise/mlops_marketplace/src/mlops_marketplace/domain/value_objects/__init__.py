"""
Value objects for the MLOps Marketplace domain.

Contains immutable value objects that represent business concepts
without identity, used throughout the domain model.
"""

from mlops_marketplace.domain.value_objects.identifiers import (
    SolutionId,
    ProviderId,
    UserId,
    SubscriptionId,
    TransactionId,
    ReviewId,
    CertificationId,
    DeploymentId,
)

from mlops_marketplace.domain.value_objects.pricing import (
    Price,
    PricingModel,
    Discount,
    BillingCycle,
)

from mlops_marketplace.domain.value_objects.rating import (
    Rating,
    RatingBreakdown,
)

from mlops_marketplace.domain.value_objects.version import (
    Version,
    VersionRange,
)

from mlops_marketplace.domain.value_objects.technical import (
    TechnicalSpecification,
    SystemRequirements,
    PerformanceMetric,
    ResourceRequirement,
)

from mlops_marketplace.domain.value_objects.compliance import (
    ComplianceRequirement,
    SecurityStandard,
    CertificationLevel,
)

from mlops_marketplace.domain.value_objects.geographic import (
    Country,
    Region,
    TimeZone,
)

from mlops_marketplace.domain.value_objects.contact import (
    EmailAddress,
    PhoneNumber,
    Address,
)

__all__ = [
    # Identifiers
    "SolutionId",
    "ProviderId",
    "UserId",
    "SubscriptionId",
    "TransactionId",
    "ReviewId",
    "CertificationId",
    "DeploymentId",
    
    # Pricing
    "Price",
    "PricingModel",
    "Discount",
    "BillingCycle",
    
    # Rating
    "Rating",
    "RatingBreakdown",
    
    # Version
    "Version",
    "VersionRange",
    
    # Technical
    "TechnicalSpecification",
    "SystemRequirements",
    "PerformanceMetric",
    "ResourceRequirement",
    
    # Compliance
    "ComplianceRequirement",
    "SecurityStandard",
    "CertificationLevel",
    
    # Geographic
    "Country",
    "Region",
    "TimeZone",
    
    # Contact
    "EmailAddress",
    "PhoneNumber",
    "Address",
]