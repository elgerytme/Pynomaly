"""
Provider entities for the MLOps Marketplace.

Defines entities related to solution providers who publish and maintain
solutions in the marketplace.
"""

from datetime import datetime
from decimal import Decimal
from enum import Enum
from typing import Dict, List, Optional, Set
from uuid import UUID, uuid4

from pydantic import BaseModel, Field, validator

from mlops_marketplace.domain.value_objects import ProviderId


class ProviderType(str, Enum):
    """Types of solution providers."""
    INDIVIDUAL = "individual"
    ORGANIZATION = "organization"
    ENTERPRISE = "enterprise"
    ACADEMIC = "academic"
    OPEN_SOURCE = "open_source"


class ProviderStatus(str, Enum):
    """Status of a provider account."""
    PENDING = "pending"
    ACTIVE = "active"
    SUSPENDED = "suspended"
    DEACTIVATED = "deactivated"
    BANNED = "banned"


class ProviderTier(str, Enum):
    """Provider tier levels with different privileges."""
    BASIC = "basic"
    PREMIUM = "premium"
    ENTERPRISE = "enterprise"
    PARTNER = "partner"


class VerificationLevel(str, Enum):
    """Provider verification levels."""
    UNVERIFIED = "unverified"
    EMAIL_VERIFIED = "email_verified"
    IDENTITY_VERIFIED = "identity_verified"
    BUSINESS_VERIFIED = "business_verified"
    PARTNER_VERIFIED = "partner_verified"


class ProviderMetrics(BaseModel):
    """Metrics and statistics for a provider."""
    
    # Solution metrics
    total_solutions: int = Field(default=0, ge=0)
    published_solutions: int = Field(default=0, ge=0)
    featured_solutions: int = Field(default=0, ge=0)
    
    # Engagement metrics
    total_downloads: int = Field(default=0, ge=0)
    total_deployments: int = Field(default=0, ge=0)
    total_reviews: int = Field(default=0, ge=0)
    average_rating: float = Field(default=0.0, ge=0, le=5)
    
    # Revenue metrics
    total_revenue: Decimal = Field(default=Decimal('0.00'), ge=0)
    monthly_revenue: Decimal = Field(default=Decimal('0.00'), ge=0)
    active_subscriptions: int = Field(default=0, ge=0)
    
    # Quality metrics
    average_quality_score: float = Field(default=0.0, ge=0, le=100)
    certification_count: int = Field(default=0, ge=0)
    support_response_time: Optional[float] = None  # in hours
    
    # Activity metrics
    last_activity_at: Optional[datetime] = None
    solutions_updated_last_30_days: int = Field(default=0, ge=0)
    
    def update_solution_metrics(
        self, 
        solutions: int = 0, 
        published: int = 0, 
        featured: int = 0
    ) -> None:
        """Update solution count metrics."""
        self.total_solutions += solutions
        self.published_solutions += published
        self.featured_solutions += featured
    
    def update_engagement_metrics(
        self, 
        downloads: int = 0, 
        deployments: int = 0,
        reviews: int = 0,
        rating: Optional[float] = None
    ) -> None:
        """Update engagement metrics."""
        self.total_downloads += downloads
        self.total_deployments += deployments
        self.total_reviews += reviews
        
        if rating is not None and rating > 0:
            # Update average rating
            if self.total_reviews > 0:
                total_rating_points = self.average_rating * (self.total_reviews - reviews)
                total_rating_points += rating
                self.average_rating = total_rating_points / self.total_reviews
            else:
                self.average_rating = rating
    
    def update_revenue_metrics(self, revenue: Decimal, monthly: Decimal) -> None:
        """Update revenue metrics."""
        self.total_revenue += revenue
        self.monthly_revenue = monthly
    
    class Config:
        """Pydantic configuration."""
        json_encoders = {
            Decimal: str,
            datetime: lambda v: v.isoformat(),
        }


class ProviderProfile(BaseModel):
    """Extended profile information for providers."""
    
    # Basic information
    display_name: str = Field(..., min_length=1, max_length=100)
    bio: Optional[str] = Field(None, max_length=1000)
    avatar_url: Optional[str] = None
    banner_url: Optional[str] = None
    website_url: Optional[str] = None
    
    # Location and contact
    country: Optional[str] = None
    timezone: Optional[str] = None
    languages: List[str] = Field(default_factory=list)
    
    # Social links
    github_url: Optional[str] = None
    linkedin_url: Optional[str] = None
    twitter_url: Optional[str] = None
    
    # Organization details (for non-individual providers)
    organization_size: Optional[str] = None
    industry: Optional[str] = None
    founded_year: Optional[int] = None
    
    # Expertise and specializations
    expertise_areas: List[str] = Field(default_factory=list)
    specializations: List[str] = Field(default_factory=list)
    certifications: List[str] = Field(default_factory=list)
    
    # Support information
    support_email: Optional[str] = None
    support_url: Optional[str] = None
    documentation_url: Optional[str] = None
    
    @validator('languages', 'expertise_areas', 'specializations')
    def validate_string_lists(cls, v: List[str]) -> List[str]:
        """Validate string lists."""
        return [item.strip() for item in v if item and item.strip()]
    
    @validator('founded_year')
    def validate_founded_year(cls, v: Optional[int]) -> Optional[int]:
        """Validate founded year."""
        if v is not None:
            current_year = datetime.utcnow().year
            if v < 1900 or v > current_year:
                raise ValueError(f"Founded year must be between 1900 and {current_year}")
        return v


class SolutionProvider(BaseModel):
    """Solution provider entity."""
    
    id: ProviderId
    
    # Account information
    email: str = Field(..., regex=r"^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$")
    username: str = Field(..., min_length=3, max_length=50, regex=r"^[a-zA-Z0-9_-]+$")
    provider_type: ProviderType
    tier: ProviderTier = Field(default=ProviderTier.BASIC)
    
    # Status and verification
    status: ProviderStatus = Field(default=ProviderStatus.PENDING)
    verification_level: VerificationLevel = Field(default=VerificationLevel.UNVERIFIED)
    is_verified: bool = Field(default=False)
    is_featured: bool = Field(default=False)
    
    # Profile and metrics
    profile: ProviderProfile
    metrics: ProviderMetrics = Field(default_factory=ProviderMetrics)
    
    # Settings and preferences
    public_profile: bool = Field(default=True)
    email_notifications: bool = Field(default=True)
    marketing_emails: bool = Field(default=False)
    
    # Financial information
    stripe_account_id: Optional[str] = None
    tax_id: Optional[str] = None
    billing_address: Dict[str, str] = Field(default_factory=dict)
    
    # API access
    api_keys: List[UUID] = Field(default_factory=list)
    rate_limit_per_hour: int = Field(default=1000)
    
    # Timestamps
    created_at: datetime = Field(default_factory=datetime.utcnow)
    updated_at: datetime = Field(default_factory=datetime.utcnow)
    last_login_at: Optional[datetime] = None
    verified_at: Optional[datetime] = None
    
    def verify_provider(self, verification_level: VerificationLevel) -> None:
        """Verify the provider at a specific level."""
        self.verification_level = verification_level
        self.is_verified = True
        self.verified_at = datetime.utcnow()
        self.updated_at = datetime.utcnow()
        
        # Upgrade tier if applicable
        if verification_level in [VerificationLevel.BUSINESS_VERIFIED, VerificationLevel.PARTNER_VERIFIED]:
            if self.tier == ProviderTier.BASIC:
                self.tier = ProviderTier.PREMIUM
    
    def upgrade_tier(self, new_tier: ProviderTier) -> None:
        """Upgrade provider tier."""
        if new_tier.value > self.tier.value:  # Assuming enum values are ordered
            self.tier = new_tier
            self.updated_at = datetime.utcnow()
            
            # Update rate limits based on tier
            rate_limits = {
                ProviderTier.BASIC: 1000,
                ProviderTier.PREMIUM: 5000,
                ProviderTier.ENTERPRISE: 20000,
                ProviderTier.PARTNER: 50000,
            }
            self.rate_limit_per_hour = rate_limits.get(new_tier, 1000)
    
    def suspend(self, reason: str = "") -> None:
        """Suspend the provider account."""
        self.status = ProviderStatus.SUSPENDED
        self.updated_at = datetime.utcnow()
    
    def activate(self) -> None:
        """Activate the provider account."""
        self.status = ProviderStatus.ACTIVE
        self.updated_at = datetime.utcnow()
    
    def update_login(self) -> None:
        """Update last login timestamp."""
        self.last_login_at = datetime.utcnow()
        self.metrics.last_activity_at = datetime.utcnow()
    
    def add_api_key(self, key_id: UUID) -> None:
        """Add a new API key."""
        if key_id not in self.api_keys:
            self.api_keys.append(key_id)
            self.updated_at = datetime.utcnow()
    
    def remove_api_key(self, key_id: UUID) -> None:
        """Remove an API key."""
        if key_id in self.api_keys:
            self.api_keys.remove(key_id)
            self.updated_at = datetime.utcnow()
    
    def can_publish_solutions(self) -> bool:
        """Check if provider can publish solutions."""
        return (
            self.status == ProviderStatus.ACTIVE
            and self.is_verified
            and self.verification_level != VerificationLevel.UNVERIFIED
        )
    
    def can_create_premium_solutions(self) -> bool:
        """Check if provider can create premium solutions."""
        return (
            self.can_publish_solutions()
            and self.tier in [ProviderTier.PREMIUM, ProviderTier.ENTERPRISE, ProviderTier.PARTNER]
        )
    
    def get_solution_limit(self) -> int:
        """Get the maximum number of solutions this provider can create."""
        limits = {
            ProviderTier.BASIC: 5,
            ProviderTier.PREMIUM: 50,
            ProviderTier.ENTERPRISE: 500,
            ProviderTier.PARTNER: -1,  # Unlimited
        }
        return limits.get(self.tier, 5)
    
    def has_reached_solution_limit(self) -> bool:
        """Check if provider has reached their solution creation limit."""
        limit = self.get_solution_limit()
        return limit != -1 and self.metrics.total_solutions >= limit
    
    @validator('username')
    def validate_username(cls, v: str) -> str:
        """Validate username format."""
        if not v or len(v.strip()) < 3:
            raise ValueError("Username must be at least 3 characters long")
        
        # Check for reserved usernames
        reserved = {'admin', 'api', 'www', 'mail', 'ftp', 'localhost', 'marketplace'}
        if v.lower() in reserved:
            raise ValueError("Username is reserved")
        
        return v.strip().lower()
    
    @validator('billing_address')
    def validate_billing_address(cls, v: Dict[str, str]) -> Dict[str, str]:
        """Validate billing address."""
        if v:
            required_fields = {'street', 'city', 'country', 'postal_code'}
            if not all(field in v for field in required_fields):
                raise ValueError("Billing address missing required fields")
        return v
    
    class Config:
        """Pydantic configuration."""
        json_encoders = {
            UUID: str,
            datetime: lambda v: v.isoformat(),
            Decimal: str,
        }