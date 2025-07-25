"""
Solution entities for the MLOps Marketplace.

Defines the core solution entities that represent ML solutions, models, and tools
available in the marketplace.
"""

from datetime import datetime
from decimal import Decimal
from enum import Enum
from typing import Any, Dict, List, Optional, Set
from uuid import UUID, uuid4

from pydantic import BaseModel, Field, validator

from mlops_marketplace.domain.value_objects import (
    SolutionId,
    ProviderId,
    Price,
    Version,
    TechnicalSpecification,
)


class SolutionType(str, Enum):
    """Types of solutions available in the marketplace."""
    MODEL = "model"
    ALGORITHM = "algorithm"
    PIPELINE = "pipeline"
    FRAMEWORK = "framework"
    TOOL = "tool"
    DATASET = "dataset"
    CONNECTOR = "connector"
    TEMPLATE = "template"


class SolutionStatus(str, Enum):
    """Status of a solution in the marketplace."""
    DRAFT = "draft"
    PENDING_REVIEW = "pending_review"
    UNDER_REVIEW = "under_review"
    APPROVED = "approved"
    PUBLISHED = "published"
    DEPRECATED = "deprecated"
    SUSPENDED = "suspended"
    ARCHIVED = "archived"


class LicenseType(str, Enum):
    """License types for solutions."""
    MIT = "mit"
    APACHE_2_0 = "apache_2_0"
    BSD_3_CLAUSE = "bsd_3_clause"
    GPL_V3 = "gpl_v3"
    LGPL_V3 = "lgpl_v3"
    COMMERCIAL = "commercial"
    PROPRIETARY = "proprietary"
    CUSTOM = "custom"


class SolutionCategory(BaseModel):
    """Category classification for solutions."""
    
    id: UUID = Field(default_factory=uuid4)
    name: str = Field(..., min_length=1, max_length=100)
    description: str = Field(..., max_length=500)
    parent_id: Optional[UUID] = None
    slug: str = Field(..., regex=r"^[a-z0-9-]+$")
    icon: Optional[str] = None
    sort_order: int = Field(default=0)
    is_active: bool = Field(default=True)
    created_at: datetime = Field(default_factory=datetime.utcnow)
    updated_at: datetime = Field(default_factory=datetime.utcnow)
    
    class Config:
        """Pydantic configuration."""
        json_encoders = {
            UUID: str,
            datetime: lambda v: v.isoformat(),
        }


class SolutionDependency(BaseModel):
    """Dependency specification for a solution."""
    
    id: UUID = Field(default_factory=uuid4)
    name: str = Field(..., min_length=1, max_length=100)
    version_constraint: str = Field(..., description="Version constraint (e.g., >=1.0.0)")
    dependency_type: str = Field(..., description="Type of dependency (python, system, etc.)")
    is_optional: bool = Field(default=False)
    description: Optional[str] = None
    
    @validator('version_constraint')
    def validate_version_constraint(cls, v: str) -> str:
        """Validate version constraint format."""
        # Basic validation - could be enhanced with proper semver parsing
        if not v or len(v.strip()) == 0:
            raise ValueError("Version constraint cannot be empty")
        return v.strip()


class SolutionMetadata(BaseModel):
    """Extended metadata for solutions."""
    
    tags: List[str] = Field(default_factory=list)
    keywords: List[str] = Field(default_factory=list)
    industry_applications: List[str] = Field(default_factory=list)
    supported_frameworks: List[str] = Field(default_factory=list)
    supported_languages: List[str] = Field(default_factory=list)
    min_python_version: Optional[str] = None
    max_python_version: Optional[str] = None
    hardware_requirements: Dict[str, Any] = Field(default_factory=dict)
    performance_benchmarks: Dict[str, Any] = Field(default_factory=dict)
    use_cases: List[str] = Field(default_factory=list)
    limitations: List[str] = Field(default_factory=list)
    
    @validator('tags', 'keywords', 'industry_applications')
    def validate_string_lists(cls, v: List[str]) -> List[str]:
        """Validate string lists are not empty and contain valid strings."""
        return [item.strip().lower() for item in v if item and item.strip()]


class SolutionVersion(BaseModel):
    """Version of a solution with specific implementation details."""
    
    id: UUID = Field(default_factory=uuid4)
    solution_id: SolutionId
    version: Version
    status: SolutionStatus = Field(default=SolutionStatus.DRAFT)
    
    # Version-specific content
    title: str = Field(..., min_length=1, max_length=200)
    description: str = Field(..., min_length=10, max_length=2000)
    changelog: str = Field(default="")
    documentation_url: Optional[str] = None
    source_code_url: Optional[str] = None
    
    # Technical specifications
    technical_spec: TechnicalSpecification
    dependencies: List[SolutionDependency] = Field(default_factory=list)
    
    # Pricing and licensing
    price: Optional[Price] = None
    license_type: LicenseType = Field(default=LicenseType.MIT)
    license_url: Optional[str] = None
    
    # Quality metrics
    quality_score: Optional[float] = Field(None, ge=0, le=100)
    security_score: Optional[float] = Field(None, ge=0, le=100)
    performance_score: Optional[float] = Field(None, ge=0, le=100)
    
    # Deployment information
    container_image: Optional[str] = None
    deployment_config: Dict[str, Any] = Field(default_factory=dict)
    api_specification: Dict[str, Any] = Field(default_factory=dict)
    
    # Metrics
    download_count: int = Field(default=0, ge=0)
    usage_count: int = Field(default=0, ge=0)
    
    # Timestamps
    published_at: Optional[datetime] = None
    created_at: datetime = Field(default_factory=datetime.utcnow)
    updated_at: datetime = Field(default_factory=datetime.utcnow)
    
    def is_published(self) -> bool:
        """Check if this version is published."""
        return self.status == SolutionStatus.PUBLISHED
    
    def can_be_deployed(self) -> bool:
        """Check if this version can be deployed."""
        return (
            self.status == SolutionStatus.PUBLISHED 
            and self.container_image is not None
            and self.quality_score is not None
            and self.quality_score >= 70
        )
    
    def update_metrics(self, downloads: int = 0, usage: int = 0) -> None:
        """Update usage metrics."""
        self.download_count += downloads
        self.usage_count += usage
        self.updated_at = datetime.utcnow()
    
    class Config:
        """Pydantic configuration."""
        json_encoders = {
            UUID: str,
            datetime: lambda v: v.isoformat(),
            Decimal: str,
        }


class Solution(BaseModel):
    """Core solution entity representing an ML solution in the marketplace."""
    
    id: SolutionId
    provider_id: ProviderId
    
    # Basic information
    name: str = Field(..., min_length=1, max_length=100, regex=r"^[a-zA-Z0-9\s\-_\.]+$")
    slug: str = Field(..., regex=r"^[a-z0-9-]+$")
    solution_type: SolutionType
    category_id: UUID
    
    # Content
    short_description: str = Field(..., min_length=10, max_length=300)
    long_description: str = Field(..., min_length=50, max_length=5000)
    logo_url: Optional[str] = None
    banner_url: Optional[str] = None
    screenshots: List[str] = Field(default_factory=list)
    
    # Metadata
    metadata: SolutionMetadata = Field(default_factory=SolutionMetadata)
    
    # Versions
    versions: List[SolutionVersion] = Field(default_factory=list)
    latest_version_id: Optional[UUID] = None
    
    # Status and visibility
    status: SolutionStatus = Field(default=SolutionStatus.DRAFT)
    is_featured: bool = Field(default=False)
    is_verified: bool = Field(default=False)
    
    # Metrics and ratings
    average_rating: float = Field(default=0.0, ge=0, le=5)
    total_reviews: int = Field(default=0, ge=0)
    total_downloads: int = Field(default=0, ge=0)
    total_deployments: int = Field(default=0, ge=0)
    
    # Timestamps
    created_at: datetime = Field(default_factory=datetime.utcnow)
    updated_at: datetime = Field(default_factory=datetime.utcnow)
    published_at: Optional[datetime] = None
    
    def add_version(self, version: SolutionVersion) -> None:
        """Add a new version to this solution."""
        if version.solution_id != self.id:
            raise ValueError("Version solution_id must match solution id")
        
        self.versions.append(version)
        if version.status == SolutionStatus.PUBLISHED:
            self.latest_version_id = version.id
        self.updated_at = datetime.utcnow()
    
    def get_latest_version(self) -> Optional[SolutionVersion]:
        """Get the latest published version."""
        if not self.latest_version_id:
            return None
        
        return next(
            (v for v in self.versions if v.id == self.latest_version_id),
            None
        )
    
    def get_version(self, version_id: UUID) -> Optional[SolutionVersion]:
        """Get a specific version by ID."""
        return next((v for v in self.versions if v.id == version_id), None)
    
    def get_published_versions(self) -> List[SolutionVersion]:
        """Get all published versions."""
        return [v for v in self.versions if v.status == SolutionStatus.PUBLISHED]
    
    def is_published(self) -> bool:
        """Check if solution has any published versions."""
        return any(v.status == SolutionStatus.PUBLISHED for v in self.versions)
    
    def update_rating(self, new_rating: float, review_count_change: int = 1) -> None:
        """Update the average rating with a new rating."""
        if review_count_change > 0:
            total_rating_points = self.average_rating * self.total_reviews
            total_rating_points += new_rating
            self.total_reviews += review_count_change
            self.average_rating = total_rating_points / self.total_reviews
        elif review_count_change < 0 and self.total_reviews > 0:
            # Handle rating removal
            total_rating_points = self.average_rating * self.total_reviews
            total_rating_points -= new_rating
            self.total_reviews += review_count_change  # review_count_change is negative
            
            if self.total_reviews > 0:
                self.average_rating = total_rating_points / self.total_reviews
            else:
                self.average_rating = 0.0
        
        self.updated_at = datetime.utcnow()
    
    def increment_downloads(self, count: int = 1) -> None:
        """Increment download count."""
        self.total_downloads += count
        self.updated_at = datetime.utcnow()
    
    def increment_deployments(self, count: int = 1) -> None:
        """Increment deployment count."""
        self.total_deployments += count
        self.updated_at = datetime.utcnow()
    
    def can_be_published(self) -> bool:
        """Check if solution meets publishing requirements."""
        return (
            len(self.name.strip()) >= 3
            and len(self.short_description.strip()) >= 10
            and len(self.long_description.strip()) >= 50
            and len(self.versions) > 0
            and any(v.status == SolutionStatus.APPROVED for v in self.versions)
        )
    
    def publish(self) -> None:
        """Publish the solution."""
        if not self.can_be_published():
            raise ValueError("Solution does not meet publishing requirements")
        
        self.status = SolutionStatus.PUBLISHED
        self.published_at = datetime.utcnow()
        self.updated_at = datetime.utcnow()
        
        # Publish the latest approved version
        for version in self.versions:
            if version.status == SolutionStatus.APPROVED:
                version.status = SolutionStatus.PUBLISHED
                version.published_at = datetime.utcnow()
                if not self.latest_version_id:
                    self.latest_version_id = version.id
                break
    
    @validator('slug')
    def validate_slug(cls, v: str) -> str:
        """Validate and format slug."""
        if not v or len(v.strip()) == 0:
            raise ValueError("Slug cannot be empty")
        return v.strip().lower()
    
    @validator('screenshots')
    def validate_screenshots(cls, v: List[str]) -> List[str]:
        """Validate screenshot URLs."""
        if len(v) > 10:
            raise ValueError("Maximum 10 screenshots allowed")
        return [url.strip() for url in v if url and url.strip()]
    
    class Config:
        """Pydantic configuration."""
        json_encoders = {
            UUID: str,
            datetime: lambda v: v.isoformat(),
        }