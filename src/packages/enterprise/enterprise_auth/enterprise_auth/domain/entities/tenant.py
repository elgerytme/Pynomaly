"""
Tenant domain entities for multi-tenancy support.
"""

from datetime import datetime
from enum import Enum
from typing import Dict, List, Optional
from uuid import UUID, uuid4

from pydantic import BaseModel, Field, validator


class TenantStatus(str, Enum):
    """Tenant status enumeration."""
    ACTIVE = "active"
    INACTIVE = "inactive"
    SUSPENDED = "suspended"
    TRIAL = "trial"
    EXPIRED = "expired"
    DELETED = "deleted"


class TenantPlan(str, Enum):
    """Tenant subscription plan enumeration."""
    FREE = "free"
    STARTER = "starter"
    PROFESSIONAL = "professional"
    ENTERPRISE = "enterprise"
    CUSTOM = "custom"


class TenantFeature(str, Enum):
    """Tenant feature enumeration."""
    SSO = "sso"
    SAML = "saml"
    LDAP = "ldap"
    AUDIT_LOGS = "audit_logs"
    ADVANCED_ANALYTICS = "advanced_analytics"
    API_ACCESS = "api_access"
    CUSTOM_ROLES = "custom_roles"
    BULK_OPERATIONS = "bulk_operations"
    PRIORITY_SUPPORT = "priority_support"
    SLA_GUARANTEE = "sla_guarantee"
    WHITE_LABELING = "white_labeling"
    CUSTOM_INTEGRATIONS = "custom_integrations"


class Tenant(BaseModel):
    """
    Tenant domain entity representing an organization in a multi-tenant system.
    
    Each tenant represents an isolated organization with its own users,
    data, and configuration settings.
    """
    
    id: UUID = Field(default_factory=uuid4, description="Unique tenant identifier")
    
    # Basic Information
    name: str = Field(..., min_length=1, max_length=200, description="Tenant name")
    slug: str = Field(..., min_length=3, max_length=50, description="URL-friendly identifier")
    description: Optional[str] = Field(None, max_length=1000)
    
    # Contact Information
    admin_email: str = Field(..., description="Primary admin email")
    support_email: Optional[str] = Field(None)
    website: Optional[str] = Field(None)
    phone: Optional[str] = Field(None)
    
    # Subscription & Billing
    plan: TenantPlan = Field(default=TenantPlan.FREE)
    status: TenantStatus = Field(default=TenantStatus.TRIAL)
    trial_ends_at: Optional[datetime] = Field(None)
    subscription_starts_at: Optional[datetime] = Field(None)
    subscription_ends_at: Optional[datetime] = Field(None)
    
    # Features & Limits
    enabled_features: List[TenantFeature] = Field(default_factory=list)
    max_users: int = Field(default=5, description="Maximum allowed users")
    max_data_retention_days: int = Field(default=30, description="Data retention period")
    max_api_calls_per_month: int = Field(default=1000, description="API rate limit")
    max_storage_gb: int = Field(default=1, description="Storage limit in GB")
    
    # Configuration
    settings: Dict[str, any] = Field(default_factory=dict, description="Tenant-specific settings")
    custom_domain: Optional[str] = Field(None, description="Custom domain for tenant")
    logo_url: Optional[str] = Field(None, description="Tenant logo URL")
    theme_config: Dict[str, any] = Field(default_factory=dict, description="UI theme configuration")
    
    # Security Settings
    password_policy: Dict[str, any] = Field(default_factory=dict)
    session_timeout_minutes: int = Field(default=480, description="Session timeout in minutes")
    mfa_required: bool = Field(default=False, description="Require MFA for all users")
    ip_whitelist: List[str] = Field(default_factory=list, description="Allowed IP addresses")
    
    # SSO Configuration
    sso_enabled: bool = Field(default=False)
    saml_config: Optional[Dict[str, any]] = Field(None)
    oauth_config: Optional[Dict[str, any]] = Field(None)
    ldap_config: Optional[Dict[str, any]] = Field(None)
    
    # Usage Statistics
    current_user_count: int = Field(default=0)
    current_storage_gb: float = Field(default=0.0)
    api_calls_this_month: int = Field(default=0)
    last_activity_at: Optional[datetime] = Field(None)
    
    # Audit Fields
    created_at: datetime = Field(default_factory=datetime.utcnow)
    updated_at: datetime = Field(default_factory=datetime.utcnow)
    created_by: Optional[UUID] = Field(None)
    updated_by: Optional[UUID] = Field(None)
    
    # Soft Delete
    deleted_at: Optional[datetime] = Field(None)
    deleted_by: Optional[UUID] = Field(None)
    
    class Config:
        """Pydantic configuration."""
        use_enum_values = True
        validate_assignment = True
        json_encoders = {
            datetime: lambda v: v.isoformat(),
            UUID: lambda v: str(v),
        }
    
    @validator('slug')
    def slug_must_be_valid(cls, v):
        """Validate slug format."""
        if not v.replace('-', '').replace('_', '').isalnum():
            raise ValueError('Slug must contain only alphanumeric characters, hyphens, and underscores')
        return v.lower()
    
    @validator('admin_email')
    def admin_email_must_be_lowercase(cls, v):
        """Ensure admin email is lowercase."""
        return v.lower()
    
    @validator('enabled_features', pre=True)
    def validate_features(cls, v):
        """Validate enabled features."""
        if isinstance(v, str):
            return [v]
        return v or []
    
    def has_feature(self, feature: TenantFeature) -> bool:
        """Check if tenant has a specific feature enabled."""
        return feature in self.enabled_features
    
    def enable_feature(self, feature: TenantFeature) -> None:
        """Enable a feature for the tenant."""
        if feature not in self.enabled_features:
            self.enabled_features.append(feature)
            self.updated_at = datetime.utcnow()
    
    def disable_feature(self, feature: TenantFeature) -> None:
        """Disable a feature for the tenant."""
        if feature in self.enabled_features:
            self.enabled_features.remove(feature)
            self.updated_at = datetime.utcnow()
    
    def is_active(self) -> bool:
        """Check if tenant is active."""
        return (
            self.status in [TenantStatus.ACTIVE, TenantStatus.TRIAL] and
            self.deleted_at is None and
            not self.is_expired()
        )
    
    def is_expired(self) -> bool:
        """Check if tenant subscription is expired."""
        if self.status == TenantStatus.TRIAL and self.trial_ends_at:
            return datetime.utcnow() > self.trial_ends_at
        
        if self.subscription_ends_at:
            return datetime.utcnow() > self.subscription_ends_at
        
        return False
    
    def is_trial(self) -> bool:
        """Check if tenant is on trial."""
        return self.status == TenantStatus.TRIAL
    
    def can_add_user(self) -> bool:
        """Check if tenant can add more users."""
        return self.current_user_count < self.max_users
    
    def can_use_storage(self, additional_gb: float = 0) -> bool:
        """Check if tenant can use more storage."""
        return (self.current_storage_gb + additional_gb) <= self.max_storage_gb
    
    def can_make_api_call(self) -> bool:
        """Check if tenant can make more API calls this month."""
        return self.api_calls_this_month < self.max_api_calls_per_month
    
    def increment_user_count(self) -> None:
        """Increment current user count."""
        self.current_user_count += 1
        self.updated_at = datetime.utcnow()
    
    def decrement_user_count(self) -> None:
        """Decrement current user count."""
        if self.current_user_count > 0:
            self.current_user_count -= 1
            self.updated_at = datetime.utcnow()
    
    def increment_api_calls(self, count: int = 1) -> None:
        """Increment API call count."""
        self.api_calls_this_month += count
        self.last_activity_at = datetime.utcnow()
        self.updated_at = datetime.utcnow()
    
    def update_storage_usage(self, storage_gb: float) -> None:
        """Update current storage usage."""
        self.current_storage_gb = storage_gb
        self.updated_at = datetime.utcnow()
    
    def upgrade_plan(self, new_plan: TenantPlan, updated_by: Optional[UUID] = None) -> None:
        """Upgrade tenant plan."""
        old_plan = self.plan
        self.plan = new_plan
        self.updated_by = updated_by
        self.updated_at = datetime.utcnow()
        
        # Update limits based on plan
        self._update_plan_limits(new_plan)
        
        # Enable features based on plan
        self._update_plan_features(new_plan)
        
        # If upgrading from trial, activate subscription
        if old_plan == TenantPlan.FREE and new_plan != TenantPlan.FREE:
            self.status = TenantStatus.ACTIVE
            self.subscription_starts_at = datetime.utcnow()
    
    def suspend(self, suspended_by: Optional[UUID] = None, reason: Optional[str] = None) -> None:
        """Suspend tenant."""
        self.status = TenantStatus.SUSPENDED
        self.updated_by = suspended_by
        self.updated_at = datetime.utcnow()
        
        # Add suspension reason to settings
        if reason:
            self.settings['suspension_reason'] = reason
            self.settings['suspended_at'] = datetime.utcnow().isoformat()
    
    def reactivate(self, reactivated_by: Optional[UUID] = None) -> None:
        """Reactivate suspended tenant."""
        self.status = TenantStatus.ACTIVE
        self.updated_by = reactivated_by
        self.updated_at = datetime.utcnow()
        
        # Remove suspension info
        self.settings.pop('suspension_reason', None)
        self.settings.pop('suspended_at', None)
    
    def soft_delete(self, deleted_by: Optional[UUID] = None) -> None:
        """Soft delete the tenant."""
        self.deleted_at = datetime.utcnow()
        self.deleted_by = deleted_by
        self.status = TenantStatus.DELETED
        self.updated_at = datetime.utcnow()
    
    def restore(self) -> None:
        """Restore a soft-deleted tenant."""
        self.deleted_at = None
        self.deleted_by = None
        self.status = TenantStatus.INACTIVE  # Require re-activation
        self.updated_at = datetime.utcnow()
    
    def reset_monthly_usage(self) -> None:
        """Reset monthly usage counters."""
        self.api_calls_this_month = 0
        self.updated_at = datetime.utcnow()
    
    def _update_plan_limits(self, plan: TenantPlan) -> None:
        """Update limits based on plan."""
        plan_limits = {
            TenantPlan.FREE: {
                'max_users': 5,
                'max_data_retention_days': 30,
                'max_api_calls_per_month': 1000,
                'max_storage_gb': 1,
            },
            TenantPlan.STARTER: {
                'max_users': 25,
                'max_data_retention_days': 90,
                'max_api_calls_per_month': 10000,
                'max_storage_gb': 10,
            },
            TenantPlan.PROFESSIONAL: {
                'max_users': 100,
                'max_data_retention_days': 365,
                'max_api_calls_per_month': 100000,
                'max_storage_gb': 100,
            },
            TenantPlan.ENTERPRISE: {
                'max_users': 1000,
                'max_data_retention_days': 1095,  # 3 years
                'max_api_calls_per_month': 1000000,
                'max_storage_gb': 1000,
            },
            TenantPlan.CUSTOM: {
                # Custom plans maintain current limits
            }
        }
        
        if plan != TenantPlan.CUSTOM and plan in plan_limits:
            for key, value in plan_limits[plan].items():
                setattr(self, key, value)
    
    def _update_plan_features(self, plan: TenantPlan) -> None:
        """Update features based on plan."""
        plan_features = {
            TenantPlan.FREE: [
                TenantFeature.API_ACCESS,
            ],
            TenantPlan.STARTER: [
                TenantFeature.API_ACCESS,
                TenantFeature.CUSTOM_ROLES,
            ],
            TenantPlan.PROFESSIONAL: [
                TenantFeature.API_ACCESS,
                TenantFeature.CUSTOM_ROLES,
                TenantFeature.SSO,
                TenantFeature.AUDIT_LOGS,
                TenantFeature.ADVANCED_ANALYTICS,
                TenantFeature.BULK_OPERATIONS,
            ],
            TenantPlan.ENTERPRISE: [
                TenantFeature.API_ACCESS,
                TenantFeature.CUSTOM_ROLES,
                TenantFeature.SSO,
                TenantFeature.SAML,
                TenantFeature.LDAP,
                TenantFeature.AUDIT_LOGS,
                TenantFeature.ADVANCED_ANALYTICS,
                TenantFeature.BULK_OPERATIONS,
                TenantFeature.PRIORITY_SUPPORT,
                TenantFeature.SLA_GUARANTEE,
                TenantFeature.WHITE_LABELING,
                TenantFeature.CUSTOM_INTEGRATIONS,
            ],
        }
        
        if plan in plan_features:
            self.enabled_features = plan_features[plan]
    
    @property
    def is_deleted(self) -> bool:
        """Check if tenant is deleted."""
        return self.deleted_at is not None
    
    @property 
    def days_until_trial_expiry(self) -> Optional[int]:
        """Get days until trial expiry."""
        if not self.is_trial() or not self.trial_ends_at:
            return None
        
        delta = self.trial_ends_at - datetime.utcnow()
        return max(0, delta.days)
    
    @property
    def storage_usage_percentage(self) -> float:
        """Get storage usage as percentage."""
        if self.max_storage_gb <= 0:
            return 0.0
        return (self.current_storage_gb / self.max_storage_gb) * 100
    
    @property
    def api_usage_percentage(self) -> float:
        """Get API usage as percentage."""
        if self.max_api_calls_per_month <= 0:
            return 0.0
        return (self.api_calls_this_month / self.max_api_calls_per_month) * 100
    
    def to_dict(self, include_sensitive: bool = False) -> Dict[str, any]:
        """
        Convert tenant to dictionary.
        
        Args:
            include_sensitive: Whether to include sensitive fields
        """
        data = self.dict()
        
        if not include_sensitive:
            # Remove sensitive fields
            sensitive_fields = [
                'saml_config', 'oauth_config', 'ldap_config',
                'password_policy', 'ip_whitelist'
            ]
            for field in sensitive_fields:
                if field in data:
                    data[field] = "***REDACTED***"
        
        return data


class TenantInvitation(BaseModel):
    """Tenant invitation entity for inviting users to join a tenant."""
    
    id: UUID = Field(default_factory=uuid4)
    tenant_id: UUID = Field(...)
    
    # Invitation Details
    email: str = Field(...)
    role: str = Field(default="viewer")
    invited_by: UUID = Field(...)
    invitation_token: str = Field(...)
    
    # Status
    status: str = Field(default="pending")  # pending, accepted, expired, revoked
    expires_at: datetime = Field(...)
    
    # Optional Message
    message: Optional[str] = Field(None, max_length=500)
    
    # Audit Fields
    created_at: datetime = Field(default_factory=datetime.utcnow)
    accepted_at: Optional[datetime] = Field(None)
    revoked_at: Optional[datetime] = Field(None)
    
    class Config:
        """Pydantic configuration."""
        validate_assignment = True
        json_encoders = {
            datetime: lambda v: v.isoformat(),
            UUID: lambda v: str(v),
        }
    
    def is_valid(self) -> bool:
        """Check if invitation is valid."""
        return (
            self.status == "pending" and
            datetime.utcnow() < self.expires_at
        )
    
    def accept(self) -> None:
        """Accept the invitation."""
        self.status = "accepted"
        self.accepted_at = datetime.utcnow()
    
    def revoke(self) -> None:
        """Revoke the invitation."""
        self.status = "revoked"
        self.revoked_at = datetime.utcnow()
    
    def is_expired(self) -> bool:
        """Check if invitation is expired."""
        return datetime.utcnow() > self.expires_at