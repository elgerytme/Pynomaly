"""
Authentication Data Transfer Objects (DTOs) for API communication.
"""

from datetime import datetime
from typing import Dict, List, Optional
from uuid import UUID

from pydantic import BaseModel, EmailStr, Field, validator


class LoginRequest(BaseModel):
    """User login request."""
    
    email: EmailStr = Field(..., description="User email")
    password: str = Field(..., min_length=1, description="User password")
    mfa_code: Optional[str] = Field(None, description="MFA verification code")
    remember_me: bool = Field(default=False, description="Remember login")
    device_info: Optional[Dict[str, str]] = Field(None, description="Device information")
    
    @validator('mfa_code')
    def validate_mfa_code(cls, v):
        """Validate MFA code format."""
        if v and not v.isdigit():
            raise ValueError('MFA code must be numeric')
        return v


class LoginResponse(BaseModel):
    """User login response."""
    
    success: bool = Field(..., description="Authentication success")
    user_id: Optional[UUID] = Field(None, description="User ID")
    tenant_id: Optional[UUID] = Field(None, description="Tenant ID")
    access_token: Optional[str] = Field(None, description="JWT access token")
    refresh_token: Optional[str] = Field(None, description="Refresh token")
    expires_at: Optional[str] = Field(None, description="Token expiration time")
    requires_mfa: bool = Field(default=False, description="MFA verification required")
    message: Optional[str] = Field(None, description="Response message")
    user_info: Optional[Dict[str, any]] = Field(None, description="User information")


class RegisterRequest(BaseModel):
    """User registration request."""
    
    tenant_id: UUID = Field(..., description="Tenant ID")
    email: EmailStr = Field(..., description="User email")
    username: Optional[str] = Field(None, min_length=3, max_length=50)
    first_name: str = Field(..., min_length=1, max_length=100)
    last_name: str = Field(..., min_length=1, max_length=100)
    password: str = Field(..., min_length=8, description="User password")
    confirm_password: str = Field(..., description="Password confirmation")
    role: Optional[str] = Field(None, description="Initial user role")
    terms_accepted: bool = Field(..., description="Terms and conditions accepted")
    
    @validator('confirm_password')
    def passwords_match(cls, v, values):
        """Ensure passwords match."""
        if 'password' in values and v != values['password']:
            raise ValueError('Passwords do not match')
        return v
    
    @validator('username')
    def validate_username(cls, v):
        """Validate username format."""
        if v and not v.replace('_', '').replace('-', '').isalnum():
            raise ValueError('Username must contain only alphanumeric characters, underscores, and hyphens')
        return v.lower() if v else v


class RegisterResponse(BaseModel):
    """User registration response."""
    
    success: bool = Field(..., description="Registration success")
    user_id: Optional[UUID] = Field(None, description="Created user ID")
    message: str = Field(..., description="Registration message")


class RefreshTokenRequest(BaseModel):
    """Token refresh request."""
    
    refresh_token: str = Field(..., description="Refresh token")


class PasswordResetRequest(BaseModel):
    """Password reset request."""
    
    email: EmailStr = Field(..., description="User email")


class PasswordResetConfirmRequest(BaseModel):
    """Password reset confirmation request."""
    
    token: str = Field(..., description="Reset token")
    new_password: str = Field(..., min_length=8, description="New password")
    confirm_password: str = Field(..., description="Password confirmation")
    
    @validator('confirm_password')
    def passwords_match(cls, v, values):
        """Ensure passwords match."""
        if 'new_password' in values and v != values['new_password']:
            raise ValueError('Passwords do not match')
        return v


class PasswordChangeRequest(BaseModel):
    """Password change request."""
    
    current_password: str = Field(..., description="Current password")
    new_password: str = Field(..., min_length=8, description="New password")
    confirm_password: str = Field(..., description="Password confirmation")
    
    @validator('confirm_password')
    def passwords_match(cls, v, values):
        """Ensure passwords match."""
        if 'new_password' in values and v != values['new_password']:
            raise ValueError('Passwords do not match')
        return v


class MFASetupRequest(BaseModel):
    """MFA setup request."""
    
    password: str = Field(..., description="Current password for verification")


class MFAVerifyRequest(BaseModel):
    """MFA verification request."""
    
    code: str = Field(..., min_length=6, max_length=6, description="6-digit MFA code")
    
    @validator('code')
    def validate_code(cls, v):
        """Validate MFA code format."""
        if not v.isdigit():
            raise ValueError('MFA code must be numeric')
        return v


class MFADisableRequest(BaseModel):
    """MFA disable request."""
    
    password: str = Field(..., description="Current password for verification")
    backup_code: Optional[str] = Field(None, description="Backup code (alternative to password)")


class SAMLAuthRequest(BaseModel):
    """SAML authentication request."""
    
    tenant_id: UUID = Field(..., description="Tenant ID")
    saml_response: str = Field(..., description="SAML response")
    relay_state: Optional[str] = Field(None, description="Relay state parameter")


class OAuth2AuthRequest(BaseModel):
    """OAuth2/OIDC authentication request."""
    
    tenant_id: UUID = Field(..., description="Tenant ID")
    provider: str = Field(..., description="OAuth2 provider name")
    code: str = Field(..., description="Authorization code")
    redirect_uri: str = Field(..., description="Redirect URI")
    state: Optional[str] = Field(None, description="State parameter")


class UserProfileRequest(BaseModel):
    """User profile update request."""
    
    first_name: Optional[str] = Field(None, min_length=1, max_length=100)
    last_name: Optional[str] = Field(None, min_length=1, max_length=100)
    display_name: Optional[str] = Field(None, max_length=200)
    phone: Optional[str] = Field(None)
    timezone: Optional[str] = Field(None)
    language: Optional[str] = Field(None)
    preferences: Optional[Dict[str, any]] = Field(None)


class UserProfileResponse(BaseModel):
    """User profile response."""
    
    id: UUID
    email: str
    username: str
    first_name: str
    last_name: str
    display_name: Optional[str]
    phone: Optional[str]
    avatar_url: Optional[str]
    timezone: str
    language: str
    preferences: Dict[str, any]
    roles: List[str]
    permissions: List[str]
    mfa_enabled: bool
    created_at: datetime
    updated_at: datetime
    
    class Config:
        """Pydantic configuration."""
        json_encoders = {
            datetime: lambda v: v.isoformat(),
            UUID: lambda v: str(v),
        }


class SessionInfoResponse(BaseModel):
    """User session information response."""
    
    id: UUID
    user_id: UUID
    tenant_id: UUID
    device_name: Optional[str]
    ip_address: str
    user_agent: Optional[str]
    location: Optional[Dict[str, str]]
    created_at: datetime
    last_accessed_at: datetime
    expires_at: datetime
    is_current: bool = Field(default=False, description="Is this the current session")
    
    class Config:
        """Pydantic configuration."""
        json_encoders = {
            datetime: lambda v: v.isoformat(),
            UUID: lambda v: str(v),
        }


class RevokeSessionRequest(BaseModel):
    """Session revocation request."""
    
    session_id: Optional[UUID] = Field(None, description="Specific session to revoke")
    revoke_all: bool = Field(default=False, description="Revoke all sessions except current")


class TokenValidationResponse(BaseModel):
    """Token validation response."""
    
    valid: bool = Field(..., description="Token validity")
    user_id: Optional[UUID] = Field(None, description="User ID from token")
    tenant_id: Optional[UUID] = Field(None, description="Tenant ID from token")
    roles: List[str] = Field(default_factory=list, description="User roles")
    permissions: List[str] = Field(default_factory=list, description="User permissions")
    expires_at: Optional[datetime] = Field(None, description="Token expiration")
    error: Optional[str] = Field(None, description="Validation error message")
    
    class Config:
        """Pydantic configuration."""
        json_encoders = {
            datetime: lambda v: v.isoformat(),
            UUID: lambda v: str(v),
        }


class AuthEventRequest(BaseModel):
    """Authentication event logging request."""
    
    event_type: str = Field(..., description="Event type")
    user_id: Optional[UUID] = Field(None, description="User ID")
    tenant_id: Optional[UUID] = Field(None, description="Tenant ID")
    ip_address: Optional[str] = Field(None, description="Client IP address")
    user_agent: Optional[str] = Field(None, description="User agent")
    details: Optional[Dict[str, any]] = Field(None, description="Additional event details")


class AuthStatsResponse(BaseModel):
    """Authentication statistics response."""
    
    total_logins_today: int = Field(default=0)
    failed_logins_today: int = Field(default=0)
    active_sessions: int = Field(default=0)
    mfa_enabled_users: int = Field(default=0)
    sso_logins_today: int = Field(default=0)
    locked_accounts: int = Field(default=0)
    
    # Time series data
    hourly_logins: List[Dict[str, int]] = Field(default_factory=list)
    daily_logins: List[Dict[str, int]] = Field(default_factory=list)
    
    # Provider breakdown
    auth_provider_stats: Dict[str, int] = Field(default_factory=dict)


class BulkUserActionRequest(BaseModel):
    """Bulk user action request."""
    
    user_ids: List[UUID] = Field(..., description="List of user IDs")
    action: str = Field(..., description="Action to perform")
    parameters: Optional[Dict[str, any]] = Field(None, description="Action parameters")
    
    @validator('action')
    def validate_action(cls, v):
        """Validate allowed bulk actions."""
        allowed_actions = [
            'activate', 'deactivate', 'suspend', 'unlock',
            'reset_mfa', 'force_password_reset', 'add_role', 'remove_role'
        ]
        if v not in allowed_actions:
            raise ValueError(f'Action must be one of: {", ".join(allowed_actions)}')
        return v


class BulkUserActionResponse(BaseModel):
    """Bulk user action response."""
    
    success: bool = Field(..., description="Overall operation success")
    total_users: int = Field(..., description="Total users processed")
    successful_updates: int = Field(..., description="Successful updates")
    failed_updates: int = Field(..., description="Failed updates")
    results: List[Dict[str, any]] = Field(..., description="Individual results")
    errors: List[str] = Field(default_factory=list, description="Error messages")