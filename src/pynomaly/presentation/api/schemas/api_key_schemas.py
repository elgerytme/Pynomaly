"""Pydantic schemas for API key management."""

from datetime import datetime
from typing import List, Optional

from pydantic import BaseModel, Field


class ApiKeyCreate(BaseModel):
    """Schema for creating a new API key."""
    
    name: str = Field(..., description="Human-readable name for the API key")
    description: Optional[str] = Field(None, description="Optional description")
    scopes: List[str] = Field(default=[], description="List of scopes/permissions")
    expires_at: Optional[datetime] = Field(None, description="Optional expiration date")
    rate_limit: Optional[int] = Field(None, description="Custom rate limit per minute")


class ApiKeyUpdate(BaseModel):
    """Schema for updating an API key."""
    
    name: Optional[str] = Field(None, description="Update name")
    description: Optional[str] = Field(None, description="Update description")
    scopes: Optional[List[str]] = Field(None, description="Update scopes")
    expires_at: Optional[datetime] = Field(None, description="Update expiration")
    rate_limit: Optional[int] = Field(None, description="Update rate limit")
    is_active: Optional[bool] = Field(None, description="Enable/disable key")


class ApiKeyResponse(BaseModel):
    """Schema for API key response."""
    
    id: str = Field(..., description="Unique API key ID")
    name: str = Field(..., description="API key name")
    description: Optional[str] = Field(None, description="Description")
    scopes: List[str] = Field(..., description="Assigned scopes")
    is_active: bool = Field(..., description="Whether key is active")
    created_at: datetime = Field(..., description="Creation timestamp")
    updated_at: datetime = Field(..., description="Last update timestamp")
    last_used_at: Optional[datetime] = Field(None, description="Last usage timestamp")
    expires_at: Optional[datetime] = Field(None, description="Expiration timestamp")
    rate_limit: Optional[int] = Field(None, description="Rate limit per minute")
    usage_count: int = Field(default=0, description="Total usage count")


class ApiKeyWithSecret(ApiKeyResponse):
    """Schema for API key with secret (only returned on creation)."""
    
    secret: str = Field(..., description="API key secret (only shown once)")


class ApiKeyList(BaseModel):
    """Schema for paginated API key list."""
    
    items: List[ApiKeyResponse] = Field(..., description="List of API keys")
    total: int = Field(..., description="Total number of keys")
    page: int = Field(..., description="Current page")
    page_size: int = Field(..., description="Items per page")


class ApiKeyUsageStats(BaseModel):
    """Schema for API key usage statistics."""
    
    key_id: str = Field(..., description="API key ID")
    daily_usage: int = Field(..., description="Usage count today")
    weekly_usage: int = Field(..., description="Usage count this week")
    monthly_usage: int = Field(..., description="Usage count this month")
    last_hour_usage: int = Field(..., description="Usage count in last hour")
    rate_limit_hits: int = Field(..., description="Number of rate limit violations")
    success_rate: float = Field(..., description="Success rate percentage")
