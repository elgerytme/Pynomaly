"""Pydantic schemas for rate limit management."""

from typing import List, Optional

from pydantic import BaseModel, Field


class RateLimitRuleCreate(BaseModel):
    """Schema for creating a new rate limit rule."""
    
    path_regex: str = Field(..., description="Regex to match API path")
    limit: int = Field(..., description="Number of requests allowed")
    period_seconds: int = Field(..., description="Time period in seconds")
    description: Optional[str] = Field(None, description="Rule description")


class RateLimitRuleUpdate(BaseModel):
    """Schema for updating a rate limit rule."""
    
    path_regex: Optional[str] = Field(None, description="Update path regex")
    limit: Optional[int] = Field(None, description="Update request limit")
    period_seconds: Optional[int] = Field(None, description="Update time period")
    description: Optional[str] = Field(None, description="Update description")
    is_active: Optional[bool] = Field(None, description="Enable/disable rule")


class RateLimitRuleResponse(BaseModel):
    """Schema for rate limit rule response."""
    
    id: str = Field(..., description="Unique rule ID")
    path_regex: str = Field(..., description="API path regex")
    limit: int = Field(..., description="Request limit")
    period_seconds: int = Field(..., description="Time period in seconds")
    description: Optional[str] = Field(None, description="Description")
    is_active: bool = Field(..., description="Whether rule is active")
    created_at: str = Field(..., description="Creation timestamp")
    updated_at: str = Field(..., description="Last update timestamp")


class RateLimitRuleList(BaseModel):
    """Schema for paginated rate limit rule list."""
    
    items: List[RateLimitRuleResponse] = Field(..., description="List of rules")
    total: int = Field(..., description="Total number of rules")
    page: int = Field(..., description="Current page")
    page_size: int = Field(..., description="Items per page")

