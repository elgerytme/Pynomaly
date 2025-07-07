"""Pydantic schemas for user management."""

from datetime import datetime
from typing import List, Optional

from pydantic import BaseModel, EmailStr, Field


class UserCreate(BaseModel):
    """Schema for creating a new user."""
    
    username: str = Field(..., description="Unique username")
    email: EmailStr = Field(..., description="User email address")
    password: str = Field(..., description="User password")
    full_name: Optional[str] = Field(None, description="Full name")
    roles: List[str] = Field(default=[], description="User roles")
    is_active: bool = Field(default=True, description="Whether user is active")


class UserUpdate(BaseModel):
    """Schema for updating a user."""
    
    email: Optional[EmailStr] = Field(None, description="Update email")
    full_name: Optional[str] = Field(None, description="Update full name")
    roles: Optional[List[str]] = Field(None, description="Update roles")
    is_active: Optional[bool] = Field(None, description="Enable/disable user")


class UserPasswordUpdate(BaseModel):
    """Schema for updating user password."""
    
    current_password: str = Field(..., description="Current password")
    new_password: str = Field(..., description="New password")


class UserResponse(BaseModel):
    """Schema for user response."""
    
    id: str = Field(..., description="Unique user ID")
    username: str = Field(..., description="Username")
    email: str = Field(..., description="Email address")
    full_name: Optional[str] = Field(None, description="Full name")
    roles: List[str] = Field(..., description="User roles")
    is_active: bool = Field(..., description="Whether user is active")
    created_at: datetime = Field(..., description="Creation timestamp")
    updated_at: datetime = Field(..., description="Last update timestamp")
    last_login_at: Optional[datetime] = Field(None, description="Last login timestamp")


class UserList(BaseModel):
    """Schema for paginated user list."""
    
    items: List[UserResponse] = Field(..., description="List of users")
    total: int = Field(..., description="Total number of users")
    page: int = Field(..., description="Current page")
    page_size: int = Field(..., description="Items per page")


class UserActivityLog(BaseModel):
    """Schema for user activity log entry."""
    
    id: str = Field(..., description="Log entry ID")
    user_id: str = Field(..., description="User ID")
    action: str = Field(..., description="Action performed")
    resource: Optional[str] = Field(None, description="Resource affected")
    timestamp: datetime = Field(..., description="Action timestamp")
    ip_address: Optional[str] = Field(None, description="IP address")
    user_agent: Optional[str] = Field(None, description="User agent")


class UserSession(BaseModel):
    """Schema for user session."""
    
    id: str = Field(..., description="Session ID")
    user_id: str = Field(..., description="User ID")
    created_at: datetime = Field(..., description="Session creation time")
    last_activity: datetime = Field(..., description="Last activity time")
    ip_address: Optional[str] = Field(None, description="IP address")
    user_agent: Optional[str] = Field(None, description="User agent")
    is_active: bool = Field(..., description="Whether session is active")
