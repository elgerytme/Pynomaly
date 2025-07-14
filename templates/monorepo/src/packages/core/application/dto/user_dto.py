"""User data transfer objects."""

from datetime import datetime
from typing import Optional, Set

from pydantic import BaseModel, EmailStr, Field

from ..entities.user import UserRole, UserStatus


class CreateUserDTO(BaseModel):
    """DTO for creating a new user."""
    
    email: EmailStr
    username: str = Field(..., min_length=3, max_length=50)
    full_name: str = Field(..., min_length=1, max_length=100)
    password: str = Field(..., min_length=8)
    roles: Set[UserRole] = Field(default_factory=set)


class UpdateUserDTO(BaseModel):
    """DTO for updating user information."""
    
    full_name: Optional[str] = Field(None, min_length=1, max_length=100)
    roles: Optional[Set[UserRole]] = None
    status: Optional[UserStatus] = None


class UserDTO(BaseModel):
    """DTO for user information."""
    
    id: str
    email: str
    username: str
    full_name: str
    roles: Set[UserRole]
    status: UserStatus
    created_at: datetime
    updated_at: Optional[datetime] = None
    last_login: Optional[datetime] = None
    email_verified: bool
    failed_login_attempts: int
    is_locked: bool


class UserLoginDTO(BaseModel):
    """DTO for user login."""
    
    email: EmailStr
    password: str
    remember_me: bool = False


class UserTokenDTO(BaseModel):
    """DTO for user authentication token."""
    
    access_token: str
    token_type: str = "bearer"
    expires_in: int
    user: UserDTO