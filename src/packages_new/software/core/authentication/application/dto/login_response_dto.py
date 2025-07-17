"""
Login response DTO
"""
from dataclasses import dataclass
from typing import Optional
from uuid import UUID

@dataclass
class LoginResponseDto:
    """Data transfer object for login response"""
    
    success: bool
    error: Optional[str] = None
    access_token: Optional[str] = None
    refresh_token: Optional[str] = None
    user_id: Optional[UUID] = None
    expires_in: Optional[int] = None  # Token expiration in seconds