"""
User entity for authentication domain
"""
from dataclasses import dataclass
from datetime import datetime
from typing import Optional
from uuid import UUID

@dataclass
class User:
    """Core user entity with authentication-specific properties"""
    
    id: UUID
    email: str
    username: str
    password_hash: str
    created_at: datetime
    last_login: Optional[datetime] = None
    is_active: bool = True
    is_verified: bool = False
    failed_login_attempts: int = 0
    locked_until: Optional[datetime] = None
    
    def can_login(self) -> bool:
        """Business rule: User can login if active, verified, and not locked"""
        if not self.is_active:
            return False
        
        if not self.is_verified:
            return False
        
        if self.locked_until and self.locked_until > datetime.now():
            return False
        
        return True
    
    def increment_failed_attempts(self) -> None:
        """Business rule: Increment failed login attempts"""
        self.failed_login_attempts += 1
        
        # Lock account after 5 failed attempts
        if self.failed_login_attempts >= 5:
            self.locked_until = datetime.now() + timedelta(minutes=30)
    
    def reset_failed_attempts(self) -> None:
        """Business rule: Reset failed attempts on successful login"""
        self.failed_login_attempts = 0
        self.locked_until = None
        self.last_login = datetime.now()
    
    def verify_account(self) -> None:
        """Business rule: Verify user account"""
        self.is_verified = True
    
    def deactivate(self) -> None:
        """Business rule: Deactivate user account"""
        self.is_active = False