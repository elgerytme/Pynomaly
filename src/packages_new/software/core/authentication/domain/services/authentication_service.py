"""
Authentication domain service
"""
import hashlib
import secrets
from datetime import datetime, timedelta
from typing import Optional, Tuple
from ..entities.user import User
from ..value_objects.email import Email

class AuthenticationService:
    """Domain service for authentication logic"""
    
    def __init__(self):
        self.password_salt_length = 32
        self.min_password_length = 8
    
    def hash_password(self, password: str) -> str:
        """Hash password with salt"""
        salt = secrets.token_hex(self.password_salt_length)
        password_hash = hashlib.pbkdf2_hmac(
            'sha256',
            password.encode('utf-8'),
            salt.encode('utf-8'),
            100000
        )
        return f"{salt}:{password_hash.hex()}"
    
    def verify_password(self, password: str, password_hash: str) -> bool:
        """Verify password against hash"""
        try:
            salt, stored_hash = password_hash.split(':')
            password_hash = hashlib.pbkdf2_hmac(
                'sha256',
                password.encode('utf-8'),
                salt.encode('utf-8'),
                100000
            )
            return stored_hash == password_hash.hex()
        except ValueError:
            return False
    
    def validate_password_strength(self, password: str) -> Tuple[bool, list[str]]:
        """Validate password strength"""
        errors = []
        
        if len(password) < self.min_password_length:
            errors.append(f"Password must be at least {self.min_password_length} characters")
        
        if not any(c.isupper() for c in password):
            errors.append("Password must contain at least one uppercase letter")
        
        if not any(c.islower() for c in password):
            errors.append("Password must contain at least one lowercase letter")
        
        if not any(c.isdigit() for c in password):
            errors.append("Password must contain at least one digit")
        
        if not any(c in "!@#$%^&*()_+-=[]{}|;:,.<>?" for c in password):
            errors.append("Password must contain at least one special character")
        
        return len(errors) == 0, errors
    
    def authenticate_user(self, user: User, password: str) -> bool:
        """Authenticate user with password"""
        if not user.can_login():
            return False
        
        if not self.verify_password(password, user.password_hash):
            user.increment_failed_attempts()
            return False
        
        user.reset_failed_attempts()
        return True
    
    def generate_reset_token(self) -> str:
        """Generate password reset token"""
        return secrets.token_urlsafe(32)
    
    def is_reset_token_valid(self, token: str, created_at: datetime) -> bool:
        """Check if reset token is still valid"""
        expiry_time = created_at + timedelta(hours=1)
        return datetime.now() < expiry_time