"""
Token service for authentication application layer
"""
import jwt
from datetime import datetime, timedelta
from typing import Dict, Optional
from uuid import UUID
from ...domain.entities.user import User

class TokenService:
    """Application service for JWT token management"""
    
    def __init__(self, secret_key: str, access_token_expiry: int = 3600):
        self.secret_key = secret_key
        self.access_token_expiry = access_token_expiry  # seconds
        self.refresh_token_expiry = 86400 * 7  # 7 days
        self.algorithm = 'HS256'
    
    def generate_access_token(self, user: User) -> str:
        """Generate JWT access token"""
        payload = {
            'user_id': str(user.id),
            'username': user.username,
            'email': user.email,
            'iat': datetime.utcnow(),
            'exp': datetime.utcnow() + timedelta(seconds=self.access_token_expiry),
            'type': 'access'
        }
        
        return jwt.encode(payload, self.secret_key, algorithm=self.algorithm)
    
    def generate_refresh_token(self, user: User) -> str:
        """Generate JWT refresh token"""
        payload = {
            'user_id': str(user.id),
            'iat': datetime.utcnow(),
            'exp': datetime.utcnow() + timedelta(seconds=self.refresh_token_expiry),
            'type': 'refresh'
        }
        
        return jwt.encode(payload, self.secret_key, algorithm=self.algorithm)
    
    def decode_token(self, token: str) -> Optional[Dict]:
        """Decode and validate JWT token"""
        try:
            payload = jwt.decode(token, self.secret_key, algorithms=[self.algorithm])
            return payload
        except jwt.ExpiredSignatureError:
            return None
        except jwt.InvalidTokenError:
            return None
    
    def is_token_valid(self, token: str) -> bool:
        """Check if token is valid"""
        payload = self.decode_token(token)
        return payload is not None
    
    def get_user_id_from_token(self, token: str) -> Optional[UUID]:
        """Extract user ID from token"""
        payload = self.decode_token(token)
        if payload:
            try:
                return UUID(payload.get('user_id'))
            except (ValueError, TypeError):
                return None
        return None
    
    def is_access_token(self, token: str) -> bool:
        """Check if token is an access token"""
        payload = self.decode_token(token)
        return payload and payload.get('type') == 'access'
    
    def is_refresh_token(self, token: str) -> bool:
        """Check if token is a refresh token"""
        payload = self.decode_token(token)
        return payload and payload.get('type') == 'refresh'