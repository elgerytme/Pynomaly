"""JWT token handling utilities."""

import jwt
import secrets
from datetime import datetime, timedelta
from typing import Dict, Any, Optional


class JWTHandler:
    """JWT token creation and validation."""
    
    def __init__(self, secret_key: Optional[str] = None, algorithm: str = "HS256"):
        """Initialize JWT handler."""
        self.secret_key = secret_key or secrets.token_urlsafe(32)
        self.algorithm = algorithm
        
    def create_token(
        self,
        payload: Dict[str, Any],
        expires_in: int = 3600
    ) -> str:
        """Create a JWT token."""
        now = datetime.utcnow()
        payload.update({
            'iat': now,
            'exp': now + timedelta(seconds=expires_in),
            'jti': secrets.token_hex(16)
        })
        
        return jwt.encode(payload, self.secret_key, algorithm=self.algorithm)
    
    def verify_token(self, token: str) -> Optional[Dict[str, Any]]:
        """Verify and decode a JWT token."""
        try:
            payload = jwt.decode(token, self.secret_key, algorithms=[self.algorithm])
            return payload
        except jwt.ExpiredSignatureError:
            return None
        except jwt.InvalidTokenError:
            return None
    
    def refresh_token(self, token: str, expires_in: int = 3600) -> Optional[str]:
        """Refresh a JWT token."""
        payload = self.verify_token(token)
        if payload:
            # Remove old timing claims
            payload.pop('iat', None)
            payload.pop('exp', None)
            payload.pop('jti', None)
            return self.create_token(payload, expires_in)
        return None