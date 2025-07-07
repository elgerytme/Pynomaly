"""Service for managing API keys."""

import secrets
import hashlib
from datetime import datetime, timedelta
from typing import List, Optional

from pydantic import BaseModel


class ApiKey(BaseModel):
    """API Key model."""
    id: str
    name: str
    description: Optional[str] = None
    scopes: List[str] = []
    is_active: bool = True
    created_at: datetime
    updated_at: datetime
    last_used_at: Optional[datetime] = None
    expires_at: Optional[datetime] = None
    rate_limit: Optional[int] = None
    usage_count: int = 0
    secret_hash: str


class ApiKeyService:
    """Service for managing API keys."""
    
    def __init__(self):
        self._keys = {}
    
    def create_key(
        self,
        name: str,
        description: Optional[str] = None,
        scopes: List[str] = None,
        expires_at: Optional[datetime] = None,
        rate_limit: Optional[int] = None
    ) -> tuple[ApiKey, str]:
        """Create a new API key.
        
        Returns:
            Tuple of (ApiKey, secret) where secret is only returned once
        """
        key_id = f"ak_{secrets.token_hex(16)}"
        secret = f"sk_{secrets.token_hex(32)}"
        secret_hash = hashlib.sha256(secret.encode()).hexdigest()
        
        now = datetime.utcnow()
        
        api_key = ApiKey(
            id=key_id,
            name=name,
            description=description,
            scopes=scopes or [],
            is_active=True,
            created_at=now,
            updated_at=now,
            expires_at=expires_at,
            rate_limit=rate_limit,
            usage_count=0,
            secret_hash=secret_hash
        )
        
        self._keys[key_id] = api_key
        return api_key, secret
    
    def get_key(self, key_id: str) -> Optional[ApiKey]:
        """Get API key by ID."""
        return self._keys.get(key_id)
    
    def list_keys(self, limit: int = 10, offset: int = 0) -> List[ApiKey]:
        """List API keys with pagination."""
        keys = list(self._keys.values())
        return keys[offset:offset + limit]
    
    def update_key(
        self,
        key_id: str,
        name: Optional[str] = None,
        description: Optional[str] = None,
        scopes: Optional[List[str]] = None,
        expires_at: Optional[datetime] = None,
        rate_limit: Optional[int] = None,
        is_active: Optional[bool] = None
    ) -> Optional[ApiKey]:
        """Update an existing API key."""
        if key_id not in self._keys:
            return None
        
        key = self._keys[key_id]
        
        if name is not None:
            key.name = name
        if description is not None:
            key.description = description
        if scopes is not None:
            key.scopes = scopes
        if expires_at is not None:
            key.expires_at = expires_at
        if rate_limit is not None:
            key.rate_limit = rate_limit
        if is_active is not None:
            key.is_active = is_active
        
        key.updated_at = datetime.utcnow()
        return key
    
    def delete_key(self, key_id: str) -> bool:
        """Delete an API key."""
        if key_id in self._keys:
            del self._keys[key_id]
            return True
        return False
    
    def verify_key(self, secret: str) -> Optional[ApiKey]:
        """Verify API key secret and return the key if valid."""
        secret_hash = hashlib.sha256(secret.encode()).hexdigest()
        
        for key in self._keys.values():
            if key.secret_hash == secret_hash and key.is_active:
                # Check expiration
                if key.expires_at and datetime.utcnow() > key.expires_at:
                    continue
                
                # Update usage
                key.last_used_at = datetime.utcnow()
                key.usage_count += 1
                
                return key
        
        return None
