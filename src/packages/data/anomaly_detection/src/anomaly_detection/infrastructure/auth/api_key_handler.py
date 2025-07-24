"""API key handling utilities."""

import secrets
import hashlib
from typing import Dict, Optional, Set
from datetime import datetime, timedelta


class APIKeyHandler:
    """API key creation and validation."""
    
    def __init__(self):
        """Initialize API key handler."""
        self.api_keys: Dict[str, Dict[str, Any]] = {}
        
    def generate_api_key(
        self,
        name: str,
        permissions: Optional[Set[str]] = None,
        expires_in_days: Optional[int] = None
    ) -> str:
        """Generate a new API key."""
        api_key = f"ak_{secrets.token_urlsafe(32)}"
        
        key_data = {
            'name': name,
            'permissions': permissions or set(),
            'created_at': datetime.utcnow(),
            'expires_at': None,
            'last_used': None,
            'usage_count': 0
        }
        
        if expires_in_days:
            key_data['expires_at'] = datetime.utcnow() + timedelta(days=expires_in_days)
        
        # Store hashed version
        hashed_key = self._hash_key(api_key)
        self.api_keys[hashed_key] = key_data
        
        return api_key
    
    def validate_api_key(self, api_key: str) -> Optional[Dict[str, Any]]:
        """Validate an API key."""
        hashed_key = self._hash_key(api_key)
        key_data = self.api_keys.get(hashed_key)
        
        if not key_data:
            return None
            
        # Check if expired
        if key_data['expires_at'] and datetime.utcnow() > key_data['expires_at']:
            return None
            
        # Update usage
        key_data['last_used'] = datetime.utcnow()
        key_data['usage_count'] += 1
        
        return key_data
    
    def revoke_api_key(self, api_key: str) -> bool:
        """Revoke an API key."""
        hashed_key = self._hash_key(api_key)
        if hashed_key in self.api_keys:
            del self.api_keys[hashed_key]
            return True
        return False
    
    def list_api_keys(self) -> Dict[str, Dict[str, Any]]:
        """List all API keys (without the actual keys)."""
        return {
            key_hash[:8] + "...": {
                'name': data['name'],
                'created_at': data['created_at'],
                'expires_at': data['expires_at'],
                'last_used': data['last_used'],
                'usage_count': data['usage_count']
            }
            for key_hash, data in self.api_keys.items()
        }
    
    def _hash_key(self, api_key: str) -> str:
        """Hash an API key for storage."""
        return hashlib.sha256(api_key.encode()).hexdigest()