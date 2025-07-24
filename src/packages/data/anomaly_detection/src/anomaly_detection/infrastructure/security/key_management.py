"""Key management utilities."""

import secrets
import base64
from typing import Dict, Optional
from cryptography.fernet import Fernet


class KeyManager:
    """Manages encryption keys and key rotation."""
    
    def __init__(self):
        """Initialize key manager."""
        self._keys: Dict[str, bytes] = {}
        self._current_key_id: Optional[str] = None
    
    def generate_key(self, key_id: str) -> str:
        """Generate a new encryption key."""
        key = Fernet.generate_key()
        self._keys[key_id] = key
        self._current_key_id = key_id
        return base64.urlsafe_b64encode(key).decode()
    
    def get_key(self, key_id: str) -> Optional[bytes]:
        """Get a key by ID."""
        return self._keys.get(key_id)
    
    def get_current_key(self) -> Optional[bytes]:
        """Get the current active key."""
        if self._current_key_id:
            return self._keys.get(self._current_key_id)
        return None
    
    def rotate_key(self) -> str:
        """Rotate to a new key."""
        new_key_id = secrets.token_hex(16)
        return self.generate_key(new_key_id)
    
    def list_keys(self) -> list[str]:
        """List all key IDs."""
        return list(self._keys.keys())
    
    def delete_key(self, key_id: str) -> bool:
        """Delete a key."""
        if key_id in self._keys:
            del self._keys[key_id]
            if self._current_key_id == key_id:
                self._current_key_id = None
            return True
        return False