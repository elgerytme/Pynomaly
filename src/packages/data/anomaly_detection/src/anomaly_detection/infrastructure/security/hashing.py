"""Secure hashing utilities."""

import hashlib
import hmac
import secrets
from typing import Optional


class SecureHasher:
    """Secure password and data hashing utility."""
    
    def __init__(self, algorithm: str = "sha256"):
        """Initialize with specified hashing algorithm."""
        self.algorithm = algorithm
        
    def hash_password(self, password: str, salt: Optional[str] = None) -> tuple[str, str]:
        """Hash a password with salt."""
        if salt is None:
            salt = secrets.token_hex(32)
        
        hash_obj = hashlib.new(self.algorithm)
        hash_obj.update((password + salt).encode('utf-8'))
        hashed = hash_obj.hexdigest()
        
        return hashed, salt
    
    def verify_password(self, password: str, hashed: str, salt: str) -> bool:
        """Verify a password against its hash."""
        test_hash, _ = self.hash_password(password, salt)
        return hmac.compare_digest(test_hash, hashed)
    
    def hash_data(self, data: str) -> str:
        """Hash arbitrary data."""
        hash_obj = hashlib.new(self.algorithm)
        hash_obj.update(data.encode('utf-8'))
        return hash_obj.hexdigest()
    
    def generate_salt(self) -> str:
        """Generate a random salt."""
        return secrets.token_hex(32)