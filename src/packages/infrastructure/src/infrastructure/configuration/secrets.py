"""Secrets management for infrastructure components."""

from __future__ import annotations

import os
import json
import base64
from typing import Any, Dict, Optional
from cryptography.fernet import Fernet
from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.primitives.kdf.pbkdf2 import PBKDF2HMAC


class SecretsManager:
    """Manages encrypted secrets for infrastructure components."""
    
    def __init__(self, master_key: Optional[str] = None):
        """Initialize secrets manager with optional master key."""
        self._master_key = master_key or os.getenv("SECRETS_MASTER_KEY")
        self._cipher_suite = None
        
        if self._master_key:
            self._initialize_cipher()
    
    def _initialize_cipher(self) -> None:
        """Initialize the cipher suite for encryption/decryption."""
        if not self._master_key:
            raise ValueError("Master key required for encryption operations")
        
        # Derive key from master key
        kdf = PBKDF2HMAC(
            algorithm=hashes.SHA256(),
            length=32,
            salt=b'infrastructure_salt',  # In production, use random salt
            iterations=100000,
        )
        key = base64.urlsafe_b64encode(kdf.derive(self._master_key.encode()))
        self._cipher_suite = Fernet(key)
    
    def encrypt_secret(self, secret: str) -> str:
        """Encrypt a secret value."""
        if not self._cipher_suite:
            raise ValueError("Cipher suite not initialized")
        
        encrypted_data = self._cipher_suite.encrypt(secret.encode())
        return base64.urlsafe_b64encode(encrypted_data).decode()
    
    def decrypt_secret(self, encrypted_secret: str) -> str:
        """Decrypt an encrypted secret value."""
        if not self._cipher_suite:
            raise ValueError("Cipher suite not initialized")
        
        encrypted_data = base64.urlsafe_b64decode(encrypted_secret.encode())
        decrypted_data = self._cipher_suite.decrypt(encrypted_data)
        return decrypted_data.decode()
    
    def get_secret(self, key: str, default: Optional[str] = None) -> Optional[str]:
        """Get a secret value from environment or encrypted storage."""
        # First try environment variable
        env_value = os.getenv(key)
        if env_value:
            # Check if it's encrypted (starts with 'encrypted:')
            if env_value.startswith('encrypted:'):
                encrypted_value = env_value[10:]  # Remove 'encrypted:' prefix
                return self.decrypt_secret(encrypted_value)
            return env_value
        
        # Try encrypted secrets file
        secrets_file = os.getenv("SECRETS_FILE", ".secrets.json")
        if os.path.exists(secrets_file):
            try:
                with open(secrets_file, 'r') as f:
                    secrets_data = json.load(f)
                
                if key in secrets_data:
                    encrypted_value = secrets_data[key]
                    return self.decrypt_secret(encrypted_value)
            except (IOError, json.JSONDecodeError, ValueError):
                pass  # Fall through to default
        
        return default
    
    def store_secret(self, key: str, value: str, file_path: Optional[str] = None) -> None:
        """Store an encrypted secret to file."""
        if not self._cipher_suite:
            raise ValueError("Cipher suite not initialized")
        
        secrets_file = file_path or os.getenv("SECRETS_FILE", ".secrets.json")
        
        # Load existing secrets
        secrets_data = {}
        if os.path.exists(secrets_file):
            try:
                with open(secrets_file, 'r') as f:
                    secrets_data = json.load(f)
            except (IOError, json.JSONDecodeError):
                pass
        
        # Encrypt and store the secret
        encrypted_value = self.encrypt_secret(value)
        secrets_data[key] = encrypted_value
        
        # Write back to file
        with open(secrets_file, 'w') as f:
            json.dump(secrets_data, f, indent=2)
    
    def rotate_secret(self, key: str, new_value: str) -> str:
        """Rotate a secret by storing new value and returning old one."""
        old_value = self.get_secret(key)
        self.store_secret(key, new_value)
        return old_value or ""
    
    def delete_secret(self, key: str, file_path: Optional[str] = None) -> bool:
        """Delete a secret from encrypted storage."""
        secrets_file = file_path or os.getenv("SECRETS_FILE", ".secrets.json")
        
        if not os.path.exists(secrets_file):
            return False
        
        try:
            with open(secrets_file, 'r') as f:
                secrets_data = json.load(f)
            
            if key in secrets_data:
                del secrets_data[key]
                
                with open(secrets_file, 'w') as f:
                    json.dump(secrets_data, f, indent=2)
                
                return True
        except (IOError, json.JSONDecodeError):
            pass
        
        return False
    
    def list_secrets(self, file_path: Optional[str] = None) -> list[str]:
        """List all secret keys (not values) from encrypted storage."""
        secrets_file = file_path or os.getenv("SECRETS_FILE", ".secrets.json")
        
        if not os.path.exists(secrets_file):
            return []
        
        try:
            with open(secrets_file, 'r') as f:
                secrets_data = json.load(f)
            
            return list(secrets_data.keys())
        except (IOError, json.JSONDecodeError):
            return []
    
    def validate_secret(self, key: str) -> bool:
        """Validate that a secret exists and can be decrypted."""
        try:
            secret = self.get_secret(key)
            return secret is not None
        except Exception:
            return False
    
    @classmethod
    def generate_master_key(cls) -> str:
        """Generate a new master key for secrets encryption."""
        return base64.urlsafe_b64encode(os.urandom(32)).decode()
    
    def health_check(self) -> Dict[str, Any]:
        """Perform health check on secrets management."""
        status = {
            "status": "healthy",
            "cipher_initialized": self._cipher_suite is not None,
            "master_key_available": self._master_key is not None,
            "secrets_file_exists": False,
            "secrets_count": 0
        }
        
        secrets_file = os.getenv("SECRETS_FILE", ".secrets.json")
        if os.path.exists(secrets_file):
            status["secrets_file_exists"] = True
            try:
                with open(secrets_file, 'r') as f:
                    secrets_data = json.load(f)
                status["secrets_count"] = len(secrets_data)
            except (IOError, json.JSONDecodeError):
                status["status"] = "degraded"
                status["error"] = "Failed to read secrets file"
        
        return status


# Global secrets manager instance
_secrets_manager: Optional[SecretsManager] = None


def get_secrets_manager() -> SecretsManager:
    """Get the global secrets manager instance."""
    global _secrets_manager
    if _secrets_manager is None:
        _secrets_manager = SecretsManager()
    return _secrets_manager


def get_secret(key: str, default: Optional[str] = None) -> Optional[str]:
    """Convenience function to get a secret."""
    return get_secrets_manager().get_secret(key, default)