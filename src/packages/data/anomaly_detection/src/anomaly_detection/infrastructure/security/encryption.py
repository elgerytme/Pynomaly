"""Data encryption module for secure data handling."""

import hashlib
import hmac
import secrets
import base64
from typing import Dict, Any, Optional, Tuple
from cryptography.hazmat.primitives import hashes, serialization
from cryptography.hazmat.primitives.asymmetric import rsa, padding
from cryptography.hazmat.primitives.ciphers import Cipher, algorithms, modes
from cryptography.hazmat.primitives.kdf.pbkdf2 import PBKDF2HMAC
import structlog

logger = structlog.get_logger(__name__)


class DataEncryption:
    """Data encryption and decryption service."""
    
    def __init__(self, key: Optional[bytes] = None):
        """Initialize encryption service."""
        self.key = key or self._generate_key()
        logger.info("DataEncryption initialized")
    
    def _generate_key(self) -> bytes:
        """Generate a new encryption key."""
        return secrets.token_bytes(32)
    
    def encrypt_data(self, data: bytes) -> bytes:
        """Encrypt data using AES encryption."""
        # Generate a random initialization vector
        iv = secrets.token_bytes(16)
        
        # Create cipher
        cipher = Cipher(algorithms.AES(self.key), modes.CBC(iv))
        encryptor = cipher.encryptor()
        
        # Pad data to match block size
        padding_length = 16 - (len(data) % 16)
        padded_data = data + bytes([padding_length] * padding_length)
        
        # Encrypt
        encrypted_data = encryptor.update(padded_data) + encryptor.finalize()
        
        # Combine IV and encrypted data
        return iv + encrypted_data
    
    def decrypt_data(self, encrypted_data: bytes) -> bytes:
        """Decrypt data using AES decryption."""
        # Extract IV and encrypted data
        iv = encrypted_data[:16]
        ciphertext = encrypted_data[16:]
        
        # Create cipher
        cipher = Cipher(algorithms.AES(self.key), modes.CBC(iv))
        decryptor = cipher.decryptor()
        
        # Decrypt
        padded_data = decryptor.update(ciphertext) + decryptor.finalize()
        
        # Remove padding
        padding_length = padded_data[-1]
        return padded_data[:-padding_length]
    
    def hash_data(self, data: bytes, salt: Optional[bytes] = None) -> Tuple[bytes, bytes]:
        """Hash data with optional salt."""
        if salt is None:
            salt = secrets.token_bytes(32)
        
        # Use PBKDF2 for secure hashing
        kdf = PBKDF2HMAC(
            algorithm=hashes.SHA256(),
            length=32,
            salt=salt,
            iterations=100000,
        )
        hash_value = kdf.derive(data)
        
        return hash_value, salt
    
    def verify_hash(self, data: bytes, hash_value: bytes, salt: bytes) -> bool:
        """Verify data against a hash."""
        try:
            kdf = PBKDF2HMAC(
                algorithm=hashes.SHA256(),
                length=32,
                salt=salt,
                iterations=100000,
            )
            kdf.verify(data, hash_value)
            return True
        except Exception:
            return False
    
    def encrypt_string(self, text: str) -> str:
        """Encrypt a string and return base64 encoded result."""
        encrypted_bytes = self.encrypt_data(text.encode('utf-8'))
        return base64.b64encode(encrypted_bytes).decode('utf-8')
    
    def decrypt_string(self, encrypted_text: str) -> str:
        """Decrypt a base64 encoded string."""
        encrypted_bytes = base64.b64decode(encrypted_text.encode('utf-8'))
        decrypted_bytes = self.decrypt_data(encrypted_bytes)
        return decrypted_bytes.decode('utf-8')


# Global instance
_data_encryption: Optional[DataEncryption] = None


def get_data_encryption() -> DataEncryption:
    """Get the global data encryption instance."""
    global _data_encryption
    
    if _data_encryption is None:
        _data_encryption = DataEncryption()
    
    return _data_encryption