"""Advanced encryption services for data protection.

This module provides comprehensive encryption capabilities including:
- Data-at-rest encryption
- Field-level encryption
- Key management
- Symmetric and asymmetric encryption
- Secure key derivation
"""

from __future__ import annotations

import base64
import logging
import os
import time
from enum import Enum
from typing import Any

from cryptography.fernet import Fernet
from cryptography.hazmat.backends import default_backend
from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.primitives.ciphers import Cipher, algorithms, modes
from cryptography.hazmat.primitives.kdf.pbkdf2 import PBKDF2HMAC
from cryptography.hazmat.primitives.kdf.scrypt import Scrypt
from pydantic import BaseModel, Field, field_validator, ConfigDict

logger = logging.getLogger(__name__)


class EncryptionAlgorithm(str, Enum):
    """Supported encryption algorithms."""

    FERNET = "fernet"  # Symmetric, recommended for most use cases
    AES_GCM = "aes_gcm"  # AES with Galois/Counter Mode
    AES_CBC = "aes_cbc"  # AES with Cipher Block Chaining
    RSA = "rsa"  # RSA asymmetric encryption
    CHACHA20 = "chacha20"  # ChaCha20 symmetric encryption


class KeyDerivationMethod(str, Enum):
    """Key derivation methods."""

    PBKDF2 = "pbkdf2"
    SCRYPT = "scrypt"


class EncryptionConfig(BaseModel):
    """Configuration for encryption services."""

    algorithm: EncryptionAlgorithm = EncryptionAlgorithm.FERNET
    key_derivation: KeyDerivationMethod = KeyDerivationMethod.PBKDF2
    key_length: int = Field(default=32, ge=16, le=64)  # 256 bits default
    iterations: int = Field(default=100000, ge=1000)  # PBKDF2 iterations
    salt_length: int = Field(default=16, ge=8)  # Salt length in bytes
    enable_compression: bool = False
    enable_key_rotation: bool = True
    max_key_age_days: int = Field(default=90, ge=1)

    @field_validator("key_length")
    @classmethod
    def validate_key_length(cls, v):
        """Validate key length based on algorithm."""
        if v not in [16, 24, 32]:
            raise ValueError("Key length must be 16, 24, or 32 bytes")
        return v


class EncryptionError(Exception):
    """Base exception for encryption errors."""

    pass


class KeyManagementError(EncryptionError):
    """Exception for key management errors."""

    pass


class DecryptionError(EncryptionError):
    """Exception for decryption errors."""

    pass


class EncryptionKey(BaseModel):
    """Represents an encryption key with metadata."""

    key_id: str
    algorithm: EncryptionAlgorithm
    key_data: bytes
    salt: bytes | None = None
    created_at: float  # Unix timestamp
    expires_at: float | None = None
    is_active: bool = True
    metadata: dict[str, Any] = Field(default_factory=dict)

    model_config = ConfigDict(arbitrary_types_allowed = True)


class EncryptionService:
    """Service for handling encryption operations."""

    def __init__(self, config: EncryptionConfig | None = None):
        """Initialize encryption service.

        Args:
            config: Encryption configuration
        """
        self.config = config or EncryptionConfig()
        self.backend = default_backend()
        self._keys: dict[str, EncryptionKey] = {}
        self._active_key_id: str | None = None

        # Initialize master key if not provided
        self._master_key = self._load_or_generate_master_key()

    def _load_or_generate_master_key(self) -> bytes:
        """Load or generate the master encryption key."""
        # In production, this should be loaded from a secure key management service
        # For now, we'll use environment variable or generate one
        master_key_env = os.environ.get("PYNOMALY_MASTER_KEY")

        if master_key_env:
            try:
                return base64.b64decode(master_key_env)
            except Exception as e:
                logger.warning(f"Failed to decode master key from environment: {e}")

        # Generate a new master key
        master_key = os.urandom(32)  # 256 bits
        encoded_key = base64.b64encode(master_key).decode()

        logger.warning(
            "Generated new master key. In production, store this securely: "
            f"PYNOMALY_MASTER_KEY={encoded_key}"
        )

        return master_key

    def derive_key(
        self, password: str, salt: bytes | None = None
    ) -> tuple[bytes, bytes]:
        """Derive encryption key from password.

        Args:
            password: Password to derive key from
            salt: Optional salt (random salt generated if None)

        Returns:
            Tuple of (derived_key, salt)
        """
        if salt is None:
            salt = os.urandom(self.config.salt_length)

        if self.config.key_derivation == KeyDerivationMethod.PBKDF2:
            kdf = PBKDF2HMAC(
                algorithm=hashes.SHA256(),
                length=self.config.key_length,
                salt=salt,
                iterations=self.config.iterations,
                backend=self.backend,
            )
        else:  # SCRYPT
            kdf = Scrypt(
                algorithm=hashes.SHA256(),
                length=self.config.key_length,
                salt=salt,
                n=2**14,  # CPU/memory cost parameter
                r=8,  # Block size parameter
                p=1,  # Parallelization parameter
                backend=self.backend,
            )

        key = kdf.derive(password.encode())
        return key, salt

    def create_key(self, password: str | None = None, key_id: str | None = None) -> str:
        """Create a new encryption key.

        Args:
            password: Optional password for key derivation
            key_id: Optional key identifier

        Returns:
            Key ID
        """
        if key_id is None:
            key_id = f"key_{len(self._keys) + 1}_{int(os.urandom(4).hex(), 16)}"

        if password:
            key_data, salt = self.derive_key(password)
        else:
            key_data = os.urandom(self.config.key_length)
            salt = None

        # Calculate expiration if key rotation is enabled
        expires_at = None
        if self.config.enable_key_rotation:
            import time

            expires_at = time.time() + (self.config.max_key_age_days * 24 * 3600)

        encryption_key = EncryptionKey(
            key_id=key_id,
            algorithm=self.config.algorithm,
            key_data=key_data,
            salt=salt,
            created_at=time.time(),
            expires_at=expires_at,
        )

        self._keys[key_id] = encryption_key

        # Set as active key if none exists
        if self._active_key_id is None:
            self._active_key_id = key_id

        logger.info(f"Created encryption key: {key_id}")
        return key_id

    def get_active_key(self) -> EncryptionKey:
        """Get the active encryption key.

        Returns:
            Active encryption key

        Raises:
            KeyManagementError: If no active key exists
        """
        if not self._active_key_id or self._active_key_id not in self._keys:
            raise KeyManagementError("No active encryption key available")

        key = self._keys[self._active_key_id]

        # Check if key has expired
        if key.expires_at and time.time() > key.expires_at:
            logger.warning(f"Active key {self._active_key_id} has expired")
            # Could trigger automatic key rotation here

        return key

    def encrypt(self, data: str | bytes, key_id: str | None = None) -> dict[str, Any]:
        """Encrypt data.

        Args:
            data: Data to encrypt
            key_id: Optional key ID (uses active key if None)

        Returns:
            Dictionary with encrypted data and metadata

        Raises:
            EncryptionError: If encryption fails
        """
        if isinstance(data, str):
            data = data.encode()

        # Get encryption key
        if key_id:
            if key_id not in self._keys:
                raise KeyManagementError(f"Key not found: {key_id}")
            key = self._keys[key_id]
        else:
            key = self.get_active_key()

        try:
            if key.algorithm == EncryptionAlgorithm.FERNET:
                encrypted_data = self._encrypt_fernet(data, key.key_data)
            elif key.algorithm == EncryptionAlgorithm.AES_GCM:
                encrypted_data = self._encrypt_aes_gcm(data, key.key_data)
            elif key.algorithm == EncryptionAlgorithm.AES_CBC:
                encrypted_data = self._encrypt_aes_cbc(data, key.key_data)
            else:
                raise EncryptionError(f"Unsupported algorithm: {key.algorithm}")

            return {
                "data": encrypted_data,
                "key_id": key.key_id,
                "algorithm": key.algorithm,
                "encrypted_at": time.time(),
            }

        except Exception as e:
            raise EncryptionError(f"Encryption failed: {e}")

    def decrypt(self, encrypted_data: dict[str, Any]) -> bytes:
        """Decrypt data.

        Args:
            encrypted_data: Encrypted data dictionary from encrypt()

        Returns:
            Decrypted data

        Raises:
            DecryptionError: If decryption fails
        """
        key_id = encrypted_data.get("key_id")
        if not key_id or key_id not in self._keys:
            raise KeyManagementError(f"Decryption key not found: {key_id}")

        key = self._keys[key_id]
        algorithm = encrypted_data.get("algorithm", key.algorithm)
        data = encrypted_data["data"]

        try:
            if algorithm == EncryptionAlgorithm.FERNET:
                return self._decrypt_fernet(data, key.key_data)
            elif algorithm == EncryptionAlgorithm.AES_GCM:
                return self._decrypt_aes_gcm(data, key.key_data)
            elif algorithm == EncryptionAlgorithm.AES_CBC:
                return self._decrypt_aes_cbc(data, key.key_data)
            else:
                raise DecryptionError(f"Unsupported algorithm: {algorithm}")

        except Exception as e:
            raise DecryptionError(f"Decryption failed: {e}")

    def _encrypt_fernet(self, data: bytes, key: bytes) -> bytes:
        """Encrypt using Fernet."""
        # Fernet requires a 32-byte key
        if len(key) != 32:
            # Derive 32-byte key from provided key
            digest = hashes.Hash(hashes.SHA256(), backend=self.backend)
            digest.update(key)
            key = digest.finalize()

        fernet_key = base64.urlsafe_b64encode(key)
        f = Fernet(fernet_key)
        return f.encrypt(data)

    def _decrypt_fernet(self, data: bytes, key: bytes) -> bytes:
        """Decrypt using Fernet."""
        if len(key) != 32:
            digest = hashes.Hash(hashes.SHA256(), backend=self.backend)
            digest.update(key)
            key = digest.finalize()

        fernet_key = base64.urlsafe_b64encode(key)
        f = Fernet(fernet_key)
        return f.decrypt(data)

    def _encrypt_aes_gcm(self, data: bytes, key: bytes) -> bytes:
        """Encrypt using AES-GCM."""
        iv = os.urandom(12)  # 96-bit IV for GCM
        cipher = Cipher(algorithms.AES(key), modes.GCM(iv), backend=self.backend)
        encryptor = cipher.encryptor()

        ciphertext = encryptor.update(data) + encryptor.finalize()

        # Return IV + tag + ciphertext
        return iv + encryptor.tag + ciphertext

    def _decrypt_aes_gcm(self, data: bytes, key: bytes) -> bytes:
        """Decrypt using AES-GCM."""
        iv = data[:12]
        tag = data[12:28]
        ciphertext = data[28:]

        cipher = Cipher(algorithms.AES(key), modes.GCM(iv, tag), backend=self.backend)
        decryptor = cipher.decryptor()

        return decryptor.update(ciphertext) + decryptor.finalize()

    def _encrypt_aes_cbc(self, data: bytes, key: bytes) -> bytes:
        """Encrypt using AES-CBC."""
        # Pad data to block size
        block_size = 16
        padding_length = block_size - (len(data) % block_size)
        padded_data = data + bytes([padding_length] * padding_length)

        iv = os.urandom(16)  # 128-bit IV
        cipher = Cipher(algorithms.AES(key), modes.CBC(iv), backend=self.backend)
        encryptor = cipher.encryptor()

        ciphertext = encryptor.update(padded_data) + encryptor.finalize()

        # Return IV + ciphertext
        return iv + ciphertext

    def _decrypt_aes_cbc(self, data: bytes, key: bytes) -> bytes:
        """Decrypt using AES-CBC."""
        iv = data[:16]
        ciphertext = data[16:]

        cipher = Cipher(algorithms.AES(key), modes.CBC(iv), backend=self.backend)
        decryptor = cipher.decryptor()

        padded_data = decryptor.update(ciphertext) + decryptor.finalize()

        # Remove padding
        padding_length = padded_data[-1]
        return padded_data[:-padding_length]


class FieldEncryption:
    """Service for field-level encryption in databases."""

    def __init__(self, encryption_service: EncryptionService):
        """Initialize field encryption.

        Args:
            encryption_service: Base encryption service
        """
        self.encryption_service = encryption_service
        self._encrypted_fields: dict[str, list[str]] = {}  # table -> [fields]

    def register_encrypted_field(self, table: str, field: str) -> None:
        """Register a field for automatic encryption.

        Args:
            table: Database table name
            field: Field name to encrypt
        """
        if table not in self._encrypted_fields:
            self._encrypted_fields[table] = []

        if field not in self._encrypted_fields[table]:
            self._encrypted_fields[table].append(field)
            logger.info(f"Registered encrypted field: {table}.{field}")

    def encrypt_record(self, table: str, record: dict[str, Any]) -> dict[str, Any]:
        """Encrypt sensitive fields in a database record.

        Args:
            table: Table name
            record: Database record

        Returns:
            Record with encrypted fields
        """
        if table not in self._encrypted_fields:
            return record

        encrypted_record = record.copy()

        for field in self._encrypted_fields[table]:
            if field in record and record[field] is not None:
                # Convert to string if needed
                value = str(record[field])

                # Encrypt the field
                encrypted_data = self.encryption_service.encrypt(value)

                # Store as JSON string or use a structured format
                import json

                encrypted_record[field] = json.dumps(encrypted_data)

                logger.debug(f"Encrypted field {table}.{field}")

        return encrypted_record

    def decrypt_record(self, table: str, record: dict[str, Any]) -> dict[str, Any]:
        """Decrypt sensitive fields in a database record.

        Args:
            table: Table name
            record: Database record with encrypted fields

        Returns:
            Record with decrypted fields
        """
        if table not in self._encrypted_fields:
            return record

        decrypted_record = record.copy()

        for field in self._encrypted_fields[table]:
            if field in record and record[field] is not None:
                try:
                    # Parse encrypted data
                    import json

                    encrypted_data = json.loads(record[field])

                    # Decrypt the field
                    decrypted_bytes = self.encryption_service.decrypt(encrypted_data)
                    decrypted_record[field] = decrypted_bytes.decode()

                    logger.debug(f"Decrypted field {table}.{field}")

                except (json.JSONDecodeError, DecryptionError) as e:
                    logger.error(f"Failed to decrypt field {table}.{field}: {e}")
                    # Keep original value or set to None based on requirements
                    decrypted_record[field] = None

        return decrypted_record


class DataEncryption:
    """High-level service for data encryption in the application."""

    def __init__(self, config: EncryptionConfig | None = None):
        """Initialize data encryption service.

        Args:
            config: Encryption configuration
        """
        self.encryption_service = EncryptionService(config)
        self.field_encryption = FieldEncryption(self.encryption_service)

        # Create default encryption key
        self.default_key_id = self.encryption_service.create_key()

    def encrypt_sensitive_data(self, data: Any) -> str:
        """Encrypt sensitive data for storage.

        Args:
            data: Data to encrypt

        Returns:
            Base64-encoded encrypted data
        """
        import json

        # Convert data to JSON string
        if isinstance(data, dict | list):
            json_data = json.dumps(data)
        else:
            json_data = str(data)

        # Encrypt
        encrypted = self.encryption_service.encrypt(json_data)

        # Return as base64 string for easy storage
        import json

        return base64.b64encode(json.dumps(encrypted).encode()).decode()

    def decrypt_sensitive_data(
        self, encrypted_data: str, return_type: type = str
    ) -> Any:
        """Decrypt sensitive data from storage.

        Args:
            encrypted_data: Base64-encoded encrypted data
            return_type: Expected return type

        Returns:
            Decrypted data
        """
        import json

        # Decode from base64
        encrypted_dict = json.loads(base64.b64decode(encrypted_data).decode())

        # Decrypt
        decrypted_bytes = self.encryption_service.decrypt(encrypted_dict)
        decrypted_str = decrypted_bytes.decode()

        # Convert to appropriate type
        if return_type in (dict, list):
            return json.loads(decrypted_str)
        else:
            return decrypted_str


# Global encryption service instance
_encryption_service: DataEncryption | None = None


def get_encryption_service() -> DataEncryption:
    """Get global encryption service instance."""
    global _encryption_service
    if _encryption_service is None:
        _encryption_service = DataEncryption()
    return _encryption_service


def init_encryption_service(
    config: EncryptionConfig | None = None,
) -> DataEncryption:
    """Initialize global encryption service.

    Args:
        config: Encryption configuration

    Returns:
        Encryption service instance
    """
    global _encryption_service
    _encryption_service = DataEncryption(config)
    return _encryption_service
