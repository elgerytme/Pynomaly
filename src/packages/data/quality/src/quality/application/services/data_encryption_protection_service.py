"""
Data Encryption and Protection Service - Comprehensive data encryption and protection for enterprise security.

This service provides end-to-end encryption, key management, data protection at rest and in transit,
and enterprise-grade cryptographic security features.
"""

import asyncio
import logging
import os
import base64
import json
from typing import Dict, List, Optional, Any, Union, Tuple, Set
from datetime import datetime, timedelta
from dataclasses import dataclass, field
from enum import Enum
import hashlib
import secrets
from cryptography.fernet import Fernet
from cryptography.hazmat.primitives import hashes, serialization
from cryptography.hazmat.primitives.asymmetric import rsa, padding
from cryptography.hazmat.primitives.ciphers import Cipher, algorithms, modes
from cryptography.hazmat.primitives.kdf.pbkdf2 import PBKDF2HMAC

from ...domain.interfaces.data_quality_interface import DataQualityInterface

logger = logging.getLogger(__name__)


class EncryptionAlgorithm(Enum):
    """Supported encryption algorithms."""
    AES_256_GCM = "aes_256_gcm"
    AES_256_CBC = "aes_256_cbc"
    CHACHA20_POLY1305 = "chacha20_poly1305"
    FERNET = "fernet"
    RSA_2048 = "rsa_2048"
    RSA_4096 = "rsa_4096"


class KeyType(Enum):
    """Types of encryption keys."""
    SYMMETRIC = "symmetric"
    ASYMMETRIC_PUBLIC = "asymmetric_public"
    ASYMMETRIC_PRIVATE = "asymmetric_private"
    MASTER_KEY = "master_key"
    DATA_ENCRYPTION_KEY = "data_encryption_key"
    KEY_ENCRYPTION_KEY = "key_encryption_key"


class ProtectionLevel(Enum):
    """Data protection levels."""
    PUBLIC = "public"
    INTERNAL = "internal"
    CONFIDENTIAL = "confidential"
    SECRET = "secret"
    TOP_SECRET = "top_secret"


@dataclass
class EncryptionKey:
    """Encryption key metadata and data."""
    key_id: str
    key_type: KeyType
    algorithm: EncryptionAlgorithm
    key_data: bytes
    protection_level: ProtectionLevel = ProtectionLevel.CONFIDENTIAL
    purpose: str = "general"
    created_at: datetime = field(default_factory=datetime.utcnow)
    expires_at: Optional[datetime] = None
    last_used: Optional[datetime] = None
    usage_count: int = 0
    is_active: bool = True
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class EncryptedData:
    """Encrypted data container."""
    data_id: str
    encrypted_data: bytes
    encryption_algorithm: EncryptionAlgorithm
    key_id: str
    initialization_vector: Optional[bytes] = None
    authentication_tag: Optional[bytes] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    encrypted_at: datetime = field(default_factory=datetime.utcnow)
    data_size_bytes: int = 0
    checksum: str = ""


@dataclass
class EncryptionPolicy:
    """Data encryption policy."""
    policy_id: str
    name: str
    description: str
    data_classification: ProtectionLevel
    required_algorithm: EncryptionAlgorithm
    key_rotation_days: int = 90
    require_encryption_at_rest: bool = True
    require_encryption_in_transit: bool = True
    require_field_level_encryption: bool = False
    allowed_key_types: List[KeyType] = field(default_factory=lambda: [KeyType.SYMMETRIC])
    compliance_requirements: List[str] = field(default_factory=list)
    created_at: datetime = field(default_factory=datetime.utcnow)
    is_active: bool = True


@dataclass
class KeyRotationEvent:
    """Key rotation event record."""
    event_id: str
    old_key_id: str
    new_key_id: str
    rotation_reason: str
    affected_data_count: int = 0
    rotation_time_seconds: float = 0.0
    created_at: datetime = field(default_factory=datetime.utcnow)
    completed_at: Optional[datetime] = None
    status: str = "in_progress"  # in_progress, completed, failed


class DataEncryptionProtectionService:
    """Comprehensive data encryption and protection service."""
    
    def __init__(self):
        """Initialize the data encryption protection service."""
        self.encryption_keys: Dict[str, EncryptionKey] = {}
        self.encrypted_data_registry: Dict[str, EncryptedData] = {}
        self.encryption_policies: Dict[str, EncryptionPolicy] = {}
        self.key_rotation_history: List[KeyRotationEvent] = []
        self.master_key: Optional[bytes] = None
        
        # Configuration
        self.default_key_size = 32  # 256 bits
        self.key_derivation_iterations = 100000
        self.auto_key_rotation_enabled = True
        self.key_rotation_check_interval_hours = 24
        
        # Initialize master key and default policies
        self._initialize_master_key()
        self._initialize_default_policies()
        
        # Start background tasks
        asyncio.create_task(self._key_rotation_monitor())
        asyncio.create_task(self._encryption_health_monitor())
        
        logger.info("Data Encryption and Protection Service initialized")
    
    def _initialize_master_key(self):
        """Initialize or load the master key."""
        # In production, this would load from secure key management system
        master_key_path = "./master_key.key"
        
        if os.path.exists(master_key_path):
            with open(master_key_path, "rb") as f:
                self.master_key = f.read()
        else:
            # Generate new master key
            self.master_key = secrets.token_bytes(32)
            
            # Save master key securely (in production, use HSM or KMS)
            os.makedirs(os.path.dirname(master_key_path) if os.path.dirname(master_key_path) else ".", exist_ok=True)
            with open(master_key_path, "wb") as f:
                f.write(self.master_key)
            
            logger.info("Generated new master key")
    
    def _initialize_default_policies(self):
        """Initialize default encryption policies."""
        policies = [
            EncryptionPolicy(
                policy_id="pol_public",
                name="Public Data Policy",
                description="Policy for public data - no encryption required",
                data_classification=ProtectionLevel.PUBLIC,
                required_algorithm=EncryptionAlgorithm.AES_256_GCM,
                require_encryption_at_rest=False,
                require_encryption_in_transit=True,
                key_rotation_days=365
            ),
            EncryptionPolicy(
                policy_id="pol_internal",
                name="Internal Data Policy", 
                description="Policy for internal data - basic encryption",
                data_classification=ProtectionLevel.INTERNAL,
                required_algorithm=EncryptionAlgorithm.AES_256_GCM,
                require_encryption_at_rest=True,
                require_encryption_in_transit=True,
                key_rotation_days=180
            ),
            EncryptionPolicy(
                policy_id="pol_confidential",
                name="Confidential Data Policy",
                description="Policy for confidential data - strong encryption",
                data_classification=ProtectionLevel.CONFIDENTIAL,
                required_algorithm=EncryptionAlgorithm.AES_256_GCM,
                require_encryption_at_rest=True,
                require_encryption_in_transit=True,
                require_field_level_encryption=True,
                key_rotation_days=90,
                compliance_requirements=["GDPR", "HIPAA"]
            ),
            EncryptionPolicy(
                policy_id="pol_secret",
                name="Secret Data Policy",
                description="Policy for secret data - maximum encryption",
                data_classification=ProtectionLevel.SECRET,
                required_algorithm=EncryptionAlgorithm.CHACHA20_POLY1305,
                require_encryption_at_rest=True,
                require_encryption_in_transit=True,
                require_field_level_encryption=True,
                key_rotation_days=30,
                compliance_requirements=["GDPR", "HIPAA", "SOX"]
            ),
            EncryptionPolicy(
                policy_id="pol_top_secret",
                name="Top Secret Data Policy",
                description="Policy for top secret data - maximum security",
                data_classification=ProtectionLevel.TOP_SECRET,
                required_algorithm=EncryptionAlgorithm.AES_256_GCM,
                require_encryption_at_rest=True,
                require_encryption_in_transit=True,
                require_field_level_encryption=True,
                allowed_key_types=[KeyType.ASYMMETRIC_PUBLIC, KeyType.ASYMMETRIC_PRIVATE],
                key_rotation_days=7,
                compliance_requirements=["GDPR", "HIPAA", "SOX", "FIPS-140-2"]
            )
        ]
        
        for policy in policies:
            self.encryption_policies[policy.policy_id] = policy
    
    # Error handling would be managed by interface implementation
    async def generate_encryption_key(
        self,
        key_type: KeyType,
        algorithm: EncryptionAlgorithm,
        protection_level: ProtectionLevel = ProtectionLevel.CONFIDENTIAL,
        purpose: str = "general",
        expires_in_days: Optional[int] = None
    ) -> EncryptionKey:
        """
        Generate a new encryption key.
        
        Args:
            key_type: Type of key to generate
            algorithm: Encryption algorithm
            protection_level: Protection level for the key
            purpose: Purpose/usage description
            expires_in_days: Expiration period in days
            
        Returns:
            Generated encryption key
        """
        key_id = f"key_{secrets.token_hex(16)}"
        
        # Generate key data based on algorithm
        if algorithm == EncryptionAlgorithm.AES_256_GCM:
            key_data = secrets.token_bytes(32)  # 256 bits
        elif algorithm == EncryptionAlgorithm.AES_256_CBC:
            key_data = secrets.token_bytes(32)  # 256 bits
        elif algorithm == EncryptionAlgorithm.CHACHA20_POLY1305:
            key_data = secrets.token_bytes(32)  # 256 bits
        elif algorithm == EncryptionAlgorithm.FERNET:
            key_data = Fernet.generate_key()
        elif algorithm in [EncryptionAlgorithm.RSA_2048, EncryptionAlgorithm.RSA_4096]:
            key_size = 2048 if algorithm == EncryptionAlgorithm.RSA_2048 else 4096
            private_key = rsa.generate_private_key(
                public_exponent=65537,
                key_size=key_size
            )
            
            if key_type == KeyType.ASYMMETRIC_PRIVATE:
                key_data = private_key.private_bytes(
                    encoding=serialization.Encoding.PEM,
                    format=serialization.PrivateFormat.PKCS8,
                    encryption_algorithm=serialization.NoEncryption()
                )
            else:  # PUBLIC
                public_key = private_key.public_key()
                key_data = public_key.public_bytes(
                    encoding=serialization.Encoding.PEM,
                    format=serialization.PublicFormat.SubjectPublicKeyInfo
                )
        else:
            raise ValueError(f"Unsupported algorithm: {algorithm}")
        
        # Set expiration
        expires_at = None
        if expires_in_days:
            expires_at = datetime.utcnow() + timedelta(days=expires_in_days)
        
        # Create key object
        encryption_key = EncryptionKey(
            key_id=key_id,
            key_type=key_type,
            algorithm=algorithm,
            key_data=key_data,
            protection_level=protection_level,
            purpose=purpose,
            expires_at=expires_at
        )
        
        # Encrypt key data with master key if needed
        if protection_level in [ProtectionLevel.SECRET, ProtectionLevel.TOP_SECRET]:
            encryption_key.key_data = await self._encrypt_with_master_key(key_data)
            encryption_key.metadata["encrypted_with_master"] = True
        
        # Store key
        self.encryption_keys[key_id] = encryption_key
        
        logger.info(f"Generated encryption key: {key_id} ({algorithm.value})")
        return encryption_key
    
    async def _encrypt_with_master_key(self, data: bytes) -> bytes:
        """Encrypt data with the master key."""
        if not self.master_key:
            raise ValueError("Master key not available")
        
        # Use Fernet for master key encryption
        fernet = Fernet(base64.urlsafe_b64encode(self.master_key))
        return fernet.encrypt(data)
    
    async def _decrypt_with_master_key(self, encrypted_data: bytes) -> bytes:
        """Decrypt data with the master key."""
        if not self.master_key:
            raise ValueError("Master key not available")
        
        # Use Fernet for master key decryption
        fernet = Fernet(base64.urlsafe_b64encode(self.master_key))
        return fernet.decrypt(encrypted_data)
    
    # Error handling would be managed by interface implementation
    async def encrypt_data(
        self,
        data: Union[bytes, str],
        key_id: Optional[str] = None,
        protection_level: ProtectionLevel = ProtectionLevel.CONFIDENTIAL,
        metadata: Optional[Dict[str, Any]] = None
    ) -> EncryptedData:
        """
        Encrypt data using specified key or protection level.
        
        Args:
            data: Data to encrypt
            key_id: Specific key ID to use
            protection_level: Required protection level
            metadata: Additional metadata
            
        Returns:
            EncryptedData object
        """
        # Convert string to bytes if needed
        if isinstance(data, str):
            data_bytes = data.encode('utf-8')
        else:
            data_bytes = data
        
        # Get encryption key
        if key_id:
            if key_id not in self.encryption_keys:
                raise ValueError(f"Key {key_id} not found")
            encryption_key = self.encryption_keys[key_id]
        else:
            # Find appropriate key for protection level
            encryption_key = await self._get_key_for_protection_level(protection_level)
        
        # Get actual key data
        actual_key_data = encryption_key.key_data
        if encryption_key.metadata.get("encrypted_with_master"):
            actual_key_data = await self._decrypt_with_master_key(actual_key_data)
        
        # Encrypt based on algorithm
        encrypted_data, iv, auth_tag = await self._encrypt_with_algorithm(
            data_bytes, actual_key_data, encryption_key.algorithm
        )
        
        # Calculate checksum
        checksum = hashlib.sha256(data_bytes).hexdigest()
        
        # Create encrypted data object
        data_id = f"data_{secrets.token_hex(16)}"
        encrypted_data_obj = EncryptedData(
            data_id=data_id,
            encrypted_data=encrypted_data,
            encryption_algorithm=encryption_key.algorithm,
            key_id=encryption_key.key_id,
            initialization_vector=iv,
            authentication_tag=auth_tag,
            metadata=metadata or {},
            data_size_bytes=len(data_bytes),
            checksum=checksum
        )
        
        # Store in registry
        self.encrypted_data_registry[data_id] = encrypted_data_obj
        
        # Update key usage
        encryption_key.last_used = datetime.utcnow()
        encryption_key.usage_count += 1
        
        logger.debug(f"Encrypted data: {data_id} using key {encryption_key.key_id}")
        return encrypted_data_obj
    
    async def _get_key_for_protection_level(self, protection_level: ProtectionLevel) -> EncryptionKey:
        """Get or create an appropriate key for the protection level."""
        # Find existing active key
        for key in self.encryption_keys.values():
            if (key.protection_level == protection_level and 
                key.is_active and 
                key.key_type == KeyType.SYMMETRIC and
                (not key.expires_at or key.expires_at > datetime.utcnow())):
                return key
        
        # Create new key if none found
        policy = self._get_policy_for_protection_level(protection_level)
        algorithm = policy.required_algorithm
        
        return await self.generate_encryption_key(
            key_type=KeyType.SYMMETRIC,
            algorithm=algorithm,
            protection_level=protection_level,
            purpose=f"auto_generated_for_{protection_level.value}"
        )
    
    def _get_policy_for_protection_level(self, protection_level: ProtectionLevel) -> EncryptionPolicy:
        """Get encryption policy for protection level."""
        policy_mapping = {
            ProtectionLevel.PUBLIC: "pol_public",
            ProtectionLevel.INTERNAL: "pol_internal", 
            ProtectionLevel.CONFIDENTIAL: "pol_confidential",
            ProtectionLevel.SECRET: "pol_secret",
            ProtectionLevel.TOP_SECRET: "pol_top_secret"
        }
        
        policy_id = policy_mapping.get(protection_level, "pol_confidential")
        return self.encryption_policies[policy_id]
    
    async def _encrypt_with_algorithm(
        self,
        data: bytes,
        key: bytes,
        algorithm: EncryptionAlgorithm
    ) -> Tuple[bytes, Optional[bytes], Optional[bytes]]:
        """Encrypt data with specific algorithm."""
        
        if algorithm == EncryptionAlgorithm.AES_256_GCM:
            from cryptography.hazmat.primitives.ciphers.aead import AESGCM
            
            aesgcm = AESGCM(key)
            nonce = os.urandom(12)  # 96-bit nonce for GCM
            encrypted_data = aesgcm.encrypt(nonce, data, None)
            
            # Split encrypted data and auth tag
            ciphertext = encrypted_data[:-16]
            auth_tag = encrypted_data[-16:]
            
            return ciphertext, nonce, auth_tag
        
        elif algorithm == EncryptionAlgorithm.AES_256_CBC:
            from cryptography.hazmat.primitives.ciphers import Cipher, algorithms, modes
            from cryptography.hazmat.primitives import padding
            
            # Pad data to block size
            padder = padding.PKCS7(128).padder()
            padded_data = padder.update(data) + padder.finalize()
            
            iv = os.urandom(16)  # 128-bit IV for CBC
            cipher = Cipher(algorithms.AES(key), modes.CBC(iv))
            encryptor = cipher.encryptor()
            
            encrypted_data = encryptor.update(padded_data) + encryptor.finalize()
            return encrypted_data, iv, None
        
        elif algorithm == EncryptionAlgorithm.CHACHA20_POLY1305:
            from cryptography.hazmat.primitives.ciphers.aead import ChaCha20Poly1305
            
            chacha = ChaCha20Poly1305(key)
            nonce = os.urandom(12)  # 96-bit nonce
            encrypted_data = chacha.encrypt(nonce, data, None)
            
            # Split encrypted data and auth tag
            ciphertext = encrypted_data[:-16]
            auth_tag = encrypted_data[-16:]
            
            return ciphertext, nonce, auth_tag
        
        elif algorithm == EncryptionAlgorithm.FERNET:
            fernet = Fernet(key)
            encrypted_data = fernet.encrypt(data)
            return encrypted_data, None, None
        
        else:
            raise ValueError(f"Unsupported encryption algorithm: {algorithm}")
    
    # Error handling would be managed by interface implementation
    async def decrypt_data(
        self,
        encrypted_data_obj: EncryptedData,
        verify_checksum: bool = True
    ) -> bytes:
        """
        Decrypt encrypted data.
        
        Args:
            encrypted_data_obj: Encrypted data object
            verify_checksum: Whether to verify data integrity
            
        Returns:
            Decrypted data as bytes
        """
        # Get encryption key
        key_id = encrypted_data_obj.key_id
        if key_id not in self.encryption_keys:
            raise ValueError(f"Encryption key {key_id} not found")
        
        encryption_key = self.encryption_keys[key_id]
        
        # Check if key is active and not expired
        if not encryption_key.is_active:
            raise ValueError(f"Encryption key {key_id} is not active")
        
        if encryption_key.expires_at and encryption_key.expires_at <= datetime.utcnow():
            raise ValueError(f"Encryption key {key_id} has expired")
        
        # Get actual key data
        actual_key_data = encryption_key.key_data
        if encryption_key.metadata.get("encrypted_with_master"):
            actual_key_data = await self._decrypt_with_master_key(actual_key_data)
        
        # Decrypt based on algorithm
        decrypted_data = await self._decrypt_with_algorithm(
            encrypted_data_obj.encrypted_data,
            actual_key_data,
            encrypted_data_obj.encryption_algorithm,
            encrypted_data_obj.initialization_vector,
            encrypted_data_obj.authentication_tag
        )
        
        # Verify checksum if requested
        if verify_checksum and encrypted_data_obj.checksum:
            calculated_checksum = hashlib.sha256(decrypted_data).hexdigest()
            if calculated_checksum != encrypted_data_obj.checksum:
                raise ValueError("Data integrity check failed - checksum mismatch")
        
        # Update key usage
        encryption_key.last_used = datetime.utcnow()
        encryption_key.usage_count += 1
        
        logger.debug(f"Decrypted data: {encrypted_data_obj.data_id}")
        return decrypted_data
    
    async def _decrypt_with_algorithm(
        self,
        encrypted_data: bytes,
        key: bytes,
        algorithm: EncryptionAlgorithm,
        iv: Optional[bytes] = None,
        auth_tag: Optional[bytes] = None
    ) -> bytes:
        """Decrypt data with specific algorithm."""
        
        if algorithm == EncryptionAlgorithm.AES_256_GCM:
            from cryptography.hazmat.primitives.ciphers.aead import AESGCM
            
            aesgcm = AESGCM(key)
            # Reconstruct the full encrypted data with auth tag
            full_encrypted_data = encrypted_data + auth_tag
            decrypted_data = aesgcm.decrypt(iv, full_encrypted_data, None)
            return decrypted_data
        
        elif algorithm == EncryptionAlgorithm.AES_256_CBC:
            from cryptography.hazmat.primitives.ciphers import Cipher, algorithms, modes
            from cryptography.hazmat.primitives import padding
            
            cipher = Cipher(algorithms.AES(key), modes.CBC(iv))
            decryptor = cipher.decryptor()
            
            padded_data = decryptor.update(encrypted_data) + decryptor.finalize()
            
            # Remove padding
            unpadder = padding.PKCS7(128).unpadder()
            decrypted_data = unpadder.update(padded_data) + unpadder.finalize()
            
            return decrypted_data
        
        elif algorithm == EncryptionAlgorithm.CHACHA20_POLY1305:
            from cryptography.hazmat.primitives.ciphers.aead import ChaCha20Poly1305
            
            chacha = ChaCha20Poly1305(key)
            # Reconstruct the full encrypted data with auth tag
            full_encrypted_data = encrypted_data + auth_tag
            decrypted_data = chacha.decrypt(iv, full_encrypted_data, None)
            return decrypted_data
        
        elif algorithm == EncryptionAlgorithm.FERNET:
            fernet = Fernet(key)
            decrypted_data = fernet.decrypt(encrypted_data)
            return decrypted_data
        
        else:
            raise ValueError(f"Unsupported decryption algorithm: {algorithm}")
    
    # Error handling would be managed by interface implementation
    async def rotate_key(
        self,
        key_id: str,
        reason: str = "scheduled_rotation"
    ) -> Tuple[EncryptionKey, KeyRotationEvent]:
        """
        Rotate an encryption key.
        
        Args:
            key_id: ID of key to rotate
            reason: Reason for rotation
            
        Returns:
            Tuple of (new_key, rotation_event)
        """
        start_time = datetime.utcnow()
        
        # Get old key
        if key_id not in self.encryption_keys:
            raise ValueError(f"Key {key_id} not found")
        
        old_key = self.encryption_keys[key_id]
        
        # Generate new key with same properties
        new_key = await self.generate_encryption_key(
            key_type=old_key.key_type,
            algorithm=old_key.algorithm,
            protection_level=old_key.protection_level,
            purpose=old_key.purpose
        )
        
        # Create rotation event
        event_id = f"rot_{secrets.token_hex(8)}"
        rotation_event = KeyRotationEvent(
            event_id=event_id,
            old_key_id=key_id,
            new_key_id=new_key.key_id,
            rotation_reason=reason
        )
        
        # Re-encrypt all data using the old key
        affected_data_count = 0
        
        for data_id, encrypted_data_obj in self.encrypted_data_registry.items():
            if encrypted_data_obj.key_id == key_id:
                try:
                    # Decrypt with old key
                    decrypted_data = await self.decrypt_data(encrypted_data_obj, verify_checksum=False)
                    
                    # Re-encrypt with new key
                    new_encrypted_data = await self.encrypt_data(
                        decrypted_data,
                        key_id=new_key.key_id,
                        metadata=encrypted_data_obj.metadata
                    )
                    
                    # Update registry
                    self.encrypted_data_registry[data_id] = new_encrypted_data
                    affected_data_count += 1
                    
                except Exception as e:
                    logger.error(f"Failed to re-encrypt data {data_id} during key rotation: {e}")
        
        # Mark old key as inactive
        old_key.is_active = False
        old_key.metadata["rotated_at"] = datetime.utcnow().isoformat()
        old_key.metadata["rotated_to"] = new_key.key_id
        
        # Complete rotation event
        rotation_event.affected_data_count = affected_data_count
        rotation_event.rotation_time_seconds = (datetime.utcnow() - start_time).total_seconds()
        rotation_event.completed_at = datetime.utcnow()
        rotation_event.status = "completed"
        
        # Store rotation event
        self.key_rotation_history.append(rotation_event)
        
        logger.info(f"Key rotation completed: {key_id} -> {new_key.key_id}, {affected_data_count} data items re-encrypted")
        return new_key, rotation_event
    
    async def _key_rotation_monitor(self):
        """Background task to monitor and trigger key rotations."""
        while True:
            try:
                await asyncio.sleep(self.key_rotation_check_interval_hours * 3600)
                
                if not self.auto_key_rotation_enabled:
                    continue
                
                # Check for keys that need rotation
                current_time = datetime.utcnow()
                
                for key in self.encryption_keys.values():
                    if not key.is_active:
                        continue
                    
                    # Get policy for this key's protection level
                    policy = self._get_policy_for_protection_level(key.protection_level)
                    
                    # Check if key is due for rotation
                    key_age_days = (current_time - key.created_at).days
                    
                    if key_age_days >= policy.key_rotation_days:
                        logger.info(f"Triggering automatic rotation for key {key.key_id} (age: {key_age_days} days)")
                        await self.rotate_key(key.key_id, "automatic_scheduled_rotation")
                
            except Exception as e:
                logger.error(f"Key rotation monitor error: {e}")
    
    async def _encryption_health_monitor(self):
        """Background task to monitor encryption health."""
        while True:
            try:
                await asyncio.sleep(3600)  # Check every hour
                
                # Check for expired keys
                current_time = datetime.utcnow()
                expired_keys = [
                    key for key in self.encryption_keys.values()
                    if key.expires_at and key.expires_at <= current_time and key.is_active
                ]
                
                for key in expired_keys:
                    logger.warning(f"Key {key.key_id} has expired")
                    key.is_active = False
                
                # Check for unused keys (older than 90 days with no usage)
                unused_cutoff = current_time - timedelta(days=90)
                unused_keys = [
                    key for key in self.encryption_keys.values()
                    if (not key.last_used or key.last_used < unused_cutoff) and
                    key.created_at < unused_cutoff and key.is_active
                ]
                
                for key in unused_keys:
                    logger.info(f"Marking unused key {key.key_id} as inactive")
                    key.is_active = False
                
            except Exception as e:
                logger.error(f"Encryption health monitor error: {e}")
    
    # Error handling would be managed by interface implementation
    async def get_encryption_dashboard(self) -> Dict[str, Any]:
        """Get comprehensive encryption and protection dashboard."""
        # Calculate statistics
        total_keys = len(self.encryption_keys)
        active_keys = len([k for k in self.encryption_keys.values() if k.is_active])
        expired_keys = len([
            k for k in self.encryption_keys.values() 
            if k.expires_at and k.expires_at <= datetime.utcnow()
        ])
        
        # Key usage statistics
        keys_by_algorithm = {}
        keys_by_protection_level = {}
        
        for key in self.encryption_keys.values():
            algo = key.algorithm.value
            level = key.protection_level.value
            
            keys_by_algorithm[algo] = keys_by_algorithm.get(algo, 0) + 1
            keys_by_protection_level[level] = keys_by_protection_level.get(level, 0) + 1
        
        # Encrypted data statistics
        total_encrypted_data = len(self.encrypted_data_registry)
        total_encrypted_bytes = sum(data.data_size_bytes for data in self.encrypted_data_registry.values())
        
        # Recent key rotations
        recent_rotations = [
            {
                "event_id": event.event_id,
                "old_key_id": event.old_key_id,
                "new_key_id": event.new_key_id,
                "reason": event.rotation_reason,
                "affected_data_count": event.affected_data_count,
                "completed_at": event.completed_at.isoformat() if event.completed_at else None
            }
            for event in self.key_rotation_history[-10:]  # Last 10 rotations
        ]
        
        return {
            "timestamp": datetime.utcnow().isoformat(),
            "key_management": {
                "total_keys": total_keys,
                "active_keys": active_keys,
                "expired_keys": expired_keys,
                "keys_by_algorithm": keys_by_algorithm,
                "keys_by_protection_level": keys_by_protection_level
            },
            "encrypted_data": {
                "total_encrypted_items": total_encrypted_data,
                "total_encrypted_bytes": total_encrypted_bytes,
                "encryption_policies": len(self.encryption_policies)
            },
            "key_rotation": {
                "auto_rotation_enabled": self.auto_key_rotation_enabled,
                "total_rotations": len(self.key_rotation_history),
                "recent_rotations": recent_rotations
            },
            "security_metrics": {
                "master_key_available": self.master_key is not None,
                "supported_algorithms": [algo.value for algo in EncryptionAlgorithm],
                "protection_levels": [level.value for level in ProtectionLevel]
            },
            "compliance": {
                "policies_configured": len(self.encryption_policies),
                "fips_compliance": False,  # Would check actual FIPS compliance
                "key_escrow_enabled": False  # Would check if key escrow is configured
            }
        }