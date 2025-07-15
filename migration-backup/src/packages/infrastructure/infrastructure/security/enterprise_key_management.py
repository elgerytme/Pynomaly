"""
Enterprise Key Management System for Pynomaly.

This module implements a comprehensive key management system with support for
Hardware Security Modules (HSM), quantum-resistant cryptography, and envelope encryption.
"""

from __future__ import annotations

import asyncio
import base64
import hashlib
import logging
import secrets
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from typing import Any
from uuid import UUID, uuid4

from cryptography.fernet import Fernet
from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.primitives.asymmetric import rsa
from cryptography.hazmat.primitives.ciphers import Cipher, algorithms, modes

logger = logging.getLogger(__name__)


class KeyType(Enum):
    """Types of cryptographic keys."""

    SYMMETRIC = "symmetric"
    ASYMMETRIC = "asymmetric"
    SIGNING = "signing"
    ENCRYPTION = "encryption"
    MASTER = "master"
    DATA = "data"


class KeyStatus(Enum):
    """Key lifecycle status."""

    ACTIVE = "active"
    INACTIVE = "inactive"
    EXPIRED = "expired"
    REVOKED = "revoked"
    COMPROMISED = "compromised"


class CryptoAlgorithm(Enum):
    """Supported cryptographic algorithms."""

    AES_256_GCM = "aes_256_gcm"
    AES_256_CBC = "aes_256_cbc"
    RSA_4096 = "rsa_4096"
    RSA_2048 = "rsa_2048"
    ECDSA_P256 = "ecdsa_p256"
    ECDSA_P384 = "ecdsa_p384"
    CHACHA20_POLY1305 = "chacha20_poly1305"
    KYBER_1024 = "kyber_1024"  # Post-quantum
    DILITHIUM_5 = "dilithium_5"  # Post-quantum


@dataclass
class KeyMetadata:
    """Key metadata for management."""

    key_id: UUID = field(default_factory=uuid4)
    name: str = ""
    description: str = ""
    key_type: KeyType = KeyType.SYMMETRIC
    algorithm: CryptoAlgorithm = CryptoAlgorithm.AES_256_GCM
    status: KeyStatus = KeyStatus.ACTIVE
    created_at: datetime = field(default_factory=datetime.now)
    created_by: str = ""
    expires_at: datetime | None = None
    last_rotation: datetime | None = None
    rotation_interval_days: int = 90
    usage_count: int = 0
    max_usage_count: int | None = None
    tags: list[str] = field(default_factory=list)
    context: dict[str, Any] = field(default_factory=dict)
    hsm_backed: bool = False
    exportable: bool = False


@dataclass
class EncryptionContext:
    """Context for encryption operations."""

    key_id: UUID
    algorithm: CryptoAlgorithm
    additional_data: bytes | None = None
    context_data: dict[str, str] = field(default_factory=dict)
    timestamp: datetime = field(default_factory=datetime.now)


@dataclass
class EncryptionResult:
    """Result of encryption operation."""

    ciphertext: bytes
    context: EncryptionContext
    nonce: bytes | None = None
    tag: bytes | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


class HardwareSecurityModule:
    """Hardware Security Module interface."""

    def __init__(self, hsm_config: dict[str, Any]):
        self.hsm_config = hsm_config
        self.connected = False
        self.slots = {}

    async def connect(self) -> bool:
        """Connect to HSM."""
        try:
            # Simulated HSM connection
            # In production, use actual HSM SDK (CloudHSM, Luna, etc.)
            logger.info("Connecting to HSM...")

            # Simulate connection process
            await asyncio.sleep(0.1)

            self.connected = True
            logger.info("HSM connection established")
            return True

        except Exception as e:
            logger.error(f"HSM connection failed: {e}")
            return False

    async def generate_key(self, key_metadata: KeyMetadata) -> bytes:
        """Generate key in HSM."""
        if not self.connected:
            raise RuntimeError("HSM not connected")

        try:
            # Simulate HSM key generation
            if key_metadata.key_type == KeyType.SYMMETRIC:
                if key_metadata.algorithm == CryptoAlgorithm.AES_256_GCM:
                    key_material = secrets.token_bytes(32)  # 256 bits
                elif key_metadata.algorithm == CryptoAlgorithm.CHACHA20_POLY1305:
                    key_material = secrets.token_bytes(32)  # 256 bits
                else:
                    raise ValueError(
                        f"Unsupported symmetric algorithm: {key_metadata.algorithm}"
                    )

            elif key_metadata.key_type == KeyType.ASYMMETRIC:
                if key_metadata.algorithm == CryptoAlgorithm.RSA_4096:
                    # Generate RSA key pair in HSM
                    private_key = rsa.generate_private_key(
                        public_exponent=65537, key_size=4096
                    )
                    key_material = private_key.private_bytes(
                        encoding=serialization.Encoding.PEM,
                        format=serialization.PrivateFormat.PKCS8,
                        encryption_algorithm=serialization.NoEncryption(),
                    )
                else:
                    raise ValueError(
                        f"Unsupported asymmetric algorithm: {key_metadata.algorithm}"
                    )

            else:
                raise ValueError(f"Unsupported key type: {key_metadata.key_type}")

            # Store key reference in HSM
            slot_id = str(uuid4())
            self.slots[str(key_metadata.key_id)] = {
                "slot_id": slot_id,
                "key_material": key_material,
                "metadata": key_metadata,
            }

            logger.info(f"Generated key in HSM: {key_metadata.key_id}")
            return key_material

        except Exception as e:
            logger.error(f"HSM key generation failed: {e}")
            raise

    async def encrypt_data(
        self, key_id: UUID, plaintext: bytes, context: EncryptionContext
    ) -> EncryptionResult:
        """Encrypt data using HSM key."""
        if not self.connected:
            raise RuntimeError("HSM not connected")

        try:
            slot_info = self.slots.get(str(key_id))
            if not slot_info:
                raise ValueError(f"Key not found in HSM: {key_id}")

            key_material = slot_info["key_material"]
            metadata = slot_info["metadata"]

            if metadata.algorithm == CryptoAlgorithm.AES_256_GCM:
                # AES-GCM encryption
                nonce = secrets.token_bytes(12)  # 96-bit nonce for GCM
                cipher = Cipher(algorithms.AES(key_material), modes.GCM(nonce))
                encryptor = cipher.encryptor()

                if context.additional_data:
                    encryptor.authenticate_additional_data(context.additional_data)

                ciphertext = encryptor.update(plaintext) + encryptor.finalize()

                return EncryptionResult(
                    ciphertext=ciphertext,
                    context=context,
                    nonce=nonce,
                    tag=encryptor.tag,
                )

            else:
                raise ValueError(
                    f"Unsupported encryption algorithm: {metadata.algorithm}"
                )

        except Exception as e:
            logger.error(f"HSM encryption failed: {e}")
            raise

    async def decrypt_data(
        self, key_id: UUID, encryption_result: EncryptionResult
    ) -> bytes:
        """Decrypt data using HSM key."""
        if not self.connected:
            raise RuntimeError("HSM not connected")

        try:
            slot_info = self.slots.get(str(key_id))
            if not slot_info:
                raise ValueError(f"Key not found in HSM: {key_id}")

            key_material = slot_info["key_material"]
            metadata = slot_info["metadata"]

            if metadata.algorithm == CryptoAlgorithm.AES_256_GCM:
                # AES-GCM decryption
                cipher = Cipher(
                    algorithms.AES(key_material),
                    modes.GCM(encryption_result.nonce, encryption_result.tag),
                )
                decryptor = cipher.decryptor()

                if encryption_result.context.additional_data:
                    decryptor.authenticate_additional_data(
                        encryption_result.context.additional_data
                    )

                plaintext = (
                    decryptor.update(encryption_result.ciphertext)
                    + decryptor.finalize()
                )
                return plaintext

            else:
                raise ValueError(
                    f"Unsupported decryption algorithm: {metadata.algorithm}"
                )

        except Exception as e:
            logger.error(f"HSM decryption failed: {e}")
            raise


class QuantumResistantCrypto:
    """Quantum-resistant cryptographic operations."""

    def __init__(self):
        self.supported_algorithms = [
            CryptoAlgorithm.KYBER_1024,
            CryptoAlgorithm.DILITHIUM_5,
        ]

    async def generate_keypair(self, algorithm: CryptoAlgorithm) -> tuple[bytes, bytes]:
        """Generate quantum-resistant key pair."""
        try:
            if algorithm == CryptoAlgorithm.KYBER_1024:
                # Simulate Kyber-1024 key generation
                # In production, use actual post-quantum crypto library
                private_key = secrets.token_bytes(3168)  # Kyber-1024 private key size
                public_key = secrets.token_bytes(1568)  # Kyber-1024 public key size

            elif algorithm == CryptoAlgorithm.DILITHIUM_5:
                # Simulate Dilithium-5 key generation
                private_key = secrets.token_bytes(4864)  # Dilithium-5 private key size
                public_key = secrets.token_bytes(2592)  # Dilithium-5 public key size

            else:
                raise ValueError(
                    f"Unsupported quantum-resistant algorithm: {algorithm}"
                )

            logger.info(f"Generated quantum-resistant keypair: {algorithm.value}")
            return private_key, public_key

        except Exception as e:
            logger.error(f"Quantum-resistant key generation failed: {e}")
            raise

    async def encrypt(
        self, public_key: bytes, plaintext: bytes, algorithm: CryptoAlgorithm
    ) -> bytes:
        """Encrypt data using quantum-resistant algorithm."""
        try:
            if algorithm == CryptoAlgorithm.KYBER_1024:
                # Simulate Kyber-1024 encryption
                # In production, use actual Kyber implementation
                ciphertext = hashlib.sha256(public_key + plaintext).digest() + plaintext

            else:
                raise ValueError(f"Unsupported encryption algorithm: {algorithm}")

            return ciphertext

        except Exception as e:
            logger.error(f"Quantum-resistant encryption failed: {e}")
            raise

    async def decrypt(
        self, private_key: bytes, ciphertext: bytes, algorithm: CryptoAlgorithm
    ) -> bytes:
        """Decrypt data using quantum-resistant algorithm."""
        try:
            if algorithm == CryptoAlgorithm.KYBER_1024:
                # Simulate Kyber-1024 decryption
                # In production, use actual Kyber implementation
                plaintext = ciphertext[32:]  # Remove hash prefix

            else:
                raise ValueError(f"Unsupported decryption algorithm: {algorithm}")

            return plaintext

        except Exception as e:
            logger.error(f"Quantum-resistant decryption failed: {e}")
            raise


class EnvelopeEncryption:
    """Envelope encryption implementation."""

    def __init__(self, master_key_id: UUID):
        self.master_key_id = master_key_id

    async def encrypt(
        self, plaintext: bytes, context: dict[str, str]
    ) -> dict[str, Any]:
        """Encrypt data using envelope encryption."""
        try:
            # Generate data encryption key (DEK)
            dek = Fernet.generate_key()

            # Encrypt plaintext with DEK
            fernet = Fernet(dek)
            encrypted_data = fernet.encrypt(plaintext)

            # Encrypt DEK with master key (simplified - would use actual KMS)
            encrypted_dek = self._encrypt_dek(dek)

            envelope = {
                "encrypted_data": base64.b64encode(encrypted_data).decode(),
                "encrypted_dek": base64.b64encode(encrypted_dek).decode(),
                "master_key_id": str(self.master_key_id),
                "context": context,
                "timestamp": datetime.now().isoformat(),
                "algorithm": "envelope_aes_256",
            }

            logger.info("Envelope encryption completed")
            return envelope

        except Exception as e:
            logger.error(f"Envelope encryption failed: {e}")
            raise

    async def decrypt(self, envelope: dict[str, Any]) -> bytes:
        """Decrypt data using envelope encryption."""
        try:
            # Decrypt DEK with master key
            encrypted_dek = base64.b64decode(envelope["encrypted_dek"])
            dek = self._decrypt_dek(encrypted_dek)

            # Decrypt data with DEK
            encrypted_data = base64.b64decode(envelope["encrypted_data"])
            fernet = Fernet(dek)
            plaintext = fernet.decrypt(encrypted_data)

            logger.info("Envelope decryption completed")
            return plaintext

        except Exception as e:
            logger.error(f"Envelope decryption failed: {e}")
            raise

    def _encrypt_dek(self, dek: bytes) -> bytes:
        """Encrypt data encryption key with master key."""
        # Simplified implementation - in production, use actual KMS/HSM
        master_key = self._get_master_key()
        fernet = Fernet(master_key)
        return fernet.encrypt(dek)

    def _decrypt_dek(self, encrypted_dek: bytes) -> bytes:
        """Decrypt data encryption key with master key."""
        # Simplified implementation - in production, use actual KMS/HSM
        master_key = self._get_master_key()
        fernet = Fernet(master_key)
        return fernet.decrypt(encrypted_dek)

    def _get_master_key(self) -> bytes:
        """Get master encryption key."""
        # Simplified implementation - in production, fetch from secure storage
        return base64.urlsafe_b64encode(
            hashlib.sha256(str(self.master_key_id).encode()).digest()
        )


class AutomatedKeyRotation:
    """Automated key rotation management."""

    def __init__(self, key_manager):
        self.key_manager = key_manager
        self.rotation_jobs = {}
        self.rotation_history = []

    async def schedule_rotation(self, key_id: UUID, interval_days: int = 90):
        """Schedule automatic key rotation."""
        try:
            key_metadata = await self.key_manager.get_key_metadata(key_id)
            if not key_metadata:
                raise ValueError(f"Key not found: {key_id}")

            next_rotation = datetime.now() + timedelta(days=interval_days)

            job_id = str(uuid4())
            self.rotation_jobs[job_id] = {
                "key_id": key_id,
                "interval_days": interval_days,
                "next_rotation": next_rotation,
                "created_at": datetime.now(),
                "status": "scheduled",
            }

            logger.info(f"Scheduled key rotation for {key_id} in {interval_days} days")
            return job_id

        except Exception as e:
            logger.error(f"Failed to schedule key rotation: {e}")
            raise

    async def rotate_key(self, key_id: UUID, preserve_old: bool = True) -> UUID:
        """Rotate a cryptographic key."""
        try:
            # Get current key metadata
            current_metadata = await self.key_manager.get_key_metadata(key_id)
            if not current_metadata:
                raise ValueError(f"Key not found: {key_id}")

            # Generate new key with same properties
            new_metadata = KeyMetadata(
                name=f"{current_metadata.name}_rotated",
                description=f"Rotated version of {current_metadata.name}",
                key_type=current_metadata.key_type,
                algorithm=current_metadata.algorithm,
                created_by="key_rotation_service",
                expires_at=current_metadata.expires_at,
                rotation_interval_days=current_metadata.rotation_interval_days,
                tags=current_metadata.tags + ["rotated"],
                context=current_metadata.context,
                hsm_backed=current_metadata.hsm_backed,
                exportable=current_metadata.exportable,
            )

            # Generate new key
            new_key_id = await self.key_manager.generate_key(new_metadata)

            # Update old key status
            if preserve_old:
                await self.key_manager.update_key_status(key_id, KeyStatus.INACTIVE)
            else:
                await self.key_manager.revoke_key(key_id)

            # Record rotation
            rotation_record = {
                "rotation_id": str(uuid4()),
                "old_key_id": str(key_id),
                "new_key_id": str(new_key_id),
                "rotated_at": datetime.now(),
                "reason": "scheduled_rotation",
                "preserve_old": preserve_old,
            }
            self.rotation_history.append(rotation_record)

            logger.info(f"Key rotated: {key_id} -> {new_key_id}")
            return new_key_id

        except Exception as e:
            logger.error(f"Key rotation failed: {e}")
            raise

    async def check_rotation_schedule(self):
        """Check for keys that need rotation."""
        try:
            current_time = datetime.now()
            rotations_needed = []

            for job_id, job in self.rotation_jobs.items():
                if (
                    job["status"] == "scheduled"
                    and current_time >= job["next_rotation"]
                ):
                    rotations_needed.append(job)

            for job in rotations_needed:
                try:
                    new_key_id = await self.rotate_key(job["key_id"])

                    # Reschedule next rotation
                    next_rotation = current_time + timedelta(days=job["interval_days"])
                    job["next_rotation"] = next_rotation
                    job["last_rotation"] = current_time

                    logger.info(
                        f"Automatic key rotation completed: {job['key_id']} -> {new_key_id}"
                    )

                except Exception as e:
                    logger.error(
                        f"Automatic key rotation failed for {job['key_id']}: {e}"
                    )
                    job["status"] = "failed"
                    job["error"] = str(e)

        except Exception as e:
            logger.error(f"Rotation schedule check failed: {e}")


class EnterpriseKMS:
    """Enterprise Key Management System."""

    def __init__(self, hsm_config: dict[str, Any] | None = None):
        self.key_store = {}
        self.key_metadata_store = {}
        self.audit_log = []

        # Initialize components
        self.hsm = HardwareSecurityModule(hsm_config or {}) if hsm_config else None
        self.quantum_crypto = QuantumResistantCrypto()
        self.key_rotation = AutomatedKeyRotation(self)

        # Master key for envelope encryption
        self.master_key_id = uuid4()
        self.envelope_encryption = EnvelopeEncryption(self.master_key_id)

    async def initialize(self):
        """Initialize the KMS."""
        try:
            if self.hsm:
                await self.hsm.connect()

            # Generate master key
            await self._generate_master_key()

            logger.info("Enterprise KMS initialized successfully")

        except Exception as e:
            logger.error(f"KMS initialization failed: {e}")
            raise

    async def generate_key(self, metadata: KeyMetadata) -> UUID:
        """Generate a new cryptographic key."""
        try:
            key_id = metadata.key_id

            # Generate key material
            if metadata.hsm_backed and self.hsm:
                key_material = await self.hsm.generate_key(metadata)
            else:
                key_material = await self._generate_key_material(metadata)

            # Store key and metadata
            self.key_store[str(key_id)] = key_material
            self.key_metadata_store[str(key_id)] = metadata

            # Schedule automatic rotation
            if metadata.rotation_interval_days > 0:
                await self.key_rotation.schedule_rotation(
                    key_id, metadata.rotation_interval_days
                )

            # Audit log
            await self._log_audit_event(
                "key_generated",
                {
                    "key_id": str(key_id),
                    "key_type": metadata.key_type.value,
                    "algorithm": metadata.algorithm.value,
                    "hsm_backed": metadata.hsm_backed,
                    "created_by": metadata.created_by,
                },
            )

            logger.info(f"Generated key: {key_id}")
            return key_id

        except Exception as e:
            logger.error(f"Key generation failed: {e}")
            raise

    async def encrypt_data(
        self, key_id: UUID, plaintext: bytes, context: dict[str, str] = None
    ) -> dict[str, Any]:
        """Encrypt data with specified key."""
        try:
            metadata = self.key_metadata_store.get(str(key_id))
            if not metadata:
                raise ValueError(f"Key not found: {key_id}")

            if metadata.status != KeyStatus.ACTIVE:
                raise ValueError(f"Key is not active: {key_id}")

            # Update usage count
            metadata.usage_count += 1
            if (
                metadata.max_usage_count
                and metadata.usage_count > metadata.max_usage_count
            ):
                raise ValueError(f"Key usage limit exceeded: {key_id}")

            encryption_context = EncryptionContext(
                key_id=key_id, algorithm=metadata.algorithm, context_data=context or {}
            )

            # Encrypt based on backing store
            if metadata.hsm_backed and self.hsm:
                result = await self.hsm.encrypt_data(
                    key_id, plaintext, encryption_context
                )
                encrypted_data = {
                    "ciphertext": base64.b64encode(result.ciphertext).decode(),
                    "nonce": base64.b64encode(result.nonce).decode()
                    if result.nonce
                    else None,
                    "tag": base64.b64encode(result.tag).decode()
                    if result.tag
                    else None,
                    "key_id": str(key_id),
                    "algorithm": metadata.algorithm.value,
                    "context": context or {},
                    "hsm_backed": True,
                    "timestamp": datetime.now().isoformat(),
                }
            else:
                # Use envelope encryption for software keys
                encrypted_data = await self.envelope_encryption.encrypt(
                    plaintext, context or {}
                )
                encrypted_data["key_id"] = str(key_id)
                encrypted_data["hsm_backed"] = False

            # Audit log
            await self._log_audit_event(
                "data_encrypted",
                {
                    "key_id": str(key_id),
                    "data_size": len(plaintext),
                    "algorithm": metadata.algorithm.value,
                    "context": context or {},
                },
            )

            return encrypted_data

        except Exception as e:
            logger.error(f"Data encryption failed: {e}")
            raise

    async def decrypt_data(self, encrypted_data: dict[str, Any]) -> bytes:
        """Decrypt data using stored key."""
        try:
            key_id = UUID(encrypted_data["key_id"])
            metadata = self.key_metadata_store.get(str(key_id))

            if not metadata:
                raise ValueError(f"Key not found: {key_id}")

            if metadata.status not in [KeyStatus.ACTIVE, KeyStatus.INACTIVE]:
                raise ValueError(f"Key cannot be used for decryption: {key_id}")

            # Decrypt based on backing store
            if encrypted_data.get("hsm_backed", False) and self.hsm:
                # Reconstruct encryption result
                encryption_result = EncryptionResult(
                    ciphertext=base64.b64decode(encrypted_data["ciphertext"]),
                    context=EncryptionContext(
                        key_id=key_id,
                        algorithm=CryptoAlgorithm(encrypted_data["algorithm"]),
                        context_data=encrypted_data.get("context", {}),
                    ),
                    nonce=base64.b64decode(encrypted_data["nonce"])
                    if encrypted_data.get("nonce")
                    else None,
                    tag=base64.b64decode(encrypted_data["tag"])
                    if encrypted_data.get("tag")
                    else None,
                )

                plaintext = await self.hsm.decrypt_data(key_id, encryption_result)
            else:
                # Use envelope decryption
                plaintext = await self.envelope_encryption.decrypt(encrypted_data)

            # Audit log
            await self._log_audit_event(
                "data_decrypted",
                {
                    "key_id": str(key_id),
                    "data_size": len(plaintext),
                    "algorithm": encrypted_data.get("algorithm", "unknown"),
                },
            )

            return plaintext

        except Exception as e:
            logger.error(f"Data decryption failed: {e}")
            raise

    async def get_key_metadata(self, key_id: UUID) -> KeyMetadata | None:
        """Get key metadata."""
        return self.key_metadata_store.get(str(key_id))

    async def update_key_status(self, key_id: UUID, status: KeyStatus):
        """Update key status."""
        try:
            metadata = self.key_metadata_store.get(str(key_id))
            if not metadata:
                raise ValueError(f"Key not found: {key_id}")

            old_status = metadata.status
            metadata.status = status

            # Audit log
            await self._log_audit_event(
                "key_status_updated",
                {
                    "key_id": str(key_id),
                    "old_status": old_status.value,
                    "new_status": status.value,
                },
            )

            logger.info(f"Updated key status: {key_id} -> {status.value}")

        except Exception as e:
            logger.error(f"Key status update failed: {e}")
            raise

    async def revoke_key(self, key_id: UUID, reason: str = ""):
        """Revoke a key."""
        try:
            await self.update_key_status(key_id, KeyStatus.REVOKED)

            # Clear key material for revoked keys
            if str(key_id) in self.key_store:
                del self.key_store[str(key_id)]

            # Audit log
            await self._log_audit_event(
                "key_revoked", {"key_id": str(key_id), "reason": reason}
            )

            logger.info(f"Revoked key: {key_id}")

        except Exception as e:
            logger.error(f"Key revocation failed: {e}")
            raise

    async def list_keys(
        self, status: KeyStatus | None = None, key_type: KeyType | None = None
    ) -> list[KeyMetadata]:
        """List keys with optional filtering."""
        try:
            keys = []

            for metadata in self.key_metadata_store.values():
                if status and metadata.status != status:
                    continue
                if key_type and metadata.key_type != key_type:
                    continue

                keys.append(metadata)

            return keys

        except Exception as e:
            logger.error(f"Key listing failed: {e}")
            raise

    async def get_audit_log(
        self, start_time: datetime | None = None, end_time: datetime | None = None
    ) -> list[dict[str, Any]]:
        """Get audit log entries."""
        try:
            filtered_log = []

            for entry in self.audit_log:
                entry_time = datetime.fromisoformat(entry["timestamp"])

                if start_time and entry_time < start_time:
                    continue
                if end_time and entry_time > end_time:
                    continue

                filtered_log.append(entry)

            return filtered_log

        except Exception as e:
            logger.error(f"Audit log retrieval failed: {e}")
            raise

    async def _generate_master_key(self):
        """Generate master key for envelope encryption."""
        master_metadata = KeyMetadata(
            key_id=self.master_key_id,
            name="master_key",
            description="Master key for envelope encryption",
            key_type=KeyType.MASTER,
            algorithm=CryptoAlgorithm.AES_256_GCM,
            created_by="system",
            hsm_backed=True if self.hsm else False,
            exportable=False,
        )

        if self.hsm:
            key_material = await self.hsm.generate_key(master_metadata)
        else:
            key_material = secrets.token_bytes(32)  # 256-bit key

        self.key_store[str(self.master_key_id)] = key_material
        self.key_metadata_store[str(self.master_key_id)] = master_metadata

        logger.info("Master key generated")

    async def _generate_key_material(self, metadata: KeyMetadata) -> bytes:
        """Generate key material based on algorithm."""
        try:
            if metadata.algorithm == CryptoAlgorithm.AES_256_GCM:
                return secrets.token_bytes(32)  # 256 bits
            elif metadata.algorithm == CryptoAlgorithm.AES_256_CBC:
                return secrets.token_bytes(32)  # 256 bits
            elif metadata.algorithm == CryptoAlgorithm.CHACHA20_POLY1305:
                return secrets.token_bytes(32)  # 256 bits
            elif metadata.algorithm in [
                CryptoAlgorithm.RSA_2048,
                CryptoAlgorithm.RSA_4096,
            ]:
                key_size = (
                    2048 if metadata.algorithm == CryptoAlgorithm.RSA_2048 else 4096
                )
                private_key = rsa.generate_private_key(
                    public_exponent=65537, key_size=key_size
                )
                return private_key.private_bytes(
                    encoding=serialization.Encoding.PEM,
                    format=serialization.PrivateFormat.PKCS8,
                    encryption_algorithm=serialization.NoEncryption(),
                )
            elif metadata.algorithm in [
                CryptoAlgorithm.KYBER_1024,
                CryptoAlgorithm.DILITHIUM_5,
            ]:
                private_key, public_key = await self.quantum_crypto.generate_keypair(
                    metadata.algorithm
                )
                return private_key  # Store private key, public key derivable
            else:
                raise ValueError(f"Unsupported algorithm: {metadata.algorithm}")

        except Exception as e:
            logger.error(f"Key material generation failed: {e}")
            raise

    async def _log_audit_event(self, event_type: str, details: dict[str, Any]):
        """Log audit event."""
        try:
            audit_entry = {
                "event_id": str(uuid4()),
                "event_type": event_type,
                "timestamp": datetime.now().isoformat(),
                "details": details,
                "source": "enterprise_kms",
            }

            self.audit_log.append(audit_entry)

            # In production, also send to external audit system
            logger.debug(f"Audit event logged: {event_type}")

        except Exception as e:
            logger.error(f"Audit logging failed: {e}")


# Example usage and testing
if __name__ == "__main__":

    async def test_enterprise_kms():
        """Test Enterprise KMS functionality."""

        # Initialize KMS
        kms = EnterpriseKMS()
        await kms.initialize()

        # Generate symmetric key
        symmetric_metadata = KeyMetadata(
            name="test_symmetric_key",
            description="Test AES key for data encryption",
            key_type=KeyType.SYMMETRIC,
            algorithm=CryptoAlgorithm.AES_256_GCM,
            created_by="test_user",
            rotation_interval_days=30,
            tags=["test", "symmetric"],
        )

        sym_key_id = await kms.generate_key(symmetric_metadata)
        print(f"Generated symmetric key: {sym_key_id}")

        # Encrypt data
        test_data = b"This is sensitive data that needs encryption!"
        context = {"purpose": "test", "department": "security"}

        encrypted_result = await kms.encrypt_data(sym_key_id, test_data, context)
        print(f"Encrypted data: {encrypted_result['ciphertext'][:50]}...")

        # Decrypt data
        decrypted_data = await kms.decrypt_data(encrypted_result)
        print(f"Decrypted data: {decrypted_data.decode()}")

        # Test key rotation
        print(f"Original key ID: {sym_key_id}")
        new_key_id = await kms.key_rotation.rotate_key(sym_key_id)
        print(f"Rotated key ID: {new_key_id}")

        # List keys
        active_keys = await kms.list_keys(status=KeyStatus.ACTIVE)
        print(f"Active keys: {len(active_keys)}")

        # Get audit log
        audit_entries = await kms.get_audit_log()
        print(f"Audit entries: {len(audit_entries)}")
        for entry in audit_entries[-3:]:  # Show last 3 entries
            print(f"  {entry['event_type']}: {entry['timestamp']}")

    # Run test
    asyncio.run(test_enterprise_kms())
