"""
Encryption and cryptographic utilities for Software API.

This module provides:
- Data encryption at rest and in transit
- Key management
- Digital signatures
- Secure random generation
- Hash functions
"""

import base64
import hashlib
import hmac
import logging
import os
import secrets
from datetime import datetime
from typing import Any

from cryptography.fernet import Fernet
from cryptography.hazmat.backends import default_backend
from cryptography.hazmat.primitives import hashes, serialization
from cryptography.hazmat.primitives.asymmetric import padding, rsa
from cryptography.hazmat.primitives.kdf.pbkdf2 import PBKDF2HMAC

logger = logging.getLogger(__name__)


class EncryptionManager:
    """Comprehensive encryption management."""

    def __init__(self, master_key: str | None = None):
        self.master_key = master_key or self._generate_master_key()
        self.fernet = Fernet(
            self.master_key.encode()
            if isinstance(self.master_key, str)
            else self.master_key
        )
        self.key_cache = {}
        self.algorithm_config = {
            "symmetric": "AES-256-GCM",
            "asymmetric": "RSA-2048",
            "hash": "SHA-256",
            "kdf": "PBKDF2",
        }

    def _generate_master_key(self) -> bytes:
        """Generate master encryption key."""
        return Fernet.generate_key()

    def encrypt_data(self, data: str, key_id: str | None = None) -> dict[str, str]:
        """Encrypt data with optional key rotation."""
        try:
            if key_id and key_id in self.key_cache:
                fernet = self.key_cache[key_id]
            else:
                fernet = self.fernet
                key_id = "master"

            # Convert string to bytes
            if isinstance(data, str):
                data_bytes = data.encode("utf-8")
            else:
                data_bytes = data

            # Encrypt data
            encrypted_data = fernet.encrypt(data_bytes)

            # Return encrypted data with metadata
            return {
                "encrypted_data": base64.b64encode(encrypted_data).decode("utf-8"),
                "key_id": key_id,
                "algorithm": "Fernet",
                "timestamp": datetime.utcnow().isoformat(),
            }

        except Exception as e:
            logger.error(f"Encryption failed: {e}")
            raise EncryptionError(f"Failed to encrypt data: {e}")

    def decrypt_data(self, encrypted_data: dict[str, str]) -> str:
        """Decrypt data using appropriate key."""
        try:
            key_id = encrypted_data.get("key_id", "master")

            if key_id in self.key_cache:
                fernet = self.key_cache[key_id]
            else:
                fernet = self.fernet

            # Decode and decrypt
            encrypted_bytes = base64.b64decode(encrypted_data["encrypted_data"])
            decrypted_bytes = fernet.decrypt(encrypted_bytes)

            return decrypted_bytes.decode("utf-8")

        except Exception as e:
            logger.error(f"Decryption failed: {e}")
            raise DecryptionError(f"Failed to decrypt data: {e}")

    def rotate_key(self, old_key_id: str) -> str:
        """Rotate encryption key."""
        new_key = Fernet.generate_key()
        new_key_id = f"key_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}"

        # Store old key for decryption
        if old_key_id == "master":
            self.key_cache[f"master_old_{new_key_id}"] = self.fernet

        # Create new Fernet instance
        self.key_cache[new_key_id] = Fernet(new_key)

        logger.info(f"Key rotated: {old_key_id} -> {new_key_id}")
        return new_key_id

    def generate_data_key(self) -> tuple[str, bytes]:
        """Generate data encryption key."""
        key = Fernet.generate_key()
        key_id = f"data_key_{secrets.token_hex(8)}"
        self.key_cache[key_id] = Fernet(key)
        return key_id, key


class AsymmetricCrypto:
    """Asymmetric cryptography for key exchange and digital signatures."""

    def __init__(self):
        self.private_key = None
        self.public_key = None
        self.key_pairs = {}

    def generate_key_pair(self, key_size: int = 2048) -> dict[str, str]:
        """Generate RSA key pair."""
        private_key = rsa.generate_private_key(
            public_exponent=65537, key_size=key_size, backend=default_backend()
        )

        public_key = private_key.public_key()

        # Serialize keys
        private_pem = private_key.private_bytes(
            encoding=serialization.Encoding.PEM,
            format=serialization.PrivateFormat.PKCS8,
            encryption_algorithm=serialization.NoEncryption(),
        )

        public_pem = public_key.public_bytes(
            encoding=serialization.Encoding.PEM,
            format=serialization.PublicFormat.SubjectPublicKeyInfo,
        )

        key_id = f"keypair_{secrets.token_hex(8)}"
        self.key_pairs[key_id] = {"private": private_key, "public": public_key}

        return {
            "key_id": key_id,
            "private_key": private_pem.decode("utf-8"),
            "public_key": public_pem.decode("utf-8"),
        }

    def encrypt_with_public_key(self, data: str, public_key_pem: str) -> str:
        """Encrypt data with RSA public key."""
        try:
            # Load public key
            public_key = serialization.load_pem_public_key(
                public_key_pem.encode("utf-8"), backend=default_backend()
            )

            # Encrypt data
            encrypted = public_key.encrypt(
                data.encode("utf-8"),
                padding.OAEP(
                    mgf=padding.MGF1(algorithm=hashes.SHA256()),
                    algorithm=hashes.SHA256(),
                    label=None,
                ),
            )

            return base64.b64encode(encrypted).decode("utf-8")

        except Exception as e:
            logger.error(f"Public key encryption failed: {e}")
            raise EncryptionError(f"Failed to encrypt with public key: {e}")

    def decrypt_with_private_key(
        self, encrypted_data: str, private_key_pem: str
    ) -> str:
        """Decrypt data with RSA private key."""
        try:
            # Load private key
            private_key = serialization.load_pem_private_key(
                private_key_pem.encode("utf-8"),
                password=None,
                backend=default_backend(),
            )

            # Decrypt data
            encrypted_bytes = base64.b64decode(encrypted_data)
            decrypted = private_key.decrypt(
                encrypted_bytes,
                padding.OAEP(
                    mgf=padding.MGF1(algorithm=hashes.SHA256()),
                    algorithm=hashes.SHA256(),
                    label=None,
                ),
            )

            return decrypted.decode("utf-8")

        except Exception as e:
            logger.error(f"Private key decryption failed: {e}")
            raise DecryptionError(f"Failed to decrypt with private key: {e}")

    def sign_data(self, data: str, private_key_pem: str) -> str:
        """Create digital signature."""
        try:
            # Load private key
            private_key = serialization.load_pem_private_key(
                private_key_pem.encode("utf-8"),
                password=None,
                backend=default_backend(),
            )

            # Sign data
            signature = private_key.sign(
                data.encode("utf-8"),
                padding.PSS(
                    mgf=padding.MGF1(hashes.SHA256()),
                    salt_length=padding.PSS.MAX_LENGTH,
                ),
                hashes.SHA256(),
            )

            return base64.b64encode(signature).decode("utf-8")

        except Exception as e:
            logger.error(f"Signing failed: {e}")
            raise SignatureError(f"Failed to sign data: {e}")

    def verify_signature(self, data: str, signature: str, public_key_pem: str) -> bool:
        """Verify digital signature."""
        try:
            # Load public key
            public_key = serialization.load_pem_public_key(
                public_key_pem.encode("utf-8"), backend=default_backend()
            )

            # Verify signature
            signature_bytes = base64.b64decode(signature)
            public_key.verify(
                signature_bytes,
                data.encode("utf-8"),
                padding.PSS(
                    mgf=padding.MGF1(hashes.SHA256()),
                    salt_length=padding.PSS.MAX_LENGTH,
                ),
                hashes.SHA256(),
            )

            return True

        except Exception as e:
            logger.warning(f"Signature verification failed: {e}")
            return False


class HashUtilities:
    """Cryptographic hash functions and utilities."""

    @staticmethod
    def sha256_hash(data: str) -> str:
        """Generate SHA-256 hash."""
        return hashlib.sha256(data.encode("utf-8")).hexdigest()

    @staticmethod
    def sha512_hash(data: str) -> str:
        """Generate SHA-512 hash."""
        return hashlib.sha512(data.encode("utf-8")).hexdigest()

    @staticmethod
    def hmac_signature(data: str, key: str, algorithm: str = "sha256") -> str:
        """Generate HMAC signature."""
        if algorithm == "sha256":
            hash_func = hashlib.sha256
        elif algorithm == "sha512":
            hash_func = hashlib.sha512
        else:
            raise ValueError(f"Unsupported HMAC algorithm: {algorithm}")

        signature = hmac.new(
            key.encode("utf-8"), data.encode("utf-8"), hash_func
        ).hexdigest()

        return signature

    @staticmethod
    def verify_hmac(
        data: str, signature: str, key: str, algorithm: str = "sha256"
    ) -> bool:
        """Verify HMAC signature."""
        expected_signature = HashUtilities.hmac_signature(data, key, algorithm)
        return hmac.compare_digest(signature, expected_signature)

    @staticmethod
    def derive_key(password: str, salt: bytes, iterations: int = 100000) -> bytes:
        """Derive key from password using PBKDF2."""
        kdf = PBKDF2HMAC(
            algorithm=hashes.SHA256(),
            length=32,
            salt=salt,
            iterations=iterations,
            backend=default_backend(),
        )

        return kdf.derive(password.encode("utf-8"))

    @staticmethod
    def generate_salt(length: int = 32) -> bytes:
        """Generate random salt."""
        return os.urandom(length)


class SecureRandom:
    """Secure random number and string generation."""

    @staticmethod
    def generate_token(length: int = 32) -> str:
        """Generate secure random token."""
        return secrets.token_hex(length)

    @staticmethod
    def generate_url_safe_token(length: int = 32) -> str:
        """Generate URL-safe random token."""
        return secrets.token_urlsafe(length)

    @staticmethod
    def generate_password(length: int = 16, include_symbols: bool = True) -> str:
        """Generate secure random password."""
        import string

        characters = string.ascii_letters + string.digits
        if include_symbols:
            characters += "!@#$%^&*"

        return "".join(secrets.choice(characters) for _ in range(length))

    @staticmethod
    def generate_uuid() -> str:
        """Generate secure UUID."""
        import uuid

        return str(uuid.uuid4())

    @staticmethod
    def generate_api_key(prefix: str = "pk") -> str:
        """Generate API key with prefix."""
        return f"{prefix}_{secrets.token_hex(16)}"


class FieldLevelEncryption:
    """Field-level encryption for sensitive data."""

    def __init__(self, encryption_manager: EncryptionManager):
        self.encryption_manager = encryption_manager
        self.encrypted_fields = {
            "email",
            "phone",
            "ssn",
            "credit_card",
            "api_key",
            "password_hash",
        }

    def encrypt_record(self, record: dict[str, Any]) -> dict[str, Any]:
        """Encrypt sensitive fields in record."""
        encrypted_record = record.copy()

        for field, value in record.items():
            if field in self.encrypted_fields and value is not None:
                encrypted_data = self.encryption_manager.encrypt_data(str(value))
                encrypted_record[f"{field}_encrypted"] = encrypted_data
                # Remove original field for security
                del encrypted_record[field]

        return encrypted_record

    def decrypt_record(self, record: dict[str, Any]) -> dict[str, Any]:
        """Decrypt sensitive fields in record."""
        decrypted_record = record.copy()

        # Find encrypted fields
        encrypted_fields = [
            field for field in record.keys() if field.endswith("_encrypted")
        ]

        for encrypted_field in encrypted_fields:
            original_field = encrypted_field.replace("_encrypted", "")

            try:
                encrypted_data = record[encrypted_field]
                decrypted_value = self.encryption_manager.decrypt_data(encrypted_data)
                decrypted_record[original_field] = decrypted_value
                # Remove encrypted field
                del decrypted_record[encrypted_field]
            except Exception as e:
                logger.error(f"Failed to decrypt field {original_field}: {e}")
                # Keep encrypted field if decryption fails
                continue

        return decrypted_record


class CertificateManager:
    """SSL/TLS certificate management."""

    def __init__(self):
        self.certificates = {}
        self.ca_certificates = {}

    def load_certificate(self, cert_id: str, cert_path: str, key_path: str) -> None:
        """Load SSL certificate and private key."""
        try:
            with open(cert_path, "rb") as cert_file:
                cert_data = cert_file.read()

            with open(key_path, "rb") as key_file:
                key_data = key_file.read()

            self.certificates[cert_id] = {
                "certificate": cert_data,
                "private_key": key_data,
                "loaded_at": datetime.utcnow(),
            }

            logger.info(f"Certificate loaded: {cert_id}")

        except Exception as e:
            logger.error(f"Failed to load certificate {cert_id}: {e}")
            raise CertificateError(f"Failed to load certificate: {e}")

    def get_certificate(self, cert_id: str) -> dict[str, bytes] | None:
        """Get certificate and private key."""
        return self.certificates.get(cert_id)

    def verify_certificate_chain(self, cert_data: bytes, ca_cert_data: bytes) -> bool:
        """Verify certificate chain (basic implementation)."""
        try:
            # Load certificates
            from cryptography import x509

            cert = x509.load_pem_x509_certificate(cert_data, default_backend())
            ca_cert = x509.load_pem_x509_certificate(ca_cert_data, default_backend())

            # Basic verification (in production, use proper chain validation)
            issuer = cert.issuer
            ca_subject = ca_cert.subject

            return issuer == ca_subject

        except Exception as e:
            logger.error(f"Certificate verification failed: {e}")
            return False


# Custom exceptions
class EncryptionError(Exception):
    """Raised when encryption fails."""

    pass


class DecryptionError(Exception):
    """Raised when decryption fails."""

    pass


class SignatureError(Exception):
    """Raised when digital signing fails."""

    pass


class CertificateError(Exception):
    """Raised when certificate operations fail."""

    pass


def create_encryption_config() -> dict[str, Any]:
    """Create default encryption configuration."""
    return {
        "encryption": {
            "master_key": os.getenv("PYNOMALY_MASTER_KEY"),
            "key_rotation_interval": 86400 * 30,  # 30 days
            "algorithm": "AES-256-GCM",
        },
        "hashing": {"algorithm": "SHA-256", "salt_length": 32, "iterations": 100000},
        "certificates": {
            "cert_path": "/etc/ssl/certs/monorepo.crt",
            "key_path": "/etc/ssl/private/monorepo.key",
            "ca_path": "/etc/ssl/certs/ca-bundle.crt",
        },
    }
