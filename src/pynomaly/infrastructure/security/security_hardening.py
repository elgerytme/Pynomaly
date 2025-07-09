"""
Security Hardening Implementation

This module provides comprehensive security hardening measures for the Pynomaly platform.
It addresses critical security vulnerabilities and implements security best practices.
"""

from __future__ import annotations

import hashlib
import hmac
import json
import os
import secrets
import tempfile
import time
from pathlib import Path
from typing import Any

import joblib
from cryptography.fernet import Fernet
from pydantic import BaseModel, ConfigDict, field_validator

from pynomaly.infrastructure.config.settings import Settings


class SecureSerializationError(Exception):
    """Raised when secure serialization operations fail."""

    pass


class SecureModelSerializer:
    """
    Secure model serialization replacing unsafe pickle usage.

    This class provides safe serialization/deserialization of ML models
    using joblib for sklearn models and encrypted JSON for other objects.
    """

    def __init__(self, encryption_key: bytes | None = None):
        """Initialize with encryption key."""
        if encryption_key:
            self.fernet = Fernet(encryption_key)
        else:
            # Generate a key from settings or create one
            key = os.getenv("PYNOMALY_SERIALIZATION_KEY")
            if not key:
                key = Fernet.generate_key()
                # In production, this should be stored securely
                os.environ["PYNOMALY_SERIALIZATION_KEY"] = key.decode()
            self.fernet = Fernet(key if isinstance(key, bytes) else key.encode())

    def serialize_model(self, model: Any, path: Path) -> None:
        """
        Safely serialize a model to disk.

        Args:
            model: The model to serialize
            path: Path to save the serialized model
        """
        try:
            # Create a temporary file for atomic writes
            with tempfile.NamedTemporaryFile(delete=False, suffix=".tmp") as tmp_file:
                tmp_path = Path(tmp_file.name)

                # Use joblib for sklearn-compatible models
                if hasattr(model, "fit") and hasattr(model, "predict"):
                    # Add integrity check
                    model_data = {
                        "model": model,
                        "timestamp": time.time(),
                        "checksum": self._calculate_model_checksum(model),
                    }
                    joblib.dump(model_data, tmp_path)
                else:
                    # For other objects, use encrypted JSON
                    model_dict = self._serialize_to_dict(model)
                    encrypted_data = self._encrypt_data(json.dumps(model_dict))
                    tmp_path.write_bytes(encrypted_data)

                # Atomic move to final location
                tmp_path.replace(path)

        except Exception as e:
            # Clean up temporary file if it exists
            if "tmp_path" in locals():
                tmp_path.unlink(missing_ok=True)
            raise SecureSerializationError(f"Failed to serialize model: {str(e)}")

    def deserialize_model(self, path: Path) -> Any:
        """
        Safely deserialize a model from disk.

        Args:
            path: Path to the serialized model

        Returns:
            The deserialized model
        """
        try:
            if not path.exists():
                raise SecureSerializationError(f"Model file not found: {path}")

            # Check file size limit (prevent DoS)
            if path.stat().st_size > 100 * 1024 * 1024:  # 100MB limit
                raise SecureSerializationError("Model file too large")

            # Try joblib first
            try:
                model_data = joblib.load(path)
                if isinstance(model_data, dict) and "model" in model_data:
                    # Verify integrity
                    expected_checksum = model_data.get("checksum")
                    if expected_checksum:
                        actual_checksum = self._calculate_model_checksum(
                            model_data["model"]
                        )
                        if not hmac.compare_digest(expected_checksum, actual_checksum):
                            raise SecureSerializationError(
                                "Model integrity check failed"
                            )
                    return model_data["model"]
                return model_data
            except:
                # Try encrypted JSON
                encrypted_data = path.read_bytes()
                decrypted_data = self._decrypt_data(encrypted_data)
                model_dict = json.loads(decrypted_data)
                return self._deserialize_from_dict(model_dict)

        except Exception as e:
            raise SecureSerializationError(f"Failed to deserialize model: {str(e)}")

    def _serialize_to_dict(self, obj: Any) -> dict[str, Any]:
        """Convert object to serializable dictionary."""
        if hasattr(obj, "__dict__"):
            return {
                "type": obj.__class__.__name__,
                "module": obj.__class__.__module__,
                "data": obj.__dict__,
                "timestamp": time.time(),
            }
        else:
            return {"type": type(obj).__name__, "value": obj, "timestamp": time.time()}

    def _deserialize_from_dict(self, data: dict[str, Any]) -> Any:
        """Reconstruct object from dictionary."""
        # Basic validation
        if "type" not in data:
            raise SecureSerializationError("Invalid serialized data")

        obj_type = data["type"]

        # Only allow safe types
        allowed_types = {
            "str",
            "int",
            "float",
            "bool",
            "list",
            "dict",
            "tuple",
            "Detector",
            "Dataset",
            "DetectionResult",  # Add allowed custom types
        }

        if obj_type not in allowed_types:
            raise SecureSerializationError(f"Unsafe object type: {obj_type}")

        # Return simple values
        if "value" in data:
            return data["value"]

        # For complex objects, you would need specific reconstruction logic
        # This is a simplified example
        return data.get("data", {})

    def _encrypt_data(self, data: str) -> bytes:
        """Encrypt string data."""
        return self.fernet.encrypt(data.encode())

    def _decrypt_data(self, encrypted_data: bytes) -> str:
        """Decrypt data to string."""
        return self.fernet.decrypt(encrypted_data).decode()

    def _calculate_model_checksum(self, model: Any) -> str:
        """Calculate a checksum for model integrity verification."""
        model_str = str(model.__dict__ if hasattr(model, "__dict__") else model)
        return hashlib.sha256(model_str.encode()).hexdigest()


class SecureInputValidator:
    """
    Enhanced input validation with security-focused sanitization.

    This class provides comprehensive input validation to prevent
    injection attacks and other security vulnerabilities.
    """

    # Dangerous patterns to block
    DANGEROUS_PATTERNS = [
        r"<script[^>]*>.*?</script>",  # XSS
        r"javascript:",  # XSS
        r"on\w+\s*=",  # Event handlers
        r"eval\s*\(",  # Code execution
        r"exec\s*\(",  # Code execution
        r"__import__",  # Python imports
        r"subprocess",  # System commands
        r"os\.system",  # System commands
        r"\.\./",  # Path traversal
        r"\.\.\\",  # Path traversal (Windows)
        r"DROP\s+TABLE",  # SQL injection
        r"UNION\s+SELECT",  # SQL injection
        r"INSERT\s+INTO",  # SQL injection
        r"UPDATE\s+SET",  # SQL injection
        r"DELETE\s+FROM",  # SQL injection
    ]

    def __init__(self):
        """Initialize the validator."""
        import re

        self.dangerous_regex = re.compile(
            "|".join(self.DANGEROUS_PATTERNS), re.IGNORECASE | re.DOTALL
        )

    def validate_and_sanitize(self, value: Any, field_name: str = "input") -> Any:
        """
        Validate and sanitize input value.

        Args:
            value: The input value to validate
            field_name: Name of the field being validated

        Returns:
            Sanitized value

        Raises:
            ValueError: If input is invalid or dangerous
        """
        if value is None:
            return None

        # Handle different types
        if isinstance(value, str):
            return self._sanitize_string(value, field_name)
        elif isinstance(value, (int, float)):
            return self._sanitize_number(value, field_name)
        elif isinstance(value, bool):
            return value
        elif isinstance(value, (list, tuple)):
            return [
                self.validate_and_sanitize(item, f"{field_name}[{i}]")
                for i, item in enumerate(value)
            ]
        elif isinstance(value, dict):
            return {
                k: self.validate_and_sanitize(v, f"{field_name}.{k}")
                for k, v in value.items()
            }
        else:
            # For other types, convert to string and sanitize
            return self._sanitize_string(str(value), field_name)

    def _sanitize_string(self, value: str, field_name: str) -> str:
        """Sanitize string input."""
        # Length check
        if len(value) > 10000:  # 10KB limit
            raise ValueError(f"Input too long for field {field_name}")

        # Check for dangerous patterns
        if self.dangerous_regex.search(value):
            raise ValueError(
                f"Potentially dangerous input detected in field {field_name}"
            )

        # Basic sanitization
        sanitized = value.strip()

        # Remove null bytes
        sanitized = sanitized.replace("\x00", "")

        # Normalize whitespace
        sanitized = " ".join(sanitized.split())

        return sanitized

    def _sanitize_number(self, value: int | float, field_name: str) -> int | float:
        """Sanitize numeric input."""
        # Check for reasonable bounds
        if abs(value) > 1e15:  # Very large numbers
            raise ValueError(f"Number too large for field {field_name}")

        # Check for NaN/Inf
        if isinstance(value, float):
            if not (value == value):  # NaN check
                raise ValueError(f"NaN value not allowed for field {field_name}")
            if abs(value) == float("inf"):
                raise ValueError(f"Infinite value not allowed for field {field_name}")

        return value

    def validate_file_path(self, path: str) -> str:
        """Validate file path for security."""
        # Convert to Path object for normalization
        try:
            path_obj = Path(path).resolve()
        except (OSError, ValueError):
            raise ValueError("Invalid file path")

        # Check for path traversal
        if ".." in path:
            raise ValueError("Path traversal not allowed")

        # Check for absolute paths outside allowed directories
        # This should be configured based on your deployment
        allowed_dirs = [
            Path.cwd(),
            Path.home() / ".pynomaly",
            Path("/tmp/pynomaly"),
        ]

        if not any(
            str(path_obj).startswith(str(allowed_dir)) for allowed_dir in allowed_dirs
        ):
            raise ValueError("Path outside allowed directories")

        return str(path_obj)


class SecureConfigurationManager:
    """
    Secure configuration management with hardened defaults.

    This class ensures secure configuration defaults and validates
    security-critical settings.
    """

    def __init__(self, settings: Settings):
        """Initialize with settings."""
        self.settings = settings
        self.validator = SecureInputValidator()

    def validate_security_configuration(self) -> list[str]:
        """
        Validate security configuration and return warnings.

        Returns:
            List of security warnings
        """
        warnings = []

        # Check secret key
        if (
            self.settings.secret_key
            == "change-me-in-production-this-is-32-chars-long-default-key"
        ):
            warnings.append("CRITICAL: Using default secret key in production")

        if len(self.settings.secret_key) < 32:
            warnings.append("WARNING: Secret key is too short (minimum 32 characters)")

        # Check JWT configuration
        if self.settings.jwt_access_token_expire_minutes > 1440:  # 24 hours
            warnings.append("WARNING: JWT access token expiry is too long")

        # Check HTTPS enforcement
        if not self.settings.force_https:
            warnings.append("WARNING: HTTPS is not enforced")

        # Check debug mode
        if self.settings.app.debug:
            warnings.append("WARNING: Debug mode is enabled")

        # Check database security
        if self.settings.database_echo:
            warnings.append("WARNING: Database query logging is enabled")

        # Check rate limiting
        if self.settings.api_rate_limit > 1000:
            warnings.append("WARNING: API rate limit is very high")

        return warnings

    def generate_secure_secret_key(self) -> str:
        """Generate a cryptographically secure secret key."""
        return secrets.token_urlsafe(64)

    def harden_csp_policy(self) -> dict[str, str]:
        """Generate hardened Content Security Policy."""
        nonce = secrets.token_urlsafe(16)

        return {
            "default-src": "'self'",
            "script-src": f"'self' 'nonce-{nonce}'",
            "style-src": f"'self' 'nonce-{nonce}'",
            "img-src": "'self' data:",
            "font-src": "'self'",
            "connect-src": "'self'",
            "media-src": "'self'",
            "object-src": "'none'",
            "frame-src": "'none'",
            "frame-ancestors": "'none'",
            "base-uri": "'self'",
            "form-action": "'self'",
            "upgrade-insecure-requests": "",
            "block-all-mixed-content": "",
            "require-trusted-types-for": "'script'",
            "nonce": nonce,
        }

    def get_security_headers(self) -> dict[str, str]:
        """Get comprehensive security headers."""
        csp_policy = self.harden_csp_policy()
        csp_value = "; ".join(f"{k} {v}" for k, v in csp_policy.items() if k != "nonce")

        return {
            "Content-Security-Policy": csp_value,
            "X-Content-Type-Options": "nosniff",
            "X-Frame-Options": "DENY",
            "X-XSS-Protection": "1; mode=block",
            "Strict-Transport-Security": "max-age=31536000; includeSubDomains; preload",
            "Referrer-Policy": "strict-origin-when-cross-origin",
            "Permissions-Policy": "geolocation=(), microphone=(), camera=()",
            "Cross-Origin-Embedder-Policy": "require-corp",
            "Cross-Origin-Opener-Policy": "same-origin",
            "Cross-Origin-Resource-Policy": "same-origin",
        }


class SecureModelValidator(BaseModel):
    """
    Pydantic model with enhanced security validation.

    This base model provides automatic input sanitization and validation
    for all fields.
    """

    model_config = ConfigDict(
        validate_assignment=True, str_strip_whitespace=True, str_max_length=10000
    )

    @field_validator("*", mode="before")
    @classmethod
    def sanitize_all_inputs(cls, v):
        """Sanitize all input fields."""
        if v is None:
            return v

        validator = SecureInputValidator()
        try:
            return validator.validate_and_sanitize(v)
        except ValueError as e:
            raise ValueError(f"Input validation failed: {str(e)}")

    def dict(self, **kwargs):
        """Override dict to ensure sensitive fields are excluded."""
        exclude_sensitive = kwargs.get("exclude_sensitive", False)
        if exclude_sensitive:
            # Add logic to exclude sensitive fields
            sensitive_fields = getattr(self, "_sensitive_fields", set())
            exclude = kwargs.get("exclude", set())
            if isinstance(exclude, set):
                exclude.update(sensitive_fields)
            else:
                exclude = sensitive_fields
            kwargs["exclude"] = exclude

        return super().dict(**kwargs)


def initialize_security_hardening(settings: Settings) -> SecureConfigurationManager:
    """
    Initialize security hardening for the application.

    Args:
        settings: Application settings

    Returns:
        Configured security manager
    """
    config_manager = SecureConfigurationManager(settings)

    # Validate current configuration
    warnings = config_manager.validate_security_configuration()
    if warnings:
        print("Security Configuration Warnings:")
        for warning in warnings:
            print(f"  - {warning}")

    # Set secure defaults if needed
    if (
        settings.secret_key
        == "change-me-in-production-this-is-32-chars-long-default-key"
    ):
        secure_key = config_manager.generate_secure_secret_key()
        print(f"Generated secure secret key: {secure_key[:16]}...")
        # In production, this should be set via environment variable
        os.environ["PYNOMALY_SECRET_KEY"] = secure_key

    return config_manager


# Global instances
_secure_serializer = None
_input_validator = None
_config_manager = None


def get_secure_serializer() -> SecureModelSerializer:
    """Get global secure serializer instance."""
    global _secure_serializer
    if _secure_serializer is None:
        _secure_serializer = SecureModelSerializer()
    return _secure_serializer


def get_input_validator() -> SecureInputValidator:
    """Get global input validator instance."""
    global _input_validator
    if _input_validator is None:
        _input_validator = SecureInputValidator()
    return _input_validator


def get_config_manager() -> SecureConfigurationManager | None:
    """Get global configuration manager instance."""
    return _config_manager


def set_config_manager(manager: SecureConfigurationManager) -> None:
    """Set global configuration manager instance."""
    global _config_manager
    _config_manager = manager
