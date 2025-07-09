"""
Security Configuration for Pynomaly Production Environment
"""

from __future__ import annotations

import os

from pydantic import BaseModel, Field

from pynomaly.infrastructure.security.authentication import SecurityConfig
from pynomaly.infrastructure.security.encryption import EncryptionConfig


class SecuritySettings(BaseModel):
    """Comprehensive security settings for production deployment."""

    # Environment
    environment: str = Field(default="production", description="Environment type")
    debug_mode: bool = Field(default=False, description="Enable debug mode")

    # API Security
    api_key_required: bool = Field(default=True, description="Require API key for all requests")
    rate_limiting_enabled: bool = Field(default=True, description="Enable rate limiting")
    cors_enabled: bool = Field(default=True, description="Enable CORS")
    cors_allowed_origins: list[str] = Field(default_factory=lambda: ["https://app.pynomaly.com"])

    # Authentication
    jwt_required: bool = Field(default=True, description="Require JWT for authenticated endpoints")
    session_management: bool = Field(default=True, description="Enable session management")
    two_factor_auth: bool = Field(default=True, description="Enable 2FA")

    # Authorization
    rbac_enabled: bool = Field(default=True, description="Enable role-based access control")
    permission_checks: bool = Field(default=True, description="Enable permission checks")

    # Data Protection
    encryption_at_rest: bool = Field(default=True, description="Enable encryption at rest")
    encryption_in_transit: bool = Field(default=True, description="Enable encryption in transit")
    field_level_encryption: bool = Field(default=True, description="Enable field-level encryption")

    # Monitoring
    security_monitoring: bool = Field(default=True, description="Enable security monitoring")
    audit_logging: bool = Field(default=True, description="Enable audit logging")
    threat_detection: bool = Field(default=True, description="Enable threat detection")

    # Compliance
    gdpr_compliance: bool = Field(default=True, description="Enable GDPR compliance features")
    data_retention_days: int = Field(default=365, description="Data retention period in days")

    # SSL/TLS
    ssl_required: bool = Field(default=True, description="Require SSL/TLS")
    ssl_cert_path: str | None = Field(default=None, description="SSL certificate path")
    ssl_key_path: str | None = Field(default=None, description="SSL private key path")

    # Database Security
    database_ssl: bool = Field(default=True, description="Enable database SSL")
    database_encryption: bool = Field(default=True, description="Enable database encryption")

    # File Security
    file_upload_restrictions: bool = Field(default=True, description="Enable file upload restrictions")
    max_file_size_mb: int = Field(default=100, description="Maximum file size in MB")
    allowed_file_types: list[str] = Field(default_factory=lambda: [".csv", ".json", ".parquet"])

    # Network Security
    ip_whitelist_enabled: bool = Field(default=False, description="Enable IP whitelist")
    ip_whitelist: list[str] = Field(default_factory=list, description="Allowed IP addresses")

    # Password Policy
    password_policy_enabled: bool = Field(default=True, description="Enable password policy")
    password_min_length: int = Field(default=12, description="Minimum password length")
    password_complexity: bool = Field(default=True, description="Require password complexity")

    # Session Security
    session_timeout_minutes: int = Field(default=30, description="Session timeout in minutes")
    max_concurrent_sessions: int = Field(default=3, description="Maximum concurrent sessions")

    # Security Headers
    security_headers_enabled: bool = Field(default=True, description="Enable security headers")
    content_security_policy: str = Field(
        default="default-src 'self'; script-src 'self' 'unsafe-inline'; style-src 'self' 'unsafe-inline'",
        description="Content Security Policy"
    )


class ProductionSecurityConfig:
    """Production security configuration manager."""

    def __init__(self, config_path: str | None = None):
        self.config_path = config_path or os.getenv("SECURITY_CONFIG_PATH", "config/security")
        self.settings = SecuritySettings()
        self._load_from_environment()

    def _load_from_environment(self) -> None:
        """Load security settings from environment variables."""
        # API Security
        if os.getenv("API_KEY_REQUIRED"):
            self.settings.api_key_required = os.getenv("API_KEY_REQUIRED").lower() == "true"

        if os.getenv("RATE_LIMITING_ENABLED"):
            self.settings.rate_limiting_enabled = os.getenv("RATE_LIMITING_ENABLED").lower() == "true"

        # CORS
        if os.getenv("CORS_ALLOWED_ORIGINS"):
            self.settings.cors_allowed_origins = os.getenv("CORS_ALLOWED_ORIGINS").split(",")

        # Authentication
        if os.getenv("JWT_REQUIRED"):
            self.settings.jwt_required = os.getenv("JWT_REQUIRED").lower() == "true"

        if os.getenv("TWO_FACTOR_AUTH"):
            self.settings.two_factor_auth = os.getenv("TWO_FACTOR_AUTH").lower() == "true"

        # SSL/TLS
        if os.getenv("SSL_CERT_PATH"):
            self.settings.ssl_cert_path = os.getenv("SSL_CERT_PATH")

        if os.getenv("SSL_KEY_PATH"):
            self.settings.ssl_key_path = os.getenv("SSL_KEY_PATH")

        # Database
        if os.getenv("DATABASE_SSL"):
            self.settings.database_ssl = os.getenv("DATABASE_SSL").lower() == "true"

        # File Security
        if os.getenv("MAX_FILE_SIZE_MB"):
            self.settings.max_file_size_mb = int(os.getenv("MAX_FILE_SIZE_MB"))

        if os.getenv("ALLOWED_FILE_TYPES"):
            self.settings.allowed_file_types = os.getenv("ALLOWED_FILE_TYPES").split(",")

        # Network Security
        if os.getenv("IP_WHITELIST"):
            self.settings.ip_whitelist = os.getenv("IP_WHITELIST").split(",")
            self.settings.ip_whitelist_enabled = True

        # Password Policy
        if os.getenv("PASSWORD_MIN_LENGTH"):
            self.settings.password_min_length = int(os.getenv("PASSWORD_MIN_LENGTH"))

        # Session Security
        if os.getenv("SESSION_TIMEOUT_MINUTES"):
            self.settings.session_timeout_minutes = int(os.getenv("SESSION_TIMEOUT_MINUTES"))

        if os.getenv("MAX_CONCURRENT_SESSIONS"):
            self.settings.max_concurrent_sessions = int(os.getenv("MAX_CONCURRENT_SESSIONS"))

    def get_security_config(self) -> SecurityConfig:
        """Get authentication security configuration."""
        return SecurityConfig(
            jwt_secret_key=os.getenv("JWT_SECRET_KEY", ""),
            jwt_algorithm="HS256",
            jwt_expiration_hours=24,
            refresh_token_expiration_days=7,
            password_min_length=self.settings.password_min_length,
            password_require_uppercase=True,
            password_require_lowercase=True,
            password_require_numbers=True,
            password_require_symbols=True,
            api_key_length=32,
            api_key_prefix="pyn_",
            rate_limit_requests_per_minute=100,
            rate_limit_burst_size=20,
            session_timeout_minutes=self.settings.session_timeout_minutes,
            max_concurrent_sessions=self.settings.max_concurrent_sessions,
            encryption_key=os.getenv("ENCRYPTION_KEY", ""),
            audit_log_enabled=self.settings.audit_logging,
            audit_log_retention_days=90
        )

    def get_encryption_config(self) -> EncryptionConfig:
        """Get encryption configuration."""
        return EncryptionConfig(
            symmetric_key=os.getenv("SYMMETRIC_KEY"),
            fernet_key=os.getenv("FERNET_KEY"),
            kdf_salt_size=32,
            kdf_iterations=100000,
            hash_algorithm="sha256",
            hmac_key=os.getenv("HMAC_KEY"),
            database_encryption_key=os.getenv("DATABASE_ENCRYPTION_KEY"),
            file_encryption_key=os.getenv("FILE_ENCRYPTION_KEY"),
            pii_encryption_key=os.getenv("PII_ENCRYPTION_KEY"),
            sensitive_data_key=os.getenv("SENSITIVE_DATA_KEY")
        )

    def get_security_headers(self) -> dict[str, str]:
        """Get security headers configuration."""
        headers = {}

        if self.settings.security_headers_enabled:
            headers.update({
                "X-Frame-Options": "DENY",
                "X-Content-Type-Options": "nosniff",
                "X-XSS-Protection": "1; mode=block",
                "Strict-Transport-Security": "max-age=31536000; includeSubDomains",
                "Content-Security-Policy": self.settings.content_security_policy,
                "Referrer-Policy": "strict-origin-when-cross-origin",
                "Permissions-Policy": "geolocation=(), microphone=(), camera=()"
            })

        return headers

    def validate_configuration(self) -> list[str]:
        """Validate security configuration and return warnings."""
        warnings = []

        # Check for required environment variables
        required_vars = [
            "JWT_SECRET_KEY",
            "ENCRYPTION_KEY",
            "HMAC_KEY"
        ]

        for var in required_vars:
            if not os.getenv(var):
                warnings.append(f"Missing required environment variable: {var}")

        # Check SSL configuration
        if self.settings.ssl_required:
            if not self.settings.ssl_cert_path or not self.settings.ssl_key_path:
                warnings.append("SSL is required but certificate paths are not configured")

        # Check password policy
        if self.settings.password_min_length < 8:
            warnings.append("Password minimum length should be at least 8 characters")

        # Check session timeout
        if self.settings.session_timeout_minutes > 60:
            warnings.append("Session timeout is set to more than 60 minutes")

        # Check CORS configuration
        if self.settings.cors_enabled and not self.settings.cors_allowed_origins:
            warnings.append("CORS is enabled but no allowed origins are configured")

        return warnings

    def generate_security_report(self) -> dict[str, any]:
        """Generate a comprehensive security report."""
        warnings = self.validate_configuration()

        return {
            "configuration_status": "valid" if not warnings else "warnings",
            "warnings": warnings,
            "security_features": {
                "api_key_required": self.settings.api_key_required,
                "rate_limiting_enabled": self.settings.rate_limiting_enabled,
                "jwt_required": self.settings.jwt_required,
                "two_factor_auth": self.settings.two_factor_auth,
                "rbac_enabled": self.settings.rbac_enabled,
                "encryption_at_rest": self.settings.encryption_at_rest,
                "encryption_in_transit": self.settings.encryption_in_transit,
                "field_level_encryption": self.settings.field_level_encryption,
                "security_monitoring": self.settings.security_monitoring,
                "audit_logging": self.settings.audit_logging,
                "threat_detection": self.settings.threat_detection,
                "gdpr_compliance": self.settings.gdpr_compliance,
                "ssl_required": self.settings.ssl_required,
                "database_ssl": self.settings.database_ssl,
                "file_upload_restrictions": self.settings.file_upload_restrictions,
                "ip_whitelist_enabled": self.settings.ip_whitelist_enabled,
                "password_policy_enabled": self.settings.password_policy_enabled,
                "security_headers_enabled": self.settings.security_headers_enabled
            },
            "configuration_details": {
                "environment": self.settings.environment,
                "debug_mode": self.settings.debug_mode,
                "password_min_length": self.settings.password_min_length,
                "session_timeout_minutes": self.settings.session_timeout_minutes,
                "max_concurrent_sessions": self.settings.max_concurrent_sessions,
                "max_file_size_mb": self.settings.max_file_size_mb,
                "allowed_file_types": self.settings.allowed_file_types,
                "data_retention_days": self.settings.data_retention_days,
                "cors_allowed_origins": self.settings.cors_allowed_origins,
                "ip_whitelist": self.settings.ip_whitelist if self.settings.ip_whitelist_enabled else []
            }
        }

    def export_security_config(self, output_path: str) -> None:
        """Export security configuration to a file."""
        config_data = {
            "security_settings": self.settings.dict(),
            "security_report": self.generate_security_report()
        }

        import json
        with open(output_path, 'w') as f:
            json.dump(config_data, f, indent=2, default=str)

    def create_environment_template(self, output_path: str) -> None:
        """Create a template .env file with security variables."""
        template = """# Pynomaly Security Configuration
# Copy this file to .env and fill in the values

# JWT Configuration
JWT_SECRET_KEY=your-jwt-secret-key-here
JWT_ALGORITHM=HS256

# Encryption Keys
ENCRYPTION_KEY=your-encryption-key-here
FERNET_KEY=your-fernet-key-here
HMAC_KEY=your-hmac-key-here
SYMMETRIC_KEY=your-symmetric-key-here

# Database Encryption
DATABASE_ENCRYPTION_KEY=your-database-encryption-key-here

# Field-Level Encryption
PII_ENCRYPTION_KEY=your-pii-encryption-key-here
SENSITIVE_DATA_KEY=your-sensitive-data-key-here
FILE_ENCRYPTION_KEY=your-file-encryption-key-here

# API Security
API_KEY_REQUIRED=true
RATE_LIMITING_ENABLED=true

# CORS Configuration
CORS_ALLOWED_ORIGINS=https://app.pynomaly.com,https://dashboard.pynomaly.com

# Authentication
JWT_REQUIRED=true
TWO_FACTOR_AUTH=true

# SSL/TLS
SSL_CERT_PATH=/path/to/cert.pem
SSL_KEY_PATH=/path/to/key.pem

# Database Security
DATABASE_SSL=true

# File Security
MAX_FILE_SIZE_MB=100
ALLOWED_FILE_TYPES=.csv,.json,.parquet

# Network Security (optional)
# IP_WHITELIST=192.168.1.0/24,10.0.0.0/8

# Password Policy
PASSWORD_MIN_LENGTH=12

# Session Security
SESSION_TIMEOUT_MINUTES=30
MAX_CONCURRENT_SESSIONS=3

# Environment
ENVIRONMENT=production
DEBUG_MODE=false
"""

        with open(output_path, 'w') as f:
            f.write(template)


# Global security configuration instance
security_config = ProductionSecurityConfig()
