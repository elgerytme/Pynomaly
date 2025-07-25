"""Security configuration management."""

from __future__ import annotations

import os
from typing import Optional, Dict, Any, List
from pydantic import BaseModel, Field, validator
from pathlib import Path

from ...shared.infrastructure.config.base_settings import BasePackageSettings


class JWTConfig(BaseModel):
    """JWT token configuration."""
    secret_key: str = Field(..., min_length=32)
    algorithm: str = Field(default="HS256")
    access_token_expire_minutes: int = Field(default=30)
    refresh_token_expire_days: int = Field(default=7)
    issuer: str = Field(default="enterprise-security")
    audience: str = Field(default="enterprise-api")


class OAuth2Config(BaseModel):
    """OAuth2 configuration."""
    client_id: str
    client_secret: str
    authorize_url: str
    token_url: str
    redirect_uri: str
    scopes: List[str] = Field(default_factory=list)


class ComplianceConfig(BaseModel):
    """Compliance and audit configuration."""
    enable_audit_logging: bool = Field(default=True)
    audit_log_retention_days: int = Field(default=90)
    enable_gdpr_compliance: bool = Field(default=True)
    enable_hipaa_compliance: bool = Field(default=False)
    enable_sox_compliance: bool = Field(default=False)
    data_classification_levels: List[str] = Field(
        default_factory=lambda: ["public", "internal", "confidential", "restricted"]
    )


class SecurityPolicyConfig(BaseModel):
    """Security policy configuration."""
    password_min_length: int = Field(default=12, ge=8)
    password_require_uppercase: bool = Field(default=True)
    password_require_lowercase: bool = Field(default=True)
    password_require_digits: bool = Field(default=True)
    password_require_special: bool = Field(default=True)
    password_expire_days: int = Field(default=90)
    max_login_attempts: int = Field(default=5, ge=1)
    lockout_duration_minutes: int = Field(default=30, ge=1)
    enable_mfa: bool = Field(default=True)
    session_timeout_minutes: int = Field(default=480, ge=1)


class EncryptionConfig(BaseModel):
    """Encryption configuration."""
    at_rest_encryption_key: str = Field(..., min_length=32)
    in_transit_encryption_enabled: bool = Field(default=True)
    encryption_algorithm: str = Field(default="AES-256-GCM")
    key_rotation_enabled: bool = Field(default=True)
    key_rotation_days: int = Field(default=90, ge=1)


class MonitoringConfig(BaseModel):
    """Security monitoring configuration."""
    enable_threat_detection: bool = Field(default=True)
    enable_anomaly_detection: bool = Field(default=True)
    enable_behavioral_analysis: bool = Field(default=True)
    alert_webhook_url: Optional[str] = None
    metrics_retention_days: int = Field(default=30, ge=1)
    log_level: str = Field(default="INFO", regex="^(DEBUG|INFO|WARN|ERROR|CRITICAL)$")


class SecurityConfig(BasePackageSettings):
    """Comprehensive security configuration."""
    
    def __init__(self):
        super().__init__(package_name="enterprise_security", env_prefix="SECURITY_")
        
        # Load security-specific configurations
        self.jwt = self._load_jwt_config()
        self.oauth2 = self._load_oauth2_config()
        self.compliance = self._load_compliance_config()
        self.policy = self._load_security_policy_config()
        self.encryption = self._load_encryption_config()
        self.monitoring = self._load_monitoring_config()
        
        # Additional security settings
        self.enable_security_headers: bool = self._get_bool("ENABLE_SECURITY_HEADERS", True)
        self.enable_cors: bool = self._get_bool("ENABLE_CORS", True)
        self.cors_origins: List[str] = self._get_list("CORS_ORIGINS", ["https://localhost:3000"])
        self.rate_limit_enabled: bool = self._get_bool("RATE_LIMIT_ENABLED", True)
        self.rate_limit_requests_per_minute: int = self._get_int("RATE_LIMIT_RPM", 100)
        
    def _load_jwt_config(self) -> JWTConfig:
        """Load JWT configuration."""
        return JWTConfig(
            secret_key=os.getenv(f"{self.env_prefix}JWT_SECRET_KEY", self._generate_secret()),
            algorithm=os.getenv(f"{self.env_prefix}JWT_ALGORITHM", "HS256"),
            access_token_expire_minutes=self._get_int("JWT_ACCESS_TOKEN_EXPIRE_MINUTES", 30),
            refresh_token_expire_days=self._get_int("JWT_REFRESH_TOKEN_EXPIRE_DAYS", 7),
            issuer=os.getenv(f"{self.env_prefix}JWT_ISSUER", "enterprise-security"),
            audience=os.getenv(f"{self.env_prefix}JWT_AUDIENCE", "enterprise-api")
        )
    
    def _load_oauth2_config(self) -> Optional[OAuth2Config]:
        """Load OAuth2 configuration if available."""
        client_id = os.getenv(f"{self.env_prefix}OAUTH2_CLIENT_ID")
        if not client_id:
            return None
            
        return OAuth2Config(
            client_id=client_id,
            client_secret=os.getenv(f"{self.env_prefix}OAUTH2_CLIENT_SECRET", ""),
            authorize_url=os.getenv(f"{self.env_prefix}OAUTH2_AUTHORIZE_URL", ""),
            token_url=os.getenv(f"{self.env_prefix}OAUTH2_TOKEN_URL", ""),
            redirect_uri=os.getenv(f"{self.env_prefix}OAUTH2_REDIRECT_URI", ""),
            scopes=self._get_list("OAUTH2_SCOPES", ["openid", "profile", "email"])
        )
    
    def _load_compliance_config(self) -> ComplianceConfig:
        """Load compliance configuration."""
        return ComplianceConfig(
            enable_audit_logging=self._get_bool("COMPLIANCE_AUDIT_LOGGING", True),
            audit_log_retention_days=self._get_int("COMPLIANCE_AUDIT_RETENTION_DAYS", 90),
            enable_gdpr_compliance=self._get_bool("COMPLIANCE_GDPR_ENABLED", True),
            enable_hipaa_compliance=self._get_bool("COMPLIANCE_HIPAA_ENABLED", False),
            enable_sox_compliance=self._get_bool("COMPLIANCE_SOX_ENABLED", False),
            data_classification_levels=self._get_list(
                "COMPLIANCE_DATA_CLASSIFICATION_LEVELS", 
                ["public", "internal", "confidential", "restricted"]
            )
        )
    
    def _load_security_policy_config(self) -> SecurityPolicyConfig:
        """Load security policy configuration."""
        return SecurityPolicyConfig(
            password_min_length=self._get_int("POLICY_PASSWORD_MIN_LENGTH", 12),
            password_require_uppercase=self._get_bool("POLICY_PASSWORD_REQUIRE_UPPERCASE", True),
            password_require_lowercase=self._get_bool("POLICY_PASSWORD_REQUIRE_LOWERCASE", True),
            password_require_digits=self._get_bool("POLICY_PASSWORD_REQUIRE_DIGITS", True),
            password_require_special=self._get_bool("POLICY_PASSWORD_REQUIRE_SPECIAL", True),
            password_expire_days=self._get_int("POLICY_PASSWORD_EXPIRE_DAYS", 90),
            max_login_attempts=self._get_int("POLICY_MAX_LOGIN_ATTEMPTS", 5),
            lockout_duration_minutes=self._get_int("POLICY_LOCKOUT_DURATION_MINUTES", 30),
            enable_mfa=self._get_bool("POLICY_ENABLE_MFA", True),
            session_timeout_minutes=self._get_int("POLICY_SESSION_TIMEOUT_MINUTES", 480)
        )
    
    def _load_encryption_config(self) -> EncryptionConfig:
        """Load encryption configuration."""
        return EncryptionConfig(
            at_rest_encryption_key=os.getenv(
                f"{self.env_prefix}ENCRYPTION_KEY", 
                self._generate_secret()
            ),
            in_transit_encryption_enabled=self._get_bool("ENCRYPTION_IN_TRANSIT_ENABLED", True),
            encryption_algorithm=os.getenv(f"{self.env_prefix}ENCRYPTION_ALGORITHM", "AES-256-GCM"),
            key_rotation_enabled=self._get_bool("ENCRYPTION_KEY_ROTATION_ENABLED", True),
            key_rotation_days=self._get_int("ENCRYPTION_KEY_ROTATION_DAYS", 90)
        )
    
    def _load_monitoring_config(self) -> MonitoringConfig:
        """Load monitoring configuration."""
        return MonitoringConfig(
            enable_threat_detection=self._get_bool("MONITORING_THREAT_DETECTION_ENABLED", True),
            enable_anomaly_detection=self._get_bool("MONITORING_ANOMALY_DETECTION_ENABLED", True),
            enable_behavioral_analysis=self._get_bool("MONITORING_BEHAVIORAL_ANALYSIS_ENABLED", True),
            alert_webhook_url=os.getenv(f"{self.env_prefix}MONITORING_ALERT_WEBHOOK_URL"),
            metrics_retention_days=self._get_int("MONITORING_METRICS_RETENTION_DAYS", 30),
            log_level=os.getenv(f"{self.env_prefix}MONITORING_LOG_LEVEL", "INFO")
        )
    
    def _generate_secret(self) -> str:
        """Generate a secure random secret."""
        import secrets
        return secrets.token_urlsafe(32)
    
    def _get_bool(self, key: str, default: bool = False) -> bool:
        """Get boolean environment variable."""
        value = os.getenv(f"{self.env_prefix}{key}")
        if value is None:
            return default
        return value.lower() in ("true", "1", "yes", "on")
    
    def _get_int(self, key: str, default: int = 0) -> int:
        """Get integer environment variable."""
        value = os.getenv(f"{self.env_prefix}{key}")
        if value is None:
            return default
        try:
            return int(value)
        except ValueError:
            return default
    
    def _get_list(self, key: str, default: List[str] = None) -> List[str]:
        """Get list environment variable (comma-separated)."""
        if default is None:
            default = []
        value = os.getenv(f"{self.env_prefix}{key}")
        if value is None:
            return default
        return [item.strip() for item in value.split(",") if item.strip()]