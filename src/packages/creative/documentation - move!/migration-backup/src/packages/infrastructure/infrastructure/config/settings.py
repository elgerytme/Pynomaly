"""Application settings using pydantic."""

from __future__ import annotations

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict

from .api_settings import APISettings
from .app_settings import AppSettings
from .database_settings import DatabaseSettings
from .ml_settings import MLSettings
from .monitoring_settings import MonitoringSettings
from .resilience_settings import ResilienceSettings
from .security_auth_settings import SecurityAuthSettings
from .storage_settings import StorageSettings
from ..messaging.config.messaging_settings import MessagingSettings


class SecuritySettings(SecurityAuthSettings):
    """Extended security settings including advanced features."""

    # Input sanitization
    sanitization_level: str = "moderate"  # strict, moderate, permissive
    max_input_length: int = 10000
    allow_html: bool = False

    # Encryption
    encryption_algorithm: str = "fernet"  # fernet, aes_gcm, aes_cbc
    encryption_key_length: int = 32
    enable_key_rotation: bool = True
    key_rotation_days: int = 90

    # Audit logging
    enable_audit_logging: bool = True
    enable_compliance_logging: bool = False
    audit_retention_days: int = 2555  # 7 years

    # Security monitoring
    enable_security_monitoring: bool = True
    threat_detection_enabled: bool = True

    # Rate limiting
    enable_advanced_rate_limiting: bool = True
    brute_force_max_attempts: int = 5
    brute_force_time_window: int = 300  # 5 minutes

    # Headers and CORS
    security_headers_enabled: bool = True
    csp_enabled: bool = True
    hsts_enabled: bool = True

    # Session management
    session_timeout: int = 3600  # 1 hour
    max_concurrent_sessions: int = 5

    def get_monitoring_providers(self) -> list[str]:
        """Get configured monitoring providers."""
        providers = []
        if self.enable_audit_logging:
            providers.append("audit")
        if self.enable_security_monitoring:
            providers.append("security")
        if self.threat_detection_enabled:
            providers.append("threat_detection")
        return providers


class Settings(BaseSettings):
    """Application settings with environment variable support."""

    model_config = SettingsConfigDict(
        env_file=".env", env_prefix="PYNOMALY_", case_sensitive=False, extra="ignore"
    )

    # Configuration modules
    app: AppSettings = Field(default_factory=AppSettings)
    api: APISettings = Field(default_factory=APISettings)
    database: DatabaseSettings = Field(default_factory=DatabaseSettings)
    storage: StorageSettings = Field(default_factory=StorageSettings)
    ml: MLSettings = Field(default_factory=MLSettings)
    monitoring: MonitoringSettings = Field(default_factory=MonitoringSettings)
    resilience: ResilienceSettings = Field(default_factory=ResilienceSettings)
    security: SecuritySettings = Field(default_factory=SecuritySettings)
    messaging: MessagingSettings = Field(default_factory=MessagingSettings)

    @property
    def is_production(self) -> bool:
        """Check if running in production mode."""
        return self.app.is_production

    def get_database_config(self):
        """Get database configuration."""
        return self.database.get_database_config(self.app.debug)

    def get_logging_config(self):
        """Get logging configuration."""
        return self.monitoring.get_logging_config()

    def get_cors_config(self):
        """Get CORS configuration for API."""
        return self.api.get_cors_config()

    def get_redis_url(self) -> str:
        """Get Redis URL for cache configuration."""
        return self.storage.redis_url or "redis://localhost:6379/0"

    @property
    def cache_ttl(self) -> int:
        """Get cache TTL setting."""
        return self.storage.cache_ttl

    def get_messaging_config(self) -> dict[str, any]:
        """Get messaging configuration."""
        return self.messaging.get_queue_config()

    def get_queue_url(self) -> str:
        """Get message queue URL."""
        return self.messaging.get_redis_url()


# Global settings instance
_settings: Settings | None = None


def get_settings() -> Settings:
    """Get application settings singleton."""
    global _settings
    if _settings is None:
        _settings = Settings()
    return _settings
