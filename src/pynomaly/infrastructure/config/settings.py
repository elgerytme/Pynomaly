"""Application settings using pydantic."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Optional

from pydantic import BaseModel, Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class AppSettings(BaseModel):
    """Application-specific settings."""
    name: str = "Pynomaly"
    version: str = "0.1.0"
    environment: str = "development"
    debug: bool = False


class MonitoringSettings(BaseModel):
    """Monitoring and observability settings."""
    metrics_enabled: bool = True
    tracing_enabled: bool = False
    prometheus_enabled: bool = True
    prometheus_port: int = 9090
    otlp_endpoint: Optional[str] = None
    otlp_insecure: bool = True
    log_level: str = "INFO"
    log_format: str = "json"
    host_name: str = "localhost"
    instrument_fastapi: bool = True
    instrument_sqlalchemy: bool = True


class SecuritySettings(BaseModel):
    """Security and audit settings."""
    
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
    
    @field_validator("sanitization_level")
    @classmethod
    def validate_sanitization_level(cls, v: str) -> str:
        """Validate sanitization level."""
        valid_levels = ["strict", "moderate", "permissive"]
        if v not in valid_levels:
            raise ValueError(f"Sanitization level must be one of: {valid_levels}")
        return v
    
    @field_validator("encryption_algorithm")
    @classmethod
    def validate_encryption_algorithm(cls, v: str) -> str:
        """Validate encryption algorithm."""
        valid_algorithms = ["fernet", "aes_gcm", "aes_cbc"]
        if v not in valid_algorithms:
            raise ValueError(f"Encryption algorithm must be one of: {valid_algorithms}")
        return v


class Settings(BaseSettings):
    """Application settings with environment variable support."""
    
    model_config = SettingsConfigDict(
        env_file=".env",
        env_prefix="PYNOMALY_",
        case_sensitive=False,
        extra="ignore"
    )
    
    # Application settings
    app: AppSettings = Field(default_factory=AppSettings)
    
    # API settings
    api_host: str = "0.0.0.0"
    api_port: int = 8000
    api_workers: int = 1
    api_cors_origins: list[str] = ["*"]
    api_rate_limit: int = 100  # requests per minute
    
    # Storage settings
    storage_path: Path = Path("./storage")
    model_storage_path: Path = Path("./storage/models")
    experiment_storage_path: Path = Path("./storage/experiments")
    temp_path: Path = Path("./storage/temp")
    
    # Database settings
    database_url: Optional[str] = None
    database_pool_size: int = 10
    database_max_overflow: int = 20
    database_pool_timeout: int = 30
    database_pool_recycle: int = 3600
    database_echo: bool = False
    database_echo_pool: bool = False
    
    # Repository selection
    use_database_repositories: bool = False  # Default to in-memory for backward compatibility
    
    @property
    def database_configured(self) -> bool:
        """Check if database is configured."""
        return self.database_url is not None
    
    # Cache settings
    cache_enabled: bool = True
    cache_ttl: int = 3600  # seconds
    redis_url: Optional[str] = None
    
    # Security settings
    secret_key: str = Field(default="change-me-in-production")
    auth_enabled: bool = False
    jwt_algorithm: str = "HS256"
    jwt_expiration: int = 3600  # seconds
    
    # Monitoring settings
    monitoring: MonitoringSettings = Field(default_factory=MonitoringSettings)
    
    # Security settings
    security: SecuritySettings = Field(default_factory=SecuritySettings)
    
    # Algorithm settings
    default_contamination_rate: float = 0.1
    max_parallel_detectors: int = 4
    detector_timeout: int = 300  # seconds
    
    # Data processing settings
    max_dataset_size_mb: int = 1000
    chunk_size: int = 10000
    max_features: int = 1000
    
    # ML settings
    random_seed: int = 42
    gpu_enabled: bool = False
    gpu_memory_fraction: float = 0.8
    
    # Streaming settings
    kafka_bootstrap_servers: str = "localhost:9092"
    kafka_topic_prefix: str = "pynomaly"
    streaming_enabled: bool = False
    max_streaming_sessions: int = 10
    
    @field_validator("storage_path", "model_storage_path", "experiment_storage_path", "temp_path")
    @classmethod
    def create_directories(cls, v: Path) -> Path:
        """Ensure directories exist."""
        v.mkdir(parents=True, exist_ok=True)
        return v
    
    @field_validator("default_contamination_rate")
    @classmethod
    def validate_contamination_rate(cls, v: float) -> float:
        """Validate contamination rate is in valid range."""
        if not 0 <= v <= 1:
            raise ValueError("Contamination rate must be between 0 and 1")
        return v
    
    @property
    def is_production(self) -> bool:
        """Check if running in production mode."""
        return not self.app.debug
    
    def get_database_config(self) -> Dict[str, Any]:
        """Get database configuration."""
        if not self.database_url:
            return {}
        
        config = {
            "url": self.database_url,
            "pool_size": self.database_pool_size,
            "max_overflow": self.database_max_overflow,
            "pool_timeout": self.database_pool_timeout,
            "pool_recycle": self.database_pool_recycle,
            "pool_pre_ping": True,
            "echo": self.database_echo or self.app.debug,
            "echo_pool": self.database_echo_pool
        }
        
        # Add database-specific configurations
        if self.database_url.startswith('sqlite:'):
            config.update({
                "connect_args": {"check_same_thread": False},
                "poolclass": "StaticPool"
            })
        elif self.database_url.startswith('postgresql:'):
            config.update({
                "pool_size": max(self.database_pool_size, 5),
                "max_overflow": max(self.database_max_overflow, 10)
            })
        
        return config
    
    def get_logging_config(self) -> Dict[str, Any]:
        """Get logging configuration."""
        return {
            "version": 1,
            "disable_existing_loggers": False,
            "formatters": {
                "json": {
                    "()": "structlog.stdlib.ProcessorFormatter",
                    "processor": "structlog.processors.JSONRenderer()"
                },
                "text": {
                    "format": "%(asctime)s [%(levelname)s] %(name)s: %(message)s"
                }
            },
            "handlers": {
                "console": {
                    "class": "logging.StreamHandler",
                    "formatter": self.monitoring.log_format,
                    "stream": "ext://sys.stdout"
                }
            },
            "root": {
                "level": self.monitoring.log_level,
                "handlers": ["console"]
            }
        }
    
    def get_cors_config(self) -> Dict[str, Any]:
        """Get CORS configuration for API."""
        return {
            "allow_origins": self.api_cors_origins,
            "allow_credentials": True,
            "allow_methods": ["*"],
            "allow_headers": ["*"],
        }