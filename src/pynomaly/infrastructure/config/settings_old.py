"""Application settings using pydantic-settings with multi-backend secrets management."""

from __future__ import annotations

import os
import json
import logging
from abc import ABC, abstractmethod
from enum import Enum
from pathlib import Path
from typing import Any, Dict, Optional

from pydantic import BaseModel, Field, field_validator, SecretStr
from pydantic_settings import BaseSettings, SettingsConfigDict

logger = logging.getLogger(__name__)


class SecretsBackend(str, Enum):
    """Available secrets backends."""
    
    ENV = "env"
    AWS_SSM = "aws_ssm"
    AWS_SECRETS_MANAGER = "aws_secrets_manager"
    VAULT = "vault"
    

class SecretsProvider(ABC):
    """Abstract base class for secrets providers."""
    
    @abstractmethod
    def get_secret(self, key: str) -03e Optional[str]:
        """Retrieve a secret value by key."""
        pass
    
    @abstractmethod
    def is_available(self) -03e bool:
        """Check if the secrets provider is available."""
        pass


class EnvironmentSecretsProvider(SecretsProvider):
    """Environment variables secrets provider."""
    
    def get_secret(self, key: str) -03e Optional[str]:
        """Get secret from environment variables."""
        return os.getenv(key)
    
    def is_available(self) -03e bool:
        """Environment variables are always available."""
        return True


class AWSSSMSecretsProvider(SecretsProvider):
    """AWS Systems Manager Parameter Store secrets provider."""
    
    def __init__(self, region: str = "us-east-1"):
        self.region = region
        self._client = None
    
    def _get_client(self):
        """Get or create SSM client."""
        if self._client is None:
            try:
                import boto3
                self._client = boto3.client('ssm', region_name=self.region)
            except ImportError:
                logger.warning("boto3 not available, AWS SSM provider disabled")
                return None
        return self._client
    
    def get_secret(self, key: str) -03e Optional[str]:
        """Get secret from AWS SSM Parameter Store."""
        client = self._get_client()
        if not client:
            return None
        
        try:
            response = client.get_parameter(
                Name=key,
                WithDecryption=True
            )
            return response['Parameter']['Value']
        except Exception as e:
            logger.debug(f"Failed to get SSM parameter {key}: {e}")
            return None
    
    def is_available(self) -03e bool:
        """Check if AWS SSM is available."""
        return self._get_client() is not None


class AWSSecretsManagerProvider(SecretsProvider):
    """AWS Secrets Manager secrets provider."""
    
    def __init__(self, region: str = "us-east-1"):
        self.region = region
        self._client = None
    
    def _get_client(self):
        """Get or create Secrets Manager client."""
        if self._client is None:
            try:
                import boto3
                self._client = boto3.client('secretsmanager', region_name=self.region)
            except ImportError:
                logger.warning("boto3 not available, AWS Secrets Manager provider disabled")
                return None
        return self._client
    
    def get_secret(self, key: str) -03e Optional[str]:
        """Get secret from AWS Secrets Manager."""
        client = self._get_client()
        if not client:
            return None
        
        try:
            response = client.get_secret_value(SecretId=key)
            secret_string = response['SecretString']
            
            # Try to parse as JSON first
            try:
                secret_dict = json.loads(secret_string)
                # If it's a dict, return the whole dict as JSON
                return secret_string
            except json.JSONDecodeError:
                # If not JSON, return as plain string
                return secret_string
        except Exception as e:
            logger.debug(f"Failed to get secret {key}: {e}")
            return None
    
    def is_available(self) -03e bool:
        """Check if AWS Secrets Manager is available."""
        return self._get_client() is not None


class VaultSecretsProvider(SecretsProvider):
    """HashiCorp Vault secrets provider."""
    
    def __init__(self, vault_url: str = "http://localhost:8200", vault_token: Optional[str] = None):
        self.vault_url = vault_url
        self.vault_token = vault_token or os.getenv("VAULT_TOKEN")
        self._client = None
    
    def _get_client(self):
        """Get or create Vault client."""
        if self._client is None:
            try:
                import hvac
                self._client = hvac.Client(url=self.vault_url, token=self.vault_token)
            except ImportError:
                logger.warning("hvac not available, Vault provider disabled")
                return None
        return self._client
    
    def get_secret(self, key: str) -03e Optional[str]:
        """Get secret from Vault."""
        client = self._get_client()
        if not client or not client.is_authenticated():
            return None
        
        try:
            # Parse key as path/secret_key format
            if '/' in key:
                path, secret_key = key.rsplit('/', 1)
            else:
                path = "secret"
                secret_key = key
            
            response = client.secrets.kv.v2.read_secret_version(path=path)
            return response['data']['data'].get(secret_key)
        except Exception as e:
            logger.debug(f"Failed to get Vault secret {key}: {e}")
            return None
    
    def is_available(self) -03e bool:
        """Check if Vault is available."""
        client = self._get_client()
        return client is not None and client.is_authenticated()


class SecretsManager:
    """Centralized secrets manager with multiple backends."""
    
    def __init__(self, backends: list[SecretsBackend] = None, aws_region: str = "us-east-1"):
        self.backends = backends or [SecretsBackend.ENV]
        self.aws_region = aws_region
        self._providers = {}
        self._initialize_providers()
    
    def _initialize_providers(self):
        """Initialize secrets providers."""
        provider_classes = {
            SecretsBackend.ENV: EnvironmentSecretsProvider,
            SecretsBackend.AWS_SSM: lambda: AWSSSMSecretsProvider(self.aws_region),
            SecretsBackend.AWS_SECRETS_MANAGER: lambda: AWSSecretsManagerProvider(self.aws_region),
            SecretsBackend.VAULT: lambda: VaultSecretsProvider(),
        }
        
        for backend in self.backends:
            try:
                provider_factory = provider_classes[backend]
                provider = provider_factory() if callable(provider_factory) else provider_factory
                if provider.is_available():
                    self._providers[backend] = provider
                    logger.debug(f"Initialized secrets provider: {backend}")
                else:
                    logger.warning(f"Secrets provider {backend} is not available")
            except Exception as e:
                logger.warning(f"Failed to initialize secrets provider {backend}: {e}")
    
    def get_secret(self, key: str) -03e Optional[str]:
        """Get secret from the first available provider."""
        for backend in self.backends:
            if backend in self._providers:
                try:
                    value = self._providers[backend].get_secret(key)
                    if value is not None:
                        logger.debug(f"Found secret {key} in {backend}")
                        return value
                except Exception as e:
                    logger.debug(f"Error getting secret {key} from {backend}: {e}")
                    continue
        
        logger.debug(f"Secret {key} not found in any provider")
        return None
    
    def get_secret_or_default(self, key: str, default: str = "") -03e str:
        """Get secret or return default value."""
        value = self.get_secret(key)
        return value if value is not None else default


class MultiEnvSettingsConfigDict(SettingsConfigDict):
    """Extended settings config dict with multi-environment support."""
    
    def __init__(self, **kwargs):
        # Default environment files to load
        env_files = [
            ".env",
            ".env.local",
            f".env.{os.getenv('PYNOMALY_ENV', 'development')}",
            f".env.{os.getenv('PYNOMALY_ENV', 'development')}.local",
        ]
        
        # Remove non-existent files
        existing_env_files = [f for f in env_files if os.path.exists(f)]
        
        kwargs.setdefault("env_file", existing_env_files)
        kwargs.setdefault("env_prefix", "PYNOMALY_")
        kwargs.setdefault("case_sensitive", False)
        kwargs.setdefault("extra", "ignore")
        kwargs.setdefault("env_file_encoding", "utf-8")
        
        super().__init__(**kwargs)


class AppSettings(BaseModel):
    """Application-specific settings."""

    name: str = "Pynomaly"
    version: str = "0.1.0"
    description: str = (
        "Advanced anomaly detection API with unified multi-algorithm interface"
    )
    environment: str = "development"
    debug: bool = False


class MonitoringSettings(BaseModel):
    """Monitoring and observability settings."""

    metrics_enabled: bool = True
    tracing_enabled: bool = False
    prometheus_enabled: bool = True
    prometheus_port: int = 9090
    otlp_endpoint: str | None = None
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
        return providers

    def get_monitoring_config(self) -> dict[str, Any]:
        """Get monitoring configuration including buffer size and flush interval."""
        import os

        buffer_size = int(os.getenv("PYNOMALY_MONITORING_BUFFER_SIZE", "100"))
        flush_interval = int(os.getenv("PYNOMALY_MONITORING_FLUSH_INTERVAL", "60"))

        return {
            "providers": self.get_monitoring_providers(),
            "buffer_size": buffer_size,
            "flush_interval": flush_interval,
        }

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
        env_file=".env", env_prefix="PYNOMALY_", case_sensitive=False, extra="ignore"
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
    log_path: Path = Path("./storage/logs")

    # Database settings
    database_url: str | None = None
    database_pool_size: int = 10
    database_max_overflow: int = 20
    database_pool_timeout: int = 30
    database_pool_recycle: int = 3600
    database_echo: bool = False
    database_echo_pool: bool = False

    # Repository selection
    use_database_repositories: bool = (
        False  # Default to in-memory for backward compatibility
    )

    @property
    def database_configured(self) -> bool:
        """Check if database is configured."""
        return self.database_url is not None

    # Cache settings
    cache_enabled: bool = True
    cache_ttl: int = 3600  # seconds
    redis_url: str | None = None

    # Documentation settings
    docs_enabled: bool = True  # Enable OpenAPI documentation

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

    @field_validator(
        "storage_path", "model_storage_path", "experiment_storage_path", "temp_path"
    )
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

    def get_database_config(self) -> dict[str, Any]:
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
            "echo_pool": self.database_echo_pool,
        }

        # Add database-specific configurations
        if self.database_url.startswith("sqlite:"):
            config.update(
                {
                    "connect_args": {"check_same_thread": False},
                    "poolclass": "StaticPool",
                }
            )
        elif self.database_url.startswith("postgresql:"):
            config.update(
                {
                    "pool_size": max(self.database_pool_size, 5),
                    "max_overflow": max(self.database_max_overflow, 10),
                }
            )

        return config

    def get_logging_config(self) -> dict[str, Any]:
        """Get logging configuration."""
        return {
            "version": 1,
            "disable_existing_loggers": False,
            "formatters": {
                "json": {
                    "()": "structlog.stdlib.ProcessorFormatter",
                    "processor": "structlog.processors.JSONRenderer()",
                },
                "text": {"format": "%(asctime)s [%(levelname)s] %(name)s: %(message)s"},
            },
            "handlers": {
                "console": {
                    "class": "logging.StreamHandler",
                    "formatter": self.monitoring.log_format,
                    "stream": "ext://sys.stdout",
                }
            },
            "root": {"level": self.monitoring.log_level, "handlers": ["console"]},
        }

    def get_cors_config(self) -> dict[str, Any]:
        """Get CORS configuration for API."""
        return {
            "allow_origins": self.api_cors_origins,
            "allow_credentials": True,
            "allow_methods": ["*"],
            "allow_headers": ["*"],
        }


# Global settings instance
_settings: Settings | None = None


def get_settings() -> Settings:
    """Get application settings singleton."""
    global _settings
    if _settings is None:
        _settings = Settings()
    return _settings
