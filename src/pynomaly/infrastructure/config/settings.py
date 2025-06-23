"""Application settings using pydantic."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Optional

from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Application settings with environment variable support."""
    
    model_config = SettingsConfigDict(
        env_file=".env",
        env_prefix="PYNOMALY_",
        case_sensitive=False,
        extra="ignore"
    )
    
    # Application settings
    app_name: str = "Pynomaly"
    version: str = "0.1.0"
    debug: bool = False
    
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
    
    # Database settings (for future use)
    database_url: Optional[str] = None
    database_pool_size: int = 5
    
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
    metrics_enabled: bool = True
    tracing_enabled: bool = False
    log_level: str = "INFO"
    log_format: str = "json"  # json or text
    
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
        return not self.debug
    
    def get_database_config(self) -> Dict[str, Any]:
        """Get database configuration."""
        if not self.database_url:
            return {}
        
        return {
            "url": self.database_url,
            "pool_size": self.database_pool_size,
            "pool_pre_ping": True,
            "echo": self.debug
        }
    
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
                    "formatter": self.log_format,
                    "stream": "ext://sys.stdout"
                }
            },
            "root": {
                "level": self.log_level,
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