"""Standardized configuration for data quality package using the shared framework."""

import os
from typing import Dict, Any, Optional
from pathlib import Path

# Import from shared infrastructure
import sys
sys.path.append(str(Path(__file__).parents[6] / "shared" / "infrastructure"))

try:
    from config.base_settings import (
        BasePackageSettings, 
        DatabaseSettings,
        LoggingSettings,
        APISettings,
        MonitoringSettings,
        get_database_settings,
        get_logging_settings,
        get_api_settings,
        get_monitoring_settings,
        PYDANTIC_AVAILABLE
    )
except ImportError:
    # Fallback if shared infrastructure is not available
    PYDANTIC_AVAILABLE = False
    
    class BasePackageSettings:
        def __init__(self, package_name: str, env_prefix: str = ""):
            self.package_name = package_name
            self.env_prefix = env_prefix
        
        @classmethod
        def load(cls):
            return cls("quality")


if PYDANTIC_AVAILABLE:
    from pydantic import Field, validator
    from pydantic_settings import BaseSettings
    
    class ValidationEngineSettings(BaseSettings):
        """Configuration for validation engine using Pydantic."""
        
        # Performance settings
        enable_parallel_processing: bool = Field(default=True, env="QUALITY_VALIDATION_PARALLEL")
        max_workers: int = Field(default=4, env="QUALITY_VALIDATION_MAX_WORKERS")
        timeout_seconds: int = Field(default=300, env="QUALITY_VALIDATION_TIMEOUT")
        
        # Caching settings
        enable_caching: bool = Field(default=True, env="QUALITY_VALIDATION_CACHING")
        cache_size: int = Field(default=1000, env="QUALITY_VALIDATION_CACHE_SIZE")
        cache_ttl_seconds: int = Field(default=3600, env="QUALITY_VALIDATION_CACHE_TTL")
        
        # Sampling settings
        enable_sampling: bool = Field(default=False, env="QUALITY_VALIDATION_SAMPLING")
        sample_size: int = Field(default=10000, env="QUALITY_VALIDATION_SAMPLE_SIZE")
        sample_random_state: int = Field(default=42, env="QUALITY_VALIDATION_RANDOM_STATE")
        
        # Quality thresholds
        warning_threshold: float = Field(default=0.8, env="QUALITY_WARNING_THRESHOLD")
        error_threshold: float = Field(default=0.6, env="QUALITY_ERROR_THRESHOLD")
        critical_threshold: float = Field(default=0.4, env="QUALITY_CRITICAL_THRESHOLD")
        
        # Rule execution settings
        max_rule_execution_time_ms: float = Field(default=5000.0, env="QUALITY_MAX_RULE_EXECUTION_TIME")
        enable_rule_timeout: bool = Field(default=True, env="QUALITY_ENABLE_RULE_TIMEOUT")
        
        # Data source settings
        max_data_size_mb: int = Field(default=100, env="QUALITY_MAX_DATA_SIZE_MB")
        enable_data_streaming: bool = Field(default=True, env="QUALITY_ENABLE_DATA_STREAMING")
        streaming_chunk_size: int = Field(default=1000, env="QUALITY_STREAMING_CHUNK_SIZE")
        
        # Additional configuration
        enable_metrics: bool = Field(default=True, env="QUALITY_ENABLE_METRICS")
        enable_detailed_logging: bool = Field(default=False, env="QUALITY_DETAILED_LOGGING")
        enable_profiling: bool = Field(default=False, env="QUALITY_ENABLE_PROFILING")
        
        class Config:
            env_prefix = "QUALITY_"
            case_sensitive = False
            
        @validator('max_workers')
        def validate_max_workers(cls, v):
            if v < 1:
                raise ValueError('Max workers must be at least 1')
            if v > 32:
                raise ValueError('Max workers should not exceed 32')
            return v
            
        @validator('timeout_seconds')
        def validate_timeout(cls, v):
            if v < 1:
                raise ValueError('Timeout must be at least 1 second')
            return v
            
        @validator('cache_size')
        def validate_cache_size(cls, v):
            if v < 0:
                raise ValueError('Cache size cannot be negative')
            return v
            
        @validator('sample_size')
        def validate_sample_size(cls, v):
            if v < 1:
                raise ValueError('Sample size must be at least 1')
            return v
            
        @validator('warning_threshold', 'error_threshold', 'critical_threshold')
        def validate_thresholds(cls, v):
            if not 0.0 <= v <= 1.0:
                raise ValueError('Thresholds must be between 0.0 and 1.0')
            return v

    class ProfilingSettings(BaseSettings):
        """Configuration for data profiling features."""
        
        # Profile computation settings
        enable_basic_stats: bool = Field(default=True, env="QUALITY_PROFILE_BASIC_STATS")
        enable_advanced_stats: bool = Field(default=False, env="QUALITY_PROFILE_ADVANCED_STATS")
        enable_distribution_analysis: bool = Field(default=True, env="QUALITY_PROFILE_DISTRIBUTION")
        enable_correlation_analysis: bool = Field(default=False, env="QUALITY_PROFILE_CORRELATION")
        
        # Performance settings for profiling
        max_categorical_values: int = Field(default=50, env="QUALITY_PROFILE_MAX_CATEGORICAL")
        histogram_bins: int = Field(default=20, env="QUALITY_PROFILE_HISTOGRAM_BINS")
        percentiles: list = Field(default=[0.25, 0.5, 0.75, 0.95, 0.99], env="QUALITY_PROFILE_PERCENTILES")
        
        # Profile storage settings
        store_profiles: bool = Field(default=True, env="QUALITY_PROFILE_STORE")
        profile_retention_days: int = Field(default=30, env="QUALITY_PROFILE_RETENTION_DAYS")
        
        class Config:
            env_prefix = "QUALITY_"
            case_sensitive = False
            
        @validator('max_categorical_values')
        def validate_max_categorical(cls, v):
            if v < 1:
                raise ValueError('Max categorical values must be at least 1')
            return v
            
        @validator('histogram_bins')
        def validate_histogram_bins(cls, v):
            if v < 1:
                raise ValueError('Histogram bins must be at least 1')
            return v

else:
    # Fallback implementations without Pydantic
    class ValidationEngineSettings:
        def __init__(self):
            self.enable_parallel_processing = os.getenv("QUALITY_VALIDATION_PARALLEL", "true").lower() == "true"
            self.max_workers = int(os.getenv("QUALITY_VALIDATION_MAX_WORKERS", "4"))
            self.timeout_seconds = int(os.getenv("QUALITY_VALIDATION_TIMEOUT", "300"))
            self.enable_caching = os.getenv("QUALITY_VALIDATION_CACHING", "true").lower() == "true"
            self.cache_size = int(os.getenv("QUALITY_VALIDATION_CACHE_SIZE", "1000"))
            self.cache_ttl_seconds = int(os.getenv("QUALITY_VALIDATION_CACHE_TTL", "3600"))
            self.enable_sampling = os.getenv("QUALITY_VALIDATION_SAMPLING", "false").lower() == "true"
            self.sample_size = int(os.getenv("QUALITY_VALIDATION_SAMPLE_SIZE", "10000"))
            self.sample_random_state = int(os.getenv("QUALITY_VALIDATION_RANDOM_STATE", "42"))
            self.warning_threshold = float(os.getenv("QUALITY_WARNING_THRESHOLD", "0.8"))
            self.error_threshold = float(os.getenv("QUALITY_ERROR_THRESHOLD", "0.6"))
            self.critical_threshold = float(os.getenv("QUALITY_CRITICAL_THRESHOLD", "0.4"))
            self.max_rule_execution_time_ms = float(os.getenv("QUALITY_MAX_RULE_EXECUTION_TIME", "5000"))
            self.enable_rule_timeout = os.getenv("QUALITY_ENABLE_RULE_TIMEOUT", "true").lower() == "true"
            self.max_data_size_mb = int(os.getenv("QUALITY_MAX_DATA_SIZE_MB", "100"))
            self.enable_data_streaming = os.getenv("QUALITY_ENABLE_DATA_STREAMING", "true").lower() == "true"
            self.streaming_chunk_size = int(os.getenv("QUALITY_STREAMING_CHUNK_SIZE", "1000"))
            self.enable_metrics = os.getenv("QUALITY_ENABLE_METRICS", "true").lower() == "true"
            self.enable_detailed_logging = os.getenv("QUALITY_DETAILED_LOGGING", "false").lower() == "true"
            self.enable_profiling = os.getenv("QUALITY_ENABLE_PROFILING", "false").lower() == "true"

    class ProfilingSettings:
        def __init__(self):
            self.enable_basic_stats = os.getenv("QUALITY_PROFILE_BASIC_STATS", "true").lower() == "true"
            self.enable_advanced_stats = os.getenv("QUALITY_PROFILE_ADVANCED_STATS", "false").lower() == "true"
            self.enable_distribution_analysis = os.getenv("QUALITY_PROFILE_DISTRIBUTION", "true").lower() == "true"
            self.enable_correlation_analysis = os.getenv("QUALITY_PROFILE_CORRELATION", "false").lower() == "true"
            self.max_categorical_values = int(os.getenv("QUALITY_PROFILE_MAX_CATEGORICAL", "50"))
            self.histogram_bins = int(os.getenv("QUALITY_PROFILE_HISTOGRAM_BINS", "20"))
            self.percentiles = [0.25, 0.5, 0.75, 0.95, 0.99]  # Could be made configurable
            self.store_profiles = os.getenv("QUALITY_PROFILE_STORE", "true").lower() == "true"
            self.profile_retention_days = int(os.getenv("QUALITY_PROFILE_RETENTION_DAYS", "30"))


class QualitySettings(BasePackageSettings):
    """Main configuration class for data quality package."""
    
    def __init__(self):
        super().__init__("quality", "QUALITY_")
        
        # Quality-specific settings
        self.validation_engine = ValidationEngineSettings()
        self.profiling = ProfilingSettings()
        
        # Security settings
        self.enable_encryption = os.getenv("QUALITY_ENABLE_ENCRYPTION", "true").lower() == "true"
        self.encryption_key = os.getenv("QUALITY_ENCRYPTION_KEY")
        
        # Data retention settings
        self.data_retention_days = int(os.getenv("QUALITY_DATA_RETENTION_DAYS", "90"))
        self.check_result_retention_days = int(os.getenv("QUALITY_CHECK_RESULT_RETENTION_DAYS", "365"))
        
        # Integration settings
        self.enable_webhooks = os.getenv("QUALITY_ENABLE_WEBHOOKS", "false").lower() == "true"
        self.webhook_timeout_seconds = int(os.getenv("QUALITY_WEBHOOK_TIMEOUT", "30"))
        self.webhook_retry_attempts = int(os.getenv("QUALITY_WEBHOOK_RETRY_ATTEMPTS", "3"))
        
        # Override database settings for quality-specific defaults
        self.database.database = os.getenv("QUALITY_DB_DATABASE", "quality_db")
        self.database.host = os.getenv("QUALITY_DB_HOST", "localhost")
        self.database.port = int(os.getenv("QUALITY_DB_PORT", "5432"))
        
        # Override API settings for quality-specific defaults  
        self.api.port = int(os.getenv("QUALITY_API_PORT", "8001"))
        
    @classmethod
    def load(cls) -> 'QualitySettings':
        """Load quality settings from environment and configuration files."""
        instance = cls()
        
        # Load from configuration files using the parent method
        config_data = instance.load_from_multiple_sources()
        if config_data:
            instance.update_from_dict(config_data)
            
        return instance
    
    def update_from_dict(self, data: Dict[str, Any]) -> None:
        """Update settings from dictionary data."""
        # Call parent method for standard settings
        super().update_from_dict(data)
        
        # Update quality-specific settings
        if "validation_engine" in data:
            validation_data = data["validation_engine"]
            for key, value in validation_data.items():
                if hasattr(self.validation_engine, key):
                    setattr(self.validation_engine, key, value)
        
        if "profiling" in data:
            profiling_data = data["profiling"]
            for key, value in profiling_data.items():
                if hasattr(self.profiling, key):
                    setattr(self.profiling, key, value)
        
        # Update other quality-specific settings
        for key in ["enable_encryption", "encryption_key", "data_retention_days", 
                   "check_result_retention_days", "enable_webhooks", 
                   "webhook_timeout_seconds", "webhook_retry_attempts"]:
            if key in data:
                setattr(self, key, data[key])
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert settings to dictionary."""
        result = super().to_dict()
        
        # Add quality-specific settings
        result.update({
            "validation_engine": self.validation_engine.__dict__,
            "profiling": self.profiling.__dict__,
            "enable_encryption": self.enable_encryption,
            "encryption_key": self.encryption_key,
            "data_retention_days": self.data_retention_days,
            "check_result_retention_days": self.check_result_retention_days,
            "enable_webhooks": self.enable_webhooks,
            "webhook_timeout_seconds": self.webhook_timeout_seconds,
            "webhook_retry_attempts": self.webhook_retry_attempts,
        })
        
        return result
    
    def validate_settings(self) -> None:
        """Validate configuration settings."""
        errors = []
        
        # Validate thresholds are in correct order
        if (self.validation_engine.warning_threshold <= self.validation_engine.error_threshold or
            self.validation_engine.error_threshold <= self.validation_engine.critical_threshold):
            errors.append("Quality thresholds must be in descending order: warning > error > critical")
        
        # Validate retention settings
        if self.data_retention_days < 1:
            errors.append("Data retention days must be at least 1")
            
        if self.check_result_retention_days < 1:
            errors.append("Check result retention days must be at least 1")
        
        # Validate webhook settings
        if self.enable_webhooks:
            if self.webhook_timeout_seconds < 1:
                errors.append("Webhook timeout must be at least 1 second")
            if self.webhook_retry_attempts < 0:
                errors.append("Webhook retry attempts cannot be negative")
        
        if errors:
            from shared.infrastructure.exceptions.base_exceptions import ConfigurationError
            raise ConfigurationError(
                message=f"Configuration validation failed: {'; '.join(errors)}",
                details={"validation_errors": errors}
            )


# Global settings instance
_settings: Optional[QualitySettings] = None


def get_quality_settings() -> QualitySettings:
    """Get the global quality settings instance."""
    global _settings
    if _settings is None:
        _settings = QualitySettings.load()
        _settings.validate_settings()
    return _settings


def reload_quality_settings() -> QualitySettings:
    """Reload settings from all sources."""
    global _settings
    _settings = QualitySettings.load()
    _settings.validate_settings()
    return _settings


def create_example_quality_config() -> None:
    """Create example configuration files for the quality package."""
    settings = QualitySettings()
    
    example_config = {
        "environment": "development",
        "debug": False,
        "version": "1.0.0",
        
        "database": {
            "host": "localhost",
            "port": 5432,
            "database": "quality_db",
            "username": "postgres",
            "password": "",
            "pool_size": 10,
            "async_mode": True
        },
        
        "logging": {
            "level": "INFO",
            "format": "json",
            "enable_structured_logging": True,
            "enable_performance_logging": True,
            "slow_query_threshold_ms": 1000.0
        },
        
        "api": {
            "host": "0.0.0.0",
            "port": 8001,
            "cors_origins": ["*"],
            "rate_limit_enabled": True,
            "requests_per_minute": 60
        },
        
        "monitoring": {
            "enable_metrics": True,
            "metrics_port": 9091,
            "enable_health_checks": True,
            "health_check_interval": 30
        },
        
        "validation_engine": {
            "enable_parallel_processing": True,
            "max_workers": 4,
            "timeout_seconds": 300,
            "enable_caching": True,
            "cache_size": 1000,
            "warning_threshold": 0.8,
            "error_threshold": 0.6,
            "critical_threshold": 0.4,
            "max_data_size_mb": 100
        },
        
        "profiling": {
            "enable_basic_stats": True,
            "enable_advanced_stats": False,
            "enable_distribution_analysis": True,
            "max_categorical_values": 50,
            "histogram_bins": 20,
            "store_profiles": True,
            "profile_retention_days": 30
        },
        
        "enable_encryption": True,
        "data_retention_days": 90,
        "check_result_retention_days": 365,
        "enable_webhooks": False,
        "webhook_timeout_seconds": 30,
        "webhook_retry_attempts": 3
    }
    
    # Write example YAML file
    try:
        import yaml
        with open("quality.example.yml", "w", encoding="utf-8") as f:
            yaml.dump(example_config, f, default_flow_style=False, indent=2)
        print("✅ Created example configuration: quality.example.yml")
    except ImportError:
        import json
        with open("quality.example.json", "w", encoding="utf-8") as f:
            json.dump(example_config, f, indent=2)
        print("✅ Created example configuration: quality.example.json")
    
    # Create example environment file
    example_env = """# Data Quality Environment Variables

# Application
QUALITY_ENVIRONMENT=development
QUALITY_DEBUG=false
QUALITY_VERSION=1.0.0
QUALITY_SECRET_KEY=dev-secret-key-change-in-production

# Database
QUALITY_DB_HOST=localhost
QUALITY_DB_PORT=5432
QUALITY_DB_DATABASE=quality_db
QUALITY_DB_USERNAME=postgres
QUALITY_DB_PASSWORD=
QUALITY_DB_POOL_SIZE=10
QUALITY_DB_ASYNC_MODE=true

# Logging  
QUALITY_LOG_LEVEL=INFO
QUALITY_LOG_FORMAT=json
QUALITY_LOG_STRUCTURED=true
QUALITY_LOG_PERFORMANCE=true
QUALITY_LOG_SLOW_QUERY_THRESHOLD_MS=1000

# API
QUALITY_API_HOST=0.0.0.0
QUALITY_API_PORT=8001
QUALITY_API_CORS_ORIGINS=*
QUALITY_API_RATE_LIMIT_ENABLED=true
QUALITY_API_REQUESTS_PER_MINUTE=60

# Monitoring
QUALITY_MONITORING_ENABLE_METRICS=true
QUALITY_MONITORING_METRICS_PORT=9091
QUALITY_MONITORING_ENABLE_HEALTH_CHECKS=true
QUALITY_MONITORING_HEALTH_CHECK_INTERVAL=30

# Validation Engine
QUALITY_VALIDATION_PARALLEL=true
QUALITY_VALIDATION_MAX_WORKERS=4
QUALITY_VALIDATION_TIMEOUT=300
QUALITY_VALIDATION_CACHING=true
QUALITY_VALIDATION_CACHE_SIZE=1000
QUALITY_WARNING_THRESHOLD=0.8
QUALITY_ERROR_THRESHOLD=0.6
QUALITY_CRITICAL_THRESHOLD=0.4
QUALITY_MAX_DATA_SIZE_MB=100

# Profiling
QUALITY_PROFILE_BASIC_STATS=true
QUALITY_PROFILE_ADVANCED_STATS=false
QUALITY_PROFILE_DISTRIBUTION=true
QUALITY_PROFILE_MAX_CATEGORICAL=50
QUALITY_PROFILE_HISTOGRAM_BINS=20
QUALITY_PROFILE_STORE=true
QUALITY_PROFILE_RETENTION_DAYS=30

# Security & Retention
QUALITY_ENABLE_ENCRYPTION=true
QUALITY_DATA_RETENTION_DAYS=90
QUALITY_CHECK_RESULT_RETENTION_DAYS=365

# Webhooks (optional)
QUALITY_ENABLE_WEBHOOKS=false
QUALITY_WEBHOOK_TIMEOUT=30
QUALITY_WEBHOOK_RETRY_ATTEMPTS=3
"""
    
    with open("quality.example.env", "w", encoding="utf-8") as f:
        f.write(example_env)
    print("✅ Created example environment file: quality.example.env")


if __name__ == "__main__":
    # Create example config files when run directly
    create_example_quality_config()
    
    # Print current settings
    settings = get_quality_settings()
    print(f"\n📋 Current Quality Settings:")
    print(f"   Environment: {settings.environment}")
    print(f"   Debug: {settings.debug}")
    print(f"   Database: {settings.database.url}")
    print(f"   API Port: {settings.api.port}")
    print(f"   Max Workers: {settings.validation_engine.max_workers}")
    print(f"   Warning Threshold: {settings.validation_engine.warning_threshold}")
    print(f"   Pydantic Available: {PYDANTIC_AVAILABLE}")


__all__ = [
    "QualitySettings",
    "ValidationEngineSettings", 
    "ProfilingSettings",
    "get_quality_settings",
    "reload_quality_settings",
    "create_example_quality_config",
    "PYDANTIC_AVAILABLE"
]