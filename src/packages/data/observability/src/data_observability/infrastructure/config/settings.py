"""Configuration settings for data observability."""

from typing import Optional

from pydantic import Field, validator
from pydantic_settings import BaseSettings


class DatabaseSettings(BaseSettings):
    """Database configuration settings."""
    
    host: str = Field(default="localhost", description="Database host")
    port: int = Field(default=5432, description="Database port")
    name: str = Field(default="data_observability", description="Database name")
    user: str = Field(default="postgres", description="Database user")
    password: str = Field(default="", description="Database password")
    
    @property
    def url(self) -> str:
        """Get database connection URL."""
        return f"postgresql://{self.user}:{self.password}@{self.host}:{self.port}/{self.name}"
    
    @property
    def async_url(self) -> str:
        """Get async database connection URL."""
        return f"postgresql+asyncpg://{self.user}:{self.password}@{self.host}:{self.port}/{self.name}"
    
    class Config:
        env_prefix = "DB_"


class SecuritySettings(BaseSettings):
    """Security configuration settings."""
    
    secret_key: str = Field(
        default="your-secret-key-change-in-production",
        description="Secret key for JWT tokens"
    )
    algorithm: str = Field(default="HS256", description="JWT algorithm")
    access_token_expire_minutes: int = Field(
        default=30, 
        description="Access token expiration time in minutes"
    )
    
    @validator("secret_key")
    def validate_secret_key(cls, v: str) -> str:
        """Validate secret key."""
        if len(v) < 32:
            raise ValueError("Secret key must be at least 32 characters long")
        return v
    
    class Config:
        env_prefix = "SECURITY_"


class APISettings(BaseSettings):
    """API configuration settings."""
    
    title: str = Field(default="Data Observability API", description="API title")
    description: str = Field(
        default="REST API for data observability operations", 
        description="API description"
    )
    version: str = Field(default="1.0.0", description="API version")
    host: str = Field(default="0.0.0.0", description="API host")
    port: int = Field(default=8000, description="API port")
    debug: bool = Field(default=False, description="Debug mode")
    reload: bool = Field(default=False, description="Auto-reload on changes")
    
    class Config:
        env_prefix = "API_"


class WorkerSettings(BaseSettings):
    """Worker configuration settings."""
    
    health_check_interval: int = Field(
        default=300, 
        description="Health check interval in seconds"
    )
    quality_check_interval: int = Field(
        default=3600, 
        description="Quality check interval in seconds"
    )
    lineage_discovery_interval: int = Field(
        default=7200, 
        description="Lineage discovery interval in seconds"
    )
    
    class Config:
        env_prefix = "WORKER_"


class LoggingSettings(BaseSettings):
    """Logging configuration settings."""
    
    level: str = Field(default="INFO", description="Log level")
    format: str = Field(
        default="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        description="Log format"
    )
    file_path: Optional[str] = Field(default=None, description="Log file path")
    max_file_size: int = Field(default=10485760, description="Max log file size in bytes")  # 10MB
    backup_count: int = Field(default=5, description="Number of backup log files")
    
    class Config:
        env_prefix = "LOG_"


class MonitoringSettings(BaseSettings):
    """Monitoring and metrics configuration."""
    
    enable_prometheus: bool = Field(default=True, description="Enable Prometheus metrics")
    prometheus_port: int = Field(default=9090, description="Prometheus metrics port")
    health_check_endpoint: str = Field(default="/health", description="Health check endpoint")
    
    class Config:
        env_prefix = "MONITORING_"


class Settings(BaseSettings):
    """Main application settings."""
    
    environment: str = Field(default="development", description="Environment")
    debug: bool = Field(default=False, description="Debug mode")
    
    # Sub-settings
    database: DatabaseSettings = Field(default_factory=DatabaseSettings)
    security: SecuritySettings = Field(default_factory=SecuritySettings)
    api: APISettings = Field(default_factory=APISettings)
    worker: WorkerSettings = Field(default_factory=WorkerSettings)
    logging: LoggingSettings = Field(default_factory=LoggingSettings)
    monitoring: MonitoringSettings = Field(default_factory=MonitoringSettings)
    
    @validator("environment")
    def validate_environment(cls, v: str) -> str:
        """Validate environment."""
        allowed = ["development", "testing", "staging", "production"]
        if v not in allowed:
            raise ValueError(f"Environment must be one of: {allowed}")
        return v
    
    class Config:
        env_file = ".env"
        case_sensitive = False
        validate_assignment = True
    
    def is_development(self) -> bool:
        """Check if running in development mode."""
        return self.environment == "development"
    
    def is_production(self) -> bool:
        """Check if running in production mode."""
        return self.environment == "production"


# Global settings instance
settings = Settings()