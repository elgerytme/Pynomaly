"""Application settings configuration."""

from typing import List, Optional

from pydantic import Field, validator
from pydantic_settings import BaseSettings


class DatabaseSettings(BaseSettings):
    """Database configuration."""
    
    url: str = Field(default="sqlite:///./app.db", description="Database URL")
    echo: bool = Field(default=False, description="Enable SQL logging")
    pool_size: int = Field(default=5, description="Connection pool size")
    max_overflow: int = Field(default=10, description="Maximum pool overflow")
    
    class Config:
        env_prefix = "DATABASE_"


class RedisSettings(BaseSettings):
    """Redis configuration."""
    
    url: str = Field(default="redis://localhost:6379", description="Redis URL")
    db: int = Field(default=0, description="Redis database number")
    password: Optional[str] = Field(default=None, description="Redis password")
    
    class Config:
        env_prefix = "REDIS_"


class SecuritySettings(BaseSettings):
    """Security configuration."""
    
    secret_key: str = Field(..., description="Secret key for JWT tokens")
    algorithm: str = Field(default="HS256", description="JWT algorithm")
    access_token_expire_minutes: int = Field(default=30, description="Access token expiration")
    refresh_token_expire_days: int = Field(default=7, description="Refresh token expiration")
    password_reset_expire_minutes: int = Field(default=60, description="Password reset expiration")
    
    class Config:
        env_prefix = "SECURITY_"


class CORSSettings(BaseSettings):
    """CORS configuration."""
    
    allow_origins: List[str] = Field(default=["*"], description="Allowed origins")
    allow_credentials: bool = Field(default=True, description="Allow credentials")
    allow_methods: List[str] = Field(default=["*"], description="Allowed methods")
    allow_headers: List[str] = Field(default=["*"], description="Allowed headers")
    
    class Config:
        env_prefix = "CORS_"


class LoggingSettings(BaseSettings):
    """Logging configuration."""
    
    level: str = Field(default="INFO", description="Log level")
    format: str = Field(default="json", description="Log format")
    file_path: Optional[str] = Field(default=None, description="Log file path")
    
    class Config:
        env_prefix = "LOGGING_"


class Settings(BaseSettings):
    """Application settings."""
    
    app_name: str = Field(default="My App", description="Application name")
    app_version: str = Field(default="0.1.0", description="Application version")
    debug: bool = Field(default=False, description="Debug mode")
    environment: str = Field(default="development", description="Environment")
    host: str = Field(default="0.0.0.0", description="Host address")
    port: int = Field(default=8000, description="Port number")
    
    # Sub-configurations
    database: DatabaseSettings = Field(default_factory=DatabaseSettings)
    redis: RedisSettings = Field(default_factory=RedisSettings)
    security: SecuritySettings = Field(default_factory=SecuritySettings)
    cors: CORSSettings = Field(default_factory=CORSSettings)
    logging: LoggingSettings = Field(default_factory=LoggingSettings)
    
    @validator('environment')
    def validate_environment(cls, v: str) -> str:
        """Validate environment value."""
        allowed_environments = ["development", "staging", "production", "testing"]
        if v not in allowed_environments:
            raise ValueError(f"Environment must be one of: {allowed_environments}")
        return v
    
    @property
    def is_development(self) -> bool:
        """Check if in development environment."""
        return self.environment == "development"
    
    @property
    def is_production(self) -> bool:
        """Check if in production environment."""
        return self.environment == "production"
    
    @property
    def is_testing(self) -> bool:
        """Check if in testing environment."""
        return self.environment == "testing"
    
    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"
        case_sensitive = False


# Global settings instance
settings = Settings()