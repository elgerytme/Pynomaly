"""Application configuration."""

from __future__ import annotations

from functools import lru_cache
from typing import List

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Application settings."""
    
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
    )
    
    # App settings
    debug: bool = False
    app_name: str = "HTMX Tailwind App"
    version: str = "0.1.0"
    
    # Security
    secret_key: str = "your-secret-key-change-in-production"
    allowed_hosts: List[str] = ["*"]
    
    # Database
    database_url: str = "sqlite:///./app.db"
    
    # Redis
    redis_url: str = "redis://localhost:6379"
    
    # HTMX settings
    htmx_version: str = "1.9.9"
    enable_htmx_debug: bool = False
    
    # Static files
    static_url: str = "/static"
    media_url: str = "/media"
    
    # Templates
    template_debug: bool = False
    template_auto_reload: bool = True


@lru_cache()
def get_settings() -> Settings:
    """Get cached settings instance."""
    return Settings()