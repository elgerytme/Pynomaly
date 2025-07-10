"""Security and authentication configuration settings."""

from __future__ import annotations

import os
from typing import Any

from pydantic import BaseModel, Field, field_validator


class SecurityAuthSettings(BaseModel):
    """Security and authentication configuration settings."""

    # Security key
    secret_key: str = Field(
        default="change-me-in-production-this-is-32-chars-long-default-key"
    )
    auth_enabled: bool = True  # Enable authentication by default
    jwt_algorithm: str = "HS256"
    jwt_expiration: int = 3600  # seconds

    # Email settings
    smtp_server: str | None = None
    smtp_port: int = 587
    smtp_username: str | None = None
    smtp_password: str | None = None
    smtp_use_tls: bool = True
    sender_email: str | None = None
    sender_name: str = "Pynomaly System"
    base_url: str = "http://localhost:8000"

    @field_validator("secret_key")
    @classmethod
    def validate_secret_key(cls, v: str) -> str:
        """Validate secret key security."""
        if v == "change-me-in-production":
            # Check if we're in production environment
            env = os.getenv("PYNOMALY_APP_ENVIRONMENT", "development")
            if env in ["production", "prod"]:
                raise ValueError(
                    "Must set a secure secret key in production environment. "
                    "Set PYNOMALY_SECRET_KEY environment variable."
                )

        if len(v) < 32:
            raise ValueError("Secret key must be at least 32 characters long")

        return v

    @field_validator("jwt_algorithm")
    @classmethod
    def validate_jwt_algorithm(cls, v: str) -> str:
        """Validate JWT algorithm."""
        valid_algorithms = ["HS256", "HS384", "HS512", "RS256", "RS384", "RS512"]
        if v not in valid_algorithms:
            raise ValueError(f"JWT algorithm must be one of: {valid_algorithms}")
        return v

    @field_validator("jwt_expiration")
    @classmethod
    def validate_jwt_expiration(cls, v: int) -> int:
        """Validate JWT expiration time."""
        if v < 300:  # 5 minutes minimum
            raise ValueError("JWT expiration must be at least 300 seconds (5 minutes)")
        if v > 86400:  # 24 hours maximum
            raise ValueError("JWT expiration must be at most 86400 seconds (24 hours)")
        return v