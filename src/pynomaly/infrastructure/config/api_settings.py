"""API configuration settings."""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel


class APISettings(BaseModel):
    """API configuration settings."""

    api_host: str = "0.0.0.0"
    api_port: int = 8000
    api_workers: int = 1
    api_cors_origins: list[str] = ["*"]
    api_rate_limit: int = 100  # requests per minute

    # Documentation settings
    docs_enabled: bool = True  # Enable OpenAPI documentation

    def get_cors_config(self) -> dict[str, Any]:
        """Get CORS configuration for API."""
        return {
            "allow_origins": self.api_cors_origins,
            "allow_credentials": True,
            "allow_methods": ["*"],
            "allow_headers": ["*"],
        }