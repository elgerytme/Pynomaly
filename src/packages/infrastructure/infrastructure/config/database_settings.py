"""Database configuration settings."""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel


class DatabaseSettings(BaseModel):
    """Database configuration settings."""

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

    def get_database_config(self, debug: bool = False) -> dict[str, Any]:
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
            "echo": self.database_echo or debug,
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