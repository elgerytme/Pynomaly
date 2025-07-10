"""Application-specific settings."""

from __future__ import annotations

from pydantic import BaseModel


class AppSettings(BaseModel):
    """Application-specific settings."""

    name: str = "Pynomaly"
    version: str = "0.1.0"
    description: str = (
        "Advanced anomaly detection API with unified multi-algorithm interface"
    )
    environment: str = "development"
    debug: bool = False

    @property
    def is_production(self) -> bool:
        """Check if running in production mode."""
        return not self.debug