"""Storage configuration settings."""

from __future__ import annotations

from pathlib import Path

from pydantic import BaseModel, field_validator


class StorageSettings(BaseModel):
    """Storage configuration settings."""

    # Storage paths
    storage_path: Path = Path("./storage")
    model_storage_path: Path = Path("./storage/models")
    experiment_storage_path: Path = Path("./storage/experiments")
    temp_path: Path = Path("./storage/temp")
    log_path: Path = Path("./storage/logs")

    # Cache settings
    cache_enabled: bool = True
    cache_ttl: int = 3600  # seconds
    redis_url: str | None = None

    @field_validator(
        "storage_path", "model_storage_path", "experiment_storage_path", "temp_path"
    )
    @classmethod
    def create_directories(cls, v: Path) -> Path:
        """Ensure directories exist."""
        v.mkdir(parents=True, exist_ok=True)
        return v