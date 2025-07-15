"""Storage configuration settings."""

from __future__ import annotations

from pathlib import Path
from typing import Optional

from pydantic import BaseModel, field_validator


class StorageSettings(BaseModel):
    """Storage configuration settings."""

    model_config = {"protected_namespaces": ()}

    # Local storage paths
    storage_path: Path = Path("./storage")
    model_storage_path: Path = Path("./storage/models")
    experiment_storage_path: Path = Path("./storage/experiments")
    temp_path: Path = Path("./storage/temp")
    log_path: Path = Path("./storage/logs")
    local_storage_path: Path = Path("./storage")

    # Cloud storage provider configuration
    default_provider: str = "local"  # local, s3, azure, gcs, minio
    
    # AWS S3 settings
    s3_bucket_name: Optional[str] = None
    s3_region: str = "us-east-1"
    s3_endpoint_url: Optional[str] = None  # For MinIO compatibility
    
    # Azure Blob Storage settings
    azure_container_name: Optional[str] = None
    azure_account_name: Optional[str] = None
    azure_connection_string: Optional[str] = None
    
    # Google Cloud Storage settings
    gcs_bucket_name: Optional[str] = None
    gcs_project_id: Optional[str] = None
    
    # MinIO settings
    minio_endpoint: str = "http://localhost:9000"
    minio_bucket_name: str = "pynomaly"
    minio_use_ssl: bool = False

    # Cache settings
    cache_enabled: bool = True
    cache_ttl: int = 3600  # seconds
    redis_url: str | None = None

    @field_validator(
        "storage_path", "model_storage_path", "experiment_storage_path", 
        "temp_path", "log_path", "local_storage_path"
    )
    @classmethod
    def create_directories(cls, v: Path) -> Path:
        """Ensure directories exist."""
        v.mkdir(parents=True, exist_ok=True)
        return v
    
    @field_validator("default_provider")
    @classmethod
    def validate_provider(cls, v: str) -> str:
        """Validate storage provider."""
        valid_providers = {"local", "s3", "azure", "gcs", "minio"}
        if v not in valid_providers:
            raise ValueError(f"Invalid provider: {v}. Must be one of {valid_providers}")
        return v
