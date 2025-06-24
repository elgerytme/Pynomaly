"""Model storage information value object."""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from enum import Enum
from typing import Optional


class StorageBackend(Enum):
    """Supported storage backends for model persistence."""
    LOCAL_FILESYSTEM = "local_filesystem"
    AWS_S3 = "aws_s3"
    AZURE_BLOB = "azure_blob"
    GCP_STORAGE = "gcp_storage"
    MLFLOW = "mlflow"
    HUGGINGFACE_HUB = "huggingface_hub"
    DATABASE = "database"
    REDIS = "redis"


class SerializationFormat(Enum):
    """Supported model serialization formats."""
    PICKLE = "pickle"
    JOBLIB = "joblib"
    ONNX = "onnx"
    TENSORFLOW_SAVEDMODEL = "tensorflow_savedmodel"
    PYTORCH_STATE_DICT = "pytorch_state_dict"
    HUGGINGFACE = "huggingface"
    MLFLOW_MODEL = "mlflow_model"
    SCIKIT_LEARN_PICKLE = "scikit_learn_pickle"
    JAX_PARAMS = "jax_params"


@dataclass(frozen=True)
class ModelStorageInfo:
    """Information about model storage location and format.
    
    This value object encapsulates all information needed to locate,
    retrieve, and validate a stored model.
    
    Attributes:
        storage_backend: Backend system where model is stored
        storage_path: Path or identifier within the storage backend
        format: Serialization format used for the model
        size_bytes: Size of stored model in bytes
        checksum: SHA-256 checksum for integrity verification
        encryption_key_id: ID of encryption key if model is encrypted
        compression_type: Type of compression applied (if any)
        metadata_path: Path to additional metadata file
    """
    
    storage_backend: StorageBackend
    storage_path: str
    format: SerializationFormat
    size_bytes: int
    checksum: str
    encryption_key_id: Optional[str] = None
    compression_type: Optional[str] = None
    metadata_path: Optional[str] = None
    
    def __post_init__(self) -> None:
        """Validate storage information."""
        if not isinstance(self.storage_backend, StorageBackend):
            raise TypeError(
                f"Storage backend must be StorageBackend enum, "
                f"got {type(self.storage_backend)}"
            )
        
        if not isinstance(self.format, SerializationFormat):
            raise TypeError(
                f"Format must be SerializationFormat enum, "
                f"got {type(self.format)}"
            )
        
        if not self.storage_path:
            raise ValueError("Storage path cannot be empty")
        
        if not isinstance(self.size_bytes, int) or self.size_bytes < 0:
            raise ValueError(f"Size must be non-negative integer, got {self.size_bytes}")
        
        if not self.checksum:
            raise ValueError("Checksum cannot be empty")
        
        # Validate checksum format (SHA-256 should be 64 hex characters)
        if not self._is_valid_sha256(self.checksum):
            raise ValueError(f"Invalid SHA-256 checksum format: {self.checksum}")
    
    @staticmethod
    def _is_valid_sha256(checksum: str) -> bool:
        """Validate SHA-256 checksum format."""
        if not isinstance(checksum, str):
            return False
        
        if len(checksum) != 64:
            return False
        
        try:
            int(checksum, 16)
            return True
        except ValueError:
            return False
    
    @classmethod
    def create_for_local_file(
        cls,
        file_path: str,
        format: SerializationFormat,
        size_bytes: int,
        checksum: str,
        compression_type: Optional[str] = None
    ) -> ModelStorageInfo:
        """Create storage info for local filesystem storage.
        
        Args:
            file_path: Path to the model file
            format: Serialization format
            size_bytes: File size in bytes
            checksum: SHA-256 checksum
            compression_type: Optional compression type
            
        Returns:
            ModelStorageInfo instance for local storage
        """
        return cls(
            storage_backend=StorageBackend.LOCAL_FILESYSTEM,
            storage_path=file_path,
            format=format,
            size_bytes=size_bytes,
            checksum=checksum,
            compression_type=compression_type
        )
    
    @classmethod
    def create_for_s3(
        cls,
        bucket: str,
        key: str,
        format: SerializationFormat,
        size_bytes: int,
        checksum: str,
        encryption_key_id: Optional[str] = None
    ) -> ModelStorageInfo:
        """Create storage info for AWS S3 storage.
        
        Args:
            bucket: S3 bucket name
            key: S3 object key
            format: Serialization format
            size_bytes: Object size in bytes
            checksum: SHA-256 checksum
            encryption_key_id: Optional KMS key ID
            
        Returns:
            ModelStorageInfo instance for S3 storage
        """
        storage_path = f"s3://{bucket}/{key}"
        
        return cls(
            storage_backend=StorageBackend.AWS_S3,
            storage_path=storage_path,
            format=format,
            size_bytes=size_bytes,
            checksum=checksum,
            encryption_key_id=encryption_key_id
        )
    
    @classmethod
    def create_for_mlflow(
        cls,
        experiment_id: str,
        run_id: str,
        artifact_path: str,
        format: SerializationFormat,
        size_bytes: int,
        checksum: str
    ) -> ModelStorageInfo:
        """Create storage info for MLflow storage.
        
        Args:
            experiment_id: MLflow experiment ID
            run_id: MLflow run ID
            artifact_path: Path within the run artifacts
            format: Serialization format
            size_bytes: Artifact size in bytes
            checksum: SHA-256 checksum
            
        Returns:
            ModelStorageInfo instance for MLflow storage
        """
        storage_path = f"runs:/{run_id}/{artifact_path}"
        
        return cls(
            storage_backend=StorageBackend.MLFLOW,
            storage_path=storage_path,
            format=format,
            size_bytes=size_bytes,
            checksum=checksum
        )
    
    @property
    def is_encrypted(self) -> bool:
        """Check if the model is encrypted."""
        return self.encryption_key_id is not None
    
    @property
    def is_compressed(self) -> bool:
        """Check if the model is compressed."""
        return self.compression_type is not None
    
    @property
    def is_cloud_storage(self) -> bool:
        """Check if stored in cloud storage."""
        cloud_backends = {
            StorageBackend.AWS_S3,
            StorageBackend.AZURE_BLOB,
            StorageBackend.GCP_STORAGE
        }
        return self.storage_backend in cloud_backends
    
    @property
    def is_local_storage(self) -> bool:
        """Check if stored in local filesystem."""
        return self.storage_backend == StorageBackend.LOCAL_FILESYSTEM
    
    @property
    def format_supports_streaming(self) -> bool:
        """Check if format supports streaming/lazy loading."""
        streaming_formats = {
            SerializationFormat.HUGGINGFACE,
            SerializationFormat.MLFLOW_MODEL,
            SerializationFormat.TENSORFLOW_SAVEDMODEL
        }
        return self.format in streaming_formats
    
    @property
    def format_is_portable(self) -> bool:
        """Check if format is cross-platform portable."""
        portable_formats = {
            SerializationFormat.ONNX,
            SerializationFormat.TENSORFLOW_SAVEDMODEL,
            SerializationFormat.HUGGINGFACE,
            SerializationFormat.MLFLOW_MODEL
        }
        return self.format in portable_formats
    
    def get_storage_url(self) -> str:
        """Get full storage URL."""
        if self.storage_backend == StorageBackend.LOCAL_FILESYSTEM:
            return f"file://{self.storage_path}"
        elif self.storage_backend in {StorageBackend.AWS_S3, StorageBackend.AZURE_BLOB, StorageBackend.GCP_STORAGE}:
            return self.storage_path  # Already includes protocol
        elif self.storage_backend == StorageBackend.MLFLOW:
            return f"mlflow://{self.storage_path}"
        elif self.storage_backend == StorageBackend.HUGGINGFACE_HUB:
            return f"hf://{self.storage_path}"
        else:
            return self.storage_path
    
    def get_size_mb(self) -> float:
        """Get size in megabytes."""
        return self.size_bytes / (1024 * 1024)
    
    def get_size_human_readable(self) -> str:
        """Get human-readable size string."""
        size = self.size_bytes
        
        if size < 1024:
            return f"{size} B"
        elif size < 1024 ** 2:
            return f"{size / 1024:.1f} KB"
        elif size < 1024 ** 3:
            return f"{size / (1024 ** 2):.1f} MB"
        else:
            return f"{size / (1024 ** 3):.1f} GB"
    
    def verify_checksum(self, data: bytes) -> bool:
        """Verify data against stored checksum.
        
        Args:
            data: Model data bytes
            
        Returns:
            True if checksum matches
        """
        computed_checksum = hashlib.sha256(data).hexdigest()
        return computed_checksum.lower() == self.checksum.lower()
    
    def to_dict(self) -> dict:
        """Convert to dictionary representation."""
        return {
            "storage_backend": self.storage_backend.value,
            "storage_path": self.storage_path,
            "format": self.format.value,
            "size_bytes": self.size_bytes,
            "size_human": self.get_size_human_readable(),
            "checksum": self.checksum,
            "encryption_key_id": self.encryption_key_id,
            "compression_type": self.compression_type,
            "metadata_path": self.metadata_path,
            "is_encrypted": self.is_encrypted,
            "is_compressed": self.is_compressed,
            "is_cloud_storage": self.is_cloud_storage,
            "storage_url": self.get_storage_url(),
        }
    
    def __str__(self) -> str:
        """Human-readable representation."""
        return (
            f"Storage({self.format.value} on {self.storage_backend.value}, "
            f"{self.get_size_human_readable()})"
        )