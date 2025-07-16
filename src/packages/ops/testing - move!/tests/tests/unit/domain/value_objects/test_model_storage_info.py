"""Tests for model storage info value object."""

import hashlib

import pytest

from monorepo.domain.value_objects.model_storage_info import (
    ModelStorageInfo,
    SerializationFormat,
    StorageBackend,
)


class TestStorageBackend:
    """Test suite for StorageBackend enum."""

    def test_enum_values(self):
        """Test all enum values are correctly defined."""
        assert StorageBackend.LOCAL_FILESYSTEM == "local_filesystem"
        assert StorageBackend.AWS_S3 == "aws_s3"
        assert StorageBackend.AZURE_BLOB == "azure_blob"
        assert StorageBackend.GCP_STORAGE == "gcp_storage"
        assert StorageBackend.MLFLOW == "mlflow"
        assert StorageBackend.HUGGINGFACE_HUB == "huggingface_hub"
        assert StorageBackend.DATABASE == "database"
        assert StorageBackend.REDIS == "redis"

    def test_enum_count(self):
        """Test enum has correct number of values."""
        assert len(StorageBackend) == 8

    def test_enum_iteration(self):
        """Test enum iteration."""
        backends = list(StorageBackend)
        assert len(backends) == 8
        assert StorageBackend.LOCAL_FILESYSTEM in backends
        assert StorageBackend.AWS_S3 in backends


class TestSerializationFormat:
    """Test suite for SerializationFormat enum."""

    def test_enum_values(self):
        """Test all enum values are correctly defined."""
        assert SerializationFormat.PICKLE == "pickle"
        assert SerializationFormat.JOBLIB == "joblib"
        assert SerializationFormat.ONNX == "onnx"
        assert SerializationFormat.TENSORFLOW_SAVEDMODEL == "tensorflow_savedmodel"
        assert SerializationFormat.PYTORCH_STATE_DICT == "pytorch_state_dict"
        assert SerializationFormat.HUGGINGFACE == "huggingface"
        assert SerializationFormat.MLFLOW_MODEL == "mlflow_model"
        assert SerializationFormat.SCIKIT_LEARN_PICKLE == "scikit_learn_pickle"
        assert SerializationFormat.JAX_PARAMS == "jax_params"

    def test_enum_count(self):
        """Test enum has correct number of values."""
        assert len(SerializationFormat) == 9

    def test_enum_iteration(self):
        """Test enum iteration."""
        formats = list(SerializationFormat)
        assert len(formats) == 9
        assert SerializationFormat.PICKLE in formats
        assert SerializationFormat.ONNX in formats


class TestModelStorageInfo:
    """Test suite for ModelStorageInfo value object."""

    def test_basic_creation(self):
        """Test basic creation of model storage info."""
        storage_info = ModelStorageInfo(
            storage_backend=StorageBackend.LOCAL_FILESYSTEM,
            storage_path="/path/to/model.pkl",
            format=SerializationFormat.PICKLE,
            size_bytes=1024,
            checksum="a" * 64,  # Valid SHA-256 format
        )

        assert storage_info.storage_backend == StorageBackend.LOCAL_FILESYSTEM
        assert storage_info.storage_path == "/path/to/model.pkl"
        assert storage_info.format == SerializationFormat.PICKLE
        assert storage_info.size_bytes == 1024
        assert storage_info.checksum == "a" * 64

    def test_creation_with_optional_fields(self):
        """Test creation with optional fields."""
        storage_info = ModelStorageInfo(
            storage_backend=StorageBackend.AWS_S3,
            storage_path="s3://bucket/model.pkl",
            format=SerializationFormat.PICKLE,
            size_bytes=1024,
            checksum="a" * 64,
            encryption_key_id="key123",
            compression_type="gzip",
            metadata_path="s3://bucket/model.meta",
        )

        assert storage_info.encryption_key_id == "key123"
        assert storage_info.compression_type == "gzip"
        assert storage_info.metadata_path == "s3://bucket/model.meta"

    def test_immutability(self):
        """Test that model storage info is immutable."""
        storage_info = ModelStorageInfo(
            storage_backend=StorageBackend.LOCAL_FILESYSTEM,
            storage_path="/path/to/model.pkl",
            format=SerializationFormat.PICKLE,
            size_bytes=1024,
            checksum="a" * 64,
        )

        # Should not be able to modify values
        with pytest.raises(AttributeError):
            storage_info.storage_backend = StorageBackend.AWS_S3

    def test_validation_storage_backend_type(self):
        """Test validation of storage backend type."""
        with pytest.raises(
            TypeError, match="Storage backend must be StorageBackend enum"
        ):
            ModelStorageInfo(
                storage_backend="local_filesystem",  # String instead of enum
                storage_path="/path/to/model.pkl",
                format=SerializationFormat.PICKLE,
                size_bytes=1024,
                checksum="a" * 64,
            )

    def test_validation_format_type(self):
        """Test validation of format type."""
        with pytest.raises(TypeError, match="Format must be SerializationFormat enum"):
            ModelStorageInfo(
                storage_backend=StorageBackend.LOCAL_FILESYSTEM,
                storage_path="/path/to/model.pkl",
                format="pickle",  # String instead of enum
                size_bytes=1024,
                checksum="a" * 64,
            )

    def test_validation_storage_path_empty(self):
        """Test validation of empty storage path."""
        with pytest.raises(ValueError, match="Storage path cannot be empty"):
            ModelStorageInfo(
                storage_backend=StorageBackend.LOCAL_FILESYSTEM,
                storage_path="",
                format=SerializationFormat.PICKLE,
                size_bytes=1024,
                checksum="a" * 64,
            )

    def test_validation_size_bytes(self):
        """Test validation of size_bytes."""
        # Valid sizes
        ModelStorageInfo(
            storage_backend=StorageBackend.LOCAL_FILESYSTEM,
            storage_path="/path/to/model.pkl",
            format=SerializationFormat.PICKLE,
            size_bytes=0,
            checksum="a" * 64,
        )

        ModelStorageInfo(
            storage_backend=StorageBackend.LOCAL_FILESYSTEM,
            storage_path="/path/to/model.pkl",
            format=SerializationFormat.PICKLE,
            size_bytes=1024,
            checksum="a" * 64,
        )

        # Invalid sizes
        with pytest.raises(ValueError, match="Size must be non-negative integer"):
            ModelStorageInfo(
                storage_backend=StorageBackend.LOCAL_FILESYSTEM,
                storage_path="/path/to/model.pkl",
                format=SerializationFormat.PICKLE,
                size_bytes=-1,
                checksum="a" * 64,
            )

        with pytest.raises(ValueError, match="Size must be non-negative integer"):
            ModelStorageInfo(
                storage_backend=StorageBackend.LOCAL_FILESYSTEM,
                storage_path="/path/to/model.pkl",
                format=SerializationFormat.PICKLE,
                size_bytes=1024.5,
                checksum="a" * 64,
            )

    def test_validation_checksum_empty(self):
        """Test validation of empty checksum."""
        with pytest.raises(ValueError, match="Checksum cannot be empty"):
            ModelStorageInfo(
                storage_backend=StorageBackend.LOCAL_FILESYSTEM,
                storage_path="/path/to/model.pkl",
                format=SerializationFormat.PICKLE,
                size_bytes=1024,
                checksum="",
            )

    def test_validation_checksum_format(self):
        """Test validation of checksum format."""
        # Valid SHA-256 checksums
        ModelStorageInfo(
            storage_backend=StorageBackend.LOCAL_FILESYSTEM,
            storage_path="/path/to/model.pkl",
            format=SerializationFormat.PICKLE,
            size_bytes=1024,
            checksum="a" * 64,  # 64 hex chars
        )

        ModelStorageInfo(
            storage_backend=StorageBackend.LOCAL_FILESYSTEM,
            storage_path="/path/to/model.pkl",
            format=SerializationFormat.PICKLE,
            size_bytes=1024,
            checksum="1234567890abcdef" * 4,  # 64 hex chars
        )

        # Invalid checksums
        with pytest.raises(ValueError, match="Invalid SHA-256 checksum format"):
            ModelStorageInfo(
                storage_backend=StorageBackend.LOCAL_FILESYSTEM,
                storage_path="/path/to/model.pkl",
                format=SerializationFormat.PICKLE,
                size_bytes=1024,
                checksum="a" * 63,  # Too short
            )

        with pytest.raises(ValueError, match="Invalid SHA-256 checksum format"):
            ModelStorageInfo(
                storage_backend=StorageBackend.LOCAL_FILESYSTEM,
                storage_path="/path/to/model.pkl",
                format=SerializationFormat.PICKLE,
                size_bytes=1024,
                checksum="a" * 65,  # Too long
            )

        with pytest.raises(ValueError, match="Invalid SHA-256 checksum format"):
            ModelStorageInfo(
                storage_backend=StorageBackend.LOCAL_FILESYSTEM,
                storage_path="/path/to/model.pkl",
                format=SerializationFormat.PICKLE,
                size_bytes=1024,
                checksum="g" * 64,  # Invalid hex character
            )

    def test_is_valid_sha256_method(self):
        """Test _is_valid_sha256 static method."""
        # Valid cases
        assert ModelStorageInfo._is_valid_sha256("a" * 64) is True
        assert ModelStorageInfo._is_valid_sha256("1234567890abcdef" * 4) is True
        assert ModelStorageInfo._is_valid_sha256("ABCDEF0123456789" * 4) is True

        # Invalid cases
        assert ModelStorageInfo._is_valid_sha256("a" * 63) is False  # Too short
        assert ModelStorageInfo._is_valid_sha256("a" * 65) is False  # Too long
        assert ModelStorageInfo._is_valid_sha256("g" * 64) is False  # Invalid hex
        assert ModelStorageInfo._is_valid_sha256(123) is False  # Not string
        assert ModelStorageInfo._is_valid_sha256(None) is False  # None

    def test_create_for_local_file_factory(self):
        """Test create_for_local_file factory method."""
        storage_info = ModelStorageInfo.create_for_local_file(
            file_path="/path/to/model.pkl",
            format=SerializationFormat.PICKLE,
            size_bytes=1024,
            checksum="a" * 64,
            compression_type="gzip",
        )

        assert storage_info.storage_backend == StorageBackend.LOCAL_FILESYSTEM
        assert storage_info.storage_path == "/path/to/model.pkl"
        assert storage_info.format == SerializationFormat.PICKLE
        assert storage_info.size_bytes == 1024
        assert storage_info.checksum == "a" * 64
        assert storage_info.compression_type == "gzip"
        assert storage_info.encryption_key_id is None
        assert storage_info.metadata_path is None

    def test_create_for_s3_factory(self):
        """Test create_for_s3 factory method."""
        storage_info = ModelStorageInfo.create_for_s3(
            bucket="my-bucket",
            key="models/model.pkl",
            format=SerializationFormat.PICKLE,
            size_bytes=2048,
            checksum="b" * 64,
            encryption_key_id="key456",
        )

        assert storage_info.storage_backend == StorageBackend.AWS_S3
        assert storage_info.storage_path == "s3://my-bucket/models/model.pkl"
        assert storage_info.format == SerializationFormat.PICKLE
        assert storage_info.size_bytes == 2048
        assert storage_info.checksum == "b" * 64
        assert storage_info.encryption_key_id == "key456"

    def test_create_for_mlflow_factory(self):
        """Test create_for_mlflow factory method."""
        storage_info = ModelStorageInfo.create_for_mlflow(
            experiment_id="exp123",
            run_id="run456",
            artifact_path="models/model",
            format=SerializationFormat.MLFLOW_MODEL,
            size_bytes=4096,
            checksum="c" * 64,
        )

        assert storage_info.storage_backend == StorageBackend.MLFLOW
        assert storage_info.storage_path == "runs:/run456/models/model"
        assert storage_info.format == SerializationFormat.MLFLOW_MODEL
        assert storage_info.size_bytes == 4096
        assert storage_info.checksum == "c" * 64

    def test_is_encrypted_property(self):
        """Test is_encrypted property."""
        # Without encryption key
        storage_info1 = ModelStorageInfo(
            storage_backend=StorageBackend.LOCAL_FILESYSTEM,
            storage_path="/path/to/model.pkl",
            format=SerializationFormat.PICKLE,
            size_bytes=1024,
            checksum="a" * 64,
        )
        assert storage_info1.is_encrypted is False

        # With encryption key
        storage_info2 = ModelStorageInfo(
            storage_backend=StorageBackend.AWS_S3,
            storage_path="s3://bucket/model.pkl",
            format=SerializationFormat.PICKLE,
            size_bytes=1024,
            checksum="a" * 64,
            encryption_key_id="key123",
        )
        assert storage_info2.is_encrypted is True

    def test_is_compressed_property(self):
        """Test is_compressed property."""
        # Without compression
        storage_info1 = ModelStorageInfo(
            storage_backend=StorageBackend.LOCAL_FILESYSTEM,
            storage_path="/path/to/model.pkl",
            format=SerializationFormat.PICKLE,
            size_bytes=1024,
            checksum="a" * 64,
        )
        assert storage_info1.is_compressed is False

        # With compression
        storage_info2 = ModelStorageInfo(
            storage_backend=StorageBackend.LOCAL_FILESYSTEM,
            storage_path="/path/to/model.pkl",
            format=SerializationFormat.PICKLE,
            size_bytes=1024,
            checksum="a" * 64,
            compression_type="gzip",
        )
        assert storage_info2.is_compressed is True

    def test_is_cloud_storage_property(self):
        """Test is_cloud_storage property."""
        # Cloud storage backends
        cloud_backends = [
            StorageBackend.AWS_S3,
            StorageBackend.AZURE_BLOB,
            StorageBackend.GCP_STORAGE,
        ]

        for backend in cloud_backends:
            storage_info = ModelStorageInfo(
                storage_backend=backend,
                storage_path="cloud://bucket/model.pkl",
                format=SerializationFormat.PICKLE,
                size_bytes=1024,
                checksum="a" * 64,
            )
            assert storage_info.is_cloud_storage is True

        # Non-cloud storage backends
        non_cloud_backends = [
            StorageBackend.LOCAL_FILESYSTEM,
            StorageBackend.MLFLOW,
            StorageBackend.DATABASE,
            StorageBackend.REDIS,
        ]

        for backend in non_cloud_backends:
            storage_info = ModelStorageInfo(
                storage_backend=backend,
                storage_path="/path/to/model.pkl",
                format=SerializationFormat.PICKLE,
                size_bytes=1024,
                checksum="a" * 64,
            )
            assert storage_info.is_cloud_storage is False

    def test_is_local_storage_property(self):
        """Test is_local_storage property."""
        # Local storage
        storage_info1 = ModelStorageInfo(
            storage_backend=StorageBackend.LOCAL_FILESYSTEM,
            storage_path="/path/to/model.pkl",
            format=SerializationFormat.PICKLE,
            size_bytes=1024,
            checksum="a" * 64,
        )
        assert storage_info1.is_local_storage is True

        # Non-local storage
        storage_info2 = ModelStorageInfo(
            storage_backend=StorageBackend.AWS_S3,
            storage_path="s3://bucket/model.pkl",
            format=SerializationFormat.PICKLE,
            size_bytes=1024,
            checksum="a" * 64,
        )
        assert storage_info2.is_local_storage is False

    def test_format_supports_streaming_property(self):
        """Test format_supports_streaming property."""
        # Streaming formats
        streaming_formats = [
            SerializationFormat.HUGGINGFACE,
            SerializationFormat.MLFLOW_MODEL,
            SerializationFormat.TENSORFLOW_SAVEDMODEL,
        ]

        for format_type in streaming_formats:
            storage_info = ModelStorageInfo(
                storage_backend=StorageBackend.LOCAL_FILESYSTEM,
                storage_path="/path/to/model",
                format=format_type,
                size_bytes=1024,
                checksum="a" * 64,
            )
            assert storage_info.format_supports_streaming is True

        # Non-streaming formats
        non_streaming_formats = [
            SerializationFormat.PICKLE,
            SerializationFormat.JOBLIB,
            SerializationFormat.ONNX,
        ]

        for format_type in non_streaming_formats:
            storage_info = ModelStorageInfo(
                storage_backend=StorageBackend.LOCAL_FILESYSTEM,
                storage_path="/path/to/model",
                format=format_type,
                size_bytes=1024,
                checksum="a" * 64,
            )
            assert storage_info.format_supports_streaming is False

    def test_format_is_portable_property(self):
        """Test format_is_portable property."""
        # Portable formats
        portable_formats = [
            SerializationFormat.ONNX,
            SerializationFormat.TENSORFLOW_SAVEDMODEL,
            SerializationFormat.HUGGINGFACE,
            SerializationFormat.MLFLOW_MODEL,
        ]

        for format_type in portable_formats:
            storage_info = ModelStorageInfo(
                storage_backend=StorageBackend.LOCAL_FILESYSTEM,
                storage_path="/path/to/model",
                format=format_type,
                size_bytes=1024,
                checksum="a" * 64,
            )
            assert storage_info.format_is_portable is True

        # Non-portable formats
        non_portable_formats = [
            SerializationFormat.PICKLE,
            SerializationFormat.JOBLIB,
            SerializationFormat.PYTORCH_STATE_DICT,
        ]

        for format_type in non_portable_formats:
            storage_info = ModelStorageInfo(
                storage_backend=StorageBackend.LOCAL_FILESYSTEM,
                storage_path="/path/to/model",
                format=format_type,
                size_bytes=1024,
                checksum="a" * 64,
            )
            assert storage_info.format_is_portable is False

    def test_get_storage_url_method(self):
        """Test get_storage_url method."""
        # Local filesystem
        local_storage = ModelStorageInfo(
            storage_backend=StorageBackend.LOCAL_FILESYSTEM,
            storage_path="/path/to/model.pkl",
            format=SerializationFormat.PICKLE,
            size_bytes=1024,
            checksum="a" * 64,
        )
        assert local_storage.get_storage_url() == "file:///path/to/model.pkl"

        # AWS S3
        s3_storage = ModelStorageInfo(
            storage_backend=StorageBackend.AWS_S3,
            storage_path="s3://bucket/model.pkl",
            format=SerializationFormat.PICKLE,
            size_bytes=1024,
            checksum="a" * 64,
        )
        assert s3_storage.get_storage_url() == "s3://bucket/model.pkl"

        # MLflow
        mlflow_storage = ModelStorageInfo(
            storage_backend=StorageBackend.MLFLOW,
            storage_path="runs:/run123/model",
            format=SerializationFormat.MLFLOW_MODEL,
            size_bytes=1024,
            checksum="a" * 64,
        )
        assert mlflow_storage.get_storage_url() == "mlflow://runs:/run123/model"

        # HuggingFace Hub
        hf_storage = ModelStorageInfo(
            storage_backend=StorageBackend.HUGGINGFACE_HUB,
            storage_path="model-name",
            format=SerializationFormat.HUGGINGFACE,
            size_bytes=1024,
            checksum="a" * 64,
        )
        assert hf_storage.get_storage_url() == "hf://model-name"

    def test_get_size_mb_method(self):
        """Test get_size_mb method."""
        storage_info = ModelStorageInfo(
            storage_backend=StorageBackend.LOCAL_FILESYSTEM,
            storage_path="/path/to/model.pkl",
            format=SerializationFormat.PICKLE,
            size_bytes=2097152,  # 2MB
            checksum="a" * 64,
        )
        assert storage_info.get_size_mb() == 2.0

        storage_info2 = ModelStorageInfo(
            storage_backend=StorageBackend.LOCAL_FILESYSTEM,
            storage_path="/path/to/model.pkl",
            format=SerializationFormat.PICKLE,
            size_bytes=1048576,  # 1MB
            checksum="a" * 64,
        )
        assert storage_info2.get_size_mb() == 1.0

    def test_get_size_human_readable_method(self):
        """Test get_size_human_readable method."""
        test_cases = [
            (500, "500 B"),
            (1024, "1.0 KB"),
            (1536, "1.5 KB"),
            (1048576, "1.0 MB"),
            (1610612736, "1.5 GB"),
            (1073741824, "1.0 GB"),
        ]

        for size_bytes, expected in test_cases:
            storage_info = ModelStorageInfo(
                storage_backend=StorageBackend.LOCAL_FILESYSTEM,
                storage_path="/path/to/model.pkl",
                format=SerializationFormat.PICKLE,
                size_bytes=size_bytes,
                checksum="a" * 64,
            )
            assert storage_info.get_size_human_readable() == expected

    def test_verify_checksum_method(self):
        """Test verify_checksum method."""
        test_data = b"test model data"
        expected_checksum = hashlib.sha256(test_data).hexdigest()

        storage_info = ModelStorageInfo(
            storage_backend=StorageBackend.LOCAL_FILESYSTEM,
            storage_path="/path/to/model.pkl",
            format=SerializationFormat.PICKLE,
            size_bytes=1024,
            checksum=expected_checksum,
        )

        # Valid checksum
        assert storage_info.verify_checksum(test_data) is True

        # Invalid checksum
        assert storage_info.verify_checksum(b"different data") is False

        # Test case insensitive comparison
        storage_info_upper = ModelStorageInfo(
            storage_backend=StorageBackend.LOCAL_FILESYSTEM,
            storage_path="/path/to/model.pkl",
            format=SerializationFormat.PICKLE,
            size_bytes=1024,
            checksum=expected_checksum.upper(),
        )
        assert storage_info_upper.verify_checksum(test_data) is True

    def test_to_dict_method(self):
        """Test to_dict method."""
        storage_info = ModelStorageInfo(
            storage_backend=StorageBackend.AWS_S3,
            storage_path="s3://bucket/model.pkl",
            format=SerializationFormat.PICKLE,
            size_bytes=1024,
            checksum="a" * 64,
            encryption_key_id="key123",
            compression_type="gzip",
            metadata_path="s3://bucket/model.meta",
        )

        result = storage_info.to_dict()

        expected = {
            "storage_backend": "aws_s3",
            "storage_path": "s3://bucket/model.pkl",
            "format": "pickle",
            "size_bytes": 1024,
            "size_human": "1.0 KB",
            "checksum": "a" * 64,
            "encryption_key_id": "key123",
            "compression_type": "gzip",
            "metadata_path": "s3://bucket/model.meta",
            "is_encrypted": True,
            "is_compressed": True,
            "is_cloud_storage": True,
            "storage_url": "s3://bucket/model.pkl",
        }

        assert result == expected

    def test_string_representation(self):
        """Test string representation."""
        storage_info = ModelStorageInfo(
            storage_backend=StorageBackend.LOCAL_FILESYSTEM,
            storage_path="/path/to/model.pkl",
            format=SerializationFormat.PICKLE,
            size_bytes=1024,
            checksum="a" * 64,
        )

        result = str(storage_info)
        expected = "Storage(pickle on local_filesystem, 1.0 KB)"
        assert result == expected

    def test_equality_comparison(self):
        """Test equality comparison."""
        storage_info1 = ModelStorageInfo(
            storage_backend=StorageBackend.LOCAL_FILESYSTEM,
            storage_path="/path/to/model.pkl",
            format=SerializationFormat.PICKLE,
            size_bytes=1024,
            checksum="a" * 64,
        )

        storage_info2 = ModelStorageInfo(
            storage_backend=StorageBackend.LOCAL_FILESYSTEM,
            storage_path="/path/to/model.pkl",
            format=SerializationFormat.PICKLE,
            size_bytes=1024,
            checksum="a" * 64,
        )

        storage_info3 = ModelStorageInfo(
            storage_backend=StorageBackend.LOCAL_FILESYSTEM,
            storage_path="/path/to/model.pkl",
            format=SerializationFormat.PICKLE,
            size_bytes=2048,  # Different size
            checksum="a" * 64,
        )

        assert storage_info1 == storage_info2
        assert storage_info1 != storage_info3

    def test_hash_behavior(self):
        """Test hash behavior for use in sets and dictionaries."""
        storage_info1 = ModelStorageInfo(
            storage_backend=StorageBackend.LOCAL_FILESYSTEM,
            storage_path="/path/to/model.pkl",
            format=SerializationFormat.PICKLE,
            size_bytes=1024,
            checksum="a" * 64,
        )

        storage_info2 = ModelStorageInfo(
            storage_backend=StorageBackend.LOCAL_FILESYSTEM,
            storage_path="/path/to/model.pkl",
            format=SerializationFormat.PICKLE,
            size_bytes=1024,
            checksum="a" * 64,
        )

        storage_info3 = ModelStorageInfo(
            storage_backend=StorageBackend.AWS_S3,
            storage_path="s3://bucket/model.pkl",
            format=SerializationFormat.PICKLE,
            size_bytes=1024,
            checksum="a" * 64,
        )

        # Same values should have same hash
        assert hash(storage_info1) == hash(storage_info2)

        # Different values should have different hash
        assert hash(storage_info1) != hash(storage_info3)

        # Test in set
        storage_set = {storage_info1, storage_info2, storage_info3}
        assert len(storage_set) == 2  # storage_info1 and storage_info2 are equal

    def test_repr_representation(self):
        """Test repr representation."""
        storage_info = ModelStorageInfo(
            storage_backend=StorageBackend.LOCAL_FILESYSTEM,
            storage_path="/path/to/model.pkl",
            format=SerializationFormat.PICKLE,
            size_bytes=1024,
            checksum="a" * 64,
        )

        repr_str = repr(storage_info)
        assert "ModelStorageInfo" in repr_str
        assert "LOCAL_FILESYSTEM" in repr_str
        assert "PICKLE" in repr_str

    def test_comprehensive_usage_scenario(self):
        """Test comprehensive usage scenario."""
        # Create storage info with all features
        storage_info = ModelStorageInfo(
            storage_backend=StorageBackend.AWS_S3,
            storage_path="s3://ml-models/anomaly-detection/v1.0.0/model.pkl",
            format=SerializationFormat.PICKLE,
            size_bytes=5242880,  # 5MB
            checksum="1234567890abcdef" * 4,
            encryption_key_id="arn:aws:kms:us-west-2:123456789012:key/12345678-1234-1234-1234-123456789012",
            compression_type="gzip",
            metadata_path="s3://ml-models/anomaly-detection/v1.0.0/model.meta.json",
        )

        # Test all properties
        assert storage_info.is_cloud_storage is True
        assert storage_info.is_local_storage is False
        assert storage_info.is_encrypted is True
        assert storage_info.is_compressed is True
        assert storage_info.format_is_portable is False
        assert storage_info.format_supports_streaming is False

        # Test size calculations
        assert storage_info.get_size_mb() == 5.0
        assert storage_info.get_size_human_readable() == "5.0 MB"

        # Test URL generation
        assert (
            storage_info.get_storage_url()
            == "s3://ml-models/anomaly-detection/v1.0.0/model.pkl"
        )

        # Test checksum verification
        test_data = b"test model data"
        actual_checksum = hashlib.sha256(test_data).hexdigest()
        assert storage_info.verify_checksum(test_data) is False  # Different checksum

        # Test dictionary conversion
        result_dict = storage_info.to_dict()
        assert result_dict["storage_backend"] == "aws_s3"
        assert result_dict["is_encrypted"] is True
        assert result_dict["is_compressed"] is True
        assert result_dict["size_human"] == "5.0 MB"

    def test_factory_methods_comprehensive(self):
        """Test all factory methods comprehensively."""
        # Local file factory
        local_info = ModelStorageInfo.create_for_local_file(
            file_path="/models/anomaly_detector.joblib",
            format=SerializationFormat.JOBLIB,
            size_bytes=2097152,
            checksum="abcdef1234567890" * 4,
            compression_type="lzma",
        )

        assert local_info.storage_backend == StorageBackend.LOCAL_FILESYSTEM
        assert local_info.format == SerializationFormat.JOBLIB
        assert local_info.compression_type == "lzma"
        assert local_info.is_local_storage is True

        # S3 factory
        s3_info = ModelStorageInfo.create_for_s3(
            bucket="ml-models-prod",
            key="anomaly-detection/isolation-forest/v2.1.0/model.onnx",
            format=SerializationFormat.ONNX,
            size_bytes=10485760,
            checksum="fedcba0987654321" * 4,
            encryption_key_id="encryption-key-id",
        )

        assert s3_info.storage_backend == StorageBackend.AWS_S3
        assert s3_info.format == SerializationFormat.ONNX
        assert s3_info.encryption_key_id == "encryption-key-id"
        assert s3_info.is_cloud_storage is True
        assert s3_info.format_is_portable is True

        # MLflow factory
        mlflow_info = ModelStorageInfo.create_for_mlflow(
            experiment_id="123456",
            run_id="abcdef123456",
            artifact_path="model/artifacts",
            format=SerializationFormat.MLFLOW_MODEL,
            size_bytes=20971520,
            checksum="1a2b3c4d5e6f7890" * 4,
        )

        assert mlflow_info.storage_backend == StorageBackend.MLFLOW
        assert mlflow_info.format == SerializationFormat.MLFLOW_MODEL
        assert mlflow_info.storage_path == "runs:/abcdef123456/model/artifacts"
        assert mlflow_info.format_supports_streaming is True
        assert mlflow_info.format_is_portable is True

    def test_edge_cases(self):
        """Test edge cases."""
        # Zero size
        zero_size_info = ModelStorageInfo(
            storage_backend=StorageBackend.LOCAL_FILESYSTEM,
            storage_path="/path/to/empty.pkl",
            format=SerializationFormat.PICKLE,
            size_bytes=0,
            checksum="a" * 64,
        )
        assert zero_size_info.get_size_mb() == 0.0
        assert zero_size_info.get_size_human_readable() == "0 B"

        # Very large size
        large_size_info = ModelStorageInfo(
            storage_backend=StorageBackend.LOCAL_FILESYSTEM,
            storage_path="/path/to/large.pkl",
            format=SerializationFormat.PICKLE,
            size_bytes=10737418240,  # 10GB
            checksum="a" * 64,
        )
        assert large_size_info.get_size_mb() == 10240.0
        assert large_size_info.get_size_human_readable() == "10.0 GB"

        # Minimum valid path
        min_path_info = ModelStorageInfo(
            storage_backend=StorageBackend.LOCAL_FILESYSTEM,
            storage_path="a",  # Single character
            format=SerializationFormat.PICKLE,
            size_bytes=1,
            checksum="a" * 64,
        )
        assert min_path_info.storage_path == "a"

    def test_format_and_backend_combinations(self):
        """Test various format and backend combinations."""
        combinations = [
            (StorageBackend.LOCAL_FILESYSTEM, SerializationFormat.PICKLE),
            (StorageBackend.AWS_S3, SerializationFormat.JOBLIB),
            (StorageBackend.AZURE_BLOB, SerializationFormat.ONNX),
            (StorageBackend.GCP_STORAGE, SerializationFormat.TENSORFLOW_SAVEDMODEL),
            (StorageBackend.MLFLOW, SerializationFormat.MLFLOW_MODEL),
            (StorageBackend.HUGGINGFACE_HUB, SerializationFormat.HUGGINGFACE),
            (StorageBackend.DATABASE, SerializationFormat.PICKLE),
            (StorageBackend.REDIS, SerializationFormat.JOBLIB),
        ]

        for backend, format_type in combinations:
            storage_info = ModelStorageInfo(
                storage_backend=backend,
                storage_path="/path/or/uri",
                format=format_type,
                size_bytes=1024,
                checksum="a" * 64,
            )

            assert storage_info.storage_backend == backend
            assert storage_info.format == format_type

            # Test that properties work for all combinations
            assert isinstance(storage_info.is_cloud_storage, bool)
            assert isinstance(storage_info.is_local_storage, bool)
            assert isinstance(storage_info.format_supports_streaming, bool)
            assert isinstance(storage_info.format_is_portable, bool)
