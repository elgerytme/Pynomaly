"""Tests for enhanced model persistence service."""

import tempfile
from pathlib import Path
from unittest.mock import AsyncMock, Mock

import numpy as np
import pytest

from pynomaly.application.services.enhanced_model_persistence_service import (
    EnhancedModelPersistenceService,
    ModelSerializationError,
    UnsupportedFormatError,
)
from pynomaly.domain.entities import Dataset, Detector
from pynomaly.domain.entities.model_version import ModelStatus, ModelVersion
from pynomaly.domain.value_objects import ContaminationRate
from pynomaly.domain.value_objects.model_storage_info import SerializationFormat
from pynomaly.domain.value_objects.performance_metrics import PerformanceMetrics
from pynomaly.domain.value_objects.semantic_version import SemanticVersion


class MockDetector(Detector):
    """Mock detector for testing."""

    def __init__(
        self,
        name: str = "test_detector",
        algorithm: str = "test_algo",
        fitted: bool = True,
    ):
        super().__init__(
            name=name,
            algorithm_name=algorithm,
            contamination_rate=ContaminationRate.auto(),
            is_fitted=fitted,
        )

    def fit(self, dataset: Dataset) -> None:
        """Mock fit method."""
        self.is_fitted = True

    def detect(self, dataset: Dataset):
        """Mock detect method."""
        from pynomaly.domain.entities import DetectionResult
        from pynomaly.domain.value_objects import AnomalyScore

        n_samples = len(dataset.data)
        scores = [AnomalyScore(0.5) for _ in range(n_samples)]
        labels = np.random.choice([0, 1], size=n_samples)

        return DetectionResult(
            detector_id=self.id,
            dataset_id=dataset.id,
            scores=scores,
            labels=labels,
            threshold=0.5,
            execution_time_ms=100,
        )

    def score(self, dataset: Dataset):
        """Mock score method."""
        from pynomaly.domain.value_objects import AnomalyScore

        n_samples = len(dataset.data)
        return [AnomalyScore(0.5) for _ in range(n_samples)]


@pytest.fixture
def mock_detector():
    """Create a mock detector for testing."""
    return MockDetector()


@pytest.fixture
def mock_detector_repository():
    """Create a mock detector repository."""
    repo = Mock()
    repo.save = AsyncMock()
    repo.get_by_id = AsyncMock()
    return repo


@pytest.fixture
def sample_performance_metrics():
    """Create sample performance metrics."""
    return PerformanceMetrics(
        accuracy=0.95,
        precision=0.90,
        recall=0.88,
        f1_score=0.89,
        training_time=120.5,
        inference_time=5.2,
        model_size=1024 * 1024,  # 1MB
        roc_auc=0.92,
        memory_usage=256.0,
        cpu_usage=75.0,
    )


@pytest.fixture
def temp_storage_path():
    """Create temporary storage path."""
    with tempfile.TemporaryDirectory() as temp_dir:
        yield Path(temp_dir)


@pytest.fixture
def persistence_service(mock_detector_repository, temp_storage_path):
    """Create enhanced model persistence service."""
    return EnhancedModelPersistenceService(
        detector_repository=mock_detector_repository, storage_path=temp_storage_path
    )


class TestEnhancedModelPersistenceService:
    """Test enhanced model persistence service."""

    @pytest.mark.asyncio
    async def test_serialize_model_pickle_format(
        self, persistence_service, mock_detector, sample_performance_metrics
    ):
        """Test model serialization with pickle format."""
        version = SemanticVersion(1, 0, 0)

        model_version = await persistence_service.serialize_model(
            detector=mock_detector,
            version=version,
            format=SerializationFormat.PICKLE,
            performance_metrics=sample_performance_metrics,
            created_by="test_user",
        )

        assert isinstance(model_version, ModelVersion)
        assert model_version.version == version
        assert model_version.detector_id == mock_detector.id
        assert model_version.created_by == "test_user"
        assert model_version.status == ModelStatus.VALIDATED
        assert model_version.storage_info.format == SerializationFormat.PICKLE
        assert model_version.performance_metrics == sample_performance_metrics

    @pytest.mark.asyncio
    async def test_serialize_model_joblib_format(
        self, persistence_service, mock_detector, sample_performance_metrics
    ):
        """Test model serialization with joblib format."""
        version = SemanticVersion(1, 0, 1)

        model_version = await persistence_service.serialize_model(
            detector=mock_detector,
            version=version,
            format=SerializationFormat.JOBLIB,
            performance_metrics=sample_performance_metrics,
        )

        assert model_version.storage_info.format == SerializationFormat.JOBLIB
        assert model_version.storage_info.storage_path.endswith(".joblib")

    @pytest.mark.asyncio
    async def test_serialize_unfitted_detector_raises_error(self, persistence_service):
        """Test that serializing unfitted detector raises error."""
        unfitted_detector = MockDetector(fitted=False)
        version = SemanticVersion(1, 0, 0)

        with pytest.raises(ModelSerializationError, match="is not fitted"):
            await persistence_service.serialize_model(
                detector=unfitted_detector, version=version
            )

    @pytest.mark.asyncio
    async def test_serialize_with_compression(
        self, persistence_service, mock_detector, sample_performance_metrics
    ):
        """Test model serialization with compression."""
        version = SemanticVersion(1, 0, 0)

        model_version = await persistence_service.serialize_model(
            detector=mock_detector,
            version=version,
            performance_metrics=sample_performance_metrics,
            compress=True,
        )

        assert model_version.storage_info.is_compressed
        assert model_version.storage_info.compression_type == "gzip"
        assert model_version.storage_info.storage_path.endswith(".gz")

    @pytest.mark.asyncio
    async def test_deserialize_model(
        self, persistence_service, mock_detector, sample_performance_metrics
    ):
        """Test model deserialization."""
        version = SemanticVersion(1, 0, 0)

        # Serialize first
        model_version = await persistence_service.serialize_model(
            detector=mock_detector,
            version=version,
            performance_metrics=sample_performance_metrics,
        )

        # Then deserialize
        deserialized_detector = await persistence_service.deserialize_model(
            model_version
        )

        assert isinstance(deserialized_detector, MockDetector)
        assert deserialized_detector.name == mock_detector.name
        assert deserialized_detector.algorithm_name == mock_detector.algorithm_name
        assert deserialized_detector.is_fitted == mock_detector.is_fitted

    @pytest.mark.asyncio
    async def test_deserialize_with_compression(
        self, persistence_service, mock_detector, sample_performance_metrics
    ):
        """Test deserialization of compressed model."""
        version = SemanticVersion(1, 0, 0)

        # Serialize with compression
        model_version = await persistence_service.serialize_model(
            detector=mock_detector,
            version=version,
            performance_metrics=sample_performance_metrics,
            compress=True,
        )

        # Deserialize
        deserialized_detector = await persistence_service.deserialize_model(
            model_version
        )

        assert isinstance(deserialized_detector, MockDetector)
        assert deserialized_detector.name == mock_detector.name

    @pytest.mark.asyncio
    async def test_checksum_verification_success(
        self, persistence_service, mock_detector, sample_performance_metrics
    ):
        """Test successful checksum verification."""
        version = SemanticVersion(1, 0, 0)

        model_version = await persistence_service.serialize_model(
            detector=mock_detector,
            version=version,
            performance_metrics=sample_performance_metrics,
        )

        # Should not raise error with verify_checksum=True
        deserialized_detector = await persistence_service.deserialize_model(
            model_version, verify_checksum=True
        )

        assert isinstance(deserialized_detector, MockDetector)

    @pytest.mark.asyncio
    async def test_list_model_versions(
        self, persistence_service, mock_detector, sample_performance_metrics
    ):
        """Test listing model versions."""
        # Create multiple versions
        versions = [
            SemanticVersion(1, 0, 0),
            SemanticVersion(1, 0, 1),
            SemanticVersion(1, 1, 0),
        ]

        created_versions = []
        for version in versions:
            model_version = await persistence_service.serialize_model(
                detector=mock_detector,
                version=version,
                performance_metrics=sample_performance_metrics,
            )
            created_versions.append(model_version)

        # List all versions
        listed_versions = await persistence_service.list_model_versions()

        assert len(listed_versions) == 3

        # Should be sorted by creation date (newest first)
        version_strings = [v.version_string for v in listed_versions]
        assert "1.1.0" in version_strings
        assert "1.0.1" in version_strings
        assert "1.0.0" in version_strings

    @pytest.mark.asyncio
    async def test_list_model_versions_with_filter(
        self, persistence_service, mock_detector, sample_performance_metrics
    ):
        """Test listing model versions with filters."""
        version = SemanticVersion(1, 0, 0)

        await persistence_service.serialize_model(
            detector=mock_detector,
            version=version,
            performance_metrics=sample_performance_metrics,
        )

        # Filter by model ID
        filtered_versions = await persistence_service.list_model_versions(
            model_id=mock_detector.id
        )

        assert len(filtered_versions) == 1
        assert filtered_versions[0].model_id == mock_detector.id

        # Filter by status
        status_filtered = await persistence_service.list_model_versions(
            status_filter=ModelStatus.VALIDATED
        )

        assert len(status_filtered) == 1
        assert status_filtered[0].status == ModelStatus.VALIDATED

    @pytest.mark.asyncio
    async def test_get_model_version(
        self, persistence_service, mock_detector, sample_performance_metrics
    ):
        """Test getting specific model version."""
        version = SemanticVersion(1, 2, 3)

        await persistence_service.serialize_model(
            detector=mock_detector,
            version=version,
            performance_metrics=sample_performance_metrics,
        )

        # Get by SemanticVersion object
        retrieved_version = await persistence_service.get_model_version(
            model_id=mock_detector.id, version=version
        )

        assert retrieved_version is not None
        assert retrieved_version.version == version
        assert retrieved_version.model_id == mock_detector.id

        # Get by version string
        retrieved_by_string = await persistence_service.get_model_version(
            model_id=mock_detector.id, version="1.2.3"
        )

        assert retrieved_by_string is not None
        assert retrieved_by_string.version == version

    @pytest.mark.asyncio
    async def test_get_latest_version(
        self, persistence_service, mock_detector, sample_performance_metrics
    ):
        """Test getting latest model version."""
        # Create multiple versions
        versions = [
            SemanticVersion(1, 0, 0),
            SemanticVersion(1, 0, 1),  # This should be latest
            SemanticVersion(0, 9, 0),  # Older version
        ]

        for version in versions:
            await persistence_service.serialize_model(
                detector=mock_detector,
                version=version,
                performance_metrics=sample_performance_metrics,
            )

        latest_version = await persistence_service.get_latest_version(
            model_id=mock_detector.id
        )

        assert latest_version is not None
        # Should get the most recently created, not necessarily highest version number
        assert latest_version.model_id == mock_detector.id

    @pytest.mark.asyncio
    async def test_compare_versions(self, persistence_service, mock_detector):
        """Test comparing model versions."""
        # Create two versions with different performance
        metrics1 = PerformanceMetrics(
            accuracy=0.90,
            precision=0.85,
            recall=0.80,
            f1_score=0.82,
            training_time=100.0,
            inference_time=5.0,
            model_size=1024,
        )

        metrics2 = PerformanceMetrics(
            accuracy=0.95,
            precision=0.90,
            recall=0.88,
            f1_score=0.89,
            training_time=120.0,
            inference_time=4.0,
            model_size=2048,
        )

        version1 = await persistence_service.serialize_model(
            detector=mock_detector,
            version=SemanticVersion(1, 0, 0),
            performance_metrics=metrics1,
        )

        version2 = await persistence_service.serialize_model(
            detector=mock_detector,
            version=SemanticVersion(1, 0, 1),
            performance_metrics=metrics2,
        )

        comparison = await persistence_service.compare_versions(version1, version2)

        assert "version1" in comparison
        assert "version2" in comparison
        assert "performance_difference" in comparison
        assert "version1_is_better" in comparison

        # Version2 should be better (higher accuracy, precision, recall, f1)
        assert not comparison["version1_is_better"]

        # Check specific performance differences
        perf_diff = comparison["performance_difference"]
        assert perf_diff["accuracy"] < 0  # version1 has lower accuracy
        assert perf_diff["precision"] < 0
        assert perf_diff["recall"] < 0
        assert perf_diff["f1_score"] < 0

    @pytest.mark.asyncio
    async def test_delete_model_version(
        self, persistence_service, mock_detector, sample_performance_metrics
    ):
        """Test deleting model version."""
        version = SemanticVersion(1, 0, 0)

        model_version = await persistence_service.serialize_model(
            detector=mock_detector,
            version=version,
            performance_metrics=sample_performance_metrics,
        )

        # Delete the version
        deleted = await persistence_service.delete_model_version(model_version)

        assert deleted is True

        # Verify it's gone
        retrieved = await persistence_service.get_model_version(
            model_id=mock_detector.id, version=version
        )

        assert retrieved is None

    @pytest.mark.asyncio
    async def test_delete_deployed_version_without_force_fails(
        self, persistence_service, mock_detector, sample_performance_metrics
    ):
        """Test that deleting deployed version without force fails."""
        version = SemanticVersion(1, 0, 0)

        model_version = await persistence_service.serialize_model(
            detector=mock_detector,
            version=version,
            performance_metrics=sample_performance_metrics,
        )

        # Mark as deployed
        model_version.update_status(ModelStatus.DEPLOYED)

        # Should raise error without force
        with pytest.raises(ValueError, match="Cannot delete deployed version"):
            await persistence_service.delete_model_version(model_version, force=False)

    @pytest.mark.asyncio
    async def test_export_model_bundle(
        self,
        persistence_service,
        mock_detector,
        sample_performance_metrics,
        temp_storage_path,
    ):
        """Test exporting model bundle."""
        version = SemanticVersion(1, 0, 0)

        model_version = await persistence_service.serialize_model(
            detector=mock_detector,
            version=version,
            performance_metrics=sample_performance_metrics,
        )

        export_path = temp_storage_path / "export"

        exported_files = await persistence_service.export_model_bundle(
            model_version=model_version,
            export_path=export_path,
            include_dependencies=True,
            include_examples=True,
        )

        # Check that all expected files were created
        expected_files = [
            "model",
            "config",
            "metadata",
            "requirements",
            "examples",
            "deploy_script",
        ]
        for file_type in expected_files:
            assert file_type in exported_files
            assert Path(exported_files[file_type]).exists()

        # Check requirements file content
        req_path = Path(exported_files["requirements"])
        with open(req_path) as f:
            requirements = f.read()

        assert "pynomaly" in requirements
        assert "numpy" in requirements
        assert "pandas" in requirements
        assert "scikit-learn" in requirements

    @pytest.mark.asyncio
    async def test_create_model_archive(
        self,
        persistence_service,
        mock_detector,
        sample_performance_metrics,
        temp_storage_path,
    ):
        """Test creating model archive."""
        version = SemanticVersion(1, 0, 0)

        model_version = await persistence_service.serialize_model(
            detector=mock_detector,
            version=version,
            performance_metrics=sample_performance_metrics,
        )

        archive_path = temp_storage_path / "model_archive.zip"

        created_archive = await persistence_service.create_model_archive(
            model_version=model_version, archive_path=archive_path
        )

        assert Path(created_archive).exists()
        assert created_archive == str(archive_path)

        # Verify archive contents
        import zipfile

        with zipfile.ZipFile(archive_path, "r") as zf:
            archive_files = zf.namelist()

            expected_files = [
                "model_config.json",
                "metadata.json",
                "requirements.txt",
                "example_usage.py",
                "deploy.py",
            ]

            for expected_file in expected_files:
                assert any(expected_file in f for f in archive_files)

    @pytest.mark.asyncio
    async def test_unsupported_format_raises_error(
        self, persistence_service, mock_detector
    ):
        """Test that unsupported format raises error."""
        version = SemanticVersion(1, 0, 0)

        with pytest.raises(UnsupportedFormatError):
            await persistence_service.serialize_model(
                detector=mock_detector,
                version=version,
                format=SerializationFormat.ONNX,  # Not implemented yet
            )


class TestSemanticVersion:
    """Test semantic version value object."""

    def test_version_string_property(self):
        """Test version string property."""
        version = SemanticVersion(1, 2, 3)
        assert version.version_string == "1.2.3"

    def test_from_string_valid(self):
        """Test creating version from valid string."""
        version = SemanticVersion.from_string("2.5.10")
        assert version.major == 2
        assert version.minor == 5
        assert version.patch == 10

    def test_from_string_with_v_prefix(self):
        """Test creating version from string with v prefix."""
        version = SemanticVersion.from_string("v1.0.0")
        assert version.major == 1
        assert version.minor == 0
        assert version.patch == 0

    def test_from_string_invalid_raises_error(self):
        """Test that invalid version string raises error."""
        with pytest.raises(ValueError, match="Invalid version string"):
            SemanticVersion.from_string("1.2")

        with pytest.raises(ValueError, match="Invalid version string"):
            SemanticVersion.from_string("1.2.3.4")

        with pytest.raises(ValueError, match="Invalid version string"):
            SemanticVersion.from_string("abc.def.ghi")

    def test_increment_major(self):
        """Test incrementing major version."""
        version = SemanticVersion(1, 2, 3)
        new_version = version.increment_major()

        assert new_version.major == 2
        assert new_version.minor == 0
        assert new_version.patch == 0

    def test_increment_minor(self):
        """Test incrementing minor version."""
        version = SemanticVersion(1, 2, 3)
        new_version = version.increment_minor()

        assert new_version.major == 1
        assert new_version.minor == 3
        assert new_version.patch == 0

    def test_increment_patch(self):
        """Test incrementing patch version."""
        version = SemanticVersion(1, 2, 3)
        new_version = version.increment_patch()

        assert new_version.major == 1
        assert new_version.minor == 2
        assert new_version.patch == 4

    def test_is_compatible_with(self):
        """Test version compatibility check."""
        v1_0_0 = SemanticVersion(1, 0, 0)
        v1_0_1 = SemanticVersion(1, 0, 1)
        v1_1_0 = SemanticVersion(1, 1, 0)
        v2_0_0 = SemanticVersion(2, 0, 0)

        # Same major version - compatible
        assert v1_1_0.is_compatible_with(v1_0_0)
        assert v1_0_1.is_compatible_with(v1_0_0)

        # Different major version - not compatible
        assert not v2_0_0.is_compatible_with(v1_0_0)
        assert not v1_0_0.is_compatible_with(v2_0_0)

    def test_is_newer_than(self):
        """Test version comparison."""
        v1_0_0 = SemanticVersion(1, 0, 0)
        v1_0_1 = SemanticVersion(1, 0, 1)
        v1_1_0 = SemanticVersion(1, 1, 0)
        v2_0_0 = SemanticVersion(2, 0, 0)

        assert v1_0_1.is_newer_than(v1_0_0)
        assert v1_1_0.is_newer_than(v1_0_1)
        assert v2_0_0.is_newer_than(v1_1_0)

        assert not v1_0_0.is_newer_than(v1_0_1)

    def test_comparison_operators(self):
        """Test comparison operators."""
        v1 = SemanticVersion(1, 0, 0)
        v2 = SemanticVersion(1, 0, 1)
        v3 = SemanticVersion(1, 0, 1)  # Same as v2

        assert v1 < v2
        assert v2 > v1
        assert v2 == v3
        assert v2 >= v3
        assert v1 <= v2

    def test_is_prerelease(self):
        """Test prerelease version check."""
        prerelease = SemanticVersion(0, 1, 0)
        stable = SemanticVersion(1, 0, 0)

        assert prerelease.is_prerelease()
        assert not stable.is_prerelease()

    def test_is_stable(self):
        """Test stable version check."""
        prerelease = SemanticVersion(0, 1, 0)
        stable = SemanticVersion(1, 0, 0)

        assert not prerelease.is_stable()
        assert stable.is_stable()


if __name__ == "__main__":
    pytest.main([__file__])
