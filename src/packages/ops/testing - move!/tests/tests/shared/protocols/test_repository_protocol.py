"""
Tests for the repository protocol implementations.

This module tests the RepositoryProtocol and all specialized repository protocols
to ensure proper contract enforcement and runtime behavior checking.
"""

import asyncio
from uuid import UUID, uuid4

import pytest

from monorepo.shared.protocols.repository_protocol import (
    AlertNotificationRepositoryProtocol,
    AlertRepositoryProtocol,
    DatasetRepositoryProtocol,
    DetectionResultRepositoryProtocol,
    DetectorRepositoryProtocol,
    ExperimentRepositoryProtocol,
    ExperimentRunRepositoryProtocol,
    ModelRepositoryProtocol,
    ModelVersionRepositoryProtocol,
    PipelineRepositoryProtocol,
    PipelineRunRepositoryProtocol,
    RepositoryProtocol,
)


# Mock domain entities for testing
class MockEntity:
    """Base mock entity for testing."""

    def __init__(self, id: UUID = None, name: str = "test_entity"):
        self.id = id or uuid4()
        self.name = name


class MockDetector(MockEntity):
    """Mock detector entity."""

    def __init__(
        self,
        id: UUID = None,
        name: str = "test_detector",
        algorithm: str = "isolation_forest",
    ):
        super().__init__(id, name)
        self.algorithm = algorithm
        self.is_fitted = False


class MockDataset(MockEntity):
    """Mock dataset entity."""

    def __init__(self, id: UUID = None, name: str = "test_dataset"):
        super().__init__(id, name)
        self.metadata = {"rows": 1000, "columns": 5}


class MockDetectionResult(MockEntity):
    """Mock detection result entity."""

    def __init__(
        self, id: UUID = None, detector_id: UUID = None, dataset_id: UUID = None
    ):
        super().__init__(id, "detection_result")
        self.detector_id = detector_id or uuid4()
        self.dataset_id = dataset_id or uuid4()
        self.anomalies = []


class MockModel(MockEntity):
    """Mock model entity."""

    def __init__(self, id: UUID = None, name: str = "test_model"):
        super().__init__(id, name)
        self.stage = "development"
        self.model_type = "anomaly_detector"


class MockModelVersion(MockEntity):
    """Mock model version entity."""

    def __init__(self, id: UUID = None, model_id: UUID = None, version: str = "1.0.0"):
        super().__init__(id, f"model_version_{version}")
        self.model_id = model_id or uuid4()
        self.version = version


class MockExperiment(MockEntity):
    """Mock experiment entity."""

    def __init__(self, id: UUID = None, name: str = "test_experiment"):
        super().__init__(id, name)
        self.status = "running"
        self.experiment_type = "hyperparameter_tuning"


class MockExperimentRun(MockEntity):
    """Mock experiment run entity."""

    def __init__(self, id: UUID = None, experiment_id: UUID = None):
        super().__init__(id, "experiment_run")
        self.experiment_id = experiment_id or uuid4()
        self.status = "completed"


class MockPipeline(MockEntity):
    """Mock pipeline entity."""

    def __init__(self, id: UUID = None, name: str = "test_pipeline"):
        super().__init__(id, name)
        self.environment = "development"
        self.status = "active"
        self.pipeline_type = "training"


class MockPipelineRun(MockEntity):
    """Mock pipeline run entity."""

    def __init__(self, id: UUID = None, pipeline_id: UUID = None):
        super().__init__(id, "pipeline_run")
        self.pipeline_id = pipeline_id or uuid4()
        self.status = "completed"


class MockAlert(MockEntity):
    """Mock alert entity."""

    def __init__(self, id: UUID = None, name: str = "test_alert"):
        super().__init__(id, name)
        self.status = "active"
        self.alert_type = "threshold"
        self.severity = "high"


class MockAlertNotification(MockEntity):
    """Mock alert notification entity."""

    def __init__(self, id: UUID = None, alert_id: UUID = None):
        super().__init__(id, "alert_notification")
        self.alert_id = alert_id or uuid4()
        self.status = "sent"


# Base repository implementation for testing
class MockRepository(RepositoryProtocol[MockEntity]):
    """Mock implementation of RepositoryProtocol for testing."""

    def __init__(self):
        self.storage: dict[UUID, MockEntity] = {}
        self.operation_count = 0
        self.should_fail_operations = False

    async def save(self, entity: MockEntity) -> None:
        """Save an entity to the repository."""
        if self.should_fail_operations:
            raise ValueError("Save operation failed")

        self.operation_count += 1
        self.storage[entity.id] = entity

    async def find_by_id(self, entity_id: UUID) -> MockEntity | None:
        """Find an entity by its ID."""
        if self.should_fail_operations:
            raise ValueError("Find operation failed")

        self.operation_count += 1
        return self.storage.get(entity_id)

    async def find_all(self) -> list[MockEntity]:
        """Find all entities in the repository."""
        if self.should_fail_operations:
            raise ValueError("Find all operation failed")

        self.operation_count += 1
        return list(self.storage.values())

    async def delete(self, entity_id: UUID) -> bool:
        """Delete an entity by its ID."""
        if self.should_fail_operations:
            raise ValueError("Delete operation failed")

        self.operation_count += 1
        if entity_id in self.storage:
            del self.storage[entity_id]
            return True
        return False

    async def exists(self, entity_id: UUID) -> bool:
        """Check if an entity exists."""
        if self.should_fail_operations:
            raise ValueError("Exists operation failed")

        self.operation_count += 1
        return entity_id in self.storage

    async def count(self) -> int:
        """Count total number of entities."""
        if self.should_fail_operations:
            raise ValueError("Count operation failed")

        self.operation_count += 1
        return len(self.storage)


# Specialized repository implementations for testing
class MockDetectorRepository(DetectorRepositoryProtocol):
    """Mock detector repository implementation."""

    def __init__(self):
        self.storage: dict[UUID, MockDetector] = {}
        self.operation_count = 0
        self.should_fail_operations = False

    async def save(self, entity: MockDetector) -> None:
        self.operation_count += 1
        self.storage[entity.id] = entity

    async def find_by_id(self, entity_id: UUID) -> MockDetector | None:
        self.operation_count += 1
        return self.storage.get(entity_id)

    async def find_all(self) -> list[MockDetector]:
        self.operation_count += 1
        return list(self.storage.values())

    async def delete(self, entity_id: UUID) -> bool:
        self.operation_count += 1
        if entity_id in self.storage:
            del self.storage[entity_id]
            return True
        return False

    async def exists(self, entity_id: UUID) -> bool:
        self.operation_count += 1
        return entity_id in self.storage

    async def count(self) -> int:
        self.operation_count += 1
        return len(self.storage)

    # Specialized methods
    async def find_by_name(self, name: str) -> MockDetector | None:
        self.operation_count += 1
        for detector in self.storage.values():
            if detector.name == name:
                return detector
        return None

    async def find_by_algorithm(self, algorithm_name: str) -> list[MockDetector]:
        self.operation_count += 1
        return [d for d in self.storage.values() if d.algorithm == algorithm_name]

    async def find_fitted(self) -> list[MockDetector]:
        self.operation_count += 1
        return [d for d in self.storage.values() if d.is_fitted]

    async def save_model_artifact(self, detector_id: UUID, artifact: bytes) -> None:
        self.operation_count += 1
        # Mock implementation - would save to storage in real implementation
        pass

    async def load_model_artifact(self, detector_id: UUID) -> bytes | None:
        self.operation_count += 1
        # Mock implementation - would load from storage in real implementation
        return b"mock_artifact_data"


@pytest.mark.asyncio
class TestRepositoryProtocol:
    """Test suite for base RepositoryProtocol contract enforcement."""

    @pytest.fixture
    def mock_repository(self):
        """Create mock repository for testing."""
        return MockRepository()

    @pytest.fixture
    def sample_entity(self):
        """Create sample entity for testing."""
        return MockEntity(name="test_entity")

    async def test_protocol_compliance(self, mock_repository):
        """Test that mock repository implements the protocol correctly."""
        assert isinstance(mock_repository, RepositoryProtocol)

        # Check that all required methods exist
        required_methods = [
            "save",
            "find_by_id",
            "find_all",
            "delete",
            "exists",
            "count",
        ]
        for method in required_methods:
            assert hasattr(mock_repository, method)
            assert callable(getattr(mock_repository, method))

    async def test_save_operation(self, mock_repository, sample_entity):
        """Test save operation."""
        await mock_repository.save(sample_entity)

        assert mock_repository.operation_count == 1
        assert sample_entity.id in mock_repository.storage
        assert mock_repository.storage[sample_entity.id] == sample_entity

    async def test_find_by_id_success(self, mock_repository, sample_entity):
        """Test successful find by ID operation."""
        await mock_repository.save(sample_entity)

        found_entity = await mock_repository.find_by_id(sample_entity.id)

        assert found_entity is not None
        assert found_entity.id == sample_entity.id
        assert found_entity.name == sample_entity.name

    async def test_find_by_id_not_found(self, mock_repository):
        """Test find by ID when entity doesn't exist."""
        non_existent_id = uuid4()

        found_entity = await mock_repository.find_by_id(non_existent_id)

        assert found_entity is None

    async def test_find_all_empty(self, mock_repository):
        """Test find all when repository is empty."""
        entities = await mock_repository.find_all()

        assert isinstance(entities, list)
        assert len(entities) == 0

    async def test_find_all_with_entities(self, mock_repository):
        """Test find all with multiple entities."""
        entities = [MockEntity(name=f"entity_{i}") for i in range(3)]

        for entity in entities:
            await mock_repository.save(entity)

        found_entities = await mock_repository.find_all()

        assert len(found_entities) == 3
        assert all(isinstance(e, MockEntity) for e in found_entities)

    async def test_delete_success(self, mock_repository, sample_entity):
        """Test successful delete operation."""
        await mock_repository.save(sample_entity)

        result = await mock_repository.delete(sample_entity.id)

        assert result is True
        assert sample_entity.id not in mock_repository.storage

    async def test_delete_not_found(self, mock_repository):
        """Test delete when entity doesn't exist."""
        non_existent_id = uuid4()

        result = await mock_repository.delete(non_existent_id)

        assert result is False

    async def test_exists_true(self, mock_repository, sample_entity):
        """Test exists when entity exists."""
        await mock_repository.save(sample_entity)

        exists = await mock_repository.exists(sample_entity.id)

        assert exists is True

    async def test_exists_false(self, mock_repository):
        """Test exists when entity doesn't exist."""
        non_existent_id = uuid4()

        exists = await mock_repository.exists(non_existent_id)

        assert exists is False

    async def test_count_empty(self, mock_repository):
        """Test count when repository is empty."""
        count = await mock_repository.count()

        assert count == 0

    async def test_count_with_entities(self, mock_repository):
        """Test count with multiple entities."""
        entities = [MockEntity(name=f"entity_{i}") for i in range(5)]

        for entity in entities:
            await mock_repository.save(entity)

        count = await mock_repository.count()

        assert count == 5

    async def test_operation_error_handling(self, mock_repository, sample_entity):
        """Test error handling in repository operations."""
        mock_repository.should_fail_operations = True

        # Test that all operations raise errors when configured to fail
        with pytest.raises(ValueError, match="Save operation failed"):
            await mock_repository.save(sample_entity)

        with pytest.raises(ValueError, match="Find operation failed"):
            await mock_repository.find_by_id(sample_entity.id)

        with pytest.raises(ValueError, match="Find all operation failed"):
            await mock_repository.find_all()

        with pytest.raises(ValueError, match="Delete operation failed"):
            await mock_repository.delete(sample_entity.id)

        with pytest.raises(ValueError, match="Exists operation failed"):
            await mock_repository.exists(sample_entity.id)

        with pytest.raises(ValueError, match="Count operation failed"):
            await mock_repository.count()

    async def test_concurrent_operations(self, mock_repository):
        """Test concurrent repository operations."""
        entities = [MockEntity(name=f"concurrent_entity_{i}") for i in range(10)]

        # Save entities concurrently
        save_tasks = [mock_repository.save(entity) for entity in entities]
        await asyncio.gather(*save_tasks)

        # Verify all entities were saved
        count = await mock_repository.count()
        assert count == 10

        # Find entities concurrently
        find_tasks = [mock_repository.find_by_id(entity.id) for entity in entities]
        found_entities = await asyncio.gather(*find_tasks)

        assert len(found_entities) == 10
        assert all(entity is not None for entity in found_entities)


@pytest.mark.asyncio
class TestDetectorRepositoryProtocol:
    """Test suite for DetectorRepositoryProtocol."""

    @pytest.fixture
    def detector_repository(self):
        """Create mock detector repository for testing."""
        return MockDetectorRepository()

    @pytest.fixture
    def sample_detectors(self):
        """Create sample detectors for testing."""
        return [
            MockDetector(name="detector_1", algorithm="isolation_forest"),
            MockDetector(name="detector_2", algorithm="one_class_svm"),
            MockDetector(name="detector_3", algorithm="isolation_forest"),
        ]

    async def test_protocol_compliance(self, detector_repository):
        """Test that detector repository implements the protocol correctly."""
        assert isinstance(detector_repository, DetectorRepositoryProtocol)
        assert isinstance(detector_repository, RepositoryProtocol)

        # Check specialized methods
        specialized_methods = [
            "find_by_name",
            "find_by_algorithm",
            "find_fitted",
            "save_model_artifact",
            "load_model_artifact",
        ]
        for method in specialized_methods:
            assert hasattr(detector_repository, method)
            assert callable(getattr(detector_repository, method))

    async def test_find_by_name_success(self, detector_repository, sample_detectors):
        """Test successful find by name."""
        detector = sample_detectors[0]
        await detector_repository.save(detector)

        found_detector = await detector_repository.find_by_name(detector.name)

        assert found_detector is not None
        assert found_detector.name == detector.name
        assert found_detector.id == detector.id

    async def test_find_by_name_not_found(self, detector_repository):
        """Test find by name when detector doesn't exist."""
        found_detector = await detector_repository.find_by_name("nonexistent")

        assert found_detector is None

    async def test_find_by_algorithm(self, detector_repository, sample_detectors):
        """Test find by algorithm."""
        for detector in sample_detectors:
            await detector_repository.save(detector)

        # Find isolation forest detectors
        isolation_detectors = await detector_repository.find_by_algorithm(
            "isolation_forest"
        )

        assert len(isolation_detectors) == 2
        assert all(d.algorithm == "isolation_forest" for d in isolation_detectors)

        # Find one class SVM detectors
        svm_detectors = await detector_repository.find_by_algorithm("one_class_svm")

        assert len(svm_detectors) == 1
        assert svm_detectors[0].algorithm == "one_class_svm"

    async def test_find_fitted(self, detector_repository, sample_detectors):
        """Test find fitted detectors."""
        # Mark some detectors as fitted
        sample_detectors[0].is_fitted = True
        sample_detectors[2].is_fitted = True

        for detector in sample_detectors:
            await detector_repository.save(detector)

        fitted_detectors = await detector_repository.find_fitted()

        assert len(fitted_detectors) == 2
        assert all(d.is_fitted for d in fitted_detectors)

    async def test_model_artifact_operations(
        self, detector_repository, sample_detectors
    ):
        """Test model artifact save and load operations."""
        detector = sample_detectors[0]
        await detector_repository.save(detector)

        artifact_data = b"test_model_artifact_data"

        # Save artifact
        await detector_repository.save_model_artifact(detector.id, artifact_data)

        # Load artifact
        loaded_artifact = await detector_repository.load_model_artifact(detector.id)

        assert loaded_artifact is not None
        assert isinstance(loaded_artifact, bytes)


@pytest.mark.asyncio
class TestRepositoryProtocolEnforcement:
    """Test protocol enforcement and runtime checking."""

    def test_incomplete_implementation_detection(self):
        """Test that incomplete implementations are detected."""

        class IncompleteRepository:
            """Incomplete repository missing required methods."""

            async def save(self, entity):
                pass

            # Missing other required methods

        incomplete = IncompleteRepository()

        # Should not be considered a valid implementation
        assert not isinstance(incomplete, RepositoryProtocol)

    def test_protocol_method_signatures(self):
        """Test that protocols define correct method signatures."""
        # Test base protocol
        base_methods = (
            RepositoryProtocol.__annotations__.keys()
            if hasattr(RepositoryProtocol, "__annotations__")
            else set()
        )

        # Test specialized protocols exist
        specialized_protocols = [
            DetectorRepositoryProtocol,
            DatasetRepositoryProtocol,
            DetectionResultRepositoryProtocol,
            ModelRepositoryProtocol,
            ModelVersionRepositoryProtocol,
            ExperimentRepositoryProtocol,
            ExperimentRunRepositoryProtocol,
            PipelineRepositoryProtocol,
            PipelineRunRepositoryProtocol,
            AlertRepositoryProtocol,
            AlertNotificationRepositoryProtocol,
        ]

        for protocol in specialized_protocols:
            assert issubclass(protocol, RepositoryProtocol)

    def test_protocol_runtime_checking(self):
        """Test runtime protocol checking works correctly."""
        mock_repo = MockRepository()
        mock_detector_repo = MockDetectorRepository()

        # Should pass runtime checks
        assert isinstance(mock_repo, RepositoryProtocol)
        assert isinstance(mock_detector_repo, DetectorRepositoryProtocol)
        assert isinstance(mock_detector_repo, RepositoryProtocol)

        # Protocol methods should be callable
        base_methods = ["save", "find_by_id", "find_all", "delete", "exists", "count"]
        for method in base_methods:
            assert callable(getattr(mock_repo, method))
            assert callable(getattr(mock_detector_repo, method))

        # Specialized methods should be callable
        specialized_methods = [
            "find_by_name",
            "find_by_algorithm",
            "find_fitted",
            "save_model_artifact",
            "load_model_artifact",
        ]
        for method in specialized_methods:
            assert callable(getattr(mock_detector_repo, method))


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
