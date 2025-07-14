"""
Comprehensive tests for Repository Protocol definitions.
Tests protocol conformance, type checking, and CRUD operation contracts.
"""

import inspect
from unittest.mock import AsyncMock, Mock
from uuid import UUID, uuid4

import pytest

from pynomaly.domain.entities import (
    Alert,
    Dataset,
    DetectionResult,
    Detector,
    Experiment,
    Model,
    Pipeline,
)
from pynomaly.shared.protocols.repository_protocol import (
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


class TestRepositoryProtocol:
    """Test suite for base RepositoryProtocol."""

    def test_protocol_is_runtime_checkable(self):
        """Test that RepositoryProtocol is runtime checkable."""
        assert hasattr(RepositoryProtocol, "__runtime_checkable__")
        assert RepositoryProtocol.__runtime_checkable__ is True

    def test_protocol_defines_crud_methods(self):
        """Test that protocol defines all CRUD methods."""
        crud_methods = ["save", "find_by_id", "find_all", "delete", "exists", "count"]

        for method in crud_methods:
            assert hasattr(RepositoryProtocol, method)
            assert callable(getattr(RepositoryProtocol, method))

    def test_protocol_method_signatures(self):
        """Test that protocol methods have correct signatures."""
        # Test save method signature
        save_sig = inspect.signature(RepositoryProtocol.save)
        assert "entity" in save_sig.parameters
        assert save_sig.return_annotation is None or save_sig.return_annotation == type(
            None
        )

        # Test find_by_id signature
        find_by_id_sig = inspect.signature(RepositoryProtocol.find_by_id)
        assert "entity_id" in find_by_id_sig.parameters

        # Test delete signature
        delete_sig = inspect.signature(RepositoryProtocol.delete)
        assert "entity_id" in delete_sig.parameters
        assert delete_sig.return_annotation == bool

        # Test count signature
        count_sig = inspect.signature(RepositoryProtocol.count)
        assert count_sig.return_annotation == int

    def test_protocol_is_generic(self):
        """Test that RepositoryProtocol is generic."""
        # Should have a type parameter
        assert hasattr(RepositoryProtocol, "__parameters__")

    @pytest.mark.asyncio
    async def test_mock_repository_implementation(self):
        """Test mock repository implementation."""
        mock_repo = AsyncMock(spec=RepositoryProtocol[Dataset])
        mock_entity = Mock(spec=Dataset)
        entity_id = uuid4()

        # Configure mock returns
        mock_repo.save.return_value = None
        mock_repo.find_by_id.return_value = mock_entity
        mock_repo.find_all.return_value = [mock_entity]
        mock_repo.delete.return_value = True
        mock_repo.exists.return_value = True
        mock_repo.count.return_value = 5

        # Test CRUD operations
        await mock_repo.save(mock_entity)
        mock_repo.save.assert_called_with(mock_entity)

        found_entity = await mock_repo.find_by_id(entity_id)
        assert found_entity == mock_entity

        all_entities = await mock_repo.find_all()
        assert len(all_entities) == 1
        assert all_entities[0] == mock_entity

        deleted = await mock_repo.delete(entity_id)
        assert deleted is True

        exists = await mock_repo.exists(entity_id)
        assert exists is True

        count = await mock_repo.count()
        assert count == 5

    @pytest.mark.asyncio
    async def test_concrete_repository_implementation(self):
        """Test a concrete repository implementation."""

        class ConcreteRepository:
            def __init__(self):
                self._entities = {}

            async def save(self, entity) -> None:
                self._entities[entity.id] = entity

            async def find_by_id(self, entity_id: UUID):
                return self._entities.get(entity_id)

            async def find_all(self) -> list:
                return list(self._entities.values())

            async def delete(self, entity_id: UUID) -> bool:
                if entity_id in self._entities:
                    del self._entities[entity_id]
                    return True
                return False

            async def exists(self, entity_id: UUID) -> bool:
                return entity_id in self._entities

            async def count(self) -> int:
                return len(self._entities)

        repo = ConcreteRepository()
        assert isinstance(repo, RepositoryProtocol)

        # Test with mock entity
        mock_entity = Mock()
        mock_entity.id = uuid4()

        # Test save
        await repo.save(mock_entity)
        assert await repo.count() == 1

        # Test find_by_id
        found = await repo.find_by_id(mock_entity.id)
        assert found == mock_entity

        # Test exists
        assert await repo.exists(mock_entity.id) is True

        # Test find_all
        all_entities = await repo.find_all()
        assert len(all_entities) == 1

        # Test delete
        deleted = await repo.delete(mock_entity.id)
        assert deleted is True
        assert await repo.count() == 0


class TestDetectorRepositoryProtocol:
    """Test suite for DetectorRepositoryProtocol."""

    def test_inherits_from_repository_protocol(self):
        """Test that DetectorRepositoryProtocol inherits from RepositoryProtocol."""
        assert issubclass(DetectorRepositoryProtocol, RepositoryProtocol)

    def test_protocol_defines_detector_specific_methods(self):
        """Test protocol defines detector-specific methods."""
        detector_methods = [
            "find_by_name",
            "find_by_algorithm",
            "find_fitted",
            "save_model_artifact",
            "load_model_artifact",
        ]

        for method in detector_methods:
            assert hasattr(DetectorRepositoryProtocol, method)
            assert callable(getattr(DetectorRepositoryProtocol, method))

    def test_detector_method_signatures(self):
        """Test detector-specific method signatures."""
        # Test find_by_name signature
        find_by_name_sig = inspect.signature(DetectorRepositoryProtocol.find_by_name)
        assert "name" in find_by_name_sig.parameters

        # Test find_by_algorithm signature
        find_by_algo_sig = inspect.signature(
            DetectorRepositoryProtocol.find_by_algorithm
        )
        assert "algorithm_name" in find_by_algo_sig.parameters

        # Test save_model_artifact signature
        save_artifact_sig = inspect.signature(
            DetectorRepositoryProtocol.save_model_artifact
        )
        assert "detector_id" in save_artifact_sig.parameters
        assert "artifact" in save_artifact_sig.parameters

    @pytest.mark.asyncio
    async def test_mock_detector_repository(self):
        """Test mock detector repository implementation."""
        mock_repo = AsyncMock(spec=DetectorRepositoryProtocol)
        mock_detector = Mock(spec=Detector)
        detector_id = uuid4()

        # Configure detector-specific returns
        mock_repo.find_by_name.return_value = mock_detector
        mock_repo.find_by_algorithm.return_value = [mock_detector]
        mock_repo.find_fitted.return_value = [mock_detector]
        mock_repo.save_model_artifact.return_value = None
        mock_repo.load_model_artifact.return_value = b"model_data"

        # Test detector-specific operations
        found_by_name = await mock_repo.find_by_name("test_detector")
        assert found_by_name == mock_detector

        found_by_algo = await mock_repo.find_by_algorithm("IsolationForest")
        assert len(found_by_algo) == 1
        assert found_by_algo[0] == mock_detector

        fitted_detectors = await mock_repo.find_fitted()
        assert len(fitted_detectors) == 1

        await mock_repo.save_model_artifact(detector_id, b"model_data")
        mock_repo.save_model_artifact.assert_called_with(detector_id, b"model_data")

        artifact = await mock_repo.load_model_artifact(detector_id)
        assert artifact == b"model_data"


class TestDatasetRepositoryProtocol:
    """Test suite for DatasetRepositoryProtocol."""

    def test_inherits_from_repository_protocol(self):
        """Test that DatasetRepositoryProtocol inherits from RepositoryProtocol."""
        assert issubclass(DatasetRepositoryProtocol, RepositoryProtocol)

    def test_protocol_defines_dataset_specific_methods(self):
        """Test protocol defines dataset-specific methods."""
        dataset_methods = ["find_by_name", "find_by_metadata", "save_data", "load_data"]

        for method in dataset_methods:
            assert hasattr(DatasetRepositoryProtocol, method)
            assert callable(getattr(DatasetRepositoryProtocol, method))

    @pytest.mark.asyncio
    async def test_mock_dataset_repository(self):
        """Test mock dataset repository implementation."""
        mock_repo = AsyncMock(spec=DatasetRepositoryProtocol)
        mock_dataset = Mock(spec=Dataset)
        dataset_id = uuid4()

        # Configure dataset-specific returns
        mock_repo.find_by_name.return_value = mock_dataset
        mock_repo.find_by_metadata.return_value = [mock_dataset]
        mock_repo.save_data.return_value = "/path/to/data.parquet"
        mock_repo.load_data.return_value = mock_dataset

        # Test dataset-specific operations
        found_by_name = await mock_repo.find_by_name("test_dataset")
        assert found_by_name == mock_dataset

        found_by_metadata = await mock_repo.find_by_metadata("source", "csv")
        assert len(found_by_metadata) == 1

        saved_path = await mock_repo.save_data(dataset_id, "parquet")
        assert saved_path == "/path/to/data.parquet"

        loaded_dataset = await mock_repo.load_data(dataset_id)
        assert loaded_dataset == mock_dataset


class TestDetectionResultRepositoryProtocol:
    """Test suite for DetectionResultRepositoryProtocol."""

    def test_inherits_from_repository_protocol(self):
        """Test that DetectionResultRepositoryProtocol inherits from RepositoryProtocol."""
        assert issubclass(DetectionResultRepositoryProtocol, RepositoryProtocol)

    def test_protocol_defines_result_specific_methods(self):
        """Test protocol defines result-specific methods."""
        result_methods = [
            "find_by_detector",
            "find_by_dataset",
            "find_recent",
            "get_summary_stats",
        ]

        for method in result_methods:
            assert hasattr(DetectionResultRepositoryProtocol, method)
            assert callable(getattr(DetectionResultRepositoryProtocol, method))

    @pytest.mark.asyncio
    async def test_mock_detection_result_repository(self):
        """Test mock detection result repository implementation."""
        mock_repo = AsyncMock(spec=DetectionResultRepositoryProtocol)
        mock_result = Mock(spec=DetectionResult)
        detector_id = uuid4()
        dataset_id = uuid4()
        result_id = uuid4()

        # Configure result-specific returns
        mock_repo.find_by_detector.return_value = [mock_result]
        mock_repo.find_by_dataset.return_value = [mock_result]
        mock_repo.find_recent.return_value = [mock_result]
        mock_repo.get_summary_stats.return_value = {
            "total_anomalies": 5,
            "avg_score": 0.7,
        }

        # Test result-specific operations
        detector_results = await mock_repo.find_by_detector(detector_id)
        assert len(detector_results) == 1

        dataset_results = await mock_repo.find_by_dataset(dataset_id)
        assert len(dataset_results) == 1

        recent_results = await mock_repo.find_recent(limit=10)
        assert len(recent_results) == 1

        stats = await mock_repo.get_summary_stats(result_id)
        assert stats["total_anomalies"] == 5
        assert stats["avg_score"] == 0.7


class TestModelRepositoryProtocol:
    """Test suite for ModelRepositoryProtocol."""

    def test_inherits_from_repository_protocol(self):
        """Test that ModelRepositoryProtocol inherits from RepositoryProtocol."""
        assert issubclass(ModelRepositoryProtocol, RepositoryProtocol)

    def test_protocol_defines_model_specific_methods(self):
        """Test protocol defines model-specific methods."""
        model_methods = ["find_by_name", "find_by_stage", "find_by_type"]

        for method in model_methods:
            assert hasattr(ModelRepositoryProtocol, method)
            assert callable(getattr(ModelRepositoryProtocol, method))

    @pytest.mark.asyncio
    async def test_mock_model_repository(self):
        """Test mock model repository implementation."""
        mock_repo = AsyncMock(spec=ModelRepositoryProtocol)
        mock_model = Mock(spec=Model)

        # Configure model-specific returns
        mock_repo.find_by_name.return_value = [mock_model]
        mock_repo.find_by_stage.return_value = [mock_model]
        mock_repo.find_by_type.return_value = [mock_model]

        # Test model-specific operations
        models_by_name = await mock_repo.find_by_name("test_model")
        assert len(models_by_name) == 1

        models_by_stage = await mock_repo.find_by_stage("production")
        assert len(models_by_stage) == 1

        models_by_type = await mock_repo.find_by_type("anomaly_detection")
        assert len(models_by_type) == 1


class TestExperimentRepositoryProtocol:
    """Test suite for ExperimentRepositoryProtocol."""

    def test_inherits_from_repository_protocol(self):
        """Test that ExperimentRepositoryProtocol inherits from RepositoryProtocol."""
        assert issubclass(ExperimentRepositoryProtocol, RepositoryProtocol)

    def test_protocol_defines_experiment_specific_methods(self):
        """Test protocol defines experiment-specific methods."""
        experiment_methods = ["find_by_name", "find_by_status", "find_by_type"]

        for method in experiment_methods:
            assert hasattr(ExperimentRepositoryProtocol, method)
            assert callable(getattr(ExperimentRepositoryProtocol, method))

    @pytest.mark.asyncio
    async def test_mock_experiment_repository(self):
        """Test mock experiment repository implementation."""
        mock_repo = AsyncMock(spec=ExperimentRepositoryProtocol)
        mock_experiment = Mock(spec=Experiment)

        # Configure experiment-specific returns
        mock_repo.find_by_name.return_value = [mock_experiment]
        mock_repo.find_by_status.return_value = [mock_experiment]
        mock_repo.find_by_type.return_value = [mock_experiment]

        # Test experiment-specific operations
        experiments_by_name = await mock_repo.find_by_name("test_experiment")
        assert len(experiments_by_name) == 1

        experiments_by_status = await mock_repo.find_by_status("running")
        assert len(experiments_by_status) == 1

        experiments_by_type = await mock_repo.find_by_type("hyperparameter_tuning")
        assert len(experiments_by_type) == 1


class TestPipelineRepositoryProtocol:
    """Test suite for PipelineRepositoryProtocol."""

    def test_inherits_from_repository_protocol(self):
        """Test that PipelineRepositoryProtocol inherits from RepositoryProtocol."""
        assert issubclass(PipelineRepositoryProtocol, RepositoryProtocol)

    def test_protocol_defines_pipeline_specific_methods(self):
        """Test protocol defines pipeline-specific methods."""
        pipeline_methods = [
            "find_by_name",
            "find_by_name_and_environment",
            "find_by_status",
            "find_by_type",
        ]

        for method in pipeline_methods:
            assert hasattr(PipelineRepositoryProtocol, method)
            assert callable(getattr(PipelineRepositoryProtocol, method))

    @pytest.mark.asyncio
    async def test_mock_pipeline_repository(self):
        """Test mock pipeline repository implementation."""
        mock_repo = AsyncMock(spec=PipelineRepositoryProtocol)
        mock_pipeline = Mock(spec=Pipeline)

        # Configure pipeline-specific returns
        mock_repo.find_by_name.return_value = [mock_pipeline]
        mock_repo.find_by_name_and_environment.return_value = [mock_pipeline]
        mock_repo.find_by_status.return_value = [mock_pipeline]
        mock_repo.find_by_type.return_value = [mock_pipeline]

        # Test pipeline-specific operations
        pipelines_by_name = await mock_repo.find_by_name("test_pipeline")
        assert len(pipelines_by_name) == 1

        pipelines_by_env = await mock_repo.find_by_name_and_environment(
            "test_pipeline", "production"
        )
        assert len(pipelines_by_env) == 1

        pipelines_by_status = await mock_repo.find_by_status("active")
        assert len(pipelines_by_status) == 1

        pipelines_by_type = await mock_repo.find_by_type("training")
        assert len(pipelines_by_type) == 1


class TestAlertRepositoryProtocol:
    """Test suite for AlertRepositoryProtocol."""

    def test_inherits_from_repository_protocol(self):
        """Test that AlertRepositoryProtocol inherits from RepositoryProtocol."""
        assert issubclass(AlertRepositoryProtocol, RepositoryProtocol)

    def test_protocol_defines_alert_specific_methods(self):
        """Test protocol defines alert-specific methods."""
        alert_methods = [
            "find_by_name",
            "find_by_status",
            "find_by_type",
            "find_by_severity",
        ]

        for method in alert_methods:
            assert hasattr(AlertRepositoryProtocol, method)
            assert callable(getattr(AlertRepositoryProtocol, method))

    @pytest.mark.asyncio
    async def test_mock_alert_repository(self):
        """Test mock alert repository implementation."""
        mock_repo = AsyncMock(spec=AlertRepositoryProtocol)
        mock_alert = Mock(spec=Alert)

        # Configure alert-specific returns
        mock_repo.find_by_name.return_value = [mock_alert]
        mock_repo.find_by_status.return_value = [mock_alert]
        mock_repo.find_by_type.return_value = [mock_alert]
        mock_repo.find_by_severity.return_value = [mock_alert]

        # Test alert-specific operations
        alerts_by_name = await mock_repo.find_by_name("test_alert")
        assert len(alerts_by_name) == 1

        alerts_by_status = await mock_repo.find_by_status("active")
        assert len(alerts_by_status) == 1

        alerts_by_type = await mock_repo.find_by_type("anomaly_threshold")
        assert len(alerts_by_type) == 1

        alerts_by_severity = await mock_repo.find_by_severity("high")
        assert len(alerts_by_severity) == 1


class TestRepositoryProtocolInteroperability:
    """Test interoperability and consistency across repository protocols."""

    def test_all_protocols_inherit_from_base(self):
        """Test that all repository protocols inherit from base RepositoryProtocol."""
        repository_protocols = [
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

        for protocol in repository_protocols:
            assert issubclass(protocol, RepositoryProtocol)

    def test_all_protocols_are_runtime_checkable(self):
        """Test that all repository protocols are runtime checkable."""
        repository_protocols = [
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

        for protocol in repository_protocols:
            assert hasattr(protocol, "__runtime_checkable__")

    def test_crud_methods_consistency(self):
        """Test that all repository protocols have consistent CRUD methods."""
        base_methods = ["save", "find_by_id", "find_all", "delete", "exists", "count"]

        repository_protocols = [
            DetectorRepositoryProtocol,
            DatasetRepositoryProtocol,
            DetectionResultRepositoryProtocol,
            ModelRepositoryProtocol,
            ExperimentRepositoryProtocol,
            PipelineRepositoryProtocol,
            AlertRepositoryProtocol,
        ]

        for protocol in repository_protocols:
            for method in base_methods:
                assert hasattr(
                    protocol, method
                ), f"{protocol.__name__} missing {method}"

    @pytest.mark.asyncio
    async def test_polymorphic_repository_usage(self):
        """Test polymorphic usage of different repository types."""
        # Create mocks for different repository types
        detector_repo = AsyncMock(spec=DetectorRepositoryProtocol)
        dataset_repo = AsyncMock(spec=DatasetRepositoryProtocol)
        result_repo = AsyncMock(spec=DetectionResultRepositoryProtocol)

        repositories = [detector_repo, dataset_repo, result_repo]

        # Configure common CRUD operations
        for repo in repositories:
            repo.count.return_value = 5
            repo.exists.return_value = True

        # Test polymorphic usage
        async def count_all_entities(repos: list[RepositoryProtocol]) -> int:
            total = 0
            for repo in repos:
                total += await repo.count()
            return total

        total_count = await count_all_entities(repositories)
        assert total_count == 15  # 3 repos * 5 entities each

    def test_repository_protocol_type_parameters(self):
        """Test that repository protocols correctly use type parameters."""
        # DetectorRepositoryProtocol should work with Detector type
        detector_repo = AsyncMock(spec=DetectorRepositoryProtocol)
        assert isinstance(detector_repo, RepositoryProtocol)

        # DatasetRepositoryProtocol should work with Dataset type
        dataset_repo = AsyncMock(spec=DatasetRepositoryProtocol)
        assert isinstance(dataset_repo, RepositoryProtocol)

    @pytest.mark.asyncio
    async def test_error_handling_consistency(self):
        """Test consistent error handling across repository protocols."""
        repos = [
            AsyncMock(spec=DetectorRepositoryProtocol),
            AsyncMock(spec=DatasetRepositoryProtocol),
            AsyncMock(spec=DetectionResultRepositoryProtocol),
        ]

        # Configure all repos to raise the same exception type
        non_existent_id = uuid4()
        for repo in repos:
            repo.find_by_id.return_value = None
            repo.delete.return_value = False

        # Test consistent None returns for not found
        for repo in repos:
            result = await repo.find_by_id(non_existent_id)
            assert result is None

            deleted = await repo.delete(non_existent_id)
            assert deleted is False

    def test_method_signature_consistency(self):
        """Test that similar methods across protocols have consistent signatures."""
        # Test find_by_name methods have consistent signatures
        protocols_with_find_by_name = [
            DetectorRepositoryProtocol,
            DatasetRepositoryProtocol,
            ModelRepositoryProtocol,
            ExperimentRepositoryProtocol,
            PipelineRepositoryProtocol,
            AlertRepositoryProtocol,
        ]

        for protocol in protocols_with_find_by_name:
            method_sig = inspect.signature(protocol.find_by_name)
            assert "name" in method_sig.parameters
            # All should take a string name parameter
            name_param = method_sig.parameters["name"]
            assert name_param.annotation == str

    @pytest.mark.asyncio
    async def test_repository_protocol_contracts(self):
        """Test that repository protocol contracts are properly defined."""

        class TestRepository:
            def __init__(self):
                self._entities = {}

            async def save(self, entity) -> None:
                self._entities[entity.id] = entity

            async def find_by_id(self, entity_id: UUID):
                return self._entities.get(entity_id)

            async def find_all(self) -> list:
                return list(self._entities.values())

            async def delete(self, entity_id: UUID) -> bool:
                return self._entities.pop(entity_id, None) is not None

            async def exists(self, entity_id: UUID) -> bool:
                return entity_id in self._entities

            async def count(self) -> int:
                return len(self._entities)

        repo = TestRepository()

        # Test that the repository conforms to the protocol
        assert isinstance(repo, RepositoryProtocol)

        # Test contract behavior
        mock_entity = Mock()
        mock_entity.id = uuid4()

        # Save should persist entity
        await repo.save(mock_entity)
        assert await repo.exists(mock_entity.id)

        # Find by ID should return saved entity
        found = await repo.find_by_id(mock_entity.id)
        assert found == mock_entity

        # Count should reflect saved entities
        assert await repo.count() == 1

        # Delete should remove entity
        deleted = await repo.delete(mock_entity.id)
        assert deleted is True
        assert not await repo.exists(mock_entity.id)
        assert await repo.count() == 0
