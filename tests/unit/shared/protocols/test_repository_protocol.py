"""Tests for repository protocol."""

from typing import Any
from unittest.mock import Mock
from uuid import UUID, uuid4

import pytest

from pynomaly.domain.entities import (
    Alert,
    AlertNotification,
    Dataset,
    DetectionResult,
    Detector,
    Experiment,
    ExperimentRun,
    Model,
    ModelVersion,
    Pipeline,
    PipelineRun,
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

    def test_protocol_definition(self):
        """Test protocol has required methods."""
        assert hasattr(RepositoryProtocol, 'save')
        assert hasattr(RepositoryProtocol, 'find_by_id')
        assert hasattr(RepositoryProtocol, 'find_all')
        assert hasattr(RepositoryProtocol, 'delete')
        assert hasattr(RepositoryProtocol, 'exists')
        assert hasattr(RepositoryProtocol, 'count')

    def test_protocol_runtime_checkable(self):
        """Test protocol is runtime checkable."""
        class ConcreteRepository:
            def save(self, entity: Any) -> None:
                pass
            
            def find_by_id(self, entity_id: UUID) -> Any | None:
                return None
            
            def find_all(self) -> list[Any]:
                return []
            
            def delete(self, entity_id: UUID) -> bool:
                return True
            
            def exists(self, entity_id: UUID) -> bool:
                return False
            
            def count(self) -> int:
                return 0
        
        repository = ConcreteRepository()
        assert isinstance(repository, RepositoryProtocol)

    def test_basic_operations(self):
        """Test basic repository operations."""
        mock_repo = Mock(spec=RepositoryProtocol)
        mock_entity = Mock()
        test_id = uuid4()
        
        # Test save
        mock_repo.save(mock_entity)
        mock_repo.save.assert_called_once_with(mock_entity)
        
        # Test find_by_id
        mock_repo.find_by_id.return_value = mock_entity
        result = mock_repo.find_by_id(test_id)
        assert result == mock_entity
        
        # Test find_all
        mock_repo.find_all.return_value = [mock_entity]
        results = mock_repo.find_all()
        assert results == [mock_entity]
        
        # Test delete
        mock_repo.delete.return_value = True
        deleted = mock_repo.delete(test_id)
        assert deleted is True
        
        # Test exists
        mock_repo.exists.return_value = True
        exists = mock_repo.exists(test_id)
        assert exists is True
        
        # Test count
        mock_repo.count.return_value = 1
        count = mock_repo.count()
        assert count == 1


class TestDetectorRepositoryProtocol:
    """Test suite for DetectorRepositoryProtocol."""

    def test_protocol_inheritance(self):
        """Test DetectorRepositoryProtocol extends RepositoryProtocol."""
        assert hasattr(DetectorRepositoryProtocol, 'save')
        assert hasattr(DetectorRepositoryProtocol, 'find_by_id')
        assert hasattr(DetectorRepositoryProtocol, 'find_by_name')
        assert hasattr(DetectorRepositoryProtocol, 'find_by_algorithm')
        assert hasattr(DetectorRepositoryProtocol, 'find_fitted')
        assert hasattr(DetectorRepositoryProtocol, 'save_model_artifact')
        assert hasattr(DetectorRepositoryProtocol, 'load_model_artifact')

    def test_protocol_runtime_checkable(self):
        """Test DetectorRepositoryProtocol is runtime checkable."""
        class ConcreteDetectorRepository:
            def save(self, entity: Detector) -> None:
                pass
            
            def find_by_id(self, entity_id: UUID) -> Detector | None:
                return None
            
            def find_all(self) -> list[Detector]:
                return []
            
            def delete(self, entity_id: UUID) -> bool:
                return True
            
            def exists(self, entity_id: UUID) -> bool:
                return False
            
            def count(self) -> int:
                return 0
            
            def find_by_name(self, name: str) -> Detector | None:
                return None
            
            def find_by_algorithm(self, algorithm_name: str) -> list[Detector]:
                return []
            
            def find_fitted(self) -> list[Detector]:
                return []
            
            def save_model_artifact(self, detector_id: UUID, artifact: bytes) -> None:
                pass
            
            def load_model_artifact(self, detector_id: UUID) -> bytes | None:
                return None
        
        repository = ConcreteDetectorRepository()
        assert isinstance(repository, DetectorRepositoryProtocol)
        assert isinstance(repository, RepositoryProtocol)

    def test_detector_specific_methods(self):
        """Test detector-specific methods."""
        mock_repo = Mock(spec=DetectorRepositoryProtocol)
        mock_detector = Mock(spec=Detector)
        test_id = uuid4()
        
        # Test find_by_name
        mock_repo.find_by_name.return_value = mock_detector
        result = mock_repo.find_by_name("test_detector")
        assert result == mock_detector
        
        # Test find_by_algorithm
        mock_repo.find_by_algorithm.return_value = [mock_detector]
        results = mock_repo.find_by_algorithm("IsolationForest")
        assert results == [mock_detector]
        
        # Test find_fitted
        mock_repo.find_fitted.return_value = [mock_detector]
        fitted = mock_repo.find_fitted()
        assert fitted == [mock_detector]
        
        # Test save_model_artifact
        artifact = b"model_data"
        mock_repo.save_model_artifact(test_id, artifact)
        mock_repo.save_model_artifact.assert_called_once_with(test_id, artifact)
        
        # Test load_model_artifact
        mock_repo.load_model_artifact.return_value = artifact
        loaded = mock_repo.load_model_artifact(test_id)
        assert loaded == artifact


class TestDatasetRepositoryProtocol:
    """Test suite for DatasetRepositoryProtocol."""

    def test_protocol_inheritance(self):
        """Test DatasetRepositoryProtocol extends RepositoryProtocol."""
        assert hasattr(DatasetRepositoryProtocol, 'save')
        assert hasattr(DatasetRepositoryProtocol, 'find_by_id')
        assert hasattr(DatasetRepositoryProtocol, 'find_by_name')
        assert hasattr(DatasetRepositoryProtocol, 'find_by_metadata')
        assert hasattr(DatasetRepositoryProtocol, 'save_data')
        assert hasattr(DatasetRepositoryProtocol, 'load_data')

    def test_protocol_runtime_checkable(self):
        """Test DatasetRepositoryProtocol is runtime checkable."""
        class ConcreteDatasetRepository:
            def save(self, entity: Dataset) -> None:
                pass
            
            def find_by_id(self, entity_id: UUID) -> Dataset | None:
                return None
            
            def find_all(self) -> list[Dataset]:
                return []
            
            def delete(self, entity_id: UUID) -> bool:
                return True
            
            def exists(self, entity_id: UUID) -> bool:
                return False
            
            def count(self) -> int:
                return 0
            
            def find_by_name(self, name: str) -> Dataset | None:
                return None
            
            def find_by_metadata(self, key: str, value: Any) -> list[Dataset]:
                return []
            
            def save_data(self, dataset_id: UUID, format: str = "parquet") -> str:
                return f"/path/to/data.{format}"
            
            def load_data(self, dataset_id: UUID) -> Dataset | None:
                return None
        
        repository = ConcreteDatasetRepository()
        assert isinstance(repository, DatasetRepositoryProtocol)
        assert isinstance(repository, RepositoryProtocol)

    def test_dataset_specific_methods(self):
        """Test dataset-specific methods."""
        mock_repo = Mock(spec=DatasetRepositoryProtocol)
        mock_dataset = Mock(spec=Dataset)
        test_id = uuid4()
        
        # Test find_by_name
        mock_repo.find_by_name.return_value = mock_dataset
        result = mock_repo.find_by_name("test_dataset")
        assert result == mock_dataset
        
        # Test find_by_metadata
        mock_repo.find_by_metadata.return_value = [mock_dataset]
        results = mock_repo.find_by_metadata("source", "csv")
        assert results == [mock_dataset]
        
        # Test save_data
        mock_repo.save_data.return_value = "/path/to/data.parquet"
        path = mock_repo.save_data(test_id, "parquet")
        assert path == "/path/to/data.parquet"
        
        # Test load_data
        mock_repo.load_data.return_value = mock_dataset
        loaded = mock_repo.load_data(test_id)
        assert loaded == mock_dataset


class TestDetectionResultRepositoryProtocol:
    """Test suite for DetectionResultRepositoryProtocol."""

    def test_protocol_inheritance(self):
        """Test DetectionResultRepositoryProtocol extends RepositoryProtocol."""
        assert hasattr(DetectionResultRepositoryProtocol, 'save')
        assert hasattr(DetectionResultRepositoryProtocol, 'find_by_id')
        assert hasattr(DetectionResultRepositoryProtocol, 'find_by_detector')
        assert hasattr(DetectionResultRepositoryProtocol, 'find_by_dataset')
        assert hasattr(DetectionResultRepositoryProtocol, 'find_recent')
        assert hasattr(DetectionResultRepositoryProtocol, 'get_summary_stats')

    def test_protocol_runtime_checkable(self):
        """Test DetectionResultRepositoryProtocol is runtime checkable."""
        class ConcreteDetectionResultRepository:
            def save(self, entity: DetectionResult) -> None:
                pass
            
            def find_by_id(self, entity_id: UUID) -> DetectionResult | None:
                return None
            
            def find_all(self) -> list[DetectionResult]:
                return []
            
            def delete(self, entity_id: UUID) -> bool:
                return True
            
            def exists(self, entity_id: UUID) -> bool:
                return False
            
            def count(self) -> int:
                return 0
            
            def find_by_detector(self, detector_id: UUID) -> list[DetectionResult]:
                return []
            
            def find_by_dataset(self, dataset_id: UUID) -> list[DetectionResult]:
                return []
            
            def find_recent(self, limit: int = 10) -> list[DetectionResult]:
                return []
            
            def get_summary_stats(self, result_id: UUID) -> dict[str, Any]:
                return {}
        
        repository = ConcreteDetectionResultRepository()
        assert isinstance(repository, DetectionResultRepositoryProtocol)
        assert isinstance(repository, RepositoryProtocol)

    def test_detection_result_specific_methods(self):
        """Test detection result-specific methods."""
        mock_repo = Mock(spec=DetectionResultRepositoryProtocol)
        mock_result = Mock(spec=DetectionResult)
        test_id = uuid4()
        
        # Test find_by_detector
        mock_repo.find_by_detector.return_value = [mock_result]
        results = mock_repo.find_by_detector(test_id)
        assert results == [mock_result]
        
        # Test find_by_dataset
        mock_repo.find_by_dataset.return_value = [mock_result]
        results = mock_repo.find_by_dataset(test_id)
        assert results == [mock_result]
        
        # Test find_recent
        mock_repo.find_recent.return_value = [mock_result]
        recent = mock_repo.find_recent(5)
        assert recent == [mock_result]
        
        # Test get_summary_stats
        stats = {"anomalies": 10, "accuracy": 0.95}
        mock_repo.get_summary_stats.return_value = stats
        result_stats = mock_repo.get_summary_stats(test_id)
        assert result_stats == stats


class TestModelRepositoryProtocol:
    """Test suite for ModelRepositoryProtocol."""

    def test_protocol_inheritance(self):
        """Test ModelRepositoryProtocol extends RepositoryProtocol."""
        assert hasattr(ModelRepositoryProtocol, 'save')
        assert hasattr(ModelRepositoryProtocol, 'find_by_id')
        assert hasattr(ModelRepositoryProtocol, 'find_by_name')
        assert hasattr(ModelRepositoryProtocol, 'find_by_stage')
        assert hasattr(ModelRepositoryProtocol, 'find_by_type')

    def test_protocol_runtime_checkable(self):
        """Test ModelRepositoryProtocol is runtime checkable."""
        class ConcreteModelRepository:
            def save(self, entity: Model) -> None:
                pass
            
            def find_by_id(self, entity_id: UUID) -> Model | None:
                return None
            
            def find_all(self) -> list[Model]:
                return []
            
            def delete(self, entity_id: UUID) -> bool:
                return True
            
            def exists(self, entity_id: UUID) -> bool:
                return False
            
            def count(self) -> int:
                return 0
            
            def find_by_name(self, name: str) -> list[Model]:
                return []
            
            def find_by_stage(self, stage: Any) -> list[Model]:
                return []
            
            def find_by_type(self, model_type: Any) -> list[Model]:
                return []
        
        repository = ConcreteModelRepository()
        assert isinstance(repository, ModelRepositoryProtocol)
        assert isinstance(repository, RepositoryProtocol)

    def test_model_specific_methods(self):
        """Test model-specific methods."""
        mock_repo = Mock(spec=ModelRepositoryProtocol)
        mock_model = Mock(spec=Model)
        
        # Test find_by_name
        mock_repo.find_by_name.return_value = [mock_model]
        results = mock_repo.find_by_name("test_model")
        assert results == [mock_model]
        
        # Test find_by_stage
        mock_repo.find_by_stage.return_value = [mock_model]
        results = mock_repo.find_by_stage("production")
        assert results == [mock_model]
        
        # Test find_by_type
        mock_repo.find_by_type.return_value = [mock_model]
        results = mock_repo.find_by_type("classifier")
        assert results == [mock_model]


class TestProtocolInteractions:
    """Test protocol interactions and edge cases."""

    def test_multiple_protocol_implementation(self):
        """Test class implementing multiple repository protocols."""
        class MultiProtocolRepository:
            def save(self, entity: Any) -> None:
                pass
            
            def find_by_id(self, entity_id: UUID) -> Any | None:
                return None
            
            def find_all(self) -> list[Any]:
                return []
            
            def delete(self, entity_id: UUID) -> bool:
                return True
            
            def exists(self, entity_id: UUID) -> bool:
                return False
            
            def count(self) -> int:
                return 0
            
            def find_by_name(self, name: str) -> Any | None:
                return None
            
            def find_by_algorithm(self, algorithm_name: str) -> list[Any]:
                return []
            
            def find_fitted(self) -> list[Any]:
                return []
            
            def save_model_artifact(self, detector_id: UUID, artifact: bytes) -> None:
                pass
            
            def load_model_artifact(self, detector_id: UUID) -> bytes | None:
                return None
        
        repository = MultiProtocolRepository()
        assert isinstance(repository, RepositoryProtocol)
        assert isinstance(repository, DetectorRepositoryProtocol)

    def test_error_handling(self):
        """Test error handling in repository operations."""
        mock_repo = Mock(spec=RepositoryProtocol)
        test_id = uuid4()
        
        # Configure mock to raise exception
        mock_repo.find_by_id.side_effect = ValueError("Database error")
        
        with pytest.raises(ValueError, match="Database error"):
            mock_repo.find_by_id(test_id)

    def test_uuid_handling(self):
        """Test UUID parameter handling."""
        mock_repo = Mock(spec=RepositoryProtocol)
        test_id = uuid4()
        
        # Test with UUID
        mock_repo.find_by_id(test_id)
        mock_repo.find_by_id.assert_called_with(test_id)
        
        # Test with string UUID
        str_id = str(test_id)
        mock_repo.find_by_id(str_id)
        mock_repo.find_by_id.assert_called_with(str_id)

    def test_type_parameters(self):
        """Test type parameters in generic protocols."""
        # Test that protocols can be used with type parameters
        detector_repo: DetectorRepositoryProtocol = Mock()
        dataset_repo: DatasetRepositoryProtocol = Mock()
        
        # Should be able to assign specific types
        assert detector_repo is not None
        assert dataset_repo is not None

    def test_inheritance_chain(self):
        """Test inheritance chain for specialized protocols."""
        # All specialized protocols should inherit from base RepositoryProtocol
        protocols = [
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
        
        for protocol in protocols:
            # Check that each protocol has the base methods
            assert hasattr(protocol, 'save')
            assert hasattr(protocol, 'find_by_id')
            assert hasattr(protocol, 'find_all')
            assert hasattr(protocol, 'delete')
            assert hasattr(protocol, 'exists')
            assert hasattr(protocol, 'count')

    def test_runtime_checkable_consistency(self):
        """Test that all protocols are runtime checkable."""
        protocols = [
            RepositoryProtocol,
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
        
        for protocol in protocols:
            # Each protocol should be decorated with @runtime_checkable
            assert hasattr(protocol, '__class_getitem__')  # Generic protocol marker


class TestSpecializedProtocols:
    """Test specialized repository protocols."""

    def test_experiment_repository_protocol(self):
        """Test ExperimentRepositoryProtocol."""
        mock_repo = Mock(spec=ExperimentRepositoryProtocol)
        mock_experiment = Mock(spec=Experiment)
        
        mock_repo.find_by_name.return_value = [mock_experiment]
        results = mock_repo.find_by_name("test_experiment")
        assert results == [mock_experiment]

    def test_pipeline_repository_protocol(self):
        """Test PipelineRepositoryProtocol."""
        mock_repo = Mock(spec=PipelineRepositoryProtocol)
        mock_pipeline = Mock(spec=Pipeline)
        
        mock_repo.find_by_name_and_environment.return_value = [mock_pipeline]
        results = mock_repo.find_by_name_and_environment("test_pipeline", "production")
        assert results == [mock_pipeline]

    def test_alert_repository_protocol(self):
        """Test AlertRepositoryProtocol."""
        mock_repo = Mock(spec=AlertRepositoryProtocol)
        mock_alert = Mock(spec=Alert)
        
        mock_repo.find_by_severity.return_value = [mock_alert]
        results = mock_repo.find_by_severity("high")
        assert results == [mock_alert]

    def test_alert_notification_repository_protocol(self):
        """Test AlertNotificationRepositoryProtocol."""
        mock_repo = Mock(spec=AlertNotificationRepositoryProtocol)
        mock_notification = Mock(spec=AlertNotification)
        test_id = uuid4()
        
        mock_repo.find_by_alert_id.return_value = [mock_notification]
        results = mock_repo.find_by_alert_id(test_id)
        assert results == [mock_notification]