"""Production-ready tests for Data Transfer Objects (DTOs)."""

from __future__ import annotations

import pytest
from datetime import datetime
from uuid import uuid4, UUID

from pynomaly.application.dto import (
    CreateDetectorDTO,
    DetectorResponseDTO,
    DatasetDTO,
    DetectionRequestDTO,
    DetectionResultDTO,
    AnomalyDTO,
    CreateExperimentDTO,
    ExperimentResponseDTO,
)


class TestCreateDetectorDTO:
    """Test CreateDetectorDTO validation and serialization."""
    
    def test_valid_creation(self):
        """Test creating a valid CreateDetectorDTO."""
        dto = CreateDetectorDTO(
            name="test_detector",
            algorithm_name="IsolationForest",
            contamination_rate=0.1,
            parameters={"n_estimators": 100}
        )
        
        assert dto.name == "test_detector"
        assert dto.algorithm_name == "IsolationForest"
        assert dto.contamination_rate == 0.1
        assert dto.parameters["n_estimators"] == 100
        assert dto.metadata == {}  # default
    
    def test_contamination_rate_validation(self):
        """Test contamination rate bounds validation."""
        # Valid rates
        CreateDetectorDTO(name="test", algorithm_name="test", contamination_rate=0.0)
        CreateDetectorDTO(name="test", algorithm_name="test", contamination_rate=0.5)
        CreateDetectorDTO(name="test", algorithm_name="test", contamination_rate=1.0)
        
        # Invalid rates
        with pytest.raises(ValueError):
            CreateDetectorDTO(name="test", algorithm_name="test", contamination_rate=-0.1)
        
        with pytest.raises(ValueError):
            CreateDetectorDTO(name="test", algorithm_name="test", contamination_rate=1.1)
    
    def test_name_validation(self):
        """Test name length validation."""
        # Valid names
        CreateDetectorDTO(name="a", algorithm_name="test")  # min length 1
        CreateDetectorDTO(name="a" * 100, algorithm_name="test")  # max length 100
        
        # Invalid names
        with pytest.raises(ValueError):
            CreateDetectorDTO(name="", algorithm_name="test")
        
        with pytest.raises(ValueError):
            CreateDetectorDTO(name="a" * 101, algorithm_name="test")  # too long
    
    def test_defaults(self):
        """Test default values."""
        dto = CreateDetectorDTO(name="test", algorithm_name="IsolationForest")
        
        assert dto.contamination_rate == 0.1  # default
        assert dto.parameters == {}  # default
        assert dto.metadata == {}  # default
    
    def test_serialization(self):
        """Test JSON serialization."""
        dto = CreateDetectorDTO(
            name="test_detector",
            algorithm_name="IsolationForest",
            contamination_rate=0.2,
            parameters={"max_depth": 5},
            metadata={"version": "1.0"}
        )
        
        data = dto.model_dump()
        
        assert data["name"] == "test_detector"
        assert data["algorithm_name"] == "IsolationForest"
        assert data["contamination_rate"] == 0.2
        assert data["parameters"]["max_depth"] == 5
        assert data["metadata"]["version"] == "1.0"


class TestDetectorResponseDTO:
    """Test DetectorResponseDTO serialization and validation."""
    
    def test_valid_response(self):
        """Test creating a valid response DTO."""
        detector_id = uuid4()
        created_at = datetime.utcnow()
        
        dto = DetectorResponseDTO(
            id=detector_id,
            name="test_detector",
            algorithm_name="IsolationForest",
            contamination_rate=0.1,
            is_fitted=True,
            created_at=created_at,
            parameters={"n_estimators": 100}
        )
        
        assert dto.id == detector_id
        assert dto.name == "test_detector"
        assert dto.algorithm_name == "IsolationForest"
        assert dto.contamination_rate == 0.1
        assert dto.is_fitted is True
        assert dto.created_at == created_at
        assert dto.status == "active"  # default
        assert dto.version == "1.0.0"  # default
    
    def test_optional_fields(self):
        """Test optional field defaults."""
        dto = DetectorResponseDTO(
            id=uuid4(),
            name="test",
            algorithm_name="test",
            contamination_rate=0.1,
            is_fitted=False,
            created_at=datetime.utcnow()
        )
        
        assert dto.trained_at is None  # default
        assert dto.parameters == {}  # default
        assert dto.metadata == {}  # default
    
    def test_serialization_with_dates(self):
        """Test serialization includes proper datetime handling."""
        now = datetime.utcnow()
        dto = DetectorResponseDTO(
            id=uuid4(),
            name="test",
            algorithm_name="test",
            contamination_rate=0.1,
            is_fitted=True,
            created_at=now,
            trained_at=now
        )
        
        data = dto.model_dump()
        
        assert "id" in data
        assert "created_at" in data
        assert "trained_at" in data
        assert "status" in data
        assert "version" in data


class TestDatasetDTO:
    """Test DatasetDTO validation and serialization."""
    
    def test_valid_dataset(self):
        """Test creating a valid dataset DTO."""
        dataset_id = uuid4()
        created_at = datetime.utcnow()
        feature_names = [f"feature_{i}" for i in range(5)]
        
        dto = DatasetDTO(
            id=dataset_id,
            name="test_dataset",
            shape=(1000, 5),
            n_samples=1000,
            n_features=5,
            feature_names=feature_names,
            has_target=True,
            target_column="target",
            created_at=created_at,
            memory_usage_mb=2.5,
            numeric_features=3,
            categorical_features=2
        )
        
        assert dto.id == dataset_id
        assert dto.name == "test_dataset"
        assert dto.shape == (1000, 5)
        assert dto.n_samples == 1000
        assert dto.n_features == 5
        assert dto.feature_names == feature_names
        assert dto.has_target is True
        assert dto.target_column == "target"
        assert dto.memory_usage_mb == 2.5
    
    def test_dataset_without_target(self):
        """Test dataset without target column."""
        dto = DatasetDTO(
            id=uuid4(),
            name="unsupervised_dataset",
            shape=(500, 3),
            n_samples=500,
            n_features=3,
            feature_names=["x", "y", "z"],
            has_target=False,
            created_at=datetime.utcnow(),
            memory_usage_mb=1.0,
            numeric_features=3,
            categorical_features=0
        )
        
        assert dto.has_target is False
        assert dto.target_column is None
    
    def test_feature_type_consistency(self):
        """Test that feature type counts are consistent."""
        dto = DatasetDTO(
            id=uuid4(),
            name="test",
            shape=(100, 10),
            n_samples=100,
            n_features=10,
            feature_names=[f"f{i}" for i in range(10)],
            has_target=False,
            created_at=datetime.utcnow(),
            memory_usage_mb=0.5,
            numeric_features=7,
            categorical_features=3
        )
        
        # Should sum to total features
        assert dto.numeric_features + dto.categorical_features == dto.n_features


class TestDetectionRequestDTO:
    """Test DetectionRequestDTO validation."""
    
    def test_valid_request(self):
        """Test creating a valid detection request."""
        detector_id = uuid4()
        data = [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]
        
        dto = DetectionRequestDTO(
            detector_id=detector_id,
            data=data,
            return_scores=True,
            return_feature_importance=True,
            threshold=0.5
        )
        
        assert dto.detector_id == detector_id
        assert dto.data == data
        assert dto.return_scores is True
        assert dto.return_feature_importance is True
        assert dto.threshold == 0.5
    
    def test_default_values(self):
        """Test default parameter values."""
        dto = DetectionRequestDTO(
            detector_id=uuid4(),
            data=[[1.0, 2.0]]
        )
        
        assert dto.return_scores is True  # default
        assert dto.return_feature_importance is False  # default
        assert dto.threshold is None  # default
    
    def test_data_validation(self):
        """Test data format validation."""
        # Valid data formats
        DetectionRequestDTO(
            detector_id=uuid4(),
            data=[[1.0, 2.0]]  # 2D list
        )
        
        DetectionRequestDTO(
            detector_id=uuid4(),
            data=[[1.0], [2.0], [3.0]]  # multiple samples
        )


class TestDetectionResultDTO:
    """Test DetectionResultDTO validation and serialization."""
    
    def test_valid_result(self):
        """Test creating a valid detection result."""
        result_id = uuid4()
        detector_id = uuid4()
        dataset_id = uuid4()
        created_at = datetime.utcnow()
        
        # Create sample anomaly DTOs
        anomalies = [
            AnomalyDTO(
                id=uuid4(),
                score=0.95,
                data_point={"feature1": 1.0, "feature2": 2.0},
                detector_name="IsolationForest",
                timestamp=created_at
            )
        ]
        
        dto = DetectionResultDTO(
            id=result_id,
            detector_id=detector_id,
            dataset_id=dataset_id,
            created_at=created_at,
            duration_seconds=1.5,
            anomalies=anomalies,
            total_samples=100,
            anomaly_count=1,
            contamination_rate=0.01,
            mean_score=0.2,
            max_score=0.95,
            min_score=0.05,
            threshold=0.5
        )
        
        assert dto.id == result_id
        assert dto.detector_id == detector_id
        assert dto.dataset_id == dataset_id
        assert dto.duration_seconds == 1.5
        assert len(dto.anomalies) == 1
        assert dto.total_samples == 100
        assert dto.anomaly_count == 1
        assert dto.contamination_rate == 0.01
        assert dto.threshold == 0.5
    
    def test_contamination_rate_validation(self):
        """Test contamination rate bounds."""
        # Valid contamination rates
        base_dto_data = {
            "id": uuid4(),
            "detector_id": uuid4(),
            "dataset_id": uuid4(),
            "created_at": datetime.utcnow(),
            "duration_seconds": 1.0,
            "anomalies": [],
            "total_samples": 100,
            "anomaly_count": 5,
            "mean_score": 0.2,
            "max_score": 0.9,
            "min_score": 0.1,
            "threshold": 0.5
        }
        
        DetectionResultDTO(**base_dto_data, contamination_rate=0.0)
        DetectionResultDTO(**base_dto_data, contamination_rate=0.5)
        DetectionResultDTO(**base_dto_data, contamination_rate=1.0)
        
        # Invalid contamination rates
        with pytest.raises(ValueError):
            DetectionResultDTO(**base_dto_data, contamination_rate=-0.1)
        
        with pytest.raises(ValueError):
            DetectionResultDTO(**base_dto_data, contamination_rate=1.1)
    
    def test_optional_fields(self):
        """Test optional field defaults."""
        dto = DetectionResultDTO(
            id=uuid4(),
            detector_id=uuid4(),
            dataset_id=uuid4(),
            created_at=datetime.utcnow(),
            duration_seconds=1.0,
            anomalies=[],
            total_samples=10,
            anomaly_count=0,
            contamination_rate=0.0,
            mean_score=0.1,
            max_score=0.1,
            min_score=0.1,
            threshold=0.5
        )
        
        assert dto.run_id is None  # default
        assert dto.metadata == {}  # default
        assert dto.parameters == {}  # default


class TestAnomalyDTO:
    """Test AnomalyDTO validation and serialization."""
    
    def test_valid_anomaly(self):
        """Test creating a valid anomaly DTO."""
        anomaly_id = uuid4()
        timestamp = datetime.utcnow()
        data_point = {"age": 25, "income": 50000.0, "city": "New York"}
        
        dto = AnomalyDTO(
            id=anomaly_id,
            score=0.85,
            data_point=data_point,
            detector_name="LocalOutlierFactor",
            timestamp=timestamp
        )
        
        assert dto.id == anomaly_id
        assert dto.score == 0.85
        assert dto.data_point == data_point
        assert dto.detector_name == "LocalOutlierFactor"
        assert dto.timestamp == timestamp
    
    def test_score_validation(self):
        """Test score bounds validation."""
        base_data = {
            "id": uuid4(),
            "data_point": {},
            "detector_name": "test",
            "timestamp": datetime.utcnow()
        }
        
        # Valid scores
        AnomalyDTO(**base_data, score=0.0)
        AnomalyDTO(**base_data, score=0.5)
        AnomalyDTO(**base_data, score=1.0)
        
        # Invalid scores
        with pytest.raises(ValueError):
            AnomalyDTO(**base_data, score=-0.1)
        
        with pytest.raises(ValueError):
            AnomalyDTO(**base_data, score=1.1)
    
    def test_data_point_types(self):
        """Test various data point formats."""
        # Numeric data
        AnomalyDTO(
            id=uuid4(),
            score=0.8,
            data_point={"x": 1.0, "y": 2.0},
            detector_name="test",
            timestamp=datetime.utcnow()
        )
        
        # Mixed data types
        AnomalyDTO(
            id=uuid4(),
            score=0.9,
            data_point={"age": 30, "name": "John", "active": True},
            detector_name="test",
            timestamp=datetime.utcnow()
        )


class TestExperimentDTO:
    """Test experiment-related DTOs."""
    
    def test_create_experiment_dto(self):
        """Test creating experiment DTO."""
        dto = CreateExperimentDTO(
            name="fraud_detection_experiment",
            description="Testing various algorithms for fraud detection",
            tags=["fraud", "financial", "anomaly_detection"]
        )
        
        assert dto.name == "fraud_detection_experiment"
        assert dto.description == "Testing various algorithms for fraud detection"
        assert dto.tags == ["fraud", "financial", "anomaly_detection"]
    
    def test_minimal_experiment(self):
        """Test experiment with minimal required data."""
        dto = CreateExperimentDTO(name="minimal_test")
        
        assert dto.name == "minimal_test"
        assert dto.description is None
        assert dto.tags == []  # default empty list
    
    def test_experiment_response_dto(self):
        """Test experiment response DTO."""
        experiment_id = uuid4()
        created_at = datetime.utcnow()
        
        dto = ExperimentResponseDTO(
            id=experiment_id,
            name="test_experiment",
            description="Test description",
            tags=["test"],
            created_at=created_at,
            status="completed",
            n_runs=10
        )
        
        assert dto.id == experiment_id
        assert dto.name == "test_experiment"
        assert dto.status == "completed"
        assert dto.n_runs == 10
        assert dto.created_at == created_at


class TestDTOIntegration:
    """Test DTO integration patterns."""
    
    def test_nested_dto_serialization(self):
        """Test serialization of DTOs containing other DTOs."""
        # Create nested anomaly DTOs within a result DTO
        anomalies = [
            AnomalyDTO(
                id=uuid4(),
                score=0.9,
                data_point={"feature1": 1.0},
                detector_name="test",
                timestamp=datetime.utcnow()
            ),
            AnomalyDTO(
                id=uuid4(),
                score=0.8,
                data_point={"feature1": 2.0},
                detector_name="test",
                timestamp=datetime.utcnow()
            )
        ]
        
        result_dto = DetectionResultDTO(
            id=uuid4(),
            detector_id=uuid4(),
            dataset_id=uuid4(),
            created_at=datetime.utcnow(),
            duration_seconds=2.0,
            anomalies=anomalies,
            total_samples=100,
            anomaly_count=2,
            contamination_rate=0.02,
            mean_score=0.3,
            max_score=0.9,
            min_score=0.1,
            threshold=0.6
        )
        
        # Test serialization includes nested anomalies
        data = result_dto.model_dump()
        
        assert "anomalies" in data
        assert len(data["anomalies"]) == 2
        assert "score" in data["anomalies"][0]
        assert "data_point" in data["anomalies"][0]
    
    def test_uuid_consistency(self):
        """Test that UUIDs are handled consistently across DTOs."""
        detector_id = uuid4()
        dataset_id = uuid4()
        
        # Use same IDs in request and response
        request = DetectionRequestDTO(
            detector_id=detector_id,
            data=[[1.0, 2.0]]
        )
        
        result = DetectionResultDTO(
            id=uuid4(),
            detector_id=detector_id,
            dataset_id=dataset_id,
            created_at=datetime.utcnow(),
            duration_seconds=1.0,
            anomalies=[],
            total_samples=1,
            anomaly_count=0,
            contamination_rate=0.0,
            mean_score=0.1,
            max_score=0.1,
            min_score=0.1,
            threshold=0.5
        )
        
        assert request.detector_id == result.detector_id
    
    def test_pydantic_model_config(self):
        """Test that Pydantic configuration is working correctly."""
        # Test from_attributes configuration
        dto = DetectorResponseDTO(
            id=uuid4(),
            name="test",
            algorithm_name="test",
            contamination_rate=0.1,
            is_fitted=True,
            created_at=datetime.utcnow()
        )
        
        # Should be able to dump and validate
        data = dto.model_dump()
        
        # Should be able to recreate from dict
        dto2 = DetectorResponseDTO(**data)
        assert dto2.name == dto.name
        assert dto2.algorithm_name == dto.algorithm_name