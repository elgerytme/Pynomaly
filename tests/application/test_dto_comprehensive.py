"""Comprehensive tests for all Data Transfer Objects (DTOs)."""

from __future__ import annotations

import pytest
from datetime import datetime
from uuid import uuid4, UUID
from typing import Optional, Dict, Any

# Import DTOs with error handling for testing
try:
    from pynomaly.application.dto.detector_dto import (
        CreateDetectorDTO, DetectorResponseDTO, DetectionRequestDTO
    )
    DETECTOR_DTOS_AVAILABLE = True
except ImportError:
    DETECTOR_DTOS_AVAILABLE = False

try:
    from pynomaly.application.dto.dataset_dto import (
        DatasetDTO, CreateDatasetDTO, DataQualityReportDTO
    )
    DATASET_DTOS_AVAILABLE = True
except ImportError:
    DATASET_DTOS_AVAILABLE = False

try:
    from pynomaly.application.dto.result_dto import (
        DetectionResultDTO, AnomalyDTO
    )
    RESULT_DTOS_AVAILABLE = True
except ImportError:
    RESULT_DTOS_AVAILABLE = False

try:
    from pynomaly.application.dto.experiment_dto import (
        CreateExperimentDTO, ExperimentResponseDTO
    )
    EXPERIMENT_DTOS_AVAILABLE = True
except ImportError:
    EXPERIMENT_DTOS_AVAILABLE = False


@pytest.mark.skipif(not DETECTOR_DTOS_AVAILABLE, reason="Detector DTOs not available")
class TestDetectorDTOs:
    """Test detector-related DTOs."""
    
    def test_create_detector_dto_valid(self):
        """Test creating a valid CreateDetectorDTO."""
        dto = CreateDetectorDTO(
            name="test_detector",
            algorithm_name="IsolationForest",
            contamination_rate=0.1,
            parameters={"n_estimators": 100},
            metadata={"version": "1.0"}
        )
        
        assert dto.name == "test_detector"
        assert dto.algorithm_name == "IsolationForest"
        assert dto.contamination_rate == 0.1
        assert dto.parameters["n_estimators"] == 100
        assert dto.metadata["version"] == "1.0"
    
    def test_create_detector_dto_defaults(self):
        """Test CreateDetectorDTO with default values."""
        dto = CreateDetectorDTO(
            name="test_detector",
            algorithm_name="IsolationForest"
        )
        
        assert dto.name == "test_detector"
        assert dto.algorithm_name == "IsolationForest"
        assert dto.contamination_rate == 0.1  # default
        assert dto.parameters == {}  # default
        assert dto.metadata == {}  # default
    
    def test_create_detector_dto_validation(self):
        """Test CreateDetectorDTO validation."""
        # Test contamination rate bounds
        with pytest.raises(ValueError):
            CreateDetectorDTO(
                name="test",
                algorithm_name="test",
                contamination_rate=-0.1  # Invalid: negative
            )
        
        with pytest.raises(ValueError):
            CreateDetectorDTO(
                name="test",
                algorithm_name="test",
                contamination_rate=1.1  # Invalid: > 1
            )
        
        # Test name validation
        with pytest.raises(ValueError):
            CreateDetectorDTO(
                name="",  # Invalid: empty
                algorithm_name="test"
            )
        
        with pytest.raises(ValueError):
            CreateDetectorDTO(
                name="a" * 101,  # Invalid: too long
                algorithm_name="test"
            )
    
    def test_detector_response_dto(self):
        """Test DetectorResponseDTO."""
        dto = DetectorResponseDTO(
            id=uuid4(),
            name="test_detector",
            algorithm_name="IsolationForest",
            contamination_rate=0.1,
            is_fitted=True,
            created_at=datetime.utcnow(),
            parameters={"n_estimators": 100}
        )
        
        assert isinstance(dto.id, UUID)
        assert dto.name == "test_detector"
        assert dto.algorithm_name == "IsolationForest"
        assert dto.contamination_rate == 0.1
        assert dto.is_fitted is True
        assert dto.status == "active"  # default
        assert dto.version == "1.0.0"  # default
    
    def test_detection_request_dto(self):
        """Test DetectionRequestDTO."""
        dto = DetectionRequestDTO(
            detector_id=uuid4(),
            data=[[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]],
            return_scores=True,
            return_feature_importance=True,
            threshold=0.5
        )
        
        assert isinstance(dto.detector_id, UUID)
        assert len(dto.data) == 2
        assert len(dto.data[0]) == 3
        assert dto.return_scores is True
        assert dto.return_feature_importance is True
        assert dto.threshold == 0.5
    
    def test_detection_request_dto_defaults(self):
        """Test DetectionRequestDTO defaults."""
        dto = DetectionRequestDTO(
            detector_id=uuid4(),
            data=[[1.0, 2.0]]
        )
        
        assert dto.return_scores is True  # default
        assert dto.return_feature_importance is False  # default
        assert dto.threshold is None  # default


@pytest.mark.skipif(not DATASET_DTOS_AVAILABLE, reason="Dataset DTOs not available")
class TestDatasetDTOs:
    """Test dataset-related DTOs."""
    
    def test_dataset_dto(self):
        """Test DatasetDTO."""
        dto = DatasetDTO(
            id=uuid4(),
            name="test_dataset",
            shape=(1000, 10),
            n_samples=1000,
            n_features=10,
            feature_names=[f"feature_{i}" for i in range(10)],
            has_target=True,
            target_column="target",
            created_at=datetime.utcnow(),
            memory_usage_mb=2.5,
            numeric_features=8,
            categorical_features=2
        )
        
        assert isinstance(dto.id, UUID)
        assert dto.name == "test_dataset"
        assert dto.shape == (1000, 10)
        assert dto.n_samples == 1000
        assert dto.n_features == 10
        assert len(dto.feature_names) == 10
        assert dto.has_target is True
        assert dto.target_column == "target"
        assert dto.memory_usage_mb == 2.5
        assert dto.numeric_features == 8
        assert dto.categorical_features == 2
    
    def test_dataset_dto_without_target(self):
        """Test DatasetDTO without target."""
        dto = DatasetDTO(
            id=uuid4(),
            name="unsupervised_dataset",
            shape=(500, 5),
            n_samples=500,
            n_features=5,
            feature_names=[f"feature_{i}" for i in range(5)],
            has_target=False,
            created_at=datetime.utcnow(),
            memory_usage_mb=1.0,
            numeric_features=5,
            categorical_features=0
        )
        
        assert dto.has_target is False
        assert dto.target_column is None
    
    def test_create_dataset_dto(self):
        """Test CreateDatasetDTO."""
        dto = CreateDatasetDTO(
            name="new_dataset",
            description="A test dataset",
            target_column="target",
            file_path="/path/to/data.csv",
            file_format="csv",
            delimiter=",",
            encoding="utf-8"
        )
        
        assert dto.name == "new_dataset"
        assert dto.description == "A test dataset"
        assert dto.target_column == "target"
        assert dto.file_path == "/path/to/data.csv"
        assert dto.file_format == "csv"
        assert dto.delimiter == ","
        assert dto.encoding == "utf-8"
    
    def test_create_dataset_dto_validation(self):
        """Test CreateDatasetDTO validation."""
        # Test name length validation
        with pytest.raises(ValueError):
            CreateDatasetDTO(name="")  # Empty name
        
        with pytest.raises(ValueError):
            CreateDatasetDTO(name="a" * 101)  # Too long
        
        # Test file format validation
        with pytest.raises(ValueError):
            CreateDatasetDTO(
                name="test",
                file_format="invalid_format"
            )
    
    def test_data_quality_report_dto(self):
        """Test DataQualityReportDTO."""
        dto = DataQualityReportDTO(
            quality_score=0.85,
            n_missing_values=10,
            n_duplicates=5,
            n_outliers=15,
            missing_columns=["col1", "col2"],
            constant_columns=["col3"],
            high_cardinality_columns=["col4"],
            highly_correlated_features=[("col5", "col6", 0.95)],
            recommendations=["Remove constant columns", "Handle missing values"]
        )
        
        assert dto.quality_score == 0.85
        assert dto.n_missing_values == 10
        assert dto.n_duplicates == 5
        assert dto.n_outliers == 15
        assert len(dto.missing_columns) == 2
        assert len(dto.recommendations) == 2


@pytest.mark.skipif(not RESULT_DTOS_AVAILABLE, reason="Result DTOs not available")
class TestResultDTOs:
    """Test result-related DTOs."""
    
    def test_anomaly_dto(self):
        """Test AnomalyDTO."""
        dto = AnomalyDTO(
            id=uuid4(),
            score=0.95,
            data_point={"feature1": 1.0, "feature2": 2.0},
            detector_name="IsolationForest",
            timestamp=datetime.utcnow(),
            severity="high",
            explanation="High anomaly score detected",
            confidence_lower=0.85,
            confidence_upper=0.99
        )
        
        assert isinstance(dto.id, UUID)
        assert dto.score == 0.95
        assert dto.data_point == {"feature1": 1.0, "feature2": 2.0}
        assert dto.detector_name == "IsolationForest"
        assert dto.severity == "high"
        assert dto.explanation == "High anomaly score detected"
        assert dto.confidence_lower == 0.85
        assert dto.confidence_upper == 0.99
    
    def test_anomaly_dto_validation(self):
        """Test AnomalyDTO validation."""
        # Test score bounds
        with pytest.raises(ValueError):
            AnomalyDTO(
                id=uuid4(),
                score=-0.1,  # Invalid: negative
                data_point={},
                detector_name="test",
                timestamp=datetime.utcnow(),
                severity="low"
            )
        
        with pytest.raises(ValueError):
            AnomalyDTO(
                id=uuid4(),
                score=1.1,  # Invalid: > 1
                data_point={},
                detector_name="test",
                timestamp=datetime.utcnow(),
                severity="low"
            )
        
        # Test severity validation
        with pytest.raises(ValueError):
            AnomalyDTO(
                id=uuid4(),
                score=0.5,
                data_point={},
                detector_name="test",
                timestamp=datetime.utcnow(),
                severity="invalid"  # Invalid severity
            )
    
    def test_detection_result_dto(self):
        """Test DetectionResultDTO."""
        anomalies = [
            AnomalyDTO(
                id=uuid4(),
                score=0.95,
                data_point={"feature1": 1.0},
                detector_name="test",
                timestamp=datetime.utcnow(),
                severity="high"
            )
        ]
        
        dto = DetectionResultDTO(
            id=uuid4(),
            detector_id=uuid4(),
            dataset_id=uuid4(),
            created_at=datetime.utcnow(),
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
        
        assert isinstance(dto.id, UUID)
        assert isinstance(dto.detector_id, UUID)
        assert isinstance(dto.dataset_id, UUID)
        assert dto.duration_seconds == 1.5
        assert len(dto.anomalies) == 1
        assert dto.total_samples == 100
        assert dto.anomaly_count == 1
        assert dto.contamination_rate == 0.01
        assert dto.threshold == 0.5
    
    def test_detection_result_dto_validation(self):
        """Test DetectionResultDTO validation."""
        # Test contamination rate bounds
        with pytest.raises(ValueError):
            DetectionResultDTO(
                id=uuid4(),
                detector_id=uuid4(),
                dataset_id=uuid4(),
                created_at=datetime.utcnow(),
                duration_seconds=1.0,
                anomalies=[],
                total_samples=100,
                anomaly_count=0,
                contamination_rate=-0.1,  # Invalid: negative
                mean_score=0.1,
                max_score=0.1,
                min_score=0.1,
                threshold=0.5
            )


@pytest.mark.skipif(not EXPERIMENT_DTOS_AVAILABLE, reason="Experiment DTOs not available")
class TestExperimentDTOs:
    """Test experiment-related DTOs."""
    
    def test_create_experiment_dto(self):
        """Test CreateExperimentDTO."""
        dto = CreateExperimentDTO(
            name="fraud_detection_experiment",
            description="Testing algorithms for fraud detection",
            tags=["fraud", "financial", "anomaly_detection"]
        )
        
        assert dto.name == "fraud_detection_experiment"
        assert dto.description == "Testing algorithms for fraud detection"
        assert dto.tags == ["fraud", "financial", "anomaly_detection"]
    
    def test_create_experiment_dto_minimal(self):
        """Test CreateExperimentDTO with minimal data."""
        dto = CreateExperimentDTO(name="minimal_experiment")
        
        assert dto.name == "minimal_experiment"
        assert dto.description is None
        assert dto.tags == []  # default
    
    def test_experiment_response_dto(self):
        """Test ExperimentResponseDTO."""
        dto = ExperimentResponseDTO(
            id=uuid4(),
            name="test_experiment",
            description="Test description",
            tags=["test"],
            created_at=datetime.utcnow(),
            status="completed",
            n_runs=10
        )
        
        assert isinstance(dto.id, UUID)
        assert dto.name == "test_experiment"
        assert dto.status == "completed"
        assert dto.n_runs == 10


@pytest.mark.skipif(not (RESULT_DTOS_AVAILABLE and DETECTOR_DTOS_AVAILABLE), reason="DTOs not available for serialization tests")
class TestDTOSerialization:
    """Test DTO serialization and deserialization."""
    
    def test_nested_dto_serialization(self):
        """Test serialization of DTOs containing other DTOs."""
        anomalies = [
            AnomalyDTO(
                id=uuid4(),
                score=0.9,
                data_point={"feature1": 1.0},
                detector_name="test",
                timestamp=datetime.utcnow(),
                severity="high"
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
            anomaly_count=1,
            contamination_rate=0.01,
            mean_score=0.3,
            max_score=0.9,
            min_score=0.1,
            threshold=0.6
        )
        
        # Test serialization includes nested anomalies
        data = result_dto.model_dump()
        
        assert "anomalies" in data
        assert len(data["anomalies"]) == 1
        assert "score" in data["anomalies"][0]
        assert "data_point" in data["anomalies"][0]
    
    def test_uuid_consistency(self):
        """Test UUID handling consistency."""
        detector_id = uuid4()
        dataset_id = uuid4()
        
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
    
    def test_datetime_handling(self):
        """Test datetime serialization."""
        now = datetime.utcnow()
        dto = DetectorResponseDTO(
            id=uuid4(),
            name="test",
            algorithm_name="test",
            contamination_rate=0.1,
            is_fitted=True,
            created_at=now
        )
        
        data = dto.model_dump()
        assert "created_at" in data
        # Should be serializable
        assert data["created_at"] is not None
    
    def test_pydantic_config(self):
        """Test Pydantic configuration."""
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
        dto2 = DetectorResponseDTO(**data)
        
        assert dto2.name == dto.name
        assert dto2.algorithm_name == dto.algorithm_name
        assert dto2.contamination_rate == dto.contamination_rate