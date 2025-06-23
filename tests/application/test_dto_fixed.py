"""Comprehensive tests for Data Transfer Objects (DTOs)."""

from __future__ import annotations

import numpy as np
import pytest
from datetime import datetime
from typing import Optional, Dict, Any
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
    AutoMLRequestDTO,
    AutoMLResponseDTO,
    ExplanationRequestDTO,
    ExplanationResponseDTO,
    FeatureContributionDTO,
    LocalExplanationDTO,
)


class TestCreateDetectorDTO:
    """Test CreateDetectorDTO validation and conversion."""
    
    def test_valid_dto_creation(self):
        """Test creating a valid CreateDetectorDTO."""
        dto = CreateDetectorDTO(
            name="test_detector",
            algorithm_name="IsolationForest",
            contamination_rate=0.1,
            parameters={"n_estimators": 100, "random_state": 42}
        )
        
        assert dto.name == "test_detector"
        assert dto.algorithm_name == "IsolationForest"
        assert dto.contamination_rate == 0.1
        assert dto.parameters["n_estimators"] == 100
    
    def test_contamination_validation(self):
        """Test contamination rate validation."""
        # Valid contamination rates
        CreateDetectorDTO(name="test", algorithm_name="test", contamination_rate=0.05)
        CreateDetectorDTO(name="test", algorithm_name="test", contamination_rate=0.5)
        
        # Invalid contamination rates
        with pytest.raises(ValueError):
            CreateDetectorDTO(name="test", algorithm_name="test", contamination_rate=-0.1)
        
        with pytest.raises(ValueError):
            CreateDetectorDTO(name="test", algorithm_name="test", contamination_rate=1.5)
    
    def test_name_validation(self):
        """Test name validation."""
        # Valid names
        CreateDetectorDTO(name="valid_name", algorithm_name="test", contamination_rate=0.1)
        CreateDetectorDTO(name="Valid Name 123", algorithm_name="test", contamination_rate=0.1)
        
        # Invalid names
        with pytest.raises(ValueError):
            CreateDetectorDTO(name="", algorithm_name="test", contamination_rate=0.1)
    
    def test_algorithm_validation(self):
        """Test algorithm validation."""
        valid_algorithms = [
            "IsolationForest", "LocalOutlierFactor", "OneClassSVM",
            "LOF", "KNN", "ABOD"
        ]
        
        for algo in valid_algorithms:
            CreateDetectorDTO(name="test", algorithm_name=algo, contamination_rate=0.1)
        
        # Test with empty algorithm name should raise validation error
        with pytest.raises(ValueError):
            CreateDetectorDTO(name="test", algorithm_name="", contamination_rate=0.1)
    
    def test_parameters_optional(self):
        """Test that parameters are optional."""
        dto = CreateDetectorDTO(name="test", algorithm_name="IsolationForest", contamination_rate=0.1)
        assert dto.parameters == {}
    
    def test_json_serialization(self):
        """Test JSON serialization of DTO."""
        dto = CreateDetectorDTO(
            name="test_detector",
            algorithm_name="IsolationForest",
            contamination_rate=0.1,
            parameters={"n_estimators": 100}
        )
        
        # Test that it can be serialized
        json_data = dto.model_dump()
        
        assert json_data["name"] == "test_detector"
        assert json_data["algorithm_name"] == "IsolationForest"
        assert json_data["contamination_rate"] == 0.1
        assert json_data["parameters"] == {"n_estimators": 100}


class TestDetectorResponseDTO:
    """Test DetectorResponseDTO serialization."""
    
    def test_valid_response_dto(self):
        """Test creating a valid DetectorResponseDTO."""
        dto = DetectorResponseDTO(
            id=uuid4(),
            name="test_detector",
            algorithm_name="IsolationForest",
            contamination_rate=0.1,
            is_fitted=True,
            created_at=datetime.utcnow(),
            parameters={"n_estimators": 100}
        )
        
        assert dto.name == "test_detector"
        assert dto.algorithm_name == "IsolationForest"
        assert dto.contamination_rate == 0.1
        assert dto.is_fitted is True
        assert dto.parameters == {"n_estimators": 100}
        assert dto.status == "active"
        assert dto.version == "1.0.0"
    
    def test_serialization(self):
        """Test JSON serialization."""
        dto = DetectorResponseDTO(
            id=uuid4(),
            name="test_detector",
            algorithm_name="IsolationForest",
            contamination_rate=0.1,
            is_fitted=False,
            created_at=datetime.utcnow()
        )
        
        serialized = dto.model_dump()
        
        assert "id" in serialized
        assert "name" in serialized
        assert "algorithm_name" in serialized
        assert "contamination_rate" in serialized
        assert "created_at" in serialized
        assert "is_fitted" in serialized
        assert "status" in serialized
        assert "version" in serialized


class TestDatasetDTO:
    """Test DatasetDTO validation and conversion."""
    
    def test_valid_dataset_dto(self):
        """Test creating a valid DatasetDTO."""
        dto = DatasetDTO(
            id=uuid4(),
            name="test_dataset",
            shape=(1000, 10),
            n_samples=1000,
            n_features=10,
            feature_names=["feature_" + str(i) for i in range(10)],
            has_target=True,
            target_column="target",
            created_at=datetime.utcnow(),
            memory_usage_mb=1.5,
            numeric_features=8,
            categorical_features=2
        )
        
        assert dto.name == "test_dataset"
        assert dto.shape == (1000, 10)
        assert dto.n_samples == 1000
        assert dto.n_features == 10
        assert dto.has_target is True
        assert dto.target_column == "target"
        assert len(dto.feature_names) == 10
    
    def test_dataset_without_target(self):
        """Test dataset DTO without targets."""
        dto = DatasetDTO(
            id=uuid4(),
            name="test_dataset",
            shape=(500, 5),
            n_samples=500,
            n_features=5,
            feature_names=["feature_" + str(i) for i in range(5)],
            has_target=False,
            created_at=datetime.utcnow(),
            memory_usage_mb=0.8,
            numeric_features=5,
            categorical_features=0
        )
        
        assert dto.name == "test_dataset"
        assert dto.has_target is False
        assert dto.target_column is None
    
    def test_serialization(self):
        """Test dataset DTO serialization."""
        dto = DatasetDTO(
            id=uuid4(),
            name="test_dataset",
            shape=(100, 3),
            n_samples=100,
            n_features=3,
            feature_names=["a", "b", "c"],
            has_target=False,
            created_at=datetime.utcnow(),
            memory_usage_mb=0.1,
            numeric_features=3,
            categorical_features=0
        )
        
        serialized = dto.model_dump()
        
        assert "id" in serialized
        assert "name" in serialized
        assert "shape" in serialized
        assert "n_samples" in serialized
        assert "n_features" in serialized
        assert "feature_names" in serialized
        assert "has_target" in serialized


class TestDetectionRequestDTO:
    """Test DetectionRequestDTO validation."""
    
    def test_valid_detection_request(self):
        """Test creating a valid DetectionRequestDTO."""
        dto = DetectionRequestDTO(
            detector_id=uuid4(),
            data=[[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]],
            return_scores=True,
            return_feature_importance=False
        )
        
        assert isinstance(dto.detector_id, UUID)
        assert len(dto.data) == 2
        assert len(dto.data[0]) == 3
        assert dto.return_scores is True
        assert dto.return_feature_importance is False
    
    def test_optional_parameters(self):
        """Test optional parameters in detection request."""
        dto = DetectionRequestDTO(
            detector_id=uuid4(),
            data=[[1.0, 2.0]]
        )
        
        assert dto.return_scores is True  # default
        assert dto.return_feature_importance is False  # default
        assert dto.threshold is None  # default
    
    def test_with_threshold(self):
        """Test detection request with custom threshold."""
        dto = DetectionRequestDTO(
            detector_id=uuid4(),
            data=[[1.0, 2.0]],
            threshold=0.5
        )
        
        assert dto.threshold == 0.5


class TestDetectionResultDTO:
    """Test DetectionResultDTO validation."""
    
    def test_valid_detection_result(self):
        """Test creating a valid DetectionResultDTO."""
        dto = DetectionResultDTO(
            id=uuid4(),
            detector_id=uuid4(),
            dataset_id=uuid4(),
            anomaly_scores=[0.1, 0.8, 0.3],
            predictions=[0, 1, 0],
            threshold=0.5,
            n_anomalies=1,
            anomaly_rate=0.33,
            execution_time_ms=150.5,
            created_at=datetime.utcnow()
        )
        
        assert len(dto.anomaly_scores) == 3
        assert len(dto.predictions) == 3
        assert dto.threshold == 0.5
        assert dto.n_anomalies == 1
        assert dto.anomaly_rate == 0.33
    
    def test_serialization(self):
        """Test detection result serialization."""
        dto = DetectionResultDTO(
            id=uuid4(),
            detector_id=uuid4(),
            dataset_id=uuid4(),
            anomaly_scores=[0.1, 0.9],
            predictions=[0, 1],
            threshold=0.5,
            n_anomalies=1,
            anomaly_rate=0.5,
            execution_time_ms=100.0,
            created_at=datetime.utcnow()
        )
        
        serialized = dto.model_dump()
        
        assert "id" in serialized
        assert "detector_id" in serialized
        assert "dataset_id" in serialized
        assert "anomaly_scores" in serialized
        assert "predictions" in serialized
        assert "threshold" in serialized


class TestAnomalyDTO:
    """Test AnomalyDTO validation."""
    
    def test_valid_anomaly_dto(self):
        """Test creating a valid AnomalyDTO."""
        dto = AnomalyDTO(
            index=42,
            score=0.95,
            data_point={"feature1": 1.0, "feature2": 2.0},
            detector_name="IsolationForest",
            confidence=0.8
        )
        
        assert dto.index == 42
        assert dto.score == 0.95
        assert dto.data_point == {"feature1": 1.0, "feature2": 2.0}
        assert dto.detector_name == "IsolationForest"
        assert dto.confidence == 0.8
    
    def test_optional_confidence(self):
        """Test anomaly DTO with optional confidence."""
        dto = AnomalyDTO(
            index=0,
            score=0.9,
            data_point={},
            detector_name="LOF"
        )
        
        assert dto.confidence is None


class TestCreateExperimentDTO:
    """Test CreateExperimentDTO validation."""
    
    def test_valid_experiment_creation(self):
        """Test creating a valid CreateExperimentDTO."""
        dto = CreateExperimentDTO(
            name="test_experiment",
            description="A test experiment",
            tags=["test", "anomaly_detection"]
        )
        
        assert dto.name == "test_experiment"
        assert dto.description == "A test experiment"
        assert dto.tags == ["test", "anomaly_detection"]
    
    def test_minimal_experiment(self):
        """Test creating experiment with minimal required fields."""
        dto = CreateExperimentDTO(name="minimal_experiment")
        
        assert dto.name == "minimal_experiment"
        assert dto.description is None
        assert dto.tags == []


class TestExperimentResponseDTO:
    """Test ExperimentResponseDTO validation."""
    
    def test_valid_experiment_response(self):
        """Test creating a valid ExperimentResponseDTO."""
        dto = ExperimentResponseDTO(
            id=uuid4(),
            name="test_experiment",
            description="Test description",
            tags=["test"],
            created_at=datetime.utcnow(),
            status="running",
            n_runs=5
        )
        
        assert dto.name == "test_experiment"
        assert dto.status == "running"
        assert dto.n_runs == 5


class TestAutoMLDTO:
    """Test AutoML-related DTOs."""
    
    def test_automl_request_dto(self):
        """Test AutoMLRequestDTO creation."""
        dto = AutoMLRequestDTO(
            dataset_id=uuid4(),
            target_metric="f1_score",
            max_trials=50,
            timeout_minutes=30
        )
        
        assert isinstance(dto.dataset_id, UUID)
        assert dto.target_metric == "f1_score"
        assert dto.max_trials == 50
        assert dto.timeout_minutes == 30
    
    def test_automl_response_dto(self):
        """Test AutoMLResponseDTO creation."""
        dto = AutoMLResponseDTO(
            id=uuid4(),
            status="completed",
            best_score=0.95,
            trials_completed=45,
            execution_time_seconds=1800,
            created_at=datetime.utcnow()
        )
        
        assert dto.status == "completed"
        assert dto.best_score == 0.95
        assert dto.trials_completed == 45


class TestExplainabilityDTO:
    """Test Explainability-related DTOs."""
    
    def test_explanation_request_dto(self):
        """Test ExplanationRequestDTO creation."""
        dto = ExplanationRequestDTO(
            detector_id=uuid4(),
            dataset_id=uuid4(),
            method="shap",
            instance_indices=[0, 1, 2]
        )
        
        assert isinstance(dto.detector_id, UUID)
        assert isinstance(dto.dataset_id, UUID)
        assert dto.method == "shap"
        assert dto.instance_indices == [0, 1, 2]
    
    def test_feature_contribution_dto(self):
        """Test FeatureContributionDTO creation."""
        dto = FeatureContributionDTO(
            feature_name="age",
            contribution=0.25,
            rank=1,
            confidence=0.9
        )
        
        assert dto.feature_name == "age"
        assert dto.contribution == 0.25
        assert dto.rank == 1
        assert dto.confidence == 0.9
    
    def test_local_explanation_dto(self):
        """Test LocalExplanationDTO creation."""
        feature_contributions = [
            FeatureContributionDTO(feature_name="age", contribution=0.3, rank=1),
            FeatureContributionDTO(feature_name="income", contribution=0.2, rank=2)
        ]
        
        dto = LocalExplanationDTO(
            instance_id=42,
            prediction_score=0.8,
            feature_contributions=feature_contributions,
            explanation_quality=0.9
        )
        
        assert dto.instance_id == 42
        assert dto.prediction_score == 0.8
        assert len(dto.feature_contributions) == 2
        assert dto.explanation_quality == 0.9


class TestDTOValidation:
    """Test general DTO validation patterns."""
    
    def test_pydantic_validation_errors(self):
        """Test that Pydantic validation works correctly."""
        with pytest.raises(ValueError):
            # Missing required field
            CreateDetectorDTO(name="test")
        
        with pytest.raises(ValueError):
            # Invalid contamination rate type
            CreateDetectorDTO(
                name="test",
                algorithm_name="test",
                contamination_rate="invalid"
            )
    
    def test_uuid_validation(self):
        """Test UUID field validation."""
        # Valid UUID
        DetectionRequestDTO(
            detector_id=uuid4(),
            data=[[1.0, 2.0]]
        )
        
        # Invalid UUID should raise validation error
        with pytest.raises(ValueError):
            DetectionRequestDTO(
                detector_id="not-a-uuid",
                data=[[1.0, 2.0]]
            )
    
    def test_datetime_serialization(self):
        """Test datetime serialization in DTOs."""
        now = datetime.utcnow()
        dto = DetectorResponseDTO(
            id=uuid4(),
            name="test",
            algorithm_name="test",
            contamination_rate=0.1,
            is_fitted=False,
            created_at=now
        )
        
        serialized = dto.model_dump()
        assert "created_at" in serialized
        # Datetime should be serialized as ISO string
        assert isinstance(serialized["created_at"], str) or isinstance(serialized["created_at"], datetime)