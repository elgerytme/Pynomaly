"""Comprehensive tests for all Data Transfer Objects (DTOs) - Phase 1 Coverage Enhancement."""

from __future__ import annotations

import numpy as np
import pytest
from datetime import datetime
from typing import Optional, Dict, Any
from uuid import UUID, uuid4
from unittest.mock import Mock

from pynomaly.application.dto import (
    # Detector DTOs
    DetectorDTO,
    CreateDetectorDTO,
    UpdateDetectorDTO,
    # Dataset DTOs
    DatasetDTO,
    CreateDatasetDTO,
    DataQualityReportDTO,
    # Result DTOs  
    DetectionResultDTO,
    AnomalyDTO,
    # Experiment DTOs
    ExperimentDTO,
    RunDTO,
    CreateExperimentDTO,
    LeaderboardEntryDTO,
)
from pynomaly.domain.entities import Dataset, Detector, DetectionResult, Anomaly
from pynomaly.domain.value_objects import ContaminationRate, AnomalyScore, ConfidenceInterval


class TestDetectorDTO:
    """Comprehensive tests for DetectorDTO."""
    
    def test_detector_dto_creation(self):
        """Test creating a DetectorDTO with all fields."""
        detector_id = uuid4()
        created_at = datetime.now()
        trained_at = datetime.now()
        
        dto = DetectorDTO(
            id=detector_id,
            name="Advanced Isolation Forest",
            algorithm_name="isolation_forest",
            contamination_rate=0.05,
            is_fitted=True,
            created_at=created_at,
            trained_at=trained_at,
            parameters={"n_estimators": 200, "max_samples": 512},
            metadata={"complexity": "medium", "use_case": "fraud_detection"},
            requires_fitting=True,
            supports_streaming=False,
            supports_multivariate=True,
            time_complexity="O(n log n)",
            space_complexity="O(n)"
        )
        
        assert dto.id == detector_id
        assert dto.name == "Advanced Isolation Forest"
        assert dto.algorithm_name == "isolation_forest"
        assert dto.contamination_rate == 0.05
        assert dto.is_fitted is True
        assert dto.created_at == created_at
        assert dto.trained_at == trained_at
        assert dto.parameters["n_estimators"] == 200
        assert dto.metadata["complexity"] == "medium"
        assert dto.time_complexity == "O(n log n)"
    
    def test_detector_dto_validation(self):
        """Test DetectorDTO field validation."""
        detector_id = uuid4()
        created_at = datetime.now()
        
        # Valid contamination rate boundaries
        DetectorDTO(
            id=detector_id, name="test", algorithm_name="test",
            contamination_rate=0.0, is_fitted=False, created_at=created_at
        )
        DetectorDTO(
            id=detector_id, name="test", algorithm_name="test",
            contamination_rate=1.0, is_fitted=False, created_at=created_at
        )
        
        # Invalid contamination rates
        with pytest.raises(ValueError, match="greater than or equal to 0"):
            DetectorDTO(
                id=detector_id, name="test", algorithm_name="test",
                contamination_rate=-0.1, is_fitted=False, created_at=created_at
            )
        
        with pytest.raises(ValueError, match="less than or equal to 1"):
            DetectorDTO(
                id=detector_id, name="test", algorithm_name="test",
                contamination_rate=1.5, is_fitted=False, created_at=created_at
            )
    
    def test_detector_dto_optional_fields(self):
        """Test DetectorDTO with optional fields."""
        dto = DetectorDTO(
            id=uuid4(),
            name="test",
            algorithm_name="test",
            contamination_rate=0.1,
            is_fitted=False,
            created_at=datetime.now()
        )
        
        assert dto.trained_at is None
        assert dto.parameters == {}
        assert dto.metadata == {}
        assert dto.time_complexity is None
        assert dto.space_complexity is None


class TestCreateDetectorDTO:
    """Comprehensive tests for CreateDetectorDTO."""
    
    def test_create_detector_dto_valid(self):
        """Test creating a valid CreateDetectorDTO."""
        dto = CreateDetectorDTO(
            name="Fraud Detection Model",
            algorithm_name="isolation_forest",
            contamination_rate=0.02,
            parameters={"n_estimators": 150, "random_state": 42},
            metadata={"purpose": "credit_card_fraud", "version": "1.0"}
        )
        
        assert dto.name == "Fraud Detection Model"
        assert dto.algorithm_name == "isolation_forest"
        assert dto.contamination_rate == 0.02
        assert dto.parameters["n_estimators"] == 150
        assert dto.metadata["purpose"] == "credit_card_fraud"
    
    def test_create_detector_dto_defaults(self):
        """Test CreateDetectorDTO with default values."""
        dto = CreateDetectorDTO(
            name="Test Detector",
            algorithm_name="lof"
        )
        
        assert dto.contamination_rate == 0.1  # Default
        assert dto.parameters == {}
        assert dto.metadata == {}
    
    def test_create_detector_dto_validation(self):
        """Test CreateDetectorDTO validation."""
        # Valid name lengths
        CreateDetectorDTO(name="a", algorithm_name="test")  # Min length 1
        CreateDetectorDTO(name="a" * 100, algorithm_name="test")  # Max length 100
        
        # Invalid names
        with pytest.raises(ValueError, match="at least 1 character"):
            CreateDetectorDTO(name="", algorithm_name="test")
        
        with pytest.raises(ValueError, match="at most 100 characters"):
            CreateDetectorDTO(name="a" * 101, algorithm_name="test")
        
        # Invalid algorithm name
        with pytest.raises(ValueError, match="at least 1 character"):
            CreateDetectorDTO(name="test", algorithm_name="")
        
        # Invalid contamination rate
        with pytest.raises(ValueError):
            CreateDetectorDTO(name="test", algorithm_name="test", contamination_rate=-0.1)
        
        with pytest.raises(ValueError):
            CreateDetectorDTO(name="test", algorithm_name="test", contamination_rate=1.5)


class TestUpdateDetectorDTO:
    """Comprehensive tests for UpdateDetectorDTO."""
    
    def test_update_detector_dto_partial_update(self):
        """Test updating selected fields only."""
        dto = UpdateDetectorDTO(
            name="Updated Name",
            contamination_rate=0.15
        )
        
        assert dto.name == "Updated Name"
        assert dto.contamination_rate == 0.15
        assert dto.parameters is None
        assert dto.metadata is None
    
    def test_update_detector_dto_validation(self):
        """Test UpdateDetectorDTO validation."""
        # Valid updates
        UpdateDetectorDTO(name="a")  # Min length
        UpdateDetectorDTO(name="a" * 100)  # Max length
        UpdateDetectorDTO(contamination_rate=0.0)  # Min rate
        UpdateDetectorDTO(contamination_rate=1.0)  # Max rate
        
        # Invalid updates
        with pytest.raises(ValueError):
            UpdateDetectorDTO(name="")
        
        with pytest.raises(ValueError):
            UpdateDetectorDTO(name="a" * 101)
        
        with pytest.raises(ValueError):
            UpdateDetectorDTO(contamination_rate=-0.1)
        
        with pytest.raises(ValueError):
            UpdateDetectorDTO(contamination_rate=1.5)
    
    def test_update_detector_dto_all_none(self):
        """Test UpdateDetectorDTO with all None values."""
        dto = UpdateDetectorDTO()
        
        assert dto.name is None
        assert dto.contamination_rate is None
        assert dto.parameters is None
        assert dto.metadata is None


class TestDatasetDTO:
    """Comprehensive tests for DatasetDTO."""
    
    def test_dataset_dto_creation(self):
        """Test creating a comprehensive DatasetDTO."""
        dataset_id = uuid4()
        created_at = datetime.now()
        
        dto = DatasetDTO(
            id=dataset_id,
            name="Credit Card Transactions",
            shape=(50000, 30),
            n_samples=50000,
            n_features=30,
            feature_names=[f"feature_{i}" for i in range(30)],
            has_target=True,
            target_column="is_fraud",
            created_at=created_at,
            metadata={"source": "bank_a", "period": "2024_q1"},
            description="Credit card transaction data for fraud detection",
            memory_usage_mb=15.7,
            numeric_features=25,
            categorical_features=5
        )
        
        assert dto.id == dataset_id
        assert dto.name == "Credit Card Transactions"
        assert dto.shape == (50000, 30)
        assert dto.n_samples == 50000
        assert dto.n_features == 30
        assert len(dto.feature_names) == 30
        assert dto.has_target is True
        assert dto.target_column == "is_fraud"
        assert dto.created_at == created_at
        assert dto.metadata["source"] == "bank_a"
        assert dto.description == "Credit card transaction data for fraud detection"
        assert dto.memory_usage_mb == 15.7
        assert dto.numeric_features == 25
        assert dto.categorical_features == 5
    
    def test_dataset_dto_optional_fields(self):
        """Test DatasetDTO with optional fields."""
        dto = DatasetDTO(
            id=uuid4(),
            name="Test Dataset",
            shape=(100, 5),
            n_samples=100,
            n_features=5,
            feature_names=["f1", "f2", "f3", "f4", "f5"],
            has_target=False,
            created_at=datetime.now(),
            memory_usage_mb=0.5,
            numeric_features=5,
            categorical_features=0
        )
        
        assert dto.target_column is None
        assert dto.metadata == {}
        assert dto.description is None
    
    def test_dataset_dto_json_schema(self):
        """Test DatasetDTO JSON schema example."""
        # The DTO should be serializable with the provided schema example
        dto = DatasetDTO(
            id=UUID("123e4567-e89b-12d3-a456-426614174000"),
            name="credit_card_transactions",
            shape=(10000, 30),
            n_samples=10000,
            n_features=30,
            feature_names=["amount", "merchant_id", "time"],
            has_target=True,
            target_column="is_fraud",
            created_at=datetime(2024, 1, 1),
            memory_usage_mb=2.4,
            numeric_features=25,
            categorical_features=5
        )
        
        serialized = dto.model_dump()
        assert serialized["name"] == "credit_card_transactions"
        assert serialized["n_samples"] == 10000
        assert serialized["has_target"] is True


class TestCreateDatasetDTO:
    """Comprehensive tests for CreateDatasetDTO."""
    
    def test_create_dataset_dto_comprehensive(self):
        """Test creating a comprehensive CreateDatasetDTO."""
        dto = CreateDatasetDTO(
            name="IoT Sensor Data",
            description="Temperature and humidity sensor readings from manufacturing plant",
            target_column="anomaly_flag",
            metadata={"plant_id": "plant_001", "sensor_count": 150},
            file_path="/data/sensors/readings_2024.csv",
            file_format="csv",
            delimiter=",",
            encoding="utf-8",
            parse_dates=["timestamp", "last_calibration"],
            dtype={"sensor_id": "str", "temperature": "float32", "humidity": "float32"}
        )
        
        assert dto.name == "IoT Sensor Data"
        assert dto.description == "Temperature and humidity sensor readings from manufacturing plant"
        assert dto.target_column == "anomaly_flag"
        assert dto.metadata["plant_id"] == "plant_001"
        assert dto.file_path == "/data/sensors/readings_2024.csv"
        assert dto.file_format == "csv"
        assert dto.delimiter == ","
        assert dto.encoding == "utf-8"
        assert "timestamp" in dto.parse_dates
        assert dto.dtype["sensor_id"] == "str"
    
    def test_create_dataset_dto_validation(self):
        """Test CreateDatasetDTO validation."""
        # Valid name lengths
        CreateDatasetDTO(name="a")  # Min length 1
        CreateDatasetDTO(name="a" * 100)  # Max length 100
        
        # Valid description lengths
        CreateDatasetDTO(name="test", description="a" * 500)  # Max length 500
        
        # Valid file formats
        for format in ["csv", "parquet", "json", "excel"]:
            CreateDatasetDTO(name="test", file_format=format)
        
        # Invalid inputs
        with pytest.raises(ValueError, match="at least 1 character"):
            CreateDatasetDTO(name="")
        
        with pytest.raises(ValueError, match="at most 100 characters"):
            CreateDatasetDTO(name="a" * 101)
        
        with pytest.raises(ValueError, match="at most 500 characters"):
            CreateDatasetDTO(name="test", description="a" * 501)
        
        with pytest.raises(ValueError, match="does not match"):
            CreateDatasetDTO(name="test", file_format="invalid_format")
    
    def test_create_dataset_dto_defaults(self):
        """Test CreateDatasetDTO default values."""
        dto = CreateDatasetDTO(name="Test Dataset")
        
        assert dto.description is None
        assert dto.target_column is None
        assert dto.metadata == {}
        assert dto.file_path is None
        assert dto.file_format is None
        assert dto.delimiter == ","
        assert dto.encoding == "utf-8"
        assert dto.parse_dates is None
        assert dto.dtype is None


class TestDataQualityReportDTO:
    """Comprehensive tests for DataQualityReportDTO."""
    
    def test_data_quality_report_dto_comprehensive(self):
        """Test creating a comprehensive DataQualityReportDTO."""
        dto = DataQualityReportDTO(
            quality_score=0.85,
            n_missing_values=127,
            n_duplicates=45,
            n_outliers=89,
            missing_columns=["last_login", "optional_field"],
            constant_columns=["status", "version"],
            high_cardinality_columns=["user_id", "session_id"],
            highly_correlated_features=[
                ("feature_a", "feature_b", 0.95),
                ("price", "price_usd", 0.99)
            ],
            recommendations=[
                "Remove constant columns: status, version",
                "Handle missing values in last_login column",
                "Consider feature selection for highly correlated features"
            ]
        )
        
        assert dto.quality_score == 0.85
        assert dto.n_missing_values == 127
        assert dto.n_duplicates == 45
        assert dto.n_outliers == 89
        assert "last_login" in dto.missing_columns
        assert "status" in dto.constant_columns
        assert "user_id" in dto.high_cardinality_columns
        assert len(dto.highly_correlated_features) == 2
        assert dto.highly_correlated_features[0][2] == 0.95
        assert len(dto.recommendations) == 3
    
    def test_data_quality_report_dto_validation(self):
        """Test DataQualityReportDTO validation."""
        # Valid quality scores
        DataQualityReportDTO(quality_score=0.0, n_missing_values=0, n_duplicates=0, n_outliers=0)
        DataQualityReportDTO(quality_score=1.0, n_missing_values=0, n_duplicates=0, n_outliers=0)
        
        # Invalid quality scores
        with pytest.raises(ValueError, match="greater than or equal to 0"):
            DataQualityReportDTO(quality_score=-0.1, n_missing_values=0, n_duplicates=0, n_outliers=0)
        
        with pytest.raises(ValueError, match="less than or equal to 1"):
            DataQualityReportDTO(quality_score=1.5, n_missing_values=0, n_duplicates=0, n_outliers=0)
    
    def test_data_quality_report_dto_defaults(self):
        """Test DataQualityReportDTO default values."""
        dto = DataQualityReportDTO(
            quality_score=0.9,
            n_missing_values=0,
            n_duplicates=0,
            n_outliers=0
        )
        
        assert dto.missing_columns == []
        assert dto.constant_columns == []
        assert dto.high_cardinality_columns == []
        assert dto.highly_correlated_features == []
        assert dto.recommendations == []


class TestDetectionResultDTO:
    """Enhanced tests for DetectionResultDTO."""
    
    def test_detection_result_dto_from_domain_comprehensive(self):
        """Test comprehensive DetectionResultDTO creation from domain entity."""
        # Create domain entities
        detector = Detector(
            name="Advanced LOF",
            algorithm="local_outlier_factor",
            contamination=ContaminationRate(0.05),
            hyperparameters={"n_neighbors": 25, "algorithm": "auto"}
        )
        
        features = np.random.RandomState(42).normal(0, 1, (200, 8))
        dataset = Dataset(name="comprehensive_test_dataset", features=features)
        
        # Create realistic anomaly scores
        scores = np.random.RandomState(42).beta(2, 8, 200)  # Most scores low, few high
        
        # Create anomalies with features and confidence intervals
        anomalies = []
        high_score_indices = np.where(scores > 0.9)[0]
        for idx in high_score_indices[:5]:  # Top 5 anomalies
            confidence = ConfidenceInterval(
                lower=max(0, scores[idx] - 0.05),
                upper=min(1, scores[idx] + 0.05),
                confidence_level=0.95
            )
            anomaly = Anomaly(
                score=AnomalyScore(scores[idx]),
                index=int(idx),
                features=features[idx],
                confidence=confidence,
                explanation={"feature_importance": {f"feature_{i}": np.random.random() for i in range(8)}}
            )
            anomalies.append(anomaly)
        
        result = DetectionResult(
            detector=detector,
            dataset=dataset,
            anomalies=anomalies,
            scores=scores,
            metadata={
                "processing_time": 1.47,
                "algorithm_specific": {"n_neighbors_used": 25},
                "quality_metrics": {"stability_score": 0.92}
            }
        )
        
        dto = DetectionResultDTO.from_domain_entity(result)
        
        assert dto.detector_id == detector.id
        assert dto.dataset_name == "comprehensive_test_dataset"
        assert len(dto.anomalies) == len(anomalies)
        assert len(dto.scores) == 200
        assert dto.total_samples == 200
        assert dto.anomaly_count == len(anomalies)
        assert 0 <= dto.contamination_rate <= 1
        assert 0 <= dto.max_score <= 1
        assert 0 <= dto.min_score <= 1
        assert dto.metadata["processing_time"] == 1.47
    
    def test_detection_result_dto_statistical_accuracy(self):
        """Test statistical accuracy of DetectionResultDTO calculations."""
        detector = Detector(
            name="test", algorithm="test", contamination=ContaminationRate(0.1)
        )
        features = np.random.random((100, 5))
        dataset = Dataset(name="test", features=features)
        
        # Controlled scores for testing
        scores = np.array([0.1] * 90 + [0.9] * 10)  # 10% high scores
        anomalies = [
            Anomaly(score=AnomalyScore(0.9), index=i) 
            for i in range(90, 100)
        ]
        
        result = DetectionResult(
            detector=detector, dataset=dataset, anomalies=anomalies, scores=scores
        )
        
        dto = DetectionResultDTO.from_domain_entity(result)
        
        assert dto.total_samples == 100
        assert dto.anomaly_count == 10
        assert dto.contamination_rate == 0.1  # 10/100
        assert dto.max_score == 0.9
        assert dto.min_score == 0.1
        assert abs(dto.mean_score - np.mean(scores)) < 1e-10


class TestAnomalyDTO:
    """Enhanced tests for AnomalyDTO."""
    
    def test_anomaly_dto_comprehensive(self):
        """Test comprehensive AnomalyDTO creation."""
        features = np.array([2.5, -1.8, 0.7, 4.2, -0.3])
        confidence = ConfidenceInterval(lower=0.87, upper=0.96, confidence_level=0.95)
        explanation = {
            "feature_importance": {
                "feature_0": 0.35,
                "feature_1": 0.28,
                "feature_2": 0.15,
                "feature_3": 0.12,
                "feature_4": 0.10
            },
            "local_explanation": "High deviation in features 0 and 1",
            "shap_values": [0.15, -0.08, 0.03, 0.12, -0.02]
        }
        
        anomaly = Anomaly(
            score=AnomalyScore(0.94),
            index=127,
            features=features,
            confidence=confidence,
            explanation=explanation
        )
        
        dto = AnomalyDTO.from_domain_entity(anomaly)
        
        assert dto.score == 0.94
        assert dto.index == 127
        assert dto.features == features.tolist()
        assert dto.confidence_lower == 0.87
        assert dto.confidence_upper == 0.96
        assert dto.explanation["feature_importance"]["feature_0"] == 0.35
        assert dto.explanation["local_explanation"] == "High deviation in features 0 and 1"
        assert len(dto.explanation["shap_values"]) == 5


# Note: ExperimentDTO, RunDTO, and LeaderboardEntryDTO tests would require 
# reading their implementations first. For now, focusing on the core DTOs 
# that are most critical for Phase 1 coverage improvement.


class TestDTOIntegration:
    """Integration tests for DTOs working together."""
    
    def test_dto_serialization_roundtrip(self):
        """Test that DTOs can be serialized and deserialized correctly."""
        # Test CreateDetectorDTO serialization
        create_dto = CreateDetectorDTO(
            name="Test Detector",
            algorithm_name="isolation_forest",
            contamination_rate=0.08,
            parameters={"n_estimators": 100, "random_state": 42},
            metadata={"purpose": "testing"}
        )
        
        serialized = create_dto.model_dump()
        deserialized = CreateDetectorDTO(**serialized)
        
        assert deserialized.name == create_dto.name
        assert deserialized.algorithm_name == create_dto.algorithm_name
        assert deserialized.contamination_rate == create_dto.contamination_rate
        assert deserialized.parameters == create_dto.parameters
        assert deserialized.metadata == create_dto.metadata
    
    def test_dto_validation_edge_cases(self):
        """Test DTO validation with edge cases."""
        # Test boundary values
        CreateDetectorDTO(
            name="a",  # Minimum length
            algorithm_name="t",  # Minimum length
            contamination_rate=0.0  # Minimum value
        )
        
        CreateDetectorDTO(
            name="a" * 100,  # Maximum length
            algorithm_name="test",
            contamination_rate=1.0  # Maximum value
        )
        
        # Test data quality with edge cases
        DataQualityReportDTO(
            quality_score=0.0,  # Minimum quality
            n_missing_values=0,
            n_duplicates=0,
            n_outliers=0
        )
        
        DataQualityReportDTO(
            quality_score=1.0,  # Perfect quality
            n_missing_values=0,
            n_duplicates=0,
            n_outliers=0
        )
    
    def test_dto_type_safety(self):
        """Test DTO type safety and proper validation."""
        # Test that invalid types raise appropriate errors
        with pytest.raises(ValueError):
            DetectorDTO(
                id="not-a-uuid",  # Should be UUID
                name="test",
                algorithm_name="test",
                contamination_rate=0.1,
                is_fitted=False,
                created_at=datetime.now()
            )
        
        with pytest.raises(ValueError):
            CreateDatasetDTO(
                name="test",
                shape="not-a-tuple"  # Should be tuple
            )