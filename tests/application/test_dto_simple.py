"""Simple validation tests for DTOs to check what works."""

from __future__ import annotations

import pytest
from datetime import datetime
from uuid import uuid4
from typing import Dict, Any

# Test simple DTOs without complex dependencies
try:
    from pynomaly.application.dto.detector_dto import CreateDetectorDTO, DetectorResponseDTO, DetectionRequestDTO
    DETECTOR_AVAILABLE = True
except ImportError as e:
    print(f"Detector DTOs not available: {e}")
    DETECTOR_AVAILABLE = False

try:
    from pynomaly.application.dto.dataset_dto import CreateDatasetDTO, DataQualityReportDTO
    DATASET_AVAILABLE = True
except ImportError as e:
    print(f"Dataset DTOs not available: {e}")
    DATASET_AVAILABLE = False

try:
    from pynomaly.application.dto.result_dto import AnomalyDTO
    RESULT_AVAILABLE = True
except ImportError as e:
    print(f"Result DTOs not available: {e}")
    RESULT_AVAILABLE = False

try:
    from pynomaly.application.dto.experiment_dto import CreateExperimentDTO, ExperimentResponseDTO
    EXPERIMENT_AVAILABLE = True
except ImportError as e:
    print(f"Experiment DTOs not available: {e}")
    EXPERIMENT_AVAILABLE = False


class TestDetectorDTOs:
    """Test detector DTOs."""

    @pytest.mark.skipif(not DETECTOR_AVAILABLE, reason="Detector DTOs not available")
    def test_create_detector_dto(self):
        """Test CreateDetectorDTO creation."""
        dto = CreateDetectorDTO(
            name="test_detector",
            algorithm_name="IsolationForest",
            contamination_rate=0.1
        )
        
        assert dto.name == "test_detector"
        assert dto.algorithm_name == "IsolationForest"
        assert dto.contamination_rate == 0.1
        assert dto.parameters == {}
        assert dto.metadata == {}

    @pytest.mark.skipif(not DETECTOR_AVAILABLE, reason="Detector DTOs not available")
    def test_create_detector_dto_validation(self):
        """Test CreateDetectorDTO validation."""
        # Test contamination rate bounds
        with pytest.raises(ValueError):
            CreateDetectorDTO(
                name="test",
                algorithm_name="test",
                contamination_rate=-0.1
            )
        
        with pytest.raises(ValueError):
            CreateDetectorDTO(
                name="test",
                algorithm_name="test",
                contamination_rate=1.1
            )
        
        # Test name validation
        with pytest.raises(ValueError):
            CreateDetectorDTO(
                name="",
                algorithm_name="test"
            )

    @pytest.mark.skipif(not DETECTOR_AVAILABLE, reason="Detector DTOs not available")
    def test_detector_response_dto(self):
        """Test DetectorResponseDTO."""
        dto = DetectorResponseDTO(
            id=uuid4(),
            name="test_detector",
            algorithm_name="IsolationForest",
            contamination_rate=0.1,
            is_fitted=True,
            created_at=datetime.utcnow()
        )
        
        assert dto.name == "test_detector"
        assert dto.algorithm_name == "IsolationForest"
        assert dto.contamination_rate == 0.1
        assert dto.is_fitted is True
        assert dto.status == "active"
        assert dto.version == "1.0.0"

    @pytest.mark.skipif(not DETECTOR_AVAILABLE, reason="Detector DTOs not available")
    def test_detection_request_dto(self):
        """Test DetectionRequestDTO."""
        dto = DetectionRequestDTO(
            detector_id=uuid4(),
            data=[[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]
        )
        
        assert len(dto.data) == 2
        assert len(dto.data[0]) == 3
        assert dto.return_scores is True
        assert dto.return_feature_importance is False
        assert dto.threshold is None


class TestDatasetDTOs:
    """Test dataset DTOs."""

    @pytest.mark.skipif(not DATASET_AVAILABLE, reason="Dataset DTOs not available")
    def test_create_dataset_dto(self):
        """Test CreateDatasetDTO."""
        dto = CreateDatasetDTO(
            name="test_dataset",
            description="A test dataset"
        )
        
        assert dto.name == "test_dataset"
        assert dto.description == "A test dataset"
        assert dto.target_column is None
        assert dto.delimiter == ","
        assert dto.encoding == "utf-8"

    @pytest.mark.skipif(not DATASET_AVAILABLE, reason="Dataset DTOs not available")
    def test_create_dataset_dto_validation(self):
        """Test CreateDatasetDTO validation."""
        # Test name validation
        with pytest.raises(ValueError):
            CreateDatasetDTO(name="")
        
        with pytest.raises(ValueError):
            CreateDatasetDTO(name="a" * 101)

    @pytest.mark.skipif(not DATASET_AVAILABLE, reason="Dataset DTOs not available")
    def test_data_quality_report_dto(self):
        """Test DataQualityReportDTO."""
        dto = DataQualityReportDTO(
            quality_score=0.85,
            n_missing_values=10,
            n_duplicates=5,
            n_outliers=15
        )
        
        assert dto.quality_score == 0.85
        assert dto.n_missing_values == 10
        assert dto.n_duplicates == 5
        assert dto.n_outliers == 15
        assert dto.missing_columns == []
        assert dto.recommendations == []


class TestResultDTOs:
    """Test result DTOs."""

    @pytest.mark.skipif(not RESULT_AVAILABLE, reason="Result DTOs not available")
    def test_anomaly_dto(self):
        """Test AnomalyDTO."""
        dto = AnomalyDTO(
            id=uuid4(),
            score=0.95,
            data_point={"feature1": 1.0, "feature2": 2.0},
            detector_name="IsolationForest",
            timestamp=datetime.utcnow(),
            severity="high"
        )
        
        assert dto.score == 0.95
        assert dto.data_point == {"feature1": 1.0, "feature2": 2.0}
        assert dto.detector_name == "IsolationForest"
        assert dto.severity == "high"
        assert dto.explanation is None
        assert dto.metadata == {}

    @pytest.mark.skipif(not RESULT_AVAILABLE, reason="Result DTOs not available")
    def test_anomaly_dto_validation(self):
        """Test AnomalyDTO validation."""
        # Test score bounds
        with pytest.raises(ValueError):
            AnomalyDTO(
                id=uuid4(),
                score=-0.1,
                data_point={},
                detector_name="test",
                timestamp=datetime.utcnow(),
                severity="low"
            )
        
        with pytest.raises(ValueError):
            AnomalyDTO(
                id=uuid4(),
                score=1.1,
                data_point={},
                detector_name="test",
                timestamp=datetime.utcnow(),
                severity="low"
            )


class TestExperimentDTOs:
    """Test experiment DTOs."""

    @pytest.mark.skipif(not EXPERIMENT_AVAILABLE, reason="Experiment DTOs not available")
    def test_create_experiment_dto(self):
        """Test CreateExperimentDTO."""
        dto = CreateExperimentDTO(
            name="test_experiment",
            description="A test experiment"
        )
        
        assert dto.name == "test_experiment"
        assert dto.description == "A test experiment"
        assert dto.metadata == {}

    @pytest.mark.skipif(not EXPERIMENT_AVAILABLE, reason="Experiment DTOs not available")
    def test_create_experiment_dto_minimal(self):
        """Test CreateExperimentDTO with minimal data."""
        dto = CreateExperimentDTO(name="minimal_experiment")
        
        assert dto.name == "minimal_experiment"
        assert dto.description is None
        assert dto.metadata == {}

    @pytest.mark.skipif(not EXPERIMENT_AVAILABLE, reason="Experiment DTOs not available")
    def test_experiment_response_dto(self):
        """Test ExperimentResponseDTO."""
        dto = ExperimentResponseDTO(
            id="test_id",
            name="test_experiment",
            created_at=datetime.utcnow(),
            updated_at=datetime.utcnow()
        )
        
        assert dto.id == "test_id"
        assert dto.name == "test_experiment"
        assert dto.status == "active"
        assert dto.total_runs == 0


class TestDTOSerialization:
    """Test DTO serialization."""

    @pytest.mark.skipif(not DETECTOR_AVAILABLE, reason="Detector DTOs not available")
    def test_create_detector_serialization(self):
        """Test CreateDetectorDTO serialization."""
        dto = CreateDetectorDTO(
            name="test_detector",
            algorithm_name="IsolationForest",
            contamination_rate=0.1,
            parameters={"n_estimators": 100}
        )
        
        data = dto.model_dump()
        
        assert data["name"] == "test_detector"
        assert data["algorithm_name"] == "IsolationForest"
        assert data["contamination_rate"] == 0.1
        assert data["parameters"]["n_estimators"] == 100

    @pytest.mark.skipif(not DETECTOR_AVAILABLE, reason="Detector DTOs not available")
    def test_detector_response_serialization(self):
        """Test DetectorResponseDTO serialization."""
        dto = DetectorResponseDTO(
            id=uuid4(),
            name="test",
            algorithm_name="test",
            contamination_rate=0.1,
            is_fitted=True,
            created_at=datetime.utcnow()
        )
        
        data = dto.model_dump()
        
        assert "id" in data
        assert "created_at" in data
        assert "status" in data
        assert "version" in data


if __name__ == "__main__":
    # Simple test to check what's working
    print("Testing DTOs...")
    
    if DETECTOR_AVAILABLE:
        print("✓ Detector DTOs available")
        try:
            dto = CreateDetectorDTO(
                name="test",
                algorithm_name="IsolationForest"
            )
            print("✓ CreateDetectorDTO works")
        except Exception as e:
            print(f"✗ CreateDetectorDTO failed: {e}")
    else:
        print("✗ Detector DTOs not available")
    
    if DATASET_AVAILABLE:
        print("✓ Dataset DTOs available")
        try:
            dto = CreateDatasetDTO(name="test")
            print("✓ CreateDatasetDTO works")
        except Exception as e:
            print(f"✗ CreateDatasetDTO failed: {e}")
    else:
        print("✗ Dataset DTOs not available")
    
    if RESULT_AVAILABLE:
        print("✓ Result DTOs available")
        try:
            dto = AnomalyDTO(
                id=uuid4(),
                score=0.5,
                data_point={},
                detector_name="test",
                timestamp=datetime.utcnow(),
                severity="low"
            )
            print("✓ AnomalyDTO works")
        except Exception as e:
            print(f"✗ AnomalyDTO failed: {e}")
    else:
        print("✗ Result DTOs not available")
    
    if EXPERIMENT_AVAILABLE:
        print("✓ Experiment DTOs available")
        try:
            dto = CreateExperimentDTO(name="test")
            print("✓ CreateExperimentDTO works")
        except Exception as e:
            print(f"✗ CreateExperimentDTO failed: {e}")
    else:
        print("✗ Experiment DTOs not available")
    
    print("DTO testing complete.")