"""Tests for domain entities."""

from __future__ import annotations

from datetime import datetime
from uuid import UUID

import pandas as pd
import pytest

from pynomaly.domain.entities import Anomaly, Dataset, Detector, DetectionResult
from pynomaly.domain.exceptions import InvalidDataError, ValidationError


class TestDetector:
    """Test Detector entity."""
    
    def test_create_detector(self):
        """Test creating a detector."""
        detector = Detector(
            name="Test Detector",
            algorithm="IsolationForest",
            parameters={"contamination": 0.1}
        )
        
        assert detector.name == "Test Detector"
        assert detector.algorithm == "IsolationForest"
        assert detector.parameters["contamination"] == 0.1
        assert not detector.is_fitted
        assert isinstance(detector.id, UUID)
        assert isinstance(detector.created_at, datetime)
    
    def test_detector_validation(self):
        """Test detector validation."""
        with pytest.raises(ValueError):
            Detector(name="", algorithm="IsolationForest")
        
        with pytest.raises(ValueError):
            Detector(name="Test", algorithm="")
    
    def test_update_parameters(self):
        """Test updating detector parameters."""
        detector = Detector(name="Test", algorithm="LOF")
        detector.update_parameters({"n_neighbors": 20})
        
        assert detector.parameters["n_neighbors"] == 20
        assert isinstance(detector.updated_at, datetime)


class TestDataset:
    """Test Dataset entity."""
    
    def test_create_dataset(self):
        """Test creating a dataset."""
        df = pd.DataFrame({
            "feature1": [1, 2, 3, 4, 5],
            "feature2": [2, 4, 6, 8, 10],
            "target": [0, 0, 1, 0, 1]
        })
        
        dataset = Dataset(
            name="Test Dataset",
            data=df[["feature1", "feature2"]],
            target_column="target"
        )
        
        assert dataset.name == "Test Dataset"
        assert dataset.n_samples == 5
        assert dataset.n_features == 2
        assert dataset.has_target
        assert dataset.target_column == "target"
    
    def test_dataset_validation(self):
        """Test dataset validation."""
        with pytest.raises(ValueError):
            Dataset(name="", data=pd.DataFrame())
        
        with pytest.raises(InvalidDataError):
            Dataset(name="Test", data=pd.DataFrame())  # Empty DataFrame
    
    def test_dataset_split(self):
        """Test splitting dataset."""
        df = pd.DataFrame({
            "feature1": range(100),
            "feature2": range(100, 200)
        })
        
        dataset = Dataset(name="Test", data=df)
        train, test = dataset.split(test_size=0.2, random_state=42)
        
        assert train.n_samples == 80
        assert test.n_samples == 20
        assert train.name == "Test_train"
        assert test.name == "Test_test"
    
    def test_get_feature_types(self):
        """Test getting feature types."""
        df = pd.DataFrame({
            "numeric1": [1.0, 2.0, 3.0],
            "numeric2": [1, 2, 3],
            "categorical": ["A", "B", "C"],
            "mixed": [1, "2", 3.0]
        })
        
        dataset = Dataset(name="Test", data=df)
        
        numeric_features = dataset.get_numeric_features()
        categorical_features = dataset.get_categorical_features()
        
        assert "numeric1" in numeric_features
        assert "numeric2" in numeric_features
        assert "categorical" in categorical_features
        assert "mixed" in categorical_features


class TestAnomaly:
    """Test Anomaly entity."""
    
    def test_create_anomaly(self):
        """Test creating an anomaly."""
        anomaly = Anomaly(
            index=42,
            score=0.95,
            threshold=0.8,
            features={"feature1": 10.5, "feature2": -3.2}
        )
        
        assert anomaly.index == 42
        assert anomaly.score == 0.95
        assert anomaly.threshold == 0.8
        assert anomaly.is_anomaly
        assert anomaly.confidence == pytest.approx(0.1875)  # (0.95 - 0.8) / 0.8
    
    def test_anomaly_validation(self):
        """Test anomaly validation."""
        with pytest.raises(ValueError):
            Anomaly(index=-1, score=0.5, threshold=0.8)
        
        with pytest.raises(ValueError):
            Anomaly(index=0, score=-0.1, threshold=0.8)
        
        with pytest.raises(ValueError):
            Anomaly(index=0, score=0.5, threshold=-0.1)


class TestDetectionResult:
    """Test DetectionResult entity."""
    
    def test_create_detection_result(self):
        """Test creating a detection result."""
        detector_id = UUID("12345678-1234-5678-1234-567812345678")
        dataset_id = UUID("87654321-4321-8765-4321-876543218765")
        
        result = DetectionResult(
            detector_id=detector_id,
            dataset_id=dataset_id,
            n_samples=1000,
            n_anomalies=50,
            anomaly_rate=0.05,
            threshold=0.85,
            execution_time_ms=250,
            anomaly_indices=[1, 5, 10, 20],
            anomaly_scores=[0.9, 0.92, 0.88, 0.95]
        )
        
        assert result.detector_id == detector_id
        assert result.dataset_id == dataset_id
        assert result.n_samples == 1000
        assert result.n_anomalies == 50
        assert result.anomaly_rate == 0.05
        assert len(result.anomaly_indices) == 4
    
    def test_detection_result_validation(self):
        """Test detection result validation."""
        detector_id = UUID("12345678-1234-5678-1234-567812345678")
        dataset_id = UUID("87654321-4321-8765-4321-876543218765")
        
        with pytest.raises(ValueError):
            DetectionResult(
                detector_id=detector_id,
                dataset_id=dataset_id,
                n_samples=0,  # Invalid
                n_anomalies=10,
                anomaly_rate=0.1,
                threshold=0.8,
                execution_time_ms=100
            )
        
        with pytest.raises(ValueError):
            DetectionResult(
                detector_id=detector_id,
                dataset_id=dataset_id,
                n_samples=100,
                n_anomalies=101,  # More anomalies than samples
                anomaly_rate=1.01,
                threshold=0.8,
                execution_time_ms=100
            )