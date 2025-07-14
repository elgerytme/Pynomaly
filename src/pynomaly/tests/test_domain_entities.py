#!/usr/bin/env python3
"""
Unit tests for domain entities
"""
import pytest
import pandas as pd
import numpy as np
from datetime import datetime
from uuid import uuid4

from pynomaly.domain.entities.dataset import Dataset
from pynomaly.domain.entities.detector import Detector
from pynomaly.domain.entities.anomaly import Anomaly
from pynomaly.domain.entities.detection_result import DetectionResult
from pynomaly.domain.value_objects.anomaly_score import AnomalyScore
from pynomaly.domain.value_objects.contamination_rate import ContaminationRate
from pynomaly.domain.exceptions import ValidationError


class TestDataset:
    """Tests for Dataset entity."""
    
    def test_create_dataset_from_dataframe(self):
        """Test creating dataset from pandas DataFrame."""
        df = pd.DataFrame({
            'feature1': [1, 2, 3, 4, 5],
            'feature2': [2, 4, 6, 8, 10],
            'feature3': [1.1, 2.2, 3.3, 4.4, 5.5]
        })
        
        dataset = Dataset(name="Test Dataset", data=df)
        assert dataset.name == "Test Dataset"
        assert dataset.data.shape == (5, 3)
        assert dataset.n_samples == 5
        assert dataset.n_features == 3
        assert dataset.id is not None
        
    def test_dataset_with_metadata(self):
        """Test dataset with metadata."""
        df = pd.DataFrame({'x': [1, 2, 3], 'y': [4, 5, 6]})
        metadata = {
            "source": "test_data",
            "created_by": "unit_test",
            "version": "1.0"
        }
        
        dataset = Dataset(name="Test", data=df, metadata=metadata)
        assert dataset.metadata["source"] == "test_data"
        assert dataset.metadata["version"] == "1.0"
        
    def test_dataset_feature_types(self):
        """Test dataset feature type detection."""
        df = pd.DataFrame({
            'numeric': [1.0, 2.0, 3.0],
            'integer': [1, 2, 3],
            'categorical': ['A', 'B', 'C']
        })
        
        dataset = Dataset(name="Mixed Types", data=df)
        feature_types = dataset.get_feature_types()
        assert len(feature_types) == 3
        
    def test_dataset_basic_statistics(self):
        """Test dataset basic statistics."""
        df = pd.DataFrame({
            'x': [1, 2, 3, 4, 5],
            'y': [10, 20, 30, 40, 50]
        })
        
        dataset = Dataset(name="Stats Test", data=df)
        stats = dataset.get_basic_statistics()
        assert 'mean' in stats
        assert 'std' in stats
        assert len(stats['mean']) == 2  # Two features
        
    def test_invalid_dataset(self):
        """Test invalid dataset creation."""
        # Empty name
        with pytest.raises(ValidationError):
            Dataset(name="", data=pd.DataFrame({'x': [1, 2, 3]}))
            
        # Empty dataset
        with pytest.raises(ValidationError):
            Dataset(name="Test", data=pd.DataFrame())


class TestDetector:
    """Tests for Detector entity."""
    
    def test_create_detector(self):
        """Test creating a detector."""
        detector = Detector(
            name="Test Detector",
            algorithm="isolation_forest",
            contamination=ContaminationRate(value=0.1)
        )
        
        assert detector.name == "Test Detector"
        assert detector.algorithm == "isolation_forest"
        assert detector.contamination.value == 0.1
        assert detector.id is not None
        
    def test_detector_with_parameters(self):
        """Test detector with algorithm parameters."""
        params = {
            "n_estimators": 100,
            "max_samples": "auto",
            "random_state": 42
        }
        
        detector = Detector(
            name="Configured Detector",
            algorithm="isolation_forest",
            contamination=ContaminationRate(value=0.05),
            parameters=params
        )
        
        assert detector.parameters["n_estimators"] == 100
        assert detector.parameters["random_state"] == 42
        
    def test_detector_validation(self):
        """Test detector validation."""
        # Invalid algorithm
        with pytest.raises(ValidationError):
            Detector(
                name="Invalid",
                algorithm="unknown_algorithm",
                contamination=ContaminationRate(value=0.1)
            )
            
        # Missing name
        with pytest.raises(ValidationError):
            Detector(
                name="",
                algorithm="isolation_forest",
                contamination=ContaminationRate(value=0.1)
            )


class TestAnomaly:
    """Tests for Anomaly entity."""
    
    def test_create_anomaly(self):
        """Test creating an anomaly."""
        score = AnomalyScore(value=0.8, threshold=0.5, method="sklearn")
        data_point = {"feature1": 1.5, "feature2": -2.0}
        
        anomaly = Anomaly(
            score=score,
            data_point=data_point,
            detector_name="Test Detector"
        )
        
        assert anomaly.score.value == 0.8
        assert anomaly.data_point["feature1"] == 1.5
        assert anomaly.detector_name == "Test Detector"
        assert anomaly.id is not None
        assert isinstance(anomaly.timestamp, datetime)
        
    def test_anomaly_with_explanation(self):
        """Test anomaly with explanation."""
        score = AnomalyScore(value=0.9, threshold=0.5, method="pyod")
        data_point = {"x": 100, "y": 200}
        explanation = {
            "feature_contributions": {"x": 0.6, "y": 0.4},
            "method": "shap"
        }
        
        anomaly = Anomaly(
            score=score,
            data_point=data_point,
            detector_name="Explainable Detector",
            explanation=explanation
        )
        
        assert anomaly.explanation["method"] == "shap"
        assert anomaly.explanation["feature_contributions"]["x"] == 0.6
        
    def test_anomaly_severity_classification(self):
        """Test anomaly severity classification."""
        # High severity
        high_score = AnomalyScore(value=0.95, threshold=0.5, method="sklearn")
        high_anomaly = Anomaly(
            score=high_score,
            data_point={"x": 1},
            detector_name="Test"
        )
        assert high_anomaly.get_severity() == "high"
        
        # Medium severity
        med_score = AnomalyScore(value=0.7, threshold=0.5, method="sklearn")
        med_anomaly = Anomaly(
            score=med_score,
            data_point={"x": 1},
            detector_name="Test"
        )
        assert med_anomaly.get_severity() == "medium"
        
        # Low severity
        low_score = AnomalyScore(value=0.55, threshold=0.5, method="sklearn")
        low_anomaly = Anomaly(
            score=low_score,
            data_point={"x": 1},
            detector_name="Test"
        )
        assert low_anomaly.get_severity() == "low"


class TestDetectionResult:
    """Tests for DetectionResult entity."""
    
    def test_create_detection_result(self):
        """Test creating a detection result."""
        # Create test scores and anomalies
        scores = [
            AnomalyScore(value=0.2, threshold=0.5, method="sklearn"),
            AnomalyScore(value=0.8, threshold=0.5, method="sklearn"),
            AnomalyScore(value=0.3, threshold=0.5, method="sklearn"),
            AnomalyScore(value=0.9, threshold=0.5, method="sklearn"),
            AnomalyScore(value=0.1, threshold=0.5, method="sklearn")
        ]
        
        anomalies = [
            Anomaly(score=scores[1], data_point={"x": 1}, detector_name="Test"),
            Anomaly(score=scores[3], data_point={"x": 2}, detector_name="Test")
        ]
        
        result = DetectionResult(
            scores=scores,
            anomalies=anomalies
        )
        
        assert result.n_samples == 5
        assert result.n_anomalies == 2
        assert result.anomaly_rate == 0.4  # 2/5 = 40%
        assert result.id is not None
        
    def test_detection_result_statistics(self):
        """Test detection result statistics."""
        scores = [
            AnomalyScore(value=0.1, threshold=0.5, method="sklearn"),
            AnomalyScore(value=0.3, threshold=0.5, method="sklearn"),
            AnomalyScore(value=0.7, threshold=0.5, method="sklearn"),
            AnomalyScore(value=0.9, threshold=0.5, method="sklearn")
        ]
        
        anomalies = [
            Anomaly(score=scores[2], data_point={"x": 1}, detector_name="Test"),
            Anomaly(score=scores[3], data_point={"x": 2}, detector_name="Test")
        ]
        
        result = DetectionResult(scores=scores, anomalies=anomalies)
        
        stats = result.get_statistics()
        assert 'mean_score' in stats
        assert 'max_score' in stats
        assert 'anomaly_rate' in stats
        assert stats['anomaly_rate'] == 0.5
        
    def test_detection_result_with_metadata(self):
        """Test detection result with metadata."""
        scores = [AnomalyScore(value=0.5, threshold=0.5, method="sklearn") for _ in range(5)]
        anomalies = []
        
        metadata = {
            "algorithm": "isolation_forest",
            "execution_time": 0.123,
            "contamination": 0.1
        }
        
        result = DetectionResult(
            scores=scores,
            anomalies=anomalies,
            metadata=metadata
        )
        
        assert result.metadata["algorithm"] == "isolation_forest"
        assert result.metadata["execution_time"] == 0.123
        
    def test_invalid_detection_result(self):
        """Test invalid detection result."""
        # Empty scores
        with pytest.raises(ValidationError):
            DetectionResult(scores=[], anomalies=[])
            
        # Inconsistent anomalies
        scores = [AnomalyScore(value=0.5, threshold=0.5, method="sklearn")]
        anomalies = [
            Anomaly(
                score=AnomalyScore(value=0.8, threshold=0.5, method="sklearn"),
                data_point={"x": 1},
                detector_name="Test"
            )
        ]  # Anomaly score not in scores list
        
        # This should be valid - anomalies can have different scores
        result = DetectionResult(scores=scores, anomalies=anomalies)
        assert result.n_samples == 1
        assert result.n_anomalies == 1


def test_entity_immutability():
    """Test that entities have proper identity and comparison."""
    df = pd.DataFrame({'x': [1, 2, 3], 'y': [4, 5, 6]})
    
    dataset1 = Dataset(name="Test", data=df)
    dataset2 = Dataset(name="Test", data=df)
    
    # Different entities even with same data
    assert dataset1.id != dataset2.id
    assert dataset1 != dataset2
    
    # Same entity should equal itself
    assert dataset1 == dataset1


def test_entity_string_representation():
    """Test entity string representations."""
    df = pd.DataFrame({'x': [1, 2, 3]})
    dataset = Dataset(name="Test Dataset", data=df)
    
    str_repr = str(dataset)
    assert "Test Dataset" in str_repr
    assert "3 samples" in str_repr
    
    repr_str = repr(dataset)
    assert "Dataset" in repr_str