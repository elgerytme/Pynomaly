#!/usr/bin/env python3
"""
Basic test coverage for core components to increase overall test coverage
"""
import pytest
import pandas as pd
import numpy as np
from datetime import datetime

from pynomaly.domain.value_objects.anomaly_score import AnomalyScore
from pynomaly.domain.value_objects.contamination_rate import ContaminationRate
from pynomaly.domain.value_objects.confidence_interval import ConfidenceInterval
from pynomaly.domain.entities.dataset import Dataset
from pynomaly.domain.entities.detector import Detector
from pynomaly.domain.entities.anomaly import Anomaly
from pynomaly.domain.exceptions import ValidationError
from pynomaly.domain.exceptions.dataset_exceptions import DataValidationError


def test_anomaly_score_basic():
    """Test basic anomaly score functionality."""
    score = AnomalyScore(value=0.8, threshold=0.5, method="sklearn")
    assert score.value == 0.8
    assert score.threshold == 0.5
    assert score.method == "sklearn"
    assert score.is_anomaly() is True


def test_anomaly_score_metadata():
    """Test anomaly score with metadata."""
    metadata = {"algorithm": "isolation_forest"}
    score = AnomalyScore(value=0.3, metadata=metadata, method="pyod")
    assert score.metadata["algorithm"] == "isolation_forest"
    assert score.is_anomaly() is False


def test_contamination_rate_basic():
    """Test basic contamination rate."""
    rate = ContaminationRate(value=0.1)
    assert rate.value == 0.1
    assert rate.confidence_level == 0.8  # Default
    assert rate.source == "auto"  # Default


def test_contamination_rate_custom():
    """Test contamination rate with custom parameters."""
    rate = ContaminationRate(
        value=0.05, 
        source="domain_knowledge", 
        confidence_level=0.9
    )
    assert rate.value == 0.05
    assert rate.source == "domain_knowledge"
    assert rate.confidence_level == 0.9


def test_confidence_interval_basic():
    """Test basic confidence interval."""
    ci = ConfidenceInterval(lower=0.1, upper=0.9, confidence_level=0.95)
    assert ci.lower == 0.1
    assert ci.upper == 0.9
    assert ci.confidence_level == 0.95
    assert ci.width() == 0.8
    assert ci.midpoint() == 0.5
    assert ci.contains(0.5) is True
    assert ci.contains(0.05) is False


def test_dataset_creation():
    """Test dataset creation."""
    df = pd.DataFrame({
        'x': [1, 2, 3, 4, 5],
        'y': [2, 4, 6, 8, 10]
    })
    
    dataset = Dataset(name="Test Dataset", data=df)
    assert dataset.name == "Test Dataset"
    assert dataset.n_samples == 5
    assert dataset.n_features == 2
    assert dataset.id is not None


def test_dataset_with_metadata():
    """Test dataset with metadata."""
    df = pd.DataFrame({'a': [1, 2, 3]})
    metadata = {"source": "test", "version": "1.0"}
    
    dataset = Dataset(name="Test", data=df, metadata=metadata)
    assert dataset.metadata["source"] == "test"
    assert dataset.metadata["version"] == "1.0"


def test_detector_basic():
    """Test basic detector creation."""
    detector = Detector(
        name="Test Detector",
        algorithm_name="isolation_forest",
        contamination_rate=ContaminationRate(value=0.1)
    )
    
    assert detector.name == "Test Detector"
    assert detector.algorithm_name == "isolation_forest"
    assert detector.contamination_rate.value == 0.1
    assert detector.id is not None


def test_detector_with_parameters():
    """Test detector with parameters."""
    params = {"n_estimators": 100, "random_state": 42}
    detector = Detector(
        name="Configured Detector",
        algorithm_name="isolation_forest",
        contamination_rate=ContaminationRate(value=0.05),
        parameters=params
    )
    
    assert detector.parameters["n_estimators"] == 100
    assert detector.parameters["random_state"] == 42


def test_anomaly_creation():
    """Test anomaly creation."""
    score = AnomalyScore(value=0.8, threshold=0.5, method="sklearn")
    data_point = {"x": 1.5, "y": -2.0}
    
    anomaly = Anomaly(
        score=score,
        data_point=data_point,
        detector_name="Test Detector"
    )
    
    assert anomaly.score.value == 0.8
    assert anomaly.data_point["x"] == 1.5
    assert anomaly.detector_name == "Test Detector"
    assert anomaly.id is not None
    assert isinstance(anomaly.timestamp, datetime)


def test_anomaly_severity():
    """Test anomaly severity classification."""
    # Critical severity
    high_score = AnomalyScore(value=0.95, threshold=0.5, method="sklearn")
    high_anomaly = Anomaly(
        score=high_score,
        data_point={"x": 1},
        detector_name="Test"
    )
    assert high_anomaly.severity == "critical"
    
    # High severity
    med_score = AnomalyScore(value=0.8, threshold=0.5, method="sklearn")
    med_anomaly = Anomaly(
        score=med_score,
        data_point={"x": 1},
        detector_name="Test"
    )
    assert med_anomaly.severity == "high"


def test_value_object_immutability():
    """Test that value objects are immutable."""
    score = AnomalyScore(value=0.8, threshold=0.5, method="sklearn")
    
    # Should not be able to modify fields
    with pytest.raises(AttributeError):
        score.value = 0.9


def test_value_object_equality():
    """Test value object equality."""
    score1 = AnomalyScore(value=0.8, threshold=0.5, method="sklearn")
    score2 = AnomalyScore(value=0.8, threshold=0.5, method="sklearn")
    score3 = AnomalyScore(value=0.7, threshold=0.5, method="sklearn")
    
    assert score1 == score2
    assert score1 != score3


def test_invalid_values():
    """Test that invalid values raise appropriate errors."""
    # Invalid score value
    with pytest.raises(ValidationError):
        AnomalyScore(value=-0.1)  # Negative value
        
    # Invalid threshold
    with pytest.raises(ValidationError):
        AnomalyScore(value=0.5, threshold=1.1)  # Threshold > 1
        
    # Invalid contamination rate
    with pytest.raises(ValidationError):
        ContaminationRate(value=-0.1)  # Negative value
        
    # Invalid confidence interval
    with pytest.raises(ValidationError):
        ConfidenceInterval(lower=0.8, upper=0.2)  # Lower > upper


def test_dataset_validation():
    """Test dataset validation."""
    # Empty name
    with pytest.raises(ValueError):
        Dataset(name="", data=pd.DataFrame({'x': [1, 2, 3]}))
        
    # Empty dataset
    with pytest.raises(DataValidationError):
        Dataset(name="Test", data=pd.DataFrame())


def test_detector_validation():
    """Test detector validation."""
    # Missing name
    with pytest.raises(ValueError):
        Detector(
            name="",
            algorithm_name="isolation_forest",
            contamination_rate=ContaminationRate(value=0.1)
        )


def test_string_representations():
    """Test string representations of objects."""
    df = pd.DataFrame({'x': [1, 2, 3]})
    dataset = Dataset(name="Test Dataset", data=df)
    
    str_repr = str(dataset)
    assert "Test Dataset" in str_repr
    
    score = AnomalyScore(value=0.8, threshold=0.5, method="sklearn")
    score_str = str(score)
    assert "0.8" in score_str


def test_entity_identity():
    """Test that entities have proper identity."""
    df = pd.DataFrame({'x': [1, 2, 3]})
    
    dataset1 = Dataset(name="Test", data=df)
    dataset2 = Dataset(name="Test", data=df)
    
    # Different entities even with same data
    assert dataset1.id != dataset2.id
    assert dataset1 != dataset2
    
    # Same entity should equal itself
    assert dataset1 == dataset1