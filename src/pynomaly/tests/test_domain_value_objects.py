#!/usr/bin/env python3
"""
Unit tests for domain value objects
"""
import pytest
import numpy as np
from datetime import datetime
from typing import Any, Dict

from pynomaly.domain.value_objects.anomaly_score import AnomalyScore
from pynomaly.domain.value_objects.contamination_rate import ContaminationRate
from pynomaly.domain.value_objects.confidence_interval import ConfidenceInterval
from pynomaly.domain.value_objects.threshold_config import ThresholdConfig
from pynomaly.domain.exceptions import ValidationError


class TestAnomalyScore:
    """Tests for AnomalyScore value object."""
    
    def test_valid_anomaly_score_creation(self):
        """Test creating a valid anomaly score."""
        score = AnomalyScore(value=0.8, threshold=0.5, method="sklearn")
        assert score.value == 0.8
        assert score.threshold == 0.5
        assert score.method == "sklearn"
        assert score.is_anomaly() is True
        
    def test_anomaly_score_with_metadata(self):
        """Test anomaly score with metadata."""
        metadata = {"algorithm": "isolation_forest", "feature_count": 5}
        score = AnomalyScore(value=0.3, metadata=metadata, method="pyod")
        assert score.metadata["algorithm"] == "isolation_forest"
        assert score.metadata["feature_count"] == 5
        assert score.is_anomaly() is False
        
    def test_anomaly_score_confidence_level(self):
        """Test confidence level calculation."""
        score = AnomalyScore(value=0.9, threshold=0.5)
        confidence = score.confidence_level()
        assert confidence > 0.5  # High confidence for score well above threshold
        
    def test_invalid_score_value(self):
        """Test invalid score values."""
        with pytest.raises(ValidationError):
            AnomalyScore(value=-0.1)  # Negative value
            
        with pytest.raises(ValidationError):
            AnomalyScore(value=1.1)  # Value > 1
            
    def test_invalid_threshold(self):
        """Test invalid threshold values."""
        with pytest.raises(ValidationError):
            AnomalyScore(value=0.5, threshold=-0.1)  # Negative threshold
            
        with pytest.raises(ValidationError):
            AnomalyScore(value=0.5, threshold=1.1)  # Threshold > 1


class TestContaminationRate:
    """Tests for ContaminationRate value object."""
    
    def test_valid_contamination_rate(self):
        """Test creating valid contamination rate."""
        rate = ContaminationRate(value=0.1)
        assert rate.value == 0.1
        assert rate.is_low_contamination() is True
        
    def test_auto_contamination_rate(self):
        """Test auto contamination rate."""
        rate = ContaminationRate(value="auto")
        assert rate.value == "auto"
        
    def test_high_contamination_rate(self):
        """Test high contamination rate detection."""
        rate = ContaminationRate(value=0.3)
        assert rate.is_high_contamination() is True
        assert rate.is_low_contamination() is False
        
    def test_invalid_contamination_rate(self):
        """Test invalid contamination rates."""
        with pytest.raises(ValidationError):
            ContaminationRate(value=-0.1)  # Negative value
            
        with pytest.raises(ValidationError):
            ContaminationRate(value=0.6)  # Too high (>0.5)


class TestConfidenceInterval:
    """Tests for ConfidenceInterval value object."""
    
    def test_valid_confidence_interval(self):
        """Test creating valid confidence interval."""
        ci = ConfidenceInterval(lower=0.1, upper=0.9, confidence_level=0.95)
        assert ci.lower == 0.1
        assert ci.upper == 0.9
        assert ci.confidence_level == 0.95
        assert ci.width() == 0.8
        assert ci.midpoint() == 0.5
        
    def test_confidence_interval_contains(self):
        """Test if value is contained in interval."""
        ci = ConfidenceInterval(lower=0.2, upper=0.8, confidence_level=0.95)
        assert ci.contains(0.5) is True
        assert ci.contains(0.1) is False
        assert ci.contains(0.9) is False
        
    def test_invalid_confidence_interval(self):
        """Test invalid confidence intervals."""
        with pytest.raises(ValidationError):
            ConfidenceInterval(lower=0.8, upper=0.2)  # Lower > upper
            
        with pytest.raises(ValidationError):
            ConfidenceInterval(lower=0.1, upper=0.9, confidence_level=1.5)  # Invalid confidence level


class TestThresholdConfig:
    """Tests for ThresholdConfig value object."""
    
    def test_static_threshold(self):
        """Test static threshold configuration."""
        config = ThresholdConfig(
            strategy="static",
            value=0.5,
            metadata={"description": "Fixed threshold"}
        )
        assert config.strategy == "static"
        assert config.value == 0.5
        assert config.is_static() is True
        assert config.is_adaptive() is False
        
    def test_adaptive_threshold(self):
        """Test adaptive threshold configuration."""
        config = ThresholdConfig(
            strategy="adaptive",
            parameters={"contamination": 0.1, "method": "quantile"}
        )
        assert config.strategy == "adaptive"
        assert config.is_adaptive() is True
        assert config.parameters["contamination"] == 0.1
        
    def test_percentile_threshold(self):
        """Test percentile-based threshold."""
        config = ThresholdConfig(
            strategy="percentile",
            value=95.0,  # 95th percentile
            parameters={"window_size": 100}
        )
        assert config.strategy == "percentile"
        assert config.value == 95.0
        
    def test_invalid_threshold_config(self):
        """Test invalid threshold configurations."""
        with pytest.raises(ValidationError):
            ThresholdConfig(strategy="invalid_strategy")
            
        with pytest.raises(ValidationError):
            ThresholdConfig(strategy="static")  # Missing value for static


def test_value_objects_immutability():
    """Test that value objects are immutable."""
    score = AnomalyScore(value=0.8, threshold=0.5, method="sklearn")
    
    # Should not be able to modify fields
    with pytest.raises(AttributeError):
        score.value = 0.9
        
    with pytest.raises(AttributeError):
        score.threshold = 0.7


def test_value_objects_equality():
    """Test value object equality."""
    score1 = AnomalyScore(value=0.8, threshold=0.5, method="sklearn")
    score2 = AnomalyScore(value=0.8, threshold=0.5, method="sklearn")
    score3 = AnomalyScore(value=0.7, threshold=0.5, method="sklearn")
    
    assert score1 == score2
    assert score1 != score3


def test_value_objects_hashing():
    """Test value object hashing for use in sets/dicts."""
    score1 = AnomalyScore(value=0.8, threshold=0.5, method="sklearn")
    score2 = AnomalyScore(value=0.8, threshold=0.5, method="sklearn")
    
    # Should be hashable and equal objects should have same hash
    score_set = {score1, score2}
    assert len(score_set) == 1  # Only one unique score