"""Tests for domain value objects."""

from __future__ import annotations

import pytest

from pynomaly.domain.exceptions import InvalidValueError
from pynomaly.domain.value_objects import (
    AnomalyScore,
    ConfidenceInterval,
    ContaminationRate,
    ThresholdConfig,
)


class TestAnomalyScore:
    """Test AnomalyScore value object."""

    def test_create_anomaly_score(self):
        """Test creating anomaly score."""
        score = AnomalyScore(value=0.85)

        assert score.value == 0.85
        assert score.is_valid()
        assert str(score) == "0.85"

    def test_anomaly_score_with_confidence(self):
        """Test anomaly score with confidence interval."""
        confidence = ConfidenceInterval(lower=0.8, upper=0.9, confidence_level=0.95)
        score = AnomalyScore(value=0.85, confidence_interval=confidence)

        assert score.confidence_interval is not None
        assert score.confidence_interval.contains(0.85)

    def test_anomaly_score_comparison(self):
        """Test comparing anomaly scores."""
        score1 = AnomalyScore(value=0.7)
        score2 = AnomalyScore(value=0.9)

        assert score1 < score2
        assert score2 > score1
        assert score1 != score2

    def test_invalid_anomaly_score(self):
        """Test invalid anomaly scores."""
        with pytest.raises(InvalidValueError):
            AnomalyScore(value=-0.1)

        with pytest.raises(InvalidValueError):
            AnomalyScore(value=1.1)


class TestContaminationRate:
    """Test ContaminationRate value object."""

    def test_create_contamination_rate(self):
        """Test creating contamination rate."""
        rate = ContaminationRate(value=0.05)

        assert rate.value == 0.05
        assert rate.as_percentage() == 5.0
        assert str(rate) == "5.0%"

    def test_contamination_rate_validation(self):
        """Test contamination rate validation."""
        # Valid edge cases
        rate1 = ContaminationRate(value=0.0)
        rate2 = ContaminationRate(value=0.5)

        assert rate1.value == 0.0
        assert rate2.value == 0.5

        # Invalid cases
        with pytest.raises(InvalidValueError):
            ContaminationRate(value=-0.01)

        with pytest.raises(InvalidValueError):
            ContaminationRate(value=0.51)

        with pytest.raises(InvalidValueError):
            ContaminationRate(value=1.0)


class TestConfidenceInterval:
    """Test ConfidenceInterval value object."""

    def test_create_confidence_interval(self):
        """Test creating confidence interval."""
        ci = ConfidenceInterval(lower=0.7, upper=0.9, confidence_level=0.95)

        assert ci.lower == 0.7
        assert ci.upper == 0.9
        assert ci.confidence_level == 0.95
        assert ci.width() == 0.2
        assert ci.midpoint() == 0.8

    def test_confidence_interval_contains(self):
        """Test checking if value is in interval."""
        ci = ConfidenceInterval(lower=0.7, upper=0.9)

        assert ci.contains(0.8)
        assert ci.contains(0.7)
        assert ci.contains(0.9)
        assert not ci.contains(0.6)
        assert not ci.contains(1.0)

    def test_confidence_interval_validation(self):
        """Test confidence interval validation."""
        # Lower > upper
        with pytest.raises(InvalidValueError):
            ConfidenceInterval(lower=0.9, upper=0.7)

        # Invalid confidence level
        with pytest.raises(InvalidValueError):
            ConfidenceInterval(lower=0.7, upper=0.9, confidence_level=1.5)

        with pytest.raises(InvalidValueError):
            ConfidenceInterval(lower=0.7, upper=0.9, confidence_level=-0.1)


class TestThresholdConfig:
    """Test ThresholdConfig value object."""

    def test_create_threshold_config(self):
        """Test creating threshold config."""
        config = ThresholdConfig(method="percentile", value=95.0, auto_adjust=True)

        assert config.method == "percentile"
        assert config.value == 95.0
        assert config.auto_adjust is True

    def test_threshold_config_defaults(self):
        """Test threshold config defaults."""
        config = ThresholdConfig()

        assert config.method == "contamination"
        assert config.value is None
        assert config.auto_adjust is False

    def test_threshold_config_validation(self):
        """Test threshold config validation."""
        # Invalid method
        with pytest.raises(InvalidValueError):
            ThresholdConfig(method="invalid_method")

        # Percentile out of range
        with pytest.raises(InvalidValueError):
            ThresholdConfig(method="percentile", value=101.0)

        with pytest.raises(InvalidValueError):
            ThresholdConfig(method="percentile", value=-1.0)
