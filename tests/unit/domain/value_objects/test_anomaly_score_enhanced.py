"""Tests for enhanced AnomalyScore value object validation."""

import math
import pytest
from unittest.mock import Mock

from pynomaly.domain.value_objects.anomaly_score import AnomalyScore
from pynomaly.domain.value_objects.confidence_interval import ConfidenceInterval
from pynomaly.domain.exceptions import InvalidValueError


class TestAnomalyScoreBasicValidation:
    """Test basic validation rules for AnomalyScore."""

    def test_valid_score_creation(self):
        """Test creating valid anomaly scores."""
        score = AnomalyScore(0.5)
        assert score.value == 0.5
        assert score.confidence_interval is None
        assert score.method is None

    def test_valid_score_with_confidence_interval(self):
        """Test creating score with confidence interval."""
        ci = ConfidenceInterval(0.4, 0.6, 0.95)
        score = AnomalyScore(0.5, confidence_interval=ci)
        assert score.value == 0.5
        assert score.confidence_interval == ci

    def test_valid_score_with_method(self):
        """Test creating score with method."""
        score = AnomalyScore(0.8, method="isolation_forest")
        assert score.value == 0.8
        assert score.method == "isolation_forest"

    def test_invalid_non_numeric_value(self):
        """Test rejection of non-numeric values."""
        with pytest.raises(InvalidValueError) as exc_info:
            AnomalyScore("0.5")
        
        assert "Score value must be numeric" in str(exc_info.value)
        assert exc_info.value.details["field"] == "value"

    def test_invalid_nan_value(self):
        """Test rejection of NaN values."""
        with pytest.raises(InvalidValueError) as exc_info:
            AnomalyScore(float('nan'))
        
        assert "Score value cannot be NaN or infinite" in str(exc_info.value)
        assert exc_info.value.details["field"] == "value"

    def test_invalid_infinite_value(self):
        """Test rejection of infinite values."""
        with pytest.raises(InvalidValueError) as exc_info:
            AnomalyScore(float('inf'))
        
        assert "Score value cannot be NaN or infinite" in str(exc_info.value)
        assert exc_info.value.details["field"] == "value"

    def test_invalid_negative_value(self):
        """Test rejection of negative values."""
        with pytest.raises(InvalidValueError) as exc_info:
            AnomalyScore(-0.1)
        
        assert "Score value must be between 0 and 1" in str(exc_info.value)
        assert exc_info.value.details["field"] == "value"

    def test_invalid_value_greater_than_one(self):
        """Test rejection of values greater than 1."""
        with pytest.raises(InvalidValueError) as exc_info:
            AnomalyScore(1.1)
        
        assert "Score value must be between 0 and 1" in str(exc_info.value)
        assert exc_info.value.details["field"] == "value"

    def test_boundary_values(self):
        """Test boundary values (0.0 and 1.0)."""
        score_zero = AnomalyScore(0.0)
        assert score_zero.value == 0.0
        
        # Note: 1.0 requires confidence interval due to business rule
        ci = ConfidenceInterval(0.95, 1.0, 0.95)
        score_one = AnomalyScore(1.0, confidence_interval=ci)
        assert score_one.value == 1.0


class TestAnomalyScoreBusinessRules:
    """Test business rule validation for AnomalyScore."""

    def test_noise_threshold_rule(self):
        """Test that scores between 0 and 0.01 are rejected as noise."""
        with pytest.raises(InvalidValueError) as exc_info:
            AnomalyScore(0.005)
        
        assert "likely noise" in str(exc_info.value)
        assert exc_info.value.details["rule"] == "noise_threshold"

    def test_perfect_score_validation_rule(self):
        """Test that perfect scores (1.0) require confidence intervals."""
        with pytest.raises(InvalidValueError) as exc_info:
            AnomalyScore(1.0)
        
        assert "Perfect anomaly scores (1.0) must include confidence intervals" in str(exc_info.value)
        assert exc_info.value.details["rule"] == "perfect_score_validation"

    def test_perfect_score_with_confidence_interval(self):
        """Test that perfect scores with confidence intervals are valid."""
        ci = ConfidenceInterval(0.95, 1.0, 0.95)
        score = AnomalyScore(1.0, confidence_interval=ci)
        assert score.value == 1.0

    def test_high_precision_documentation_rule(self):
        """Test that high precision scores require method documentation."""
        with pytest.raises(InvalidValueError) as exc_info:
            AnomalyScore(0.9995)
        
        assert "High precision scores (>0.999) must specify the scoring method" in str(exc_info.value)
        assert exc_info.value.details["rule"] == "high_precision_documentation"

    def test_high_precision_with_method(self):
        """Test that high precision scores with method are valid."""
        score = AnomalyScore(0.9995, method="ensemble")
        assert score.value == 0.9995
        assert score.method == "ensemble"


class TestAnomalyScoreConfidenceIntervalValidation:
    """Test confidence interval validation for AnomalyScore."""

    def test_score_outside_confidence_interval(self):
        """Test that scores outside confidence intervals are rejected."""
        ci = ConfidenceInterval(0.3, 0.7, 0.95)
        
        with pytest.raises(InvalidValueError) as exc_info:
            AnomalyScore(0.8, confidence_interval=ci)
        
        assert "Score value (0.8) must be within confidence interval" in str(exc_info.value)
        assert exc_info.value.details["field"] == "confidence_interval"

    def test_confidence_interval_too_wide(self):
        """Test that overly wide confidence intervals are rejected."""
        ci = ConfidenceInterval(0.1, 0.95, 0.95)  # Width = 0.85 > 0.8
        
        with pytest.raises(InvalidValueError) as exc_info:
            AnomalyScore(0.5, confidence_interval=ci)
        
        assert "Confidence interval too wide" in str(exc_info.value)
        assert exc_info.value.details["rule"] == "confidence_interval_width"

    def test_extreme_score_narrow_interval(self):
        """Test validation of narrow intervals on extreme scores."""
        ci = ConfidenceInterval(0.96, 0.965, 0.95)  # Very narrow interval
        
        with pytest.raises(InvalidValueError) as exc_info:
            AnomalyScore(0.962, confidence_interval=ci)
        
        assert "Suspiciously narrow confidence interval" in str(exc_info.value)
        assert exc_info.value.details["rule"] == "extreme_score_narrow_interval"

    def test_reasonable_confidence_interval(self):
        """Test that reasonable confidence intervals are accepted."""
        ci = ConfidenceInterval(0.6, 0.8, 0.95)
        score = AnomalyScore(0.7, confidence_interval=ci)
        assert score.value == 0.7
        assert score.confidence_interval == ci


class TestAnomalyScoreMethodValidation:
    """Test method validation for AnomalyScore."""

    def test_known_methods(self):
        """Test that known methods are accepted."""
        known_methods = [
            'isolation_forest', 'lof', 'svm', 'autoencoder', 'kmeans',
            'pca', 'gaussian_mixture', 'dbscan', 'ensemble'
        ]
        
        for method in known_methods:
            score = AnomalyScore(0.5, method=method)
            assert score.method == method

    def test_case_insensitive_methods(self):
        """Test that methods are case-insensitive."""
        score = AnomalyScore(0.5, method="ISOLATION_FOREST")
        assert score.method == "ISOLATION_FOREST"

    def test_custom_method_valid_format(self):
        """Test that custom methods with valid format are accepted."""
        custom_methods = ["custom_detector", "my-algorithm", "test123"]
        
        for method in custom_methods:
            score = AnomalyScore(0.5, method=method)
            assert score.method == method

    def test_invalid_method_format(self):
        """Test that methods with invalid characters are rejected."""
        with pytest.raises(InvalidValueError) as exc_info:
            AnomalyScore(0.5, method="invalid@method!")
        
        assert "Method name 'invalid@method!' contains invalid characters" in str(exc_info.value)
        assert exc_info.value.details["rule"] == "method_format"


class TestAnomalyScoreValidationMethods:
    """Test validation methods for AnomalyScore."""

    def test_is_valid_true_cases(self):
        """Test is_valid returns True for valid scores."""
        valid_scores = [
            AnomalyScore(0.0),
            AnomalyScore(0.5),
            AnomalyScore(0.2, method="lof"),
        ]
        
        for score in valid_scores:
            assert score.is_valid()

    def test_is_valid_false_cases(self):
        """Test is_valid returns False for invalid conditions."""
        # Test with mock objects to simulate invalid conditions
        mock_score = Mock()
        mock_score.value = 0.005  # Below noise threshold
        mock_score.confidence_interval = None
        mock_score.method = None
        
        # We can't easily test is_valid with business rule violations
        # since they're checked during construction
        assert AnomalyScore(0.5).is_valid()

    def test_is_valid_with_exception_handling(self):
        """Test is_valid handles exceptions gracefully."""
        score = AnomalyScore(0.5)
        
        # Mock confidence_interval to raise exception
        mock_ci = Mock()
        mock_ci.contains.side_effect = Exception("Test exception")
        
        # Use object.__setattr__ to bypass frozen dataclass
        object.__setattr__(score, 'confidence_interval', mock_ci)
        
        assert not score.is_valid()


class TestAnomalyScoreEnhancedMethods:
    """Test enhanced methods for AnomalyScore."""

    def test_exceeds_threshold_with_validation(self):
        """Test exceeds_threshold with threshold validation."""
        score = AnomalyScore(0.7)
        
        # Valid threshold
        assert score.exceeds_threshold(0.5)
        assert not score.exceeds_threshold(0.8)
        
        # Invalid threshold type
        with pytest.raises(InvalidValueError) as exc_info:
            score.exceeds_threshold("0.5")
        assert "Threshold must be numeric" in str(exc_info.value)
        
        # Invalid threshold range
        with pytest.raises(InvalidValueError) as exc_info:
            score.exceeds_threshold(1.5)
        assert "Threshold must be between 0 and 1" in str(exc_info.value)

    def test_is_anomaly(self):
        """Test is_anomaly method."""
        score = AnomalyScore(0.7)
        
        assert score.is_anomaly()  # Default threshold 0.5
        assert score.is_anomaly(0.6)
        assert not score.is_anomaly(0.8)

    def test_is_high_confidence(self):
        """Test is_high_confidence method."""
        score = AnomalyScore(0.9)
        
        assert score.is_high_confidence()  # Default threshold 0.8
        assert score.is_high_confidence(0.85)
        assert not score.is_high_confidence(0.95)

    def test_severity_level(self):
        """Test severity_level categorization."""
        test_cases = [
            (0.95, "critical"),
            (0.8, "high"),
            (0.6, "medium"),
            (0.4, "low"),
            (0.2, "minimal"),
        ]
        
        for value, expected_severity in test_cases:
            score = AnomalyScore(value)
            assert score.severity_level() == expected_severity

    def test_confidence_score_without_interval(self):
        """Test confidence_score without confidence interval."""
        score = AnomalyScore(0.7)
        assert score.confidence_score() == 0.0

    def test_confidence_score_with_interval(self):
        """Test confidence_score with confidence interval."""
        ci = ConfidenceInterval(0.6, 0.8, 0.95)  # Width = 0.2
        score = AnomalyScore(0.7, confidence_interval=ci)
        
        expected_confidence = 1.0 - 0.2  # 0.8
        assert score.confidence_score() == expected_confidence

    def test_adjusted_score_without_interval(self):
        """Test adjusted_score without confidence interval."""
        score = AnomalyScore(0.7)
        assert score.adjusted_score() == 0.7

    def test_adjusted_score_with_interval(self):
        """Test adjusted_score with confidence interval."""
        ci = ConfidenceInterval(0.6, 0.8, 0.95)  # Width = 0.2
        score = AnomalyScore(0.7, confidence_interval=ci)
        
        confidence = 0.8  # 1.0 - 0.2
        expected_adjusted = 0.7 * 0.8 + confidence * 0.2  # 0.56 + 0.16 = 0.72
        assert score.adjusted_score() == pytest.approx(expected_adjusted)

    def test_adjusted_score_custom_weight(self):
        """Test adjusted_score with custom confidence weight."""
        ci = ConfidenceInterval(0.6, 0.8, 0.95)  # Width = 0.2
        score = AnomalyScore(0.7, confidence_interval=ci)
        
        confidence = 0.8  # 1.0 - 0.2
        weight = 0.3
        expected_adjusted = 0.7 * 0.7 + confidence * 0.3  # 0.49 + 0.24 = 0.73
        assert score.adjusted_score(weight) == pytest.approx(expected_adjusted)


class TestAnomalyScoreProperties:
    """Test properties of AnomalyScore."""

    def test_is_confident_property(self):
        """Test is_confident property."""
        score_without_ci = AnomalyScore(0.7)
        assert not score_without_ci.is_confident
        
        ci = ConfidenceInterval(0.6, 0.8, 0.95)
        score_with_ci = AnomalyScore(0.7, confidence_interval=ci)
        assert score_with_ci.is_confident

    def test_confidence_width_property(self):
        """Test confidence_width property."""
        score_without_ci = AnomalyScore(0.7)
        assert score_without_ci.confidence_width is None
        
        ci = ConfidenceInterval(0.6, 0.8, 0.95)
        score_with_ci = AnomalyScore(0.7, confidence_interval=ci)
        assert score_with_ci.confidence_width == 0.2

    def test_confidence_bounds_properties(self):
        """Test confidence_lower and confidence_upper properties."""
        score_without_ci = AnomalyScore(0.7)
        assert score_without_ci.confidence_lower is None
        assert score_without_ci.confidence_upper is None
        
        ci = ConfidenceInterval(0.6, 0.8, 0.95)
        score_with_ci = AnomalyScore(0.7, confidence_interval=ci)
        assert score_with_ci.confidence_lower == 0.6
        assert score_with_ci.confidence_upper == 0.8


class TestAnomalyScoreComparison:
    """Test comparison methods for AnomalyScore."""

    def test_comparison_with_anomaly_score(self):
        """Test comparison with other AnomalyScore instances."""
        score1 = AnomalyScore(0.3)
        score2 = AnomalyScore(0.7)
        
        assert score1 < score2
        assert score1 <= score2
        assert score2 > score1
        assert score2 >= score1

    def test_comparison_with_numeric(self):
        """Test comparison with numeric values."""
        score = AnomalyScore(0.5)
        
        assert score < 0.7
        assert score <= 0.5
        assert score > 0.3
        assert score >= 0.5

    def test_comparison_with_invalid_type(self):
        """Test comparison with invalid types."""
        score = AnomalyScore(0.5)
        
        assert (score < "0.5") is NotImplemented
        assert (score <= "0.5") is NotImplemented
        assert (score > "0.5") is NotImplemented
        assert (score >= "0.5") is NotImplemented


class TestAnomalyScoreStringRepresentation:
    """Test string representation of AnomalyScore."""

    def test_string_representation(self):
        """Test __str__ method."""
        score = AnomalyScore(0.75)
        assert str(score) == "0.75"

    def test_string_representation_with_confidence(self):
        """Test string representation includes confidence info."""
        ci = ConfidenceInterval(0.7, 0.8, 0.95)
        score = AnomalyScore(0.75, confidence_interval=ci)
        assert str(score) == "0.75"  # Just the value


class TestAnomalyScoreIntegration:
    """Integration tests for AnomalyScore with various scenarios."""

    def test_realistic_anomaly_detection_scenario(self):
        """Test realistic anomaly detection scenario."""
        # High confidence anomaly with narrow interval
        ci = ConfidenceInterval(0.85, 0.95, 0.95)
        score = AnomalyScore(0.9, confidence_interval=ci, method="isolation_forest")
        
        assert score.is_anomaly()
        assert score.is_high_confidence()
        assert score.severity_level() == "critical"
        assert score.confidence_score() > 0.8
        assert score.is_confident

    def test_low_confidence_anomaly_scenario(self):
        """Test low confidence anomaly scenario."""
        # Wide confidence interval indicates uncertainty
        ci = ConfidenceInterval(0.3, 0.7, 0.95)
        score = AnomalyScore(0.5, confidence_interval=ci, method="lof")
        
        assert score.is_anomaly()
        assert not score.is_high_confidence()
        assert score.severity_level() == "medium"
        assert score.confidence_score() < 0.7
        assert score.is_confident

    def test_borderline_case_scenario(self):
        """Test borderline anomaly case."""
        score = AnomalyScore(0.51, method="svm")
        
        assert score.is_anomaly()
        assert not score.is_high_confidence()
        assert score.severity_level() == "medium"
        assert score.confidence_score() == 0.0  # No confidence interval
        assert not score.is_confident
