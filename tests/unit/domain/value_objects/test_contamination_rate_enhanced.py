"""Tests for enhanced ContaminationRate value object validation."""

import pytest

from pynomaly.domain.value_objects.contamination_rate import ContaminationRate
from pynomaly.domain.exceptions import InvalidValueError


class TestContaminationRateBasicValidation:
    """Test basic validation rules for ContaminationRate."""

    def test_valid_contamination_rate_creation(self):
        """Test creating valid contamination rates."""
        rate = ContaminationRate(0.1)
        assert rate.value == 0.1

    def test_invalid_non_numeric_value(self):
        """Test rejection of non-numeric values."""
        with pytest.raises(InvalidValueError) as exc_info:
            ContaminationRate("0.1")
        
        assert "Contamination rate must be numeric" in str(exc_info.value)
        assert exc_info.value.details["field"] == "value"

    def test_invalid_nan_value(self):
        """Test rejection of NaN values."""
        with pytest.raises(InvalidValueError) as exc_info:
            ContaminationRate(float('nan'))
        
        assert "Contamination rate cannot be NaN or infinite" in str(exc_info.value)
        assert exc_info.value.details["field"] == "value"

    def test_invalid_infinite_value(self):
        """Test rejection of infinite values."""
        with pytest.raises(InvalidValueError) as exc_info:
            ContaminationRate(float('inf'))
        
        assert "Contamination rate cannot be NaN or infinite" in str(exc_info.value)
        assert exc_info.value.details["field"] == "value"

    def test_invalid_negative_value(self):
        """Test rejection of negative values."""
        with pytest.raises(InvalidValueError) as exc_info:
            ContaminationRate(-0.1)
        
        assert "Contamination rate must be between 0 and 0.5" in str(exc_info.value)
        assert exc_info.value.details["field"] == "value"

    def test_invalid_value_greater_than_half(self):
        """Test rejection of values greater than 0.5."""
        with pytest.raises(InvalidValueError) as exc_info:
            ContaminationRate(0.6)
        
        assert "Contamination rate must be between 0 and 0.5" in str(exc_info.value)
        assert exc_info.value.details["field"] == "value"

    def test_boundary_values(self):
        """Test boundary values (0.0 and 0.5)."""
        rate_zero = ContaminationRate(0.0)
        assert rate_zero.value == 0.0
        
        rate_half = ContaminationRate(0.5)
        assert rate_half.value == 0.5


class TestContaminationRateBusinessRules:
    """Test business rule validation for ContaminationRate."""

    def test_high_contamination_warning_rule(self):
        """Test rejection of unusually high contamination rates."""
        with pytest.raises(InvalidValueError) as exc_info:
            ContaminationRate(0.35)
        
        assert "unusually high" in str(exc_info.value)
        assert exc_info.value.details["rule"] == "high_contamination_warning"

    def test_low_contamination_warning_rule(self):
        """Test rejection of extremely low contamination rates."""
        with pytest.raises(InvalidValueError) as exc_info:
            ContaminationRate(0.0005)
        
        assert "extremely low" in str(exc_info.value)
        assert exc_info.value.details["rule"] == "low_contamination_warning"


class TestContaminationRateStandardization:
    """Test standardization methods for ContaminationRate."""

    def test_is_standard(self):
        """Test is_standard method."""
        standard_rates = [0.0, 0.05, 0.1, 0.15, 0.2, 0.25]

        for rate in standard_rates:
            assert ContaminationRate(rate).is_standard()

        assert not ContaminationRate(0.3).is_standard()


class TestContaminationRateMethods:
    """Test methods of ContaminationRate."""

    def test_as_percentage(self):
        """Test as_percentage method."""
        rate = ContaminationRate(0.1)
        assert rate.as_percentage() == 10.0

    def test_as_ratio(self):
        """Test as_ratio method."""
        rate1 = ContaminationRate(0.1)
        assert rate1.as_ratio() == "1:10"

        rate2 = ContaminationRate(0.333)
        assert rate2.as_ratio() == "1:3.0"  # Approximated ratio

    def test_severity_level(self):
        """Test severity_level method."""
        test_cases = [
            (0.0, "none"),
            (0.03, "low"),
            (0.1, "moderate"),
            (0.2, "high"),
            (0.35, "critical"),
        ]

        for value, expected_severity in test_cases:
            rate = ContaminationRate(value)
            assert rate.severity_level() == expected_severity

    def test_expected_anomalies(self):
        """Test expected_anomalies calculation."""
        rate = ContaminationRate(0.1)
        sample_size = 1000

        expected_anomalies = rate.expected_anomalies(sample_size)
        assert expected_anomalies == 100

        with pytest.raises(InvalidValueError):
            rate.expected_anomalies(-10)  # Invalid sample size

    def test_confidence_interval(self):
        """Test confidence_interval method."""
        rate = ContaminationRate(0.1)
        sample_size = 1000

        lower, upper = rate.confidence_interval(sample_size)
        assert 0.085 < lower < 0.115

        with pytest.raises(InvalidValueError):
            rate.confidence_interval(-10)  # Invalid sample size

    def test_precision_constraint(self):
        """Test precision constraint for contamination rate."""
        with pytest.raises(InvalidValueError) as exc_info:
            ContaminationRate(0.0000005)
        
        assert "excessive precision" in str(exc_info.value)
        assert exc_info.value.details["rule"] == "precision_constraint"


@pytest.mark.parametrize("value, expected_valid", [
    (0.1, True),
    (0.2, True),
    (0.0, True),
    (0.25, True),
])
def test_contamination_rate_is_valid_true_cases(value, expected_valid):
    """Test the is_valid method for ContaminationRate with valid values."""
    rate = ContaminationRate(value)
    assert rate.is_valid() == expected_valid


def test_contamination_rate_is_valid_false_cases():
    """Test cases where ContaminationRate creation fails due to business rules."""
    # These should fail during construction, not during is_valid check
    with pytest.raises(InvalidValueError):
        ContaminationRate(0.35)  # Business rule violation (unusually high)
    
    with pytest.raises(InvalidValueError):
        ContaminationRate(0.6)  # Out of range
    
    with pytest.raises(InvalidValueError):
        ContaminationRate(0.0005)  # Business rule violation (extremely low)


def test_realistic_scenario():
    """Test realistic scenario of contamination rate in an anomaly detection model."""
    rate = ContaminationRate(0.1)
    sample_size = 1000
    expected_anomalies = rate.expected_anomalies(sample_size)
    lower, upper = rate.confidence_interval(sample_size)

    assert expected_anomalies == 100
    assert 0.085 < lower < 0.115
    
    # Test high contamination rate (this should fail during construction)
    with pytest.raises(InvalidValueError):
        ContaminationRate(0.35)  # Should trigger high contamination warning
