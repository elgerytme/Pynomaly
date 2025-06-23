"""Unit tests for domain value objects."""

import pytest

from pynomaly.domain.value_objects import AnomalyScore, ConfidenceInterval, ContaminationRate


class TestAnomalyScore:
    """Test AnomalyScore value object."""
    
    def test_creation_valid(self):
        """Test creating valid anomaly score."""
        score = AnomalyScore(value=0.8)
        assert score.value == 0.8
        assert score.confidence_lower is None
        assert score.confidence_upper is None
        assert score.method is None
    
    def test_creation_with_confidence(self):
        """Test creating score with confidence intervals."""
        score = AnomalyScore(
            value=0.8,
            confidence_lower=0.7,
            confidence_upper=0.9,
            method="bootstrap"
        )
        assert score.value == 0.8
        assert score.confidence_lower == 0.7
        assert score.confidence_upper == 0.9
        assert score.method == "bootstrap"
        assert score.is_confident is True
        assert score.confidence_width == 0.2
    
    def test_invalid_value_type(self):
        """Test that non-numeric values raise error."""
        with pytest.raises(ValueError, match="Score value must be numeric"):
            AnomalyScore(value="not a number")  # type: ignore
    
    def test_invalid_confidence_bounds(self):
        """Test that invalid confidence bounds raise error."""
        with pytest.raises(ValueError, match="Lower confidence bound.*cannot be greater"):
            AnomalyScore(value=0.8, confidence_lower=0.9, confidence_upper=0.7)
    
    def test_value_outside_confidence(self):
        """Test that value outside confidence interval raises error."""
        with pytest.raises(ValueError, match="Score value.*must be within confidence interval"):
            AnomalyScore(value=0.5, confidence_lower=0.7, confidence_upper=0.9)
    
    def test_comparison_operators(self):
        """Test comparison operators."""
        score1 = AnomalyScore(value=0.8)
        score2 = AnomalyScore(value=0.6)
        score3 = AnomalyScore(value=0.8)
        
        assert score1 > score2
        assert score2 < score1
        assert score1 >= score3
        assert score1 <= score3
        assert score1 > 0.6
        assert score1 >= 0.8
        assert score2 < 0.8
        assert score2 <= 0.6
    
    def test_exceeds_threshold(self):
        """Test threshold comparison."""
        score = AnomalyScore(value=0.8)
        assert score.exceeds_threshold(0.7) is True
        assert score.exceeds_threshold(0.9) is False
        assert score.exceeds_threshold(0.8) is False


class TestContaminationRate:
    """Test ContaminationRate value object."""
    
    def test_creation_valid(self):
        """Test creating valid contamination rate."""
        rate = ContaminationRate(value=0.1)
        assert rate.value == 0.1
        assert rate.as_percentage == 10.0
        assert rate.is_auto is True
    
    def test_creation_from_percentage(self):
        """Test creating from percentage."""
        rate = ContaminationRate.from_percentage(15)
        assert rate.value == 0.15
        assert rate.as_percentage == 15.0
        assert rate.is_auto is False
    
    def test_auto_creation(self):
        """Test auto contamination rate."""
        rate = ContaminationRate.auto()
        assert rate.value == 0.1
        assert rate.is_auto is True
    
    def test_invalid_type(self):
        """Test that non-numeric values raise error."""
        with pytest.raises(TypeError, match="Contamination rate must be numeric"):
            ContaminationRate(value="invalid")  # type: ignore
    
    def test_invalid_range(self):
        """Test that values outside [0, 1] raise error."""
        with pytest.raises(ValueError, match="Contamination rate must be between 0 and 1"):
            ContaminationRate(value=1.5)
        
        with pytest.raises(ValueError, match="Contamination rate must be between 0 and 1"):
            ContaminationRate(value=-0.1)
    
    def test_invalid_percentage(self):
        """Test that invalid percentages raise error."""
        with pytest.raises(ValueError, match="Percentage must be between 0 and 100"):
            ContaminationRate.from_percentage(150)
    
    def test_calculate_threshold_index(self):
        """Test threshold index calculation."""
        rate = ContaminationRate(value=0.1)
        
        assert rate.calculate_threshold_index(100) == 10
        assert rate.calculate_threshold_index(50) == 5
        assert rate.calculate_threshold_index(5) == 1  # At least 1
        
        # Edge cases
        rate_zero = ContaminationRate(value=0.0)
        assert rate_zero.calculate_threshold_index(100) == 0
        
        rate_high = ContaminationRate(value=0.99)
        assert rate_high.calculate_threshold_index(100) == 99  # Not all samples
    
    def test_string_representations(self):
        """Test string representations."""
        rate = ContaminationRate(value=0.15)
        assert str(rate) == "15.0%"
        assert repr(rate) == "ContaminationRate(value=0.15)"


class TestConfidenceInterval:
    """Test ConfidenceInterval value object."""
    
    def test_creation_valid(self):
        """Test creating valid confidence interval."""
        ci = ConfidenceInterval(lower=0.7, upper=0.9, level=0.95)
        assert ci.lower == 0.7
        assert ci.upper == 0.9
        assert ci.level == 0.95
        assert ci.width == 0.2
        assert ci.midpoint == 0.8
        assert ci.margin_of_error == 0.1
        assert ci.as_percentage == 95
    
    def test_creation_from_point_estimate(self):
        """Test creating from point estimate."""
        ci = ConfidenceInterval.from_point_estimate(
            point=0.8, margin=0.1, level=0.95
        )
        assert ci.lower == 0.7
        assert ci.upper == 0.9
        assert ci.level == 0.95
    
    def test_creation_from_percentiles(self):
        """Test creating from data percentiles."""
        data = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
        ci = ConfidenceInterval.from_percentiles(data, level=0.8)
        
        # With 80% CI and 10 points, we exclude 1 from each end
        assert ci.lower == 0.2
        assert ci.upper == 0.9
        assert ci.level == 0.8
    
    def test_invalid_bounds(self):
        """Test that invalid bounds raise error."""
        with pytest.raises(ValueError, match="Lower bound.*cannot be greater"):
            ConfidenceInterval(lower=0.9, upper=0.7)
    
    def test_invalid_level(self):
        """Test that invalid confidence level raises error."""
        with pytest.raises(ValueError, match="Confidence level must be between 0 and 1"):
            ConfidenceInterval(lower=0.7, upper=0.9, level=1.5)
    
    def test_contains(self):
        """Test value containment check."""
        ci = ConfidenceInterval(lower=0.7, upper=0.9)
        assert ci.contains(0.8) is True
        assert ci.contains(0.7) is True
        assert ci.contains(0.9) is True
        assert ci.contains(0.6) is False
        assert ci.contains(1.0) is False
    
    def test_overlaps(self):
        """Test interval overlap check."""
        ci1 = ConfidenceInterval(lower=0.7, upper=0.9)
        ci2 = ConfidenceInterval(lower=0.8, upper=1.0)
        ci3 = ConfidenceInterval(lower=0.1, upper=0.6)
        
        assert ci1.overlaps(ci2) is True
        assert ci2.overlaps(ci1) is True
        assert ci1.overlaps(ci3) is False
    
    def test_union(self):
        """Test interval union."""
        ci1 = ConfidenceInterval(lower=0.7, upper=0.9, level=0.95)
        ci2 = ConfidenceInterval(lower=0.8, upper=1.0, level=0.90)
        
        union = ci1.union(ci2)
        assert union is not None
        assert union.lower == 0.7
        assert union.upper == 1.0
        assert union.level == 0.90  # More conservative
        
        # Non-overlapping intervals
        ci3 = ConfidenceInterval(lower=0.1, upper=0.5)
        assert ci1.union(ci3) is None
    
    def test_intersection(self):
        """Test interval intersection."""
        ci1 = ConfidenceInterval(lower=0.7, upper=0.9, level=0.95)
        ci2 = ConfidenceInterval(lower=0.8, upper=1.0, level=0.90)
        
        intersection = ci1.intersection(ci2)
        assert intersection is not None
        assert intersection.lower == 0.8
        assert intersection.upper == 0.9
        assert intersection.level == 0.95  # Less conservative
        
        # Non-overlapping intervals
        ci3 = ConfidenceInterval(lower=0.1, upper=0.5)
        assert ci1.intersection(ci3) is None
    
    def test_as_tuple(self):
        """Test tuple conversion."""
        ci = ConfidenceInterval(lower=0.7, upper=0.9)
        assert ci.as_tuple() == (0.7, 0.9)
    
    def test_string_representations(self):
        """Test string representations."""
        ci = ConfidenceInterval(lower=0.7, upper=0.9, level=0.95)
        assert str(ci) == "[0.700, 0.900] (95% CI)"
        assert repr(ci) == "ConfidenceInterval(lower=0.7, upper=0.9, level=0.95)"