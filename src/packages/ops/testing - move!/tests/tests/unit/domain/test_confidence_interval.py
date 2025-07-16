"""
Unit tests for confidence interval value object.

Tests the ConfidenceInterval value object functionality including
validation, calculations, and methods.
"""

import numpy as np
import pytest
from hypothesis import assume, given
from hypothesis import strategies as st

from monorepo.domain.exceptions import InvalidValueError
from monorepo.domain.value_objects.confidence_interval import ConfidenceInterval


@given(st.floats(min_value=0.0, max_value=1.0), st.floats(min_value=0.0, max_value=1.0))
def test_confidence_interval_random(lower, upper):
    assume(lower <= upper)
    ci = ConfidenceInterval(lower=lower, upper=upper, confidence_level=0.95)
    assert ci.is_valid()


class TestConfidenceInterval:
    """Test cases for ConfidenceInterval value object."""

    def test_create_valid_confidence_interval(self):
        """Test creating a valid confidence interval."""
        ci = ConfidenceInterval(
            lower=0.1, upper=0.9, confidence_level=0.95, method="test"
        )

        assert ci.lower == 0.1
        assert ci.upper == 0.9
        assert ci.confidence_level == 0.95
        assert ci.method == "test"
        assert ci.is_valid()

    def test_confidence_interval_validation(self):
        """Test confidence interval validation."""
        # Test invalid lower > upper
        with pytest.raises(
            InvalidValueError, match="Lower bound.*cannot be greater than upper bound"
        ):
            ConfidenceInterval(lower=0.9, upper=0.1, method="test")

        # Test invalid confidence level
        with pytest.raises(
            InvalidValueError, match="Confidence level must be between 0 and 1"
        ):
            ConfidenceInterval(
                lower=0.1, upper=0.9, confidence_level=1.5, method="test"
            )

        with pytest.raises(
            InvalidValueError, match="Confidence level must be between 0 and 1"
        ):
            ConfidenceInterval(
                lower=0.1, upper=0.9, confidence_level=-0.1, method="test"
            )

        # Test empty method
        with pytest.raises(
            InvalidValueError, match="Method must be a non-empty string"
        ):
            ConfidenceInterval(lower=0.1, upper=0.9, method="")

        # Test non-numeric bounds
        with pytest.raises(InvalidValueError, match="Lower bound must be numeric"):
            ConfidenceInterval(lower="invalid", upper=0.9, method="test")

        with pytest.raises(InvalidValueError, match="Upper bound must be numeric"):
            ConfidenceInterval(lower=0.1, upper="invalid", method="test")

    def test_confidence_interval_calculations(self):
        """Test confidence interval calculations."""
        ci = ConfidenceInterval(lower=10.0, upper=20.0, method="test")

        assert ci.width() == 10.0
        assert ci.midpoint() == 15.0
        assert ci.center == 15.0
        assert ci.margin_of_error == 5.0

    def test_contains_method(self):
        """Test contains method."""
        ci = ConfidenceInterval(lower=10.0, upper=20.0, method="test")

        assert ci.contains(15.0)
        assert ci.contains(10.0)  # Boundary case
        assert ci.contains(20.0)  # Boundary case
        assert not ci.contains(5.0)
        assert not ci.contains(25.0)

    def test_overlaps_method(self):
        """Test overlaps method."""
        ci1 = ConfidenceInterval(lower=10.0, upper=20.0, method="test1")
        ci2 = ConfidenceInterval(lower=15.0, upper=25.0, method="test2")
        ci3 = ConfidenceInterval(lower=25.0, upper=35.0, method="test3")

        assert ci1.overlaps(ci2)
        assert ci2.overlaps(ci1)
        assert not ci1.overlaps(ci3)
        assert not ci3.overlaps(ci1)

    def test_intersection_method(self):
        """Test intersection method."""
        ci1 = ConfidenceInterval(
            lower=10.0, upper=20.0, confidence_level=0.95, method="test1"
        )
        ci2 = ConfidenceInterval(
            lower=15.0, upper=25.0, confidence_level=0.90, method="test2"
        )
        ci3 = ConfidenceInterval(lower=25.0, upper=35.0, method="test3")

        # Test overlapping intervals
        intersection = ci1.intersection(ci2)
        assert intersection is not None
        assert intersection.lower == 15.0
        assert intersection.upper == 20.0
        assert intersection.confidence_level == 0.90  # More conservative
        assert "intersection" in intersection.method

        # Test non-overlapping intervals
        no_intersection = ci1.intersection(ci3)
        assert no_intersection is None

    def test_to_tuple_method(self):
        """Test to_tuple method."""
        ci = ConfidenceInterval(lower=10.0, upper=20.0, method="test")

        assert ci.to_tuple() == (10.0, 20.0)

    def test_to_dict_method(self):
        """Test to_dict method."""
        ci = ConfidenceInterval(
            lower=10.0, upper=20.0, confidence_level=0.95, method="test"
        )

        result = ci.to_dict()
        expected = {
            "lower": 10.0,
            "upper": 20.0,
            "confidence_level": 0.95,
            "method": "test",
            "width": 10.0,
            "midpoint": 15.0,
            "margin_of_error": 5.0,
        }

        assert result == expected

    def test_from_bounds_classmethod(self):
        """Test from_bounds class method."""
        ci = ConfidenceInterval.from_bounds(
            lower=5.0, upper=15.0, confidence_level=0.90, method="manual"
        )

        assert ci.lower == 5.0
        assert ci.upper == 15.0
        assert ci.confidence_level == 0.90
        assert ci.method == "manual"

    def test_from_center_and_margin_classmethod(self):
        """Test from_center_and_margin class method."""
        ci = ConfidenceInterval.from_center_and_margin(
            center=10.0, margin_of_error=3.0, confidence_level=0.90, method="manual"
        )

        assert ci.lower == 7.0
        assert ci.upper == 13.0
        assert ci.confidence_level == 0.90
        assert ci.method == "manual"

    def test_from_samples_classmethod(self):
        """Test from_samples class method."""
        samples = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]

        ci = ConfidenceInterval.from_samples(
            samples=samples, confidence_level=0.90, method="percentile"
        )

        # For 90% confidence level, we expect 5th and 95th percentiles
        assert ci.confidence_level == 0.90
        assert "percentile" in ci.method
        assert ci.lower <= ci.upper

        # Test with numpy array
        samples_array = np.array(samples)
        ci_array = ConfidenceInterval.from_samples(
            samples=samples_array, confidence_level=0.90
        )

        assert ci_array.confidence_level == 0.90

    def test_from_samples_empty_error(self):
        """Test from_samples with empty samples raises error."""
        with pytest.raises(
            InvalidValueError,
            match="Cannot create confidence interval from empty samples",
        ):
            ConfidenceInterval.from_samples(samples=[])

    def test_string_representation(self):
        """Test string representations."""
        ci = ConfidenceInterval(
            lower=10.1234, upper=20.5678, confidence_level=0.95, method="test"
        )

        str_repr = str(ci)
        assert "10.1234" in str_repr
        assert "20.5678" in str_repr
        assert "95.0%" in str_repr  # Updated to match actual format
        assert "test" in str_repr

        repr_str = repr(ci)
        assert "ConfidenceInterval" in repr_str
        assert "lower=10.1234" in repr_str
        assert "upper=20.5678" in repr_str

    def test_equality_and_hashing(self):
        """Test equality and hashing (since dataclass is frozen)."""
        ci1 = ConfidenceInterval(lower=10.0, upper=20.0, method="test")
        ci2 = ConfidenceInterval(lower=10.0, upper=20.0, method="test")
        ci3 = ConfidenceInterval(lower=10.0, upper=21.0, method="test")

        assert ci1 == ci2
        assert ci1 != ci3

        # Test that frozen dataclass can be hashed
        assert hash(ci1) == hash(ci2)
        assert hash(ci1) != hash(ci3)

    def test_default_values(self):
        """Test default values for optional parameters."""
        ci = ConfidenceInterval(lower=10.0, upper=20.0)

        assert ci.confidence_level == 0.95
        assert ci.method == "unknown"

    def test_edge_cases(self):
        """Test edge cases."""
        # Test with equal bounds (zero width)
        ci_zero = ConfidenceInterval(lower=10.0, upper=10.0, method="test")
        assert ci_zero.width() == 0.0
        assert ci_zero.margin_of_error == 0.0
        assert ci_zero.contains(10.0)

        # Test with very small bounds
        ci_small = ConfidenceInterval(lower=1e-10, upper=2e-10, method="test")
        assert ci_small.width() == 1e-10

        # Test with very large bounds
        ci_large = ConfidenceInterval(lower=1e10, upper=2e10, method="test")
        assert ci_large.width() == 1e10

    def test_confidence_levels_boundary(self):
        """Test boundary confidence levels."""
        # Test 0% confidence
        ci_0 = ConfidenceInterval(
            lower=10.0, upper=20.0, confidence_level=0.0, method="test"
        )
        assert ci_0.confidence_level == 0.0

        # Test 100% confidence
        ci_100 = ConfidenceInterval(
            lower=10.0, upper=20.0, confidence_level=1.0, method="test"
        )
        assert ci_100.confidence_level == 1.0

    def test_numeric_precision(self):
        """Test numeric precision in calculations."""
        # Test that width calculation has proper rounding
        ci = ConfidenceInterval(lower=0.1000000001, upper=0.1000000002, method="test")

        # Width should be rounded to 10 decimal places
        assert ci.width() == round(0.1000000002 - 0.1000000001, 10)
