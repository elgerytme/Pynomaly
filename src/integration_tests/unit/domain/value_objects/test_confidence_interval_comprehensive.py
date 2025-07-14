"""Comprehensive tests for ConfidenceInterval value object."""

import numpy as np
import pytest

from pynomaly.domain.exceptions import InvalidValueError
from pynomaly.domain.value_objects.confidence_interval import ConfidenceInterval


class TestConfidenceIntervalInitialization:
    """Test confidence interval initialization and validation."""

    def test_basic_initialization(self):
        """Test basic confidence interval initialization."""
        ci = ConfidenceInterval(lower=0.1, upper=0.9)

        assert ci.lower == 0.1
        assert ci.upper == 0.9
        assert ci.confidence_level == 0.95  # Default
        assert ci.method == "unknown"  # Default

    def test_initialization_with_all_parameters(self):
        """Test initialization with all parameters."""
        ci = ConfidenceInterval(
            lower=0.2, upper=0.8, confidence_level=0.9, method="bootstrap"
        )

        assert ci.lower == 0.2
        assert ci.upper == 0.8
        assert ci.confidence_level == 0.9
        assert ci.method == "bootstrap"

    def test_initialization_with_integers(self):
        """Test initialization with integer bounds."""
        ci = ConfidenceInterval(lower=1, upper=10)

        assert ci.lower == 1
        assert ci.upper == 10
        assert isinstance(ci.lower, int)
        assert isinstance(ci.upper, int)

    def test_initialization_with_equal_bounds(self):
        """Test initialization with equal lower and upper bounds."""
        ci = ConfidenceInterval(lower=0.5, upper=0.5)

        assert ci.lower == 0.5
        assert ci.upper == 0.5
        assert ci.width() == 0.0

    def test_validation_invalid_lower_bound_type(self):
        """Test validation with invalid lower bound type."""
        with pytest.raises(InvalidValueError, match="Lower bound must be numeric"):
            ConfidenceInterval(lower="0.1", upper=0.9)

    def test_validation_invalid_upper_bound_type(self):
        """Test validation with invalid upper bound type."""
        with pytest.raises(InvalidValueError, match="Upper bound must be numeric"):
            ConfidenceInterval(lower=0.1, upper="0.9")

    def test_validation_lower_greater_than_upper(self):
        """Test validation when lower bound is greater than upper bound."""
        with pytest.raises(
            InvalidValueError, match="Lower bound .* cannot be greater than upper bound"
        ):
            ConfidenceInterval(lower=0.9, upper=0.1)

    def test_validation_invalid_confidence_level(self):
        """Test validation with invalid confidence level."""
        with pytest.raises(
            InvalidValueError, match="Confidence level must be between 0 and 1"
        ):
            ConfidenceInterval(lower=0.1, upper=0.9, confidence_level=1.5)

        with pytest.raises(
            InvalidValueError, match="Confidence level must be between 0 and 1"
        ):
            ConfidenceInterval(lower=0.1, upper=0.9, confidence_level=-0.1)

    def test_validation_invalid_method(self):
        """Test validation with invalid method."""
        with pytest.raises(
            InvalidValueError, match="Method must be a non-empty string"
        ):
            ConfidenceInterval(lower=0.1, upper=0.9, method="")

        with pytest.raises(
            InvalidValueError, match="Method must be a non-empty string"
        ):
            ConfidenceInterval(lower=0.1, upper=0.9, method="   ")

        with pytest.raises(
            InvalidValueError, match="Method must be a non-empty string"
        ):
            ConfidenceInterval(lower=0.1, upper=0.9, method=123)


class TestConfidenceIntervalProperties:
    """Test confidence interval properties and methods."""

    def test_width_calculation(self):
        """Test width calculation."""
        ci = ConfidenceInterval(lower=0.2, upper=0.8)
        assert ci.width() == 0.6

    def test_width_with_equal_bounds(self):
        """Test width with equal bounds."""
        ci = ConfidenceInterval(lower=0.5, upper=0.5)
        assert ci.width() == 0.0

    def test_width_with_negative_bounds(self):
        """Test width with negative bounds."""
        ci = ConfidenceInterval(lower=-0.5, upper=0.5)
        assert ci.width() == 1.0

    def test_width_precision(self):
        """Test width calculation precision."""
        ci = ConfidenceInterval(lower=0.1111111, upper=0.9999999)
        width = ci.width()
        # Should be rounded to 10 decimal places
        assert width == 0.8888888

    def test_midpoint_calculation(self):
        """Test midpoint calculation."""
        ci = ConfidenceInterval(lower=0.2, upper=0.8)
        assert ci.midpoint() == 0.5

    def test_midpoint_with_negative_bounds(self):
        """Test midpoint with negative bounds."""
        ci = ConfidenceInterval(lower=-1.0, upper=1.0)
        assert ci.midpoint() == 0.0

    def test_midpoint_with_asymmetric_bounds(self):
        """Test midpoint with asymmetric bounds."""
        ci = ConfidenceInterval(lower=0.1, upper=0.7)
        assert ci.midpoint() == 0.4

    def test_center_property(self):
        """Test center property (alias for midpoint)."""
        ci = ConfidenceInterval(lower=0.2, upper=0.8)
        assert ci.center == 0.5
        assert ci.center == ci.midpoint()

    def test_margin_of_error_property(self):
        """Test margin of error property."""
        ci = ConfidenceInterval(lower=0.2, upper=0.8)
        assert ci.margin_of_error == 0.3  # (0.8 - 0.2) / 2

    def test_margin_of_error_with_equal_bounds(self):
        """Test margin of error with equal bounds."""
        ci = ConfidenceInterval(lower=0.5, upper=0.5)
        assert ci.margin_of_error == 0.0

    def test_is_valid_method(self):
        """Test is_valid method."""
        valid_ci = ConfidenceInterval(
            lower=0.1, upper=0.9, confidence_level=0.95, method="test"
        )
        assert valid_ci.is_valid() is True

    def test_is_valid_with_edge_cases(self):
        """Test is_valid with edge cases."""
        # Equal bounds
        ci1 = ConfidenceInterval(lower=0.5, upper=0.5)
        assert ci1.is_valid() is True

        # Confidence level at boundaries
        ci2 = ConfidenceInterval(lower=0.1, upper=0.9, confidence_level=0.0)
        assert ci2.is_valid() is True

        ci3 = ConfidenceInterval(lower=0.1, upper=0.9, confidence_level=1.0)
        assert ci3.is_valid() is True


class TestConfidenceIntervalContains:
    """Test confidence interval contains method."""

    def test_contains_value_inside(self):
        """Test contains with value inside interval."""
        ci = ConfidenceInterval(lower=0.2, upper=0.8)
        assert ci.contains(0.5) is True
        assert ci.contains(0.3) is True
        assert ci.contains(0.7) is True

    def test_contains_value_outside(self):
        """Test contains with value outside interval."""
        ci = ConfidenceInterval(lower=0.2, upper=0.8)
        assert ci.contains(0.1) is False
        assert ci.contains(0.9) is False

    def test_contains_value_at_boundaries(self):
        """Test contains with value at boundaries."""
        ci = ConfidenceInterval(lower=0.2, upper=0.8)
        assert ci.contains(0.2) is True
        assert ci.contains(0.8) is True

    def test_contains_with_equal_bounds(self):
        """Test contains with equal bounds."""
        ci = ConfidenceInterval(lower=0.5, upper=0.5)
        assert ci.contains(0.5) is True
        assert ci.contains(0.4) is False
        assert ci.contains(0.6) is False

    def test_contains_with_negative_bounds(self):
        """Test contains with negative bounds."""
        ci = ConfidenceInterval(lower=-1.0, upper=1.0)
        assert ci.contains(0.0) is True
        assert ci.contains(-0.5) is True
        assert ci.contains(0.5) is True
        assert ci.contains(-1.5) is False
        assert ci.contains(1.5) is False


class TestConfidenceIntervalOverlaps:
    """Test confidence interval overlaps method."""

    def test_overlaps_with_overlapping_intervals(self):
        """Test overlaps with overlapping intervals."""
        ci1 = ConfidenceInterval(lower=0.1, upper=0.6)
        ci2 = ConfidenceInterval(lower=0.4, upper=0.9)

        assert ci1.overlaps(ci2) is True
        assert ci2.overlaps(ci1) is True

    def test_overlaps_with_non_overlapping_intervals(self):
        """Test overlaps with non-overlapping intervals."""
        ci1 = ConfidenceInterval(lower=0.1, upper=0.4)
        ci2 = ConfidenceInterval(lower=0.6, upper=0.9)

        assert ci1.overlaps(ci2) is False
        assert ci2.overlaps(ci1) is False

    def test_overlaps_with_touching_intervals(self):
        """Test overlaps with touching intervals."""
        ci1 = ConfidenceInterval(lower=0.1, upper=0.5)
        ci2 = ConfidenceInterval(lower=0.5, upper=0.9)

        assert ci1.overlaps(ci2) is True
        assert ci2.overlaps(ci1) is True

    def test_overlaps_with_identical_intervals(self):
        """Test overlaps with identical intervals."""
        ci1 = ConfidenceInterval(lower=0.2, upper=0.8)
        ci2 = ConfidenceInterval(lower=0.2, upper=0.8)

        assert ci1.overlaps(ci2) is True
        assert ci2.overlaps(ci1) is True

    def test_overlaps_with_nested_intervals(self):
        """Test overlaps with nested intervals."""
        ci1 = ConfidenceInterval(lower=0.1, upper=0.9)
        ci2 = ConfidenceInterval(lower=0.3, upper=0.7)

        assert ci1.overlaps(ci2) is True
        assert ci2.overlaps(ci1) is True

    def test_overlaps_with_single_point_intervals(self):
        """Test overlaps with single point intervals."""
        ci1 = ConfidenceInterval(lower=0.5, upper=0.5)
        ci2 = ConfidenceInterval(lower=0.3, upper=0.7)

        assert ci1.overlaps(ci2) is True
        assert ci2.overlaps(ci1) is True


class TestConfidenceIntervalIntersection:
    """Test confidence interval intersection method."""

    def test_intersection_with_overlapping_intervals(self):
        """Test intersection with overlapping intervals."""
        ci1 = ConfidenceInterval(
            lower=0.1, upper=0.6, confidence_level=0.95, method="method1"
        )
        ci2 = ConfidenceInterval(
            lower=0.4, upper=0.9, confidence_level=0.9, method="method2"
        )

        intersection = ci1.intersection(ci2)

        assert intersection is not None
        assert intersection.lower == 0.4
        assert intersection.upper == 0.6
        assert intersection.confidence_level == 0.9  # More conservative
        assert intersection.method == "intersection(method1, method2)"

    def test_intersection_with_non_overlapping_intervals(self):
        """Test intersection with non-overlapping intervals."""
        ci1 = ConfidenceInterval(lower=0.1, upper=0.4)
        ci2 = ConfidenceInterval(lower=0.6, upper=0.9)

        intersection = ci1.intersection(ci2)

        assert intersection is None

    def test_intersection_with_touching_intervals(self):
        """Test intersection with touching intervals."""
        ci1 = ConfidenceInterval(lower=0.1, upper=0.5)
        ci2 = ConfidenceInterval(lower=0.5, upper=0.9)

        intersection = ci1.intersection(ci2)

        assert intersection is not None
        assert intersection.lower == 0.5
        assert intersection.upper == 0.5
        assert intersection.width() == 0.0

    def test_intersection_with_identical_intervals(self):
        """Test intersection with identical intervals."""
        ci1 = ConfidenceInterval(
            lower=0.2, upper=0.8, confidence_level=0.95, method="test"
        )
        ci2 = ConfidenceInterval(
            lower=0.2, upper=0.8, confidence_level=0.95, method="test"
        )

        intersection = ci1.intersection(ci2)

        assert intersection is not None
        assert intersection.lower == 0.2
        assert intersection.upper == 0.8
        assert intersection.confidence_level == 0.95
        assert intersection.method == "intersection(test, test)"

    def test_intersection_with_nested_intervals(self):
        """Test intersection with nested intervals."""
        ci1 = ConfidenceInterval(lower=0.1, upper=0.9, confidence_level=0.95)
        ci2 = ConfidenceInterval(lower=0.3, upper=0.7, confidence_level=0.9)

        intersection = ci1.intersection(ci2)

        assert intersection is not None
        assert intersection.lower == 0.3
        assert intersection.upper == 0.7
        assert intersection.confidence_level == 0.9

    def test_intersection_confidence_level_selection(self):
        """Test that intersection uses more conservative confidence level."""
        ci1 = ConfidenceInterval(lower=0.2, upper=0.8, confidence_level=0.99)
        ci2 = ConfidenceInterval(lower=0.4, upper=0.9, confidence_level=0.95)

        intersection = ci1.intersection(ci2)

        assert intersection.confidence_level == 0.95  # More conservative


class TestConfidenceIntervalConversion:
    """Test confidence interval conversion methods."""

    def test_to_tuple(self):
        """Test to_tuple conversion."""
        ci = ConfidenceInterval(lower=0.2, upper=0.8)
        result = ci.to_tuple()

        assert result == (0.2, 0.8)
        assert isinstance(result, tuple)
        assert len(result) == 2

    def test_to_dict(self):
        """Test to_dict conversion."""
        ci = ConfidenceInterval(
            lower=0.2, upper=0.8, confidence_level=0.95, method="bootstrap"
        )
        result = ci.to_dict()

        expected = {
            "lower": 0.2,
            "upper": 0.8,
            "confidence_level": 0.95,
            "method": "bootstrap",
            "width": 0.6,
            "midpoint": 0.5,
            "margin_of_error": 0.3,
        }

        assert result == expected

    def test_to_dict_with_equal_bounds(self):
        """Test to_dict with equal bounds."""
        ci = ConfidenceInterval(lower=0.5, upper=0.5)
        result = ci.to_dict()

        assert result["width"] == 0.0
        assert result["midpoint"] == 0.5
        assert result["margin_of_error"] == 0.0


class TestConfidenceIntervalClassMethods:
    """Test confidence interval class methods."""

    def test_from_bounds(self):
        """Test from_bounds class method."""
        ci = ConfidenceInterval.from_bounds(
            lower=0.1, upper=0.9, confidence_level=0.9, method="test"
        )

        assert ci.lower == 0.1
        assert ci.upper == 0.9
        assert ci.confidence_level == 0.9
        assert ci.method == "test"

    def test_from_bounds_with_defaults(self):
        """Test from_bounds with default parameters."""
        ci = ConfidenceInterval.from_bounds(lower=0.1, upper=0.9)

        assert ci.lower == 0.1
        assert ci.upper == 0.9
        assert ci.confidence_level == 0.95
        assert ci.method == "manual"

    def test_from_center_and_margin(self):
        """Test from_center_and_margin class method."""
        ci = ConfidenceInterval.from_center_and_margin(
            center=0.5, margin_of_error=0.3, confidence_level=0.9, method="test"
        )

        assert ci.lower == 0.2
        assert ci.upper == 0.8
        assert ci.confidence_level == 0.9
        assert ci.method == "test"
        assert ci.midpoint() == 0.5
        assert ci.margin_of_error == 0.3

    def test_from_center_and_margin_with_defaults(self):
        """Test from_center_and_margin with default parameters."""
        ci = ConfidenceInterval.from_center_and_margin(center=0.5, margin_of_error=0.2)

        assert ci.lower == 0.3
        assert ci.upper == 0.7
        assert ci.confidence_level == 0.95
        assert ci.method == "manual"

    def test_from_center_and_margin_zero_margin(self):
        """Test from_center_and_margin with zero margin."""
        ci = ConfidenceInterval.from_center_and_margin(center=0.5, margin_of_error=0.0)

        assert ci.lower == 0.5
        assert ci.upper == 0.5
        assert ci.width() == 0.0

    def test_from_samples_with_list(self):
        """Test from_samples with list input."""
        samples = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
        ci = ConfidenceInterval.from_samples(samples, confidence_level=0.8)

        assert ci.lower <= ci.upper
        assert ci.confidence_level == 0.8
        assert ci.method == "percentile_percentile"

    def test_from_samples_with_numpy_array(self):
        """Test from_samples with numpy array input."""
        samples = np.array([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9])
        ci = ConfidenceInterval.from_samples(samples, confidence_level=0.95)

        assert ci.lower <= ci.upper
        assert ci.confidence_level == 0.95
        assert ci.method == "percentile_percentile"

    def test_from_samples_with_empty_samples(self):
        """Test from_samples with empty samples."""
        with pytest.raises(
            InvalidValueError,
            match="Cannot create confidence interval from empty samples",
        ):
            ConfidenceInterval.from_samples([])

    def test_from_samples_with_single_value(self):
        """Test from_samples with single value."""
        ci = ConfidenceInterval.from_samples([0.5])

        assert ci.lower == 0.5
        assert ci.upper == 0.5
        assert ci.width() == 0.0

    def test_from_samples_percentile_calculation(self):
        """Test from_samples percentile calculation."""
        samples = np.arange(0, 1.01, 0.01)  # 0.00, 0.01, 0.02, ..., 1.00
        ci = ConfidenceInterval.from_samples(samples, confidence_level=0.9)

        # For 90% confidence level, we should get 5th and 95th percentiles
        expected_lower = 0.05
        expected_upper = 0.95

        assert abs(ci.lower - expected_lower) < 0.01
        assert abs(ci.upper - expected_upper) < 0.01

    def test_from_samples_custom_method(self):
        """Test from_samples with custom method name."""
        samples = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
        ci = ConfidenceInterval.from_samples(samples, method="bootstrap")

        assert ci.method == "bootstrap_percentile"


class TestConfidenceIntervalStringRepresentation:
    """Test confidence interval string representations."""

    def test_str_representation(self):
        """Test string representation."""
        ci = ConfidenceInterval(
            lower=0.2, upper=0.8, confidence_level=0.95, method="bootstrap"
        )

        str_repr = str(ci)
        assert "[0.2000, 0.8000]" in str_repr
        assert "95.0% confidence" in str_repr
        assert "bootstrap" in str_repr

    def test_str_representation_with_different_precision(self):
        """Test string representation with different precision."""
        ci = ConfidenceInterval(lower=0.123456, upper=0.876543)

        str_repr = str(ci)
        assert "[0.1235, 0.8765]" in str_repr  # 4 decimal places

    def test_repr_representation(self):
        """Test repr representation."""
        ci = ConfidenceInterval(
            lower=0.2, upper=0.8, confidence_level=0.95, method="bootstrap"
        )

        repr_str = repr(ci)
        assert "ConfidenceInterval(" in repr_str
        assert "lower=0.2" in repr_str
        assert "upper=0.8" in repr_str
        assert "confidence_level=0.95" in repr_str
        assert "method='bootstrap'" in repr_str

    def test_repr_with_default_values(self):
        """Test repr with default values."""
        ci = ConfidenceInterval(lower=0.1, upper=0.9)

        repr_str = repr(ci)
        assert "confidence_level=0.95" in repr_str
        assert "method='unknown'" in repr_str


class TestConfidenceIntervalImmutability:
    """Test confidence interval immutability."""

    def test_immutable_attributes(self):
        """Test that attributes cannot be modified."""
        ci = ConfidenceInterval(lower=0.2, upper=0.8)

        with pytest.raises(AttributeError):
            ci.lower = 0.1

        with pytest.raises(AttributeError):
            ci.upper = 0.9

        with pytest.raises(AttributeError):
            ci.confidence_level = 0.99

        with pytest.raises(AttributeError):
            ci.method = "new_method"

    def test_frozen_dataclass(self):
        """Test that the dataclass is frozen."""
        ci = ConfidenceInterval(lower=0.2, upper=0.8)

        # This should raise an error due to frozen=True
        with pytest.raises(AttributeError):
            ci.__dict__["lower"] = 0.1


class TestConfidenceIntervalEdgeCases:
    """Test confidence interval edge cases."""

    def test_very_small_intervals(self):
        """Test very small intervals."""
        ci = ConfidenceInterval(lower=0.5, upper=0.5000001)

        assert ci.width() < 0.000001
        assert ci.contains(0.5) is True
        assert ci.contains(0.5000001) is True

    def test_very_large_intervals(self):
        """Test very large intervals."""
        ci = ConfidenceInterval(lower=-1000000, upper=1000000)

        assert ci.width() == 2000000
        assert ci.midpoint() == 0.0
        assert ci.contains(0) is True
        assert ci.contains(500000) is True

    def test_negative_intervals(self):
        """Test intervals with negative values."""
        ci = ConfidenceInterval(lower=-0.8, upper=-0.2)

        assert ci.width() == 0.6
        assert ci.midpoint() == -0.5
        assert ci.contains(-0.5) is True
        assert ci.contains(-0.1) is False
        assert ci.contains(-0.9) is False

    def test_mixed_sign_intervals(self):
        """Test intervals crossing zero."""
        ci = ConfidenceInterval(lower=-0.3, upper=0.7)

        assert ci.width() == 1.0
        assert ci.midpoint() == 0.2
        assert ci.contains(0) is True
        assert ci.contains(-0.2) is True
        assert ci.contains(0.5) is True

    def test_extreme_confidence_levels(self):
        """Test extreme confidence levels."""
        ci1 = ConfidenceInterval(lower=0.1, upper=0.9, confidence_level=0.0001)
        ci2 = ConfidenceInterval(lower=0.1, upper=0.9, confidence_level=0.9999)

        assert ci1.confidence_level == 0.0001
        assert ci2.confidence_level == 0.9999

    def test_floating_point_precision(self):
        """Test floating point precision issues."""
        ci = ConfidenceInterval(lower=0.1, upper=0.1 + 1e-10)

        # Should handle very small differences
        assert ci.width() > 0
        assert ci.is_valid()

    def test_with_infinity_values(self):
        """Test handling of infinity values."""
        # Should work with finite values
        ci = ConfidenceInterval(lower=0.1, upper=float("inf"))
        assert ci.lower == 0.1
        assert ci.upper == float("inf")
        assert ci.width() == float("inf")

    def test_intersection_with_no_overlap_edge_case(self):
        """Test intersection edge case with no overlap."""
        ci1 = ConfidenceInterval(lower=0.1, upper=0.3)
        ci2 = ConfidenceInterval(lower=0.7, upper=0.9)

        intersection = ci1.intersection(ci2)
        assert intersection is None

    def test_overlaps_with_single_point_contact(self):
        """Test overlaps with single point contact."""
        ci1 = ConfidenceInterval(lower=0.1, upper=0.5)
        ci2 = ConfidenceInterval(lower=0.5, upper=0.9)

        assert ci1.overlaps(ci2) is True
        intersection = ci1.intersection(ci2)
        assert intersection is not None
        assert intersection.lower == 0.5
        assert intersection.upper == 0.5


class TestConfidenceIntervalPerformance:
    """Test confidence interval performance characteristics."""

    def test_creation_performance(self):
        """Test performance of creating many confidence intervals."""
        import time

        start_time = time.time()
        intervals = [
            ConfidenceInterval(lower=i / 10000, upper=(i + 1) / 10000)
            for i in range(1000)
        ]
        end_time = time.time()

        assert end_time - start_time < 1.0
        assert len(intervals) == 1000

    def test_overlap_checking_performance(self):
        """Test performance of overlap checking."""
        import time

        intervals = [
            ConfidenceInterval(lower=i / 1000, upper=(i + 2) / 1000) for i in range(100)
        ]

        start_time = time.time()
        overlaps = []
        for i in range(len(intervals)):
            for j in range(i + 1, len(intervals)):
                overlaps.append(intervals[i].overlaps(intervals[j]))
        end_time = time.time()

        assert end_time - start_time < 1.0
        assert len(overlaps) == 100 * 99 // 2  # Combinations

    def test_from_samples_performance(self):
        """Test performance of from_samples method."""
        import time

        large_samples = np.random.random(10000)

        start_time = time.time()
        ci = ConfidenceInterval.from_samples(large_samples)
        end_time = time.time()

        assert end_time - start_time < 1.0
        assert ci.is_valid()


class TestConfidenceIntervalIntegration:
    """Test confidence interval integration scenarios."""

    def test_multiple_interval_intersection(self):
        """Test intersection of multiple intervals."""
        ci1 = ConfidenceInterval(lower=0.1, upper=0.8)
        ci2 = ConfidenceInterval(lower=0.2, upper=0.7)
        ci3 = ConfidenceInterval(lower=0.3, upper=0.6)

        intersection1 = ci1.intersection(ci2)
        final_intersection = intersection1.intersection(ci3)

        assert final_intersection is not None
        assert final_intersection.lower == 0.3
        assert final_intersection.upper == 0.6

    def test_interval_chain_validation(self):
        """Test validation of interval chains."""
        intervals = [
            ConfidenceInterval(lower=0.1, upper=0.3),
            ConfidenceInterval(lower=0.2, upper=0.4),
            ConfidenceInterval(lower=0.3, upper=0.5),
            ConfidenceInterval(lower=0.4, upper=0.6),
        ]

        # Test chain of overlaps
        for i in range(len(intervals) - 1):
            assert intervals[i].overlaps(intervals[i + 1])

    def test_nested_interval_hierarchy(self):
        """Test nested interval hierarchy."""
        outer = ConfidenceInterval(lower=0.0, upper=1.0)
        middle = ConfidenceInterval(lower=0.2, upper=0.8)
        inner = ConfidenceInterval(lower=0.4, upper=0.6)

        # All should overlap
        assert outer.overlaps(middle)
        assert middle.overlaps(inner)
        assert outer.overlaps(inner)

        # Test containment
        assert outer.contains(middle.lower)
        assert outer.contains(middle.upper)
        assert middle.contains(inner.lower)
        assert middle.contains(inner.upper)

    def test_statistical_coverage(self):
        """Test statistical coverage properties."""
        # Generate samples from normal distribution
        np.random.seed(42)
        samples = np.random.normal(0, 1, 1000)

        # Create confidence interval
        ci = ConfidenceInterval.from_samples(samples, confidence_level=0.95)

        # Test that interval contains approximately 95% of the samples
        contained_count = sum(1 for sample in samples if ci.contains(sample))
        coverage = contained_count / len(samples)

        # Should be close to 95% (within reasonable tolerance)
        assert 0.93 <= coverage <= 0.97
