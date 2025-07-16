"""
Comprehensive tests for ContaminationRate value object.

This module tests the ContaminationRate value object to ensure proper validation,
behavior, and immutability across all use cases.
"""

import pytest

from monorepo.domain.exceptions import InvalidValueError
from monorepo.domain.value_objects import ContaminationRate


class TestContaminationRateCreation:
    """Test ContaminationRate creation and validation."""

    def test_basic_creation(self):
        """Test basic contamination rate creation."""
        rate = ContaminationRate(value=0.1)

        assert rate.value == 0.1

    def test_valid_values(self):
        """Test valid contamination rate values."""
        valid_values = [0.0, 0.05, 0.1, 0.2, 0.3, 0.4, 0.5]

        for value in valid_values:
            rate = ContaminationRate(value=value)
            assert rate.value == value

    def test_boundary_values(self):
        """Test boundary contamination rate values."""
        # Test exact boundaries
        min_rate = ContaminationRate(value=0.0)
        max_rate = ContaminationRate(value=0.5)

        assert min_rate.value == 0.0
        assert max_rate.value == 0.5

    def test_integer_values(self):
        """Test integer values are accepted."""
        rate = ContaminationRate(value=0)  # integer 0
        assert rate.value == 0.0
        assert isinstance(rate.value, (int, float))


class TestContaminationRateValidation:
    """Test validation of ContaminationRate parameters."""

    def test_invalid_values_too_low(self):
        """Test contamination rates below 0.0."""
        invalid_values = [-0.1, -1.0, -0.01]

        for value in invalid_values:
            with pytest.raises(
                InvalidValueError,
                match="Contamination rate must be between 0 and 0.5",
            ):
                ContaminationRate(value=value)

    def test_invalid_values_too_high(self):
        """Test contamination rates above 0.5."""
        invalid_values = [0.51, 0.6, 1.0, 2.0]

        for value in invalid_values:
            with pytest.raises(
                InvalidValueError,
                match="Contamination rate must be between 0 and 0.5",
            ):
                ContaminationRate(value=value)

    def test_non_numeric_values(self):
        """Test non-numeric contamination rates."""
        invalid_values = ["0.1", None, [], {}, "high", True]

        for value in invalid_values:
            with pytest.raises(
                InvalidValueError, match="Contamination rate must be numeric"
            ):
                ContaminationRate(value=value)

    def test_invalid_special_float_values(self):
        """Test special float values."""
        import math

        invalid_values = [math.nan, math.inf, -math.inf]

        for value in invalid_values:
            with pytest.raises(InvalidValueError):
                ContaminationRate(value=value)


class TestContaminationRateBehavior:
    """Test ContaminationRate behavior and methods."""

    def test_is_valid_method(self):
        """Test is_valid method."""
        valid_rate = ContaminationRate(value=0.1)
        assert valid_rate.is_valid() is True

        # Test edge cases
        edge_cases = [
            ContaminationRate(value=0.0),
            ContaminationRate(value=0.5),
        ]

        for rate in edge_cases:
            assert rate.is_valid() is True

    def test_as_percentage(self):
        """Test percentage conversion."""
        test_cases = [
            (0.0, 0.0),
            (0.05, 5.0),
            (0.1, 10.0),
            (0.2, 20.0),
            (0.5, 50.0),
        ]

        for value, expected_percentage in test_cases:
            rate = ContaminationRate(value=value)
            assert rate.as_percentage() == expected_percentage

    def test_string_representation(self):
        """Test string representation as percentage."""
        test_cases = [
            (0.0, "0.0%"),
            (0.05, "5.0%"),
            (0.1, "10.0%"),
            (0.15, "15.0%"),
            (0.2, "20.0%"),
            (0.25, "25.0%"),
            (0.5, "50.0%"),
        ]

        for value, expected_str in test_cases:
            rate = ContaminationRate(value=value)
            assert str(rate) == expected_str

    def test_string_representation_precision(self):
        """Test string representation with decimal precision."""
        # Test with one decimal place precision
        rate = ContaminationRate(value=0.123)  # 12.3%
        assert str(rate) == "12.3%"

        rate = ContaminationRate(value=0.456)  # 45.6%
        assert str(rate) == "45.6%"


class TestContaminationRateComparison:
    """Test ContaminationRate comparison operations."""

    def test_comparison_with_other_rates(self):
        """Test comparison operations with other rates."""
        rate1 = ContaminationRate(value=0.1)
        rate2 = ContaminationRate(value=0.2)
        rate3 = ContaminationRate(value=0.1)

        assert rate1 < rate2
        assert rate1 <= rate2
        assert rate2 > rate1
        assert rate2 >= rate1
        assert rate1 <= rate3
        assert rate1 >= rate3

    def test_comparison_with_numbers(self):
        """Test comparison operations with numbers."""
        rate = ContaminationRate(value=0.2)

        assert rate > 0.1
        assert rate >= 0.2
        assert rate < 0.3
        assert rate <= 0.2
        assert not (rate > 0.2)
        assert not (rate < 0.2)

    def test_comparison_with_invalid_types(self):
        """Test comparison with invalid types."""
        rate = ContaminationRate(value=0.2)

        with pytest.raises(TypeError):
            rate < "0.1"

        with pytest.raises(TypeError):
            rate > []

    def test_equality(self):
        """Test equality comparison."""
        rate1 = ContaminationRate(value=0.1)
        rate2 = ContaminationRate(value=0.1)
        rate3 = ContaminationRate(value=0.2)

        assert rate1 == rate2
        assert rate1 != rate3
        assert rate1 != 0.1  # Different types
        assert rate1 != "0.1"


class TestContaminationRateClassMethods:
    """Test ContaminationRate class methods for common rates."""

    def test_auto_method(self):
        """Test auto class method."""
        auto_rate = ContaminationRate.auto()

        assert auto_rate.value == 0.1
        assert isinstance(auto_rate, ContaminationRate)

    def test_low_method(self):
        """Test low class method."""
        low_rate = ContaminationRate.low()

        assert low_rate.value == 0.05
        assert isinstance(low_rate, ContaminationRate)

    def test_medium_method(self):
        """Test medium class method."""
        medium_rate = ContaminationRate.medium()

        assert medium_rate.value == 0.1
        assert isinstance(medium_rate, ContaminationRate)

    def test_high_method(self):
        """Test high class method."""
        high_rate = ContaminationRate.high()

        assert high_rate.value == 0.2
        assert isinstance(high_rate, ContaminationRate)

    def test_class_method_relationships(self):
        """Test relationships between class methods."""
        low = ContaminationRate.low()
        medium = ContaminationRate.medium()
        high = ContaminationRate.high()
        auto = ContaminationRate.auto()

        assert low < medium
        assert medium < high
        assert auto == medium  # auto and medium are the same


class TestContaminationRateClassConstants:
    """Test ContaminationRate class constants."""

    def test_class_constants_exist(self):
        """Test that class constants are defined."""
        assert hasattr(ContaminationRate, "AUTO")
        assert hasattr(ContaminationRate, "LOW")
        assert hasattr(ContaminationRate, "MEDIUM")
        assert hasattr(ContaminationRate, "HIGH")

    def test_class_constants_values(self):
        """Test class constant values."""
        assert ContaminationRate.AUTO.value == 0.1
        assert ContaminationRate.LOW.value == 0.05
        assert ContaminationRate.MEDIUM.value == 0.1
        assert ContaminationRate.HIGH.value == 0.2

    def test_class_constants_types(self):
        """Test class constant types."""
        assert isinstance(ContaminationRate.AUTO, ContaminationRate)
        assert isinstance(ContaminationRate.LOW, ContaminationRate)
        assert isinstance(ContaminationRate.MEDIUM, ContaminationRate)
        assert isinstance(ContaminationRate.HIGH, ContaminationRate)

    def test_class_constants_relationships(self):
        """Test relationships between class constants."""
        assert ContaminationRate.LOW < ContaminationRate.MEDIUM
        assert ContaminationRate.MEDIUM < ContaminationRate.HIGH
        assert ContaminationRate.AUTO == ContaminationRate.MEDIUM

    def test_class_constants_immutability(self):
        """Test that class constants cannot be modified."""
        original_auto = ContaminationRate.AUTO

        # Should not be able to reassign
        with pytest.raises(AttributeError):
            ContaminationRate.AUTO = ContaminationRate(0.3)

        assert ContaminationRate.AUTO == original_auto


class TestContaminationRateImmutability:
    """Test ContaminationRate immutability."""

    def test_frozen_dataclass(self):
        """Test that ContaminationRate is frozen."""
        rate = ContaminationRate(value=0.1)

        with pytest.raises(AttributeError):
            rate.value = 0.2

    def test_hash_consistency(self):
        """Test that equal rates have equal hashes."""
        rate1 = ContaminationRate(value=0.1)
        rate2 = ContaminationRate(value=0.1)
        rate3 = ContaminationRate(value=0.2)

        assert hash(rate1) == hash(rate2)
        assert hash(rate1) != hash(rate3)

    def test_use_in_sets(self):
        """Test using contamination rates in sets."""
        rate1 = ContaminationRate(value=0.1)
        rate2 = ContaminationRate(value=0.2)
        rate3 = ContaminationRate(value=0.1)  # Same as rate1

        rate_set = {rate1, rate2, rate3}

        assert len(rate_set) == 2  # rate1 and rate3 are the same
        assert rate1 in rate_set
        assert rate2 in rate_set

    def test_use_as_dict_keys(self):
        """Test using contamination rates as dictionary keys."""
        rate1 = ContaminationRate(value=0.1)
        rate2 = ContaminationRate(value=0.2)

        rate_dict = {rate1: "medium", rate2: "high"}

        assert len(rate_dict) == 2
        assert rate_dict[rate1] == "medium"
        assert rate_dict[rate2] == "high"


class TestContaminationRateEdgeCases:
    """Test ContaminationRate edge cases and boundary conditions."""

    def test_floating_point_precision(self):
        """Test floating point precision handling."""
        # Test with high precision values
        precise_value = 0.123456789
        rate = ContaminationRate(value=precise_value)

        assert rate.value == precise_value

    def test_very_small_values(self):
        """Test very small but valid values."""
        small_values = [0.001, 0.0001, 0.00001]

        for value in small_values:
            rate = ContaminationRate(value=value)
            assert rate.value == value
            assert rate.is_valid()

    def test_values_close_to_boundary(self):
        """Test values very close to boundaries."""
        # Values very close to 0.5 but still valid
        close_to_max = 0.499999999
        rate = ContaminationRate(value=close_to_max)
        assert rate.value == close_to_max

        # Values very close to 0.0
        close_to_min = 0.000000001
        rate = ContaminationRate(value=close_to_min)
        assert rate.value == close_to_min

    def test_percentage_calculation_precision(self):
        """Test percentage calculation with various precisions."""
        test_cases = [
            (0.001, 0.1),  # 0.1%
            (0.0001, 0.01),  # 0.01%
            (0.12345, 12.345),  # 12.345%
        ]

        for value, expected_percentage in test_cases:
            rate = ContaminationRate(value=value)
            assert abs(rate.as_percentage() - expected_percentage) < 1e-10

    def test_string_representation_edge_cases(self):
        """Test string representation for edge cases."""
        # Test exact integer percentages
        rate = ContaminationRate(value=0.1)  # Exactly 10%
        assert str(rate) == "10.0%"

        # Test with decimal places
        rate = ContaminationRate(value=0.125)  # 12.5%
        assert str(rate) == "12.5%"

        # Test very small value
        rate = ContaminationRate(value=0.001)  # 0.1%
        assert str(rate) == "0.1%"


class TestContaminationRateRepresentation:
    """Test ContaminationRate representation methods."""

    def test_repr_method(self):
        """Test repr representation."""
        rate = ContaminationRate(value=0.15)

        repr_str = repr(rate)

        assert "ContaminationRate" in repr_str
        assert "0.15" in repr_str

    def test_str_vs_repr(self):
        """Test difference between str and repr."""
        rate = ContaminationRate(value=0.1)

        str_repr = str(rate)  # Should be "10.0%"
        repr_repr = repr(rate)  # Should be "ContaminationRate(value=0.1)"

        assert str_repr != repr_repr
        assert "%" in str_repr
        assert "ContaminationRate" in repr_repr


class TestContaminationRateUsagePatterns:
    """Test common usage patterns for ContaminationRate."""

    def test_sorting_rates(self):
        """Test sorting contamination rates."""
        rates = [
            ContaminationRate(value=0.3),
            ContaminationRate(value=0.1),
            ContaminationRate(value=0.5),
            ContaminationRate(value=0.05),
        ]

        sorted_rates = sorted(rates)

        expected_values = [0.05, 0.1, 0.3, 0.5]
        actual_values = [rate.value for rate in sorted_rates]

        assert actual_values == expected_values

    def test_filtering_rates(self):
        """Test filtering contamination rates."""
        rates = [
            ContaminationRate(value=0.05),
            ContaminationRate(value=0.1),
            ContaminationRate(value=0.2),
            ContaminationRate(value=0.4),
        ]

        # Filter rates above 0.15
        high_rates = [rate for rate in rates if rate > 0.15]

        assert len(high_rates) == 2
        assert all(rate.value > 0.15 for rate in high_rates)

    def test_rate_categorization(self):
        """Test categorizing rates using class constants."""

        def categorize_rate(rate: ContaminationRate) -> str:
            if rate <= ContaminationRate.LOW:
                return "low"
            elif rate <= ContaminationRate.MEDIUM:
                return "medium"
            elif rate <= ContaminationRate.HIGH:
                return "high"
            else:
                return "very_high"

        test_cases = [
            (0.03, "low"),
            (0.05, "low"),
            (0.1, "medium"),
            (0.2, "high"),
            (0.4, "very_high"),
        ]

        for value, expected_category in test_cases:
            rate = ContaminationRate(value=value)
            actual_category = categorize_rate(rate)
            assert actual_category == expected_category


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
