"""Tests for contamination rate value object."""

import pytest

from monorepo.domain.exceptions import InvalidValueError
from monorepo.domain.value_objects.contamination_rate import ContaminationRate


class TestContaminationRate:
    """Test suite for ContaminationRate value object."""

    def test_valid_contamination_rate(self):
        """Test creating valid contamination rate."""
        rate = ContaminationRate(0.1)
        assert rate.value == 0.1
        assert isinstance(rate.value, float)

    def test_contamination_rate_immutability(self):
        """Test contamination rate is immutable."""
        rate = ContaminationRate(0.1)

        # Should not be able to modify value
        with pytest.raises(AttributeError):
            rate.value = 0.2

    def test_contamination_rate_validation_type(self):
        """Test contamination rate type validation."""
        # Valid types
        ContaminationRate(0.1)  # float
        ContaminationRate(0.1)  # converted from int
        ContaminationRate(0)  # int zero

        # Invalid types
        with pytest.raises(InvalidValueError, match="must be numeric"):
            ContaminationRate("0.1")

        with pytest.raises(InvalidValueError, match="must be numeric"):
            ContaminationRate(None)

        with pytest.raises(InvalidValueError, match="must be numeric"):
            ContaminationRate([0.1])

    def test_contamination_rate_validation_range(self):
        """Test contamination rate range validation."""
        # Valid range
        ContaminationRate(0.0)  # minimum
        ContaminationRate(0.25)  # middle
        ContaminationRate(0.5)  # maximum

        # Invalid range
        with pytest.raises(InvalidValueError, match="must be between 0 and 0.5"):
            ContaminationRate(-0.1)

        with pytest.raises(InvalidValueError, match="must be between 0 and 0.5"):
            ContaminationRate(0.6)

        with pytest.raises(InvalidValueError, match="must be between 0 and 0.5"):
            ContaminationRate(1.0)

    def test_contamination_rate_boundary_values(self):
        """Test contamination rate boundary values."""
        # Test exact boundaries
        min_rate = ContaminationRate(0.0)
        assert min_rate.value == 0.0

        max_rate = ContaminationRate(0.5)
        assert max_rate.value == 0.5

        # Test just inside boundaries
        just_above_min = ContaminationRate(0.001)
        assert just_above_min.value == 0.001

        just_below_max = ContaminationRate(0.499)
        assert just_below_max.value == 0.499

    def test_is_valid_method(self):
        """Test is_valid method."""
        # Valid rates
        rate1 = ContaminationRate(0.1)
        assert rate1.is_valid() is True

        rate2 = ContaminationRate(0.0)
        assert rate2.is_valid() is True

        rate3 = ContaminationRate(0.5)
        assert rate3.is_valid() is True

    def test_as_percentage_method(self):
        """Test as_percentage method."""
        rate1 = ContaminationRate(0.1)
        assert rate1.as_percentage() == 10.0

        rate2 = ContaminationRate(0.05)
        assert rate2.as_percentage() == 5.0

        rate3 = ContaminationRate(0.25)
        assert rate3.as_percentage() == 25.0

        rate4 = ContaminationRate(0.0)
        assert rate4.as_percentage() == 0.0

        rate5 = ContaminationRate(0.5)
        assert rate5.as_percentage() == 50.0

    def test_string_representation(self):
        """Test string representation."""
        rate1 = ContaminationRate(0.1)
        assert str(rate1) == "10.0%"

        rate2 = ContaminationRate(0.05)
        assert str(rate2) == "5.0%"

        rate3 = ContaminationRate(0.25)
        assert str(rate3) == "25.0%"

        rate4 = ContaminationRate(0.0)
        assert str(rate4) == "0.0%"

        rate5 = ContaminationRate(0.5)
        assert str(rate5) == "50.0%"

        # Test with decimal places
        rate6 = ContaminationRate(0.125)
        assert str(rate6) == "12.5%"

    def test_class_constants(self):
        """Test class constants."""
        assert ContaminationRate.AUTO.value == 0.1
        assert ContaminationRate.LOW.value == 0.05
        assert ContaminationRate.MEDIUM.value == 0.1
        assert ContaminationRate.HIGH.value == 0.2

        # Test constants are immutable
        with pytest.raises(AttributeError):
            ContaminationRate.AUTO.value = 0.2

    def test_class_methods(self):
        """Test class methods."""
        # Test auto method
        auto_rate = ContaminationRate.auto()
        assert auto_rate.value == 0.1
        assert auto_rate == ContaminationRate.AUTO

        # Test low method
        low_rate = ContaminationRate.low()
        assert low_rate.value == 0.05
        assert low_rate == ContaminationRate.LOW

        # Test medium method
        medium_rate = ContaminationRate.medium()
        assert medium_rate.value == 0.1
        assert medium_rate == ContaminationRate.MEDIUM

        # Test high method
        high_rate = ContaminationRate.high()
        assert high_rate.value == 0.2
        assert high_rate == ContaminationRate.HIGH

    def test_equality_comparison(self):
        """Test equality comparison."""
        rate1 = ContaminationRate(0.1)
        rate2 = ContaminationRate(0.1)
        rate3 = ContaminationRate(0.2)

        assert rate1 == rate2
        assert rate1 != rate3

        # Test with class constants
        assert rate1 == ContaminationRate.AUTO
        assert rate1 == ContaminationRate.MEDIUM

    def test_hash_behavior(self):
        """Test hash behavior for use in sets and dictionaries."""
        rate1 = ContaminationRate(0.1)
        rate2 = ContaminationRate(0.1)
        rate3 = ContaminationRate(0.2)

        # Same values should have same hash
        assert hash(rate1) == hash(rate2)

        # Different values should have different hash
        assert hash(rate1) != hash(rate3)

        # Test in set
        rate_set = {rate1, rate2, rate3}
        assert len(rate_set) == 2  # rate1 and rate2 are equal

        # Test in dictionary
        rate_dict = {rate1: "low", rate3: "high"}
        assert len(rate_dict) == 2
        assert rate_dict[rate2] == "low"  # rate2 equals rate1

    def test_ordering_comparison(self):
        """Test ordering comparison."""
        low_rate = ContaminationRate(0.05)
        medium_rate = ContaminationRate(0.1)
        high_rate = ContaminationRate(0.2)

        assert low_rate < medium_rate
        assert medium_rate < high_rate
        assert low_rate < high_rate

        assert high_rate > medium_rate
        assert medium_rate > low_rate
        assert high_rate > low_rate

        assert low_rate <= medium_rate
        assert medium_rate <= high_rate
        assert low_rate <= low_rate  # equal

        assert high_rate >= medium_rate
        assert medium_rate >= low_rate
        assert high_rate >= high_rate  # equal

    def test_repr_representation(self):
        """Test repr representation."""
        rate = ContaminationRate(0.1)
        repr_str = repr(rate)
        assert "ContaminationRate" in repr_str
        assert "0.1" in repr_str

    def test_contamination_rate_with_precision(self):
        """Test contamination rate with high precision values."""
        precise_rate = ContaminationRate(0.12345)
        assert precise_rate.value == 0.12345
        assert precise_rate.as_percentage() == 12.345

    def test_contamination_rate_arithmetic_operations(self):
        """Test that contamination rate doesn't support arithmetic operations."""
        rate = ContaminationRate(0.1)

        # ContaminationRate is immutable, but we can access the value
        assert abs(rate.value + 0.05 - 0.15) < 1e-10
        assert abs(rate.value * 2 - 0.2) < 1e-10
        assert abs(rate.value / 2 - 0.05) < 1e-10

    def test_contamination_rate_with_integer_values(self):
        """Test contamination rate with integer values."""
        rate = ContaminationRate(0)
        assert rate.value == 0
        assert rate.as_percentage() == 0.0
        assert str(rate) == "0.0%"

    def test_class_constants_uniqueness(self):
        """Test class constants are unique objects."""
        assert ContaminationRate.AUTO is not ContaminationRate.MEDIUM
        assert ContaminationRate.LOW is not ContaminationRate.HIGH

        # But they can be equal in value
        assert ContaminationRate.AUTO == ContaminationRate.MEDIUM

    def test_contamination_rate_edge_cases(self):
        """Test edge cases for contamination rate."""
        # Very small positive value
        tiny_rate = ContaminationRate(1e-10)
        assert tiny_rate.value == 1e-10
        assert tiny_rate.is_valid() is True

        # Close to maximum
        almost_max_rate = ContaminationRate(0.4999999)
        assert almost_max_rate.value == 0.4999999
        assert almost_max_rate.is_valid() is True

    def test_contamination_rate_factory_methods_independence(self):
        """Test factory methods create independent objects."""
        auto1 = ContaminationRate.auto()
        auto2 = ContaminationRate.auto()

        # They should be equal but not the same object
        assert auto1 == auto2
        assert auto1 is not auto2

    def test_contamination_rate_serialization_compatibility(self):
        """Test contamination rate works with serialization."""
        rate = ContaminationRate(0.15)

        # Can extract value for serialization
        serialized_value = rate.value
        assert serialized_value == 0.15

        # Can recreate from serialized value
        deserialized_rate = ContaminationRate(serialized_value)
        assert deserialized_rate == rate

    def test_contamination_rate_validation_error_messages(self):
        """Test specific validation error messages."""
        # Test type error message
        with pytest.raises(InvalidValueError) as exc_info:
            ContaminationRate("invalid")
        assert "must be numeric" in str(exc_info.value)
        assert "str" in str(exc_info.value)

        # Test range error message
        with pytest.raises(InvalidValueError) as exc_info:
            ContaminationRate(0.6)
        assert "must be between 0 and 0.5" in str(exc_info.value)
        assert "0.6" in str(exc_info.value)

    def test_contamination_rate_practical_usage(self):
        """Test practical usage scenarios."""

        # Common usage pattern
        def create_contamination_rate(value: float | None = None) -> ContaminationRate:
            if value is None:
                return ContaminationRate.auto()
            return ContaminationRate(value)

        # Test with None (should use auto)
        rate1 = create_contamination_rate(None)
        assert rate1 == ContaminationRate.AUTO

        # Test with specific value
        rate2 = create_contamination_rate(0.15)
        assert rate2.value == 0.15

        # Test with predefined constants
        rate3 = create_contamination_rate(0.05)
        assert rate3 == ContaminationRate.LOW

    def test_contamination_rate_comparison_with_float(self):
        """Test comparison with float values."""
        rate = ContaminationRate(0.1)

        # Direct value comparison
        assert rate.value == 0.1
        assert rate.value != 0.2
        assert rate.value < 0.2
        assert rate.value > 0.05

    def test_contamination_rate_string_formatting(self):
        """Test string formatting variations."""
        # Test integer-like percentages
        rate1 = ContaminationRate(0.1)
        assert str(rate1) == "10.0%"

        rate2 = ContaminationRate(0.2)
        assert str(rate2) == "20.0%"

        # Test decimal percentages
        rate3 = ContaminationRate(0.125)
        assert str(rate3) == "12.5%"

        rate4 = ContaminationRate(0.033)
        assert str(rate4) == "3.3%"
