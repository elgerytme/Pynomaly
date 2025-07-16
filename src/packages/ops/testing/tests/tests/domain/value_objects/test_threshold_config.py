"""
Comprehensive tests for ThresholdConfig value object.

This module tests the ThresholdConfig value object to ensure proper validation,
behavior, and immutability across all use cases.
"""

import pytest

from pynomaly.domain.exceptions import InvalidValueError
from pynomaly.domain.value_objects import ThresholdConfig


class TestThresholdConfigCreation:
    """Test ThresholdConfig creation and validation."""

    def test_default_creation(self):
        """Test default threshold config creation."""
        config = ThresholdConfig()

        assert config.method == "contamination"
        assert config.value is None
        assert config.auto_adjust is False
        assert config.min_threshold is None
        assert config.max_threshold is None

    def test_creation_with_all_parameters(self):
        """Test creation with all parameters."""
        config = ThresholdConfig(
            method="percentile",
            value=95.0,
            auto_adjust=True,
            min_threshold=0.1,
            max_threshold=0.9,
        )

        assert config.method == "percentile"
        assert config.value == 95.0
        assert config.auto_adjust is True
        assert config.min_threshold == 0.1
        assert config.max_threshold == 0.9

    def test_creation_with_partial_parameters(self):
        """Test creation with partial parameters."""
        config = ThresholdConfig(method="fixed", value=0.5)

        assert config.method == "fixed"
        assert config.value == 0.5
        assert config.auto_adjust is False  # default
        assert config.min_threshold is None  # default
        assert config.max_threshold is None  # default


class TestThresholdConfigValidation:
    """Test validation of ThresholdConfig parameters."""

    def test_valid_methods(self):
        """Test valid threshold methods."""
        valid_methods = [
            "percentile",
            "fixed",
            "iqr",
            "mad",
            "adaptive",
            "contamination",
        ]

        for method in valid_methods:
            config = ThresholdConfig(method=method)
            assert config.method == method

    def test_invalid_methods(self):
        """Test invalid threshold methods."""
        invalid_methods = [
            "invalid",
            "unknown",
            "custom",
            "",
            "PERCENTILE",  # case sensitive
            "Fixed",  # case sensitive
        ]

        for method in invalid_methods:
            with pytest.raises(
                InvalidValueError, match=f"Invalid threshold method: {method}"
            ):
                ThresholdConfig(method=method)

    def test_percentile_value_validation(self):
        """Test percentile value validation."""
        # Valid percentile values
        valid_percentiles = [0, 25, 50, 75, 90, 95, 99, 100]

        for percentile in valid_percentiles:
            config = ThresholdConfig(method="percentile", value=percentile)
            assert config.value == percentile

    def test_invalid_percentile_values(self):
        """Test invalid percentile values."""
        invalid_percentiles = [-1, -10, 101, 150, 200]

        for percentile in invalid_percentiles:
            with pytest.raises(
                InvalidValueError,
                match=f"Percentile value must be between 0 and 100, got {percentile}",
            ):
                ThresholdConfig(method="percentile", value=percentile)

    def test_percentile_value_with_other_methods(self):
        """Test that percentile validation only applies to percentile method."""
        # Values that would be invalid for percentile but valid for other methods
        other_methods = ["fixed", "iqr", "mad", "adaptive", "contamination"]

        for method in other_methods:
            # Should not raise error for non-percentile methods
            config = ThresholdConfig(method=method, value=150)
            assert config.value == 150

    def test_min_max_threshold_validation(self):
        """Test min/max threshold validation."""
        # Valid min/max combinations
        valid_combinations = [
            (0.1, 0.9),
            (0.0, 1.0),
            (0.3, 0.7),
            (0.01, 0.99),
        ]

        for min_val, max_val in valid_combinations:
            config = ThresholdConfig(min_threshold=min_val, max_threshold=max_val)
            assert config.min_threshold == min_val
            assert config.max_threshold == max_val

    def test_invalid_min_max_threshold_combinations(self):
        """Test invalid min/max threshold combinations."""
        invalid_combinations = [
            (0.9, 0.1),  # min > max
            (0.5, 0.5),  # min == max
            (0.7, 0.3),  # min > max
        ]

        for min_val, max_val in invalid_combinations:
            with pytest.raises(
                InvalidValueError,
                match=f"min_threshold \\({min_val}\\) must be less than max_threshold \\({max_val}\\)",
            ):
                ThresholdConfig(min_threshold=min_val, max_threshold=max_val)

    def test_min_threshold_only(self):
        """Test setting only min_threshold."""
        config = ThresholdConfig(min_threshold=0.1)

        assert config.min_threshold == 0.1
        assert config.max_threshold is None

    def test_max_threshold_only(self):
        """Test setting only max_threshold."""
        config = ThresholdConfig(max_threshold=0.9)

        assert config.min_threshold is None
        assert config.max_threshold == 0.9


class TestThresholdConfigImmutability:
    """Test ThresholdConfig immutability."""

    def test_frozen_dataclass(self):
        """Test that ThresholdConfig is frozen."""
        config = ThresholdConfig(method="fixed", value=0.5)

        with pytest.raises(AttributeError):
            config.method = "percentile"

        with pytest.raises(AttributeError):
            config.value = 0.8

        with pytest.raises(AttributeError):
            config.auto_adjust = True

    def test_hash_consistency(self):
        """Test that equal configs have equal hashes."""
        config1 = ThresholdConfig(method="fixed", value=0.5, auto_adjust=True)
        config2 = ThresholdConfig(method="fixed", value=0.5, auto_adjust=True)
        config3 = ThresholdConfig(method="fixed", value=0.6, auto_adjust=True)

        assert hash(config1) == hash(config2)
        assert hash(config1) != hash(config3)

    def test_use_in_sets(self):
        """Test using threshold configs in sets."""
        config1 = ThresholdConfig(method="fixed", value=0.5)
        config2 = ThresholdConfig(method="percentile", value=95)
        config3 = ThresholdConfig(method="fixed", value=0.5)  # Same as config1

        config_set = {config1, config2, config3}

        assert len(config_set) == 2  # config1 and config3 are the same
        assert config1 in config_set
        assert config2 in config_set

    def test_use_as_dict_keys(self):
        """Test using threshold configs as dictionary keys."""
        config1 = ThresholdConfig(method="fixed", value=0.5)
        config2 = ThresholdConfig(method="percentile", value=95)

        config_dict = {config1: "fixed_threshold", config2: "percentile_threshold"}

        assert len(config_dict) == 2
        assert config_dict[config1] == "fixed_threshold"
        assert config_dict[config2] == "percentile_threshold"


class TestThresholdConfigEquality:
    """Test ThresholdConfig equality and comparison."""

    def test_equality_same_configs(self):
        """Test equality of identical configs."""
        config1 = ThresholdConfig(
            method="percentile",
            value=95,
            auto_adjust=True,
            min_threshold=0.1,
            max_threshold=0.9,
        )
        config2 = ThresholdConfig(
            method="percentile",
            value=95,
            auto_adjust=True,
            min_threshold=0.1,
            max_threshold=0.9,
        )

        assert config1 == config2

    def test_inequality_different_methods(self):
        """Test inequality of different methods."""
        config1 = ThresholdConfig(method="fixed")
        config2 = ThresholdConfig(method="percentile")

        assert config1 != config2

    def test_inequality_different_values(self):
        """Test inequality of different values."""
        config1 = ThresholdConfig(method="fixed", value=0.5)
        config2 = ThresholdConfig(method="fixed", value=0.6)

        assert config1 != config2

    def test_inequality_different_auto_adjust(self):
        """Test inequality of different auto_adjust settings."""
        config1 = ThresholdConfig(auto_adjust=True)
        config2 = ThresholdConfig(auto_adjust=False)

        assert config1 != config2

    def test_inequality_different_min_threshold(self):
        """Test inequality of different min_threshold values."""
        config1 = ThresholdConfig(min_threshold=0.1)
        config2 = ThresholdConfig(min_threshold=0.2)

        assert config1 != config2

    def test_inequality_different_max_threshold(self):
        """Test inequality of different max_threshold values."""
        config1 = ThresholdConfig(max_threshold=0.8)
        config2 = ThresholdConfig(max_threshold=0.9)

        assert config1 != config2

    def test_equality_with_non_config(self):
        """Test equality with non-ThresholdConfig objects."""
        config = ThresholdConfig()

        assert config != "contamination"
        assert config != {"method": "contamination"}
        assert config != None
        assert config != 0.5


class TestThresholdConfigRepresentation:
    """Test ThresholdConfig representation methods."""

    def test_str_representation(self):
        """Test string representation."""
        config = ThresholdConfig(method="fixed", value=0.5)

        str_repr = str(config)

        assert "ThresholdConfig" in str_repr
        assert "fixed" in str_repr
        assert "0.5" in str_repr

    def test_repr_representation(self):
        """Test detailed representation."""
        config = ThresholdConfig(
            method="percentile",
            value=95,
            auto_adjust=True,
            min_threshold=0.1,
            max_threshold=0.9,
        )

        repr_str = repr(config)

        assert "ThresholdConfig" in repr_str
        assert "percentile" in repr_str
        assert "95" in repr_str
        assert "True" in repr_str
        assert "0.1" in repr_str
        assert "0.9" in repr_str


class TestThresholdConfigEdgeCases:
    """Test ThresholdConfig edge cases and boundary conditions."""

    def test_boundary_percentile_values(self):
        """Test boundary percentile values."""
        # Test exact boundaries
        config_min = ThresholdConfig(method="percentile", value=0)
        config_max = ThresholdConfig(method="percentile", value=100)

        assert config_min.value == 0
        assert config_max.value == 100

    def test_floating_point_percentiles(self):
        """Test floating point percentile values."""
        float_percentiles = [0.0, 25.5, 50.1, 75.9, 99.9, 100.0]

        for percentile in float_percentiles:
            config = ThresholdConfig(method="percentile", value=percentile)
            assert config.value == percentile

    def test_extreme_threshold_ranges(self):
        """Test extreme but valid threshold ranges."""
        # Very narrow range
        config_narrow = ThresholdConfig(min_threshold=0.499, max_threshold=0.501)
        assert config_narrow.min_threshold == 0.499
        assert config_narrow.max_threshold == 0.501

        # Very wide range
        config_wide = ThresholdConfig(min_threshold=0.001, max_threshold=0.999)
        assert config_wide.min_threshold == 0.001
        assert config_wide.max_threshold == 0.999

    def test_zero_values(self):
        """Test with zero values."""
        config = ThresholdConfig(method="fixed", value=0.0, min_threshold=0.0)

        assert config.value == 0.0
        assert config.min_threshold == 0.0

    def test_negative_values_for_non_percentile(self):
        """Test negative values for non-percentile methods."""
        # Negative values should be allowed for non-percentile methods
        config = ThresholdConfig(method="fixed", value=-0.5)
        assert config.value == -0.5

        config = ThresholdConfig(method="iqr", value=-2.0)
        assert config.value == -2.0

    def test_very_large_values(self):
        """Test very large values for non-percentile methods."""
        large_value = 1000000.0
        config = ThresholdConfig(method="adaptive", value=large_value)
        assert config.value == large_value


class TestThresholdConfigUsagePatterns:
    """Test common usage patterns for ThresholdConfig."""

    def test_contamination_config(self):
        """Test typical contamination-based configuration."""
        config = ThresholdConfig()  # Default is contamination

        assert config.method == "contamination"
        assert config.value is None  # Usually determined automatically

    def test_fixed_threshold_config(self):
        """Test fixed threshold configuration."""
        config = ThresholdConfig(method="fixed", value=0.5)

        assert config.method == "fixed"
        assert config.value == 0.5

    def test_percentile_config(self):
        """Test percentile-based configuration."""
        config = ThresholdConfig(method="percentile", value=95)

        assert config.method == "percentile"
        assert config.value == 95

    def test_adaptive_config_with_bounds(self):
        """Test adaptive configuration with bounds."""
        config = ThresholdConfig(
            method="adaptive",
            auto_adjust=True,
            min_threshold=0.1,
            max_threshold=0.9,
        )

        assert config.method == "adaptive"
        assert config.auto_adjust is True
        assert config.min_threshold == 0.1
        assert config.max_threshold == 0.9

    def test_iqr_config(self):
        """Test IQR-based configuration."""
        config = ThresholdConfig(method="iqr", value=1.5)

        assert config.method == "iqr"
        assert config.value == 1.5

    def test_mad_config(self):
        """Test MAD-based configuration."""
        config = ThresholdConfig(method="mad", value=3.0)

        assert config.method == "mad"
        assert config.value == 3.0

    def test_auto_adjust_scenarios(self):
        """Test various auto-adjust scenarios."""
        # Auto-adjust without bounds
        config1 = ThresholdConfig(method="adaptive", auto_adjust=True)
        assert config1.auto_adjust is True
        assert config1.min_threshold is None
        assert config1.max_threshold is None

        # Auto-adjust with min bound only
        config2 = ThresholdConfig(
            method="adaptive", auto_adjust=True, min_threshold=0.1
        )
        assert config2.auto_adjust is True
        assert config2.min_threshold == 0.1
        assert config2.max_threshold is None

        # Auto-adjust with max bound only
        config3 = ThresholdConfig(
            method="adaptive", auto_adjust=True, max_threshold=0.9
        )
        assert config3.auto_adjust is True
        assert config3.min_threshold is None
        assert config3.max_threshold == 0.9


class TestThresholdConfigSpecialCases:
    """Test special cases and method-specific validations."""

    def test_method_case_sensitivity(self):
        """Test that method names are case sensitive."""
        valid_method = "percentile"
        invalid_methods = ["Percentile", "PERCENTILE", "Percentile"]

        # Valid method should work
        config = ThresholdConfig(method=valid_method)
        assert config.method == valid_method

        # Invalid cases should fail
        for invalid_method in invalid_methods:
            with pytest.raises(InvalidValueError):
                ThresholdConfig(method=invalid_method)

    def test_none_values_handling(self):
        """Test handling of None values."""
        config = ThresholdConfig(
            method="contamination",
            value=None,
            min_threshold=None,
            max_threshold=None,
        )

        assert config.value is None
        assert config.min_threshold is None
        assert config.max_threshold is None

    def test_boolean_parameter_types(self):
        """Test boolean parameter handling."""
        # Test explicit True/False
        config_true = ThresholdConfig(auto_adjust=True)
        config_false = ThresholdConfig(auto_adjust=False)

        assert config_true.auto_adjust is True
        assert config_false.auto_adjust is False

        # Test that other "truthy" values don't work
        with pytest.raises(TypeError):
            ThresholdConfig(auto_adjust=1)  # type: ignore

        with pytest.raises(TypeError):
            ThresholdConfig(auto_adjust="true")  # type: ignore

    def test_numeric_type_flexibility(self):
        """Test that numeric parameters accept int and float."""
        # Test with integers
        config_int = ThresholdConfig(
            method="percentile",
            value=95,  # int
            min_threshold=0,  # int
            max_threshold=1,  # int
        )

        assert config_int.value == 95
        assert config_int.min_threshold == 0
        assert config_int.max_threshold == 1

        # Test with floats
        config_float = ThresholdConfig(
            method="percentile",
            value=95.5,  # float
            min_threshold=0.1,  # float
            max_threshold=0.9,  # float
        )

        assert config_float.value == 95.5
        assert config_float.min_threshold == 0.1
        assert config_float.max_threshold == 0.9


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
