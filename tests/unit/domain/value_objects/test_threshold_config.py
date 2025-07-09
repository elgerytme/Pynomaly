"""Tests for threshold config value object."""

import pytest

from pynomaly.domain.exceptions import InvalidValueError
from pynomaly.domain.value_objects.threshold_config import ThresholdConfig


class TestThresholdConfig:
    """Test suite for ThresholdConfig value object."""

    def test_default_creation(self):
        """Test creation with default values."""
        config = ThresholdConfig()
        
        assert config.method == "contamination"
        assert config.value is None
        assert config.auto_adjust is False
        assert config.min_threshold is None
        assert config.max_threshold is None

    def test_custom_creation(self):
        """Test creation with custom values."""
        config = ThresholdConfig(
            method="percentile",
            value=95.0,
            auto_adjust=True,
            min_threshold=0.1,
            max_threshold=0.9
        )
        
        assert config.method == "percentile"
        assert config.value == 95.0
        assert config.auto_adjust is True
        assert config.min_threshold == 0.1
        assert config.max_threshold == 0.9

    def test_immutability(self):
        """Test that threshold config is immutable."""
        config = ThresholdConfig(method="fixed", value=0.5)
        
        # Should not be able to modify values
        with pytest.raises(AttributeError):
            config.method = "percentile"
        
        with pytest.raises(AttributeError):
            config.value = 0.7

    def test_valid_methods(self):
        """Test all valid threshold methods."""
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

    def test_invalid_method(self):
        """Test invalid threshold method."""
        with pytest.raises(InvalidValueError, match="Invalid threshold method: invalid"):
            ThresholdConfig(method="invalid")
        
        with pytest.raises(InvalidValueError, match="Invalid threshold method: zscore"):
            ThresholdConfig(method="zscore")

    def test_percentile_method_validation(self):
        """Test validation for percentile method."""
        # Valid percentile values
        ThresholdConfig(method="percentile", value=0.0)
        ThresholdConfig(method="percentile", value=50.0)
        ThresholdConfig(method="percentile", value=100.0)
        ThresholdConfig(method="percentile", value=95.5)
        
        # Invalid percentile values
        with pytest.raises(InvalidValueError, match="Percentile value must be between 0 and 100"):
            ThresholdConfig(method="percentile", value=-1.0)
        
        with pytest.raises(InvalidValueError, match="Percentile value must be between 0 and 100"):
            ThresholdConfig(method="percentile", value=101.0)

    def test_percentile_method_without_value(self):
        """Test percentile method without value (should be allowed)."""
        config = ThresholdConfig(method="percentile", value=None)
        assert config.method == "percentile"
        assert config.value is None

    def test_non_percentile_method_validation(self):
        """Test that non-percentile methods don't validate value range."""
        # These should not raise errors even with values outside 0-100
        ThresholdConfig(method="fixed", value=1.5)
        ThresholdConfig(method="iqr", value=-0.5)
        ThresholdConfig(method="mad", value=200.0)
        ThresholdConfig(method="adaptive", value=1000.0)
        ThresholdConfig(method="contamination", value=0.1)

    def test_threshold_bounds_validation(self):
        """Test validation of threshold bounds."""
        # Valid bounds
        ThresholdConfig(min_threshold=0.1, max_threshold=0.9)
        ThresholdConfig(min_threshold=0.0, max_threshold=1.0)
        ThresholdConfig(min_threshold=-1.0, max_threshold=1.0)
        
        # Invalid bounds (min >= max)
        with pytest.raises(InvalidValueError, match="min_threshold.*must be less than.*max_threshold"):
            ThresholdConfig(min_threshold=0.9, max_threshold=0.1)
        
        with pytest.raises(InvalidValueError, match="min_threshold.*must be less than.*max_threshold"):
            ThresholdConfig(min_threshold=0.5, max_threshold=0.5)

    def test_threshold_bounds_none_values(self):
        """Test threshold bounds with None values."""
        # Only min_threshold set
        config1 = ThresholdConfig(min_threshold=0.1, max_threshold=None)
        assert config1.min_threshold == 0.1
        assert config1.max_threshold is None
        
        # Only max_threshold set
        config2 = ThresholdConfig(min_threshold=None, max_threshold=0.9)
        assert config2.min_threshold is None
        assert config2.max_threshold == 0.9
        
        # Both None
        config3 = ThresholdConfig(min_threshold=None, max_threshold=None)
        assert config3.min_threshold is None
        assert config3.max_threshold is None

    def test_equality_comparison(self):
        """Test equality comparison."""
        config1 = ThresholdConfig(method="percentile", value=95.0, auto_adjust=True)
        config2 = ThresholdConfig(method="percentile", value=95.0, auto_adjust=True)
        config3 = ThresholdConfig(method="percentile", value=90.0, auto_adjust=True)
        
        assert config1 == config2
        assert config1 != config3

    def test_hash_behavior(self):
        """Test hash behavior for use in sets and dictionaries."""
        config1 = ThresholdConfig(method="percentile", value=95.0, auto_adjust=True)
        config2 = ThresholdConfig(method="percentile", value=95.0, auto_adjust=True)
        config3 = ThresholdConfig(method="fixed", value=0.5, auto_adjust=False)
        
        # Same values should have same hash
        assert hash(config1) == hash(config2)
        
        # Different values should have different hash
        assert hash(config1) != hash(config3)
        
        # Test in set
        config_set = {config1, config2, config3}
        assert len(config_set) == 2  # config1 and config2 are equal

    def test_repr_representation(self):
        """Test repr representation."""
        config = ThresholdConfig(method="percentile", value=95.0, auto_adjust=True)
        repr_str = repr(config)
        assert "ThresholdConfig" in repr_str
        assert "percentile" in repr_str
        assert "95.0" in repr_str
        assert "True" in repr_str

    def test_string_representation(self):
        """Test string representation."""
        config = ThresholdConfig(method="percentile", value=95.0)
        str_repr = str(config)
        # Should contain the dataclass string representation
        assert "ThresholdConfig" in str_repr or "percentile" in str_repr

    def test_method_specific_configurations(self):
        """Test configurations specific to each method."""
        # Percentile method
        percentile_config = ThresholdConfig(method="percentile", value=95.0)
        assert percentile_config.method == "percentile"
        assert percentile_config.value == 95.0
        
        # Fixed method
        fixed_config = ThresholdConfig(method="fixed", value=0.5)
        assert fixed_config.method == "fixed"
        assert fixed_config.value == 0.5
        
        # IQR method
        iqr_config = ThresholdConfig(method="iqr", value=1.5)
        assert iqr_config.method == "iqr"
        assert iqr_config.value == 1.5
        
        # MAD method
        mad_config = ThresholdConfig(method="mad", value=2.0)
        assert mad_config.method == "mad"
        assert mad_config.value == 2.0
        
        # Adaptive method
        adaptive_config = ThresholdConfig(method="adaptive", auto_adjust=True)
        assert adaptive_config.method == "adaptive"
        assert adaptive_config.auto_adjust is True
        
        # Contamination method (default)
        contamination_config = ThresholdConfig(method="contamination", value=0.1)
        assert contamination_config.method == "contamination"
        assert contamination_config.value == 0.1

    def test_auto_adjust_combinations(self):
        """Test auto_adjust with different methods."""
        # Auto adjust enabled
        auto_config = ThresholdConfig(method="adaptive", auto_adjust=True)
        assert auto_config.auto_adjust is True
        
        # Auto adjust disabled
        manual_config = ThresholdConfig(method="fixed", auto_adjust=False)
        assert manual_config.auto_adjust is False
        
        # Auto adjust with bounds
        bounded_auto = ThresholdConfig(
            method="adaptive",
            auto_adjust=True,
            min_threshold=0.1,
            max_threshold=0.9
        )
        assert bounded_auto.auto_adjust is True
        assert bounded_auto.min_threshold == 0.1
        assert bounded_auto.max_threshold == 0.9

    def test_practical_usage_scenarios(self):
        """Test practical usage scenarios."""
        # Scenario 1: Percentile-based thresholding
        percentile_config = ThresholdConfig(
            method="percentile",
            value=95.0,
            auto_adjust=False
        )
        assert percentile_config.method == "percentile"
        assert percentile_config.value == 95.0
        
        # Scenario 2: Fixed threshold with bounds
        fixed_bounded = ThresholdConfig(
            method="fixed",
            value=0.5,
            min_threshold=0.1,
            max_threshold=0.9
        )
        assert fixed_bounded.method == "fixed"
        assert fixed_bounded.value == 0.5
        assert 0.1 <= fixed_bounded.value <= 0.9
        
        # Scenario 3: Adaptive thresholding with auto-adjustment
        adaptive_config = ThresholdConfig(
            method="adaptive",
            auto_adjust=True,
            min_threshold=0.05,
            max_threshold=0.95
        )
        assert adaptive_config.method == "adaptive"
        assert adaptive_config.auto_adjust is True
        
        # Scenario 4: IQR-based outlier detection
        iqr_config = ThresholdConfig(
            method="iqr",
            value=1.5,  # Standard IQR multiplier
            auto_adjust=False
        )
        assert iqr_config.method == "iqr"
        assert iqr_config.value == 1.5

    def test_configuration_validation_edge_cases(self):
        """Test edge cases for configuration validation."""
        # Edge case: Percentile at boundaries
        ThresholdConfig(method="percentile", value=0.0)
        ThresholdConfig(method="percentile", value=100.0)
        
        # Edge case: Very small threshold bounds
        ThresholdConfig(min_threshold=0.0001, max_threshold=0.0002)
        
        # Edge case: Large threshold bounds
        ThresholdConfig(min_threshold=-1000.0, max_threshold=1000.0)
        
        # Edge case: Negative values for non-percentile methods
        ThresholdConfig(method="fixed", value=-0.5)
        ThresholdConfig(method="iqr", value=-1.0)

    def test_method_case_sensitivity(self):
        """Test that method names are case sensitive."""
        # Valid lowercase method
        ThresholdConfig(method="percentile")
        
        # Invalid uppercase method
        with pytest.raises(InvalidValueError, match="Invalid threshold method: PERCENTILE"):
            ThresholdConfig(method="PERCENTILE")
        
        # Invalid mixed case method
        with pytest.raises(InvalidValueError, match="Invalid threshold method: Percentile"):
            ThresholdConfig(method="Percentile")

    def test_default_configuration_factory(self):
        """Test creating default configurations for different methods."""
        def create_percentile_config() -> ThresholdConfig:
            return ThresholdConfig(method="percentile", value=95.0)
        
        def create_fixed_config() -> ThresholdConfig:
            return ThresholdConfig(method="fixed", value=0.5)
        
        def create_adaptive_config() -> ThresholdConfig:
            return ThresholdConfig(
                method="adaptive",
                auto_adjust=True,
                min_threshold=0.05,
                max_threshold=0.95
            )
        
        # Test factory functions
        perc_config = create_percentile_config()
        assert perc_config.method == "percentile"
        assert perc_config.value == 95.0
        
        fixed_config = create_fixed_config()
        assert fixed_config.method == "fixed"
        assert fixed_config.value == 0.5
        
        adaptive_config = create_adaptive_config()
        assert adaptive_config.method == "adaptive"
        assert adaptive_config.auto_adjust is True

    def test_configuration_with_all_parameters(self):
        """Test configuration with all parameters set."""
        config = ThresholdConfig(
            method="percentile",
            value=90.0,
            auto_adjust=True,
            min_threshold=0.05,
            max_threshold=0.95
        )
        
        assert config.method == "percentile"
        assert config.value == 90.0
        assert config.auto_adjust is True
        assert config.min_threshold == 0.05
        assert config.max_threshold == 0.95

    def test_configuration_serialization_compatibility(self):
        """Test configuration works with serialization."""
        config = ThresholdConfig(
            method="percentile",
            value=95.0,
            auto_adjust=True,
            min_threshold=0.1,
            max_threshold=0.9
        )
        
        # Can extract values for serialization
        config_dict = {
            "method": config.method,
            "value": config.value,
            "auto_adjust": config.auto_adjust,
            "min_threshold": config.min_threshold,
            "max_threshold": config.max_threshold,
        }
        
        assert config_dict["method"] == "percentile"
        assert config_dict["value"] == 95.0
        assert config_dict["auto_adjust"] is True
        assert config_dict["min_threshold"] == 0.1
        assert config_dict["max_threshold"] == 0.9
        
        # Can recreate from serialized values
        recreated_config = ThresholdConfig(**config_dict)
        assert recreated_config == config

    def test_validation_error_messages(self):
        """Test specific validation error messages."""
        # Test method validation error
        with pytest.raises(InvalidValueError) as exc_info:
            ThresholdConfig(method="invalid_method")
        assert "Invalid threshold method: invalid_method" in str(exc_info.value)
        
        # Test percentile validation error
        with pytest.raises(InvalidValueError) as exc_info:
            ThresholdConfig(method="percentile", value=150.0)
        assert "Percentile value must be between 0 and 100" in str(exc_info.value)
        assert "150.0" in str(exc_info.value)
        
        # Test threshold bounds validation error
        with pytest.raises(InvalidValueError) as exc_info:
            ThresholdConfig(min_threshold=0.8, max_threshold=0.2)
        assert "min_threshold (0.8) must be less than max_threshold (0.2)" in str(exc_info.value)

    def test_dataclass_properties(self):
        """Test dataclass properties."""
        config = ThresholdConfig(method="percentile", value=95.0)
        
        # Should have dataclass properties
        assert hasattr(config, '__dataclass_fields__')
        assert 'method' in config.__dataclass_fields__
        assert 'value' in config.__dataclass_fields__
        assert 'auto_adjust' in config.__dataclass_fields__
        assert 'min_threshold' in config.__dataclass_fields__
        assert 'max_threshold' in config.__dataclass_fields__

    def test_method_compatibility_matrix(self):
        """Test method compatibility with different parameter combinations."""
        # Test which methods work with which parameters
        test_cases = [
            ("percentile", {"value": 95.0}, True),
            ("percentile", {"value": 150.0}, False),  # Invalid range
            ("fixed", {"value": 0.5}, True),
            ("fixed", {"value": -0.5}, True),  # Negative allowed for non-percentile
            ("iqr", {"value": 1.5}, True),
            ("mad", {"value": 2.0}, True),
            ("adaptive", {"auto_adjust": True}, True),
            ("contamination", {"value": 0.1}, True),
        ]
        
        for method, params, should_succeed in test_cases:
            if should_succeed:
                config = ThresholdConfig(method=method, **params)
                assert config.method == method
                for key, value in params.items():
                    assert getattr(config, key) == value
            else:
                with pytest.raises(InvalidValueError):
                    ThresholdConfig(method=method, **params)

    def test_configuration_immutability_comprehensive(self):
        """Test comprehensive immutability of configuration."""
        config = ThresholdConfig(
            method="percentile",
            value=95.0,
            auto_adjust=True,
            min_threshold=0.1,
            max_threshold=0.9
        )
        
        # All attributes should be immutable
        with pytest.raises(AttributeError):
            config.method = "fixed"
        
        with pytest.raises(AttributeError):
            config.value = 90.0
        
        with pytest.raises(AttributeError):
            config.auto_adjust = False
        
        with pytest.raises(AttributeError):
            config.min_threshold = 0.2
        
        with pytest.raises(AttributeError):
            config.max_threshold = 0.8