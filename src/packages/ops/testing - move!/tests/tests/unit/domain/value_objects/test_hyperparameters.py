"""Tests for hyperparameters value objects."""

import json

import pytest

from monorepo.domain.value_objects.hyperparameters import (
    DistributionType,
    HyperparameterRange,
    HyperparameterSet,
    HyperparameterSpace,
    ParameterType,
    boolean_parameter,
    categorical_parameter,
    float_parameter,
    int_parameter,
)


class TestParameterType:
    """Test suite for ParameterType enum."""

    def test_enum_values(self):
        """Test all enum values are correctly defined."""
        assert ParameterType.CATEGORICAL == "categorical"
        assert ParameterType.FLOAT == "float"
        assert ParameterType.INT == "int"
        assert ParameterType.DISCRETE == "discrete"
        assert ParameterType.BOOLEAN == "boolean"

    def test_enum_count(self):
        """Test enum has correct number of values."""
        assert len(ParameterType) == 5


class TestDistributionType:
    """Test suite for DistributionType enum."""

    def test_enum_values(self):
        """Test all enum values are correctly defined."""
        assert DistributionType.UNIFORM == "uniform"
        assert DistributionType.LOG_UNIFORM == "log_uniform"
        assert DistributionType.NORMAL == "normal"
        assert DistributionType.LOG_NORMAL == "log_normal"
        assert DistributionType.DISCRETE_UNIFORM == "discrete_uniform"

    def test_enum_count(self):
        """Test enum has correct number of values."""
        assert len(DistributionType) == 5


class TestHyperparameterRange:
    """Test suite for HyperparameterRange value object."""

    def test_categorical_parameter_creation(self):
        """Test creation of categorical parameter."""
        param = HyperparameterRange(
            name="algorithm",
            type=ParameterType.CATEGORICAL,
            choices=["rf", "svm", "xgb"],
            default_value="rf",
            description="Algorithm choice",
        )

        assert param.name == "algorithm"
        assert param.type == ParameterType.CATEGORICAL
        assert param.choices == ["rf", "svm", "xgb"]
        assert param.default_value == "rf"
        assert param.description == "Algorithm choice"

    def test_float_parameter_creation(self):
        """Test creation of float parameter."""
        param = HyperparameterRange(
            name="learning_rate",
            type=ParameterType.FLOAT,
            low=0.01,
            high=1.0,
            log=True,
            distribution=DistributionType.LOG_UNIFORM,
            default_value=0.1,
        )

        assert param.name == "learning_rate"
        assert param.type == ParameterType.FLOAT
        assert param.low == 0.01
        assert param.high == 1.0
        assert param.log is True
        assert param.distribution == DistributionType.LOG_UNIFORM
        assert param.default_value == 0.1

    def test_int_parameter_creation(self):
        """Test creation of integer parameter."""
        param = HyperparameterRange(
            name="n_estimators",
            type=ParameterType.INT,
            low=10,
            high=1000,
            step=10,
            default_value=100,
        )

        assert param.name == "n_estimators"
        assert param.type == ParameterType.INT
        assert param.low == 10
        assert param.high == 1000
        assert param.step == 10
        assert param.default_value == 100

    def test_boolean_parameter_creation(self):
        """Test creation of boolean parameter."""
        param = HyperparameterRange(
            name="bootstrap", type=ParameterType.BOOLEAN, default_value=True
        )

        assert param.name == "bootstrap"
        assert param.type == ParameterType.BOOLEAN
        assert param.choices == [True, False]
        assert param.default_value is True

    def test_immutability(self):
        """Test that hyperparameter range is immutable."""
        param = HyperparameterRange(
            name="test", type=ParameterType.FLOAT, low=0.0, high=1.0
        )

        # Should not be able to modify values
        with pytest.raises(AttributeError):
            param.name = "new_name"

    def test_validation_categorical_without_choices(self):
        """Test validation for categorical parameter without choices."""
        with pytest.raises(
            ValueError, match="Choices required for categorical parameter"
        ):
            HyperparameterRange(name="test", type=ParameterType.CATEGORICAL)

    def test_validation_discrete_without_choices(self):
        """Test validation for discrete parameter without choices."""
        with pytest.raises(ValueError, match="Choices required for discrete parameter"):
            HyperparameterRange(name="test", type=ParameterType.DISCRETE)

    def test_validation_float_without_bounds(self):
        """Test validation for float parameter without bounds."""
        with pytest.raises(
            ValueError, match="Low and high bounds required for float parameter"
        ):
            HyperparameterRange(name="test", type=ParameterType.FLOAT)

    def test_validation_int_without_bounds(self):
        """Test validation for integer parameter without bounds."""
        with pytest.raises(
            ValueError, match="Low and high bounds required for int parameter"
        ):
            HyperparameterRange(name="test", type=ParameterType.INT)

    def test_validation_invalid_bounds(self):
        """Test validation for invalid bounds."""
        with pytest.raises(ValueError, match="Low bound must be less than high bound"):
            HyperparameterRange(
                name="test", type=ParameterType.FLOAT, low=1.0, high=0.0
            )

    def test_boolean_parameter_auto_choices(self):
        """Test boolean parameter automatically gets choices."""
        param = HyperparameterRange(name="test", type=ParameterType.BOOLEAN)

        assert param.choices == [True, False]

    def test_is_valid_value_categorical(self):
        """Test is_valid_value for categorical parameters."""
        param = HyperparameterRange(
            name="test", type=ParameterType.CATEGORICAL, choices=["a", "b", "c"]
        )

        assert param.is_valid_value("a") is True
        assert param.is_valid_value("b") is True
        assert param.is_valid_value("d") is False
        assert param.is_valid_value(1) is False

    def test_is_valid_value_float(self):
        """Test is_valid_value for float parameters."""
        param = HyperparameterRange(
            name="test", type=ParameterType.FLOAT, low=0.0, high=1.0
        )

        assert param.is_valid_value(0.5) is True
        assert param.is_valid_value(0.0) is True
        assert param.is_valid_value(1.0) is True
        assert param.is_valid_value(-0.1) is False
        assert param.is_valid_value(1.1) is False
        assert param.is_valid_value("0.5") is False

    def test_is_valid_value_int(self):
        """Test is_valid_value for integer parameters."""
        param = HyperparameterRange(name="test", type=ParameterType.INT, low=1, high=10)

        assert param.is_valid_value(5) is True
        assert param.is_valid_value(1) is True
        assert param.is_valid_value(10) is True
        assert param.is_valid_value(0) is False
        assert param.is_valid_value(11) is False
        assert param.is_valid_value(5.5) is False

    def test_is_valid_value_boolean(self):
        """Test is_valid_value for boolean parameters."""
        param = HyperparameterRange(name="test", type=ParameterType.BOOLEAN)

        assert param.is_valid_value(True) is True
        assert param.is_valid_value(False) is True
        assert param.is_valid_value(1) is False
        assert param.is_valid_value(0) is False
        assert param.is_valid_value("true") is False

    def test_get_grid_values_categorical(self):
        """Test get_grid_values for categorical parameters."""
        param = HyperparameterRange(
            name="test", type=ParameterType.CATEGORICAL, choices=["a", "b", "c"]
        )

        grid_values = param.get_grid_values()
        assert grid_values == ["a", "b", "c"]

    def test_get_grid_values_float(self):
        """Test get_grid_values for float parameters."""
        param = HyperparameterRange(
            name="test", type=ParameterType.FLOAT, low=0.0, high=1.0
        )

        grid_values = param.get_grid_values(size=5)
        assert len(grid_values) == 5
        assert grid_values[0] == 0.0
        assert grid_values[-1] == 1.0

    def test_get_grid_values_float_log(self):
        """Test get_grid_values for log-scale float parameters."""
        param = HyperparameterRange(
            name="test", type=ParameterType.FLOAT, low=0.01, high=1.0, log=True
        )

        grid_values = param.get_grid_values(size=3)
        assert len(grid_values) == 3
        assert grid_values[0] == 0.01
        assert grid_values[-1] == 1.0

    def test_get_grid_values_int(self):
        """Test get_grid_values for integer parameters."""
        param = HyperparameterRange(name="test", type=ParameterType.INT, low=1, high=10)

        grid_values = param.get_grid_values(size=5)
        assert len(grid_values) == 5
        assert all(isinstance(v, int) for v in grid_values)
        assert grid_values[0] == 1
        assert grid_values[-1] == 10

    def test_sample_value_categorical(self):
        """Test sample_value for categorical parameters."""
        param = HyperparameterRange(
            name="test", type=ParameterType.CATEGORICAL, choices=["a", "b", "c"]
        )

        # Test with fixed random state
        value = param.sample_value(random_state=42)
        assert value in ["a", "b", "c"]

        # Test multiple samples
        values = [param.sample_value(random_state=i) for i in range(10)]
        assert all(v in ["a", "b", "c"] for v in values)

    def test_sample_value_float(self):
        """Test sample_value for float parameters."""
        param = HyperparameterRange(
            name="test", type=ParameterType.FLOAT, low=0.0, high=1.0
        )

        # Test with fixed random state
        value = param.sample_value(random_state=42)
        assert 0.0 <= value <= 1.0

        # Test multiple samples
        values = [param.sample_value(random_state=i) for i in range(10)]
        assert all(0.0 <= v <= 1.0 for v in values)

    def test_sample_value_float_log(self):
        """Test sample_value for log-scale float parameters."""
        param = HyperparameterRange(
            name="test", type=ParameterType.FLOAT, low=0.01, high=1.0, log=True
        )

        value = param.sample_value(random_state=42)
        assert 0.01 <= value <= 1.0

    def test_sample_value_int(self):
        """Test sample_value for integer parameters."""
        param = HyperparameterRange(name="test", type=ParameterType.INT, low=1, high=10)

        # Test with fixed random state
        value = param.sample_value(random_state=42)
        assert 1 <= value <= 10
        assert isinstance(value, int)

    def test_to_dict_method(self):
        """Test to_dict method."""
        param = HyperparameterRange(
            name="test",
            type=ParameterType.FLOAT,
            low=0.0,
            high=1.0,
            distribution=DistributionType.UNIFORM,
            default_value=0.5,
            description="Test parameter",
        )

        result = param.to_dict()
        expected = {
            "name": "test",
            "type": "float",
            "low": 0.0,
            "high": 1.0,
            "choices": None,
            "distribution": "uniform",
            "log": False,
            "step": None,
            "grid_size": None,
            "default_value": 0.5,
            "description": "Test parameter",
        }

        assert result == expected

    def test_from_dict_method(self):
        """Test from_dict method."""
        data = {
            "name": "test",
            "type": "float",
            "low": 0.0,
            "high": 1.0,
            "distribution": "uniform",
            "default_value": 0.5,
        }

        param = HyperparameterRange.from_dict(data)
        assert param.name == "test"
        assert param.type == ParameterType.FLOAT
        assert param.low == 0.0
        assert param.high == 1.0
        assert param.distribution == DistributionType.UNIFORM
        assert param.default_value == 0.5


class TestHyperparameterSpace:
    """Test suite for HyperparameterSpace value object."""

    def test_basic_creation(self):
        """Test basic creation of hyperparameter space."""
        param1 = HyperparameterRange(
            name="param1", type=ParameterType.FLOAT, low=0.0, high=1.0
        )

        param2 = HyperparameterRange(
            name="param2", type=ParameterType.CATEGORICAL, choices=["a", "b", "c"]
        )

        space = HyperparameterSpace(
            parameters={"param1": param1, "param2": param2},
            name="test_space",
            description="Test space",
        )

        assert space.name == "test_space"
        assert space.description == "Test space"
        assert len(space.parameters) == 2
        assert "param1" in space.parameters
        assert "param2" in space.parameters

    def test_immutability(self):
        """Test that hyperparameter space is immutable."""
        param = HyperparameterRange(
            name="test", type=ParameterType.FLOAT, low=0.0, high=1.0
        )

        space = HyperparameterSpace(parameters={"test": param})

        # Should not be able to modify values
        with pytest.raises(AttributeError):
            space.name = "new_name"

    def test_validation_empty_parameters(self):
        """Test validation for empty parameters."""
        with pytest.raises(ValueError, match="must contain at least one parameter"):
            HyperparameterSpace(parameters={})

    def test_validation_name_mismatch(self):
        """Test validation for parameter name mismatch."""
        param = HyperparameterRange(
            name="param1", type=ParameterType.FLOAT, low=0.0, high=1.0
        )

        with pytest.raises(ValueError, match="Parameter name mismatch"):
            HyperparameterSpace(parameters={"param2": param})

    def test_add_parameter_method(self):
        """Test add_parameter method."""
        param1 = HyperparameterRange(
            name="param1", type=ParameterType.FLOAT, low=0.0, high=1.0
        )

        param2 = HyperparameterRange(
            name="param2", type=ParameterType.CATEGORICAL, choices=["a", "b"]
        )

        space = HyperparameterSpace(parameters={"param1": param1})
        new_space = space.add_parameter(param2)

        # Original space should be unchanged
        assert len(space.parameters) == 1
        assert "param2" not in space.parameters

        # New space should have both parameters
        assert len(new_space.parameters) == 2
        assert "param1" in new_space.parameters
        assert "param2" in new_space.parameters

    def test_remove_parameter_method(self):
        """Test remove_parameter method."""
        param1 = HyperparameterRange(
            name="param1", type=ParameterType.FLOAT, low=0.0, high=1.0
        )

        param2 = HyperparameterRange(
            name="param2", type=ParameterType.CATEGORICAL, choices=["a", "b"]
        )

        space = HyperparameterSpace(parameters={"param1": param1, "param2": param2})
        new_space = space.remove_parameter("param2")

        # Original space should be unchanged
        assert len(space.parameters) == 2
        assert "param2" in space.parameters

        # New space should have only param1
        assert len(new_space.parameters) == 1
        assert "param1" in new_space.parameters
        assert "param2" not in new_space.parameters

    def test_get_parameter_method(self):
        """Test get_parameter method."""
        param = HyperparameterRange(
            name="test", type=ParameterType.FLOAT, low=0.0, high=1.0
        )

        space = HyperparameterSpace(parameters={"test": param})

        assert space.get_parameter("test") == param
        assert space.get_parameter("nonexistent") is None

    def test_validate_values_method(self):
        """Test validate_values method."""
        param1 = HyperparameterRange(
            name="param1", type=ParameterType.FLOAT, low=0.0, high=1.0
        )

        param2 = HyperparameterRange(
            name="param2", type=ParameterType.CATEGORICAL, choices=["a", "b"]
        )

        space = HyperparameterSpace(parameters={"param1": param1, "param2": param2})

        # Valid values
        valid_values = {"param1": 0.5, "param2": "a"}
        validation_results = space.validate_values(valid_values)
        assert validation_results["param1"] is True
        assert validation_results["param2"] is True

        # Invalid values
        invalid_values = {"param1": 1.5, "param2": "c"}
        validation_results = space.validate_values(invalid_values)
        assert validation_results["param1"] is False
        assert validation_results["param2"] is False

        # Missing values
        missing_values = {"param1": 0.5}
        validation_results = space.validate_values(missing_values)
        assert validation_results["param1"] is True
        assert validation_results["param2"] is False

    def test_are_valid_values_method(self):
        """Test are_valid_values method."""
        param1 = HyperparameterRange(
            name="param1", type=ParameterType.FLOAT, low=0.0, high=1.0
        )

        param2 = HyperparameterRange(
            name="param2", type=ParameterType.CATEGORICAL, choices=["a", "b"]
        )

        space = HyperparameterSpace(parameters={"param1": param1, "param2": param2})

        # Valid values
        valid_values = {"param1": 0.5, "param2": "a"}
        assert space.are_valid_values(valid_values) is True

        # Invalid values
        invalid_values = {"param1": 1.5, "param2": "a"}
        assert space.are_valid_values(invalid_values) is False

        # Missing values
        missing_values = {"param1": 0.5}
        assert space.are_valid_values(missing_values) is False

    def test_sample_values_method(self):
        """Test sample_values method."""
        param1 = HyperparameterRange(
            name="param1", type=ParameterType.FLOAT, low=0.0, high=1.0
        )

        param2 = HyperparameterRange(
            name="param2", type=ParameterType.CATEGORICAL, choices=["a", "b"]
        )

        space = HyperparameterSpace(parameters={"param1": param1, "param2": param2})

        # Sample with fixed random state
        samples = space.sample_values(random_state=42)
        assert len(samples) == 2
        assert "param1" in samples
        assert "param2" in samples
        assert 0.0 <= samples["param1"] <= 1.0
        assert samples["param2"] in ["a", "b"]

    def test_get_grid_combinations_method(self):
        """Test get_grid_combinations method."""
        param1 = HyperparameterRange(
            name="param1", type=ParameterType.CATEGORICAL, choices=["a", "b"]
        )

        param2 = HyperparameterRange(
            name="param2", type=ParameterType.CATEGORICAL, choices=["x", "y"]
        )

        space = HyperparameterSpace(parameters={"param1": param1, "param2": param2})

        combinations = space.get_grid_combinations()
        assert len(combinations) == 4

        expected_combinations = [
            {"param1": "a", "param2": "x"},
            {"param1": "a", "param2": "y"},
            {"param1": "b", "param2": "x"},
            {"param1": "b", "param2": "y"},
        ]

        for combo in expected_combinations:
            assert combo in combinations

    def test_get_grid_combinations_with_limit(self):
        """Test get_grid_combinations with max_combinations limit."""
        param1 = HyperparameterRange(
            name="param1", type=ParameterType.CATEGORICAL, choices=["a", "b", "c"]
        )

        param2 = HyperparameterRange(
            name="param2", type=ParameterType.CATEGORICAL, choices=["x", "y", "z"]
        )

        space = HyperparameterSpace(parameters={"param1": param1, "param2": param2})

        combinations = space.get_grid_combinations(max_combinations=5)
        assert len(combinations) == 5

    def test_estimate_grid_size_method(self):
        """Test estimate_grid_size method."""
        param1 = HyperparameterRange(
            name="param1", type=ParameterType.CATEGORICAL, choices=["a", "b", "c"]
        )

        param2 = HyperparameterRange(
            name="param2", type=ParameterType.FLOAT, low=0.0, high=1.0, grid_size=5
        )

        space = HyperparameterSpace(parameters={"param1": param1, "param2": param2})

        estimated_size = space.estimate_grid_size()
        assert estimated_size == 3 * 5  # 15

    def test_get_dimensionality_method(self):
        """Test get_dimensionality method."""
        param1 = HyperparameterRange(
            name="param1", type=ParameterType.FLOAT, low=0.0, high=1.0
        )

        param2 = HyperparameterRange(
            name="param2", type=ParameterType.CATEGORICAL, choices=["a", "b"]
        )

        space = HyperparameterSpace(parameters={"param1": param1, "param2": param2})

        assert space.get_dimensionality() == 2

    def test_get_parameter_types_method(self):
        """Test get_parameter_types method."""
        param1 = HyperparameterRange(
            name="param1", type=ParameterType.FLOAT, low=0.0, high=1.0
        )

        param2 = HyperparameterRange(
            name="param2", type=ParameterType.CATEGORICAL, choices=["a", "b"]
        )

        space = HyperparameterSpace(parameters={"param1": param1, "param2": param2})

        types = space.get_parameter_types()
        assert types == {"param1": "float", "param2": "categorical"}

    def test_get_numeric_parameters_method(self):
        """Test get_numeric_parameters method."""
        param1 = HyperparameterRange(
            name="param1", type=ParameterType.FLOAT, low=0.0, high=1.0
        )

        param2 = HyperparameterRange(
            name="param2", type=ParameterType.INT, low=1, high=10
        )

        param3 = HyperparameterRange(
            name="param3", type=ParameterType.CATEGORICAL, choices=["a", "b"]
        )

        space = HyperparameterSpace(
            parameters={"param1": param1, "param2": param2, "param3": param3}
        )

        numeric_params = space.get_numeric_parameters()
        assert set(numeric_params) == {"param1", "param2"}

    def test_get_categorical_parameters_method(self):
        """Test get_categorical_parameters method."""
        param1 = HyperparameterRange(
            name="param1", type=ParameterType.FLOAT, low=0.0, high=1.0
        )

        param2 = HyperparameterRange(
            name="param2", type=ParameterType.CATEGORICAL, choices=["a", "b"]
        )

        param3 = HyperparameterRange(name="param3", type=ParameterType.BOOLEAN)

        space = HyperparameterSpace(
            parameters={"param1": param1, "param2": param2, "param3": param3}
        )

        categorical_params = space.get_categorical_parameters()
        assert set(categorical_params) == {"param2", "param3"}

    def test_to_dict_method(self):
        """Test to_dict method."""
        param = HyperparameterRange(
            name="test", type=ParameterType.FLOAT, low=0.0, high=1.0
        )

        space = HyperparameterSpace(
            parameters={"test": param}, name="test_space", description="Test space"
        )

        result = space.to_dict()
        assert result["name"] == "test_space"
        assert result["description"] == "Test space"
        assert result["dimensionality"] == 1
        assert "parameters" in result
        assert "test" in result["parameters"]

    def test_from_dict_method(self):
        """Test from_dict method."""
        data = {
            "name": "test_space",
            "description": "Test space",
            "parameters": {
                "test": {
                    "name": "test",
                    "type": "float",
                    "low": 0.0,
                    "high": 1.0,
                    "distribution": "uniform",
                }
            },
        }

        space = HyperparameterSpace.from_dict(data)
        assert space.name == "test_space"
        assert space.description == "Test space"
        assert len(space.parameters) == 1
        assert "test" in space.parameters
        assert space.parameters["test"].type == ParameterType.FLOAT


class TestHyperparameterSet:
    """Test suite for HyperparameterSet value object."""

    def test_basic_creation(self):
        """Test basic creation of hyperparameter set."""
        param_set = HyperparameterSet(
            parameters={"param1": 0.5, "param2": "a"},
        )

        assert param_set.parameters == {"param1": 0.5, "param2": "a"}
        assert param_set.space is None

    def test_creation_with_space(self):
        """Test creation with space validation."""
        param1 = HyperparameterRange(
            name="param1", type=ParameterType.FLOAT, low=0.0, high=1.0
        )

        param2 = HyperparameterRange(
            name="param2", type=ParameterType.CATEGORICAL, choices=["a", "b"]
        )

        space = HyperparameterSpace(parameters={"param1": param1, "param2": param2})

        # Valid parameters
        param_set = HyperparameterSet(
            parameters={"param1": 0.5, "param2": "a"}, space=space
        )

        assert param_set.parameters == {"param1": 0.5, "param2": "a"}
        assert param_set.space == space

    def test_validation_with_invalid_parameters(self):
        """Test validation with invalid parameters."""
        param = HyperparameterRange(
            name="param1", type=ParameterType.FLOAT, low=0.0, high=1.0
        )

        space = HyperparameterSpace(parameters={"param1": param})

        with pytest.raises(ValueError, match="Invalid parameter values"):
            HyperparameterSet(
                parameters={"param1": 1.5},  # Invalid value
                space=space,
            )

    def test_immutability(self):
        """Test that hyperparameter set is immutable."""
        param_set = HyperparameterSet(parameters={"test": 0.5})

        # Should not be able to modify values
        with pytest.raises(AttributeError):
            param_set.parameters = {"new": 0.7}

    def test_get_method(self):
        """Test get method."""
        param_set = HyperparameterSet(parameters={"param1": 0.5, "param2": "a"})

        assert param_set.get("param1") == 0.5
        assert param_set.get("param2") == "a"
        assert param_set.get("nonexistent") is None
        assert param_set.get("nonexistent", "default") == "default"

    def test_set_method(self):
        """Test set method."""
        param_set = HyperparameterSet(parameters={"param1": 0.5})
        new_param_set = param_set.set("param2", "a")

        # Original should be unchanged
        assert param_set.parameters == {"param1": 0.5}

        # New set should have both parameters
        assert new_param_set.parameters == {"param1": 0.5, "param2": "a"}

    def test_update_method(self):
        """Test update method."""
        param_set = HyperparameterSet(parameters={"param1": 0.5})
        new_param_set = param_set.update({"param2": "a", "param3": True})

        # Original should be unchanged
        assert param_set.parameters == {"param1": 0.5}

        # New set should have all parameters
        assert new_param_set.parameters == {
            "param1": 0.5,
            "param2": "a",
            "param3": True,
        }

    def test_remove_method(self):
        """Test remove method."""
        param_set = HyperparameterSet(parameters={"param1": 0.5, "param2": "a"})
        new_param_set = param_set.remove("param2")

        # Original should be unchanged
        assert param_set.parameters == {"param1": 0.5, "param2": "a"}

        # New set should have only param1
        assert new_param_set.parameters == {"param1": 0.5}

    def test_keys_method(self):
        """Test keys method."""
        param_set = HyperparameterSet(parameters={"param1": 0.5, "param2": "a"})
        keys = param_set.keys()
        assert set(keys) == {"param1", "param2"}

    def test_values_method(self):
        """Test values method."""
        param_set = HyperparameterSet(parameters={"param1": 0.5, "param2": "a"})
        values = param_set.values()
        assert set(values) == {0.5, "a"}

    def test_items_method(self):
        """Test items method."""
        param_set = HyperparameterSet(parameters={"param1": 0.5, "param2": "a"})
        items = list(param_set.items())
        assert set(items) == {("param1", 0.5), ("param2", "a")}

    def test_is_complete_for_space_method(self):
        """Test is_complete_for_space method."""
        param1 = HyperparameterRange(
            name="param1", type=ParameterType.FLOAT, low=0.0, high=1.0
        )

        param2 = HyperparameterRange(
            name="param2", type=ParameterType.CATEGORICAL, choices=["a", "b"]
        )

        space = HyperparameterSpace(parameters={"param1": param1, "param2": param2})

        # Complete set
        complete_set = HyperparameterSet(parameters={"param1": 0.5, "param2": "a"})
        assert complete_set.is_complete_for_space(space) is True

        # Incomplete set
        incomplete_set = HyperparameterSet(parameters={"param1": 0.5})
        assert incomplete_set.is_complete_for_space(space) is False

    def test_get_missing_parameters_method(self):
        """Test get_missing_parameters method."""
        param1 = HyperparameterRange(
            name="param1", type=ParameterType.FLOAT, low=0.0, high=1.0
        )

        param2 = HyperparameterRange(
            name="param2", type=ParameterType.CATEGORICAL, choices=["a", "b"]
        )

        space = HyperparameterSpace(parameters={"param1": param1, "param2": param2})

        incomplete_set = HyperparameterSet(parameters={"param1": 0.5})
        missing = incomplete_set.get_missing_parameters(space)
        assert missing == ["param2"]

    def test_to_dict_method(self):
        """Test to_dict method."""
        param_set = HyperparameterSet(parameters={"param1": 0.5, "param2": "a"})

        result = param_set.to_dict()
        assert result["parameters"] == {"param1": 0.5, "param2": "a"}
        assert result["parameter_count"] == 2
        assert result["space_name"] is None

    def test_from_dict_method(self):
        """Test from_dict method."""
        data = {"parameters": {"param1": 0.5, "param2": "a"}}

        param_set = HyperparameterSet.from_dict(data)
        assert param_set.parameters == {"param1": 0.5, "param2": "a"}

    def test_to_json_method(self):
        """Test to_json method."""
        param_set = HyperparameterSet(parameters={"param1": 0.5, "param2": "a"})

        json_str = param_set.to_json()
        assert isinstance(json_str, str)

        # Should be valid JSON
        parsed = json.loads(json_str)
        assert parsed["parameters"] == {"param1": 0.5, "param2": "a"}

    def test_from_json_method(self):
        """Test from_json method."""
        json_str = '{"parameters": {"param1": 0.5, "param2": "a"}}'

        param_set = HyperparameterSet.from_json(json_str)
        assert param_set.parameters == {"param1": 0.5, "param2": "a"}


class TestUtilityFunctions:
    """Test suite for utility functions."""

    def test_categorical_parameter_function(self):
        """Test categorical_parameter utility function."""
        param = categorical_parameter(
            name="algorithm",
            choices=["rf", "svm", "xgb"],
            default="rf",
            description="Algorithm choice",
        )

        assert param.name == "algorithm"
        assert param.type == ParameterType.CATEGORICAL
        assert param.choices == ["rf", "svm", "xgb"]
        assert param.default_value == "rf"
        assert param.description == "Algorithm choice"

    def test_float_parameter_function(self):
        """Test float_parameter utility function."""
        param = float_parameter(
            name="learning_rate",
            low=0.01,
            high=1.0,
            log=True,
            distribution=DistributionType.LOG_UNIFORM,
            default=0.1,
            description="Learning rate",
        )

        assert param.name == "learning_rate"
        assert param.type == ParameterType.FLOAT
        assert param.low == 0.01
        assert param.high == 1.0
        assert param.log is True
        assert param.distribution == DistributionType.LOG_UNIFORM
        assert param.default_value == 0.1
        assert param.description == "Learning rate"

    def test_int_parameter_function(self):
        """Test int_parameter utility function."""
        param = int_parameter(
            name="n_estimators",
            low=10,
            high=1000,
            log=False,
            step=10,
            default=100,
            description="Number of estimators",
        )

        assert param.name == "n_estimators"
        assert param.type == ParameterType.INT
        assert param.low == 10
        assert param.high == 1000
        assert param.log is False
        assert param.step == 10
        assert param.default_value == 100
        assert param.description == "Number of estimators"

    def test_boolean_parameter_function(self):
        """Test boolean_parameter utility function."""
        param = boolean_parameter(
            name="bootstrap", default=True, description="Whether to use bootstrap"
        )

        assert param.name == "bootstrap"
        assert param.type == ParameterType.BOOLEAN
        assert param.choices == [True, False]
        assert param.default_value is True
        assert param.description == "Whether to use bootstrap"


class TestIntegrationScenarios:
    """Test integration scenarios."""

    def test_complete_optimization_workflow(self):
        """Test complete optimization workflow."""
        # Create parameter space
        space = HyperparameterSpace(
            parameters={
                "learning_rate": float_parameter("learning_rate", 0.01, 1.0, log=True),
                "n_estimators": int_parameter("n_estimators", 10, 1000),
                "algorithm": categorical_parameter("algorithm", ["rf", "xgb"]),
                "bootstrap": boolean_parameter("bootstrap", default=True),
            },
            name="ML_optimization",
            description="Machine learning optimization space",
        )

        # Sample parameter values
        sampled_values = space.sample_values(random_state=42)

        # Create parameter set
        param_set = HyperparameterSet(parameters=sampled_values, space=space)

        # Verify completeness
        assert param_set.is_complete_for_space(space) is True

        # Test serialization
        space_dict = space.to_dict()
        recreated_space = HyperparameterSpace.from_dict(space_dict)
        assert recreated_space.name == space.name
        assert recreated_space.get_dimensionality() == space.get_dimensionality()

        # Test parameter set serialization
        param_dict = param_set.to_dict()
        recreated_set = HyperparameterSet.from_dict(param_dict)
        assert recreated_set.parameters == param_set.parameters

    def test_grid_search_scenario(self):
        """Test grid search scenario."""
        # Create small parameter space for grid search
        space = HyperparameterSpace(
            parameters={
                "C": float_parameter("C", 0.1, 10.0),
                "kernel": categorical_parameter("kernel", ["linear", "rbf"]),
                "gamma": categorical_parameter("gamma", ["scale", "auto"]),
            }
        )

        # Get grid combinations
        combinations = space.get_grid_combinations(max_combinations=20)

        # Verify all combinations are valid
        for combo in combinations:
            assert space.are_valid_values(combo) is True

        # Test grid size estimation
        estimated_size = space.estimate_grid_size()
        assert estimated_size > 0

    def test_parameter_validation_scenarios(self):
        """Test parameter validation scenarios."""
        # Create parameter with validation
        param = float_parameter("test", 0.0, 1.0)

        # Test valid values
        assert param.is_valid_value(0.0) is True
        assert param.is_valid_value(0.5) is True
        assert param.is_valid_value(1.0) is True

        # Test invalid values
        assert param.is_valid_value(-0.1) is False
        assert param.is_valid_value(1.1) is False
        assert param.is_valid_value("0.5") is False

        # Test space validation
        space = HyperparameterSpace(parameters={"test": param})

        assert space.are_valid_values({"test": 0.5}) is True
        assert space.are_valid_values({"test": 1.5}) is False
        assert space.are_valid_values({}) is False

    def test_sampling_consistency(self):
        """Test sampling consistency with random states."""
        param = float_parameter("test", 0.0, 1.0)

        # Same random state should produce same results
        value1 = param.sample_value(random_state=42)
        value2 = param.sample_value(random_state=42)
        assert value1 == value2

        # Different random states should produce different results (with high probability)
        values = [param.sample_value(random_state=i) for i in range(10)]
        assert len(set(values)) > 1  # Should have some variation

    def test_edge_cases(self):
        """Test edge cases."""
        # Single choice categorical parameter
        param = categorical_parameter("test", ["only_choice"])
        assert param.get_grid_values() == ["only_choice"]
        assert param.sample_value(random_state=42) == "only_choice"

        # Very small integer range
        param = int_parameter("test", 1, 2)
        grid_values = param.get_grid_values(size=10)
        assert len(grid_values) == 2  # Should be limited by range
        assert set(grid_values) == {1, 2}

        # Large parameter space
        space = HyperparameterSpace(
            parameters={
                f"param_{i}": float_parameter(f"param_{i}", 0.0, 1.0) for i in range(10)
            }
        )
        assert space.get_dimensionality() == 10

        # Test with extreme values
        param = float_parameter("test", 1e-10, 1e10, log=True)
        grid_values = param.get_grid_values(size=3)
        assert len(grid_values) == 3
        assert grid_values[0] == 1e-10
        assert grid_values[-1] == 1e10
