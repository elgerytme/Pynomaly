"""
Hyperparameter Value Objects

Domain value objects for representing hyperparameters, parameter spaces,
and optimization configurations in a type-safe, immutable manner.
"""

import json
from dataclasses import dataclass, field
from enum import Enum
from typing import Any


class ParameterType(str, Enum):
    """Types of hyperparameters."""

    CATEGORICAL = "categorical"
    FLOAT = "float"
    INT = "int"
    DISCRETE = "discrete"
    BOOLEAN = "boolean"

    def __str__(self) -> str:
        """Return the string value of the enum."""
        return self.value


class DistributionType(str, Enum):
    """Parameter value distributions."""

    UNIFORM = "uniform"
    LOG_UNIFORM = "log_uniform"
    NORMAL = "normal"
    LOG_NORMAL = "log_normal"
    DISCRETE_UNIFORM = "discrete_uniform"

    def __str__(self) -> str:
        """Return the string value of the enum."""
        return self.value


@dataclass(frozen=True)
class HyperparameterRange:
    """
    Immutable value object representing a hyperparameter's allowed range.

    Defines the search space for a single hyperparameter including its type,
    bounds, distribution, and sampling constraints.
    """

    name: str
    type: ParameterType

    # Numeric parameter bounds
    low: float | None = None
    high: float | None = None

    # Categorical/discrete choices
    choices: list[Any] | None = None

    # Distribution and sampling
    distribution: DistributionType = DistributionType.UNIFORM
    log: bool = False
    step: float | None = None

    # Grid search specific
    grid_size: int | None = None

    # Constraints and metadata
    default_value: Any | None = None
    description: str | None = None

    def __post_init__(self) -> None:
        """Validate parameter range configuration."""
        if self.type in [ParameterType.CATEGORICAL, ParameterType.DISCRETE]:
            if not self.choices:
                raise ValueError(
                    f"Choices required for {self.type.value} parameter {self.name}"
                )

        elif self.type in [ParameterType.FLOAT, ParameterType.INT]:
            if self.low is None or self.high is None:
                raise ValueError(
                    f"Low and high bounds required for {self.type.value} parameter {self.name}"
                )
            if self.low >= self.high:
                raise ValueError(
                    f"Low bound must be less than high bound for parameter {self.name}"
                )

        elif self.type == ParameterType.BOOLEAN:
            if self.choices is None:
                object.__setattr__(self, "choices", [True, False])

    def is_valid_value(self, value: Any) -> bool:
        """Check if a value is valid for this parameter range."""
        try:
            if self.type == ParameterType.CATEGORICAL:
                return self.choices is not None and value in self.choices

            elif self.type == ParameterType.DISCRETE:
                return self.choices is not None and value in self.choices

            elif self.type == ParameterType.BOOLEAN:
                return isinstance(value, bool)

            elif self.type == ParameterType.FLOAT:
                if not isinstance(value, (int, float)):
                    return False
                return (
                    self.low is not None
                    and self.high is not None
                    and self.low <= value <= self.high
                )

            elif self.type == ParameterType.INT:
                if not isinstance(value, int):
                    return False
                return (
                    self.low is not None
                    and self.high is not None
                    and self.low <= value <= self.high
                )

            return False

        except (TypeError, ValueError):
            return False

    def get_grid_values(self, size: int | None = None) -> list[Any]:
        """Get grid values for this parameter."""
        grid_size = size or self.grid_size or 10

        if self.type in [
            ParameterType.CATEGORICAL,
            ParameterType.DISCRETE,
            ParameterType.BOOLEAN,
        ]:
            return list(self.choices) if self.choices is not None else []

        elif self.type == ParameterType.FLOAT:
            import numpy as np

            if self.low is None or self.high is None:
                return []

            if self.log or self.distribution == DistributionType.LOG_UNIFORM:
                return np.logspace(
                    np.log10(self.low), np.log10(self.high), grid_size
                ).tolist()
            else:
                return np.linspace(self.low, self.high, grid_size).tolist()

        elif self.type == ParameterType.INT:
            import numpy as np

            if self.low is None or self.high is None:
                return []

            max_size = min(grid_size, int(self.high - self.low + 1))
            return np.linspace(self.low, self.high, max_size, dtype=int).tolist()

        return []

    def sample_value(self, random_state: int | None = None) -> Any:
        """Sample a random value from this parameter's distribution."""
        import numpy as np

        if random_state is not None:
            np.random.seed(random_state)

        if self.type in [
            ParameterType.CATEGORICAL,
            ParameterType.DISCRETE,
            ParameterType.BOOLEAN,
        ]:
            if self.choices is None:
                return None
            return np.random.choice(self.choices)

        elif self.type == ParameterType.FLOAT:
            if self.low is None or self.high is None:
                return None
            if self.log or self.distribution == DistributionType.LOG_UNIFORM:
                log_low = np.log(self.low)
                log_high = np.log(self.high)
                return np.exp(np.random.uniform(log_low, log_high))
            elif self.distribution == DistributionType.NORMAL:
                # Use mean and std based on bounds
                mean = (self.low + self.high) / 2
                std = (self.high - self.low) / 6  # 3-sigma rule
                value = np.random.normal(mean, std)
                return np.clip(value, self.low, self.high)
            else:
                return np.random.uniform(self.low, self.high)

        elif self.type == ParameterType.INT:
            if self.low is None or self.high is None:
                return None
            if self.log:
                log_low = np.log(self.low)
                log_high = np.log(self.high)
                return int(np.exp(np.random.uniform(log_low, log_high)))
            else:
                return np.random.randint(self.low, self.high + 1)

        return self.default_value

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary representation."""
        return {
            "name": self.name,
            "type": self.type.value,
            "low": self.low,
            "high": self.high,
            "choices": self.choices,
            "distribution": self.distribution.value,
            "log": self.log,
            "step": self.step,
            "grid_size": self.grid_size,
            "default_value": self.default_value,
            "description": self.description,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "HyperparameterRange":
        """Create from dictionary representation."""
        data = data.copy()
        if "type" in data:
            data["type"] = ParameterType(data["type"])
        if "distribution" in data:
            data["distribution"] = DistributionType(data["distribution"])
        return cls(**data)


@dataclass(frozen=True)
class HyperparameterSpace:
    """
    Immutable value object representing a complete hyperparameter search space.

    Contains multiple parameter ranges and provides methods for sampling,
    validation, and search space analysis.
    """

    parameters: dict[str, HyperparameterRange] = field(default_factory=dict)
    name: str | None = None
    description: str | None = None

    def __post_init__(self):
        """Validate hyperparameter space."""
        if not self.parameters:
            raise ValueError("Hyperparameter space must contain at least one parameter")

        # Ensure all parameter names match their keys
        for param_name, param_range in self.parameters.items():
            if param_range.name != param_name:
                raise ValueError(
                    f"Parameter name mismatch: key '{param_name}' vs name '{param_range.name}'"
                )

    def add_parameter(self, param_range: HyperparameterRange) -> "HyperparameterSpace":
        """Add a parameter to the space (returns new immutable instance)."""
        new_params = self.parameters.copy()
        new_params[param_range.name] = param_range
        return HyperparameterSpace(
            parameters=new_params, name=self.name, description=self.description
        )

    def remove_parameter(self, param_name: str) -> "HyperparameterSpace":
        """Remove a parameter from the space (returns new immutable instance)."""
        new_params = self.parameters.copy()
        if param_name in new_params:
            del new_params[param_name]
        return HyperparameterSpace(
            parameters=new_params, name=self.name, description=self.description
        )

    def get_parameter(self, param_name: str) -> HyperparameterRange | None:
        """Get a parameter by name."""
        return self.parameters.get(param_name)

    def validate_values(self, values: dict[str, Any]) -> dict[str, bool]:
        """Validate a set of parameter values."""
        validation_results = {}
        for param_name, param_range in self.parameters.items():
            if param_name in values:
                validation_results[param_name] = param_range.is_valid_value(
                    values[param_name]
                )
            else:
                validation_results[param_name] = False
        return validation_results

    def are_valid_values(self, values: dict[str, Any]) -> bool:
        """Check if all parameter values are valid."""
        validation_results = self.validate_values(values)
        return all(validation_results.values())

    def sample_values(self, random_state: int | None = None) -> dict[str, Any]:
        """Sample random values for all parameters."""
        import numpy as np

        if random_state is not None:
            np.random.seed(random_state)

        sampled_values = {}
        for param_name, param_range in self.parameters.items():
            sampled_values[param_name] = param_range.sample_value()

        return sampled_values

    def get_grid_combinations(
        self, max_combinations: int | None = None
    ) -> list[dict[str, Any]]:
        """Get all grid combinations for the parameter space."""
        import itertools

        # Get grid values for each parameter
        param_grids = {}
        for param_name, param_range in self.parameters.items():
            param_grids[param_name] = param_range.get_grid_values()

        # Generate all combinations
        param_names = list(param_grids.keys())
        param_values = list(param_grids.values())

        combinations = []
        for combination in itertools.product(*param_values):
            param_dict = dict(zip(param_names, combination, strict=False))
            combinations.append(param_dict)

            # Limit combinations if specified
            if max_combinations and len(combinations) >= max_combinations:
                break

        return combinations

    def estimate_grid_size(self) -> int:
        """Estimate the total number of grid combinations."""
        total_size = 1
        for param_range in self.parameters.values():
            if param_range.type in [
                ParameterType.CATEGORICAL,
                ParameterType.DISCRETE,
                ParameterType.BOOLEAN,
            ]:
                total_size *= len(param_range.choices)
            else:
                grid_size = param_range.grid_size or 10
                total_size *= grid_size
        return total_size

    def get_dimensionality(self) -> int:
        """Get the dimensionality of the search space."""
        return len(self.parameters)

    def get_parameter_types(self) -> dict[str, str]:
        """Get parameter types mapping."""
        return {name: param.type.value for name, param in self.parameters.items()}

    def get_numeric_parameters(self) -> list[str]:
        """Get names of numeric parameters."""
        return [
            name
            for name, param in self.parameters.items()
            if param.type in [ParameterType.FLOAT, ParameterType.INT]
        ]

    def get_categorical_parameters(self) -> list[str]:
        """Get names of categorical parameters."""
        return [
            name
            for name, param in self.parameters.items()
            if param.type
            in [
                ParameterType.CATEGORICAL,
                ParameterType.DISCRETE,
                ParameterType.BOOLEAN,
            ]
        ]

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary representation."""
        return {
            "parameters": {
                name: param.to_dict() for name, param in self.parameters.items()
            },
            "name": self.name,
            "description": self.description,
            "dimensionality": self.get_dimensionality(),
            "estimated_grid_size": self.estimate_grid_size(),
            "parameter_types": self.get_parameter_types(),
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "HyperparameterSpace":
        """Create from dictionary representation."""
        parameters = {}
        if "parameters" in data:
            for name, param_data in data["parameters"].items():
                parameters[name] = HyperparameterRange.from_dict(param_data)

        return cls(
            parameters=parameters,
            name=data.get("name"),
            description=data.get("description"),
        )


@dataclass(frozen=True)
class HyperparameterSet:
    """
    Immutable value object representing a specific set of hyperparameter values.

    Represents a concrete assignment of values to hyperparameters, typically
    the result of optimization or manual configuration.
    """

    parameters: dict[str, Any] = field(default_factory=dict)
    space: HyperparameterSpace | None = None

    def __post_init__(self):
        """Validate hyperparameter set."""
        if self.space:
            if not self.space.are_valid_values(self.parameters):
                invalid_params = [
                    name
                    for name, is_valid in self.space.validate_values(
                        self.parameters
                    ).items()
                    if not is_valid
                ]
                raise ValueError(f"Invalid parameter values: {invalid_params}")

    def get(self, param_name: str, default: Any = None) -> Any:
        """Get parameter value."""
        return self.parameters.get(param_name, default)

    def set(self, param_name: str, value: Any) -> "HyperparameterSet":
        """Set parameter value (returns new immutable instance)."""
        new_params = self.parameters.copy()
        new_params[param_name] = value
        return HyperparameterSet(parameters=new_params, space=self.space)

    def update(self, updates: dict[str, Any]) -> "HyperparameterSet":
        """Update multiple parameters (returns new immutable instance)."""
        new_params = self.parameters.copy()
        new_params.update(updates)
        return HyperparameterSet(parameters=new_params, space=self.space)

    def remove(self, param_name: str) -> "HyperparameterSet":
        """Remove parameter (returns new immutable instance)."""
        new_params = self.parameters.copy()
        if param_name in new_params:
            del new_params[param_name]
        return HyperparameterSet(parameters=new_params, space=self.space)

    def keys(self) -> list[str]:
        """Get parameter names."""
        return list(self.parameters.keys())

    def values(self) -> list[Any]:
        """Get parameter values."""
        return list(self.parameters.values())

    def items(self):
        """Get parameter items."""
        return self.parameters.items()

    def is_complete_for_space(self, space: HyperparameterSpace) -> bool:
        """Check if this set provides values for all parameters in a space."""
        return all(
            param_name in self.parameters for param_name in space.parameters.keys()
        )

    def get_missing_parameters(self, space: HyperparameterSpace) -> list[str]:
        """Get list of parameters missing for a given space."""
        return [
            param_name
            for param_name in space.parameters.keys()
            if param_name not in self.parameters
        ]

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary representation."""
        return {
            "parameters": self.parameters.copy(),
            "parameter_count": len(self.parameters),
            "space_name": self.space.name if self.space else None,
        }

    @classmethod
    def from_dict(
        cls, data: dict[str, Any], space: HyperparameterSpace | None = None
    ) -> "HyperparameterSet":
        """Create from dictionary representation."""
        if isinstance(data, dict) and "parameters" in data:
            parameters = data["parameters"]
        else:
            parameters = data

        return cls(parameters=parameters, space=space)

    def to_json(self) -> str:
        """Convert to JSON string."""
        return json.dumps(self.to_dict(), default=str)

    @classmethod
    def from_json(
        cls, json_str: str, space: HyperparameterSpace | None = None
    ) -> "HyperparameterSet":
        """Create from JSON string."""
        data = json.loads(json_str)
        return cls.from_dict(data, space)


# Utility functions for creating common parameter ranges


def categorical_parameter(
    name: str, choices: list[Any], default: Any = None, description: str = None
) -> HyperparameterRange:
    """Create a categorical parameter range."""
    return HyperparameterRange(
        name=name,
        type=ParameterType.CATEGORICAL,
        choices=choices,
        default_value=default,
        description=description,
    )


def float_parameter(
    name: str,
    low: float,
    high: float,
    log: bool = False,
    distribution: DistributionType = DistributionType.UNIFORM,
    default: float = None,
    description: str = None,
) -> HyperparameterRange:
    """Create a float parameter range."""
    return HyperparameterRange(
        name=name,
        type=ParameterType.FLOAT,
        low=low,
        high=high,
        log=log,
        distribution=distribution,
        default_value=default,
        description=description,
    )


def int_parameter(
    name: str,
    low: int,
    high: int,
    log: bool = False,
    step: int = 1,
    default: int = None,
    description: str = None,
) -> HyperparameterRange:
    """Create an integer parameter range."""
    return HyperparameterRange(
        name=name,
        type=ParameterType.INT,
        low=low,
        high=high,
        log=log,
        step=step,
        default_value=default,
        description=description,
    )


def boolean_parameter(
    name: str, default: bool = None, description: str = None
) -> HyperparameterRange:
    """Create a boolean parameter range."""
    return HyperparameterRange(
        name=name,
        type=ParameterType.BOOLEAN,
        choices=[True, False],
        default_value=default,
        description=description,
    )
