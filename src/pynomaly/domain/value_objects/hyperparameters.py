"""
Hyperparameter Value Objects

Domain value objects for representing hyperparameters, parameter spaces,
and optimization configurations in a type-safe, immutable manner.
"""

import json
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, Union


class ParameterType(Enum):
    """Types of hyperparameters."""

    CATEGORICAL = "categorical"
    FLOAT = "float"
    INT = "int"
    DISCRETE = "discrete"
    BOOLEAN = "boolean"


@dataclass(frozen=True)
class HyperparameterRange:
    """
    Immutable value object representing a range of hyperparameter values.
    """
    
    min_value: Union[int, float]
    max_value: Union[int, float]
    param_type: ParameterType
    
    def __post_init__(self):
        if self.min_value >= self.max_value:
            raise ValueError("min_value must be less than max_value")
    
    def contains(self, value: Union[int, float]) -> bool:
        """Check if value is within range."""
        return self.min_value <= value <= self.max_value


@dataclass(frozen=True)
class HyperparameterSet:
    """
    Immutable value object representing a set of discrete hyperparameter values.
    """
    
    values: List[Any] = field(default_factory=list)
    param_type: ParameterType = ParameterType.CATEGORICAL
    
    def contains(self, value: Any) -> bool:
        """Check if value is in the set."""
        return value in self.values


@dataclass(frozen=True)
class HyperparameterSpace:
    """
    Immutable value object representing a hyperparameter optimization space.
    """
    
    ranges: Dict[str, HyperparameterRange] = field(default_factory=dict)
    sets: Dict[str, HyperparameterSet] = field(default_factory=dict)
    
    def validate_parameters(self, parameters: Dict[str, Any]) -> bool:
        """Validate that parameters are within the defined space."""
        for param_name, value in parameters.items():
            if param_name in self.ranges:
                if not self.ranges[param_name].contains(value):
                    return False
            elif param_name in self.sets:
                if not self.sets[param_name].contains(value):
                    return False
        return True


@dataclass(frozen=True)
class Hyperparameters:
    """
    Immutable value object representing a set of hyperparameters.
    
    Simple wrapper around a dictionary of parameter values.
    """
    
    parameters: Dict[str, Any] = field(default_factory=dict)
    
    def get(self, param_name: str, default: Any = None) -> Any:
        """Get parameter value."""
        return self.parameters.get(param_name, default)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary representation."""
        return self.parameters.copy()
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "Hyperparameters":
        """Create from dictionary representation."""
        if isinstance(data, dict) and "parameters" in data:
            parameters = data["parameters"]
        else:
            parameters = data
        return cls(parameters=parameters)
