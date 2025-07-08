"""Algorithm configuration value object."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict


@dataclass(frozen=True)
class AlgorithmConfig:
    """Value object representing algorithm configuration.
    
    This immutable value object encapsulates algorithm-specific
    configuration parameters.
    
    Attributes:
        algorithm: Name of the algorithm
        parameters: Algorithm-specific parameters
        version: Algorithm version (optional)
        description: Description of the configuration
    """
    
    algorithm: str
    parameters: Dict[str, Any] = field(default_factory=dict)
    version: str | None = None
    description: str | None = None
    
    def __post_init__(self) -> None:
        """Validate algorithm configuration after initialization."""
        if not self.algorithm:
            raise ValueError("Algorithm name cannot be empty")
        
        if not isinstance(self.parameters, dict):
            raise TypeError("Parameters must be a dictionary")
        
        # Validate parameter types
        for key, value in self.parameters.items():
            if not isinstance(key, str):
                raise TypeError(f"Parameter key must be string, got {type(key)}")
            
            # Ensure parameters are serializable
            if not self._is_serializable(value):
                raise TypeError(f"Parameter '{key}' must be serializable")
    
    def _is_serializable(self, value: Any) -> bool:
        """Check if a value is serializable."""
        try:
            import json
            json.dumps(value)
            return True
        except (TypeError, ValueError):
            return False
    
    def get_parameter(self, key: str, default: Any = None) -> Any:
        """Get a parameter value with optional default."""
        return self.parameters.get(key, default)
    
    def with_parameter(self, key: str, value: Any) -> AlgorithmConfig:
        """Create a new config with updated parameter."""
        if not self._is_serializable(value):
            raise TypeError(f"Parameter '{key}' must be serializable")
        
        new_parameters = self.parameters.copy()
        new_parameters[key] = value
        
        return AlgorithmConfig(
            algorithm=self.algorithm,
            parameters=new_parameters,
            version=self.version,
            description=self.description
        )
    
    def with_parameters(self, **params: Any) -> AlgorithmConfig:
        """Create a new config with multiple updated parameters."""
        new_parameters = self.parameters.copy()
        
        for key, value in params.items():
            if not self._is_serializable(value):
                raise TypeError(f"Parameter '{key}' must be serializable")
            new_parameters[key] = value
        
        return AlgorithmConfig(
            algorithm=self.algorithm,
            parameters=new_parameters,
            version=self.version,
            description=self.description
        )
    
    def without_parameter(self, key: str) -> AlgorithmConfig:
        """Create a new config without specified parameter."""
        new_parameters = self.parameters.copy()
        new_parameters.pop(key, None)
        
        return AlgorithmConfig(
            algorithm=self.algorithm,
            parameters=new_parameters,
            version=self.version,
            description=self.description
        )
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary representation."""
        return {
            "algorithm": self.algorithm,
            "parameters": self.parameters.copy(),
            "version": self.version,
            "description": self.description
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> AlgorithmConfig:
        """Create from dictionary representation."""
        return cls(
            algorithm=data["algorithm"],
            parameters=data.get("parameters", {}),
            version=data.get("version"),
            description=data.get("description")
        )
    
    def __str__(self) -> str:
        """String representation for users."""
        version_str = f" v{self.version}" if self.version else ""
        param_count = len(self.parameters)
        return f"{self.algorithm}{version_str} ({param_count} parameters)"
    
    def __repr__(self) -> str:
        """Developer representation."""
        return (
            f"AlgorithmConfig(algorithm='{self.algorithm}', "
            f"parameters={self.parameters}, version={self.version!r}, "
            f"description={self.description!r})"
        )
