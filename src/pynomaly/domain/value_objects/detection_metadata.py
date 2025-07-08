"""Detection metadata value object."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Dict


@dataclass(frozen=True)
class DetectionMetadata:
    """Value object representing detection execution metadata.
    
    This immutable value object encapsulates metadata about
    the detection execution process.
    
    Attributes:
        execution_time: Time taken for detection (seconds)
        parameters: Parameters used for detection
        timestamp: When the detection was performed
        environment: Environment information
        version: Algorithm/model version used
        resource_usage: Resource usage information
    """
    
    execution_time: float
    parameters: Dict[str, Any] = field(default_factory=dict)
    timestamp: datetime = field(default_factory=datetime.utcnow)
    environment: Dict[str, Any] = field(default_factory=dict)
    version: str | None = None
    resource_usage: Dict[str, Any] = field(default_factory=dict)
    
    def __post_init__(self) -> None:
        """Validate detection metadata after initialization."""
        if self.execution_time < 0:
            raise ValueError("Execution time cannot be negative")
        
        if not isinstance(self.parameters, dict):
            raise TypeError("Parameters must be a dictionary")
        
        if not isinstance(self.environment, dict):
            raise TypeError("Environment must be a dictionary")
        
        if not isinstance(self.resource_usage, dict):
            raise TypeError("Resource usage must be a dictionary")
    
    def get_parameter(self, key: str, default: Any = None) -> Any:
        """Get a parameter value with optional default."""
        return self.parameters.get(key, default)
    
    def get_environment_info(self, key: str, default: Any = None) -> Any:
        """Get environment information with optional default."""
        return self.environment.get(key, default)
    
    def get_resource_usage(self, key: str, default: Any = None) -> Any:
        """Get resource usage information with optional default."""
        return self.resource_usage.get(key, default)
    
    def with_parameter(self, key: str, value: Any) -> DetectionMetadata:
        """Create new metadata with updated parameter."""
        new_parameters = self.parameters.copy()
        new_parameters[key] = value
        
        return DetectionMetadata(
            execution_time=self.execution_time,
            parameters=new_parameters,
            timestamp=self.timestamp,
            environment=self.environment,
            version=self.version,
            resource_usage=self.resource_usage
        )
    
    def with_environment_info(self, key: str, value: Any) -> DetectionMetadata:
        """Create new metadata with updated environment info."""
        new_environment = self.environment.copy()
        new_environment[key] = value
        
        return DetectionMetadata(
            execution_time=self.execution_time,
            parameters=self.parameters,
            timestamp=self.timestamp,
            environment=new_environment,
            version=self.version,
            resource_usage=self.resource_usage
        )
    
    def with_resource_usage(self, key: str, value: Any) -> DetectionMetadata:
        """Create new metadata with updated resource usage."""
        new_resource_usage = self.resource_usage.copy()
        new_resource_usage[key] = value
        
        return DetectionMetadata(
            execution_time=self.execution_time,
            parameters=self.parameters,
            timestamp=self.timestamp,
            environment=self.environment,
            version=self.version,
            resource_usage=new_resource_usage
        )
    
    @property
    def execution_time_ms(self) -> float:
        """Get execution time in milliseconds."""
        return self.execution_time * 1000.0
    
    @property
    def memory_usage_mb(self) -> float | None:
        """Get memory usage in MB if available."""
        memory_bytes = self.resource_usage.get("memory_bytes")
        if memory_bytes is not None:
            return memory_bytes / (1024 * 1024)
        return None
    
    @property
    def cpu_usage_percent(self) -> float | None:
        """Get CPU usage percentage if available."""
        return self.resource_usage.get("cpu_percent")
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary representation."""
        return {
            "execution_time": self.execution_time,
            "parameters": self.parameters.copy(),
            "timestamp": self.timestamp.isoformat(),
            "environment": self.environment.copy(),
            "version": self.version,
            "resource_usage": self.resource_usage.copy()
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> DetectionMetadata:
        """Create from dictionary representation."""
        timestamp = data.get("timestamp")
        if isinstance(timestamp, str):
            timestamp = datetime.fromisoformat(timestamp)
        elif timestamp is None:
            timestamp = datetime.utcnow()
        
        return cls(
            execution_time=data["execution_time"],
            parameters=data.get("parameters", {}),
            timestamp=timestamp,
            environment=data.get("environment", {}),
            version=data.get("version"),
            resource_usage=data.get("resource_usage", {})
        )
    
    def __str__(self) -> str:
        """String representation for users."""
        return f"Detection completed in {self.execution_time:.3f}s at {self.timestamp}"
    
    def __repr__(self) -> str:
        """Developer representation."""
        return (
            f"DetectionMetadata(execution_time={self.execution_time}, "
            f"timestamp={self.timestamp}, version={self.version!r})"
        )
