"""Interface for data profiling domain communication."""

from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, Protocol
from dataclasses import dataclass
from enum import Enum


class ProfileType(Enum):
    """Types of data profiles."""
    SCHEMA = "schema"
    STATISTICAL = "statistical"
    PATTERN = "pattern"
    COMPREHENSIVE = "comprehensive"


@dataclass
class ColumnProfile:
    """Column-level profiling information."""
    name: str
    data_type: str
    null_count: int
    unique_count: int
    min_value: Any
    max_value: Any
    mean_value: Optional[float]
    std_deviation: Optional[float]
    patterns: List[str]
    quality_score: float


@dataclass
class DataProfile:
    """Complete data profile."""
    profile_id: str
    data_collection_id: str
    profile_type: ProfileType
    column_profiles: List[ColumnProfile]
    schema_info: Dict[str, Any]
    statistical_summary: Dict[str, Any]
    patterns_detected: List[str]
    quality_assessment: Dict[str, Any]
    metadata: Dict[str, Any]


class DataProfilingProtocol(Protocol):
    """Protocol for data profiling service interactions."""
    
    def profile_dataset(self, data: Any, profile_type: ProfileType) -> DataProfile:
        """Profile a data_collection."""
        ...
    
    def get_schema_info(self, data: Any) -> Dict[str, Any]:
        """Get schema information."""
        ...


class DataProfilingInterface(ABC):
    """Abstract interface for data profiling operations."""
    
    @abstractmethod
    def create_profile(self, dataset_id: str, profile_type: ProfileType) -> DataProfile:
        """Create a data profile for a data_collection."""
        pass
    
    @abstractmethod
    def get_profile(self, profile_id: str) -> Optional[DataProfile]:
        """Retrieve a data profile."""
        pass
    
    @abstractmethod
    def compare_profiles(self, profile1_id: str, profile2_id: str) -> Dict[str, Any]:
        """Compare two data profiles."""
        pass
    
    @abstractmethod
    def detect_schema_drift(self, profile1_id: str, profile2_id: str) -> Dict[str, Any]:
        """Detect schema drift between profiles."""
        pass