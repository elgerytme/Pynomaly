"""Value objects for ML datasets."""

from dataclasses import dataclass, field
from datetime import datetime
from typing import Dict, List, Optional, Any
from uuid import UUID, uuid4
from enum import Enum


class DatasetType(Enum):
    """Dataset type enumeration."""
    TRAINING = "training"
    VALIDATION = "validation"
    TEST = "test"
    INFERENCE = "inference"
    FULL = "full"


class DataFormat(Enum):
    """Data format enumeration."""
    CSV = "csv"
    JSON = "json"
    PARQUET = "parquet"
    AVRO = "avro"
    PICKLE = "pickle"
    HDF5 = "hdf5"
    TFRECORD = "tfrecord"
    NUMPY = "numpy"
    ARROW = "arrow"


@dataclass(frozen=True)
class DatasetId:
    """Unique identifier for ML datasets."""
    value: UUID = field(default_factory=uuid4)
    
    def __str__(self) -> str:
        return str(self.value)


@dataclass(frozen=True)
class DatasetSchema:
    """Schema definition for ML datasets."""
    columns: Dict[str, str] = field(default_factory=dict)  # column_name -> data_type
    nullable_columns: List[str] = field(default_factory=list)
    primary_key: Optional[str] = None
    foreign_keys: Dict[str, str] = field(default_factory=dict)
    constraints: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert schema to dictionary."""
        return {
            "columns": self.columns,
            "nullable_columns": self.nullable_columns,
            "primary_key": self.primary_key,
            "foreign_keys": self.foreign_keys,
            "constraints": self.constraints,
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "DatasetSchema":
        """Create schema from dictionary."""
        return cls(
            columns=data.get("columns", {}),
            nullable_columns=data.get("nullable_columns", []),
            primary_key=data.get("primary_key"),
            foreign_keys=data.get("foreign_keys", {}),
            constraints=data.get("constraints", {}),
        )


@dataclass(frozen=True)
class DatasetStatistics:
    """Statistics for ML datasets."""
    row_count: int = 0
    column_count: int = 0
    missing_values: Dict[str, int] = field(default_factory=dict)
    data_types: Dict[str, str] = field(default_factory=dict)
    numerical_stats: Dict[str, Dict[str, float]] = field(default_factory=dict)
    categorical_stats: Dict[str, Dict[str, int]] = field(default_factory=dict)
    correlation_matrix: Optional[List[List[float]]] = None
    outliers: Dict[str, List[Any]] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert statistics to dictionary."""
        return {
            "row_count": self.row_count,
            "column_count": self.column_count,
            "missing_values": self.missing_values,
            "data_types": self.data_types,
            "numerical_stats": self.numerical_stats,
            "categorical_stats": self.categorical_stats,
            "correlation_matrix": self.correlation_matrix,
            "outliers": self.outliers,
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "DatasetStatistics":
        """Create statistics from dictionary."""
        return cls(
            row_count=data.get("row_count", 0),
            column_count=data.get("column_count", 0),
            missing_values=data.get("missing_values", {}),
            data_types=data.get("data_types", {}),
            numerical_stats=data.get("numerical_stats", {}),
            categorical_stats=data.get("categorical_stats", {}),
            correlation_matrix=data.get("correlation_matrix"),
            outliers=data.get("outliers", {}),
        )


@dataclass(frozen=True)
class DatasetVersion:
    """Version information for ML datasets."""
    version: str
    created_at: datetime
    checksum: str
    size_bytes: int
    changes: List[str] = field(default_factory=list)
    parent_version: Optional[str] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert version to dictionary."""
        return {
            "version": self.version,
            "created_at": self.created_at.isoformat(),
            "checksum": self.checksum,
            "size_bytes": self.size_bytes,
            "changes": self.changes,
            "parent_version": self.parent_version,
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "DatasetVersion":
        """Create version from dictionary."""
        return cls(
            version=data["version"],
            created_at=datetime.fromisoformat(data["created_at"]),
            checksum=data["checksum"],
            size_bytes=data["size_bytes"],
            changes=data.get("changes", []),
            parent_version=data.get("parent_version"),
        )


@dataclass(frozen=True)
class DatasetMetadata:
    """Metadata for ML datasets."""
    name: str
    description: Optional[str] = None
    source: Optional[str] = None
    license: Optional[str] = None
    tags: List[str] = field(default_factory=list)
    created_by: str = ""
    data_format: DataFormat = DataFormat.CSV
    encoding: str = "utf-8"
    separator: str = ","
    location: str = ""
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert metadata to dictionary."""
        return {
            "name": self.name,
            "description": self.description,
            "source": self.source,
            "license": self.license,
            "tags": self.tags,
            "created_by": self.created_by,
            "data_format": self.data_format.value,
            "encoding": self.encoding,
            "separator": self.separator,
            "location": self.location,
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "DatasetMetadata":
        """Create metadata from dictionary."""
        return cls(
            name=data["name"],
            description=data.get("description"),
            source=data.get("source"),
            license=data.get("license"),
            tags=data.get("tags", []),
            created_by=data.get("created_by", ""),
            data_format=DataFormat(data.get("data_format", DataFormat.CSV.value)),
            encoding=data.get("encoding", "utf-8"),
            separator=data.get("separator", ","),
            location=data.get("location", ""),
        )