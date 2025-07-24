
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional, Union
from uuid import UUID, uuid4


class DataType(str, Enum):
    """Data type classifications."""
    
    STRING = "string"
    INTEGER = "integer"
    FLOAT = "float"
    BOOLEAN = "boolean"
    DATE = "date"
    DATETIME = "datetime"
    TIME = "time"
    DECIMAL = "decimal"
    BINARY = "binary"
    JSON = "json"
    ARRAY = "array"
    OBJECT = "object"
    NULL = "null"
    UNKNOWN = "unknown"


class ProfileStatus(str, Enum):
    """Status of profile generation."""
    
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    PARTIAL = "partial"


@dataclass
class ProfileStatistics:
    """Statistical measures for column data."""
    
    # Basic counts
    total_count: int = 0
    null_count: int = 0
    distinct_count: int = 0
    duplicate_count: int = 0
    
    # String statistics
    min_length: Optional[int] = None
    max_length: Optional[int] = None
    avg_length: Optional[float] = None
    
    # Numeric statistics
    min_value: Optional[Union[int, float]] = None
    max_value: Optional[Union[int, float]] = None
    mean: Optional[float] = None
    median: Optional[float] = None
    mode: Optional[Union[str, int, float]] = None
    std_dev: Optional[float] = None
    variance: Optional[float] = None
    
    # Percentiles
    p25: Optional[float] = None  # 25th percentile
    p75: Optional[float] = None  # 75th percentile
    p95: Optional[float] = None  # 95th percentile
    p99: Optional[float] = None  # 99th percentile
    
    # Distribution
    skewness: Optional[float] = None
    kurtosis: Optional[float] = None
    entropy: Optional[float] = None
    
    def __post_init__(self) -> None:
        """Validate statistics after initialization."""
        if self.total_count < 0:
            raise ValueError("Total count cannot be negative")
        
        if self.null_count < 0:
            raise ValueError("Null count cannot be negative")
        
        if self.null_count > self.total_count:
            raise ValueError("Null count cannot exceed total count")
        
        if self.distinct_count is not None and self.distinct_count < 0:
            raise ValueError("Distinct count cannot be negative")
    
    @property
    def null_rate(self) -> float:
        """Get null rate as percentage."""
        if self.total_count == 0:
            return 0.0
        return (self.null_count / self.total_count) * 100
    
    @property
    def completeness_rate(self) -> float:
        """Get completeness rate as percentage."""
        return 100.0 - self.null_rate
    
    @property
    def uniqueness_rate(self) -> float:
        """Get uniqueness rate as percentage."""
        if self.total_count == 0 or self.distinct_count is None:
            return 0.0
        return (self.distinct_count / self.total_count) * 100
    
    @property
    def duplication_rate(self) -> float:
        """Get duplication rate as percentage."""
        if self.total_count == 0:
            return 0.0
        return (self.duplicate_count / self.total_count) * 100
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert statistics to dictionary."""
        return {
            "total_count": self.total_count,
            "null_count": self.null_count,
            "distinct_count": self.distinct_count,
            "duplicate_count": self.duplicate_count,
            "min_length": self.min_length,
            "max_length": self.max_length,
            "avg_length": self.avg_length,
            "min_value": self.min_value,
            "max_value": self.max_value,
            "mean": self.mean,
            "median": self.median,
            "mode": self.mode,
            "std_dev": self.std_dev,
            "variance": self.variance,
            "p25": self.p25,
            "p75": self.p75,
            "p95": self.p95,
            "p99": self.p99,
            "skewness": self.skewness,
            "kurtosis": self.kurtosis,
            "entropy": self.entropy,
            "null_rate": self.null_rate,
            "completeness_rate": self.completeness_rate,
            "uniqueness_rate": self.uniqueness_rate,
            "duplication_rate": self.duplication_rate,
        }


@dataclass
class ColumnProfile:
    """Profile information for a single column."""
    
    id: UUID = field(default_factory=uuid4)
    column_name: str = ""
    data_type: DataType = DataType.UNKNOWN
    inferred_type: Optional[DataType] = None
    
    # Basic information
    position: int = 0
    is_nullable: bool = True
    is_primary_key: bool = False
    is_foreign_key: bool = False
    foreign_key_table: Optional[str] = None
    foreign_key_column: Optional[str] = None
    
    # Statistics
    statistics: ProfileStatistics = field(default_factory=ProfileStatistics)
    
    # Patterns and formats
    common_patterns: List[str] = field(default_factory=list)
    format_patterns: List[str] = field(default_factory=list)
    regex_patterns: List[str] = field(default_factory=list)
    
    # Value analysis
    top_values: List[Dict[str, Any]] = field(default_factory=list)  # value, count, percentage
    sample_values: List[Any] = field(default_factory=list)
    invalid_values: List[Any] = field(default_factory=list)
    
    # Data quality indicators
    quality_score: float = 0.0  # 0.0 to 1.0
    anomaly_count: int = 0
    outlier_count: int = 0
    
    # Metadata
    created_at: datetime = field(default_factory=datetime.utcnow)
    updated_at: datetime = field(default_factory=datetime.utcnow)
    
    def __post_init__(self) -> None:
        """Validate column profile after initialization."""
        if not self.column_name:
            raise ValueError("Column name cannot be empty")
        
        if not (0.0 <= self.quality_score <= 1.0):
            raise ValueError("Quality score must be between 0.0 and 1.0")
        
        if self.anomaly_count < 0:
            raise ValueError("Anomaly count cannot be negative")
        
        if self.outlier_count < 0:
            raise ValueError("Outlier count cannot be negative")
    
    @property
    def is_numeric(self) -> bool:
        """Check if column contains numeric data."""
        return self.data_type in [DataType.INTEGER, DataType.FLOAT, DataType.DECIMAL]
    
    @property
    def is_categorical(self) -> bool:
        """Check if column appears to be categorical."""
        if self.statistics.total_count == 0:
            return False
        
        # Consider categorical if unique values are less than 10% of total
        if self.statistics.distinct_count is None:
            return False
        
        uniqueness_ratio = self.statistics.distinct_count / self.statistics.total_count
        return uniqueness_ratio < 0.1 and self.statistics.distinct_count <= 50
    
    @property
    def is_high_cardinality(self) -> bool:
        """Check if column has high cardinality."""
        if self.statistics.distinct_count is None or self.statistics.total_count == 0:
            return False
        
        uniqueness_ratio = self.statistics.distinct_count / self.statistics.total_count
        return uniqueness_ratio > 0.8
    
    @property
    def has_quality_issues(self) -> bool:
        """Check if column has quality issues."""
        return (
            self.quality_score < 0.8 or
            self.anomaly_count > 0 or
            self.statistics.null_rate > 20.0
        )
    
    def add_top_value(self, value: Any, count: int) -> None:
        """Add a top value to the profile."""
        if self.statistics.total_count > 0:
            percentage = (count / self.statistics.total_count) * 100
        else:
            percentage = 0.0
        
        self.top_values.append({
            "value": value,
            "count": count,
            "percentage": percentage
        })
        
        # Keep only top 10 values
        self.top_values.sort(key=lambda x: x["count"], reverse=True)
        self.top_values = self.top_values[:10]
        
        self.updated_at = datetime.utcnow()
    
    def add_pattern(self, pattern: str, pattern_type: str = "common") -> None:
        """Add a pattern to the profile."""
        if pattern_type == "common" and pattern not in self.common_patterns:
            self.common_patterns.append(pattern)
        elif pattern_type == "format" and pattern not in self.format_patterns:
            self.format_patterns.append(pattern)
        elif pattern_type == "regex" and pattern not in self.regex_patterns:
            self.regex_patterns.append(pattern)
        
        self.updated_at = datetime.utcnow()
    
    def calculate_quality_score(self) -> float:
        """Calculate overall quality score for the column."""
        score_components = []
        
        # Completeness component (40% weight)
        completeness_score = self.statistics.completeness_rate / 100.0
        score_components.append(completeness_score * 0.4)
        
        # Consistency component (30% weight)
        consistency_score = 1.0
        if self.anomaly_count > 0:
            # Reduce score based on anomaly rate
            anomaly_rate = self.anomaly_count / max(self.statistics.total_count, 1)
            consistency_score = max(0.0, 1.0 - anomaly_rate)
        score_components.append(consistency_score * 0.3)
        
        # Validity component (20% weight)
        validity_score = 1.0
        if self.invalid_values:
            invalid_rate = len(self.invalid_values) / max(self.statistics.total_count, 1)
            validity_score = max(0.0, 1.0 - invalid_rate)
        score_components.append(validity_score * 0.2)
        
        # Uniqueness component (10% weight)
        uniqueness_score = min(1.0, self.statistics.uniqueness_rate / 100.0)
        score_components.append(uniqueness_score * 0.1)
        
        self.quality_score = sum(score_components)
        return self.quality_score
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert column profile to dictionary."""
        return {
            "id": str(self.id),
            "column_name": self.column_name,
            "data_type": self.data_type.value,
            "inferred_type": self.inferred_type.value if self.inferred_type else None,
            "position": self.position,
            "is_nullable": self.is_nullable,
            "is_primary_key": self.is_primary_key,
            "is_foreign_key": self.is_foreign_key,
            "foreign_key_table": self.foreign_key_table,
            "foreign_key_column": self.foreign_key_column,
            "statistics": self.statistics.to_dict(),
            "common_patterns": self.common_patterns,
            "format_patterns": self.format_patterns,
            "regex_patterns": self.regex_patterns,
            "top_values": self.top_values,
            "sample_values": self.sample_values[:10],  # Limit for serialization
            "invalid_values": self.invalid_values[:10],
            "quality_score": self.quality_score,
            "anomaly_count": self.anomaly_count,
            "outlier_count": self.outlier_count,
            "created_at": self.created_at.isoformat(),
            "updated_at": self.updated_at.isoformat(),
            "is_numeric": self.is_numeric,
            "is_categorical": self.is_categorical,
            "is_high_cardinality": self.is_high_cardinality,
            "has_quality_issues": self.has_quality_issues,
        }


@dataclass
class DataProfile:
    """Data profile domain entity representing dataset profiling results."""
    
    id: UUID = field(default_factory=uuid4)
    dataset_name: str = ""
    table_name: Optional[str] = None
    schema_name: Optional[str] = None
    
    # Profile status
    status: ProfileStatus = ProfileStatus.PENDING
    version: str = "1.0.0"
    
    # Dataset overview
    total_rows: int = 0
    total_columns: int = 0
    file_size_bytes: Optional[int] = None
    
    # Column profiles
    column_profiles: List[ColumnProfile] = field(default_factory=list)
    
    # Dataset-level statistics
    completeness_score: float = 0.0
    uniqueness_score: float = 0.0
    validity_score: float = 0.0
    overall_quality_score: float = 0.0
    
    # Relationships
    primary_keys: List[str] = field(default_factory=list)
    foreign_keys: List[Dict[str, str]] = field(default_factory=list)
    relationships: List[Dict[str, Any]] = field(default_factory=list)
    
    # Profiling metadata
    profiling_started_at: Optional[datetime] = None
    profiling_completed_at: Optional[datetime] = None
    profiling_duration_ms: float = 0.0
    sample_size: Optional[int] = None
    sampling_method: Optional[str] = None
    
    # Metadata
    created_at: datetime = field(default_factory=datetime.utcnow)
    created_by: str = ""
    updated_at: datetime = field(default_factory=datetime.utcnow)
    updated_by: str = ""
    
    # Configuration
    config: Dict[str, Any] = field(default_factory=dict)
    tags: List[str] = field(default_factory=list)
    
    def __post_init__(self) -> None:
        """Validate data profile after initialization."""
        if not self.dataset_name:
            raise ValueError("Dataset name cannot be empty")
        
        if self.total_rows < 0:
            raise ValueError("Total rows cannot be negative")
        
        if self.total_columns < 0:
            raise ValueError("Total columns cannot be negative")
        
        if not (0.0 <= self.overall_quality_score <= 1.0):
            raise ValueError("Overall quality score must be between 0.0 and 1.0")
    
    @property
    def is_completed(self) -> bool:
        """Check if profiling is completed."""
        return self.status == ProfileStatus.COMPLETED
    
    @property
    def is_running(self) -> bool:
        """Check if profiling is currently running."""
        return self.status == ProfileStatus.RUNNING
    
    @property
    def has_quality_issues(self) -> bool:
        """Check if dataset has quality issues."""
        return (
            self.overall_quality_score < 0.8 or
            any(cp.has_quality_issues for cp in self.column_profiles)
        )
    
    @property
    def profiling_duration_seconds(self) -> float:
        """Get profiling duration in seconds."""
        return self.profiling_duration_ms / 1000.0
    
    def add_column_profile(self, column_profile: ColumnProfile) -> None:
        """Add a column profile."""
        # Check for duplicate column names
        existing_columns = [cp.column_name for cp in self.column_profiles]
        if column_profile.column_name in existing_columns:
            raise ValueError(f"Column profile for '{column_profile.column_name}' already exists")
        
        self.column_profiles.append(column_profile)
        self.total_columns = len(self.column_profiles)
        self.updated_at = datetime.utcnow()
    
    def remove_column_profile(self, column_name: str) -> bool:
        """Remove a column profile."""
        for i, cp in enumerate(self.column_profiles):
            if cp.column_name == column_name:
                self.column_profiles.pop(i)
                self.total_columns = len(self.column_profiles)
                self.updated_at = datetime.utcnow()
                return True
        return False
    
    def get_column_profile(self, column_name: str) -> Optional[ColumnProfile]:
        """Get a column profile by name."""
        for cp in self.column_profiles:
            if cp.column_name == column_name:
                return cp
        return None
    
    def get_numeric_columns(self) -> List[ColumnProfile]:
        """Get all numeric column profiles."""
        return [cp for cp in self.column_profiles if cp.is_numeric]
    
    def get_categorical_columns(self) -> List[ColumnProfile]:
        """Get all categorical column profiles."""
        return [cp for cp in self.column_profiles if cp.is_categorical]
    
    def get_high_cardinality_columns(self) -> List[ColumnProfile]:
        """Get all high cardinality column profiles."""
        return [cp for cp in self.column_profiles if cp.is_high_cardinality]
    
    def get_columns_with_quality_issues(self) -> List[ColumnProfile]:
        """Get columns with quality issues."""
        return [cp for cp in self.column_profiles if cp.has_quality_issues]
    
    def start_profiling(self) -> None:
        """Start the profiling process."""
        self.status = ProfileStatus.RUNNING
        self.profiling_started_at = datetime.utcnow()
        self.updated_at = self.profiling_started_at
    
    def complete_profiling(self) -> None:
        """Complete the profiling process."""
        if self.status != ProfileStatus.RUNNING:
            raise ValueError("Cannot complete profiling that is not running")
        
        self.status = ProfileStatus.COMPLETED
        self.profiling_completed_at = datetime.utcnow()
        self.updated_at = self.profiling_completed_at
        
        if self.profiling_started_at:
            duration = self.profiling_completed_at - self.profiling_started_at
            self.profiling_duration_ms = duration.total_seconds() * 1000
        
        # Calculate overall scores
        self.calculate_overall_scores()
    
    def fail_profiling(self, error_message: str) -> None:
        """Mark profiling as failed."""
        self.status = ProfileStatus.FAILED
        self.config["error_message"] = error_message
        self.updated_at = datetime.utcnow()
    
    def calculate_overall_scores(self) -> None:
        """Calculate overall quality scores for the dataset."""
        if not self.column_profiles:
            return
        
        # Completeness score (average across columns)
        completeness_scores = [cp.statistics.completeness_rate / 100.0 for cp in self.column_profiles]
        self.completeness_score = sum(completeness_scores) / len(completeness_scores)
        
        # Uniqueness score (average across columns)
        uniqueness_scores = [cp.statistics.uniqueness_rate / 100.0 for cp in self.column_profiles]
        self.uniqueness_score = sum(uniqueness_scores) / len(uniqueness_scores)
        
        # Validity score (based on columns with quality issues)
        columns_with_issues = len(self.get_columns_with_quality_issues())
        self.validity_score = max(0.0, 1.0 - (columns_with_issues / len(self.column_profiles)))
        
        # Overall quality score (weighted average)
        self.overall_quality_score = (
            self.completeness_score * 0.4 +
            self.validity_score * 0.4 +
            self.uniqueness_score * 0.2
        )
    
    def add_tag(self, tag: str) -> None:
        """Add a tag to the profile."""
        if tag and tag not in self.tags:
            self.tags.append(tag)
            self.updated_at = datetime.utcnow()
    
    def remove_tag(self, tag: str) -> None:
        """Remove a tag from the profile."""
        if tag in self.tags:
            self.tags.remove(tag)
            self.updated_at = datetime.utcnow()
    
    def has_tag(self, tag: str) -> bool:
        """Check if profile has a specific tag."""
        return tag in self.tags
    
    def update_config(self, key: str, value: Any) -> None:
        """Update configuration key-value pair."""
        self.config[key] = value
        self.updated_at = datetime.utcnow()
    
    def get_target_name(self) -> str:
        """Get the full target name for the profile."""
        parts = []
        if self.schema_name:
            parts.append(self.schema_name)
        if self.table_name:
            parts.append(self.table_name)
        elif self.dataset_name:
            parts.append(self.dataset_name)
        
        return ".".join(parts)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert profile to dictionary."""
        return {
            "id": str(self.id),
            "dataset_name": self.dataset_name,
            "table_name": self.table_name,
            "schema_name": self.schema_name,
            "status": self.status.value,
            "version": self.version,
            "total_rows": self.total_rows,
            "total_columns": self.total_columns,
            "file_size_bytes": self.file_size_bytes,
            "column_profiles": [cp.to_dict() for cp in self.column_profiles],
            "completeness_score": self.completeness_score,
            "uniqueness_score": self.uniqueness_score,
            "validity_score": self.validity_score,
            "overall_quality_score": self.overall_quality_score,
            "primary_keys": self.primary_keys,
            "foreign_keys": self.foreign_keys,
            "relationships": self.relationships,
            "profiling_started_at": (
                self.profiling_started_at.isoformat()
                if self.profiling_started_at else None
            ),
            "profiling_completed_at": (
                self.profiling_completed_at.isoformat()
                if self.profiling_completed_at else None
            ),
            "profiling_duration_ms": self.profiling_duration_ms,
            "profiling_duration_seconds": self.profiling_duration_seconds,
            "sample_size": self.sample_size,
            "sampling_method": self.sampling_method,
            "created_at": self.created_at.isoformat(),
            "created_by": self.created_by,
            "updated_at": self.updated_at.isoformat(),
            "updated_by": self.updated_by,
            "config": self.config,
            "tags": self.tags,
            "is_completed": self.is_completed,
            "is_running": self.is_running,
            "has_quality_issues": self.has_quality_issues,
            "target_name": self.get_target_name(),
        }
    
    def __str__(self) -> str:
        """Human-readable string representation."""
        target = self.get_target_name()
        return f"DataProfile('{target}', rows={self.total_rows}, cols={self.total_columns})"
    
    def __repr__(self) -> str:
        """Developer string representation."""
        return (
            f"DataProfile(id={self.id}, dataset='{self.dataset_name}', "
            f"status={self.status.value}, quality={self.overall_quality_score:.2f})"
        )
