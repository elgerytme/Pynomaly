"""Core data profiling domain entities."""

from __future__ import annotations
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Dict, List, Optional, Any, Union
from uuid import uuid4


class ProfilingStatus(str, Enum):
    """Status of a profiling operation."""
    PENDING = "pending"
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


class CardinalityLevel(str, Enum):
    """Cardinality level classification."""
    UNIQUE = "unique"        # > 95% unique values
    HIGH = "high"           # 50-95% unique values
    MEDIUM = "medium"       # 10-50% unique values
    LOW = "low"            # 2-10% unique values
    CONSTANT = "constant"   # Single unique value
    UNKNOWN = "unknown"     # Unable to determine cardinality


class DataType(str, Enum):
    """Standard data types."""
    INTEGER = "integer"
    FLOAT = "float"
    STRING = "string"
    BOOLEAN = "boolean"
    DATETIME = "datetime"
    DATE = "date"
    TIME = "time"
    CATEGORICAL = "categorical"
    JSON = "json"
    BINARY = "binary"
    UNKNOWN = "unknown"


class SemanticType(str, Enum):
    """Semantic data classification."""
    PII_EMAIL = "pii_email"
    PII_PHONE = "pii_phone"
    PII_SSN = "pii_ssn"
    PII_NAME = "pii_name"
    PII_ADDRESS = "pii_address"
    FINANCIAL_AMOUNT = "financial_amount"
    FINANCIAL_ACCOUNT = "financial_account"
    GEOGRAPHIC_COORDINATE = "geographic_coordinate"
    GEOGRAPHIC_LOCATION = "geographic_location"
    IDENTIFIER = "identifier"
    URL = "url"
    IP_ADDRESS = "ip_address"
    TIMESTAMP = "timestamp"
    CATEGORICAL = "categorical"
    MEASUREMENT = "measurement"
    COUNT = "count"
    UNKNOWN = "unknown"


@dataclass(frozen=True)
class ProfileId:
    """Profile identifier value object."""
    value: str = field(default_factory=lambda: str(uuid4()))

    def __str__(self) -> str:
        return self.value


@dataclass(frozen=True)
class DatasetId:
    """Dataset identifier value object."""
    value: str = field(default_factory=lambda: str(uuid4()))

    def __init__(self, value: str = None):
        object.__setattr__(self, 'value', value or str(uuid4()))

    def __str__(self) -> str:
        return self.value


@dataclass(frozen=True)
class StatisticalSummary:
    """Statistical summary for numerical data."""
    mean: Optional[float] = None
    median: Optional[float] = None
    mode: Optional[Union[float, str]] = None
    std_dev: Optional[float] = None
    variance: Optional[float] = None
    min_value: Optional[Union[float, str]] = None
    max_value: Optional[Union[float, str]] = None
    q1: Optional[float] = None
    q3: Optional[float] = None
    skewness: Optional[float] = None
    kurtosis: Optional[float] = None
    range_value: Optional[float] = None


@dataclass(frozen=True)
class ValueDistribution:
    """Value distribution analysis."""
    value_counts: Dict[str, int] = field(default_factory=dict)
    top_values: List[Dict[str, Any]] = field(default_factory=list)
    unique_count: int = 0
    null_count: int = 0
    total_count: int = 0
    cardinality_level: CardinalityLevel = CardinalityLevel.UNKNOWN
    statistical_summary: Optional[StatisticalSummary] = None


@dataclass(frozen=True)
class Pattern:
    """Data pattern discovered in a column."""
    pattern_type: str
    regex: str
    frequency: int
    percentage: float
    examples: List[str] = field(default_factory=list)
    confidence: float = 0.0
    description: str = ""


@dataclass(frozen=True)
class ColumnProfile:
    """Comprehensive profile of a single column."""
    column_name: str
    data_type: DataType
    nullable: bool = True
    unique_count: int = 0
    null_count: int = 0
    total_count: int = 0
    completeness_ratio: float = 0.0
    cardinality: CardinalityLevel = CardinalityLevel.UNKNOWN
    distribution: Optional[ValueDistribution] = None
    patterns: List[Pattern] = field(default_factory=list)
    semantic_type: Optional[SemanticType] = None
    inferred_constraints: List[str] = field(default_factory=list)
    quality_score: float = 0.0
    
    @property
    def uniqueness_ratio(self) -> float:
        """Calculate uniqueness ratio."""
        if self.total_count == 0:
            return 0.0
        return self.unique_count / self.total_count


@dataclass(frozen=True)
class TableRelationship:
    """Relationship between tables or columns."""
    source_table: str
    source_column: str
    target_table: str
    target_column: str
    relationship_type: str  # "foreign_key", "reference", "similarity"
    confidence: float = 0.0
    cardinality: str = "unknown"  # "one_to_one", "one_to_many", "many_to_many"


@dataclass(frozen=True)
class Constraint:
    """Database or data constraint."""
    constraint_type: str  # "primary_key", "foreign_key", "unique", "check", "not_null"
    table_name: str
    column_names: List[str]
    definition: str = ""
    is_enforced: bool = True


@dataclass(frozen=True)
class IndexInfo:
    """Index information."""
    index_name: str
    table_name: str
    column_names: List[str]
    index_type: str = "btree"
    is_unique: bool = False
    is_primary: bool = False
    size_bytes: Optional[int] = None


@dataclass(frozen=True)
class SizeMetrics:
    """Size and storage metrics."""
    total_rows: int = 0
    total_columns: int = 0
    estimated_size_bytes: Optional[int] = None
    compressed_size_bytes: Optional[int] = None
    average_row_size_bytes: Optional[float] = None
    storage_efficiency: Optional[float] = None


@dataclass(frozen=True)
class SchemaEvolution:
    """Schema evolution tracking."""
    previous_version: Optional[str] = None
    current_version: str = "1.0.0"
    changes_detected: List[str] = field(default_factory=list)
    breaking_changes: List[str] = field(default_factory=list)
    last_change_date: Optional[datetime] = None


@dataclass(frozen=True)
class SchemaProfile:
    """Comprehensive schema analysis results."""
    total_tables: int = 1
    total_columns: int = 0
    total_rows: int = 0
    columns: List[ColumnProfile] = field(default_factory=list)
    relationships: List[TableRelationship] = field(default_factory=list)
    constraints: List[Constraint] = field(default_factory=list)
    indexes: List[IndexInfo] = field(default_factory=list)
    size_metrics: Optional[SizeMetrics] = None
    schema_evolution: Optional[SchemaEvolution] = None
    primary_keys: List[str] = field(default_factory=list)
    foreign_keys: List[str] = field(default_factory=list)
    estimated_size_bytes: Optional[int] = None


@dataclass(frozen=True)
class QualityIssue:
    """Data quality issue identified during profiling."""
    issue_type: str
    severity: str  # "critical", "high", "medium", "low"
    description: str
    affected_columns: List[str] = field(default_factory=list)
    affected_rows: Optional[int] = None
    impact_percentage: float = 0.0
    suggested_actions: List[str] = field(default_factory=list)
    rule_violated: Optional[str] = None


@dataclass(frozen=True)
class QualityAssessment:
    """Comprehensive data quality assessment."""
    overall_score: float = 0.0
    completeness_score: float = 0.0
    consistency_score: float = 0.0
    accuracy_score: float = 0.0
    validity_score: float = 0.0
    uniqueness_score: float = 0.0
    timeliness_score: float = 0.0
    issues: List[QualityIssue] = field(default_factory=list)
    recommendations: List[str] = field(default_factory=list)
    critical_issues: int = 0
    high_issues: int = 0
    medium_issues: int = 0
    low_issues: int = 0
    
    @property
    def total_issues(self) -> int:
        """Total number of quality issues."""
        return self.critical_issues + self.high_issues + self.medium_issues + self.low_issues


@dataclass(frozen=True)
class ProfilingMetadata:
    """Metadata about the profiling operation."""
    profiling_strategy: str = "full"  # "full", "sample", "incremental"
    sample_size: Optional[int] = None
    sample_percentage: Optional[float] = None
    execution_time_seconds: float = 0.0
    memory_usage_mb: float = 0.0
    include_patterns: bool = True
    include_statistical_analysis: bool = True
    include_quality_assessment: bool = True
    engine_version: str = "1.0.0"
    configuration: Dict[str, Any] = field(default_factory=dict)


@dataclass
class DataProfile:
    """Main data profiling entity containing all analysis results."""
    
    profile_id: ProfileId = field(default_factory=ProfileId)
    dataset_id: DatasetId = field(default_factory=DatasetId)
    source_type: str = "unknown"
    source_connection: Dict[str, Any] = field(default_factory=dict)
    
    # Core profiles
    schema_profile: Optional[SchemaProfile] = None
    quality_assessment: Optional[QualityAssessment] = None
    profiling_metadata: Optional[ProfilingMetadata] = None
    
    # Status and timing
    status: ProfilingStatus = ProfilingStatus.PENDING
    created_at: datetime = field(default_factory=datetime.now)
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    error_message: Optional[str] = None
    
    # Additional metadata
    tags: List[str] = field(default_factory=list)
    notes: str = ""
    
    def start_profiling(self) -> None:
        """Mark profiling as started."""
        self.status = ProfilingStatus.IN_PROGRESS
        self.started_at = datetime.now()
    
    def complete_profiling(self, 
                          schema_profile: SchemaProfile,
                          quality_assessment: QualityAssessment,
                          metadata: ProfilingMetadata) -> None:
        """Mark profiling as completed with results."""
        self.status = ProfilingStatus.COMPLETED
        self.completed_at = datetime.now()
        self.schema_profile = schema_profile
        self.quality_assessment = quality_assessment
        self.profiling_metadata = metadata
    
    def fail_profiling(self, error_message: str) -> None:
        """Mark profiling as failed."""
        self.status = ProfilingStatus.FAILED
        self.completed_at = datetime.now()
        self.error_message = error_message
    
    def cancel_profiling(self) -> None:
        """Mark profiling as cancelled."""
        self.status = ProfilingStatus.CANCELLED
        self.completed_at = datetime.now()
    
    @property
    def execution_time_seconds(self) -> Optional[float]:
        """Calculate execution time if profiling is completed."""
        if self.started_at and self.completed_at:
            return (self.completed_at - self.started_at).total_seconds()
        return None
    
    @property
    def is_completed(self) -> bool:
        """Check if profiling is completed successfully."""
        return self.status == ProfilingStatus.COMPLETED
    
    @property
    def has_schema_profile(self) -> bool:
        """Check if schema profile is available."""
        return self.schema_profile is not None
    
    @property
    def has_quality_assessment(self) -> bool:
        """Check if quality assessment is available."""
        return self.quality_assessment is not None


@dataclass(frozen=True)
class ProfilingJob:
    """Job entity for tracking profiling operations."""
    
    job_id: str = field(default_factory=lambda: str(uuid4()))
    profile_id: ProfileId = field(default_factory=ProfileId)
    dataset_source: Dict[str, Any] = field(default_factory=dict)
    profiling_config: Dict[str, Any] = field(default_factory=dict)
    
    # Job lifecycle
    status: ProfilingStatus = ProfilingStatus.PENDING
    created_at: datetime = field(default_factory=datetime.now)
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    
    # Progress tracking
    progress_percentage: float = 0.0
    current_step: str = ""
    estimated_completion: Optional[datetime] = None
    
    # Results and errors
    result_profile: Optional[DataProfile] = None
    error_details: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    
    # Resource usage
    memory_usage_mb: float = 0.0
    cpu_usage_percentage: float = 0.0
    
    @property
    def execution_time_seconds(self) -> Optional[float]:
        """Calculate execution time."""
        if self.started_at and self.completed_at:
            return (self.completed_at - self.started_at).total_seconds()
        return None
    
    @property
    def is_running(self) -> bool:
        """Check if job is currently running."""
        return self.status == ProfilingStatus.IN_PROGRESS
    
    @property
    def is_finished(self) -> bool:
        """Check if job is finished (completed, failed, or cancelled)."""
        return self.status in [
            ProfilingStatus.COMPLETED, 
            ProfilingStatus.FAILED, 
            ProfilingStatus.CANCELLED
        ]