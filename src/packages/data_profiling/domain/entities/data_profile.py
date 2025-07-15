"""Data Profile entity for automated data profiling operations."""

from dataclasses import dataclass
from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional
from uuid import UUID, uuid4

from pydantic import BaseModel, Field


@dataclass(frozen=True)
class ProfileId:
    """Data Profile identifier."""
    value: UUID = Field(default_factory=uuid4)


@dataclass(frozen=True)
class DatasetId:
    """Dataset identifier."""
    value: UUID = Field(default_factory=uuid4)


class ProfilingStatus(str, Enum):
    """Profiling job status enumeration."""
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


class DataType(str, Enum):
    """Data type enumeration."""
    STRING = "string"
    INTEGER = "integer"
    FLOAT = "float"
    BOOLEAN = "boolean"
    DATETIME = "datetime"
    DATE = "date"
    TIME = "time"
    CATEGORICAL = "categorical"
    UNKNOWN = "unknown"


class CardinalityLevel(str, Enum):
    """Cardinality level enumeration."""
    LOW = "low"        # < 10 unique values
    MEDIUM = "medium"  # 10-100 unique values
    HIGH = "high"      # 100-1000 unique values
    VERY_HIGH = "very_high"  # > 1000 unique values


class PatternType(str, Enum):
    """Pattern type enumeration."""
    EMAIL = "email"
    PHONE = "phone"
    URL = "url"
    DATE = "date"
    TIME = "time"
    NUMERIC = "numeric"
    ALPHANUMERIC = "alphanumeric"
    CUSTOM = "custom"


class QualityIssueType(str, Enum):
    """Quality issue type enumeration."""
    MISSING_VALUES = "missing_values"
    DUPLICATE_VALUES = "duplicate_values"
    INVALID_FORMAT = "invalid_format"
    OUTLIERS = "outliers"
    INCONSISTENT_VALUES = "inconsistent_values"
    REFERENTIAL_INTEGRITY = "referential_integrity"


class ValueDistribution(BaseModel):
    """Value distribution statistics."""
    unique_count: int
    null_count: int
    total_count: int
    completeness_ratio: float
    top_values: Dict[str, int] = Field(default_factory=dict)
    
    class Config:
        frozen = True


class StatisticalSummary(BaseModel):
    """Statistical summary for numerical data."""
    min_value: Optional[float] = None
    max_value: Optional[float] = None
    mean: Optional[float] = None
    median: Optional[float] = None
    std_dev: Optional[float] = None
    quartiles: Optional[List[float]] = None
    
    class Config:
        frozen = True


class Pattern(BaseModel):
    """Detected pattern in data."""
    pattern_type: PatternType
    regex: str
    frequency: int
    percentage: float
    examples: List[str] = Field(default_factory=list)
    confidence: float = 0.0
    
    class Config:
        frozen = True


class QualityIssue(BaseModel):
    """Data quality issue."""
    issue_type: QualityIssueType
    severity: str  # "low", "medium", "high", "critical"
    description: str
    affected_rows: int
    affected_percentage: float
    examples: List[str] = Field(default_factory=list)
    suggested_action: Optional[str] = None
    
    class Config:
        frozen = True


class ColumnProfile(BaseModel):
    """Profile for a single column."""
    column_name: str
    data_type: DataType
    inferred_type: Optional[DataType] = None
    nullable: bool = True
    
    # Distribution analysis
    distribution: ValueDistribution
    cardinality: CardinalityLevel
    
    # Statistical analysis (for numerical columns)
    statistical_summary: Optional[StatisticalSummary] = None
    
    # Pattern analysis
    patterns: List[Pattern] = Field(default_factory=list)
    
    # Quality assessment
    quality_score: float = 0.0
    quality_issues: List[QualityIssue] = Field(default_factory=list)
    
    # Semantic information
    semantic_type: Optional[str] = None  # e.g., "email", "phone", "address"
    business_meaning: Optional[str] = None
    
    class Config:
        frozen = True


class SchemaProfile(BaseModel):
    """Schema-level profile information."""
    table_name: str
    total_columns: int
    total_rows: int
    columns: List[ColumnProfile]
    
    # Relationships
    primary_keys: List[str] = Field(default_factory=list)
    foreign_keys: Dict[str, str] = Field(default_factory=dict)  # column -> referenced_table.column
    
    # Constraints
    unique_constraints: List[List[str]] = Field(default_factory=list)
    check_constraints: List[str] = Field(default_factory=list)
    
    # Size metrics
    estimated_size_bytes: Optional[int] = None
    compression_ratio: Optional[float] = None
    
    class Config:
        frozen = True


class QualityAssessment(BaseModel):
    """Overall quality assessment."""
    overall_score: float
    completeness_score: float
    consistency_score: float
    accuracy_score: float
    validity_score: float
    uniqueness_score: float
    
    # Quality dimensions weights
    dimension_weights: Dict[str, float] = Field(default_factory=dict)
    
    # Issues summary
    critical_issues: int = 0
    high_issues: int = 0
    medium_issues: int = 0
    low_issues: int = 0
    
    # Recommendations
    recommendations: List[str] = Field(default_factory=list)
    
    class Config:
        frozen = True


class ProfilingMetadata(BaseModel):
    """Profiling execution metadata."""
    profiling_strategy: str  # "full", "sample", "incremental"
    sample_size: Optional[int] = None
    sample_percentage: Optional[float] = None
    execution_time_seconds: float
    memory_usage_mb: Optional[float] = None
    
    # Configuration
    include_patterns: bool = True
    include_statistical_analysis: bool = True
    include_quality_assessment: bool = True
    
    class Config:
        frozen = True


class DataProfile(BaseModel):
    """Data profiling aggregate root."""
    
    profile_id: ProfileId
    dataset_id: DatasetId
    status: ProfilingStatus = ProfilingStatus.PENDING
    
    # Profile components
    schema_profile: Optional[SchemaProfile] = None
    quality_assessment: Optional[QualityAssessment] = None
    profiling_metadata: Optional[ProfilingMetadata] = None
    
    # Execution tracking
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    error_message: Optional[str] = None
    
    # Data source information
    source_type: str  # "database", "file", "stream", etc.
    source_connection: Dict[str, Any] = Field(default_factory=dict)
    source_query: Optional[str] = None
    
    # Audit fields
    created_at: datetime = Field(default_factory=datetime.utcnow)
    updated_at: datetime = Field(default_factory=datetime.utcnow)
    version: int = 1
    
    class Config:
        use_enum_values = True
        
    def start_profiling(self) -> None:
        """Start the profiling process."""
        self.status = ProfilingStatus.RUNNING
        self.started_at = datetime.utcnow()
        self.updated_at = datetime.utcnow()
        
    def complete_profiling(self, 
                          schema_profile: SchemaProfile,
                          quality_assessment: QualityAssessment,
                          metadata: ProfilingMetadata) -> None:
        """Complete the profiling with results."""
        self.status = ProfilingStatus.COMPLETED
        self.completed_at = datetime.utcnow()
        self.schema_profile = schema_profile
        self.quality_assessment = quality_assessment
        self.profiling_metadata = metadata
        self.updated_at = datetime.utcnow()
        self.version += 1
        
    def fail_profiling(self, error_message: str) -> None:
        """Mark profiling as failed."""
        self.status = ProfilingStatus.FAILED
        self.error_message = error_message
        self.completed_at = datetime.utcnow()
        self.updated_at = datetime.utcnow()
        self.version += 1
        
    def cancel_profiling(self) -> None:
        """Cancel the profiling process."""
        self.status = ProfilingStatus.CANCELLED
        self.completed_at = datetime.utcnow()
        self.updated_at = datetime.utcnow()
        self.version += 1
        
    def is_completed(self) -> bool:
        """Check if profiling is completed."""
        return self.status == ProfilingStatus.COMPLETED
        
    def is_failed(self) -> bool:
        """Check if profiling failed."""
        return self.status == ProfilingStatus.FAILED
        
    def is_running(self) -> bool:
        """Check if profiling is running."""
        return self.status == ProfilingStatus.RUNNING
        
    def get_column_profile(self, column_name: str) -> Optional[ColumnProfile]:
        """Get profile for a specific column."""
        if not self.schema_profile:
            return None
            
        for column in self.schema_profile.columns:
            if column.column_name == column_name:
                return column
        return None
        
    def get_quality_issues_by_severity(self, severity: str) -> List[QualityIssue]:
        """Get quality issues by severity level."""
        if not self.schema_profile:
            return []
            
        issues = []
        for column in self.schema_profile.columns:
            for issue in column.quality_issues:
                if issue.severity == severity:
                    issues.append(issue)
        return issues
        
    def get_columns_by_pattern(self, pattern_type: PatternType) -> List[str]:
        """Get columns that match a specific pattern type."""
        if not self.schema_profile:
            return []
            
        matching_columns = []
        for column in self.schema_profile.columns:
            for pattern in column.patterns:
                if pattern.pattern_type == pattern_type:
                    matching_columns.append(column.column_name)
                    break
        return matching_columns