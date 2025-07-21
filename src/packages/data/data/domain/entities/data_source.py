"""Data source entity."""

from typing import Optional, Dict, Any, List
from uuid import UUID, uuid4
from datetime import datetime
from enum import Enum
from pydantic import Field, validator
from packages.core.domain.abstractions.base_entity import BaseEntity
from ..value_objects.data_classification import DataClassification


class SourceStatus(str, Enum):
    """Data source status."""
    ACTIVE = "active"
    INACTIVE = "inactive"
    DEPRECATED = "deprecated"
    MAINTENANCE = "maintenance"
    ERROR = "error"


class AccessPattern(str, Enum):
    """Data access patterns."""
    BATCH = "batch"
    STREAMING = "streaming"
    REAL_TIME = "real_time"
    ON_DEMAND = "on_demand"
    SCHEDULED = "scheduled"


class DataSource(BaseEntity):
    """Represents a specific data source within an origin."""
    
    source_id: UUID = Field(default_factory=uuid4, description="Unique source identifier")
    origin_id: UUID = Field(..., description="Reference to data origin")
    name: str = Field(..., min_length=1, max_length=200, description="Source name")
    description: Optional[str] = Field(None, description="Source description")
    source_path: str = Field(..., description="Path or identifier within origin")
    status: SourceStatus = Field(default=SourceStatus.ACTIVE, description="Source status")
    access_pattern: AccessPattern = Field(default=AccessPattern.BATCH, description="Data access pattern")
    classification: Optional[DataClassification] = Field(None, description="Data classification")
    format: Optional[str] = Field(None, description="Data format (CSV, JSON, etc.)")
    encoding: Optional[str] = Field(None, description="Character encoding")
    compression: Optional[str] = Field(None, description="Compression method")
    size_bytes: Optional[int] = Field(None, ge=0, description="Data size in bytes")
    record_count: Optional[int] = Field(None, ge=0, description="Number of records")
    last_modified_at: Optional[datetime] = Field(None, description="Last modification timestamp")
    last_validated_at: Optional[datetime] = Field(None, description="Last validation timestamp")
    validation_status: Optional[str] = Field(None, description="Last validation result")
    refresh_frequency: Optional[str] = Field(None, description="Data refresh frequency")
    retention_policy: Optional[str] = Field(None, description="Data retention policy")
    access_permissions: List[str] = Field(default_factory=list, description="Access permission roles")
    dependencies: List[UUID] = Field(default_factory=list, description="Dependent source IDs")
    downstream_consumers: List[str] = Field(default_factory=list, description="Consumer systems")
    quality_score: float = Field(default=1.0, ge=0.0, le=1.0, description="Data quality score")
    availability_score: float = Field(default=1.0, ge=0.0, le=1.0, description="Availability score")
    performance_metrics: Dict[str, Any] = Field(default_factory=dict, description="Performance metrics")
    error_log: List[Dict[str, Any]] = Field(default_factory=list, description="Error history")
    configuration: Dict[str, Any] = Field(default_factory=dict, description="Source configuration")
    tags: List[str] = Field(default_factory=list, description="Classification tags")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Additional metadata")
    
    @validator('name')
    def validate_name(cls, v: str) -> str:
        """Validate source name format."""
        if not v.strip():
            raise ValueError("Source name cannot be empty")
        return v.strip()
    
    @validator('source_path')
    def validate_source_path(cls, v: str) -> str:
        """Validate source path is not empty."""
        if not v.strip():
            raise ValueError("Source path cannot be empty")
        return v.strip()
    
    @validator('quality_score', 'availability_score')
    def validate_scores(cls, v: float) -> float:
        """Validate scores are between 0 and 1."""
        if not 0.0 <= v <= 1.0:
            raise ValueError("Score must be between 0.0 and 1.0")
        return v
    
    def activate(self) -> None:
        """Activate the data source."""
        self.status = SourceStatus.ACTIVE
        self.updated_at = datetime.utcnow()
    
    def deactivate(self, reason: Optional[str] = None) -> None:
        """Deactivate the data source."""
        self.status = SourceStatus.INACTIVE
        self.updated_at = datetime.utcnow()
        
        if reason:
            self.log_error("DEACTIVATION", reason)
    
    def mark_deprecated(self, replacement_source_id: Optional[UUID] = None) -> None:
        """Mark source as deprecated."""
        self.status = SourceStatus.DEPRECATED
        self.updated_at = datetime.utcnow()
        
        if replacement_source_id:
            self.metadata['replacement_source_id'] = str(replacement_source_id)
    
    def enter_maintenance(self, expected_duration: Optional[str] = None) -> None:
        """Put source into maintenance mode."""
        self.status = SourceStatus.MAINTENANCE
        self.updated_at = datetime.utcnow()
        
        if expected_duration:
            self.metadata['maintenance_duration'] = expected_duration
    
    def record_validation(self, is_valid: bool, details: Optional[str] = None) -> None:
        """Record validation result."""
        self.last_validated_at = datetime.utcnow()
        self.validation_status = "valid" if is_valid else "invalid"
        self.updated_at = datetime.utcnow()
        
        if details:
            self.metadata['last_validation_details'] = details
        
        if not is_valid:
            self.log_error("VALIDATION_FAILED", details or "Validation failed")
    
    def update_size(self, size_bytes: int, record_count: Optional[int] = None) -> None:
        """Update data size information."""
        self.size_bytes = max(0, size_bytes)
        if record_count is not None:
            self.record_count = max(0, record_count)
        self.last_modified_at = datetime.utcnow()
        self.updated_at = datetime.utcnow()
    
    def update_quality_score(self, score: float) -> None:
        """Update data quality score."""
        if not 0.0 <= score <= 1.0:
            raise ValueError("Quality score must be between 0.0 and 1.0")
        
        self.quality_score = score
        self.updated_at = datetime.utcnow()
    
    def update_availability_score(self, score: float) -> None:
        """Update availability score."""
        if not 0.0 <= score <= 1.0:
            raise ValueError("Availability score must be between 0.0 and 1.0")
        
        self.availability_score = score
        self.updated_at = datetime.utcnow()
    
    def log_error(self, error_type: str, message: str, details: Optional[Dict[str, Any]] = None) -> None:
        """Log an error for this source."""
        error_entry = {
            'timestamp': datetime.utcnow().isoformat(),
            'type': error_type,
            'message': message,
            'details': details or {}
        }
        
        self.error_log.append(error_entry)
        
        # Keep only last 100 errors
        if len(self.error_log) > 100:
            self.error_log = self.error_log[-100:]
        
        self.status = SourceStatus.ERROR
        self.updated_at = datetime.utcnow()
    
    def clear_errors(self) -> None:
        """Clear error log and restore active status."""
        self.error_log.clear()
        if self.status == SourceStatus.ERROR:
            self.status = SourceStatus.ACTIVE
        self.updated_at = datetime.utcnow()
    
    def is_available(self) -> bool:
        """Check if source is currently available."""
        return self.status == SourceStatus.ACTIVE
    
    def is_streaming(self) -> bool:
        """Check if this is a streaming data source."""
        return self.access_pattern in [AccessPattern.STREAMING, AccessPattern.REAL_TIME]
    
    def is_batch(self) -> bool:
        """Check if this is a batch data source."""
        return self.access_pattern == AccessPattern.BATCH
    
    def is_high_quality(self, threshold: float = 0.8) -> bool:
        """Check if source meets high quality threshold."""
        return self.quality_score >= threshold
    
    def is_highly_available(self, threshold: float = 0.95) -> bool:
        """Check if source meets high availability threshold."""
        return self.availability_score >= threshold
    
    def get_average_record_size(self) -> Optional[float]:
        """Calculate average record size in bytes."""
        if self.size_bytes is not None and self.record_count and self.record_count > 0:
            return self.size_bytes / self.record_count
        return None
    
    def requires_special_access(self) -> bool:
        """Check if source requires special access permissions."""
        return bool(self.access_permissions) or (
            self.classification and self.classification.requires_access_controls()
        )