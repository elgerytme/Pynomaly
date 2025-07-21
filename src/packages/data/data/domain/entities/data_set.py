"""Data set entity."""

from typing import Optional, Dict, Any, List
from uuid import UUID, uuid4
from datetime import datetime
from enum import Enum
from pydantic import Field, validator
from packages.core.domain.abstractions.base_entity import BaseEntity
from ..value_objects.data_schema import DataSchema
from ..value_objects.data_classification import DataClassification, DataQualityDimension


class DataSetType(str, Enum):
    """Types of data sets."""
    TRANSACTIONAL = "transactional"
    ANALYTICAL = "analytical"
    REFERENCE = "reference"
    MASTER = "master"
    STAGING = "staging"
    AGGREGATED = "aggregated"
    REAL_TIME = "real_time"
    BATCH = "batch"
    SNAPSHOT = "snapshot"
    DELTA = "delta"


class DataSetStatus(str, Enum):
    """Data set status."""
    ACTIVE = "active"
    INACTIVE = "inactive"
    LOADING = "loading"
    VALIDATING = "validating"
    ERROR = "error"
    ARCHIVED = "archived"


class DataSet(BaseEntity):
    """Represents a specific collection of structured data."""
    
    dataset_id: UUID = Field(default_factory=uuid4, description="Unique dataset identifier")
    asset_id: UUID = Field(..., description="Parent data asset reference")
    source_id: Optional[UUID] = Field(None, description="Data source reference")
    name: str = Field(..., min_length=1, max_length=200, description="Dataset name")
    description: Optional[str] = Field(None, description="Dataset description")
    dataset_type: DataSetType = Field(..., description="Type of dataset")
    status: DataSetStatus = Field(default=DataSetStatus.ACTIVE, description="Dataset status")
    schema: DataSchema = Field(..., description="Dataset schema definition")
    classification: Optional[DataClassification] = Field(None, description="Data classification")
    partition_columns: List[str] = Field(default_factory=list, description="Partitioning columns")
    sort_columns: List[str] = Field(default_factory=list, description="Sort order columns")
    record_count: int = Field(default=0, ge=0, description="Number of records")
    size_bytes: int = Field(default=0, ge=0, description="Dataset size in bytes")
    column_count: int = Field(default=0, ge=0, description="Number of columns")
    null_record_count: int = Field(default=0, ge=0, description="Records with null values")
    duplicate_record_count: int = Field(default=0, ge=0, description="Duplicate records")
    data_freshness_hours: Optional[float] = Field(None, ge=0, description="Data age in hours")
    load_timestamp: Optional[datetime] = Field(None, description="Last load timestamp")
    validation_timestamp: Optional[datetime] = Field(None, description="Last validation timestamp")
    validation_results: Dict[str, Any] = Field(default_factory=dict, description="Validation results")
    quality_metrics: Dict[DataQualityDimension, float] = Field(default_factory=dict, description="Quality metrics")
    quality_score: float = Field(default=1.0, ge=0.0, le=1.0, description="Overall quality score")
    profile_statistics: Dict[str, Any] = Field(default_factory=dict, description="Data profiling stats")
    lineage_info: Dict[str, Any] = Field(default_factory=dict, description="Lineage information")
    processing_history: List[Dict[str, Any]] = Field(default_factory=list, description="Processing history")
    error_log: List[Dict[str, Any]] = Field(default_factory=list, description="Error history")
    access_log: List[Dict[str, Any]] = Field(default_factory=list, description="Access history")
    retention_days: Optional[int] = Field(None, ge=0, description="Retention period in days")
    compression_ratio: Optional[float] = Field(None, ge=0, description="Compression ratio")
    indexing_info: Dict[str, Any] = Field(default_factory=dict, description="Index information")
    sample_data: List[Dict[str, Any]] = Field(default_factory=list, description="Sample records")
    tags: List[str] = Field(default_factory=list, description="Dataset tags")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Additional metadata")
    
    @validator('name')
    def validate_name(cls, v: str) -> str:
        """Validate dataset name format."""
        if not v.strip():
            raise ValueError("Dataset name cannot be empty")
        return v.strip()
    
    @validator('quality_score')
    def validate_quality_score(cls, v: float) -> float:
        """Validate quality score is between 0 and 1."""
        if not 0.0 <= v <= 1.0:
            raise ValueError("Quality score must be between 0.0 and 1.0")
        return v
    
    @validator('column_count')
    def validate_column_count_matches_schema(cls, v: int, values: Dict[str, Any]) -> int:
        """Validate column count matches schema if provided."""
        schema = values.get('schema')
        if schema and len(schema.fields) != v:
            return len(schema.fields)
        return v
    
    def load_data(self, record_count: int, size_bytes: int) -> None:
        """Record successful data load."""
        self.status = DataSetStatus.LOADING
        self.record_count = max(0, record_count)
        self.size_bytes = max(0, size_bytes)
        self.load_timestamp = datetime.utcnow()
        self.updated_at = datetime.utcnow()
        
        self._log_processing("DATA_LOADED", f"Loaded {record_count} records, {size_bytes} bytes")
    
    def complete_load(self) -> None:
        """Mark data load as complete."""
        if self.status == DataSetStatus.LOADING:
            self.status = DataSetStatus.ACTIVE
            self.updated_at = datetime.utcnow()
            self._log_processing("LOAD_COMPLETED", "Data load completed successfully")
    
    def validate_data(self, validation_results: Dict[str, Any]) -> None:
        """Record data validation results."""
        self.status = DataSetStatus.VALIDATING
        self.validation_results = validation_results
        self.validation_timestamp = datetime.utcnow()
        self.updated_at = datetime.utcnow()
        
        # Update quality metrics from validation
        if 'quality_metrics' in validation_results:
            self.quality_metrics.update(validation_results['quality_metrics'])
        
        # Calculate overall quality score
        self._calculate_quality_score()
        
        self._log_processing("DATA_VALIDATED", "Data validation completed", validation_results)
    
    def complete_validation(self, is_valid: bool) -> None:
        """Complete data validation process."""
        if is_valid:
            self.status = DataSetStatus.ACTIVE
            self._log_processing("VALIDATION_PASSED", "Data validation passed")
        else:
            self.status = DataSetStatus.ERROR
            self._log_processing("VALIDATION_FAILED", "Data validation failed")
        
        self.updated_at = datetime.utcnow()
    
    def update_profile_statistics(self, statistics: Dict[str, Any]) -> None:
        """Update data profiling statistics."""
        self.profile_statistics = statistics
        self.updated_at = datetime.utcnow()
        
        # Update derived metrics
        if 'null_count' in statistics:
            self.null_record_count = statistics['null_count']
        if 'duplicate_count' in statistics:
            self.duplicate_record_count = statistics['duplicate_count']
        
        self._log_processing("PROFILE_UPDATED", "Data profile statistics updated")
    
    def update_quality_metrics(self, metrics: Dict[DataQualityDimension, float]) -> None:
        """Update quality metrics."""
        self.quality_metrics.update(metrics)
        self._calculate_quality_score()
        self.updated_at = datetime.utcnow()
        
        self._log_processing("QUALITY_METRICS_UPDATED", "Quality metrics updated", dict(metrics))
    
    def record_access(self, user: str, access_type: str, details: Optional[Dict[str, Any]] = None) -> None:
        """Record data access."""
        access_entry = {
            'timestamp': datetime.utcnow().isoformat(),
            'user': user,
            'access_type': access_type,
            'details': details or {}
        }
        
        self.access_log.append(access_entry)
        
        # Keep only last 1000 access logs
        if len(self.access_log) > 1000:
            self.access_log = self.access_log[-1000:]
        
        self.updated_at = datetime.utcnow()
    
    def log_error(self, error_type: str, message: str, details: Optional[Dict[str, Any]] = None) -> None:
        """Log an error for this dataset."""
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
        
        self.status = DataSetStatus.ERROR
        self.updated_at = datetime.utcnow()
    
    def clear_errors(self) -> None:
        """Clear error log and restore active status."""
        self.error_log.clear()
        if self.status == DataSetStatus.ERROR:
            self.status = DataSetStatus.ACTIVE
        self.updated_at = datetime.utcnow()
        
        self._log_processing("ERRORS_CLEARED", "Error log cleared")
    
    def archive(self, reason: Optional[str] = None) -> None:
        """Archive the dataset."""
        self.status = DataSetStatus.ARCHIVED
        self.updated_at = datetime.utcnow()
        
        self._log_processing("ARCHIVED", reason or "Dataset archived")
    
    def _calculate_quality_score(self) -> None:
        """Calculate overall quality score from metrics."""
        if not self.quality_metrics:
            self.quality_score = 1.0
            return
        
        # Use classification requirements if available
        if self.classification and self.classification.quality_requirements:
            self.quality_score = self.classification.get_quality_score(dict(self.quality_metrics))
        else:
            # Simple average of available metrics
            total_score = sum(self.quality_metrics.values())
            self.quality_score = total_score / len(self.quality_metrics)
    
    def _log_processing(self, operation: str, description: str, details: Optional[Dict[str, Any]] = None) -> None:
        """Log a processing operation."""
        processing_entry = {
            'timestamp': datetime.utcnow().isoformat(),
            'operation': operation,
            'description': description,
            'details': details or {}
        }
        
        self.processing_history.append(processing_entry)
        
        # Keep only last 100 processing entries
        if len(self.processing_history) > 100:
            self.processing_history = self.processing_history[-100:]
    
    def is_active(self) -> bool:
        """Check if dataset is currently active."""
        return self.status == DataSetStatus.ACTIVE
    
    def has_errors(self) -> bool:
        """Check if dataset has errors."""
        return self.status == DataSetStatus.ERROR or bool(self.error_log)
    
    def is_high_quality(self, threshold: float = 0.8) -> bool:
        """Check if dataset meets high quality threshold."""
        return self.quality_score >= threshold
    
    def is_fresh(self, max_hours: float = 24.0) -> bool:
        """Check if data is fresh within specified hours."""
        if self.data_freshness_hours is None:
            return True
        return self.data_freshness_hours <= max_hours
    
    def get_completeness_score(self) -> float:
        """Calculate data completeness score."""
        if self.record_count == 0:
            return 0.0
        
        complete_records = self.record_count - self.null_record_count
        return complete_records / self.record_count
    
    def get_uniqueness_score(self) -> float:
        """Calculate data uniqueness score."""
        if self.record_count == 0:
            return 1.0
        
        unique_records = self.record_count - self.duplicate_record_count
        return unique_records / self.record_count
    
    def get_average_record_size(self) -> float:
        """Calculate average record size in bytes."""
        if self.record_count == 0:
            return 0.0
        return self.size_bytes / self.record_count
    
    def requires_cleanup(self) -> bool:
        """Check if dataset requires data cleanup."""
        return (
            self.duplicate_record_count > 0 or
            self.null_record_count > (self.record_count * 0.1) or  # More than 10% null records
            self.quality_score < 0.7
        )