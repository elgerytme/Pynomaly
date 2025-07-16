"""DatasetProfile entity for comprehensive dataset analysis and profiling."""

from __future__ import annotations

from datetime import datetime
from typing import Any, Optional
from uuid import UUID, uuid4

from pydantic import Field, validator

from packages.core.domain.abstractions.base_entity import BaseEntity
from packages.data_science.domain.value_objects.data_distribution import DataDistribution
from packages.data_science.domain.value_objects.correlation_matrix import CorrelationMatrix


class DatasetProfile(BaseEntity):
    """Entity representing a comprehensive dataset profile.
    
    This entity captures detailed analysis and profiling information
    about a dataset including statistics, distributions, quality metrics,
    and data characteristics.
    
    Attributes:
        profile_id: Unique identifier for the dataset profile
        dataset_id: Identifier for the dataset being profiled
        dataset_name: Name of the dataset
        dataset_version: Version of the dataset
        profile_timestamp: When the profile was created
        
        # Basic dataset information
        total_rows: Total number of rows in the dataset
        total_columns: Total number of columns
        memory_usage_bytes: Memory usage in bytes
        file_size_bytes: File size in bytes
        data_types: Data types of each column
        
        # Column analysis
        column_profiles: Detailed profile for each column
        missing_values: Missing value analysis
        duplicate_analysis: Duplicate row analysis
        unique_values: Unique value counts per column
        
        # Statistical summaries
        numerical_summary: Statistical summary for numerical columns
        categorical_summary: Summary for categorical columns
        distributions: Distribution analysis for numerical columns
        correlation_matrix: Correlation analysis between features
        
        # Data quality metrics
        completeness_score: Overall data completeness (0-1)
        consistency_score: Data consistency score (0-1)
        validity_score: Data validity score (0-1)
        uniqueness_score: Data uniqueness score (0-1)
        accuracy_score: Data accuracy score (0-1)
        overall_quality_score: Overall data quality score (0-1)
        
        # Data quality issues
        quality_issues: List of identified data quality issues
        outliers: Outlier detection results
        anomalies: Anomaly detection results
        data_drift: Data drift detection results
        
        # Schema information
        schema_version: Version of the data schema
        schema_changes: Changes from previous schema version
        column_definitions: Detailed column definitions
        constraints: Data constraints and validation rules
        
        # Sampling information
        is_sample: Whether this profile is based on a sample
        sample_size: Size of the sample if applicable
        sampling_method: Method used for sampling
        sampling_parameters: Parameters for sampling
        
        # Business context
        business_rules: Business rules applied to the data
        domain_knowledge: Domain-specific insights
        use_case_suitability: Suitability for different use cases
        
        # Temporal analysis
        temporal_columns: Columns representing time/date
        temporal_patterns: Temporal patterns in the data
        seasonality: Seasonality analysis
        trend_analysis: Trend analysis results
        
        # Metadata
        tags: Tags for organization
        notes: Additional notes
        created_by: User who created the profile
        profiling_tool: Tool used for profiling
        profiling_config: Configuration used for profiling
        
        # Comparison and versioning
        previous_profile_id: Reference to previous profile version
        comparison_results: Comparison with previous profile
        change_summary: Summary of changes from previous version
        
        # Export and sharing
        export_formats: Available export formats
        sharing_permissions: Sharing permissions
        access_log: Access log for the profile
    """
    
    # Core identification
    profile_id: UUID = Field(default_factory=uuid4)
    dataset_id: str = Field(..., min_length=1)
    dataset_name: str = Field(..., min_length=1)
    dataset_version: str = Field(default="1.0.0")
    profile_timestamp: datetime = Field(default_factory=datetime.utcnow)
    
    # Basic dataset information
    total_rows: int = Field(..., ge=0)
    total_columns: int = Field(..., ge=0)
    memory_usage_bytes: Optional[int] = Field(None, ge=0)
    file_size_bytes: Optional[int] = Field(None, ge=0)
    data_types: dict[str, str] = Field(default_factory=dict)
    
    # Column analysis
    column_profiles: dict[str, dict[str, Any]] = Field(default_factory=dict)
    missing_values: dict[str, Any] = Field(default_factory=dict)
    duplicate_analysis: dict[str, Any] = Field(default_factory=dict)
    unique_values: dict[str, int] = Field(default_factory=dict)
    
    # Statistical summaries
    numerical_summary: dict[str, dict[str, float]] = Field(default_factory=dict)
    categorical_summary: dict[str, dict[str, Any]] = Field(default_factory=dict)
    distributions: dict[str, DataDistribution] = Field(default_factory=dict)
    correlation_matrix: Optional[CorrelationMatrix] = None
    
    # Data quality metrics
    completeness_score: float = Field(default=0.0, ge=0, le=1)
    consistency_score: float = Field(default=0.0, ge=0, le=1)
    validity_score: float = Field(default=0.0, ge=0, le=1)
    uniqueness_score: float = Field(default=0.0, ge=0, le=1)
    accuracy_score: float = Field(default=0.0, ge=0, le=1)
    overall_quality_score: float = Field(default=0.0, ge=0, le=1)
    
    # Data quality issues
    quality_issues: list[dict[str, Any]] = Field(default_factory=list)
    outliers: dict[str, Any] = Field(default_factory=dict)
    anomalies: dict[str, Any] = Field(default_factory=dict)
    data_drift: dict[str, Any] = Field(default_factory=dict)
    
    # Schema information
    schema_version: str = Field(default="1.0.0")
    schema_changes: list[dict[str, Any]] = Field(default_factory=list)
    column_definitions: dict[str, dict[str, Any]] = Field(default_factory=dict)
    constraints: dict[str, list[dict[str, Any]]] = Field(default_factory=dict)
    
    # Sampling information
    is_sample: bool = Field(default=False)
    sample_size: Optional[int] = Field(None, ge=0)
    sampling_method: Optional[str] = None
    sampling_parameters: dict[str, Any] = Field(default_factory=dict)
    
    # Business context
    business_rules: list[dict[str, Any]] = Field(default_factory=list)
    domain_knowledge: dict[str, Any] = Field(default_factory=dict)
    use_case_suitability: dict[str, float] = Field(default_factory=dict)
    
    # Temporal analysis
    temporal_columns: list[str] = Field(default_factory=list)
    temporal_patterns: dict[str, Any] = Field(default_factory=dict)
    seasonality: dict[str, Any] = Field(default_factory=dict)
    trend_analysis: dict[str, Any] = Field(default_factory=dict)
    
    # Metadata
    tags: list[str] = Field(default_factory=list)
    notes: str = Field(default="", max_length=5000)
    created_by: str = Field(..., min_length=1)
    profiling_tool: str = Field(default="monorepo")
    profiling_config: dict[str, Any] = Field(default_factory=dict)
    
    # Comparison and versioning
    previous_profile_id: Optional[UUID] = None
    comparison_results: dict[str, Any] = Field(default_factory=dict)
    change_summary: dict[str, Any] = Field(default_factory=dict)
    
    # Export and sharing
    export_formats: list[str] = Field(default_factory=list)
    sharing_permissions: dict[str, Any] = Field(default_factory=dict)
    access_log: list[dict[str, Any]] = Field(default_factory=list)
    
    @validator('overall_quality_score')
    def validate_overall_quality_score(cls, v: float, values: dict[str, Any]) -> float:
        """Auto-calculate overall quality score if not provided."""
        if v == 0.0:
            # Calculate as weighted average of individual scores
            completeness = values.get('completeness_score', 0.0)
            consistency = values.get('consistency_score', 0.0)
            validity = values.get('validity_score', 0.0)
            uniqueness = values.get('uniqueness_score', 0.0)
            accuracy = values.get('accuracy_score', 0.0)
            
            # Weighted average (completeness and validity are more important)
            weights = [0.25, 0.2, 0.25, 0.15, 0.15]  # Sum = 1.0
            scores = [completeness, consistency, validity, uniqueness, accuracy]
            
            return sum(score * weight for score, weight in zip(scores, weights))
        
        return v
    
    @validator('sample_size')
    def validate_sample_size(cls, v: Optional[int], values: dict[str, Any]) -> Optional[int]:
        """Validate sample size."""
        is_sample = values.get('is_sample', False)
        total_rows = values.get('total_rows', 0)
        
        if is_sample and v is None:
            raise ValueError("sample_size must be provided when is_sample is True")
        
        if v is not None and v > total_rows:
            raise ValueError("sample_size cannot exceed total_rows")
        
        return v
    
    def get_column_profile(self, column_name: str) -> Optional[dict[str, Any]]:
        """Get profile for a specific column."""
        return self.column_profiles.get(column_name)
    
    def get_missing_value_percentage(self, column_name: str) -> Optional[float]:
        """Get missing value percentage for a column."""
        column_profile = self.get_column_profile(column_name)
        if column_profile and 'missing_count' in column_profile:
            missing_count = column_profile['missing_count']
            return (missing_count / self.total_rows) * 100 if self.total_rows > 0 else 0
        return None
    
    def get_columns_by_type(self, data_type: str) -> list[str]:
        """Get columns of a specific data type."""
        return [col for col, dtype in self.data_types.items() if dtype == data_type]
    
    def get_numerical_columns(self) -> list[str]:
        """Get numerical columns."""
        numerical_types = {'int64', 'float64', 'int32', 'float32', 'number', 'integer', 'float'}
        return [col for col, dtype in self.data_types.items() if dtype in numerical_types]
    
    def get_categorical_columns(self) -> list[str]:
        """Get categorical columns."""
        categorical_types = {'object', 'string', 'category', 'categorical', 'text'}
        return [col for col, dtype in self.data_types.items() if dtype in categorical_types]
    
    def get_high_cardinality_columns(self, threshold: float = 0.8) -> list[str]:
        """Get columns with high cardinality (unique value ratio)."""
        high_cardinality = []
        
        for column, unique_count in self.unique_values.items():
            if self.total_rows > 0:
                cardinality_ratio = unique_count / self.total_rows
                if cardinality_ratio >= threshold:
                    high_cardinality.append(column)
        
        return high_cardinality
    
    def get_low_variance_columns(self, threshold: float = 0.01) -> list[str]:
        """Get columns with low variance."""
        low_variance = []
        
        for column, stats in self.numerical_summary.items():
            if 'std' in stats and stats['std'] < threshold:
                low_variance.append(column)
        
        return low_variance
    
    def get_columns_with_outliers(self) -> list[str]:
        """Get columns that contain outliers."""
        if not self.outliers:
            return []
        
        return list(self.outliers.keys())
    
    def get_quality_issues_by_severity(self, severity: str) -> list[dict[str, Any]]:
        """Get quality issues by severity level."""
        return [issue for issue in self.quality_issues if issue.get('severity') == severity]
    
    def get_critical_quality_issues(self) -> list[dict[str, Any]]:
        """Get critical quality issues."""
        return self.get_quality_issues_by_severity('critical')
    
    def add_quality_issue(self, issue_type: str, description: str, 
                         severity: str = 'medium', column: Optional[str] = None) -> None:
        """Add a quality issue to the profile."""
        issue = {
            'type': issue_type,
            'description': description,
            'severity': severity,
            'column': column,
            'timestamp': datetime.utcnow().isoformat()
        }
        
        self.quality_issues.append(issue)
    
    def calculate_data_density(self) -> float:
        """Calculate data density (non-missing values ratio)."""
        if self.total_rows == 0 or self.total_columns == 0:
            return 0.0
        
        total_cells = self.total_rows * self.total_columns
        
        # Calculate total missing values
        total_missing = 0
        for column_profile in self.column_profiles.values():
            if 'missing_count' in column_profile:
                total_missing += column_profile['missing_count']
        
        density = (total_cells - total_missing) / total_cells
        return max(0.0, min(1.0, density))
    
    def get_memory_usage_mb(self) -> Optional[float]:
        """Get memory usage in megabytes."""
        if self.memory_usage_bytes is None:
            return None
        return self.memory_usage_bytes / (1024 * 1024)
    
    def get_file_size_mb(self) -> Optional[float]:
        """Get file size in megabytes."""
        if self.file_size_bytes is None:
            return None
        return self.file_size_bytes / (1024 * 1024)
    
    def compare_with_previous(self, previous_profile: DatasetProfile) -> dict[str, Any]:
        """Compare this profile with a previous version."""
        if not isinstance(previous_profile, DatasetProfile):
            raise ValueError("Can only compare with another DatasetProfile")
        
        comparison = {
            "current_profile_id": str(self.profile_id),
            "previous_profile_id": str(previous_profile.profile_id),
            "comparison_timestamp": datetime.utcnow().isoformat(),
        }
        
        # Compare basic metrics
        comparison.update({
            "row_change": self.total_rows - previous_profile.total_rows,
            "column_change": self.total_columns - previous_profile.total_columns,
            "quality_score_change": self.overall_quality_score - previous_profile.overall_quality_score,
        })
        
        # Compare column changes
        current_columns = set(self.data_types.keys())
        previous_columns = set(previous_profile.data_types.keys())
        
        comparison.update({
            "new_columns": list(current_columns - previous_columns),
            "removed_columns": list(previous_columns - current_columns),
            "common_columns": list(current_columns & previous_columns),
        })
        
        # Compare data types
        type_changes = []
        for column in comparison["common_columns"]:
            current_type = self.data_types.get(column)
            previous_type = previous_profile.data_types.get(column)
            
            if current_type != previous_type:
                type_changes.append({
                    "column": column,
                    "previous_type": previous_type,
                    "current_type": current_type
                })
        
        comparison["type_changes"] = type_changes
        
        # Store comparison results
        self.comparison_results = comparison
        self.previous_profile_id = previous_profile.profile_id
        
        return comparison
    
    def get_suitability_score(self, use_case: str) -> Optional[float]:
        """Get suitability score for a specific use case."""
        return self.use_case_suitability.get(use_case)
    
    def set_suitability_score(self, use_case: str, score: float) -> None:
        """Set suitability score for a use case."""
        if not 0 <= score <= 1:
            raise ValueError("Suitability score must be between 0 and 1")
        
        self.use_case_suitability[use_case] = score
    
    def get_profile_summary(self) -> dict[str, Any]:
        """Get comprehensive profile summary."""
        summary = {
            "profile_id": str(self.profile_id),
            "dataset_name": self.dataset_name,
            "dataset_version": self.dataset_version,
            "total_rows": self.total_rows,
            "total_columns": self.total_columns,
            "memory_usage_mb": self.get_memory_usage_mb(),
            "data_density": self.calculate_data_density(),
            "overall_quality_score": self.overall_quality_score,
            "is_sample": self.is_sample,
            "profile_timestamp": self.profile_timestamp.isoformat(),
            "created_by": self.created_by,
        }
        
        # Add column type breakdown
        summary["column_types"] = {
            "numerical": len(self.get_numerical_columns()),
            "categorical": len(self.get_categorical_columns()),
            "temporal": len(self.temporal_columns),
        }
        
        # Add quality breakdown
        summary["quality_metrics"] = {
            "completeness": self.completeness_score,
            "consistency": self.consistency_score,
            "validity": self.validity_score,
            "uniqueness": self.uniqueness_score,
            "accuracy": self.accuracy_score,
        }
        
        # Add issues summary
        summary["quality_issues"] = {
            "total": len(self.quality_issues),
            "critical": len(self.get_critical_quality_issues()),
            "high": len(self.get_quality_issues_by_severity('high')),
            "medium": len(self.get_quality_issues_by_severity('medium')),
            "low": len(self.get_quality_issues_by_severity('low')),
        }
        
        # Add distribution summary
        if self.distributions:
            summary["distributions"] = {
                "fitted_distributions": len(self.distributions),
                "normal_distributions": len([
                    d for d in self.distributions.values() 
                    if d.is_normal_distribution()
                ]),
                "good_fits": len([
                    d for d in self.distributions.values() 
                    if d.is_good_fit()
                ])
            }
        
        return summary
    
    def to_dict(self) -> dict[str, Any]:
        """Convert profile to dictionary representation."""
        data = super().to_dict()
        
        # Convert UUIDs to strings for serialization
        data["profile_id"] = str(self.profile_id)
        if self.previous_profile_id:
            data["previous_profile_id"] = str(self.previous_profile_id)
        
        # Convert datetime objects to ISO format
        data["profile_timestamp"] = self.profile_timestamp.isoformat()
        
        # Convert distributions to dict format
        if self.distributions:
            data["distributions"] = {
                col: dist.to_dict() for col, dist in self.distributions.items()
            }
        
        # Convert correlation matrix to dict format
        if self.correlation_matrix:
            data["correlation_matrix"] = self.correlation_matrix.to_dict()
        
        return data
    
    def __str__(self) -> str:
        """String representation of the profile."""
        return f"DatasetProfile(id={self.profile_id}, dataset='{self.dataset_name}', rows={self.total_rows}, cols={self.total_columns})"
    
    def __repr__(self) -> str:
        """Detailed string representation."""
        return (
            f"DatasetProfile(profile_id={self.profile_id}, "
            f"dataset_name='{self.dataset_name}', "
            f"total_rows={self.total_rows}, total_columns={self.total_columns}, "
            f"quality_score={self.overall_quality_score:.3f}, "
            f"created_by='{self.created_by}')"
        )