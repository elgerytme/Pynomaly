"""Data Transfer Objects for configuration management."""

from __future__ import annotations

from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional, Union
from uuid import UUID, uuid4

from pydantic import BaseModel, Field, ConfigDict


# ============================================================================
# Enums and Constants
# ============================================================================

class ConfigurationSource(str, Enum):
    """Sources of configuration capture."""
    AUTOML = "automl"
    AUTONOMOUS = "autonomous"
    CLI = "cli"
    WEB_API = "web_api"
    WEB_UI = "web_ui"
    TEST = "test"
    MANUAL = "manual"
    TEMPLATE = "template"


class ConfigurationStatus(str, Enum):
    """Configuration lifecycle status."""
    DRAFT = "draft"
    ACTIVE = "active"
    COMPLETED = "completed"
    FAILED = "failed"
    ARCHIVED = "archived"
    DEPRECATED = "deprecated"


class ExportFormat(str, Enum):
    """Supported export formats."""
    YAML = "yaml"
    JSON = "json"
    PYTHON = "python"
    NOTEBOOK = "notebook"
    DOCKER = "docker"
    CONFIG_INI = "config_ini"


class ConfigurationLevel(str, Enum):
    """Configuration complexity levels."""
    BASIC = "basic"
    INTERMEDIATE = "intermediate"
    ADVANCED = "advanced"
    EXPERT = "expert"


# ============================================================================
# Core Configuration Components
# ============================================================================

class DatasetConfigDTO(BaseModel):
    """Configuration for dataset handling."""
    model_config = ConfigDict(from_attributes=True)
    
    # Dataset identification
    dataset_path: Optional[str] = Field(default=None, description="Path to dataset file")
    dataset_name: Optional[str] = Field(default=None, description="Dataset name identifier")
    dataset_id: Optional[UUID] = Field(default=None, description="Dataset UUID")
    
    # Data loading configuration
    file_format: Optional[str] = Field(default=None, description="File format (csv, parquet, json)")
    delimiter: Optional[str] = Field(default=",", description="CSV delimiter")
    encoding: Optional[str] = Field(default="utf-8", description="File encoding")
    header_row: Optional[int] = Field(default=0, description="Header row index")
    
    # Data processing
    feature_columns: Optional[List[str]] = Field(default=None, description="Feature column names")
    target_column: Optional[str] = Field(default=None, description="Target column name")
    exclude_columns: Optional[List[str]] = Field(default=None, description="Columns to exclude")
    datetime_columns: Optional[List[str]] = Field(default=None, description="Datetime column names")
    
    # Data validation
    expected_shape: Optional[tuple] = Field(default=None, description="Expected dataset shape")
    required_columns: Optional[List[str]] = Field(default=None, description="Required column names")
    data_types: Optional[Dict[str, str]] = Field(default=None, description="Expected data types")
    
    # Sampling configuration
    sample_size: Optional[int] = Field(default=None, description="Sample size for large datasets")
    sampling_method: Optional[str] = Field(default="random", description="Sampling method")
    random_seed: Optional[int] = Field(default=42, description="Random seed for reproducibility")


class AlgorithmConfigDTO(BaseModel):
    """Configuration for anomaly detection algorithms."""
    model_config = ConfigDict(from_attributes=True)
    
    # Algorithm selection
    algorithm_name: str = Field(..., description="Algorithm name")
    algorithm_family: Optional[str] = Field(default=None, description="Algorithm family (statistical, distance, etc.)")
    algorithm_version: Optional[str] = Field(default=None, description="Algorithm version")
    
    # Hyperparameters
    hyperparameters: Dict[str, Any] = Field(default_factory=dict, description="Algorithm hyperparameters")
    parameter_source: Optional[str] = Field(default="default", description="Parameter source (default, optimized, manual)")
    optimization_method: Optional[str] = Field(default=None, description="Hyperparameter optimization method")
    
    # Model configuration
    contamination: Optional[float] = Field(default=0.1, ge=0, le=0.5, description="Expected contamination rate")
    random_state: Optional[int] = Field(default=42, description="Random state for reproducibility")
    n_jobs: Optional[int] = Field(default=1, description="Number of parallel jobs")
    
    # Performance constraints
    max_training_time: Optional[float] = Field(default=None, description="Maximum training time in seconds")
    max_memory_usage: Optional[float] = Field(default=None, description="Maximum memory usage in MB")
    
    # Ensemble configuration
    is_ensemble: bool = Field(default=False, description="Whether this is an ensemble configuration")
    ensemble_method: Optional[str] = Field(default=None, description="Ensemble combination method")
    base_algorithms: Optional[List[str]] = Field(default=None, description="Base algorithms for ensemble")


class PreprocessingConfigDTO(BaseModel):
    """Configuration for data preprocessing."""
    model_config = ConfigDict(from_attributes=True)
    
    # Missing value handling
    missing_value_strategy: Optional[str] = Field(default="mean", description="Missing value imputation strategy")
    missing_value_threshold: Optional[float] = Field(default=0.5, description="Threshold for dropping columns with missing values")
    
    # Outlier handling
    outlier_detection_method: Optional[str] = Field(default="iqr", description="Outlier detection method")
    outlier_threshold: Optional[float] = Field(default=1.5, description="Outlier detection threshold")
    outlier_treatment: Optional[str] = Field(default="remove", description="Outlier treatment method")
    
    # Feature scaling
    scaling_method: Optional[str] = Field(default="standard", description="Feature scaling method")
    scaling_robust: bool = Field(default=False, description="Use robust scaling")
    
    # Feature selection
    feature_selection_method: Optional[str] = Field(default=None, description="Feature selection method")
    max_features: Optional[int] = Field(default=None, description="Maximum number of features")
    feature_importance_threshold: Optional[float] = Field(default=None, description="Feature importance threshold")
    
    # Categorical encoding
    categorical_encoding: Optional[str] = Field(default="label", description="Categorical encoding method")
    high_cardinality_threshold: Optional[int] = Field(default=10, description="High cardinality threshold")
    
    # Advanced preprocessing
    apply_pca: bool = Field(default=False, description="Apply PCA dimensionality reduction")
    pca_components: Optional[int] = Field(default=None, description="Number of PCA components")
    remove_duplicates: bool = Field(default=True, description="Remove duplicate rows")
    normalize_data: bool = Field(default=False, description="Apply data normalization")


class EvaluationConfigDTO(BaseModel):
    """Configuration for model evaluation."""
    model_config = ConfigDict(from_attributes=True)
    
    # Evaluation metrics
    primary_metric: str = Field(default="roc_auc", description="Primary evaluation metric")
    secondary_metrics: List[str] = Field(default_factory=list, description="Additional evaluation metrics")
    
    # Cross-validation
    cv_folds: int = Field(default=5, ge=2, le=20, description="Number of cross-validation folds")
    cv_strategy: str = Field(default="stratified", description="Cross-validation strategy")
    cv_random_state: int = Field(default=42, description="Random state for CV splits")
    
    # Test configuration
    test_size: Optional[float] = Field(default=0.2, ge=0.1, le=0.5, description="Test set size")
    validation_size: Optional[float] = Field(default=0.2, ge=0.1, le=0.5, description="Validation set size")
    
    # Performance thresholds
    min_accuracy: Optional[float] = Field(default=None, description="Minimum required accuracy")
    max_false_positive_rate: Optional[float] = Field(default=None, description="Maximum allowed false positive rate")
    min_recall: Optional[float] = Field(default=None, description="Minimum required recall")
    
    # Evaluation options
    calculate_feature_importance: bool = Field(default=True, description="Calculate feature importance")
    generate_plots: bool = Field(default=True, description="Generate evaluation plots")
    save_predictions: bool = Field(default=False, description="Save model predictions")
    detailed_results: bool = Field(default=True, description="Include detailed evaluation results")


class EnvironmentConfigDTO(BaseModel):
    """Configuration for execution environment."""
    model_config = ConfigDict(from_attributes=True)
    
    # Python environment
    python_version: Optional[str] = Field(default=None, description="Python version")
    dependencies: Optional[Dict[str, str]] = Field(default=None, description="Package dependencies")
    
    # Hardware configuration
    cpu_count: Optional[int] = Field(default=None, description="Number of CPU cores")
    memory_gb: Optional[float] = Field(default=None, description="Available memory in GB")
    gpu_available: bool = Field(default=False, description="GPU availability")
    gpu_model: Optional[str] = Field(default=None, description="GPU model")
    
    # Execution constraints
    max_execution_time: Optional[float] = Field(default=None, description="Maximum execution time in seconds")
    memory_limit_mb: Optional[float] = Field(default=None, description="Memory limit in MB")
    disk_space_gb: Optional[float] = Field(default=None, description="Available disk space in GB")
    
    # Platform information
    operating_system: Optional[str] = Field(default=None, description="Operating system")
    architecture: Optional[str] = Field(default=None, description="System architecture")
    container_runtime: Optional[str] = Field(default=None, description="Container runtime (docker, podman)")


# ============================================================================
# Configuration Metadata and Lineage
# ============================================================================

class ConfigurationMetadataDTO(BaseModel):
    """Metadata for configuration tracking."""
    model_config = ConfigDict(from_attributes=True)
    
    # Origin information
    created_by: Optional[str] = Field(default=None, description="User who created the configuration")
    created_at: datetime = Field(default_factory=datetime.now, description="Creation timestamp")
    source: ConfigurationSource = Field(..., description="Configuration source")
    source_version: Optional[str] = Field(default=None, description="Source version")
    
    # Classification
    tags: List[str] = Field(default_factory=list, description="Configuration tags")
    category: Optional[str] = Field(default=None, description="Configuration category")
    complexity_level: ConfigurationLevel = Field(default=ConfigurationLevel.BASIC, description="Complexity level")
    
    # Versioning
    version: str = Field(default="1.0.0", description="Configuration version")
    parent_id: Optional[UUID] = Field(default=None, description="Parent configuration ID")
    derived_from: Optional[List[UUID]] = Field(default=None, description="Configurations this was derived from")
    
    # Usage tracking
    usage_count: int = Field(default=0, description="Number of times configuration was used")
    last_used: Optional[datetime] = Field(default=None, description="Last usage timestamp")
    success_rate: Optional[float] = Field(default=None, description="Success rate of this configuration")
    
    # Documentation
    description: Optional[str] = Field(default=None, description="Configuration description")
    notes: Optional[str] = Field(default=None, description="Additional notes")
    documentation_url: Optional[str] = Field(default=None, description="Documentation URL")


class PerformanceResultsDTO(BaseModel):
    """Performance results for configuration."""
    model_config = ConfigDict(from_attributes=True)
    
    # Primary metrics
    accuracy: Optional[float] = Field(default=None, description="Model accuracy")
    precision: Optional[float] = Field(default=None, description="Model precision")
    recall: Optional[float] = Field(default=None, description="Model recall")
    f1_score: Optional[float] = Field(default=None, description="F1 score")
    roc_auc: Optional[float] = Field(default=None, description="ROC AUC score")
    
    # Performance characteristics
    training_time_seconds: Optional[float] = Field(default=None, description="Training time in seconds")
    prediction_time_ms: Optional[float] = Field(default=None, description="Prediction time in milliseconds")
    memory_usage_mb: Optional[float] = Field(default=None, description="Memory usage in MB")
    model_size_mb: Optional[float] = Field(default=None, description="Model size in MB")
    
    # Cross-validation results
    cv_scores: Optional[List[float]] = Field(default=None, description="Cross-validation scores")
    cv_mean: Optional[float] = Field(default=None, description="CV mean score")
    cv_std: Optional[float] = Field(default=None, description="CV standard deviation")
    
    # Additional metrics
    confusion_matrix: Optional[List[List[int]]] = Field(default=None, description="Confusion matrix")
    feature_importance: Optional[Dict[str, float]] = Field(default=None, description="Feature importance scores")
    anomaly_scores: Optional[List[float]] = Field(default=None, description="Anomaly scores")
    
    # Resource utilization
    cpu_usage_percent: Optional[float] = Field(default=None, description="CPU usage percentage")
    gpu_usage_percent: Optional[float] = Field(default=None, description="GPU usage percentage")
    disk_io_mb: Optional[float] = Field(default=None, description="Disk I/O in MB")
    
    # Quality metrics
    stability_score: Optional[float] = Field(default=None, description="Model stability score")
    robustness_score: Optional[float] = Field(default=None, description="Model robustness score")
    interpretability_score: Optional[float] = Field(default=None, description="Model interpretability score")


class ConfigurationLineageDTO(BaseModel):
    """Configuration lineage and relationships."""
    model_config = ConfigDict(from_attributes=True)
    
    # Relationship tracking
    parent_configurations: List[UUID] = Field(default_factory=list, description="Parent configuration IDs")
    child_configurations: List[UUID] = Field(default_factory=list, description="Child configuration IDs")
    related_configurations: List[UUID] = Field(default_factory=list, description="Related configuration IDs")
    
    # Derivation information
    derivation_method: Optional[str] = Field(default=None, description="How this configuration was derived")
    modifications_made: List[str] = Field(default_factory=list, description="Modifications from parent")
    optimization_history: List[Dict[str, Any]] = Field(default_factory=list, description="Optimization steps")
    
    # Version control
    git_commit: Optional[str] = Field(default=None, description="Git commit hash")
    git_branch: Optional[str] = Field(default=None, description="Git branch name")
    code_version: Optional[str] = Field(default=None, description="Code version")
    
    # Experiment tracking
    experiment_id: Optional[str] = Field(default=None, description="Experiment identifier")
    run_id: Optional[str] = Field(default=None, description="Run identifier")
    mlflow_run_id: Optional[str] = Field(default=None, description="MLflow run ID")


# ============================================================================
# Main Configuration DTO
# ============================================================================

class ExperimentConfigurationDTO(BaseModel):
    """Complete experiment configuration."""
    model_config = ConfigDict(from_attributes=True)
    
    # Identification
    id: UUID = Field(default_factory=uuid4, description="Configuration ID")
    name: str = Field(..., description="Configuration name")
    status: ConfigurationStatus = Field(default=ConfigurationStatus.DRAFT, description="Configuration status")
    
    # Core configuration components
    dataset_config: DatasetConfigDTO = Field(..., description="Dataset configuration")
    algorithm_config: AlgorithmConfigDTO = Field(..., description="Algorithm configuration")
    preprocessing_config: Optional[PreprocessingConfigDTO] = Field(default=None, description="Preprocessing configuration")
    evaluation_config: EvaluationConfigDTO = Field(..., description="Evaluation configuration")
    environment_config: Optional[EnvironmentConfigDTO] = Field(default=None, description="Environment configuration")
    
    # Metadata and tracking
    metadata: ConfigurationMetadataDTO = Field(..., description="Configuration metadata")
    lineage: Optional[ConfigurationLineageDTO] = Field(default=None, description="Configuration lineage")
    performance_results: Optional[PerformanceResultsDTO] = Field(default=None, description="Performance results")
    
    # Export configuration
    export_formats: List[ExportFormat] = Field(default_factory=list, description="Supported export formats")
    export_metadata: Dict[str, Any] = Field(default_factory=dict, description="Export-specific metadata")
    
    # Validation
    is_valid: bool = Field(default=True, description="Whether configuration is valid")
    validation_errors: List[str] = Field(default_factory=list, description="Validation error messages")
    validation_warnings: List[str] = Field(default_factory=list, description="Validation warnings")
    
    # Execution tracking
    last_executed: Optional[datetime] = Field(default=None, description="Last execution timestamp")
    execution_count: int = Field(default=0, description="Number of times executed")
    average_execution_time: Optional[float] = Field(default=None, description="Average execution time in seconds")


# ============================================================================
# Configuration Collection and Management DTOs
# ============================================================================

class ConfigurationCollectionDTO(BaseModel):
    """Collection of related configurations."""
    model_config = ConfigDict(from_attributes=True)
    
    # Collection identification
    id: UUID = Field(default_factory=uuid4, description="Collection ID")
    name: str = Field(..., description="Collection name")
    description: Optional[str] = Field(default=None, description="Collection description")
    
    # Configuration references
    configurations: List[UUID] = Field(default_factory=list, description="Configuration IDs in collection")
    featured_configuration: Optional[UUID] = Field(default=None, description="Featured/recommended configuration")
    
    # Organization
    tags: List[str] = Field(default_factory=list, description="Collection tags")
    category: Optional[str] = Field(default=None, description="Collection category")
    created_by: Optional[str] = Field(default=None, description="Collection creator")
    created_at: datetime = Field(default_factory=datetime.now, description="Creation timestamp")
    
    # Access control
    is_public: bool = Field(default=False, description="Whether collection is public")
    access_permissions: Dict[str, List[str]] = Field(default_factory=dict, description="Access permissions")
    
    # Statistics
    total_configurations: int = Field(default=0, description="Total number of configurations")
    success_rate: Optional[float] = Field(default=None, description="Average success rate")
    average_performance: Optional[float] = Field(default=None, description="Average performance score")


class ConfigurationTemplateDTO(BaseModel):
    """Template for creating configurations."""
    model_config = ConfigDict(from_attributes=True)
    
    # Template identification
    id: UUID = Field(default_factory=uuid4, description="Template ID")
    name: str = Field(..., description="Template name")
    description: str = Field(..., description="Template description")
    
    # Template content
    template_config: ExperimentConfigurationDTO = Field(..., description="Base configuration")
    variable_parameters: Dict[str, Any] = Field(default_factory=dict, description="Customizable parameters")
    parameter_constraints: Dict[str, Dict[str, Any]] = Field(default_factory=dict, description="Parameter constraints")
    
    # Template metadata
    use_cases: List[str] = Field(default_factory=list, description="Recommended use cases")
    difficulty_level: ConfigurationLevel = Field(..., description="Template difficulty level")
    estimated_runtime: Optional[float] = Field(default=None, description="Estimated runtime in seconds")
    
    # Usage tracking
    usage_count: int = Field(default=0, description="Template usage count")
    success_rate: Optional[float] = Field(default=None, description="Template success rate")
    rating: Optional[float] = Field(default=None, description="User rating")
    
    # Documentation
    documentation: Optional[str] = Field(default=None, description="Template documentation")
    examples: List[Dict[str, Any]] = Field(default_factory=list, description="Usage examples")
    tutorial_url: Optional[str] = Field(default=None, description="Tutorial URL")


# ============================================================================
# Request and Response DTOs
# ============================================================================

class ConfigurationCaptureRequestDTO(BaseModel):
    """Request to capture configuration from execution."""
    model_config = ConfigDict(from_attributes=True)
    
    # Source information
    source: ConfigurationSource = Field(..., description="Configuration source")
    source_context: Dict[str, Any] = Field(default_factory=dict, description="Source-specific context")
    
    # Configuration data
    raw_parameters: Dict[str, Any] = Field(..., description="Raw parameter values")
    execution_results: Optional[Dict[str, Any]] = Field(default=None, description="Execution results")
    
    # Metadata
    user_id: Optional[str] = Field(default=None, description="User identifier")
    session_id: Optional[str] = Field(default=None, description="Session identifier")
    timestamp: datetime = Field(default_factory=datetime.now, description="Capture timestamp")
    
    # Options
    auto_save: bool = Field(default=True, description="Automatically save configuration")
    generate_name: bool = Field(default=True, description="Auto-generate configuration name")
    tags: List[str] = Field(default_factory=list, description="Additional tags")


class ConfigurationExportRequestDTO(BaseModel):
    """Request to export configuration."""
    model_config = ConfigDict(from_attributes=True)
    
    # Configuration selection
    configuration_ids: List[UUID] = Field(..., description="Configuration IDs to export")
    
    # Export options
    export_format: ExportFormat = Field(..., description="Export format")
    include_metadata: bool = Field(default=True, description="Include metadata")
    include_performance: bool = Field(default=True, description="Include performance results")
    include_lineage: bool = Field(default=False, description="Include lineage information")
    
    # Output options
    output_path: Optional[str] = Field(default=None, description="Output file path")
    compress: bool = Field(default=False, description="Compress output")
    split_files: bool = Field(default=False, description="Split into separate files")
    
    # Template options
    template_name: Optional[str] = Field(default=None, description="Export as template")
    parameterize: bool = Field(default=False, description="Create parameterized template")
    include_documentation: bool = Field(default=True, description="Include documentation")


class ConfigurationSearchRequestDTO(BaseModel):
    """Request to search configurations."""
    model_config = ConfigDict(from_attributes=True)
    
    # Search criteria
    query: Optional[str] = Field(default=None, description="Search query")
    tags: Optional[List[str]] = Field(default=None, description="Required tags")
    source: Optional[ConfigurationSource] = Field(default=None, description="Configuration source filter")
    algorithm: Optional[str] = Field(default=None, description="Algorithm filter")
    
    # Date range
    created_after: Optional[datetime] = Field(default=None, description="Created after date")
    created_before: Optional[datetime] = Field(default=None, description="Created before date")
    
    # Performance filters
    min_accuracy: Optional[float] = Field(default=None, description="Minimum accuracy filter")
    max_execution_time: Optional[float] = Field(default=None, description="Maximum execution time filter")
    
    # Pagination
    offset: int = Field(default=0, ge=0, description="Search result offset")
    limit: int = Field(default=50, ge=1, le=1000, description="Maximum results to return")
    
    # Sorting
    sort_by: str = Field(default="created_at", description="Sort field")
    sort_order: str = Field(default="desc", description="Sort order (asc, desc)")


class ConfigurationResponseDTO(BaseModel):
    """Response containing configuration data."""
    model_config = ConfigDict(from_attributes=True)
    
    # Status
    success: bool = Field(..., description="Whether operation was successful")
    message: str = Field(..., description="Response message")
    
    # Data
    configuration: Optional[ExperimentConfigurationDTO] = Field(default=None, description="Configuration data")
    configurations: Optional[List[ExperimentConfigurationDTO]] = Field(default=None, description="Multiple configurations")
    
    # Export data
    export_data: Optional[str] = Field(default=None, description="Exported configuration data")
    export_files: Optional[List[str]] = Field(default=None, description="Export file paths")
    
    # Metadata
    total_count: Optional[int] = Field(default=None, description="Total number of matching configurations")
    execution_time: Optional[float] = Field(default=None, description="Operation execution time")
    
    # Errors and warnings
    errors: List[str] = Field(default_factory=list, description="Error messages")
    warnings: List[str] = Field(default_factory=list, description="Warning messages")


# ============================================================================
# Configuration Validation DTOs
# ============================================================================

class ConfigurationValidationResultDTO(BaseModel):
    """Results of configuration validation."""
    model_config = ConfigDict(from_attributes=True)
    
    # Validation status
    is_valid: bool = Field(..., description="Whether configuration is valid")
    validation_score: float = Field(..., ge=0, le=1, description="Validation score (0-1)")
    
    # Issues found
    errors: List[str] = Field(default_factory=list, description="Validation errors")
    warnings: List[str] = Field(default_factory=list, description="Validation warnings")
    suggestions: List[str] = Field(default_factory=list, description="Improvement suggestions")
    
    # Component validation
    dataset_validation: Dict[str, Any] = Field(default_factory=dict, description="Dataset validation results")
    algorithm_validation: Dict[str, Any] = Field(default_factory=dict, description="Algorithm validation results")
    preprocessing_validation: Dict[str, Any] = Field(default_factory=dict, description="Preprocessing validation results")
    
    # Compatibility checks
    compatibility_issues: List[str] = Field(default_factory=list, description="Compatibility issues")
    missing_dependencies: List[str] = Field(default_factory=list, description="Missing dependencies")
    version_conflicts: List[str] = Field(default_factory=list, description="Version conflicts")
    
    # Performance predictions
    estimated_runtime: Optional[float] = Field(default=None, description="Estimated runtime in seconds")
    estimated_memory: Optional[float] = Field(default=None, description="Estimated memory usage in MB")
    risk_assessment: Optional[str] = Field(default=None, description="Risk assessment level")


# ============================================================================
# Utility Functions
# ============================================================================

def create_basic_configuration(
    name: str,
    dataset_path: str,
    algorithm_name: str,
    source: ConfigurationSource = ConfigurationSource.MANUAL
) -> ExperimentConfigurationDTO:
    """Create a basic configuration with minimal parameters."""
    return ExperimentConfigurationDTO(
        name=name,
        dataset_config=DatasetConfigDTO(dataset_path=dataset_path),
        algorithm_config=AlgorithmConfigDTO(algorithm_name=algorithm_name),
        evaluation_config=EvaluationConfigDTO(),
        metadata=ConfigurationMetadataDTO(source=source)
    )


def merge_configurations(
    base_config: ExperimentConfigurationDTO,
    override_config: ExperimentConfigurationDTO
) -> ExperimentConfigurationDTO:
    """Merge two configurations, with override taking precedence."""
    # Create new configuration
    merged = base_config.model_copy(deep=True)
    
    # Update with override values
    merged.algorithm_config = override_config.algorithm_config
    if override_config.preprocessing_config:
        merged.preprocessing_config = override_config.preprocessing_config
    if override_config.evaluation_config:
        merged.evaluation_config = override_config.evaluation_config
    
    # Update metadata
    merged.metadata.derived_from = [base_config.id, override_config.id]
    merged.metadata.parent_id = base_config.id
    merged.metadata.version = "merged-1.0.0"
    
    return merged


def validate_configuration_compatibility(
    config1: ExperimentConfigurationDTO,
    config2: ExperimentConfigurationDTO
) -> List[str]:
    """Check compatibility between two configurations."""
    issues = []
    
    # Check algorithm compatibility
    if config1.algorithm_config.algorithm_name != config2.algorithm_config.algorithm_name:
        issues.append("Different algorithms specified")
    
    # Check dataset compatibility
    if (config1.dataset_config.dataset_path and config2.dataset_config.dataset_path and
        config1.dataset_config.dataset_path != config2.dataset_config.dataset_path):
        issues.append("Different datasets specified")
    
    # Check preprocessing compatibility
    if (config1.preprocessing_config and config2.preprocessing_config and
        config1.preprocessing_config.scaling_method != config2.preprocessing_config.scaling_method):
        issues.append("Different preprocessing scaling methods")
    
    return issues


@dataclass
class RequestConfigurationDTO:
    """Request configuration data transfer object."""
    method: str
    path: str
    query_parameters: Dict[str, Any]
    headers: Dict[str, str]
    body: Optional[Union[Dict[str, Any], str]]
    client_ip: str
    user_agent: Optional[str]
    content_type: Optional[str]
    content_length: Optional[str]


@dataclass
class ResponseConfigurationDTO:
    """Response configuration data transfer object."""
    status_code: int
    headers: Dict[str, str]
    body: Optional[Union[Dict[str, Any], str]]
    processing_time_ms: float
    content_type: Optional[str]
    content_length: Optional[str]


@dataclass
class WebAPIContextDTO:
    """Web API context data transfer object."""
    request_config: RequestConfigurationDTO
    response_config: ResponseConfigurationDTO
    endpoint: str
    api_version: Optional[str]
    client_info: Dict[str, Any]
    session_id: Optional[str]