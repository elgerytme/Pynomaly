"""Data Transfer Objects for configuration management."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from enum import Enum
from typing import Any
from uuid import UUID, uuid4

from pydantic import BaseModel, ConfigDict, Field

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


# ============================================================================
# Missing DTO Classes
# ============================================================================


class OptimizationConfigDTO(BaseModel):
    """Configuration for optimization and AutoML."""

    model_config = ConfigDict(from_attributes=True)

    enable_optimization: bool = Field(
        default=True, description="Enable hyperparameter optimization"
    )
    optimization_method: str = Field(
        default="random_search", description="Optimization method"
    )
    max_trials: int = Field(default=100, description="Maximum number of trials")
    timeout_seconds: int = Field(default=3600, description="Timeout in seconds")

    # Resource constraints
    max_memory_mb: int = Field(default=4096, description="Maximum memory usage")
    max_cpu_cores: int = Field(default=4, description="Maximum CPU cores")

    # Objectives
    primary_objective: str = Field(default="accuracy", description="Primary objective")
    secondary_objectives: list[str] = Field(
        default_factory=list, description="Secondary objectives"
    )

    # Early stopping
    early_stopping_patience: int = Field(
        default=10, description="Early stopping patience"
    )
    early_stopping_threshold: float = Field(
        default=0.01, description="Early stopping threshold"
    )


class AlgorithmConfigurationDTO(BaseModel):
    """DTO for algorithm configuration."""

    model_config = ConfigDict(from_attributes=True)

    algorithm_name: str = Field(..., description="Name of the algorithm")
    algorithm_family: str = Field(..., description="Algorithm family")
    parameters: dict[str, Any] = Field(
        default_factory=dict, description="Algorithm parameters"
    )
    hyperparameters: dict[str, Any] = Field(
        default_factory=dict, description="Hyperparameters"
    )
    optimization_config: dict[str, Any] = Field(
        default_factory=dict, description="Optimization configuration"
    )
    performance_metrics: dict[str, float] = Field(
        default_factory=dict, description="Performance metrics"
    )
    resource_requirements: dict[str, Any] = Field(
        default_factory=dict, description="Resource requirements"
    )
    configuration_source: ConfigurationSource = Field(
        default=ConfigurationSource.MANUAL, description="Source of configuration"
    )
    created_at: datetime = Field(
        default_factory=datetime.utcnow, description="Creation timestamp"
    )


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
    dataset_path: str | None = Field(default=None, description="Path to dataset file")
    dataset_name: str | None = Field(
        default=None, description="Dataset name identifier"
    )
    dataset_id: UUID | None = Field(default=None, description="Dataset UUID")

    # Data loading configuration
    file_format: str | None = Field(
        default=None, description="File format (csv, parquet, json)"
    )
    delimiter: str | None = Field(default=",", description="CSV delimiter")
    encoding: str | None = Field(default="utf-8", description="File encoding")
    header_row: int | None = Field(default=0, description="Header row index")

    # Data processing
    feature_columns: list[str] | None = Field(
        default=None, description="Feature column names"
    )
    target_column: str | None = Field(default=None, description="Target column name")
    exclude_columns: list[str] | None = Field(
        default=None, description="Columns to exclude"
    )
    datetime_columns: list[str] | None = Field(
        default=None, description="Datetime column names"
    )

    # Data validation
    expected_shape: tuple | None = Field(
        default=None, description="Expected dataset shape"
    )
    required_columns: list[str] | None = Field(
        default=None, description="Required column names"
    )
    data_types: dict[str, str] | None = Field(
        default=None, description="Expected data types"
    )

    # Sampling configuration
    sample_size: int | None = Field(
        default=None, description="Sample size for large datasets"
    )
    sampling_method: str | None = Field(default="random", description="Sampling method")
    random_seed: int | None = Field(
        default=42, description="Random seed for reproducibility"
    )


class AlgorithmConfigDTO(BaseModel):
    """Configuration for anomaly detection algorithms."""

    model_config = ConfigDict(from_attributes=True)

    # Algorithm selection
    algorithm_name: str = Field(..., description="Algorithm name")
    algorithm_family: str | None = Field(
        default=None, description="Algorithm family (statistical, distance, etc.)"
    )
    algorithm_version: str | None = Field(default=None, description="Algorithm version")

    # Hyperparameters
    hyperparameters: dict[str, Any] = Field(
        default_factory=dict, description="Algorithm hyperparameters"
    )
    parameter_source: str | None = Field(
        default="default", description="Parameter source (default, optimized, manual)"
    )
    optimization_method: str | None = Field(
        default=None, description="Hyperparameter optimization method"
    )

    # Model configuration
    contamination: float | None = Field(
        default=0.1, ge=0, le=0.5, description="Expected contamination rate"
    )
    random_state: int | None = Field(
        default=42, description="Random state for reproducibility"
    )
    n_jobs: int | None = Field(default=1, description="Number of parallel jobs")

    # Performance constraints
    max_training_time: float | None = Field(
        default=None, description="Maximum training time in seconds"
    )
    max_memory_usage: float | None = Field(
        default=None, description="Maximum memory usage in MB"
    )

    # Ensemble configuration
    is_ensemble: bool = Field(
        default=False, description="Whether this is an ensemble configuration"
    )
    ensemble_method: str | None = Field(
        default=None, description="Ensemble combination method"
    )
    base_algorithms: list[str] | None = Field(
        default=None, description="Base algorithms for ensemble"
    )


class PreprocessingConfigDTO(BaseModel):
    """Configuration for data preprocessing."""

    model_config = ConfigDict(from_attributes=True)

    # Missing value handling
    missing_value_strategy: str | None = Field(
        default="mean", description="Missing value imputation strategy"
    )
    missing_value_threshold: float | None = Field(
        default=0.5, description="Threshold for dropping columns with missing values"
    )

    # Outlier handling
    outlier_detection_method: str | None = Field(
        default="iqr", description="Outlier detection method"
    )
    outlier_threshold: float | None = Field(
        default=1.5, description="Outlier detection threshold"
    )
    outlier_treatment: str | None = Field(
        default="remove", description="Outlier treatment method"
    )

    # Feature scaling
    scaling_method: str | None = Field(
        default="standard", description="Feature scaling method"
    )
    scaling_robust: bool = Field(default=False, description="Use robust scaling")

    # Feature selection
    feature_selection_method: str | None = Field(
        default=None, description="Feature selection method"
    )
    max_features: int | None = Field(
        default=None, description="Maximum number of features"
    )
    feature_importance_threshold: float | None = Field(
        default=None, description="Feature importance threshold"
    )

    # Categorical encoding
    categorical_encoding: str | None = Field(
        default="label", description="Categorical encoding method"
    )
    high_cardinality_threshold: int | None = Field(
        default=10, description="High cardinality threshold"
    )

    # Advanced preprocessing
    apply_pca: bool = Field(
        default=False, description="Apply PCA dimensionality reduction"
    )
    pca_components: int | None = Field(
        default=None, description="Number of PCA components"
    )
    remove_duplicates: bool = Field(default=True, description="Remove duplicate rows")
    normalize_data: bool = Field(default=False, description="Apply data normalization")


class EvaluationConfigDTO(BaseModel):
    """Configuration for model evaluation."""

    model_config = ConfigDict(from_attributes=True)

    # Evaluation metrics
    primary_metric: str = Field(
        default="roc_auc", description="Primary evaluation metric"
    )
    secondary_metrics: list[str] = Field(
        default_factory=list, description="Additional evaluation metrics"
    )

    # Cross-validation
    cv_folds: int = Field(
        default=5, ge=2, le=20, description="Number of cross-validation folds"
    )
    cv_strategy: str = Field(
        default="stratified", description="Cross-validation strategy"
    )
    cv_random_state: int = Field(default=42, description="Random state for CV splits")

    # Test configuration
    test_size: float | None = Field(
        default=0.2, ge=0.1, le=0.5, description="Test set size"
    )
    validation_size: float | None = Field(
        default=0.2, ge=0.1, le=0.5, description="Validation set size"
    )

    # Performance thresholds
    min_accuracy: float | None = Field(
        default=None, description="Minimum required accuracy"
    )
    max_false_positive_rate: float | None = Field(
        default=None, description="Maximum allowed false positive rate"
    )
    min_recall: float | None = Field(
        default=None, description="Minimum required recall"
    )

    # Evaluation options
    calculate_feature_importance: bool = Field(
        default=True, description="Calculate feature importance"
    )
    generate_plots: bool = Field(default=True, description="Generate evaluation plots")
    save_predictions: bool = Field(default=False, description="Save model predictions")
    detailed_results: bool = Field(
        default=True, description="Include detailed evaluation results"
    )


class EnvironmentConfigDTO(BaseModel):
    """Configuration for execution environment."""

    model_config = ConfigDict(from_attributes=True)

    # Python environment
    python_version: str | None = Field(default=None, description="Python version")
    dependencies: dict[str, str] | None = Field(
        default=None, description="Package dependencies"
    )

    # Hardware configuration
    cpu_count: int | None = Field(default=None, description="Number of CPU cores")
    memory_gb: float | None = Field(default=None, description="Available memory in GB")
    gpu_available: bool = Field(default=False, description="GPU availability")
    gpu_model: str | None = Field(default=None, description="GPU model")

    # Execution constraints
    max_execution_time: float | None = Field(
        default=None, description="Maximum execution time in seconds"
    )
    memory_limit_mb: float | None = Field(
        default=None, description="Memory limit in MB"
    )
    disk_space_gb: float | None = Field(
        default=None, description="Available disk space in GB"
    )

    # Platform information
    operating_system: str | None = Field(default=None, description="Operating system")
    architecture: str | None = Field(default=None, description="System architecture")
    container_runtime: str | None = Field(
        default=None, description="Container runtime (docker, podman)"
    )


# ============================================================================
# Configuration Metadata and Lineage
# ============================================================================


class ConfigurationMetadataDTO(BaseModel):
    """Metadata for configuration tracking."""

    model_config = ConfigDict(from_attributes=True)

    # Origin information
    created_by: str | None = Field(
        default=None, description="User who created the configuration"
    )
    created_at: datetime = Field(
        default_factory=datetime.now, description="Creation timestamp"
    )
    source: ConfigurationSource = Field(..., description="Configuration source")
    source_version: str | None = Field(default=None, description="Source version")

    # Classification
    tags: list[str] = Field(default_factory=list, description="Configuration tags")
    category: str | None = Field(default=None, description="Configuration category")
    complexity_level: ConfigurationLevel = Field(
        default=ConfigurationLevel.BASIC, description="Complexity level"
    )

    # Versioning
    version: str = Field(default="1.0.0", description="Configuration version")
    parent_id: UUID | None = Field(default=None, description="Parent configuration ID")
    derived_from: list[UUID] | None = Field(
        default=None, description="Configurations this was derived from"
    )

    # Usage tracking
    usage_count: int = Field(
        default=0, description="Number of times configuration was used"
    )
    last_used: datetime | None = Field(default=None, description="Last usage timestamp")
    success_rate: float | None = Field(
        default=None, description="Success rate of this configuration"
    )

    # Documentation
    description: str | None = Field(
        default=None, description="Configuration description"
    )
    notes: str | None = Field(default=None, description="Additional notes")
    documentation_url: str | None = Field(default=None, description="Documentation URL")


class PerformanceResultsDTO(BaseModel):
    """Performance results for configuration."""

    model_config = ConfigDict(from_attributes=True)

    # Primary metrics
    accuracy: float | None = Field(default=None, description="Model accuracy")
    precision: float | None = Field(default=None, description="Model precision")
    recall: float | None = Field(default=None, description="Model recall")
    f1_score: float | None = Field(default=None, description="F1 score")
    roc_auc: float | None = Field(default=None, description="ROC AUC score")

    # Performance characteristics
    training_time_seconds: float | None = Field(
        default=None, description="Training time in seconds"
    )
    prediction_time_ms: float | None = Field(
        default=None, description="Prediction time in milliseconds"
    )
    memory_usage_mb: float | None = Field(
        default=None, description="Memory usage in MB"
    )
    model_size_mb: float | None = Field(default=None, description="Model size in MB")

    # Cross-validation results
    cv_scores: list[float] | None = Field(
        default=None, description="Cross-validation scores"
    )
    cv_mean: float | None = Field(default=None, description="CV mean score")
    cv_std: float | None = Field(default=None, description="CV standard deviation")

    # Additional metrics
    confusion_matrix: list[list[int]] | None = Field(
        default=None, description="Confusion matrix"
    )
    feature_importance: dict[str, float] | None = Field(
        default=None, description="Feature importance scores"
    )
    anomaly_scores: list[float] | None = Field(
        default=None, description="Anomaly scores"
    )

    # Resource utilization
    cpu_usage_percent: float | None = Field(
        default=None, description="CPU usage percentage"
    )
    gpu_usage_percent: float | None = Field(
        default=None, description="GPU usage percentage"
    )
    disk_io_mb: float | None = Field(default=None, description="Disk I/O in MB")

    # Quality metrics
    stability_score: float | None = Field(
        default=None, description="Model stability score"
    )
    robustness_score: float | None = Field(
        default=None, description="Model robustness score"
    )
    interpretability_score: float | None = Field(
        default=None, description="Model interpretability score"
    )


class ConfigurationLineageDTO(BaseModel):
    """Configuration lineage and relationships."""

    model_config = ConfigDict(from_attributes=True)

    # Relationship tracking
    parent_configurations: list[UUID] = Field(
        default_factory=list, description="Parent configuration IDs"
    )
    child_configurations: list[UUID] = Field(
        default_factory=list, description="Child configuration IDs"
    )
    related_configurations: list[UUID] = Field(
        default_factory=list, description="Related configuration IDs"
    )

    # Derivation information
    derivation_method: str | None = Field(
        default=None, description="How this configuration was derived"
    )
    modifications_made: list[str] = Field(
        default_factory=list, description="Modifications from parent"
    )
    optimization_history: list[dict[str, Any]] = Field(
        default_factory=list, description="Optimization steps"
    )

    # Version control
    git_commit: str | None = Field(default=None, description="Git commit hash")
    git_branch: str | None = Field(default=None, description="Git branch name")
    code_version: str | None = Field(default=None, description="Code version")

    # Experiment tracking
    experiment_id: str | None = Field(default=None, description="Experiment identifier")
    run_id: str | None = Field(default=None, description="Run identifier")
    mlflow_run_id: str | None = Field(default=None, description="MLflow run ID")


# ============================================================================
# Main Configuration DTO
# ============================================================================


class ExperimentConfigurationDTO(BaseModel):
    """Complete experiment configuration."""

    model_config = ConfigDict(from_attributes=True)

    # Identification
    id: UUID = Field(default_factory=uuid4, description="Configuration ID")
    name: str = Field(..., description="Configuration name")
    status: ConfigurationStatus = Field(
        default=ConfigurationStatus.DRAFT, description="Configuration status"
    )

    # Core configuration components
    dataset_config: DatasetConfigDTO = Field(..., description="Dataset configuration")
    algorithm_config: AlgorithmConfigDTO = Field(
        ..., description="Algorithm configuration"
    )
    preprocessing_config: PreprocessingConfigDTO | None = Field(
        default=None, description="Preprocessing configuration"
    )
    evaluation_config: EvaluationConfigDTO = Field(
        ..., description="Evaluation configuration"
    )
    environment_config: EnvironmentConfigDTO | None = Field(
        default=None, description="Environment configuration"
    )

    # Metadata and tracking
    metadata: ConfigurationMetadataDTO = Field(
        ..., description="Configuration metadata"
    )
    lineage: ConfigurationLineageDTO | None = Field(
        default=None, description="Configuration lineage"
    )
    performance_results: PerformanceResultsDTO | None = Field(
        default=None, description="Performance results"
    )

    # Export configuration
    export_formats: list[ExportFormat] = Field(
        default_factory=list, description="Supported export formats"
    )
    export_metadata: dict[str, Any] = Field(
        default_factory=dict, description="Export-specific metadata"
    )

    # Validation
    is_valid: bool = Field(default=True, description="Whether configuration is valid")
    validation_errors: list[str] = Field(
        default_factory=list, description="Validation error messages"
    )
    validation_warnings: list[str] = Field(
        default_factory=list, description="Validation warnings"
    )

    # Execution tracking
    last_executed: datetime | None = Field(
        default=None, description="Last execution timestamp"
    )
    execution_count: int = Field(default=0, description="Number of times executed")
    average_execution_time: float | None = Field(
        default=None, description="Average execution time in seconds"
    )


# ============================================================================
# Backward Compatibility DTOs
# ============================================================================


# Aliases for backward compatibility
DatasetConfigurationDTO = DatasetConfigDTO
AlgorithmConfigurationDTO = AlgorithmConfigDTO
PreprocessingConfigurationDTO = PreprocessingConfigDTO
EvaluationConfigurationDTO = EvaluationConfigDTO


# ============================================================================
# Configuration Collection and Management DTOs
# ============================================================================


class ConfigurationCollectionDTO(BaseModel):
    """Collection of related configurations."""

    model_config = ConfigDict(from_attributes=True)

    # Collection identification
    id: UUID = Field(default_factory=uuid4, description="Collection ID")
    name: str = Field(..., description="Collection name")
    description: str | None = Field(default=None, description="Collection description")

    # Configuration references
    configurations: list[UUID] = Field(
        default_factory=list, description="Configuration IDs in collection"
    )
    featured_configuration: UUID | None = Field(
        default=None, description="Featured/recommended configuration"
    )

    # Organization
    tags: list[str] = Field(default_factory=list, description="Collection tags")
    category: str | None = Field(default=None, description="Collection category")
    created_by: str | None = Field(default=None, description="Collection creator")
    created_at: datetime = Field(
        default_factory=datetime.now, description="Creation timestamp"
    )

    # Access control
    is_public: bool = Field(default=False, description="Whether collection is public")
    access_permissions: dict[str, list[str]] = Field(
        default_factory=dict, description="Access permissions"
    )

    # Statistics
    total_configurations: int = Field(
        default=0, description="Total number of configurations"
    )
    success_rate: float | None = Field(default=None, description="Average success rate")
    average_performance: float | None = Field(
        default=None, description="Average performance score"
    )


class ConfigurationTemplateDTO(BaseModel):
    """Template for creating configurations."""

    model_config = ConfigDict(from_attributes=True)

    # Template identification
    id: UUID = Field(default_factory=uuid4, description="Template ID")
    name: str = Field(..., description="Template name")
    description: str = Field(..., description="Template description")

    # Template content
    template_config: ExperimentConfigurationDTO = Field(
        ..., description="Base configuration"
    )
    variable_parameters: dict[str, Any] = Field(
        default_factory=dict, description="Customizable parameters"
    )
    parameter_constraints: dict[str, dict[str, Any]] = Field(
        default_factory=dict, description="Parameter constraints"
    )

    # Template metadata
    use_cases: list[str] = Field(
        default_factory=list, description="Recommended use cases"
    )
    difficulty_level: ConfigurationLevel = Field(
        ..., description="Template difficulty level"
    )
    estimated_runtime: float | None = Field(
        default=None, description="Estimated runtime in seconds"
    )

    # Usage tracking
    usage_count: int = Field(default=0, description="Template usage count")
    success_rate: float | None = Field(
        default=None, description="Template success rate"
    )
    rating: float | None = Field(default=None, description="User rating")

    # Documentation
    documentation: str | None = Field(
        default=None, description="Template documentation"
    )
    examples: list[dict[str, Any]] = Field(
        default_factory=list, description="Usage examples"
    )
    tutorial_url: str | None = Field(default=None, description="Tutorial URL")


# ============================================================================
# Request and Response DTOs
# ============================================================================


class ConfigurationCaptureRequestDTO(BaseModel):
    """Request to capture configuration from execution."""

    model_config = ConfigDict(from_attributes=True)

    # Source information
    source: ConfigurationSource = Field(..., description="Configuration source")
    source_context: dict[str, Any] = Field(
        default_factory=dict, description="Source-specific context"
    )

    # Configuration data
    raw_parameters: dict[str, Any] = Field(..., description="Raw parameter values")
    execution_results: dict[str, Any] | None = Field(
        default=None, description="Execution results"
    )

    # Metadata
    user_id: str | None = Field(default=None, description="User identifier")
    session_id: str | None = Field(default=None, description="Session identifier")
    timestamp: datetime = Field(
        default_factory=datetime.now, description="Capture timestamp"
    )

    # Options
    auto_save: bool = Field(
        default=True, description="Automatically save configuration"
    )
    generate_name: bool = Field(
        default=True, description="Auto-generate configuration name"
    )
    tags: list[str] = Field(default_factory=list, description="Additional tags")


class ConfigurationExportRequestDTO(BaseModel):
    """Request to export configuration."""

    model_config = ConfigDict(from_attributes=True)

    # Configuration selection
    configuration_ids: list[UUID] = Field(
        ..., description="Configuration IDs to export"
    )

    # Export options
    export_format: ExportFormat = Field(..., description="Export format")
    include_metadata: bool = Field(default=True, description="Include metadata")
    include_performance: bool = Field(
        default=True, description="Include performance results"
    )
    include_lineage: bool = Field(
        default=False, description="Include lineage information"
    )

    # Output options
    output_path: str | None = Field(default=None, description="Output file path")
    compress: bool = Field(default=False, description="Compress output")
    split_files: bool = Field(default=False, description="Split into separate files")

    # Template options
    template_name: str | None = Field(default=None, description="Export as template")
    parameterize: bool = Field(
        default=False, description="Create parameterized template"
    )
    include_documentation: bool = Field(
        default=True, description="Include documentation"
    )


class ConfigurationSearchRequestDTO(BaseModel):
    """Request to search configurations."""

    model_config = ConfigDict(from_attributes=True)

    # Search criteria
    query: str | None = Field(default=None, description="Search query")
    tags: list[str] | None = Field(default=None, description="Required tags")
    source: ConfigurationSource | None = Field(
        default=None, description="Configuration source filter"
    )
    algorithm: str | None = Field(default=None, description="Algorithm filter")

    # Date range
    created_after: datetime | None = Field(
        default=None, description="Created after date"
    )
    created_before: datetime | None = Field(
        default=None, description="Created before date"
    )

    # Performance filters
    min_accuracy: float | None = Field(
        default=None, description="Minimum accuracy filter"
    )
    max_execution_time: float | None = Field(
        default=None, description="Maximum execution time filter"
    )

    # Pagination
    offset: int = Field(default=0, ge=0, description="Search result offset")
    limit: int = Field(
        default=50, ge=1, le=1000, description="Maximum results to return"
    )

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
    configuration: ExperimentConfigurationDTO | None = Field(
        default=None, description="Configuration data"
    )
    configurations: list[ExperimentConfigurationDTO] | None = Field(
        default=None, description="Multiple configurations"
    )

    # Export data
    export_data: str | None = Field(
        default=None, description="Exported configuration data"
    )
    export_files: list[str] | None = Field(
        default=None, description="Export file paths"
    )

    # Metadata
    total_count: int | None = Field(
        default=None, description="Total number of matching configurations"
    )
    execution_time: float | None = Field(
        default=None, description="Operation execution time"
    )

    # Errors and warnings
    errors: list[str] = Field(default_factory=list, description="Error messages")
    warnings: list[str] = Field(default_factory=list, description="Warning messages")


# ============================================================================
# Configuration Validation DTOs
# ============================================================================


class ConfigurationValidationResultDTO(BaseModel):
    """Results of configuration validation."""

    model_config = ConfigDict(from_attributes=True)

    # Validation status
    is_valid: bool = Field(..., description="Whether configuration is valid")
    validation_score: float = Field(
        ..., ge=0, le=1, description="Validation score (0-1)"
    )

    # Issues found
    errors: list[str] = Field(default_factory=list, description="Validation errors")
    warnings: list[str] = Field(default_factory=list, description="Validation warnings")
    suggestions: list[str] = Field(
        default_factory=list, description="Improvement suggestions"
    )

    # Component validation
    dataset_validation: dict[str, Any] = Field(
        default_factory=dict, description="Dataset validation results"
    )
    algorithm_validation: dict[str, Any] = Field(
        default_factory=dict, description="Algorithm validation results"
    )
    preprocessing_validation: dict[str, Any] = Field(
        default_factory=dict, description="Preprocessing validation results"
    )

    # Compatibility checks
    compatibility_issues: list[str] = Field(
        default_factory=list, description="Compatibility issues"
    )
    missing_dependencies: list[str] = Field(
        default_factory=list, description="Missing dependencies"
    )
    version_conflicts: list[str] = Field(
        default_factory=list, description="Version conflicts"
    )

    # Performance predictions
    estimated_runtime: float | None = Field(
        default=None, description="Estimated runtime in seconds"
    )
    estimated_memory: float | None = Field(
        default=None, description="Estimated memory usage in MB"
    )
    risk_assessment: str | None = Field(
        default=None, description="Risk assessment level"
    )


# ============================================================================
# Utility Functions
# ============================================================================


def create_basic_configuration(
    name: str,
    dataset_path: str,
    algorithm_name: str,
    source: ConfigurationSource = ConfigurationSource.MANUAL,
) -> ExperimentConfigurationDTO:
    """Create a basic configuration with minimal parameters."""
    return ExperimentConfigurationDTO(
        name=name,
        dataset_config=DatasetConfigDTO(dataset_path=dataset_path),
        algorithm_config=AlgorithmConfigDTO(algorithm_name=algorithm_name),
        evaluation_config=EvaluationConfigDTO(),
        metadata=ConfigurationMetadataDTO(source=source),
    )


def merge_configurations(
    base_config: ExperimentConfigurationDTO, override_config: ExperimentConfigurationDTO
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
    config1: ExperimentConfigurationDTO, config2: ExperimentConfigurationDTO
) -> list[str]:
    """Check compatibility between two configurations."""
    issues = []

    # Check algorithm compatibility
    if (
        config1.algorithm_config.algorithm_name
        != config2.algorithm_config.algorithm_name
    ):
        issues.append("Different algorithms specified")

    # Check dataset compatibility
    if (
        config1.dataset_config.dataset_path
        and config2.dataset_config.dataset_path
        and config1.dataset_config.dataset_path != config2.dataset_config.dataset_path
    ):
        issues.append("Different datasets specified")

    # Check preprocessing compatibility
    if (
        config1.preprocessing_config
        and config2.preprocessing_config
        and config1.preprocessing_config.scaling_method
        != config2.preprocessing_config.scaling_method
    ):
        issues.append("Different preprocessing scaling methods")

    return issues


@dataclass
class RequestConfigurationDTO:
    """Request configuration data transfer object."""

    method: str
    path: str
    query_parameters: dict[str, Any]
    headers: dict[str, str]
    body: dict[str, Any] | str | None
    client_ip: str
    user_agent: str | None
    content_type: str | None
    content_length: str | None


@dataclass
class ResponseConfigurationDTO:
    """Response configuration data transfer object."""

    status_code: int
    headers: dict[str, str]
    body: dict[str, Any] | str | None
    processing_time_ms: float
    content_type: str | None
    content_length: str | None


@dataclass
class WebAPIContextDTO:
    """Web API context data transfer object."""

    request_config: RequestConfigurationDTO
    response_config: ResponseConfigurationDTO
    endpoint: str
    api_version: str | None
    client_info: dict[str, Any]
    session_id: str | None
