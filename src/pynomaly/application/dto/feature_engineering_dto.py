"""
Feature Engineering Data Transfer Objects

DTOs for feature engineering requests and responses.
"""

from typing import Dict, List, Optional, Any, Union
from uuid import UUID
from datetime import datetime
from pydantic import BaseModel, Field, validator
from enum import Enum


class FeatureTypeEnum(str, Enum):
    """Feature data types."""
    NUMERIC = "numeric"
    CATEGORICAL = "categorical"
    TEXT = "text"
    DATETIME = "datetime"
    BOOLEAN = "boolean"
    DERIVED = "derived"


class SelectionMethodEnum(str, Enum):
    """Feature selection methods."""
    STATISTICAL = "statistical"
    RFE = "rfe"  # Recursive Feature Elimination
    LASSO = "lasso"
    SEQUENTIAL = "sequential"
    GENETIC = "genetic"
    MUTUAL_INFO = "mutual_info"
    CHI_SQUARE = "chi_square"
    ANOVA = "anova"


class ImportanceMethodEnum(str, Enum):
    """Feature importance calculation methods."""
    TREE_BASED = "tree_based"
    PERMUTATION = "permutation"
    SHAP = "shap"
    LINEAR_COEFFICIENTS = "linear_coefficients"
    MUTUAL_INFORMATION = "mutual_information"


class TransformationTypeEnum(str, Enum):
    """Feature transformation types."""
    SCALING = "scaling"
    NORMALIZATION = "normalization"
    LOG_TRANSFORM = "log_transform"
    SQRT_TRANSFORM = "sqrt_transform"
    BOX_COX = "box_cox"
    YEO_JOHNSON = "yeo_johnson"
    POLYNOMIAL = "polynomial"
    BINNING = "binning"
    ONE_HOT_ENCODING = "one_hot_encoding"
    LABEL_ENCODING = "label_encoding"
    TARGET_ENCODING = "target_encoding"


class ComplexityLevelEnum(str, Enum):
    """Auto feature engineering complexity levels."""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    EXTREME = "extreme"


# Feature Creation DTOs

class FeatureSpecificationDTO(BaseModel):
    """Specification for creating a new feature."""
    name: str = Field(..., description="Feature name")
    expression: str = Field(..., description="Feature creation expression")
    type: FeatureTypeEnum = Field(default=FeatureTypeEnum.DERIVED, description="Feature type")
    source_columns: List[str] = Field(..., description="Source columns used")
    parameters: Optional[Dict[str, Any]] = Field(default_factory=dict, description="Additional parameters")
    description: Optional[str] = Field(None, description="Feature description")
    
    @validator("name")
    def validate_name(cls, v):
        if not v or len(v.strip()) == 0:
            raise ValueError("Feature name cannot be empty")
        if not v.replace("_", "").replace("-", "").isalnum():
            raise ValueError("Feature name must be alphanumeric with underscores or hyphens")
        return v.strip()


class FeatureCreationRequestDTO(BaseModel):
    """Request for creating new features."""
    dataset_id: UUID = Field(..., description="Dataset identifier")
    feature_specifications: List[FeatureSpecificationDTO] = Field(..., description="Feature specifications")
    validation_enabled: bool = Field(True, description="Enable feature validation")
    create_pipeline: bool = Field(False, description="Create feature engineering pipeline")
    
    @validator("feature_specifications")
    def validate_features(cls, v):
        if not v:
            raise ValueError("At least one feature specification required")
        feature_names = [spec.name for spec in v]
        if len(feature_names) != len(set(feature_names)):
            raise ValueError("Feature names must be unique")
        return v


class FeatureMetadataDTO(BaseModel):
    """Feature metadata information."""
    feature_id: UUID = Field(..., description="Feature identifier")
    name: str = Field(..., description="Feature name")
    type: FeatureTypeEnum = Field(..., description="Feature type")
    source_columns: List[str] = Field(..., description="Source columns")
    creation_expression: str = Field(..., description="Creation expression")
    quality_score: float = Field(..., description="Quality score (0-1)")
    null_percentage: float = Field(..., description="Percentage of null values")
    unique_values: int = Field(..., description="Number of unique values")
    created_at: datetime = Field(..., description="Creation timestamp")


class FeatureCreationResponseDTO(BaseModel):
    """Response for feature creation."""
    creation_id: UUID = Field(..., description="Creation job identifier")
    dataset_id: UUID = Field(..., description="Dataset identifier")
    status: str = Field(..., description="Creation status")
    created_features: List[FeatureMetadataDTO] = Field(..., description="Created features metadata")
    feature_pipeline: Optional[Dict[str, Any]] = Field(None, description="Feature engineering pipeline")
    quality_metrics: Dict[str, float] = Field(..., description="Overall quality metrics")
    error_message: Optional[str] = Field(None, description="Error message if failed")
    execution_time_seconds: float = Field(..., description="Execution time in seconds")
    created_at: datetime = Field(..., description="Creation timestamp")


# Feature Importance DTOs

class FeatureImportanceRequestDTO(BaseModel):
    """Request for feature importance analysis."""
    dataset_id: UUID = Field(..., description="Dataset identifier")
    feature_columns: List[str] = Field(..., description="Features to analyze")
    target_column: str = Field(..., description="Target column for importance calculation")
    methods: List[ImportanceMethodEnum] = Field(
        default=[ImportanceMethodEnum.TREE_BASED, ImportanceMethodEnum.PERMUTATION],
        description="Importance calculation methods"
    )
    model_type: str = Field(default="random_forest", description="Model type for importance calculation")
    cross_validation: bool = Field(True, description="Use cross-validation for stability")
    n_iterations: int = Field(10, description="Number of iterations for stability assessment")
    
    @validator("n_iterations")
    def validate_iterations(cls, v):
        if v < 1 or v > 100:
            raise ValueError("Number of iterations must be between 1 and 100")
        return v


class FeatureImportanceScoreDTO(BaseModel):
    """Individual feature importance score."""
    feature_name: str = Field(..., description="Feature name")
    importance_score: float = Field(..., description="Importance score")
    rank: int = Field(..., description="Importance rank")
    stability_score: float = Field(..., description="Stability across iterations")
    confidence_interval: Optional[List[float]] = Field(None, description="95% confidence interval")


class FeatureImportanceResponseDTO(BaseModel):
    """Response for feature importance analysis."""
    analysis_id: UUID = Field(..., description="Analysis identifier")
    dataset_id: UUID = Field(..., description="Dataset identifier")
    target_column: str = Field(..., description="Target column")
    importance_scores: Dict[str, List[FeatureImportanceScoreDTO]] = Field(
        ..., description="Importance scores by method"
    )
    aggregated_rankings: List[FeatureImportanceScoreDTO] = Field(
        ..., description="Aggregated rankings across methods"
    )
    method_comparison: Dict[str, float] = Field(..., description="Method correlation scores")
    insights: List[str] = Field(default_factory=list, description="Generated insights")
    status: str = Field(..., description="Analysis status")
    error_message: Optional[str] = Field(None, description="Error message if failed")
    execution_time_seconds: float = Field(..., description="Execution time in seconds")
    created_at: datetime = Field(..., description="Analysis timestamp")


# Feature Selection DTOs

class FeatureSelectionRequestDTO(BaseModel):
    """Request for feature selection."""
    dataset_id: UUID = Field(..., description="Dataset identifier")
    feature_columns: List[str] = Field(..., description="Features to select from")
    target_column: str = Field(..., description="Target column for supervised selection")
    selection_method: SelectionMethodEnum = Field(..., description="Selection method")
    max_features: int = Field(..., description="Maximum number of features to select")
    selection_criteria: str = Field(default="performance", description="Selection criteria")
    cross_validation_folds: int = Field(5, description="Cross-validation folds")
    performance_metric: str = Field(default="accuracy", description="Performance metric for evaluation")
    
    @validator("max_features")
    def validate_max_features(cls, v):
        if v < 1:
            raise ValueError("Maximum features must be at least 1")
        return v
    
    @validator("cross_validation_folds")
    def validate_cv_folds(cls, v):
        if v < 2 or v > 10:
            raise ValueError("Cross-validation folds must be between 2 and 10")
        return v


class FeatureSelectionResultDTO(BaseModel):
    """Individual feature selection result."""
    feature_name: str = Field(..., description="Feature name")
    selected: bool = Field(..., description="Whether feature was selected")
    selection_score: float = Field(..., description="Selection score")
    rank: int = Field(..., description="Feature rank")
    selection_reason: str = Field(..., description="Reason for selection/rejection")


class FeatureSelectionResponseDTO(BaseModel):
    """Response for feature selection."""
    selection_id: UUID = Field(..., description="Selection job identifier")
    dataset_id: UUID = Field(..., description="Dataset identifier")
    selected_features: List[str] = Field(..., description="Selected feature names")
    selection_results: List[FeatureSelectionResultDTO] = Field(..., description="Detailed selection results")
    performance_improvement: float = Field(..., description="Performance improvement percentage")
    feature_reduction: float = Field(..., description="Feature reduction percentage")
    selection_stability: float = Field(..., description="Selection stability score")
    insights: List[str] = Field(default_factory=list, description="Generated insights")
    status: str = Field(..., description="Selection status")
    error_message: Optional[str] = Field(None, description="Error message if failed")
    execution_time_seconds: float = Field(..., description="Execution time in seconds")
    created_at: datetime = Field(..., description="Selection timestamp")


# Auto Feature Engineering DTOs

class AutoFeatureEngineeringRequestDTO(BaseModel):
    """Request for automated feature engineering."""
    dataset_id: UUID = Field(..., description="Dataset identifier")
    target_column: Optional[str] = Field(None, description="Target column for supervised feature engineering")
    feature_columns: Optional[List[str]] = Field(None, description="Features to engineer from")
    complexity_level: ComplexityLevelEnum = Field(
        default=ComplexityLevelEnum.MEDIUM,
        description="Feature engineering complexity level"
    )
    max_features: int = Field(100, description="Maximum number of features to generate")
    time_budget_minutes: int = Field(30, description="Time budget for feature engineering")
    include_transformations: bool = Field(True, description="Include transformation-based features")
    include_interactions: bool = Field(True, description="Include feature interactions")
    include_aggregations: bool = Field(True, description="Include aggregation features")
    validation_split: float = Field(0.2, description="Validation split for feature evaluation")
    
    @validator("max_features")
    def validate_max_features(cls, v):
        if v < 1 or v > 1000:
            raise ValueError("Maximum features must be between 1 and 1000")
        return v
    
    @validator("time_budget_minutes")
    def validate_time_budget(cls, v):
        if v < 1 or v > 120:
            raise ValueError("Time budget must be between 1 and 120 minutes")
        return v
    
    @validator("validation_split")
    def validate_validation_split(cls, v):
        if v < 0.1 or v > 0.5:
            raise ValueError("Validation split must be between 0.1 and 0.5")
        return v


class GeneratedFeatureDTO(BaseModel):
    """Generated feature information."""
    feature_id: UUID = Field(..., description="Generated feature identifier")
    name: str = Field(..., description="Feature name")
    type: str = Field(..., description="Feature type")
    generation_method: str = Field(..., description="Generation method used")
    source_features: List[str] = Field(..., description="Source features used")
    importance_score: float = Field(..., description="Estimated importance score")
    quality_score: float = Field(..., description="Quality score")
    complexity_score: float = Field(..., description="Complexity score")
    interpretability_score: float = Field(..., description="Interpretability score")
    expression: str = Field(..., description="Feature expression or formula")


class AutoFeatureEngineeringResponseDTO(BaseModel):
    """Response for automated feature engineering."""
    engineering_id: UUID = Field(..., description="Engineering job identifier")
    dataset_id: UUID = Field(..., description="Dataset identifier")
    generated_features: List[GeneratedFeatureDTO] = Field(..., description="Generated features")
    feature_generation_summary: Dict[str, int] = Field(..., description="Summary by generation method")
    performance_impact: Dict[str, float] = Field(..., description="Performance impact metrics")
    feature_engineering_pipeline: Dict[str, Any] = Field(..., description="Engineering pipeline")
    insights: List[str] = Field(default_factory=list, description="Generated insights")
    status: str = Field(..., description="Engineering status")
    error_message: Optional[str] = Field(None, description="Error message if failed")
    execution_time_seconds: float = Field(..., description="Execution time in seconds")
    created_at: datetime = Field(..., description="Engineering timestamp")


# Feature Transformation DTOs

class TransformationSpecificationDTO(BaseModel):
    """Specification for feature transformation."""
    name: str = Field(..., description="Transformation name")
    type: TransformationTypeEnum = Field(..., description="Transformation type")
    parameters: Optional[Dict[str, Any]] = Field(default_factory=dict, description="Transformation parameters")


class FeatureTransformationRequestDTO(BaseModel):
    """Request for feature transformation."""
    dataset_id: UUID = Field(..., description="Dataset identifier")
    feature_columns: List[str] = Field(..., description="Features to transform")
    transformations: List[TransformationSpecificationDTO] = Field(..., description="Transformations to apply")
    validation_enabled: bool = Field(True, description="Enable transformation validation")
    create_inverse: bool = Field(False, description="Create inverse transformation")
    
    @validator("transformations")
    def validate_transformations(cls, v):
        if not v:
            raise ValueError("At least one transformation required")
        return v


class TransformationResultDTO(BaseModel):
    """Individual transformation result."""
    original_feature: str = Field(..., description="Original feature name")
    transformed_feature: str = Field(..., description="Transformed feature name")
    transformation_type: str = Field(..., description="Transformation type applied")
    transformation_parameters: Dict[str, Any] = Field(..., description="Parameters used")
    quality_improvement: float = Field(..., description="Quality improvement score")
    distribution_change: Dict[str, float] = Field(..., description="Distribution change metrics")


class FeatureTransformationResponseDTO(BaseModel):
    """Response for feature transformation."""
    transformation_id: UUID = Field(..., description="Transformation job identifier")
    dataset_id: UUID = Field(..., description="Dataset identifier")
    transformed_features: List[str] = Field(..., description="Transformed feature names")
    transformation_results: List[TransformationResultDTO] = Field(..., description="Detailed transformation results")
    transformation_pipeline: Dict[str, Any] = Field(..., description="Transformation pipeline")
    quality_metrics: Dict[str, float] = Field(..., description="Overall quality metrics")
    execution_time_seconds: float = Field(..., description="Execution time in seconds")
    created_at: datetime = Field(..., description="Transformation timestamp")


# Feature Management DTOs

class FeatureListRequestDTO(BaseModel):
    """Request for listing features."""
    dataset_id: UUID = Field(..., description="Dataset identifier")
    feature_type: Optional[FeatureTypeEnum] = Field(None, description="Filter by feature type")
    include_derived: bool = Field(True, description="Include derived features")
    include_metadata: bool = Field(True, description="Include feature metadata")
    sort_by: str = Field(default="name", description="Sort field")
    sort_order: str = Field(default="asc", description="Sort order")
    limit: int = Field(100, description="Maximum results")
    offset: int = Field(0, description="Results offset")


class FeatureListResponseDTO(BaseModel):
    """Response for listing features."""
    dataset_id: UUID = Field(..., description="Dataset identifier")
    features: List[FeatureMetadataDTO] = Field(..., description="Feature list")
    total_count: int = Field(..., description="Total number of features")
    original_features: int = Field(..., description="Number of original features")
    derived_features: int = Field(..., description="Number of derived features")
    feature_types: Dict[str, int] = Field(..., description="Feature count by type")


class FeatureDeletionRequestDTO(BaseModel):
    """Request for deleting features."""
    feature_ids: List[UUID] = Field(..., description="Feature identifiers to delete")
    cascade_delete: bool = Field(False, description="Delete dependent features")
    backup_before_delete: bool = Field(True, description="Create backup before deletion")


class FeatureDeletionResponseDTO(BaseModel):
    """Response for feature deletion."""
    deletion_id: UUID = Field(..., description="Deletion job identifier")
    deleted_features: List[UUID] = Field(..., description="Successfully deleted feature IDs")
    failed_deletions: List[Dict[str, Any]] = Field(default_factory=list, description="Failed deletions with reasons")
    cascade_deleted: List[UUID] = Field(default_factory=list, description="Cascade deleted feature IDs")
    backup_location: Optional[str] = Field(None, description="Backup file location")
    execution_time_seconds: float = Field(..., description="Execution time in seconds")
    created_at: datetime = Field(..., description="Deletion timestamp")


# Feature Pipeline DTOs

class FeaturePipelineDTO(BaseModel):
    """Feature engineering pipeline definition."""
    pipeline_id: UUID = Field(..., description="Pipeline identifier")
    name: str = Field(..., description="Pipeline name")
    steps: List[Dict[str, Any]] = Field(..., description="Pipeline steps")
    input_features: List[str] = Field(..., description="Input features")
    output_features: List[str] = Field(..., description="Output features")
    created_at: datetime = Field(..., description="Pipeline creation timestamp")
    created_by: UUID = Field(..., description="Creator user ID")


class FeaturePipelineExecutionRequestDTO(BaseModel):
    """Request for executing feature pipeline."""
    pipeline_id: UUID = Field(..., description="Pipeline identifier")
    dataset_id: UUID = Field(..., description="Dataset to apply pipeline to")
    validation_enabled: bool = Field(True, description="Enable validation")
    save_results: bool = Field(True, description="Save pipeline results")


class FeaturePipelineExecutionResponseDTO(BaseModel):
    """Response for feature pipeline execution."""
    execution_id: UUID = Field(..., description="Execution identifier")
    pipeline_id: UUID = Field(..., description="Pipeline identifier")
    dataset_id: UUID = Field(..., description="Dataset identifier")
    output_features: List[str] = Field(..., description="Generated feature names")
    execution_status: str = Field(..., description="Execution status")
    quality_metrics: Dict[str, float] = Field(..., description="Quality metrics")
    execution_time_seconds: float = Field(..., description="Execution time in seconds")
    created_at: datetime = Field(..., description="Execution timestamp")