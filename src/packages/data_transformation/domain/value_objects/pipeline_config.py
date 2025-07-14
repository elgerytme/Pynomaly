"""Pipeline configuration value object."""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Union
from enum import Enum

from pydantic import BaseModel, Field, validator


class SourceType(str, Enum):
    """Supported data source types."""
    CSV = "csv"
    JSON = "json"
    PARQUET = "parquet"
    EXCEL = "excel"
    DATABASE = "database"
    API = "api"
    STREAM = "stream"
    S3 = "s3"
    GCS = "gcs"
    AZURE = "azure"


class CleaningStrategy(str, Enum):
    """Data cleaning strategies."""
    NONE = "none"
    AUTO = "auto"
    CONSERVATIVE = "conservative"
    AGGRESSIVE = "aggressive"
    CUSTOM = "custom"


class ScalingMethod(str, Enum):
    """Feature scaling methods."""
    NONE = "none"
    STANDARD = "standard"
    MINMAX = "minmax"
    ROBUST = "robust"
    QUANTILE = "quantile"
    POWER = "power"


class EncodingStrategy(str, Enum):
    """Categorical encoding strategies."""
    NONE = "none"
    ONEHOT = "onehot"
    LABEL = "label"
    TARGET = "target"
    ORDINAL = "ordinal"
    FREQUENCY = "frequency"
    HASH = "hash"


class FeatureSelectionMethod(str, Enum):
    """Feature selection methods."""
    NONE = "none"
    VARIANCE = "variance"
    UNIVARIATE = "univariate"
    RECURSIVE = "recursive"
    L1_REGULARIZATION = "l1"
    TREE_IMPORTANCE = "tree"
    CORRELATION = "correlation"


class OutputFormat(str, Enum):
    """Output format options."""
    PANDAS = "pandas"
    POLARS = "polars"
    NUMPY = "numpy"
    CSV = "csv"
    PARQUET = "parquet"
    JSON = "json"


class PipelineConfig(BaseModel):
    """
    Configuration value object for transformation pipelines.
    
    Encapsulates all configuration options for data transformation,
    cleaning, preprocessing, and feature engineering operations.
    """
    
    # Data source configuration
    source_type: SourceType
    source_path: Optional[str] = None
    source_params: Dict[str, Any] = Field(default_factory=dict)
    
    # Data cleaning configuration
    cleaning_strategy: CleaningStrategy = CleaningStrategy.AUTO
    missing_value_threshold: float = Field(default=0.5, ge=0.0, le=1.0)
    outlier_detection_method: str = "iqr"
    outlier_threshold: float = Field(default=1.5, gt=0.0)
    remove_duplicates: bool = True
    
    # Preprocessing configuration
    scaling_method: ScalingMethod = ScalingMethod.STANDARD
    encoding_strategy: EncodingStrategy = EncodingStrategy.ONEHOT
    handle_categorical_na: str = "mode"  # mode, constant, drop
    handle_numerical_na: str = "median"  # mean, median, mode, constant, drop
    
    # Feature engineering configuration
    feature_engineering: bool = True
    polynomial_features: bool = False
    polynomial_degree: int = Field(default=2, ge=1, le=5)
    interaction_features: bool = False
    time_features: bool = False
    domain_features: bool = False
    
    # Feature selection configuration
    feature_selection: bool = False
    feature_selection_method: FeatureSelectionMethod = FeatureSelectionMethod.VARIANCE
    max_features: Optional[int] = None
    feature_selection_threshold: float = 0.01
    
    # Performance configuration
    chunk_size: Optional[int] = None
    use_gpu: bool = False
    parallel_processing: bool = True
    memory_optimize: bool = True
    
    # Validation configuration
    validation_split: float = Field(default=0.2, ge=0.0, le=0.5)
    stratify: bool = False
    random_state: int = 42
    
    # Output configuration
    output_format: str = "pandas"  # pandas, polars, arrow, numpy
    save_intermediate: bool = False
    compression: Optional[str] = None  # gzip, bz2, xz
    
    # Advanced options
    custom_transformers: List[str] = Field(default_factory=list)
    pipeline_cache: bool = True
    verbose: bool = False
    
    class Config:
        """Pydantic configuration."""
        use_enum_values = True
        validate_assignment = True
    
    @validator("missing_value_threshold")
    def validate_missing_threshold(cls, v: float) -> float:
        """Validate missing value threshold is between 0 and 1."""
        if not 0.0 <= v <= 1.0:
            raise ValueError("Missing value threshold must be between 0 and 1")
        return v
    
    @validator("validation_split")
    def validate_validation_split(cls, v: float) -> float:
        """Validate validation split is reasonable."""
        if not 0.0 <= v <= 0.5:
            raise ValueError("Validation split must be between 0 and 0.5")
        return v
    
    @validator("polynomial_degree")
    def validate_polynomial_degree(cls, v: int) -> int:
        """Validate polynomial degree is reasonable."""
        if not 1 <= v <= 5:
            raise ValueError("Polynomial degree must be between 1 and 5")
        return v
    
    @classmethod
    def create_default(cls, source_type: SourceType) -> PipelineConfig:
        """Create default configuration for a source type."""
        return cls(source_type=source_type)
    
    @classmethod
    def create_minimal(cls, source_type: SourceType, source_path: str) -> PipelineConfig:
        """Create minimal configuration with just source info."""
        return cls(
            source_type=source_type,
            source_path=source_path,
            cleaning_strategy=CleaningStrategy.CONSERVATIVE,
            feature_engineering=False,
            feature_selection=False,
        )
    
    @classmethod
    def create_comprehensive(cls, source_type: SourceType, source_path: str) -> PipelineConfig:
        """Create comprehensive configuration with all features enabled."""
        return cls(
            source_type=source_type,
            source_path=source_path,
            cleaning_strategy=CleaningStrategy.AUTO,
            scaling_method=ScalingMethod.STANDARD,
            encoding_strategy=EncodingStrategy.ONEHOT,
            feature_engineering=True,
            polynomial_features=True,
            interaction_features=True,
            time_features=True,
            feature_selection=True,
            feature_selection_method=FeatureSelectionMethod.UNIVARIATE,
            parallel_processing=True,
            memory_optimize=True,
        )
    
    def enable_gpu_acceleration(self) -> PipelineConfig:
        """Enable GPU acceleration if available."""
        config = self.model_copy()
        config.use_gpu = True
        return config
    
    def set_performance_mode(self, mode: str) -> PipelineConfig:
        """Set performance optimization mode."""
        config = self.model_copy()
        
        if mode == "speed":
            config.chunk_size = 10000
            config.parallel_processing = True
            config.memory_optimize = False
            config.pipeline_cache = True
        elif mode == "memory":
            config.chunk_size = 1000
            config.parallel_processing = False
            config.memory_optimize = True
            config.pipeline_cache = False
        elif mode == "balanced":
            config.chunk_size = 5000
            config.parallel_processing = True
            config.memory_optimize = True
            config.pipeline_cache = True
        
        return config
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert configuration to dictionary."""
        return self.model_dump()