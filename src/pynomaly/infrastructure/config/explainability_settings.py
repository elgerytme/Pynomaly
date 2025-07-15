"""Explainability service configuration settings."""

from pydantic import BaseSettings, Field


class ExplainabilitySettings(BaseSettings):
    """Configuration for explainability services."""
    
    # General explainability settings
    enable_explainability: bool = Field(
        default=True,
        description="Enable explainability features"
    )
    
    max_features: int = Field(
        default=10,
        description="Maximum number of features to include in explanations"
    )
    
    background_samples: int = Field(
        default=100,
        description="Number of background samples for SHAP explanations"
    )
    
    # Caching settings
    enable_caching: bool = Field(
        default=True,
        description="Enable explanation result caching"
    )
    
    cache_ttl: int = Field(
        default=3600,
        description="Cache TTL in seconds (1 hour default)"
    )
    
    max_cache_size: int = Field(
        default=1000,
        description="Maximum number of cached explanations"
    )
    
    # Performance settings
    n_jobs: int = Field(
        default=-1,
        description="Number of parallel jobs (-1 for all cores)"
    )
    
    batch_size: int = Field(
        default=100,
        description="Batch size for explanation generation"
    )
    
    timeout_seconds: int = Field(
        default=300,
        description="Timeout for explanation generation (5 minutes)"
    )
    
    # SHAP-specific settings
    shap_algorithm: str = Field(
        default="auto",
        description="SHAP algorithm: auto, tree, linear, kernel, deep"
    )
    
    shap_feature_perturbation: str = Field(
        default="interventional", 
        description="SHAP feature perturbation method"
    )
    
    shap_check_additivity: bool = Field(
        default=False,
        description="Check SHAP additivity (expensive for large datasets)"
    )
    
    shap_linearize_link: bool = Field(
        default=True,
        description="Linearize link function for SHAP"
    )
    
    # LIME-specific settings
    lime_num_samples: int = Field(
        default=5000,
        description="Number of samples for LIME explanations"
    )
    
    lime_feature_selection: str = Field(
        default="auto",
        description="LIME feature selection method"
    )
    
    lime_discretize_continuous: bool = Field(
        default=True,
        description="Discretize continuous features in LIME"
    )
    
    lime_discretizer: str = Field(
        default="quartiles",
        description="LIME discretizer method: quartiles, deciles, entropy"
    )
    
    lime_distance_metric: str = Field(
        default="euclidean",
        description="Distance metric for LIME"
    )
    
    # Visualization settings
    enable_visualizations: bool = Field(
        default=True,
        description="Enable explanation visualizations"
    )
    
    plot_format: str = Field(
        default="png",
        description="Default plot format: png, svg, pdf"
    )
    
    plot_dpi: int = Field(
        default=300,
        description="Plot DPI for image exports"
    )
    
    plot_width: int = Field(
        default=10,
        description="Plot width in inches"
    )
    
    plot_height: int = Field(
        default=6,
        description="Plot height in inches"
    )
    
    # Quality and validation settings
    min_feature_importance_threshold: float = Field(
        default=0.01,
        description="Minimum feature importance to include in explanations"
    )
    
    explanation_confidence_threshold: float = Field(
        default=0.7,
        description="Minimum confidence threshold for explanations"
    )
    
    enable_explanation_validation: bool = Field(
        default=True,
        description="Enable explanation validation checks"
    )
    
    # Explanation types to enable
    enable_local_explanations: bool = Field(
        default=True,
        description="Enable local (instance-level) explanations"
    )
    
    enable_global_explanations: bool = Field(
        default=True,
        description="Enable global (model-level) explanations"
    )
    
    enable_cohort_explanations: bool = Field(
        default=True,
        description="Enable cohort explanations"
    )
    
    enable_counterfactual_explanations: bool = Field(
        default=False,
        description="Enable counterfactual explanations (experimental)"
    )
    
    # Storage settings
    explanation_storage_path: str = Field(
        default="./explanations",
        description="Path to store explanation artifacts"
    )
    
    enable_explanation_persistence: bool = Field(
        default=True,
        description="Enable persistent storage of explanations"
    )
    
    explanation_retention_days: int = Field(
        default=30,
        description="Days to retain stored explanations"
    )
    
    class Config:
        """Pydantic configuration."""
        env_prefix = "PYNOMALY_EXPLAINABILITY_"
        case_sensitive = False