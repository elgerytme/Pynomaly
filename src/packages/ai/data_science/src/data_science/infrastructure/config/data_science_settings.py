"""Settings and configuration for data science package."""

from typing import Any, Optional
from pydantic import ConfigDict, Field
from pydantic_settings import BaseSettings


class DataScienceSettings(BaseSettings):
    """Configuration settings for data science package."""
    
    # Model Training Settings
    default_training_timeout_minutes: int = Field(
        default=120,
        description="Default timeout for model training in minutes"
    )
    
    max_training_duration_hours: int = Field(
        default=24,
        description="Maximum allowed training duration in hours"
    )
    
    default_validation_split: float = Field(
        default=0.2,
        ge=0.1,
        le=0.5,
        description="Default validation split ratio"
    )
    
    default_test_split: float = Field(
        default=0.1,
        ge=0.05,
        le=0.3,
        description="Default test split ratio"
    )
    
    default_cross_validation_folds: int = Field(
        default=5,
        ge=2,
        le=10,
        description="Default number of cross-validation folds"
    )
    
    # Model Registry Settings
    model_registry_enabled: bool = Field(
        default=True,
        description="Whether to use model registry for tracking"
    )
    
    model_artifact_storage_path: str = Field(
        default="./model_artifacts",
        description="Path for storing model artifacts"
    )
    
    model_versioning_strategy: str = Field(
        default="semantic",
        description="Model versioning strategy (semantic, timestamp, sequential)"
    )
    
    max_model_versions_per_name: int = Field(
        default=10,
        ge=1,
        description="Maximum number of versions to keep per model name"
    )
    
    # Performance Monitoring Settings
    performance_monitoring_enabled: bool = Field(
        default=True,
        description="Whether to enable performance monitoring"
    )
    
    drift_detection_threshold: float = Field(
        default=0.05,
        ge=0.01,
        le=0.2,
        description="Threshold for detecting model drift"
    )
    
    performance_degradation_threshold: float = Field(
        default=0.1,
        ge=0.01,
        le=0.5,
        description="Threshold for performance degradation alerts"
    )
    
    monitoring_window_days: int = Field(
        default=30,
        ge=1,
        le=365,
        description="Window for performance monitoring in days"
    )
    
    # Feature Store Settings
    feature_store_enabled: bool = Field(
        default=True,
        description="Whether to use feature store"
    )
    
    feature_cache_ttl_hours: int = Field(
        default=24,
        ge=1,
        le=168,
        description="Feature cache TTL in hours"
    )
    
    feature_validation_enabled: bool = Field(
        default=True,
        description="Whether to validate features"
    )
    
    max_feature_groups_per_store: int = Field(
        default=100,
        ge=1,
        description="Maximum number of feature groups per store"
    )
    
    # Experiment Tracking Settings
    experiment_tracking_enabled: bool = Field(
        default=True,
        description="Whether to enable experiment tracking"
    )
    
    experiment_auto_archive_days: int = Field(
        default=90,
        ge=30,
        le=365,
        description="Days after which to auto-archive experiments"
    )
    
    max_concurrent_experiments: int = Field(
        default=10,
        ge=1,
        le=100,
        description="Maximum number of concurrent experiments"
    )
    
    experiment_result_retention_days: int = Field(
        default=365,
        ge=90,
        description="Days to retain experiment results"
    )
    
    # Data Validation Settings
    data_validation_enabled: bool = Field(
        default=True,
        description="Whether to enable data validation"
    )
    
    min_dataset_size: int = Field(
        default=100,
        ge=10,
        description="Minimum required dataset size"
    )
    
    max_missing_value_ratio: float = Field(
        default=0.3,
        ge=0.0,
        le=0.9,
        description="Maximum allowed missing value ratio"
    )
    
    outlier_detection_enabled: bool = Field(
        default=True,
        description="Whether to enable outlier detection"
    )
    
    # Algorithm Specific Settings
    algorithm_configs: dict[str, dict[str, Any]] = Field(
        default_factory=lambda: {
            "random_forest": {
                "default_n_estimators": 100,
                "max_n_estimators": 1000,
                "default_max_depth": None,
                "max_max_depth": 50
            },
            "xgboost": {
                "default_n_estimators": 100,
                "max_n_estimators": 1000,
                "default_learning_rate": 0.1,
                "min_learning_rate": 0.001,
                "max_learning_rate": 1.0
            },
            "neural_network": {
                "default_epochs": 100,
                "max_epochs": 1000,
                "default_batch_size": 32,
                "max_batch_size": 1024,
                "default_learning_rate": 0.001
            }
        },
        description="Algorithm-specific configuration settings"
    )
    
    # Resource Management Settings
    resource_monitoring_enabled: bool = Field(
        default=True,
        description="Whether to monitor resource usage"
    )
    
    max_memory_usage_gb: int = Field(
        default=32,
        ge=1,
        description="Maximum memory usage allowed in GB"
    )
    
    max_cpu_cores: int = Field(
        default=8,
        ge=1,
        description="Maximum CPU cores to use"
    )
    
    gpu_enabled: bool = Field(
        default=False,
        description="Whether GPU acceleration is available"
    )
    
    max_gpu_memory_gb: int = Field(
        default=16,
        ge=1,
        description="Maximum GPU memory in GB"
    )
    
    # Deployment Settings
    deployment_validation_enabled: bool = Field(
        default=True,
        description="Whether to validate models before deployment"
    )
    
    deployment_approval_required: bool = Field(
        default=True,
        description="Whether deployment requires approval"
    )
    
    canary_deployment_enabled: bool = Field(
        default=True,
        description="Whether to use canary deployments"
    )
    
    canary_traffic_percentage: float = Field(
        default=0.1,
        ge=0.01,
        le=0.5,
        description="Percentage of traffic for canary deployment"
    )
    
    # Security Settings
    model_encryption_enabled: bool = Field(
        default=True,
        description="Whether to encrypt model artifacts"
    )
    
    audit_logging_enabled: bool = Field(
        default=True,
        description="Whether to enable audit logging"
    )
    
    access_control_enabled: bool = Field(
        default=True,
        description="Whether to enable access control"
    )
    
    # Backup and Recovery Settings
    backup_enabled: bool = Field(
        default=True,
        description="Whether to enable automatic backups"
    )
    
    backup_retention_days: int = Field(
        default=30,
        ge=7,
        le=365,
        description="Days to retain backups"
    )
    
    backup_interval_hours: int = Field(
        default=24,
        ge=1,
        le=168,
        description="Backup interval in hours"
    )        """Pydantic configuration."""
        env_prefix = "DS_"
        env_file = ".env"
        case_sensitive = False
        
    def get_algorithm_config(self, algorithm: str) -> dict[str, Any]:
        """Get configuration for a specific algorithm.
        
        Args:
            algorithm: Algorithm name
            
        Returns:
            Algorithm configuration dictionary
        """
        return self.algorithm_configs.get(algorithm.lower(), {})
    
    def get_resource_limits(self) -> dict[str, Any]:
        """Get resource limits configuration.
        
        Returns:
            Resource limits dictionary
        """
        return {
            "memory_gb": self.max_memory_usage_gb,
            "cpu_cores": self.max_cpu_cores,
            "gpu_enabled": self.gpu_enabled,
            "gpu_memory_gb": self.max_gpu_memory_gb if self.gpu_enabled else 0
        }
    
    def get_validation_thresholds(self) -> dict[str, float]:
        """Get data validation thresholds.
        
        Returns:
            Validation thresholds dictionary
        """
        return {
            "drift_detection": self.drift_detection_threshold,
            "performance_degradation": self.performance_degradation_threshold,
            "max_missing_values": self.max_missing_value_ratio
        }
    
    def is_feature_store_enabled(self) -> bool:
        """Check if feature store is enabled.
        
        Returns:
            True if feature store is enabled
        """
        return self.feature_store_enabled
    
    def is_experiment_tracking_enabled(self) -> bool:
        """Check if experiment tracking is enabled.
        
        Returns:
            True if experiment tracking is enabled
        """
        return self.experiment_tracking_enabled
    
    def should_enable_monitoring(self) -> bool:
        """Check if monitoring should be enabled.
        
        Returns:
            True if monitoring should be enabled
        """
        return (
            self.performance_monitoring_enabled and 
            self.resource_monitoring_enabled
        )