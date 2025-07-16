"""MLOps Platform Settings

Configuration settings for the MLOps platform.
"""

import os
from dataclasses import dataclass
from typing import Optional, Dict, Any

from .database import DatabaseConfig


@dataclass
class StorageConfig:
    """Storage configuration settings."""
    
    # Storage type: 'local' or 's3'
    backend: str = "local"
    
    # Local storage settings
    local_path: str = "./mlops_models"
    
    # S3 storage settings
    s3_bucket: Optional[str] = None
    s3_region: str = "us-east-1"
    s3_endpoint_url: Optional[str] = None
    s3_access_key_id: Optional[str] = None
    s3_secret_access_key: Optional[str] = None
    s3_prefix: str = "mlops/models"
    
    @classmethod
    def from_env(cls) -> 'StorageConfig':
        """Create configuration from environment variables.
        
        Returns:
            StorageConfig instance
        """
        return cls(
            backend=os.getenv('MLOPS_STORAGE_BACKEND', 'local'),
            
            local_path=os.getenv('MLOPS_STORAGE_LOCAL_PATH', './mlops_models'),
            
            s3_bucket=os.getenv('MLOPS_S3_BUCKET'),
            s3_region=os.getenv('MLOPS_S3_REGION', 'us-east-1'),
            s3_endpoint_url=os.getenv('MLOPS_S3_ENDPOINT_URL'),
            s3_access_key_id=os.getenv('MLOPS_S3_ACCESS_KEY_ID'),
            s3_secret_access_key=os.getenv('MLOPS_S3_SECRET_ACCESS_KEY'),
            s3_prefix=os.getenv('MLOPS_S3_PREFIX', 'mlops/models'),
        )


@dataclass
class SecurityConfig:
    """Security configuration settings."""
    
    # Authentication
    secret_key: str = "your-secret-key-change-this-in-production"
    algorithm: str = "HS256"
    access_token_expire_minutes: int = 30
    
    # API rate limiting
    rate_limit_enabled: bool = True
    rate_limit_requests_per_minute: int = 60
    
    # CORS settings
    cors_enabled: bool = True
    cors_origins: list = None
    
    @classmethod
    def from_env(cls) -> 'SecurityConfig':
        """Create configuration from environment variables.
        
        Returns:
            SecurityConfig instance
        """
        cors_origins = os.getenv('MLOPS_CORS_ORIGINS')
        if cors_origins:
            cors_origins = [origin.strip() for origin in cors_origins.split(',')]
        else:
            cors_origins = ["*"]  # Allow all origins in development
        
        return cls(
            secret_key=os.getenv('MLOPS_SECRET_KEY', 'your-secret-key-change-this-in-production'),
            algorithm=os.getenv('MLOPS_ALGORITHM', 'HS256'),
            access_token_expire_minutes=int(os.getenv('MLOPS_ACCESS_TOKEN_EXPIRE_MINUTES', '30')),
            
            rate_limit_enabled=os.getenv('MLOPS_RATE_LIMIT_ENABLED', 'true').lower() == 'true',
            rate_limit_requests_per_minute=int(os.getenv('MLOPS_RATE_LIMIT_RPM', '60')),
            
            cors_enabled=os.getenv('MLOPS_CORS_ENABLED', 'true').lower() == 'true',
            cors_origins=cors_origins,
        )


@dataclass
class MonitoringConfig:
    """Monitoring configuration settings."""
    
    # Metrics collection
    metrics_enabled: bool = True
    metrics_port: int = 8090
    
    # Health checks
    health_check_enabled: bool = True
    health_check_interval_seconds: int = 30
    
    # Logging
    log_level: str = "INFO"
    log_format: str = "json"
    
    @classmethod
    def from_env(cls) -> 'MonitoringConfig':
        """Create configuration from environment variables.
        
        Returns:
            MonitoringConfig instance
        """
        return cls(
            metrics_enabled=os.getenv('MLOPS_METRICS_ENABLED', 'true').lower() == 'true',
            metrics_port=int(os.getenv('MLOPS_METRICS_PORT', '8090')),
            
            health_check_enabled=os.getenv('MLOPS_HEALTH_CHECK_ENABLED', 'true').lower() == 'true',
            health_check_interval_seconds=int(os.getenv('MLOPS_HEALTH_CHECK_INTERVAL', '30')),
            
            log_level=os.getenv('MLOPS_LOG_LEVEL', 'INFO'),
            log_format=os.getenv('MLOPS_LOG_FORMAT', 'json'),
        )


@dataclass
class MLOpsSettings:
    """MLOps platform configuration settings."""
    
    # Environment
    environment: str = "development"
    debug: bool = False
    
    # Service configuration
    database: DatabaseConfig = None
    storage: StorageConfig = None
    security: SecurityConfig = None
    monitoring: MonitoringConfig = None
    
    # Application settings
    app_name: str = "MLOps Platform"
    app_version: str = "1.0.0"
    
    # Feature flags
    features: Dict[str, bool] = None
    
    def __post_init__(self):
        """Initialize nested configurations."""
        if self.database is None:
            self.database = DatabaseConfig.from_env()
        if self.storage is None:
            self.storage = StorageConfig.from_env()
        if self.security is None:
            self.security = SecurityConfig.from_env()
        if self.monitoring is None:
            self.monitoring = MonitoringConfig.from_env()
        if self.features is None:
            self.features = {
                'model_versioning': True,
                'experiment_tracking': True,
                'pipeline_orchestration': True,
                'deployment_management': True,
                'model_promotion': True,
                'artifact_storage': True,
                'monitoring': True,
                'security': True,
            }
    
    @classmethod
    def from_env(cls) -> 'MLOpsSettings':
        """Create configuration from environment variables.
        
        Returns:
            MLOpsSettings instance
        """
        return cls(
            environment=os.getenv('MLOPS_ENVIRONMENT', 'development'),
            debug=os.getenv('MLOPS_DEBUG', 'false').lower() == 'true',
            
            app_name=os.getenv('MLOPS_APP_NAME', 'MLOps Platform'),
            app_version=os.getenv('MLOPS_APP_VERSION', '1.0.0'),
        )
    
    @property
    def is_development(self) -> bool:
        """Check if running in development environment."""
        return self.environment.lower() == 'development'
    
    @property
    def is_production(self) -> bool:
        """Check if running in production environment."""
        return self.environment.lower() == 'production'
    
    def get_feature_flag(self, feature_name: str, default: bool = False) -> bool:
        """Get feature flag value.
        
        Args:
            feature_name: Name of the feature
            default: Default value if feature not found
            
        Returns:
            Feature flag value
        """
        return self.features.get(feature_name, default)