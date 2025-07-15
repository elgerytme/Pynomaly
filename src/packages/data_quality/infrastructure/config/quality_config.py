"""Configuration classes for data quality services."""

from dataclasses import dataclass, field
from typing import Dict, Any, Optional


@dataclass
class ValidationEngineConfig:
    """Configuration for validation engine."""
    
    # Performance settings
    enable_parallel_processing: bool = True
    max_workers: int = 4
    timeout_seconds: int = 300
    
    # Caching settings
    enable_caching: bool = True
    cache_size: int = 1000
    
    # Sampling settings
    enable_sampling: bool = False
    sample_size: int = 10000
    sample_random_state: int = 42
    
    # Additional configuration
    enable_metrics: bool = True
    enable_detailed_logging: bool = False
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert config to dictionary."""
        return {
            'enable_parallel_processing': self.enable_parallel_processing,
            'max_workers': self.max_workers,
            'timeout_seconds': self.timeout_seconds,
            'enable_caching': self.enable_caching,
            'cache_size': self.cache_size,
            'enable_sampling': self.enable_sampling,
            'sample_size': self.sample_size,
            'sample_random_state': self.sample_random_state,
            'enable_metrics': self.enable_metrics,
            'enable_detailed_logging': self.enable_detailed_logging
        }
    
    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]) -> 'ValidationEngineConfig':
        """Create config from dictionary."""
        return cls(**config_dict)


@dataclass
class QualityConfig:
    """Main configuration for data quality package."""
    
    validation_engine: ValidationEngineConfig = field(default_factory=ValidationEngineConfig)
    
    # Database settings
    database_url: Optional[str] = None
    connection_pool_size: int = 10
    
    # Logging settings
    log_level: str = "INFO"
    log_format: str = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    
    # Security settings
    enable_encryption: bool = True
    encryption_key: Optional[str] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert config to dictionary."""
        return {
            'validation_engine': self.validation_engine.to_dict(),
            'database_url': self.database_url,
            'connection_pool_size': self.connection_pool_size,
            'log_level': self.log_level,
            'log_format': self.log_format,
            'enable_encryption': self.enable_encryption,
            'encryption_key': self.encryption_key
        }
    
    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]) -> 'QualityConfig':
        """Create config from dictionary."""
        validation_engine_dict = config_dict.pop('validation_engine', {})
        validation_engine = ValidationEngineConfig.from_dict(validation_engine_dict)
        
        return cls(
            validation_engine=validation_engine,
            **config_dict
        )