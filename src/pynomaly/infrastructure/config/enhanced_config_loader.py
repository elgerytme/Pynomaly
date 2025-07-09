"""
Enhanced Configuration Loader for Core Architecture Components

This module provides a comprehensive configuration loading system that supports
the three main sub-tasks: data_ingestion, anomaly_detection, and alerting.
"""

import os
import json
import yaml
from typing import Dict, Any, Optional, List
from pathlib import Path
from dataclasses import dataclass, field
from enum import Enum
import logging

from pynomaly.shared.exceptions import ConfigurationError
from pynomaly.domain.value_objects.semantic_version import SemanticVersion


class ConfigurationEnvironment(Enum):
    """Configuration environment types."""
    DEVELOPMENT = "development"
    TESTING = "testing"
    STAGING = "staging"
    PRODUCTION = "production"


@dataclass
class DataIngestionConfig:
    """Configuration for data ingestion component."""
    
    # Data source configurations
    sources: Dict[str, Dict[str, Any]] = field(default_factory=dict)
    
    # Batch processing settings
    batch_size: int = 1000
    max_batch_size: int = 10000
    batch_timeout: int = 30
    
    # Streaming settings
    stream_buffer_size: int = 1000
    stream_flush_interval: int = 5
    
    # Data validation settings
    validation_enabled: bool = True
    validation_rules: Dict[str, Any] = field(default_factory=dict)
    
    # Performance settings
    parallel_workers: int = 4
    memory_limit_mb: int = 1024
    
    # Retry and error handling
    retry_attempts: int = 3
    retry_delay: int = 1
    error_threshold: float = 0.1


@dataclass
class AnomalyDetectionConfig:
    """Configuration for anomaly detection component."""
    
    # Algorithm settings
    default_algorithm: str = "isolation_forest"
    available_algorithms: List[str] = field(default_factory=lambda: [
        "isolation_forest", "local_outlier_factor", "one_class_svm", 
        "elliptic_envelope", "autoencoder"
    ])
    
    # Model settings
    model_cache_size: int = 10
    model_cache_ttl: int = 3600
    auto_retrain_enabled: bool = True
    retrain_threshold: float = 0.1
    
    # Detection parameters
    contamination: float = 0.1
    confidence_threshold: float = 0.8
    ensemble_enabled: bool = True
    ensemble_size: int = 5
    
    # Performance optimization
    batch_prediction: bool = True
    prediction_batch_size: int = 1000
    async_processing: bool = True
    
    # Explainability
    explainability_enabled: bool = True
    shap_enabled: bool = True
    lime_enabled: bool = True


@dataclass
class AlertingConfig:
    """Configuration for alerting component."""
    
    # Alert channels
    channels: Dict[str, Dict[str, Any]] = field(default_factory=dict)
    
    # Alert levels
    severity_levels: List[str] = field(default_factory=lambda: [
        "low", "medium", "high", "critical"
    ])
    
    # Alert thresholds
    thresholds: Dict[str, float] = field(default_factory=lambda: {
        "low": 0.3,
        "medium": 0.5,
        "high": 0.7,
        "critical": 0.9
    })
    
    # Alert aggregation
    aggregation_window: int = 60
    max_alerts_per_minute: int = 10
    deduplication_enabled: bool = True
    
    # Alert routing
    routing_rules: List[Dict[str, Any]] = field(default_factory=list)
    
    # Notification settings
    notification_delay: int = 0
    retry_failed_notifications: bool = True
    notification_timeout: int = 30


@dataclass
class CoreArchitectureConfig:
    """Main configuration for core architecture components."""
    
    # Environment and metadata
    environment: ConfigurationEnvironment = ConfigurationEnvironment.DEVELOPMENT
    version: SemanticVersion = field(default_factory=lambda: SemanticVersion(1, 0, 0))
    debug: bool = False
    
    # Component configurations
    data_ingestion: DataIngestionConfig = field(default_factory=DataIngestionConfig)
    anomaly_detection: AnomalyDetectionConfig = field(default_factory=AnomalyDetectionConfig)
    alerting: AlertingConfig = field(default_factory=AlertingConfig)
    
    # System-wide settings
    logging_level: str = "INFO"
    metrics_enabled: bool = True
    health_check_enabled: bool = True
    
    # Security settings
    encryption_enabled: bool = True
    authentication_required: bool = True
    rate_limiting_enabled: bool = True
    
    # Performance settings
    max_concurrent_requests: int = 100
    request_timeout: int = 30
    connection_pool_size: int = 10


class EnhancedConfigLoader:
    """Enhanced configuration loader with validation and environment support."""
    
    def __init__(self, config_path: Optional[str] = None):
        """Initialize the enhanced config loader.
        
        Args:
            config_path: Optional path to configuration file
        """
        self.logger = logging.getLogger(__name__)
        self.config_path = config_path or self._get_default_config_path()
        self._config_cache: Optional[CoreArchitectureConfig] = None
        
    def _get_default_config_path(self) -> str:
        """Get the default configuration path."""
        # Check environment variable first
        env_path = os.getenv("PYNOMALY_CONFIG_PATH")
        if env_path:
            return env_path
            
        # Check standard locations
        standard_paths = [
            "config/core_architecture.yml",
            "config/core_architecture.yaml",
            "config/core_architecture.json",
            "/etc/pynomaly/config.yml",
            os.path.expanduser("~/.pynomaly/config.yml")
        ]
        
        for path in standard_paths:
            if os.path.exists(path):
                return path
                
        # Return default path
        return "config/core_architecture.yml"
    
    def load_config(self, force_reload: bool = False) -> CoreArchitectureConfig:
        """Load configuration with caching and validation.
        
        Args:
            force_reload: Force reload from file even if cached
            
        Returns:
            CoreArchitectureConfig: Loaded and validated configuration
            
        Raises:
            ConfigurationError: If configuration is invalid
        """
        if self._config_cache is not None and not force_reload:
            return self._config_cache
            
        try:
            config_data = self._load_from_file()
            environment_overrides = self._load_environment_overrides()
            
            # Merge configurations
            merged_config = self._merge_configs(config_data, environment_overrides)
            
            # Create configuration object
            config = self._create_config_object(merged_config)
            
            # Validate configuration
            self._validate_config(config)
            
            # Cache the configuration
            self._config_cache = config
            
            self.logger.info(f"Configuration loaded successfully from {self.config_path}")
            return config
            
        except Exception as e:
            self.logger.error(f"Failed to load configuration: {e}")
            raise ConfigurationError(f"Configuration loading failed: {e}")
    
    def _load_from_file(self) -> Dict[str, Any]:
        """Load configuration from file."""
        if not os.path.exists(self.config_path):
            self.logger.warning(f"Configuration file not found: {self.config_path}")
            return {}
            
        try:
            with open(self.config_path, 'r', encoding='utf-8') as f:
                if self.config_path.endswith(('.yml', '.yaml')):
                    return yaml.safe_load(f) or {}
                elif self.config_path.endswith('.json'):
                    return json.load(f) or {}
                else:
                    raise ConfigurationError(f"Unsupported configuration file format: {self.config_path}")
        except Exception as e:
            raise ConfigurationError(f"Failed to parse configuration file: {e}")
    
    def _load_environment_overrides(self) -> Dict[str, Any]:
        """Load configuration overrides from environment variables."""
        overrides = {}
        
        # Environment mapping
        env_mappings = {
            'PYNOMALY_ENV': 'environment',
            'PYNOMALY_DEBUG': 'debug',
            'PYNOMALY_LOG_LEVEL': 'logging_level',
            'PYNOMALY_METRICS_ENABLED': 'metrics_enabled',
            'PYNOMALY_HEALTH_CHECK_ENABLED': 'health_check_enabled',
            
            # Data ingestion
            'PYNOMALY_BATCH_SIZE': 'data_ingestion.batch_size',
            'PYNOMALY_PARALLEL_WORKERS': 'data_ingestion.parallel_workers',
            'PYNOMALY_MEMORY_LIMIT_MB': 'data_ingestion.memory_limit_mb',
            
            # Anomaly detection
            'PYNOMALY_DEFAULT_ALGORITHM': 'anomaly_detection.default_algorithm',
            'PYNOMALY_CONTAMINATION': 'anomaly_detection.contamination',
            'PYNOMALY_CONFIDENCE_THRESHOLD': 'anomaly_detection.confidence_threshold',
            
            # Alerting
            'PYNOMALY_AGGREGATION_WINDOW': 'alerting.aggregation_window',
            'PYNOMALY_MAX_ALERTS_PER_MINUTE': 'alerting.max_alerts_per_minute',
        }
        
        for env_var, config_path in env_mappings.items():
            value = os.getenv(env_var)
            if value is not None:
                self._set_nested_value(overrides, config_path, self._convert_env_value(value))
                
        return overrides
    
    def _convert_env_value(self, value: str) -> Any:
        """Convert environment variable value to appropriate type."""
        # Boolean conversion
        if value.lower() in ('true', 'false'):
            return value.lower() == 'true'
            
        # Integer conversion
        try:
            return int(value)
        except ValueError:
            pass
            
        # Float conversion
        try:
            return float(value)
        except ValueError:
            pass
            
        # Return as string
        return value
    
    def _set_nested_value(self, config: Dict[str, Any], path: str, value: Any) -> None:
        """Set nested configuration value using dot notation."""
        keys = path.split('.')
        current = config
        
        for key in keys[:-1]:
            if key not in current:
                current[key] = {}
            current = current[key]
            
        current[keys[-1]] = value
    
    def _merge_configs(self, base: Dict[str, Any], overrides: Dict[str, Any]) -> Dict[str, Any]:
        """Merge configuration dictionaries."""
        result = base.copy()
        
        for key, value in overrides.items():
            if key in result and isinstance(result[key], dict) and isinstance(value, dict):
                result[key] = self._merge_configs(result[key], value)
            else:
                result[key] = value
                
        return result
    
    def _create_config_object(self, config_data: Dict[str, Any]) -> CoreArchitectureConfig:
        """Create configuration object from dictionary."""
        # Handle environment enum
        if 'environment' in config_data:
            env_str = config_data['environment']
            if isinstance(env_str, str):
                try:
                    config_data['environment'] = ConfigurationEnvironment(env_str)
                except ValueError:
                    config_data['environment'] = ConfigurationEnvironment.DEVELOPMENT
        
        # Handle version
        if 'version' in config_data:
            version_str = config_data['version']
            if isinstance(version_str, str):
                parts = version_str.split('.')
                if len(parts) >= 2:
                    config_data['version'] = SemanticVersion(
                        int(parts[0]),
                        int(parts[1]),
                        int(parts[2]) if len(parts) > 2 else 0
                    )
        
        # Create nested configurations
        data_ingestion_data = config_data.get('data_ingestion', {})
        anomaly_detection_data = config_data.get('anomaly_detection', {})
        alerting_data = config_data.get('alerting', {})
        
        # Create component configurations
        data_ingestion_config = DataIngestionConfig(**data_ingestion_data)
        anomaly_detection_config = AnomalyDetectionConfig(**anomaly_detection_data)
        alerting_config = AlertingConfig(**alerting_data)
        
        # Create main configuration
        main_config_data = {k: v for k, v in config_data.items() 
                           if k not in ['data_ingestion', 'anomaly_detection', 'alerting']}
        
        return CoreArchitectureConfig(
            data_ingestion=data_ingestion_config,
            anomaly_detection=anomaly_detection_config,
            alerting=alerting_config,
            **main_config_data
        )
    
    def _validate_config(self, config: CoreArchitectureConfig) -> None:
        """Validate configuration object."""
        # Validate data ingestion
        if config.data_ingestion.batch_size <= 0:
            raise ConfigurationError("Data ingestion batch_size must be positive")
        if config.data_ingestion.parallel_workers <= 0:
            raise ConfigurationError("Data ingestion parallel_workers must be positive")
        if config.data_ingestion.memory_limit_mb <= 0:
            raise ConfigurationError("Data ingestion memory_limit_mb must be positive")
        
        # Validate anomaly detection
        if not (0 < config.anomaly_detection.contamination < 1):
            raise ConfigurationError("Anomaly detection contamination must be between 0 and 1")
        if not (0 < config.anomaly_detection.confidence_threshold < 1):
            raise ConfigurationError("Anomaly detection confidence_threshold must be between 0 and 1")
        
        # Validate alerting
        if config.alerting.aggregation_window <= 0:
            raise ConfigurationError("Alerting aggregation_window must be positive")
        if config.alerting.max_alerts_per_minute <= 0:
            raise ConfigurationError("Alerting max_alerts_per_minute must be positive")
        
        # Validate system settings
        if config.max_concurrent_requests <= 0:
            raise ConfigurationError("max_concurrent_requests must be positive")
        if config.request_timeout <= 0:
            raise ConfigurationError("request_timeout must be positive")
        if config.connection_pool_size <= 0:
            raise ConfigurationError("connection_pool_size must be positive")
    
    def save_config(self, config: CoreArchitectureConfig, output_path: Optional[str] = None) -> None:
        """Save configuration to file.
        
        Args:
            config: Configuration to save
            output_path: Optional output path (defaults to current config path)
        """
        output_path = output_path or self.config_path
        
        # Convert to dictionary
        config_dict = self._config_to_dict(config)
        
        # Ensure directory exists
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        
        # Save based on file extension
        try:
            with open(output_path, 'w', encoding='utf-8') as f:
                if output_path.endswith(('.yml', '.yaml')):
                    yaml.dump(config_dict, f, default_flow_style=False, indent=2)
                elif output_path.endswith('.json'):
                    json.dump(config_dict, f, indent=2)
                else:
                    raise ConfigurationError(f"Unsupported output format: {output_path}")
                    
            self.logger.info(f"Configuration saved to {output_path}")
            
        except Exception as e:
            raise ConfigurationError(f"Failed to save configuration: {e}")
    
    def _config_to_dict(self, config: CoreArchitectureConfig) -> Dict[str, Any]:
        """Convert configuration object to dictionary."""
        return {
            'environment': config.environment.value,
            'version': f"{config.version.major}.{config.version.minor}.{config.version.patch}",
            'debug': config.debug,
            'logging_level': config.logging_level,
            'metrics_enabled': config.metrics_enabled,
            'health_check_enabled': config.health_check_enabled,
            'encryption_enabled': config.encryption_enabled,
            'authentication_required': config.authentication_required,
            'rate_limiting_enabled': config.rate_limiting_enabled,
            'max_concurrent_requests': config.max_concurrent_requests,
            'request_timeout': config.request_timeout,
            'connection_pool_size': config.connection_pool_size,
            'data_ingestion': {
                'sources': config.data_ingestion.sources,
                'batch_size': config.data_ingestion.batch_size,
                'max_batch_size': config.data_ingestion.max_batch_size,
                'batch_timeout': config.data_ingestion.batch_timeout,
                'stream_buffer_size': config.data_ingestion.stream_buffer_size,
                'stream_flush_interval': config.data_ingestion.stream_flush_interval,
                'validation_enabled': config.data_ingestion.validation_enabled,
                'validation_rules': config.data_ingestion.validation_rules,
                'parallel_workers': config.data_ingestion.parallel_workers,
                'memory_limit_mb': config.data_ingestion.memory_limit_mb,
                'retry_attempts': config.data_ingestion.retry_attempts,
                'retry_delay': config.data_ingestion.retry_delay,
                'error_threshold': config.data_ingestion.error_threshold,
            },
            'anomaly_detection': {
                'default_algorithm': config.anomaly_detection.default_algorithm,
                'available_algorithms': config.anomaly_detection.available_algorithms,
                'model_cache_size': config.anomaly_detection.model_cache_size,
                'model_cache_ttl': config.anomaly_detection.model_cache_ttl,
                'auto_retrain_enabled': config.anomaly_detection.auto_retrain_enabled,
                'retrain_threshold': config.anomaly_detection.retrain_threshold,
                'contamination': config.anomaly_detection.contamination,
                'confidence_threshold': config.anomaly_detection.confidence_threshold,
                'ensemble_enabled': config.anomaly_detection.ensemble_enabled,
                'ensemble_size': config.anomaly_detection.ensemble_size,
                'batch_prediction': config.anomaly_detection.batch_prediction,
                'prediction_batch_size': config.anomaly_detection.prediction_batch_size,
                'async_processing': config.anomaly_detection.async_processing,
                'explainability_enabled': config.anomaly_detection.explainability_enabled,
                'shap_enabled': config.anomaly_detection.shap_enabled,
                'lime_enabled': config.anomaly_detection.lime_enabled,
            },
            'alerting': {
                'channels': config.alerting.channels,
                'severity_levels': config.alerting.severity_levels,
                'thresholds': config.alerting.thresholds,
                'aggregation_window': config.alerting.aggregation_window,
                'max_alerts_per_minute': config.alerting.max_alerts_per_minute,
                'deduplication_enabled': config.alerting.deduplication_enabled,
                'routing_rules': config.alerting.routing_rules,
                'notification_delay': config.alerting.notification_delay,
                'retry_failed_notifications': config.alerting.retry_failed_notifications,
                'notification_timeout': config.alerting.notification_timeout,
            }
        }
    
    def get_config_info(self) -> Dict[str, Any]:
        """Get configuration information."""
        config = self.load_config()
        return {
            'environment': config.environment.value,
            'version': f"{config.version.major}.{config.version.minor}.{config.version.patch}",
            'config_path': self.config_path,
            'components': {
                'data_ingestion': {
                    'sources': len(config.data_ingestion.sources),
                    'batch_size': config.data_ingestion.batch_size,
                    'parallel_workers': config.data_ingestion.parallel_workers,
                },
                'anomaly_detection': {
                    'default_algorithm': config.anomaly_detection.default_algorithm,
                    'available_algorithms': len(config.anomaly_detection.available_algorithms),
                    'ensemble_enabled': config.anomaly_detection.ensemble_enabled,
                },
                'alerting': {
                    'channels': len(config.alerting.channels),
                    'severity_levels': len(config.alerting.severity_levels),
                    'thresholds': len(config.alerting.thresholds),
                }
            }
        }


# Global configuration loader instance
_config_loader: Optional[EnhancedConfigLoader] = None


def get_config_loader() -> EnhancedConfigLoader:
    """Get the global configuration loader instance."""
    global _config_loader
    if _config_loader is None:
        _config_loader = EnhancedConfigLoader()
    return _config_loader


def get_config() -> CoreArchitectureConfig:
    """Get the current configuration."""
    return get_config_loader().load_config()


def reload_config() -> CoreArchitectureConfig:
    """Reload configuration from file."""
    return get_config_loader().load_config(force_reload=True)
