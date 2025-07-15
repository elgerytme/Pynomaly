"""
Enhanced Configuration Loader for Core Architecture Components

This module provides a comprehensive configuration loading system that supports
the three main sub-tasks: data_ingestion, anomaly_detection, and alerting.
"""

import json
import logging
import os
from dataclasses import dataclass, field
from enum import Enum
from typing import Any

import yaml

from pynomaly.domain.value_objects.semantic_version import SemanticVersion
from pynomaly.shared.exceptions import ConfigurationError


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
    sources: dict[str, dict[str, Any]] = field(default_factory=dict)

    # Batch processing settings
    batch_size: int = 1000
    max_batch_size: int = 10000
    batch_timeout: int = 30

    # Streaming settings
    stream_buffer_size: int = 1000
    stream_flush_interval: int = 5

    # Data validation settings
    validation_enabled: bool = True
    validation_rules: dict[str, Any] = field(default_factory=dict)

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
    available_algorithms: list[str] = field(
        default_factory=lambda: [
            "isolation_forest",
            "local_outlier_factor",
            "one_class_svm",
            "elliptic_envelope",
            "autoencoder",
        ]
    )

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
    channels: dict[str, dict[str, Any]] = field(default_factory=dict)

    # Alert levels
    severity_levels: list[str] = field(
        default_factory=lambda: ["low", "medium", "high", "critical"]
    )

    # Alert thresholds
    thresholds: dict[str, float] = field(
        default_factory=lambda: {
            "low": 0.3,
            "medium": 0.5,
            "high": 0.7,
            "critical": 0.9,
        }
    )

    # Alert aggregation
    aggregation_window: int = 60
    max_alerts_per_minute: int = 10
    deduplication_enabled: bool = True

    # Alert routing
    routing_rules: list[dict[str, Any]] = field(default_factory=list)

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
    anomaly_detection: AnomalyDetectionConfig = field(
        default_factory=AnomalyDetectionConfig
    )
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

    def __init__(self, config_path: str | None = None):
        """Initialize the enhanced config loader.

        Args:
            config_path: Optional path to configuration file
        """
        self.logger = logging.getLogger(__name__)
        self.config_path = config_path or self._get_default_config_path()
        self._config_cache: CoreArchitectureConfig | None = None

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
            os.path.expanduser("~/.pynomaly/config.yml"),
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
            # Load from file or create defaults
            config_data = self._load_from_file()

            # Create configuration object with defaults
            config = CoreArchitectureConfig()

            # Apply any loaded data
            if config_data:
                config = self._apply_config_data(config, config_data)

            # Validate configuration
            self._validate_config(config)

            # Cache the configuration
            self._config_cache = config

            self.logger.info(
                f"Configuration loaded successfully from {self.config_path}"
            )
            return config

        except Exception as e:
            self.logger.error(f"Failed to load configuration: {e}")
            raise ConfigurationError(f"Configuration loading failed: {e}")

    def _load_from_file(self) -> dict[str, Any]:
        """Load configuration from file."""
        if not os.path.exists(self.config_path):
            self.logger.warning(f"Configuration file not found: {self.config_path}")
            return {}

        try:
            with open(self.config_path, encoding="utf-8") as f:
                if self.config_path.endswith((".yml", ".yaml")):
                    return yaml.safe_load(f) or {}
                elif self.config_path.endswith(".json"):
                    return json.load(f) or {}
                else:
                    raise ConfigurationError(
                        f"Unsupported configuration file format: {self.config_path}"
                    )
        except Exception as e:
            raise ConfigurationError(f"Failed to parse configuration file: {e}")

    def _apply_config_data(
        self, config: CoreArchitectureConfig, data: dict[str, Any]
    ) -> CoreArchitectureConfig:
        """Apply loaded configuration data to config object."""
        # This is a simplified implementation
        # In a real scenario, you would need more sophisticated merging logic

        # Apply top-level settings
        if "debug" in data:
            config.debug = data["debug"]
        if "logging_level" in data:
            config.logging_level = data["logging_level"]

        # Apply component configurations
        if "data_ingestion" in data:
            di_config = data["data_ingestion"]
            if "batch_size" in di_config:
                config.data_ingestion.batch_size = di_config["batch_size"]
            if "parallel_workers" in di_config:
                config.data_ingestion.parallel_workers = di_config["parallel_workers"]

        if "anomaly_detection" in data:
            ad_config = data["anomaly_detection"]
            if "default_algorithm" in ad_config:
                config.anomaly_detection.default_algorithm = ad_config[
                    "default_algorithm"
                ]
            if "contamination" in ad_config:
                config.anomaly_detection.contamination = ad_config["contamination"]

        if "alerting" in data:
            alert_config = data["alerting"]
            if "channels" in alert_config:
                config.alerting.channels = alert_config["channels"]
            if "thresholds" in alert_config:
                config.alerting.thresholds = alert_config["thresholds"]

        return config

    def _validate_config(self, config: CoreArchitectureConfig) -> None:
        """Validate configuration object."""
        # Validate data ingestion
        if config.data_ingestion.batch_size <= 0:
            raise ConfigurationError("Data ingestion batch_size must be positive")
        if config.data_ingestion.parallel_workers <= 0:
            raise ConfigurationError("Data ingestion parallel_workers must be positive")

        # Validate anomaly detection
        if not (0 < config.anomaly_detection.contamination < 1):
            raise ConfigurationError(
                "Anomaly detection contamination must be between 0 and 1"
            )
        if not (0 < config.anomaly_detection.confidence_threshold < 1):
            raise ConfigurationError(
                "Anomaly detection confidence_threshold must be between 0 and 1"
            )

        # Validate alerting
        if config.alerting.aggregation_window <= 0:
            raise ConfigurationError("Alerting aggregation_window must be positive")

        # Validate system settings
        if config.max_concurrent_requests <= 0:
            raise ConfigurationError("max_concurrent_requests must be positive")


# Global configuration loader instance
_config_loader: EnhancedConfigLoader | None = None


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
