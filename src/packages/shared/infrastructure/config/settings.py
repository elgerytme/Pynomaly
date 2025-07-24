"""Configuration management for anomaly detection package."""

from __future__ import annotations

import os
import json
from pathlib import Path
from typing import Any, Optional, Dict, List
from dataclasses import dataclass, field


@dataclass
class DatabaseSettings:
    """Database configuration settings."""
    
    host: str = "localhost"
    port: int = 5432
    database: str = "anomaly_detection"
    username: str = "postgres"
    password: str = ""
    pool_size: int = 10
    
    @property
    def url(self) -> str:
        """Get database URL."""
        if self.password:
            return f"postgresql://{self.username}:{self.password}@{self.host}:{self.port}/{self.database}"
        else:
            return f"postgresql://{self.username}@{self.host}:{self.port}/{self.database}"
    
    @classmethod
    def from_env(cls) -> DatabaseSettings:
        """Load from environment variables."""
        return cls(
            host=os.getenv("DB_HOST", "localhost"),
            port=int(os.getenv("DB_PORT", "5432")),
            database=os.getenv("DB_DATABASE", "anomaly_detection"),
            username=os.getenv("DB_USERNAME", "postgres"),
            password=os.getenv("DB_PASSWORD", ""),
            pool_size=int(os.getenv("DB_POOL_SIZE", "10"))
        )


@dataclass
class LoggingSettings:
    """Enhanced logging configuration settings."""
    
    level: str = "INFO"
    format: str = "json"
    output: str = "console"
    file_path: Optional[str] = None
    max_file_size: str = "10MB"
    backup_count: int = 5
    file_enabled: bool = True
    
    # Enhanced logging features
    enable_structured_logging: bool = True
    enable_request_tracking: bool = True
    enable_performance_logging: bool = True
    enable_error_tracking: bool = True
    enable_metrics_logging: bool = True
    
    # Performance logging thresholds
    slow_query_threshold_ms: float = 1000.0
    slow_detection_threshold_ms: float = 5000.0
    slow_model_training_threshold_ms: float = 30000.0
    
    # Error handling settings
    log_stack_traces: bool = True
    include_request_context: bool = True
    sanitize_sensitive_data: bool = True
    
    @classmethod
    def from_env(cls) -> LoggingSettings:
        """Load from environment variables."""
        return cls(
            level=os.getenv("LOG_LEVEL", "INFO"),
            format=os.getenv("LOG_FORMAT", "json"),
            output=os.getenv("LOG_OUTPUT", "console"),
            file_path=os.getenv("LOG_FILE_PATH"),
            max_file_size=os.getenv("LOG_MAX_FILE_SIZE", "10MB"),
            backup_count=int(os.getenv("LOG_BACKUP_COUNT", "5")),
            file_enabled=os.getenv("LOG_FILE_ENABLED", "true").lower() == "true",
            
            enable_structured_logging=os.getenv("LOG_STRUCTURED", "true").lower() == "true",
            enable_request_tracking=os.getenv("LOG_REQUEST_TRACKING", "true").lower() == "true",
            enable_performance_logging=os.getenv("LOG_PERFORMANCE", "true").lower() == "true",
            enable_error_tracking=os.getenv("LOG_ERROR_TRACKING", "true").lower() == "true",
            enable_metrics_logging=os.getenv("LOG_METRICS", "true").lower() == "true",
            
            slow_query_threshold_ms=float(os.getenv("LOG_SLOW_QUERY_THRESHOLD_MS", "1000")),
            slow_detection_threshold_ms=float(os.getenv("LOG_SLOW_DETECTION_THRESHOLD_MS", "5000")),
            slow_model_training_threshold_ms=float(os.getenv("LOG_SLOW_TRAINING_THRESHOLD_MS", "30000")),
            
            log_stack_traces=os.getenv("LOG_STACK_TRACES", "true").lower() == "true",
            include_request_context=os.getenv("LOG_REQUEST_CONTEXT", "true").lower() == "true",
            sanitize_sensitive_data=os.getenv("LOG_SANITIZE_DATA", "true").lower() == "true"
        )


@dataclass
class DetectionSettings:
    """Detection algorithm configuration settings."""
    
    default_algorithm: str = "isolation_forest"
    default_contamination: float = 0.1
    max_samples: int = 100000
    timeout_seconds: int = 300
    
    # Algorithm-specific settings
    isolation_forest_estimators: int = 100
    lof_neighbors: int = 20
    ocsvm_kernel: str = "rbf"
    
    @classmethod
    def from_env(cls) -> DetectionSettings:
        """Load from environment variables."""
        return cls(
            default_algorithm=os.getenv("DETECTION_DEFAULT_ALGORITHM", "isolation_forest"),
            default_contamination=float(os.getenv("DETECTION_DEFAULT_CONTAMINATION", "0.1")),
            max_samples=int(os.getenv("DETECTION_MAX_SAMPLES", "100000")),
            timeout_seconds=int(os.getenv("DETECTION_TIMEOUT_SECONDS", "300")),
            isolation_forest_estimators=int(os.getenv("DETECTION_IF_ESTIMATORS", "100")),
            lof_neighbors=int(os.getenv("DETECTION_LOF_NEIGHBORS", "20")),
            ocsvm_kernel=os.getenv("DETECTION_OCSVM_KERNEL", "rbf")
        )


@dataclass
class StreamingSettings:
    """Streaming detection configuration settings."""
    
    buffer_size: int = 1000
    update_frequency: int = 100
    concept_drift_threshold: float = 0.05
    max_buffer_size: int = 10000
    
    @classmethod
    def from_env(cls) -> StreamingSettings:
        """Load from environment variables."""
        return cls(
            buffer_size=int(os.getenv("STREAMING_BUFFER_SIZE", "1000")),
            update_frequency=int(os.getenv("STREAMING_UPDATE_FREQUENCY", "100")),
            concept_drift_threshold=float(os.getenv("STREAMING_DRIFT_THRESHOLD", "0.05")),
            max_buffer_size=int(os.getenv("STREAMING_MAX_BUFFER_SIZE", "10000"))
        )


@dataclass
class APISettings:
    """API server configuration settings."""
    
    host: str = "0.0.0.0"
    port: int = 8000
    workers: int = 1
    reload: bool = False
    debug: bool = False
    cors_origins: List[str] = field(default_factory=lambda: ["*"])
    
    @classmethod
    def from_env(cls) -> APISettings:
        """Load from environment variables."""
        cors_origins_str = os.getenv("API_CORS_ORIGINS", "*")
        cors_origins = [origin.strip() for origin in cors_origins_str.split(",")]
        
        return cls(
            host=os.getenv("API_HOST", "0.0.0.0"),
            port=int(os.getenv("API_PORT", "8000")),
            workers=int(os.getenv("API_WORKERS", "1")),
            reload=os.getenv("API_RELOAD", "false").lower() == "true",
            debug=os.getenv("API_DEBUG", "false").lower() == "true",
            cors_origins=cors_origins
        )


@dataclass
class MonitoringSettings:
    """Monitoring and observability settings."""
    
    enable_metrics: bool = True
    metrics_port: int = 9090
    enable_tracing: bool = False
    jaeger_endpoint: Optional[str] = None
    
    @classmethod
    def from_env(cls) -> MonitoringSettings:
        """Load from environment variables."""
        return cls(
            enable_metrics=os.getenv("MONITORING_ENABLE_METRICS", "true").lower() == "true",
            metrics_port=int(os.getenv("MONITORING_METRICS_PORT", "9090")),
            enable_tracing=os.getenv("MONITORING_ENABLE_TRACING", "false").lower() == "true",
            jaeger_endpoint=os.getenv("MONITORING_JAEGER_ENDPOINT")
        )


@dataclass
class Settings:
    """Main application settings."""
    
    # Environment and deployment
    environment: str = "development"
    debug: bool = False
    app_name: str = "Anomaly Detection Service"
    version: str = "0.1.0"
    
    # Component settings
    database: DatabaseSettings = field(default_factory=DatabaseSettings)
    logging: LoggingSettings = field(default_factory=LoggingSettings)
    detection: DetectionSettings = field(default_factory=DetectionSettings)
    streaming: StreamingSettings = field(default_factory=StreamingSettings)
    api: APISettings = field(default_factory=APISettings)
    monitoring: MonitoringSettings = field(default_factory=MonitoringSettings)
    
    # Security settings
    secret_key: str = "dev-secret-key-change-in-production"
    api_key: Optional[str] = None
    
    @classmethod
    def load(cls) -> Settings:
        """Load settings from environment and configuration files."""
        settings = cls()
        
        # Load from environment
        settings.environment = os.getenv("ENVIRONMENT", "development")
        settings.debug = os.getenv("DEBUG", "false").lower() == "true"
        settings.app_name = os.getenv("APP_NAME", "Anomaly Detection Service")
        settings.version = os.getenv("VERSION", "0.1.0")
        settings.secret_key = os.getenv("SECRET_KEY", "dev-secret-key-change-in-production")
        settings.api_key = os.getenv("API_KEY")
        
        # Load component settings from environment
        settings.database = DatabaseSettings.from_env()
        settings.logging = LoggingSettings.from_env()
        settings.detection = DetectionSettings.from_env()
        settings.streaming = StreamingSettings.from_env()
        settings.api = APISettings.from_env()
        settings.monitoring = MonitoringSettings.from_env()
        
        # Try to load from configuration file
        config_data = _load_config_file()
        if config_data:
            settings = _update_from_dict(settings, config_data)
        
        return settings


def _load_config_file() -> Optional[Dict[str, Any]]:
    """Load configuration from JSON file."""
    config_paths = [
        Path("config.json"),
        Path("config/config.json"),
        Path(".config/anomaly_detection/config.json"),
        Path.home() / ".config" / "anomaly_detection" / "config.json",
    ]
    
    for config_path in config_paths:
        if config_path.exists():
            try:
                with open(config_path, 'r', encoding='utf-8') as f:
                    return json.load(f)
            except Exception as e:
                print(f"Warning: Could not load config from {config_path}: {e}")
    
    return None


def _update_from_dict(settings: Settings, data: Dict[str, Any]) -> Settings:
    """Update settings from dictionary."""
    # This is a simple implementation - in production you might want
    # more sophisticated merging logic
    
    # Update top-level settings
    for key in ["environment", "debug", "app_name", "version", "secret_key", "api_key"]:
        if key in data:
            setattr(settings, key, data[key])
    
    # Update component settings
    component_map = {
        "database": DatabaseSettings,
        "logging": LoggingSettings,
        "detection": DetectionSettings,
        "streaming": StreamingSettings,
        "api": APISettings,
        "monitoring": MonitoringSettings
    }
    
    for component_name, component_class in component_map.items():
        if component_name in data:
            component_data = data[component_name]
            current_component = getattr(settings, component_name)
            
            # Update fields in the component
            for field_name, field_value in component_data.items():
                if hasattr(current_component, field_name):
                    setattr(current_component, field_name, field_value)
    
    return settings


# Global settings instance
settings = Settings.load()


def get_settings() -> Settings:
    """Get the global settings instance."""
    return settings


def reload_settings() -> Settings:
    """Reload settings from all sources."""
    global settings
    settings = Settings.load()
    return settings


def create_example_config() -> None:
    """Create example configuration files."""
    
    example_config = {
        "environment": "development",
        "debug": False,
        "app_name": "Anomaly Detection Service",
        "version": "0.1.0",
        "database": {
            "host": "localhost",
            "port": 5432,
            "database": "anomaly_detection",
            "username": "postgres",
            "password": ""
        },
        "logging": {
            "level": "INFO",
            "format": "json",
            "output": "console"
        },
        "detection": {
            "default_algorithm": "isolation_forest",
            "default_contamination": 0.1,
            "max_samples": 100000,
            "timeout_seconds": 300
        },
        "streaming": {
            "buffer_size": 1000,
            "update_frequency": 100,
            "concept_drift_threshold": 0.05
        },
        "api": {
            "host": "0.0.0.0",
            "port": 8000,
            "workers": 1,
            "cors_origins": ["*"]
        },
        "monitoring": {
            "enable_metrics": True,
            "metrics_port": 9090,
            "enable_tracing": False
        }
    }
    
    example_env = """# Anomaly Detection Environment Variables

# Application
ENVIRONMENT=development
DEBUG=false
APP_NAME="Anomaly Detection Service"
SECRET_KEY=dev-secret-key-change-in-production

# Database
DB_HOST=localhost
DB_PORT=5432
DB_DATABASE=anomaly_detection
DB_USERNAME=postgres
DB_PASSWORD=

# Logging
LOG_LEVEL=INFO
LOG_FORMAT=json
LOG_OUTPUT=console

# Detection
DETECTION_DEFAULT_ALGORITHM=isolation_forest
DETECTION_DEFAULT_CONTAMINATION=0.1
DETECTION_MAX_SAMPLES=100000

# API
API_HOST=0.0.0.0
API_PORT=8000
API_WORKERS=1

# Monitoring
MONITORING_ENABLE_METRICS=true
MONITORING_METRICS_PORT=9090
"""
    
    # Write example files
    with open("config.example.json", "w", encoding="utf-8") as f:
        json.dump(example_config, f, indent=2)
    
    with open(".env.example", "w", encoding="utf-8") as f:
        f.write(example_env)
    
    print("✅ Created example configuration files:")
    print("   - config.example.json")
    print("   - .env.example")


if __name__ == "__main__":
    # Create example config files when run directly
    create_example_config()
    
    # Print current settings
    print("\n📋 Current Settings:")
    print(f"   Environment: {settings.environment}")
    print(f"   Debug: {settings.debug}")
    print(f"   Default Algorithm: {settings.detection.default_algorithm}")
    print(f"   API Port: {settings.api.port}")
    print(f"   Database URL: {settings.database.url}")