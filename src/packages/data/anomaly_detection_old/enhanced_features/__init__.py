"""Enhanced features and capabilities for anomaly detection."""

from .model_persistence import ModelPersistence, ModelMetadata
from .advanced_explainability import AdvancedExplainability, ExplanationResult, GlobalExplanation
from .integration_adapters import (
    IntegrationManager, 
    create_adapter,
    FileSystemAdapter,
    DatabaseAdapter,
    APIAdapter,
    StreamingAdapter,
    IntegrationConfig
)
from .monitoring_alerting import (
    MonitoringAlertingSystem,
    Alert,
    AlertRule,
    AlertSeverity,
    AlertStatus,
    console_notification_handler,
    email_notification_handler,
    slack_notification_handler
)

__all__ = [
    # Model persistence
    "ModelPersistence",
    "ModelMetadata",
    
    # Explainability
    "AdvancedExplainability",
    "ExplanationResult", 
    "GlobalExplanation",
    
    # Integration adapters
    "IntegrationManager",
    "create_adapter",
    "FileSystemAdapter",
    "DatabaseAdapter", 
    "APIAdapter",
    "StreamingAdapter",
    "IntegrationConfig",
    
    # Monitoring and alerting
    "MonitoringAlertingSystem",
    "Alert",
    "AlertRule",
    "AlertSeverity",
    "AlertStatus",
    "console_notification_handler",
    "email_notification_handler",
    "slack_notification_handler",
]