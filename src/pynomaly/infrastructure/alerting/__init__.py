"""
Pynomaly Real-time Alerting and Notification System.

This module provides comprehensive alerting capabilities including:
- Alert rule management
- Real-time metric processing
- Multi-channel notifications (Email, Slack, Discord, Teams, PagerDuty)
- WebSocket-based real-time alerts
- Prometheus metrics integration
- SQLAlchemy database persistence
"""

# Legacy intelligent alerting engine (kept for backward compatibility)
# New real-time alerting system components
from .alert_manager import (
    AlertInfo,
    AlertManager,
    AlertRuleCreate,
    AlertRuleUpdate,
    AlertStatus,
    get_alert_manager,
)
from .alert_manager import (
    AlertRule as NewAlertRule,
)
from .alert_manager import (
    AlertSeverity as NewAlertSeverity,
)
from .alert_manager import (
    NotificationChannel as NewNotificationChannel,
)
from .alerting_service import (
    AlertRuleResponse,
    AlertSystemStatus,
    MetricSubmission,
    websocket_connections,
)
from .alerting_service import (
    router as alerting_router,
)
from .intelligent_alerting_engine import (
    Alert,
    AlertCategory,
    AlertRule,
    AlertSeverity,
    EmailNotificationProvider,
    IntelligentAlertingEngine,
    NotificationChannel,
    NotificationProvider,
    SlackNotificationProvider,
    WebhookNotificationProvider,
    create_default_alert_rules,
)
from .metric_collector import (
    MetricCollector,
    MetricInfo,
    get_metric_collector,
)

__all__ = [
    # Legacy components (backward compatibility)
    "IntelligentAlertingEngine",
    "AlertRule",
    "Alert",
    "AlertSeverity",
    "AlertCategory",
    "NotificationChannel",
    "NotificationProvider",
    "EmailNotificationProvider",
    "SlackNotificationProvider",
    "WebhookNotificationProvider",
    "create_default_alert_rules",

    # New real-time alerting system
    "AlertManager",
    "AlertInfo",
    "NewAlertRule",
    "AlertRuleCreate",
    "AlertRuleUpdate",
    "NewAlertSeverity",
    "AlertStatus",
    "NewNotificationChannel",
    "get_alert_manager",

    # Metric Collector
    "MetricCollector",
    "MetricInfo",
    "get_metric_collector",

    # Alerting Service
    "alerting_router",
    "AlertSystemStatus",
    "MetricSubmission",
    "AlertRuleResponse",
    "websocket_connections",
]
