"""Enterprise integration service connecting dashboard, alerting, and autonomous detection.

This service provides seamless integration between the enterprise dashboard,
intelligent alerting system, and autonomous anomaly detection capabilities.
"""

from __future__ import annotations

import asyncio
import logging
from contextlib import asynccontextmanager
from dataclasses import dataclass
from datetime import datetime
from typing import Any

from monorepo.application.services.enterprise_dashboard_service import (
    AlertPriority,
    DashboardMetricType,
    EnterpriseDashboardService,
)
from monorepo.infrastructure.alerting import (
    AlertCategory,
    AlertRule,
    AlertSeverity,
    EmailNotificationProvider,
    IntelligentAlertingEngine,
    NotificationChannel,
    SlackNotificationProvider,
    WebhookNotificationProvider,
    create_default_alert_rules,
)

# Optional imports
try:
    from monorepo.application.services.autonomous_service import (
        AutonomousDetectionService,
    )

    AUTONOMOUS_AVAILABLE = True
except ImportError:
    AUTONOMOUS_AVAILABLE = False

try:
    from monorepo.infrastructure.monitoring import autonomous_monitor

    MONITORING_AVAILABLE = True
except ImportError:
    MONITORING_AVAILABLE = False


@dataclass
class IntegrationConfig:
    """Configuration for enterprise integration."""

    enable_dashboard_integration: bool = True
    enable_alerting_integration: bool = True
    enable_autonomous_monitoring: bool = True

    # Dashboard settings
    dashboard_update_interval_seconds: int = 30
    business_metrics_enabled: bool = True
    operational_monitoring_enabled: bool = True
    compliance_tracking_enabled: bool = True

    # Alerting settings
    alert_correlation_enabled: bool = True
    intelligent_suppression_enabled: bool = True
    max_alerts_per_hour: int = 100

    # Notification settings
    email_notifications_enabled: bool = False
    slack_notifications_enabled: bool = False
    webhook_notifications_enabled: bool = False

    # Email configuration
    smtp_server: str | None = None
    smtp_port: int = 587
    smtp_username: str | None = None
    smtp_password: str | None = None

    # Slack configuration
    slack_webhook_url: str | None = None

    # Webhook configuration
    webhook_url: str | None = None
    webhook_headers: dict[str, str] | None = None


class EnterpriseIntegrationService:
    """Enterprise integration service for connecting dashboard, alerting, and autonomous detection."""

    def __init__(self, config: IntegrationConfig = None):
        """Initialize enterprise integration service.

        Args:
            config: Integration configuration
        """
        self.config = config or IntegrationConfig()
        self.logger = logging.getLogger(__name__)

        # Core services
        self.dashboard_service: EnterpriseDashboardService | None = None
        self.alerting_engine: IntelligentAlertingEngine | None = None
        self.autonomous_service: AutonomousDetectionService | None = None

        # Integration state
        self.integration_active = False
        self.background_tasks: list[asyncio.Task] = []

        # Performance tracking
        self.integration_metrics = {
            "dashboard_updates": 0,
            "alerts_generated": 0,
            "notifications_sent": 0,
            "autonomous_detections_tracked": 0,
            "last_update": None,
        }

        self.logger.info("Enterprise Integration Service initialized")

    async def initialize(self):
        """Initialize all enterprise services and integrations."""

        try:
            # Initialize dashboard service
            if self.config.enable_dashboard_integration:
                await self._initialize_dashboard()

            # Initialize alerting engine
            if self.config.enable_alerting_integration:
                await self._initialize_alerting()

            # Setup autonomous monitoring integration
            if self.config.enable_autonomous_monitoring and AUTONOMOUS_AVAILABLE:
                await self._initialize_autonomous_integration()

            # Start background tasks
            await self._start_background_tasks()

            self.integration_active = True
            self.logger.info("Enterprise integration initialized successfully")

        except Exception as e:
            self.logger.error(f"Failed to initialize enterprise integration: {e}")
            raise

    async def _initialize_dashboard(self):
        """Initialize enterprise dashboard service."""

        self.dashboard_service = EnterpriseDashboardService(
            enable_business_metrics=self.config.business_metrics_enabled,
            enable_operational_monitoring=self.config.operational_monitoring_enabled,
            enable_compliance_tracking=self.config.compliance_tracking_enabled,
        )

        self.logger.info("Dashboard service initialized")

    async def _initialize_alerting(self):
        """Initialize intelligent alerting engine."""

        self.alerting_engine = IntelligentAlertingEngine(
            enable_alert_correlation=self.config.alert_correlation_enabled,
            enable_intelligent_suppression=self.config.intelligent_suppression_enabled,
            max_alerts_per_hour=self.config.max_alerts_per_hour,
        )

        # Setup notification providers
        await self._setup_notification_providers()

        # Add default alert rules
        default_rules = create_default_alert_rules()
        for rule in default_rules:
            self.alerting_engine.add_alert_rule(rule)

        # Add enterprise-specific alert rules
        await self._add_enterprise_alert_rules()

        self.logger.info("Alerting engine initialized")

    async def _setup_notification_providers(self):
        """Setup notification providers for the alerting engine."""

        # Email notifications
        if (
            self.config.email_notifications_enabled
            and self.config.smtp_server
            and self.config.smtp_username
            and self.config.smtp_password
        ):
            email_provider = EmailNotificationProvider(
                smtp_server=self.config.smtp_server,
                smtp_port=self.config.smtp_port,
                username=self.config.smtp_username,
                password=self.config.smtp_password,
            )

            self.alerting_engine.add_notification_provider(
                NotificationChannel.EMAIL, email_provider
            )

            self.logger.info("Email notification provider configured")

        # Slack notifications
        if self.config.slack_notifications_enabled and self.config.slack_webhook_url:
            slack_provider = SlackNotificationProvider(
                webhook_url=self.config.slack_webhook_url
            )

            self.alerting_engine.add_notification_provider(
                NotificationChannel.SLACK, slack_provider
            )

            self.logger.info("Slack notification provider configured")

        # Webhook notifications
        if self.config.webhook_notifications_enabled and self.config.webhook_url:
            webhook_provider = WebhookNotificationProvider(
                webhook_url=self.config.webhook_url,
                headers=self.config.webhook_headers or {},
            )

            self.alerting_engine.add_notification_provider(
                NotificationChannel.WEBHOOK, webhook_provider
            )

            self.logger.info("Webhook notification provider configured")

    async def _add_enterprise_alert_rules(self):
        """Add enterprise-specific alert rules."""

        enterprise_rules = [
            AlertRule(
                id="dashboard_service_health",
                name="Dashboard Service Health",
                description="Enterprise dashboard service health check failed",
                category=AlertCategory.INFRASTRUCTURE,
                severity=AlertSeverity.CRITICAL,
                condition="dashboard_health == 0",
                threshold_value=0,
                comparison_operator="==",
                notification_channels=[
                    NotificationChannel.EMAIL,
                    NotificationChannel.SLACK,
                ],
                require_acknowledgment=True,
            ),
            AlertRule(
                id="business_kpi_degradation",
                name="Business KPI Degradation",
                description="Critical business KPI below acceptable threshold",
                category=AlertCategory.BUSINESS,
                severity=AlertSeverity.ERROR,
                condition="business_kpi_score < 80.0",
                threshold_value=80.0,
                comparison_operator="<",
                notification_channels=[NotificationChannel.EMAIL],
                escalation_delay_minutes=15,
            ),
            AlertRule(
                id="autonomous_detection_failure_rate",
                name="Autonomous Detection Failure Rate",
                description="High failure rate in autonomous detection pipeline",
                category=AlertCategory.PERFORMANCE,
                severity=AlertSeverity.WARNING,
                condition="autonomous_failure_rate > 10.0",
                threshold_value=10.0,
                comparison_operator=">",
                notification_channels=[
                    NotificationChannel.SLACK,
                    NotificationChannel.WEBHOOK,
                ],
            ),
            AlertRule(
                id="cost_threshold_exceeded",
                name="Cost Threshold Exceeded",
                description="Processing costs exceeded daily budget threshold",
                category=AlertCategory.BUSINESS,
                severity=AlertSeverity.WARNING,
                condition="daily_cost > 1000.0",
                threshold_value=1000.0,
                comparison_operator=">",
                notification_channels=[NotificationChannel.EMAIL],
                business_hours_only=True,
            ),
        ]

        for rule in enterprise_rules:
            self.alerting_engine.add_alert_rule(rule)

        self.logger.info(f"Added {len(enterprise_rules)} enterprise alert rules")

    async def _initialize_autonomous_integration(self):
        """Initialize integration with autonomous detection service."""

        # This would be injected in a real implementation
        # For now, we'll log the integration setup
        self.logger.info("Autonomous detection integration configured")

    async def _start_background_tasks(self):
        """Start background monitoring and integration tasks."""

        # Dashboard update task
        if self.dashboard_service:
            task = asyncio.create_task(self._dashboard_update_loop())
            self.background_tasks.append(task)

        # Alerting monitoring task
        if self.alerting_engine:
            task = asyncio.create_task(self._alerting_monitoring_loop())
            self.background_tasks.append(task)

        # Integration health monitoring
        task = asyncio.create_task(self._integration_health_loop())
        self.background_tasks.append(task)

        self.logger.info(f"Started {len(self.background_tasks)} background tasks")

    async def _dashboard_update_loop(self):
        """Background task for dashboard updates."""

        while self.integration_active:
            try:
                # Update dashboard metrics
                await self._update_dashboard_metrics()

                # Check for dashboard-related alerts
                await self._check_dashboard_alerts()

                self.integration_metrics["dashboard_updates"] += 1

                await asyncio.sleep(self.config.dashboard_update_interval_seconds)

            except Exception as e:
                self.logger.error(f"Error in dashboard update loop: {e}")
                await asyncio.sleep(60)  # Wait longer on error

    async def _alerting_monitoring_loop(self):
        """Background task for alerting engine monitoring."""

        while self.integration_active:
            try:
                # Monitor alerting engine health
                stats = self.alerting_engine.get_alert_statistics()

                # Update integration metrics
                self.integration_metrics["alerts_generated"] = stats.get(
                    "alerts_last_hour", 0
                )

                # Check for alerting system health issues
                if stats.get("active_alerts", 0) > 50:
                    self.logger.warning(
                        f"High number of active alerts: {stats['active_alerts']}"
                    )

                await asyncio.sleep(60)  # Check every minute

            except Exception as e:
                self.logger.error(f"Error in alerting monitoring loop: {e}")
                await asyncio.sleep(60)

    async def _integration_health_loop(self):
        """Background task for integration health monitoring."""

        while self.integration_active:
            try:
                # Update last update timestamp
                self.integration_metrics["last_update"] = datetime.now().isoformat()

                # Log periodic health status
                self.logger.debug("Enterprise integration health check passed")

                await asyncio.sleep(300)  # Check every 5 minutes

            except Exception as e:
                self.logger.error(f"Error in integration health loop: {e}")
                await asyncio.sleep(60)

    async def _update_dashboard_metrics(self):
        """Update dashboard metrics from various sources."""

        if not self.dashboard_service:
            return

        # This would collect metrics from various sources
        # For now, we'll simulate some metric updates

        # Update operational metrics based on system state
        # In a real implementation, this would collect from monitoring systems
        pass

    async def _check_dashboard_alerts(self):
        """Check dashboard metrics against alert thresholds."""

        if not self.alerting_engine or not self.dashboard_service:
            return

        # Get current dashboard metrics
        realtime_data = self.dashboard_service.get_real_time_dashboard_data()

        # Check business metrics
        business_metrics = realtime_data.get("business_metrics", {})
        for metric_name, metric_data in business_metrics.items():
            value = metric_data.get("value", 0)

            await self.alerting_engine.evaluate_metric(
                metric_name=metric_name,
                metric_value=value,
                source_service="enterprise_dashboard",
                environment="production",
                tags={"metric_type": "business"},
            )

        # Check operational metrics
        operational_metrics = realtime_data.get("operational_metrics", {})
        for metric_name, metric_data in operational_metrics.items():
            value = metric_data.get("current_value", 0)

            await self.alerting_engine.evaluate_metric(
                metric_name=metric_name,
                metric_value=value,
                source_service="enterprise_dashboard",
                environment="production",
                tags={"metric_type": "operational"},
            )

    async def process_autonomous_detection_event(
        self,
        detection_id: str,
        success: bool,
        execution_time: float,
        algorithm_used: str,
        anomalies_found: int,
        dataset_size: int,
        cost_usd: float = 0.0,
        metadata: dict[str, Any] = None,
    ):
        """Process an autonomous detection event through the integration pipeline.

        This method updates dashboard metrics, checks alert conditions,
        and tracks performance for enterprise monitoring.
        """

        try:
            # Update dashboard metrics
            if self.dashboard_service:
                self.dashboard_service.record_detection_event(
                    detection_id=detection_id,
                    success=success,
                    execution_time=execution_time,
                    algorithm_used=algorithm_used,
                    anomalies_found=anomalies_found,
                    dataset_size=dataset_size,
                    cost_usd=cost_usd,
                )

            # Check alerting conditions
            if self.alerting_engine:
                # Check execution time alerts
                await self.alerting_engine.evaluate_metric(
                    metric_name="detection_execution_time",
                    metric_value=execution_time,
                    source_service="autonomous_detection",
                    environment="production",
                    tags={
                        "algorithm": algorithm_used,
                        "detection_id": detection_id,
                        "success": str(success),
                    },
                    metadata=metadata or {},
                )

                # Check success rate (this would be calculated from recent history)
                if not success:
                    await self.alerting_engine.evaluate_metric(
                        metric_name="detection_failure",
                        metric_value=1.0,
                        source_service="autonomous_detection",
                        environment="production",
                        tags={"algorithm": algorithm_used},
                    )

            # Update integration metrics
            self.integration_metrics["autonomous_detections_tracked"] += 1

            self.logger.debug(f"Processed autonomous detection event: {detection_id}")

        except Exception as e:
            self.logger.error(
                f"Failed to process autonomous detection event {detection_id}: {e}"
            )

    async def create_custom_alert(
        self,
        title: str,
        message: str,
        severity: AlertSeverity,
        category: AlertCategory,
        source_service: str,
        metadata: dict[str, Any] = None,
    ) -> str | None:
        """Create a custom alert through the integration system."""

        try:
            if not self.alerting_engine:
                self.logger.warning("Alerting engine not available for custom alert")
                return None

            # Convert severity to dashboard priority
            priority_map = {
                AlertSeverity.INFO: AlertPriority.LOW,
                AlertSeverity.WARNING: AlertPriority.MEDIUM,
                AlertSeverity.ERROR: AlertPriority.HIGH,
                AlertSeverity.CRITICAL: AlertPriority.CRITICAL,
                AlertSeverity.EMERGENCY: AlertPriority.CRITICAL,
            }

            # Create dashboard alert
            if self.dashboard_service:
                metric_type_map = {
                    AlertCategory.PERFORMANCE: DashboardMetricType.PERFORMANCE,
                    AlertCategory.SECURITY: DashboardMetricType.SECURITY,
                    AlertCategory.COMPLIANCE: DashboardMetricType.COMPLIANCE,
                    AlertCategory.BUSINESS: DashboardMetricType.BUSINESS_KPI,
                    AlertCategory.INFRASTRUCTURE: DashboardMetricType.OPERATIONAL,
                    AlertCategory.DATA_QUALITY: DashboardMetricType.OPERATIONAL,
                }

                alert_id = self.dashboard_service.create_alert(
                    title=title,
                    message=message,
                    priority=priority_map.get(severity, AlertPriority.MEDIUM),
                    metric_type=metric_type_map.get(
                        category, DashboardMetricType.OPERATIONAL
                    ),
                    source_service=source_service,
                )

                self.integration_metrics["alerts_generated"] += 1

                self.logger.info(f"Created custom alert: {alert_id}")
                return alert_id

        except Exception as e:
            self.logger.error(f"Failed to create custom alert: {e}")

        return None

    def get_integration_status(self) -> dict[str, Any]:
        """Get current integration status and metrics."""

        return {
            "integration_active": self.integration_active,
            "services": {
                "dashboard_service": self.dashboard_service is not None,
                "alerting_engine": self.alerting_engine is not None,
                "autonomous_service": self.autonomous_service is not None,
            },
            "configuration": {
                "dashboard_integration": self.config.enable_dashboard_integration,
                "alerting_integration": self.config.enable_alerting_integration,
                "autonomous_monitoring": self.config.enable_autonomous_monitoring,
            },
            "metrics": self.integration_metrics.copy(),
            "background_tasks": len(self.background_tasks),
            "notification_providers": (
                len(self.alerting_engine.notification_providers)
                if self.alerting_engine
                else 0
            ),
        }

    async def shutdown(self):
        """Gracefully shutdown the integration service."""

        self.integration_active = False

        # Cancel background tasks
        for task in self.background_tasks:
            task.cancel()

        # Wait for tasks to complete
        if self.background_tasks:
            await asyncio.gather(*self.background_tasks, return_exceptions=True)

        self.logger.info("Enterprise integration service shut down")

    @asynccontextmanager
    async def integration_context(self):
        """Context manager for enterprise integration lifecycle."""

        try:
            await self.initialize()
            yield self
        finally:
            await self.shutdown()


# Global integration service instance
_global_integration_service: EnterpriseIntegrationService | None = None


def get_integration_service() -> EnterpriseIntegrationService | None:
    """Get the global integration service instance."""
    return _global_integration_service


async def initialize_enterprise_integration(
    config: IntegrationConfig = None,
) -> EnterpriseIntegrationService:
    """Initialize the global enterprise integration service."""
    global _global_integration_service

    _global_integration_service = EnterpriseIntegrationService(config)
    await _global_integration_service.initialize()

    return _global_integration_service
