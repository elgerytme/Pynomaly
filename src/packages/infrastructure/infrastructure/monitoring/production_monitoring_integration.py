"""Production monitoring integration for Pynomaly.

This module provides a unified interface to integrate all monitoring components:
- Prometheus metrics
- Advanced alerting
- Real-time dashboard
- Health checks
- Performance monitoring
"""

import asyncio
import os
from datetime import datetime
from pathlib import Path
from typing import Any

from prometheus_client import CollectorRegistry, Counter, Gauge, Histogram
from prometheus_client.exposition import start_http_server

from ...shared.logging import get_logger
from .advanced_alerting_service import (
    AdvancedAlertingService,
    AlertChannel,
    AlertRule,
    AlertSeverity,
    NotificationChannel,
    ThresholdCondition,
)
from .health_checks import HealthCheckManager
from .metrics_service import MetricsService
from .performance_monitor import PerformanceMonitor
from .realtime_dashboard import RealtimeDashboard

logger = get_logger(__name__)


class ProductionMonitoringIntegration:
    """Unified production monitoring system integration."""

    def __init__(self, config: dict[str, Any] = None):
        self.config = config or {}
        self.registry = CollectorRegistry()

        # Core services
        self.alerting_service: AdvancedAlertingService | None = None
        self.dashboard: RealtimeDashboard | None = None
        self.health_manager: HealthCheckManager | None = None
        self.performance_monitor: PerformanceMonitor | None = None
        self.metrics_service: MetricsService | None = None

        # Prometheus metrics
        self.setup_core_metrics()

        # Background tasks
        self._monitoring_tasks: list[asyncio.Task] = []
        self._running = False

    def setup_core_metrics(self):
        """Set up core Prometheus metrics."""
        # Application metrics
        self.http_requests_total = Counter(
            'http_requests_total',
            'Total HTTP requests',
            ['method', 'endpoint', 'status'],
            registry=self.registry
        )

        self.http_request_duration = Histogram(
            'http_request_duration_seconds',
            'HTTP request duration',
            ['method', 'endpoint'],
            registry=self.registry
        )

        self.anomaly_detection_total = Counter(
            'anomaly_detection_total',
            'Total anomaly detections',
            ['algorithm', 'status'],
            registry=self.registry
        )

        self.anomaly_detection_duration = Histogram(
            'anomaly_detection_duration_seconds',
            'Anomaly detection duration',
            ['algorithm'],
            registry=self.registry
        )

        self.model_training_total = Counter(
            'model_training_total',
            'Total model training attempts',
            ['algorithm', 'status'],
            registry=self.registry
        )

        self.model_accuracy = Gauge(
            'model_accuracy',
            'Current model accuracy',
            ['algorithm'],
            registry=self.registry
        )

        # System metrics
        self.active_connections = Gauge(
            'active_connections',
            'Active connections',
            registry=self.registry
        )

        self.memory_usage_bytes = Gauge(
            'memory_usage_bytes',
            'Memory usage in bytes',
            registry=self.registry
        )

        self.cpu_usage_percent = Gauge(
            'cpu_usage_percent',
            'CPU usage percentage',
            registry=self.registry
        )

        # Alert metrics
        self.alerts_total = Counter(
            'alerts_total',
            'Total alerts',
            ['severity', 'rule'],
            registry=self.registry
        )

        self.active_alerts = Gauge(
            'active_alerts',
            'Current active alerts',
            ['severity'],
            registry=self.registry
        )

    async def initialize(self):
        """Initialize all monitoring components."""
        logger.info("Initializing production monitoring integration...")

        try:
            # Initialize alerting service
            await self._initialize_alerting()

            # Initialize dashboard
            await self._initialize_dashboard()

            # Initialize health checks
            await self._initialize_health_checks()

            # Initialize performance monitoring
            await self._initialize_performance_monitoring()

            # Initialize metrics service
            await self._initialize_metrics_service()

            # Setup default alert rules
            await self._setup_default_alert_rules()

            # Setup notification channels
            await self._setup_notification_channels()

            # Start Prometheus metrics server
            self._start_metrics_server()

            # Start background monitoring tasks
            await self._start_background_tasks()

            self._running = True
            logger.info("Production monitoring integration initialized successfully")

        except Exception as e:
            logger.error(f"Failed to initialize monitoring integration: {e}")
            raise

    async def _initialize_alerting(self):
        """Initialize advanced alerting service."""
        self.alerting_service = AdvancedAlertingService()
        await self.alerting_service.initialize()
        logger.info("Alerting service initialized")

    async def _initialize_dashboard(self):
        """Initialize real-time dashboard."""
        dashboard_config = self.config.get('dashboard', {})
        host = dashboard_config.get('host', '0.0.0.0')
        port = dashboard_config.get('port', 8080)

        self.dashboard = RealtimeDashboard(host=host, port=port)
        logger.info(f"Real-time dashboard initialized on {host}:{port}")

    async def _initialize_health_checks(self):
        """Initialize health check manager."""
        health_config = self.config.get('health_checks', {})
        self.health_manager = HealthCheckManager(health_config)
        await self.health_manager.initialize()
        logger.info("Health check manager initialized")

    async def _initialize_performance_monitoring(self):
        """Initialize performance monitor."""
        perf_config = self.config.get('performance', {})
        self.performance_monitor = PerformanceMonitor(perf_config)
        await self.performance_monitor.initialize()
        logger.info("Performance monitor initialized")

    async def _initialize_metrics_service(self):
        """Initialize metrics service."""
        metrics_config = self.config.get('metrics', {})
        self.metrics_service = MetricsService(
            registry=self.registry,
            config=metrics_config
        )
        await self.metrics_service.initialize()
        logger.info("Metrics service initialized")

    async def _setup_default_alert_rules(self):
        """Set up default alert rules for production."""
        if not self.alerting_service:
            return

        # High error rate alert
        error_rate_rule = AlertRule(
            name="High Error Rate",
            description="HTTP error rate exceeds 10% for 5 minutes",
            conditions=[
                ThresholdCondition(
                    metric_name="http_error_rate",
                    operator=">",
                    value=0.1,
                    window_minutes=5
                )
            ],
            severity=AlertSeverity.CRITICAL,
            channels=[AlertChannel.EMAIL, AlertChannel.SLACK],
            suppress_duration_minutes=30,
            escalation_enabled=True,
            escalation_delay_minutes=15
        )
        self.alerting_service.add_alert_rule(error_rate_rule)

        # High response time alert
        response_time_rule = AlertRule(
            name="High Response Time",
            description="95th percentile response time exceeds 2 seconds",
            conditions=[
                ThresholdCondition(
                    metric_name="http_response_time_p95",
                    operator=">",
                    value=2.0,
                    window_minutes=5
                )
            ],
            severity=AlertSeverity.WARNING,
            channels=[AlertChannel.EMAIL, AlertChannel.SLACK],
            suppress_duration_minutes=60
        )
        self.alerting_service.add_alert_rule(response_time_rule)

        # Anomaly detection failures
        anomaly_failure_rule = AlertRule(
            name="Anomaly Detection Failures",
            description="Multiple anomaly detection failures",
            conditions=[
                ThresholdCondition(
                    metric_name="anomaly_detection_failures",
                    operator=">=",
                    value=5,
                    window_minutes=60
                )
            ],
            severity=AlertSeverity.HIGH,
            channels=[AlertChannel.EMAIL, AlertChannel.SLACK],
            suppress_duration_minutes=120
        )
        self.alerting_service.add_alert_rule(anomaly_failure_rule)

        # Low model accuracy
        accuracy_rule = AlertRule(
            name="Low Model Accuracy",
            description="Model accuracy below 80%",
            conditions=[
                ThresholdCondition(
                    metric_name="model_accuracy",
                    operator="<",
                    value=0.8,
                    window_minutes=10
                )
            ],
            severity=AlertSeverity.WARNING,
            channels=[AlertChannel.EMAIL],
            suppress_duration_minutes=180
        )
        self.alerting_service.add_alert_rule(accuracy_rule)

        # System resource alerts
        cpu_rule = AlertRule(
            name="High CPU Usage",
            description="CPU usage above 80% for 10 minutes",
            conditions=[
                ThresholdCondition(
                    metric_name="cpu_usage_percent",
                    operator=">",
                    value=80,
                    window_minutes=10
                )
            ],
            severity=AlertSeverity.WARNING,
            channels=[AlertChannel.EMAIL],
            suppress_duration_minutes=60
        )
        self.alerting_service.add_alert_rule(cpu_rule)

        memory_rule = AlertRule(
            name="High Memory Usage",
            description="Memory usage above 85% for 10 minutes",
            conditions=[
                ThresholdCondition(
                    metric_name="memory_usage_percent",
                    operator=">",
                    value=85,
                    window_minutes=10
                )
            ],
            severity=AlertSeverity.WARNING,
            channels=[AlertChannel.EMAIL],
            suppress_duration_minutes=60
        )
        self.alerting_service.add_alert_rule(memory_rule)

        logger.info("Default alert rules configured")

    async def _setup_notification_channels(self):
        """Set up notification channels."""
        if not self.alerting_service:
            return

        # Email channel
        email_config = self.config.get('notifications', {}).get('email', {})
        if email_config.get('enabled'):
            email_channel = NotificationChannel(
                channel_type=AlertChannel.EMAIL,
                name="primary_email",
                enabled=True,
                config={
                    "smtp_server": email_config.get('smtp_server'),
                    "smtp_port": email_config.get('smtp_port', 587),
                    "username": email_config.get('username'),
                    "password": email_config.get('password'),
                    "from_email": email_config.get('from_email'),
                    "recipients": email_config.get('recipients', [])
                },
                severity_filter=[AlertSeverity.HIGH, AlertSeverity.CRITICAL]
            )
            self.alerting_service.add_notification_channel(email_channel)

        # Slack channel
        slack_config = self.config.get('notifications', {}).get('slack', {})
        if slack_config.get('enabled'):
            slack_channel = NotificationChannel(
                channel_type=AlertChannel.SLACK,
                name="primary_slack",
                enabled=True,
                config={
                    "webhook_url": slack_config.get('webhook_url'),
                    "channel": slack_config.get('channel'),
                    "recipients": slack_config.get('recipients', [])
                }
            )
            self.alerting_service.add_notification_channel(slack_channel)

        # Webhook channel
        webhook_config = self.config.get('notifications', {}).get('webhook', {})
        if webhook_config.get('enabled'):
            webhook_channel = NotificationChannel(
                channel_type=AlertChannel.WEBHOOK,
                name="primary_webhook",
                enabled=True,
                config={
                    "webhook_url": webhook_config.get('url'),
                    "auth_header": webhook_config.get('auth_header'),
                    "recipients": webhook_config.get('recipients', [])
                }
            )
            self.alerting_service.add_notification_channel(webhook_channel)

        logger.info("Notification channels configured")

    def _start_metrics_server(self):
        """Start Prometheus metrics HTTP server."""
        metrics_port = self.config.get('metrics', {}).get('port', 8000)
        try:
            start_http_server(metrics_port, registry=self.registry)
            logger.info(f"Prometheus metrics server started on port {metrics_port}")
        except Exception as e:
            logger.error(f"Failed to start metrics server: {e}")

    async def _start_background_tasks(self):
        """Start background monitoring tasks."""
        # System metrics collection
        self._monitoring_tasks.append(
            asyncio.create_task(self._collect_system_metrics())
        )

        # Dashboard broadcast task
        if self.dashboard:
            self._monitoring_tasks.append(
                asyncio.create_task(self.dashboard.start_background_tasks())
            )

        # Health checks
        if self.health_manager:
            self._monitoring_tasks.append(
                asyncio.create_task(self.health_manager.start_monitoring())
            )

        # Performance monitoring
        if self.performance_monitor:
            self._monitoring_tasks.append(
                asyncio.create_task(self.performance_monitor.start_monitoring())
            )

        logger.info("Background monitoring tasks started")

    async def _collect_system_metrics(self):
        """Background task to collect system metrics."""
        import psutil

        while self._running:
            try:
                # CPU usage
                cpu_percent = psutil.cpu_percent(interval=1)
                self.cpu_usage_percent.set(cpu_percent)

                # Memory usage
                memory = psutil.virtual_memory()
                self.memory_usage_bytes.set(memory.used)

                # Record metrics for alerting
                if self.alerting_service:
                    await self.alerting_service.record_metric(
                        "cpu_usage_percent", cpu_percent
                    )
                    await self.alerting_service.record_metric(
                        "memory_usage_percent", memory.percent
                    )

                await asyncio.sleep(30)  # Collect every 30 seconds

            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Error collecting system metrics: {e}")
                await asyncio.sleep(60)

    # Monitoring interface methods

    async def record_http_request(self, method: str, endpoint: str,
                                status_code: int, duration: float):
        """Record HTTP request metrics."""
        # Prometheus metrics
        self.http_requests_total.labels(
            method=method,
            endpoint=endpoint,
            status=str(status_code)
        ).inc()

        self.http_request_duration.labels(
            method=method,
            endpoint=endpoint
        ).observe(duration)

        # Calculate error rate for alerting
        if self.alerting_service and status_code >= 500:
            await self.alerting_service.record_metric(
                "http_error_rate", 1 if status_code >= 500 else 0
            )

    async def record_anomaly_detection(self, algorithm: str,
                                     duration: float, success: bool):
        """Record anomaly detection metrics."""
        status = "success" if success else "failure"

        self.anomaly_detection_total.labels(
            algorithm=algorithm,
            status=status
        ).inc()

        if success:
            self.anomaly_detection_duration.labels(
                algorithm=algorithm
            ).observe(duration)

        # Record for alerting
        if self.alerting_service and not success:
            await self.alerting_service.record_metric(
                "anomaly_detection_failures", 1
            )

    async def record_model_training(self, algorithm: str,
                                  success: bool, accuracy: float = None):
        """Record model training metrics."""
        status = "success" if success else "failure"

        self.model_training_total.labels(
            algorithm=algorithm,
            status=status
        ).inc()

        if success and accuracy is not None:
            self.model_accuracy.labels(algorithm=algorithm).set(accuracy)

            # Record for alerting
            if self.alerting_service:
                await self.alerting_service.record_metric(
                    "model_accuracy", accuracy
                )

    async def record_alert(self, severity: str, rule_name: str):
        """Record alert metrics."""
        self.alerts_total.labels(severity=severity, rule=rule_name).inc()

        # Update active alerts gauge
        if self.alerting_service:
            stats = self.alerting_service.get_alert_statistics()
            for sev, count in stats.get("alerts_by_severity", {}).items():
                self.active_alerts.labels(severity=sev).set(count)

    async def get_monitoring_status(self) -> dict[str, Any]:
        """Get overall monitoring system status."""
        status = {
            "timestamp": datetime.now().isoformat(),
            "status": "healthy",
            "components": {}
        }

        # Alerting service status
        if self.alerting_service:
            alert_stats = self.alerting_service.get_alert_statistics()
            status["components"]["alerting"] = {
                "status": "healthy",
                "stats": alert_stats
            }

        # Dashboard status
        if self.dashboard:
            status["components"]["dashboard"] = {
                "status": "healthy",
                "websocket_connections": len(self.dashboard.websocket_connections)
            }

        # Health checks status
        if self.health_manager:
            health_status = await self.health_manager.get_overall_status()
            status["components"]["health_checks"] = health_status

        # Performance monitor status
        if self.performance_monitor:
            perf_status = await self.performance_monitor.get_status()
            status["components"]["performance"] = perf_status

        # Determine overall status
        component_statuses = [
            comp.get("status", "unknown")
            for comp in status["components"].values()
        ]

        if "unhealthy" in component_statuses:
            status["status"] = "unhealthy"
        elif "degraded" in component_statuses:
            status["status"] = "degraded"

        return status

    async def trigger_test_alert(self, severity: AlertSeverity = AlertSeverity.LOW):
        """Trigger a test alert for testing purposes."""
        if not self.alerting_service:
            return False

        test_rule = AlertRule(
            name="Test Alert",
            description="This is a test alert to verify notification channels",
            conditions=[
                ThresholdCondition(
                    metric_name="test_metric",
                    operator=">",
                    value=0,
                    window_minutes=1
                )
            ],
            severity=severity,
            channels=[AlertChannel.EMAIL, AlertChannel.SLACK],
            suppress_duration_minutes=1
        )

        self.alerting_service.add_alert_rule(test_rule)
        await self.alerting_service.record_metric("test_metric", 1)

        # Clean up test rule after a short delay
        async def cleanup():
            await asyncio.sleep(60)
            self.alerting_service.remove_alert_rule(test_rule.rule_id)

        asyncio.create_task(cleanup())
        return True

    async def shutdown(self):
        """Shutdown monitoring integration."""
        logger.info("Shutting down production monitoring integration...")

        self._running = False

        # Cancel background tasks
        for task in self._monitoring_tasks:
            if not task.done():
                task.cancel()
                try:
                    await task
                except asyncio.CancelledError:
                    pass

        # Shutdown components
        if self.alerting_service:
            await self.alerting_service.shutdown()

        if self.health_manager:
            await self.health_manager.shutdown()

        if self.performance_monitor:
            await self.performance_monitor.shutdown()

        if self.metrics_service:
            await self.metrics_service.shutdown()

        logger.info("Production monitoring integration shutdown complete")

    async def run_dashboard(self):
        """Run the real-time dashboard server."""
        if self.dashboard:
            await self.dashboard.run()


def create_production_monitoring(config_file: str = None) -> ProductionMonitoringIntegration:
    """Create production monitoring integration from config file."""
    config = {}

    if config_file and Path(config_file).exists():
        import yaml
        with open(config_file) as f:
            config = yaml.safe_load(f)
    else:
        # Default configuration
        config = {
            "dashboard": {
                "host": "0.0.0.0",
                "port": 8080
            },
            "metrics": {
                "port": 8000
            },
            "notifications": {
                "email": {
                    "enabled": os.getenv("EMAIL_NOTIFICATIONS_ENABLED", "false").lower() == "true",
                    "smtp_server": os.getenv("SMTP_SERVER"),
                    "smtp_port": int(os.getenv("SMTP_PORT", "587")),
                    "username": os.getenv("SMTP_USERNAME"),
                    "password": os.getenv("SMTP_PASSWORD"),
                    "from_email": os.getenv("ALERT_EMAIL_FROM"),
                    "recipients": os.getenv("ALERT_EMAIL_TO", "").split(",")
                },
                "slack": {
                    "enabled": os.getenv("SLACK_NOTIFICATIONS_ENABLED", "false").lower() == "true",
                    "webhook_url": os.getenv("SLACK_WEBHOOK_URL"),
                    "channel": os.getenv("SLACK_CHANNEL", "#alerts")
                },
                "webhook": {
                    "enabled": os.getenv("WEBHOOK_NOTIFICATIONS_ENABLED", "false").lower() == "true",
                    "url": os.getenv("WEBHOOK_URL"),
                    "auth_header": os.getenv("WEBHOOK_AUTH_HEADER")
                }
            }
        }

    return ProductionMonitoringIntegration(config)
