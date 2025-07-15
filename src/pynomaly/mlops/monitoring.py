#!/usr/bin/env python3
"""
Comprehensive Monitoring and Observability Stack for Pynomaly.
Provides metrics collection, alerting, and observability for ML operations.
"""

import asyncio
import json
import logging
import uuid
from collections import defaultdict, deque
from collections.abc import Callable
from dataclasses import asdict, dataclass
from datetime import datetime, timedelta
from enum import Enum
from pathlib import Path
from typing import Any

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

logger = logging.getLogger(__name__)


class MetricType(Enum):
    """Metric type enumeration."""

    COUNTER = "counter"
    GAUGE = "gauge"
    HISTOGRAM = "histogram"
    SUMMARY = "summary"


class AlertSeverity(Enum):
    """Alert severity levels."""

    INFO = "info"
    WARNING = "warning"
    ERROR = "error"
    CRITICAL = "critical"


class AlertStatus(Enum):
    """Alert status."""

    ACTIVE = "active"
    RESOLVED = "resolved"
    ACKNOWLEDGED = "acknowledged"
    SILENCED = "silenced"


@dataclass
class Metric:
    """Metric data point."""

    name: str
    value: float | int
    metric_type: MetricType
    tags: dict[str, str]
    timestamp: datetime
    unit: str | None = None
    description: str | None = None


@dataclass
class Alert:
    """Alert definition."""

    alert_id: str
    name: str
    description: str
    severity: AlertSeverity
    status: AlertStatus
    created_at: datetime
    updated_at: datetime
    resolved_at: datetime | None
    condition: str
    threshold: float
    metric_name: str
    current_value: float | None
    metadata: dict[str, Any]
    notification_channels: list[str]


@dataclass
class Dashboard:
    """Dashboard configuration."""

    dashboard_id: str
    name: str
    description: str
    created_at: datetime
    updated_at: datetime
    widgets: list[dict[str, Any]]
    layout: dict[str, Any]
    author: str
    public: bool
    tags: list[str]


class MetricsCollector:
    """Collects and stores metrics from various sources."""

    def __init__(self, max_datapoints: int = 10000):
        self.max_datapoints = max_datapoints
        self.metrics_storage: dict[str, deque] = defaultdict(
            lambda: deque(maxlen=max_datapoints)
        )
        self.metric_metadata: dict[str, dict[str, Any]] = {}
        self.collection_intervals: dict[str, int] = {}
        self.custom_collectors: dict[str, Callable] = {}

    def record_metric(
        self,
        name: str,
        value: float | int,
        metric_type: MetricType = MetricType.GAUGE,
        tags: dict[str, str] = None,
        unit: str = None,
        description: str = None,
    ):
        """Record a metric value."""
        metric = Metric(
            name=name,
            value=value,
            metric_type=metric_type,
            tags=tags or {},
            timestamp=datetime.now(),
            unit=unit,
            description=description,
        )

        # Store metric
        metric_key = self._get_metric_key(name, tags or {})
        self.metrics_storage[metric_key].append(metric)

        # Store metadata
        if name not in self.metric_metadata:
            self.metric_metadata[name] = {
                "type": metric_type.value,
                "unit": unit,
                "description": description,
                "tags_schema": list((tags or {}).keys()),
            }

        logger.debug(f"Recorded metric: {name}={value} (tags: {tags})")

    def increment_counter(
        self, name: str, tags: dict[str, str] = None, increment: float = 1
    ):
        """Increment a counter metric."""
        # Get current value
        metric_key = self._get_metric_key(name, tags or {})
        current_metrics = list(self.metrics_storage[metric_key])

        if current_metrics:
            last_value = current_metrics[-1].value
            new_value = last_value + increment
        else:
            new_value = increment

        self.record_metric(name, new_value, MetricType.COUNTER, tags)

    def set_gauge(self, name: str, value: float | int, tags: dict[str, str] = None):
        """Set a gauge metric value."""
        self.record_metric(name, value, MetricType.GAUGE, tags)

    def record_histogram(
        self, name: str, value: float | int, tags: dict[str, str] = None
    ):
        """Record a histogram metric value."""
        self.record_metric(name, value, MetricType.HISTOGRAM, tags)

    def get_metrics(
        self,
        name: str = None,
        tags: dict[str, str] = None,
        start_time: datetime = None,
        end_time: datetime = None,
    ) -> list[Metric]:
        """Get metrics with optional filtering."""
        metrics = []

        for metric_key, metric_deque in self.metrics_storage.items():
            for metric in metric_deque:
                # Filter by name
                if name and metric.name != name:
                    continue

                # Filter by tags
                if tags:
                    if not all(metric.tags.get(k) == v for k, v in tags.items()):
                        continue

                # Filter by time range
                if start_time and metric.timestamp < start_time:
                    continue
                if end_time and metric.timestamp > end_time:
                    continue

                metrics.append(metric)

        # Sort by timestamp
        metrics.sort(key=lambda m: m.timestamp)
        return metrics

    def get_metric_names(self) -> list[str]:
        """Get list of all metric names."""
        return list(self.metric_metadata.keys())

    def get_metric_metadata(self, name: str) -> dict[str, Any]:
        """Get metadata for a metric."""
        return self.metric_metadata.get(name, {})

    def register_collector(
        self, name: str, collector_func: Callable, interval_seconds: int = 60
    ):
        """Register a custom metric collector."""
        self.custom_collectors[name] = collector_func
        self.collection_intervals[name] = interval_seconds
        logger.info(f"Registered collector: {name} (interval: {interval_seconds}s)")

    def start_collection(self):
        """Start automatic metric collection."""

        async def collect_metrics():
            while True:
                for name, collector in self.custom_collectors.items():
                    try:
                        interval = self.collection_intervals.get(name, 60)
                        await asyncio.sleep(interval)

                        # Run collector
                        result = collector()
                        if isinstance(result, dict):
                            for metric_name, value in result.items():
                                self.record_metric(f"{name}.{metric_name}", value)
                        elif isinstance(result, (int, float)):
                            self.record_metric(name, result)

                    except Exception as e:
                        logger.error(f"Collector {name} failed: {e}")

        # Start background task
        asyncio.create_task(collect_metrics())

    def _get_metric_key(self, name: str, tags: dict[str, str]) -> str:
        """Generate a unique key for metric storage."""
        tag_str = "&".join(f"{k}={v}" for k, v in sorted(tags.items()))
        return f"{name}#{tag_str}" if tag_str else name


class AlertManager:
    """Manages alerting rules and notifications."""

    def __init__(self, metrics_collector: MetricsCollector):
        self.metrics_collector = metrics_collector
        self.alert_rules: dict[str, dict[str, Any]] = {}
        self.active_alerts: dict[str, Alert] = {}
        self.alert_history: list[Alert] = []
        self.notification_channels: dict[str, Callable] = {}

    def add_alert_rule(
        self,
        name: str,
        metric_name: str,
        condition: str,
        threshold: float,
        severity: AlertSeverity = AlertSeverity.WARNING,
        description: str = "",
        evaluation_interval: int = 60,
        notification_channels: list[str] = None,
    ):
        """Add an alerting rule."""
        rule_id = str(uuid.uuid4())

        self.alert_rules[rule_id] = {
            "name": name,
            "metric_name": metric_name,
            "condition": condition,
            "threshold": threshold,
            "severity": severity,
            "description": description,
            "evaluation_interval": evaluation_interval,
            "notification_channels": notification_channels or [],
            "last_evaluation": None,
            "created_at": datetime.now(),
        }

        logger.info(f"Added alert rule: {name} (threshold: {threshold})")
        return rule_id

    def register_notification_channel(self, name: str, handler: Callable):
        """Register a notification channel."""
        self.notification_channels[name] = handler
        logger.info(f"Registered notification channel: {name}")

    async def evaluate_alerts(self):
        """Evaluate all alert rules."""
        for rule_id, rule in self.alert_rules.items():
            try:
                await self._evaluate_rule(rule_id, rule)
            except Exception as e:
                logger.error(f"Failed to evaluate rule {rule['name']}: {e}")

    async def _evaluate_rule(self, rule_id: str, rule: dict[str, Any]):
        """Evaluate a single alert rule."""
        # Get recent metrics
        now = datetime.now()
        start_time = now - timedelta(minutes=5)  # Look at last 5 minutes

        metrics = self.metrics_collector.get_metrics(
            name=rule["metric_name"], start_time=start_time, end_time=now
        )

        if not metrics:
            return

        # Get current value (latest metric)
        current_value = metrics[-1].value

        # Evaluate condition
        alert_triggered = self._evaluate_condition(
            current_value, rule["condition"], rule["threshold"]
        )

        # Check if alert exists
        existing_alert = None
        for alert in self.active_alerts.values():
            if alert.name == rule["name"]:
                existing_alert = alert
                break

        if alert_triggered and not existing_alert:
            # Create new alert
            alert = Alert(
                alert_id=str(uuid.uuid4()),
                name=rule["name"],
                description=rule["description"],
                severity=rule["severity"],
                status=AlertStatus.ACTIVE,
                created_at=now,
                updated_at=now,
                resolved_at=None,
                condition=rule["condition"],
                threshold=rule["threshold"],
                metric_name=rule["metric_name"],
                current_value=current_value,
                metadata={"rule_id": rule_id},
                notification_channels=rule["notification_channels"],
            )

            self.active_alerts[alert.alert_id] = alert
            self.alert_history.append(alert)

            # Send notifications
            await self._send_notifications(alert)

            logger.warning(f"Alert triggered: {alert.name} (value: {current_value})")

        elif not alert_triggered and existing_alert:
            # Resolve alert
            existing_alert.status = AlertStatus.RESOLVED
            existing_alert.resolved_at = now
            existing_alert.updated_at = now

            # Remove from active alerts
            if existing_alert.alert_id in self.active_alerts:
                del self.active_alerts[existing_alert.alert_id]

            # Send resolution notification
            await self._send_notifications(existing_alert, resolved=True)

            logger.info(f"Alert resolved: {existing_alert.name}")

        # Update last evaluation time
        rule["last_evaluation"] = now

    def _evaluate_condition(
        self, value: float, condition: str, threshold: float
    ) -> bool:
        """Evaluate alert condition."""
        if condition == ">":
            return value > threshold
        elif condition == ">=":
            return value >= threshold
        elif condition == "<":
            return value < threshold
        elif condition == "<=":
            return value <= threshold
        elif condition == "==":
            return value == threshold
        elif condition == "!=":
            return value != threshold
        else:
            logger.error(f"Unknown condition: {condition}")
            return False

    async def _send_notifications(self, alert: Alert, resolved: bool = False):
        """Send alert notifications."""
        for channel_name in alert.notification_channels:
            if channel_name in self.notification_channels:
                try:
                    handler = self.notification_channels[channel_name]
                    await handler(alert, resolved)
                except Exception as e:
                    logger.error(f"Failed to send notification via {channel_name}: {e}")

    def get_active_alerts(self) -> list[Alert]:
        """Get all active alerts."""
        return list(self.active_alerts.values())

    def get_alert_history(self, limit: int = 100) -> list[Alert]:
        """Get alert history."""
        return sorted(self.alert_history, key=lambda a: a.created_at, reverse=True)[
            :limit
        ]

    def acknowledge_alert(self, alert_id: str) -> bool:
        """Acknowledge an alert."""
        if alert_id in self.active_alerts:
            self.active_alerts[alert_id].status = AlertStatus.ACKNOWLEDGED
            self.active_alerts[alert_id].updated_at = datetime.now()
            logger.info(f"Alert acknowledged: {alert_id}")
            return True
        return False

    def silence_alert(self, alert_id: str, duration_minutes: int = 60) -> bool:
        """Silence an alert for a specified duration."""
        if alert_id in self.active_alerts:
            alert = self.active_alerts[alert_id]
            alert.status = AlertStatus.SILENCED
            alert.updated_at = datetime.now()
            alert.metadata["silenced_until"] = (
                datetime.now() + timedelta(minutes=duration_minutes)
            ).isoformat()
            logger.info(f"Alert silenced: {alert_id} for {duration_minutes} minutes")
            return True
        return False


class DashboardManager:
    """Manages monitoring dashboards."""

    def __init__(self, metrics_collector: MetricsCollector):
        self.metrics_collector = metrics_collector
        self.dashboards: dict[str, Dashboard] = {}
        self.dashboard_path = Path("mlops/monitoring/dashboards")
        self.dashboard_path.mkdir(parents=True, exist_ok=True)

    def create_dashboard(
        self,
        name: str,
        description: str = "",
        author: str = "system",
        public: bool = True,
        tags: list[str] = None,
    ) -> str:
        """Create a new dashboard."""
        dashboard_id = str(uuid.uuid4())

        dashboard = Dashboard(
            dashboard_id=dashboard_id,
            name=name,
            description=description,
            created_at=datetime.now(),
            updated_at=datetime.now(),
            widgets=[],
            layout={"columns": 12, "rows": []},
            author=author,
            public=public,
            tags=tags or [],
        )

        self.dashboards[dashboard_id] = dashboard
        self._save_dashboard(dashboard)

        logger.info(f"Created dashboard: {name} ({dashboard_id})")
        return dashboard_id

    def add_widget(
        self,
        dashboard_id: str,
        widget_type: str,
        title: str,
        config: dict[str, Any],
        position: dict[str, int] = None,
    ) -> bool:
        """Add a widget to a dashboard."""
        if dashboard_id not in self.dashboards:
            return False

        widget = {
            "widget_id": str(uuid.uuid4()),
            "type": widget_type,
            "title": title,
            "config": config,
            "position": position or {"x": 0, "y": 0, "w": 6, "h": 4},
            "created_at": datetime.now().isoformat(),
        }

        dashboard = self.dashboards[dashboard_id]
        dashboard.widgets.append(widget)
        dashboard.updated_at = datetime.now()

        self._save_dashboard(dashboard)
        logger.info(f"Added widget to dashboard {dashboard_id}: {title}")
        return True

    def get_dashboard_data(self, dashboard_id: str) -> dict[str, Any]:
        """Get dashboard data including widget data."""
        if dashboard_id not in self.dashboards:
            return {}

        dashboard = self.dashboards[dashboard_id]
        dashboard_data = asdict(dashboard)

        # Populate widget data
        for widget in dashboard_data["widgets"]:
            widget["data"] = self._get_widget_data(widget)

        return dashboard_data

    def _get_widget_data(self, widget: dict[str, Any]) -> dict[str, Any]:
        """Get data for a specific widget."""
        widget_type = widget["type"]
        config = widget["config"]

        if widget_type == "metric_chart":
            # Time series chart
            metric_name = config.get("metric_name")
            if metric_name:
                now = datetime.now()
                start_time = now - timedelta(hours=config.get("time_range_hours", 1))

                metrics = self.metrics_collector.get_metrics(
                    name=metric_name, start_time=start_time, end_time=now
                )

                return {
                    "series": [
                        {"x": m.timestamp.isoformat(), "y": m.value} for m in metrics
                    ]
                }

        elif widget_type == "stat":
            # Single stat widget
            metric_name = config.get("metric_name")
            if metric_name:
                recent_metrics = self.metrics_collector.get_metrics(
                    name=metric_name, start_time=datetime.now() - timedelta(minutes=5)
                )
                if recent_metrics:
                    return {"value": recent_metrics[-1].value}

        elif widget_type == "alert_list":
            # Alert list widget
            from .monitoring import alert_manager

            alerts = alert_manager.get_active_alerts()
            return {
                "alerts": [
                    {
                        "name": alert.name,
                        "severity": alert.severity.value,
                        "status": alert.status.value,
                        "created_at": alert.created_at.isoformat(),
                        "current_value": alert.current_value,
                    }
                    for alert in alerts[: config.get("max_alerts", 10)]
                ]
            }

        return {}

    def _save_dashboard(self, dashboard: Dashboard):
        """Save dashboard to file."""
        try:
            dashboard_file = self.dashboard_path / f"{dashboard.dashboard_id}.json"
            with open(dashboard_file, "w") as f:
                json.dump(asdict(dashboard), f, indent=2, default=str)
        except Exception as e:
            logger.error(f"Failed to save dashboard: {e}")

    def list_dashboards(self) -> list[Dashboard]:
        """List all dashboards."""
        return list(self.dashboards.values())


class MLOpsMonitor:
    """Comprehensive MLOps monitoring system."""

    def __init__(self):
        self.metrics_collector = MetricsCollector()
        self.alert_manager = AlertManager(self.metrics_collector)
        self.dashboard_manager = DashboardManager(self.metrics_collector)

        # Setup default collectors
        self._setup_default_collectors()
        self._setup_default_alerts()
        self._setup_default_dashboards()

    def _setup_default_collectors(self):
        """Setup default system and ML metrics collectors."""

        # System metrics collector
        def system_metrics():
            try:
                import psutil

                return {
                    "cpu_percent": psutil.cpu_percent(),
                    "memory_percent": psutil.virtual_memory().percent,
                    "disk_percent": psutil.disk_usage("/").percent,
                }
            except ImportError:
                return {"cpu_percent": 0, "memory_percent": 0, "disk_percent": 0}

        # Model performance collector
        def model_performance():
            try:
                from .model_deployment import deployment_manager

                deployments = deployment_manager.list_deployments()

                metrics = {}
                for deployment in deployments:
                    if deployment.status.value == "active":
                        health = deployment_manager.get_deployment_health(
                            deployment.deployment_id
                        )
                        if health:
                            prefix = f"model.{deployment.model_id}"
                            metrics[f"{prefix}.predictions_count"] = (
                                health.predictions_count
                            )
                            metrics[f"{prefix}.errors_count"] = health.errors_count
                            metrics[f"{prefix}.response_time_ms"] = (
                                health.response_time_ms
                            )
                            metrics[f"{prefix}.uptime_seconds"] = health.uptime_seconds

                return metrics
            except Exception as e:
                logger.warning(f"Model performance collection failed: {e}")
                return {}

        # Register collectors
        self.metrics_collector.register_collector("system", system_metrics, 30)
        self.metrics_collector.register_collector(
            "model_performance", model_performance, 60
        )

    def _setup_default_alerts(self):
        """Setup default alerting rules."""

        # High CPU usage
        self.alert_manager.add_alert_rule(
            name="High CPU Usage",
            metric_name="system.cpu_percent",
            condition=">",
            threshold=80.0,
            severity=AlertSeverity.WARNING,
            description="CPU usage is above 80%",
        )

        # High memory usage
        self.alert_manager.add_alert_rule(
            name="High Memory Usage",
            metric_name="system.memory_percent",
            condition=">",
            threshold=90.0,
            severity=AlertSeverity.ERROR,
            description="Memory usage is above 90%",
        )

        # Model error rate
        self.alert_manager.add_alert_rule(
            name="High Model Error Rate",
            metric_name="model_performance.errors_count",
            condition=">",
            threshold=10.0,
            severity=AlertSeverity.ERROR,
            description="Model error count is too high",
        )

    def _setup_default_dashboards(self):
        """Setup default monitoring dashboards."""

        # System overview dashboard
        system_dashboard_id = self.dashboard_manager.create_dashboard(
            name="System Overview",
            description="System resource monitoring",
            tags=["system", "overview"],
        )

        self.dashboard_manager.add_widget(
            system_dashboard_id,
            "metric_chart",
            "CPU Usage",
            {"metric_name": "system.cpu_percent", "time_range_hours": 1},
            {"x": 0, "y": 0, "w": 6, "h": 4},
        )

        self.dashboard_manager.add_widget(
            system_dashboard_id,
            "metric_chart",
            "Memory Usage",
            {"metric_name": "system.memory_percent", "time_range_hours": 1},
            {"x": 6, "y": 0, "w": 6, "h": 4},
        )

        # ML models dashboard
        ml_dashboard_id = self.dashboard_manager.create_dashboard(
            name="ML Models",
            description="Model performance monitoring",
            tags=["ml", "models"],
        )

        self.dashboard_manager.add_widget(
            ml_dashboard_id,
            "alert_list",
            "Active Alerts",
            {"max_alerts": 5},
            {"x": 0, "y": 0, "w": 12, "h": 3},
        )

        logger.info("Default monitoring setup completed")

    async def start_monitoring(self):
        """Start the monitoring system."""
        # Start metric collection
        self.metrics_collector.start_collection()

        # Start alert evaluation
        async def alert_loop():
            while True:
                try:
                    await self.alert_manager.evaluate_alerts()
                    await asyncio.sleep(60)  # Evaluate every minute
                except Exception as e:
                    logger.error(f"Alert evaluation error: {e}")
                    await asyncio.sleep(60)

        asyncio.create_task(alert_loop())
        logger.info("Monitoring system started")

    def record_model_prediction(
        self,
        model_id: str,
        prediction_time_ms: float,
        success: bool = True,
        error: str = None,
    ):
        """Record model prediction metrics."""
        tags = {"model_id": model_id, "status": "success" if success else "error"}

        # Record prediction count
        self.metrics_collector.increment_counter("model.predictions_total", tags)

        # Record prediction time
        if success:
            self.metrics_collector.record_histogram(
                "model.prediction_time_ms", prediction_time_ms, tags
            )
        else:
            self.metrics_collector.increment_counter("model.errors_total", tags)
            if error:
                self.metrics_collector.record_metric(
                    "model.error_type", 1, tags={**tags, "error_type": error}
                )

    def record_training_metrics(
        self,
        model_id: str,
        training_time_minutes: float,
        accuracy: float,
        loss: float,
        dataset_size: int,
    ):
        """Record model training metrics."""
        tags = {"model_id": model_id}

        self.metrics_collector.record_metric(
            "training.duration_minutes", training_time_minutes, tags=tags
        )
        self.metrics_collector.set_gauge("training.accuracy", accuracy, tags)
        self.metrics_collector.set_gauge("training.loss", loss, tags)
        self.metrics_collector.set_gauge("training.dataset_size", dataset_size, tags)

    def record_data_drift(self, model_id: str, drift_score: float, threshold: float):
        """Record data drift metrics."""
        tags = {"model_id": model_id}

        self.metrics_collector.set_gauge("data_drift.score", drift_score, tags)
        self.metrics_collector.set_gauge("data_drift.threshold", threshold, tags)

        # Record drift detection
        drift_detected = drift_score > threshold
        self.metrics_collector.set_gauge(
            "data_drift.detected", 1 if drift_detected else 0, tags
        )


# Global monitoring instance
mlops_monitor = MLOpsMonitor()

# FastAPI app for monitoring endpoints
monitoring_app = FastAPI(
    title="Pynomaly Monitoring API",
    description="Monitoring and observability endpoints",
    version="1.0.0",
)


class MetricResponse(BaseModel):
    name: str
    value: float
    metric_type: str
    tags: dict[str, str]
    timestamp: datetime
    unit: str | None


class AlertResponse(BaseModel):
    alert_id: str
    name: str
    severity: str
    status: str
    created_at: datetime
    current_value: float | None
    threshold: float


@monitoring_app.get("/metrics", response_model=list[MetricResponse])
async def get_metrics(
    name: str = None, start_time: datetime = None, end_time: datetime = None
):
    """Get metrics with optional filtering."""
    metrics = mlops_monitor.metrics_collector.get_metrics(
        name, start_time=start_time, end_time=end_time
    )
    return [
        MetricResponse(
            name=m.name,
            value=m.value,
            metric_type=m.metric_type.value,
            tags=m.tags,
            timestamp=m.timestamp,
            unit=m.unit,
        )
        for m in metrics
    ]


@monitoring_app.get("/alerts", response_model=list[AlertResponse])
async def get_alerts():
    """Get active alerts."""
    alerts = mlops_monitor.alert_manager.get_active_alerts()
    return [
        AlertResponse(
            alert_id=a.alert_id,
            name=a.name,
            severity=a.severity.value,
            status=a.status.value,
            created_at=a.created_at,
            current_value=a.current_value,
            threshold=a.threshold,
        )
        for a in alerts
    ]


@monitoring_app.post("/alerts/{alert_id}/acknowledge")
async def acknowledge_alert(alert_id: str):
    """Acknowledge an alert."""
    success = mlops_monitor.alert_manager.acknowledge_alert(alert_id)
    if success:
        return {"status": "acknowledged"}
    else:
        raise HTTPException(status_code=404, detail="Alert not found")


@monitoring_app.get("/dashboards")
async def get_dashboards():
    """Get all dashboards."""
    dashboards = mlops_monitor.dashboard_manager.list_dashboards()
    return [asdict(d) for d in dashboards]


@monitoring_app.get("/dashboards/{dashboard_id}")
async def get_dashboard(dashboard_id: str):
    """Get dashboard with data."""
    dashboard_data = mlops_monitor.dashboard_manager.get_dashboard_data(dashboard_id)
    if not dashboard_data:
        raise HTTPException(status_code=404, detail="Dashboard not found")
    return dashboard_data


@monitoring_app.get("/health")
async def monitoring_health():
    """Monitoring system health check."""
    return {
        "status": "healthy",
        "timestamp": datetime.now(),
        "metrics_count": len(mlops_monitor.metrics_collector.metrics_storage),
        "active_alerts": len(mlops_monitor.alert_manager.active_alerts),
        "dashboards_count": len(mlops_monitor.dashboard_manager.dashboards),
    }


# Export for use
__all__ = [
    "MLOpsMonitor",
    "MetricsCollector",
    "AlertManager",
    "DashboardManager",
    "MetricType",
    "AlertSeverity",
    "mlops_monitor",
    "monitoring_app",
]
