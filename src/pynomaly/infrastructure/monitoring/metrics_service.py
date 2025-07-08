"""Metrics collection and monitoring service for production observability."""

from __future__ import annotations

import asyncio
import logging
import time
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional, Union
from uuid import UUID, uuid4

import psutil
import requests
from prometheus_client import CollectorRegistry, Counter, Gauge, Histogram, Summary

from pynomaly.domain.models.monitoring import (
    Alert,
    AlertRule,
    AlertSeverity,
    AlertStatus,
    HealthCheck,
    Metric,
    MetricPoint,
    MetricType,
    ServiceStatus,
)


class MetricsService:
    """Comprehensive metrics collection and monitoring service."""

    def __init__(self, service_name: str = "pynomaly", service_version: str = "1.0.0"):
        self.service_name = service_name
        self.service_version = service_version
        self.logger = logging.getLogger(__name__)

        # Prometheus registry for external metrics export
        self.prometheus_registry = CollectorRegistry()

        # Internal metrics storage
        self.metrics: Dict[str, Metric] = {}
        self.alert_rules: Dict[UUID, AlertRule] = {}
        self.active_alerts: Dict[UUID, Alert] = {}
        self.health_checks: Dict[UUID, HealthCheck] = {}

        # Service status tracking
        self.service_status = ServiceStatus(
            service_name=service_name,
            service_version=service_version,
        )

        # Prometheus metrics
        self._initialize_prometheus_metrics()

        # Background tasks
        self.monitoring_tasks: set[asyncio.Task] = set()
        self.is_monitoring = False

        self.logger.info(f"Metrics service initialized for {service_name} v{service_version}")

    def _initialize_prometheus_metrics(self) -> None:
        """Initialize Prometheus metrics."""

        # Application metrics
        self.prom_request_count = Counter(
            'pynomaly_requests_total',
            'Total number of requests',
            ['method', 'endpoint', 'status'],
            registry=self.prometheus_registry
        )

        self.prom_request_duration = Histogram(
            'pynomaly_request_duration_seconds',
            'Request duration in seconds',
            ['method', 'endpoint'],
            registry=self.prometheus_registry
        )

        self.prom_error_count = Counter(
            'pynomaly_errors_total',
            'Total number of errors',
            ['error_type', 'component'],
            registry=self.prometheus_registry
        )

        # ML model metrics
        self.prom_model_predictions = Counter(
            'pynomaly_model_predictions_total',
            'Total number of model predictions',
            ['model_name', 'model_version'],
            registry=self.prometheus_registry
        )

        self.prom_model_accuracy = Gauge(
            'pynomaly_model_accuracy',
            'Current model accuracy',
            ['model_name', 'model_version'],
            registry=self.prometheus_registry
        )

        self.prom_anomaly_detection_rate = Gauge(
            'pynomaly_anomaly_detection_rate',
            'Rate of anomalies detected',
            ['detector_type'],
            registry=self.prometheus_registry
        )

        # System metrics
        self.prom_cpu_usage = Gauge(
            'pynomaly_cpu_usage_percent',
            'CPU usage percentage',
            registry=self.prometheus_registry
        )

        self.prom_memory_usage = Gauge(
            'pynomaly_memory_usage_bytes',
            'Memory usage in bytes',
            registry=self.prometheus_registry
        )

        self.prom_disk_usage = Gauge(
            'pynomaly_disk_usage_percent',
            'Disk usage percentage',
            registry=self.prometheus_registry
        )

    async def start_monitoring(self) -> None:
        """Start background monitoring tasks."""

        if self.is_monitoring:
            return

        self.is_monitoring = True

        # Start monitoring tasks
        tasks = [
            asyncio.create_task(self._system_metrics_collector()),
            asyncio.create_task(self._health_check_runner()),
            asyncio.create_task(self._alert_evaluator()),
            asyncio.create_task(self._metrics_cleanup()),
        ]

        self.monitoring_tasks.update(tasks)

        self.logger.info("Started monitoring tasks")

    async def stop_monitoring(self) -> None:
        """Stop background monitoring tasks."""

        self.is_monitoring = False

        for task in self.monitoring_tasks:
            task.cancel()

        await asyncio.gather(*self.monitoring_tasks, return_exceptions=True)
        self.monitoring_tasks.clear()

        self.logger.info("Stopped monitoring tasks")

    def create_metric(
        self,
        name: str,
        metric_type: MetricType,
        description: str,
        unit: str = "",
        labels: Optional[Dict[str, str]] = None,
    ) -> Metric:
        """Create a new metric."""

        if name in self.metrics:
            return self.metrics[name]

        metric = Metric(
            metric_id=uuid4(),
            name=name,
            metric_type=metric_type,
            description=description,
            unit=unit,
            labels=labels or {},
        )

        self.metrics[name] = metric

        self.logger.debug(f"Created metric: {name}")
        return metric

    def record_metric(
        self,
        name: str,
        value: Union[float, int, str],
        labels: Optional[Dict[str, str]] = None,
    ) -> None:
        """Record a metric value."""

        if name not in self.metrics:
            # Auto-create metric if it doesn't exist
            metric_type = MetricType.GAUGE if isinstance(value, (int, float)) else MetricType.COUNTER
            self.create_metric(name, metric_type, f"Auto-created metric: {name}")

        metric = self.metrics[name]
        metric.add_data_point(value, labels)

        # Update Prometheus metrics if applicable
        self._update_prometheus_metric(name, value, labels)

    def record_request_metrics(
        self,
        method: str,
        endpoint: str,
        status_code: int,
        duration: float,
    ) -> None:
        """Record HTTP request metrics."""

        # Prometheus metrics
        self.prom_request_count.labels(
            method=method,
            endpoint=endpoint,
            status=str(status_code)
        ).inc()

        self.prom_request_duration.labels(
            method=method,
            endpoint=endpoint
        ).observe(duration)

        # Internal metrics
        self.record_metric("http_requests_total", 1, {
            "method": method,
            "endpoint": endpoint,
            "status": str(status_code),
        })

        self.record_metric("http_request_duration_seconds", duration, {
            "method": method,
            "endpoint": endpoint,
        })

        if status_code >= 400:
            self.prom_error_count.labels(
                error_type="http_error",
                component="api"
            ).inc()

            self.record_metric("http_errors_total", 1, {
                "status": str(status_code),
                "endpoint": endpoint,
            })

    def record_model_metrics(
        self,
        model_name: str,
        model_version: str,
        prediction_count: int = 1,
        accuracy: Optional[float] = None,
        inference_time: Optional[float] = None,
    ) -> None:
        """Record ML model metrics."""

        # Prometheus metrics
        self.prom_model_predictions.labels(
            model_name=model_name,
            model_version=model_version
        ).inc(prediction_count)

        if accuracy is not None:
            self.prom_model_accuracy.labels(
                model_name=model_name,
                model_version=model_version
            ).set(accuracy)

        # Internal metrics
        self.record_metric("model_predictions_total", prediction_count, {
            "model_name": model_name,
            "model_version": model_version,
        })

        if accuracy is not None:
            self.record_metric("model_accuracy", accuracy, {
                "model_name": model_name,
                "model_version": model_version,
            })

        if inference_time is not None:
            self.record_metric("model_inference_duration_seconds", inference_time, {
                "model_name": model_name,
                "model_version": model_version,
            })

    def record_anomaly_detection_metrics(
        self,
        detector_type: str,
        anomalies_detected: int,
        total_samples: int,
        detection_accuracy: Optional[float] = None,
    ) -> None:
        """Record anomaly detection specific metrics."""

        detection_rate = anomalies_detected / max(total_samples, 1) * 100

        # Prometheus metrics
        self.prom_anomaly_detection_rate.labels(
            detector_type=detector_type
        ).set(detection_rate)

        # Internal metrics
        self.record_metric("anomalies_detected_total", anomalies_detected, {
            "detector_type": detector_type,
        })

        self.record_metric("anomaly_detection_rate", detection_rate, {
            "detector_type": detector_type,
        })

        if detection_accuracy is not None:
            self.record_metric("anomaly_detection_accuracy", detection_accuracy, {
                "detector_type": detector_type,
            })

    def create_alert_rule(
        self,
        name: str,
        metric_name: str,
        condition: str,
        threshold: Union[float, int, str],
        severity: AlertSeverity = AlertSeverity.WARNING,
        evaluation_window: timedelta = timedelta(minutes=5),
    ) -> AlertRule:
        """Create a new alert rule."""

        rule = AlertRule(
            rule_id=uuid4(),
            name=name,
            description=f"Alert when {metric_name} {condition} {threshold}",
            metric_name=metric_name,
            condition=condition,
            threshold=threshold,
            severity=severity,
            evaluation_window=evaluation_window,
        )

        self.alert_rules[rule.rule_id] = rule

        self.logger.info(f"Created alert rule: {name}")
        return rule

    def add_health_check(
        self,
        name: str,
        check_type: str,
        target: str,
        timeout: int = 30,
        interval: int = 60,
    ) -> HealthCheck:
        """Add a health check."""

        health_check = HealthCheck(
            check_id=uuid4(),
            name=name,
            description=f"{check_type} health check for {target}",
            check_type=check_type,
            target=target,
            timeout=timeout,
            interval=interval,
        )

        self.health_checks[health_check.check_id] = health_check
        self.service_status.add_health_check(health_check)

        self.logger.info(f"Added health check: {name}")
        return health_check

    async def get_metric_value(
        self,
        name: str,
        aggregation: str = "latest",
        time_range: Optional[timedelta] = None,
    ) -> Optional[Union[float, int, str]]:
        """Get metric value with optional aggregation."""

        if name not in self.metrics:
            return None

        metric = self.metrics[name]

        if aggregation == "latest":
            return metric.get_latest_value()

        elif aggregation in ["avg", "average"]:
            start_time = None
            if time_range:
                start_time = datetime.utcnow() - time_range
            return metric.calculate_average(start_time=start_time)

        elif aggregation == "sum" and time_range:
            start_time = datetime.utcnow() - time_range
            points = metric.get_values_in_range(start_time, datetime.utcnow())
            numeric_values = [p.value for p in points if isinstance(p.value, (int, float))]
            return sum(numeric_values) if numeric_values else None

        return metric.get_latest_value()

    async def get_service_health(self) -> Dict[str, Any]:
        """Get comprehensive service health status."""

        # Update service status
        await self._update_service_status()

        return self.service_status.get_status_summary()

    async def get_metrics_summary(
        self,
        time_range: timedelta = timedelta(hours=1)
    ) -> Dict[str, Any]:
        """Get metrics summary for dashboard."""

        start_time = datetime.utcnow() - time_range

        summary = {
            "timestamp": datetime.utcnow().isoformat(),
            "time_range_hours": time_range.total_seconds() / 3600,
            "metrics": {},
            "alerts": {
                "active": len([a for a in self.active_alerts.values() if a.status == AlertStatus.ACTIVE]),
                "critical": len([a for a in self.active_alerts.values()
                               if a.status == AlertStatus.ACTIVE and a.severity == AlertSeverity.CRITICAL]),
                "warning": len([a for a in self.active_alerts.values()
                              if a.status == AlertStatus.ACTIVE and a.severity == AlertSeverity.WARNING]),
            },
            "health_checks": {
                "total": len(self.health_checks),
                "healthy": sum(1 for hc in self.health_checks.values() if hc.is_healthy),
                "unhealthy": sum(1 for hc in self.health_checks.values() if not hc.is_healthy),
            },
        }

        # Include key metrics
        key_metrics = [
            "http_requests_total",
            "http_request_duration_seconds",
            "model_predictions_total",
            "anomalies_detected_total",
            "cpu_usage_percent",
            "memory_usage_percent",
        ]

        for metric_name in key_metrics:
            if metric_name in self.metrics:
                metric = self.metrics[metric_name]

                latest_value = metric.get_latest_value()
                avg_value = metric.calculate_average(start_time=start_time)

                summary["metrics"][metric_name] = {
                    "latest": latest_value,
                    "average": avg_value,
                    "unit": metric.unit,
                    "description": metric.description,
                }

        return summary

    async def _system_metrics_collector(self) -> None:
        """Background task to collect system metrics."""

        while self.is_monitoring:
            try:
                # CPU usage
                cpu_percent = psutil.cpu_percent(interval=1)
                self.prom_cpu_usage.set(cpu_percent)
                self.record_metric("cpu_usage_percent", cpu_percent)
                self.service_status.cpu_usage = cpu_percent

                # Memory usage
                memory = psutil.virtual_memory()
                self.prom_memory_usage.set(memory.used)
                self.record_metric("memory_usage_percent", memory.percent)
                self.record_metric("memory_usage_bytes", memory.used)
                self.service_status.memory_usage = memory.percent

                # Disk usage
                disk = psutil.disk_usage('/')
                disk_percent = (disk.used / disk.total) * 100
                self.prom_disk_usage.set(disk_percent)
                self.record_metric("disk_usage_percent", disk_percent)
                self.service_status.disk_usage = disk_percent

                # Network I/O
                network = psutil.net_io_counters()
                self.record_metric("network_bytes_sent", network.bytes_sent)
                self.record_metric("network_bytes_recv", network.bytes_recv)

                # Process metrics
                process = psutil.Process()
                self.record_metric("process_cpu_percent", process.cpu_percent())
                self.record_metric("process_memory_rss", process.memory_info().rss)
                self.record_metric("process_num_threads", process.num_threads())

            except Exception as e:
                self.logger.error(f"Error collecting system metrics: {e}")

            await asyncio.sleep(30)  # Collect every 30 seconds

    async def _health_check_runner(self) -> None:
        """Background task to run health checks."""

        while self.is_monitoring:
            try:
                for health_check in self.health_checks.values():
                    if not health_check.enabled:
                        continue

                    # Check if it's time to run this health check
                    if (health_check.last_check_at and
                        datetime.utcnow() - health_check.last_check_at < timedelta(seconds=health_check.interval)):
                        continue

                    await self._run_health_check(health_check)

            except Exception as e:
                self.logger.error(f"Error running health checks: {e}")

            await asyncio.sleep(10)  # Check every 10 seconds

    async def _run_health_check(self, health_check: HealthCheck) -> None:
        """Run a single health check."""

        start_time = time.time()

        try:
            if health_check.check_type == "http":
                response = requests.get(
                    health_check.target,
                    timeout=health_check.timeout
                )
                response.raise_for_status()

                response_time = time.time() - start_time
                health_check.record_success(response_time)

                health_check.last_result = {
                    "status_code": response.status_code,
                    "response_time": response_time,
                    "response_size": len(response.content),
                }

            elif health_check.check_type == "tcp":
                import socket

                host, port = health_check.target.split(":")
                sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
                sock.settimeout(health_check.timeout)

                result = sock.connect_ex((host, int(port)))
                sock.close()

                if result == 0:
                    response_time = time.time() - start_time
                    health_check.record_success(response_time)
                    health_check.last_result = {"tcp_connect": "success"}
                else:
                    health_check.record_failure(f"TCP connection failed to {health_check.target}")

            elif health_check.check_type == "database":
                # Database health check would be implemented here
                # For now, simulate success
                response_time = time.time() - start_time
                health_check.record_success(response_time)
                health_check.last_result = {"database": "connected"}

        except Exception as e:
            health_check.record_failure(str(e))
            self.logger.warning(f"Health check failed for {health_check.name}: {e}")

    async def _alert_evaluator(self) -> None:
        """Background task to evaluate alert rules."""

        while self.is_monitoring:
            try:
                for rule in self.alert_rules.values():
                    if not rule.is_enabled:
                        continue

                    await self._evaluate_alert_rule(rule)

            except Exception as e:
                self.logger.error(f"Error evaluating alerts: {e}")

            await asyncio.sleep(30)  # Evaluate every 30 seconds

    async def _evaluate_alert_rule(self, rule: AlertRule) -> None:
        """Evaluate a single alert rule."""

        # Get metric value
        metric_value = await self.get_metric_value(
            rule.metric_name,
            aggregation=rule.aggregation_function,
            time_range=rule.evaluation_window,
        )

        if metric_value is None:
            return

        # Check if alert condition is met
        should_alert = rule.evaluate(metric_value)

        # Find existing alert for this rule
        existing_alert = None
        for alert in self.active_alerts.values():
            if alert.rule_id == rule.rule_id and alert.status == AlertStatus.ACTIVE:
                existing_alert = alert
                break

        if should_alert:
            if not existing_alert:
                # Create new alert
                alert = Alert(
                    alert_id=uuid4(),
                    rule_id=rule.rule_id,
                    rule_name=rule.name,
                    metric_name=rule.metric_name,
                    metric_value=metric_value,
                    threshold=rule.threshold,
                    severity=rule.severity,
                    message=rule.generate_alert_message(metric_value),
                )

                self.active_alerts[alert.alert_id] = alert

                # Update service status
                self.service_status.active_alerts += 1
                if alert.severity == AlertSeverity.CRITICAL:
                    self.service_status.critical_alerts += 1

                self.logger.warning(f"Alert triggered: {alert.message}")

                # Send notifications (would be implemented)
                await self._send_alert_notification(alert)

        else:
            if existing_alert:
                # Resolve alert
                existing_alert.resolve(resolved_by=uuid4(), resolution_note="Condition no longer met")

                # Update service status
                self.service_status.active_alerts -= 1
                if existing_alert.severity == AlertSeverity.CRITICAL:
                    self.service_status.critical_alerts -= 1

                self.logger.info(f"Alert resolved: {existing_alert.message}")

    async def _send_alert_notification(self, alert: Alert) -> None:
        """Send alert notification (placeholder for implementation)."""

        # This would integrate with notification systems like:
        # - Slack
        # - PagerDuty
        # - Email
        # - SMS
        # - Discord
        # - Microsoft Teams

        self.logger.info(f"Would send notification for alert: {alert.message}")

    async def _metrics_cleanup(self) -> None:
        """Background task to clean up old metrics data."""

        while self.is_monitoring:
            try:
                cutoff_time = datetime.utcnow() - timedelta(days=7)  # Keep 7 days of detailed data

                for metric in self.metrics.values():
                    initial_count = len(metric.data_points)
                    metric.data_points = [
                        point for point in metric.data_points
                        if point.timestamp > cutoff_time
                    ]

                    removed_count = initial_count - len(metric.data_points)
                    if removed_count > 0:
                        self.logger.debug(f"Cleaned up {removed_count} old data points for metric {metric.name}")

                # Clean up resolved alerts older than 30 days
                alert_cutoff = datetime.utcnow() - timedelta(days=30)
                alerts_to_remove = []

                for alert_id, alert in self.active_alerts.items():
                    if (alert.status == AlertStatus.RESOLVED and
                        alert.resolved_at and
                        alert.resolved_at < alert_cutoff):
                        alerts_to_remove.append(alert_id)

                for alert_id in alerts_to_remove:
                    del self.active_alerts[alert_id]

                if alerts_to_remove:
                    self.logger.info(f"Cleaned up {len(alerts_to_remove)} old resolved alerts")

            except Exception as e:
                self.logger.error(f"Error during metrics cleanup: {e}")

            await asyncio.sleep(3600)  # Clean up every hour

    async def _update_service_status(self) -> None:
        """Update overall service status."""

        # Calculate response time percentiles from recent request metrics
        if "http_request_duration_seconds" in self.metrics:
            duration_metric = self.metrics["http_request_duration_seconds"]
            recent_time = datetime.utcnow() - timedelta(minutes=5)
            recent_points = duration_metric.get_values_in_range(recent_time, datetime.utcnow())

            if recent_points:
                durations = [p.value for p in recent_points if isinstance(p.value, (int, float))]
                if durations:
                    durations.sort()
                    n = len(durations)

                    self.service_status.response_time_p50 = durations[int(n * 0.5)]
                    self.service_status.response_time_p95 = durations[int(n * 0.95)]
                    self.service_status.response_time_p99 = durations[int(n * 0.99)]

        # Calculate error rate
        if "http_requests_total" in self.metrics and "http_errors_total" in self.metrics:
            requests_metric = self.metrics["http_requests_total"]
            errors_metric = self.metrics["http_errors_total"]

            recent_time = datetime.utcnow() - timedelta(minutes=5)

            recent_requests = requests_metric.get_values_in_range(recent_time, datetime.utcnow())
            recent_errors = errors_metric.get_values_in_range(recent_time, datetime.utcnow())

            total_requests = sum(p.value for p in recent_requests if isinstance(p.value, (int, float)))
            total_errors = sum(p.value for p in recent_errors if isinstance(p.value, (int, float)))

            if total_requests > 0:
                self.service_status.error_rate = (total_errors / total_requests) * 100
                self.service_status.throughput = total_requests / 300  # requests per second

        # Update component statuses
        for health_check in self.health_checks.values():
            self.service_status.update_component_status(health_check.name, health_check.is_healthy)

        # Update alert counts
        self.service_status.active_alerts = len([
            a for a in self.active_alerts.values() if a.status == AlertStatus.ACTIVE
        ])
        self.service_status.critical_alerts = len([
            a for a in self.active_alerts.values()
            if a.status == AlertStatus.ACTIVE and a.severity == AlertSeverity.CRITICAL
        ])

    def _update_prometheus_metric(
        self,
        name: str,
        value: Union[float, int, str],
        labels: Optional[Dict[str, str]] = None,
    ) -> None:
        """Update corresponding Prometheus metric if it exists."""

        # This would map internal metrics to Prometheus metrics
        # For now, just log the update
        self.logger.debug(f"Updated metric {name} = {value}")

    def get_prometheus_metrics(self) -> str:
        """Get Prometheus formatted metrics."""

        from prometheus_client import generate_latest
        return generate_latest(self.prometheus_registry).decode('utf-8')

    def get_active_alerts(self) -> List[Alert]:
        """Get list of active alerts."""

        return [
            alert for alert in self.active_alerts.values()
            if alert.status == AlertStatus.ACTIVE
        ]

    def acknowledge_alert(self, alert_id: UUID, acknowledged_by: UUID) -> bool:
        """Acknowledge an alert."""

        if alert_id not in self.active_alerts:
            return False

        alert = self.active_alerts[alert_id]
        alert.acknowledge(acknowledged_by)

        self.logger.info(f"Alert acknowledged: {alert.message}")
        return True

    def resolve_alert(self, alert_id: UUID, resolved_by: UUID, resolution_note: Optional[str] = None) -> bool:
        """Resolve an alert."""

        if alert_id not in self.active_alerts:
            return False

        alert = self.active_alerts[alert_id]
        alert.resolve(resolved_by, resolution_note)

        # Update service status
        self.service_status.active_alerts -= 1
        if alert.severity == AlertSeverity.CRITICAL:
            self.service_status.critical_alerts -= 1

        self.logger.info(f"Alert resolved: {alert.message}")
        return True
