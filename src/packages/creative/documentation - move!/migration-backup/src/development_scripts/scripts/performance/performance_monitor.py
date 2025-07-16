#!/usr/bin/env python3
"""
Real-time Performance Monitoring System for Pynomaly Production

Monitors performance metrics in real-time and triggers alerts when
thresholds are exceeded, especially for high-load scenarios.
"""

import asyncio
import json
import logging
import random
import time
from collections.abc import Callable
from dataclasses import asdict, dataclass
from datetime import datetime, timedelta
from enum import Enum
from pathlib import Path
from typing import Any

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


class AlertSeverity(Enum):
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


class MetricType(Enum):
    RESPONSE_TIME = "response_time"
    THROUGHPUT = "throughput"
    ERROR_RATE = "error_rate"
    RESOURCE_USAGE = "resource_usage"
    USER_LOAD = "user_load"
    AVAILABILITY = "availability"


@dataclass
class PerformanceThreshold:
    metric_name: str
    metric_type: MetricType
    warning_threshold: float
    critical_threshold: float
    unit: str
    description: str
    check_duration_seconds: int = 60
    alert_cooldown_seconds: int = 300


@dataclass
class PerformanceAlert:
    alert_id: str
    metric_name: str
    severity: AlertSeverity
    current_value: float
    threshold_value: float
    message: str
    timestamp: datetime
    resolved: bool = False
    resolved_at: datetime | None = None
    duration_seconds: int = 0
    metadata: dict[str, Any] = None

    def __post_init__(self):
        if self.metadata is None:
            self.metadata = {}


@dataclass
class MetricDataPoint:
    metric_name: str
    value: float
    timestamp: datetime
    tags: dict[str, str] = None

    def __post_init__(self):
        if self.tags is None:
            self.tags = {}


class PerformanceMonitor:
    """Real-time performance monitoring with intelligent alerting."""

    def __init__(self, project_root: Path):
        self.project_root = project_root
        self.monitor_id = f"perf_monitor_{int(time.time())}"
        self.thresholds = []
        self.active_alerts = []
        self.metric_history = []
        self.alert_handlers = []
        self.monitoring_active = False
        self.max_history_points = 10000

        self._setup_default_thresholds()

    def _setup_default_thresholds(self):
        """Set up default performance thresholds for production."""
        default_thresholds = [
            PerformanceThreshold(
                metric_name="api_response_time_ms",
                metric_type=MetricType.RESPONSE_TIME,
                warning_threshold=500,
                critical_threshold=1000,
                unit="ms",
                description="API response time threshold for optimal user experience",
                check_duration_seconds=30,
            ),
            PerformanceThreshold(
                metric_name="api_p95_response_time_ms",
                metric_type=MetricType.RESPONSE_TIME,
                warning_threshold=800,
                critical_threshold=1500,
                unit="ms",
                description="95th percentile response time threshold",
                check_duration_seconds=60,
            ),
            PerformanceThreshold(
                metric_name="error_rate_percent",
                metric_type=MetricType.ERROR_RATE,
                warning_threshold=1.0,
                critical_threshold=3.0,
                unit="%",
                description="API error rate threshold",
                check_duration_seconds=60,
            ),
            PerformanceThreshold(
                metric_name="concurrent_users",
                metric_type=MetricType.USER_LOAD,
                warning_threshold=400,
                critical_threshold=600,
                unit="users",
                description="Concurrent user load threshold",
                check_duration_seconds=30,
            ),
            PerformanceThreshold(
                metric_name="cpu_usage_percent",
                metric_type=MetricType.RESOURCE_USAGE,
                warning_threshold=70,
                critical_threshold=85,
                unit="%",
                description="CPU usage threshold",
                check_duration_seconds=120,
            ),
            PerformanceThreshold(
                metric_name="memory_usage_percent",
                metric_type=MetricType.RESOURCE_USAGE,
                warning_threshold=80,
                critical_threshold=90,
                unit="%",
                description="Memory usage threshold",
                check_duration_seconds=120,
            ),
            PerformanceThreshold(
                metric_name="database_query_time_ms",
                metric_type=MetricType.RESPONSE_TIME,
                warning_threshold=200,
                critical_threshold=500,
                unit="ms",
                description="Database query response time threshold",
                check_duration_seconds=60,
            ),
            PerformanceThreshold(
                metric_name="cache_hit_rate_percent",
                metric_type=MetricType.AVAILABILITY,
                warning_threshold=85,  # Alert if below 85%
                critical_threshold=70,  # Critical if below 70%
                unit="%",
                description="Cache hit rate threshold (alerts when below threshold)",
                check_duration_seconds=180,
            ),
            PerformanceThreshold(
                metric_name="requests_per_second",
                metric_type=MetricType.THROUGHPUT,
                warning_threshold=100,  # Alert if below expected throughput
                critical_threshold=50,
                unit="req/s",
                description="Minimum expected throughput threshold",
                check_duration_seconds=60,
            ),
            PerformanceThreshold(
                metric_name="availability_percent",
                metric_type=MetricType.AVAILABILITY,
                warning_threshold=99.5,  # Alert if below 99.5%
                critical_threshold=99.0,  # Critical if below 99%
                unit="%",
                description="System availability threshold",
                check_duration_seconds=300,
            ),
        ]

        self.thresholds = default_thresholds
        logger.info(f"📊 Configured {len(default_thresholds)} performance thresholds")

    def add_alert_handler(self, handler: Callable[[PerformanceAlert], None]):
        """Add a custom alert handler function."""
        self.alert_handlers.append(handler)

    def record_metric(
        self, metric_name: str, value: float, tags: dict[str, str] = None
    ):
        """Record a performance metric data point."""
        data_point = MetricDataPoint(
            metric_name=metric_name,
            value=value,
            timestamp=datetime.now(),
            tags=tags or {},
        )

        self.metric_history.append(data_point)

        # Keep history size manageable
        if len(self.metric_history) > self.max_history_points:
            self.metric_history = self.metric_history[-self.max_history_points :]

        # Check thresholds for this metric
        self._check_metric_thresholds(data_point)

    def _check_metric_thresholds(self, data_point: MetricDataPoint):
        """Check if a metric data point violates any thresholds."""
        relevant_thresholds = [
            t for t in self.thresholds if t.metric_name == data_point.metric_name
        ]

        for threshold in relevant_thresholds:
            # Get recent data points for this metric
            recent_points = [
                dp
                for dp in self.metric_history
                if (
                    dp.metric_name == data_point.metric_name
                    and dp.timestamp
                    >= datetime.now()
                    - timedelta(seconds=threshold.check_duration_seconds)
                )
            ]

            if len(recent_points) < 3:  # Need enough data points
                continue

            avg_value = sum(dp.value for dp in recent_points) / len(recent_points)

            # Determine if threshold is violated
            severity = None
            threshold_value = None

            if threshold.metric_type in [
                MetricType.AVAILABILITY,
                MetricType.THROUGHPUT,
            ]:
                # For availability and throughput, alert when BELOW threshold
                if avg_value < threshold.critical_threshold:
                    severity = AlertSeverity.CRITICAL
                    threshold_value = threshold.critical_threshold
                elif avg_value < threshold.warning_threshold:
                    severity = AlertSeverity.HIGH
                    threshold_value = threshold.warning_threshold
            else:
                # For other metrics, alert when ABOVE threshold
                if avg_value > threshold.critical_threshold:
                    severity = AlertSeverity.CRITICAL
                    threshold_value = threshold.critical_threshold
                elif avg_value > threshold.warning_threshold:
                    severity = AlertSeverity.HIGH
                    threshold_value = threshold.warning_threshold

            if severity:
                self._trigger_alert(threshold, avg_value, threshold_value, severity)

    def _trigger_alert(
        self,
        threshold: PerformanceThreshold,
        current_value: float,
        threshold_value: float,
        severity: AlertSeverity,
    ):
        """Trigger a performance alert."""

        # Check if similar alert already exists and is recent
        existing_alert = next(
            (
                alert
                for alert in self.active_alerts
                if (
                    alert.metric_name == threshold.metric_name
                    and not alert.resolved
                    and alert.timestamp
                    >= datetime.now()
                    - timedelta(seconds=threshold.alert_cooldown_seconds)
                )
            ),
            None,
        )

        if existing_alert:
            return  # Don't spam alerts

        # Create new alert
        alert = PerformanceAlert(
            alert_id=f"alert_{int(time.time())}_{hash(threshold.metric_name) % 1000}",
            metric_name=threshold.metric_name,
            severity=severity,
            current_value=current_value,
            threshold_value=threshold_value,
            message=self._generate_alert_message(
                threshold, current_value, threshold_value, severity
            ),
            timestamp=datetime.now(),
            metadata={
                "threshold_description": threshold.description,
                "metric_type": threshold.metric_type.value,
                "check_duration_seconds": threshold.check_duration_seconds,
            },
        )

        self.active_alerts.append(alert)

        # Execute alert handlers
        for handler in self.alert_handlers:
            try:
                handler(alert)
            except Exception as e:
                logger.error(f"Alert handler failed: {e}")

        logger.warning(f"🚨 {severity.value.upper()} ALERT: {alert.message}")

    def _generate_alert_message(
        self,
        threshold: PerformanceThreshold,
        current_value: float,
        threshold_value: float,
        severity: AlertSeverity,
    ) -> str:
        """Generate a descriptive alert message."""

        if threshold.metric_type in [MetricType.AVAILABILITY, MetricType.THROUGHPUT]:
            # Below threshold alerts
            message = (
                f"{threshold.metric_name.replace('_', ' ').title()} is below threshold: "
                f"{current_value:.2f}{threshold.unit} < {threshold_value:.2f}{threshold.unit}"
            )
        else:
            # Above threshold alerts
            message = (
                f"{threshold.metric_name.replace('_', ' ').title()} exceeded threshold: "
                f"{current_value:.2f}{threshold.unit} > {threshold_value:.2f}{threshold.unit}"
            )

        if severity == AlertSeverity.CRITICAL:
            message += " - IMMEDIATE ACTION REQUIRED"
        elif severity == AlertSeverity.HIGH:
            message += " - Attention needed"

        return message

    def resolve_alert(self, alert_id: str):
        """Manually resolve an alert."""
        alert = next((a for a in self.active_alerts if a.alert_id == alert_id), None)
        if alert and not alert.resolved:
            alert.resolved = True
            alert.resolved_at = datetime.now()
            alert.duration_seconds = int(
                (alert.resolved_at - alert.timestamp).total_seconds()
            )
            logger.info(f"✅ Alert resolved: {alert.alert_id}")

    def auto_resolve_alerts(self):
        """Auto-resolve alerts when conditions improve."""
        for alert in self.active_alerts:
            if alert.resolved:
                continue

            threshold = next(
                (t for t in self.thresholds if t.metric_name == alert.metric_name), None
            )
            if not threshold:
                continue

            # Get recent data points
            recent_points = [
                dp
                for dp in self.metric_history
                if (
                    dp.metric_name == alert.metric_name
                    and dp.timestamp
                    >= datetime.now()
                    - timedelta(seconds=threshold.check_duration_seconds)
                )
            ]

            if len(recent_points) < 3:
                continue

            avg_value = sum(dp.value for dp in recent_points) / len(recent_points)

            # Check if condition has improved
            should_resolve = False

            if threshold.metric_type in [
                MetricType.AVAILABILITY,
                MetricType.THROUGHPUT,
            ]:
                # Should resolve if value is back above warning threshold
                should_resolve = avg_value >= threshold.warning_threshold
            else:
                # Should resolve if value is back below warning threshold
                should_resolve = avg_value <= threshold.warning_threshold

            if should_resolve:
                self.resolve_alert(alert.alert_id)

    async def simulate_production_metrics(self, duration_seconds: int = 300):
        """Simulate real-time production metrics for demonstration."""
        logger.info(
            f"📊 Starting production metrics simulation for {duration_seconds}s..."
        )

        start_time = time.time()

        while time.time() - start_time < duration_seconds:
            # Simulate various load scenarios
            current_minute = int((time.time() - start_time) / 60)

            # Simulate load patterns (traffic spikes, etc.)
            if current_minute % 5 == 0:  # Traffic spike every 5 minutes
                load_multiplier = random.uniform(1.5, 2.5)
            else:
                load_multiplier = random.uniform(0.8, 1.2)

            # Simulate concurrent users
            base_users = 200
            concurrent_users = base_users * load_multiplier + random.uniform(-50, 50)
            self.record_metric("concurrent_users", max(0, concurrent_users))

            # Simulate API response times (affected by load)
            base_response_time = 150
            response_time = base_response_time * (
                1 + (concurrent_users - 200) / 1000
            ) + random.uniform(-20, 40)
            self.record_metric("api_response_time_ms", max(50, response_time))

            # Simulate P95 response time
            p95_response_time = response_time * 1.8 + random.uniform(0, 100)
            self.record_metric("api_p95_response_time_ms", p95_response_time)

            # Simulate error rate (increases with high load)
            base_error_rate = 0.2
            error_rate = (
                base_error_rate
                + max(0, (concurrent_users - 400) / 1000)
                + random.uniform(-0.1, 0.3)
            )
            self.record_metric("error_rate_percent", max(0, error_rate))

            # Simulate resource usage
            cpu_usage = 30 + (concurrent_users / 10) + random.uniform(-5, 10)
            self.record_metric("cpu_usage_percent", max(10, min(100, cpu_usage)))

            memory_usage = 40 + (concurrent_users / 8) + random.uniform(-5, 8)
            self.record_metric("memory_usage_percent", max(20, min(100, memory_usage)))

            # Simulate database performance
            db_query_time = 80 + (concurrent_users / 5) + random.uniform(-10, 20)
            self.record_metric("database_query_time_ms", max(30, db_query_time))

            # Simulate cache hit rate (decreases slightly under load)
            cache_hit_rate = 92 - (concurrent_users - 200) / 100 + random.uniform(-2, 2)
            self.record_metric(
                "cache_hit_rate_percent", max(70, min(100, cache_hit_rate))
            )

            # Simulate throughput
            throughput = concurrent_users * 2.5 + random.uniform(-20, 20)
            self.record_metric("requests_per_second", max(10, throughput))

            # Simulate availability
            availability = 99.8 if error_rate < 1.0 else (99.5 - error_rate / 2)
            self.record_metric("availability_percent", max(95, availability))

            # Auto-resolve alerts that have improved
            self.auto_resolve_alerts()

            await asyncio.sleep(5)  # Check every 5 seconds

    def get_active_alerts(self) -> list[PerformanceAlert]:
        """Get all active (unresolved) alerts."""
        return [alert for alert in self.active_alerts if not alert.resolved]

    def get_metric_summary(self, hours: int = 1) -> dict[str, Any]:
        """Get summary of metrics for the specified time period."""
        cutoff_time = datetime.now() - timedelta(hours=hours)
        recent_metrics = [
            dp for dp in self.metric_history if dp.timestamp >= cutoff_time
        ]

        if not recent_metrics:
            return {"message": "No recent metrics available"}

        # Group by metric name
        metric_groups = {}
        for dp in recent_metrics:
            if dp.metric_name not in metric_groups:
                metric_groups[dp.metric_name] = []
            metric_groups[dp.metric_name].append(dp.value)

        summary = {}
        for metric_name, values in metric_groups.items():
            summary[metric_name] = {
                "count": len(values),
                "avg": sum(values) / len(values),
                "min": min(values),
                "max": max(values),
                "latest": values[-1],
            }

        return {
            "period_hours": hours,
            "total_data_points": len(recent_metrics),
            "metrics": summary,
            "active_alerts": len(self.get_active_alerts()),
            "timestamp": datetime.now().isoformat(),
        }

    def generate_monitoring_report(self) -> dict[str, Any]:
        """Generate comprehensive monitoring report."""
        active_alerts = self.get_active_alerts()
        resolved_alerts = [alert for alert in self.active_alerts if alert.resolved]

        # Alert statistics
        alert_stats = {
            "total_alerts": len(self.active_alerts),
            "active_alerts": len(active_alerts),
            "resolved_alerts": len(resolved_alerts),
            "critical_alerts": len(
                [a for a in active_alerts if a.severity == AlertSeverity.CRITICAL]
            ),
            "high_alerts": len(
                [a for a in active_alerts if a.severity == AlertSeverity.HIGH]
            ),
        }

        # Recent performance summary
        performance_summary = self.get_metric_summary(hours=1)

        report = {
            "monitor_id": self.monitor_id,
            "generated_at": datetime.now().isoformat(),
            "monitoring_duration_hours": (
                datetime.now()
                - datetime.fromtimestamp(int(self.monitor_id.split("_")[-1]))
            ).total_seconds()
            / 3600,
            "alert_statistics": alert_stats,
            "performance_summary": performance_summary,
            "active_alerts": [asdict(alert) for alert in active_alerts],
            "threshold_configuration": [
                asdict(threshold) for threshold in self.thresholds
            ],
            "recommendations": self._generate_monitoring_recommendations(
                active_alerts, performance_summary
            ),
        }

        return report

    def _generate_monitoring_recommendations(
        self, active_alerts: list[PerformanceAlert], performance_summary: dict[str, Any]
    ) -> list[str]:
        """Generate monitoring recommendations based on current state."""
        recommendations = []

        if len(active_alerts) == 0:
            recommendations.append(
                "✅ No active alerts - system performing within thresholds"
            )
        else:
            critical_alerts = [
                a for a in active_alerts if a.severity == AlertSeverity.CRITICAL
            ]
            if critical_alerts:
                recommendations.append(
                    f"🚨 {len(critical_alerts)} CRITICAL alerts require immediate attention"
                )

            high_alerts = [a for a in active_alerts if a.severity == AlertSeverity.HIGH]
            if high_alerts:
                recommendations.append(
                    f"⚠️ {len(high_alerts)} HIGH priority alerts need investigation"
                )

        # Check performance trends
        metrics = performance_summary.get("metrics", {})

        if "api_response_time_ms" in metrics:
            avg_response = metrics["api_response_time_ms"]["avg"]
            if avg_response > 300:
                recommendations.append(
                    "📈 API response times elevated - consider performance optimization"
                )

        if "concurrent_users" in metrics:
            max_users = metrics["concurrent_users"]["max"]
            if max_users > 400:
                recommendations.append(
                    "👥 High user load detected - monitor for capacity limits"
                )

        if "error_rate_percent" in metrics:
            avg_errors = metrics["error_rate_percent"]["avg"]
            if avg_errors > 0.5:
                recommendations.append(
                    "❌ Error rate above baseline - investigate error patterns"
                )

        # General recommendations
        recommendations.extend(
            [
                "📊 Continue real-time monitoring of key performance indicators",
                "🔄 Review alert thresholds weekly and adjust based on usage patterns",
                "📋 Set up automated alert escalation for critical issues",
                "📈 Implement performance trending analysis for proactive optimization",
            ]
        )

        return recommendations


def email_alert_handler(alert: PerformanceAlert):
    """Example alert handler that would send email notifications."""
    logger.info(f"📧 EMAIL ALERT: {alert.severity.value.upper()} - {alert.message}")
    # In production, this would integrate with email service


def slack_alert_handler(alert: PerformanceAlert):
    """Example alert handler that would send Slack notifications."""
    logger.info(f"💬 SLACK ALERT: {alert.severity.value.upper()} - {alert.message}")
    # In production, this would integrate with Slack API


def pagerduty_alert_handler(alert: PerformanceAlert):
    """Example alert handler for PagerDuty integration."""
    if alert.severity in [AlertSeverity.CRITICAL, AlertSeverity.HIGH]:
        logger.info(
            f"📟 PAGERDUTY ALERT: {alert.severity.value.upper()} - {alert.message}"
        )
        # In production, this would integrate with PagerDuty API


async def main():
    """Main monitoring execution."""
    project_root = Path(__file__).parent.parent.parent
    monitor = PerformanceMonitor(project_root)

    print("📊 Real-time Performance Monitoring System")
    print("=" * 50)
    print("🎯 Monitoring production performance with intelligent alerting")

    # Add alert handlers
    monitor.add_alert_handler(email_alert_handler)
    monitor.add_alert_handler(slack_alert_handler)
    monitor.add_alert_handler(pagerduty_alert_handler)

    print(f"\n⚠️ Configured {len(monitor.thresholds)} performance thresholds:")
    for threshold in monitor.thresholds[:5]:
        print(
            f"  🔹 {threshold.metric_name}: Warning {threshold.warning_threshold}{threshold.unit}, Critical {threshold.critical_threshold}{threshold.unit}"
        )

    print("\n🚀 Starting real-time monitoring simulation...")

    # Run monitoring simulation
    await monitor.simulate_production_metrics(duration_seconds=120)  # 2 minutes demo

    # Generate final report
    report = monitor.generate_monitoring_report()

    print("\n📊 Monitoring Summary:")
    print(f"  ⏱️ Duration: {report['monitoring_duration_hours']:.2f} hours")
    print(
        f"  📈 Total data points: {report['performance_summary'].get('total_data_points', 0)}"
    )
    print(f"  🚨 Active alerts: {report['alert_statistics']['active_alerts']}")
    print(f"  ✅ Resolved alerts: {report['alert_statistics']['resolved_alerts']}")

    if report["alert_statistics"]["critical_alerts"] > 0:
        print(f"  🚨 CRITICAL alerts: {report['alert_statistics']['critical_alerts']}")

    print("\n📋 Key Recommendations:")
    for rec in report["recommendations"][:4]:
        print(f"  {rec}")

    # Save detailed report
    report_file = (
        project_root / f"performance_monitoring_report_{int(time.time())}.json"
    )
    with open(report_file, "w") as f:
        json.dump(report, f, indent=2, default=str)

    print(f"\n📄 Full monitoring report saved to: {report_file}")

    if report["alert_statistics"]["critical_alerts"] == 0:
        print("\n✅ Performance monitoring completed successfully!")
        return 0
    else:
        print("\n⚠️ Critical alerts detected - system needs attention.")
        return 1


if __name__ == "__main__":
    import sys

    sys.exit(asyncio.run(main()))
