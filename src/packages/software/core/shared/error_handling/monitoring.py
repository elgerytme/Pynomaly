"""Error monitoring and alerting system for comprehensive error tracking."""

from __future__ import annotations

import asyncio
import logging
import statistics
import time
from collections import defaultdict, deque
from collections.abc import Callable
from dataclasses import dataclass, field
from enum import Enum
from typing import Any

from .resilience import get_resilience_manager
from .unified_exceptions import PynamolyError

logger = logging.getLogger(__name__)


class AlertLevel(Enum):
    """Alert severity levels."""

    INFO = "info"
    WARNING = "warning"
    ERROR = "error"
    CRITICAL = "critical"


@dataclass
class ErrorMetrics:
    """Error measurements for monitoring."""

    total_errors: int = 0
    error_rate: float = 0.0  # errors per minute
    avg_error_severity: float = 0.0
    error_categories: dict[str, int] = field(default_factory=dict)
    error_codes: dict[str, int] = field(default_factory=dict)
    components_affected: set[str] = field(default_factory=set)
    recent_errors: deque = field(default_factory=lambda: deque(maxlen=100))

    def add_error(self, error: PynamolyError) -> None:
        """Add error to measurements."""
        self.total_errors += 1
        self.recent_errors.append(
            {
                "timestamp": time.time(),
                "error_code": error.details.error_code,
                "category": error.details.category.value,
                "severity": error.details.severity.value,
                "component": error.details.context.component,
                "message": error.details.message,
            }
        )

        # Update category counts
        category = error.details.category.value
        self.error_categories[category] = self.error_categories.get(category, 0) + 1

        # Update error code counts
        error_code = error.details.error_code
        self.error_codes[error_code] = self.error_codes.get(error_code, 0) + 1

        # Track affected components
        if error.details.context.component:
            self.components_affected.add(error.details.context.component)

        # Calculate error rate (errors per minute)
        current_time = time.time()
        recent_errors_count = sum(
            1
            for error_data in self.recent_errors
            if current_time - error_data["timestamp"] <= 60
        )
        self.error_rate = recent_errors_count

        # Calculate average severity
        severity_values = {"low": 1, "medium": 2, "high": 3, "critical": 4}

        if self.recent_errors:
            avg_severity = statistics.mean(
                severity_values.get(error_data["severity"], 2)
                for error_data in self.recent_errors
            )
            self.avg_error_severity = avg_severity


@dataclass
class AlertRule:
    """Alert rule configuration."""

    name: str
    condition: str  # Python expression to evaluate
    alert_level: AlertLevel
    cooldown_minutes: int = 5
    description: str = ""
    enabled: bool = True
    last_triggered: float | None = None
    trigger_count: int = 0


@dataclass
class Alert:
    """Alert notification."""

    id: str
    rule_name: str
    level: AlertLevel
    message: str
    timestamp: float
    context: dict[str, Any]
    resolved: bool = False
    resolved_at: float | None = None


class ErrorMonitor:
    """Comprehensive error monitoring system."""

    def __init__(
        self,
        alert_callback: Callable[[Alert], None] | None = None,
        monitoring_interval: float = 60.0,  # seconds
        retention_hours: int = 24,
    ):
        """Initialize error monitor.

        Args:
            alert_callback: Function to call when alerts are triggered
            monitoring_interval: How often to check alert conditions
            retention_hours: How long to keep error history
        """
        self.alert_callback = alert_callback
        self.monitoring_interval = monitoring_interval
        self.retention_hours = retention_hours

        # Error tracking
        self.global_measurements = ErrorMetrics()
        self.component_measurements: dict[str, ErrorMetrics] = defaultdict(ErrorMetrics)
        self.user_measurements: dict[str, ErrorMetrics] = defaultdict(ErrorMetrics)

        # Alert system
        self.alert_rules: list[AlertRule] = []
        self.active_alerts: dict[str, Alert] = {}
        self.alert_history: list[Alert] = []

        # Monitoring state
        self.monitoring_task: asyncio.Task | None = None
        self.is_monitoring = False

        # Initialize default alert rules
        self._initialize_default_rules()

    def _initialize_default_rules(self) -> None:
        """Initialize default alert rules."""
        default_rules = [
            AlertRule(
                name="high_error_rate",
                condition="measurements.error_rate > 10",
                alert_level=AlertLevel.WARNING,
                cooldown_minutes=5,
                description="High error rate detected (>10 errors/minute)",
            ),
            AlertRule(
                name="critical_error_rate",
                condition="measurements.error_rate > 50",
                alert_level=AlertLevel.CRITICAL,
                cooldown_minutes=2,
                description="Critical error rate detected (>50 errors/minute)",
            ),
            AlertRule(
                name="high_severity_errors",
                condition="measurements.avg_error_severity > 3.0",
                alert_level=AlertLevel.ERROR,
                cooldown_minutes=10,
                description="High average error severity detected",
            ),
            AlertRule(
                name="multiple_components_affected",
                condition="len(measurements.components_affected) > 5",
                alert_level=AlertLevel.WARNING,
                cooldown_minutes=15,
                description="Multiple components affected by errors",
            ),
            AlertRule(
                name="data_integrity_errors",
                condition="measurements.error_categories.get('data_integrity', 0) > 0",
                alert_level=AlertLevel.CRITICAL,
                cooldown_minutes=1,
                description="Data integrity errors detected",
            ),
            AlertRule(
                name="authentication_failures",
                condition="measurements.error_categories.get('authentication', 0) > 5",
                alert_level=AlertLevel.WARNING,
                cooldown_minutes=5,
                description="Multiple authentication failures",
            ),
        ]

        self.alert_rules.extend(default_rules)

    def add_alert_rule(self, rule: AlertRule) -> None:
        """Add custom alert rule."""
        self.alert_rules.append(rule)
        logger.info(f"Added alert rule: {rule.name}")

    def remove_alert_rule(self, rule_name: str) -> bool:
        """Remove alert rule by name."""
        for i, rule in enumerate(self.alert_rules):
            if rule.name == rule_name:
                self.alert_rules.pop(i)
                logger.info(f"Removed alert rule: {rule_name}")
                return True
        return False

    def track_error(self, error: PynamolyError) -> None:
        """Track error in monitoring system."""
        try:
            # Add to global measurements
            self.global_measurements.add_error(error)

            # Add to component measurements
            if error.details.context.component:
                self.component_measurements[error.details.context.component].add_error(error)

            # Add to user measurements
            if error.details.context.user_id:
                self.user_measurements[error.details.context.user_id].add_error(error)

            # Log error for structured logging
            logger.error(
                "Error tracked in monitoring system",
                extra={
                    "error_id": error.details.context.error_id,
                    "error_code": error.details.error_code,
                    "category": error.details.category.value,
                    "severity": error.details.severity.value,
                    "component": error.details.context.component,
                    "user_id": error.details.context.user_id,
                    "message": error.details.message,
                },
            )

        except Exception as e:
            logger.error(f"Failed to track error: {e}")

    def start_monitoring(self) -> None:
        """Start error monitoring."""
        if not self.is_monitoring:
            self.is_monitoring = True
            self.monitoring_task = asyncio.create_task(self._monitoring_loop())
            logger.info("Error monitoring started")

    async def stop_monitoring(self) -> None:
        """Stop error monitoring."""
        if self.is_monitoring:
            self.is_monitoring = False
            if self.monitoring_task:
                self.monitoring_task.cancel()
                try:
                    await self.monitoring_task
                except asyncio.CancelledError:
                    pass
            logger.info("Error monitoring stopped")

    async def _monitoring_loop(self) -> None:
        """Main monitoring loop."""
        while self.is_monitoring:
            try:
                await asyncio.sleep(self.monitoring_interval)
                await self._check_alert_conditions()
                await self._cleanup_old_data()
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Error in monitoring loop: {e}")

    async def _check_alert_conditions(self) -> None:
        """Check all alert conditions."""
        current_time = time.time()

        for rule in self.alert_rules:
            if not rule.enabled:
                continue

            # Check cooldown
            if (
                rule.last_triggered
                and current_time - rule.last_triggered < rule.cooldown_minutes * 60
            ):
                continue

            try:
                # Evaluate condition
                if await self._evaluate_condition(rule.condition):
                    await self._trigger_alert(rule)

            except Exception as e:
                logger.error(f"Error evaluating alert rule '{rule.name}': {e}")

    async def _evaluate_condition(self, condition: str) -> bool:
        """Evaluate alert condition."""
        try:
            # Create context for condition evaluation
            context = {
                "measurements": self.global_measurements,
                "component_measurements": self.component_measurements,
                "user_measurements": self.user_measurements,
                "time": time.time(),
                "len": len,
                "sum": sum,
                "max": max,
                "min": min,
                "statistics": statistics,
            }

            # Safely evaluate condition
            return eval(condition, {"__builtins__": {}}, context)

        except Exception as e:
            logger.error(f"Error evaluating condition '{condition}': {e}")
            return False

    async def _trigger_alert(self, rule: AlertRule) -> None:
        """Trigger alert for rule."""
        current_time = time.time()

        # Create alert
        alert = Alert(
            id=f"{rule.name}_{int(current_time)}",
            rule_name=rule.name,
            level=rule.alert_level,
            message=rule.description,
            timestamp=current_time,
            context=self._get_alert_context(),
        )

        # Update rule state
        rule.last_triggered = current_time
        rule.trigger_count += 1

        # Store alert
        self.active_alerts[alert.id] = alert
        self.alert_history.append(alert)

        # Call alert callback
        if self.alert_callback:
            try:
                self.alert_callback(alert)
            except Exception as e:
                logger.error(f"Error in alert callback: {e}")

        logger.warning(
            f"Alert triggered: {rule.name}",
            extra={
                "alert_id": alert.id,
                "alert_level": alert.level.value,
                "message": alert.message,
                "context": alert.context,
            },
        )

    def _get_alert_context(self) -> dict[str, Any]:
        """Get context information for alerts."""
        resilience_manager = get_resilience_manager()

        return {
            "global_measurements": {
                "total_errors": self.global_measurements.total_errors,
                "error_rate": self.global_measurements.error_rate,
                "avg_error_severity": self.global_measurements.avg_error_severity,
                "error_categories": dict(self.global_measurements.error_categories),
                "components_affected": list(self.global_measurements.components_affected),
            },
            "top_error_codes": dict(
                sorted(
                    self.global_measurements.error_codes.items(),
                    key=lambda x: x[1],
                    reverse=True,
                )[:10]
            ),
            "resilience_stats": resilience_manager.get_comprehensive_stats(),
            "timestamp": time.time(),
        }

    def resolve_alert(self, alert_id: str) -> bool:
        """Resolve an active alert."""
        if alert_id in self.active_alerts:
            alert = self.active_alerts[alert_id]
            alert.resolved = True
            alert.resolved_at = time.time()
            del self.active_alerts[alert_id]

            logger.info(f"Alert resolved: {alert_id}")
            return True
        return False

    async def _cleanup_old_data(self) -> None:
        """Clean up old monitoring data."""
        current_time = time.time()
        retention_seconds = self.retention_hours * 3600

        # Clean up alert history
        self.alert_history = [
            alert
            for alert in self.alert_history
            if current_time - alert.timestamp < retention_seconds
        ]

        # Clean up error measurements (recent_errors are already limited by deque maxlen)
        # This is handled automatically by the deque with maxlen

    def get_dashboard_data(self) -> dict[str, Any]:
        """Get comprehensive dashboard data."""
        current_time = time.time()

        # Recent alerts (last 24 hours)
        recent_alerts = [
            {
                "id": alert.id,
                "rule_name": alert.rule_name,
                "level": alert.level.value,
                "message": alert.message,
                "timestamp": alert.timestamp,
                "resolved": alert.resolved,
                "resolved_at": alert.resolved_at,
            }
            for alert in self.alert_history
            if current_time - alert.timestamp < 86400  # 24 hours
        ]

        # Top error codes
        top_error_codes = dict(
            sorted(
                self.global_measurements.error_codes.items(),
                key=lambda x: x[1],
                reverse=True,
            )[:10]
        )

        # Component health
        component_health = {}
        for component, measurements in self.component_measurements.items():
            component_health[component] = {
                "total_errors": measurements.total_errors,
                "error_rate": measurements.error_rate,
                "avg_severity": measurements.avg_error_severity,
                "health_status": self._calculate_component_health(measurements),
            }

        # Alert rule status
        alert_rules_status = [
            {
                "name": rule.name,
                "enabled": rule.enabled,
                "description": rule.description,
                "alert_level": rule.alert_level.value,
                "trigger_count": rule.trigger_count,
                "last_triggered": rule.last_triggered,
                "cooldown_minutes": rule.cooldown_minutes,
            }
            for rule in self.alert_rules
        ]

        return {
            "monitoring_status": {
                "is_monitoring": self.is_monitoring,
                "monitoring_interval": self.monitoring_interval,
                "retention_hours": self.retention_hours,
            },
            "global_measurements": {
                "total_errors": self.global_measurements.total_errors,
                "error_rate": self.global_measurements.error_rate,
                "avg_error_severity": self.global_measurements.avg_error_severity,
                "error_categories": dict(self.global_measurements.error_categories),
                "components_affected": list(self.global_measurements.components_affected),
            },
            "top_error_codes": top_error_codes,
            "component_health": component_health,
            "active_alerts": len(self.active_alerts),
            "recent_alerts": recent_alerts,
            "alert_rules": alert_rules_status,
            "timestamp": current_time,
        }

    def _calculate_component_health(self, metrics: ErrorMetrics) -> str:
        """Calculate component health status."""
        if measurements.error_rate > 20:
            return "critical"
        elif measurements.error_rate > 10:
            return "degraded"
        elif measurements.avg_error_severity > 3.0:
            return "warning"
        elif measurements.total_errors > 0:
            return "healthy_with_errors"
        else:
            return "healthy"

    def get_error_trends(self, hours: int = 24) -> dict[str, Any]:
        """Get error trends over time."""
        current_time = time.time()
        start_time = current_time - (hours * 3600)

        # Group errors by hour
        hourly_errors = defaultdict(int)
        hourly_severity = defaultdict(list)

        for error_data in self.global_measurements.recent_errors:
            if error_data["timestamp"] >= start_time:
                hour = int(error_data["timestamp"] // 3600)
                hourly_errors[hour] += 1

                severity_values = {"low": 1, "medium": 2, "high": 3, "critical": 4}
                hourly_severity[hour].append(
                    severity_values.get(error_data["severity"], 2)
                )

        # Calculate averages
        trends = []
        for hour in sorted(hourly_errors.keys()):
            avg_severity = (
                statistics.mean(hourly_severity[hour]) if hourly_severity[hour] else 0
            )
            trends.append(
                {
                    "hour": hour,
                    "timestamp": hour * 3600,
                    "error_count": hourly_errors[hour],
                    "avg_severity": avg_severity,
                }
            )

        return {
            "trends": trends,
            "summary": {
                "total_hours": hours,
                "total_errors": sum(hourly_errors.values()),
                "avg_errors_per_hour": statistics.mean(hourly_errors.values())
                if hourly_errors
                else 0,
                "peak_hour": max(hourly_errors.items(), key=lambda x: x[1])
                if hourly_errors
                else None,
            },
        }


# Global error monitor instance
_error_monitor: ErrorMonitor | None = None


def get_error_monitor(
    alert_callback: Callable[[Alert], None] | None = None,
    monitoring_interval: float = 60.0,
    retention_hours: int = 24,
) -> ErrorMonitor:
    """Get or create global error monitor."""
    global _error_monitor

    if _error_monitor is None:
        _error_monitor = ErrorMonitor(
            alert_callback=alert_callback,
            monitoring_interval=monitoring_interval,
            retention_hours=retention_hours,
        )

    return _error_monitor


def track_error(error: PynamolyError) -> None:
    """Track error in global error monitor."""
    monitor = get_error_monitor()
    monitor.track_error(error)


def start_error_monitoring() -> None:
    """Start global error monitoring."""
    monitor = get_error_monitor()
    monitor.start_monitoring()


async def stop_error_monitoring() -> None:
    """Stop global error monitoring."""
    global _error_monitor

    if _error_monitor:
        await _error_monitor.stop_monitoring()
        _error_monitor = None
