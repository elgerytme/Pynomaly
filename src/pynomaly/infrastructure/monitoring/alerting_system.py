"""Comprehensive alerting system for anomaly detection processing."""

import asyncio
import json
import logging
import re
import smtplib
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from enum import Enum
from typing import Any, Dict, List, Optional, Callable
from urllib.parse import urlparse
import time

try:
    import requests
    REQUESTS_AVAILABLE = True
except ImportError:
    REQUESTS_AVAILABLE = False

from .prometheus_metrics_enhanced import get_metrics_collector

logger = logging.getLogger(__name__)


class AlertSeverity(Enum):
    """Alert severity levels."""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


class AlertType(Enum):
    """Types of alerts."""
    JOB_FAILURE = "job_failure"
    SLA_BREACH = "sla_breach"
    HIGH_ERROR_RATE = "high_error_rate"
    MEMORY_USAGE = "memory_usage"
    PERFORMANCE_DEGRADATION = "performance_degradation"
    ANOMALY_SPIKE = "anomaly_spike"
    SYSTEM_HEALTH = "system_health"
    RESOURCE_EXHAUSTION = "resource_exhaustion"


class NotificationType(Enum):
    """Types of notifications."""
    EMAIL = "email"
    WEBHOOK = "webhook"
    SLACK = "slack"
    TEAMS = "teams"
    PAGERDUTY = "pagerduty"
    LOG = "log"


@dataclass
class AlertRule:
    """Alert rule configuration."""
    name: str
    description: str
    alert_type: AlertType
    severity: AlertSeverity
    condition: str  # Expression to evaluate
    threshold: float
    duration_seconds: int = 300  # How long condition must be true
    cooldown_seconds: int = 3600  # Cooldown period between alerts
    enabled: bool = True
    
    # Custom fields
    metadata: Dict[str, Any] = field(default_factory=dict)
    tags: List[str] = field(default_factory=list)
    
    def __post_init__(self):
        """Validate rule configuration."""
        if self.duration_seconds < 0:
            raise ValueError("Duration must be non-negative")
        if self.cooldown_seconds < 0:
            raise ValueError("Cooldown must be non-negative")
        if self.threshold < 0:
            raise ValueError("Threshold must be non-negative")


@dataclass
class Alert:
    """Alert instance."""
    id: str
    rule: AlertRule
    triggered_at: datetime
    resolved_at: Optional[datetime] = None
    value: float = 0.0
    message: str = ""
    context: Dict[str, Any] = field(default_factory=dict)
    
    @property
    def is_resolved(self) -> bool:
        """Check if alert is resolved."""
        return self.resolved_at is not None
    
    @property
    def duration(self) -> timedelta:
        """Get alert duration."""
        end_time = self.resolved_at or datetime.now()
        return end_time - self.triggered_at


@dataclass
class NotificationConfig:
    """Notification configuration."""
    type: NotificationType
    enabled: bool = True
    config: Dict[str, Any] = field(default_factory=dict)
    
    # Filtering
    min_severity: AlertSeverity = AlertSeverity.LOW
    alert_types: List[AlertType] = field(default_factory=list)
    tags: List[str] = field(default_factory=list)


class AlertingSystem:
    """Comprehensive alerting system for anomaly detection processing."""
    
    def __init__(self):
        """Initialize alerting system."""
        self.metrics_collector = get_metrics_collector()
        
        # Alert rules and active alerts
        self.rules: Dict[str, AlertRule] = {}
        self.active_alerts: Dict[str, Alert] = {}
        self.alert_history: List[Alert] = []
        
        # Notification configs
        self.notification_configs: List[NotificationConfig] = []
        
        # Monitoring
        self._monitoring_task: Optional[asyncio.Task] = None
        self._monitoring_active = False
        
        # Log patterns for log-based alerting
        self.log_patterns: Dict[str, Dict[str, Any]] = {}
        
        # SLA configurations
        self.sla_configs: Dict[str, Dict[str, Any]] = {}
        
        logger.info("Alerting system initialized")
    
    def add_rule(self, rule: AlertRule) -> None:
        """Add alert rule."""
        self.rules[rule.name] = rule
        logger.info(f"Added alert rule: {rule.name}")
    
    def remove_rule(self, rule_name: str) -> bool:
        """Remove alert rule."""
        if rule_name in self.rules:
            del self.rules[rule_name]
            logger.info(f"Removed alert rule: {rule_name}")
            return True
        return False
    
    def add_notification_config(self, config: NotificationConfig) -> None:
        """Add notification configuration."""
        self.notification_configs.append(config)
        logger.info(f"Added notification config: {config.type.value}")
    
    def add_log_pattern(self, name: str, pattern: str, severity: AlertSeverity, 
                       threshold: int = 1, window_seconds: int = 300) -> None:
        """Add log pattern for monitoring."""
        self.log_patterns[name] = {
            "pattern": re.compile(pattern),
            "severity": severity,
            "threshold": threshold,
            "window_seconds": window_seconds,
            "matches": []
        }
        logger.info(f"Added log pattern: {name}")
    
    def add_sla_config(self, name: str, metric: str, threshold: float, 
                      duration_seconds: int = 300) -> None:
        """Add SLA configuration."""
        self.sla_configs[name] = {
            "metric": metric,
            "threshold": threshold,
            "duration_seconds": duration_seconds,
            "violations": []
        }
        logger.info(f"Added SLA config: {name}")
    
    async def start_monitoring(self) -> None:
        """Start alert monitoring."""
        if self._monitoring_active:
            return
        
        self._monitoring_active = True
        self._monitoring_task = asyncio.create_task(self._monitoring_loop())
        logger.info("Started alert monitoring")
    
    async def stop_monitoring(self) -> None:
        """Stop alert monitoring."""
        if not self._monitoring_active:
            return
        
        self._monitoring_active = False
        if self._monitoring_task:
            self._monitoring_task.cancel()
            try:
                await self._monitoring_task
            except asyncio.CancelledError:
                pass
        
        logger.info("Stopped alert monitoring")
    
    async def _monitoring_loop(self) -> None:
        """Main monitoring loop."""
        while self._monitoring_active:
            try:
                await self._check_alert_rules()
                await self._check_sla_violations()
                await self._cleanup_old_alerts()
                
                await asyncio.sleep(30)  # Check every 30 seconds
                
            except Exception as e:
                logger.error(f"Error in monitoring loop: {e}")
                await asyncio.sleep(60)
    
    async def _check_alert_rules(self) -> None:
        """Check all alert rules."""
        for rule_name, rule in self.rules.items():
            if not rule.enabled:
                continue
            
            try:
                # Check if rule condition is met
                if await self._evaluate_rule_condition(rule):
                    await self._trigger_alert(rule)
                else:
                    await self._resolve_alert(rule_name)
            
            except Exception as e:
                logger.error(f"Error checking rule {rule_name}: {e}")
    
    async def _evaluate_rule_condition(self, rule: AlertRule) -> bool:
        """Evaluate rule condition."""
        # This is a simplified evaluation - in production, you'd want
        # a more sophisticated expression evaluator
        
        if rule.alert_type == AlertType.JOB_FAILURE:
            # Check job failure rate
            return await self._check_job_failure_rate(rule)
        
        elif rule.alert_type == AlertType.HIGH_ERROR_RATE:
            # Check error rate
            return await self._check_error_rate(rule)
        
        elif rule.alert_type == AlertType.MEMORY_USAGE:
            # Check memory usage
            return await self._check_memory_usage(rule)
        
        elif rule.alert_type == AlertType.PERFORMANCE_DEGRADATION:
            # Check performance metrics
            return await self._check_performance_degradation(rule)
        
        elif rule.alert_type == AlertType.ANOMALY_SPIKE:
            # Check anomaly detection spike
            return await self._check_anomaly_spike(rule)
        
        return False
    
    async def _check_job_failure_rate(self, rule: AlertRule) -> bool:
        """Check job failure rate."""
        # This would query your metrics backend
        # For now, we'll simulate with a simple check
        return False
    
    async def _check_error_rate(self, rule: AlertRule) -> bool:
        """Check error rate."""
        # This would query your metrics backend
        return False
    
    async def _check_memory_usage(self, rule: AlertRule) -> bool:
        """Check memory usage."""
        # This would query your metrics backend
        return False
    
    async def _check_performance_degradation(self, rule: AlertRule) -> bool:
        """Check performance degradation."""
        # This would query your metrics backend
        return False
    
    async def _check_anomaly_spike(self, rule: AlertRule) -> bool:
        """Check anomaly spike."""
        # This would query your metrics backend
        return False
    
    async def _check_sla_violations(self) -> None:
        """Check SLA violations."""
        current_time = datetime.now()
        
        for sla_name, sla_config in self.sla_configs.items():
            # Check if SLA is violated
            if await self._is_sla_violated(sla_config):
                # Record SLA violation
                self.metrics_collector.increment_sla_violations(
                    "processing",
                    sla_name,
                    "medium"
                )
                
                # Create SLA breach alert
                rule = AlertRule(
                    name=f"sla_breach_{sla_name}",
                    description=f"SLA breach for {sla_name}",
                    alert_type=AlertType.SLA_BREACH,
                    severity=AlertSeverity.HIGH,
                    condition=f"{sla_config['metric']} > {sla_config['threshold']}",
                    threshold=sla_config['threshold']
                )
                
                await self._trigger_alert(rule)
    
    async def _is_sla_violated(self, sla_config: Dict[str, Any]) -> bool:
        """Check if SLA is violated."""
        # This would check against your metrics
        return False
    
    async def _trigger_alert(self, rule: AlertRule) -> None:
        """Trigger an alert."""
        alert_id = f"{rule.name}_{int(time.time())}"
        
        # Check cooldown
        if await self._is_in_cooldown(rule):
            return
        
        # Create alert
        alert = Alert(
            id=alert_id,
            rule=rule,
            triggered_at=datetime.now(),
            message=f"Alert triggered: {rule.description}"
        )
        
        self.active_alerts[alert_id] = alert
        self.alert_history.append(alert)
        
        # Record alert metrics
        self.metrics_collector.increment_alert_count(
            rule.alert_type.value,
            rule.severity.value,
            "alerting_system"
        )\n        
        # Send notifications
        await self._send_notifications(alert)
        
        logger.warning(f"Alert triggered: {rule.name} - {rule.description}")
    
    async def _resolve_alert(self, rule_name: str) -> None:
        """Resolve an alert."""
        alerts_to_resolve = [
            alert for alert in self.active_alerts.values()
            if alert.rule.name == rule_name and not alert.is_resolved
        ]
        
        for alert in alerts_to_resolve:
            alert.resolved_at = datetime.now()
            del self.active_alerts[alert.id]
            logger.info(f"Alert resolved: {alert.rule.name}")
    
    async def _is_in_cooldown(self, rule: AlertRule) -> bool:
        """Check if rule is in cooldown period."""
        if rule.cooldown_seconds == 0:
            return False
        
        cutoff_time = datetime.now() - timedelta(seconds=rule.cooldown_seconds)
        
        for alert in self.alert_history:
            if (alert.rule.name == rule.name and 
                alert.triggered_at > cutoff_time):
                return True
        
        return False
    
    async def _send_notifications(self, alert: Alert) -> None:
        """Send notifications for an alert."""
        for config in self.notification_configs:
            if not config.enabled:
                continue
            
            # Check if alert matches notification criteria
            if not self._matches_notification_criteria(alert, config):
                continue
            
            try:
                if config.type == NotificationType.EMAIL:
                    await self._send_email_notification(alert, config)
                elif config.type == NotificationType.WEBHOOK:
                    await self._send_webhook_notification(alert, config)
                elif config.type == NotificationType.SLACK:
                    await self._send_slack_notification(alert, config)
                elif config.type == NotificationType.LOG:
                    await self._send_log_notification(alert, config)
                
                # Record notification metrics
                self.metrics_collector.increment_notification_count(
                    config.type.value,
                    config.config.get("destination", "unknown"),
                    "success"
                )
                
            except Exception as e:
                logger.error(f"Failed to send {config.type.value} notification: {e}")
                self.metrics_collector.increment_notification_count(
                    config.type.value,
                    config.config.get("destination", "unknown"),
                    "failed"
                )
    
    def _matches_notification_criteria(self, alert: Alert, config: NotificationConfig) -> bool:
        """Check if alert matches notification criteria."""
        # Check severity
        severity_levels = {
            AlertSeverity.LOW: 1,
            AlertSeverity.MEDIUM: 2,
            AlertSeverity.HIGH: 3,
            AlertSeverity.CRITICAL: 4
        }
        
        if severity_levels[alert.rule.severity] < severity_levels[config.min_severity]:
            return False
        
        # Check alert types
        if config.alert_types and alert.rule.alert_type not in config.alert_types:
            return False
        
        # Check tags
        if config.tags:
            if not any(tag in alert.rule.tags for tag in config.tags):
                return False
        
        return True
    
    async def _send_email_notification(self, alert: Alert, config: NotificationConfig) -> None:
        """Send email notification."""
        email_config = config.config
        
        msg = MIMEMultipart()
        msg['From'] = email_config['from']
        msg['To'] = email_config['to']
        msg['Subject'] = f"[{alert.rule.severity.value.upper()}] {alert.rule.name}"
        
        body = f"""
        Alert: {alert.rule.name}
        Severity: {alert.rule.severity.value}
        Description: {alert.rule.description}
        Triggered at: {alert.triggered_at}
        
        Message: {alert.message}
        
        Context: {json.dumps(alert.context, indent=2)}
        """
        
        msg.attach(MIMEText(body, 'plain'))
        
        server = smtplib.SMTP(email_config['smtp_server'], email_config['smtp_port'])
        if email_config.get('use_tls', True):
            server.starttls()
        if email_config.get('username') and email_config.get('password'):
            server.login(email_config['username'], email_config['password'])
        
        server.send_message(msg)
        server.quit()
    
    async def _send_webhook_notification(self, alert: Alert, config: NotificationConfig) -> None:
        """Send webhook notification."""
        if not REQUESTS_AVAILABLE:
            raise ImportError("requests library required for webhook notifications")
        
        webhook_config = config.config
        
        payload = {
            "alert_id": alert.id,
            "rule_name": alert.rule.name,
            "severity": alert.rule.severity.value,
            "alert_type": alert.rule.alert_type.value,
            "description": alert.rule.description,
            "triggered_at": alert.triggered_at.isoformat(),
            "message": alert.message,
            "context": alert.context
        }
        
        headers = webhook_config.get('headers', {})
        headers['Content-Type'] = 'application/json'
        
        response = requests.post(
            webhook_config['url'],
            json=payload,
            headers=headers,
            timeout=webhook_config.get('timeout', 30)
        )
        
        response.raise_for_status()
    
    async def _send_slack_notification(self, alert: Alert, config: NotificationConfig) -> None:
        """Send Slack notification."""
        if not REQUESTS_AVAILABLE:
            raise ImportError("requests library required for Slack notifications")
        
        slack_config = config.config
        
        color_map = {
            AlertSeverity.LOW: "good",
            AlertSeverity.MEDIUM: "warning",
            AlertSeverity.HIGH: "danger",
            AlertSeverity.CRITICAL: "danger"
        }
        
        payload = {
            "attachments": [
                {
                    "color": color_map.get(alert.rule.severity, "warning"),
                    "title": f"Alert: {alert.rule.name}",
                    "text": alert.rule.description,
                    "fields": [
                        {"title": "Severity", "value": alert.rule.severity.value, "short": True},
                        {"title": "Type", "value": alert.rule.alert_type.value, "short": True},
                        {"title": "Triggered", "value": alert.triggered_at.isoformat(), "short": True},
                        {"title": "Message", "value": alert.message, "short": False}
                    ],
                    "footer": "Pynomaly Alerting System",
                    "ts": int(alert.triggered_at.timestamp())
                }
            ]
        }
        
        response = requests.post(
            slack_config['webhook_url'],
            json=payload,
            timeout=slack_config.get('timeout', 30)
        )
        
        response.raise_for_status()
    
    async def _send_log_notification(self, alert: Alert, config: NotificationConfig) -> None:
        """Send log notification."""
        log_config = config.config
        log_level = log_config.get('level', 'WARNING')
        
        message = f"ALERT [{alert.rule.severity.value.upper()}] {alert.rule.name}: {alert.message}"
        
        if log_level.upper() == 'ERROR':
            logger.error(message)
        elif log_level.upper() == 'WARNING':
            logger.warning(message)
        elif log_level.upper() == 'INFO':
            logger.info(message)
        else:
            logger.debug(message)
    
    async def _cleanup_old_alerts(self) -> None:
        """Clean up old alerts from history."""
        cutoff_time = datetime.now() - timedelta(days=7)  # Keep 7 days of history
        
        self.alert_history = [
            alert for alert in self.alert_history
            if alert.triggered_at > cutoff_time
        ]
    
    def process_log_entry(self, log_entry: str) -> None:
        """Process a log entry for pattern matching."""
        for pattern_name, pattern_config in self.log_patterns.items():
            if pattern_config["pattern"].search(log_entry):
                current_time = datetime.now()
                
                # Add match to window
                pattern_config["matches"].append(current_time)
                
                # Clean old matches outside window
                window_start = current_time - timedelta(seconds=pattern_config["window_seconds"])
                pattern_config["matches"] = [
                    match_time for match_time in pattern_config["matches"]
                    if match_time > window_start
                ]
                
                # Check if threshold is exceeded
                if len(pattern_config["matches"]) >= pattern_config["threshold"]:
                    # Create alert rule and trigger
                    rule = AlertRule(
                        name=f"log_pattern_{pattern_name}",
                        description=f"Log pattern {pattern_name} threshold exceeded",
                        alert_type=AlertType.HIGH_ERROR_RATE,
                        severity=pattern_config["severity"],
                        condition=f"log_pattern_{pattern_name} >= {pattern_config['threshold']}",
                        threshold=pattern_config["threshold"]
                    )
                    
                    # Trigger alert asynchronously
                    asyncio.create_task(self._trigger_alert(rule))
    
    def get_active_alerts(self) -> List[Alert]:
        """Get all active alerts."""
        return list(self.active_alerts.values())
    
    def get_alert_history(self, hours: int = 24) -> List[Alert]:
        """Get alert history for the specified number of hours."""
        cutoff_time = datetime.now() - timedelta(hours=hours)
        return [
            alert for alert in self.alert_history
            if alert.triggered_at > cutoff_time
        ]
    
    def get_alert_statistics(self) -> Dict[str, Any]:
        """Get alert statistics."""
        active_count = len(self.active_alerts)
        
        # Count by severity
        severity_counts = {}
        for alert in self.active_alerts.values():
            severity = alert.rule.severity.value
            severity_counts[severity] = severity_counts.get(severity, 0) + 1
        
        # Count by type
        type_counts = {}
        for alert in self.active_alerts.values():
            alert_type = alert.rule.alert_type.value
            type_counts[alert_type] = type_counts.get(alert_type, 0) + 1
        
        return {
            "active_alerts": active_count,
            "total_rules": len(self.rules),
            "enabled_rules": len([r for r in self.rules.values() if r.enabled]),
            "severity_counts": severity_counts,
            "type_counts": type_counts,
            "notification_configs": len(self.notification_configs)
        }


# Global alerting system instance
_alerting_system: Optional[AlertingSystem] = None


def get_alerting_system() -> AlertingSystem:
    """Get the global alerting system instance."""
    global _alerting_system
    if _alerting_system is None:
        _alerting_system = AlertingSystem()
    return _alerting_system


def setup_default_alerts() -> None:
    """Setup default alert rules."""
    alerting_system = get_alerting_system()
    
    # Job failure rate alert
    alerting_system.add_rule(AlertRule(
        name="high_job_failure_rate",
        description="High job failure rate detected",
        alert_type=AlertType.JOB_FAILURE,
        severity=AlertSeverity.HIGH,
        condition="job_failure_rate > 0.1",
        threshold=0.1,
        duration_seconds=300
    ))
    
    # Memory usage alert
    alerting_system.add_rule(AlertRule(
        name="high_memory_usage",
        description="High memory usage detected",
        alert_type=AlertType.MEMORY_USAGE,
        severity=AlertSeverity.MEDIUM,
        condition="memory_usage > 0.8",
        threshold=0.8,
        duration_seconds=600
    ))
    
    # Error rate alert
    alerting_system.add_rule(AlertRule(
        name="high_error_rate",
        description="High error rate detected",
        alert_type=AlertType.HIGH_ERROR_RATE,
        severity=AlertSeverity.MEDIUM,
        condition="error_rate > 0.05",
        threshold=0.05,
        duration_seconds=300
    ))
    
    # Add common log patterns
    alerting_system.add_log_pattern(
        "error_pattern",
        r"ERROR|FATAL|CRITICAL",
        AlertSeverity.MEDIUM,
        threshold=5,
        window_seconds=300
    )
    
    alerting_system.add_log_pattern(
        "memory_error_pattern",
        r"OutOfMemoryError|MemoryError|out of memory",
        AlertSeverity.HIGH,
        threshold=1,
        window_seconds=60
    )
    
    logger.info("Default alerts setup completed")
