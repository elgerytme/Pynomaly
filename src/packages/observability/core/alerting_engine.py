"""
Intelligent Alerting Engine

Provides advanced alerting capabilities with anomaly detection,
smart routing, escalation policies, and multi-channel notifications.
"""

import asyncio
import logging
import smtplib
import json
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional, Union, Callable, Set
from dataclasses import dataclass, field
from enum import Enum
from email.mime.text import MimeText
from email.mime.multipart import MimeMultipart

import numpy as np
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler
import aiohttp
import slack_sdk

logger = logging.getLogger(__name__)


class AlertSeverity(Enum):
    """Alert severity levels."""
    CRITICAL = "critical"
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"
    INFO = "info"


class AlertStatus(Enum):
    """Alert status states."""
    ACTIVE = "active"
    ACKNOWLEDGED = "acknowledged"
    RESOLVED = "resolved"
    SUPPRESSED = "suppressed"


class NotificationChannel(Enum):
    """Supported notification channels."""
    EMAIL = "email"
    SLACK = "slack"
    WEBHOOK = "webhook"
    SMS = "sms"
    PAGERDUTY = "pagerduty"


class AnomalyMethod(Enum):
    """Anomaly detection methods."""
    ISOLATION_FOREST = "isolation_forest"
    STATISTICAL_THRESHOLD = "statistical_threshold"
    MOVING_AVERAGE = "moving_average"
    SEASONAL_DECOMPOSITION = "seasonal_decomposition"


@dataclass
class AlertCondition:
    """Alert condition configuration."""
    metric_name: str
    operator: str  # >, <, >=, <=, ==, !=, anomaly
    threshold: Union[float, str]
    evaluation_window: timedelta
    comparison_type: str = "current"  # current, average, sum, min, max
    anomaly_method: Optional[AnomalyMethod] = None
    sensitivity: float = 0.95  # For anomaly detection


@dataclass
class EscalationPolicy:
    """Escalation policy configuration."""
    id: str
    name: str
    levels: List[Dict[str, Any]]  # Each level has delay and notifications
    repeat_interval: Optional[timedelta] = None
    max_escalations: int = 3


@dataclass
class NotificationConfig:
    """Notification configuration."""
    channel: NotificationChannel
    config: Dict[str, Any]  # Channel-specific configuration
    enabled: bool = True


@dataclass
class Alert:
    """Alert instance."""
    id: str
    rule_id: str
    severity: AlertSeverity
    status: AlertStatus
    title: str
    description: str
    metric_name: str
    current_value: Union[float, str]
    threshold: Union[float, str]
    triggered_at: datetime
    acknowledged_at: Optional[datetime] = None
    resolved_at: Optional[datetime] = None
    escalation_level: int = 0
    last_notification: Optional[datetime] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    tags: Set[str] = field(default_factory=set)


@dataclass
class AlertRule:
    """Alert rule configuration."""
    id: str
    name: str
    description: str
    conditions: List[AlertCondition]
    severity: AlertSeverity
    escalation_policy_id: Optional[str] = None
    notification_configs: List[NotificationConfig] = field(default_factory=list)
    enabled: bool = True
    tags: Set[str] = field(default_factory=set)
    suppression_window: Optional[timedelta] = None
    auto_resolve_after: Optional[timedelta] = None
    dependencies: List[str] = field(default_factory=list)  # Other rule IDs


class AnomalyDetector:
    """Advanced anomaly detection for metrics."""
    
    def __init__(self, method: AnomalyMethod = AnomalyMethod.ISOLATION_FOREST):
        self.method = method
        self.models: Dict[str, Any] = {}
        self.scalers: Dict[str, StandardScaler] = {}
        self.history: Dict[str, List[float]] = {}
        self.window_size = 100  # Number of data points for training
    
    def fit(self, metric_name: str, data: List[float]) -> None:
        """Fit anomaly detection model."""
        if len(data) < 10:  # Need minimum data points
            return
        
        self.history[metric_name] = data[-self.window_size:]
        
        if self.method == AnomalyMethod.ISOLATION_FOREST:
            # Prepare data
            X = np.array(data).reshape(-1, 1)
            
            # Scale data
            scaler = StandardScaler()
            X_scaled = scaler.fit_transform(X)
            self.scalers[metric_name] = scaler
            
            # Train isolation forest
            model = IsolationForest(contamination=0.1, random_state=42)
            model.fit(X_scaled)
            self.models[metric_name] = model
            
        elif self.method == AnomalyMethod.STATISTICAL_THRESHOLD:
            # Calculate statistical thresholds
            mean_val = np.mean(data)
            std_val = np.std(data)
            
            self.models[metric_name] = {
                "mean": mean_val,
                "std": std_val,
                "upper_threshold": mean_val + (3 * std_val),
                "lower_threshold": mean_val - (3 * std_val)
            }
    
    def detect(self, metric_name: str, value: float, sensitivity: float = 0.95) -> bool:
        """Detect if value is anomalous."""
        if metric_name not in self.models:
            return False
        
        if self.method == AnomalyMethod.ISOLATION_FOREST:
            model = self.models[metric_name]
            scaler = self.scalers[metric_name]
            
            # Scale the value
            X = np.array([[value]])
            X_scaled = scaler.transform(X)
            
            # Predict anomaly (-1 for anomaly, 1 for normal)
            prediction = model.predict(X_scaled)[0]
            anomaly_score = model.decision_function(X_scaled)[0]
            
            # Adjust threshold based on sensitivity
            threshold = np.percentile(
                [model.decision_function(scaler.transform([[h]])[0]) for h in self.history[metric_name]],
                (1 - sensitivity) * 100
            )
            
            return anomaly_score < threshold
            
        elif self.method == AnomalyMethod.STATISTICAL_THRESHOLD:
            thresholds = self.models[metric_name]
            return value > thresholds["upper_threshold"] or value < thresholds["lower_threshold"]
        
        return False
    
    def update(self, metric_name: str, value: float) -> None:
        """Update model with new data point."""
        if metric_name not in self.history:
            self.history[metric_name] = []
        
        self.history[metric_name].append(value)
        
        # Keep only recent data
        if len(self.history[metric_name]) > self.window_size:
            self.history[metric_name] = self.history[metric_name][-self.window_size:]
        
        # Retrain if enough data
        if len(self.history[metric_name]) >= 20:
            self.fit(metric_name, self.history[metric_name])


class AlertingEngine:
    """
    Comprehensive alerting engine with anomaly detection,
    intelligent routing, and multi-channel notifications.
    """
    
    def __init__(
        self,
        smtp_host: Optional[str] = None,
        smtp_port: int = 587,
        smtp_username: Optional[str] = None,
        smtp_password: Optional[str] = None,
        slack_token: Optional[str] = None,
        enable_anomaly_detection: bool = True
    ):
        # SMTP configuration
        self.smtp_host = smtp_host
        self.smtp_port = smtp_port
        self.smtp_username = smtp_username
        self.smtp_password = smtp_password
        
        # Slack configuration
        self.slack_client = None
        if slack_token:
            self.slack_client = slack_sdk.WebClient(token=slack_token)
        
        # Storage
        self.alert_rules: Dict[str, AlertRule] = {}
        self.escalation_policies: Dict[str, EscalationPolicy] = {}
        self.active_alerts: Dict[str, Alert] = {}
        self.alert_history: List[Alert] = []
        
        # Anomaly detection
        self.anomaly_detector = AnomalyDetector() if enable_anomaly_detection else None
        
        # Notification handlers
        self.notification_handlers: Dict[NotificationChannel, Callable] = {
            NotificationChannel.EMAIL: self._send_email_notification,
            NotificationChannel.SLACK: self._send_slack_notification,
            NotificationChannel.WEBHOOK: self._send_webhook_notification
        }
        
        # Metrics tracking
        self.metrics_cache: Dict[str, List[Dict[str, Any]]] = {}
        
        # Background tasks
        self.running = False
        self.evaluation_task = None
        self.escalation_task = None
    
    async def start(self) -> None:
        """Start the alerting engine."""
        self.running = True
        
        # Start background tasks
        self.evaluation_task = asyncio.create_task(self._evaluation_loop())
        self.escalation_task = asyncio.create_task(self._escalation_loop())
        
        logger.info("Alerting engine started")
    
    async def stop(self) -> None:
        """Stop the alerting engine."""
        self.running = False
        
        if self.evaluation_task:
            self.evaluation_task.cancel()
        
        if self.escalation_task:
            self.escalation_task.cancel()
        
        logger.info("Alerting engine stopped")
    
    def add_alert_rule(self, rule: AlertRule) -> None:
        """Add an alert rule."""
        self.alert_rules[rule.id] = rule
        logger.info(f"Added alert rule: {rule.name}")
    
    def remove_alert_rule(self, rule_id: str) -> bool:
        """Remove an alert rule."""
        if rule_id in self.alert_rules:
            del self.alert_rules[rule_id]
            logger.info(f"Removed alert rule: {rule_id}")
            return True
        return False
    
    def add_escalation_policy(self, policy: EscalationPolicy) -> None:
        """Add an escalation policy."""
        self.escalation_policies[policy.id] = policy
        logger.info(f"Added escalation policy: {policy.name}")
    
    async def update_metric(self, metric_name: str, value: float, timestamp: Optional[datetime] = None) -> None:
        """Update metric value and check for alerts."""
        timestamp = timestamp or datetime.utcnow()
        
        # Store metric value
        if metric_name not in self.metrics_cache:
            self.metrics_cache[metric_name] = []
        
        self.metrics_cache[metric_name].append({
            "value": value,
            "timestamp": timestamp
        })
        
        # Keep only recent data (last 1000 points)
        if len(self.metrics_cache[metric_name]) > 1000:
            self.metrics_cache[metric_name] = self.metrics_cache[metric_name][-1000:]
        
        # Update anomaly detector
        if self.anomaly_detector:
            self.anomaly_detector.update(metric_name, value)
        
        # Check alert conditions
        await self._check_alert_conditions(metric_name, value, timestamp)
    
    async def _check_alert_conditions(self, metric_name: str, value: float, timestamp: datetime) -> None:
        """Check if metric value triggers any alert conditions."""
        for rule in self.alert_rules.values():
            if not rule.enabled:
                continue
            
            # Check if any condition matches this metric
            for condition in rule.conditions:
                if condition.metric_name != metric_name:
                    continue
                
                # Check if condition is met
                if await self._evaluate_condition(condition, value, timestamp):
                    await self._trigger_alert(rule, condition, value, timestamp)
    
    async def _evaluate_condition(self, condition: AlertCondition, value: float, timestamp: datetime) -> bool:
        """Evaluate if a condition is met."""
        if condition.operator == "anomaly":
            # Use anomaly detection
            if not self.anomaly_detector:
                return False
            
            return self.anomaly_detector.detect(
                condition.metric_name,
                value,
                condition.sensitivity
            )
        
        # Get comparison value based on evaluation window
        comparison_value = await self._get_comparison_value(condition, timestamp)
        if comparison_value is None:
            return False
        
        # Evaluate condition
        if condition.operator == ">":
            return comparison_value > float(condition.threshold)
        elif condition.operator == "<":
            return comparison_value < float(condition.threshold)
        elif condition.operator == ">=":
            return comparison_value >= float(condition.threshold)
        elif condition.operator == "<=":
            return comparison_value <= float(condition.threshold)
        elif condition.operator == "==":
            return comparison_value == float(condition.threshold)
        elif condition.operator == "!=":
            return comparison_value != float(condition.threshold)
        
        return False
    
    async def _get_comparison_value(self, condition: AlertCondition, timestamp: datetime) -> Optional[float]:
        """Get comparison value for condition evaluation."""
        metric_name = condition.metric_name
        
        if metric_name not in self.metrics_cache:
            return None
        
        # Get data within evaluation window
        start_time = timestamp - condition.evaluation_window
        relevant_data = [
            point["value"] for point in self.metrics_cache[metric_name]
            if start_time <= point["timestamp"] <= timestamp
        ]
        
        if not relevant_data:
            return None
        
        # Calculate comparison value based on type
        if condition.comparison_type == "current":
            return relevant_data[-1]
        elif condition.comparison_type == "average":
            return sum(relevant_data) / len(relevant_data)
        elif condition.comparison_type == "sum":
            return sum(relevant_data)
        elif condition.comparison_type == "min":
            return min(relevant_data)
        elif condition.comparison_type == "max":
            return max(relevant_data)
        
        return relevant_data[-1]  # Default to current
    
    async def _trigger_alert(self, rule: AlertRule, condition: AlertCondition, value: float, timestamp: datetime) -> None:
        """Trigger an alert."""
        # Check if alert already exists and is active
        existing_alert = None
        for alert in self.active_alerts.values():
            if alert.rule_id == rule.id and alert.metric_name == condition.metric_name:
                existing_alert = alert
                break
        
        if existing_alert:
            # Update existing alert
            existing_alert.current_value = value
            existing_alert.triggered_at = timestamp
            return
        
        # Check dependencies
        if rule.dependencies and not await self._check_dependencies(rule.dependencies):
            logger.info(f"Alert {rule.name} suppressed due to dependencies")
            return
        
        # Check suppression window
        if rule.suppression_window and await self._is_suppressed(rule, timestamp):
            logger.info(f"Alert {rule.name} suppressed due to suppression window")
            return
        
        # Create new alert
        alert = Alert(
            id=f"alert_{len(self.active_alerts)}_{timestamp.strftime('%Y%m%d%H%M%S')}",
            rule_id=rule.id,
            severity=rule.severity,
            status=AlertStatus.ACTIVE,
            title=rule.name,
            description=rule.description,
            metric_name=condition.metric_name,
            current_value=value,
            threshold=condition.threshold,
            triggered_at=timestamp,
            tags=rule.tags.copy()
        )
        
        # Store alert
        self.active_alerts[alert.id] = alert
        self.alert_history.append(alert)
        
        # Send notifications
        await self._send_notifications(alert, rule)
        
        logger.warning(f"Alert triggered: {rule.name} (value: {value}, threshold: {condition.threshold})")
    
    async def _check_dependencies(self, dependencies: List[str]) -> bool:
        """Check if dependent alerts are active."""
        for dep_rule_id in dependencies:
            # Check if any alert from the dependent rule is active
            for alert in self.active_alerts.values():
                if alert.rule_id == dep_rule_id and alert.status == AlertStatus.ACTIVE:
                    return False  # Dependency is active, suppress this alert
        return True
    
    async def _is_suppressed(self, rule: AlertRule, timestamp: datetime) -> bool:
        """Check if alert is within suppression window."""
        if not rule.suppression_window:
            return False
        
        # Check recent alerts for this rule
        suppression_start = timestamp - rule.suppression_window
        for alert in self.alert_history:
            if (alert.rule_id == rule.id and 
                alert.triggered_at >= suppression_start and 
                alert.status == AlertStatus.RESOLVED):
                return True
        
        return False
    
    async def _send_notifications(self, alert: Alert, rule: AlertRule) -> None:
        """Send notifications for an alert."""
        for notification_config in rule.notification_configs:
            if not notification_config.enabled:
                continue
            
            try:
                handler = self.notification_handlers.get(notification_config.channel)
                if handler:
                    await handler(alert, notification_config.config)
                else:
                    logger.error(f"No handler for notification channel: {notification_config.channel}")
                    
            except Exception as e:
                logger.error(f"Failed to send notification via {notification_config.channel}: {e}")
    
    async def _send_email_notification(self, alert: Alert, config: Dict[str, Any]) -> None:
        """Send email notification."""
        if not self.smtp_host:
            logger.warning("SMTP not configured, cannot send email notification")
            return
        
        recipients = config.get("recipients", [])
        if not recipients:
            return
        
        # Create email message
        msg = MimeMultipart()
        msg["From"] = config.get("from", self.smtp_username)
        msg["To"] = ", ".join(recipients)
        msg["Subject"] = f"[{alert.severity.value.upper()}] {alert.title}"
        
        # Email body
        body = f"""
Alert: {alert.title}
Severity: {alert.severity.value.upper()}
Description: {alert.description}
Metric: {alert.metric_name}
Current Value: {alert.current_value}
Threshold: {alert.threshold}
Triggered At: {alert.triggered_at.isoformat()}

Tags: {', '.join(alert.tags)}
        """
        
        msg.attach(MimeText(body, "plain"))
        
        # Send email
        try:
            with smtplib.SMTP(self.smtp_host, self.smtp_port) as server:
                if self.smtp_username and self.smtp_password:
                    server.starttls()
                    server.login(self.smtp_username, self.smtp_password)
                
                server.send_message(msg)
                
        except Exception as e:
            logger.error(f"Failed to send email notification: {e}")
    
    async def _send_slack_notification(self, alert: Alert, config: Dict[str, Any]) -> None:
        """Send Slack notification."""
        if not self.slack_client:
            logger.warning("Slack not configured, cannot send notification")
            return
        
        channel = config.get("channel", "#alerts")
        
        # Create Slack message
        color = {
            AlertSeverity.CRITICAL: "danger",
            AlertSeverity.HIGH: "danger",
            AlertSeverity.MEDIUM: "warning",
            AlertSeverity.LOW: "good",
            AlertSeverity.INFO: "good"
        }.get(alert.severity, "warning")
        
        attachment = {
            "color": color,
            "title": f"[{alert.severity.value.upper()}] {alert.title}",
            "text": alert.description,
            "fields": [
                {"title": "Metric", "value": alert.metric_name, "short": True},
                {"title": "Current Value", "value": str(alert.current_value), "short": True},
                {"title": "Threshold", "value": str(alert.threshold), "short": True},
                {"title": "Triggered At", "value": alert.triggered_at.isoformat(), "short": True}
            ],
            "footer": f"Tags: {', '.join(alert.tags)}"
        }
        
        try:
            response = self.slack_client.chat_postMessage(
                channel=channel,
                text=f"Alert: {alert.title}",
                attachments=[attachment]
            )
            
            if not response["ok"]:
                logger.error(f"Failed to send Slack notification: {response['error']}")
                
        except Exception as e:
            logger.error(f"Failed to send Slack notification: {e}")
    
    async def _send_webhook_notification(self, alert: Alert, config: Dict[str, Any]) -> None:
        """Send webhook notification."""
        url = config.get("url")
        if not url:
            return
        
        # Prepare payload
        payload = {
            "alert_id": alert.id,
            "severity": alert.severity.value,
            "status": alert.status.value,
            "title": alert.title,
            "description": alert.description,
            "metric_name": alert.metric_name,
            "current_value": alert.current_value,
            "threshold": alert.threshold,
            "triggered_at": alert.triggered_at.isoformat(),
            "tags": list(alert.tags)
        }
        
        headers = config.get("headers", {"Content-Type": "application/json"})
        
        try:
            async with aiohttp.ClientSession() as session:
                async with session.post(url, json=payload, headers=headers) as response:
                    if response.status >= 400:
                        logger.error(f"Webhook notification failed: {response.status}")
                        
        except Exception as e:
            logger.error(f"Failed to send webhook notification: {e}")
    
    async def _evaluation_loop(self) -> None:
        """Background loop for alert evaluation."""
        while self.running:
            try:
                # Auto-resolve alerts
                await self._auto_resolve_alerts()
                
                # Check for stale metrics
                await self._check_stale_metrics()
                
                await asyncio.sleep(30)  # Check every 30 seconds
                
            except Exception as e:
                logger.error(f"Error in evaluation loop: {e}")
                await asyncio.sleep(30)
    
    async def _escalation_loop(self) -> None:
        """Background loop for alert escalation."""
        while self.running:
            try:
                await self._process_escalations()
                await asyncio.sleep(60)  # Check every minute
                
            except Exception as e:
                logger.error(f"Error in escalation loop: {e}")
                await asyncio.sleep(60)
    
    async def _auto_resolve_alerts(self) -> None:
        """Auto-resolve alerts based on rules."""
        current_time = datetime.utcnow()
        
        for alert_id, alert in list(self.active_alerts.items()):
            if alert.status != AlertStatus.ACTIVE:
                continue
            
            rule = self.alert_rules.get(alert.rule_id)
            if not rule or not rule.auto_resolve_after:
                continue
            
            # Check if alert should auto-resolve
            if current_time - alert.triggered_at >= rule.auto_resolve_after:
                await self.resolve_alert(alert_id, "system", "Auto-resolved after timeout")
    
    async def _check_stale_metrics(self) -> None:
        """Check for stale metrics and alert if necessary."""
        current_time = datetime.utcnow()
        stale_threshold = timedelta(minutes=5)
        
        for metric_name, data_points in self.metrics_cache.items():
            if not data_points:
                continue
            
            last_update = data_points[-1]["timestamp"]
            if current_time - last_update > stale_threshold:
                # Create stale metric alert
                await self._create_stale_metric_alert(metric_name, last_update)
    
    async def _create_stale_metric_alert(self, metric_name: str, last_update: datetime) -> None:
        """Create alert for stale metric."""
        alert_id = f"stale_metric_{metric_name}_{datetime.utcnow().strftime('%Y%m%d%H%M%S')}"
        
        # Check if stale metric alert already exists
        for alert in self.active_alerts.values():
            if f"stale_metric_{metric_name}" in alert.id:
                return  # Already have a stale metric alert
        
        alert = Alert(
            id=alert_id,
            rule_id="stale_metric_rule",
            severity=AlertSeverity.MEDIUM,
            status=AlertStatus.ACTIVE,
            title=f"Stale Metric: {metric_name}",
            description=f"Metric {metric_name} has not been updated since {last_update.isoformat()}",
            metric_name=metric_name,
            current_value="stale",
            threshold="5 minutes",
            triggered_at=datetime.utcnow(),
            tags={"stale_metric", "monitoring"}
        )
        
        self.active_alerts[alert_id] = alert
        self.alert_history.append(alert)
        
        logger.warning(f"Stale metric alert: {metric_name}")
    
    async def _process_escalations(self) -> None:
        """Process alert escalations."""
        current_time = datetime.utcnow()
        
        for alert in self.active_alerts.values():
            if alert.status != AlertStatus.ACTIVE:
                continue
            
            rule = self.alert_rules.get(alert.rule_id)
            if not rule or not rule.escalation_policy_id:
                continue
            
            policy = self.escalation_policies.get(rule.escalation_policy_id)
            if not policy:
                continue
            
            # Check if escalation is needed
            await self._check_escalation(alert, policy)
    
    async def _check_escalation(self, alert: Alert, policy: EscalationPolicy) -> None:
        """Check if alert needs escalation."""
        current_time = datetime.utcnow()
        
        # Get current escalation level
        if alert.escalation_level >= len(policy.levels):
            return  # Already at max escalation
        
        level_config = policy.levels[alert.escalation_level]
        escalation_delay = timedelta(seconds=level_config.get("delay_seconds", 300))  # Default 5 minutes
        
        # Check if enough time has passed for escalation
        last_notification_time = alert.last_notification or alert.triggered_at
        if current_time - last_notification_time >= escalation_delay:
            # Escalate alert
            alert.escalation_level += 1
            alert.last_notification = current_time
            
            # Send escalation notifications
            notifications = level_config.get("notifications", [])
            for notification in notifications:
                await self._send_escalation_notification(alert, notification)
            
            logger.warning(f"Escalated alert {alert.id} to level {alert.escalation_level}")
    
    async def _send_escalation_notification(self, alert: Alert, notification: Dict[str, Any]) -> None:
        """Send escalation notification."""
        channel = notification.get("channel")
        config = notification.get("config", {})
        
        # Add escalation context to alert description
        original_description = alert.description
        alert.description = f"[ESCALATION LEVEL {alert.escalation_level}] {original_description}"
        
        # Send notification
        handler = self.notification_handlers.get(NotificationChannel(channel))
        if handler:
            try:
                await handler(alert, config)
            except Exception as e:
                logger.error(f"Failed to send escalation notification: {e}")
        
        # Restore original description
        alert.description = original_description
    
    async def acknowledge_alert(self, alert_id: str, acknowledged_by: str, note: str = "") -> bool:
        """Acknowledge an alert."""
        alert = self.active_alerts.get(alert_id)
        if not alert:
            return False
        
        alert.status = AlertStatus.ACKNOWLEDGED
        alert.acknowledged_at = datetime.utcnow()
        alert.metadata["acknowledged_by"] = acknowledged_by
        alert.metadata["acknowledgment_note"] = note
        
        logger.info(f"Alert {alert_id} acknowledged by {acknowledged_by}")
        return True
    
    async def resolve_alert(self, alert_id: str, resolved_by: str, note: str = "") -> bool:
        """Resolve an alert."""
        alert = self.active_alerts.get(alert_id)
        if not alert:
            return False
        
        alert.status = AlertStatus.RESOLVED
        alert.resolved_at = datetime.utcnow()
        alert.metadata["resolved_by"] = resolved_by
        alert.metadata["resolution_note"] = note
        
        # Remove from active alerts
        del self.active_alerts[alert_id]
        
        logger.info(f"Alert {alert_id} resolved by {resolved_by}")
        return True
    
    def get_active_alerts(self, severity: Optional[AlertSeverity] = None) -> List[Alert]:
        """Get active alerts, optionally filtered by severity."""
        alerts = list(self.active_alerts.values())
        
        if severity:
            alerts = [alert for alert in alerts if alert.severity == severity]
        
        return sorted(alerts, key=lambda x: x.triggered_at, reverse=True)
    
    def get_alert_stats(self) -> Dict[str, Any]:
        """Get alerting statistics."""
        active_alerts = list(self.active_alerts.values())
        
        stats = {
            "total_rules": len(self.alert_rules),
            "active_alerts": len(active_alerts),
            "alerts_by_severity": {},
            "escalation_policies": len(self.escalation_policies),
            "total_alert_history": len(self.alert_history),
            "anomaly_detection_enabled": self.anomaly_detector is not None
        }
        
        # Count alerts by severity
        for alert in active_alerts:
            severity = alert.severity.value
            stats["alerts_by_severity"][severity] = stats["alerts_by_severity"].get(severity, 0) + 1
        
        return stats


# Predefined alert rules for common scenarios
DEFAULT_ALERT_RULES = [
    AlertRule(
        id="high_cpu_usage",
        name="High CPU Usage",
        description="CPU usage is above 80%",
        conditions=[
            AlertCondition(
                metric_name="system_cpu_usage_percent",
                operator=">",
                threshold=80.0,
                evaluation_window=timedelta(minutes=5),
                comparison_type="average"
            )
        ],
        severity=AlertSeverity.HIGH,
        tags={"system", "cpu"}
    ),
    AlertRule(
        id="high_error_rate",
        name="High API Error Rate",
        description="API error rate is above 5%",
        conditions=[
            AlertCondition(
                metric_name="api_error_rate_percent",
                operator=">",
                threshold=5.0,
                evaluation_window=timedelta(minutes=5),
                comparison_type="average"
            )
        ],
        severity=AlertSeverity.CRITICAL,
        tags={"api", "errors"}
    ),
    AlertRule(
        id="model_accuracy_drop",
        name="Model Accuracy Drop",
        description="Model accuracy has dropped below acceptable threshold",
        conditions=[
            AlertCondition(
                metric_name="ml_model_accuracy_score",
                operator="<",
                threshold=0.8,
                evaluation_window=timedelta(minutes=30),
                comparison_type="current"
            )
        ],
        severity=AlertSeverity.HIGH,
        tags={"ml", "accuracy"}
    ),
    AlertRule(
        id="data_drift_detected",
        name="Data Drift Detected",
        description="Significant data drift detected in model inputs",
        conditions=[
            AlertCondition(
                metric_name="ml_model_drift_score",
                operator="anomaly",
                threshold="anomaly",
                evaluation_window=timedelta(hours=1),
                anomaly_method=AnomalyMethod.ISOLATION_FOREST,
                sensitivity=0.95
            )
        ],
        severity=AlertSeverity.MEDIUM,
        tags={"ml", "drift"}
    )
]