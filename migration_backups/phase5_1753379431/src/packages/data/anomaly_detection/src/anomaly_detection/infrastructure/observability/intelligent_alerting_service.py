"""Intelligent alerting system with ML-based anomaly detection on metrics."""

import logging
import json
import uuid
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Union, Callable, Tuple
from dataclasses import dataclass, asdict
from enum import Enum
import numpy as np
import pandas as pd
from abc import ABC, abstractmethod
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler
import asyncio
import threading
import time

from anomaly_detection.domain.services.detection_service import DetectionService


class AlertSeverity(Enum):
    """Alert severity levels."""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


class AlertStatus(Enum):
    """Alert status."""
    ACTIVE = "active"
    ACKNOWLEDGED = "acknowledged"
    RESOLVED = "resolved"
    SUPPRESSED = "suppressed"


class MetricType(Enum):
    """Types of metrics to monitor."""
    COUNTER = "counter"
    GAUGE = "gauge"
    HISTOGRAM = "histogram"
    SUMMARY = "summary"


@dataclass
class MetricPoint:
    """A single metric data point."""
    timestamp: datetime
    value: float
    labels: Dict[str, str]
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary format."""
        return {
            "timestamp": self.timestamp.isoformat(),
            "value": self.value,
            "labels": self.labels
        }


@dataclass
class AlertRule:
    """Configuration for an alert rule."""
    rule_id: str
    name: str
    description: str
    metric_name: str
    metric_labels: Dict[str, str]
    severity: AlertSeverity
    threshold_type: str  # "static", "dynamic", "ml_based"
    threshold_value: Optional[float] = None
    threshold_percentile: Optional[float] = None
    window_duration: str = "5m"  # e.g., "5m", "1h", "1d"
    evaluation_interval: str = "1m"
    ml_model_params: Dict[str, Any] = None
    notification_channels: List[str] = None
    suppress_duration: str = "10m"
    enabled: bool = True
    created_at: datetime = None
    updated_at: datetime = None
    
    def __post_init__(self):
        if self.ml_model_params is None:
            self.ml_model_params = {}
        if self.notification_channels is None:
            self.notification_channels = []
        if self.created_at is None:
            self.created_at = datetime.now()
        if self.updated_at is None:
            self.updated_at = datetime.now()
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary format."""
        return {
            "rule_id": self.rule_id,
            "name": self.name,
            "description": self.description,
            "metric_name": self.metric_name,
            "metric_labels": self.metric_labels,
            "severity": self.severity.value,
            "threshold_type": self.threshold_type,
            "threshold_value": self.threshold_value,
            "threshold_percentile": self.threshold_percentile,
            "window_duration": self.window_duration,
            "evaluation_interval": self.evaluation_interval,
            "ml_model_params": self.ml_model_params,
            "notification_channels": self.notification_channels,
            "suppress_duration": self.suppress_duration,
            "enabled": self.enabled,
            "created_at": self.created_at.isoformat(),
            "updated_at": self.updated_at.isoformat()
        }


@dataclass
class Alert:
    """An alert instance."""
    alert_id: str
    rule_id: str
    rule_name: str
    severity: AlertSeverity
    status: AlertStatus
    message: str
    metric_name: str
    metric_labels: Dict[str, str]
    triggered_at: datetime
    resolved_at: Optional[datetime] = None
    acknowledged_at: Optional[datetime] = None
    acknowledged_by: Optional[str] = None
    current_value: Optional[float] = None
    threshold_value: Optional[float] = None
    anomaly_score: Optional[float] = None
    context: Dict[str, Any] = None
    
    def __post_init__(self):
        if self.context is None:
            self.context = {}
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary format."""
        return {
            "alert_id": self.alert_id,
            "rule_id": self.rule_id,
            "rule_name": self.rule_name,
            "severity": self.severity.value,
            "status": self.status.value,
            "message": self.message,
            "metric_name": self.metric_name,
            "metric_labels": self.metric_labels,
            "triggered_at": self.triggered_at.isoformat(),
            "resolved_at": self.resolved_at.isoformat() if self.resolved_at else None,
            "acknowledged_at": self.acknowledged_at.isoformat() if self.acknowledged_at else None,
            "acknowledged_by": self.acknowledged_by,
            "current_value": self.current_value,
            "threshold_value": self.threshold_value,
            "anomaly_score": self.anomaly_score,
            "context": self.context
        }


class NotificationChannel(ABC):
    """Abstract base class for notification channels."""
    
    @abstractmethod
    async def send_notification(self, alert: Alert) -> bool:
        """Send notification for an alert."""
        pass


class EmailNotificationChannel(NotificationChannel):
    """Email notification channel."""
    
    def __init__(self, smtp_config: Dict[str, Any]):
        """Initialize email notification channel.
        
        Args:
            smtp_config: SMTP configuration
        """
        self.smtp_config = smtp_config
        self.logger = logging.getLogger(__name__)
    
    async def send_notification(self, alert: Alert) -> bool:
        """Send email notification."""
        try:
            # Placeholder - implement actual email sending
            self.logger.info(f"Email notification sent for alert {alert.alert_id}: {alert.message}")
            return True
        except Exception as e:
            self.logger.error(f"Failed to send email notification: {e}")
            return False


class SlackNotificationChannel(NotificationChannel):
    """Slack notification channel."""
    
    def __init__(self, webhook_url: str):
        """Initialize Slack notification channel.
        
        Args:
            webhook_url: Slack webhook URL
        """
        self.webhook_url = webhook_url
        self.logger = logging.getLogger(__name__)
    
    async def send_notification(self, alert: Alert) -> bool:
        """Send Slack notification."""
        try:
            # Placeholder - implement actual Slack notification
            self.logger.info(f"Slack notification sent for alert {alert.alert_id}: {alert.message}")
            return True
        except Exception as e:
            self.logger.error(f"Failed to send Slack notification: {e}")
            return False


class PagerDutyNotificationChannel(NotificationChannel):
    """PagerDuty notification channel."""
    
    def __init__(self, integration_key: str):
        """Initialize PagerDuty notification channel.
        
        Args:
            integration_key: PagerDuty integration key
        """
        self.integration_key = integration_key
        self.logger = logging.getLogger(__name__)
    
    async def send_notification(self, alert: Alert) -> bool:
        """Send PagerDuty notification."""
        try:
            # Placeholder - implement actual PagerDuty notification
            self.logger.info(f"PagerDuty notification sent for alert {alert.alert_id}: {alert.message}")
            return True
        except Exception as e:
            self.logger.error(f"Failed to send PagerDuty notification: {e}")
            return False


class MetricAnomalyDetector:
    """ML-based anomaly detector for metrics."""
    
    def __init__(self, 
                 contamination: float = 0.1,
                 window_size: int = 100,
                 retrain_interval: int = 1000):
        """Initialize metric anomaly detector.
        
        Args:
            contamination: Expected proportion of anomalies
            window_size: Size of sliding window for training
            retrain_interval: How often to retrain model (in data points)
        """
        self.contamination = contamination
        self.window_size = window_size
        self.retrain_interval = retrain_interval
        self.logger = logging.getLogger(__name__)
        
        # Model components
        self.scaler = StandardScaler()
        self.model = IsolationForest(contamination=contamination, random_state=42)
        
        # Training data buffer
        self._training_buffer: List[float] = []
        self._last_retrain_count = 0
        self._is_trained = False
    
    def add_data_point(self, value: float, timestamp: datetime) -> Optional[Tuple[bool, float]]:
        """Add a data point and check for anomaly.
        
        Args:
            value: Metric value
            timestamp: Timestamp of the metric
            
        Returns:
            Tuple of (is_anomaly, anomaly_score) or None if not enough data
        """
        self._training_buffer.append(value)
        
        # Keep buffer at maximum window size
        if len(self._training_buffer) > self.window_size * 2:
            self._training_buffer = self._training_buffer[-self.window_size:]
        
        # Check if we should retrain
        if (len(self._training_buffer) - self._last_retrain_count >= self.retrain_interval or
            not self._is_trained):
            self._retrain_model()
        
        # Make prediction if model is trained
        if self._is_trained and len(self._training_buffer) >= 10:
            return self._predict_anomaly(value)
        
        return None
    
    def _retrain_model(self):
        """Retrain the anomaly detection model."""
        if len(self._training_buffer) < 10:
            return
        
        try:
            # Prepare training data with features
            features = self._extract_features(self._training_buffer)
            
            if len(features) < 5:
                return
            
            # Fit scaler and model
            features_scaled = self.scaler.fit_transform(features)
            self.model.fit(features_scaled)
            
            self._is_trained = True
            self._last_retrain_count = len(self._training_buffer)
            
            self.logger.debug(f"Retrained anomaly detection model with {len(features)} samples")
            
        except Exception as e:
            self.logger.error(f"Failed to retrain anomaly detection model: {e}")
    
    def _extract_features(self, data: List[float]) -> np.ndarray:
        """Extract features from time series data.
        
        Args:
            data: Time series data
            
        Returns:
            Feature matrix
        """
        if len(data) < 5:
            return np.array([])
        
        features = []
        
        for i in range(4, len(data)):
            # Use sliding window to create features
            window = data[i-4:i+1]  # 5-point window
            
            feature_vector = [
                np.mean(window),        # Mean
                np.std(window),         # Standard deviation
                np.min(window),         # Minimum
                np.max(window),         # Maximum
                window[-1] - window[0], # Trend (last - first)
                window[-1]              # Current value
            ]
            
            features.append(feature_vector)
        
        return np.array(features)
    
    def _predict_anomaly(self, value: float) -> Tuple[bool, float]:
        """Predict if current value is anomalous.
        
        Args:
            value: Current metric value
            
        Returns:
            Tuple of (is_anomaly, anomaly_score)
        """
        try:
            # Extract features for current value
            recent_data = self._training_buffer[-5:]  # Last 5 points including current
            if len(recent_data) < 5:
                return False, 0.0
            
            features = np.array([
                np.mean(recent_data),
                np.std(recent_data),
                np.min(recent_data),
                np.max(recent_data),
                recent_data[-1] - recent_data[0],
                recent_data[-1]
            ]).reshape(1, -1)
            
            # Scale features
            features_scaled = self.scaler.transform(features)
            
            # Predict anomaly
            prediction = self.model.predict(features_scaled)[0]
            anomaly_score = self.model.decision_function(features_scaled)[0]
            
            # Convert to boolean and normalize score
            is_anomaly = prediction == -1
            normalized_score = max(0, min(1, (0.5 - anomaly_score) * 2))  # Normalize to 0-1
            
            return is_anomaly, normalized_score
            
        except Exception as e:
            self.logger.error(f"Failed to predict anomaly: {e}")
            return False, 0.0


class IntelligentAlertingService:
    """Intelligent alerting service with ML-based anomaly detection."""
    
    def __init__(self):
        """Initialize intelligent alerting service."""
        self.logger = logging.getLogger(__name__)
        
        # Storage
        self._alert_rules: Dict[str, AlertRule] = {}
        self._active_alerts: Dict[str, Alert] = {}
        self._alert_history: List[Alert] = []
        self._notification_channels: Dict[str, NotificationChannel] = {}
        
        # Metric storage and anomaly detectors
        self._metrics_buffer: Dict[str, List[MetricPoint]] = {}
        self._anomaly_detectors: Dict[str, MetricAnomalyDetector] = {}
        
        # Background processing
        self._running = False
        self._evaluation_thread: Optional[threading.Thread] = None
        self._evaluation_interval = 60  # seconds
    
    def start(self):
        """Start the alerting service."""
        if self._running:
            return
        
        self._running = True
        self._evaluation_thread = threading.Thread(target=self._evaluation_loop, daemon=True)
        self._evaluation_thread.start()
        
        self.logger.info("Intelligent alerting service started")
    
    def stop(self):
        """Stop the alerting service."""
        self._running = False
        
        if self._evaluation_thread:
            self._evaluation_thread.join(timeout=5)
        
        self.logger.info("Intelligent alerting service stopped")
    
    def add_notification_channel(self, name: str, channel: NotificationChannel):
        """Add a notification channel.
        
        Args:
            name: Channel name
            channel: Notification channel instance
        """
        self._notification_channels[name] = channel
        self.logger.info(f"Added notification channel: {name}")
    
    def create_alert_rule(self, 
                         name: str,
                         description: str,
                         metric_name: str,
                         severity: AlertSeverity,
                         threshold_type: str = "static",
                         **kwargs) -> str:
        """Create a new alert rule.
        
        Args:
            name: Rule name
            description: Rule description
            metric_name: Name of metric to monitor
            severity: Alert severity
            threshold_type: Type of threshold ("static", "dynamic", "ml_based")
            **kwargs: Additional rule parameters
            
        Returns:
            Rule ID
        """
        rule_id = str(uuid.uuid4())
        
        alert_rule = AlertRule(
            rule_id=rule_id,
            name=name,
            description=description,
            metric_name=metric_name,
            severity=severity,
            threshold_type=threshold_type,
            metric_labels=kwargs.get('metric_labels', {}),
            **{k: v for k, v in kwargs.items() if k != 'metric_labels'}
        )
        
        self._alert_rules[rule_id] = alert_rule
        
        # Initialize anomaly detector if using ML-based threshold
        if threshold_type == "ml_based":
            detector_key = self._get_detector_key(metric_name, alert_rule.metric_labels)
            if detector_key not in self._anomaly_detectors:
                ml_params = alert_rule.ml_model_params
                self._anomaly_detectors[detector_key] = MetricAnomalyDetector(
                    contamination=ml_params.get('contamination', 0.1),
                    window_size=ml_params.get('window_size', 100),
                    retrain_interval=ml_params.get('retrain_interval', 1000)
                )
        
        self.logger.info(f"Created alert rule: {name} ({rule_id})")
        return rule_id
    
    def update_alert_rule(self, rule_id: str, **kwargs) -> bool:
        """Update an existing alert rule.
        
        Args:
            rule_id: Rule ID to update
            **kwargs: Fields to update
            
        Returns:
            True if updated successfully
        """
        if rule_id not in self._alert_rules:
            return False
        
        rule = self._alert_rules[rule_id]
        
        # Update fields
        for key, value in kwargs.items():
            if hasattr(rule, key):
                setattr(rule, key, value)
        
        rule.updated_at = datetime.now()
        
        self.logger.info(f"Updated alert rule: {rule.name}")
        return True
    
    def delete_alert_rule(self, rule_id: str) -> bool:
        """Delete an alert rule.
        
        Args:
            rule_id: Rule ID to delete
            
        Returns:
            True if deleted successfully
        """
        if rule_id not in self._alert_rules:
            return False
        
        rule = self._alert_rules[rule_id]
        del self._alert_rules[rule_id]
        
        # Resolve any active alerts for this rule
        alerts_to_resolve = [
            alert for alert in self._active_alerts.values()
            if alert.rule_id == rule_id
        ]
        
        for alert in alerts_to_resolve:
            self.resolve_alert(alert.alert_id, "Rule deleted")
        
        self.logger.info(f"Deleted alert rule: {rule.name}")
        return True
    
    def ingest_metric(self, 
                     metric_name: str,
                     value: float,
                     labels: Optional[Dict[str, str]] = None,
                     timestamp: Optional[datetime] = None):
        """Ingest a metric data point.
        
        Args:
            metric_name: Name of the metric
            value: Metric value
            labels: Optional metric labels
            timestamp: Optional timestamp (defaults to now)
        """
        if timestamp is None:
            timestamp = datetime.now()
        
        if labels is None:
            labels = {}
        
        metric_point = MetricPoint(
            timestamp=timestamp,
            value=value,
            labels=labels
        )
        
        # Store metric point
        metric_key = self._get_metric_key(metric_name, labels)
        if metric_key not in self._metrics_buffer:
            self._metrics_buffer[metric_key] = []
        
        self._metrics_buffer[metric_key].append(metric_point)
        
        # Keep buffer size reasonable
        max_buffer_size = 10000
        if len(self._metrics_buffer[metric_key]) > max_buffer_size:
            self._metrics_buffer[metric_key] = self._metrics_buffer[metric_key][-max_buffer_size//2:]
        
        # Update ML-based anomaly detectors
        detector_key = self._get_detector_key(metric_name, labels)
        if detector_key in self._anomaly_detectors:
            result = self._anomaly_detectors[detector_key].add_data_point(value, timestamp)
            
            # Store anomaly prediction for rule evaluation
            if result:
                is_anomaly, anomaly_score = result
                metric_point.anomaly_detected = is_anomaly
                metric_point.anomaly_score = anomaly_score
    
    def _get_metric_key(self, metric_name: str, labels: Dict[str, str]) -> str:
        """Generate a unique key for a metric with labels."""
        sorted_labels = sorted(labels.items())
        label_str = ",".join(f"{k}={v}" for k, v in sorted_labels)
        return f"{metric_name}[{label_str}]"
    
    def _get_detector_key(self, metric_name: str, labels: Dict[str, str]) -> str:
        """Generate a unique key for an anomaly detector."""
        return self._get_metric_key(metric_name, labels)
    
    def _evaluation_loop(self):
        """Background loop for evaluating alert rules."""
        while self._running:
            try:
                self._evaluate_alert_rules()
                time.sleep(self._evaluation_interval)
            except Exception as e:
                self.logger.error(f"Error in alert evaluation loop: {e}")
                time.sleep(self._evaluation_interval)
    
    def _evaluate_alert_rules(self):
        """Evaluate all active alert rules."""
        current_time = datetime.now()
        
        for rule in self._alert_rules.values():
            if not rule.enabled:
                continue
            
            try:
                self._evaluate_single_rule(rule, current_time)
            except Exception as e:
                self.logger.error(f"Error evaluating rule {rule.name}: {e}")
    
    def _evaluate_single_rule(self, rule: AlertRule, current_time: datetime):
        """Evaluate a single alert rule.
        
        Args:
            rule: Alert rule to evaluate
            current_time: Current timestamp
        """
        # Get relevant metrics
        metric_points = self._get_metric_points_for_rule(rule, current_time)
        
        if not metric_points:
            return
        
        # Check threshold based on type
        threshold_exceeded = False
        current_value = None
        threshold_value = None
        anomaly_score = None
        
        if rule.threshold_type == "static":
            current_value = metric_points[-1].value
            threshold_value = rule.threshold_value
            threshold_exceeded = current_value > threshold_value
            
        elif rule.threshold_type == "dynamic":
            # Use percentile-based threshold
            values = [p.value for p in metric_points]
            threshold_value = np.percentile(values, rule.threshold_percentile or 95)
            current_value = values[-1]
            threshold_exceeded = current_value > threshold_value
            
        elif rule.threshold_type == "ml_based":
            # Use ML-based anomaly detection
            latest_point = metric_points[-1]
            if hasattr(latest_point, 'anomaly_detected'):
                threshold_exceeded = latest_point.anomaly_detected
                current_value = latest_point.value
                anomaly_score = getattr(latest_point, 'anomaly_score', None)
        
        # Handle alert state changes
        existing_alert = self._find_active_alert_for_rule(rule.rule_id)
        
        if threshold_exceeded and not existing_alert:
            # Create new alert
            self._create_alert(rule, current_value, threshold_value, anomaly_score, current_time)
            
        elif not threshold_exceeded and existing_alert:
            # Resolve existing alert
            self.resolve_alert(existing_alert.alert_id, "Threshold no longer exceeded")
    
    def _get_metric_points_for_rule(self, rule: AlertRule, current_time: datetime) -> List[MetricPoint]:
        """Get metric points relevant to a rule.
        
        Args:
            rule: Alert rule
            current_time: Current timestamp
            
        Returns:
            List of relevant metric points
        """
        # Calculate time window
        window_duration = self._parse_duration(rule.window_duration)
        start_time = current_time - window_duration
        
        # Find matching metrics
        matching_points = []
        
        for metric_key, points in self._metrics_buffer.items():
            if not metric_key.startswith(rule.metric_name):
                continue
            
            # Check if labels match
            if not self._labels_match(rule.metric_labels, points[0].labels if points else {}):
                continue
            
            # Filter by time window
            for point in points:
                if start_time <= point.timestamp <= current_time:
                    matching_points.append(point)
        
        # Sort by timestamp
        matching_points.sort(key=lambda p: p.timestamp)
        return matching_points
    
    def _labels_match(self, rule_labels: Dict[str, str], metric_labels: Dict[str, str]) -> bool:
        """Check if metric labels match rule criteria.
        
        Args:
            rule_labels: Labels specified in the rule
            metric_labels: Labels from the metric
            
        Returns:
            True if labels match
        """
        for key, value in rule_labels.items():
            if key not in metric_labels or metric_labels[key] != value:
                return False
        return True
    
    def _parse_duration(self, duration_str: str) -> timedelta:
        """Parse duration string to timedelta.
        
        Args:
            duration_str: Duration string (e.g., "5m", "1h", "1d")
            
        Returns:
            timedelta object
        """
        if duration_str.endswith('s'):
            return timedelta(seconds=int(duration_str[:-1]))
        elif duration_str.endswith('m'):
            return timedelta(minutes=int(duration_str[:-1]))
        elif duration_str.endswith('h'):
            return timedelta(hours=int(duration_str[:-1]))
        elif duration_str.endswith('d'):
            return timedelta(days=int(duration_str[:-1]))
        else:
            # Default to minutes
            return timedelta(minutes=int(duration_str))
    
    def _find_active_alert_for_rule(self, rule_id: str) -> Optional[Alert]:
        """Find active alert for a rule.
        
        Args:
            rule_id: Rule ID
            
        Returns:
            Active alert or None
        """
        for alert in self._active_alerts.values():
            if alert.rule_id == rule_id and alert.status == AlertStatus.ACTIVE:
                return alert
        return None
    
    def _create_alert(self,
                     rule: AlertRule,
                     current_value: Optional[float],
                     threshold_value: Optional[float],
                     anomaly_score: Optional[float],
                     triggered_at: datetime):
        """Create a new alert.
        
        Args:
            rule: Alert rule that triggered
            current_value: Current metric value
            threshold_value: Threshold that was exceeded
            anomaly_score: Anomaly score if ML-based
            triggered_at: When the alert was triggered
        """
        alert_id = str(uuid.uuid4())
        
        # Generate alert message
        if rule.threshold_type == "ml_based":
            message = f"Anomaly detected in {rule.metric_name} (score: {anomaly_score:.3f})"
        else:
            message = f"{rule.metric_name} exceeded threshold: {current_value} > {threshold_value}"
        
        alert = Alert(
            alert_id=alert_id,
            rule_id=rule.rule_id,
            rule_name=rule.name,
            severity=rule.severity,
            status=AlertStatus.ACTIVE,
            message=message,
            metric_name=rule.metric_name,
            metric_labels=rule.metric_labels,
            triggered_at=triggered_at,
            current_value=current_value,
            threshold_value=threshold_value,
            anomaly_score=anomaly_score
        )
        
        self._active_alerts[alert_id] = alert
        self._alert_history.append(alert)
        
        # Send notifications
        asyncio.create_task(self._send_alert_notifications(alert, rule))
        
        self.logger.warning(f"Alert triggered: {alert.message}")
    
    async def _send_alert_notifications(self, alert: Alert, rule: AlertRule):
        """Send notifications for an alert.
        
        Args:
            alert: Alert to send notifications for
            rule: Alert rule configuration
        """
        for channel_name in rule.notification_channels:
            if channel_name in self._notification_channels:
                try:
                    await self._notification_channels[channel_name].send_notification(alert)
                except Exception as e:
                    self.logger.error(f"Failed to send notification via {channel_name}: {e}")
    
    def acknowledge_alert(self, alert_id: str, acknowledged_by: str) -> bool:
        """Acknowledge an alert.
        
        Args:
            alert_id: Alert ID
            acknowledged_by: User who acknowledged the alert
            
        Returns:
            True if acknowledged successfully
        """
        if alert_id not in self._active_alerts:
            return False
        
        alert = self._active_alerts[alert_id]
        alert.status = AlertStatus.ACKNOWLEDGED
        alert.acknowledged_at = datetime.now()
        alert.acknowledged_by = acknowledged_by
        
        self.logger.info(f"Alert {alert_id} acknowledged by {acknowledged_by}")
        return True
    
    def resolve_alert(self, alert_id: str, resolution_reason: str = "") -> bool:
        """Resolve an alert.
        
        Args:
            alert_id: Alert ID
            resolution_reason: Reason for resolution
            
        Returns:
            True if resolved successfully
        """
        if alert_id not in self._active_alerts:
            return False
        
        alert = self._active_alerts[alert_id]
        alert.status = AlertStatus.RESOLVED
        alert.resolved_at = datetime.now()
        
        if resolution_reason:
            alert.context["resolution_reason"] = resolution_reason
        
        # Move from active to history (it's already in history)
        del self._active_alerts[alert_id]
        
        self.logger.info(f"Alert {alert_id} resolved: {resolution_reason}")
        return True
    
    def get_active_alerts(self, severity: Optional[AlertSeverity] = None) -> List[Alert]:
        """Get active alerts.
        
        Args:
            severity: Optional severity filter
            
        Returns:
            List of active alerts
        """
        alerts = list(self._active_alerts.values())
        
        if severity:
            alerts = [a for a in alerts if a.severity == severity]
        
        return sorted(alerts, key=lambda a: a.triggered_at, reverse=True)
    
    def get_alert_statistics(self) -> Dict[str, Any]:
        """Get alert statistics.
        
        Returns:
            Dictionary with alert statistics
        """
        total_alerts = len(self._alert_history)
        active_alerts = len(self._active_alerts)
        
        # Count by severity
        severity_counts = {}
        for severity in AlertSeverity:
            count = len([a for a in self._active_alerts.values() if a.severity == severity])
            severity_counts[severity.value] = count
        
        # Alert frequency in last 24 hours
        last_24h = datetime.now() - timedelta(hours=24)
        recent_alerts = [a for a in self._alert_history if a.triggered_at >= last_24h]
        
        return {
            "total_alerts": total_alerts,
            "active_alerts": active_alerts,
            "alerts_by_severity": severity_counts,
            "alerts_last_24h": len(recent_alerts),
            "total_rules": len(self._alert_rules),
            "enabled_rules": len([r for r in self._alert_rules.values() if r.enabled]),
            "ml_based_rules": len([r for r in self._alert_rules.values() if r.threshold_type == "ml_based"])
        }
    
    def health_check(self) -> Dict[str, Any]:
        """Health check for alerting service.
        
        Returns:
            Health status information
        """
        return {
            "status": "healthy" if self._running else "stopped",
            "active_alerts": len(self._active_alerts),
            "total_rules": len(self._alert_rules),
            "metrics_tracked": len(self._metrics_buffer),
            "anomaly_detectors": len(self._anomaly_detectors),
            "notification_channels": len(self._notification_channels)
        }