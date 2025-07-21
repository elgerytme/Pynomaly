"""Advanced monitoring and alerting system for anomaly detection."""

from __future__ import annotations

import time
import json
from typing import Any, Dict, List, Optional, Callable, Union
from dataclasses import dataclass, field, asdict
from datetime import datetime, timedelta
from collections import deque
from enum import Enum
import threading
import warnings

from simplified_services.core_detection_service import DetectionResult


class AlertSeverity(Enum):
    """Alert severity levels."""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


class AlertStatus(Enum):
    """Alert status types."""
    ACTIVE = "active"
    ACKNOWLEDGED = "acknowledged"
    RESOLVED = "resolved"
    SUPPRESSED = "suppressed"


@dataclass
class Alert:
    """Anomaly detection alert."""
    alert_id: str
    severity: AlertSeverity
    title: str
    description: str
    timestamp: float
    source: str
    detection_result: Optional[DetectionResult] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    status: AlertStatus = AlertStatus.ACTIVE
    acknowledged_by: Optional[str] = None
    acknowledged_at: Optional[float] = None
    resolved_at: Optional[float] = None
    suppressed_until: Optional[float] = None


@dataclass
class MonitoringMetrics:
    """System monitoring metrics."""
    total_samples_processed: int = 0
    total_anomalies_detected: int = 0
    anomaly_rate: float = 0.0
    processing_rate_per_second: float = 0.0
    average_processing_time: float = 0.0
    active_alerts: int = 0
    last_detection_time: Optional[float] = None
    uptime_seconds: float = 0.0
    error_count: int = 0
    
    # Performance metrics
    memory_usage_mb: float = 0.0
    cpu_usage_percent: float = 0.0
    
    # Detection algorithm metrics
    algorithm_performance: Dict[str, Dict[str, float]] = field(default_factory=dict)
    
    # Time series metrics (last 24 hours)
    hourly_anomaly_counts: List[int] = field(default_factory=lambda: [0] * 24)
    hourly_processing_counts: List[int] = field(default_factory=lambda: [0] * 24)


@dataclass
class AlertRule:
    """Configuration for alert rules."""
    rule_id: str
    name: str
    description: str
    condition: str  # e.g., "anomaly_rate > 0.15"
    severity: AlertSeverity
    enabled: bool = True
    threshold_value: float = 0.0
    time_window_minutes: int = 5
    min_samples: int = 10
    cooldown_minutes: int = 30  # Prevent spam alerts
    notification_channels: List[str] = field(default_factory=list)


class MonitoringAlertingSystem:
    """Advanced monitoring and alerting system.
    
    This system provides comprehensive monitoring and alerting capabilities:
    - Real-time metrics collection and tracking
    - Configurable alert rules and thresholds
    - Alert lifecycle management (creation, acknowledgment, resolution)
    - Performance monitoring and trend analysis
    - Notification routing and escalation
    - Alert suppression and cooldown mechanisms
    """
    
    def __init__(self, retention_hours: int = 24):
        """Initialize monitoring and alerting system.
        
        Args:
            retention_hours: How long to retain historical data
        """
        self.retention_hours = retention_hours
        self.start_time = time.time()
        
        # Core data structures
        self.metrics = MonitoringMetrics()
        self.active_alerts: Dict[str, Alert] = {}
        self.alert_history: List[Alert] = []
        self.alert_rules: Dict[str, AlertRule] = {}
        
        # Time series data (using deques for efficiency)
        self.processing_times: deque = deque(maxlen=1000)
        self.anomaly_rates: deque = deque(maxlen=1000)
        self.timestamps: deque = deque(maxlen=1000)
        
        # Notification handlers
        self.notification_handlers: Dict[str, Callable] = {}
        
        # Background monitoring
        self._monitoring_thread: Optional[threading.Thread] = None
        self._stop_monitoring = False
        
        # Initialize default alert rules
        self._setup_default_alert_rules()
    
    def record_detection_result(
        self,
        detection_result: DetectionResult,
        processing_time: float,
        source: str = "unknown"
    ) -> None:
        """Record a detection result and update metrics.
        
        Args:
            detection_result: The detection result to record
            processing_time: Time taken for detection in seconds
            source: Source identifier for the detection
        """
        current_time = time.time()
        
        # Update basic metrics
        self.metrics.total_samples_processed += detection_result.n_samples
        self.metrics.total_anomalies_detected += detection_result.n_anomalies
        self.metrics.last_detection_time = current_time
        
        # Calculate rates
        if self.metrics.total_samples_processed > 0:
            self.metrics.anomaly_rate = (
                self.metrics.total_anomalies_detected / self.metrics.total_samples_processed
            )
        
        # Update processing time metrics
        self.processing_times.append(processing_time)
        if self.processing_times:
            self.metrics.average_processing_time = sum(self.processing_times) / len(self.processing_times)
        
        # Update time series data
        self.timestamps.append(current_time)
        current_anomaly_rate = detection_result.n_anomalies / detection_result.n_samples if detection_result.n_samples > 0 else 0
        self.anomaly_rates.append(current_anomaly_rate)
        
        # Update hourly counters
        hour = int((current_time % 86400) // 3600)  # Current hour of day
        self.metrics.hourly_anomaly_counts[hour] += detection_result.n_anomalies
        self.metrics.hourly_processing_counts[hour] += detection_result.n_samples
        
        # Update algorithm performance
        algo = detection_result.algorithm
        if algo not in self.metrics.algorithm_performance:
            self.metrics.algorithm_performance[algo] = {
                "total_samples": 0,
                "total_anomalies": 0,
                "avg_processing_time": 0.0,
                "anomaly_rate": 0.0
            }
        
        algo_stats = self.metrics.algorithm_performance[algo]
        algo_stats["total_samples"] += detection_result.n_samples
        algo_stats["total_anomalies"] += detection_result.n_anomalies
        algo_stats["anomaly_rate"] = algo_stats["total_anomalies"] / algo_stats["total_samples"]
        
        # Update uptime
        self.metrics.uptime_seconds = current_time - self.start_time
        
        # Check alert rules
        self._check_alert_rules(detection_result, source)
        
        print(f"📊 Metrics updated: {detection_result.n_anomalies}/{detection_result.n_samples} anomalies, rate: {self.metrics.anomaly_rate:.3f}")
    
    def add_alert_rule(self, rule: AlertRule) -> None:
        """Add or update an alert rule.
        
        Args:
            rule: Alert rule configuration
        """
        self.alert_rules[rule.rule_id] = rule
        print(f"🚨 Alert rule added: {rule.name}")
    
    def create_alert(
        self,
        severity: AlertSeverity,
        title: str,
        description: str,
        source: str,
        detection_result: Optional[DetectionResult] = None,
        metadata: Optional[Dict[str, Any]] = None
    ) -> str:
        """Create a new alert.
        
        Args:
            severity: Alert severity level
            title: Alert title
            description: Detailed description
            source: Source that triggered the alert
            detection_result: Related detection result
            metadata: Additional metadata
            
        Returns:
            Alert ID
        """
        alert_id = f"alert_{int(time.time())}_{len(self.active_alerts)}"
        
        alert = Alert(
            alert_id=alert_id,
            severity=severity,
            title=title,
            description=description,
            timestamp=time.time(),
            source=source,
            detection_result=detection_result,
            metadata=metadata or {}
        )
        
        self.active_alerts[alert_id] = alert
        self.alert_history.append(alert)
        self.metrics.active_alerts = len(self.active_alerts)
        
        # Send notifications
        self._send_notifications(alert)
        
        print(f"🚨 Alert created: {title} ({severity.value.upper()})")
        return alert_id
    
    def acknowledge_alert(self, alert_id: str, acknowledged_by: str) -> bool:
        """Acknowledge an alert.
        
        Args:
            alert_id: Alert identifier
            acknowledged_by: Who acknowledged the alert
            
        Returns:
            True if successful
        """
        if alert_id in self.active_alerts:
            alert = self.active_alerts[alert_id]
            alert.status = AlertStatus.ACKNOWLEDGED
            alert.acknowledged_by = acknowledged_by
            alert.acknowledged_at = time.time()
            
            print(f"✅ Alert acknowledged: {alert_id} by {acknowledged_by}")
            return True
        
        return False
    
    def resolve_alert(self, alert_id: str) -> bool:
        """Resolve an alert.
        
        Args:
            alert_id: Alert identifier
            
        Returns:
            True if successful
        """
        if alert_id in self.active_alerts:
            alert = self.active_alerts[alert_id]
            alert.status = AlertStatus.RESOLVED
            alert.resolved_at = time.time()
            
            # Remove from active alerts
            del self.active_alerts[alert_id]
            self.metrics.active_alerts = len(self.active_alerts)
            
            print(f"✅ Alert resolved: {alert_id}")
            return True
        
        return False
    
    def suppress_alert(self, alert_id: str, duration_minutes: int) -> bool:
        """Suppress an alert for a specified duration.
        
        Args:
            alert_id: Alert identifier
            duration_minutes: Suppression duration in minutes
            
        Returns:
            True if successful
        """
        if alert_id in self.active_alerts:
            alert = self.active_alerts[alert_id]
            alert.status = AlertStatus.SUPPRESSED
            alert.suppressed_until = time.time() + (duration_minutes * 60)
            
            print(f"🔇 Alert suppressed: {alert_id} for {duration_minutes} minutes")
            return True
        
        return False
    
    def get_active_alerts(
        self,
        severity: Optional[AlertSeverity] = None,
        source: Optional[str] = None
    ) -> List[Alert]:
        """Get active alerts with optional filtering.
        
        Args:
            severity: Filter by severity
            source: Filter by source
            
        Returns:
            List of matching active alerts
        """
        alerts = list(self.active_alerts.values())
        
        if severity:
            alerts = [a for a in alerts if a.severity == severity]
        
        if source:
            alerts = [a for a in alerts if a.source == source]
        
        # Remove suppressed alerts that have expired
        current_time = time.time()
        for alert in alerts[:]:  # Copy list to avoid modification during iteration
            if (alert.status == AlertStatus.SUPPRESSED and 
                alert.suppressed_until and 
                current_time > alert.suppressed_until):
                alert.status = AlertStatus.ACTIVE
                alert.suppressed_until = None
        
        return alerts
    
    def get_alert_history(
        self,
        hours: int = 24,
        severity: Optional[AlertSeverity] = None
    ) -> List[Alert]:
        """Get alert history for specified time period.
        
        Args:
            hours: Number of hours to look back
            severity: Filter by severity
            
        Returns:
            List of historical alerts
        """
        cutoff_time = time.time() - (hours * 3600)
        history = [a for a in self.alert_history if a.timestamp >= cutoff_time]
        
        if severity:
            history = [a for a in history if a.severity == severity]
        
        return sorted(history, key=lambda a: a.timestamp, reverse=True)
    
    def get_current_metrics(self) -> MonitoringMetrics:
        """Get current monitoring metrics."""
        # Update real-time metrics
        if self.timestamps:
            recent_timestamps = [t for t in self.timestamps if time.time() - t <= 60]
            if recent_timestamps:
                self.metrics.processing_rate_per_second = len(recent_timestamps) / 60
        
        return self.metrics
    
    def get_performance_summary(self) -> Dict[str, Any]:
        """Get performance summary statistics."""
        current_time = time.time()
        
        return {
            "uptime_hours": (current_time - self.start_time) / 3600,
            "total_samples": self.metrics.total_samples_processed,
            "total_anomalies": self.metrics.total_anomalies_detected,
            "overall_anomaly_rate": self.metrics.anomaly_rate,
            "avg_processing_time_ms": self.metrics.average_processing_time * 1000,
            "processing_rate_per_second": self.metrics.processing_rate_per_second,
            "active_alerts": self.metrics.active_alerts,
            "error_count": self.metrics.error_count,
            "algorithm_performance": self.metrics.algorithm_performance,
            "last_detection": datetime.fromtimestamp(self.metrics.last_detection_time).isoformat() if self.metrics.last_detection_time else None
        }
    
    def register_notification_handler(
        self,
        channel: str,
        handler: Callable[[Alert], None]
    ) -> None:
        """Register a notification handler for alerts.
        
        Args:
            channel: Notification channel name
            handler: Function to handle notifications
        """
        self.notification_handlers[channel] = handler
        print(f"📢 Notification handler registered: {channel}")
    
    def start_background_monitoring(self) -> None:
        """Start background monitoring thread."""
        if self._monitoring_thread is not None:
            return
        
        self._stop_monitoring = False
        self._monitoring_thread = threading.Thread(target=self._background_monitor, daemon=True)
        self._monitoring_thread.start()
        print("🔄 Background monitoring started")
    
    def stop_background_monitoring(self) -> None:
        """Stop background monitoring thread."""
        if self._monitoring_thread is None:
            return
        
        self._stop_monitoring = True
        self._monitoring_thread.join(timeout=5.0)
        self._monitoring_thread = None
        print("⏹️  Background monitoring stopped")
    
    def export_metrics(self, output_path: str) -> None:
        """Export metrics to JSON file.
        
        Args:
            output_path: Path to save metrics
        """
        export_data = {
            "timestamp": time.time(),
            "metrics": asdict(self.metrics),
            "active_alerts": [asdict(alert) for alert in self.active_alerts.values()],
            "recent_history": [asdict(alert) for alert in self.get_alert_history(hours=1)],
            "performance_summary": self.get_performance_summary()
        }
        
        # Convert numpy arrays and enums to serializable formats
        export_data = self._make_serializable(export_data)
        
        with open(output_path, 'w') as f:
            json.dump(export_data, f, indent=2)
        
        print(f"📊 Metrics exported to: {output_path}")
    
    def _setup_default_alert_rules(self) -> None:
        """Setup default alert rules."""
        default_rules = [
            AlertRule(
                rule_id="high_anomaly_rate",
                name="High Anomaly Rate",
                description="Alert when anomaly rate exceeds threshold",
                condition="anomaly_rate > 0.15",
                severity=AlertSeverity.HIGH,
                threshold_value=0.15,
                time_window_minutes=5,
                cooldown_minutes=30
            ),
            AlertRule(
                rule_id="critical_anomaly_rate",
                name="Critical Anomaly Rate",
                description="Alert when anomaly rate is critically high",
                condition="anomaly_rate > 0.25",
                severity=AlertSeverity.CRITICAL,
                threshold_value=0.25,
                time_window_minutes=3,
                cooldown_minutes=15
            ),
            AlertRule(
                rule_id="slow_processing",
                name="Slow Processing",
                description="Alert when processing time is too slow",
                condition="avg_processing_time > 5.0",
                severity=AlertSeverity.MEDIUM,
                threshold_value=5.0,
                time_window_minutes=10,
                cooldown_minutes=60
            ),
            AlertRule(
                rule_id="no_recent_data",
                name="No Recent Data",
                description="Alert when no data processed recently",
                condition="time_since_last_detection > 3600",
                severity=AlertSeverity.MEDIUM,
                threshold_value=3600,
                time_window_minutes=60,
                cooldown_minutes=30
            )
        ]
        
        for rule in default_rules:
            self.add_alert_rule(rule)
    
    def _check_alert_rules(
        self,
        detection_result: DetectionResult,
        source: str
    ) -> None:
        """Check if any alert rules are triggered."""
        current_time = time.time()
        
        for rule in self.alert_rules.values():
            if not rule.enabled:
                continue
            
            # Check cooldown period
            cooldown_key = f"{rule.rule_id}_{source}"
            if hasattr(self, '_rule_cooldowns') and cooldown_key in self._rule_cooldowns:
                if current_time < self._rule_cooldowns[cooldown_key]:
                    continue
            else:
                if not hasattr(self, '_rule_cooldowns'):
                    self._rule_cooldowns = {}
            
            # Evaluate rule condition
            triggered = self._evaluate_rule_condition(rule, detection_result)
            
            if triggered:
                # Create alert
                alert_id = self.create_alert(
                    severity=rule.severity,
                    title=rule.name,
                    description=f"{rule.description}. Current value exceeds threshold of {rule.threshold_value}",
                    source=source,
                    detection_result=detection_result,
                    metadata={"rule_id": rule.rule_id, "threshold": rule.threshold_value}
                )
                
                # Set cooldown
                self._rule_cooldowns[cooldown_key] = current_time + (rule.cooldown_minutes * 60)
    
    def _evaluate_rule_condition(
        self,
        rule: AlertRule,
        detection_result: DetectionResult
    ) -> bool:
        """Evaluate if a rule condition is met."""
        try:
            # Create evaluation context
            context = {
                "anomaly_rate": self.metrics.anomaly_rate,
                "avg_processing_time": self.metrics.average_processing_time,
                "active_alerts": self.metrics.active_alerts,
                "current_anomaly_rate": detection_result.n_anomalies / detection_result.n_samples if detection_result.n_samples > 0 else 0,
                "time_since_last_detection": time.time() - self.metrics.last_detection_time if self.metrics.last_detection_time else float('inf')
            }
            
            # Simple condition evaluation
            if rule.rule_id == "high_anomaly_rate":
                return context["anomaly_rate"] > rule.threshold_value
            elif rule.rule_id == "critical_anomaly_rate":
                return context["anomaly_rate"] > rule.threshold_value
            elif rule.rule_id == "slow_processing":
                return context["avg_processing_time"] > rule.threshold_value
            elif rule.rule_id == "no_recent_data":
                return context["time_since_last_detection"] > rule.threshold_value
            
            return False
            
        except Exception as e:
            print(f"Error evaluating rule {rule.rule_id}: {e}")
            return False
    
    def _send_notifications(self, alert: Alert) -> None:
        """Send notifications for an alert."""
        for rule_id, rule in self.alert_rules.items():
            if (hasattr(rule, 'notification_channels') and 
                rule.rule_id == alert.metadata.get('rule_id')):
                
                for channel in rule.notification_channels:
                    if channel in self.notification_handlers:
                        try:
                            self.notification_handlers[channel](alert)
                        except Exception as e:
                            print(f"Error sending notification to {channel}: {e}")
    
    def _background_monitor(self) -> None:
        """Background monitoring thread."""
        while not self._stop_monitoring:
            try:
                # Update system metrics (simplified)
                self._update_system_metrics()
                
                # Clean up old data
                self._cleanup_old_data()
                
                # Check for stale alerts
                self._check_stale_alerts()
                
                time.sleep(60)  # Run every minute
                
            except Exception as e:
                print(f"Background monitoring error: {e}")
                time.sleep(60)
    
    def _update_system_metrics(self) -> None:
        """Update system performance metrics."""
        # In a real implementation, these would get actual system metrics
        self.metrics.memory_usage_mb = 100.0  # Placeholder
        self.metrics.cpu_usage_percent = 25.0  # Placeholder
    
    def _cleanup_old_data(self) -> None:
        """Clean up old historical data."""
        cutoff_time = time.time() - (self.retention_hours * 3600)
        
        # Clean up alert history
        self.alert_history = [a for a in self.alert_history if a.timestamp >= cutoff_time]
        
        # Clean up time series data
        while self.timestamps and self.timestamps[0] < cutoff_time:
            self.timestamps.popleft()
            if self.anomaly_rates:
                self.anomaly_rates.popleft()
            if self.processing_times:
                self.processing_times.popleft()
    
    def _check_stale_alerts(self) -> None:
        """Check for alerts that should be auto-resolved."""
        current_time = time.time()
        stale_alerts = []
        
        for alert_id, alert in self.active_alerts.items():
            # Auto-resolve very old alerts (24 hours)
            if current_time - alert.timestamp > 86400:
                stale_alerts.append(alert_id)
        
        for alert_id in stale_alerts:
            self.resolve_alert(alert_id)
            print(f"🕐 Auto-resolved stale alert: {alert_id}")
    
    def _make_serializable(self, obj: Any) -> Any:
        """Make object JSON serializable."""
        if isinstance(obj, dict):
            return {k: self._make_serializable(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [self._make_serializable(v) for v in obj]
        elif isinstance(obj, (AlertSeverity, AlertStatus)):
            return obj.value
        elif hasattr(obj, 'tolist'):  # NumPy array
            return obj.tolist()
        else:
            return obj


# Example notification handlers
def console_notification_handler(alert: Alert) -> None:
    """Simple console notification handler."""
    severity_emoji = {
        AlertSeverity.LOW: "ℹ️",
        AlertSeverity.MEDIUM: "⚠️",
        AlertSeverity.HIGH: "🚨",
        AlertSeverity.CRITICAL: "🔥"
    }
    
    emoji = severity_emoji.get(alert.severity, "📢")
    timestamp = datetime.fromtimestamp(alert.timestamp).strftime("%Y-%m-%d %H:%M:%S")
    
    print(f"{emoji} ALERT [{alert.severity.value.upper()}] {timestamp}")
    print(f"   Title: {alert.title}")
    print(f"   Source: {alert.source}")
    print(f"   Description: {alert.description}")
    if alert.detection_result:
        print(f"   Anomalies: {alert.detection_result.n_anomalies}/{alert.detection_result.n_samples}")


def email_notification_handler(alert: Alert) -> None:
    """Email notification handler (mock implementation)."""
    print(f"📧 EMAIL ALERT: {alert.title}")
    print(f"   To: admin@example.com")
    print(f"   Subject: [{alert.severity.value.upper()}] Anomaly Detection Alert")
    print(f"   Body: {alert.description}")


def slack_notification_handler(alert: Alert) -> None:
    """Slack notification handler (mock implementation)."""
    print(f"💬 SLACK ALERT: #{alert.severity.value}-alerts")
    print(f"   Message: {alert.title} - {alert.description}")
    if alert.detection_result:
        print(f"   Details: {alert.detection_result.n_anomalies} anomalies detected")