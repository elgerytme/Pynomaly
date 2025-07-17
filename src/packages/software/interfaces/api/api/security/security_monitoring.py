"""
Security monitoring and logging for Software API.

This module provides:
- Security event logging
- Intrusion processing
- Anomaly processing for security events
- Real-time alerting
- Security measurements and reporting
"""

import json
import logging
import threading
import time
from collections import defaultdict, deque
from collections.abc import Callable
from dataclasses import dataclass
from datetime import datetime, timedelta
from enum import Enum
from typing import Any

logger = logging.getLogger(__name__)


class SecurityEventType(str, Enum):
    """Security event types."""

    AUTHENTICATION_FAILURE = "authentication_failure"
    AUTHENTICATION_SUCCESS = "authentication_success"
    AUTHORIZATION_FAILURE = "authorization_failure"
    RATE_LIMIT_EXCEEDED = "rate_limit_exceeded"
    SUSPICIOUS_REQUEST = "suspicious_request"
    SQL_INJECTION_ATTEMPT = "sql_injection_attempt"
    XSS_ATTEMPT = "xss_attempt"
    BRUTE_FORCE_ATTACK = "brute_force_attack"
    ACCOUNT_LOCKOUT = "account_lockout"
    PRIVILEGE_ESCALATION = "privilege_escalation"
    DATA_ACCESS_VIOLATION = "data_access_violation"
    MALICIOUS_FILE_UPLOAD = "malicious_file_upload"
    DDOS_ATTACK = "ddos_attack"
    SESSION_HIJACKING = "session_hijacking"
    API_ABUSE = "api_abuse"


class SecuritySeverity(str, Enum):
    """Security event severity levels."""

    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


@dataclass
class SecurityEvent:
    """Security event data structure."""

    event_id: str
    event_type: SecurityEventType
    severity: SecuritySeverity
    timestamp: datetime
    source_ip: str
    user_id: str | None
    endpoint: str
    user_agent: str
    details: dict[str, Any]
    risk_score: float
    mitigated: bool = False
    mitigation_actions: list[str] = None

    def __post_init__(self):
        if self.mitigation_actions is None:
            self.mitigation_actions = []


class SecurityMonitor:
    """Comprehensive security monitoring system."""

    def __init__(self):
        self.events = deque(maxlen=10000)  # Keep last 10k events
        self.event_counts = defaultdict(int)
        self.ip_events = defaultdict(list)
        self.user_events = defaultdict(list)
        self.alert_thresholds = self._initialize_thresholds()
        self.alert_callbacks = []
        self.monitoring_rules = []
        self.measurements = SecurityMetrics()
        self.lock = threading.Lock()

    def _initialize_thresholds(self) -> dict[SecurityEventType, dict[str, Any]]:
        """Initialize alert thresholds for different event types."""
        return {
            SecurityEventType.AUTHENTICATION_FAILURE: {
                "count": 5,
                "window": 300,  # 5 minutes
                "action": "alert",
            },
            SecurityEventType.RATE_LIMIT_EXCEEDED: {
                "count": 10,
                "window": 600,  # 10 minutes
                "action": "block_ip",
            },
            SecurityEventType.SQL_INJECTION_ATTEMPT: {
                "count": 1,
                "window": 60,
                "action": "immediate_alert",
            },
            SecurityEventType.XSS_ATTEMPT: {
                "count": 1,
                "window": 60,
                "action": "immediate_alert",
            },
            SecurityEventType.BRUTE_FORCE_ATTACK: {
                "count": 3,
                "window": 900,  # 15 minutes
                "action": "block_ip",
            },
            SecurityEventType.DDOS_ATTACK: {
                "count": 1,
                "window": 60,
                "action": "immediate_block",
            },
        }

    def log_security_event(self, event: SecurityEvent) -> None:
        """Log security event and trigger monitoring rules."""
        with self.lock:
            # Add event to storage
            self.events.append(event)
            self.event_counts[event.event_type] += 1

            # Track by IP and user
            self.ip_events[event.source_ip].append(event)
            if event.user_id:
                self.user_events[event.user_id].append(event)

            # Update measurements
            self.measurements.record_event(event)

            # Log to application logger
            self._log_event_to_file(event)

            # Check monitoring rules
            self._check_monitoring_rules(event)

            # Clean old events
            self._cleanup_old_events()

    def _log_event_to_file(self, event: SecurityEvent) -> None:
        """Log security event to file."""
        event_data = {
            "event_id": event.event_id,
            "event_type": event.event_type,
            "severity": event.severity,
            "timestamp": event.timestamp.isoformat(),
            "source_ip": event.source_ip,
            "user_id": event.user_id,
            "endpoint": event.endpoint,
            "user_agent": event.user_agent,
            "details": event.details,
            "risk_score": event.risk_score,
            "mitigated": event.mitigated,
            "mitigation_actions": event.mitigation_actions,
        }

        # Use structured logging
        if event.severity in [SecuritySeverity.HIGH, SecuritySeverity.CRITICAL]:
            logger.critical(f"SECURITY_EVENT: {json.dumps(event_data)}")
        elif event.severity == SecuritySeverity.MEDIUM:
            logger.warning(f"SECURITY_EVENT: {json.dumps(event_data)}")
        else:
            logger.info(f"SECURITY_EVENT: {json.dumps(event_data)}")

    def _check_monitoring_rules(self, event: SecurityEvent) -> None:
        """Check monitoring rules and trigger alerts."""
        event_type = event.event_type

        if event_type in self.alert_thresholds:
            threshold = self.alert_thresholds[event_type]
            self._check_threshold_violation(event, threshold)

        # Check custom monitoring rules
        for rule in self.monitoring_rules:
            if rule.matches(event):
                rule.execute(event, self)

    def _check_threshold_violation(
        self, event: SecurityEvent, threshold: dict[str, Any]
    ) -> None:
        """Check if event violates threshold and trigger action."""
        window_start = event.timestamp - timedelta(seconds=threshold["window"])

        # Count recent events of same type from same IP
        recent_events = [
            e
            for e in self.ip_events[event.source_ip]
            if e.event_type == event.event_type and e.timestamp >= window_start
        ]

        if len(recent_events) >= threshold["count"]:
            action = threshold["action"]
            self._execute_mitigation_action(action, event, recent_events)

    def _execute_mitigation_action(
        self, action: str, event: SecurityEvent, related_events: list[SecurityEvent]
    ) -> None:
        """Execute mitigation action."""
        if action == "alert":
            self._send_alert(event, related_events)
        elif action == "immediate_alert":
            self._send_immediate_alert(event)
        elif action == "block_ip":
            self._block_ip(event.source_ip, event)
        elif action == "immediate_block":
            self._emergency_block_ip(event.source_ip, event)

        # Mark events as mitigated
        for e in related_events:
            e.mitigated = True
            e.mitigation_actions.append(action)

    def _send_alert(
        self, event: SecurityEvent, related_events: list[SecurityEvent]
    ) -> None:
        """Send security alert."""
        alert_data = {
            "alert_type": "security_threshold_exceeded",
            "primary_event": event,
            "related_events": related_events,
            "risk_level": max(e.risk_score for e in related_events),
            "timestamp": datetime.utcnow().isoformat(),
        }

        # Execute alert callbacks
        for callback in self.alert_callbacks:
            try:
                callback(alert_data)
            except Exception as e:
                logger.error(f"Alert callback failed: {e}")

    def _send_immediate_alert(self, event: SecurityEvent) -> None:
        """Send immediate high-priority alert."""
        alert_data = {
            "alert_type": "immediate_security_threat",
            "event": event,
            "risk_level": event.risk_score,
            "timestamp": datetime.utcnow().isoformat(),
            "priority": "high",
        }

        # Execute alert callbacks with high priority
        for callback in self.alert_callbacks:
            try:
                callback(alert_data)
            except Exception as e:
                logger.error(f"Immediate alert callback failed: {e}")

    def _block_ip(self, ip: str, event: SecurityEvent) -> None:
        """Block IP address."""
        # This would integrate with firewall/load balancer
        logger.critical(f"IP {ip} blocked due to security event: {event.event_type}")

        # Create blocking event
        block_event = SecurityEvent(
            event_id=f"block_{int(time.time())}",
            event_type=SecurityEventType.DDOS_ATTACK,  # Generic blocking event
            severity=SecuritySeverity.HIGH,
            timestamp=datetime.utcnow(),
            source_ip=ip,
            user_id=event.user_id,
            endpoint="security_system",
            user_agent="security_monitor",
            details={"action": "ip_blocked", "reason": event.event_type},
            risk_score=0.9,
        )

        self.log_security_event(block_event)

    def _emergency_block_ip(self, ip: str, event: SecurityEvent) -> None:
        """Emergency IP blocking for critical threats."""
        logger.critical(
            f"EMERGENCY: IP {ip} immediately blocked due to: {event.event_type}"
        )
        self._block_ip(ip, event)

        # Send immediate alert
        self._send_immediate_alert(event)

    def _cleanup_old_events(self) -> None:
        """Clean up old events from IP and user tracking."""
        cutoff_time = datetime.utcnow() - timedelta(hours=24)

        # Clean IP events
        for ip in list(self.ip_events.keys()):
            self.ip_events[ip] = [
                e for e in self.ip_events[ip] if e.timestamp > cutoff_time
            ]
            if not self.ip_events[ip]:
                del self.ip_events[ip]

        # Clean user events
        for user_id in list(self.user_events.keys()):
            self.user_events[user_id] = [
                e for e in self.user_events[user_id] if e.timestamp > cutoff_time
            ]
            if not self.user_events[user_id]:
                del self.user_events[user_id]

    def add_alert_callback(self, callback: Callable[[dict[str, Any]], None]) -> None:
        """Add alert callback function."""
        self.alert_callbacks.append(callback)

    def add_monitoring_rule(self, rule: "MonitoringRule") -> None:
        """Add custom monitoring rule."""
        self.monitoring_rules.append(rule)

    def get_security_summary(self, hours: int = 24) -> dict[str, Any]:
        """Get security summary for specified time period."""
        cutoff_time = datetime.utcnow() - timedelta(hours=hours)

        with self.lock:
            recent_events = [e for e in self.events if e.timestamp > cutoff_time]

            summary = {
                "total_events": len(recent_events),
                "events_by_type": defaultdict(int),
                "events_by_severity": defaultdict(int),
                "top_source_ips": defaultdict(int),
                "high_risk_events": [],
                "mitigation_actions": defaultdict(int),
                "average_risk_score": 0.0,
            }

            total_risk = 0
            for event in recent_events:
                summary["events_by_type"][event.event_type] += 1
                summary["events_by_severity"][event.severity] += 1
                summary["top_source_ips"][event.source_ip] += 1
                total_risk += event.risk_score

                if event.risk_score > 0.7:
                    summary["high_risk_events"].append(event)

                for action in event.mitigation_actions:
                    summary["mitigation_actions"][action] += 1

            if recent_events:
                summary["average_risk_score"] = total_risk / len(recent_events)

            # Convert defaultdicts to regular dicts for JSON serialization
            for key in [
                "events_by_type",
                "events_by_severity",
                "top_source_ips",
                "mitigation_actions",
            ]:
                summary[key] = dict(summary[key])

            return summary


class MonitoringRule:
    """Custom monitoring rule for security events."""

    def __init__(
        self,
        name: str,
        condition: Callable[[SecurityEvent], bool],
        action: Callable[[SecurityEvent, SecurityMonitor], None],
    ):
        self.name = name
        self.condition = condition
        self.action = action

    def matches(self, event: SecurityEvent) -> bool:
        """Check if event matches rule condition."""
        try:
            return self.condition(event)
        except Exception as e:
            logger.error(f"Monitoring rule {self.name} condition failed: {e}")
            return False

    def execute(self, event: SecurityEvent, monitor: SecurityMonitor) -> None:
        """Execute rule action."""
        try:
            self.action(event, monitor)
            logger.info(
                f"Monitoring rule {self.name} executed for event {event.event_id}"
            )
        except Exception as e:
            logger.error(f"Monitoring rule {self.name} action failed: {e}")


class SecurityMetrics:
    """Security measurements collection and analysis."""

    def __init__(self):
        self.event_measurements = defaultdict(int)
        self.risk_scores = deque(maxlen=1000)
        self.response_times = deque(maxlen=1000)
        self.blocked_ips = set()
        self.false_positives = 0
        self.true_positives = 0

    def record_event(self, event: SecurityEvent) -> None:
        """Record event measurements."""
        self.event_measurements[event.event_type] += 1
        self.event_measurements[f"{event.event_type}_{event.severity}"] += 1
        self.risk_scores.append(event.risk_score)

    def record_response_time(self, response_time: float) -> None:
        """Record security response time."""
        self.response_times.append(response_time)

    def record_false_positive(self) -> None:
        """Record false positive processing."""
        self.false_positives += 1

    def record_true_positive(self) -> None:
        """Record true positive processing."""
        self.true_positives += 1

    def get_accuracy_metrics(self) -> dict[str, float]:
        """Get processing accuracy measurements."""
        total_processings = self.true_positives + self.false_positives

        if total_processings == 0:
            return {"accuracy": 0.0, "precision": 0.0, "false_positive_rate": 0.0}

        accuracy = self.true_positives / total_processings
        precision = self.true_positives / total_processings
        false_positive_rate = self.false_positives / total_processings

        return {
            "accuracy": accuracy,
            "precision": precision,
            "false_positive_rate": false_positive_rate,
        }

    def get_performance_metrics(self) -> dict[str, Any]:
        """Get performance measurements."""
        if not self.response_times:
            return {"average_response_time": 0.0, "max_response_time": 0.0}

        return {
            "average_response_time": sum(self.response_times)
            / len(self.response_times),
            "max_response_time": max(self.response_times),
            "min_response_time": min(self.response_times),
        }


class IntrusionDetectionSystem:
    """Intrusion processing system for advanced threat processing."""

    def __init__(self, security_monitor: SecurityMonitor):
        self.security_monitor = security_monitor
        self.behavior_baselines = {}
        self.anomaly_threshold = 0.7
        self.learning_period = timedelta(days=7)

    def analyze_request_pattern(
        self, ip: str, user_id: str | None, endpoint: str, timestamp: datetime
    ) -> float:
        """Analyze request pattern for anomalies."""
        anomaly_score = 0.0

        # Check IP behavior
        if ip in self.security_monitor.ip_events:
            ip_events = self.security_monitor.ip_events[ip]
            anomaly_score += self._analyze_ip_pattern(ip_events, timestamp)

        # Check user behavior
        if user_id and user_id in self.security_monitor.user_events:
            user_events = self.security_monitor.user_events[user_id]
            anomaly_score += self._analyze_user_pattern(user_events, timestamp)

        # Check endpoint access pattern
        anomaly_score += self._analyze_endpoint_pattern(endpoint, timestamp)

        return min(1.0, anomaly_score)

    def _analyze_ip_pattern(
        self, events: list[SecurityEvent], timestamp: datetime
    ) -> float:
        """Analyze IP access pattern."""
        if len(events) < 5:
            return 0.0

        # Check request frequency
        recent_events = [
            e for e in events if (timestamp - e.timestamp).total_seconds() < 3600
        ]

        if len(recent_events) > 100:  # More than 100 requests per hour
            return 0.5

        # Check for pattern diversity
        endpoints = set(e.endpoint for e in recent_events)
        if len(endpoints) == 1 and len(recent_events) > 10:  # Same endpoint repeatedly
            return 0.3

        return 0.0

    def _analyze_user_pattern(
        self, events: list[SecurityEvent], timestamp: datetime
    ) -> float:
        """Analyze user behavior pattern."""
        if len(events) < 3:
            return 0.0

        # Check for unusual activity hours
        hour = timestamp.hour
        typical_hours = set(e.timestamp.hour for e in events[-20:])  # Last 20 events

        if hour not in typical_hours and len(typical_hours) > 3:
            return 0.2

        return 0.0

    def _analyze_endpoint_pattern(self, endpoint: str, timestamp: datetime) -> float:
        """Analyze endpoint access pattern."""
        # Check for suspicious endpoints
        suspicious_patterns = [
            "/admin",
            "/.env",
            "/config",
            "/wp-admin",
            "/phpmyadmin",
            "eval(",
            "exec(",
            "system(",
            "shell_exec(",
        ]

        for pattern in suspicious_patterns:
            if pattern in endpoint.lower():
                return 0.6

        return 0.0


def create_default_monitoring_rules() -> list[MonitoringRule]:
    """Create default monitoring rules."""
    rules = []

    # SQL injection processing rule
    def sql_injection_condition(event: SecurityEvent) -> bool:
        return event.event_type == SecurityEventType.SQL_INJECTION_ATTEMPT

    def sql_injection_action(event: SecurityEvent, monitor: SecurityMonitor) -> None:
        monitor._send_immediate_alert(event)
        monitor._block_ip(event.source_ip, event)

    rules.append(
        MonitoringRule(
            "SQL Injection Response", sql_injection_condition, sql_injection_action
        )
    )

    # Brute force processing rule
    def brute_force_condition(event: SecurityEvent) -> bool:
        if event.event_type != SecurityEventType.AUTHENTICATION_FAILURE:
            return False

        # Check for multiple failures from same IP
        recent_failures = [
            e
            for e in monitor.ip_events[event.source_ip]
            if e.event_type == SecurityEventType.AUTHENTICATION_FAILURE
            and (event.timestamp - e.timestamp).total_seconds() < 300
        ]

        return len(recent_failures) >= 5

    def brute_force_action(event: SecurityEvent, monitor: SecurityMonitor) -> None:
        # Create brute force event
        bf_event = SecurityEvent(
            event_id=f"bf_{int(time.time())}",
            event_type=SecurityEventType.BRUTE_FORCE_ATTACK,
            severity=SecuritySeverity.HIGH,
            timestamp=datetime.utcnow(),
            source_ip=event.source_ip,
            user_id=event.user_id,
            endpoint=event.endpoint,
            user_agent=event.user_agent,
            details={"detected_by": "brute_force_rule"},
            risk_score=0.8,
        )

        monitor.log_security_event(bf_event)
        monitor._block_ip(event.source_ip, bf_event)

    rules.append(
        MonitoringRule(
            "Brute Force Processing", brute_force_condition, brute_force_action
        )
    )

    return rules
