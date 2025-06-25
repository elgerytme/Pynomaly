"""Security monitoring and alerting system.

This module provides comprehensive security monitoring including:
- Suspicious activity detection
- Failed authentication tracking
- Rate limiting violations
- Security metrics collection
- Real-time alerting
- Threat intelligence integration
"""

from __future__ import annotations

import asyncio
import logging
import time
from abc import ABC, abstractmethod
from collections import defaultdict, deque
from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import UTC, datetime, timedelta
from enum import Enum
from typing import Any

from .audit_logger import AuditLevel, AuditLogger, SecurityEventType

logger = logging.getLogger(__name__)


class ThreatLevel(str, Enum):
    """Threat severity levels."""

    LOW = "LOW"
    MEDIUM = "MEDIUM"
    HIGH = "HIGH"
    CRITICAL = "CRITICAL"


class AlertType(str, Enum):
    """Types of security alerts."""

    BRUTE_FORCE = "brute_force"
    ANOMALOUS_ACCESS = "anomalous_access"
    PRIVILEGE_ESCALATION = "privilege_escalation"
    DATA_EXFILTRATION = "data_exfiltration"
    INJECTION_ATTACK = "injection_attack"
    RATE_LIMIT_VIOLATION = "rate_limit_violation"
    SUSPICIOUS_LOCATION = "suspicious_location"
    UNUSUAL_BEHAVIOR = "unusual_behavior"
    MALWARE_DETECTED = "malware_detected"
    SYSTEM_COMPROMISE = "system_compromise"


class SecurityMetricType(str, Enum):
    """Types of security metrics."""

    FAILED_LOGINS = "failed_logins"
    SUCCESSFUL_LOGINS = "successful_logins"
    ACCESS_DENIED = "access_denied"
    API_ERRORS = "api_errors"
    RATE_LIMIT_HITS = "rate_limit_hits"
    INJECTION_ATTEMPTS = "injection_attempts"
    SUSPICIOUS_IPS = "suspicious_ips"
    BLOCKED_REQUESTS = "blocked_requests"


@dataclass
class SecurityMetric:
    """Security metric data point."""

    metric_type: SecurityMetricType
    value: float
    timestamp: datetime
    labels: dict[str, str] = field(default_factory=dict)
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class SecurityMetrics:
    """Collection of security metrics."""

    total_requests: int = 0
    failed_authentications: int = 0
    blocked_requests: int = 0
    threat_detections: int = 0
    active_sessions: int = 0
    suspicious_activities: int = 0
    metrics_by_type: dict[SecurityMetricType, list[SecurityMetric]] = field(
        default_factory=dict
    )
    last_updated: datetime = field(default_factory=lambda: datetime.now(UTC))


@dataclass
class SecurityAlert:
    """Security alert data structure."""

    alert_id: str
    alert_type: AlertType
    threat_level: ThreatLevel
    title: str
    description: str
    timestamp: datetime

    # Context
    source_ip: str | None = None
    user_id: str | None = None
    user_agent: str | None = None
    session_id: str | None = None

    # Technical details
    indicators: dict[str, Any] = field(default_factory=dict)
    evidence: list[str] = field(default_factory=list)
    affected_resources: list[str] = field(default_factory=list)

    # Response
    recommended_actions: list[str] = field(default_factory=list)
    auto_mitigated: bool = False
    mitigation_actions: list[str] = field(default_factory=list)

    # Metadata
    correlation_id: str | None = None
    related_alerts: list[str] = field(default_factory=list)
    false_positive_score: float = 0.0  # 0.0 = likely real, 1.0 = likely false positive


class ThreatDetector(ABC):
    """Abstract base class for threat detectors."""

    def __init__(self, name: str):
        self.name = name
        self.enabled = True

    @abstractmethod
    async def analyze(self, event_data: dict[str, Any]) -> SecurityAlert | None:
        """Analyze event data for threats.

        Args:
            event_data: Event data to analyze

        Returns:
            Security alert if threat detected, None otherwise
        """
        pass

    @abstractmethod
    def get_configuration(self) -> dict[str, Any]:
        """Get detector configuration."""
        pass

    @abstractmethod
    def update_configuration(self, config: dict[str, Any]) -> None:
        """Update detector configuration."""
        pass


class BruteForceDetector(ThreatDetector):
    """Detector for brute force attacks."""

    def __init__(self):
        super().__init__("brute_force")
        self.failed_attempts: dict[str, deque] = defaultdict(lambda: deque(maxlen=100))
        self.blocked_ips: set[str] = set()

        # Configuration
        self.max_attempts = 5
        self.time_window = 300  # 5 minutes
        self.block_duration = 3600  # 1 hour

    async def analyze(self, event_data: dict[str, Any]) -> SecurityAlert | None:
        """Analyze for brute force patterns."""
        event_type = event_data.get("event_type")

        if event_type == SecurityEventType.AUTH_LOGIN_FAILURE:
            ip_address = event_data.get("ip_address")
            if not ip_address:
                return None

            current_time = time.time()

            # Add failed attempt
            self.failed_attempts[ip_address].append(current_time)

            # Clean old attempts
            cutoff_time = current_time - self.time_window
            attempts = self.failed_attempts[ip_address]
            while attempts and attempts[0] < cutoff_time:
                attempts.popleft()

            # Check if threshold exceeded
            if len(attempts) >= self.max_attempts:
                self.blocked_ips.add(ip_address)

                return SecurityAlert(
                    alert_id=f"bf_{ip_address}_{int(current_time)}",
                    alert_type=AlertType.BRUTE_FORCE,
                    threat_level=ThreatLevel.HIGH,
                    title="Brute Force Attack Detected",
                    description=f"Multiple failed login attempts from IP {ip_address}",
                    timestamp=datetime.now(UTC),
                    source_ip=ip_address,
                    indicators={
                        "failed_attempts": len(attempts),
                        "time_window": self.time_window,
                        "threshold": self.max_attempts,
                    },
                    evidence=[
                        f"Failed login attempts: {len(attempts)} in {self.time_window}s"
                    ],
                    recommended_actions=[
                        "Block IP address",
                        "Review authentication logs",
                        "Check for compromised accounts",
                        "Consider implementing CAPTCHA",
                    ],
                    auto_mitigated=True,
                    mitigation_actions=["IP blocked for brute force"],
                )

        return None

    def get_configuration(self) -> dict[str, Any]:
        """Get detector configuration."""
        return {
            "max_attempts": self.max_attempts,
            "time_window": self.time_window,
            "block_duration": self.block_duration,
        }

    def update_configuration(self, config: dict[str, Any]) -> None:
        """Update detector configuration."""
        self.max_attempts = config.get("max_attempts", self.max_attempts)
        self.time_window = config.get("time_window", self.time_window)
        self.block_duration = config.get("block_duration", self.block_duration)

    def is_ip_blocked(self, ip_address: str) -> bool:
        """Check if IP is currently blocked."""
        return ip_address in self.blocked_ips


class AnomalousAccessDetector(ThreatDetector):
    """Detector for anomalous access patterns."""

    def __init__(self):
        super().__init__("anomalous_access")
        self.user_patterns: dict[str, dict[str, Any]] = defaultdict(dict)
        self.learning_period = 7 * 24 * 3600  # 7 days

    async def analyze(self, event_data: dict[str, Any]) -> SecurityAlert | None:
        """Analyze for anomalous access patterns."""
        event_type = event_data.get("event_type")
        user_id = event_data.get("user_id")
        ip_address = event_data.get("ip_address")
        user_agent = event_data.get("user_agent")

        if not user_id or event_type != SecurityEventType.AUTH_LOGIN_SUCCESS:
            return None

        current_time = time.time()
        user_pattern = self.user_patterns[user_id]

        # Initialize pattern if new user
        if "first_seen" not in user_pattern:
            user_pattern["first_seen"] = current_time
            user_pattern["known_ips"] = {ip_address}
            user_pattern["known_user_agents"] = {user_agent} if user_agent else set()
            user_pattern["login_times"] = [current_time]
            return None

        # Skip analysis during learning period
        if current_time - user_pattern["first_seen"] < self.learning_period:
            # Update patterns during learning
            if ip_address:
                user_pattern["known_ips"].add(ip_address)
            if user_agent:
                user_pattern["known_user_agents"].add(user_agent)
            user_pattern["login_times"].append(current_time)
            return None

        # Analyze for anomalies
        anomalies = []

        # Check for new IP address
        if ip_address and ip_address not in user_pattern["known_ips"]:
            anomalies.append(f"Login from unknown IP: {ip_address}")

        # Check for new user agent
        if user_agent and user_agent not in user_pattern["known_user_agents"]:
            anomalies.append(f"Login with unknown user agent: {user_agent[:100]}")

        # Check for unusual time
        hour = datetime.fromtimestamp(current_time).hour
        usual_hours = {
            datetime.fromtimestamp(t).hour for t in user_pattern["login_times"][-50:]
        }
        if hour not in usual_hours and len(usual_hours) > 5:
            anomalies.append(f"Login at unusual time: {hour:02d}:00")

        if anomalies:
            return SecurityAlert(
                alert_id=f"aa_{user_id}_{int(current_time)}",
                alert_type=AlertType.ANOMALOUS_ACCESS,
                threat_level=ThreatLevel.MEDIUM
                if len(anomalies) == 1
                else ThreatLevel.HIGH,
                title="Anomalous Access Pattern Detected",
                description=f"User {user_id} login shows unusual patterns",
                timestamp=datetime.now(UTC),
                source_ip=ip_address,
                user_id=user_id,
                user_agent=user_agent,
                indicators={
                    "anomalies_count": len(anomalies),
                    "learning_period_days": self.learning_period / (24 * 3600),
                },
                evidence=anomalies,
                recommended_actions=[
                    "Verify user identity",
                    "Check for account compromise",
                    "Review recent user activity",
                    "Consider requiring additional authentication",
                ],
            )

        # Update patterns
        if ip_address:
            user_pattern["known_ips"].add(ip_address)
        if user_agent:
            user_pattern["known_user_agents"].add(user_agent)
        user_pattern["login_times"].append(current_time)

        return None

    def get_configuration(self) -> dict[str, Any]:
        """Get detector configuration."""
        return {"learning_period_days": self.learning_period / (24 * 3600)}

    def update_configuration(self, config: dict[str, Any]) -> None:
        """Update detector configuration."""
        if "learning_period_days" in config:
            self.learning_period = config["learning_period_days"] * 24 * 3600


class InjectionAttackDetector(ThreatDetector):
    """Detector for injection attacks."""

    def __init__(self):
        super().__init__("injection_attack")
        self.detection_count: dict[str, int] = defaultdict(int)

    async def analyze(self, event_data: dict[str, Any]) -> SecurityAlert | None:
        """Analyze for injection attacks."""
        event_type = event_data.get("event_type")

        if event_type == SecurityEventType.SECURITY_SQL_INJECTION:
            ip_address = event_data.get("ip_address")
            query = event_data.get("details", {}).get("suspicious_query", "")

            self.detection_count[ip_address] += 1

            return SecurityAlert(
                alert_id=f"inj_{ip_address}_{int(time.time())}",
                alert_type=AlertType.INJECTION_ATTACK,
                threat_level=ThreatLevel.CRITICAL,
                title="SQL Injection Attack Detected",
                description=f"SQL injection attempt from IP {ip_address}",
                timestamp=datetime.now(UTC),
                source_ip=ip_address,
                indicators={
                    "query_snippet": query[:100],
                    "total_attempts": self.detection_count[ip_address],
                },
                evidence=[f"Malicious query: {query[:200]}"],
                affected_resources=["database"],
                recommended_actions=[
                    "Block IP address immediately",
                    "Review application security",
                    "Check database integrity",
                    "Update input validation",
                    "Implement prepared statements",
                ],
                auto_mitigated=True,
                mitigation_actions=["Request blocked", "IP flagged for monitoring"],
            )

        return None

    def get_configuration(self) -> dict[str, Any]:
        """Get detector configuration."""
        return {}

    def update_configuration(self, config: dict[str, Any]) -> None:
        """Update detector configuration."""
        pass


class SecurityMonitor:
    """Main security monitoring service."""

    def __init__(self, audit_logger: AuditLogger | None = None):
        """Initialize security monitor.

        Args:
            audit_logger: Audit logger instance
        """
        self.audit_logger = audit_logger or AuditLogger()
        self.detectors: dict[str, ThreatDetector] = {}
        self.alert_handlers: list[Callable[[SecurityAlert], None]] = []
        self.metrics: dict[SecurityMetricType, list[SecurityMetric]] = defaultdict(list)

        # Alert management
        self.active_alerts: dict[str, SecurityAlert] = {}
        self.alert_history: list[SecurityAlert] = []
        self.max_history_size = 10000

        # Monitoring state
        self.monitoring_enabled = True
        self.processing_queue: asyncio.Queue = asyncio.Queue()
        self.worker_task: asyncio.Task | None = None

        # Initialize default detectors
        self._initialize_detectors()

    def _initialize_detectors(self) -> None:
        """Initialize default threat detectors."""
        self.register_detector(BruteForceDetector())
        self.register_detector(AnomalousAccessDetector())
        self.register_detector(InjectionAttackDetector())

        # Register advanced threat detectors
        try:
            from .advanced_threat_detection import create_advanced_threat_detectors

            advanced_detectors = create_advanced_threat_detectors()
            for detector in advanced_detectors:
                self.register_detector(detector)
            logger.info(
                f"Registered {len(advanced_detectors)} advanced threat detectors"
            )
        except ImportError as e:
            logger.warning(f"Advanced threat detection not available: {e}")

    def register_detector(self, detector: ThreatDetector) -> None:
        """Register a threat detector.

        Args:
            detector: Threat detector to register
        """
        self.detectors[detector.name] = detector
        logger.info(f"Registered threat detector: {detector.name}")

    def register_alert_handler(self, handler: Callable[[SecurityAlert], None]) -> None:
        """Register an alert handler.

        Args:
            handler: Function to handle security alerts
        """
        self.alert_handlers.append(handler)
        logger.info("Registered security alert handler")

    async def start_monitoring(self) -> None:
        """Start the security monitoring service."""
        if self.worker_task is None:
            self.monitoring_enabled = True
            self.worker_task = asyncio.create_task(self._monitoring_worker())
            logger.info("Security monitoring started")

    async def stop_monitoring(self) -> None:
        """Stop the security monitoring service."""
        self.monitoring_enabled = False
        if self.worker_task:
            self.worker_task.cancel()
            try:
                await self.worker_task
            except asyncio.CancelledError:
                pass
            self.worker_task = None
        logger.info("Security monitoring stopped")

    async def process_event(self, event_data: dict[str, Any]) -> None:
        """Process a security event.

        Args:
            event_data: Event data to process
        """
        if not self.monitoring_enabled:
            return

        await self.processing_queue.put(event_data)

    async def _monitoring_worker(self) -> None:
        """Background worker for processing security events."""
        while self.monitoring_enabled:
            try:
                # Get event from queue with timeout
                event_data = await asyncio.wait_for(
                    self.processing_queue.get(), timeout=1.0
                )

                await self._process_event_internal(event_data)

            except TimeoutError:
                # Normal timeout, continue monitoring
                continue
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Error in security monitoring worker: {e}")

    async def _process_event_internal(self, event_data: dict[str, Any]) -> None:
        """Internal event processing logic."""
        # Update metrics
        self._update_metrics(event_data)

        # Run threat detection
        for detector_name, detector in self.detectors.items():
            if not detector.enabled:
                continue

            try:
                alert = await detector.analyze(event_data)
                if alert:
                    await self._handle_alert(alert)

            except Exception as e:
                logger.error(f"Error in detector {detector_name}: {e}")

    def _update_metrics(self, event_data: dict[str, Any]) -> None:
        """Update security metrics based on event."""
        event_type = event_data.get("event_type")
        timestamp = datetime.now(UTC)

        # Map events to metrics
        metric_mapping = {
            SecurityEventType.AUTH_LOGIN_FAILURE: SecurityMetricType.FAILED_LOGINS,
            SecurityEventType.AUTH_LOGIN_SUCCESS: SecurityMetricType.SUCCESSFUL_LOGINS,
            SecurityEventType.AUTHZ_ACCESS_DENIED: SecurityMetricType.ACCESS_DENIED,
            SecurityEventType.SECURITY_SQL_INJECTION: SecurityMetricType.INJECTION_ATTEMPTS,
            SecurityEventType.SECURITY_RATE_LIMIT_EXCEEDED: SecurityMetricType.RATE_LIMIT_HITS,
        }

        metric_type = metric_mapping.get(event_type)
        if metric_type:
            metric = SecurityMetric(
                metric_type=metric_type,
                value=1.0,
                timestamp=timestamp,
                labels={
                    "ip_address": event_data.get("ip_address", ""),
                    "user_id": event_data.get("user_id", ""),
                },
                metadata=event_data.get("details", {}),
            )

            self.metrics[metric_type].append(metric)

            # Keep only recent metrics (last hour)
            cutoff_time = timestamp - timedelta(hours=1)
            self.metrics[metric_type] = [
                m for m in self.metrics[metric_type] if m.timestamp > cutoff_time
            ]

    async def _handle_alert(self, alert: SecurityAlert) -> None:
        """Handle a security alert."""
        # Store alert
        self.active_alerts[alert.alert_id] = alert
        self.alert_history.append(alert)

        # Limit history size
        if len(self.alert_history) > self.max_history_size:
            self.alert_history = self.alert_history[-self.max_history_size :]

        # Log the alert
        self.audit_logger.log_security_event(
            SecurityEventType.SECURITY_SUSPICIOUS_ACTIVITY,
            f"Security alert: {alert.title}",
            level=AuditLevel.WARNING
            if alert.threat_level in [ThreatLevel.LOW, ThreatLevel.MEDIUM]
            else AuditLevel.ERROR,
            details={
                "alert_id": alert.alert_id,
                "alert_type": alert.alert_type,
                "threat_level": alert.threat_level,
                "indicators": alert.indicators,
                "evidence": alert.evidence,
            },
            risk_score=self._calculate_risk_score(alert.threat_level),
        )

        # Notify handlers
        for handler in self.alert_handlers:
            try:
                if asyncio.iscoroutinefunction(handler):
                    await handler(alert)
                else:
                    handler(alert)
            except Exception as e:
                logger.error(f"Error in alert handler: {e}")

        logger.warning(
            f"Security alert generated: {alert.title} (ID: {alert.alert_id})"
        )

    def _calculate_risk_score(self, threat_level: ThreatLevel) -> int:
        """Calculate risk score from threat level."""
        risk_scores = {
            ThreatLevel.LOW: 25,
            ThreatLevel.MEDIUM: 50,
            ThreatLevel.HIGH: 75,
            ThreatLevel.CRITICAL: 100,
        }
        return risk_scores.get(threat_level, 50)

    def get_active_alerts(
        self,
        threat_level: ThreatLevel | None = None,
        alert_type: AlertType | None = None,
    ) -> list[SecurityAlert]:
        """Get active security alerts.

        Args:
            threat_level: Filter by threat level
            alert_type: Filter by alert type

        Returns:
            List of matching active alerts
        """
        alerts = list(self.active_alerts.values())

        if threat_level:
            alerts = [a for a in alerts if a.threat_level == threat_level]

        if alert_type:
            alerts = [a for a in alerts if a.alert_type == alert_type]

        return sorted(alerts, key=lambda a: a.timestamp, reverse=True)

    def get_security_metrics(
        self, metric_type: SecurityMetricType, time_range: timedelta | None = None
    ) -> list[SecurityMetric]:
        """Get security metrics.

        Args:
            metric_type: Type of metrics to retrieve
            time_range: Time range for metrics (default: last hour)

        Returns:
            List of security metrics
        """
        if time_range is None:
            time_range = timedelta(hours=1)

        cutoff_time = datetime.now(UTC) - time_range

        metrics = self.metrics.get(metric_type, [])
        return [m for m in metrics if m.timestamp > cutoff_time]

    def acknowledge_alert(self, alert_id: str) -> bool:
        """Acknowledge a security alert.

        Args:
            alert_id: Alert ID to acknowledge

        Returns:
            True if alert was acknowledged
        """
        if alert_id in self.active_alerts:
            del self.active_alerts[alert_id]
            logger.info(f"Security alert acknowledged: {alert_id}")
            return True
        return False

    def get_security_summary(self) -> dict[str, Any]:
        """Get security monitoring summary.

        Returns:
            Security summary statistics
        """
        now = datetime.now(UTC)
        hour_ago = now - timedelta(hours=1)

        # Count recent alerts by threat level
        recent_alerts = [a for a in self.alert_history if a.timestamp > hour_ago]
        alert_counts = defaultdict(int)
        for alert in recent_alerts:
            alert_counts[alert.threat_level] += 1

        # Count recent metrics
        metric_counts = {}
        for metric_type in SecurityMetricType:
            recent_metrics = self.get_security_metrics(metric_type, timedelta(hours=1))
            metric_counts[metric_type] = sum(m.value for m in recent_metrics)

        return {
            "monitoring_enabled": self.monitoring_enabled,
            "active_alerts_count": len(self.active_alerts),
            "recent_alerts": dict(alert_counts),
            "recent_metrics": metric_counts,
            "detector_status": {
                name: detector.enabled for name, detector in self.detectors.items()
            },
            "summary_timestamp": now.isoformat(),
        }


# Global security monitor instance
_security_monitor: SecurityMonitor | None = None


def get_security_monitor() -> SecurityMonitor:
    """Get global security monitor instance."""
    global _security_monitor
    if _security_monitor is None:
        _security_monitor = SecurityMonitor()
    return _security_monitor


def init_security_monitor(
    audit_logger: AuditLogger | None = None,
) -> SecurityMonitor:
    """Initialize global security monitor.

    Args:
        audit_logger: Audit logger instance

    Returns:
        Security monitor instance
    """
    global _security_monitor
    _security_monitor = SecurityMonitor(audit_logger)
    return _security_monitor


async def start_security_monitoring() -> None:
    """Start security monitoring service."""
    monitor = get_security_monitor()
    await monitor.start_monitoring()


async def stop_security_monitoring() -> None:
    """Stop security monitoring service."""
    monitor = get_security_monitor()
    await monitor.stop_monitoring()
