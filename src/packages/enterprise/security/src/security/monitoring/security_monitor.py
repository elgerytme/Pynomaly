"""Enterprise security monitoring and threat detection."""

from __future__ import annotations

import json
import time
from datetime import datetime, timezone, timedelta
from typing import Dict, List, Optional, Any, Set, Callable
from dataclasses import dataclass, field
from enum import Enum
from collections import defaultdict, deque
from threading import Lock
import hashlib

import structlog
from prometheus_client import Counter, Histogram, Gauge, CollectorRegistry

from ..config.security_config import SecurityConfig
from ...shared.infrastructure.exceptions.base_exceptions import (
    BaseApplicationError,
    ErrorCategory,
    ErrorSeverity
)
from ...shared.infrastructure.logging.structured_logging import StructuredLogger


logger = structlog.get_logger()


class ThreatLevel(Enum):
    """Threat severity levels."""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


class EventType(Enum):
    """Security event types."""
    AUTHENTICATION_SUCCESS = "auth_success"
    AUTHENTICATION_FAILURE = "auth_failure"
    AUTHORIZATION_DENIED = "authz_denied"
    SUSPICIOUS_ACTIVITY = "suspicious_activity"
    BRUTE_FORCE_ATTEMPT = "brute_force"
    ANOMALOUS_BEHAVIOR = "anomalous_behavior"
    DATA_ACCESS_VIOLATION = "data_access_violation"
    PRIVILEGE_ESCALATION = "privilege_escalation"
    MALICIOUS_REQUEST = "malicious_request"
    SYSTEM_INTRUSION = "system_intrusion"


@dataclass
class SecurityEvent:
    """Security event data structure."""
    event_id: str
    event_type: EventType
    threat_level: ThreatLevel
    timestamp: datetime
    user_id: Optional[str] = None
    ip_address: Optional[str] = None
    user_agent: Optional[str] = None
    resource: Optional[str] = None
    action: Optional[str] = None
    details: Dict[str, Any] = field(default_factory=dict)
    tags: Set[str] = field(default_factory=set)


@dataclass
class ThreatSignature:
    """Threat detection signature."""
    name: str
    description: str
    pattern: Dict[str, Any]
    threat_level: ThreatLevel
    enabled: bool = True
    created_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))


@dataclass
class BehaviorProfile:
    """User behavior profile for anomaly detection."""
    user_id: str
    typical_login_times: List[int] = field(default_factory=list)  # Hours of day
    typical_ip_addresses: Set[str] = field(default_factory=set)
    typical_user_agents: Set[str] = field(default_factory=set)
    typical_resources: Set[str] = field(default_factory=set)
    login_frequency: Dict[str, int] = field(default_factory=dict)  # Day -> count
    last_updated: datetime = field(default_factory=lambda: datetime.now(timezone.utc))


@dataclass
class SecurityMetrics:
    """Security monitoring metrics."""
    total_events: int = 0
    threat_events: int = 0
    blocked_attempts: int = 0
    false_positives: int = 0
    response_time_ms: float = 0.0
    active_threats: int = 0


class SecurityMonitor:
    """Enterprise security monitoring and threat detection system.
    
    Provides comprehensive security monitoring including:
    - Real-time threat detection
    - Behavioral anomaly detection
    - Brute force protection
    - Suspicious activity monitoring
    - Security metrics and alerting
    - Audit logging
    """
    
    def __init__(self, config: SecurityConfig):
        self.config = config
        self.logger = StructuredLogger(config.logging)
        self._lock = Lock()
        
        # Event storage (in production, use external storage)
        self._events: deque = deque(maxlen=10000)
        self._event_index: Dict[str, SecurityEvent] = {}
        
        # Threat detection
        self._threat_signatures: Dict[str, ThreatSignature] = {}
        self._behavior_profiles: Dict[str, BehaviorProfile] = {}
        
        # Rate limiting and tracking
        self._ip_attempts: Dict[str, deque] = {}
        self._user_attempts: Dict[str, deque] = {}
        self._blocked_ips: Dict[str, datetime] = {}
        
        # Metrics
        self._metrics = SecurityMetrics()
        self._prometheus_registry = CollectorRegistry()
        self._setup_prometheus_metrics()
        
        # Alert handlers
        self._alert_handlers: List[Callable[[SecurityEvent], None]] = []
        
        # Initialize default threat signatures
        self._initialize_threat_signatures()
    
    def record_event(
        self,
        event_type: EventType,
        user_id: Optional[str] = None,
        ip_address: Optional[str] = None,
        user_agent: Optional[str] = None,
        resource: Optional[str] = None,
        action: Optional[str] = None,
        details: Dict[str, Any] = None
    ) -> SecurityEvent:
        """Record a security event."""
        try:
            event_id = self._generate_event_id()
            event = SecurityEvent(
                event_id=event_id,
                event_type=event_type,
                threat_level=ThreatLevel.LOW,  # Will be updated by threat detection
                timestamp=datetime.now(timezone.utc),
                user_id=user_id,
                ip_address=ip_address,
                user_agent=user_agent,
                resource=resource,
                action=action,
                details=details or {}
            )
            
            # Store event
            with self._lock:
                self._events.append(event)
                self._event_index[event_id] = event
                self._metrics.total_events += 1
            
            # Analyze for threats
            self._analyze_event(event)
            
            # Update behavior profile
            if user_id:
                self._update_behavior_profile(event)
            
            # Update metrics
            self._prometheus_events_total.labels(
                event_type=event_type.value,
                threat_level=event.threat_level.value
            ).inc()
            
            self.logger.info(
                "Security event recorded",
                event_id=event_id,
                event_type=event_type.value,
                threat_level=event.threat_level.value,
                user_id=user_id,
                ip_address=ip_address
            )
            
            return event
            
        except Exception as e:
            self.logger.error("Failed to record security event", error=str(e))
            raise
    
    def detect_brute_force(self, ip_address: str, user_id: Optional[str] = None) -> bool:
        """Detect brute force attacks."""
        try:
            current_time = time.time()
            window_seconds = 300  # 5 minutes
            max_attempts = 10
            
            # Check IP-based attempts
            if ip_address not in self._ip_attempts:
                self._ip_attempts[ip_address] = deque()
            
            # Clean old attempts
            ip_attempts = self._ip_attempts[ip_address]
            while ip_attempts and current_time - ip_attempts[0] > window_seconds:
                ip_attempts.popleft()
            
            # Add current attempt
            ip_attempts.append(current_time)
            
            # Check if threshold exceeded
            if len(ip_attempts) > max_attempts:
                self._block_ip(ip_address, duration_minutes=30)
                
                # Record threat event
                self.record_event(
                    EventType.BRUTE_FORCE_ATTEMPT,
                    user_id=user_id,
                    ip_address=ip_address,
                    details={
                        "attempts_in_window": len(ip_attempts),
                        "window_seconds": window_seconds
                    }
                )
                
                return True
            
            return False
            
        except Exception as e:
            self.logger.error("Brute force detection failed", error=str(e))
            return False
    
    def detect_anomalous_behavior(self, event: SecurityEvent) -> bool:
        """Detect anomalous user behavior."""
        try:
            if not event.user_id:
                return False
            
            profile = self._behavior_profiles.get(event.user_id)
            if not profile:
                return False  # No baseline yet
            
            anomalies = []
            
            # Check login time anomaly
            if event.event_type == EventType.AUTHENTICATION_SUCCESS:
                current_hour = event.timestamp.hour
                if profile.typical_login_times and current_hour not in profile.typical_login_times:
                    # Check if this hour is significantly different
                    min_hour = min(profile.typical_login_times)
                    max_hour = max(profile.typical_login_times)
                    if current_hour < min_hour - 3 or current_hour > max_hour + 3:
                        anomalies.append("unusual_login_time")
            
            # Check IP address anomaly
            if event.ip_address and event.ip_address not in profile.typical_ip_addresses:
                # Check if this is a completely new location (simplified)
                if len(profile.typical_ip_addresses) > 0:
                    anomalies.append("unusual_ip_address")
            
            # Check user agent anomaly
            if event.user_agent and event.user_agent not in profile.typical_user_agents:
                if len(profile.typical_user_agents) > 0:
                    anomalies.append("unusual_user_agent")
            
            # Check resource access anomaly
            if event.resource and event.resource not in profile.typical_resources:
                if len(profile.typical_resources) > 5:  # Only check if we have enough data
                    anomalies.append("unusual_resource_access")
            
            # If multiple anomalies detected, flag as suspicious
            if len(anomalies) >= 2:
                event.threat_level = ThreatLevel.MEDIUM
                event.tags.update(anomalies)
                
                self.record_event(
                    EventType.ANOMALOUS_BEHAVIOR,
                    user_id=event.user_id,
                    ip_address=event.ip_address,
                    details={
                        "anomalies": anomalies,
                        "original_event_id": event.event_id
                    }
                )
                
                return True
            
            return False
            
        except Exception as e:
            self.logger.error("Anomaly detection failed", error=str(e))
            return False
    
    def check_threat_signatures(self, event: SecurityEvent) -> List[ThreatSignature]:
        """Check event against threat signatures."""
        try:
            matched_signatures = []
            
            for signature in self._threat_signatures.values():
                if not signature.enabled:
                    continue
                
                if self._matches_signature(event, signature):
                    matched_signatures.append(signature)
                    event.threat_level = max(event.threat_level, signature.threat_level, key=lambda x: x.value)
                    event.tags.add(f"signature:{signature.name}")
            
            return matched_signatures
            
        except Exception as e:
            self.logger.error("Threat signature check failed", error=str(e))
            return []
    
    def is_ip_blocked(self, ip_address: str) -> bool:
        """Check if IP address is blocked."""
        if ip_address not in self._blocked_ips:
            return False
        
        # Check if block has expired
        block_time = self._blocked_ips[ip_address]
        if datetime.now(timezone.utc) > block_time:
            del self._blocked_ips[ip_address]
            return False
        
        return True
    
    def add_threat_signature(self, signature: ThreatSignature) -> None:
        """Add a new threat detection signature."""
        self._threat_signatures[signature.name] = signature
        self.logger.info("Threat signature added", signature=signature.name)
    
    def add_alert_handler(self, handler: Callable[[SecurityEvent], None]) -> None:
        """Add an alert handler for security events."""
        self._alert_handlers.append(handler)
    
    def get_security_metrics(self) -> SecurityMetrics:
        """Get current security metrics."""
        return self._metrics
    
    def get_recent_events(
        self, 
        limit: int = 100, 
        event_types: List[EventType] = None,
        threat_levels: List[ThreatLevel] = None
    ) -> List[SecurityEvent]:
        """Get recent security events with optional filtering."""
        events = list(self._events)
        
        # Filter by event types
        if event_types:
            events = [e for e in events if e.event_type in event_types]
        
        # Filter by threat levels
        if threat_levels:
            events = [e for e in events if e.threat_level in threat_levels]
        
        # Sort by timestamp (newest first) and limit
        events.sort(key=lambda x: x.timestamp, reverse=True)
        return events[:limit]
    
    def generate_security_report(self, hours: int = 24) -> Dict[str, Any]:
        """Generate security report for the specified time period."""
        try:
            cutoff_time = datetime.now(timezone.utc) - timedelta(hours=hours)
            recent_events = [e for e in self._events if e.timestamp >= cutoff_time]
            
            # Count events by type
            event_counts = defaultdict(int)
            threat_counts = defaultdict(int)
            
            for event in recent_events:
                event_counts[event.event_type.value] += 1
                threat_counts[event.threat_level.value] += 1
            
            # Top IPs with most events
            ip_counts = defaultdict(int)
            for event in recent_events:
                if event.ip_address:
                    ip_counts[event.ip_address] += 1
            
            top_ips = sorted(ip_counts.items(), key=lambda x: x[1], reverse=True)[:10]
            
            # Top users with most events
            user_counts = defaultdict(int)
            for event in recent_events:
                if event.user_id:
                    user_counts[event.user_id] += 1
            
            top_users = sorted(user_counts.items(), key=lambda x: x[1], reverse=True)[:10]
            
            return {
                "report_period_hours": hours,
                "total_events": len(recent_events),
                "event_counts": dict(event_counts),
                "threat_counts": dict(threat_counts),
                "top_ips": top_ips,
                "top_users": top_users,
                "blocked_ips": len(self._blocked_ips),
                "active_behavior_profiles": len(self._behavior_profiles),
                "generated_at": datetime.now(timezone.utc).isoformat()
            }
            
        except Exception as e:
            self.logger.error("Security report generation failed", error=str(e))
            return {}
    
    # Private helper methods
    
    def _generate_event_id(self) -> str:
        """Generate unique event ID."""
        import uuid
        return str(uuid.uuid4())
    
    def _analyze_event(self, event: SecurityEvent) -> None:
        """Analyze event for threats."""
        try:
            # Check threat signatures
            matched_signatures = self.check_threat_signatures(event)
            
            # Check for anomalous behavior
            self.detect_anomalous_behavior(event)
            
            # Check for brute force
            if event.event_type == EventType.AUTHENTICATION_FAILURE and event.ip_address:
                self.detect_brute_force(event.ip_address, event.user_id)
            
            # Update metrics
            if event.threat_level in [ThreatLevel.HIGH, ThreatLevel.CRITICAL]:
                self._metrics.threat_events += 1
                self._metrics.active_threats += 1
            
            # Send alerts for high-priority events
            if event.threat_level in [ThreatLevel.HIGH, ThreatLevel.CRITICAL]:
                self._send_alerts(event)
            
        except Exception as e:
            self.logger.error("Event analysis failed", error=str(e))
    
    def _update_behavior_profile(self, event: SecurityEvent) -> None:
        """Update user behavior profile."""
        try:
            if not event.user_id:
                return
            
            if event.user_id not in self._behavior_profiles:
                self._behavior_profiles[event.user_id] = BehaviorProfile(user_id=event.user_id)
            
            profile = self._behavior_profiles[event.user_id]
            
            # Update login times
            if event.event_type == EventType.AUTHENTICATION_SUCCESS:
                hour = event.timestamp.hour
                if hour not in profile.typical_login_times:
                    profile.typical_login_times.append(hour)
                    # Keep only recent patterns (last 30 days worth)
                    if len(profile.typical_login_times) > 24:
                        profile.typical_login_times = profile.typical_login_times[-24:]
            
            # Update IP addresses
            if event.ip_address:
                profile.typical_ip_addresses.add(event.ip_address)
                # Keep only recent IPs (max 10)
                if len(profile.typical_ip_addresses) > 10:
                    profile.typical_ip_addresses = set(list(profile.typical_ip_addresses)[-10:])
            
            # Update user agents
            if event.user_agent:
                profile.typical_user_agents.add(event.user_agent)
                # Keep only recent user agents (max 5)
                if len(profile.typical_user_agents) > 5:
                    profile.typical_user_agents = set(list(profile.typical_user_agents)[-5:])
            
            # Update resources
            if event.resource:
                profile.typical_resources.add(event.resource)
                # Keep only recent resources (max 20)
                if len(profile.typical_resources) > 20:
                    profile.typical_resources = set(list(profile.typical_resources)[-20:])
            
            profile.last_updated = datetime.now(timezone.utc)
            
        except Exception as e:
            self.logger.error("Behavior profile update failed", error=str(e))
    
    def _matches_signature(self, event: SecurityEvent, signature: ThreatSignature) -> bool:
        """Check if event matches threat signature."""
        try:
            pattern = signature.pattern
            
            # Check event type
            if "event_type" in pattern:
                if event.event_type.value != pattern["event_type"]:
                    return False
            
            # Check patterns in details
            if "details_pattern" in pattern:
                for key, expected_value in pattern["details_pattern"].items():
                    if key not in event.details or event.details[key] != expected_value:
                        return False
            
            # Check user agent patterns
            if "user_agent_pattern" in pattern and event.user_agent:
                import re
                if not re.search(pattern["user_agent_pattern"], event.user_agent):
                    return False
            
            # Check resource patterns
            if "resource_pattern" in pattern and event.resource:
                import re
                if not re.search(pattern["resource_pattern"], event.resource):
                    return False
            
            return True
            
        except Exception:
            return False
    
    def _block_ip(self, ip_address: str, duration_minutes: int = 30) -> None:
        """Block IP address for specified duration."""
        block_until = datetime.now(timezone.utc) + timedelta(minutes=duration_minutes)
        self._blocked_ips[ip_address] = block_until
        self._metrics.blocked_attempts += 1
        
        self.logger.warning(
            "IP address blocked",
            ip_address=ip_address,
            duration_minutes=duration_minutes,
            blocked_until=block_until.isoformat()
        )
    
    def _send_alerts(self, event: SecurityEvent) -> None:
        """Send alerts for security events."""
        try:
            for handler in self._alert_handlers:
                handler(event)
        except Exception as e:
            self.logger.error("Alert sending failed", error=str(e))
    
    def _setup_prometheus_metrics(self) -> None:
        """Setup Prometheus metrics."""
        self._prometheus_events_total = Counter(
            'security_events_total',
            'Total number of security events',
            ['event_type', 'threat_level'],
            registry=self._prometheus_registry
        )
        
        self._prometheus_blocked_attempts = Counter(
            'security_blocked_attempts_total',
            'Total number of blocked attempts',
            registry=self._prometheus_registry
        )
        
        self._prometheus_active_threats = Gauge(
            'security_active_threats',
            'Number of active threats',
            registry=self._prometheus_registry
        )
    
    def _initialize_threat_signatures(self) -> None:
        """Initialize default threat detection signatures."""
        signatures = [
            ThreatSignature(
                name="sql_injection_attempt",
                description="Detect SQL injection attempts",
                pattern={
                    "details_pattern": {"contains_sql_keywords": True}
                },
                threat_level=ThreatLevel.HIGH
            ),
            ThreatSignature(
                name="privilege_escalation",
                description="Detect privilege escalation attempts",
                pattern={
                    "event_type": "authorization_denied",
                    "resource_pattern": r"\/admin\/.*"
                },
                threat_level=ThreatLevel.HIGH
            ),
            ThreatSignature(
                name="suspicious_user_agent",
                description="Detect suspicious user agents",
                pattern={
                    "user_agent_pattern": r"(sqlmap|nikto|nmap|masscan|python-requests)"
                },
                threat_level=ThreatLevel.MEDIUM
            )
        ]
        
        for signature in signatures:
            self.add_threat_signature(signature)