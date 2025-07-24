"""Advanced threat detection and security monitoring system."""

from __future__ import annotations

import asyncio
import hashlib
import json
import re
import time
from abc import ABC, abstractmethod
from collections import defaultdict, deque
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from typing import Any, Dict, List, Optional, Set, Tuple, Union, Callable
from pathlib import Path
import threading

import numpy as np
from pydantic import BaseModel

from ....infrastructure.logging import get_logger
from ....infrastructure.monitoring import get_metrics_collector

logger = get_logger(__name__)
metrics_collector = get_metrics_collector()


class ThreatType(Enum):
    """Types of security threats."""
    BRUTE_FORCE = "brute_force"
    SQL_INJECTION = "sql_injection"
    XSS_ATTACK = "xss_attack"
    CSRF_ATTACK = "csrf_attack"
    DOS_ATTACK = "dos_attack"
    DDOS_ATTACK = "ddos_attack"
    UNAUTHORIZED_ACCESS = "unauthorized_access"
    DATA_EXFILTRATION = "data_exfiltration"
    MALWARE = "malware"
    PRIVILEGE_ESCALATION = "privilege_escalation"
    SUSPICIOUS_ACTIVITY = "suspicious_activity"
    ANOMALOUS_TRAFFIC = "anomalous_traffic"
    API_ABUSE = "api_abuse"
    AUTHENTICATION_BYPASS = "authentication_bypass"
    INSIDER_THREAT = "insider_threat"


class ThreatSeverity(Enum):
    """Threat severity levels."""
    CRITICAL = "critical"
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"
    INFO = "info"


class ThreatStatus(Enum):
    """Threat detection status."""
    ACTIVE = "active"
    MITIGATED = "mitigated"
    FALSE_POSITIVE = "false_positive"
    INVESTIGATING = "investigating"
    RESOLVED = "resolved"


@dataclass
class ThreatEvent:
    """Individual threat event."""
    id: str
    threat_type: ThreatType
    severity: ThreatSeverity
    title: str
    description: str
    source_ip: Optional[str] = None
    target: Optional[str] = None
    user_agent: Optional[str] = None
    payload: Optional[str] = None
    confidence: float = 1.0  # 0-1
    risk_score: float = 0.0  # 0-100
    timestamp: datetime = field(default_factory=datetime.utcnow)
    status: ThreatStatus = ThreatStatus.ACTIVE
    evidence: Dict[str, Any] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)
    remediation_actions: List[str] = field(default_factory=list)


@dataclass
class SecurityAlert:
    """Security alert aggregating multiple threat events."""
    id: str
    title: str
    description: str
    severity: ThreatSeverity
    threat_events: List[ThreatEvent]
    start_time: datetime
    last_updated: datetime
    status: ThreatStatus = ThreatStatus.ACTIVE
    assignee: Optional[str] = None
    resolution_notes: Optional[str] = None
    escalation_level: int = 0
    affected_systems: List[str] = field(default_factory=list)
    
    @property
    def event_count(self) -> int:
        """Get number of threat events in this alert."""
        return len(self.threat_events)
    
    @property
    def max_risk_score(self) -> float:
        """Get maximum risk score among events."""
        if not self.threat_events:
            return 0.0
        return max(event.risk_score for event in self.threat_events)


class BaseThreatDetector(ABC):
    """Base class for threat detectors."""

    def __init__(self, name: str, config: Optional[Dict[str, Any]] = None):
        self.name = name
        self.config = config or {}
        self.enabled = self.config.get("enabled", True)
        self.sensitivity = self.config.get("sensitivity", 0.7)  # 0-1

    @abstractmethod
    async def detect_threats(self, data: Dict[str, Any]) -> List[ThreatEvent]:
        """Detect threats in the provided data."""
        pass

    def generate_threat_id(self, threat_type: str, source: str) -> str:
        """Generate unique threat ID."""
        timestamp = str(int(time.time()))
        content = f"{self.name}_{threat_type}_{source}_{timestamp}"
        return hashlib.md5(content.encode()).hexdigest()[:12]


class BruteForceDetector(BaseThreatDetector):
    """Detector for brute force attacks."""

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        super().__init__("brute_force_detector", config)
        self.max_attempts = self.config.get("max_attempts", 5)
        self.time_window = self.config.get("time_window_minutes", 5)
        self.attempt_history = defaultdict(deque)
        self._lock = threading.Lock()

    async def detect_threats(self, data: Dict[str, Any]) -> List[ThreatEvent]:
        """Detect brute force attacks."""
        threats = []
        
        # Extract authentication data
        auth_events = data.get("authentication_events", [])
        
        for event in auth_events:
            if event.get("success") is False:
                source_ip = event.get("source_ip")
                username = event.get("username")
                timestamp = datetime.fromisoformat(event.get("timestamp", datetime.utcnow().isoformat()))
                
                if source_ip:
                    threat = await self._check_brute_force(source_ip, username, timestamp)
                    if threat:
                        threats.append(threat)
        
        return threats

    async def _check_brute_force(
        self,
        source_ip: str,
        username: Optional[str],
        timestamp: datetime
    ) -> Optional[ThreatEvent]:
        """Check for brute force pattern."""
        with self._lock:
            key = f"{source_ip}_{username or 'unknown'}"
            
            # Add current attempt
            self.attempt_history[key].append(timestamp)
            
            # Remove old attempts outside time window
            cutoff_time = timestamp - timedelta(minutes=self.time_window)
            while (self.attempt_history[key] and 
                   self.attempt_history[key][0] < cutoff_time):
                self.attempt_history[key].popleft()
            
            # Check if threshold exceeded
            if len(self.attempt_history[key]) >= self.max_attempts:
                return ThreatEvent(
                    id=self.generate_threat_id("brute_force", source_ip),
                    threat_type=ThreatType.BRUTE_FORCE,
                    severity=ThreatSeverity.HIGH,
                    title=f"Brute Force Attack from {source_ip}",
                    description=f"Multiple failed login attempts ({len(self.attempt_history[key])}) detected from {source_ip}",
                    source_ip=source_ip,
                    target=username,
                    confidence=0.9,
                    risk_score=80.0,
                    evidence={
                        "attempt_count": len(self.attempt_history[key]),
                        "time_window_minutes": self.time_window,
                        "targeted_username": username
                    },
                    remediation_actions=[
                        f"Block IP address {source_ip}",
                        "Review authentication logs",
                        "Consider implementing CAPTCHA",
                        "Alert security team"
                    ]
                )
        
        return None


class InjectionDetector(BaseThreatDetector):
    """Detector for injection attacks (SQL, XSS, etc.)."""

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        super().__init__("injection_detector", config)
        
        # SQL injection patterns
        self.sql_patterns = [
            r"(?i)(union\s+select|select\s+\*\s+from|insert\s+into|update\s+\w+\s+set|delete\s+from)",
            r"(?i)('|\")(\s*;\s*)(drop|truncate|delete|update|insert)",
            r"(?i)(or\s+1\s*=\s*1|and\s+1\s*=\s*1)",
            r"(?i)(\|\||&&|\s+or\s+|\s+and\s+).*(\'\s*=\s*\'|\"\s*=\s*\")",
            r"(?i)(exec\s*\(|execute\s*\(|sp_executesql)",
            r"(?i)(information_schema|sysobjects|sys\.tables)"
        ]
        
        # XSS patterns
        self.xss_patterns = [
            r"(?i)<script[^>]*>.*?</script>",
            r"(?i)javascript:",
            r"(?i)on\w+\s*=\s*[\"'][^\"']*[\"']",
            r"(?i)<iframe[^>]*>",
            r"(?i)(<img[^>]*onerror|<body[^>]*onload)",
            r"(?i)eval\s*\(|setTimeout\s*\(|setInterval\s*\("
        ]
        
        # Command injection patterns
        self.cmd_patterns = [
            r"(?i)(;\s*|\|\s*|&&\s*)(ls|ps|cat|echo|whoami|id|uname)",
            r"(?i)(;\s*|\|\s*|&&\s*)(\.\./){2,}",
            r"(?i)(system\s*\(|exec\s*\(|shell_exec\s*\(|passthru\s*\()",
            r"(?i)(nc|netcat|telnet|ssh)\s+-"
        ]

    async def detect_threats(self, data: Dict[str, Any]) -> List[ThreatEvent]:
        """Detect injection attacks."""
        threats = []
        
        # Check HTTP requests
        http_requests = data.get("http_requests", [])
        
        for request in http_requests:
            # Check parameters, headers, and body
            for location, content in [
                ("parameters", request.get("parameters", {})),
                ("headers", request.get("headers", {})),
                ("body", {"body": request.get("body", "")})
            ]:
                threats.extend(await self._check_injection_patterns(
                    request, location, content
                ))
        
        return threats

    async def _check_injection_patterns(
        self,
        request: Dict[str, Any],
        location: str,
        content: Dict[str, Any]
    ) -> List[ThreatEvent]:
        """Check for injection patterns in content."""
        threats = []
        
        for key, value in content.items():
            if not isinstance(value, str):
                value = str(value)
            
            # Check SQL injection
            for pattern in self.sql_patterns:
                if re.search(pattern, value):
                    threat = ThreatEvent(
                        id=self.generate_threat_id("sql_injection", request.get("source_ip", "unknown")),
                        threat_type=ThreatType.SQL_INJECTION,
                        severity=ThreatSeverity.HIGH,
                        title="SQL Injection Attempt Detected",
                        description=f"SQL injection pattern detected in {location}.{key}",
                        source_ip=request.get("source_ip"),
                        target=request.get("endpoint"),
                        user_agent=request.get("user_agent"),
                        payload=value[:500],  # Truncate long payloads
                        confidence=0.8,
                        risk_score=85.0,
                        evidence={
                            "location": location,
                            "parameter": key,
                            "pattern_matched": pattern,
                            "full_payload": value
                        },
                        remediation_actions=[
                            "Block the source IP",
                            "Review and sanitize input validation",
                            "Implement parameterized queries",
                            "Alert development team"
                        ]
                    )
                    threats.append(threat)
                    break
            
            # Check XSS
            for pattern in self.xss_patterns:
                if re.search(pattern, value):
                    threat = ThreatEvent(
                        id=self.generate_threat_id("xss_attack", request.get("source_ip", "unknown")),
                        threat_type=ThreatType.XSS_ATTACK,
                        severity=ThreatSeverity.MEDIUM,
                        title="Cross-Site Scripting (XSS) Attempt Detected",
                        description=f"XSS pattern detected in {location}.{key}",
                        source_ip=request.get("source_ip"),
                        target=request.get("endpoint"),
                        user_agent=request.get("user_agent"),
                        payload=value[:500],
                        confidence=0.7,
                        risk_score=65.0,
                        evidence={
                            "location": location,
                            "parameter": key,
                            "pattern_matched": pattern,
                            "full_payload": value
                        },
                        remediation_actions=[
                            "Implement output encoding",
                            "Review input validation",
                            "Consider Content Security Policy",
                            "Block the source IP if persistent"
                        ]
                    )
                    threats.append(threat)
                    break
            
            # Check command injection
            for pattern in self.cmd_patterns:
                if re.search(pattern, value):
                    threat = ThreatEvent(
                        id=self.generate_threat_id("cmd_injection", request.get("source_ip", "unknown")),
                        threat_type=ThreatType.PRIVILEGE_ESCALATION,
                        severity=ThreatSeverity.CRITICAL,
                        title="Command Injection Attempt Detected",
                        description=f"Command injection pattern detected in {location}.{key}",
                        source_ip=request.get("source_ip"),
                        target=request.get("endpoint"),
                        user_agent=request.get("user_agent"),
                        payload=value[:500],
                        confidence=0.9,
                        risk_score=95.0,
                        evidence={
                            "location": location,
                            "parameter": key,
                            "pattern_matched": pattern,
                            "full_payload": value
                        },
                        remediation_actions=[
                            "Immediately block the source IP",
                            "Review system for compromise",
                            "Implement strict input validation",
                            "Escalate to security team immediately"
                        ]
                    )
                    threats.append(threat)
                    break
        
        return threats


class AnomalousTrafficDetector(BaseThreatDetector):
    """Detector for anomalous traffic patterns."""

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        super().__init__("anomalous_traffic_detector", config)
        self.traffic_baseline = {}
        self.traffic_history = defaultdict(list)
        self.baseline_window = self.config.get("baseline_window_hours", 24)
        self.anomaly_threshold = self.config.get("anomaly_threshold", 3.0)  # Standard deviations

    async def detect_threats(self, data: Dict[str, Any]) -> List[ThreatEvent]:
        """Detect anomalous traffic patterns."""
        threats = []
        
        # Analyze traffic metrics
        traffic_data = data.get("traffic_metrics", {})
        
        for metric_name, current_value in traffic_data.items():
            threat = await self._check_traffic_anomaly(metric_name, current_value)
            if threat:
                threats.append(threat)
        
        return threats

    async def _check_traffic_anomaly(
        self,
        metric_name: str,
        current_value: float
    ) -> Optional[ThreatEvent]:
        """Check for traffic anomalies."""
        now = datetime.utcnow()
        
        # Add current value to history
        self.traffic_history[metric_name].append((now, current_value))
        
        # Remove old values outside baseline window
        cutoff_time = now - timedelta(hours=self.baseline_window)
        self.traffic_history[metric_name] = [
            (timestamp, value) for timestamp, value in self.traffic_history[metric_name]
            if timestamp > cutoff_time
        ]
        
        # Need at least 10 data points for baseline
        if len(self.traffic_history[metric_name]) < 10:
            return None
        
        # Calculate baseline statistics
        values = [value for _, value in self.traffic_history[metric_name][:-1]]  # Exclude current
        baseline_mean = np.mean(values)
        baseline_std = np.std(values)
        
        # Check for anomaly
        if baseline_std > 0:
            z_score = abs(current_value - baseline_mean) / baseline_std
            
            if z_score > self.anomaly_threshold:
                severity = ThreatSeverity.HIGH if z_score > 5 else ThreatSeverity.MEDIUM
                
                return ThreatEvent(
                    id=self.generate_threat_id("traffic_anomaly", metric_name),
                    threat_type=ThreatType.ANOMALOUS_TRAFFIC,
                    severity=severity,
                    title=f"Anomalous Traffic Pattern: {metric_name}",
                    description=f"Unusual {metric_name} detected: {current_value:.2f} (baseline: {baseline_mean:.2f}±{baseline_std:.2f})",
                    confidence=min(0.9, z_score / 10),  # Higher z-score = higher confidence
                    risk_score=min(100.0, z_score * 15),
                    evidence={
                        "metric": metric_name,
                        "current_value": current_value,
                        "baseline_mean": baseline_mean,
                        "baseline_std": baseline_std,
                        "z_score": z_score,
                        "data_points": len(values)
                    },
                    remediation_actions=[
                        "Investigate traffic source",
                        "Check for DDoS patterns",
                        "Review system performance",
                        "Consider rate limiting"
                    ]
                )
        
        return None


class APIAbuseDetector(BaseThreatDetector):
    """Detector for API abuse patterns."""

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        super().__init__("api_abuse_detector", config)
        self.rate_limits = self.config.get("rate_limits", {
            "default": 100,  # requests per minute
            "sensitive": 10   # for sensitive endpoints
        })
        self.request_history = defaultdict(deque)
        self.sensitive_endpoints = self.config.get("sensitive_endpoints", [
            "/api/v1/admin",
            "/api/v1/users",
            "/api/v1/auth"
        ])

    async def detect_threats(self, data: Dict[str, Any]) -> List[ThreatEvent]:
        """Detect API abuse patterns."""
        threats = []
        
        # Check API requests
        api_requests = data.get("api_requests", [])
        
        for request in api_requests:
            threat = await self._check_rate_abuse(request)
            if threat:
                threats.append(threat)
            
            # Check for suspicious patterns
            pattern_threat = await self._check_suspicious_patterns(request)
            if pattern_threat:
                threats.append(pattern_threat)
        
        return threats

    async def _check_rate_abuse(self, request: Dict[str, Any]) -> Optional[ThreatEvent]:
        """Check for rate limiting abuse."""
        source_ip = request.get("source_ip")
        endpoint = request.get("endpoint", "")
        timestamp = datetime.fromisoformat(request.get("timestamp", datetime.utcnow().isoformat()))
        
        if not source_ip:
            return None
        
        # Determine rate limit
        is_sensitive = any(sensitive in endpoint for sensitive in self.sensitive_endpoints)
        rate_limit = self.rate_limits["sensitive"] if is_sensitive else self.rate_limits["default"]
        
        key = f"{source_ip}_{endpoint}"
        
        # Add current request
        self.request_history[key].append(timestamp)
        
        # Remove requests older than 1 minute
        cutoff_time = timestamp - timedelta(minutes=1)
        while (self.request_history[key] and 
               self.request_history[key][0] < cutoff_time):
            self.request_history[key].popleft()
        
        # Check if rate limit exceeded
        if len(self.request_history[key]) > rate_limit:
            return ThreatEvent(
                id=self.generate_threat_id("api_abuse", source_ip),
                threat_type=ThreatType.API_ABUSE,
                severity=ThreatSeverity.MEDIUM,
                title=f"API Rate Limit Exceeded from {source_ip}",
                description=f"Rate limit exceeded for {endpoint}: {len(self.request_history[key])} requests/minute (limit: {rate_limit})",
                source_ip=source_ip,
                target=endpoint,
                confidence=0.9,
                risk_score=60.0,
                evidence={
                    "request_count": len(self.request_history[key]),
                    "rate_limit": rate_limit,
                    "endpoint": endpoint,
                    "is_sensitive_endpoint": is_sensitive
                },
                remediation_actions=[
                    f"Implement rate limiting for {source_ip}",
                    "Review API usage patterns",
                    "Consider temporary IP blocking",
                    "Monitor for continued abuse"
                ]
            )
        
        return None

    async def _check_suspicious_patterns(self, request: Dict[str, Any]) -> Optional[ThreatEvent]:
        """Check for suspicious API usage patterns."""
        # Check for automated tools in user agent
        user_agent = request.get("user_agent", "").lower()
        suspicious_agents = [
            "bot", "crawler", "spider", "scan", "test", "exploit",
            "nmap", "sqlmap", "burp", "nikto", "gobuster"
        ]
        
        if any(agent in user_agent for agent in suspicious_agents):
            return ThreatEvent(
                id=self.generate_threat_id("suspicious_agent", request.get("source_ip", "unknown")),
                threat_type=ThreatType.SUSPICIOUS_ACTIVITY,
                severity=ThreatSeverity.MEDIUM,
                title="Suspicious User Agent Detected",
                description=f"Potentially malicious user agent: {user_agent}",
                source_ip=request.get("source_ip"),
                target=request.get("endpoint"),
                user_agent=user_agent,
                confidence=0.7,
                risk_score=55.0,
                evidence={
                    "user_agent": user_agent,
                    "matched_patterns": [agent for agent in suspicious_agents if agent in user_agent]
                },
                remediation_actions=[
                    "Monitor source IP for additional suspicious activity",
                    "Review request patterns",
                    "Consider blocking automated tools"
                ]
            )
        
        return None


class ThreatDetectionSystem:
    """Main threat detection system orchestrator."""

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """Initialize threat detection system.
        
        Args:
            config: System configuration dictionary
        """
        self.config = config or {}
        
        # Initialize detectors
        self.detectors = {
            "brute_force": BruteForceDetector(self.config.get("brute_force", {})),
            "injection": InjectionDetector(self.config.get("injection", {})),
            "anomalous_traffic": AnomalousTrafficDetector(self.config.get("anomalous_traffic", {})),
            "api_abuse": APIAbuseDetector(self.config.get("api_abuse", {}))
        }
        
        # Filter enabled detectors
        self.enabled_detectors = {
            name: detector for name, detector in self.detectors.items()
            if detector.enabled
        }
        
        # Alert management
        self.active_alerts = {}
        self.alert_aggregation_window = self.config.get("alert_aggregation_minutes", 5)
        self.max_alerts_per_hour = self.config.get("max_alerts_per_hour", 100)
        
        # Monitoring
        self.monitoring_active = False
        self.monitoring_thread = None
        
        logger.info("ThreatDetectionSystem initialized",
                   enabled_detectors=list(self.enabled_detectors.keys()))

    async def detect_threats(self, data: Dict[str, Any]) -> List[ThreatEvent]:
        """Run threat detection across all enabled detectors.
        
        Args:
            data: Input data for threat detection
            
        Returns:
            List of detected threat events
        """
        all_threats = []
        
        # Run each enabled detector
        for detector_name, detector in self.enabled_detectors.items():
            try:
                threats = await detector.detect_threats(data)
                all_threats.extend(threats)
                
                # Record metrics
                metrics_collector.record_metric(
                    f"security.threats.{detector_name}.count",
                    len(threats),
                    {"detector": detector_name}
                )
                
                logger.debug(f"Detector {detector_name} found {len(threats)} threats")
                
            except Exception as e:
                logger.error(f"Detector {detector_name} failed",
                           error=str(e), detector=detector_name)
        
        # Process and aggregate threats into alerts
        alerts = await self._process_threats_into_alerts(all_threats)
        
        # Record overall metrics
        metrics_collector.record_metric(
            "security.threats.total_count",
            len(all_threats)
        )
        
        metrics_collector.record_metric(
            "security.alerts.total_count",
            len(alerts)
        )
        
        logger.info("Threat detection completed",
                   total_threats=len(all_threats),
                   total_alerts=len(alerts))
        
        return all_threats

    async def _process_threats_into_alerts(self, threats: List[ThreatEvent]) -> List[SecurityAlert]:
        """Process threat events into security alerts."""
        if not threats:
            return []
        
        new_alerts = []
        current_time = datetime.utcnow()
        
        # Group threats by similarity for aggregation
        threat_groups = self._group_similar_threats(threats)
        
        for group_key, group_threats in threat_groups.items():
            # Check if we have an existing alert for this group
            existing_alert = self._find_existing_alert(group_key, group_threats[0])
            
            if existing_alert:
                # Update existing alert
                existing_alert.threat_events.extend(group_threats)
                existing_alert.last_updated = current_time
                
                # Update severity if needed
                max_severity = max(
                    [event.severity for event in existing_alert.threat_events],
                    key=lambda s: ["info", "low", "medium", "high", "critical"].index(s.value)
                )
                existing_alert.severity = max_severity
                
            else:
                # Create new alert
                alert_id = self._generate_alert_id(group_threats[0])
                
                alert = SecurityAlert(
                    id=alert_id,
                    title=self._generate_alert_title(group_threats),
                    description=self._generate_alert_description(group_threats),
                    severity=self._determine_alert_severity(group_threats),
                    threat_events=group_threats,
                    start_time=min(event.timestamp for event in group_threats),
                    last_updated=current_time,
                    affected_systems=list(set(
                        event.target for event in group_threats 
                        if event.target
                    ))
                )
                
                self.active_alerts[alert_id] = alert
                new_alerts.append(alert)
        
        return new_alerts

    def _group_similar_threats(self, threats: List[ThreatEvent]) -> Dict[str, List[ThreatEvent]]:
        """Group similar threats for aggregation."""
        groups = defaultdict(list)
        
        for threat in threats:
            # Create grouping key based on threat characteristics
            group_key = f"{threat.threat_type.value}_{threat.source_ip}_{threat.target}"
            groups[group_key].append(threat)
        
        return dict(groups)

    def _find_existing_alert(self, group_key: str, sample_threat: ThreatEvent) -> Optional[SecurityAlert]:
        """Find existing alert for similar threats."""
        current_time = datetime.utcnow()
        aggregation_window = timedelta(minutes=self.alert_aggregation_window)
        
        for alert in self.active_alerts.values():
            if (alert.status == ThreatStatus.ACTIVE and
                current_time - alert.last_updated < aggregation_window):
                
                # Check if threat matches alert characteristics
                sample_event = alert.threat_events[0]
                if (sample_event.threat_type == sample_threat.threat_type and
                    sample_event.source_ip == sample_threat.source_ip):
                    return alert
        
        return None

    def _generate_alert_id(self, sample_threat: ThreatEvent) -> str:
        """Generate unique alert ID."""
        timestamp = str(int(time.time()))
        content = f"alert_{sample_threat.threat_type.value}_{sample_threat.source_ip}_{timestamp}"
        return hashlib.md5(content.encode()).hexdigest()[:12]

    def _generate_alert_title(self, threats: List[ThreatEvent]) -> str:
        """Generate alert title from threat events."""
        if not threats:
            return "Security Alert"
        
        threat_type = threats[0].threat_type
        source_ip = threats[0].source_ip
        
        if len(threats) == 1:
            return threats[0].title
        else:
            return f"Multiple {threat_type.value.replace('_', ' ').title()} Attempts from {source_ip}"

    def _generate_alert_description(self, threats: List[ThreatEvent]) -> str:
        """Generate alert description from threat events."""
        if not threats:
            return "Security alert generated"
        
        threat_type = threats[0].threat_type
        source_ip = threats[0].source_ip
        
        if len(threats) == 1:
            return threats[0].description
        else:
            return (f"Detected {len(threats)} {threat_type.value.replace('_', ' ')} attempts "
                   f"from {source_ip} within {self.alert_aggregation_window} minutes")

    def _determine_alert_severity(self, threats: List[ThreatEvent]) -> ThreatSeverity:
        """Determine alert severity from threat events."""
        if not threats:
            return ThreatSeverity.LOW
        
        # Use highest severity among threats
        severity_order = {
            ThreatSeverity.CRITICAL: 4,
            ThreatSeverity.HIGH: 3,
            ThreatSeverity.MEDIUM: 2,
            ThreatSeverity.LOW: 1,
            ThreatSeverity.INFO: 0
        }
        
        max_severity = max(threats, key=lambda t: severity_order[t.severity]).severity
        
        # Escalate severity if multiple events
        if len(threats) > 5 and max_severity != ThreatSeverity.CRITICAL:
            severity_levels = list(severity_order.keys())
            current_index = severity_levels.index(max_severity)
            if current_index < len(severity_levels) - 1:
                max_severity = severity_levels[current_index + 1]
        
        return max_severity

    def get_active_alerts(
        self,
        severity_filter: Optional[ThreatSeverity] = None,
        limit: int = 100
    ) -> List[SecurityAlert]:
        """Get active security alerts.
        
        Args:
            severity_filter: Filter by severity level
            limit: Maximum number of alerts to return
            
        Returns:
            List of active security alerts
        """
        alerts = list(self.active_alerts.values())
        
        # Filter by severity if specified
        if severity_filter:
            alerts = [alert for alert in alerts if alert.severity == severity_filter]
        
        # Filter active alerts
        alerts = [alert for alert in alerts if alert.status == ThreatStatus.ACTIVE]
        
        # Sort by severity and last updated
        severity_order = {
            ThreatSeverity.CRITICAL: 4,
            ThreatSeverity.HIGH: 3,
            ThreatSeverity.MEDIUM: 2,
            ThreatSeverity.LOW: 1,
            ThreatSeverity.INFO: 0
        }
        
        alerts.sort(
            key=lambda a: (severity_order[a.severity], a.last_updated),
            reverse=True
        )
        
        return alerts[:limit]

    def resolve_alert(self, alert_id: str, resolution_notes: str = "") -> bool:
        """Resolve a security alert.
        
        Args:
            alert_id: Alert ID to resolve
            resolution_notes: Notes about the resolution
            
        Returns:
            True if alert was resolved, False if not found
        """
        if alert_id in self.active_alerts:
            alert = self.active_alerts[alert_id]
            alert.status = ThreatStatus.RESOLVED
            alert.resolution_notes = resolution_notes
            alert.last_updated = datetime.utcnow()
            
            logger.info("Security alert resolved",
                       alert_id=alert_id,
                       resolution_notes=resolution_notes)
            
            return True
        
        return False

    def generate_threat_report(
        self,
        time_period_hours: int = 24,
        include_resolved: bool = False
    ) -> Dict[str, Any]:
        """Generate comprehensive threat detection report.
        
        Args:
            time_period_hours: Time period for the report
            include_resolved: Include resolved alerts
            
        Returns:
            Comprehensive threat report
        """
        cutoff_time = datetime.utcnow() - timedelta(hours=time_period_hours)
        
        # Filter alerts by time period
        relevant_alerts = [
            alert for alert in self.active_alerts.values()
            if alert.last_updated >= cutoff_time and
            (include_resolved or alert.status == ThreatStatus.ACTIVE)
        ]
        
        # Aggregate statistics
        total_threats = sum(alert.event_count for alert in relevant_alerts)
        threat_types = defaultdict(int)
        severity_counts = defaultdict(int)
        source_ips = defaultdict(int)
        
        for alert in relevant_alerts:
            severity_counts[alert.severity.value] += 1
            
            for event in alert.threat_events:
                threat_types[event.threat_type.value] += 1
                if event.source_ip:
                    source_ips[event.source_ip] += 1
        
        report = {
            "report_metadata": {
                "generated_at": datetime.utcnow().isoformat(),
                "time_period_hours": time_period_hours,
                "include_resolved": include_resolved
            },
            "summary": {
                "total_alerts": len(relevant_alerts),
                "total_threat_events": total_threats,
                "active_alerts": len([a for a in relevant_alerts if a.status == ThreatStatus.ACTIVE]),
                "resolved_alerts": len([a for a in relevant_alerts if a.status == ThreatStatus.RESOLVED]),
                "critical_alerts": severity_counts.get("critical", 0),
                "high_alerts": severity_counts.get("high", 0)
            },
            "threat_analysis": {
                "threat_types": dict(threat_types),
                "severity_distribution": dict(severity_counts),
                "top_source_ips": dict(sorted(source_ips.items(), key=lambda x: x[1], reverse=True)[:10])
            },
            "active_alerts": [
                {
                    "id": alert.id,
                    "title": alert.title,
                    "severity": alert.severity.value,
                    "event_count": alert.event_count,
                    "start_time": alert.start_time.isoformat(),
                    "last_updated": alert.last_updated.isoformat(),
                    "max_risk_score": alert.max_risk_score,
                    "affected_systems": alert.affected_systems
                }
                for alert in relevant_alerts
                if alert.status == ThreatStatus.ACTIVE
            ],
            "recommendations": self._generate_threat_recommendations(relevant_alerts)
        }
        
        return report

    def _generate_threat_recommendations(self, alerts: List[SecurityAlert]) -> List[str]:
        """Generate threat mitigation recommendations."""
        recommendations = []
        
        # Analyze alert patterns
        threat_types = defaultdict(int)
        high_risk_ips = set()
        
        for alert in alerts:
            if alert.status == ThreatStatus.ACTIVE:
                for event in alert.threat_events:
                    threat_types[event.threat_type] += 1
                    if event.risk_score > 80 and event.source_ip:
                        high_risk_ips.add(event.source_ip)
        
        # Generate specific recommendations
        if threat_types.get(ThreatType.BRUTE_FORCE, 0) > 0:
            recommendations.append("Implement account lockout policies and CAPTCHA for login attempts")
        
        if threat_types.get(ThreatType.SQL_INJECTION, 0) > 0:
            recommendations.append("Review and strengthen input validation, use parameterized queries")
        
        if threat_types.get(ThreatType.API_ABUSE, 0) > 0:
            recommendations.append("Implement API rate limiting and authentication controls")
        
        if high_risk_ips:
            recommendations.append(f"Consider blocking {len(high_risk_ips)} high-risk IP addresses")
        
        if not recommendations:
            recommendations.append("Continue monitoring for new threats and maintain security best practices")
        
        return recommendations


# Global threat detection system instance
_threat_detection_system: Optional[ThreatDetectionSystem] = None


def get_threat_detection_system(config: Optional[Dict[str, Any]] = None) -> ThreatDetectionSystem:
    """Get the global threat detection system instance."""
    global _threat_detection_system
    
    if _threat_detection_system is None or config is not None:
        _threat_detection_system = ThreatDetectionSystem(config)
    
    return _threat_detection_system


def initialize_threat_detection_system(config: Optional[Dict[str, Any]] = None) -> ThreatDetectionSystem:
    """Initialize the global threat detection system."""
    global _threat_detection_system
    _threat_detection_system = ThreatDetectionSystem(config)
    return _threat_detection_system