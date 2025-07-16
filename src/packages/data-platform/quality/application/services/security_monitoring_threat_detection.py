"""
Security Monitoring and Threat Detection Service - Advanced security monitoring and real-time threat detection.

This service provides comprehensive security monitoring, anomaly detection, threat intelligence,
intrusion detection, and automated incident response for enterprise security.
"""

import asyncio
import logging
import json
from typing import Dict, List, Optional, Any, Union, Tuple, Set
from datetime import datetime, timedelta
from dataclasses import dataclass, field
from enum import Enum
import numpy as np
import pandas as pd
from collections import defaultdict, deque
import hashlib
import ipaddress

from interfaces.data_quality_interface import DataQualityInterface

logger = logging.getLogger(__name__)


class ThreatSeverity(Enum):
    """Threat severity levels."""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


class ThreatType(Enum):
    """Types of security threats."""
    BRUTE_FORCE = "brute_force"
    SQL_INJECTION = "sql_injection"
    XSS = "xss"
    DDoS = "ddos"
    MALWARE = "malware"
    PHISHING = "phishing"
    INSIDER_THREAT = "insider_threat"
    DATA_EXFILTRATION = "data_exfiltration"
    PRIVILEGE_ESCALATION = "privilege_escalation"
    LATERAL_MOVEMENT = "lateral_movement"
    COMMAND_INJECTION = "command_injection"
    SUSPICIOUS_LOGIN = "suspicious_login"
    ANOMALOUS_BEHAVIOR = "anomalous_behavior"
    POLICY_VIOLATION = "policy_violation"


class SecurityEvent(Enum):
    """Types of security events."""
    LOGIN_ATTEMPT = "login_attempt"
    LOGIN_SUCCESS = "login_success"
    LOGIN_FAILURE = "login_failure"
    PERMISSION_DENIED = "permission_denied"
    DATA_ACCESS = "data_access"
    DATA_EXPORT = "data_export"
    CONFIGURATION_CHANGE = "configuration_change"
    FILE_ACCESS = "file_access"
    NETWORK_CONNECTION = "network_connection"
    PROCESS_EXECUTION = "process_execution"
    SYSTEM_ALERT = "system_alert"


@dataclass
class SecurityEventRecord:
    """Security event record."""
    event_id: str
    event_type: SecurityEvent
    timestamp: datetime = field(default_factory=datetime.utcnow)
    user_id: Optional[str] = None
    ip_address: Optional[str] = None
    user_agent: Optional[str] = None
    resource_id: Optional[str] = None
    action: Optional[str] = None
    success: bool = True
    details: Dict[str, Any] = field(default_factory=dict)
    risk_score: float = 0.0
    geolocation: Optional[Dict[str, str]] = None


@dataclass
class ThreatIndicator:
    """Threat indicator definition."""
    indicator_id: str
    name: str
    threat_type: ThreatType
    pattern: str
    severity: ThreatSeverity
    confidence: float = 0.8
    description: str = ""
    created_at: datetime = field(default_factory=datetime.utcnow)
    last_seen: Optional[datetime] = None
    count: int = 0
    is_active: bool = True


@dataclass
class SecurityIncident:
    """Security incident record."""
    incident_id: str
    title: str
    threat_type: ThreatType
    severity: ThreatSeverity
    status: str = "open"  # open, investigating, resolved, closed
    affected_users: Set[str] = field(default_factory=set)
    affected_resources: Set[str] = field(default_factory=set)
    source_events: List[str] = field(default_factory=list)
    indicators: List[str] = field(default_factory=list)
    timeline: List[Dict[str, Any]] = field(default_factory=list)
    remediation_actions: List[str] = field(default_factory=list)
    assigned_to: Optional[str] = None
    created_at: datetime = field(default_factory=datetime.utcnow)
    updated_at: datetime = field(default_factory=datetime.utcnow)
    resolved_at: Optional[datetime] = None


@dataclass
class SecurityBaseline:
    """Security behavior baseline."""
    user_id: str
    typical_login_hours: List[int] = field(default_factory=list)
    typical_ip_ranges: List[str] = field(default_factory=list)
    typical_user_agents: Set[str] = field(default_factory=set)
    average_session_duration: float = 0.0
    typical_resources_accessed: Set[str] = field(default_factory=set)
    login_frequency_per_day: float = 0.0
    data_access_volume_mb: float = 0.0
    last_updated: datetime = field(default_factory=datetime.utcnow)


@dataclass
class SecurityMetrics:
    """Security monitoring metrics."""
    timestamp: datetime = field(default_factory=datetime.utcnow)
    total_events: int = 0
    failed_logins: int = 0
    successful_logins: int = 0
    blocked_attacks: int = 0
    active_incidents: int = 0
    high_risk_events: int = 0
    unique_attackers: int = 0
    threat_indicators_triggered: int = 0
    average_incident_resolution_hours: float = 0.0
    security_score: float = 100.0


class SecurityMonitoringThreatDetection:
    """Comprehensive security monitoring and threat detection service."""
    
    def __init__(self):
        """Initialize the security monitoring service."""
        self.security_events: deque = deque(maxlen=100000)  # Rolling window of events
        self.threat_indicators: Dict[str, ThreatIndicator] = {}
        self.security_incidents: Dict[str, SecurityIncident] = {}
        self.user_baselines: Dict[str, SecurityBaseline] = {}
        self.blocked_ips: Set[str] = set()
        self.security_metrics_history: List[SecurityMetrics] = []
        
        # Configuration
        self.anomaly_threshold = 2.0  # Standard deviations
        self.max_failed_logins_per_hour = 10
        self.max_login_attempts_per_ip = 50
        self.suspicious_login_locations = {"TOR", "VPN", "Anonymous Proxy"}
        self.baseline_learning_days = 30
        
        # Initialize threat indicators
        self._initialize_threat_indicators()
        
        # Start background monitoring tasks
        asyncio.create_task(self._continuous_monitoring_task())
        asyncio.create_task(self._baseline_learning_task())
        asyncio.create_task(self._metrics_collection_task())
        
        logger.info("Security Monitoring and Threat Detection Service initialized")
    
    def _initialize_threat_indicators(self):
        """Initialize default threat indicators."""
        indicators = [
            ThreatIndicator(
                indicator_id="ti_001",
                name="Brute Force Login Attempts",
                threat_type=ThreatType.BRUTE_FORCE,
                pattern="failed_login_attempts > 10 AND time_window < 60_minutes",
                severity=ThreatSeverity.HIGH,
                confidence=0.9,
                description="Multiple failed login attempts from same IP"
            ),
            ThreatIndicator(
                indicator_id="ti_002",
                name="SQL Injection Pattern",
                threat_type=ThreatType.SQL_INJECTION,
                pattern="query CONTAINS ('UNION', 'DROP', '; --', 'OR 1=1')",
                severity=ThreatSeverity.CRITICAL,
                confidence=0.85,
                description="SQL injection patterns detected in input"
            ),
            ThreatIndicator(
                indicator_id="ti_003",
                name="Unusual Login Location",
                threat_type=ThreatType.SUSPICIOUS_LOGIN,
                pattern="login_location NOT IN user_baseline_locations",
                severity=ThreatSeverity.MEDIUM,
                confidence=0.7,
                description="Login from unusual geographic location"
            ),
            ThreatIndicator(
                indicator_id="ti_004",
                name="Off-Hours Data Access",
                threat_type=ThreatType.INSIDER_THREAT,
                pattern="data_access_time NOT IN business_hours AND data_volume > baseline * 3",
                severity=ThreatSeverity.HIGH,
                confidence=0.8,
                description="Large data access outside business hours"
            ),
            ThreatIndicator(
                indicator_id="ti_005",
                name="Privilege Escalation Attempt",
                threat_type=ThreatType.PRIVILEGE_ESCALATION,
                pattern="permission_denied_count > 5 AND escalated_access_attempt = true",
                severity=ThreatSeverity.CRITICAL,
                confidence=0.9,
                description="Multiple privilege escalation attempts"
            ),
            ThreatIndicator(
                indicator_id="ti_006",
                name="Data Exfiltration Pattern",
                threat_type=ThreatType.DATA_EXFILTRATION,
                pattern="data_export_volume > baseline * 10 AND export_frequency > normal",
                severity=ThreatSeverity.CRITICAL,
                confidence=0.85,
                description="Unusual data export patterns suggesting exfiltration"
            ),
            ThreatIndicator(
                indicator_id="ti_007",
                name="DDoS Attack Pattern",
                threat_type=ThreatType.DDoS,
                pattern="request_rate > 1000/minute AND unique_ips < 10",
                severity=ThreatSeverity.HIGH,
                confidence=0.9,
                description="High request rate from few IPs indicating DDoS"
            )
        ]
        
        for indicator in indicators:
            self.threat_indicators[indicator.indicator_id] = indicator
    
    # Error handling would be managed by interface implementation
    async def log_security_event(
        self,
        event_type: SecurityEvent,
        user_id: Optional[str] = None,
        ip_address: Optional[str] = None,
        user_agent: Optional[str] = None,
        resource_id: Optional[str] = None,
        action: Optional[str] = None,
        success: bool = True,
        details: Optional[Dict[str, Any]] = None
    ) -> SecurityEventRecord:
        """
        Log a security event for monitoring and analysis.
        
        Args:
            event_type: Type of security event
            user_id: User involved in the event
            ip_address: Source IP address
            user_agent: User agent string
            resource_id: Resource being accessed
            action: Action being performed
            success: Whether the action was successful
            details: Additional event details
            
        Returns:
            SecurityEventRecord
        """
        import secrets
        
        event_id = f"evt_{secrets.token_hex(8)}"
        
        event = SecurityEventRecord(
            event_id=event_id,
            event_type=event_type,
            user_id=user_id,
            ip_address=ip_address,
            user_agent=user_agent,
            resource_id=resource_id,
            action=action,
            success=success,
            details=details or {},
            geolocation=await self._get_geolocation(ip_address) if ip_address else None
        )
        
        # Calculate risk score
        event.risk_score = await self._calculate_risk_score(event)
        
        # Add to event queue
        self.security_events.append(event)
        
        # Trigger real-time threat detection
        await self._analyze_event_for_threats(event)
        
        return event
    
    async def _get_geolocation(self, ip_address: str) -> Optional[Dict[str, str]]:
        """Get geolocation for IP address (simplified implementation)."""
        try:
            # In production, integrate with actual geolocation service
            ip = ipaddress.ip_address(ip_address)
            
            if ip.is_private:
                return {"country": "Internal", "city": "Private Network", "region": "LAN"}
            
            # Simplified geolocation (would use actual service in production)
            return {
                "country": "Unknown",
                "city": "Unknown", 
                "region": "Unknown",
                "isp": "Unknown"
            }
        except Exception:
            return None
    
    async def _calculate_risk_score(self, event: SecurityEventRecord) -> float:
        """Calculate risk score for a security event."""
        risk_score = 0.0
        
        # Base score based on event type
        event_risk_scores = {
            SecurityEvent.LOGIN_FAILURE: 3.0,
            SecurityEvent.PERMISSION_DENIED: 4.0,
            SecurityEvent.DATA_EXPORT: 6.0,
            SecurityEvent.CONFIGURATION_CHANGE: 7.0,
            SecurityEvent.LOGIN_SUCCESS: 1.0,
            SecurityEvent.DATA_ACCESS: 2.0
        }
        
        risk_score += event_risk_scores.get(event.event_type, 2.0)
        
        # Increase risk for failed events
        if not event.success:
            risk_score *= 2.0
        
        # IP-based risk factors
        if event.ip_address:
            if event.ip_address in self.blocked_ips:
                risk_score += 10.0
            
            # Check for recent failed attempts from same IP
            recent_failures = await self._count_recent_failures_from_ip(event.ip_address)
            risk_score += min(recent_failures * 0.5, 5.0)
        
        # User-based risk factors
        if event.user_id:
            baseline = self.user_baselines.get(event.user_id)
            if baseline:
                # Check against user baseline
                risk_score += await self._calculate_baseline_deviation_risk(event, baseline)
        
        # Time-based risk factors
        current_hour = event.timestamp.hour
        if current_hour < 6 or current_hour > 22:  # Off-hours access
            risk_score += 2.0
        
        return min(risk_score, 10.0)  # Cap at 10.0
    
    async def _count_recent_failures_from_ip(self, ip_address: str) -> int:
        """Count recent failed events from an IP address."""
        cutoff_time = datetime.utcnow() - timedelta(hours=1)
        
        count = 0
        for event in reversed(self.security_events):
            if event.timestamp < cutoff_time:
                break
            if event.ip_address == ip_address and not event.success:
                count += 1
        
        return count
    
    async def _calculate_baseline_deviation_risk(
        self,
        event: SecurityEventRecord,
        baseline: SecurityBaseline
    ) -> float:
        """Calculate risk based on deviation from user baseline."""
        risk = 0.0
        
        # Time-based deviation
        current_hour = event.timestamp.hour
        if current_hour not in baseline.typical_login_hours:
            risk += 1.5
        
        # IP range deviation
        if event.ip_address and baseline.typical_ip_ranges:
            ip_in_range = any(
                self._ip_in_range(event.ip_address, ip_range)
                for ip_range in baseline.typical_ip_ranges
            )
            if not ip_in_range:
                risk += 2.0
        
        # User agent deviation
        if event.user_agent and event.user_agent not in baseline.typical_user_agents:
            risk += 1.0
        
        return risk
    
    def _ip_in_range(self, ip_address: str, ip_range: str) -> bool:
        """Check if IP address is in range (simplified)."""
        try:
            return ipaddress.ip_address(ip_address) in ipaddress.ip_network(ip_range, strict=False)
        except:
            return False
    
    async def _analyze_event_for_threats(self, event: SecurityEventRecord):
        """Analyze event against threat indicators."""
        triggered_indicators = []
        
        for indicator in self.threat_indicators.values():
            if not indicator.is_active:
                continue
            
            if await self._check_indicator_match(event, indicator):
                triggered_indicators.append(indicator)
                indicator.count += 1
                indicator.last_seen = event.timestamp
        
        # Create incident if high-severity indicators triggered
        if triggered_indicators:
            critical_indicators = [i for i in triggered_indicators if i.severity == ThreatSeverity.CRITICAL]
            high_indicators = [i for i in triggered_indicators if i.severity == ThreatSeverity.HIGH]
            
            if critical_indicators or len(high_indicators) >= 2:
                await self._create_security_incident(event, triggered_indicators)
    
    async def _check_indicator_match(self, event: SecurityEventRecord, indicator: ThreatIndicator) -> bool:
        """Check if event matches threat indicator pattern."""
        # Simplified pattern matching (would use proper rule engine in production)
        
        if indicator.threat_type == ThreatType.BRUTE_FORCE:
            if event.event_type == SecurityEvent.LOGIN_FAILURE:
                recent_failures = await self._count_recent_failures_from_ip(event.ip_address)
                return recent_failures >= 10
        
        elif indicator.threat_type == ThreatType.SQL_INJECTION:
            if "query" in event.details:
                query = event.details["query"].lower()
                suspicious_patterns = ["union", "drop", "'; --", "or 1=1", "select *"]
                return any(pattern in query for pattern in suspicious_patterns)
        
        elif indicator.threat_type == ThreatType.SUSPICIOUS_LOGIN:
            if event.event_type == SecurityEvent.LOGIN_SUCCESS:
                if event.geolocation:
                    country = event.geolocation.get("country", "")
                    return country in self.suspicious_login_locations
        
        elif indicator.threat_type == ThreatType.DATA_EXFILTRATION:
            if event.event_type == SecurityEvent.DATA_EXPORT:
                volume = event.details.get("data_volume_mb", 0)
                if event.user_id:
                    baseline = self.user_baselines.get(event.user_id)
                    if baseline and volume > baseline.data_access_volume_mb * 5:
                        return True
        
        elif indicator.threat_type == ThreatType.DDoS:
            if event.ip_address:
                # Count recent requests from this IP
                recent_requests = await self._count_recent_requests_from_ip(event.ip_address)
                return recent_requests > 100  # 100 requests in last minute
        
        return False
    
    async def _count_recent_requests_from_ip(self, ip_address: str) -> int:
        """Count recent requests from an IP address."""
        cutoff_time = datetime.utcnow() - timedelta(minutes=1)
        
        count = 0
        for event in reversed(self.security_events):
            if event.timestamp < cutoff_time:
                break
            if event.ip_address == ip_address:
                count += 1
        
        return count
    
    async def _create_security_incident(
        self,
        triggering_event: SecurityEventRecord,
        indicators: List[ThreatIndicator]
    ):
        """Create a security incident."""
        import secrets
        
        incident_id = f"inc_{secrets.token_hex(8)}"
        
        # Determine severity (highest from indicators)
        severity = max((i.severity for i in indicators), key=lambda s: ["low", "medium", "high", "critical"].index(s.value))
        
        # Determine threat type (most common or highest severity)
        threat_type = indicators[0].threat_type
        
        incident = SecurityIncident(
            incident_id=incident_id,
            title=f"{threat_type.value.replace('_', ' ').title()} Detected",
            threat_type=threat_type,
            severity=severity,
            affected_users={triggering_event.user_id} if triggering_event.user_id else set(),
            affected_resources={triggering_event.resource_id} if triggering_event.resource_id else set(),
            source_events=[triggering_event.event_id],
            indicators=[i.indicator_id for i in indicators],
            timeline=[{
                "timestamp": triggering_event.timestamp.isoformat(),
                "action": "incident_created",
                "description": f"Incident created based on triggered indicators"
            }],
            remediation_actions=await self._generate_remediation_actions(threat_type, severity)
        )
        
        self.security_incidents[incident_id] = incident
        
        # Automatic response for critical incidents
        if severity == ThreatSeverity.CRITICAL:
            await self._execute_automatic_response(incident, triggering_event)
        
        logger.warning(f"Security incident created: {incident_id} - {incident.title}")
    
    async def _generate_remediation_actions(self, threat_type: ThreatType, severity: ThreatSeverity) -> List[str]:
        """Generate remediation actions for a threat."""
        actions = []
        
        if threat_type == ThreatType.BRUTE_FORCE:
            actions.extend([
                "Block source IP address temporarily",
                "Force password reset for affected accounts",
                "Enable additional authentication factors",
                "Review access logs for successful compromises"
            ])
        
        elif threat_type == ThreatType.SQL_INJECTION:
            actions.extend([
                "Block malicious requests at WAF level",
                "Review and patch vulnerable application code",
                "Validate all user inputs",
                "Implement prepared statements"
            ])
        
        elif threat_type == ThreatType.DATA_EXFILTRATION:
            actions.extend([
                "Suspend user account immediately",
                "Review data access patterns",
                "Implement DLP controls",
                "Conduct forensic analysis"
            ])
        
        elif threat_type == ThreatType.DDoS:
            actions.extend([
                "Enable DDoS protection",
                "Rate limit requests per IP",
                "Scale infrastructure if needed",
                "Block attacking IP ranges"
            ])
        
        # Add severity-specific actions
        if severity == ThreatSeverity.CRITICAL:
            actions.extend([
                "Notify security team immediately",
                "Isolate affected systems",
                "Preserve evidence for investigation"
            ])
        
        return actions
    
    async def _execute_automatic_response(self, incident: SecurityIncident, event: SecurityEventRecord):
        """Execute automatic response for critical incidents."""
        logger.info(f"Executing automatic response for critical incident: {incident.incident_id}")
        
        # Block IP if applicable
        if event.ip_address and incident.threat_type in [ThreatType.BRUTE_FORCE, ThreatType.DDoS]:
            self.blocked_ips.add(event.ip_address)
            logger.warning(f"Automatically blocked IP: {event.ip_address}")
        
        # Additional automatic responses can be added here
        
        incident.timeline.append({
            "timestamp": datetime.utcnow().isoformat(),
            "action": "automatic_response_executed",
            "description": "Automatic security response executed"
        })
    
    async def _continuous_monitoring_task(self):
        """Background task for continuous security monitoring."""
        while True:
            try:
                await asyncio.sleep(60)  # Check every minute
                
                # Check for IP addresses with too many failed attempts
                await self._check_for_brute_force_attacks()
                
                # Check for unusual patterns
                await self._check_for_anomalous_patterns()
                
                # Update threat indicators
                await self._update_threat_indicators()
                
            except Exception as e:
                logger.error(f"Continuous monitoring error: {e}")
    
    async def _check_for_brute_force_attacks(self):
        """Check for brute force attack patterns."""
        cutoff_time = datetime.utcnow() - timedelta(hours=1)
        
        # Count failed logins per IP
        ip_failures = defaultdict(int)
        
        for event in reversed(self.security_events):
            if event.timestamp < cutoff_time:
                break
            
            if (event.event_type == SecurityEvent.LOGIN_FAILURE and 
                event.ip_address and 
                not event.success):
                ip_failures[event.ip_address] += 1
        
        # Block IPs with excessive failures
        for ip_address, failure_count in ip_failures.items():
            if failure_count >= self.max_login_attempts_per_ip:
                if ip_address not in self.blocked_ips:
                    self.blocked_ips.add(ip_address)
                    logger.warning(f"Blocked IP {ip_address} for {failure_count} failed login attempts")
    
    async def _check_for_anomalous_patterns(self):
        """Check for anomalous behavior patterns."""
        # This would implement more sophisticated anomaly detection
        # For now, just log that we're checking
        logger.debug("Checking for anomalous patterns")
    
    async def _update_threat_indicators(self):
        """Update threat indicator statistics."""
        for indicator in self.threat_indicators.values():
            # Reset count periodically (daily)
            if indicator.last_seen and (datetime.utcnow() - indicator.last_seen).days >= 1:
                indicator.count = 0
    
    async def _baseline_learning_task(self):
        """Background task for learning user behavior baselines."""
        while True:
            try:
                await asyncio.sleep(3600)  # Update every hour
                await self._update_user_baselines()
            except Exception as e:
                logger.error(f"Baseline learning error: {e}")
    
    async def _update_user_baselines(self):
        """Update user behavior baselines based on historical data."""
        cutoff_time = datetime.utcnow() - timedelta(days=self.baseline_learning_days)
        
        # Group events by user
        user_events = defaultdict(list)
        
        for event in self.security_events:
            if event.timestamp > cutoff_time and event.user_id:
                user_events[event.user_id].append(event)
        
        # Update baselines for each user
        for user_id, events in user_events.items():
            if len(events) < 10:  # Need minimum number of events
                continue
            
            baseline = self.user_baselines.get(user_id, SecurityBaseline(user_id=user_id))
            
            # Update typical login hours
            login_events = [e for e in events if e.event_type == SecurityEvent.LOGIN_SUCCESS]
            if login_events:
                baseline.typical_login_hours = list(set(e.timestamp.hour for e in login_events))
            
            # Update typical IP ranges
            ip_addresses = [e.ip_address for e in events if e.ip_address]
            if ip_addresses:
                # Simplified - in production, would cluster IPs into ranges
                baseline.typical_ip_ranges = list(set(ip_addresses))[:10]  # Keep top 10
            
            # Update typical user agents
            user_agents = [e.user_agent for e in events if e.user_agent]
            if user_agents:
                baseline.typical_user_agents = set(user_agents)
            
            baseline.last_updated = datetime.utcnow()
            self.user_baselines[user_id] = baseline
    
    async def _metrics_collection_task(self):
        """Background task for collecting security metrics."""
        while True:
            try:
                await asyncio.sleep(900)  # Collect every 15 minutes
                metrics = await self._calculate_security_metrics()
                self.security_metrics_history.append(metrics)
                
                # Keep only last 30 days of metrics
                cutoff_time = datetime.utcnow() - timedelta(days=30)
                self.security_metrics_history = [
                    m for m in self.security_metrics_history 
                    if m.timestamp > cutoff_time
                ]
            except Exception as e:
                logger.error(f"Metrics collection error: {e}")
    
    async def _calculate_security_metrics(self) -> SecurityMetrics:
        """Calculate current security metrics."""
        cutoff_time = datetime.utcnow() - timedelta(hours=24)  # Last 24 hours
        
        recent_events = [e for e in self.security_events if e.timestamp > cutoff_time]
        
        total_events = len(recent_events)
        failed_logins = len([e for e in recent_events 
                            if e.event_type == SecurityEvent.LOGIN_FAILURE])
        successful_logins = len([e for e in recent_events 
                               if e.event_type == SecurityEvent.LOGIN_SUCCESS])
        high_risk_events = len([e for e in recent_events if e.risk_score >= 7.0])
        
        active_incidents = len([i for i in self.security_incidents.values() 
                              if i.status in ["open", "investigating"]])
        
        # Calculate security score (simplified)
        security_score = max(0, 100 - (failed_logins * 0.5) - (high_risk_events * 2) - (active_incidents * 10))
        
        return SecurityMetrics(
            total_events=total_events,
            failed_logins=failed_logins,
            successful_logins=successful_logins,
            high_risk_events=high_risk_events,
            active_incidents=active_incidents,
            security_score=security_score
        )
    
    # Error handling would be managed by interface implementation
    async def get_security_dashboard(self) -> Dict[str, Any]:
        """Get comprehensive security monitoring dashboard."""
        current_metrics = await self._calculate_security_metrics()
        
        # Recent high-risk events
        cutoff_time = datetime.utcnow() - timedelta(hours=24)
        high_risk_events = [
            {
                "event_id": e.event_id,
                "event_type": e.event_type.value,
                "risk_score": e.risk_score,
                "user_id": e.user_id,
                "ip_address": e.ip_address,
                "timestamp": e.timestamp.isoformat()
            }
            for e in self.security_events 
            if e.timestamp > cutoff_time and e.risk_score >= 7.0
        ]
        
        # Active incidents
        active_incidents = [
            {
                "incident_id": i.incident_id,
                "title": i.title,
                "threat_type": i.threat_type.value,
                "severity": i.severity.value,
                "status": i.status,
                "created_at": i.created_at.isoformat()
            }
            for i in self.security_incidents.values()
            if i.status in ["open", "investigating"]
        ]
        
        return {
            "timestamp": datetime.utcnow().isoformat(),
            "security_metrics": {
                "security_score": current_metrics.security_score,
                "total_events_24h": current_metrics.total_events,
                "failed_logins_24h": current_metrics.failed_logins,
                "successful_logins_24h": current_metrics.successful_logins,
                "high_risk_events_24h": current_metrics.high_risk_events,
                "active_incidents": current_metrics.active_incidents,
                "blocked_ips": len(self.blocked_ips)
            },
            "threat_detection": {
                "total_indicators": len(self.threat_indicators),
                "active_indicators": len([i for i in self.threat_indicators.values() if i.is_active]),
                "indicators_triggered_24h": len([i for i in self.threat_indicators.values() 
                                               if i.last_seen and i.last_seen > cutoff_time])
            },
            "incident_management": {
                "total_incidents": len(self.security_incidents),
                "open_incidents": len([i for i in self.security_incidents.values() if i.status == "open"]),
                "critical_incidents": len([i for i in self.security_incidents.values() 
                                         if i.severity == ThreatSeverity.CRITICAL and i.status in ["open", "investigating"]])
            },
            "user_behavior": {
                "users_with_baselines": len(self.user_baselines),
                "baseline_learning_enabled": True
            },
            "recent_high_risk_events": high_risk_events[-10:],  # Last 10
            "active_incidents": active_incidents,
            "threat_types": [t.value for t in ThreatType],
            "supported_event_types": [e.value for e in SecurityEvent]
        }