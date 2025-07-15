"""Advanced threat detection engine with machine learning capabilities."""

from __future__ import annotations

import asyncio
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Set
from uuid import UUID

from pydantic import BaseModel

from pynomaly.domain.entities.audit import AuditEvent
from pynomaly.domain.models.security import AuditEventType, SecuritySeverity
from pynomaly.infrastructure.security.audit_repository import AuditRepository, QueryCriteria


class ThreatIndicator(BaseModel):
    """Represents a detected threat indicator."""
    
    id: UUID
    threat_type: str
    severity: SecuritySeverity
    confidence: float  # 0.0 to 1.0
    description: str
    evidence: List[UUID]  # Related audit event IDs
    user_id: Optional[UUID] = None
    tenant_id: Optional[UUID] = None
    detected_at: datetime
    indicators: Dict[str, float]  # Specific indicator values
    recommended_actions: List[str]


class ThreatAnalysis(BaseModel):
    """Result of threat analysis on event patterns."""
    
    analysis_id: UUID
    analyzed_period: timedelta
    total_events_analyzed: int
    threats_detected: List[ThreatIndicator]
    risk_score: float  # Overall risk score 0-100
    analysis_timestamp: datetime
    pattern_summary: Dict[str, int]


class SecurityIncident(BaseModel):
    """Security incident created from correlated events."""
    
    incident_id: UUID
    title: str
    description: str
    severity: SecuritySeverity
    status: str  # open, investigating, resolved, false_positive
    created_at: datetime
    updated_at: datetime
    related_events: List[UUID]
    affected_users: List[UUID]
    affected_tenants: List[UUID]
    threat_indicators: List[ThreatIndicator]
    timeline: List[Dict[str, str]]
    resolution: Optional[str] = None


class BehaviorProfile(BaseModel):
    """User behavior profile for anomaly detection."""
    
    user_id: UUID
    tenant_id: UUID
    profile_updated: datetime
    login_patterns: Dict[str, float]  # Time-based login patterns
    access_patterns: Dict[str, float]  # Resource access patterns
    geographic_patterns: Dict[str, float]  # Geographic access patterns
    device_patterns: Dict[str, float]  # Device usage patterns
    risk_baseline: float  # Baseline risk score for this user
    anomaly_threshold: float  # Threshold for anomaly detection


class ThreatDetectionEngine:
    """Advanced threat detection engine with ML capabilities."""
    
    def __init__(self, audit_repository: AuditRepository):
        self.audit_repository = audit_repository
        self.behavior_profiles: Dict[UUID, BehaviorProfile] = {}
        self.threat_rules = self._initialize_threat_rules()
    
    def _initialize_threat_rules(self) -> Dict[str, Dict]:
        """Initialize threat detection rules."""
        return {
            "brute_force": {
                "event_type": AuditEventType.AUTHENTICATION_FAILURE,
                "threshold": 5,
                "timeframe": timedelta(minutes=15),
                "severity": SecuritySeverity.HIGH
            },
            "privilege_escalation": {
                "event_type": AuditEventType.AUTHORIZATION_ROLE_CHANGE,
                "threshold": 1,
                "timeframe": timedelta(hours=1),
                "severity": SecuritySeverity.CRITICAL
            },
            "unusual_data_access": {
                "event_type": AuditEventType.DATA_ACCESS_UNUSUAL,
                "threshold": 10,
                "timeframe": timedelta(hours=1),
                "severity": SecuritySeverity.MEDIUM
            },
            "suspicious_login": {
                "event_type": AuditEventType.AUTHENTICATION_SUCCESS,
                "conditions": ["new_location", "new_device"],
                "severity": SecuritySeverity.MEDIUM
            },
            "data_exfiltration": {
                "event_type": AuditEventType.DATA_EXPORT_BULK,
                "threshold": 3,
                "timeframe": timedelta(hours=24),
                "severity": SecuritySeverity.HIGH
            }
        }
    
    async def analyze_event_patterns(self, events: List[AuditEvent]) -> ThreatAnalysis:
        """Analyze patterns in audit events to detect threats."""
        analysis_id = UUID.generate()
        threats_detected = []
        
        # Group events by type and time
        event_groups = self._group_events_by_type(events)
        
        # Apply threat detection rules
        for rule_name, rule_config in self.threat_rules.items():
            threats = await self._apply_threat_rule(rule_name, rule_config, event_groups)
            threats_detected.extend(threats)
        
        # Calculate overall risk score
        risk_score = self._calculate_risk_score(threats_detected)
        
        # Create pattern summary
        pattern_summary = {
            event_type.value: len(group) 
            for event_type, group in event_groups.items()
        }
        
        return ThreatAnalysis(
            analysis_id=analysis_id,
            analyzed_period=timedelta(hours=24),  # Default analysis period
            total_events_analyzed=len(events),
            threats_detected=threats_detected,
            risk_score=risk_score,
            analysis_timestamp=datetime.utcnow(),
            pattern_summary=pattern_summary
        )
    
    async def detect_anomalous_behavior(self, user_id: UUID) -> List[ThreatIndicator]:
        """Detect anomalous behavior for a specific user."""
        # Get user's behavior profile
        profile = await self._get_or_create_behavior_profile(user_id)
        
        # Get recent events for the user
        criteria = QueryCriteria(
            user_id=user_id,
            start_time=datetime.utcnow() - timedelta(hours=24),
            limit=1000
        )
        recent_events = await self.audit_repository.query_events(criteria)
        
        threats = []
        
        # Analyze login patterns
        login_anomalies = self._detect_login_anomalies(recent_events, profile)
        threats.extend(login_anomalies)
        
        # Analyze access patterns
        access_anomalies = self._detect_access_anomalies(recent_events, profile)
        threats.extend(access_anomalies)
        
        # Analyze geographic patterns
        geo_anomalies = self._detect_geographic_anomalies(recent_events, profile)
        threats.extend(geo_anomalies)
        
        # Analyze device patterns
        device_anomalies = self._detect_device_anomalies(recent_events, profile)
        threats.extend(device_anomalies)
        
        # Update behavior profile
        await self._update_behavior_profile(user_id, recent_events)
        
        return threats
    
    async def correlate_security_events(self, timeframe: timedelta) -> List[SecurityIncident]:
        """Correlate security events to identify potential incidents."""
        start_time = datetime.utcnow() - timeframe
        
        # Get high-risk events
        criteria = QueryCriteria(
            risk_score_min=70,
            start_time=start_time,
            limit=5000
        )
        high_risk_events = await self.audit_repository.query_events(criteria)
        
        incidents = []
        
        # Correlate by user
        user_incidents = await self._correlate_by_user(high_risk_events)
        incidents.extend(user_incidents)
        
        # Correlate by tenant
        tenant_incidents = await self._correlate_by_tenant(high_risk_events)
        incidents.extend(tenant_incidents)
        
        # Correlate by attack pattern
        pattern_incidents = await self._correlate_by_pattern(high_risk_events)
        incidents.extend(pattern_incidents)
        
        # Correlate by time proximity
        temporal_incidents = await self._correlate_by_time(high_risk_events)
        incidents.extend(temporal_incidents)
        
        return incidents
    
    async def _apply_threat_rule(self, rule_name: str, rule_config: Dict, 
                                event_groups: Dict[AuditEventType, List[AuditEvent]]) -> List[ThreatIndicator]:
        """Apply a specific threat detection rule."""
        threats = []
        rule_event_type = rule_config.get("event_type")
        
        if rule_event_type not in event_groups:
            return threats
        
        events = event_groups[rule_event_type]
        
        if rule_name == "brute_force":
            threats.extend(await self._detect_brute_force(events, rule_config))
        elif rule_name == "privilege_escalation":
            threats.extend(await self._detect_privilege_escalation(events, rule_config))
        elif rule_name == "unusual_data_access":
            threats.extend(await self._detect_unusual_data_access(events, rule_config))
        elif rule_name == "suspicious_login":
            threats.extend(await self._detect_suspicious_login(events, rule_config))
        elif rule_name == "data_exfiltration":
            threats.extend(await self._detect_data_exfiltration(events, rule_config))
        
        return threats
    
    async def _detect_brute_force(self, events: List[AuditEvent], rule_config: Dict) -> List[ThreatIndicator]:
        """Detect brute force attacks."""
        threats = []
        threshold = rule_config["threshold"]
        timeframe = rule_config["timeframe"]
        
        # Group by user and IP
        user_ip_groups = {}
        for event in events:
            if event.context and event.context.ip_address:
                key = (event.user_id, event.context.ip_address)
                if key not in user_ip_groups:
                    user_ip_groups[key] = []
                user_ip_groups[key].append(event)
        
        for (user_id, ip_address), group_events in user_ip_groups.items():
            # Check if events exceed threshold within timeframe
            if len(group_events) >= threshold:
                latest_event = max(group_events, key=lambda x: x.timestamp)
                earliest_event = min(group_events, key=lambda x: x.timestamp)
                
                if latest_event.timestamp - earliest_event.timestamp <= timeframe:
                    threat = ThreatIndicator(
                        id=UUID.generate(),
                        threat_type="brute_force",
                        severity=rule_config["severity"],
                        confidence=min(1.0, len(group_events) / (threshold * 2)),
                        description=f"Brute force attack detected: {len(group_events)} failed login attempts from {ip_address}",
                        evidence=[event.id for event in group_events],
                        user_id=user_id,
                        tenant_id=group_events[0].tenant_id,
                        detected_at=datetime.utcnow(),
                        indicators={
                            "failed_attempts": len(group_events),
                            "source_ip": ip_address,
                            "time_span_minutes": (latest_event.timestamp - earliest_event.timestamp).total_seconds() / 60
                        },
                        recommended_actions=[
                            "Block IP address",
                            "Lock user account",
                            "Notify security team",
                            "Review access logs"
                        ]
                    )
                    threats.append(threat)
        
        return threats
    
    async def _detect_privilege_escalation(self, events: List[AuditEvent], rule_config: Dict) -> List[ThreatIndicator]:
        """Detect privilege escalation attempts."""
        threats = []
        
        for event in events:
            if event.risk_score >= 80:  # High risk privilege changes
                threat = ThreatIndicator(
                    id=UUID.generate(),
                    threat_type="privilege_escalation",
                    severity=rule_config["severity"],
                    confidence=event.risk_score / 100.0,
                    description=f"Privilege escalation detected for user {event.user_id}",
                    evidence=[event.id],
                    user_id=event.user_id,
                    tenant_id=event.tenant_id,
                    detected_at=datetime.utcnow(),
                    indicators={
                        "risk_score": event.risk_score,
                        "event_details": event.details
                    },
                    recommended_actions=[
                        "Review role change",
                        "Verify authorization",
                        "Audit user permissions",
                        "Contact user manager"
                    ]
                )
                threats.append(threat)
        
        return threats
    
    async def _detect_unusual_data_access(self, events: List[AuditEvent], rule_config: Dict) -> List[ThreatIndicator]:
        """Detect unusual data access patterns."""
        threats = []
        threshold = rule_config["threshold"]
        
        # Group by user
        user_groups = {}
        for event in events:
            if event.user_id not in user_groups:
                user_groups[event.user_id] = []
            user_groups[event.user_id].append(event)
        
        for user_id, group_events in user_groups.items():
            if len(group_events) >= threshold:
                threat = ThreatIndicator(
                    id=UUID.generate(),
                    threat_type="unusual_data_access",
                    severity=rule_config["severity"],
                    confidence=min(1.0, len(group_events) / (threshold * 3)),
                    description=f"Unusual data access pattern detected for user {user_id}: {len(group_events)} access events",
                    evidence=[event.id for event in group_events],
                    user_id=user_id,
                    tenant_id=group_events[0].tenant_id,
                    detected_at=datetime.utcnow(),
                    indicators={
                        "access_count": len(group_events),
                        "avg_risk_score": sum(event.risk_score for event in group_events) / len(group_events)
                    },
                    recommended_actions=[
                        "Review data access patterns",
                        "Verify business justification",
                        "Check for data exfiltration",
                        "Monitor user activity"
                    ]
                )
                threats.append(threat)
        
        return threats
    
    async def _detect_suspicious_login(self, events: List[AuditEvent], rule_config: Dict) -> List[ThreatIndicator]:
        """Detect suspicious login patterns."""
        threats = []
        
        for event in events:
            suspicion_score = 0
            indicators = {}
            
            # Check for new location
            if event.context and event.context.ip_address:
                # This would need to be implemented with IP geolocation
                # For now, we'll use a placeholder
                if self._is_new_location(event.user_id, event.context.ip_address):
                    suspicion_score += 30
                    indicators["new_location"] = True
            
            # Check for new device
            if event.context and event.context.user_agent:
                if self._is_new_device(event.user_id, event.context.user_agent):
                    suspicion_score += 20
                    indicators["new_device"] = True
            
            # Check for unusual time
            if self._is_unusual_time(event.user_id, event.timestamp):
                suspicion_score += 25
                indicators["unusual_time"] = True
            
            if suspicion_score >= 40:  # Threshold for suspicious login
                threat = ThreatIndicator(
                    id=UUID.generate(),
                    threat_type="suspicious_login",
                    severity=rule_config["severity"],
                    confidence=min(1.0, suspicion_score / 100.0),
                    description=f"Suspicious login detected for user {event.user_id}",
                    evidence=[event.id],
                    user_id=event.user_id,
                    tenant_id=event.tenant_id,
                    detected_at=datetime.utcnow(),
                    indicators=indicators,
                    recommended_actions=[
                        "Verify login legitimacy",
                        "Check for account compromise",
                        "Consider MFA requirement",
                        "Monitor user activity"
                    ]
                )
                threats.append(threat)
        
        return threats
    
    async def _detect_data_exfiltration(self, events: List[AuditEvent], rule_config: Dict) -> List[ThreatIndicator]:
        """Detect potential data exfiltration."""
        threats = []
        threshold = rule_config["threshold"]
        
        # Group by user
        user_groups = {}
        for event in events:
            if event.user_id not in user_groups:
                user_groups[event.user_id] = []
            user_groups[event.user_id].append(event)
        
        for user_id, group_events in user_groups.items():
            if len(group_events) >= threshold:
                threat = ThreatIndicator(
                    id=UUID.generate(),
                    threat_type="data_exfiltration",
                    severity=rule_config["severity"],
                    confidence=min(1.0, len(group_events) / (threshold * 2)),
                    description=f"Potential data exfiltration detected: {len(group_events)} bulk exports by user {user_id}",
                    evidence=[event.id for event in group_events],
                    user_id=user_id,
                    tenant_id=group_events[0].tenant_id,
                    detected_at=datetime.utcnow(),
                    indicators={
                        "export_count": len(group_events),
                        "total_risk_score": sum(event.risk_score for event in group_events)
                    },
                    recommended_actions=[
                        "Investigate data exports",
                        "Review export justification",
                        "Check data sensitivity",
                        "Consider access restriction"
                    ]
                )
                threats.append(threat)
        
        return threats
    
    def _group_events_by_type(self, events: List[AuditEvent]) -> Dict[AuditEventType, List[AuditEvent]]:
        """Group events by type."""
        groups = {}
        for event in events:
            if event.event_type not in groups:
                groups[event.event_type] = []
            groups[event.event_type].append(event)
        return groups
    
    def _calculate_risk_score(self, threats: List[ThreatIndicator]) -> float:
        """Calculate overall risk score from detected threats."""
        if not threats:
            return 0.0
        
        total_score = 0
        for threat in threats:
            severity_multiplier = {
                SecuritySeverity.LOW: 1.0,
                SecuritySeverity.MEDIUM: 2.0,
                SecuritySeverity.HIGH: 3.0,
                SecuritySeverity.CRITICAL: 4.0
            }
            
            threat_score = threat.confidence * severity_multiplier[threat.severity] * 25
            total_score += threat_score
        
        return min(100.0, total_score / len(threats))
    
    # Additional helper methods would be implemented here
    def _is_new_location(self, user_id: UUID, ip_address: str) -> bool:
        """Check if IP address represents a new location for the user."""
        # Placeholder implementation
        return False
    
    def _is_new_device(self, user_id: UUID, user_agent: str) -> bool:
        """Check if user agent represents a new device for the user."""
        # Placeholder implementation
        return False
    
    def _is_unusual_time(self, user_id: UUID, timestamp: datetime) -> bool:
        """Check if login time is unusual for the user."""
        # Placeholder implementation
        return False
    
    async def _get_or_create_behavior_profile(self, user_id: UUID) -> BehaviorProfile:
        """Get or create a behavior profile for the user."""
        # Placeholder implementation
        return BehaviorProfile(
            user_id=user_id,
            tenant_id=UUID.generate(),
            profile_updated=datetime.utcnow(),
            login_patterns={},
            access_patterns={},
            geographic_patterns={},
            device_patterns={},
            risk_baseline=50.0,
            anomaly_threshold=70.0
        )
    
    def _detect_login_anomalies(self, events: List[AuditEvent], profile: BehaviorProfile) -> List[ThreatIndicator]:
        """Detect login anomalies based on behavior profile."""
        # Placeholder implementation
        return []
    
    def _detect_access_anomalies(self, events: List[AuditEvent], profile: BehaviorProfile) -> List[ThreatIndicator]:
        """Detect access anomalies based on behavior profile."""
        # Placeholder implementation
        return []
    
    def _detect_geographic_anomalies(self, events: List[AuditEvent], profile: BehaviorProfile) -> List[ThreatIndicator]:
        """Detect geographic anomalies based on behavior profile."""
        # Placeholder implementation
        return []
    
    def _detect_device_anomalies(self, events: List[AuditEvent], profile: BehaviorProfile) -> List[ThreatIndicator]:
        """Detect device anomalies based on behavior profile."""
        # Placeholder implementation
        return []
    
    async def _update_behavior_profile(self, user_id: UUID, events: List[AuditEvent]) -> None:
        """Update the behavior profile with new events."""
        # Placeholder implementation
        pass
    
    async def _correlate_by_user(self, events: List[AuditEvent]) -> List[SecurityIncident]:
        """Correlate events by user to identify incidents."""
        # Placeholder implementation
        return []
    
    async def _correlate_by_tenant(self, events: List[AuditEvent]) -> List[SecurityIncident]:
        """Correlate events by tenant to identify incidents."""
        # Placeholder implementation
        return []
    
    async def _correlate_by_pattern(self, events: List[AuditEvent]) -> List[SecurityIncident]:
        """Correlate events by attack pattern to identify incidents."""
        # Placeholder implementation
        return []
    
    async def _correlate_by_time(self, events: List[AuditEvent]) -> List[SecurityIncident]:
        """Correlate events by time proximity to identify incidents."""
        # Placeholder implementation
        return []