"""
Advanced Threat Detection System for Pynomaly.

This module implements AI-powered threat detection using machine learning,
behavioral analysis, and threat intelligence to identify sophisticated attacks.
"""

from __future__ import annotations

import asyncio
import logging
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any
from uuid import UUID, uuid4

import numpy as np
import pandas as pd
from sklearn.cluster import DBSCAN
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler

logger = logging.getLogger(__name__)


class ThreatSeverity(Enum):
    """Threat severity levels."""

    INFO = "info"
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


class ThreatCategory(Enum):
    """Threat categories."""

    MALWARE = "malware"
    PHISHING = "phishing"
    INTRUSION = "intrusion"
    DATA_EXFILTRATION = "data_exfiltration"
    PRIVILEGE_ESCALATION = "privilege_escalation"
    LATERAL_MOVEMENT = "lateral_movement"
    PERSISTENCE = "persistence"
    COMMAND_CONTROL = "command_control"
    DENIAL_OF_SERVICE = "denial_of_service"
    INSIDER_THREAT = "insider_threat"


class AttackPhase(Enum):
    """MITRE ATT&CK framework phases."""

    RECONNAISSANCE = "reconnaissance"
    RESOURCE_DEVELOPMENT = "resource_development"
    INITIAL_ACCESS = "initial_access"
    EXECUTION = "execution"
    PERSISTENCE = "persistence"
    PRIVILEGE_ESCALATION = "privilege_escalation"
    DEFENSE_EVASION = "defense_evasion"
    CREDENTIAL_ACCESS = "credential_access"
    DISCOVERY = "discovery"
    LATERAL_MOVEMENT = "lateral_movement"
    COLLECTION = "collection"
    COMMAND_CONTROL = "command_control"
    EXFILTRATION = "exfiltration"
    IMPACT = "impact"


@dataclass
class SecurityEvent:
    """Security event for analysis."""

    event_id: UUID = field(default_factory=uuid4)
    timestamp: datetime = field(default_factory=datetime.now)
    event_type: str = ""
    source_ip: str = ""
    destination_ip: str = ""
    user_id: str = ""
    session_id: str = ""
    resource: str = ""
    action: str = ""
    user_agent: str = ""
    request_size: int = 0
    response_code: int = 0
    response_time: float = 0.0
    payload: dict[str, Any] = field(default_factory=dict)
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class ThreatIndicator:
    """Threat indicator of compromise (IoC)."""

    indicator_id: UUID = field(default_factory=uuid4)
    indicator_type: str = ""  # ip, domain, hash, etc.
    value: str = ""
    threat_types: list[ThreatCategory] = field(default_factory=list)
    confidence: float = 0.0
    severity: ThreatSeverity = ThreatSeverity.INFO
    first_seen: datetime = field(default_factory=datetime.now)
    last_seen: datetime = field(default_factory=datetime.now)
    source: str = ""
    context: dict[str, Any] = field(default_factory=dict)


@dataclass
class ThreatAlert:
    """Security threat alert."""

    alert_id: UUID = field(default_factory=uuid4)
    title: str = ""
    description: str = ""
    severity: ThreatSeverity = ThreatSeverity.INFO
    category: ThreatCategory = ThreatCategory.INTRUSION
    attack_phase: AttackPhase = AttackPhase.INITIAL_ACCESS
    confidence: float = 0.0
    risk_score: float = 0.0
    affected_entities: list[str] = field(default_factory=list)
    indicators: list[ThreatIndicator] = field(default_factory=list)
    events: list[SecurityEvent] = field(default_factory=list)
    mitre_techniques: list[str] = field(default_factory=list)
    recommended_actions: list[str] = field(default_factory=list)
    created_at: datetime = field(default_factory=datetime.now)
    status: str = "open"
    investigation_notes: list[dict[str, Any]] = field(default_factory=list)


class MLThreatDetector:
    """Machine learning-based threat detection."""

    def __init__(self):
        self.models = {}
        self.feature_scalers = {}
        self.baseline_profiles = {}
        self.is_trained = False

    async def train_models(self, training_data: pd.DataFrame):
        """Train ML models for threat detection."""
        try:
            logger.info("Training ML threat detection models")

            # Prepare features
            features = self._extract_features(training_data)

            # Train anomaly detection model
            self.models["anomaly"] = IsolationForest(
                contamination=0.1, random_state=42, n_estimators=200
            )

            # Scale features
            self.feature_scalers["anomaly"] = StandardScaler()
            scaled_features = self.feature_scalers["anomaly"].fit_transform(features)

            # Train model
            self.models["anomaly"].fit(scaled_features)

            # Train clustering model for behavioral analysis
            self.models["clustering"] = DBSCAN(eps=0.5, min_samples=5)
            cluster_labels = self.models["clustering"].fit_predict(scaled_features)

            # Create baseline behavioral profiles
            self._create_baseline_profiles(training_data, cluster_labels)

            self.is_trained = True
            logger.info("ML threat detection models trained successfully")

        except Exception as e:
            logger.error(f"Failed to train ML models: {e}")
            raise

    async def detect_threats(self, events: list[SecurityEvent]) -> list[ThreatAlert]:
        """Detect threats using ML models."""
        try:
            if not self.is_trained:
                logger.warning("ML models not trained, using rule-based detection")
                return await self._rule_based_detection(events)

            alerts = []

            # Convert events to DataFrame
            event_data = self._events_to_dataframe(events)

            if event_data.empty:
                return alerts

            # Extract features
            features = self._extract_features(event_data)

            if features.empty:
                return alerts

            # Scale features
            scaled_features = self.feature_scalers["anomaly"].transform(features)

            # Detect anomalies
            anomaly_scores = self.models["anomaly"].decision_function(scaled_features)
            anomaly_predictions = self.models["anomaly"].predict(scaled_features)

            # Analyze each event
            for i, event in enumerate(events):
                if i >= len(anomaly_predictions):
                    continue

                is_anomaly = anomaly_predictions[i] == -1
                anomaly_score = abs(anomaly_scores[i])

                if is_anomaly:
                    alert = await self._create_anomaly_alert(event, anomaly_score)
                    alerts.append(alert)

            # Behavioral analysis
            behavioral_alerts = await self._behavioral_threat_analysis(
                events, scaled_features
            )
            alerts.extend(behavioral_alerts)

            # Pattern-based detection
            pattern_alerts = await self._pattern_based_detection(events)
            alerts.extend(pattern_alerts)

            logger.info(
                f"ML threat detection completed: {len(alerts)} alerts generated"
            )
            return alerts

        except Exception as e:
            logger.error(f"ML threat detection failed: {e}")
            return []

    def _extract_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """Extract features for ML analysis."""
        try:
            features = pd.DataFrame()

            if data.empty:
                return features

            # Time-based features
            if "timestamp" in data.columns:
                data["timestamp"] = pd.to_datetime(data["timestamp"])
                features["hour"] = data["timestamp"].dt.hour
                features["day_of_week"] = data["timestamp"].dt.dayofweek
                features["is_weekend"] = (data["timestamp"].dt.dayofweek >= 5).astype(
                    int
                )
                features["is_business_hours"] = (
                    (data["timestamp"].dt.hour >= 9) & (data["timestamp"].dt.hour <= 17)
                ).astype(int)

            # Request features
            if "request_size" in data.columns:
                features["request_size_log"] = np.log1p(data["request_size"].fillna(0))

            if "response_time" in data.columns:
                features["response_time_log"] = np.log1p(
                    data["response_time"].fillna(0)
                )

            if "response_code" in data.columns:
                features["is_error"] = (data["response_code"] >= 400).astype(int)
                features["is_server_error"] = (data["response_code"] >= 500).astype(int)

            # User behavior features
            if "user_id" in data.columns:
                user_request_counts = data.groupby("user_id").size()
                features["user_request_frequency"] = (
                    data["user_id"].map(user_request_counts).fillna(0)
                )

            # IP-based features
            if "source_ip" in data.columns:
                ip_request_counts = data.groupby("source_ip").size()
                features["ip_request_frequency"] = (
                    data["source_ip"].map(ip_request_counts).fillna(0)
                )

            # Resource access patterns
            if "resource" in data.columns:
                features["resource_sensitivity"] = data["resource"].apply(
                    self._assess_resource_sensitivity
                )

            # Action patterns
            if "action" in data.columns:
                features["action_risk"] = data["action"].apply(self._assess_action_risk)

            # Fill any remaining NaN values
            features = features.fillna(0)

            return features

        except Exception as e:
            logger.error(f"Feature extraction failed: {e}")
            return pd.DataFrame()

    def _assess_resource_sensitivity(self, resource: str) -> float:
        """Assess resource sensitivity level."""
        sensitive_patterns = {
            "/admin/": 1.0,
            "/api/v1/users/": 0.8,
            "/api/v1/models/": 0.7,
            "/api/v1/governance/": 0.9,
            "/api/v1/security/": 0.9,
            "/login": 0.6,
            "/auth/": 0.6,
        }

        for pattern, score in sensitive_patterns.items():
            if pattern in resource:
                return score

        return 0.2  # Default low sensitivity

    def _assess_action_risk(self, action: str) -> float:
        """Assess action risk level."""
        risk_scores = {
            "delete": 1.0,
            "admin": 0.9,
            "execute": 0.8,
            "create": 0.6,
            "update": 0.5,
            "read": 0.2,
            "view": 0.1,
            "list": 0.1,
        }

        return risk_scores.get(action.lower(), 0.3)

    def _events_to_dataframe(self, events: list[SecurityEvent]) -> pd.DataFrame:
        """Convert security events to DataFrame."""
        try:
            data = []
            for event in events:
                row = {
                    "timestamp": event.timestamp,
                    "event_type": event.event_type,
                    "source_ip": event.source_ip,
                    "destination_ip": event.destination_ip,
                    "user_id": event.user_id,
                    "resource": event.resource,
                    "action": event.action,
                    "request_size": event.request_size,
                    "response_code": event.response_code,
                    "response_time": event.response_time,
                }
                data.append(row)

            return pd.DataFrame(data)

        except Exception as e:
            logger.error(f"Failed to convert events to DataFrame: {e}")
            return pd.DataFrame()

    async def _create_anomaly_alert(
        self, event: SecurityEvent, anomaly_score: float
    ) -> ThreatAlert:
        """Create threat alert for anomaly."""
        severity = self._calculate_severity(anomaly_score)
        confidence = min(anomaly_score * 100, 95)  # Convert to percentage, cap at 95%

        alert = ThreatAlert(
            title=f"Anomalous Behavior Detected - {event.event_type}",
            description=f"Unusual {event.event_type} activity detected from {event.source_ip}",
            severity=severity,
            category=ThreatCategory.INTRUSION,
            attack_phase=AttackPhase.INITIAL_ACCESS,
            confidence=confidence,
            risk_score=anomaly_score,
            affected_entities=[event.user_id, event.source_ip],
            events=[event],
            recommended_actions=self._get_recommended_actions(severity),
        )

        return alert

    def _calculate_severity(self, score: float) -> ThreatSeverity:
        """Calculate threat severity based on score."""
        if score >= 0.9:
            return ThreatSeverity.CRITICAL
        elif score >= 0.7:
            return ThreatSeverity.HIGH
        elif score >= 0.5:
            return ThreatSeverity.MEDIUM
        elif score >= 0.3:
            return ThreatSeverity.LOW
        else:
            return ThreatSeverity.INFO

    def _get_recommended_actions(self, severity: ThreatSeverity) -> list[str]:
        """Get recommended actions based on severity."""
        actions = {
            ThreatSeverity.CRITICAL: [
                "Immediately isolate affected systems",
                "Block source IP addresses",
                "Escalate to incident response team",
                "Preserve forensic evidence",
                "Notify executive leadership",
            ],
            ThreatSeverity.HIGH: [
                "Investigate immediately",
                "Consider blocking source IP",
                "Increase monitoring",
                "Notify security team",
                "Review access logs",
            ],
            ThreatSeverity.MEDIUM: [
                "Investigate within 4 hours",
                "Increase user monitoring",
                "Review authentication logs",
                "Consider additional MFA",
            ],
            ThreatSeverity.LOW: [
                "Monitor for additional activity",
                "Review user behavior",
                "Document findings",
            ],
            ThreatSeverity.INFO: [
                "Log for future reference",
                "Include in security metrics",
            ],
        }

        return actions.get(severity, ["Monitor situation"])

    async def _behavioral_threat_analysis(
        self, events: list[SecurityEvent], features: np.ndarray
    ) -> list[ThreatAlert]:
        """Analyze behavioral threats using clustering."""
        alerts = []

        try:
            if len(features) < 2:
                return alerts

            # Perform clustering analysis
            cluster_labels = self.models["clustering"].fit_predict(features)

            # Analyze clusters for anomalous behavior
            unique_labels = np.unique(cluster_labels)

            for label in unique_labels:
                if label == -1:  # Noise points (potential anomalies)
                    cluster_events = [
                        event
                        for i, event in enumerate(events)
                        if i < len(cluster_labels) and cluster_labels[i] == label
                    ]

                    if cluster_events:
                        alert = await self._create_behavioral_alert(cluster_events)
                        alerts.append(alert)

            return alerts

        except Exception as e:
            logger.error(f"Behavioral analysis failed: {e}")
            return alerts

    async def _create_behavioral_alert(
        self, events: list[SecurityEvent]
    ) -> ThreatAlert:
        """Create alert for behavioral anomalies."""
        affected_users = list(set(event.user_id for event in events if event.user_id))
        affected_ips = list(set(event.source_ip for event in events if event.source_ip))

        alert = ThreatAlert(
            title="Behavioral Anomaly Cluster Detected",
            description=f"Cluster of {len(events)} anomalous events detected",
            severity=ThreatSeverity.MEDIUM,
            category=ThreatCategory.INSIDER_THREAT,
            attack_phase=AttackPhase.DISCOVERY,
            confidence=75.0,
            risk_score=0.6,
            affected_entities=affected_users + affected_ips,
            events=events,
            recommended_actions=[
                "Analyze event cluster patterns",
                "Review affected user activities",
                "Check for coordinated attacks",
            ],
        )

        return alert

    async def _pattern_based_detection(
        self, events: list[SecurityEvent]
    ) -> list[ThreatAlert]:
        """Detect threats using pattern analysis."""
        alerts = []

        try:
            # Group events for pattern analysis
            event_groups = self._group_events_for_analysis(events)

            # Detect various attack patterns
            alerts.extend(await self._detect_brute_force_attacks(event_groups))
            alerts.extend(await self._detect_data_exfiltration(event_groups))
            alerts.extend(await self._detect_privilege_escalation(event_groups))
            alerts.extend(await self._detect_lateral_movement(event_groups))

            return alerts

        except Exception as e:
            logger.error(f"Pattern-based detection failed: {e}")
            return alerts

    def _group_events_for_analysis(
        self, events: list[SecurityEvent]
    ) -> dict[str, list[SecurityEvent]]:
        """Group events for pattern analysis."""
        groups = {"by_user": {}, "by_ip": {}, "by_resource": {}}

        for event in events:
            # Group by user
            if event.user_id:
                if event.user_id not in groups["by_user"]:
                    groups["by_user"][event.user_id] = []
                groups["by_user"][event.user_id].append(event)

            # Group by IP
            if event.source_ip:
                if event.source_ip not in groups["by_ip"]:
                    groups["by_ip"][event.source_ip] = []
                groups["by_ip"][event.source_ip].append(event)

            # Group by resource
            if event.resource:
                if event.resource not in groups["by_resource"]:
                    groups["by_resource"][event.resource] = []
                groups["by_resource"][event.resource].append(event)

        return groups

    async def _detect_brute_force_attacks(
        self, event_groups: dict[str, list[SecurityEvent]]
    ) -> list[ThreatAlert]:
        """Detect brute force attacks."""
        alerts = []

        for ip, events in event_groups["by_ip"].items():
            # Look for multiple failed login attempts
            failed_logins = [
                e
                for e in events
                if e.event_type == "authentication" and e.response_code in [401, 403]
            ]

            if len(failed_logins) >= 5:  # Threshold for brute force
                time_window = max(e.timestamp for e in failed_logins) - min(
                    e.timestamp for e in failed_logins
                )

                if time_window.total_seconds() <= 300:  # 5 minutes
                    alert = ThreatAlert(
                        title=f"Brute Force Attack Detected from {ip}",
                        description=f"{len(failed_logins)} failed login attempts in {time_window.total_seconds():.0f} seconds",
                        severity=ThreatSeverity.HIGH,
                        category=ThreatCategory.INTRUSION,
                        attack_phase=AttackPhase.CREDENTIAL_ACCESS,
                        confidence=90.0,
                        risk_score=0.8,
                        affected_entities=[ip],
                        events=failed_logins,
                        mitre_techniques=["T1110"],  # Brute Force
                        recommended_actions=[
                            f"Block IP address {ip}",
                            "Review affected user accounts",
                            "Implement account lockout policies",
                            "Enable additional MFA",
                        ],
                    )
                    alerts.append(alert)

        return alerts

    async def _detect_data_exfiltration(
        self, event_groups: dict[str, list[SecurityEvent]]
    ) -> list[ThreatAlert]:
        """Detect potential data exfiltration."""
        alerts = []

        for user, events in event_groups["by_user"].items():
            # Look for large data access patterns
            large_requests = [e for e in events if e.request_size > 10000000]  # >10MB

            if len(large_requests) >= 3:
                total_size = sum(e.request_size for e in large_requests)

                alert = ThreatAlert(
                    title=f"Potential Data Exfiltration by User {user}",
                    description=f"User accessed {len(large_requests)} large datasets totaling {total_size:,} bytes",
                    severity=ThreatSeverity.HIGH,
                    category=ThreatCategory.DATA_EXFILTRATION,
                    attack_phase=AttackPhase.EXFILTRATION,
                    confidence=75.0,
                    risk_score=0.7,
                    affected_entities=[user],
                    events=large_requests,
                    mitre_techniques=["T1041"],  # Exfiltration Over C2 Channel
                    recommended_actions=[
                        f"Investigate user {user} data access patterns",
                        "Review data classification and access controls",
                        "Monitor network traffic for unusual outbound data",
                        "Consider suspending user access pending investigation",
                    ],
                )
                alerts.append(alert)

        return alerts

    async def _detect_privilege_escalation(
        self, event_groups: dict[str, list[SecurityEvent]]
    ) -> list[ThreatAlert]:
        """Detect privilege escalation attempts."""
        alerts = []

        for user, events in event_groups["by_user"].items():
            # Look for admin resource access by non-admin users
            admin_accesses = [e for e in events if "/admin/" in e.resource]

            if admin_accesses:
                # Check if user typically accesses admin resources
                regular_accesses = [e for e in events if "/admin/" not in e.resource]

                if (
                    len(regular_accesses) > len(admin_accesses) * 2
                ):  # Mostly non-admin access
                    alert = ThreatAlert(
                        title=f"Potential Privilege Escalation by User {user}",
                        description=f"User accessed {len(admin_accesses)} admin resources unexpectedly",
                        severity=ThreatSeverity.HIGH,
                        category=ThreatCategory.PRIVILEGE_ESCALATION,
                        attack_phase=AttackPhase.PRIVILEGE_ESCALATION,
                        confidence=70.0,
                        risk_score=0.8,
                        affected_entities=[user],
                        events=admin_accesses,
                        mitre_techniques=[
                            "T1068"
                        ],  # Exploitation for Privilege Escalation
                        recommended_actions=[
                            f"Review user {user} access permissions",
                            "Audit recent privilege changes",
                            "Check for unauthorized access grants",
                            "Consider revoking elevated privileges",
                        ],
                    )
                    alerts.append(alert)

        return alerts

    async def _detect_lateral_movement(
        self, event_groups: dict[str, list[SecurityEvent]]
    ) -> list[ThreatAlert]:
        """Detect lateral movement patterns."""
        alerts = []

        for user, events in event_groups["by_user"].items():
            # Look for access to multiple different resources in short time
            unique_resources = set(e.resource for e in events)

            if len(unique_resources) >= 10:  # Accessing many different resources
                time_span = max(e.timestamp for e in events) - min(
                    e.timestamp for e in events
                )

                if time_span.total_seconds() <= 3600:  # Within 1 hour
                    alert = ThreatAlert(
                        title=f"Potential Lateral Movement by User {user}",
                        description=f"User accessed {len(unique_resources)} different resources in {time_span.total_seconds()/60:.0f} minutes",
                        severity=ThreatSeverity.MEDIUM,
                        category=ThreatCategory.LATERAL_MOVEMENT,
                        attack_phase=AttackPhase.LATERAL_MOVEMENT,
                        confidence=65.0,
                        risk_score=0.6,
                        affected_entities=[user],
                        events=events,
                        mitre_techniques=["T1021"],  # Remote Services
                        recommended_actions=[
                            f"Monitor user {user} access patterns",
                            "Review resource access justification",
                            "Check for unauthorized lateral access",
                            "Implement network segmentation",
                        ],
                    )
                    alerts.append(alert)

        return alerts

    async def _rule_based_detection(
        self, events: list[SecurityEvent]
    ) -> list[ThreatAlert]:
        """Fallback rule-based detection when ML models aren't available."""
        alerts = []

        for event in events:
            # Simple rule-based checks
            if event.response_code >= 500:
                alert = ThreatAlert(
                    title="Server Error Detected",
                    description=f"Server error {event.response_code} detected",
                    severity=ThreatSeverity.LOW,
                    category=ThreatCategory.DENIAL_OF_SERVICE,
                    confidence=50.0,
                    events=[event],
                )
                alerts.append(alert)

        return alerts

    def _create_baseline_profiles(
        self, training_data: pd.DataFrame, cluster_labels: np.ndarray
    ):
        """Create baseline behavioral profiles."""
        try:
            # Create profiles for normal behavior clusters
            for label in np.unique(cluster_labels):
                if label != -1:  # Exclude noise
                    cluster_data = training_data[cluster_labels == label]

                    profile = {
                        "cluster_id": int(label),
                        "size": len(cluster_data),
                        "typical_hours": cluster_data["timestamp"]
                        .dt.hour.value_counts()
                        .to_dict()
                        if "timestamp" in cluster_data.columns
                        else {},
                        "typical_actions": cluster_data["action"]
                        .value_counts()
                        .to_dict()
                        if "action" in cluster_data.columns
                        else {},
                        "avg_request_size": cluster_data["request_size"].mean()
                        if "request_size" in cluster_data.columns
                        else 0,
                        "avg_response_time": cluster_data["response_time"].mean()
                        if "response_time" in cluster_data.columns
                        else 0,
                    }

                    self.baseline_profiles[label] = profile

            logger.info(
                f"Created {len(self.baseline_profiles)} baseline behavioral profiles"
            )

        except Exception as e:
            logger.error(f"Failed to create baseline profiles: {e}")


class ThreatIntelligencePlatform:
    """Threat intelligence platform for enrichment and context."""

    def __init__(self):
        self.ioc_database = {}
        self.threat_feeds = []
        self.attribution_database = {}

    async def enrich_event(self, event: SecurityEvent) -> dict[str, Any]:
        """Enrich security event with threat intelligence."""
        enrichment = {
            "threat_indicators": [],
            "attribution": {},
            "context": {},
            "risk_score": 0.0,
        }

        try:
            # Check IP reputation
            if event.source_ip:
                ip_intel = await self._check_ip_reputation(event.source_ip)
                if ip_intel:
                    enrichment["threat_indicators"].append(ip_intel)
                    enrichment["risk_score"] += ip_intel.confidence * 0.3

            # Check for known attack patterns
            pattern_intel = await self._check_attack_patterns(event)
            if pattern_intel:
                enrichment["context"].update(pattern_intel)
                enrichment["risk_score"] += 0.2

            # Geolocation analysis
            geo_intel = await self._analyze_geolocation(event.source_ip)
            if geo_intel:
                enrichment["context"]["geolocation"] = geo_intel
                if geo_intel.get("is_high_risk", False):
                    enrichment["risk_score"] += 0.3

            return enrichment

        except Exception as e:
            logger.error(f"Threat intelligence enrichment failed: {e}")
            return enrichment

    async def _check_ip_reputation(self, ip_address: str) -> ThreatIndicator | None:
        """Check IP address reputation."""
        # Check local IOC database
        if ip_address in self.ioc_database:
            return self.ioc_database[ip_address]

        # Simulate threat intelligence lookup
        # In production, integrate with real threat intelligence feeds
        if self._is_suspicious_ip(ip_address):
            indicator = ThreatIndicator(
                indicator_type="ip",
                value=ip_address,
                threat_types=[ThreatCategory.COMMAND_CONTROL],
                confidence=0.7,
                severity=ThreatSeverity.MEDIUM,
                source="threat_intelligence",
            )

            self.ioc_database[ip_address] = indicator
            return indicator

        return None

    def _is_suspicious_ip(self, ip_address: str) -> bool:
        """Simple suspicious IP check (placeholder)."""
        # In production, check against real threat feeds
        suspicious_patterns = [
            "192.0.2.",  # RFC 5737 test network
            "198.51.100.",  # RFC 5737 test network
            "203.0.113.",  # RFC 5737 test network
        ]

        return any(ip_address.startswith(pattern) for pattern in suspicious_patterns)

    async def _check_attack_patterns(self, event: SecurityEvent) -> dict[str, Any]:
        """Check for known attack patterns."""
        patterns = {}

        # SQL injection patterns
        if "payload" in event.metadata:
            payload = str(event.metadata["payload"]).lower()
            sql_patterns = ["union select", "drop table", "1=1", "or 1=1"]

            if any(pattern in payload for pattern in sql_patterns):
                patterns["sql_injection_detected"] = True
                patterns["attack_type"] = "sql_injection"

        # XSS patterns
        if event.user_agent:
            xss_patterns = ["<script>", "javascript:", "onerror="]
            if any(pattern in event.user_agent.lower() for pattern in xss_patterns):
                patterns["xss_detected"] = True
                patterns["attack_type"] = "xss"

        return patterns

    async def _analyze_geolocation(self, ip_address: str) -> dict[str, Any]:
        """Analyze IP geolocation for risk assessment."""
        # Simplified geolocation analysis
        # In production, use real geolocation services

        geo_info = {
            "country": "Unknown",
            "city": "Unknown",
            "is_high_risk": False,
            "is_tor": False,
            "is_vpn": False,
        }

        # Simple country risk assessment
        if ip_address.startswith("192.0.2."):
            geo_info.update(
                {"country": "TestLand", "city": "TestCity", "is_high_risk": True}
            )

        return geo_info


class SecurityOrchestration:
    """Security orchestration and automated response (SOAR)."""

    def __init__(self):
        self.response_playbooks = {}
        self.automated_actions = {}
        self.escalation_rules = {}
        self._load_default_playbooks()

    def _load_default_playbooks(self):
        """Load default response playbooks."""
        self.response_playbooks = {
            "brute_force": {
                "actions": ["block_ip", "notify_soc", "increase_monitoring"],
                "escalation_threshold": "high",
                "auto_execute": True,
            },
            "malware_detected": {
                "actions": [
                    "isolate_system",
                    "notify_incident_response",
                    "preserve_evidence",
                ],
                "escalation_threshold": "critical",
                "auto_execute": False,
            },
            "data_exfiltration": {
                "actions": ["block_user", "notify_dpo", "audit_data_access"],
                "escalation_threshold": "high",
                "auto_execute": False,
            },
        }

    async def respond_to_threat(self, alert: ThreatAlert) -> dict[str, Any]:
        """Orchestrate response to threat alert."""
        response = {
            "alert_id": str(alert.alert_id),
            "actions_taken": [],
            "escalated": False,
            "response_time": datetime.now(),
        }

        try:
            # Determine appropriate playbook
            playbook = self._select_playbook(alert)

            if playbook:
                # Execute automated actions
                if playbook.get("auto_execute", False):
                    for action in playbook["actions"]:
                        result = await self._execute_action(action, alert)
                        response["actions_taken"].append(
                            {
                                "action": action,
                                "result": result,
                                "timestamp": datetime.now(),
                            }
                        )

                # Check escalation criteria
                if self._should_escalate(alert, playbook):
                    await self._escalate_alert(alert)
                    response["escalated"] = True

            logger.info(f"Automated response completed for alert {alert.alert_id}")
            return response

        except Exception as e:
            logger.error(f"Automated response failed: {e}")
            response["error"] = str(e)
            return response

    def _select_playbook(self, alert: ThreatAlert) -> dict[str, Any] | None:
        """Select appropriate response playbook."""
        category_playbook_map = {
            ThreatCategory.INTRUSION: "brute_force",
            ThreatCategory.MALWARE: "malware_detected",
            ThreatCategory.DATA_EXFILTRATION: "data_exfiltration",
        }

        playbook_name = category_playbook_map.get(alert.category)
        if playbook_name:
            return self.response_playbooks.get(playbook_name)

        return None

    async def _execute_action(self, action: str, alert: ThreatAlert) -> str:
        """Execute automated response action."""
        try:
            if action == "block_ip":
                # Block IP addresses
                ips_to_block = [
                    entity
                    for entity in alert.affected_entities
                    if self._is_ip_address(entity)
                ]
                if ips_to_block:
                    # In production, integrate with firewall/WAF
                    logger.info(f"Blocking IPs: {ips_to_block}")
                    return f"Blocked {len(ips_to_block)} IP addresses"

            elif action == "block_user":
                # Block user accounts
                users_to_block = [
                    entity
                    for entity in alert.affected_entities
                    if not self._is_ip_address(entity)
                ]
                if users_to_block:
                    # In production, integrate with identity management
                    logger.info(f"Blocking users: {users_to_block}")
                    return f"Blocked {len(users_to_block)} user accounts"

            elif action == "notify_soc":
                # Notify Security Operations Center
                logger.info(f"SOC notification sent for alert {alert.alert_id}")
                return "SOC notified"

            elif action == "increase_monitoring":
                # Increase monitoring for affected entities
                logger.info(f"Increased monitoring for: {alert.affected_entities}")
                return "Monitoring increased"

            else:
                logger.warning(f"Unknown action: {action}")
                return f"Unknown action: {action}"

        except Exception as e:
            logger.error(f"Action execution failed: {e}")
            return f"Action failed: {str(e)}"

    def _is_ip_address(self, entity: str) -> bool:
        """Check if entity is an IP address."""
        parts = entity.split(".")
        if len(parts) == 4:
            return all(part.isdigit() and 0 <= int(part) <= 255 for part in parts)
        return False

    def _should_escalate(self, alert: ThreatAlert, playbook: dict[str, Any]) -> bool:
        """Determine if alert should be escalated."""
        escalation_threshold = playbook.get("escalation_threshold", "high")

        severity_levels = {"info": 1, "low": 2, "medium": 3, "high": 4, "critical": 5}

        alert_level = severity_levels.get(alert.severity.value, 1)
        threshold_level = severity_levels.get(escalation_threshold, 4)

        return alert_level >= threshold_level

    async def _escalate_alert(self, alert: ThreatAlert):
        """Escalate alert to higher tier."""
        logger.info(
            f"Escalating alert {alert.alert_id} due to severity: {alert.severity.value}"
        )
        # In production, integrate with ticketing system and notification channels


class AdvancedSOC:
    """Advanced Security Operations Center with AI-powered threat detection."""

    def __init__(self):
        self.ml_threat_detector = MLThreatDetector()
        self.threat_intelligence = ThreatIntelligencePlatform()
        self.security_orchestration = SecurityOrchestration()
        self.active_alerts = {}
        self.investigation_queue = []

    async def analyze_security_events(
        self, events: list[SecurityEvent]
    ) -> list[ThreatAlert]:
        """Analyze security events and generate threat alerts."""
        try:
            logger.info(f"Analyzing {len(events)} security events")

            # ML-based threat detection
            ml_alerts = await self.ml_threat_detector.detect_threats(events)

            # Enrich alerts with threat intelligence
            enriched_alerts = []
            for alert in ml_alerts:
                for event in alert.events:
                    enrichment = await self.threat_intelligence.enrich_event(event)

                    # Update alert with enrichment data
                    alert.confidence = min(
                        95, alert.confidence + enrichment["risk_score"] * 10
                    )
                    alert.indicators.extend(enrichment["threat_indicators"])

                    # Add context to alert description
                    if enrichment["context"]:
                        alert.description += f" | Context: {enrichment['context']}"

                enriched_alerts.append(alert)

            # Store active alerts
            for alert in enriched_alerts:
                self.active_alerts[str(alert.alert_id)] = alert

                # Trigger automated response
                await self.security_orchestration.respond_to_threat(alert)

            logger.info(f"Generated {len(enriched_alerts)} threat alerts")
            return enriched_alerts

        except Exception as e:
            logger.error(f"Security event analysis failed: {e}")
            return []

    async def get_security_dashboard(self) -> dict[str, Any]:
        """Get security operations dashboard data."""
        dashboard = {
            "timestamp": datetime.now().isoformat(),
            "active_alerts": len(self.active_alerts),
            "alerts_by_severity": {},
            "alerts_by_category": {},
            "top_threats": [],
            "affected_entities": set(),
            "investigation_queue_size": len(self.investigation_queue),
        }

        # Analyze active alerts
        for alert in self.active_alerts.values():
            # Count by severity
            severity = alert.severity.value
            dashboard["alerts_by_severity"][severity] = (
                dashboard["alerts_by_severity"].get(severity, 0) + 1
            )

            # Count by category
            category = alert.category.value
            dashboard["alerts_by_category"][category] = (
                dashboard["alerts_by_category"].get(category, 0) + 1
            )

            # Collect affected entities
            dashboard["affected_entities"].update(alert.affected_entities)

        # Convert set to list for JSON serialization
        dashboard["affected_entities"] = list(dashboard["affected_entities"])

        # Get top threats by risk score
        sorted_alerts = sorted(
            self.active_alerts.values(), key=lambda x: x.risk_score, reverse=True
        )
        dashboard["top_threats"] = [
            {
                "alert_id": str(alert.alert_id),
                "title": alert.title,
                "severity": alert.severity.value,
                "risk_score": alert.risk_score,
                "confidence": alert.confidence,
            }
            for alert in sorted_alerts[:5]
        ]

        return dashboard


# Example usage
if __name__ == "__main__":

    async def test_threat_detection():
        """Test advanced threat detection system."""
        soc = AdvancedSOC()

        # Create test security events
        test_events = [
            SecurityEvent(
                event_type="authentication",
                source_ip="192.0.2.100",
                user_id="test_user",
                resource="/login",
                action="authenticate",
                response_code=401,
                response_time=0.5,
            ),
            SecurityEvent(
                event_type="api_access",
                source_ip="192.0.2.100",
                user_id="test_user",
                resource="/admin/users",
                action="list",
                response_code=200,
                response_time=1.2,
                request_size=15000000,  # Large request
            ),
        ]

        # Analyze events
        alerts = await soc.analyze_security_events(test_events)

        print("Threat Detection Results:")
        for alert in alerts:
            print(f"- {alert.title} (Severity: {alert.severity.value})")
            print(f"  Confidence: {alert.confidence:.1f}%")
            print(f"  Risk Score: {alert.risk_score:.3f}")
            print(f"  Affected Entities: {alert.affected_entities}")
            print(f"  Recommended Actions: {alert.recommended_actions}")
            print()

        # Get dashboard
        dashboard = await soc.get_security_dashboard()
        print("Security Dashboard:")
        print(f"- Active Alerts: {dashboard['active_alerts']}")
        print(f"- Alerts by Severity: {dashboard['alerts_by_severity']}")
        print(f"- Top Threats: {len(dashboard['top_threats'])}")

    # Run test
    asyncio.run(test_threat_detection())
