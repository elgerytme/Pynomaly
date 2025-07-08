"""Advanced threat detection and behavioral analysis for security hardening."""

from __future__ import annotations

import hashlib
import ipaddress
import logging
import time
from collections import defaultdict, deque
from dataclasses import dataclass, field
from datetime import UTC, datetime
from enum import Enum
from typing import Any

from .audit_logger import SecurityEventType
from .security_monitor import AlertType, SecurityAlert, ThreatDetector, ThreatLevel
from .threat_detection_config import get_threat_detection_manager

logger = logging.getLogger(__name__)


class ThreatIntelligenceSource(str, Enum):
    """Threat intelligence data sources."""

    MALWARE_DOMAINS = "malware_domains"
    KNOWN_BAD_IPS = "known_bad_ips"
    TOR_EXIT_NODES = "tor_exit_nodes"
    COMPROMISED_ACCOUNTS = "compromised_accounts"
    SUSPICIOUS_USER_AGENTS = "suspicious_user_agents"


@dataclass
class ThreatIntelligence:
    """Threat intelligence data."""

    source: ThreatIntelligenceSource
    indicator: str
    threat_type: str
    confidence: float  # 0.0 - 1.0
    first_seen: datetime
    last_updated: datetime
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class BehaviorProfile:
    """User behavior profile for anomaly detection."""

    user_id: str

    # Login patterns
    typical_login_hours: set[int] = field(default_factory=set)
    typical_ips: set[str] = field(default_factory=set)
    typical_user_agents: set[str] = field(default_factory=set)
    typical_locations: set[str] = field(default_factory=set)

    # API usage patterns
    typical_endpoints: dict[str, int] = field(default_factory=dict)
    avg_requests_per_hour: float = 0.0
    max_requests_per_hour: int = 0

    # Data access patterns
    typical_data_size: float = 0.0  # Average data accessed in MB
    max_data_accessed: float = 0.0  # Maximum data accessed in one session

    # Profile metadata
    created_at: datetime = field(default_factory=lambda: datetime.now(UTC))
    last_updated: datetime = field(default_factory=lambda: datetime.now(UTC))
    confidence_score: float = 0.0  # How well established this profile is
    sample_count: int = 0


class AdvancedBehaviorAnalyzer(ThreatDetector):
    """Advanced behavioral analysis threat detector."""

    def __init__(self):
        super().__init__("advanced_behavior")
        self.user_profiles: dict[str, BehaviorProfile] = {}
        self.session_data: dict[str, dict[str, Any]] = defaultdict(dict)

        # Load configuration from manager
        config_manager = get_threat_detection_manager()
        detector_config = config_manager.get_detector_config("advanced_behavior")
        if detector_config:
            self.learning_period_days = detector_config.config.get(
                "learning_period_days", 14
            )
            self.anomaly_threshold = detector_config.config.get(
                "anomaly_threshold", 0.7
            )
            self.min_samples_for_profile = detector_config.config.get(
                "min_samples_for_profile", 50
            )
        else:
            # Default configuration
            self.learning_period_days = 14
            self.anomaly_threshold = 0.7
            self.min_samples_for_profile = 50

    async def analyze(self, event_data: dict[str, Any]) -> SecurityAlert | None:
        """Analyze for advanced behavioral anomalies."""
        user_id = event_data.get("user_id")
        if not user_id:
            return None

        event_data.get("event_type")

        # Update behavior profile
        await self._update_behavior_profile(user_id, event_data)

        # Analyze for anomalies
        profile = self.user_profiles.get(user_id)
        if not profile or profile.sample_count < self.min_samples_for_profile:
            return None  # Not enough data for analysis

        anomalies = await self._detect_behavioral_anomalies(
            user_id, event_data, profile
        )

        if (
            anomalies
            and self._calculate_anomaly_confidence(anomalies) > self.anomaly_threshold
        ):
            return SecurityAlert(
                alert_id=f"behavior_{user_id}_{int(time.time())}",
                alert_type=AlertType.UNUSUAL_BEHAVIOR,
                threat_level=self._calculate_threat_level(anomalies),
                title="Unusual Behavior Pattern Detected",
                description=f"User {user_id} showing atypical behavior patterns",
                timestamp=datetime.now(UTC),
                user_id=user_id,
                source_ip=event_data.get("ip_address"),
                user_agent=event_data.get("user_agent"),
                indicators={
                    "anomaly_count": len(anomalies),
                    "confidence_score": self._calculate_anomaly_confidence(anomalies),
                    "profile_confidence": profile.confidence_score,
                },
                evidence=anomalies,
                recommended_actions=[
                    "Verify user identity",
                    "Review recent user activity",
                    "Check for account compromise",
                    "Consider requiring re-authentication",
                    "Monitor user closely",
                ],
            )

        return None

    async def _update_behavior_profile(
        self, user_id: str, event_data: dict[str, Any]
    ) -> None:
        """Update user behavior profile with new event data."""
        if user_id not in self.user_profiles:
            self.user_profiles[user_id] = BehaviorProfile(user_id=user_id)

        profile = self.user_profiles[user_id]
        event_type = event_data.get("event_type")
        current_time = datetime.now(UTC)

        # Update login patterns
        if event_type == SecurityEventType.AUTH_LOGIN_SUCCESS:
            profile.typical_login_hours.add(current_time.hour)

            if event_data.get("ip_address"):
                profile.typical_ips.add(event_data["ip_address"])

            if event_data.get("user_agent"):
                # Store hash of user agent to save space
                ua_hash = hashlib.sha256(event_data["user_agent"].encode()).hexdigest()[
                    :16
                ]
                profile.typical_user_agents.add(ua_hash)

        # Update API usage patterns
        if event_type == SecurityEventType.API_REQUEST:
            endpoint = event_data.get("endpoint", "")
            profile.typical_endpoints[endpoint] = (
                profile.typical_endpoints.get(endpoint, 0) + 1
            )

        # Update profile metadata
        profile.last_updated = current_time
        profile.sample_count += 1

        # Calculate confidence score based on sample count and time
        days_active = (current_time - profile.created_at).days
        profile.confidence_score = min(
            1.0,
            (profile.sample_count / self.min_samples_for_profile)
            * min(1.0, days_active / self.learning_period_days),
        )

    async def _detect_behavioral_anomalies(
        self, user_id: str, event_data: dict[str, Any], profile: BehaviorProfile
    ) -> list[str]:
        """Detect behavioral anomalies for a user."""
        anomalies = []
        current_time = datetime.now(UTC)

        # Check login time anomaly
        if event_data.get("event_type") == SecurityEventType.AUTH_LOGIN_SUCCESS:
            if (
                current_time.hour not in profile.typical_login_hours
                and len(profile.typical_login_hours) > 3
            ):
                anomalies.append(f"Login at unusual time: {current_time.hour:02d}:00")

        # Check IP address anomaly
        ip_address = event_data.get("ip_address")
        if (
            ip_address
            and ip_address not in profile.typical_ips
            and len(profile.typical_ips) > 2
        ):
            # Check if it's in the same subnet as known IPs
            is_similar_network = False
            try:
                current_ip = ipaddress.ip_address(ip_address)
                for known_ip in profile.typical_ips:
                    try:
                        known_ip_addr = ipaddress.ip_address(known_ip)
                        if current_ip.version == known_ip_addr.version:
                            # Check if in same /24 subnet
                            if isinstance(current_ip, ipaddress.IPv4Address):
                                current_network = ipaddress.ip_network(
                                    f"{current_ip}/24", strict=False
                                )
                                known_network = ipaddress.ip_network(
                                    f"{known_ip_addr}/24", strict=False
                                )
                                if current_network == known_network:
                                    is_similar_network = True
                                    break
                    except ValueError:
                        continue
            except ValueError:
                pass

            if not is_similar_network:
                anomalies.append(f"Login from unknown IP address: {ip_address}")

        # Check user agent anomaly
        user_agent = event_data.get("user_agent")
        if user_agent:
            ua_hash = hashlib.sha256(user_agent.encode()).hexdigest()[:16]
            if (
                ua_hash not in profile.typical_user_agents
                and len(profile.typical_user_agents) > 1
            ):
                anomalies.append("Login with unusual user agent")

        # Check API usage anomaly
        if event_data.get("event_type") == SecurityEventType.API_REQUEST:
            endpoint = event_data.get("endpoint", "")

            # Check for access to unusual endpoints
            if (
                endpoint not in profile.typical_endpoints
                and len(profile.typical_endpoints) > 5
            ):
                anomalies.append(f"Access to unusual endpoint: {endpoint}")

        return anomalies

    def _calculate_anomaly_confidence(self, anomalies: list[str]) -> float:
        """Calculate confidence score for detected anomalies."""
        if not anomalies:
            return 0.0

        # Base confidence increases with number of anomalies
        base_confidence = min(0.9, len(anomalies) * 0.3)

        # Increase confidence for specific high-risk anomalies
        high_risk_patterns = ["unknown IP", "unusual endpoint", "unusual time"]
        for anomaly in anomalies:
            for pattern in high_risk_patterns:
                if pattern in anomaly:
                    base_confidence = min(1.0, base_confidence + 0.2)
                    break

        return base_confidence

    def _calculate_threat_level(self, anomalies: list[str]) -> ThreatLevel:
        """Calculate threat level based on anomalies."""
        confidence = self._calculate_anomaly_confidence(anomalies)

        if confidence >= 0.9:
            return ThreatLevel.HIGH
        elif confidence >= 0.7:
            return ThreatLevel.MEDIUM
        else:
            return ThreatLevel.LOW

    def get_configuration(self) -> dict[str, Any]:
        """Get detector configuration."""
        return {
            "learning_period_days": self.learning_period_days,
            "anomaly_threshold": self.anomaly_threshold,
            "min_samples_for_profile": self.min_samples_for_profile,
        }

    def update_configuration(self, config: dict[str, Any]) -> None:
        """Update detector configuration."""
        self.learning_period_days = config.get(
            "learning_period_days", self.learning_period_days
        )
        self.anomaly_threshold = config.get("anomaly_threshold", self.anomaly_threshold)
        self.min_samples_for_profile = config.get(
            "min_samples_for_profile", self.min_samples_for_profile
        )


class ThreatIntelligenceDetector(ThreatDetector):
    """Threat intelligence-based detection."""

    def __init__(self):
        super().__init__("threat_intelligence")
        self.threat_feeds: dict[
            ThreatIntelligenceSource, list[ThreatIntelligence]
        ] = defaultdict(list)
        self.last_update: dict[ThreatIntelligenceSource, datetime] = {}

        # Load configuration from manager
        config_manager = get_threat_detection_manager()
        detector_config = config_manager.get_detector_config("threat_intelligence")
        if detector_config:
            self.update_interval = detector_config.config.get("update_interval", 3600)
            self.confidence_threshold = detector_config.config.get(
                "confidence_threshold", 0.7
            )
        else:
            # Default configuration
            self.update_interval = 3600  # 1 hour
            self.confidence_threshold = 0.7

        # Initialize with some basic threat intelligence
        self._initialize_threat_feeds()

    def _initialize_threat_feeds(self) -> None:
        """Initialize threat intelligence feeds with basic known bad indicators."""
        current_time = datetime.now(UTC)

        # Known malicious IPs (examples - in production, use real threat feeds)
        malicious_ips = [
            "192.168.1.100",  # Example malicious IP
            "10.0.0.50",  # Example malicious IP
        ]

        for ip in malicious_ips:
            threat_intel = ThreatIntelligence(
                source=ThreatIntelligenceSource.KNOWN_BAD_IPS,
                indicator=ip,
                threat_type="malicious_ip",
                confidence=0.9,
                first_seen=current_time,
                last_updated=current_time,
                metadata={"source": "internal_blocklist"},
            )
            self.threat_feeds[ThreatIntelligenceSource.KNOWN_BAD_IPS].append(
                threat_intel
            )

        # Suspicious user agents
        suspicious_uas = [
            "sqlmap",
            "nikto",
            "nmap",
            "masscan",
            "w3af",
            "burp",
            "dirbuster",
        ]

        for ua in suspicious_uas:
            threat_intel = ThreatIntelligence(
                source=ThreatIntelligenceSource.SUSPICIOUS_USER_AGENTS,
                indicator=ua.lower(),
                threat_type="malicious_tool",
                confidence=0.95,
                first_seen=current_time,
                last_updated=current_time,
                metadata={"category": "penetration_testing_tool"},
            )
            self.threat_feeds[ThreatIntelligenceSource.SUSPICIOUS_USER_AGENTS].append(
                threat_intel
            )

    async def analyze(self, event_data: dict[str, Any]) -> SecurityAlert | None:
        """Analyze against threat intelligence feeds."""
        # Check IP address against threat intelligence
        ip_address = event_data.get("ip_address")
        if ip_address:
            for threat_intel in self.threat_feeds[
                ThreatIntelligenceSource.KNOWN_BAD_IPS
            ]:
                if (
                    ip_address == threat_intel.indicator
                    and threat_intel.confidence >= self.confidence_threshold
                ):
                    return SecurityAlert(
                        alert_id=f"ti_ip_{ip_address}_{int(time.time())}",
                        alert_type=AlertType.SYSTEM_COMPROMISE,
                        threat_level=ThreatLevel.CRITICAL,
                        title="Known Malicious IP Detected",
                        description=f"Request from known malicious IP: {ip_address}",
                        timestamp=datetime.now(UTC),
                        source_ip=ip_address,
                        user_id=event_data.get("user_id"),
                        indicators={
                            "threat_type": threat_intel.threat_type,
                            "confidence": threat_intel.confidence,
                            "source": threat_intel.source,
                        },
                        evidence=[
                            f"IP {ip_address} found in threat intelligence feed",
                            f"Threat type: {threat_intel.threat_type}",
                            f"Confidence: {threat_intel.confidence}",
                        ],
                        recommended_actions=[
                            "Block IP address immediately",
                            "Review all activity from this IP",
                            "Check for system compromise",
                            "Update firewall rules",
                            "Investigate related accounts",
                        ],
                        auto_mitigated=True,
                        mitigation_actions=["IP blocked based on threat intelligence"],
                    )

        # Check user agent against threat intelligence
        user_agent = event_data.get("user_agent", "").lower()
        if user_agent:
            for threat_intel in self.threat_feeds[
                ThreatIntelligenceSource.SUSPICIOUS_USER_AGENTS
            ]:
                if (
                    threat_intel.indicator in user_agent
                    and threat_intel.confidence >= self.confidence_threshold
                ):
                    return SecurityAlert(
                        alert_id=f"ti_ua_{hash(user_agent)}_{int(time.time())}",
                        alert_type=AlertType.MALWARE_DETECTED,
                        threat_level=ThreatLevel.HIGH,
                        title="Malicious Tool Detected",
                        description="Request from suspicious user agent",
                        timestamp=datetime.now(UTC),
                        source_ip=ip_address,
                        user_agent=event_data.get("user_agent"),
                        indicators={
                            "detected_tool": threat_intel.indicator,
                            "threat_type": threat_intel.threat_type,
                            "confidence": threat_intel.confidence,
                        },
                        evidence=[
                            f"User agent contains: {threat_intel.indicator}",
                            f"Tool category: {threat_intel.metadata.get('category', 'unknown')}",
                        ],
                        recommended_actions=[
                            "Block source IP",
                            "Review access logs",
                            "Check for vulnerability scanning",
                            "Investigate potential attack",
                            "Update security monitoring",
                        ],
                    )

        return None

    def add_threat_intelligence(self, threat_intel: ThreatIntelligence) -> None:
        """Add threat intelligence to feeds."""
        self.threat_feeds[threat_intel.source].append(threat_intel)
        self.last_update[threat_intel.source] = datetime.now(UTC)
        logger.info(
            f"Added threat intelligence: {threat_intel.indicator} from {threat_intel.source}"
        )

    def get_configuration(self) -> dict[str, Any]:
        """Get detector configuration."""
        return {
            "update_interval": self.update_interval,
            "confidence_threshold": self.confidence_threshold,
            "feed_counts": {
                source.value: len(indicators)
                for source, indicators in self.threat_feeds.items()
            },
        }

    def update_configuration(self, config: dict[str, Any]) -> None:
        """Update detector configuration."""
        self.update_interval = config.get("update_interval", self.update_interval)
        self.confidence_threshold = config.get(
            "confidence_threshold", self.confidence_threshold
        )


class SessionHijackingDetector(ThreatDetector):
    """Detector for session hijacking and abuse."""

    def __init__(self):
        super().__init__("session_hijacking")
        self.session_tracking: dict[str, dict[str, Any]] = defaultdict(dict)

        # Load configuration from manager
        config_manager = get_threat_detection_manager()
        detector_config = config_manager.get_detector_config("session_hijacking")
        if detector_config:
            self.max_ip_changes = detector_config.config.get("max_ip_changes", 3)
            self.time_window = detector_config.config.get("time_window_seconds", 3600)
            self.geographic_distance_km = detector_config.config.get(
                "geographic_distance_km", 1000
            )
        else:
            # Default configuration
            self.max_ip_changes = 3  # More than 3 IP changes in time window
            self.time_window = 3600  # 1 hour
            self.geographic_distance_km = 1000  # Impossible travel distance

    async def analyze(self, event_data: dict[str, Any]) -> SecurityAlert | None:
        """Analyze for session hijacking indicators."""
        session_id = event_data.get("session_id")
        user_id = event_data.get("user_id")
        ip_address = event_data.get("ip_address")

        if not session_id or not ip_address:
            return None

        current_time = time.time()

        # Initialize session tracking if new
        if session_id not in self.session_tracking:
            self.session_tracking[session_id] = {
                "user_id": user_id,
                "start_time": current_time,
                "ip_history": [],
                "last_activity": current_time,
            }

        session_info = self.session_tracking[session_id]

        # Update last activity
        session_info["last_activity"] = current_time

        # Track IP changes
        if ip_address not in [entry["ip"] for entry in session_info["ip_history"]]:
            session_info["ip_history"].append(
                {
                    "ip": ip_address,
                    "timestamp": current_time,
                    "user_agent": event_data.get("user_agent", ""),
                }
            )

            # Check for suspicious IP changes
            recent_ips = [
                entry
                for entry in session_info["ip_history"]
                if current_time - entry["timestamp"] <= self.time_window
            ]

            if len(recent_ips) > self.max_ip_changes:
                return SecurityAlert(
                    alert_id=f"hijack_{session_id}_{int(current_time)}",
                    alert_type=AlertType.SESSION_HIJACK,
                    threat_level=ThreatLevel.HIGH,
                    title="Potential Session Hijacking",
                    description=f"Session {session_id} showing multiple IP changes",
                    timestamp=datetime.now(UTC),
                    user_id=user_id,
                    source_ip=ip_address,
                    session_id=session_id,
                    indicators={
                        "ip_change_count": len(recent_ips),
                        "max_allowed": self.max_ip_changes,
                        "session_duration_minutes": (
                            current_time - session_info["start_time"]
                        )
                        / 60,
                    },
                    evidence=[
                        f"Session changed IPs {len(recent_ips)} times in {self.time_window / 60} minutes",
                        f"IP sequence: {' -> '.join([entry['ip'] for entry in recent_ips])}",
                        f"User agents: {set(entry['user_agent'] for entry in recent_ips if entry['user_agent'])}",
                    ],
                    recommended_actions=[
                        "Terminate session immediately",
                        "Force user re-authentication",
                        "Review session activity logs",
                        "Check for account compromise",
                        "Implement stricter session controls",
                    ],
                )

        return None

    def get_configuration(self) -> dict[str, Any]:
        """Get detector configuration."""
        return {
            "max_ip_changes": self.max_ip_changes,
            "time_window_seconds": self.time_window,
            "geographic_distance_km": self.geographic_distance_km,
        }

    def update_configuration(self, config: dict[str, Any]) -> None:
        """Update detector configuration."""
        self.max_ip_changes = config.get("max_ip_changes", self.max_ip_changes)
        self.time_window = config.get("time_window_seconds", self.time_window)
        self.geographic_distance_km = config.get(
            "geographic_distance_km", self.geographic_distance_km
        )


class DataExfiltrationDetector(ThreatDetector):
    """Detector for potential data exfiltration."""

    def __init__(self):
        super().__init__("data_exfiltration")
        self.user_data_access: dict[str, deque] = defaultdict(lambda: deque(maxlen=100))

        # Load configuration from manager
        config_manager = get_threat_detection_manager()
        detector_config = config_manager.get_detector_config("data_exfiltration")
        if detector_config:
            self.size_threshold_mb = detector_config.config.get(
                "size_threshold_mb", 100
            )
            self.time_window = detector_config.config.get("time_window_seconds", 300)
            self.request_count_threshold = detector_config.config.get(
                "request_count_threshold", 50
            )
        else:
            # Default configuration
            self.size_threshold_mb = 100  # Alert if user accesses > 100MB in short time
            self.time_window = 300  # 5 minutes
            self.request_count_threshold = 50  # > 50 requests in time window

    async def analyze(self, event_data: dict[str, Any]) -> SecurityAlert | None:
        """Analyze for potential data exfiltration."""
        user_id = event_data.get("user_id")
        if not user_id:
            return None

        event_type = event_data.get("event_type")

        if event_type == SecurityEventType.DATA_ACCESS:
            current_time = time.time()
            data_size = event_data.get("details", {}).get("data_size_bytes", 0)

            # Record data access
            self.user_data_access[user_id].append(
                {
                    "timestamp": current_time,
                    "size_bytes": data_size,
                    "endpoint": event_data.get("endpoint", ""),
                    "ip_address": event_data.get("ip_address"),
                }
            )

            # Analyze recent access pattern
            recent_accesses = [
                access
                for access in self.user_data_access[user_id]
                if current_time - access["timestamp"] <= self.time_window
            ]

            if len(recent_accesses) >= 2:  # Need at least 2 accesses to analyze
                total_size_mb = sum(
                    access["size_bytes"] for access in recent_accesses
                ) / (1024 * 1024)
                request_count = len(recent_accesses)

                # Check for excessive data access
                if (
                    total_size_mb > self.size_threshold_mb
                    or request_count > self.request_count_threshold
                ):
                    return SecurityAlert(
                        alert_id=f"exfil_{user_id}_{int(current_time)}",
                        alert_type=AlertType.DATA_EXFILTRATION,
                        threat_level=ThreatLevel.HIGH,
                        title="Potential Data Exfiltration Detected",
                        description=f"User {user_id} accessing large amounts of data",
                        timestamp=datetime.now(UTC),
                        user_id=user_id,
                        source_ip=event_data.get("ip_address"),
                        indicators={
                            "total_data_mb": total_size_mb,
                            "request_count": request_count,
                            "time_window_minutes": self.time_window / 60,
                            "threshold_mb": self.size_threshold_mb,
                        },
                        evidence=[
                            f"Accessed {total_size_mb:.2f} MB in {self.time_window / 60} minutes",
                            f"Made {request_count} data requests",
                            f"Endpoints: {', '.join({a['endpoint'] for a in recent_accesses})}",
                        ],
                        affected_resources=["data", "database"],
                        recommended_actions=[
                            "Review user data access permissions",
                            "Check if access is legitimate",
                            "Monitor user activity closely",
                            "Consider restricting data access",
                            "Investigate potential insider threat",
                        ],
                    )

        return None

    def get_configuration(self) -> dict[str, Any]:
        """Get detector configuration."""
        return {
            "size_threshold_mb": self.size_threshold_mb,
            "time_window_seconds": self.time_window,
            "request_count_threshold": self.request_count_threshold,
        }

    def update_configuration(self, config: dict[str, Any]) -> None:
        """Update detector configuration."""
        self.size_threshold_mb = config.get("size_threshold_mb", self.size_threshold_mb)
        self.time_window = config.get("time_window_seconds", self.time_window)
        self.request_count_threshold = config.get(
            "request_count_threshold", self.request_count_threshold
        )


# Factory function to create all advanced detectors
def create_advanced_threat_detectors() -> list[ThreatDetector]:
    """Create all advanced threat detectors.

    Returns:
        List of configured threat detectors
    """
    return [
        AdvancedBehaviorAnalyzer(),
        ThreatIntelligenceDetector(),
        SessionHijackingDetector(),
        DataExfiltrationDetector(),
    ]
