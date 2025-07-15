"""
Zero-Trust Security Framework for Pynomaly.

This module implements a comprehensive zero-trust architecture that continuously
verifies every request based on identity, device, network, and behavioral factors.
"""

from __future__ import annotations

import asyncio
import hashlib
import logging
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from typing import Any
from uuid import UUID, uuid4

logger = logging.getLogger(__name__)


class TrustLevel(Enum):
    """Trust levels for zero-trust decisions."""

    DENIED = "denied"
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    VERIFIED = "verified"


class RiskFactor(Enum):
    """Risk factors for trust assessment."""

    DEVICE_UNKNOWN = "device_unknown"
    LOCATION_ANOMALY = "location_anomaly"
    BEHAVIOR_ANOMALY = "behavior_anomaly"
    TIME_ANOMALY = "time_anomaly"
    AUTHENTICATION_WEAKNESS = "auth_weakness"
    NETWORK_UNTRUSTED = "network_untrusted"
    PRIVILEGE_ESCALATION = "privilege_escalation"
    DATA_SENSITIVITY = "data_sensitivity"


@dataclass
class DeviceProfile:
    """Device profile for trust assessment."""

    device_id: str
    device_type: str
    operating_system: str
    browser_fingerprint: str
    last_seen: datetime
    trust_score: float = 0.5
    certificates: list[str] = field(default_factory=list)
    security_features: dict[str, bool] = field(default_factory=dict)

    def is_trusted(self) -> bool:
        """Check if device meets trust requirements."""
        return (
            self.trust_score >= 0.7
            and self.security_features.get("encryption_enabled", False)
            and self.security_features.get("screen_lock_enabled", False)
            and len(self.certificates) > 0
        )


@dataclass
class NetworkContext:
    """Network context for trust assessment."""

    ip_address: str
    geolocation: dict[str, Any]
    isp: str
    network_type: str  # corporate, home, public, vpn
    threat_intelligence_score: float = 0.0
    is_tor: bool = False
    is_vpn: bool = False
    is_corporate: bool = False

    def is_trusted(self) -> bool:
        """Check if network meets trust requirements."""
        return (
            self.threat_intelligence_score < 0.3
            and not self.is_tor
            and (self.is_corporate or self.is_vpn)
        )


@dataclass
class BehaviorProfile:
    """User behavior profile for anomaly detection."""

    user_id: str
    typical_login_times: list[int] = field(default_factory=list)
    typical_locations: list[str] = field(default_factory=list)
    typical_devices: list[str] = field(default_factory=list)
    access_patterns: dict[str, int] = field(default_factory=dict)
    risk_events: list[dict[str, Any]] = field(default_factory=list)

    def calculate_behavior_score(self, context: dict[str, Any]) -> float:
        """Calculate behavior anomaly score (0=normal, 1=highly anomalous)."""
        anomaly_score = 0.0

        # Time-based analysis
        current_hour = datetime.now().hour
        if self.typical_login_times and current_hour not in self.typical_login_times:
            anomaly_score += 0.3

        # Location analysis
        current_location = context.get("location", "")
        if self.typical_locations and current_location not in self.typical_locations:
            anomaly_score += 0.4

        # Device analysis
        current_device = context.get("device_id", "")
        if self.typical_devices and current_device not in self.typical_devices:
            anomaly_score += 0.3

        return min(anomaly_score, 1.0)


@dataclass
class SecurityRequest:
    """Security request for zero-trust evaluation."""

    request_id: UUID = field(default_factory=uuid4)
    user_id: str = ""
    resource: str = ""
    action: str = ""
    timestamp: datetime = field(default_factory=datetime.now)
    device_profile: DeviceProfile | None = None
    network_context: NetworkContext | None = None
    authentication_method: str = ""
    session_context: dict[str, Any] = field(default_factory=dict)
    risk_factors: set[RiskFactor] = field(default_factory=set)


@dataclass
class TrustDecision:
    """Zero-trust access decision."""

    decision: TrustLevel
    confidence: float
    risk_score: float
    trust_factors: dict[str, float]
    required_actions: list[str] = field(default_factory=list)
    expires_at: datetime | None = None
    reason: str = ""


class ContinuousIdentityVerification:
    """Continuous identity verification with behavioral analysis."""

    def __init__(self):
        self.behavior_profiles: dict[str, BehaviorProfile] = {}
        self.verification_cache: dict[str, TrustDecision] = {}

    async def verify_identity(self, request: SecurityRequest) -> float:
        """Verify identity with continuous assessment."""
        try:
            # Get or create behavior profile
            profile = self.behavior_profiles.get(
                request.user_id, BehaviorProfile(user_id=request.user_id)
            )

            # Analyze current behavior
            context = {
                "location": request.network_context.geolocation
                if request.network_context
                else {},
                "device_id": request.device_profile.device_id
                if request.device_profile
                else "",
                "time": request.timestamp.hour,
                "resource": request.resource,
                "action": request.action,
            }

            behavior_score = profile.calculate_behavior_score(context)

            # Authentication strength assessment
            auth_strength = self._assess_authentication_strength(
                request.authentication_method
            )

            # Session context analysis
            session_score = self._analyze_session_context(request.session_context)

            # Combined identity confidence (higher = more confident)
            identity_confidence = (
                (1.0 - behavior_score) * 0.4  # Behavior normality
                + auth_strength * 0.4  # Auth strength
                + session_score * 0.2  # Session validity
            )

            # Update behavior profile
            self._update_behavior_profile(profile, context)
            self.behavior_profiles[request.user_id] = profile

            return identity_confidence

        except Exception as e:
            logger.error(f"Identity verification failed: {e}")
            return 0.0

    def _assess_authentication_strength(self, auth_method: str) -> float:
        """Assess authentication method strength."""
        strength_map = {
            "password": 0.2,
            "password_mfa": 0.6,
            "certificate": 0.8,
            "biometric": 0.9,
            "certificate_biometric": 1.0,
        }
        return strength_map.get(auth_method, 0.1)

    def _analyze_session_context(self, session_context: dict[str, Any]) -> float:
        """Analyze session context for anomalies."""
        score = 0.5  # Base score

        # Session age
        session_age = session_context.get("age_minutes", 0)
        if session_age > 480:  # 8 hours
            score -= 0.3

        # Session activity
        last_activity = session_context.get("last_activity_minutes", 0)
        if last_activity > 30:  # 30 minutes idle
            score -= 0.2

        # Multiple sessions
        active_sessions = session_context.get("active_sessions", 1)
        if active_sessions > 3:
            score -= 0.2

        return max(score, 0.0)

    def _update_behavior_profile(
        self, profile: BehaviorProfile, context: dict[str, Any]
    ):
        """Update user behavior profile with new data."""
        # Update typical login times
        current_hour = context["time"]
        if current_hour not in profile.typical_login_times:
            profile.typical_login_times.append(current_hour)
            profile.typical_login_times = profile.typical_login_times[
                -10:
            ]  # Keep last 10

        # Update typical locations
        location_str = str(context["location"])
        if location_str not in profile.typical_locations:
            profile.typical_locations.append(location_str)
            profile.typical_locations = profile.typical_locations[-5:]  # Keep last 5

        # Update typical devices
        device_id = context["device_id"]
        if device_id and device_id not in profile.typical_devices:
            profile.typical_devices.append(device_id)
            profile.typical_devices = profile.typical_devices[-3:]  # Keep last 3


class DeviceSecurityManager:
    """Device security assessment and management."""

    def __init__(self):
        self.trusted_devices: dict[str, DeviceProfile] = {}
        self.device_certificates: dict[str, str] = {}

    async def assess_device_trust(self, device_profile: DeviceProfile) -> float:
        """Assess device trustworthiness."""
        try:
            trust_score = 0.0

            # Certificate validation
            if self._validate_device_certificate(device_profile):
                trust_score += 0.4

            # Security features check
            security_score = self._assess_security_features(device_profile)
            trust_score += security_score * 0.3

            # Device reputation
            reputation_score = self._check_device_reputation(device_profile)
            trust_score += reputation_score * 0.2

            # Device age and stability
            stability_score = self._assess_device_stability(device_profile)
            trust_score += stability_score * 0.1

            # Update device profile
            device_profile.trust_score = trust_score
            self.trusted_devices[device_profile.device_id] = device_profile

            return trust_score

        except Exception as e:
            logger.error(f"Device trust assessment failed: {e}")
            return 0.0

    def _validate_device_certificate(self, device_profile: DeviceProfile) -> bool:
        """Validate device certificates."""
        if not device_profile.certificates:
            return False

        for cert in device_profile.certificates:
            if not self._verify_certificate(cert):
                return False

        return True

    def _verify_certificate(self, certificate: str) -> bool:
        """Verify device certificate validity."""
        # Simplified certificate verification
        # In production, use proper X.509 certificate validation
        return len(certificate) > 100 and certificate.startswith(
            "-----BEGIN CERTIFICATE-----"
        )

    def _assess_security_features(self, device_profile: DeviceProfile) -> float:
        """Assess device security features."""
        features = device_profile.security_features
        score = 0.0

        security_checks = [
            "encryption_enabled",
            "screen_lock_enabled",
            "antivirus_installed",
            "firewall_enabled",
            "auto_updates_enabled",
            "secure_boot_enabled",
        ]

        enabled_features = sum(
            1 for check in security_checks if features.get(check, False)
        )
        score = enabled_features / len(security_checks)

        return score

    def _check_device_reputation(self, device_profile: DeviceProfile) -> float:
        """Check device reputation against threat intelligence."""
        # Simplified reputation check
        # In production, integrate with threat intelligence feeds
        device_id_hash = hashlib.sha256(device_profile.device_id.encode()).hexdigest()

        # Check against known compromised device patterns
        suspicious_patterns = ["suspicious", "malware", "compromised"]
        for pattern in suspicious_patterns:
            if pattern in device_profile.device_type.lower():
                return 0.0

        return 0.8  # Default good reputation

    def _assess_device_stability(self, device_profile: DeviceProfile) -> float:
        """Assess device stability and age."""
        if not device_profile.last_seen:
            return 0.5

        # Devices seen recently are more stable
        time_diff = datetime.now() - device_profile.last_seen
        if time_diff.days < 1:
            return 1.0
        elif time_diff.days < 7:
            return 0.8
        elif time_diff.days < 30:
            return 0.6
        else:
            return 0.3


class NetworkMicrosegmentation:
    """Network microsegmentation and trust assessment."""

    def __init__(self):
        self.trusted_networks: set[str] = set()
        self.threat_intelligence: dict[str, float] = {}

    async def assess_network_trust(self, network_context: NetworkContext) -> float:
        """Assess network trustworthiness."""
        try:
            trust_score = 0.0

            # Corporate network check
            if network_context.is_corporate:
                trust_score += 0.4
            elif network_context.is_vpn:
                trust_score += 0.3
            else:
                trust_score += 0.1  # Public networks get low trust

            # Threat intelligence check
            threat_score = self._check_threat_intelligence(network_context.ip_address)
            trust_score += (1.0 - threat_score) * 0.3

            # Geolocation analysis
            geo_score = self._assess_geolocation_risk(network_context.geolocation)
            trust_score += geo_score * 0.2

            # Network type assessment
            network_type_score = self._assess_network_type(network_context)
            trust_score += network_type_score * 0.1

            return min(trust_score, 1.0)

        except Exception as e:
            logger.error(f"Network trust assessment failed: {e}")
            return 0.0

    def _check_threat_intelligence(self, ip_address: str) -> float:
        """Check IP address against threat intelligence."""
        # Check cached threat intelligence
        if ip_address in self.threat_intelligence:
            return self.threat_intelligence[ip_address]

        # Simplified threat check
        # In production, integrate with real threat intelligence feeds
        threat_score = 0.0

        # Check for known malicious IP patterns
        if ip_address.startswith("127.") or ip_address.startswith("192.168."):
            threat_score = 0.0  # Local networks
        elif ip_address.startswith("10."):
            threat_score = 0.0  # Private networks
        else:
            threat_score = 0.1  # Default low threat for public IPs

        self.threat_intelligence[ip_address] = threat_score
        return threat_score

    def _assess_geolocation_risk(self, geolocation: dict[str, Any]) -> float:
        """Assess geolocation-based risk."""
        if not geolocation:
            return 0.3  # Unknown location = medium risk

        country = geolocation.get("country", "").upper()

        # High-risk countries (simplified list)
        high_risk_countries = {"XX", "YY", "ZZ"}  # Placeholder
        medium_risk_countries = {"AA", "BB", "CC"}  # Placeholder

        if country in high_risk_countries:
            return 0.2
        elif country in medium_risk_countries:
            return 0.5
        else:
            return 0.8  # Low risk countries

    def _assess_network_type(self, network_context: NetworkContext) -> float:
        """Assess network type trustworthiness."""
        network_scores = {
            "corporate": 1.0,
            "vpn": 0.8,
            "home": 0.6,
            "public": 0.2,
            "unknown": 0.3,
        }

        return network_scores.get(network_context.network_type, 0.3)


class PrivilegedAccessManagement:
    """Privileged access management with just-in-time access."""

    def __init__(self):
        self.privileged_sessions: dict[str, dict[str, Any]] = {}
        self.access_requests: dict[str, dict[str, Any]] = {}

    async def request_privileged_access(
        self,
        user_id: str,
        resource: str,
        justification: str,
        duration_minutes: int = 60,
    ) -> str:
        """Request privileged access with approval workflow."""
        request_id = str(uuid4())

        access_request = {
            "request_id": request_id,
            "user_id": user_id,
            "resource": resource,
            "justification": justification,
            "duration_minutes": duration_minutes,
            "requested_at": datetime.now(),
            "status": "pending",
            "approvals": [],
        }

        self.access_requests[request_id] = access_request

        # In production, trigger approval workflow
        logger.info(f"Privileged access requested: {request_id}")

        return request_id

    async def approve_privileged_access(
        self, request_id: str, approver_id: str, approved: bool = True
    ) -> bool:
        """Approve or deny privileged access request."""
        if request_id not in self.access_requests:
            return False

        request = self.access_requests[request_id]

        approval = {
            "approver_id": approver_id,
            "approved": approved,
            "approved_at": datetime.now(),
        }

        request["approvals"].append(approval)

        # Check if sufficient approvals
        required_approvals = 1  # Configurable
        approved_count = sum(1 for a in request["approvals"] if a["approved"])

        if approved_count >= required_approvals:
            request["status"] = "approved"

            # Create privileged session
            await self._create_privileged_session(request)
            return True
        elif any(not a["approved"] for a in request["approvals"]):
            request["status"] = "denied"

        return False

    async def _create_privileged_session(self, request: dict[str, Any]):
        """Create privileged access session."""
        session_id = str(uuid4())
        expires_at = datetime.now() + timedelta(minutes=request["duration_minutes"])

        session = {
            "session_id": session_id,
            "user_id": request["user_id"],
            "resource": request["resource"],
            "granted_at": datetime.now(),
            "expires_at": expires_at,
            "activities": [],
        }

        self.privileged_sessions[session_id] = session
        logger.info(f"Privileged session created: {session_id}")

    def check_privileged_access(self, user_id: str, resource: str) -> bool:
        """Check if user has active privileged access to resource."""
        now = datetime.now()

        for session in self.privileged_sessions.values():
            if (
                session["user_id"] == user_id
                and session["resource"] == resource
                and session["expires_at"] > now
            ):
                return True

        return False


class ZeroTrustFramework:
    """Main zero-trust framework orchestrator."""

    def __init__(self):
        self.identity_verification = ContinuousIdentityVerification()
        self.device_security = DeviceSecurityManager()
        self.network_microsegmentation = NetworkMicrosegmentation()
        self.privileged_access = PrivilegedAccessManagement()

        # Trust policy configuration
        self.trust_thresholds = {
            TrustLevel.VERIFIED: 0.9,
            TrustLevel.HIGH: 0.8,
            TrustLevel.MEDIUM: 0.6,
            TrustLevel.LOW: 0.4,
        }

    async def evaluate_trust(self, request: SecurityRequest) -> TrustDecision:
        """Evaluate trust for a security request."""
        try:
            trust_factors = {}

            # Identity verification
            identity_score = await self.identity_verification.verify_identity(request)
            trust_factors["identity"] = identity_score

            # Device assessment
            device_score = 0.5  # Default if no device profile
            if request.device_profile:
                device_score = await self.device_security.assess_device_trust(
                    request.device_profile
                )
            trust_factors["device"] = device_score

            # Network assessment
            network_score = 0.3  # Default if no network context
            if request.network_context:
                network_score = (
                    await self.network_microsegmentation.assess_network_trust(
                        request.network_context
                    )
                )
            trust_factors["network"] = network_score

            # Calculate overall trust score
            overall_trust = self._calculate_overall_trust(trust_factors, request)

            # Determine trust level
            trust_level = self._determine_trust_level(overall_trust)

            # Calculate risk score (inverse of trust)
            risk_score = 1.0 - overall_trust

            # Determine required actions
            required_actions = self._determine_required_actions(trust_level, request)

            # Create decision
            decision = TrustDecision(
                decision=trust_level,
                confidence=overall_trust,
                risk_score=risk_score,
                trust_factors=trust_factors,
                required_actions=required_actions,
                expires_at=datetime.now() + timedelta(minutes=15),  # 15-minute TTL
                reason=self._generate_decision_reason(trust_level, trust_factors),
            )

            logger.info(
                f"Zero-trust decision: {trust_level.value} (score: {overall_trust:.3f})"
            )
            return decision

        except Exception as e:
            logger.error(f"Zero-trust evaluation failed: {e}")
            return TrustDecision(
                decision=TrustLevel.DENIED,
                confidence=0.0,
                risk_score=1.0,
                trust_factors={},
                reason="Evaluation failed",
            )

    def _calculate_overall_trust(
        self, trust_factors: dict[str, float], request: SecurityRequest
    ) -> float:
        """Calculate overall trust score with weighted factors."""
        # Base weights
        weights = {"identity": 0.4, "device": 0.3, "network": 0.2, "context": 0.1}

        # Adjust weights based on request sensitivity
        if self._is_sensitive_resource(request.resource):
            weights["identity"] += 0.1
            weights["device"] += 0.1
            weights["network"] -= 0.1
            weights["context"] -= 0.1

        # Calculate context score
        context_score = self._assess_context_trust(request)
        trust_factors["context"] = context_score

        # Weighted average
        overall_trust = sum(
            trust_factors.get(factor, 0.0) * weight
            for factor, weight in weights.items()
        )

        # Apply risk factor penalties
        risk_penalty = len(request.risk_factors) * 0.1
        overall_trust = max(0.0, overall_trust - risk_penalty)

        return min(overall_trust, 1.0)

    def _is_sensitive_resource(self, resource: str) -> bool:
        """Check if resource is considered sensitive."""
        sensitive_patterns = [
            "/admin/",
            "/api/v1/users/",
            "/api/v1/models/",
            "/api/v1/governance/",
            "/api/v1/security/",
        ]

        return any(pattern in resource for pattern in sensitive_patterns)

    def _assess_context_trust(self, request: SecurityRequest) -> float:
        """Assess contextual trust factors."""
        context_score = 0.5  # Base score

        # Time-based assessment
        current_hour = request.timestamp.hour
        if 6 <= current_hour <= 22:  # Business hours
            context_score += 0.2
        else:
            context_score -= 0.2

        # Action assessment
        safe_actions = ["read", "view", "list"]
        risky_actions = ["delete", "admin", "execute"]

        if request.action in safe_actions:
            context_score += 0.2
        elif request.action in risky_actions:
            context_score -= 0.3

        return max(0.0, min(context_score, 1.0))

    def _determine_trust_level(self, trust_score: float) -> TrustLevel:
        """Determine trust level based on score."""
        for level, threshold in sorted(
            self.trust_thresholds.items(), key=lambda x: x[1], reverse=True
        ):
            if trust_score >= threshold:
                return level

        return TrustLevel.DENIED

    def _determine_required_actions(
        self, trust_level: TrustLevel, request: SecurityRequest
    ) -> list[str]:
        """Determine required actions based on trust level."""
        actions = []

        if trust_level == TrustLevel.DENIED:
            actions.append("deny_access")

        elif trust_level == TrustLevel.LOW:
            actions.extend(
                ["require_additional_mfa", "limit_access_scope", "increase_monitoring"]
            )

        elif trust_level == TrustLevel.MEDIUM:
            actions.extend(["require_step_up_auth", "monitor_session"])

        # Check for privileged access requirements
        if self._requires_privileged_access(request.resource):
            if not self.privileged_access.check_privileged_access(
                request.user_id, request.resource
            ):
                actions.append("require_privileged_access_approval")

        return actions

    def _requires_privileged_access(self, resource: str) -> bool:
        """Check if resource requires privileged access."""
        privileged_patterns = [
            "/admin/",
            "/api/v1/governance/",
            "/api/v1/security/",
            "/api/v1/users/admin",
            "/api/v1/models/deploy",
        ]

        return any(pattern in resource for pattern in privileged_patterns)

    def _generate_decision_reason(
        self, trust_level: TrustLevel, trust_factors: dict[str, float]
    ) -> str:
        """Generate human-readable decision reason."""
        if trust_level == TrustLevel.DENIED:
            return "Access denied due to insufficient trust factors"
        elif trust_level == TrustLevel.LOW:
            return f"Low trust access granted with restrictions (identity: {trust_factors.get('identity', 0):.2f})"
        elif trust_level == TrustLevel.MEDIUM:
            return f"Medium trust access granted with monitoring (device: {trust_factors.get('device', 0):.2f})"
        elif trust_level == TrustLevel.HIGH:
            return f"High trust access granted (network: {trust_factors.get('network', 0):.2f})"
        else:
            return "Verified access granted with full trust"


# Example usage and testing
if __name__ == "__main__":

    async def test_zero_trust():
        """Test zero-trust framework."""
        framework = ZeroTrustFramework()

        # Create test request
        device_profile = DeviceProfile(
            device_id="device_123",
            device_type="laptop",
            operating_system="Windows 11",
            browser_fingerprint="chrome_fingerprint",
            last_seen=datetime.now(),
            certificates=["-----BEGIN CERTIFICATE-----test-----END CERTIFICATE-----"],
            security_features={
                "encryption_enabled": True,
                "screen_lock_enabled": True,
                "antivirus_installed": True,
                "firewall_enabled": True,
            },
        )

        network_context = NetworkContext(
            ip_address="192.168.1.100",
            geolocation={"country": "US", "city": "San Francisco"},
            isp="Corporate ISP",
            network_type="corporate",
            is_corporate=True,
        )

        request = SecurityRequest(
            user_id="user_123",
            resource="/api/v1/models/deploy",
            action="create",
            device_profile=device_profile,
            network_context=network_context,
            authentication_method="certificate_biometric",
        )

        # Evaluate trust
        decision = await framework.evaluate_trust(request)

        print(f"Trust Decision: {decision.decision.value}")
        print(f"Confidence: {decision.confidence:.3f}")
        print(f"Risk Score: {decision.risk_score:.3f}")
        print(f"Trust Factors: {decision.trust_factors}")
        print(f"Required Actions: {decision.required_actions}")
        print(f"Reason: {decision.reason}")

    # Run test
    asyncio.run(test_zero_trust())
