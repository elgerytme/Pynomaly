"""
Enhanced rate limiter with intelligent adaptive throttling and DDoS protection.
"""

import logging
import time
from collections import defaultdict, deque
from dataclasses import dataclass, field
from enum import Enum

import redis
from fastapi import Request

from pynomaly.infrastructure.config import Settings
from pynomaly.infrastructure.security.audit_logger import (
    AuditLevel,
    SecurityEventType,
    get_audit_logger,
)

logger = logging.getLogger(__name__)


class ThreatLevel(str, Enum):
    """Threat levels for rate limiting."""

    BENIGN = "benign"
    SUSPICIOUS = "suspicious"
    MALICIOUS = "malicious"
    CRITICAL = "critical"


class RequestPattern(str, Enum):
    """Request patterns for analysis."""

    NORMAL = "normal"
    BURST = "burst"
    SUSTAINED = "sustained"
    DISTRIBUTED = "distributed"


@dataclass
class ClientProfile:
    """Client behavior profile."""

    ip: str
    first_seen: float
    last_seen: float
    request_count: int = 0
    blocked_count: int = 0
    threat_level: ThreatLevel = ThreatLevel.BENIGN
    request_pattern: RequestPattern = RequestPattern.NORMAL
    user_agents: set[str] = field(default_factory=set)
    endpoints: set[str] = field(default_factory=set)
    request_sizes: list[int] = field(default_factory=list)
    response_times: list[float] = field(default_factory=list)
    reputation_score: int = 100  # Start with good reputation

    def update_profile(self, request: Request, response_time: float = None):
        """Update client profile with new request data."""
        self.last_seen = time.time()
        self.request_count += 1

        # Track user agents
        user_agent = request.headers.get("User-Agent", "")
        if user_agent:
            self.user_agents.add(user_agent)

        # Track endpoints
        self.endpoints.add(request.url.path)

        # Track request size
        content_length = request.headers.get("Content-Length")
        if content_length:
            try:
                self.request_sizes.append(int(content_length))
                # Keep only last 100 entries
                if len(self.request_sizes) > 100:
                    self.request_sizes = self.request_sizes[-100:]
            except ValueError:
                pass

        # Track response time
        if response_time is not None:
            self.response_times.append(response_time)
            if len(self.response_times) > 100:
                self.response_times = self.response_times[-100:]

    def calculate_anomaly_score(self) -> float:
        """Calculate anomaly score based on behavior patterns."""
        score = 0.0

        # Multiple user agents (potential bot rotation)
        if len(self.user_agents) > 10:
            score += 0.3
        elif len(self.user_agents) > 5:
            score += 0.1

        # Too many different endpoints
        if len(self.endpoints) > 50:
            score += 0.2

        # Unusual request patterns
        if self.request_count > 1000:
            # High volume client
            if len(self.user_agents) == 1:
                score += 0.2  # Single UA with high volume = bot

        # Request size patterns
        if self.request_sizes:
            avg_size = sum(self.request_sizes) / len(self.request_sizes)
            if avg_size > 100000:  # Very large requests
                score += 0.1

        return min(score, 1.0)


@dataclass
class AdaptiveLimit:
    """Adaptive rate limit configuration."""

    base_limit: int
    current_limit: int
    window: int
    burst_tolerance: int = 0
    adaptive_factor: float = 1.0
    last_adjustment: float = field(default_factory=time.time)

    def adjust_limit(self, load_factor: float, threat_level: ThreatLevel):
        """Adjust rate limit based on system load and threat level."""
        now = time.time()
        if now - self.last_adjustment < 60:  # Don't adjust more than once per minute
            return

        # Base adjustment based on system load
        if load_factor > 0.8:  # High load
            self.adaptive_factor *= 0.8
        elif load_factor < 0.3:  # Low load
            self.adaptive_factor *= 1.1

        # Threat-based adjustment
        threat_multipliers = {
            ThreatLevel.BENIGN: 1.0,
            ThreatLevel.SUSPICIOUS: 0.7,
            ThreatLevel.MALICIOUS: 0.3,
            ThreatLevel.CRITICAL: 0.1,
        }

        threat_factor = threat_multipliers[threat_level]
        self.current_limit = int(self.base_limit * self.adaptive_factor * threat_factor)
        self.current_limit = max(
            self.current_limit, 1
        )  # Always allow at least 1 request

        self.last_adjustment = now


class EnhancedRateLimiter:
    """Enhanced rate limiter with behavioral analysis and adaptive limits."""

    def __init__(self, redis_client: redis.Redis, settings: Settings):
        self.redis = redis_client
        self.settings = settings
        self.audit_logger = get_audit_logger()

        # Client profiles
        self.client_profiles: dict[str, ClientProfile] = {}
        self.ip_request_windows: dict[str, deque] = defaultdict(
            lambda: deque(maxlen=1000)
        )

        # Adaptive limits
        self.adaptive_limits = {
            "global": AdaptiveLimit(base_limit=10000, current_limit=10000, window=60),
            "per_ip": AdaptiveLimit(base_limit=100, current_limit=100, window=60),
            "per_endpoint": AdaptiveLimit(base_limit=200, current_limit=200, window=60),
        }

        # DDoS detection
        self.ddos_threshold = 1000  # requests per minute for DDoS detection
        self.suspicious_ips: set[str] = set()
        self.blocked_ips: set[str] = set()

        # Attack patterns
        self.attack_patterns = {
            "slow_loris": self._detect_slow_loris,
            "http_flood": self._detect_http_flood,
            "cc_attack": self._detect_cc_attack,
            "bot_network": self._detect_bot_network,
        }

        # System metrics
        self.system_load = 0.0
        self.global_request_count = 0
        self.last_metrics_update = time.time()

    async def check_request_allowed(
        self, request: Request, response_time: float = None
    ) -> tuple[bool, str, dict]:
        """Check if request should be allowed with enhanced analysis."""
        client_ip = self._get_client_ip(request)
        endpoint = request.url.path

        # Update client profile
        profile = self._get_or_create_profile(client_ip)
        profile.update_profile(request, response_time)

        # Update request window
        now = time.time()
        self.ip_request_windows[client_ip].append(now)

        # Update system metrics
        self._update_system_metrics()

        # Check if IP is blocked
        if client_ip in self.blocked_ips:
            return (
                False,
                "IP_BLOCKED",
                {"reason": "IP previously blocked for malicious activity"},
            )

        # Perform behavioral analysis
        threat_level = await self._analyze_threat_level(profile, request)

        # Adjust adaptive limits
        self._adjust_adaptive_limits(threat_level)

        # Check various rate limits
        limit_checks = [
            await self._check_global_limit(),
            await self._check_ip_limit(client_ip),
            await self._check_endpoint_limit(endpoint),
            await self._check_user_limit(request),
        ]

        for allowed, reason, details in limit_checks:
            if not allowed:
                profile.blocked_count += 1
                await self._handle_rate_limit_violation(profile, reason, details)
                return False, reason, details

        # Check for attack patterns
        attack_detected = await self._detect_attack_patterns(profile, request)
        if attack_detected:
            attack_type, details = attack_detected
            profile.threat_level = ThreatLevel.MALICIOUS
            await self._handle_attack_detection(profile, attack_type, details)
            return False, f"ATTACK_DETECTED_{attack_type.upper()}", details

        # DDoS detection
        if await self._detect_ddos_attack(client_ip):
            profile.threat_level = ThreatLevel.CRITICAL
            self.blocked_ips.add(client_ip)
            await self._handle_ddos_attack(profile)
            return False, "DDOS_DETECTED", {"reason": "DDoS attack pattern detected"}

        return True, "ALLOWED", {"threat_level": threat_level.value}

    def _get_client_ip(self, request: Request) -> str:
        """Extract client IP from request."""
        # Check for forwarded headers
        forwarded_for = request.headers.get("X-Forwarded-For")
        if forwarded_for:
            return forwarded_for.split(",")[0].strip()

        real_ip = request.headers.get("X-Real-IP")
        if real_ip:
            return real_ip

        return request.client.host if request.client else "unknown"

    def _get_or_create_profile(self, ip: str) -> ClientProfile:
        """Get or create client profile."""
        if ip not in self.client_profiles:
            self.client_profiles[ip] = ClientProfile(
                ip=ip, first_seen=time.time(), last_seen=time.time()
            )
        return self.client_profiles[ip]

    def _update_system_metrics(self):
        """Update system performance metrics."""
        now = time.time()
        self.global_request_count += 1

        if now - self.last_metrics_update > 60:  # Update every minute
            # Calculate system load (simplified)
            minute_requests = sum(
                1
                for windows in self.ip_request_windows.values()
                for timestamp in windows
                if now - timestamp < 60
            )

            # Normalize load (0.0 to 1.0)
            max_capacity = self.adaptive_limits["global"].current_limit
            self.system_load = min(minute_requests / max_capacity, 1.0)

            self.last_metrics_update = now

    async def _analyze_threat_level(
        self, profile: ClientProfile, request: Request
    ) -> ThreatLevel:
        """Analyze threat level based on behavioral patterns."""
        anomaly_score = profile.calculate_anomaly_score()

        # Request frequency analysis
        now = time.time()
        recent_requests = [
            timestamp
            for timestamp in self.ip_request_windows[profile.ip]
            if now - timestamp < 300  # Last 5 minutes
        ]

        request_rate = len(recent_requests) / 300  # Requests per second

        # User agent analysis
        user_agent = request.headers.get("User-Agent", "")
        suspicious_ua_patterns = [
            "python",
            "curl",
            "wget",
            "scrapy",
            "bot",
            "crawler",
            "scanner",
            "test",
            "benchmark",
            "siege",
            "httperf",
        ]

        ua_suspicious = any(
            pattern in user_agent.lower() for pattern in suspicious_ua_patterns
        )

        # Threat level calculation
        if anomaly_score > 0.8 or request_rate > 5 or profile.blocked_count > 10:
            return ThreatLevel.CRITICAL
        elif (
            anomaly_score > 0.5
            or request_rate > 2
            or ua_suspicious
            or profile.blocked_count > 5
        ):
            return ThreatLevel.MALICIOUS
        elif anomaly_score > 0.3 or request_rate > 1 or profile.blocked_count > 0:
            return ThreatLevel.SUSPICIOUS
        else:
            return ThreatLevel.BENIGN

    def _adjust_adaptive_limits(self, threat_level: ThreatLevel):
        """Adjust rate limits based on current threat level and system load."""
        for limit_name, adaptive_limit in self.adaptive_limits.items():
            adaptive_limit.adjust_limit(self.system_load, threat_level)

    async def _check_global_limit(self) -> tuple[bool, str, dict]:
        """Check global rate limit."""
        limit = self.adaptive_limits["global"].current_limit
        window = self.adaptive_limits["global"].window

        try:
            key = "rate_limit:global"
            current = await self._get_current_count(key, window)

            if current >= limit:
                return (
                    False,
                    "GLOBAL_RATE_LIMIT",
                    {"limit": limit, "current": current, "window": window},
                )

            await self._increment_count(key, window)
            return True, "OK", {}

        except Exception as e:
            logger.error(f"Global rate limit check failed: {e}")
            return True, "OK", {}  # Fail open

    async def _check_ip_limit(self, ip: str) -> tuple[bool, str, dict]:
        """Check per-IP rate limit."""
        limit = self.adaptive_limits["per_ip"].current_limit
        window = self.adaptive_limits["per_ip"].window

        # Apply threat-based scaling
        profile = self.client_profiles.get(ip)
        if profile:
            threat_multipliers = {
                ThreatLevel.BENIGN: 1.0,
                ThreatLevel.SUSPICIOUS: 0.5,
                ThreatLevel.MALICIOUS: 0.2,
                ThreatLevel.CRITICAL: 0.1,
            }
            limit = int(limit * threat_multipliers[profile.threat_level])

        try:
            key = f"rate_limit:ip:{ip}"
            current = await self._get_current_count(key, window)

            if current >= limit:
                return (
                    False,
                    "IP_RATE_LIMIT",
                    {"ip": ip, "limit": limit, "current": current, "window": window},
                )

            await self._increment_count(key, window)
            return True, "OK", {}

        except Exception as e:
            logger.error(f"IP rate limit check failed: {e}")
            return True, "OK", {}  # Fail open

    async def _check_endpoint_limit(self, endpoint: str) -> tuple[bool, str, dict]:
        """Check per-endpoint rate limit."""
        limit = self.adaptive_limits["per_endpoint"].current_limit
        window = self.adaptive_limits["per_endpoint"].window

        try:
            key = f"rate_limit:endpoint:{endpoint}"
            current = await self._get_current_count(key, window)

            if current >= limit:
                return (
                    False,
                    "ENDPOINT_RATE_LIMIT",
                    {
                        "endpoint": endpoint,
                        "limit": limit,
                        "current": current,
                        "window": window,
                    },
                )

            await self._increment_count(key, window)
            return True, "OK", {}

        except Exception as e:
            logger.error(f"Endpoint rate limit check failed: {e}")
            return True, "OK", {}  # Fail open

    async def _check_user_limit(self, request: Request) -> tuple[bool, str, dict]:
        """Check per-user rate limit for authenticated requests."""
        # This would integrate with your authentication system
        # For now, return True
        return True, "OK", {}

    async def _get_current_count(self, key: str, window: int) -> int:
        """Get current count for rate limiting key."""
        try:
            count = self.redis.get(key)
            return int(count) if count else 0
        except:
            return 0

    async def _increment_count(self, key: str, window: int) -> int:
        """Increment count for rate limiting key."""
        try:
            pipe = self.redis.pipeline()
            pipe.incr(key)
            pipe.expire(key, window)
            results = pipe.execute()
            return results[0]
        except:
            return 0

    async def _detect_attack_patterns(
        self, profile: ClientProfile, request: Request
    ) -> tuple[str, dict] | None:
        """Detect various attack patterns."""
        for pattern_name, detector in self.attack_patterns.items():
            result = await detector(profile, request)
            if result:
                return pattern_name, result
        return None

    async def _detect_slow_loris(
        self, profile: ClientProfile, request: Request
    ) -> dict | None:
        """Detect Slow Loris attacks (slow HTTP requests)."""
        # This would need integration with request timing
        return None

    async def _detect_http_flood(
        self, profile: ClientProfile, request: Request
    ) -> dict | None:
        """Detect HTTP flood attacks."""
        now = time.time()
        recent_requests = [
            t
            for t in self.ip_request_windows[profile.ip]
            if now - t < 60  # Last minute
        ]

        if len(recent_requests) > 300:  # More than 5 requests/second sustained
            return {
                "pattern": "http_flood",
                "request_rate": len(recent_requests) / 60,
                "threshold": 5.0,
            }
        return None

    async def _detect_cc_attack(
        self, profile: ClientProfile, request: Request
    ) -> dict | None:
        """Detect Challenge Collapsar (CC) attacks."""
        # CC attacks typically target specific pages repeatedly
        endpoint_requests = sum(
            1 for endpoint in profile.endpoints if endpoint == request.url.path
        )

        if endpoint_requests > 100 and len(profile.endpoints) == 1:
            return {
                "pattern": "cc_attack",
                "endpoint": request.url.path,
                "requests": endpoint_requests,
            }
        return None

    async def _detect_bot_network(
        self, profile: ClientProfile, request: Request
    ) -> dict | None:
        """Detect coordinated bot network attacks."""
        # Look for similar behavior patterns across multiple IPs
        similar_profiles = 0
        for other_ip, other_profile in self.client_profiles.items():
            if other_ip != profile.ip:
                # Check for similar user agents and endpoints
                ua_overlap = len(profile.user_agents & other_profile.user_agents)
                endpoint_overlap = len(profile.endpoints & other_profile.endpoints)

                if ua_overlap > 0 and endpoint_overlap > 5:
                    similar_profiles += 1

        if similar_profiles > 10:  # Many IPs with similar behavior
            return {
                "pattern": "bot_network",
                "similar_profiles": similar_profiles,
                "common_endpoints": len(profile.endpoints),
            }
        return None

    async def _detect_ddos_attack(self, ip: str) -> bool:
        """Detect DDoS attacks based on global request patterns."""
        now = time.time()

        # Check global request rate
        total_recent_requests = sum(
            len([t for t in windows if now - t < 60])
            for windows in self.ip_request_windows.values()
        )

        if total_recent_requests > self.ddos_threshold:
            # Check if this IP contributes significantly
            ip_recent_requests = len(
                [t for t in self.ip_request_windows[ip] if now - t < 60]
            )

            contribution = ip_recent_requests / total_recent_requests
            if contribution > 0.1:  # IP contributes more than 10% of traffic
                return True

        return False

    async def _handle_rate_limit_violation(
        self, profile: ClientProfile, reason: str, details: dict
    ):
        """Handle rate limit violations."""
        self.audit_logger.log_security_event(
            SecurityEventType.SECURITY_RATE_LIMIT_EXCEEDED,
            f"Rate limit violated: {reason} for IP {profile.ip}",
            level=AuditLevel.WARNING,
            details={
                "ip": profile.ip,
                "reason": reason,
                "threat_level": profile.threat_level.value,
                **details,
            },
            risk_score=50,
        )

    async def _handle_attack_detection(
        self, profile: ClientProfile, attack_type: str, details: dict
    ):
        """Handle attack pattern detection."""
        self.audit_logger.log_security_event(
            SecurityEventType.SECURITY_INTRUSION_DETECTED,
            f"Attack pattern detected: {attack_type} from IP {profile.ip}",
            level=AuditLevel.CRITICAL,
            details={
                "ip": profile.ip,
                "attack_type": attack_type,
                "threat_level": profile.threat_level.value,
                **details,
            },
            risk_score=90,
        )

        # Auto-block for certain attack types
        if attack_type in ["http_flood", "cc_attack"]:
            self.blocked_ips.add(profile.ip)

    async def _handle_ddos_attack(self, profile: ClientProfile):
        """Handle DDoS attack detection."""
        self.audit_logger.log_security_event(
            SecurityEventType.SECURITY_INTRUSION_DETECTED,
            f"DDoS attack detected from IP {profile.ip}",
            level=AuditLevel.CRITICAL,
            details={
                "ip": profile.ip,
                "attack_type": "ddos",
                "threat_level": "critical",
            },
            risk_score=100,
        )

    def get_statistics(self) -> dict:
        """Get rate limiter statistics."""
        now = time.time()

        # Calculate active clients (seen in last hour)
        active_clients = sum(
            1
            for profile in self.client_profiles.values()
            if now - profile.last_seen < 3600
        )

        # Calculate threat distribution
        threat_distribution = defaultdict(int)
        for profile in self.client_profiles.values():
            if now - profile.last_seen < 3600:  # Active clients only
                threat_distribution[profile.threat_level.value] += 1

        return {
            "total_clients": len(self.client_profiles),
            "active_clients": active_clients,
            "blocked_ips": len(self.blocked_ips),
            "suspicious_ips": len(self.suspicious_ips),
            "threat_distribution": dict(threat_distribution),
            "system_load": self.system_load,
            "global_requests": self.global_request_count,
            "adaptive_limits": {
                name: {
                    "base_limit": limit.base_limit,
                    "current_limit": limit.current_limit,
                    "adaptive_factor": limit.adaptive_factor,
                }
                for name, limit in self.adaptive_limits.items()
            },
        }
