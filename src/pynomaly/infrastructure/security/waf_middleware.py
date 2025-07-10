"""
Web Application Firewall (WAF) middleware for advanced threat detection and prevention.
Provides comprehensive protection against OWASP Top 10 and advanced attack vectors.
"""

import json
import logging
import re
import time
from collections import defaultdict, deque
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Any

import redis
from fastapi import Request, Response, status
from fastapi.responses import JSONResponse
from starlette.middleware.base import BaseHTTPMiddleware

from pynomaly.infrastructure.config import Settings
from pynomaly.infrastructure.security.audit_logger import (
    AuditLevel,
    SecurityEventType,
    get_audit_logger,
)
from pynomaly.infrastructure.security.input_sanitizer import (
    InputSanitizer,
    SanitizationConfig,
)

logger = logging.getLogger(__name__)


class ThreatLevel(str, Enum):
    """Threat severity levels."""

    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


class AttackType(str, Enum):
    """Types of attacks the WAF can detect."""

    SQL_INJECTION = "sql_injection"
    XSS = "xss"
    COMMAND_INJECTION = "command_injection"
    PATH_TRAVERSAL = "path_traversal"
    CSRF = "csrf"
    SCANNER = "scanner"
    BRUTE_FORCE = "brute_force"
    DDOS = "ddos"
    MALICIOUS_BOT = "malicious_bot"
    PROTOCOL_VIOLATION = "protocol_violation"
    ANOMALY = "anomaly"
    MALWARE = "malware"


@dataclass
class ThreatSignature:
    """Threat detection signature."""

    name: str
    pattern: str
    attack_type: AttackType
    threat_level: ThreatLevel
    description: str
    regex_flags: int = re.IGNORECASE
    enabled: bool = True
    custom: bool = False


@dataclass
class AttackAttempt:
    """Detected attack attempt."""

    ip: str
    timestamp: float
    attack_type: AttackType
    threat_level: ThreatLevel
    details: dict[str, Any]
    signature: str
    blocked: bool = False
    risk_score: int = 0


@dataclass
class WAFRule:
    """WAF rule configuration."""

    name: str
    enabled: bool
    action: str  # block, monitor, sanitize
    conditions: list[str]
    exceptions: list[str] = None
    priority: int = 100


class WAFStats:
    """WAF statistics tracking."""

    def __init__(self):
        self.total_requests = 0
        self.blocked_requests = 0
        self.attacks_detected = 0
        self.attack_types = defaultdict(int)
        self.top_attackers = defaultdict(int)
        self.recent_attacks = deque(maxlen=1000)
        self.start_time = time.time()

    def record_request(self):
        """Record a processed request."""
        self.total_requests += 1

    def record_attack(self, attack: AttackAttempt):
        """Record a detected attack."""
        self.attacks_detected += 1
        self.attack_types[attack.attack_type.value] += 1
        self.top_attackers[attack.ip] += 1
        self.recent_attacks.append(attack)

        if attack.blocked:
            self.blocked_requests += 1

    def get_stats(self) -> dict[str, Any]:
        """Get current statistics."""
        uptime = time.time() - self.start_time
        return {
            "uptime_seconds": uptime,
            "total_requests": self.total_requests,
            "blocked_requests": self.blocked_requests,
            "attacks_detected": self.attacks_detected,
            "attack_types": dict(self.attack_types),
            "top_attackers": dict(
                sorted(self.top_attackers.items(), key=lambda x: x[1], reverse=True)[
                    :10
                ]
            ),
            "recent_attacks": len(self.recent_attacks),
            "requests_per_second": self.total_requests / uptime if uptime > 0 else 0,
            "attack_rate": (self.attacks_detected / self.total_requests * 100)
            if self.total_requests > 0
            else 0,
        }


class WAFMiddleware(BaseHTTPMiddleware):
    """Advanced Web Application Firewall middleware."""

    def __init__(self, app, settings: Settings):
        super().__init__(app)
        self.settings = settings
        self.redis_client = redis.from_url(settings.redis_url)
        self.audit_logger = get_audit_logger()
        self.stats = WAFStats()

        # Initialize components
        self.sanitizer = InputSanitizer(
            SanitizationConfig(level="moderate", max_length=50000, allow_html=False)
        )

        # Threat detection signatures
        self.signatures = self._load_signatures()
        self.compiled_patterns = self._compile_patterns()

        # IP reputation and blocking
        self.blocked_ips: set[str] = set()
        self.suspicious_ips: dict[str, int] = {}  # IP -> risk score
        self.ip_activity: dict[str, list[float]] = defaultdict(list)  # IP -> timestamps

        # Anomaly detection
        self.baseline_metrics = {
            "avg_request_size": 1024,
            "avg_response_time": 100,
            "common_user_agents": set(),
            "common_paths": set(),
        }

        # Load configuration
        self.config = self._load_config()

        logger.info(
            "WAF middleware initialized with %d signatures", len(self.signatures)
        )

    def _load_signatures(self) -> list[ThreatSignature]:
        """Load threat detection signatures."""
        signatures = [
            # SQL Injection
            ThreatSignature(
                name="SQL Injection - Union Based",
                pattern=r"(?:union\s+select|select\s+.*\s+from|insert\s+into|drop\s+table|delete\s+from)",
                attack_type=AttackType.SQL_INJECTION,
                threat_level=ThreatLevel.HIGH,
                description="Detects SQL injection attempts using UNION or SELECT statements",
            ),
            ThreatSignature(
                name="SQL Injection - Blind",
                pattern=r"(?:or\s+1=1|and\s+1=1|or\s+\d+=\d+|and\s+\d+=\d+|'(?:\s*or\s*|\s*and\s*)')",
                attack_type=AttackType.SQL_INJECTION,
                threat_level=ThreatLevel.HIGH,
                description="Detects blind SQL injection attempts",
            ),
            ThreatSignature(
                name="SQL Injection - Error Based",
                pattern=r"(?:benchmark\s*\(|pg_sleep\s*\(|waitfor\s+delay|convert\s*\(|cast\s*\()",
                attack_type=AttackType.SQL_INJECTION,
                threat_level=ThreatLevel.HIGH,
                description="Detects error-based SQL injection attempts",
            ),
            # XSS
            ThreatSignature(
                name="XSS - Script Tags",
                pattern=r"<script[^>]*>.*?</script>",
                attack_type=AttackType.XSS,
                threat_level=ThreatLevel.HIGH,
                description="Detects XSS attempts using script tags",
            ),
            ThreatSignature(
                name="XSS - Event Handlers",
                pattern=r"on(?:load|error|click|focus|blur|change|submit|mouseover)\s*=",
                attack_type=AttackType.XSS,
                threat_level=ThreatLevel.MEDIUM,
                description="Detects XSS attempts using event handlers",
            ),
            ThreatSignature(
                name="XSS - JavaScript Protocol",
                pattern=r"javascript\s*:\s*(?:alert|confirm|prompt|eval)\s*\(",
                attack_type=AttackType.XSS,
                threat_level=ThreatLevel.HIGH,
                description="Detects XSS attempts using javascript protocol",
            ),
            # Command Injection
            ThreatSignature(
                name="Command Injection - Shell Commands",
                pattern=r"(?:;|\||\|\||&&|\$\(|`)\s*(?:cat|ls|pwd|whoami|id|uname|nc|netcat|curl|wget|ping)",
                attack_type=AttackType.COMMAND_INJECTION,
                threat_level=ThreatLevel.CRITICAL,
                description="Detects command injection attempts",
            ),
            ThreatSignature(
                name="Command Injection - Windows",
                pattern=r"(?:;|\||\|\||&&)\s*(?:dir|type|net|tasklist|systeminfo|ipconfig)",
                attack_type=AttackType.COMMAND_INJECTION,
                threat_level=ThreatLevel.CRITICAL,
                description="Detects Windows command injection attempts",
            ),
            # Path Traversal
            ThreatSignature(
                name="Path Traversal - Directory Traversal",
                pattern=r"(?:\.\./|\.\.\\|%2e%2e%2f|%2e%2e%5c){2,}",
                attack_type=AttackType.PATH_TRAVERSAL,
                threat_level=ThreatLevel.HIGH,
                description="Detects directory traversal attempts",
            ),
            ThreatSignature(
                name="Path Traversal - System Files",
                pattern=r"(?:/etc/passwd|/etc/shadow|/proc/|/sys/|c:\\windows\\system32|c:\\boot\.ini)",
                attack_type=AttackType.PATH_TRAVERSAL,
                threat_level=ThreatLevel.HIGH,
                description="Detects attempts to access system files",
            ),
            # Scanner Detection
            ThreatSignature(
                name="Scanner - Nikto",
                pattern=r"nikto|nmap|dirb|dirbuster|gobuster|wpscan|sqlmap",
                attack_type=AttackType.SCANNER,
                threat_level=ThreatLevel.MEDIUM,
                description="Detects security scanner user agents",
            ),
            ThreatSignature(
                name="Scanner - Suspicious Paths",
                pattern=r"(?:\.php|\.asp|\.jsp|admin|login|config|backup|test|dev|staging|phpmyadmin)",
                attack_type=AttackType.SCANNER,
                threat_level=ThreatLevel.LOW,
                description="Detects scanning for common files/paths",
            ),
            # Malware/Webshell
            ThreatSignature(
                name="Webshell - PHP",
                pattern=r"(?:eval\s*\(|base64_decode\s*\(|system\s*\(|exec\s*\(|shell_exec\s*\(|passthru\s*\()",
                attack_type=AttackType.MALWARE,
                threat_level=ThreatLevel.CRITICAL,
                description="Detects PHP webshell patterns",
            ),
            ThreatSignature(
                name="Webshell - ASP",
                pattern=r"(?:wscript\.shell|cmd\.exe|powershell\.exe|execute\s*\(|createobject\s*\()",
                attack_type=AttackType.MALWARE,
                threat_level=ThreatLevel.CRITICAL,
                description="Detects ASP webshell patterns",
            ),
            # Protocol Violations
            ThreatSignature(
                name="Protocol Violation - HTTP Method",
                pattern=r"(?:TRACE|TRACK|DEBUG|OPTIONS|CONNECT)",
                attack_type=AttackType.PROTOCOL_VIOLATION,
                threat_level=ThreatLevel.LOW,
                description="Detects unusual HTTP methods",
            ),
            ThreatSignature(
                name="Protocol Violation - Header Injection",
                pattern=r"(?:\r\n|\n|\r)(?:content-type|set-cookie|location):",
                attack_type=AttackType.PROTOCOL_VIOLATION,
                threat_level=ThreatLevel.MEDIUM,
                description="Detects HTTP header injection attempts",
            ),
        ]

        # Load custom signatures from config if available
        custom_signatures = self._load_custom_signatures()
        signatures.extend(custom_signatures)

        return signatures

    def _load_custom_signatures(self) -> list[ThreatSignature]:
        """Load custom threat signatures from configuration."""
        custom_sigs = []
        config_path = Path("config/security/waf_signatures.json")

        if config_path.exists():
            try:
                with open(config_path) as f:
                    data = json.load(f)

                for sig_data in data.get("signatures", []):
                    sig = ThreatSignature(
                        name=sig_data["name"],
                        pattern=sig_data["pattern"],
                        attack_type=AttackType(sig_data["attack_type"]),
                        threat_level=ThreatLevel(sig_data["threat_level"]),
                        description=sig_data.get("description", ""),
                        enabled=sig_data.get("enabled", True),
                        custom=True,
                    )
                    custom_sigs.append(sig)

            except Exception as e:
                logger.warning(f"Failed to load custom signatures: {e}")

        return custom_sigs

    def _compile_patterns(self) -> dict[str, re.Pattern]:
        """Compile regex patterns for performance."""
        patterns = {}

        for sig in self.signatures:
            if sig.enabled:
                try:
                    patterns[sig.name] = re.compile(sig.pattern, sig.regex_flags)
                except re.error as e:
                    logger.error(
                        f"Invalid regex pattern in signature '{sig.name}': {e}"
                    )

        return patterns

    def _load_config(self) -> dict[str, Any]:
        """Load WAF configuration."""
        config_path = Path("config/security/waf_config.json")
        default_config = {
            "blocking_enabled": True,
            "monitoring_enabled": True,
            "ip_reputation_enabled": True,
            "anomaly_detection_enabled": True,
            "auto_block_threshold": 5,
            "block_duration": 3600,  # 1 hour
            "whitelist_ips": [],
            "blacklist_ips": [],
            "rate_limit_requests": 100,
            "rate_limit_window": 60,
            "max_request_size": 1048576,  # 1MB
            "allowed_extensions": [
                ".jpg",
                ".jpeg",
                ".png",
                ".gif",
                ".css",
                ".js",
                ".html",
                ".htm",
                ".pdf",
            ],
            "blocked_extensions": [
                ".exe",
                ".bat",
                ".cmd",
                ".sh",
                ".php",
                ".asp",
                ".jsp",
            ],
            "sensitive_paths": ["/admin", "/config", "/backup", "/test"],
            "rules": [],
        }

        if config_path.exists():
            try:
                with open(config_path) as f:
                    loaded_config = json.load(f)
                    default_config.update(loaded_config)
            except Exception as e:
                logger.warning(f"Failed to load WAF config: {e}")

        return default_config

    async def dispatch(self, request: Request, call_next):
        """Process request through WAF."""
        start_time = time.time()
        self.stats.record_request()

        try:
            # Skip WAF for whitelisted paths
            if self._is_whitelisted_path(request.url.path):
                return await call_next(request)

            # Get client info
            client_ip = self._get_client_ip(request)
            user_agent = request.headers.get("User-Agent", "")

            # Check IP reputation
            if self._is_blocked_ip(client_ip):
                attack = AttackAttempt(
                    ip=client_ip,
                    timestamp=time.time(),
                    attack_type=AttackType.DDOS,
                    threat_level=ThreatLevel.HIGH,
                    details={"reason": "IP blocked due to reputation"},
                    signature="ip_reputation",
                    blocked=True,
                    risk_score=90,
                )
                self.stats.record_attack(attack)
                return self._create_block_response(attack)

            # Basic request validation
            validation_result = await self._validate_request(request)
            if validation_result:
                attack = AttackAttempt(
                    ip=client_ip,
                    timestamp=time.time(),
                    attack_type=validation_result["attack_type"],
                    threat_level=validation_result["threat_level"],
                    details=validation_result["details"],
                    signature=validation_result["signature"],
                    blocked=True,
                    risk_score=validation_result["risk_score"],
                )
                self.stats.record_attack(attack)
                await self._handle_attack(attack)
                return self._create_block_response(attack)

            # Analyze request content
            content_analysis = await self._analyze_request_content(request)
            if content_analysis:
                attack = AttackAttempt(
                    ip=client_ip,
                    timestamp=time.time(),
                    attack_type=content_analysis["attack_type"],
                    threat_level=content_analysis["threat_level"],
                    details=content_analysis["details"],
                    signature=content_analysis["signature"],
                    blocked=self.config["blocking_enabled"],
                    risk_score=content_analysis["risk_score"],
                )
                self.stats.record_attack(attack)
                await self._handle_attack(attack)

                if self.config["blocking_enabled"]:
                    return self._create_block_response(attack)

            # Check for anomalies
            anomaly_result = await self._detect_anomalies(request)
            if anomaly_result:
                attack = AttackAttempt(
                    ip=client_ip,
                    timestamp=time.time(),
                    attack_type=AttackType.ANOMALY,
                    threat_level=ThreatLevel.MEDIUM,
                    details=anomaly_result,
                    signature="anomaly_detection",
                    blocked=False,
                    risk_score=anomaly_result["risk_score"],
                )
                self.stats.record_attack(attack)
                await self._handle_attack(attack)

            # Process request
            response = await call_next(request)

            # Add security headers
            self._add_security_headers(response)

            # Log successful request
            processing_time = time.time() - start_time
            self._log_request(request, response, processing_time)

            return response

        except Exception as e:
            logger.error(f"WAF processing error: {e}")
            # Continue without WAF protection on error
            return await call_next(request)

    def _is_whitelisted_path(self, path: str) -> bool:
        """Check if path is whitelisted."""
        whitelist = ["/health", "/api/v1/health/", "/metrics", "/favicon.ico"]
        return path in whitelist

    def _get_client_ip(self, request: Request) -> str:
        """Get client IP address."""
        # Check for forwarded headers
        forwarded_for = request.headers.get("X-Forwarded-For")
        if forwarded_for:
            return forwarded_for.split(",")[0].strip()

        real_ip = request.headers.get("X-Real-IP")
        if real_ip:
            return real_ip

        return request.client.host if request.client else "unknown"

    def _is_blocked_ip(self, ip: str) -> bool:
        """Check if IP is blocked."""
        # Check static blacklist
        if ip in self.config.get("blacklist_ips", []):
            return True

        # Check dynamic blocks
        if ip in self.blocked_ips:
            return True

        # Check Redis cache
        try:
            blocked = self.redis_client.get(f"waf:blocked:{ip}")
            return blocked is not None
        except:
            return False

    async def _validate_request(self, request: Request) -> dict[str, Any] | None:
        """Validate basic request properties."""
        # Check request size
        if hasattr(request, "headers") and "content-length" in request.headers:
            content_length = int(request.headers.get("content-length", 0))
            if content_length > self.config["max_request_size"]:
                return {
                    "attack_type": AttackType.DDOS,
                    "threat_level": ThreatLevel.MEDIUM,
                    "details": {
                        "content_length": content_length,
                        "max_allowed": self.config["max_request_size"],
                    },
                    "signature": "request_size_limit",
                    "risk_score": 50,
                }

        # Check file extensions
        path = request.url.path
        if "." in path:
            ext = Path(path).suffix.lower()
            if ext in self.config["blocked_extensions"]:
                return {
                    "attack_type": AttackType.MALWARE,
                    "threat_level": ThreatLevel.HIGH,
                    "details": {"extension": ext, "path": path},
                    "signature": "blocked_extension",
                    "risk_score": 80,
                }

        # Check sensitive paths
        for sensitive_path in self.config["sensitive_paths"]:
            if path.startswith(sensitive_path):
                return {
                    "attack_type": AttackType.SCANNER,
                    "threat_level": ThreatLevel.MEDIUM,
                    "details": {"path": path, "sensitive_path": sensitive_path},
                    "signature": "sensitive_path_access",
                    "risk_score": 60,
                }

        # Check HTTP method
        if request.method in ["TRACE", "TRACK", "DEBUG"]:
            return {
                "attack_type": AttackType.PROTOCOL_VIOLATION,
                "threat_level": ThreatLevel.LOW,
                "details": {"method": request.method},
                "signature": "suspicious_http_method",
                "risk_score": 30,
            }

        return None

    async def _analyze_request_content(self, request: Request) -> dict[str, Any] | None:
        """Analyze request content for threats."""
        # Collect all request data
        content_parts = []

        # URL and query parameters
        content_parts.append(str(request.url))

        # Headers
        for name, value in request.headers.items():
            content_parts.append(f"{name}: {value}")

        # Body (if present)
        if request.method in ["POST", "PUT", "PATCH"]:
            try:
                body = await request.body()
                if body:
                    content_parts.append(body.decode("utf-8", errors="ignore"))
            except:
                pass

        # Combine all content
        combined_content = "\n".join(content_parts)

        # Run through signatures
        for sig in self.signatures:
            if not sig.enabled:
                continue

            pattern = self.compiled_patterns.get(sig.name)
            if not pattern:
                continue

            match = pattern.search(combined_content)
            if match:
                risk_score = self._calculate_risk_score(
                    sig.threat_level, sig.attack_type
                )

                return {
                    "attack_type": sig.attack_type,
                    "threat_level": sig.threat_level,
                    "details": {
                        "signature": sig.name,
                        "description": sig.description,
                        "match": match.group(0)[:100],  # First 100 chars
                        "position": match.start(),
                    },
                    "signature": sig.name,
                    "risk_score": risk_score,
                }

        return None

    async def _detect_anomalies(self, request: Request) -> dict[str, Any] | None:
        """Detect request anomalies."""
        if not self.config["anomaly_detection_enabled"]:
            return None

        anomalies = []
        risk_score = 0

        # Check user agent
        user_agent = request.headers.get("User-Agent", "")
        if not user_agent:
            anomalies.append("Missing User-Agent")
            risk_score += 20
        elif len(user_agent) > 500:
            anomalies.append("Unusually long User-Agent")
            risk_score += 15

        # Check for suspicious headers
        suspicious_headers = [
            "X-Forwarded-For",
            "X-Real-IP",
            "X-Remote-IP",
            "X-Client-IP",
        ]
        for header in suspicious_headers:
            if header in request.headers:
                value = request.headers[header]
                if self._is_suspicious_header_value(value):
                    anomalies.append(f"Suspicious {header}: {value}")
                    risk_score += 10

        # Check request pattern
        path = request.url.path
        if len(path) > 200:
            anomalies.append("Unusually long request path")
            risk_score += 10

        # Check for encoded characters
        if "%" in path and self._has_excessive_encoding(path):
            anomalies.append("Excessive URL encoding")
            risk_score += 15

        if anomalies:
            return {"anomalies": anomalies, "risk_score": min(risk_score, 100)}

        return None

    def _is_suspicious_header_value(self, value: str) -> bool:
        """Check if header value is suspicious."""
        # Check for multiple IPs (potential spoofing)
        if "," in value and len(value.split(",")) > 3:
            return True

        # Check for suspicious characters
        if any(char in value for char in ["<", ">", '"', "'", "\\", "\n", "\r"]):
            return True

        return False

    def _has_excessive_encoding(self, path: str) -> bool:
        """Check if path has excessive URL encoding."""
        encoded_chars = path.count("%")
        return encoded_chars > len(path) * 0.3  # More than 30% encoded

    def _calculate_risk_score(
        self, threat_level: ThreatLevel, attack_type: AttackType
    ) -> int:
        """Calculate risk score for an attack."""
        base_scores = {
            ThreatLevel.LOW: 25,
            ThreatLevel.MEDIUM: 50,
            ThreatLevel.HIGH: 75,
            ThreatLevel.CRITICAL: 100,
        }

        attack_multipliers = {
            AttackType.SQL_INJECTION: 1.2,
            AttackType.XSS: 1.1,
            AttackType.COMMAND_INJECTION: 1.3,
            AttackType.MALWARE: 1.3,
            AttackType.DDOS: 1.1,
            AttackType.BRUTE_FORCE: 1.0,
            AttackType.SCANNER: 0.8,
            AttackType.ANOMALY: 0.7,
        }

        base_score = base_scores.get(threat_level, 50)
        multiplier = attack_multipliers.get(attack_type, 1.0)

        return min(int(base_score * multiplier), 100)

    async def _handle_attack(self, attack: AttackAttempt):
        """Handle detected attack."""
        # Log security event
        self.audit_logger.log_security_event(
            SecurityEventType.SECURITY_INTRUSION_DETECTED,
            f"WAF detected {attack.attack_type.value} attack from {attack.ip}",
            level=AuditLevel.WARNING
            if attack.threat_level in [ThreatLevel.LOW, ThreatLevel.MEDIUM]
            else AuditLevel.CRITICAL,
            details={
                "ip": attack.ip,
                "attack_type": attack.attack_type.value,
                "threat_level": attack.threat_level.value,
                "signature": attack.signature,
                "blocked": attack.blocked,
                "risk_score": attack.risk_score,
                **attack.details,
            },
            risk_score=attack.risk_score,
        )

        # Update IP reputation
        self._update_ip_reputation(attack.ip, attack.risk_score)

        # Auto-block if threshold reached
        await self._check_auto_block(attack.ip)

    def _update_ip_reputation(self, ip: str, risk_score: int):
        """Update IP reputation score."""
        current_score = self.suspicious_ips.get(ip, 0)
        new_score = min(current_score + risk_score, 100)
        self.suspicious_ips[ip] = new_score

        # Store in Redis with TTL
        try:
            self.redis_client.setex(f"waf:reputation:{ip}", 3600, new_score)
        except:
            pass

    async def _check_auto_block(self, ip: str):
        """Check if IP should be auto-blocked."""
        if ip in self.config.get("whitelist_ips", []):
            return

        reputation_score = self.suspicious_ips.get(ip, 0)
        if (
            reputation_score >= self.config["auto_block_threshold"] * 20
        ):  # 20 points per violation
            await self._block_ip(ip, "Auto-blocked due to reputation score")

    async def _block_ip(self, ip: str, reason: str):
        """Block an IP address."""
        self.blocked_ips.add(ip)

        # Store in Redis with TTL
        try:
            self.redis_client.setex(
                f"waf:blocked:{ip}",
                self.config["block_duration"],
                json.dumps(
                    {
                        "reason": reason,
                        "timestamp": time.time(),
                        "duration": self.config["block_duration"],
                    }
                ),
            )
        except:
            pass

        logger.warning(f"IP {ip} blocked: {reason}")

    def _create_block_response(self, attack: AttackAttempt) -> JSONResponse:
        """Create response for blocked request."""
        return JSONResponse(
            status_code=status.HTTP_403_FORBIDDEN,
            content={
                "error": "Request blocked by WAF",
                "message": "Your request was blocked due to security policy",
                "reference": f"WAF-{int(time.time())}-{attack.ip.replace('.', '')}",
            },
            headers={
                "X-WAF-Block": "true",
                "X-WAF-Signature": attack.signature,
                "X-WAF-Threat-Level": attack.threat_level.value,
            },
        )

    def _add_security_headers(self, response: Response):
        """Add security headers to response."""
        response.headers["X-WAF-Protected"] = "true"
        response.headers["X-Content-Type-Options"] = "nosniff"
        response.headers["X-Frame-Options"] = "DENY"
        response.headers["X-XSS-Protection"] = "1; mode=block"

    def _log_request(
        self, request: Request, response: Response, processing_time: float
    ):
        """Log request details."""
        if processing_time > 1.0:  # Log slow requests
            logger.warning(
                f"Slow WAF processing: {processing_time:.2f}s for {request.method} {request.url.path}"
            )

    def get_stats(self) -> dict[str, Any]:
        """Get WAF statistics."""
        return {
            "waf_stats": self.stats.get_stats(),
            "blocked_ips": len(self.blocked_ips),
            "suspicious_ips": len(self.suspicious_ips),
            "active_signatures": len([s for s in self.signatures if s.enabled]),
            "config": {
                "blocking_enabled": self.config["blocking_enabled"],
                "monitoring_enabled": self.config["monitoring_enabled"],
                "auto_block_threshold": self.config["auto_block_threshold"],
            },
        }

    async def unblock_ip(self, ip: str) -> bool:
        """Unblock an IP address."""
        if ip in self.blocked_ips:
            self.blocked_ips.remove(ip)

        if ip in self.suspicious_ips:
            del self.suspicious_ips[ip]

        try:
            self.redis_client.delete(f"waf:blocked:{ip}")
            self.redis_client.delete(f"waf:reputation:{ip}")
            return True
        except:
            return False
