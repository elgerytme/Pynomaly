"""
Advanced security features for web UI.
Implements rate limiting, WAF, IP blocking, and security monitoring.
"""

import asyncio
import hashlib
import ipaddress
import re
import time
from collections import defaultdict, deque
from dataclasses import dataclass
from datetime import datetime
from enum import Enum
from functools import wraps
from typing import Any

from fastapi import HTTPException, Request, status
from fastapi.responses import JSONResponse
from starlette.middleware.base import BaseHTTPMiddleware

try:
    from pynomaly.core.config import get_settings
except ImportError:
    # Fallback for testing - create a mock settings function
    def get_settings():
        from types import SimpleNamespace

        return SimpleNamespace(debug=False, testing=True, log_level="INFO")


try:
    from pynomaly.presentation.web.error_handling import (
        ErrorCode,
        ErrorLevel,
        WebUIError,
        get_web_ui_logger,
    )
except ImportError:
    # Fallback for testing - create mock classes
    import logging
    from enum import Enum

    class ErrorCode(Enum):
        INTERNAL_SERVER_ERROR = "internal_server_error"

    class ErrorLevel(Enum):
        ERROR = "error"

    class WebUIError(Exception):
        def __init__(self, message, error_code, error_level, **kwargs):
            self.message = message
            self.error_code = error_code
            self.error_level = error_level
            super().__init__(message)

    def get_web_ui_logger():
        return logging.getLogger(__name__)


class SecurityThreatLevel(Enum):
    """Security threat levels"""

    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


class SecurityEventType(Enum):
    """Types of security events"""

    RATE_LIMIT_EXCEEDED = "rate_limit_exceeded"
    SUSPICIOUS_PATTERN = "suspicious_pattern"
    IP_BLOCKED = "ip_blocked"
    WAF_TRIGGERED = "waf_triggered"
    BRUTE_FORCE_ATTEMPT = "brute_force_attempt"
    SQL_INJECTION_ATTEMPT = "sql_injection_attempt"
    XSS_ATTEMPT = "xss_attempt"
    CSRF_VIOLATION = "csrf_violation"
    UNUSUAL_REQUEST_PATTERN = "unusual_request_pattern"
    AUTHENTICATION_FAILURE = "authentication_failure"


@dataclass
class SecurityEvent:
    """Security event data structure"""

    event_type: SecurityEventType
    threat_level: SecurityThreatLevel
    ip_address: str
    user_agent: str
    request_path: str
    request_method: str
    timestamp: datetime
    details: dict[str, Any]
    event_id: str
    blocked: bool = False
    action_taken: str | None = None


@dataclass
class RateLimitRule:
    """Rate limiting rule configuration"""

    requests_per_minute: int
    requests_per_hour: int
    requests_per_day: int
    burst_limit: int
    window_size: int = 60  # seconds
    penalty_duration: int = 300  # seconds


class RateLimiter:
    """Advanced rate limiter with burst protection and adaptive limits"""

    def __init__(self):
        self.settings = get_settings()
        self.logger = get_web_ui_logger()

        # Rate limiting storage
        self.request_counts = defaultdict(lambda: defaultdict(int))
        self.request_timestamps = defaultdict(lambda: deque(maxlen=10000))
        self.blocked_ips = defaultdict(lambda: {"until": None, "count": 0})
        self.burst_counts = defaultdict(int)

        # Rate limiting rules
        self.default_rules = RateLimitRule(
            requests_per_minute=60,
            requests_per_hour=1000,
            requests_per_day=10000,
            burst_limit=10,
            window_size=60,
            penalty_duration=300,
        )

        # Endpoint-specific rules
        self.endpoint_rules = {
            "/api/auth/login": RateLimitRule(5, 20, 100, 3, 60, 900),
            "/api/auth/register": RateLimitRule(3, 10, 50, 2, 60, 1800),
            "/api/auth/reset-password": RateLimitRule(3, 10, 20, 1, 60, 3600),
            "/api/upload": RateLimitRule(10, 50, 200, 5, 60, 300),
            "/api/data/export": RateLimitRule(5, 20, 100, 2, 60, 600),
        }

        # Cleanup task
        self.cleanup_task = None
        self.start_cleanup_task()

    def start_cleanup_task(self):
        """Start background cleanup task"""
        if self.cleanup_task is None:
            self.cleanup_task = asyncio.create_task(self._cleanup_expired_entries())

    async def _cleanup_expired_entries(self):
        """Clean up expired rate limit entries"""
        while True:
            try:
                current_time = time.time()

                # Clean up expired blocked IPs
                expired_ips = []
                for ip, data in self.blocked_ips.items():
                    if data["until"] and current_time > data["until"]:
                        expired_ips.append(ip)

                for ip in expired_ips:
                    del self.blocked_ips[ip]

                # Clean up old request timestamps
                for ip in list(self.request_timestamps.keys()):
                    timestamps = self.request_timestamps[ip]
                    while (
                        timestamps and current_time - timestamps[0] > 86400
                    ):  # 24 hours
                        timestamps.popleft()

                    if not timestamps:
                        del self.request_timestamps[ip]

                # Clean up old request counts
                for ip in list(self.request_counts.keys()):
                    counts = self.request_counts[ip]
                    for window in list(counts.keys()):
                        if current_time - window > 86400:  # 24 hours
                            del counts[window]

                    if not counts:
                        del self.request_counts[ip]

                await asyncio.sleep(300)  # Clean up every 5 minutes

            except Exception as e:
                self.logger.log(f"Error in rate limiter cleanup: {e}", level="ERROR")
                await asyncio.sleep(60)

    def is_rate_limited(self, ip: str, endpoint: str) -> tuple[bool, str | None]:
        """Check if IP is rate limited for endpoint"""
        current_time = time.time()

        # Check if IP is currently blocked
        if ip in self.blocked_ips:
            block_data = self.blocked_ips[ip]
            if block_data["until"] and current_time < block_data["until"]:
                return (
                    True,
                    f"IP blocked until {datetime.fromtimestamp(block_data['until'])}",
                )
            elif block_data["until"] and current_time >= block_data["until"]:
                # Unblock IP
                del self.blocked_ips[ip]

        # Get rate limit rules for endpoint
        rules = self.endpoint_rules.get(endpoint, self.default_rules)

        # Check burst limit
        if self.burst_counts[ip] > rules.burst_limit:
            self._block_ip(ip, rules.penalty_duration)
            return (
                True,
                f"Burst limit exceeded: {self.burst_counts[ip]} > {rules.burst_limit}",
            )

        # Check rate limits
        timestamps = self.request_timestamps[ip]

        # Check per-minute limit
        recent_requests = sum(1 for ts in timestamps if current_time - ts < 60)
        if recent_requests >= rules.requests_per_minute:
            self._block_ip(ip, rules.penalty_duration)
            return (
                True,
                f"Rate limit exceeded: {recent_requests} requests in last minute",
            )

        # Check per-hour limit
        hour_requests = sum(1 for ts in timestamps if current_time - ts < 3600)
        if hour_requests >= rules.requests_per_hour:
            self._block_ip(ip, rules.penalty_duration)
            return True, f"Rate limit exceeded: {hour_requests} requests in last hour"

        # Check per-day limit
        day_requests = sum(1 for ts in timestamps if current_time - ts < 86400)
        if day_requests >= rules.requests_per_day:
            self._block_ip(ip, rules.penalty_duration * 2)
            return True, f"Rate limit exceeded: {day_requests} requests in last day"

        return False, None

    def record_request(self, ip: str, endpoint: str) -> None:
        """Record a request for rate limiting"""
        current_time = time.time()

        # Record timestamp
        self.request_timestamps[ip].append(current_time)

        # Update burst count
        recent_requests = sum(
            1 for ts in self.request_timestamps[ip] if current_time - ts < 10
        )
        self.burst_counts[ip] = recent_requests

        # Update request counts
        window = int(current_time // 60) * 60  # 1-minute window
        self.request_counts[ip][window] += 1

    def _block_ip(self, ip: str, duration: int) -> None:
        """Block IP for specified duration"""
        current_time = time.time()
        self.blocked_ips[ip] = {
            "until": current_time + duration,
            "count": self.blocked_ips[ip]["count"] + 1 if ip in self.blocked_ips else 1,
        }

        self.logger.log(f"Blocked IP {ip} for {duration} seconds", level="WARNING")

    def get_rate_limit_status(self, ip: str, endpoint: str) -> dict[str, Any]:
        """Get current rate limit status for IP"""
        current_time = time.time()
        timestamps = self.request_timestamps[ip]
        rules = self.endpoint_rules.get(endpoint, self.default_rules)

        minute_requests = sum(1 for ts in timestamps if current_time - ts < 60)
        hour_requests = sum(1 for ts in timestamps if current_time - ts < 3600)
        day_requests = sum(1 for ts in timestamps if current_time - ts < 86400)

        return {
            "ip": ip,
            "endpoint": endpoint,
            "current_requests": {
                "minute": minute_requests,
                "hour": hour_requests,
                "day": day_requests,
                "burst": self.burst_counts[ip],
            },
            "limits": {
                "minute": rules.requests_per_minute,
                "hour": rules.requests_per_hour,
                "day": rules.requests_per_day,
                "burst": rules.burst_limit,
            },
            "remaining": {
                "minute": max(0, rules.requests_per_minute - minute_requests),
                "hour": max(0, rules.requests_per_hour - hour_requests),
                "day": max(0, rules.requests_per_day - day_requests),
                "burst": max(0, rules.burst_limit - self.burst_counts[ip]),
            },
            "blocked": ip in self.blocked_ips,
            "block_expires": self.blocked_ips[ip]["until"]
            if ip in self.blocked_ips
            else None,
        }


class WebApplicationFirewall:
    """Web Application Firewall with pattern matching and threat detection"""

    def __init__(self):
        self.settings = get_settings()
        self.logger = get_web_ui_logger()

        # Load WAF rules
        self.sql_injection_patterns = self._load_sql_injection_patterns()
        self.xss_patterns = self._load_xss_patterns()
        self.path_traversal_patterns = self._load_path_traversal_patterns()
        self.command_injection_patterns = self._load_command_injection_patterns()

        # Blocked patterns
        self.blocked_user_agents = self._load_blocked_user_agents()
        self.blocked_ips = set()
        self.blocked_countries = set()

        # Whitelist
        self.whitelisted_ips = self._load_whitelisted_ips()

        # Threat scoring
        self.threat_scores = defaultdict(int)
        self.threat_threshold = 100

    def _load_sql_injection_patterns(self) -> list[re.Pattern]:
        """Load SQL injection detection patterns"""
        patterns = [
            r"(\bUNION\b.*\bSELECT\b)",
            r"(\bSELECT\b.*\bFROM\b.*\bWHERE\b.*\bOR\b.*=.*)",
            r"(\bINSERT\b.*\bINTO\b.*\bVALUES\b)",
            r"(\bUPDATE\b.*\bSET\b.*\bWHERE\b)",
            r"(\bDELETE\b.*\bFROM\b.*\bWHERE\b)",
            r"(\bDROP\b.*\bTABLE\b)",
            r"(\bALTER\b.*\bTABLE\b)",
            r"(\bCREATE\b.*\bTABLE\b)",
            r"(\bEXEC\b.*\bSP_)",
            r"(\bSHOW\b.*\bTABLES\b)",
            r"(\bDESCRIBE\b.*\bTABLE\b)",
            r"('.*OR.*'.*'.*=.*')",
            r"(\".*OR.*\".*\".*=.*\")",
            r"(\bAND\b.*\b1\b.*=.*\b1\b)",
            r"(\bOR\b.*\b1\b.*=.*\b1\b)",
            r"(;\s*--)",
            r"(/\*.*\*/)",
            r"(\bxp_cmdshell\b)",
            r"(\bsp_executesql\b)",
        ]

        return [re.compile(pattern, re.IGNORECASE) for pattern in patterns]

    def _load_xss_patterns(self) -> list[re.Pattern]:
        """Load XSS detection patterns"""
        patterns = [
            r"(<script[^>]*>.*?</script>)",
            r"(<iframe[^>]*>.*?</iframe>)",
            r"(<object[^>]*>.*?</object>)",
            r"(<embed[^>]*>.*?</embed>)",
            r"(<form[^>]*>.*?</form>)",
            r"(javascript:)",
            r"(vbscript:)",
            r"(onload\s*=)",
            r"(onerror\s*=)",
            r"(onclick\s*=)",
            r"(onmouseover\s*=)",
            r"(onfocus\s*=)",
            r"(onblur\s*=)",
            r"(onchange\s*=)",
            r"(onsubmit\s*=)",
            r"(expression\s*\()",
            r"(eval\s*\()",
            r"(alert\s*\()",
            r"(confirm\s*\()",
            r"(prompt\s*\()",
            r"(document\.cookie)",
            r"(document\.write)",
            r"(window\.location)",
            r"(<img[^>]*src\s*=\s*[\"']?javascript:)",
            r"(<link[^>]*href\s*=\s*[\"']?javascript:)",
        ]

        return [re.compile(pattern, re.IGNORECASE) for pattern in patterns]

    def _load_path_traversal_patterns(self) -> list[re.Pattern]:
        """Load path traversal detection patterns"""
        patterns = [
            r"(\.\.\/)",
            r"(\.\.\\)",
            r"(%2e%2e%2f)",
            r"(%2e%2e%5c)",
            r"(%252e%252e%252f)",
            r"(%252e%252e%255c)",
            r"(\.\.%2f)",
            r"(\.\.%5c)",
            r"(%2e%2e\/)",
            r"(%2e%2e\\)",
            r"(\/etc\/passwd)",
            r"(\/etc\/shadow)",
            r"(\/windows\/system32)",
            r"(\/boot\.ini)",
            r"(\/etc\/hosts)",
            r"(\/proc\/self\/environ)",
        ]

        return [re.compile(pattern, re.IGNORECASE) for pattern in patterns]

    def _load_command_injection_patterns(self) -> list[re.Pattern]:
        """Load command injection detection patterns"""
        patterns = [
            r"(;\s*cat\s+)",
            r"(;\s*ls\s+)",
            r"(;\s*pwd\s*)",
            r"(;\s*id\s*)",
            r"(;\s*uname\s+)",
            r"(;\s*wget\s+)",
            r"(;\s*curl\s+)",
            r"(;\s*nc\s+)",
            r"(;\s*netcat\s+)",
            r"(;\s*rm\s+)",
            r"(;\s*chmod\s+)",
            r"(;\s*chown\s+)",
            r"(\|\s*cat\s+)",
            r"(\|\s*ls\s+)",
            r"(\|\s*pwd\s*)",
            r"(\|\s*id\s*)",
            r"(\|\s*uname\s+)",
            r"(\&\&\s*cat\s+)",
            r"(\&\&\s*ls\s+)",
            r"(\&\&\s*pwd\s*)",
            r"(\&\&\s*id\s*)",
            r"(`.*`)",
            r"(\$\(.*\))",
        ]

        return [re.compile(pattern, re.IGNORECASE) for pattern in patterns]

    def _load_blocked_user_agents(self) -> list[re.Pattern]:
        """Load blocked user agent patterns"""
        patterns = [
            r"(sqlmap)",
            r"(nikto)",
            r"(nmap)",
            r"(masscan)",
            r"(zap)",
            r"(burp)",
            r"(w3af)",
            r"(havij)",
            r"(beef)",
            r"(metasploit)",
            r"(python-requests)",
            r"(curl)",
            r"(wget)",
            r"(bot)",
            r"(crawler)",
            r"(spider)",
            r"(scraper)",
            r"(scanner)",
            r"(hack)",
            r"(exploit)",
            r"(attack)",
            r"(malware)",
            r"(virus)",
            r"(trojan)",
        ]

        return [re.compile(pattern, re.IGNORECASE) for pattern in patterns]

    def _load_whitelisted_ips(self) -> set[str]:
        """Load whitelisted IP addresses"""
        # Add common internal IP ranges
        whitelist = {
            "127.0.0.1",
            "::1",
            "localhost",
        }

        # Add private IP ranges
        private_ranges = [
            "10.0.0.0/8",
            "172.16.0.0/12",
            "192.168.0.0/16",
        ]

        for range_str in private_ranges:
            try:
                network = ipaddress.ip_network(range_str)
                whitelist.update(str(ip) for ip in network.hosts())
            except ValueError:
                continue

        return whitelist

    def is_whitelisted(self, ip: str) -> bool:
        """Check if IP is whitelisted"""
        return ip in self.whitelisted_ips

    def check_sql_injection(self, request_data: str) -> tuple[bool, list[str]]:
        """Check for SQL injection patterns"""
        matches = []
        for pattern in self.sql_injection_patterns:
            match = pattern.search(request_data)
            if match:
                matches.append(match.group(0))

        return len(matches) > 0, matches

    def check_xss(self, request_data: str) -> tuple[bool, list[str]]:
        """Check for XSS patterns"""
        matches = []
        for pattern in self.xss_patterns:
            match = pattern.search(request_data)
            if match:
                matches.append(match.group(0))

        return len(matches) > 0, matches

    def check_path_traversal(self, request_data: str) -> tuple[bool, list[str]]:
        """Check for path traversal patterns"""
        matches = []
        for pattern in self.path_traversal_patterns:
            match = pattern.search(request_data)
            if match:
                matches.append(match.group(0))

        return len(matches) > 0, matches

    def check_command_injection(self, request_data: str) -> tuple[bool, list[str]]:
        """Check for command injection patterns"""
        matches = []
        for pattern in self.command_injection_patterns:
            match = pattern.search(request_data)
            if match:
                matches.append(match.group(0))

        return len(matches) > 0, matches

    def check_user_agent(self, user_agent: str) -> tuple[bool, list[str]]:
        """Check for blocked user agent patterns"""
        matches = []
        for pattern in self.blocked_user_agents:
            match = pattern.search(user_agent)
            if match:
                matches.append(match.group(0))

        return len(matches) > 0, matches

    def analyze_request(self, request: Request) -> tuple[bool, list[SecurityEvent]]:
        """Analyze request for security threats"""
        events = []
        blocked = False

        # Get request data
        ip = request.client.host
        user_agent = request.headers.get("user-agent", "")
        path = request.url.path
        method = request.method

        # Skip whitelisted IPs
        if self.is_whitelisted(ip):
            return False, []

        # Collect all request data for analysis
        request_data = f"{path} {str(request.query_params)} {user_agent}"

        # Check for SQL injection
        has_sql, sql_matches = self.check_sql_injection(request_data)
        if has_sql:
            event = SecurityEvent(
                event_type=SecurityEventType.SQL_INJECTION_ATTEMPT,
                threat_level=SecurityThreatLevel.HIGH,
                ip_address=ip,
                user_agent=user_agent,
                request_path=path,
                request_method=method,
                timestamp=datetime.utcnow(),
                details={"patterns": sql_matches, "request_data": request_data},
                event_id=self._generate_event_id(),
                blocked=True,
                action_taken="Request blocked - SQL injection detected",
            )
            events.append(event)
            blocked = True

        # Check for XSS
        has_xss, xss_matches = self.check_xss(request_data)
        if has_xss:
            event = SecurityEvent(
                event_type=SecurityEventType.XSS_ATTEMPT,
                threat_level=SecurityThreatLevel.HIGH,
                ip_address=ip,
                user_agent=user_agent,
                request_path=path,
                request_method=method,
                timestamp=datetime.utcnow(),
                details={"patterns": xss_matches, "request_data": request_data},
                event_id=self._generate_event_id(),
                blocked=True,
                action_taken="Request blocked - XSS detected",
            )
            events.append(event)
            blocked = True

        # Check for path traversal
        has_traversal, traversal_matches = self.check_path_traversal(request_data)
        if has_traversal:
            event = SecurityEvent(
                event_type=SecurityEventType.SUSPICIOUS_PATTERN,
                threat_level=SecurityThreatLevel.MEDIUM,
                ip_address=ip,
                user_agent=user_agent,
                request_path=path,
                request_method=method,
                timestamp=datetime.utcnow(),
                details={
                    "patterns": traversal_matches,
                    "request_data": request_data,
                    "type": "path_traversal",
                },
                event_id=self._generate_event_id(),
                blocked=True,
                action_taken="Request blocked - Path traversal detected",
            )
            events.append(event)
            blocked = True

        # Check for command injection
        has_command, command_matches = self.check_command_injection(request_data)
        if has_command:
            event = SecurityEvent(
                event_type=SecurityEventType.SUSPICIOUS_PATTERN,
                threat_level=SecurityThreatLevel.HIGH,
                ip_address=ip,
                user_agent=user_agent,
                request_path=path,
                request_method=method,
                timestamp=datetime.utcnow(),
                details={
                    "patterns": command_matches,
                    "request_data": request_data,
                    "type": "command_injection",
                },
                event_id=self._generate_event_id(),
                blocked=True,
                action_taken="Request blocked - Command injection detected",
            )
            events.append(event)
            blocked = True

        # Check user agent
        has_blocked_agent, agent_matches = self.check_user_agent(user_agent)
        if has_blocked_agent:
            event = SecurityEvent(
                event_type=SecurityEventType.SUSPICIOUS_PATTERN,
                threat_level=SecurityThreatLevel.MEDIUM,
                ip_address=ip,
                user_agent=user_agent,
                request_path=path,
                request_method=method,
                timestamp=datetime.utcnow(),
                details={
                    "patterns": agent_matches,
                    "user_agent": user_agent,
                    "type": "blocked_user_agent",
                },
                event_id=self._generate_event_id(),
                blocked=True,
                action_taken="Request blocked - Blocked user agent detected",
            )
            events.append(event)
            blocked = True

        # Update threat score
        if events:
            self.threat_scores[ip] += len(events) * 10
            if self.threat_scores[ip] > self.threat_threshold:
                self.blocked_ips.add(ip)

        return blocked, events

    def _generate_event_id(self) -> str:
        """Generate unique event ID"""
        return hashlib.md5(f"{time.time()}{hash(self)}".encode()).hexdigest()


class SecurityMiddleware(BaseHTTPMiddleware):
    """Security middleware combining rate limiting and WAF"""

    def __init__(self, app, rate_limiter: RateLimiter, waf: WebApplicationFirewall):
        super().__init__(app)
        self.rate_limiter = rate_limiter
        self.waf = waf
        self.logger = get_web_ui_logger()
        self.security_events = deque(maxlen=10000)

    async def dispatch(self, request: Request, call_next):
        """Process request through security filters"""
        start_time = time.time()

        # Get client info
        ip = request.client.host
        user_agent = request.headers.get("user-agent", "")
        path = request.url.path
        method = request.method

        try:
            # Skip security checks for health endpoints
            if path in ["/health", "/healthz", "/ready"]:
                return await call_next(request)

            # Rate limiting check
            is_rate_limited, rate_limit_reason = self.rate_limiter.is_rate_limited(
                ip, path
            )
            if is_rate_limited:
                # Record security event
                event = SecurityEvent(
                    event_type=SecurityEventType.RATE_LIMIT_EXCEEDED,
                    threat_level=SecurityThreatLevel.MEDIUM,
                    ip_address=ip,
                    user_agent=user_agent,
                    request_path=path,
                    request_method=method,
                    timestamp=datetime.utcnow(),
                    details={"reason": rate_limit_reason},
                    event_id=self._generate_event_id(),
                    blocked=True,
                    action_taken="Request blocked - Rate limit exceeded",
                )
                self.security_events.append(event)

                self.logger.log(
                    f"Rate limit exceeded for {ip} on {path}: {rate_limit_reason}",
                    level="WARNING",
                )

                return JSONResponse(
                    status_code=status.HTTP_429_TOO_MANY_REQUESTS,
                    content={
                        "error": "Rate limit exceeded",
                        "message": "Too many requests. Please try again later.",
                        "retry_after": 60,
                        "event_id": event.event_id,
                    },
                )

            # WAF analysis
            is_blocked, waf_events = self.waf.analyze_request(request)
            if is_blocked:
                # Record security events
                for event in waf_events:
                    self.security_events.append(event)

                self.logger.log(
                    f"WAF blocked request from {ip} on {path}: {[e.event_type.value for e in waf_events]}",
                    level="CRITICAL",
                )

                return JSONResponse(
                    status_code=status.HTTP_403_FORBIDDEN,
                    content={
                        "error": "Security violation detected",
                        "message": "Your request has been blocked due to security policy violations.",
                        "event_ids": [e.event_id for e in waf_events],
                    },
                )

            # Record successful request
            self.rate_limiter.record_request(ip, path)

            # Process request
            response = await call_next(request)

            # Add security headers
            response.headers["X-Content-Type-Options"] = "nosniff"
            response.headers["X-Frame-Options"] = "DENY"
            response.headers["X-XSS-Protection"] = "1; mode=block"
            response.headers["Strict-Transport-Security"] = (
                "max-age=31536000; includeSubDomains"
            )
            response.headers["Referrer-Policy"] = "strict-origin-when-cross-origin"
            response.headers["Content-Security-Policy"] = (
                "default-src 'self'; script-src 'self' 'unsafe-inline'; style-src 'self' 'unsafe-inline'"
            )

            # Add rate limit headers
            rate_limit_status = self.rate_limiter.get_rate_limit_status(ip, path)
            response.headers["X-RateLimit-Limit"] = str(
                rate_limit_status["limits"]["minute"]
            )
            response.headers["X-RateLimit-Remaining"] = str(
                rate_limit_status["remaining"]["minute"]
            )
            response.headers["X-RateLimit-Reset"] = str(int(time.time()) + 60)

            return response

        except Exception as e:
            self.logger.log(f"Error in security middleware: {e}", level="ERROR")

            # Record security event for middleware error
            event = SecurityEvent(
                event_type=SecurityEventType.UNUSUAL_REQUEST_PATTERN,
                threat_level=SecurityThreatLevel.MEDIUM,
                ip_address=ip,
                user_agent=user_agent,
                request_path=path,
                request_method=method,
                timestamp=datetime.utcnow(),
                details={"error": str(e), "type": "middleware_error"},
                event_id=self._generate_event_id(),
                blocked=False,
                action_taken="Request processed with error",
            )
            self.security_events.append(event)

            return JSONResponse(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                content={"error": "Internal server error", "event_id": event.event_id},
            )

    def _generate_event_id(self) -> str:
        """Generate unique event ID"""
        return hashlib.md5(f"{time.time()}{hash(self)}".encode()).hexdigest()

    def get_security_events(self, limit: int = 100) -> list[dict[str, Any]]:
        """Get recent security events"""
        events = list(self.security_events)[-limit:]
        return [
            {
                "event_id": event.event_id,
                "event_type": event.event_type.value,
                "threat_level": event.threat_level.value,
                "ip_address": event.ip_address,
                "user_agent": event.user_agent,
                "request_path": event.request_path,
                "request_method": event.request_method,
                "timestamp": event.timestamp.isoformat(),
                "details": event.details,
                "blocked": event.blocked,
                "action_taken": event.action_taken,
            }
            for event in events
        ]

    def get_security_metrics(self) -> dict[str, Any]:
        """Get security metrics"""
        events = list(self.security_events)
        current_time = datetime.utcnow()

        # Count events by type
        event_counts = defaultdict(int)
        threat_counts = defaultdict(int)
        ip_counts = defaultdict(int)

        # Recent events (last hour)
        recent_events = [
            e for e in events if (current_time - e.timestamp).total_seconds() < 3600
        ]
        blocked_events = [e for e in events if e.blocked]

        for event in events:
            event_counts[event.event_type.value] += 1
            threat_counts[event.threat_level.value] += 1
            ip_counts[event.ip_address] += 1

        return {
            "total_events": len(events),
            "blocked_requests": len(blocked_events),
            "recent_events": len(recent_events),
            "events_by_type": dict(event_counts),
            "events_by_threat_level": dict(threat_counts),
            "top_ips": dict(
                sorted(ip_counts.items(), key=lambda x: x[1], reverse=True)[:10]
            ),
            "rate_limiter_status": {
                "blocked_ips": len(self.rate_limiter.blocked_ips),
                "total_requests_tracked": sum(
                    len(ts) for ts in self.rate_limiter.request_timestamps.values()
                ),
            },
            "waf_status": {
                "blocked_ips": len(self.waf.blocked_ips),
                "threat_scores": len(self.waf.threat_scores),
            },
        }


# Global instances
_rate_limiter = None
_waf = None
_security_middleware = None


def get_rate_limiter() -> RateLimiter:
    """Get global rate limiter instance"""
    global _rate_limiter
    if _rate_limiter is None:
        _rate_limiter = RateLimiter()
    return _rate_limiter


def get_waf() -> WebApplicationFirewall:
    """Get global WAF instance"""
    global _waf
    if _waf is None:
        _waf = WebApplicationFirewall()
    return _waf


def get_security_middleware() -> SecurityMiddleware:
    """Get global security middleware instance"""
    global _security_middleware
    if _security_middleware is None:
        rate_limiter = get_rate_limiter()
        waf = get_waf()
        _security_middleware = SecurityMiddleware(None, rate_limiter, waf)
    return _security_middleware


def rate_limit(requests_per_minute: int = 60, requests_per_hour: int = 1000):
    """Decorator for endpoint-specific rate limiting"""

    def decorator(func):
        @wraps(func)
        async def wrapper(*args, **kwargs):
            # Get request from args/kwargs
            request = None
            for arg in args:
                if isinstance(arg, Request):
                    request = arg
                    break

            if not request:
                return await func(*args, **kwargs)

            # Check rate limit
            rate_limiter = get_rate_limiter()
            ip = request.client.host
            endpoint = request.url.path

            is_limited, reason = rate_limiter.is_rate_limited(ip, endpoint)
            if is_limited:
                raise HTTPException(
                    status_code=status.HTTP_429_TOO_MANY_REQUESTS,
                    detail=f"Rate limit exceeded: {reason}",
                )

            # Record request
            rate_limiter.record_request(ip, endpoint)

            return await func(*args, **kwargs)

        return wrapper

    return decorator
