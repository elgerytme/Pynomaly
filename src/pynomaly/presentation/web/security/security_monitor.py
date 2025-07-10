"""
Comprehensive Security Monitoring and Alerting System
Implements real-time security event detection, analysis, and response
"""

import asyncio
import hashlib
import json
import logging
import time
from collections import defaultdict, deque
from collections.abc import Callable
from dataclasses import asdict, dataclass
from datetime import datetime
from enum import Enum
from typing import Any

from fastapi import Request, Response
from starlette.middleware.base import BaseHTTPMiddleware

# Setup logging
logger = logging.getLogger(__name__)


class SecurityEventType(Enum):
    """Types of security events"""

    XSS_ATTEMPT = "xss_attempt"
    SQL_INJECTION = "sql_injection"
    CSRF_VIOLATION = "csrf_violation"
    BRUTE_FORCE = "brute_force"
    DDOS_ATTEMPT = "ddos_attempt"
    SUSPICIOUS_REQUEST = "suspicious_request"
    AUTH_FAILURE = "auth_failure"
    PRIVILEGE_ESCALATION = "privilege_escalation"
    DATA_EXFILTRATION = "data_exfiltration"
    MALWARE_UPLOAD = "malware_upload"
    SCANNER_DETECTED = "scanner_detected"
    CLICKJACKING = "clickjacking"
    SESSION_HIJACK = "session_hijack"
    ANOMALOUS_BEHAVIOR = "anomalous_behavior"
    CSP_VIOLATION = "csp_violation"
    RATE_LIMIT_EXCEEDED = "rate_limit_exceeded"


class SecurityThreatLevel(Enum):
    """Security threat severity levels"""

    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


class SecurityResponse(Enum):
    """Security response actions"""

    LOG_ONLY = "log_only"
    MONITOR = "monitor"
    BLOCK_REQUEST = "block_request"
    BLOCK_IP = "block_ip"
    ALERT_ADMIN = "alert_admin"
    AUTO_REMEDIATE = "auto_remediate"


@dataclass
class SecurityEvent:
    """Security event data structure"""

    event_id: str
    event_type: SecurityEventType
    threat_level: SecurityThreatLevel
    timestamp: float
    source_ip: str
    user_agent: str
    request_path: str
    request_method: str
    details: dict[str, Any]
    risk_score: int
    response_action: SecurityResponse
    blocked: bool = False
    mitigated: bool = False


@dataclass
class SecurityMetrics:
    """Security monitoring metrics"""

    total_events: int = 0
    events_by_type: dict[str, int] = None
    events_by_severity: dict[str, int] = None
    blocked_requests: int = 0
    blocked_ips: int = 0
    average_risk_score: float = 0.0
    peak_events_per_minute: int = 0
    last_updated: float = 0

    def __post_init__(self):
        if self.events_by_type is None:
            self.events_by_type = defaultdict(int)
        if self.events_by_severity is None:
            self.events_by_severity = defaultdict(int)


class SecurityAnalyzer:
    """Analyzes security events and determines threat levels"""

    def __init__(self):
        self.threat_patterns = {
            # XSS patterns
            SecurityEventType.XSS_ATTEMPT: [
                r"<script[^>]*>.*?</script>",
                r"javascript:",
                r"on\w+\s*=",
                r"eval\s*\(",
                r"document\.write",
                r"innerHTML\s*=",
            ],
            # SQL Injection patterns
            SecurityEventType.SQL_INJECTION: [
                r"union\s+select",
                r"drop\s+table",
                r"insert\s+into",
                r"delete\s+from",
                r"update\s+.*\s+set",
                r"exec\s*\(",
                r"@@version",
                r"information_schema",
            ],
            # Command injection patterns
            SecurityEventType.SUSPICIOUS_REQUEST: [
                r";\s*(rm|del|format)",
                r"\|\s*(curl|wget)",
                r"&&\s*(cat|type)",
                r"`.*`",
                r"\$\(.*\)",
            ],
        }

        self.risk_weights = {
            SecurityEventType.XSS_ATTEMPT: 70,
            SecurityEventType.SQL_INJECTION: 90,
            SecurityEventType.CSRF_VIOLATION: 60,
            SecurityEventType.BRUTE_FORCE: 50,
            SecurityEventType.DDOS_ATTEMPT: 80,
            SecurityEventType.MALWARE_UPLOAD: 95,
            SecurityEventType.DATA_EXFILTRATION: 100,
            SecurityEventType.PRIVILEGE_ESCALATION: 85,
            SecurityEventType.SESSION_HIJACK: 90,
            SecurityEventType.SCANNER_DETECTED: 40,
            SecurityEventType.CLICKJACKING: 30,
            SecurityEventType.CSP_VIOLATION: 45,
            SecurityEventType.RATE_LIMIT_EXCEEDED: 25,
            SecurityEventType.AUTH_FAILURE: 35,
            SecurityEventType.ANOMALOUS_BEHAVIOR: 50,
            SecurityEventType.SUSPICIOUS_REQUEST: 65,
        }

    def analyze_request(
        self, request: Request, content: str = ""
    ) -> list[SecurityEvent]:
        """Analyze request for security threats"""
        events = []

        # Combine request data for analysis
        analysis_data = self._extract_request_data(request, content)

        # Check for known attack patterns
        for event_type, patterns in self.threat_patterns.items():
            for pattern in patterns:
                import re

                if re.search(pattern, analysis_data, re.IGNORECASE):
                    event = self._create_security_event(
                        event_type=event_type,
                        request=request,
                        details={
                            "pattern_matched": pattern,
                            "content_sample": analysis_data[:200],
                        },
                        risk_score=self.risk_weights.get(event_type, 50),
                    )
                    events.append(event)
                    break  # Only report first match per type

        # Additional heuristic checks
        events.extend(self._check_heuristics(request, analysis_data))

        return events

    def _extract_request_data(self, request: Request, content: str) -> str:
        """Extract all request data for analysis"""
        data_parts = [
            str(request.url),
            request.method,
            " ".join([f"{k}:{v}" for k, v in request.headers.items()]),
            content,
        ]

        return " ".join(data_parts)

    def _check_heuristics(self, request: Request, data: str) -> list[SecurityEvent]:
        """Additional heuristic security checks"""
        events = []

        # Check for suspicious user agents
        user_agent = request.headers.get("User-Agent", "")
        if self._is_suspicious_user_agent(user_agent):
            events.append(
                self._create_security_event(
                    event_type=SecurityEventType.SCANNER_DETECTED,
                    request=request,
                    details={"user_agent": user_agent},
                    risk_score=40,
                )
            )

        # Check for excessive parameter count
        if hasattr(request, "query_params") and len(request.query_params) > 50:
            events.append(
                self._create_security_event(
                    event_type=SecurityEventType.SUSPICIOUS_REQUEST,
                    request=request,
                    details={"parameter_count": len(request.query_params)},
                    risk_score=30,
                )
            )

        # Check for suspicious file extensions in path
        path = request.url.path
        suspicious_extensions = [".php", ".asp", ".jsp", ".exe", ".bat", ".cmd"]
        for ext in suspicious_extensions:
            if path.endswith(ext):
                events.append(
                    self._create_security_event(
                        event_type=SecurityEventType.SUSPICIOUS_REQUEST,
                        request=request,
                        details={"suspicious_extension": ext, "path": path},
                        risk_score=45,
                    )
                )
                break

        return events

    def _is_suspicious_user_agent(self, user_agent: str) -> bool:
        """Check if user agent is suspicious"""
        suspicious_patterns = [
            "nikto",
            "nmap",
            "dirb",
            "dirbuster",
            "gobuster",
            "sqlmap",
            "burp",
            "wpscan",
            "masscan",
            "zap",
            "w3af",
            "acunetix",
            "nessus",
            "openvas",
        ]

        return any(pattern in user_agent.lower() for pattern in suspicious_patterns)

    def _create_security_event(
        self,
        event_type: SecurityEventType,
        request: Request,
        details: dict[str, Any],
        risk_score: int,
    ) -> SecurityEvent:
        """Create a security event"""

        # Determine threat level based on risk score
        if risk_score >= 80:
            threat_level = SecurityThreatLevel.CRITICAL
        elif risk_score >= 60:
            threat_level = SecurityThreatLevel.HIGH
        elif risk_score >= 40:
            threat_level = SecurityThreatLevel.MEDIUM
        else:
            threat_level = SecurityThreatLevel.LOW

        # Determine response action
        response_action = self._determine_response(event_type, threat_level, risk_score)

        return SecurityEvent(
            event_id=self._generate_event_id(),
            event_type=event_type,
            threat_level=threat_level,
            timestamp=time.time(),
            source_ip=self._get_client_ip(request),
            user_agent=request.headers.get("User-Agent", ""),
            request_path=request.url.path,
            request_method=request.method,
            details=details,
            risk_score=risk_score,
            response_action=response_action,
            blocked=response_action
            in [SecurityResponse.BLOCK_REQUEST, SecurityResponse.BLOCK_IP],
        )

    def _determine_response(
        self,
        event_type: SecurityEventType,
        threat_level: SecurityThreatLevel,
        risk_score: int,
    ) -> SecurityResponse:
        """Determine appropriate response action"""

        # Critical threats - block immediately
        if threat_level == SecurityThreatLevel.CRITICAL:
            if event_type in [
                SecurityEventType.SQL_INJECTION,
                SecurityEventType.MALWARE_UPLOAD,
            ]:
                return SecurityResponse.BLOCK_IP
            return SecurityResponse.BLOCK_REQUEST

        # High threats - block request and alert
        elif threat_level == SecurityThreatLevel.HIGH:
            return SecurityResponse.BLOCK_REQUEST

        # Medium threats - monitor and potentially block
        elif threat_level == SecurityThreatLevel.MEDIUM:
            if event_type in [
                SecurityEventType.BRUTE_FORCE,
                SecurityEventType.DDOS_ATTEMPT,
            ]:
                return SecurityResponse.BLOCK_REQUEST
            return SecurityResponse.MONITOR

        # Low threats - log only
        else:
            return SecurityResponse.LOG_ONLY

    def _generate_event_id(self) -> str:
        """Generate unique event ID"""
        timestamp = str(time.time())
        return hashlib.md5(timestamp.encode()).hexdigest()[:12]

    def _get_client_ip(self, request: Request) -> str:
        """Get client IP address"""
        forwarded = request.headers.get("X-Forwarded-For")
        if forwarded:
            return forwarded.split(",")[0].strip()

        real_ip = request.headers.get("X-Real-IP")
        if real_ip:
            return real_ip

        return request.client.host if request.client else "unknown"


class SecurityAlertManager:
    """Manages security alerts and notifications"""

    def __init__(self):
        self.alert_handlers: list[Callable] = []
        self.alert_thresholds = {
            SecurityThreatLevel.CRITICAL: 1,  # Alert immediately
            SecurityThreatLevel.HIGH: 3,  # Alert after 3 events
            SecurityThreatLevel.MEDIUM: 10,  # Alert after 10 events
            SecurityThreatLevel.LOW: 50,  # Alert after 50 events
        }
        self.alert_counts = defaultdict(int)
        self.last_alert_time = defaultdict(float)
        self.alert_cooldown = 300  # 5 minutes between alerts of same type

    def register_alert_handler(self, handler: Callable):
        """Register an alert handler function"""
        self.alert_handlers.append(handler)

    async def process_event(self, event: SecurityEvent):
        """Process security event and trigger alerts if needed"""
        # Increment alert count for this threat level
        self.alert_counts[event.threat_level] += 1

        # Check if we should trigger an alert
        threshold = self.alert_thresholds.get(event.threat_level, 10)

        if self.alert_counts[event.threat_level] >= threshold:
            await self._trigger_alert(event)
            self.alert_counts[event.threat_level] = 0  # Reset counter

    async def _trigger_alert(self, event: SecurityEvent):
        """Trigger security alert"""
        # Check cooldown
        alert_key = f"{event.event_type.value}:{event.threat_level.value}"
        now = time.time()

        if (now - self.last_alert_time[alert_key]) < self.alert_cooldown:
            return  # Still in cooldown period

        self.last_alert_time[alert_key] = now

        # Create alert data
        alert_data = {
            "alert_id": self._generate_alert_id(),
            "timestamp": datetime.utcnow().isoformat(),
            "event": asdict(event),
            "severity": event.threat_level.value,
            "message": self._generate_alert_message(event),
            "recommended_actions": self._get_recommended_actions(event),
        }

        # Send to all registered handlers
        for handler in self.alert_handlers:
            try:
                await handler(alert_data)
            except Exception as e:
                logger.error(f"Alert handler failed: {e}")

    def _generate_alert_id(self) -> str:
        """Generate unique alert ID"""
        timestamp = str(time.time())
        return f"ALERT-{hashlib.md5(timestamp.encode()).hexdigest()[:8].upper()}"

    def _generate_alert_message(self, event: SecurityEvent) -> str:
        """Generate human-readable alert message"""
        messages = {
            SecurityEventType.XSS_ATTEMPT: f"XSS attack attempt detected from {event.source_ip}",
            SecurityEventType.SQL_INJECTION: f"SQL injection attack detected from {event.source_ip}",
            SecurityEventType.BRUTE_FORCE: f"Brute force attack detected from {event.source_ip}",
            SecurityEventType.DDOS_ATTEMPT: f"DDoS attack attempt detected from {event.source_ip}",
            SecurityEventType.MALWARE_UPLOAD: f"Malware upload attempt detected from {event.source_ip}",
            SecurityEventType.DATA_EXFILTRATION: f"Data exfiltration attempt detected from {event.source_ip}",
        }

        return messages.get(
            event.event_type,
            f"Security event {event.event_type.value} "
            f"detected from {event.source_ip}",
        )

    def _get_recommended_actions(self, event: SecurityEvent) -> list[str]:
        """Get recommended actions for security event"""
        actions = {
            SecurityEventType.XSS_ATTEMPT: [
                "Review and sanitize user inputs",
                "Check Content Security Policy configuration",
                "Verify output encoding is properly implemented",
            ],
            SecurityEventType.SQL_INJECTION: [
                "Review database query parameterization",
                "Check input validation and sanitization",
                "Consider implementing SQL injection detection rules",
            ],
            SecurityEventType.BRUTE_FORCE: [
                "Implement account lockout policies",
                "Enable multi-factor authentication",
                "Consider IP-based rate limiting",
            ],
            SecurityEventType.DDOS_ATTEMPT: [
                "Enable DDoS protection",
                "Implement rate limiting",
                "Consider using a CDN for traffic filtering",
            ],
        }

        return actions.get(
            event.event_type, ["Review security logs", "Investigate the incident"]
        )


class SecurityMonitor:
    """Main security monitoring system"""

    def __init__(self, config: dict[str, Any] = None):
        self.config = config or {}
        self.analyzer = SecurityAnalyzer()
        self.alert_manager = SecurityAlertManager()
        self.metrics = SecurityMetrics()

        # Event storage (in-memory for now, could be extended to database)
        self.recent_events = deque(maxlen=10000)
        self.events_by_ip = defaultdict(list)
        self.blocked_ips = set()

        # Rate limiting tracking
        self.request_counts = defaultdict(lambda: defaultdict(int))
        self.rate_limit_windows = defaultdict(float)

        # Performance metrics
        self.events_per_minute = deque(maxlen=60)  # Track last 60 minutes

        # Start background tasks
        self._start_background_tasks()

    def _start_background_tasks(self):
        """Start background monitoring tasks"""
        # Cleanup old events
        asyncio.create_task(self._cleanup_old_events())

        # Update metrics
        asyncio.create_task(self._update_metrics())

        # Process event queue
        asyncio.create_task(self._process_event_queue())

    async def process_request(
        self, request: Request, content: str = ""
    ) -> list[SecurityEvent]:
        """Process incoming request for security analysis"""
        # Rate limiting check
        client_ip = self._get_client_ip(request)
        if self._check_rate_limit(client_ip):
            rate_limit_event = SecurityEvent(
                event_id=self.analyzer._generate_event_id(),
                event_type=SecurityEventType.RATE_LIMIT_EXCEEDED,
                threat_level=SecurityThreatLevel.MEDIUM,
                timestamp=time.time(),
                source_ip=client_ip,
                user_agent=request.headers.get("User-Agent", ""),
                request_path=request.url.path,
                request_method=request.method,
                details={"rate_limit_exceeded": True},
                risk_score=40,
                response_action=SecurityResponse.BLOCK_REQUEST,
                blocked=True,
            )
            await self._record_event(rate_limit_event)
            return [rate_limit_event]

        # Analyze request for security threats
        events = self.analyzer.analyze_request(request, content)

        # Record all events
        for event in events:
            await self._record_event(event)

        return events

    async def _record_event(self, event: SecurityEvent):
        """Record security event"""
        # Add to recent events
        self.recent_events.append(event)

        # Track by IP
        self.events_by_ip[event.source_ip].append(event)

        # Update metrics
        self.metrics.total_events += 1
        self.metrics.events_by_type[event.event_type.value] += 1
        self.metrics.events_by_severity[event.threat_level.value] += 1

        if event.blocked:
            self.metrics.blocked_requests += 1

        # Update risk score average
        total_score = sum(e.risk_score for e in self.recent_events)
        self.metrics.average_risk_score = total_score / len(self.recent_events)

        # Process through alert manager
        await self.alert_manager.process_event(event)

        # Auto-block IP if needed
        if event.response_action == SecurityResponse.BLOCK_IP:
            self.blocked_ips.add(event.source_ip)
            self.metrics.blocked_ips = len(self.blocked_ips)

        logger.info(
            f"Security event recorded: {event.event_type.value} from {event.source_ip}"
        )

    def _check_rate_limit(self, client_ip: str) -> bool:
        """Check if client IP exceeds rate limit"""
        now = time.time()
        minute = int(now // 60)

        # Default rate limit: 100 requests per minute
        rate_limit = self.config.get("rate_limit_per_minute", 100)

        self.request_counts[client_ip][minute] += 1

        # Clean old entries
        cutoff = minute - 1
        for ip in list(self.request_counts.keys()):
            self.request_counts[ip] = {
                m: count for m, count in self.request_counts[ip].items() if m > cutoff
            }

        return self.request_counts[client_ip][minute] > rate_limit

    def _get_client_ip(self, request: Request) -> str:
        """Get client IP address"""
        return self.analyzer._get_client_ip(request)

    def is_blocked_ip(self, ip: str) -> bool:
        """Check if IP is blocked"""
        return ip in self.blocked_ips

    def unblock_ip(self, ip: str) -> bool:
        """Unblock an IP address"""
        if ip in self.blocked_ips:
            self.blocked_ips.remove(ip)
            self.metrics.blocked_ips = len(self.blocked_ips)
            return True
        return False

    def get_metrics(self) -> dict[str, Any]:
        """Get current security metrics"""
        self.metrics.last_updated = time.time()
        return asdict(self.metrics)

    def get_recent_events(self, limit: int = 100) -> list[dict[str, Any]]:
        """Get recent security events"""
        events = list(self.recent_events)[-limit:]
        return [asdict(event) for event in events]

    def get_ip_summary(self, ip: str) -> dict[str, Any]:
        """Get security summary for specific IP"""
        events = self.events_by_ip.get(ip, [])

        if not events:
            return {"ip": ip, "events": 0, "risk_score": 0, "blocked": False}

        total_risk = sum(event.risk_score for event in events)
        avg_risk = total_risk / len(events)

        event_types = defaultdict(int)
        for event in events:
            event_types[event.event_type.value] += 1

        return {
            "ip": ip,
            "events": len(events),
            "avg_risk_score": avg_risk,
            "total_risk_score": total_risk,
            "blocked": ip in self.blocked_ips,
            "event_types": dict(event_types),
            "first_seen": min(event.timestamp for event in events),
            "last_seen": max(event.timestamp for event in events),
        }

    async def _cleanup_old_events(self):
        """Cleanup old events periodically"""
        while True:
            try:
                await asyncio.sleep(3600)  # Run every hour

                cutoff_time = time.time() - (24 * 3600)  # 24 hours ago

                # Clean events by IP
                for ip in list(self.events_by_ip.keys()):
                    self.events_by_ip[ip] = [
                        event
                        for event in self.events_by_ip[ip]
                        if event.timestamp > cutoff_time
                    ]

                    if not self.events_by_ip[ip]:
                        del self.events_by_ip[ip]

                logger.info("Cleaned up old security events")

            except Exception as e:
                logger.error(f"Error cleaning up events: {e}")

    async def _update_metrics(self):
        """Update performance metrics periodically"""
        while True:
            try:
                await asyncio.sleep(60)  # Update every minute

                # Count events in last minute
                now = time.time()
                minute_ago = now - 60

                events_last_minute = sum(
                    1 for event in self.recent_events if event.timestamp > minute_ago
                )

                self.events_per_minute.append(events_last_minute)
                self.metrics.peak_events_per_minute = max(self.events_per_minute)

            except Exception as e:
                logger.error(f"Error updating metrics: {e}")

    async def _process_event_queue(self):
        """Process event queue for additional analysis"""
        while True:
            try:
                await asyncio.sleep(30)  # Process every 30 seconds

                # Look for attack patterns across multiple IPs
                await self._detect_distributed_attacks()

                # Analyze traffic patterns
                await self._analyze_traffic_patterns()

            except Exception as e:
                logger.error(f"Error processing event queue: {e}")

    async def _detect_distributed_attacks(self):
        """Detect distributed attacks across multiple IPs"""
        # Look for coordinated attacks from multiple IPs
        recent_time = time.time() - 300  # Last 5 minutes
        recent_events = [e for e in self.recent_events if e.timestamp > recent_time]

        # Group by event type and time window
        attack_groups = defaultdict(list)
        for event in recent_events:
            time_window = int(event.timestamp // 60)  # 1-minute windows
            key = f"{event.event_type.value}:{time_window}"
            attack_groups[key].append(event)

        # Look for suspicious patterns
        for key, events in attack_groups.items():
            if len(events) >= 5:  # 5 or more similar events in same minute
                unique_ips = len(set(event.source_ip for event in events))
                if unique_ips >= 3:  # From 3+ different IPs
                    # This looks like a coordinated attack
                    logger.warning(
                        f"Distributed attack detected: {key} " f"from {unique_ips} IPs"
                    )

    async def _analyze_traffic_patterns(self):
        """Analyze traffic patterns for anomalies"""
        # Simple anomaly detection based on request patterns
        now = time.time()
        hour_ago = now - 3600

        recent_events = [e for e in self.recent_events if e.timestamp > hour_ago]

        if len(recent_events) > 100:  # Only analyze if we have enough data
            # Check for unusual spikes
            events_per_minute = defaultdict(int)
            for event in recent_events:
                minute = int(event.timestamp // 60)
                events_per_minute[minute] += 1

            if events_per_minute:
                avg_per_minute = sum(events_per_minute.values()) / len(
                    events_per_minute
                )
                max_per_minute = max(events_per_minute.values())

                # If max is 3x average, consider it suspicious
                if max_per_minute > avg_per_minute * 3:
                    logger.warning(
                        f"Traffic spike detected: {max_per_minute} "
                        f"events/minute (avg: {avg_per_minute:.1f})"
                    )


class SecurityMonitoringMiddleware(BaseHTTPMiddleware):
    """Middleware for real-time security monitoring"""

    def __init__(self, app, security_monitor: SecurityMonitor):
        super().__init__(app)
        self.security_monitor = security_monitor

    async def dispatch(self, request: Request, call_next):
        """Process request through security monitor"""
        # Check if IP is blocked
        client_ip = self.security_monitor._get_client_ip(request)
        if self.security_monitor.is_blocked_ip(client_ip):
            return Response(
                content="Access denied due to security policy",
                status_code=403,
                headers={"X-Security-Block": "IP-BLOCKED"},
            )

        # Get request content for analysis
        content = ""
        if request.method in ["POST", "PUT", "PATCH"]:
            try:
                body = await request.body()
                content = body.decode("utf-8", errors="ignore")
                # Restore body for downstream processing
                request._body = body
            except:
                pass

        # Analyze request
        events = await self.security_monitor.process_request(request, content)

        # Check if any events require blocking
        blocked_events = [e for e in events if e.blocked]
        if blocked_events:
            highest_risk = max(e.risk_score for e in blocked_events)
            return Response(
                content=json.dumps(
                    {
                        "error": "Request blocked by security policy",
                        "risk_score": highest_risk,
                        "event_types": [e.event_type.value for e in blocked_events],
                    }
                ),
                status_code=403,
                headers={
                    "Content-Type": "application/json",
                    "X-Security-Block": "REQUEST-BLOCKED",
                },
            )

        # Continue with normal processing
        response = await call_next(request)

        # Add security headers
        response.headers["X-Security-Monitor"] = "active"
        response.headers["X-Request-Risk-Score"] = str(
            max((e.risk_score for e in events), default=0)
        )

        return response


# Global security monitor instance
_security_monitor = None


def get_security_monitor(config: dict[str, Any] = None) -> SecurityMonitor:
    """Get or create security monitor instance"""
    global _security_monitor
    if _security_monitor is None:
        _security_monitor = SecurityMonitor(config)
    return _security_monitor


# Email alert handler
async def email_alert_handler(alert_data: dict[str, Any]):
    """Send security alert via email"""
    # Implementation would integrate with email service
    logger.critical(f"SECURITY ALERT: {alert_data['message']}")


# Slack alert handler
async def slack_alert_handler(alert_data: dict[str, Any]):
    """Send security alert to Slack"""
    # Implementation would integrate with Slack API
    logger.critical(f"SLACK ALERT: {alert_data['message']}")


# Setup function
def setup_security_monitoring(app, config: dict[str, Any] = None):
    """Setup security monitoring for FastAPI app"""
    monitor = get_security_monitor(config)

    # Register alert handlers
    monitor.alert_manager.register_alert_handler(email_alert_handler)
    monitor.alert_manager.register_alert_handler(slack_alert_handler)

    # Add middleware
    app.add_middleware(SecurityMonitoringMiddleware, security_monitor=monitor)

    return monitor
