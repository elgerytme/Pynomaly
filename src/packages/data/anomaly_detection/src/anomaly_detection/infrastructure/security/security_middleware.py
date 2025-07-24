"""Comprehensive security middleware for API protection and request validation."""

from __future__ import annotations

import hashlib
import hmac
import json
import re
import time
from collections import defaultdict, deque
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional, Set, Tuple, Callable, Union
from urllib.parse import urlparse
import secrets
import ipaddress

from fastapi import Request, Response, HTTPException, status
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.responses import JSONResponse
import jwt

from ..logging import get_logger
from ...application.services.security.threat_detector import (
    get_threat_detection_system,
    ThreatEvent,
    ThreatType,
    ThreatSeverity
)

logger = get_logger(__name__)

# Lazy import metrics collector to avoid None issues
def get_safe_metrics_collector():
    """Get metrics collector with safe fallback."""
    try:
        from ..monitoring import get_metrics_collector
        return get_metrics_collector()
    except Exception:
        class MockMetricsCollector:
            def record_metric(self, *args, **kwargs):
                pass
        return MockMetricsCollector()


class SecurityConfig:
    """Security middleware configuration."""
    
    def __init__(self):
        # Rate limiting
        self.rate_limit_requests = 100  # requests per minute
        self.rate_limit_window = 60  # seconds
        self.rate_limit_burst = 20  # burst allowance
        
        # IP filtering
        self.blocked_ips: Set[str] = set()
        self.allowed_ips: Optional[Set[str]] = None  # If set, only these IPs allowed
        self.blocked_countries: Set[str] = set()
        
        # Request validation
        self.max_request_size = 10 * 1024 * 1024  # 10MB
        self.max_header_size = 8 * 1024  # 8KB
        self.max_url_length = 2048
        self.blocked_user_agents: Set[str] = {
            "sqlmap", "nikto", "gobuster", "dirb", "nmap", "masscan"
        }
        
        # Security headers
        self.security_headers = {
            "X-Content-Type-Options": "nosniff",
            "X-Frame-Options": "DENY",
            "X-XSS-Protection": "1; mode=block",
            "Strict-Transport-Security": "max-age=31536000; includeSubDomains",
            "Content-Security-Policy": "default-src 'self'; script-src 'self'; style-src 'self' 'unsafe-inline'",
            "Referrer-Policy": "strict-origin-when-cross-origin",
            "Permissions-Policy": "microphone=(), camera=(), geolocation=()"
        }
        
        # Authentication
        self.jwt_secret_key: Optional[str] = None
        self.jwt_algorithm = "HS256"
        self.jwt_expiry_minutes = 60
        self.require_auth_paths: Set[str] = {"/api/v1/admin", "/api/v1/models"}
        self.public_paths: Set[str] = {"/health", "/docs", "/openapi.json"}
        
        # Input validation
        self.sql_injection_patterns = [
            r"(?i)(union\s+select|select\s+\*\s+from|insert\s+into)",
            r"(?i)('|\")(\s*;\s*)(drop|truncate|delete|update)",
            r"(?i)(or\s+1\s*=\s*1|and\s+1\s*=\s*1)",
            r"(?i)(exec\s*\(|execute\s*\(|sp_executesql)"
        ]
        
        self.xss_patterns = [
            r"(?i)<script[^>]*>.*?</script>",
            r"(?i)javascript:",
            r"(?i)on\w+\s*=\s*[\"'][^\"']*[\"']",
            r"(?i)<iframe[^>]*>"
        ]
        
        # Threat detection
        self.enable_threat_detection = True
        self.threat_detection_sampling = 1.0  # 100% sampling


class RateLimiter:
    """Advanced rate limiter with sliding window and burst protection."""
    
    def __init__(self, requests_per_minute: int = 100, burst_allowance: int = 20):
        self.requests_per_minute = requests_per_minute
        self.burst_allowance = burst_allowance
        self.requests: Dict[str, deque] = defaultdict(deque)
        self.burst_tokens: Dict[str, int] = defaultdict(lambda: burst_allowance)
        self.last_refill: Dict[str, float] = defaultdict(time.time)
    
    def is_allowed(self, client_id: str) -> Tuple[bool, Dict[str, Any]]:
        """Check if request is allowed under rate limit."""
        now = time.time()
        minute_ago = now - 60
        
        # Clean old requests
        while self.requests[client_id] and self.requests[client_id][0] < minute_ago:
            self.requests[client_id].popleft()
        
        # Refill burst tokens
        time_since_refill = now - self.last_refill[client_id]
        if time_since_refill > 1:  # Refill every second
            tokens_to_add = int(time_since_refill * (self.burst_allowance / 60))
            self.burst_tokens[client_id] = min(
                self.burst_allowance,
                self.burst_tokens[client_id] + tokens_to_add
            )
            self.last_refill[client_id] = now
        
        # Check rate limit
        current_requests = len(self.requests[client_id])
        
        # Allow if under normal rate limit
        if current_requests < self.requests_per_minute:
            self.requests[client_id].append(now)
            return True, {
                "allowed": True,
                "requests_remaining": self.requests_per_minute - current_requests - 1,
                "reset_time": minute_ago + 60,
                "burst_tokens": self.burst_tokens[client_id]
            }
        
        # Check burst allowance
        if self.burst_tokens[client_id] > 0:
            self.burst_tokens[client_id] -= 1
            self.requests[client_id].append(now)
            return True, {
                "allowed": True,
                "requests_remaining": 0,
                "burst_tokens": self.burst_tokens[client_id],
                "reset_time": minute_ago + 60
            }
        
        # Rate limited
        return False, {
            "allowed": False,
            "requests_remaining": 0,
            "burst_tokens": 0,
            "reset_time": minute_ago + 60,
            "retry_after": 60
        }


class IPFilter:
    """IP-based access control with geolocation and reputation filtering."""
    
    def __init__(self, config: SecurityConfig):
        self.config = config
        self.blocked_networks: List[ipaddress.IPv4Network] = []
        self.allowed_networks: List[ipaddress.IPv4Network] = []
        
        # Add common malicious networks (example)
        self._load_threat_intelligence()
    
    def _load_threat_intelligence(self):
        """Load threat intelligence data."""
        # This would integrate with threat intelligence feeds
        # For now, add some common suspicious ranges
        suspicious_ranges = [
            "0.0.0.0/8",      # "This" network
            "127.0.0.0/8",    # Loopback (except for development)
            "169.254.0.0/16", # Link-local
            "224.0.0.0/4",    # Multicast
        ]
        
        for range_str in suspicious_ranges:
            try:
                network = ipaddress.ip_network(range_str)
                if network.version == 4:
                    self.blocked_networks.append(network)
            except ValueError:
                logger.warning(f"Invalid IP range in threat intelligence: {range_str}")
    
    def is_allowed(self, ip_address: str) -> Tuple[bool, str]:
        """Check if IP address is allowed."""
        try:
            ip = ipaddress.ip_address(ip_address)
            
            # Check explicit blocks
            if ip_address in self.config.blocked_ips:
                return False, "IP explicitly blocked"
            
            # Check allow list (if configured)
            if self.config.allowed_ips and ip_address not in self.config.allowed_ips:
                return False, "IP not in allow list"
            
            # Check network blocks
            for network in self.blocked_networks:
                if ip in network:
                    return False, f"IP in blocked network {network}"
            
            # Check allowed networks (if configured)
            if self.allowed_networks:
                allowed = any(ip in network for network in self.allowed_networks)
                if not allowed:
                    return False, "IP not in allowed networks"
            
            return True, "IP allowed"
            
        except ValueError:
            return False, "Invalid IP address"


class InputValidator:
    """Advanced input validation and sanitization."""
    
    def __init__(self, config: SecurityConfig):
        self.config = config
        self.compiled_sql_patterns = [re.compile(pattern) for pattern in config.sql_injection_patterns]
        self.compiled_xss_patterns = [re.compile(pattern) for pattern in config.xss_patterns]
    
    def validate_request(self, request: Request) -> Tuple[bool, List[str]]:
        """Validate request for security issues."""
        issues = []
        
        # Check request size
        content_length = request.headers.get("content-length")
        if content_length and int(content_length) > self.config.max_request_size:
            issues.append(f"Request size too large: {content_length}")
        
        # Check URL length
        if len(str(request.url)) > self.config.max_url_length:
            issues.append("URL too long")
        
        # Check headers
        total_header_size = sum(len(k) + len(v) for k, v in request.headers.items())
        if total_header_size > self.config.max_header_size:
            issues.append("Headers too large")
        
        # Check user agent
        user_agent = request.headers.get("user-agent", "").lower()
        for blocked_agent in self.config.blocked_user_agents:
            if blocked_agent in user_agent:
                issues.append(f"Blocked user agent: {blocked_agent}")
        
        return len(issues) == 0, issues
    
    async def validate_input_data(self, data: Any) -> Tuple[bool, List[str]]:
        """Validate input data for injection attacks."""
        issues = []
        
        if isinstance(data, dict):
            for key, value in data.items():
                key_issues = await self._check_string_for_threats(str(key))
                value_issues = await self._check_string_for_threats(str(value))
                issues.extend([f"Key '{key}': {issue}" for issue in key_issues])
                issues.extend([f"Value for '{key}': {issue}" for issue in value_issues])
        
        elif isinstance(data, (list, tuple)):
            for i, item in enumerate(data):
                item_valid, item_issues = await self.validate_input_data(item)
                issues.extend([f"Item {i}: {issue}" for issue in item_issues])
        
        elif isinstance(data, str):
            issues.extend(await self._check_string_for_threats(data))
        
        return len(issues) == 0, issues
    
    async def _check_string_for_threats(self, text: str) -> List[str]:
        """Check string for common attack patterns."""
        issues = []
        
        # SQL injection check
        for pattern in self.compiled_sql_patterns:
            if pattern.search(text):
                issues.append(f"Potential SQL injection: {pattern.pattern[:50]}...")
                break
        
        # XSS check
        for pattern in self.compiled_xss_patterns:
            if pattern.search(text):
                issues.append(f"Potential XSS: {pattern.pattern[:50]}...")
                break
        
        # Command injection check
        cmd_patterns = [r"[;&|`$()]", r"\.\./", r"\\x[0-9a-fA-F]{2}"]
        for pattern_str in cmd_patterns:
            if re.search(pattern_str, text):
                issues.append(f"Potential command injection: {pattern_str}")
                break
        
        return issues


class AuthenticationManager:
    """JWT-based authentication manager with enhanced security."""
    
    def __init__(self, config: SecurityConfig):
        self.config = config
        self.security = HTTPBearer()
        self.revoked_tokens: Set[str] = set()  # Token blacklist
        self.failed_attempts: Dict[str, List[datetime]] = defaultdict(list)
        self.lockout_duration = timedelta(minutes=15)
        self.max_failed_attempts = 5
    
    def create_token(self, user_data: Dict[str, Any]) -> str:
        """Create JWT token with enhanced security."""
        if not self.config.jwt_secret_key:
            raise ValueError("JWT secret key not configured")
        
        now = datetime.utcnow()
        payload = {
            "sub": user_data.get("username"),
            "iat": now,
            "exp": now + timedelta(minutes=self.config.jwt_expiry_minutes),
            "jti": secrets.token_urlsafe(16),  # Unique token ID
            "roles": user_data.get("roles", []),
            "permissions": user_data.get("permissions", [])
        }
        
        return jwt.encode(payload, self.config.jwt_secret_key, algorithm=self.config.jwt_algorithm)
    
    def verify_token(self, token: str) -> Tuple[bool, Dict[str, Any]]:
        """Verify JWT token with security checks."""
        try:
            if not self.config.jwt_secret_key:
                return False, {"error": "JWT not configured"}
            
            # Check token blacklist
            if token in self.revoked_tokens:
                return False, {"error": "Token revoked"}
            
            # Decode token
            payload = jwt.decode(
                token,
                self.config.jwt_secret_key,
                algorithms=[self.config.jwt_algorithm]
            )
            
            # Additional security checks
            if "jti" not in payload:
                return False, {"error": "Invalid token format"}
            
            return True, payload
            
        except jwt.ExpiredSignatureError:
            return False, {"error": "Token expired"}
        except jwt.InvalidTokenError:
            return False, {"error": "Invalid token"}
        except Exception as e:
            logger.error(f"Token verification error: {e}")
            return False, {"error": "Token verification failed"}
    
    def revoke_token(self, token: str) -> bool:
        """Revoke a token (add to blacklist)."""
        try:
            # Decode to get JTI
            payload = jwt.decode(
                token,
                self.config.jwt_secret_key,
                algorithms=[self.config.jwt_algorithm],
                verify=False  # We just need the JTI
            )
            
            jti = payload.get("jti")
            if jti:
                self.revoked_tokens.add(jti)
                return True
                
        except Exception as e:
            logger.error(f"Token revocation error: {e}")
        
        return False
    
    def is_user_locked_out(self, user_identifier: str) -> bool:
        """Check if user is locked out due to failed attempts."""
        now = datetime.utcnow()
        cutoff = now - self.lockout_duration
        
        # Clean old attempts
        self.failed_attempts[user_identifier] = [
            attempt for attempt in self.failed_attempts[user_identifier]
            if attempt > cutoff
        ]
        
        return len(self.failed_attempts[user_identifier]) >= self.max_failed_attempts
    
    def record_failed_attempt(self, user_identifier: str):
        """Record a failed authentication attempt."""
        self.failed_attempts[user_identifier].append(datetime.utcnow())


class SecurityMiddleware(BaseHTTPMiddleware):
    """Comprehensive security middleware for FastAPI applications."""
    
    def __init__(self, app, config: Optional[SecurityConfig] = None):
        super().__init__(app)
        self.config = config or SecurityConfig()
        
        # Initialize components
        self.rate_limiter = RateLimiter(
            self.config.rate_limit_requests,
            self.config.rate_limit_burst
        )
        self.ip_filter = IPFilter(self.config)
        self.input_validator = InputValidator(self.config)
        self.auth_manager = AuthenticationManager(self.config)
        
        # Threat detection
        if self.config.enable_threat_detection:
            self.threat_detector = get_threat_detection_system()
        else:
            self.threat_detector = None
        
        logger.info("Security middleware initialized")
    
    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        """Main middleware dispatch method."""
        start_time = time.time()
        
        try:
            # Extract client information
            client_ip = self._get_client_ip(request)
            client_id = self._get_client_id(request, client_ip)
            
            # 1. IP Filtering
            ip_allowed, ip_reason = self.ip_filter.is_allowed(client_ip)
            if not ip_allowed:
                logger.warning(f"IP blocked: {client_ip} - {ip_reason}")
                
                # Record threat event
                if self.threat_detector:
                    await self._record_threat_event(
                        ThreatType.UNAUTHORIZED_ACCESS,
                        f"Blocked IP access: {client_ip}",
                        client_ip,
                        str(request.url)
                    )
                
                return self._create_error_response(
                    status.HTTP_403_FORBIDDEN,
                    "Access denied"
                )
            
            # 2. Rate Limiting
            rate_allowed, rate_info = self.rate_limiter.is_allowed(client_id)
            if not rate_allowed:
                logger.warning(f"Rate limit exceeded: {client_id}")
                
                # Record threat event
                if self.threat_detector:
                    await self._record_threat_event(
                        ThreatType.DOS_ATTACK,
                        f"Rate limit exceeded: {client_id}",
                        client_ip,
                        str(request.url)
                    )
                
                response = self._create_error_response(
                    status.HTTP_429_TOO_MANY_REQUESTS,
                    "Rate limit exceeded"
                )
                response.headers["Retry-After"] = str(rate_info.get("retry_after", 60))
                return response
            
            # 3. Request Validation
            request_valid, validation_issues = self.input_validator.validate_request(request)
            if not request_valid:
                logger.warning(f"Request validation failed: {validation_issues}")
                
                # Record threat event
                if self.threat_detector:
                    await self._record_threat_event(
                        ThreatType.SUSPICIOUS_ACTIVITY,
                        f"Request validation failed: {validation_issues}",
                        client_ip,
                        str(request.url)
                    )
                
                return self._create_error_response(
                    status.HTTP_400_BAD_REQUEST,
                    "Invalid request"
                )
            
            # 4. Authentication Check
            auth_required = self._is_auth_required(request.url.path)
            if auth_required:
                auth_valid, auth_info = await self._validate_authentication(request)
                if not auth_valid:
                    logger.warning(f"Authentication failed: {auth_info}")
                    
                    # Record failed attempt
                    user_id = auth_info.get("user_id", client_ip)
                    self.auth_manager.record_failed_attempt(user_id)
                    
                    # Record threat event
                    if self.threat_detector:
                        await self._record_threat_event(
                            ThreatType.AUTHENTICATION_BYPASS,
                            f"Authentication failed: {auth_info}",
                            client_ip,
                            str(request.url)
                        )
                    
                    return self._create_error_response(
                        status.HTTP_401_UNAUTHORIZED,
                        "Authentication required"
                    )
            
            # 5. Input Data Validation (for POST/PUT requests)
            if request.method in ["POST", "PUT", "PATCH"]:
                try:
                    # Read and validate request body
                    body = await request.body()
                    if body:
                        try:
                            data = json.loads(body)
                            data_valid, data_issues = await self.input_validator.validate_input_data(data)
                            if not data_valid:
                                logger.warning(f"Input data validation failed: {data_issues}")
                                
                                # Record threat event
                                if self.threat_detector:
                                    await self._record_threat_event(
                                        ThreatType.SQL_INJECTION,
                                        f"Suspicious input detected: {data_issues[:3]}",  # First 3 issues
                                        client_ip,
                                        str(request.url)
                                    )
                                
                                return self._create_error_response(
                                    status.HTTP_400_BAD_REQUEST,
                                    "Invalid input data"
                                )
                        except json.JSONDecodeError:
                            # Not JSON data, skip validation
                            pass
                except Exception as e:
                    logger.error(f"Input validation error: {e}")
            
            # 6. Process Request
            response = await call_next(request)
            
            # 7. Add Security Headers
            self._add_security_headers(response)
            
            # 8. Log Request
            processing_time = time.time() - start_time
            await self._log_request(request, response, client_ip, processing_time)
            
            return response
            
        except Exception as e:
            logger.error(f"Security middleware error: {e}")
            
            # Record as potential attack
            if self.threat_detector:
                await self._record_threat_event(
                    ThreatType.SUSPICIOUS_ACTIVITY,
                    f"Middleware error: {str(e)[:100]}",
                    self._get_client_ip(request),
                    str(request.url)
                )
            
            return self._create_error_response(
                status.HTTP_500_INTERNAL_SERVER_ERROR,
                "Internal server error"
            )
    
    def _get_client_ip(self, request: Request) -> str:
        """Extract client IP address with proxy support."""
        # Check for forwarded headers (in order of preference)
        forwarded_headers = [
            "X-Forwarded-For",
            "X-Real-IP",
            "CF-Connecting-IP",  # Cloudflare
            "X-Cluster-Client-IP"
        ]
        
        for header in forwarded_headers:
            if header in request.headers:
                # Take the first IP if comma-separated
                ip = request.headers[header].split(",")[0].strip()
                if ip and ip != "unknown":
                    return ip
        
        # Fall back to direct connection
        return request.client.host if request.client else "unknown"
    
    def _get_client_id(self, request: Request, client_ip: str) -> str:
        """Generate client identifier for rate limiting."""
        # Use combination of IP and User-Agent for better granularity
        user_agent = request.headers.get("user-agent", "")
        client_id = f"{client_ip}_{hashlib.md5(user_agent.encode()).hexdigest()[:8]}"
        return client_id
    
    def _is_auth_required(self, path: str) -> bool:
        """Check if authentication is required for the path."""
        # Check public paths first
        if any(path.startswith(public_path) for public_path in self.config.public_paths):
            return False
        
        # Check auth required paths
        return any(path.startswith(auth_path) for auth_path in self.config.require_auth_paths)
    
    async def _validate_authentication(self, request: Request) -> Tuple[bool, Dict[str, Any]]:
        """Validate request authentication."""
        try:
            # Check for Authorization header
            auth_header = request.headers.get("authorization")
            if not auth_header:
                return False, {"error": "No authorization header"}
            
            # Extract token
            if not auth_header.startswith("Bearer "):
                return False, {"error": "Invalid auth header format"}
            
            token = auth_header[7:]  # Remove "Bearer "
            
            # Verify token
            token_valid, token_info = self.auth_manager.verify_token(token)
            if not token_valid:
                return False, token_info
            
            # Check user lockout
            user_id = token_info.get("sub")
            if user_id and self.auth_manager.is_user_locked_out(user_id):
                return False, {"error": "User locked out", "user_id": user_id}
            
            return True, token_info
            
        except Exception as e:
            logger.error(f"Authentication validation error: {e}")
            return False, {"error": "Authentication validation failed"}
    
    def _add_security_headers(self, response: Response):
        """Add security headers to response."""
        for header, value in self.config.security_headers.items():
            response.headers[header] = value
    
    def _create_error_response(self, status_code: int, message: str) -> JSONResponse:
        """Create standardized error response."""
        return JSONResponse(
            status_code=status_code,
            content={
                "error": message,
                "timestamp": datetime.utcnow().isoformat(),
                "status": status_code
            }
        )
    
    async def _record_threat_event(
        self,
        threat_type: ThreatType,
        description: str,
        source_ip: str,
        target: str
    ):
        """Record threat event for monitoring."""
        if not self.threat_detector:
            return
        
        try:
            # Sample based on configuration
            if secrets.randbelow(100) / 100.0 > self.config.threat_detection_sampling:
                return
            
            threat_data = {
                "http_requests": [{
                    "source_ip": source_ip,
                    "endpoint": target,
                    "timestamp": datetime.utcnow().isoformat(),
                    "threat_type": threat_type.value,
                    "description": description
                }]
            }
            
            await self.threat_detector.detect_threats(threat_data)
            
        except Exception as e:
            logger.error(f"Failed to record threat event: {e}")
    
    async def _log_request(
        self,
        request: Request,
        response: Response,
        client_ip: str,
        processing_time: float
    ):
        """Log request for security monitoring."""
        log_data = {
            "client_ip": client_ip,
            "method": request.method,
            "path": request.url.path,
            "status_code": response.status_code,
            "processing_time": processing_time,
            "user_agent": request.headers.get("user-agent", ""),
            "timestamp": datetime.utcnow().isoformat()
        }
        
        # Log based on status code
        if response.status_code >= 400:
            logger.warning("Request failed", **log_data)
        else:
            logger.info("Request processed", **log_data)
        
        # Record metrics
        metrics = get_safe_metrics_collector()
        metrics.record_metric(
            "security.requests.total",
            1,
            {
                "method": request.method,
                "status_code": str(response.status_code),
                "client_ip": client_ip
            }
        )
        
        metrics.record_metric(
            "security.requests.processing_time",
            processing_time,
            {"endpoint": request.url.path}
        )


def create_security_middleware(config: Optional[SecurityConfig] = None) -> SecurityMiddleware:
    """Factory function to create security middleware."""
    return SecurityMiddleware(None, config)


def get_security_config() -> SecurityConfig:
    """Get default security configuration."""
    config = SecurityConfig()
    
    # Set JWT secret from environment or generate
    import os
    config.jwt_secret_key = os.getenv("JWT_SECRET_KEY", secrets.token_urlsafe(32))
    
    return config