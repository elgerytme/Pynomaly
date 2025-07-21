"""
Centralized security manager for Pynomaly API.

This module integrates all security components and provides a unified interface
for security operations throughout the application.
"""

import logging
import os
from datetime import datetime
from typing import Any

import yaml

from .authentication import (
    AuthenticationManager,
    JWTManager,
    MFAManager,
    SessionManager,
)
from .authorization import AuthorizationManager, RoleBasedAccessControl
from .encryption import EncryptionManager, FieldLevelEncryption
from .input_validation import SecurityValidator
from .rate_limiting import CircuitBreaker, DDoSProtection, RateLimiter
from .security_headers import CORSPolicy, SecurityHeaders, SecurityMiddleware
from .security_monitoring import (
    SecurityEvent,
    SecurityEventType,
    SecurityMonitor,
    SecuritySeverity,
)
from .vulnerability_scanner import VulnerabilityScanner

logger = logging.getLogger(__name__)


class SecurityManager:
    """Centralized security manager for the Pynomaly platform."""

    def __init__(self, config_path: str | None = None):
        """Initialize security manager with configuration."""
        self.config = self._load_security_config(config_path)
        self.environment = os.getenv("PYNOMALY_ENV", "development")

        # Apply environment-specific overrides
        self._apply_environment_config()

        # Initialize security components
        self._initialize_components()

        # Setup integration
        self._setup_integrations()

        logger.info(f"Security manager initialized for {self.environment} environment")

    def _load_security_config(self, config_path: str | None) -> dict[str, Any]:
        """Load security configuration from file."""
        if config_path is None:
            config_path = "config/security.yaml"

        try:
            with open(config_path) as f:
                config = yaml.safe_load(f)

            # Expand environment variables
            config = self._expand_env_vars(config)
            return config

        except FileNotFoundError:
            logger.warning(
                f"Security config file not found: {config_path}, using defaults"
            )
            return self._get_default_config()
        except Exception as e:
            logger.error(f"Failed to load security config: {e}")
            return self._get_default_config()

    def _expand_env_vars(self, config: Any) -> Any:
        """Recursively expand environment variables in config."""
        if isinstance(config, dict):
            return {key: self._expand_env_vars(value) for key, value in config.items()}
        elif isinstance(config, list):
            return [self._expand_env_vars(item) for item in config]
        elif (
            isinstance(config, str) and config.startswith("${") and config.endswith("}")
        ):
            env_var = config[2:-1]
            return os.getenv(env_var, config)
        else:
            return config

    def _get_default_config(self) -> dict[str, Any]:
        """Get default security configuration."""
        return {
            "authentication": {
                "jwt": {"secret_key": "default-secret-key", "algorithm": "HS256"},
                "password_policy": {"min_length": 8},
                "account_lockout": {"max_failed_attempts": 5},
            },
            "authorization": {"rbac": {"enabled": True}},
            "rate_limiting": {
                "enabled": True,
                "per_user_limits": {"requests_per_hour": 1000},
            },
            "security_headers": {"enabled": True},
            "encryption": {"enabled": True},
            "security_monitoring": {"enabled": True},
            "vulnerability_scanning": {"enabled": True},
        }

    def _apply_environment_config(self) -> None:
        """Apply environment-specific configuration overrides."""
        env_config = self.config.get("environments", {}).get(self.environment, {})

        if env_config:
            self._deep_merge(self.config, env_config)
            logger.info(f"Applied {self.environment} environment configuration")

    def _deep_merge(self, base: dict[str, Any], override: dict[str, Any]) -> None:
        """Deep merge configuration dictionaries."""
        for key, value in override.items():
            if key in base and isinstance(base[key], dict) and isinstance(value, dict):
                self._deep_merge(base[key], value)
            else:
                base[key] = value

    def _initialize_components(self) -> None:
        """Initialize all security components."""
        # Authentication
        jwt_config = self.config.get("authentication", {}).get("jwt", {})
        self.jwt_manager = JWTManager(
            secret_key=jwt_config.get("secret_key", "default-secret"),
            algorithm=jwt_config.get("algorithm", "HS256"),
        )

        self.auth_manager = AuthenticationManager(self.jwt_manager)
        self.mfa_manager = MFAManager()
        self.session_manager = SessionManager()

        # Authorization
        self.auth_z_manager = AuthorizationManager()
        self.rbac = RoleBasedAccessControl(self.auth_z_manager)

        # Initialize default roles
        self._setup_default_roles()

        # Input validation
        self.security_validator = SecurityValidator()

        # Rate limiting and DDoS protection
        self.rate_limiter = RateLimiter()
        self.ddos_protection = DDoSProtection()
        self.circuit_breaker = CircuitBreaker()

        # Security headers and CORS
        self.security_headers = SecurityHeaders()
        self.cors_policy = CORSPolicy()
        self.security_middleware = SecurityMiddleware()

        # Apply security headers configuration
        self._configure_security_headers()
        self._configure_cors_policy()

        # Encryption
        encryption_config = self.config.get("encryption", {})
        master_key = encryption_config.get("master_key")
        self.encryption_manager = EncryptionManager(master_key)
        self.field_encryption = FieldLevelEncryption(self.encryption_manager)

        # Security monitoring
        self.security_monitor = SecurityMonitor()
        self._setup_monitoring_rules()

        # Vulnerability scanning
        self.vulnerability_scanner = VulnerabilityScanner()

        logger.info("All security components initialized")

    def _setup_default_roles(self) -> None:
        """Setup default roles from configuration."""
        default_roles = self.config.get("authorization", {}).get("default_roles", [])

        for role_config in default_roles:
            role_name = role_config.get("name")
            permissions = role_config.get("permissions", [])

            # Add role to authorization manager (simplified)
            logger.info(
                f"Configured role: {role_name} with {len(permissions)} permissions"
            )

    def _configure_security_headers(self) -> None:
        """Configure security headers from configuration."""
        headers_config = self.config.get("security_headers", {})

        if not headers_config.get("enabled", True):
            return

        # Configure HSTS
        hsts_config = headers_config.get("hsts", {})
        if hsts_config.get("enabled", True):
            max_age = hsts_config.get("max_age", 31536000)
            include_subdomains = hsts_config.get("include_subdomains", True)
            preload = hsts_config.get("preload", True)

            hsts_value = f"max-age={max_age}"
            if include_subdomains:
                hsts_value += "; includeSubDomains"
            if preload:
                hsts_value += "; preload"

            self.security_headers.add_custom_header(
                "Strict-Transport-Security", hsts_value
            )

        # Configure CSP
        csp_config = headers_config.get("csp", {})
        if csp_config.get("enabled", True):
            csp_directives = []
            for directive, value in csp_config.items():
                if directive != "enabled" and isinstance(value, str):
                    csp_directives.append(f"{directive.replace('_', '-')} {value}")

            if csp_directives:
                csp_value = "; ".join(csp_directives)
                self.security_headers.add_custom_header(
                    "Content-Security-Policy", csp_value
                )

        # Configure other headers
        other_headers = headers_config.get("other_headers", {})
        for header_name, header_value in other_headers.items():
            header_name_formatted = header_name.replace("_", "-").title()
            self.security_headers.add_custom_header(header_name_formatted, header_value)

    def _configure_cors_policy(self) -> None:
        """Configure CORS policy from configuration."""
        cors_config = self.config.get("cors", {})

        if not cors_config.get("enabled", True):
            return

        # Set policy type
        policy = cors_config.get("policy", "restrictive")
        self.cors_policy.set_policy(policy)

        # Add allowed origins
        allowed_origins = cors_config.get("allowed_origins", [])
        for origin in allowed_origins:
            self.cors_policy.add_allowed_origin(origin)

        # Configure other CORS settings
        self.cors_policy.allowed_methods = set(cors_config.get("allowed_methods", []))
        self.cors_policy.allowed_headers = set(cors_config.get("allowed_headers", []))
        self.cors_policy.exposed_headers = set(cors_config.get("expose_headers", []))
        self.cors_policy.max_age = cors_config.get("max_age", 86400)
        self.cors_policy.allow_credentials = cors_config.get("allow_credentials", True)

    def _setup_monitoring_rules(self) -> None:
        """Setup security monitoring rules."""
        # Add alert callback
        self.security_monitor.add_alert_callback(self._handle_security_alert)

        # Import and add default monitoring rules
        try:
            from .security_monitoring import create_default_monitoring_rules

            rules = create_default_monitoring_rules()
            for rule in rules:
                self.security_monitor.add_monitoring_rule(rule)
        except ImportError:
            logger.warning("Could not import default monitoring rules")

    def _handle_security_alert(self, alert_data: dict[str, Any]) -> None:
        """Handle security alerts."""
        logger.critical(f"SECURITY ALERT: {alert_data}")

        # In production, would send to:
        # - Email notifications
        # - Slack/Teams channels
        # - External monitoring systems
        # - Incident response systems

    def _setup_integrations(self) -> None:
        """Setup integrations between security components."""
        # Integrate security middleware with other components
        self.security_middleware.security_headers = self.security_headers
        self.security_middleware.cors_policy = self.cors_policy

        # Setup rate limiting integration
        rate_config = self.config.get("rate_limiting", {})
        if rate_config.get("enabled", True):
            # Configure rate limits from config
            per_user = rate_config.get("per_user_limits", {})
            per_ip = rate_config.get("per_ip_limits", {})

            # Apply configurations to rate limiter
            logger.info("Rate limiting configured")

    # Public interface methods

    def authenticate_request(
        self, username: str, password: str, mfa_code: str | None = None
    ) -> dict[str, Any]:
        """Authenticate user request."""
        # Primary authentication
        auth_result = self.auth_manager.authenticate_user(username, password)

        if not auth_result.get("status") == "success":
            # Log failed authentication
            self._log_security_event(
                event_type=SecurityEventType.AUTHENTICATION_FAILURE,
                severity=SecuritySeverity.MEDIUM,
                user_id=username,
                details={"reason": auth_result.get("error", "unknown")},
            )
            return auth_result

        # MFA verification if enabled
        mfa_config = self.config.get("authentication", {}).get("mfa", {})
        if mfa_config.get("enabled", False) and mfa_code:
            if not self.mfa_manager.verify_mfa_code(username, mfa_code):
                self._log_security_event(
                    event_type=SecurityEventType.AUTHENTICATION_FAILURE,
                    severity=SecuritySeverity.HIGH,
                    user_id=username,
                    details={"reason": "mfa_failed"},
                )
                return {"error": "mfa_failed", "message": "Invalid MFA code"}

        # Log successful authentication
        self._log_security_event(
            event_type=SecurityEventType.AUTHENTICATION_SUCCESS,
            severity=SecuritySeverity.LOW,
            user_id=username,
            details={"login_time": datetime.utcnow().isoformat()},
        )

        return auth_result

    def authorize_request(self, user_id: str, resource: str, action: str) -> bool:
        """Authorize user request."""
        authorized = self.auth_z_manager.authorize_request(user_id, resource, action)

        if not authorized:
            self._log_security_event(
                event_type=SecurityEventType.AUTHORIZATION_FAILURE,
                severity=SecuritySeverity.MEDIUM,
                user_id=user_id,
                details={"resource": resource, "action": action},
            )

        return authorized

    def validate_input(
        self, data: dict[str, Any], schema_name: str | None = None
    ) -> dict[str, Any]:
        """Validate and sanitize input data."""
        try:
            # Check for security threats
            threats = self.security_validator.check_security_threats(data)
            if threats:
                self._log_security_event(
                    event_type=SecurityEventType.SUSPICIOUS_REQUEST,
                    severity=SecuritySeverity.HIGH,
                    details={"threats": threats, "data_keys": list(data.keys())},
                )
                raise ValueError(f"Security threats detected: {threats}")

            # Validate and sanitize
            return self.security_validator.validate_and_sanitize(data, schema_name)

        except Exception as e:
            logger.error(f"Input validation failed: {e}")
            raise

    def check_rate_limit(
        self, user_id: str, ip_address: str, endpoint: str
    ) -> Tuple[bool, dict[str, Any]]:
        """Check rate limits for request."""
        from .rate_limiting import RateLimitType

        # Check user rate limit
        user_allowed, user_status = self.rate_limiter.is_allowed(
            user_id, RateLimitType.PER_USER, endpoint
        )

        # Check IP rate limit
        ip_allowed, ip_status = self.rate_limiter.is_allowed(
            ip_address, RateLimitType.PER_IP, endpoint
        )

        allowed = user_allowed and ip_allowed

        if not allowed:
            self._log_security_event(
                event_type=SecurityEventType.RATE_LIMIT_EXCEEDED,
                severity=SecuritySeverity.MEDIUM,
                user_id=user_id,
                source_ip=ip_address,
                endpoint=endpoint,
                details={"user_status": user_status, "ip_status": ip_status},
            )

        return allowed, {"user": user_status, "ip": ip_status}

    def analyze_request_security(self, request_data: dict[str, Any]) -> dict[str, Any]:
        """Comprehensive security analysis of request."""
        analysis = {
            "risk_score": 0.0,
            "threats": [],
            "recommendations": [],
            "blocked": False,
        }

        ip_address = request_data.get("ip_address", "")
        user_agent = request_data.get("user_agent", "")
        endpoint = request_data.get("endpoint", "")

        # DDoS analysis
        ddos_analysis = self.ddos_protection.analyze_request(
            ip_address, user_agent, endpoint
        )
        analysis["risk_score"] += ddos_analysis["risk_score"]
        analysis["threats"].extend(ddos_analysis["reasons"])

        if not ddos_analysis["allowed"]:
            analysis["blocked"] = True
            analysis["recommendations"].append("IP address has been blocked")

        # Input validation
        payload = request_data.get("payload", {})
        if payload:
            threats = self.security_validator.check_security_threats(payload)
            if threats:
                analysis["threats"].extend(threats)
                analysis["risk_score"] += 0.5

        # User agent analysis
        suspicious_agents = ["bot", "crawler", "scanner", "hack"]
        if any(agent in user_agent.lower() for agent in suspicious_agents):
            analysis["threats"].append("Suspicious user agent")
            analysis["risk_score"] += 0.3

        # Endpoint analysis
        sensitive_endpoints = ["/admin", "/config", "/users", "/api/internal"]
        if any(sensitive in endpoint for sensitive in sensitive_endpoints):
            analysis["risk_score"] += 0.2

        # Generate recommendations
        if analysis["risk_score"] > 0.7:
            analysis["recommendations"].append("Consider additional authentication")
        if analysis["risk_score"] > 0.5:
            analysis["recommendations"].append("Monitor this request closely")

        return analysis

    def encrypt_sensitive_data(self, data: dict[str, Any]) -> dict[str, Any]:
        """Encrypt sensitive fields in data."""
        if not self.config.get("encryption", {}).get("enabled", True):
            return data

        return self.field_encryption.encrypt_record(data)

    def decrypt_sensitive_data(self, data: dict[str, Any]) -> dict[str, Any]:
        """Decrypt sensitive fields in data."""
        if not self.config.get("encryption", {}).get("enabled", True):
            return data

        return self.field_encryption.decrypt_record(data)

    def get_security_headers(
        self, endpoint: str = "", request_origin: str = ""
    ) -> dict[str, str]:
        """Get security headers for response."""
        headers = self.security_headers.get_headers(endpoint)

        # Add CORS headers if origin is provided
        if request_origin:
            cors_headers = self.cors_policy.get_cors_headers(request_origin)
            headers.update(cors_headers)

        return headers

    def run_vulnerability_scan(self, scan_paths: list[str]) -> dict[str, Any]:
        """Run vulnerability scan."""
        config = self.config.get("vulnerability_scanning", {})

        if not config.get("enabled", True):
            return {"error": "Vulnerability scanning disabled"}

        # Get configuration and code paths
        app_config = {
            "DEBUG": os.getenv("DEBUG", "False").lower() == "true",
            "SECRET_KEY": os.getenv("SECRET_KEY", ""),
            "SECURITY_HEADERS": self.config.get("security_headers", {}),
            "SSL_VERIFY": True,
        }

        return self.vulnerability_scanner.scan_all(
            config=app_config,
            code_paths=scan_paths,
            requirements_file="requirements.txt",
        )

    def get_security_metrics(self) -> dict[str, Any]:
        """Get comprehensive security metrics."""
        return {
            "monitoring": self.security_monitor.get_security_summary(),
            "authentication": {
                "failed_attempts": len(self.auth_manager.failed_attempts),
                "locked_accounts": len(self.auth_manager.lockout_times),
                "active_sessions": len(self.session_manager.active_sessions),
            },
            "rate_limiting": {
                "rate_limit_status": "active"
                if self.config.get("rate_limiting", {}).get("enabled")
                else "disabled"
            },
            "vulnerability_scanning": {
                "last_scan": self.vulnerability_scanner.scan_history[-1]
                if self.vulnerability_scanner.scan_history
                else None,
                "total_scans": len(self.vulnerability_scanner.scan_history),
            },
        }

    def _log_security_event(
        self,
        event_type: SecurityEventType,
        severity: SecuritySeverity,
        user_id: str | None = None,
        source_ip: str = "",
        endpoint: str = "",
        user_agent: str = "",
        details: dict[str, Any] = None,
    ) -> None:
        """Log security event."""
        import secrets

        event = SecurityEvent(
            event_id=secrets.token_hex(8),
            event_type=event_type,
            severity=severity,
            timestamp=datetime.utcnow(),
            source_ip=source_ip,
            user_id=user_id,
            endpoint=endpoint,
            user_agent=user_agent,
            details=details or {},
            risk_score=self._calculate_risk_score(event_type, severity),
        )

        self.security_monitor.log_security_event(event)

    def _calculate_risk_score(
        self, event_type: SecurityEventType, severity: SecuritySeverity
    ) -> float:
        """Calculate risk score for security event."""
        base_scores = {
            SecuritySeverity.LOW: 0.2,
            SecuritySeverity.MEDIUM: 0.5,
            SecuritySeverity.HIGH: 0.8,
            SecuritySeverity.CRITICAL: 1.0,
        }

        multipliers = {
            SecurityEventType.AUTHENTICATION_FAILURE: 1.0,
            SecurityEventType.AUTHORIZATION_FAILURE: 1.2,
            SecurityEventType.SQL_INJECTION_ATTEMPT: 1.5,
            SecurityEventType.XSS_ATTEMPT: 1.5,
            SecurityEventType.BRUTE_FORCE_ATTACK: 1.3,
            SecurityEventType.DDOS_ATTACK: 1.4,
        }

        base_score = base_scores.get(severity, 0.5)
        multiplier = multipliers.get(event_type, 1.0)

        return min(1.0, base_score * multiplier)


# Global security manager instance
_security_manager: SecurityManager | None = None


def get_security_manager() -> SecurityManager:
    """Get global security manager instance."""
    global _security_manager

    if _security_manager is None:
        _security_manager = SecurityManager()

    return _security_manager


def initialize_security(config_path: str | None = None) -> SecurityManager:
    """Initialize global security manager."""
    global _security_manager

    _security_manager = SecurityManager(config_path)
    return _security_manager
