"""
Advanced Security Manager that orchestrates all security components.
"""

import asyncio
import logging
import time
from typing import Any

import redis
from fastapi import Request, Response

from monorepo.infrastructure.config import Settings
from monorepo.infrastructure.security.audit_logger import get_audit_logger
from monorepo.infrastructure.security.enhanced_rate_limiter import EnhancedRateLimiter
from monorepo.infrastructure.security.security_monitor import SecurityMonitor
from monorepo.infrastructure.security.waf_middleware import WAFMiddleware

logger = logging.getLogger(__name__)


class AdvancedSecurityManager:
    """
    Advanced security manager that coordinates WAF, rate limiting,
    monitoring, and threat intelligence.
    """

    def __init__(self, settings: Settings):
        self.settings = settings
        self.redis_client = redis.from_url(settings.redis_url)
        self.audit_logger = get_audit_logger()

        # Initialize security components
        self.waf_middleware = None
        self.rate_limiter = None
        self.security_monitor = None

        # Security state
        self.security_level = "normal"  # normal, elevated, high, critical
        self.active_incidents = []
        self.threat_intelligence = {}

        # Configuration
        self.config = {
            "auto_response_enabled": True,
            "threat_sharing_enabled": True,
            "ml_detection_enabled": False,
            "geo_blocking_enabled": False,
            "honeypot_enabled": False,
        }

        logger.info("Advanced Security Manager initialized")

    async def initialize(self, app):
        """Initialize all security components."""
        try:
            # Initialize WAF
            self.waf_middleware = WAFMiddleware(app, self.settings)

            # Initialize Rate Limiter
            self.rate_limiter = EnhancedRateLimiter(self.redis_client, self.settings)

            # Initialize Security Monitor
            self.security_monitor = SecurityMonitor(self.settings)

            # Start background tasks
            asyncio.create_task(self._threat_intelligence_updater())
            asyncio.create_task(self._security_level_monitor())
            asyncio.create_task(self._incident_response_processor())

            logger.info("All security components initialized successfully")

        except Exception as e:
            logger.error(f"Failed to initialize security components: {e}")
            raise

    async def process_request(self, request: Request) -> Response | None:
        """
        Process request through all security layers.
        Returns Response if request should be blocked, None if allowed.
        """
        start_time = time.time()

        try:
            # Get client information
            client_ip = self._get_client_ip(request)
            user_agent = request.headers.get("User-Agent", "")

            # Check if IP is in emergency block list
            if await self._is_emergency_blocked(client_ip):
                return self._create_emergency_block_response(client_ip)

            # Apply geo-blocking if enabled
            if self.config["geo_blocking_enabled"]:
                geo_result = await self._check_geo_blocking(client_ip)
                if geo_result:
                    return geo_result

            # Process through rate limiter
            rate_limit_result = await self.rate_limiter.check_request_allowed(request)
            if not rate_limit_result[0]:
                await self._handle_rate_limit_violation(client_ip, rate_limit_result)
                return self._create_rate_limit_response(rate_limit_result)

            # Process through WAF
            waf_result = await self.waf_middleware.dispatch(
                request, lambda req: Response()
            )
            if isinstance(waf_result, Response) and waf_result.status_code == 403:
                return waf_result

            # Update security monitoring
            await self._update_security_metrics(request, time.time() - start_time)

            return None  # Allow request

        except Exception as e:
            logger.error(f"Error in security processing: {e}")
            return None  # Fail open

    async def _is_emergency_blocked(self, ip: str) -> bool:
        """Check if IP is in emergency block list."""
        try:
            blocked = await self.redis_client.get(f"emergency_block:{ip}")
            return blocked is not None
        except:
            return False

    async def _check_geo_blocking(self, ip: str) -> Response | None:
        """Check geo-blocking rules."""
        # This would integrate with a geo-IP service
        # For now, return None (allow)
        return None

    async def _handle_rate_limit_violation(self, ip: str, result: tuple):
        """Handle rate limit violations."""
        reason, details = result[1], result[2]

        # Log the violation
        self.audit_logger.log_security_event(
            event_type="RATE_LIMIT_VIOLATION",
            message=f"Rate limit violation from {ip}: {reason}",
            level="WARNING",
            details={"ip": ip, "reason": reason, **details},
        )

        # Check if this should trigger escalation
        await self._check_escalation(ip, "rate_limit", details)

    async def _check_escalation(self, ip: str, violation_type: str, details: dict):
        """Check if security violation should trigger escalation."""
        # Count recent violations from this IP
        try:
            key = f"violations:{ip}"
            violations = await self.redis_client.incr(key)
            await self.redis_client.expire(key, 3600)  # 1 hour window

            if violations >= 10:  # Threshold for escalation
                await self._escalate_threat(ip, violation_type, violations)

        except Exception as e:
            logger.error(f"Error checking escalation: {e}")

    async def _escalate_threat(self, ip: str, violation_type: str, count: int):
        """Escalate threat to higher security level."""
        # Add to emergency block list
        try:
            await self.redis_client.setex(
                f"emergency_block:{ip}",
                86400,  # 24 hours
                f"Auto-escalated due to {count} {violation_type} violations",
            )

            # Log escalation
            self.audit_logger.log_security_event(
                event_type="THREAT_ESCALATION",
                message=f"Threat escalated for IP {ip}",
                level="CRITICAL",
                details={
                    "ip": ip,
                    "violation_type": violation_type,
                    "violation_count": count,
                    "action": "emergency_block",
                },
            )

            # Update security level if needed
            await self._update_security_level("elevated")

        except Exception as e:
            logger.error(f"Error escalating threat: {e}")

    async def _update_security_level(self, new_level: str):
        """Update overall security level."""
        if new_level != self.security_level:
            old_level = self.security_level
            self.security_level = new_level

            self.audit_logger.log_security_event(
                event_type="SECURITY_LEVEL_CHANGE",
                message=f"Security level changed from {old_level} to {new_level}",
                level="INFO",
                details={
                    "old_level": old_level,
                    "new_level": new_level,
                    "timestamp": time.time(),
                },
            )

            # Adjust security parameters based on level
            await self._adjust_security_parameters(new_level)

    async def _adjust_security_parameters(self, security_level: str):
        """Adjust security parameters based on current threat level."""
        adjustments = {
            "normal": {
                "rate_limit_multiplier": 1.0,
                "waf_sensitivity": "normal",
                "auto_block_threshold": 5,
            },
            "elevated": {
                "rate_limit_multiplier": 0.7,
                "waf_sensitivity": "high",
                "auto_block_threshold": 3,
            },
            "high": {
                "rate_limit_multiplier": 0.5,
                "waf_sensitivity": "strict",
                "auto_block_threshold": 2,
            },
            "critical": {
                "rate_limit_multiplier": 0.3,
                "waf_sensitivity": "maximum",
                "auto_block_threshold": 1,
            },
        }

        params = adjustments.get(security_level, adjustments["normal"])

        # Apply adjustments to rate limiter
        if self.rate_limiter:
            for limit_type, adaptive_limit in self.rate_limiter.adaptive_limits.items():
                adaptive_limit.adaptive_factor *= params["rate_limit_multiplier"]

        # Apply adjustments to WAF
        if self.waf_middleware:
            self.waf_middleware.config["auto_block_threshold"] = params[
                "auto_block_threshold"
            ]

    async def _update_security_metrics(self, request: Request, processing_time: float):
        """Update security monitoring metrics."""
        if self.security_monitor:
            await self.security_monitor.record_request(request, processing_time)

    async def _threat_intelligence_updater(self):
        """Background task to update threat intelligence."""
        while True:
            try:
                # Update threat intelligence from various sources
                await self._update_threat_feeds()
                await asyncio.sleep(3600)  # Update every hour
            except Exception as e:
                logger.error(f"Error updating threat intelligence: {e}")
                await asyncio.sleep(300)  # Retry in 5 minutes

    async def _update_threat_feeds(self):
        """Update threat intelligence feeds."""
        # This would integrate with external threat intelligence sources
        # For now, we'll simulate with some basic logic

        try:
            # Get recent attack patterns
            attack_patterns = await self._analyze_recent_attacks()

            # Update threat signatures
            if attack_patterns:
                await self._update_dynamic_signatures(attack_patterns)

            # Update IP reputation
            await self._update_ip_reputation()

        except Exception as e:
            logger.error(f"Error updating threat feeds: {e}")

    async def _analyze_recent_attacks(self) -> list[dict]:
        """Analyze recent attacks to identify patterns."""
        patterns = []

        if self.waf_middleware:
            recent_attacks = list(self.waf_middleware.stats.recent_attacks)[-100:]

            # Analyze for common patterns
            for attack in recent_attacks:
                # Look for new attack patterns
                if attack.signature not in self.threat_intelligence:
                    patterns.append(
                        {
                            "signature": attack.signature,
                            "attack_type": attack.attack_type.value,
                            "threat_level": attack.threat_level.value,
                            "frequency": 1,
                        }
                    )

        return patterns

    async def _update_dynamic_signatures(self, patterns: list[dict]):
        """Update WAF with dynamic signatures based on attack patterns."""
        # This would update the WAF with new signatures
        pass

    async def _update_ip_reputation(self):
        """Update IP reputation scores."""
        # This would integrate with IP reputation services
        pass

    async def _security_level_monitor(self):
        """Monitor and automatically adjust security level."""
        while True:
            try:
                current_metrics = await self._get_current_threat_metrics()
                new_level = await self._calculate_security_level(current_metrics)

                if new_level != self.security_level:
                    await self._update_security_level(new_level)

                await asyncio.sleep(300)  # Check every 5 minutes

            except Exception as e:
                logger.error(f"Error in security level monitor: {e}")
                await asyncio.sleep(60)

    async def _get_current_threat_metrics(self) -> dict:
        """Get current threat metrics."""
        metrics = {
            "attack_rate": 0,
            "blocked_ips": 0,
            "threat_diversity": 0,
            "system_load": 0,
        }

        if self.waf_middleware:
            waf_stats = self.waf_middleware.get_stats()
            metrics["attack_rate"] = waf_stats.get("waf_stats", {}).get(
                "attack_rate", 0
            )
            metrics["blocked_ips"] = waf_stats.get("blocked_ips", 0)

        if self.rate_limiter:
            rl_stats = self.rate_limiter.get_statistics()
            metrics["system_load"] = rl_stats.get("system_load", 0)

        return metrics

    async def _calculate_security_level(self, metrics: dict) -> str:
        """Calculate appropriate security level based on metrics."""
        score = 0

        # Attack rate factor
        attack_rate = metrics.get("attack_rate", 0)
        if attack_rate > 20:
            score += 3
        elif attack_rate > 10:
            score += 2
        elif attack_rate > 5:
            score += 1

        # Blocked IPs factor
        blocked_ips = metrics.get("blocked_ips", 0)
        if blocked_ips > 100:
            score += 2
        elif blocked_ips > 50:
            score += 1

        # System load factor
        system_load = metrics.get("system_load", 0)
        if system_load > 0.8:
            score += 1

        # Determine level
        if score >= 5:
            return "critical"
        elif score >= 3:
            return "high"
        elif score >= 2:
            return "elevated"
        else:
            return "normal"

    async def _incident_response_processor(self):
        """Process security incidents and coordinate response."""
        while True:
            try:
                # Check for active incidents
                incidents = await self._get_active_incidents()

                for incident in incidents:
                    await self._process_incident(incident)

                await asyncio.sleep(60)  # Process every minute

            except Exception as e:
                logger.error(f"Error in incident response processor: {e}")
                await asyncio.sleep(30)

    async def _get_active_incidents(self) -> list[dict]:
        """Get list of active security incidents."""
        # This would query incident database or cache
        return self.active_incidents

    async def _process_incident(self, incident: dict):
        """Process a security incident."""
        # Implement incident response logic
        pass

    def _get_client_ip(self, request: Request) -> str:
        """Get client IP from request."""
        forwarded_for = request.headers.get("X-Forwarded-For")
        if forwarded_for:
            return forwarded_for.split(",")[0].strip()

        real_ip = request.headers.get("X-Real-IP")
        if real_ip:
            return real_ip

        return request.client.host if request.client else "unknown"

    def _create_emergency_block_response(self, ip: str) -> Response:
        """Create response for emergency blocked IP."""
        return Response(
            content="Access Denied - Security Policy Violation",
            status_code=403,
            headers={
                "X-Security-Block": "emergency",
                "X-Block-Reason": "automated_threat_response",
            },
        )

    def _create_rate_limit_response(self, result: tuple) -> Response:
        """Create response for rate limited request."""
        return Response(
            content="Rate Limit Exceeded",
            status_code=429,
            headers={
                "X-RateLimit-Limit": "100",
                "X-RateLimit-Remaining": "0",
                "X-RateLimit-Reset": str(int(time.time()) + 60),
                "Retry-After": "60",
            },
        )

    async def get_security_overview(self) -> dict[str, Any]:
        """Get comprehensive security overview."""
        overview = {
            "security_level": self.security_level,
            "components_status": {},
            "threat_intelligence": self.threat_intelligence,
            "active_incidents": len(self.active_incidents),
            "configuration": self.config,
        }

        # Get component statuses
        if self.waf_middleware:
            overview["components_status"]["waf"] = self.waf_middleware.get_stats()

        if self.rate_limiter:
            overview["components_status"]["rate_limiter"] = (
                self.rate_limiter.get_statistics()
            )

        return overview

    async def emergency_block_ip(self, ip: str, reason: str, duration: int = 86400):
        """Emergency block an IP address."""
        try:
            await self.redis_client.setex(f"emergency_block:{ip}", duration, reason)

            self.audit_logger.log_security_event(
                event_type="EMERGENCY_IP_BLOCK",
                message=f"Emergency block applied to IP {ip}",
                level="CRITICAL",
                details={"ip": ip, "reason": reason, "duration": duration},
            )

            return True
        except Exception as e:
            logger.error(f"Error applying emergency block: {e}")
            return False

    async def remove_emergency_block(self, ip: str) -> bool:
        """Remove emergency block from IP address."""
        try:
            await self.redis_client.delete(f"emergency_block:{ip}")

            self.audit_logger.log_security_event(
                event_type="EMERGENCY_BLOCK_REMOVED",
                message=f"Emergency block removed from IP {ip}",
                level="INFO",
                details={"ip": ip},
            )

            return True
        except Exception as e:
            logger.error(f"Error removing emergency block: {e}")
            return False
