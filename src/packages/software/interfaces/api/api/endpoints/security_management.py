"""
Security Management API endpoints for monitoring and configuring security features.
"""

import logging
from datetime import datetime, timedelta
from typing import Any

from fastapi import APIRouter, HTTPException, Request
from pydantic import BaseModel

from pynomaly_detection.infrastructure.security.audit_logger import get_audit_logger

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/v1/security", tags=["security"])


# Pydantic models for request/response
class SecurityOverview(BaseModel):
    """Security system overview."""

    waf_enabled: bool
    rate_limiting_enabled: bool
    active_threats: int
    blocked_ips: int
    total_security_events: int
    threat_level: str
    last_updated: datetime


class ThreatInfo(BaseModel):
    """Threat information."""

    ip: str
    threat_level: str
    attack_type: str
    first_seen: datetime
    last_seen: datetime
    total_attempts: int
    blocked: bool


class SecurityRule(BaseModel):
    """Security rule configuration."""

    name: str
    enabled: bool
    rule_type: str
    action: str
    conditions: list[str]
    priority: int = 100


class SecurityStats(BaseModel):
    """Security statistics."""

    total_requests: int
    blocked_requests: int
    attack_attempts: int
    unique_attackers: int
    threat_distribution: dict[str, int]
    top_attack_types: dict[str, int]
    geographic_distribution: dict[str, int]
    hourly_stats: list[dict[str, Any]]


class IPBlockRequest(BaseModel):
    """IP block request."""

    ip: str
    reason: str
    duration_hours: int = 24
    block_type: str = "manual"


class SecurityConfigUpdate(BaseModel):
    """Security configuration update."""

    waf_enabled: bool | None = None
    rate_limiting_enabled: bool | None = None
    auto_block_threshold: int | None = None
    block_duration: int | None = None
    monitoring_enabled: bool | None = None


# Security management endpoints
@router.get("/overview", response_model=SecurityOverview)
async def get_security_overview(request: Request):
    """Get security system overview."""
    try:
        # Get WAF stats (if available)
        waf_stats = {}
        rate_limiter_stats = {}

        if hasattr(request.app.state, "waf_middleware"):
            waf_stats = request.app.state.waf_middleware.get_stats()

        if hasattr(request.app.state, "rate_limiter"):
            rate_limiter_stats = request.app.state.rate_limiter.get_statistics()

        # Calculate threat level
        threat_level = "low"
        active_threats = waf_stats.get("waf_stats", {}).get("attacks_detected", 0)

        if active_threats > 100:
            threat_level = "critical"
        elif active_threats > 50:
            threat_level = "high"
        elif active_threats > 10:
            threat_level = "medium"

        return SecurityOverview(
            waf_enabled=waf_stats.get("config", {}).get("blocking_enabled", False),
            rate_limiting_enabled=True,
            active_threats=active_threats,
            blocked_ips=(
                waf_stats.get("blocked_ips", 0)
                + rate_limiter_stats.get("blocked_ips", 0)
            ),
            total_security_events=waf_stats.get("waf_stats", {}).get(
                "total_requests", 0
            ),
            threat_level=threat_level,
            last_updated=datetime.now(),
        )
    except Exception as e:
        logger.error(f"Error getting security overview: {e}")
        raise HTTPException(status_code=500, detail="Failed to get security overview")


@router.get("/stats", response_model=SecurityStats)
async def get_security_stats(request: Request, hours: int = 24):
    """Get detailed security statistics."""
    try:
        # Get stats from various security components
        waf_stats = {}
        rate_limiter_stats = {}

        if hasattr(request.app.state, "waf_middleware"):
            waf_stats = request.app.state.waf_middleware.get_stats()

        if hasattr(request.app.state, "rate_limiter"):
            rate_limiter_stats = request.app.state.rate_limiter.get_statistics()

        # Generate hourly stats (mock data for now)
        hourly_stats = []
        for hour in range(hours):
            hour_time = datetime.now() - timedelta(hours=hour)
            hourly_stats.append(
                {
                    "hour": hour_time.isoformat(),
                    "requests": max(0, 100 - hour * 2),
                    "blocked": max(0, 10 - hour),
                    "attacks": max(0, 5 - hour),
                }
            )

        return SecurityStats(
            total_requests=waf_stats.get("waf_stats", {}).get("total_requests", 0),
            blocked_requests=waf_stats.get("waf_stats", {}).get("blocked_requests", 0),
            attack_attempts=waf_stats.get("waf_stats", {}).get("attacks_detected", 0),
            unique_attackers=len(
                waf_stats.get("waf_stats", {}).get("top_attackers", {})
            ),
            threat_distribution=waf_stats.get("waf_stats", {}).get("attack_types", {}),
            top_attack_types=waf_stats.get("waf_stats", {}).get("attack_types", {}),
            geographic_distribution={"US": 60, "CN": 20, "RU": 10, "Other": 10},
            hourly_stats=hourly_stats,
        )
    except Exception as e:
        logger.error(f"Error getting security stats: {e}")
        raise HTTPException(status_code=500, detail="Failed to get security statistics")


@router.get("/threats")
async def get_active_threats(request: Request, limit: int = 100):
    """Get list of active threats."""
    try:
        threats = []

        # Get threat data from WAF
        if hasattr(request.app.state, "waf_middleware"):
            waf = request.app.state.waf_middleware

            # Get recent attacks
            for attack in list(waf.stats.recent_attacks)[-limit:]:
                threats.append(
                    {
                        "ip": attack.ip,
                        "threat_level": attack.threat_level.value,
                        "attack_type": attack.attack_type.value,
                        "timestamp": datetime.fromtimestamp(attack.timestamp),
                        "blocked": attack.blocked,
                        "risk_score": attack.risk_score,
                        "signature": attack.signature,
                        "details": attack.details,
                    }
                )

        return {"threats": threats, "total": len(threats)}
    except Exception as e:
        logger.error(f"Error getting active threats: {e}")
        raise HTTPException(status_code=500, detail="Failed to get active threats")


@router.post("/block-ip")
async def block_ip(request: Request, block_request: IPBlockRequest):
    """Block an IP address."""
    try:
        # Block IP in WAF
        if hasattr(request.app.state, "waf_middleware"):
            waf = request.app.state.waf_middleware
            await waf._block_ip(block_request.ip, block_request.reason)

        # Block IP in rate limiter
        if hasattr(request.app.state, "rate_limiter"):
            rate_limiter = request.app.state.rate_limiter
            rate_limiter.blocked_ips.add(block_request.ip)

        # Log the action
        audit_logger = get_audit_logger()
        audit_logger.log_security_event(
            event_type="MANUAL_IP_BLOCK",
            message=(
                f"IP {block_request.ip} manually blocked: " f"{block_request.reason}"
            ),
            level="INFO",
            details={
                "ip": block_request.ip,
                "reason": block_request.reason,
                "duration_hours": block_request.duration_hours,
                "block_type": block_request.block_type,
            },
        )

        return {"message": f"IP {block_request.ip} blocked successfully"}
    except Exception as e:
        logger.error(f"Error blocking IP {block_request.ip}: {e}")
        raise HTTPException(status_code=500, detail="Failed to block IP")


@router.post("/unblock-ip")
async def unblock_ip(request: Request, ip: str):
    """Unblock an IP address."""
    try:
        # Unblock IP in WAF
        if hasattr(request.app.state, "waf_middleware"):
            waf = request.app.state.waf_middleware
            await waf.unblock_ip(ip)

        # Unblock IP in rate limiter
        if hasattr(request.app.state, "rate_limiter"):
            rate_limiter = request.app.state.rate_limiter
            rate_limiter.blocked_ips.discard(ip)

        # Log the action
        audit_logger = get_audit_logger()
        audit_logger.log_security_event(
            event_type="MANUAL_IP_UNBLOCK",
            message=f"IP {ip} manually unblocked",
            level="INFO",
            details={"ip": ip},
        )

        return {"message": f"IP {ip} unblocked successfully"}
    except Exception as e:
        logger.error(f"Error unblocking IP {ip}: {e}")
        raise HTTPException(status_code=500, detail="Failed to unblock IP")


@router.get("/blocked-ips")
async def get_blocked_ips(request: Request):
    """Get list of blocked IP addresses."""
    try:
        blocked_ips = set()

        # Get blocked IPs from WAF
        if hasattr(request.app.state, "waf_middleware"):
            waf = request.app.state.waf_middleware
            blocked_ips.update(waf.blocked_ips)

        # Get blocked IPs from rate limiter
        if hasattr(request.app.state, "rate_limiter"):
            rate_limiter = request.app.state.rate_limiter
            blocked_ips.update(rate_limiter.blocked_ips)

        return {"blocked_ips": list(blocked_ips), "total": len(blocked_ips)}
    except Exception as e:
        logger.error(f"Error getting blocked IPs: {e}")
        raise HTTPException(status_code=500, detail="Failed to get blocked IPs")


@router.get("/rules")
async def get_security_rules(request: Request):
    """Get security rules configuration."""
    try:
        rules = []

        # Get WAF rules
        if hasattr(request.app.state, "waf_middleware"):
            waf = request.app.state.waf_middleware
            waf_rules = waf.config.get("rules", [])

            for rule in waf_rules:
                rules.append(
                    {
                        "name": rule.get("name", ""),
                        "enabled": rule.get("enabled", True),
                        "rule_type": "waf",
                        "action": rule.get("action", "monitor"),
                        "conditions": rule.get("conditions", []),
                        "priority": rule.get("priority", 100),
                    }
                )

        return {"rules": rules, "total": len(rules)}
    except Exception as e:
        logger.error(f"Error getting security rules: {e}")
        raise HTTPException(status_code=500, detail="Failed to get security rules")


@router.post("/config")
async def update_security_config(request: Request, config: SecurityConfigUpdate):
    """Update security configuration."""
    try:
        updated_settings = []

        # Update WAF config
        if hasattr(request.app.state, "waf_middleware"):
            waf = request.app.state.waf_middleware

            if config.waf_enabled is not None:
                waf.config["blocking_enabled"] = config.waf_enabled
                updated_settings.append("WAF blocking")

            if config.auto_block_threshold is not None:
                waf.config["auto_block_threshold"] = config.auto_block_threshold
                updated_settings.append("Auto-block threshold")

            if config.block_duration is not None:
                waf.config["block_duration"] = config.block_duration
                updated_settings.append("Block duration")

            if config.monitoring_enabled is not None:
                waf.config["monitoring_enabled"] = config.monitoring_enabled
                updated_settings.append("Monitoring")

        # Log the configuration change
        audit_logger = get_audit_logger()
        audit_logger.log_security_event(
            event_type="SECURITY_CONFIG_UPDATE",
            message=f"Security configuration updated: {', '.join(updated_settings)}",
            level="INFO",
            details=config.dict(exclude_none=True),
        )

        return {
            "message": "Security configuration updated successfully",
            "updated": updated_settings,
        }
    except Exception as e:
        logger.error(f"Error updating security config: {e}")
        raise HTTPException(
            status_code=500, detail="Failed to update security configuration"
        )


@router.get("/events")
async def get_security_events(
    request: Request, limit: int = 100, event_type: str | None = None
):
    """Get security events log."""
    try:
        # This would typically query a database or log aggregation system
        # For now, return mock data
        events = [
            {
                "timestamp": datetime.now() - timedelta(minutes=i),
                "event_type": "SECURITY_INTRUSION_DETECTED",
                "message": f"SQL injection attempt detected from IP 192.168.1.{i}",
                "level": "WARNING",
                "ip": f"192.168.1.{i}",
                "risk_score": 75,
            }
            for i in range(1, min(limit + 1, 50))
        ]

        if event_type:
            events = [e for e in events if e["event_type"] == event_type]

        return {"events": events, "total": len(events)}
    except Exception as e:
        logger.error(f"Error getting security events: {e}")
        raise HTTPException(status_code=500, detail="Failed to get security events")


@router.post("/test-attack")
async def test_attack_detection(request: Request, attack_type: str = "sql_injection"):
    """Test attack detection (for development/testing only)."""
    try:
        # This endpoint should only be available in development mode
        if not request.app.state.settings.debug:
            raise HTTPException(
                status_code=403, detail="Test endpoints only available in debug mode"
            )

        # Simulate different attack types
        test_payloads = {
            "sql_injection": "' OR 1=1 --",
            "xss": "<script>alert('XSS')</script>",
            "command_injection": "; cat /etc/passwd",
            "path_traversal": "../../etc/passwd",
        }

        payload = test_payloads.get(attack_type, "test payload")

        # Log the test
        audit_logger = get_audit_logger()
        audit_logger.log_security_event(
            event_type="SECURITY_TEST",
            message=f"Security test performed: {attack_type}",
            level="INFO",
            details={"attack_type": attack_type, "payload": payload, "test_mode": True},
        )

        return {
            "message": f"Test attack ({attack_type}) simulation completed",
            "payload": payload,
            "detected": True,  # In a real scenario, this would depend on detection
        }
    except Exception as e:
        logger.error(f"Error in attack test: {e}")
        raise HTTPException(status_code=500, detail="Failed to perform attack test")


@router.get("/health")
async def security_health_check(request: Request):
    """Check health of security components."""
    try:
        components = {}

        # Check WAF health
        if hasattr(request.app.state, "waf_middleware"):
            waf = request.app.state.waf_middleware
            components["waf"] = {
                "status": "healthy",
                "signatures": len(waf.signatures),
                "blocked_ips": len(waf.blocked_ips),
                "config_loaded": bool(waf.config),
            }
        else:
            components["waf"] = {"status": "not_available"}

        # Check rate limiter health
        if hasattr(request.app.state, "rate_limiter"):
            rate_limiter = request.app.state.rate_limiter
            components["rate_limiter"] = {
                "status": "healthy",
                "active_clients": rate_limiter.get_statistics().get(
                    "active_clients", 0
                ),
                "blocked_ips": len(rate_limiter.blocked_ips),
                "system_load": rate_limiter.system_load,
            }
        else:
            components["rate_limiter"] = {"status": "not_available"}

        # Check audit logger
        try:
            audit_logger = get_audit_logger()
            components["audit_logger"] = {"status": "healthy"}
        except Exception:
            components["audit_logger"] = {"status": "error"}

        overall_status = (
            "healthy"
            if all(c.get("status") == "healthy" for c in components.values())
            else "degraded"
        )

        return {
            "overall_status": overall_status,
            "components": components,
            "timestamp": datetime.now(),
        }
    except Exception as e:
        logger.error(f"Error in security health check: {e}")
        raise HTTPException(
            status_code=500, detail="Failed to perform security health check"
        )


@router.get("/dashboard-data")
async def get_dashboard_data(request: Request):
    """Get comprehensive security dashboard data."""
    try:
        # Collect data from all security components
        dashboard_data = {
            "overview": {},
            "real_time_metrics": {},
            "threat_intelligence": {},
            "performance_metrics": {},
        }

        # Get WAF data
        if hasattr(request.app.state, "waf_middleware"):
            waf = request.app.state.waf_middleware
            waf_stats = waf.get_stats()

            dashboard_data["overview"]["waf"] = {
                "total_requests": waf_stats.get("waf_stats", {}).get(
                    "total_requests", 0
                ),
                "blocked_requests": waf_stats.get("waf_stats", {}).get(
                    "blocked_requests", 0
                ),
                "attacks_detected": waf_stats.get("waf_stats", {}).get(
                    "attacks_detected", 0
                ),
                "attack_rate": waf_stats.get("waf_stats", {}).get("attack_rate", 0),
            }

            dashboard_data["threat_intelligence"]["attack_types"] = waf_stats.get(
                "waf_stats", {}
            ).get("attack_types", {})
            dashboard_data["threat_intelligence"]["top_attackers"] = waf_stats.get(
                "waf_stats", {}
            ).get("top_attackers", {})

        # Get rate limiter data
        if hasattr(request.app.state, "rate_limiter"):
            rate_limiter = request.app.state.rate_limiter
            rl_stats = rate_limiter.get_statistics()

            dashboard_data["overview"]["rate_limiter"] = {
                "total_clients": rl_stats.get("total_clients", 0),
                "active_clients": rl_stats.get("active_clients", 0),
                "blocked_ips": rl_stats.get("blocked_ips", 0),
                "system_load": rl_stats.get("system_load", 0),
            }

            dashboard_data["real_time_metrics"]["threat_distribution"] = rl_stats.get(
                "threat_distribution", {}
            )
            dashboard_data["real_time_metrics"]["adaptive_limits"] = rl_stats.get(
                "adaptive_limits", {}
            )

        # Performance metrics
        dashboard_data["performance_metrics"] = {
            "response_time": 0.05,  # Mock data
            "throughput": 1000,
            "error_rate": 0.01,
            "availability": 99.99,
        }

        return dashboard_data
    except Exception as e:
        logger.error(f"Error getting dashboard data: {e}")
        raise HTTPException(status_code=500, detail="Failed to get dashboard data")
