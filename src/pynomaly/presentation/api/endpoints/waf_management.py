"""
WAF Management API endpoints for monitoring and administration.
"""

from datetime import datetime, timedelta
from typing import Any

from fastapi import APIRouter, Depends, HTTPException, Query, status
from pydantic import BaseModel, Field

from pynomaly.infrastructure.auth.enhanced_dependencies import (
    get_current_admin_user,
    get_current_user,
)
from pynomaly.infrastructure.config import Settings, get_settings
from pynomaly.infrastructure.security.waf_middleware import (
    AttackType,
    ThreatLevel,
    ThreatSignature,
    WAFMiddleware,
)

router = APIRouter(prefix="/api/v1/waf", tags=["WAF Management"])


class WAFStatusResponse(BaseModel):
    """WAF status response model."""

    uptime_seconds: float
    total_requests: int
    blocked_requests: int
    attacks_detected: int
    attack_rate: float
    blocked_ips: int
    suspicious_ips: int
    active_signatures: int
    blocking_enabled: bool
    monitoring_enabled: bool


class BlockedIPResponse(BaseModel):
    """Blocked IP response model."""

    ip: str
    reason: str
    timestamp: float
    duration: int
    expires_at: datetime | None = None


class SuspiciousIPResponse(BaseModel):
    """Suspicious IP response model."""

    ip: str
    reputation_score: int
    risk_level: str
    last_seen: datetime | None = None


class AttackAttemptResponse(BaseModel):
    """Attack attempt response model."""

    timestamp: datetime
    ip: str
    attack_type: str
    threat_level: str
    signature: str
    blocked: bool
    risk_score: int
    details: dict[str, Any]


class SignatureResponse(BaseModel):
    """Signature response model."""

    name: str
    pattern: str
    attack_type: str
    threat_level: str
    description: str
    enabled: bool
    custom: bool


class CreateSignatureRequest(BaseModel):
    """Create signature request model."""

    name: str = Field(..., min_length=1, max_length=100)
    pattern: str = Field(..., min_length=1, max_length=1000)
    attack_type: AttackType
    threat_level: ThreatLevel
    description: str = Field("", max_length=500)
    enabled: bool = True


class UpdateSignatureRequest(BaseModel):
    """Update signature request model."""

    enabled: bool


class BlockIPRequest(BaseModel):
    """Block IP request model."""

    ip: str = Field(..., regex=r"^(?:[0-9]{1,3}\.){3}[0-9]{1,3}$")
    reason: str = Field("Manual block", max_length=200)
    duration: int = Field(3600, ge=60, le=86400)  # 1 minute to 24 hours


class WAFConfigUpdateRequest(BaseModel):
    """WAF configuration update request model."""

    blocking_enabled: bool | None = None
    monitoring_enabled: bool | None = None
    auto_block_threshold: int | None = Field(None, ge=1, le=100)
    block_duration: int | None = Field(None, ge=300, le=86400)
    max_request_size: int | None = Field(None, ge=1024, le=104857600)


class WAFReportResponse(BaseModel):
    """WAF report response model."""

    report_period: str
    generated_at: datetime
    total_requests: int
    blocked_requests: int
    attacks_detected: int
    attack_rate: float
    top_attack_types: dict[str, int]
    top_attackers: dict[str, int]
    threat_distribution: dict[str, int]
    blocked_ips_count: int
    suspicious_ips_count: int


# Global WAF middleware instance (would be injected in production)
_waf_middleware: WAFMiddleware | None = None


def get_waf_middleware(settings: Settings = Depends(get_settings)) -> WAFMiddleware:
    """Get WAF middleware instance."""
    global _waf_middleware
    if _waf_middleware is None:
        _waf_middleware = WAFMiddleware(None, settings)
    return _waf_middleware


@router.get("/status", response_model=WAFStatusResponse)
async def get_waf_status(
    current_user: dict = Depends(get_current_user),
    waf: WAFMiddleware = Depends(get_waf_middleware)
):
    """Get WAF status and statistics."""
    stats = waf.get_stats()
    waf_stats = stats["waf_stats"]
    config = stats["config"]

    return WAFStatusResponse(
        uptime_seconds=waf_stats["uptime_seconds"],
        total_requests=waf_stats["total_requests"],
        blocked_requests=waf_stats["blocked_requests"],
        attacks_detected=waf_stats["attacks_detected"],
        attack_rate=waf_stats["attack_rate"],
        blocked_ips=stats["blocked_ips"],
        suspicious_ips=stats["suspicious_ips"],
        active_signatures=stats["active_signatures"],
        blocking_enabled=config["blocking_enabled"],
        monitoring_enabled=config["monitoring_enabled"]
    )


@router.get("/blocked-ips", response_model=list[BlockedIPResponse])
async def get_blocked_ips(
    current_user: dict = Depends(get_current_admin_user),
    waf: WAFMiddleware = Depends(get_waf_middleware)
):
    """Get list of blocked IP addresses."""
    blocked_ips = []

    try:
        keys = waf.redis_client.keys("waf:blocked:*")
        for key in keys:
            ip = key.decode('utf-8').split(':')[-1]
            data = waf.redis_client.get(key)
            if data:
                import json
                block_info = json.loads(data)
                expires_at = None
                if "timestamp" in block_info and "duration" in block_info:
                    expires_at = datetime.fromtimestamp(
                        block_info["timestamp"] + block_info["duration"]
                    )

                blocked_ips.append(BlockedIPResponse(
                    ip=ip,
                    reason=block_info.get("reason", "Unknown"),
                    timestamp=block_info.get("timestamp", 0),
                    duration=block_info.get("duration", 0),
                    expires_at=expires_at
                ))
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to retrieve blocked IPs: {str(e)}"
        )

    return blocked_ips


@router.get("/suspicious-ips", response_model=list[SuspiciousIPResponse])
async def get_suspicious_ips(
    current_user: dict = Depends(get_current_admin_user),
    waf: WAFMiddleware = Depends(get_waf_middleware)
):
    """Get list of suspicious IP addresses."""
    suspicious_ips = []

    try:
        keys = waf.redis_client.keys("waf:reputation:*")
        for key in keys:
            ip = key.decode('utf-8').split(':')[-1]
            score = waf.redis_client.get(key)
            if score:
                reputation_score = int(score)
                risk_level = (
                    "Critical" if reputation_score >= 80 else
                    "High" if reputation_score >= 60 else
                    "Medium" if reputation_score >= 40 else
                    "Low"
                )

                suspicious_ips.append(SuspiciousIPResponse(
                    ip=ip,
                    reputation_score=reputation_score,
                    risk_level=risk_level
                ))
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to retrieve suspicious IPs: {str(e)}"
        )

    return suspicious_ips


@router.post("/block-ip", status_code=status.HTTP_201_CREATED)
async def block_ip(
    request: BlockIPRequest,
    current_user: dict = Depends(get_current_admin_user),
    waf: WAFMiddleware = Depends(get_waf_middleware)
):
    """Block an IP address."""
    try:
        await waf._block_ip(request.ip, request.reason)
        return {"message": f"IP {request.ip} blocked successfully"}
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to block IP: {str(e)}"
        )


@router.delete("/blocked-ips/{ip}")
async def unblock_ip(
    ip: str,
    current_user: dict = Depends(get_current_admin_user),
    waf: WAFMiddleware = Depends(get_waf_middleware)
):
    """Unblock an IP address."""
    try:
        success = await waf.unblock_ip(ip)
        if success:
            return {"message": f"IP {ip} unblocked successfully"}
        else:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"IP {ip} not found in blocked list"
            )
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to unblock IP: {str(e)}"
        )


@router.get("/attacks", response_model=list[AttackAttemptResponse])
async def get_recent_attacks(
    limit: int = Query(100, ge=1, le=1000),
    attack_type: str | None = Query(None),
    threat_level: str | None = Query(None),
    since: datetime | None = Query(None),
    current_user: dict = Depends(get_current_admin_user),
    waf: WAFMiddleware = Depends(get_waf_middleware)
):
    """Get recent attack attempts."""
    attacks = []

    try:
        # Get attacks from WAF middleware
        recent_attacks = list(waf.stats.recent_attacks)

        # Apply filters
        if attack_type:
            recent_attacks = [a for a in recent_attacks if a.attack_type.value == attack_type]

        if threat_level:
            recent_attacks = [a for a in recent_attacks if a.threat_level.value == threat_level]

        if since:
            cutoff_timestamp = since.timestamp()
            recent_attacks = [a for a in recent_attacks if a.timestamp > cutoff_timestamp]

        # Convert to response format
        for attack in recent_attacks[-limit:]:
            attacks.append(AttackAttemptResponse(
                timestamp=datetime.fromtimestamp(attack.timestamp),
                ip=attack.ip,
                attack_type=attack.attack_type.value,
                threat_level=attack.threat_level.value,
                signature=attack.signature,
                blocked=attack.blocked,
                risk_score=attack.risk_score,
                details=attack.details
            ))

    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to retrieve attacks: {str(e)}"
        )

    return attacks


@router.get("/signatures", response_model=list[SignatureResponse])
async def get_signatures(
    enabled_only: bool = Query(False),
    attack_type: str | None = Query(None),
    current_user: dict = Depends(get_current_user),
    waf: WAFMiddleware = Depends(get_waf_middleware)
):
    """Get threat detection signatures."""
    signatures = []

    for sig in waf.signatures:
        if enabled_only and not sig.enabled:
            continue

        if attack_type and sig.attack_type.value != attack_type:
            continue

        signatures.append(SignatureResponse(
            name=sig.name,
            pattern=sig.pattern,
            attack_type=sig.attack_type.value,
            threat_level=sig.threat_level.value,
            description=sig.description,
            enabled=sig.enabled,
            custom=sig.custom
        ))

    return signatures


@router.post("/signatures", status_code=status.HTTP_201_CREATED)
async def create_signature(
    request: CreateSignatureRequest,
    current_user: dict = Depends(get_current_admin_user),
    waf: WAFMiddleware = Depends(get_waf_middleware)
):
    """Create a new threat detection signature."""
    try:
        # Test regex pattern
        import re
        re.compile(request.pattern)

        # Create signature
        signature = ThreatSignature(
            name=request.name,
            pattern=request.pattern,
            attack_type=request.attack_type,
            threat_level=request.threat_level,
            description=request.description,
            enabled=request.enabled,
            custom=True
        )

        # Add to WAF (this would typically save to database/config)
        waf.signatures.append(signature)
        waf.compiled_patterns[signature.name] = re.compile(
            signature.pattern, signature.regex_flags
        )

        return {"message": f"Signature '{request.name}' created successfully"}

    except re.error as e:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Invalid regex pattern: {str(e)}"
        )
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to create signature: {str(e)}"
        )


@router.patch("/signatures/{signature_name}")
async def update_signature(
    signature_name: str,
    request: UpdateSignatureRequest,
    current_user: dict = Depends(get_current_admin_user),
    waf: WAFMiddleware = Depends(get_waf_middleware)
):
    """Update a threat detection signature."""
    try:
        # Find signature
        for sig in waf.signatures:
            if sig.name == signature_name:
                sig.enabled = request.enabled

                # Update compiled patterns
                if request.enabled:
                    import re
                    waf.compiled_patterns[sig.name] = re.compile(
                        sig.pattern, sig.regex_flags
                    )
                else:
                    waf.compiled_patterns.pop(sig.name, None)

                return {"message": f"Signature '{signature_name}' updated successfully"}

        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Signature '{signature_name}' not found"
        )

    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to update signature: {str(e)}"
        )


@router.delete("/signatures/{signature_name}")
async def delete_signature(
    signature_name: str,
    current_user: dict = Depends(get_current_admin_user),
    waf: WAFMiddleware = Depends(get_waf_middleware)
):
    """Delete a custom threat detection signature."""
    try:
        # Find and remove signature
        for i, sig in enumerate(waf.signatures):
            if sig.name == signature_name:
                if not sig.custom:
                    raise HTTPException(
                        status_code=status.HTTP_400_BAD_REQUEST,
                        detail="Cannot delete built-in signatures"
                    )

                waf.signatures.pop(i)
                waf.compiled_patterns.pop(signature_name, None)

                return {"message": f"Signature '{signature_name}' deleted successfully"}

        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Signature '{signature_name}' not found"
        )

    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to delete signature: {str(e)}"
        )


@router.patch("/config")
async def update_config(
    request: WAFConfigUpdateRequest,
    current_user: dict = Depends(get_current_admin_user),
    waf: WAFMiddleware = Depends(get_waf_middleware)
):
    """Update WAF configuration."""
    try:
        updates = request.dict(exclude_unset=True)

        # Update configuration
        waf.config.update(updates)

        return {"message": "WAF configuration updated successfully"}

    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to update configuration: {str(e)}"
        )


@router.get("/report", response_model=WAFReportResponse)
async def generate_report(
    hours: int = Query(24, ge=1, le=168),  # 1 hour to 1 week
    current_user: dict = Depends(get_current_admin_user),
    waf: WAFMiddleware = Depends(get_waf_middleware)
):
    """Generate WAF security report."""
    try:
        stats = waf.get_stats()
        waf_stats = stats["waf_stats"]

        # Calculate attack statistics
        cutoff_time = datetime.now() - timedelta(hours=hours)
        recent_attacks = [
            attack for attack in waf.stats.recent_attacks
            if datetime.fromtimestamp(attack.timestamp) > cutoff_time
        ]

        # Calculate top attack types
        attack_types = {}
        for attack in recent_attacks:
            attack_type = attack.attack_type.value
            attack_types[attack_type] = attack_types.get(attack_type, 0) + 1

        # Calculate top attackers
        attackers = {}
        for attack in recent_attacks:
            ip = attack.ip
            attackers[ip] = attackers.get(ip, 0) + 1

        # Calculate threat distribution
        threat_distribution = {}
        for attack in recent_attacks:
            level = attack.threat_level.value
            threat_distribution[level] = threat_distribution.get(level, 0) + 1

        return WAFReportResponse(
            report_period=f"{hours} hours",
            generated_at=datetime.now(),
            total_requests=waf_stats["total_requests"],
            blocked_requests=waf_stats["blocked_requests"],
            attacks_detected=len(recent_attacks),
            attack_rate=waf_stats["attack_rate"],
            top_attack_types=dict(sorted(attack_types.items(), key=lambda x: x[1], reverse=True)),
            top_attackers=dict(sorted(attackers.items(), key=lambda x: x[1], reverse=True)[:10]),
            threat_distribution=threat_distribution,
            blocked_ips_count=stats["blocked_ips"],
            suspicious_ips_count=stats["suspicious_ips"]
        )

    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to generate report: {str(e)}"
        )


@router.get("/health")
async def waf_health_check(
    waf: WAFMiddleware = Depends(get_waf_middleware)
):
    """WAF health check endpoint."""
    try:
        # Basic health checks
        redis_connected = waf.redis_client.ping()
        signatures_loaded = len(waf.signatures) > 0

        status = "healthy" if redis_connected and signatures_loaded else "unhealthy"

        return {
            "status": status,
            "redis_connected": redis_connected,
            "signatures_loaded": signatures_loaded,
            "signature_count": len(waf.signatures),
            "active_signatures": len([s for s in waf.signatures if s.enabled])
        }

    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"WAF health check failed: {str(e)}"
        )
