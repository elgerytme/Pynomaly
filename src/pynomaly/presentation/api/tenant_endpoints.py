"""FastAPI endpoints for multi-tenant management."""

from datetime import datetime, timezone
from typing import Any
from uuid import UUID

from fastapi import APIRouter, Body, Depends, HTTPException, Query, status
from pydantic import BaseModel, EmailStr, Field

from pynomaly.application.services.multi_tenant_service import MultiTenantService
from pynomaly.domain.entities.tenant import (
    ResourceQuotaType,
    SubscriptionTier,
    TenantStatus,
)

# Pydantic models for API requests/responses


class TenantCreateRequest(BaseModel):
    """Request model for creating a tenant."""

    name: str = Field(..., min_length=3, max_length=50, regex=r"^[a-zA-Z0-9_-]+$")
    display_name: str = Field(..., min_length=1, max_length=100)
    contact_email: EmailStr
    subscription_tier: SubscriptionTier = SubscriptionTier.FREE
    description: str | None = Field(None, max_length=500)
    admin_user_id: UUID | None = None
    auto_activate: bool = False
    custom_config: dict[str, Any] | None = None


class TenantResponse(BaseModel):
    """Response model for tenant information."""

    tenant_id: UUID
    name: str
    display_name: str
    description: str
    status: TenantStatus
    subscription_tier: SubscriptionTier
    contact_email: str
    admin_user_id: UUID | None
    billing_contact: str
    created_at: datetime
    activated_at: datetime | None
    last_activity: datetime | None
    updated_at: datetime
    database_schema: str
    encryption_key_id: str
    total_api_requests: int
    total_cpu_hours: float
    total_storage_gb: float
    last_billing_date: datetime | None
    tags: dict[str, str]

    class Config:
        from_attributes = True


class TenantConfiguration(BaseModel):
    """Tenant configuration model."""

    max_concurrent_jobs: int
    max_model_size_mb: int
    allowed_algorithms: list[str]
    allowed_data_formats: list[str]
    enable_auto_scaling: bool
    enable_gpu_access: bool
    enable_advanced_analytics: bool
    data_retention_days: int
    backup_enabled: bool
    monitoring_level: str
    custom_settings: dict[str, Any]


class TenantDetailResponse(TenantResponse):
    """Detailed tenant response with configuration."""

    configuration: TenantConfiguration


class TenantListResponse(BaseModel):
    """Response model for tenant list."""

    tenants: list[TenantResponse]
    total_count: int
    page: int
    page_size: int


class QuotaResponse(BaseModel):
    """Response model for resource quota."""

    quota_type: str
    limit: int
    used: int
    remaining: int
    usage_percentage: float
    is_exceeded: bool
    is_unlimited: bool
    period_start: datetime
    period_end: datetime | None


class TenantUsageResponse(BaseModel):
    """Response model for tenant usage summary."""

    tenant_id: UUID
    name: str
    subscription_tier: str
    status: str
    quotas: dict[str, QuotaResponse]
    real_time_usage: dict[str, float]
    active_jobs: int
    billing: dict[str, Any]
    overall_usage_percentage: float


class TenantMetricsResponse(BaseModel):
    """Response model for comprehensive tenant metrics."""

    tenant_info: dict[str, Any]
    resource_usage: dict[str, float]
    quota_status: dict[str, dict[str, Any]]
    active_jobs: int
    billing: dict[str, Any]
    configuration: dict[str, Any]


class QuotaConsumptionRequest(BaseModel):
    """Request model for quota consumption."""

    quota_type: ResourceQuotaType
    amount: int = Field(..., gt=0)


class SubscriptionUpgradeRequest(BaseModel):
    """Request model for subscription upgrade."""

    new_tier: SubscriptionTier
    reason: str | None = None


class TenantStatusUpdateRequest(BaseModel):
    """Request model for tenant status updates."""

    status: TenantStatus
    reason: str | None = None


# API Router
router = APIRouter(prefix="/api/v1/tenants", tags=["tenants"])


# Dependency to get multi-tenant service
def get_tenant_service() -> MultiTenantService:
    """Get multi-tenant service instance."""
    return MultiTenantService()


@router.post(
    "/", response_model=TenantDetailResponse, status_code=status.HTTP_201_CREATED
)
async def create_tenant(
    request: TenantCreateRequest,
    tenant_service: MultiTenantService = Depends(get_tenant_service),
):
    """Create a new tenant."""
    try:
        tenant = await tenant_service.create_tenant(
            name=request.name,
            display_name=request.display_name,
            contact_email=request.contact_email,
            subscription_tier=request.subscription_tier,
            admin_user_id=request.admin_user_id,
            description=request.description or "",
            custom_config=request.custom_config,
        )

        # Auto-activate if requested
        if request.auto_activate:
            await tenant_service.activate_tenant(tenant.tenant_id)
            # Refresh tenant to get updated status
            tenant = await tenant_service.get_tenant(tenant.tenant_id)

        return _convert_to_detail_response(tenant)

    except ValueError as e:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(e))
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error creating tenant: {str(e)}",
        )


@router.get("/", response_model=TenantListResponse)
async def list_tenants(
    status_filter: TenantStatus | None = Query(None, alias="status"),
    tier_filter: SubscriptionTier | None = Query(None, alias="tier"),
    page: int = Query(1, ge=1),
    page_size: int = Query(50, ge=1, le=100),
    tenant_service: MultiTenantService = Depends(get_tenant_service),
):
    """List tenants with optional filtering and pagination."""
    try:
        offset = (page - 1) * page_size

        tenants = await tenant_service.list_tenants(
            status=status_filter,
            subscription_tier=tier_filter,
            limit=page_size,
            offset=offset,
        )

        # Get total count (simplified - in production you'd have a separate count method)
        all_tenants = await tenant_service.list_tenants(
            status=status_filter,
            subscription_tier=tier_filter,
            limit=10000,  # Large number to get all
        )
        total_count = len(all_tenants)

        tenant_responses = [_convert_to_response(tenant) for tenant in tenants]

        return TenantListResponse(
            tenants=tenant_responses,
            total_count=total_count,
            page=page,
            page_size=page_size,
        )

    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error listing tenants: {str(e)}",
        )


@router.get("/{tenant_id}", response_model=TenantDetailResponse)
async def get_tenant(
    tenant_id: UUID, tenant_service: MultiTenantService = Depends(get_tenant_service)
):
    """Get tenant by ID."""
    tenant = await tenant_service.get_tenant(tenant_id)

    if not tenant:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Tenant {tenant_id} not found",
        )

    return _convert_to_detail_response(tenant)


@router.get("/by-name/{tenant_name}", response_model=TenantDetailResponse)
async def get_tenant_by_name(
    tenant_name: str, tenant_service: MultiTenantService = Depends(get_tenant_service)
):
    """Get tenant by name."""
    tenant = await tenant_service.get_tenant_by_name(tenant_name)

    if not tenant:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Tenant '{tenant_name}' not found",
        )

    return _convert_to_detail_response(tenant)


@router.patch("/{tenant_id}/status", response_model=TenantDetailResponse)
async def update_tenant_status(
    tenant_id: UUID,
    request: TenantStatusUpdateRequest,
    tenant_service: MultiTenantService = Depends(get_tenant_service),
):
    """Update tenant status."""
    tenant = await tenant_service.get_tenant(tenant_id)
    if not tenant:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Tenant {tenant_id} not found",
        )

    try:
        if request.status == TenantStatus.ACTIVE:
            success = await tenant_service.activate_tenant(tenant_id)
        elif request.status == TenantStatus.SUSPENDED:
            success = await tenant_service.suspend_tenant(
                tenant_id, request.reason or ""
            )
        elif request.status == TenantStatus.DEACTIVATED:
            success = await tenant_service.deactivate_tenant(tenant_id)
        else:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Invalid status transition to {request.status}",
            )

        if not success:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Failed to update tenant status to {request.status}",
            )

        # Return updated tenant
        updated_tenant = await tenant_service.get_tenant(tenant_id)
        return _convert_to_detail_response(updated_tenant)

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error updating tenant status: {str(e)}",
        )


@router.patch("/{tenant_id}/subscription", response_model=TenantDetailResponse)
async def upgrade_subscription(
    tenant_id: UUID,
    request: SubscriptionUpgradeRequest,
    tenant_service: MultiTenantService = Depends(get_tenant_service),
):
    """Upgrade tenant subscription tier."""
    tenant = await tenant_service.get_tenant(tenant_id)
    if not tenant:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Tenant {tenant_id} not found",
        )

    try:
        success = await tenant_service.upgrade_tenant_subscription(
            tenant_id, request.new_tier
        )

        if not success:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Failed to upgrade subscription",
            )

        # Return updated tenant
        updated_tenant = await tenant_service.get_tenant(tenant_id)
        return _convert_to_detail_response(updated_tenant)

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error upgrading subscription: {str(e)}",
        )


@router.get("/{tenant_id}/usage", response_model=TenantUsageResponse)
async def get_tenant_usage(
    tenant_id: UUID, tenant_service: MultiTenantService = Depends(get_tenant_service)
):
    """Get tenant usage summary."""
    usage_summary = await tenant_service.get_tenant_usage_summary(tenant_id)

    if not usage_summary:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Tenant {tenant_id} not found",
        )

    # Convert quotas to QuotaResponse objects
    quota_responses = {}
    for quota_type, quota_info in usage_summary["quotas"].items():
        quota_responses[quota_type] = QuotaResponse(
            quota_type=quota_type,
            limit=quota_info["limit"] if quota_info["limit"] != "unlimited" else -1,
            used=quota_info["used"],
            remaining=(
                quota_info["remaining"]
                if quota_info["remaining"] != "unlimited"
                else -1
            ),
            usage_percentage=quota_info["usage_percentage"],
            is_exceeded=quota_info["is_exceeded"],
            is_unlimited=quota_info["limit"] == "unlimited",
            period_start=datetime.now(timezone.utc),  # Simplified
            period_end=None,
        )

    return TenantUsageResponse(
        tenant_id=UUID(usage_summary["tenant_id"]),
        name=usage_summary["name"],
        subscription_tier=usage_summary["subscription_tier"],
        status=usage_summary["status"],
        quotas=quota_responses,
        real_time_usage=usage_summary.get("real_time_usage", {}),
        active_jobs=usage_summary.get("active_jobs", 0),
        billing=usage_summary.get("billing", {}),
        overall_usage_percentage=usage_summary.get("overall_usage_percentage", 0.0),
    )


@router.get("/{tenant_id}/metrics", response_model=TenantMetricsResponse)
async def get_tenant_metrics(
    tenant_id: UUID, tenant_service: MultiTenantService = Depends(get_tenant_service)
):
    """Get comprehensive tenant metrics."""
    metrics = await tenant_service.get_tenant_metrics(tenant_id)

    if not metrics:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Tenant {tenant_id} not found",
        )

    return TenantMetricsResponse(**metrics)


@router.post("/{tenant_id}/quota/consume", response_model=dict[str, Any])
async def consume_quota(
    tenant_id: UUID,
    request: QuotaConsumptionRequest,
    tenant_service: MultiTenantService = Depends(get_tenant_service),
):
    """Consume tenant resource quota."""
    # Check if consumption is possible
    can_consume = await tenant_service.check_resource_quota(
        tenant_id, request.quota_type, request.amount
    )

    if not can_consume:
        raise HTTPException(
            status_code=status.HTTP_429_TOO_MANY_REQUESTS,
            detail=f"Insufficient {request.quota_type.value} quota",
        )

    # Consume the quota
    success = await tenant_service.consume_resource_quota(
        tenant_id, request.quota_type, request.amount
    )

    if not success:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST, detail="Failed to consume quota"
        )

    return {
        "success": True,
        "quota_type": request.quota_type.value,
        "amount_consumed": request.amount,
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }


@router.post("/{tenant_id}/quota/check", response_model=dict[str, Any])
async def check_quota(
    tenant_id: UUID,
    quota_type: ResourceQuotaType = Body(...),
    amount: int = Body(..., gt=0),
    tenant_service: MultiTenantService = Depends(get_tenant_service),
):
    """Check if tenant has sufficient quota."""
    can_consume = await tenant_service.check_resource_quota(
        tenant_id, quota_type, amount
    )

    # Get current quota info
    tenant = await tenant_service.get_tenant(tenant_id)
    if not tenant:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Tenant {tenant_id} not found",
        )

    quota = tenant.get_quota(quota_type)

    return {
        "can_consume": can_consume,
        "quota_type": quota_type.value,
        "requested_amount": amount,
        "current_usage": quota.used if quota else 0,
        "quota_limit": quota.limit if quota and not quota.is_unlimited else "unlimited",
        "remaining": quota.remaining if quota else 0,
    }


@router.post("/{tenant_id}/quota/reset", response_model=dict[str, Any])
async def reset_quotas(
    tenant_id: UUID, tenant_service: MultiTenantService = Depends(get_tenant_service)
):
    """Reset all quotas for a tenant (new billing period)."""
    success = await tenant_service.reset_tenant_quotas(tenant_id)

    if not success:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Tenant {tenant_id} not found",
        )

    return {
        "success": True,
        "tenant_id": str(tenant_id),
        "reset_timestamp": datetime.now(timezone.utc).isoformat(),
        "message": "All quotas reset for new billing period",
    }


@router.post("/{tenant_id}/validate-access", response_model=dict[str, Any])
async def validate_access(
    tenant_id: UUID,
    resource_path: str = Body(...),
    operation: str = Body("read"),
    tenant_service: MultiTenantService = Depends(get_tenant_service),
):
    """Validate tenant access to a resource."""
    is_valid = await tenant_service.validate_tenant_access(
        tenant_id, resource_path, operation
    )

    return {
        "tenant_id": str(tenant_id),
        "resource_path": resource_path,
        "operation": operation,
        "access_granted": is_valid,
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }


@router.get("/stats/overview", response_model=dict[str, Any])
async def get_tenant_statistics(
    tenant_service: MultiTenantService = Depends(get_tenant_service),
):
    """Get overall tenant statistics."""
    try:
        # Get all tenants for statistics
        all_tenants = await tenant_service.list_tenants(limit=10000)

        stats = {
            "total_tenants": len(all_tenants),
            "by_status": {},
            "by_tier": {},
            "active_tenants": 0,
            "total_usage": {"api_requests": 0, "cpu_hours": 0.0, "storage_gb": 0.0},
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }

        for tenant in all_tenants:
            # Count by status
            status = tenant.status.value
            stats["by_status"][status] = stats["by_status"].get(status, 0) + 1

            # Count by tier
            tier = tenant.subscription_tier.value
            stats["by_tier"][tier] = stats["by_tier"].get(tier, 0) + 1

            # Count active tenants
            if tenant.is_active():
                stats["active_tenants"] += 1

            # Aggregate usage
            stats["total_usage"]["api_requests"] += tenant.total_api_requests
            stats["total_usage"]["cpu_hours"] += tenant.total_cpu_hours
            stats["total_usage"]["storage_gb"] += tenant.total_storage_gb

        return stats

    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error getting tenant statistics: {str(e)}",
        )


# Helper functions


def _convert_to_response(tenant) -> TenantResponse:
    """Convert tenant entity to response model."""
    return TenantResponse(
        tenant_id=tenant.tenant_id,
        name=tenant.name,
        display_name=tenant.display_name,
        description=tenant.description,
        status=tenant.status,
        subscription_tier=tenant.subscription_tier,
        contact_email=tenant.contact_email,
        admin_user_id=tenant.admin_user_id,
        billing_contact=tenant.billing_contact,
        created_at=tenant.created_at,
        activated_at=tenant.activated_at,
        last_activity=tenant.last_activity,
        updated_at=tenant.updated_at,
        database_schema=tenant.database_schema,
        encryption_key_id=tenant.encryption_key_id,
        total_api_requests=tenant.total_api_requests,
        total_cpu_hours=tenant.total_cpu_hours,
        total_storage_gb=tenant.total_storage_gb,
        last_billing_date=tenant.last_billing_date,
        tags=tenant.tags,
    )


def _convert_to_detail_response(tenant) -> TenantDetailResponse:
    """Convert tenant entity to detailed response model."""
    base_response = _convert_to_response(tenant)

    config = TenantConfiguration(
        max_concurrent_jobs=tenant.configuration.max_concurrent_jobs,
        max_model_size_mb=tenant.configuration.max_model_size_mb,
        allowed_algorithms=list(tenant.configuration.allowed_algorithms),
        allowed_data_formats=list(tenant.configuration.allowed_data_formats),
        enable_auto_scaling=tenant.configuration.enable_auto_scaling,
        enable_gpu_access=tenant.configuration.enable_gpu_access,
        enable_advanced_analytics=tenant.configuration.enable_advanced_analytics,
        data_retention_days=tenant.configuration.data_retention_days,
        backup_enabled=tenant.configuration.backup_enabled,
        monitoring_level=tenant.configuration.monitoring_level,
        custom_settings=tenant.configuration.custom_settings,
    )

    return TenantDetailResponse(**base_response.dict(), configuration=config)
