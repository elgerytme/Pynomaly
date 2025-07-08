"""
FastAPI router for third-party integrations management.
"""

from datetime import datetime
from typing import List, Optional, Dict, Any
from uuid import UUID

from fastapi import APIRouter, Depends, HTTPException, status
from pydantic import BaseModel, HttpUrl

from pynomaly.application.services.integration_service import IntegrationService
from pynomaly.domain.entities.integrations import (
    IntegrationType, IntegrationStatus, IntegrationConfig, NotificationLevel,
    TriggerType, NotificationPayload, NotificationTemplate
)
from pynomaly.domain.entities.user import User
from pynomaly.shared.exceptions import (
    ValidationError, IntegrationError, NotificationError, AuthenticationError
)
from pynomaly.shared.types import TenantId, UserId

# Router setup
router = APIRouter(prefix="/api/integrations", tags=["Integrations"])

# Request/Response Models
class CreateIntegrationRequest(BaseModel):
    name: str
    integration_type: IntegrationType
    config: Dict[str, Any]
    credentials: Optional[Dict[str, Any]] = None


class IntegrationConfigRequest(BaseModel):
    enabled: bool = True
    notification_levels: List[NotificationLevel]
    triggers: List[TriggerType]
    retry_count: int = 3
    retry_delay_seconds: int = 60
    timeout_seconds: int = 30
    rate_limit_per_minute: int = 60
    template_id: Optional[str] = None
    custom_template: Optional[Dict[str, str]] = None
    include_charts: bool = False
    include_raw_data: bool = False
    filters: Dict[str, Any] = {}
    settings: Dict[str, Any] = {}


class IntegrationResponse(BaseModel):
    id: str
    name: str
    integration_type: IntegrationType
    status: IntegrationStatus
    config: Dict[str, Any]
    created_at: datetime
    updated_at: datetime
    last_triggered: Optional[datetime]
    trigger_count: int
    success_count: int
    error_count: int
    success_rate: float
    is_healthy: bool


class UpdateCredentialsRequest(BaseModel):
    credentials: Dict[str, Any]


class SendNotificationRequest(BaseModel):
    trigger_type: TriggerType
    level: NotificationLevel
    title: str
    message: str
    data: Dict[str, Any] = {}
    integration_types: Optional[List[IntegrationType]] = None


class CreateTemplateRequest(BaseModel):
    name: str
    integration_type: IntegrationType
    trigger_types: List[TriggerType]
    title_template: str
    message_template: str
    variables: List[str] = []


class TemplateResponse(BaseModel):
    id: str
    name: str
    integration_type: IntegrationType
    trigger_types: List[TriggerType]
    title_template: str
    message_template: str
    is_default: bool
    variables: List[str]
    created_at: datetime
    updated_at: datetime


class NotificationHistoryResponse(BaseModel):
    id: str
    integration_id: str
    trigger_type: TriggerType
    level: NotificationLevel
    title: str
    message: str
    response_status: int
    was_successful: bool
    sent_at: datetime
    delivery_time_ms: Optional[int]
    retry_count: int
    error_message: Optional[str]


class IntegrationMetricsResponse(BaseModel):
    integration_id: str
    total_notifications: int
    successful_notifications: int
    failed_notifications: int
    success_rate: float
    failure_rate: float
    average_delivery_time_ms: float
    last_success: Optional[datetime]
    last_failure: Optional[datetime]
    uptime_percentage: float
    rate_limit_hits: int


# Dependencies
async def get_integration_service() -> IntegrationService:
    """Get integration service instance."""
    # TODO: Implement proper dependency injection
    pass


async def get_current_user() -> User:
    """Get current authenticated user."""
    # TODO: Implement authentication
    pass


async def require_tenant_access(tenant_id: UUID, current_user: User = Depends(get_current_user)):
    """Require access to specific tenant."""
    if not (current_user.is_super_admin() or
            current_user.has_role_in_tenant(TenantId(str(tenant_id)), ["data_scientist", "tenant_admin"])):
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Access denied to tenant"
        )
    return current_user


# Integration Management Endpoints
@router.post("/tenants/{tenant_id}/integrations", response_model=IntegrationResponse, status_code=status.HTTP_201_CREATED)
async def create_integration(
    tenant_id: UUID,
    request: CreateIntegrationRequest,
    current_user: User = Depends(require_tenant_access),
    integration_service: IntegrationService = Depends(get_integration_service)
):
    """Create a new integration."""
    try:
        # Convert config dict to IntegrationConfig object
        config = IntegrationConfig(**request.config)

        integration = await integration_service.create_integration(
            name=request.name,
            integration_type=request.integration_type,
            tenant_id=TenantId(str(tenant_id)),
            user_id=UserId(current_user.id),
            config=config,
            credentials=request.credentials
        )

        return IntegrationResponse(
            id=integration.id,
            name=integration.name,
            integration_type=integration.integration_type,
            status=integration.status,
            config=integration.config.__dict__,
            created_at=integration.created_at,
            updated_at=integration.updated_at,
            last_triggered=integration.last_triggered,
            trigger_count=integration.trigger_count,
            success_count=integration.success_count,
            error_count=integration.error_count,
            success_rate=integration.success_rate,
            is_healthy=integration.is_healthy()
        )
    except (ValidationError, IntegrationError) as e:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(e)
        )


@router.get("/tenants/{tenant_id}/integrations", response_model=List[IntegrationResponse])
async def list_integrations(
    tenant_id: UUID,
    current_user: User = Depends(require_tenant_access),
    integration_service: IntegrationService = Depends(get_integration_service)
):
    """List all integrations for a tenant."""
    try:
        integrations = await integration_service.get_integrations_for_tenant(TenantId(str(tenant_id)))

        return [
            IntegrationResponse(
                id=integration.id,
                name=integration.name,
                integration_type=integration.integration_type,
                status=integration.status,
                config=integration.config.__dict__,
                created_at=integration.created_at,
                updated_at=integration.updated_at,
                last_triggered=integration.last_triggered,
                trigger_count=integration.trigger_count,
                success_count=integration.success_count,
                error_count=integration.error_count,
                success_rate=integration.success_rate,
                is_healthy=integration.is_healthy()
            )
            for integration in integrations
        ]
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to retrieve integrations: {str(e)}"
        )


@router.get("/tenants/{tenant_id}/integrations/{integration_id}", response_model=IntegrationResponse)
async def get_integration(
    tenant_id: UUID,
    integration_id: str,
    current_user: User = Depends(require_tenant_access),
    integration_service: IntegrationService = Depends(get_integration_service)
):
    """Get a specific integration."""
    try:
        # TODO: Implement get_integration_by_id in service
        raise HTTPException(
            status_code=status.HTTP_501_NOT_IMPLEMENTED,
            detail="Get integration not implemented yet"
        )
    except IntegrationError:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Integration not found"
        )


@router.put("/tenants/{tenant_id}/integrations/{integration_id}/credentials")
async def update_integration_credentials(
    tenant_id: UUID,
    integration_id: str,
    request: UpdateCredentialsRequest,
    current_user: User = Depends(require_tenant_access),
    integration_service: IntegrationService = Depends(get_integration_service)
):
    """Update integration credentials."""
    try:
        integration = await integration_service.update_integration_credentials(
            integration_id=integration_id,
            user_id=UserId(current_user.id),
            credentials=request.credentials
        )

        return {
            "message": "Credentials updated successfully",
            "integration_id": integration.id,
            "status": integration.status.value
        }
    except (AuthenticationError, IntegrationError) as e:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(e)
        )


@router.delete("/tenants/{tenant_id}/integrations/{integration_id}")
async def delete_integration(
    tenant_id: UUID,
    integration_id: str,
    current_user: User = Depends(require_tenant_access),
    integration_service: IntegrationService = Depends(get_integration_service)
):
    """Delete an integration."""
    try:
        success = await integration_service.delete_integration(
            integration_id=integration_id,
            user_id=UserId(current_user.id)
        )

        if success:
            return {"message": "Integration deleted successfully"}
        else:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Integration not found"
            )
    except AuthenticationError as e:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail=str(e)
        )


# Notification Endpoints
@router.post("/tenants/{tenant_id}/notifications/send")
async def send_notification(
    tenant_id: UUID,
    request: SendNotificationRequest,
    current_user: User = Depends(require_tenant_access),
    integration_service: IntegrationService = Depends(get_integration_service)
):
    """Send a notification to all matching integrations."""
    try:
        payload = NotificationPayload(
            trigger_type=request.trigger_type,
            level=request.level,
            title=request.title,
            message=request.message,
            tenant_id=TenantId(str(tenant_id)),
            user_id=UserId(current_user.id),
            data=request.data
        )

        results = await integration_service.send_notification(
            tenant_id=TenantId(str(tenant_id)),
            payload=payload,
            integration_types=request.integration_types
        )

        successful_count = sum(1 for success in results.values() if success)
        total_count = len(results)

        return {
            "message": "Notifications sent",
            "total_integrations": total_count,
            "successful_deliveries": successful_count,
            "failed_deliveries": total_count - successful_count,
            "results": results
        }
    except (ValidationError, NotificationError) as e:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(e)
        )


@router.get("/tenants/{tenant_id}/integrations/{integration_id}/history", response_model=List[NotificationHistoryResponse])
async def get_notification_history(
    tenant_id: UUID,
    integration_id: str,
    limit: int = 50,
    offset: int = 0,
    current_user: User = Depends(require_tenant_access),
    integration_service: IntegrationService = Depends(get_integration_service)
):
    """Get notification history for an integration."""
    try:
        history = await integration_service.get_notification_history(
            integration_id=integration_id,
            limit=limit,
            offset=offset
        )

        return [
            NotificationHistoryResponse(
                id=record.id,
                integration_id=record.integration_id,
                trigger_type=record.payload.trigger_type,
                level=record.payload.level,
                title=record.payload.title,
                message=record.payload.message,
                response_status=record.response_status,
                was_successful=record.was_successful,
                sent_at=record.sent_at,
                delivery_time_ms=record.delivery_time_ms,
                retry_count=record.retry_count,
                error_message=record.error_message
            )
            for record in history
        ]
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to retrieve notification history: {str(e)}"
        )


@router.get("/tenants/{tenant_id}/integrations/{integration_id}/metrics", response_model=IntegrationMetricsResponse)
async def get_integration_metrics(
    tenant_id: UUID,
    integration_id: str,
    current_user: User = Depends(require_tenant_access),
    integration_service: IntegrationService = Depends(get_integration_service)
):
    """Get performance metrics for an integration."""
    try:
        metrics = await integration_service.get_integration_metrics(integration_id)

        if not metrics:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Integration metrics not found"
            )

        return IntegrationMetricsResponse(
            integration_id=metrics.integration_id,
            total_notifications=metrics.total_notifications,
            successful_notifications=metrics.successful_notifications,
            failed_notifications=metrics.failed_notifications,
            success_rate=metrics.success_rate,
            failure_rate=metrics.failure_rate,
            average_delivery_time_ms=metrics.average_delivery_time_ms,
            last_success=metrics.last_success,
            last_failure=metrics.last_failure,
            uptime_percentage=metrics.uptime_percentage,
            rate_limit_hits=metrics.rate_limit_hits
        )
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to retrieve integration metrics: {str(e)}"
        )


@router.post("/tenants/{tenant_id}/notifications/{history_id}/retry")
async def retry_notification(
    tenant_id: UUID,
    history_id: str,
    current_user: User = Depends(require_tenant_access),
    integration_service: IntegrationService = Depends(get_integration_service)
):
    """Retry a failed notification."""
    try:
        success = await integration_service.retry_failed_notification(history_id)

        if success:
            return {"message": "Notification retry successful"}
        else:
            return {"message": "Notification retry failed"}
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to retry notification: {str(e)}"
        )


# Template Management Endpoints
@router.post("/tenants/{tenant_id}/templates", response_model=TemplateResponse, status_code=status.HTTP_201_CREATED)
async def create_notification_template(
    tenant_id: UUID,
    request: CreateTemplateRequest,
    current_user: User = Depends(require_tenant_access),
    integration_service: IntegrationService = Depends(get_integration_service)
):
    """Create a new notification template."""
    try:
        template = NotificationTemplate(
            id="",  # Will be set by service
            name=request.name,
            integration_type=request.integration_type,
            trigger_types=request.trigger_types,
            title_template=request.title_template,
            message_template=request.message_template,
            tenant_id=TenantId(str(tenant_id)),
            created_by=UserId(current_user.id),
            variables=request.variables
        )

        created_template = await integration_service.create_notification_template(template)

        return TemplateResponse(
            id=created_template.id,
            name=created_template.name,
            integration_type=created_template.integration_type,
            trigger_types=created_template.trigger_types,
            title_template=created_template.title_template,
            message_template=created_template.message_template,
            is_default=created_template.is_default,
            variables=created_template.variables,
            created_at=created_template.created_at,
            updated_at=created_template.updated_at
        )
    except ValidationError as e:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(e)
        )


@router.get("/tenants/{tenant_id}/templates", response_model=List[TemplateResponse])
async def list_notification_templates(
    tenant_id: UUID,
    integration_type: Optional[IntegrationType] = None,
    current_user: User = Depends(require_tenant_access),
    integration_service: IntegrationService = Depends(get_integration_service)
):
    """List notification templates for a tenant."""
    try:
        if integration_type:
            templates = await integration_service.get_templates_for_integration(
                integration_type=integration_type,
                tenant_id=TenantId(str(tenant_id))
            )
        else:
            # TODO: Implement get_all_templates method
            templates = []

        return [
            TemplateResponse(
                id=template.id,
                name=template.name,
                integration_type=template.integration_type,
                trigger_types=template.trigger_types,
                title_template=template.title_template,
                message_template=template.message_template,
                is_default=template.is_default,
                variables=template.variables,
                created_at=template.created_at,
                updated_at=template.updated_at
            )
            for template in templates
        ]
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to retrieve templates: {str(e)}"
        )


# Test Endpoints
@router.post("/tenants/{tenant_id}/integrations/{integration_id}/test")
async def test_integration(
    tenant_id: UUID,
    integration_id: str,
    current_user: User = Depends(require_tenant_access),
    integration_service: IntegrationService = Depends(get_integration_service)
):
    """Test an integration by sending a test notification."""
    try:
        test_payload = NotificationPayload(
            trigger_type=TriggerType.CUSTOM_ALERT,
            level=NotificationLevel.INFO,
            title="Pynomaly Integration Test",
            message="This is a test notification to verify your integration is working correctly.",
            tenant_id=TenantId(str(tenant_id)),
            user_id=UserId(current_user.id),
            data={"test": True}
        )

        results = await integration_service.send_notification(
            tenant_id=TenantId(str(tenant_id)),
            payload=test_payload,
            integration_types=None  # Will send to all integrations
        )

        integration_result = results.get(integration_id, False)

        return {
            "message": "Test notification sent",
            "success": integration_result,
            "integration_id": integration_id
        }
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Test failed: {str(e)}"
        )


# Quick Setup Endpoints
@router.post("/tenants/{tenant_id}/integrations/slack/quick-setup")
async def quick_setup_slack(
    tenant_id: UUID,
    webhook_url: HttpUrl,
    name: str = "Slack Integration",
    current_user: User = Depends(require_tenant_access),
    integration_service: IntegrationService = Depends(get_integration_service)
):
    """Quick setup for Slack integration."""
    try:
        config = IntegrationConfig(
            notification_levels=[NotificationLevel.WARNING, NotificationLevel.ERROR, NotificationLevel.CRITICAL],
            triggers=[TriggerType.ANOMALY_DETECTED, TriggerType.SYSTEM_ERROR, TriggerType.THRESHOLD_EXCEEDED]
        )

        credentials = {"webhook_url": str(webhook_url)}

        integration = await integration_service.create_integration(
            name=name,
            integration_type=IntegrationType.SLACK,
            tenant_id=TenantId(str(tenant_id)),
            user_id=UserId(current_user.id),
            config=config,
            credentials=credentials
        )

        return {
            "message": "Slack integration created successfully",
            "integration_id": integration.id,
            "status": integration.status.value
        }
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Failed to setup Slack integration: {str(e)}"
        )


@router.post("/tenants/{tenant_id}/integrations/pagerduty/quick-setup")
async def quick_setup_pagerduty(
    tenant_id: UUID,
    routing_key: str,
    name: str = "PagerDuty Integration",
    current_user: User = Depends(require_tenant_access),
    integration_service: IntegrationService = Depends(get_integration_service)
):
    """Quick setup for PagerDuty integration."""
    try:
        config = IntegrationConfig(
            notification_levels=[NotificationLevel.ERROR, NotificationLevel.CRITICAL],
            triggers=[TriggerType.ANOMALY_DETECTED, TriggerType.SYSTEM_ERROR, TriggerType.PERFORMANCE_DEGRADATION]
        )

        credentials = {"routing_key": routing_key}

        integration = await integration_service.create_integration(
            name=name,
            integration_type=IntegrationType.PAGERDUTY,
            tenant_id=TenantId(str(tenant_id)),
            user_id=UserId(current_user.id),
            config=config,
            credentials=credentials
        )

        return {
            "message": "PagerDuty integration created successfully",
            "integration_id": integration.id,
            "status": integration.status.value
        }
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Failed to setup PagerDuty integration: {str(e)}"
        )
