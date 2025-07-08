"""
FastAPI router for reporting and business metrics dashboard.
"""

from datetime import datetime, timedelta
from typing import List, Optional, Dict, Any
from uuid import UUID

from fastapi import APIRouter, Depends, HTTPException, status, Query
from pydantic import BaseModel

from pynomaly.application.services.reporting_service import ReportingService
from pynomaly.domain.entities.reporting import ReportType, TimeGranularity, MetricType
from pynomaly.domain.entities.user import User
from pynomaly.shared.exceptions import (
    ValidationError,
    AuthorizationError,
    ReportNotFoundError,
)
from pynomaly.shared.types import TenantId, UserId

# Router setup
router = APIRouter(prefix="/api/reporting", tags=["Business Reporting"])


# Request/Response Models
class GenerateReportRequest(BaseModel):
    report_type: ReportType
    title: Optional[str] = None
    description: Optional[str] = None
    start_date: Optional[datetime] = None
    end_date: Optional[datetime] = None
    dataset_ids: List[UUID] = []
    detector_ids: List[UUID] = []


class ReportResponse(BaseModel):
    id: str
    title: str
    description: str
    report_type: ReportType
    status: str
    created_at: datetime
    completed_at: Optional[datetime]
    expires_at: Optional[datetime]
    sections: List[Dict[str, Any]]
    metadata: Dict[str, Any]


class DashboardResponse(BaseModel):
    id: str
    name: str
    description: str
    widgets: List[Dict[str, Any]]
    layout: Dict[str, Any]
    refresh_interval: int
    is_public: bool
    created_at: datetime
    updated_at: datetime
    last_accessed: Optional[datetime]


class CreateDashboardRequest(BaseModel):
    name: str
    description: str = ""
    widgets: List[Dict[str, Any]] = []
    is_public: bool = False


class UpdateDashboardRequest(BaseModel):
    widgets: List[Dict[str, Any]]
    layout: Optional[Dict[str, Any]] = None


class MetricResponse(BaseModel):
    id: str
    name: str
    description: str
    metric_type: MetricType
    current_value: Any
    formatted_value: str
    last_updated: datetime
    tags: Dict[str, str]


class CreateAlertRequest(BaseModel):
    name: str
    description: str = ""
    metric_id: str
    condition: str
    threshold: float
    notification_channels: List[str]


class AlertResponse(BaseModel):
    id: str
    name: str
    description: str
    metric_id: str
    condition: str
    threshold: float
    is_active: bool
    notification_channels: List[str]
    last_triggered: Optional[datetime]
    trigger_count: int
    created_at: datetime


# Dependencies
async def get_reporting_service() -> ReportingService:
    """Get reporting service instance."""
    # TODO: Implement proper dependency injection
    pass


async def get_current_user() -> User:
    """Get current authenticated user."""
    # TODO: Implement authentication
    pass


async def require_tenant_access(
    tenant_id: UUID, current_user: User = Depends(get_current_user)
):
    """Require access to specific tenant."""
    if not (
        current_user.is_super_admin()
        or current_user.has_role_in_tenant(
            TenantId(str(tenant_id)),
            ["viewer", "analyst", "data_scientist", "tenant_admin"],
        )
    ):
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN, detail="Access denied to tenant"
        )
    return current_user


# Report Generation Endpoints
@router.post(
    "/tenants/{tenant_id}/reports",
    response_model=ReportResponse,
    status_code=status.HTTP_201_CREATED,
)
async def generate_report(
    tenant_id: UUID,
    request: GenerateReportRequest,
    current_user: User = Depends(require_tenant_access),
    reporting_service: ReportingService = Depends(get_reporting_service),
):
    """Generate a new business report."""
    try:
        from pynomaly.domain.entities.reporting import ReportFilter

        # Create filters
        filters = ReportFilter(
            start_date=request.start_date or datetime.utcnow() - timedelta(days=30),
            end_date=request.end_date or datetime.utcnow(),
            tenant_ids=[TenantId(str(tenant_id))],
            dataset_ids=[str(id) for id in request.dataset_ids],
            detector_ids=[str(id) for id in request.detector_ids],
        )

        report = await reporting_service.generate_report(
            report_type=request.report_type,
            tenant_id=TenantId(str(tenant_id)),
            user_id=UserId(current_user.id),
            filters=filters,
            title=request.title,
            description=request.description,
        )

        return ReportResponse(
            id=report.id,
            title=report.title,
            description=report.description,
            report_type=report.report_type,
            status=report.status.value,
            created_at=report.created_at,
            completed_at=report.completed_at,
            expires_at=report.expires_at,
            sections=[
                {
                    "id": section.id,
                    "title": section.title,
                    "description": section.description,
                    "metrics": [
                        {
                            "id": metric.id,
                            "name": metric.name,
                            "current_value": metric.current_value,
                            "formatted_value": (
                                metric.latest_value.format_value()
                                if metric.latest_value
                                else "N/A"
                            ),
                            "type": metric.metric_type.value,
                        }
                        for metric in section.metrics
                    ],
                    "charts": section.charts,
                    "insights": section.insights,
                    "order": section.order,
                }
                for section in report.sections
            ],
            metadata=report.metadata,
        )
    except AuthorizationError as e:
        raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail=str(e))
    except ValidationError as e:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(e))


@router.get("/tenants/{tenant_id}/reports", response_model=List[ReportResponse])
async def list_reports(
    tenant_id: UUID,
    limit: int = Query(10, ge=1, le=100),
    offset: int = Query(0, ge=0),
    current_user: User = Depends(require_tenant_access),
    reporting_service: ReportingService = Depends(get_reporting_service),
):
    """List reports for a tenant."""
    # TODO: Implement repository method for listing reports
    return []


@router.get("/tenants/{tenant_id}/reports/{report_id}", response_model=ReportResponse)
async def get_report(
    tenant_id: UUID,
    report_id: str,
    current_user: User = Depends(require_tenant_access),
    reporting_service: ReportingService = Depends(get_reporting_service),
):
    """Get a specific report."""
    try:
        # TODO: Implement get_report method
        raise HTTPException(
            status_code=status.HTTP_501_NOT_IMPLEMENTED,
            detail="Report retrieval not implemented yet",
        )
    except ReportNotFoundError:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND, detail="Report not found"
        )


# Dashboard Management Endpoints
@router.post(
    "/tenants/{tenant_id}/dashboards",
    response_model=DashboardResponse,
    status_code=status.HTTP_201_CREATED,
)
async def create_dashboard(
    tenant_id: UUID,
    request: CreateDashboardRequest,
    current_user: User = Depends(require_tenant_access),
    reporting_service: ReportingService = Depends(get_reporting_service),
):
    """Create a new dashboard."""
    try:
        dashboard = await reporting_service.create_dashboard(
            name=request.name,
            tenant_id=TenantId(str(tenant_id)),
            user_id=UserId(current_user.id),
            description=request.description,
            widgets=request.widgets,
        )

        return DashboardResponse(
            id=dashboard.id,
            name=dashboard.name,
            description=dashboard.description,
            widgets=dashboard.widgets,
            layout=dashboard.layout,
            refresh_interval=dashboard.refresh_interval,
            is_public=dashboard.is_public,
            created_at=dashboard.created_at,
            updated_at=dashboard.updated_at,
            last_accessed=dashboard.last_accessed,
        )
    except ValidationError as e:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(e))


@router.get("/tenants/{tenant_id}/dashboards", response_model=List[DashboardResponse])
async def list_dashboards(
    tenant_id: UUID,
    current_user: User = Depends(require_tenant_access),
    reporting_service: ReportingService = Depends(get_reporting_service),
):
    """List dashboards for a tenant."""
    # TODO: Implement repository method for listing dashboards
    return []


@router.get(
    "/tenants/{tenant_id}/dashboards/{dashboard_id}", response_model=DashboardResponse
)
async def get_dashboard(
    tenant_id: UUID,
    dashboard_id: str,
    current_user: User = Depends(require_tenant_access),
    reporting_service: ReportingService = Depends(get_reporting_service),
):
    """Get a specific dashboard."""
    try:
        dashboard = await reporting_service.get_dashboard(
            dashboard_id, UserId(current_user.id)
        )

        return DashboardResponse(
            id=dashboard.id,
            name=dashboard.name,
            description=dashboard.description,
            widgets=dashboard.widgets,
            layout=dashboard.layout,
            refresh_interval=dashboard.refresh_interval,
            is_public=dashboard.is_public,
            created_at=dashboard.created_at,
            updated_at=dashboard.updated_at,
            last_accessed=dashboard.last_accessed,
        )
    except ReportNotFoundError:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND, detail="Dashboard not found"
        )
    except AuthorizationError as e:
        raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail=str(e))


@router.put(
    "/tenants/{tenant_id}/dashboards/{dashboard_id}", response_model=DashboardResponse
)
async def update_dashboard(
    tenant_id: UUID,
    dashboard_id: str,
    request: UpdateDashboardRequest,
    current_user: User = Depends(require_tenant_access),
    reporting_service: ReportingService = Depends(get_reporting_service),
):
    """Update dashboard widgets."""
    try:
        dashboard = await reporting_service.update_dashboard_widgets(
            dashboard_id=dashboard_id,
            user_id=UserId(current_user.id),
            widgets=request.widgets,
        )

        if request.layout:
            dashboard.layout = request.layout
            # TODO: Update dashboard layout in repository

        return DashboardResponse(
            id=dashboard.id,
            name=dashboard.name,
            description=dashboard.description,
            widgets=dashboard.widgets,
            layout=dashboard.layout,
            refresh_interval=dashboard.refresh_interval,
            is_public=dashboard.is_public,
            created_at=dashboard.created_at,
            updated_at=dashboard.updated_at,
            last_accessed=dashboard.last_accessed,
        )
    except (ReportNotFoundError, AuthorizationError) as e:
        raise HTTPException(
            status_code=(
                status.HTTP_404_NOT_FOUND
                if isinstance(e, ReportNotFoundError)
                else status.HTTP_403_FORBIDDEN
            ),
            detail=str(e),
        )


@router.post(
    "/tenants/{tenant_id}/dashboards/standard",
    response_model=DashboardResponse,
    status_code=status.HTTP_201_CREATED,
)
async def create_standard_dashboard(
    tenant_id: UUID,
    current_user: User = Depends(require_tenant_access),
    reporting_service: ReportingService = Depends(get_reporting_service),
):
    """Create a standard dashboard with common widgets."""
    try:
        dashboard = await reporting_service.create_standard_dashboard(
            tenant_id=TenantId(str(tenant_id)), user_id=UserId(current_user.id)
        )

        return DashboardResponse(
            id=dashboard.id,
            name=dashboard.name,
            description=dashboard.description,
            widgets=dashboard.widgets,
            layout=dashboard.layout,
            refresh_interval=dashboard.refresh_interval,
            is_public=dashboard.is_public,
            created_at=dashboard.created_at,
            updated_at=dashboard.updated_at,
            last_accessed=dashboard.last_accessed,
        )
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to create standard dashboard: {str(e)}",
        )


# Real-time Metrics Endpoints
@router.get("/tenants/{tenant_id}/metrics/realtime")
async def get_realtime_metrics(
    tenant_id: UUID,
    metric_ids: List[str] = Query(..., description="List of metric IDs to fetch"),
    current_user: User = Depends(require_tenant_access),
    reporting_service: ReportingService = Depends(get_reporting_service),
):
    """Get real-time metrics for dashboard widgets."""
    try:
        metrics = await reporting_service.get_real_time_metrics(
            tenant_id=TenantId(str(tenant_id)), metric_ids=metric_ids
        )

        return {"timestamp": datetime.utcnow().isoformat(), "metrics": metrics}
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to fetch real-time metrics: {str(e)}",
        )


@router.get("/tenants/{tenant_id}/metrics/{metric_id}/history")
async def get_metric_history(
    tenant_id: UUID,
    metric_id: str,
    start_date: datetime = Query(...),
    end_date: datetime = Query(...),
    granularity: TimeGranularity = Query(TimeGranularity.HOUR),
    current_user: User = Depends(require_tenant_access),
    reporting_service: ReportingService = Depends(get_reporting_service),
):
    """Get metric history for charting."""
    try:
        history = await reporting_service.get_metric_history(
            tenant_id=TenantId(str(tenant_id)),
            metric_id=metric_id,
            start_date=start_date,
            end_date=end_date,
            granularity=granularity,
        )

        return {
            "metric_id": metric_id,
            "start_date": start_date.isoformat(),
            "end_date": end_date.isoformat(),
            "granularity": granularity.value,
            "data": history,
        }
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to fetch metric history: {str(e)}",
        )


# Alert Management Endpoints
@router.post(
    "/tenants/{tenant_id}/alerts",
    response_model=AlertResponse,
    status_code=status.HTTP_201_CREATED,
)
async def create_alert(
    tenant_id: UUID,
    request: CreateAlertRequest,
    current_user: User = Depends(require_tenant_access),
    reporting_service: ReportingService = Depends(get_reporting_service),
):
    """Create a new metric alert."""
    try:
        alert = await reporting_service.create_alert(
            name=request.name,
            metric_id=request.metric_id,
            tenant_id=TenantId(str(tenant_id)),
            condition=request.condition,
            threshold=request.threshold,
            notification_channels=request.notification_channels,
            description=request.description,
        )

        return AlertResponse(
            id=alert.id,
            name=alert.name,
            description=alert.description,
            metric_id=alert.metric_id,
            condition=alert.condition,
            threshold=alert.threshold,
            is_active=alert.is_active,
            notification_channels=alert.notification_channels,
            last_triggered=alert.last_triggered,
            trigger_count=alert.trigger_count,
            created_at=alert.created_at,
        )
    except ValidationError as e:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(e))


@router.get("/tenants/{tenant_id}/alerts/check")
async def check_alerts(
    tenant_id: UUID,
    current_user: User = Depends(require_tenant_access),
    reporting_service: ReportingService = Depends(get_reporting_service),
):
    """Check all active alerts and return triggered ones."""
    try:
        triggered_alerts = await reporting_service.check_alerts(
            TenantId(str(tenant_id))
        )

        return {
            "timestamp": datetime.utcnow().isoformat(),
            "tenant_id": str(tenant_id),
            "triggered_alerts": triggered_alerts,
            "count": len(triggered_alerts),
        }
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to check alerts: {str(e)}",
        )


# Business Insights Endpoints
@router.get("/tenants/{tenant_id}/insights/summary")
async def get_business_insights_summary(
    tenant_id: UUID,
    days: int = Query(30, ge=1, le=365, description="Number of days to analyze"),
    current_user: User = Depends(require_tenant_access),
    reporting_service: ReportingService = Depends(get_reporting_service),
):
    """Get high-level business insights summary."""
    try:
        end_date = datetime.utcnow()
        start_date = end_date - timedelta(days=days)

        # TODO: Implement comprehensive insights generation
        return {
            "tenant_id": str(tenant_id),
            "period": {
                "start_date": start_date.isoformat(),
                "end_date": end_date.isoformat(),
                "days": days,
            },
            "insights": [
                {
                    "category": "detection_performance",
                    "title": "Detection Performance",
                    "summary": "Anomaly detection is performing well with 95% success rate",
                    "impact": "positive",
                    "recommendations": [
                        "Continue monitoring model performance",
                        "Consider expanding to additional data sources",
                    ],
                },
                {
                    "category": "cost_optimization",
                    "title": "Cost Optimization",
                    "summary": "Significant cost savings identified through early anomaly detection",
                    "impact": "positive",
                    "value": "$15,000",
                    "recommendations": [
                        "Implement automated response workflows",
                        "Expand monitoring coverage",
                    ],
                },
            ],
        }
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to generate business insights: {str(e)}",
        )
