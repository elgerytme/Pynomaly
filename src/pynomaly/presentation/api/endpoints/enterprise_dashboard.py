"""Enterprise dashboard API endpoints for real-time monitoring and business intelligence.

This module provides REST API endpoints for accessing enterprise dashboard metrics,
real-time monitoring data, and business intelligence reports.
"""

import logging
from datetime import datetime
from typing import Any

from fastapi import APIRouter, Depends, HTTPException, Path, Query
from fastapi.responses import JSONResponse

from pynomaly.application.services.enterprise_dashboard_service import (
    AlertPriority,
    BusinessMetric,
    DashboardAlert,
    DashboardMetricType,
    EnterpriseDashboardService,
    ExecutiveSummary,
    OperationalMetric,
    get_dashboard_service,
)

# Optional authentication
try:
    from pynomaly.infrastructure.auth import get_current_user, require_permission

    AUTH_AVAILABLE = True
except ImportError:
    AUTH_AVAILABLE = False

    # Fallback dependencies
    def get_current_user():
        return {"username": "anonymous", "roles": ["user"]}

    def require_permission(permission: str):
        def dependency():
            return True

        return dependency


router = APIRouter(prefix="/api/v1/enterprise", tags=["Enterprise Dashboard"])
logger = logging.getLogger(__name__)


@router.get("/dashboard/summary")
async def get_executive_summary(
    current_user: dict = Depends(get_current_user),
    dashboard_service: EnterpriseDashboardService = Depends(get_dashboard_service),
) -> ExecutiveSummary:
    """Get executive summary for C-level reporting.

    Returns comprehensive business metrics, KPIs, and key insights
    for executive decision making.
    """
    try:
        summary = dashboard_service.get_executive_summary()

        logger.info(
            f"Executive summary requested by {current_user.get('username', 'unknown')}"
        )
        return summary

    except Exception as e:
        logger.error(f"Failed to get executive summary: {e}")
        raise HTTPException(
            status_code=500, detail="Failed to retrieve executive summary"
        )


@router.get("/dashboard/realtime")
async def get_realtime_dashboard_data(
    current_user: dict = Depends(get_current_user),
    dashboard_service: EnterpriseDashboardService = Depends(get_dashboard_service),
) -> dict[str, Any]:
    """Get real-time dashboard data for monitoring interfaces.

    Returns current operational metrics, business KPIs, active alerts,
    and system status for real-time dashboard display.
    """
    try:
        data = dashboard_service.get_real_time_dashboard_data()

        logger.debug(
            f"Real-time dashboard data requested by {current_user.get('username', 'unknown')}"
        )
        return data

    except Exception as e:
        logger.error(f"Failed to get real-time dashboard data: {e}")
        raise HTTPException(status_code=500, detail="Failed to retrieve real-time data")


@router.get("/dashboard/metrics/business")
async def get_business_metrics(
    current_user: dict = Depends(get_current_user),
    dashboard_service: EnterpriseDashboardService = Depends(get_dashboard_service),
) -> dict[str, BusinessMetric]:
    """Get business intelligence metrics.

    Returns detailed business metrics including cost savings,
    automation coverage, detection accuracy, and ROI analysis.
    """
    try:
        if not dashboard_service.enable_business_metrics:
            raise HTTPException(status_code=503, detail="Business metrics not enabled")

        metrics = dashboard_service.business_metrics

        logger.info(
            f"Business metrics requested by {current_user.get('username', 'unknown')}"
        )
        return metrics

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to get business metrics: {e}")
        raise HTTPException(
            status_code=500, detail="Failed to retrieve business metrics"
        )


@router.get("/dashboard/metrics/operational")
async def get_operational_metrics(
    current_user: dict = Depends(get_current_user),
    dashboard_service: EnterpriseDashboardService = Depends(get_dashboard_service),
) -> dict[str, OperationalMetric]:
    """Get operational monitoring metrics.

    Returns real-time operational health metrics including
    system performance, availability, and resource utilization.
    """
    try:
        if not dashboard_service.enable_operational_monitoring:
            raise HTTPException(
                status_code=503, detail="Operational monitoring not enabled"
            )

        metrics = dashboard_service.operational_metrics

        logger.debug(
            f"Operational metrics requested by {current_user.get('username', 'unknown')}"
        )
        return metrics

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to get operational metrics: {e}")
        raise HTTPException(
            status_code=500, detail="Failed to retrieve operational metrics"
        )


@router.get("/dashboard/alerts")
async def get_active_alerts(
    priority: AlertPriority | None = Query(
        None, description="Filter by alert priority"
    ),
    metric_type: DashboardMetricType | None = Query(
        None, description="Filter by metric type"
    ),
    limit: int = Query(
        50, ge=1, le=500, description="Maximum number of alerts to return"
    ),
    current_user: dict = Depends(get_current_user),
    dashboard_service: EnterpriseDashboardService = Depends(get_dashboard_service),
) -> list[DashboardAlert]:
    """Get active dashboard alerts.

    Returns current active alerts with optional filtering by
    priority and metric type.
    """
    try:
        alerts = list(dashboard_service.active_alerts.values())

        # Apply filters
        if priority:
            alerts = [alert for alert in alerts if alert.priority == priority]

        if metric_type:
            alerts = [alert for alert in alerts if alert.metric_type == metric_type]

        # Sort by priority and timestamp
        priority_order = {
            AlertPriority.CRITICAL: 0,
            AlertPriority.HIGH: 1,
            AlertPriority.MEDIUM: 2,
            AlertPriority.LOW: 3,
        }

        alerts.sort(
            key=lambda x: (priority_order.get(x.priority, 99), x.timestamp),
            reverse=True,
        )

        # Apply limit
        alerts = alerts[:limit]

        logger.debug(
            f"Active alerts requested by {current_user.get('username', 'unknown')}"
        )
        return alerts

    except Exception as e:
        logger.error(f"Failed to get active alerts: {e}")
        raise HTTPException(status_code=500, detail="Failed to retrieve active alerts")


@router.post("/dashboard/alerts/{alert_id}/acknowledge")
async def acknowledge_alert(
    alert_id: str = Path(..., description="Alert ID to acknowledge"),
    current_user: dict = Depends(get_current_user),
    dashboard_service: EnterpriseDashboardService = Depends(get_dashboard_service),
) -> dict[str, str]:
    """Acknowledge a dashboard alert.

    Marks an alert as acknowledged by the current user.
    """
    try:
        username = current_user.get("username", "unknown")
        success = dashboard_service.acknowledge_alert(alert_id, username)

        if success:
            logger.info(f"Alert {alert_id} acknowledged by {username}")
            return {"status": "success", "message": f"Alert {alert_id} acknowledged"}
        else:
            raise HTTPException(status_code=404, detail="Alert not found")

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to acknowledge alert {alert_id}: {e}")
        raise HTTPException(status_code=500, detail="Failed to acknowledge alert")


@router.post("/dashboard/alerts/{alert_id}/resolve")
async def resolve_alert(
    alert_id: str = Path(..., description="Alert ID to resolve"),
    current_user: dict = Depends(get_current_user),
    dashboard_service: EnterpriseDashboardService = Depends(get_dashboard_service),
) -> dict[str, str]:
    """Resolve a dashboard alert.

    Marks an alert as resolved by the current user.
    """
    try:
        username = current_user.get("username", "unknown")
        success = dashboard_service.resolve_alert(alert_id, username)

        if success:
            logger.info(f"Alert {alert_id} resolved by {username}")
            return {"status": "success", "message": f"Alert {alert_id} resolved"}
        else:
            raise HTTPException(status_code=404, detail="Alert not found")

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to resolve alert {alert_id}: {e}")
        raise HTTPException(status_code=500, detail="Failed to resolve alert")


@router.get("/dashboard/compliance")
async def get_compliance_report(
    current_user: dict = Depends(get_current_user),
    _: bool = (
        Depends(require_permission("compliance:read")) if AUTH_AVAILABLE else None
    ),
    dashboard_service: EnterpriseDashboardService = Depends(get_dashboard_service),
) -> dict[str, Any]:
    """Get compliance and governance report.

    Returns comprehensive compliance metrics, audit trail status,
    and regulatory adherence information.

    Requires compliance:read permission.
    """
    try:
        if not dashboard_service.enable_compliance_tracking:
            raise HTTPException(
                status_code=503, detail="Compliance tracking not enabled"
            )

        report = dashboard_service.get_compliance_report()

        logger.info(
            f"Compliance report requested by {current_user.get('username', 'unknown')}"
        )
        return report

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to get compliance report: {e}")
        raise HTTPException(
            status_code=500, detail="Failed to retrieve compliance report"
        )


@router.get("/dashboard/algorithms/performance")
async def get_algorithm_performance(
    algorithm: str | None = Query(None, description="Filter by algorithm name"),
    limit: int = Query(
        10, ge=1, le=50, description="Maximum number of algorithms to return"
    ),
    current_user: dict = Depends(get_current_user),
    dashboard_service: EnterpriseDashboardService = Depends(get_dashboard_service),
) -> dict[str, dict[str, Any]]:
    """Get algorithm performance metrics.

    Returns detailed performance metrics for anomaly detection algorithms
    including execution times, success rates, and trends.
    """
    try:
        performance_data = dashboard_service.algorithm_performance

        # Apply algorithm filter
        if algorithm:
            if algorithm in performance_data:
                performance_data = {algorithm: performance_data[algorithm]}
            else:
                performance_data = {}

        # Apply limit and sort by executions
        sorted_algorithms = sorted(
            performance_data.items(), key=lambda x: x[1]["executions"], reverse=True
        )[:limit]

        result = dict(sorted_algorithms)

        logger.debug(
            f"Algorithm performance requested by {current_user.get('username', 'unknown')}"
        )
        return result

    except Exception as e:
        logger.error(f"Failed to get algorithm performance: {e}")
        raise HTTPException(
            status_code=500, detail="Failed to retrieve algorithm performance"
        )


@router.get("/dashboard/cost-analysis")
async def get_cost_analysis(
    current_user: dict = Depends(get_current_user),
    _: bool = Depends(require_permission("finance:read")) if AUTH_AVAILABLE else None,
    dashboard_service: EnterpriseDashboardService = Depends(get_dashboard_service),
) -> dict[str, Any]:
    """Get cost analysis and ROI metrics.

    Returns detailed cost analysis including processing costs,
    automation savings, and ROI calculations.

    Requires finance:read permission.
    """
    try:
        cost_metrics = dashboard_service.cost_metrics.copy()

        # Calculate additional ROI metrics
        processing_cost = cost_metrics["processing_cost_usd"]
        savings = cost_metrics["savings_from_automation"]

        roi_percentage = ((savings - processing_cost) / max(processing_cost, 1)) * 100

        cost_analysis = {
            **cost_metrics,
            "roi_percentage": roi_percentage,
            "net_savings": savings - processing_cost,
            "cost_per_detection": processing_cost
            / max(dashboard_service.detection_stats["today"]["total"], 1),
            "analysis_timestamp": datetime.now().isoformat(),
        }

        logger.info(
            f"Cost analysis requested by {current_user.get('username', 'unknown')}"
        )
        return cost_analysis

    except Exception as e:
        logger.error(f"Failed to get cost analysis: {e}")
        raise HTTPException(status_code=500, detail="Failed to retrieve cost analysis")


@router.post("/dashboard/detection-event")
async def record_detection_event(
    detection_data: dict[str, Any],
    current_user: dict = Depends(get_current_user),
    dashboard_service: EnterpriseDashboardService = Depends(get_dashboard_service),
) -> dict[str, str]:
    """Record a detection event for dashboard metrics.

    Updates dashboard metrics based on a completed anomaly detection event.
    Used by autonomous detection services to update real-time metrics.
    """
    try:
        # Validate required fields
        required_fields = [
            "detection_id",
            "success",
            "execution_time",
            "algorithm_used",
            "anomalies_found",
            "dataset_size",
        ]
        missing_fields = [
            field for field in required_fields if field not in detection_data
        ]

        if missing_fields:
            raise HTTPException(
                status_code=400,
                detail=f"Missing required fields: {', '.join(missing_fields)}",
            )

        # Record the event
        dashboard_service.record_detection_event(
            detection_id=detection_data["detection_id"],
            success=detection_data["success"],
            execution_time=detection_data["execution_time"],
            algorithm_used=detection_data["algorithm_used"],
            anomalies_found=detection_data["anomalies_found"],
            dataset_size=detection_data["dataset_size"],
            cost_usd=detection_data.get("cost_usd", 0.0),
        )

        logger.debug(f"Detection event recorded: {detection_data['detection_id']}")
        return {"status": "success", "message": "Detection event recorded"}

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to record detection event: {e}")
        raise HTTPException(status_code=500, detail="Failed to record detection event")


@router.get("/dashboard/export")
async def export_dashboard_data(
    format: str = Query("json", pattern="^(json|csv)$", description="Export format"),
    current_user: dict = Depends(get_current_user),
    _: bool = Depends(require_permission("data:export")) if AUTH_AVAILABLE else None,
    dashboard_service: EnterpriseDashboardService = Depends(get_dashboard_service),
) -> JSONResponse:
    """Export dashboard data for analysis and reporting.

    Exports comprehensive dashboard data including metrics,
    alerts, performance data, and compliance reports.

    Requires data:export permission.
    """
    try:
        exported_data = dashboard_service.export_dashboard_data(format)

        # Set appropriate headers for file download
        if format == "json":
            media_type = "application/json"
            filename = (
                f"dashboard_export_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
            )
        else:
            media_type = "text/csv"
            filename = (
                f"dashboard_export_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
            )

        headers = {"Content-Disposition": f"attachment; filename={filename}"}

        logger.info(
            f"Dashboard data exported by {current_user.get('username', 'unknown')} in {format} format"
        )
        return JSONResponse(
            content=exported_data if format == "json" else {"data": exported_data},
            media_type=media_type,
            headers=headers,
        )

    except Exception as e:
        logger.error(f"Failed to export dashboard data: {e}")
        raise HTTPException(status_code=500, detail="Failed to export dashboard data")


@router.get("/dashboard/health")
async def get_dashboard_health() -> dict[str, Any]:
    """Get dashboard service health status.

    Returns health information about the dashboard service
    and its dependencies.
    """
    try:
        dashboard_service = get_dashboard_service()

        health_status = {
            "status": "healthy",
            "timestamp": datetime.now().isoformat(),
            "services": {
                "dashboard_service": "healthy",
                "business_metrics": (
                    "enabled"
                    if dashboard_service.enable_business_metrics
                    else "disabled"
                ),
                "operational_monitoring": (
                    "enabled"
                    if dashboard_service.enable_operational_monitoring
                    else "disabled"
                ),
                "compliance_tracking": (
                    "enabled"
                    if dashboard_service.enable_compliance_tracking
                    else "disabled"
                ),
            },
            "metrics": {
                "active_alerts": len(dashboard_service.active_alerts),
                "business_metrics_count": len(dashboard_service.business_metrics),
                "operational_metrics_count": len(dashboard_service.operational_metrics),
                "total_detections_today": dashboard_service.detection_stats["today"][
                    "total"
                ],
            },
        }

        return health_status

    except Exception as e:
        logger.error(f"Failed to get dashboard health: {e}")
        return {
            "status": "unhealthy",
            "timestamp": datetime.now().isoformat(),
            "error": str(e),
        }
