"""Advanced health monitoring API endpoints."""

from typing import Dict, List, Any, Optional
from datetime import datetime, timedelta

from fastapi import APIRouter, Depends, HTTPException, status, Query
from pydantic import BaseModel, Field

from ...domain.services.health_monitoring_service import (
    HealthMonitoringService, HealthStatus, AlertSeverity
)
from ...infrastructure.logging import get_logger

router = APIRouter()
logger = get_logger(__name__)

# Global health monitoring service instance
_health_service: Optional[HealthMonitoringService] = None


def get_health_service() -> HealthMonitoringService:
    """Get or create health monitoring service instance."""
    global _health_service
    if _health_service is None:
        _health_service = HealthMonitoringService(check_interval=30)
        logger.info("Created new health monitoring service instance")
    return _health_service


class HealthReportResponse(BaseModel):
    """Health report response model."""
    overall_status: str = Field(..., description="Overall system health status")
    overall_score: float = Field(..., description="Overall health score (0-100)")
    metrics: List[Dict[str, Any]] = Field(..., description="Individual health metrics")
    active_alerts: List[Dict[str, Any]] = Field(..., description="Active system alerts")
    performance_summary: Dict[str, Any] = Field(..., description="Performance metrics summary")
    uptime_seconds: float = Field(..., description="System uptime in seconds")
    timestamp: str = Field(..., description="Report generation timestamp")
    recommendations: List[str] = Field(..., description="Health improvement recommendations")


class AlertResponse(BaseModel):
    """Alert response model."""
    alert_id: str = Field(..., description="Unique alert identifier")
    severity: str = Field(..., description="Alert severity level")
    title: str = Field(..., description="Alert title")
    message: str = Field(..., description="Alert message")
    metric_name: str = Field(..., description="Related metric name")
    current_value: float = Field(..., description="Current metric value")
    threshold_value: float = Field(..., description="Threshold value that was exceeded")
    timestamp: str = Field(..., description="Alert timestamp")
    resolved: bool = Field(..., description="Whether alert is resolved")
    resolved_at: Optional[str] = Field(None, description="Resolution timestamp")


class PerformanceMetricsResponse(BaseModel):
    """Performance metrics response model."""
    response_time_stats: Dict[str, float] = Field(..., description="Response time statistics")
    error_stats: Dict[str, Any] = Field(..., description="Error statistics")
    throughput_stats: Dict[str, float] = Field(..., description="Throughput statistics")
    data_points: int = Field(..., description="Number of data points collected")


class ThresholdUpdateRequest(BaseModel):
    """Threshold update request model."""
    metric_name: str = Field(..., description="Metric name to update")
    warning_threshold: float = Field(..., description="Warning threshold value")
    critical_threshold: float = Field(..., description="Critical threshold value")


@router.get("/report", response_model=HealthReportResponse)
async def get_health_report(
    include_history: bool = Query(False, description="Include historical data"),
    health_service: HealthMonitoringService = Depends(get_health_service)
) -> HealthReportResponse:
    """Get comprehensive health report."""
    try:
        report = await health_service.get_health_report(include_history=include_history)
        
        return HealthReportResponse(
            overall_status=report.overall_status.value,
            overall_score=report.overall_score,
            metrics=[metric.to_dict() for metric in report.metrics],
            active_alerts=[alert.to_dict() for alert in report.active_alerts],
            performance_summary=report.performance_summary,
            uptime_seconds=report.uptime_seconds,
            timestamp=report.timestamp.isoformat(),
            recommendations=report.recommendations
        )
        
    except Exception as e:
        logger.error("Failed to get health report", error=str(e))
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to get health report: {str(e)}"
        )


@router.get("/status")
async def get_health_status(
    health_service: HealthMonitoringService = Depends(get_health_service)
) -> Dict[str, str]:
    """Get simple health status check."""
    try:
        report = await health_service.get_health_report()
        
        return {
            "status": report.overall_status.value,
            "score": f"{report.overall_score:.1f}",
            "timestamp": report.timestamp.isoformat(),
            "uptime": f"{report.uptime_seconds:.0f}s"
        }
        
    except Exception as e:
        logger.error("Health status check failed", error=str(e))
        return {
            "status": "unknown",
            "error": str(e),
            "timestamp": datetime.utcnow().isoformat()
        }


@router.get("/alerts", response_model=List[AlertResponse])
async def get_active_alerts(
    severity: Optional[str] = Query(None, description="Filter by severity (info, warning, error, critical)"),
    health_service: HealthMonitoringService = Depends(get_health_service)
) -> List[AlertResponse]:
    """Get active system alerts."""
    try:
        severity_filter = None
        if severity:
            try:
                severity_filter = AlertSeverity(severity.lower())
            except ValueError:
                raise HTTPException(
                    status_code=status.HTTP_400_BAD_REQUEST,
                    detail=f"Invalid severity: {severity}. Valid values: info, warning, error, critical"
                )
        
        alerts = await health_service.alert_manager.get_active_alerts(severity_filter)
        
        return [
            AlertResponse(
                alert_id=alert.alert_id,
                severity=alert.severity.value,
                title=alert.title,
                message=alert.message,
                metric_name=alert.metric_name,
                current_value=alert.current_value,
                threshold_value=alert.threshold_value,
                timestamp=alert.timestamp.isoformat(),
                resolved=alert.resolved,
                resolved_at=alert.resolved_at.isoformat() if alert.resolved_at else None
            )
            for alert in alerts
        ]
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error("Failed to get active alerts", error=str(e))
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to get active alerts: {str(e)}"
        )


@router.get("/alerts/history", response_model=List[AlertResponse])
async def get_alert_history(
    hours: int = Query(24, description="Hours of history to retrieve", ge=1, le=168),
    health_service: HealthMonitoringService = Depends(get_health_service)
) -> List[AlertResponse]:
    """Get alert history for specified time period."""
    try:
        alerts = await health_service.get_alert_history(hours)
        
        return [
            AlertResponse(
                alert_id=alert.alert_id,
                severity=alert.severity.value,
                title=alert.title,
                message=alert.message,
                metric_name=alert.metric_name,
                current_value=alert.current_value,
                threshold_value=alert.threshold_value,
                timestamp=alert.timestamp.isoformat(),
                resolved=alert.resolved,
                resolved_at=alert.resolved_at.isoformat() if alert.resolved_at else None
            )
            for alert in alerts
        ]
        
    except Exception as e:
        logger.error("Failed to get alert history", error=str(e))
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to get alert history: {str(e)}"
        )


@router.post("/alerts/{alert_id}/resolve")
async def resolve_alert(
    alert_id: str,
    health_service: HealthMonitoringService = Depends(get_health_service)
) -> Dict[str, str]:
    """Resolve an active alert."""
    try:
        success = await health_service.alert_manager.resolve_alert(alert_id)
        
        if success:
            return {
                "message": f"Alert {alert_id} resolved successfully",
                "timestamp": datetime.utcnow().isoformat()
            }
        else:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Alert {alert_id} not found or already resolved"
            )
            
    except HTTPException:
        raise
    except Exception as e:
        logger.error("Failed to resolve alert", alert_id=alert_id, error=str(e))
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to resolve alert: {str(e)}"
        )


@router.get("/metrics/performance", response_model=PerformanceMetricsResponse)
async def get_performance_metrics(
    health_service: HealthMonitoringService = Depends(get_health_service)
) -> PerformanceMetricsResponse:
    """Get performance metrics summary."""
    try:
        summary = await health_service.performance_tracker.get_performance_summary()
        
        return PerformanceMetricsResponse(
            response_time_stats=summary.get('response_time_stats', {}),
            error_stats=summary.get('error_stats', {}),
            throughput_stats=summary.get('throughput_stats', {}),
            data_points=summary.get('data_points', 0)
        )
        
    except Exception as e:
        logger.error("Failed to get performance metrics", error=str(e))
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to get performance metrics: {str(e)}"
        )


@router.post("/monitoring/start")
async def start_monitoring(
    health_service: HealthMonitoringService = Depends(get_health_service)
) -> Dict[str, str]:
    """Start health monitoring service."""
    try:
        await health_service.start_monitoring()
        
        return {
            "message": "Health monitoring started successfully",
            "check_interval": f"{health_service.check_interval}s",
            "timestamp": datetime.utcnow().isoformat()
        }
        
    except Exception as e:
        logger.error("Failed to start health monitoring", error=str(e))
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to start health monitoring: {str(e)}"
        )


@router.post("/monitoring/stop")
async def stop_monitoring(
    health_service: HealthMonitoringService = Depends(get_health_service)
) -> Dict[str, str]:
    """Stop health monitoring service."""
    try:
        await health_service.stop_monitoring()
        
        return {
            "message": "Health monitoring stopped successfully",
            "timestamp": datetime.utcnow().isoformat()
        }
        
    except Exception as e:
        logger.error("Failed to stop health monitoring", error=str(e))
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to stop health monitoring: {str(e)}"
        )


@router.post("/thresholds")
async def update_threshold(
    request: ThresholdUpdateRequest,
    health_service: HealthMonitoringService = Depends(get_health_service)
) -> Dict[str, str]:
    """Update health metric thresholds."""
    try:
        if request.warning_threshold >= request.critical_threshold:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Warning threshold must be less than critical threshold"
            )
        
        health_service.set_threshold(
            request.metric_name,
            request.warning_threshold,
            request.critical_threshold
        )
        
        return {
            "message": f"Thresholds updated for {request.metric_name}",
            "metric": request.metric_name,
            "warning_threshold": str(request.warning_threshold),
            "critical_threshold": str(request.critical_threshold),
            "timestamp": datetime.utcnow().isoformat()
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error("Failed to update thresholds", error=str(e))
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to update thresholds: {str(e)}"
        )


@router.get("/thresholds")
async def get_thresholds(
    health_service: HealthMonitoringService = Depends(get_health_service)
) -> Dict[str, Dict[str, float]]:
    """Get current health metric thresholds."""
    try:
        return health_service.thresholds
        
    except Exception as e:
        logger.error("Failed to get thresholds", error=str(e))
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to get thresholds: {str(e)}"
        )


@router.post("/metrics/record")
async def record_api_call(
    response_time_ms: float = Query(..., description="Response time in milliseconds"),
    success: bool = Query(True, description="Whether the call was successful"),
    health_service: HealthMonitoringService = Depends(get_health_service)
) -> Dict[str, str]:
    """Record API call metrics for monitoring."""
    try:
        await health_service.record_api_call(response_time_ms, success)
        
        return {
            "message": "API call metrics recorded",
            "response_time_ms": str(response_time_ms),
            "success": str(success),
            "timestamp": datetime.utcnow().isoformat()
        }
        
    except Exception as e:
        logger.error("Failed to record API call metrics", error=str(e))
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to record API call metrics: {str(e)}"
        )