"""Mobile API endpoints for data quality monitoring.

RESTful API endpoints optimized for mobile data quality monitoring,
providing efficient data transfer and mobile-specific features.
"""

from fastapi import APIRouter, HTTPException, Depends, Query, BackgroundTasks
from fastapi.security import HTTPBearer
from typing import Dict, Any, List, Optional
from datetime import datetime
import logging

from ..quality_monitoring.mobile_quality_service import (
    MobileQualityMonitoringService, AlertSeverity, AlertType
)
from ...core.shared.response_models import APIResponse, create_success_response, create_error_response

logger = logging.getLogger(__name__)

# Security
security = HTTPBearer()

# Router
mobile_quality_router = APIRouter(
    prefix="/api/mobile/quality",
    tags=["Mobile Quality Monitoring"]
)

# Service instance (would be injected via dependency injection)
mobile_quality_service = MobileQualityMonitoringService()


async def get_current_user(token: str = Depends(security)):
    """Get current user from token."""
    # Mock user for now - would implement real authentication
    return {"user_id": "mobile_user_123", "username": "mobile_user"}


@mobile_quality_router.get("/dashboard/{dashboard_id}")
async def get_mobile_dashboard(
    dashboard_id: str,
    current_user: dict = Depends(get_current_user)
) -> APIResponse:
    """Get mobile quality dashboard.
    
    Args:
        dashboard_id: Dashboard identifier
        current_user: Current authenticated user
        
    Returns:
        Mobile quality dashboard data
    """
    try:
        dashboard = await mobile_quality_service.get_mobile_dashboard(dashboard_id)
        
        if not dashboard:
            raise HTTPException(status_code=404, detail="Dashboard not found")
        
        return create_success_response(
            data=dashboard.to_mobile_dict(),
            message="Dashboard retrieved successfully"
        )
    
    except Exception as e:
        logger.error(f"Error retrieving mobile dashboard {dashboard_id}: {e}")
        return create_error_response(
            message="Failed to retrieve dashboard",
            details=str(e)
        )


@mobile_quality_router.post("/dashboard")
async def create_mobile_dashboard(
    favorite_datasets: List[str] = None,
    current_user: dict = Depends(get_current_user)
) -> APIResponse:
    """Create mobile quality dashboard.
    
    Args:
        favorite_datasets: List of favorite dataset IDs
        current_user: Current authenticated user
        
    Returns:
        Created mobile quality dashboard
    """
    try:
        dashboard = await mobile_quality_service.create_mobile_dashboard(
            user_id=current_user["user_id"],
            favorite_datasets=favorite_datasets or []
        )
        
        return create_success_response(
            data=dashboard.to_mobile_dict(),
            message="Dashboard created successfully"
        )
    
    except Exception as e:
        logger.error(f"Error creating mobile dashboard: {e}")
        return create_error_response(
            message="Failed to create dashboard",
            details=str(e)
        )


@mobile_quality_router.post("/dashboard/{dashboard_id}/refresh")
async def refresh_mobile_dashboard(
    dashboard_id: str,
    current_user: dict = Depends(get_current_user)
) -> APIResponse:
    """Refresh mobile quality dashboard.
    
    Args:
        dashboard_id: Dashboard identifier
        current_user: Current authenticated user
        
    Returns:
        Refreshed mobile quality dashboard
    """
    try:
        dashboard = await mobile_quality_service.refresh_mobile_dashboard(dashboard_id)
        
        if not dashboard:
            raise HTTPException(status_code=404, detail="Dashboard not found")
        
        return create_success_response(
            data=dashboard.to_mobile_dict(),
            message="Dashboard refreshed successfully"
        )
    
    except Exception as e:
        logger.error(f"Error refreshing mobile dashboard {dashboard_id}: {e}")
        return create_error_response(
            message="Failed to refresh dashboard",
            details=str(e)
        )


@mobile_quality_router.get("/alerts")
async def get_mobile_alerts(
    limit: int = Query(default=50, ge=1, le=200),
    severity: Optional[str] = Query(default=None),
    current_user: dict = Depends(get_current_user)
) -> APIResponse:
    """Get mobile quality alerts.
    
    Args:
        limit: Maximum number of alerts to return
        severity: Filter by severity (low, medium, high, critical, emergency)
        current_user: Current authenticated user
        
    Returns:
        List of mobile quality alerts
    """
    try:
        severity_filter = None
        if severity:
            try:
                severity_filter = AlertSeverity(severity.lower())
            except ValueError:
                raise HTTPException(status_code=400, detail="Invalid severity level")
        
        alerts = await mobile_quality_service.get_mobile_alerts(
            user_id=current_user["user_id"],
            limit=limit,
            severity_filter=severity_filter
        )
        
        return create_success_response(
            data=alerts,
            message=f"Retrieved {len(alerts)} mobile alerts"
        )
    
    except Exception as e:
        logger.error(f"Error retrieving mobile alerts: {e}")
        return create_error_response(
            message="Failed to retrieve alerts",
            details=str(e)
        )


@mobile_quality_router.post("/alerts")
async def create_mobile_alert(
    alert_data: Dict[str, Any],
    current_user: dict = Depends(get_current_user)
) -> APIResponse:
    """Create mobile quality alert.
    
    Args:
        alert_data: Alert data
        current_user: Current authenticated user
        
    Returns:
        Created mobile quality alert
    """
    try:
        # Validate required fields
        required_fields = ['dataset_id', 'metric_name', 'alert_type', 'severity', 'title', 'message']
        for field in required_fields:
            if field not in alert_data:
                raise HTTPException(status_code=400, detail=f"Missing required field: {field}")
        
        # Convert string enums
        alert_type = AlertType(alert_data['alert_type'].lower())
        severity = AlertSeverity(alert_data['severity'].lower())
        
        alert = await mobile_quality_service.create_mobile_alert(
            dataset_id=alert_data['dataset_id'],
            metric_name=alert_data['metric_name'],
            alert_type=alert_type,
            severity=severity,
            title=alert_data['title'],
            message=alert_data['message'],
            metadata=alert_data.get('metadata', {})
        )
        
        return create_success_response(
            data=alert.to_mobile_dict(),
            message="Mobile alert created successfully"
        )
    
    except Exception as e:
        logger.error(f"Error creating mobile alert: {e}")
        return create_error_response(
            message="Failed to create alert",
            details=str(e)
        )


@mobile_quality_router.post("/alerts/{alert_id}/resolve")
async def resolve_mobile_alert(
    alert_id: str,
    resolution_data: Dict[str, Any],
    current_user: dict = Depends(get_current_user)
) -> APIResponse:
    """Resolve mobile quality alert.
    
    Args:
        alert_id: Alert identifier
        resolution_data: Resolution data
        current_user: Current authenticated user
        
    Returns:
        Success response
    """
    try:
        success = await mobile_quality_service.resolve_mobile_alert(
            alert_id=alert_id,
            resolution_notes=resolution_data.get('resolution_notes', '')
        )
        
        if not success:
            raise HTTPException(status_code=404, detail="Alert not found")
        
        return create_success_response(
            data={"alert_id": alert_id, "resolved": True},
            message="Alert resolved successfully"
        )
    
    except Exception as e:
        logger.error(f"Error resolving mobile alert {alert_id}: {e}")
        return create_error_response(
            message="Failed to resolve alert",
            details=str(e)
        )


@mobile_quality_router.get("/incidents")
async def get_mobile_incidents(
    limit: int = Query(default=20, ge=1, le=100),
    status: Optional[str] = Query(default=None),
    current_user: dict = Depends(get_current_user)
) -> APIResponse:
    """Get mobile quality incidents.
    
    Args:
        limit: Maximum number of incidents to return
        status: Filter by status (open, in_progress, resolved, closed)
        current_user: Current authenticated user
        
    Returns:
        List of mobile quality incidents
    """
    try:
        incidents = await mobile_quality_service.get_mobile_incidents(
            user_id=current_user["user_id"],
            limit=limit,
            status_filter=status
        )
        
        return create_success_response(
            data=incidents,
            message=f"Retrieved {len(incidents)} mobile incidents"
        )
    
    except Exception as e:
        logger.error(f"Error retrieving mobile incidents: {e}")
        return create_error_response(
            message="Failed to retrieve incidents",
            details=str(e)
        )


@mobile_quality_router.post("/incidents")
async def create_mobile_incident(
    incident_data: Dict[str, Any],
    current_user: dict = Depends(get_current_user)
) -> APIResponse:
    """Create mobile quality incident.
    
    Args:
        incident_data: Incident data
        current_user: Current authenticated user
        
    Returns:
        Created mobile quality incident
    """
    try:
        # Validate required fields
        required_fields = ['title', 'description', 'severity', 'affected_datasets']
        for field in required_fields:
            if field not in incident_data:
                raise HTTPException(status_code=400, detail=f"Missing required field: {field}")
        
        # Convert string enum
        severity = AlertSeverity(incident_data['severity'].lower())
        
        incident = await mobile_quality_service.create_mobile_incident(
            title=incident_data['title'],
            description=incident_data['description'],
            severity=severity,
            affected_datasets=incident_data['affected_datasets'],
            affected_metrics=incident_data.get('affected_metrics', [])
        )
        
        return create_success_response(
            data=incident.to_mobile_dict(),
            message="Mobile incident created successfully"
        )
    
    except Exception as e:
        logger.error(f"Error creating mobile incident: {e}")
        return create_error_response(
            message="Failed to create incident",
            details=str(e)
        )


@mobile_quality_router.post("/incidents/{incident_id}/update")
async def update_mobile_incident(
    incident_id: str,
    update_data: Dict[str, Any],
    current_user: dict = Depends(get_current_user)
) -> APIResponse:
    """Update mobile quality incident.
    
    Args:
        incident_id: Incident identifier
        update_data: Update data
        current_user: Current authenticated user
        
    Returns:
        Success response
    """
    try:
        success = await mobile_quality_service.update_incident_status(
            incident_id=incident_id,
            status=update_data.get('status', 'in_progress'),
            completed_step=update_data.get('completed_step'),
            notes=update_data.get('notes', '')
        )
        
        if not success:
            raise HTTPException(status_code=404, detail="Incident not found")
        
        return create_success_response(
            data={"incident_id": incident_id, "updated": True},
            message="Incident updated successfully"
        )
    
    except Exception as e:
        logger.error(f"Error updating mobile incident {incident_id}: {e}")
        return create_error_response(
            message="Failed to update incident",
            details=str(e)
        )


@mobile_quality_router.get("/metrics/{dataset_id}")
async def get_mobile_quality_metrics(
    dataset_id: str,
    current_user: dict = Depends(get_current_user)
) -> APIResponse:
    """Get quality metrics for mobile display.
    
    Args:
        dataset_id: Dataset identifier
        current_user: Current authenticated user
        
    Returns:
        Mobile-optimized quality metrics
    """
    try:
        metrics = await mobile_quality_service.get_quality_metrics_for_mobile(dataset_id)
        
        return create_success_response(
            data=metrics,
            message=f"Retrieved mobile quality metrics for dataset {dataset_id}"
        )
    
    except Exception as e:
        logger.error(f"Error retrieving mobile quality metrics for {dataset_id}: {e}")
        return create_error_response(
            message="Failed to retrieve quality metrics",
            details=str(e)
        )


@mobile_quality_router.post("/sync/offline")
async def sync_offline_data(
    current_user: dict = Depends(get_current_user)
) -> APIResponse:
    """Sync data for offline mobile access.
    
    Args:
        current_user: Current authenticated user
        
    Returns:
        Offline data package
    """
    try:
        offline_data = await mobile_quality_service.sync_offline_data(
            user_id=current_user["user_id"]
        )
        
        return create_success_response(
            data=offline_data,
            message="Offline data synchronized successfully"
        )
    
    except Exception as e:
        logger.error(f"Error syncing offline data: {e}")
        return create_error_response(
            message="Failed to sync offline data",
            details=str(e)
        )


@mobile_quality_router.get("/summary")
async def get_mobile_summary(
    current_user: dict = Depends(get_current_user)
) -> APIResponse:
    """Get mobile quality summary.
    
    Args:
        current_user: Current authenticated user
        
    Returns:
        Mobile quality summary
    """
    try:
        summary = await mobile_quality_service.get_mobile_summary(
            user_id=current_user["user_id"]
        )
        
        return create_success_response(
            data=summary,
            message="Mobile quality summary retrieved successfully"
        )
    
    except Exception as e:
        logger.error(f"Error retrieving mobile summary: {e}")
        return create_error_response(
            message="Failed to retrieve mobile summary",
            details=str(e)
        )


@mobile_quality_router.post("/push/subscribe")
async def subscribe_to_push_notifications(
    subscription_data: Dict[str, Any],
    current_user: dict = Depends(get_current_user)
) -> APIResponse:
    """Subscribe to push notifications.
    
    Args:
        subscription_data: Push subscription data
        current_user: Current authenticated user
        
    Returns:
        Success response
    """
    try:
        # Store push subscription (would integrate with push service)
        logger.info(f"Push subscription registered for user {current_user['user_id']}")
        
        return create_success_response(
            data={"subscribed": True, "user_id": current_user["user_id"]},
            message="Push notifications subscription successful"
        )
    
    except Exception as e:
        logger.error(f"Error subscribing to push notifications: {e}")
        return create_error_response(
            message="Failed to subscribe to push notifications",
            details=str(e)
        )


@mobile_quality_router.post("/push/unsubscribe")
async def unsubscribe_from_push_notifications(
    current_user: dict = Depends(get_current_user)
) -> APIResponse:
    """Unsubscribe from push notifications.
    
    Args:
        current_user: Current authenticated user
        
    Returns:
        Success response
    """
    try:
        # Remove push subscription (would integrate with push service)
        logger.info(f"Push subscription removed for user {current_user['user_id']}")
        
        return create_success_response(
            data={"unsubscribed": True, "user_id": current_user["user_id"]},
            message="Push notifications unsubscription successful"
        )
    
    except Exception as e:
        logger.error(f"Error unsubscribing from push notifications: {e}")
        return create_error_response(
            message="Failed to unsubscribe from push notifications",
            details=str(e)
        )


@mobile_quality_router.get("/health")
async def mobile_api_health() -> APIResponse:
    """Health check for mobile API.
    
    Returns:
        Health status
    """
    return create_success_response(
        data={
            "status": "healthy",
            "timestamp": datetime.utcnow().isoformat(),
            "version": "1.0.0",
            "features": {
                "push_notifications": True,
                "offline_sync": True,
                "real_time_updates": True,
                "incident_management": True
            }
        },
        message="Mobile Quality API is healthy"
    )


# WebSocket endpoint for real-time updates
@mobile_quality_router.websocket("/ws/{user_id}")
async def mobile_quality_websocket(websocket, user_id: str):
    """WebSocket endpoint for real-time mobile quality updates.
    
    Args:
        websocket: WebSocket connection
        user_id: User identifier
    """
    await websocket.accept()
    
    try:
        while True:
            # Send periodic updates
            await asyncio.sleep(30)  # Update every 30 seconds
            
            # Get current summary
            summary = await mobile_quality_service.get_mobile_summary(user_id)
            
            # Send update
            await websocket.send_json({
                "type": "quality_update",
                "data": summary,
                "timestamp": datetime.utcnow().isoformat()
            })
    
    except Exception as e:
        logger.error(f"WebSocket error for user {user_id}: {e}")
    finally:
        await websocket.close()


# Background task endpoints
@mobile_quality_router.post("/tasks/generate-report")
async def generate_mobile_report(
    report_data: Dict[str, Any],
    background_tasks: BackgroundTasks,
    current_user: dict = Depends(get_current_user)
) -> APIResponse:
    """Generate mobile quality report in background.
    
    Args:
        report_data: Report configuration
        background_tasks: Background task manager
        current_user: Current authenticated user
        
    Returns:
        Task initiation response
    """
    try:
        task_id = f"mobile_report_{current_user['user_id']}_{datetime.utcnow().timestamp()}"
        
        # Add background task
        background_tasks.add_task(
            _generate_mobile_report,
            task_id=task_id,
            user_id=current_user["user_id"],
            report_config=report_data
        )
        
        return create_success_response(
            data={"task_id": task_id, "status": "initiated"},
            message="Mobile report generation started"
        )
    
    except Exception as e:
        logger.error(f"Error initiating mobile report generation: {e}")
        return create_error_response(
            message="Failed to initiate report generation",
            details=str(e)
        )


async def _generate_mobile_report(task_id: str, user_id: str, report_config: Dict[str, Any]):
    """Background task to generate mobile report."""
    try:
        logger.info(f"Generating mobile report {task_id} for user {user_id}")
        
        # Simulate report generation
        await asyncio.sleep(5)
        
        # Get data for report
        summary = await mobile_quality_service.get_mobile_summary(user_id)
        alerts = await mobile_quality_service.get_mobile_alerts(user_id, limit=100)
        incidents = await mobile_quality_service.get_mobile_incidents(user_id, limit=50)
        
        # Generate report (would create PDF/Excel file)
        report_data = {
            "report_id": task_id,
            "user_id": user_id,
            "generated_at": datetime.utcnow().isoformat(),
            "summary": summary,
            "alerts": alerts,
            "incidents": incidents,
            "config": report_config
        }
        
        logger.info(f"Mobile report {task_id} generated successfully")
        
        # Would save report and notify user
        
    except Exception as e:
        logger.error(f"Error generating mobile report {task_id}: {e}")


# Include router in main app
def include_mobile_quality_routes(app):
    """Include mobile quality routes in FastAPI app."""
    app.include_router(mobile_quality_router)