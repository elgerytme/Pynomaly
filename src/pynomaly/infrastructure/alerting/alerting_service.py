#!/usr/bin/env python3
"""
Alerting Service for Pynomaly Real-time Alerting System.
This module provides a FastAPI router for alert management and real-time alerting.
"""

import asyncio
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional

import structlog
from fastapi import APIRouter, Depends, HTTPException, BackgroundTasks, WebSocket, WebSocketDisconnect
from pydantic import BaseModel

from .alert_manager import (
    AlertManager,
    AlertInfo,
    AlertRuleCreate,
    AlertRuleUpdate,
    AlertSeverity,
    AlertStatus,
    NotificationChannel,
    get_alert_manager,
)
from .metric_collector import MetricCollector, get_metric_collector

# Configure structured logging
structlog.configure(
    processors=[
        structlog.stdlib.filter_by_level,
        structlog.stdlib.add_logger_name,
        structlog.stdlib.add_log_level,
        structlog.stdlib.PositionalArgumentsFormatter(),
        structlog.processors.TimeStamper(fmt="iso"),
        structlog.processors.StackInfoRenderer(),
        structlog.processors.format_exc_info,
        structlog.processors.UnicodeDecoder(),
        structlog.processors.JSONRenderer(),
    ],
    context_class=dict,
    logger_factory=structlog.stdlib.LoggerFactory(),
    wrapper_class=structlog.stdlib.BoundLogger,
    cache_logger_on_first_use=True,
)

logger = structlog.get_logger(__name__)

# Create FastAPI router
router = APIRouter(prefix="/alerting", tags=["alerting"])

# WebSocket connections for real-time alerts
websocket_connections: Dict[str, WebSocket] = {}


# Pydantic models for API
class AlertRuleResponse(BaseModel):
    """Alert rule response model."""
    
    id: str
    name: str
    description: Optional[str]
    metric_name: str
    condition: str
    threshold: str
    duration: int
    severity: AlertSeverity
    enabled: bool
    notification_channels: List[NotificationChannel]
    notification_template: Optional[str]
    cooldown_period: int
    created_at: datetime
    updated_at: datetime


class AlertSystemStatus(BaseModel):
    """Alert system status model."""
    
    status: str
    timestamp: datetime
    active_alerts: int
    total_rules: int
    enabled_rules: int
    notifications_sent_24h: int
    system_health: Dict[str, str]


class MetricSubmission(BaseModel):
    """Metric submission model."""
    
    metric_name: str
    value: float
    timestamp: Optional[datetime] = None
    metadata: Optional[Dict[str, str]] = None


# Alert Rule Management Endpoints
@router.post("/rules", response_model=AlertRuleResponse)
async def create_alert_rule(
    rule_data: AlertRuleCreate,
    alert_manager: AlertManager = Depends(get_alert_manager)
):
    """Create a new alert rule."""
    try:
        rule_id = await alert_manager.create_alert_rule(rule_data)
        
        # Get created rule details
        rules = await alert_manager.get_alert_rules()
        rule = next((r for r in rules if r.id == rule_id), None)
        
        if not rule:
            raise HTTPException(status_code=500, detail="Failed to retrieve created rule")
        
        return AlertRuleResponse(
            id=rule.id,
            name=rule.name,
            description=rule.description,
            metric_name=rule.metric_name,
            condition=rule.condition,
            threshold=rule.threshold,
            duration=rule.duration,
            severity=AlertSeverity(rule.severity),
            enabled=rule.enabled,
            notification_channels=[NotificationChannel(ch) for ch in rule.notification_channels.split(",")],
            notification_template=rule.notification_template,
            cooldown_period=rule.cooldown_period,
            created_at=rule.created_at,
            updated_at=rule.updated_at,
        )
        
    except Exception as e:
        logger.error(f"Failed to create alert rule: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/rules", response_model=List[AlertRuleResponse])
async def get_alert_rules(
    alert_manager: AlertManager = Depends(get_alert_manager)
):
    """Get all alert rules."""
    try:
        rules = await alert_manager.get_alert_rules()
        
        return [
            AlertRuleResponse(
                id=rule.id,
                name=rule.name,
                description=rule.description,
                metric_name=rule.metric_name,
                condition=rule.condition,
                threshold=rule.threshold,
                duration=rule.duration,
                severity=AlertSeverity(rule.severity),
                enabled=rule.enabled,
                notification_channels=[NotificationChannel(ch) for ch in rule.notification_channels.split(",")],
                notification_template=rule.notification_template,
                cooldown_period=rule.cooldown_period,
                created_at=rule.created_at,
                updated_at=rule.updated_at,
            )
            for rule in rules
        ]
        
    except Exception as e:
        logger.error(f"Failed to get alert rules: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.put("/rules/{rule_id}", response_model=AlertRuleResponse)
async def update_alert_rule(
    rule_id: str,
    rule_data: AlertRuleUpdate,
    alert_manager: AlertManager = Depends(get_alert_manager)
):
    """Update an alert rule."""
    try:
        success = await alert_manager.update_alert_rule(rule_id, rule_data)
        
        if not success:
            raise HTTPException(status_code=404, detail="Alert rule not found")
        
        # Get updated rule details
        rules = await alert_manager.get_alert_rules()
        rule = next((r for r in rules if r.id == rule_id), None)
        
        if not rule:
            raise HTTPException(status_code=500, detail="Failed to retrieve updated rule")
        
        return AlertRuleResponse(
            id=rule.id,
            name=rule.name,
            description=rule.description,
            metric_name=rule.metric_name,
            condition=rule.condition,
            threshold=rule.threshold,
            duration=rule.duration,
            severity=AlertSeverity(rule.severity),
            enabled=rule.enabled,
            notification_channels=[NotificationChannel(ch) for ch in rule.notification_channels.split(",")],
            notification_template=rule.notification_template,
            cooldown_period=rule.cooldown_period,
            created_at=rule.created_at,
            updated_at=rule.updated_at,
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to update alert rule: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.delete("/rules/{rule_id}")
async def delete_alert_rule(
    rule_id: str,
    alert_manager: AlertManager = Depends(get_alert_manager)
):
    """Delete an alert rule."""
    try:
        success = await alert_manager.delete_alert_rule(rule_id)
        
        if not success:
            raise HTTPException(status_code=404, detail="Alert rule not found")
        
        return {"message": "Alert rule deleted successfully"}
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to delete alert rule: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# Alert Management Endpoints
@router.get("/alerts", response_model=List[AlertInfo])
async def get_active_alerts(
    alert_manager: AlertManager = Depends(get_alert_manager)
):
    """Get all active alerts."""
    try:
        alerts = await alert_manager.get_active_alerts()
        return alerts
        
    except Exception as e:
        logger.error(f"Failed to get active alerts: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/alerts/{alert_id}/acknowledge")
async def acknowledge_alert(
    alert_id: str,
    alert_manager: AlertManager = Depends(get_alert_manager)
):
    """Acknowledge an alert."""
    try:
        success = await alert_manager.acknowledge_alert(alert_id)
        
        if not success:
            raise HTTPException(status_code=404, detail="Alert not found")
        
        return {"message": "Alert acknowledged successfully"}
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to acknowledge alert: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/alerts/{alert_id}/resolve")
async def resolve_alert(
    alert_id: str,
    alert_manager: AlertManager = Depends(get_alert_manager)
):
    """Resolve an alert."""
    try:
        success = await alert_manager.resolve_alert(alert_id)
        
        if not success:
            raise HTTPException(status_code=404, detail="Alert not found")
        
        return {"message": "Alert resolved successfully"}
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to resolve alert: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# Metric Submission Endpoints
@router.post("/metrics")
async def submit_metric(
    metric: MetricSubmission,
    background_tasks: BackgroundTasks,
    alert_manager: AlertManager = Depends(get_alert_manager)
):
    """Submit a metric for alert processing."""
    try:
        # Process metric in background
        background_tasks.add_task(
            alert_manager.process_metric,
            metric.metric_name,
            metric.value,
            metric.metadata or {}
        )
        
        return {"message": "Metric submitted successfully"}
        
    except Exception as e:
        logger.error(f"Failed to submit metric: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/metrics/batch")
async def submit_metrics_batch(
    metrics: List[MetricSubmission],
    background_tasks: BackgroundTasks,
    alert_manager: AlertManager = Depends(get_alert_manager)
):
    """Submit multiple metrics for alert processing."""
    try:
        # Process metrics in background
        for metric in metrics:
            background_tasks.add_task(
                alert_manager.process_metric,
                metric.metric_name,
                metric.value,
                metric.metadata or {}
            )
        
        return {"message": f"Submitted {len(metrics)} metrics successfully"}
        
    except Exception as e:
        logger.error(f"Failed to submit metrics batch: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# System Status Endpoints
@router.get("/status", response_model=AlertSystemStatus)
async def get_system_status(
    alert_manager: AlertManager = Depends(get_alert_manager)
):
    """Get alert system status."""
    try:
        # Get active alerts
        active_alerts = await alert_manager.get_active_alerts()
        
        # Get alert rules
        rules = await alert_manager.get_alert_rules()
        enabled_rules = [r for r in rules if r.enabled]
        
        # Calculate notifications sent in last 24 hours
        # This would require additional database queries in a real implementation
        notifications_sent_24h = 0
        
        return AlertSystemStatus(
            status="healthy",
            timestamp=datetime.utcnow(),
            active_alerts=len(active_alerts),
            total_rules=len(rules),
            enabled_rules=len(enabled_rules),
            notifications_sent_24h=notifications_sent_24h,
            system_health={
                "alert_manager": "healthy",
                "notification_channels": "healthy",
                "database": "healthy",
                "metrics_collection": "healthy"
            }
        )
        
    except Exception as e:
        logger.error(f"Failed to get system status: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/health")
async def health_check():
    """Health check endpoint."""
    return {
        "status": "healthy",
        "timestamp": datetime.utcnow().isoformat(),
        "service": "pynomaly-alerting"
    }


# WebSocket for Real-time Alerts
@router.websocket("/ws/{client_id}")
async def websocket_endpoint(
    websocket: WebSocket,
    client_id: str,
    alert_manager: AlertManager = Depends(get_alert_manager)
):
    """WebSocket endpoint for real-time alert notifications."""
    await websocket.accept()
    websocket_connections[client_id] = websocket
    
    logger.info(f"WebSocket client connected: {client_id}")
    
    try:
        # Send initial active alerts
        active_alerts = await alert_manager.get_active_alerts()
        await websocket.send_json({
            "type": "initial_alerts",
            "alerts": [alert.dict() for alert in active_alerts]
        })
        
        # Keep connection alive and listen for messages
        while True:
            try:
                message = await websocket.receive_text()
                
                # Handle client messages (e.g., acknowledge, resolve)
                if message == "ping":
                    await websocket.send_text("pong")
                    
            except WebSocketDisconnect:
                break
                
    except Exception as e:
        logger.error(f"WebSocket error for client {client_id}: {e}")
        
    finally:
        # Clean up connection
        if client_id in websocket_connections:
            del websocket_connections[client_id]
        logger.info(f"WebSocket client disconnected: {client_id}")


# Background task for alert streaming
async def stream_alerts_to_websockets(alert_manager: AlertManager):
    """Stream alerts to WebSocket clients."""
    while True:
        try:
            if websocket_connections:
                # Get current active alerts
                active_alerts = await alert_manager.get_active_alerts()
                
                # Send to all connected clients
                for client_id, websocket in list(websocket_connections.items()):
                    try:
                        await websocket.send_json({
                            "type": "alert_update",
                            "timestamp": datetime.utcnow().isoformat(),
                            "alerts": [alert.dict() for alert in active_alerts]
                        })
                    except Exception as e:
                        logger.error(f"Failed to send alert to client {client_id}: {e}")
                        # Remove disconnected client
                        if client_id in websocket_connections:
                            del websocket_connections[client_id]
            
            # Wait before next update
            await asyncio.sleep(30)  # Update every 30 seconds
            
        except Exception as e:
            logger.error(f"Error in alert streaming: {e}")
            await asyncio.sleep(60)  # Wait longer on error


# Startup and shutdown events
@router.on_event("startup")
async def startup_event():
    """Initialize alerting service on startup."""
    logger.info("Starting alerting service...")
    
    try:
        # Initialize alert manager
        alert_manager = get_alert_manager()
        await alert_manager.start()
        
        # Initialize metric collector
        metric_collector = get_metric_collector()
        await metric_collector.start()
        
        # Start background alert streaming
        asyncio.create_task(stream_alerts_to_websockets(alert_manager))
        
        logger.info("Alerting service started successfully")
        
    except Exception as e:
        logger.error(f"Failed to start alerting service: {e}")
        raise


@router.on_event("shutdown")
async def shutdown_event():
    """Cleanup on shutdown."""
    logger.info("Stopping alerting service...")
    
    try:
        # Stop alert manager
        alert_manager = get_alert_manager()
        await alert_manager.stop()
        
        # Stop metric collector
        metric_collector = get_metric_collector()
        await metric_collector.stop()
        
        # Close WebSocket connections
        for client_id, websocket in websocket_connections.items():
            try:
                await websocket.close()
            except:
                pass
        
        websocket_connections.clear()
        
        logger.info("Alerting service stopped successfully")
        
    except Exception as e:
        logger.error(f"Error during alerting service shutdown: {e}")


# Demo endpoints for testing
@router.post("/demo/trigger-alert")
async def trigger_demo_alert(
    metric_name: str = "demo.metric",
    value: float = 100.0,
    alert_manager: AlertManager = Depends(get_alert_manager)
):
    """Trigger a demo alert for testing."""
    try:
        await alert_manager.process_metric(metric_name, value, {"source": "demo"})
        return {"message": f"Demo alert triggered for {metric_name}={value}"}
        
    except Exception as e:
        logger.error(f"Failed to trigger demo alert: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/demo/create-rule")
async def create_demo_rule(
    alert_manager: AlertManager = Depends(get_alert_manager)
):
    """Create a demo alert rule for testing."""
    try:
        rule_data = AlertRuleCreate(
            name="Demo High CPU Alert",
            description="Alert when CPU usage exceeds 80%",
            metric_name="system.cpu.usage",
            condition=">",
            threshold="80.0",
            duration=60,
            severity=AlertSeverity.HIGH,
            enabled=True,
            notification_channels=[NotificationChannel.EMAIL, NotificationChannel.SLACK],
            cooldown_period=300,
        )
        
        rule_id = await alert_manager.create_alert_rule(rule_data)
        
        return {"message": f"Demo rule created with ID: {rule_id}"}
        
    except Exception as e:
        logger.error(f"Failed to create demo rule: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# Make components available for import
__all__ = [
    "router",
    "AlertSystemStatus",
    "MetricSubmission",
    "AlertRuleResponse",
    "websocket_connections",
]