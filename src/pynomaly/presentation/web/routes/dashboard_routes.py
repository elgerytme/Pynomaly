"""Dashboard routes implementation using pure models."""

import logging
from datetime import datetime
from typing import Dict, List, Optional

from fastapi import APIRouter, Depends, HTTPException, Request
from fastapi.responses import HTMLResponse

from ..models.dashboard_models import (
    DashboardStats,
    DashboardTab,
    MetricCard,
    NavigationItem,
    RecentActivity,
    SystemStatus,
)
from ..models.ui_models import DashboardData, DetectionResult, UserInfo

logger = logging.getLogger(__name__)


def create_dashboard_router() -> APIRouter:
    """Create dashboard router with pure model dependencies.
    
    Returns:
        Configured dashboard router
    """
    router = APIRouter()
    
    @router.get("/", response_class=HTMLResponse)
    async def dashboard_overview(request: Request):
        """Dashboard overview page."""
        # Create mock dashboard data using pure models
        dashboard_data = DashboardData(
            detector_count=5,
            dataset_count=3,
            result_count=25,
            recent_results=[],
            current_user="test_user",
            auth_enabled=False
        )
        
        # Mock dashboard stats
        stats = DashboardStats(
            total_detectors=5,
            total_datasets=3,
            total_experiments=10,
            active_detections=2,
            avg_anomaly_rate=0.15,
            system_health="healthy",
            last_updated=datetime.utcnow(),
            uptime_seconds=3600
        )
        
        # Mock recent activity
        activity = RecentActivity(
            detections_last_hour=5,
            new_experiments_today=2,
            trained_models_today=1,
            alerts_today=0,
            last_detection=datetime.utcnow(),
            last_training=datetime.utcnow()
        )
        
        return {
            "dashboard_data": dashboard_data,
            "stats": stats,
            "activity": activity
        }
    
    @router.get("/metrics")
    async def get_dashboard_metrics():
        """Get dashboard metrics."""
        metrics = [
            MetricCard(
                title="Total Detectors",
                value=5,
                unit="detectors",
                change_percent=10.0,
                trend="up",
                color="blue",
                icon="detector"
            ),
            MetricCard(
                title="Active Detections",
                value=2,
                unit="running",
                change_percent=0.0,
                trend="neutral",
                color="green",
                icon="activity"
            ),
            MetricCard(
                title="Anomaly Rate",
                value=15.2,
                unit="%",
                change_percent=-2.1,
                trend="down",
                color="orange",
                icon="warning"
            ),
            MetricCard(
                title="System Health",
                value="Healthy",
                unit="",
                change_percent=0.0,
                trend="neutral",
                color="green",
                icon="health"
            )
        ]
        
        return {"metrics": metrics}
    
    @router.get("/status")
    async def get_system_status():
        """Get system status."""
        status = SystemStatus(
            status="healthy",
            cpu_usage=45.2,
            memory_usage=62.1,
            disk_usage=38.7,
            active_connections=15,
            error_rate=0.01,
            response_time_ms=125.5,
            last_health_check=datetime.utcnow()
        )
        
        return {"status": status}
    
    @router.get("/navigation")
    async def get_navigation_items():
        """Get navigation items for dashboard."""
        items = [
            NavigationItem(
                id="overview",
                label="Overview",
                route="/dashboard",
                icon="dashboard",
                order=1
            ),
            NavigationItem(
                id="detectors",
                label="Detectors",
                route="/dashboard/detectors",
                icon="detector",
                order=2
            ),
            NavigationItem(
                id="datasets",
                label="Datasets",
                route="/dashboard/datasets",
                icon="data",
                order=3
            ),
            NavigationItem(
                id="experiments",
                label="Experiments",
                route="/dashboard/experiments",
                icon="experiment",
                order=4
            )
        ]
        
        return {"navigation": items}
    
    return router
