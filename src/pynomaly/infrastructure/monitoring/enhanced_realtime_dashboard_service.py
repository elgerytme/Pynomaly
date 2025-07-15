"""Enhanced real-time dashboard service with customization and advanced features."""

from __future__ import annotations

import asyncio
import logging
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional, Union
from uuid import UUID, uuid4

import psutil
from pydantic import BaseModel, Field

from pynomaly.domain.models.monitoring import (
    Dashboard,
    DashboardType,
    DashboardWidget,
)
from pynomaly.infrastructure.monitoring.dashboard_service import DashboardService


class DashboardCustomization(BaseModel):
    """Dashboard customization settings."""
    
    theme: str = "light"  # light, dark, auto
    layout: str = "grid"  # grid, flow, compact
    refresh_interval: int = Field(default=5, ge=1, le=300)  # seconds
    auto_refresh: bool = True
    show_legends: bool = True
    show_tooltips: bool = True
    animation_enabled: bool = True
    color_scheme: str = "default"  # default, colorblind, high_contrast
    density: str = "comfortable"  # compact, comfortable, spacious


class WidgetConfiguration(BaseModel):
    """Widget configuration for customizable dashboards."""
    
    widget_id: UUID = Field(default_factory=uuid4)
    title: str
    widget_type: str  # chart, gauge, table, text, alert_list, heatmap, map
    chart_type: str = "line"  # line, bar, pie, area, scatter, gauge, heatmap
    data_source: str  # metrics endpoint or query
    metrics: List[str] = Field(default_factory=list)
    time_range: str = "1h"
    refresh_interval: int = Field(default=30, ge=5)  # seconds
    position: Dict[str, int] = Field(default_factory=dict)  # x, y, width, height
    filters: Dict[str, Any] = Field(default_factory=dict)
    aggregation: str = "avg"  # avg, sum, min, max, count
    thresholds: List[Dict[str, Any]] = Field(default_factory=list)
    display_options: Dict[str, Any] = Field(default_factory=dict)
    

class UserPreferences(BaseModel):
    """User-specific dashboard preferences."""
    
    user_id: UUID
    default_time_range: str = "1h"
    preferred_theme: str = "light"
    auto_refresh_enabled: bool = True
    notification_preferences: Dict[str, bool] = Field(default_factory=dict)
    favorite_dashboards: List[UUID] = Field(default_factory=list)
    dashboard_customizations: Dict[UUID, DashboardCustomization] = Field(default_factory=dict)


class RealTimeMetricsCollector:
    """Enhanced metrics collector for real-time data."""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self._cache: Dict[str, Any] = {}
        self._last_update: Dict[str, datetime] = {}
        
    async def get_system_metrics(self) -> Dict[str, Any]:
        """Get real-time system metrics."""
        try:
            cpu_percent = psutil.cpu_percent(interval=0.1)
            memory = psutil.virtual_memory()
            disk = psutil.disk_usage('/')
            
            # Network statistics
            network = psutil.net_io_counters()
            
            metrics = {
                "cpu_usage_percent": cpu_percent,
                "memory_usage_percent": memory.percent,
                "memory_used_gb": memory.used / (1024**3),
                "memory_total_gb": memory.total / (1024**3),
                "disk_usage_percent": disk.percent,
                "disk_used_gb": disk.used / (1024**3),
                "disk_total_gb": disk.total / (1024**3),
                "network_bytes_sent": network.bytes_sent,
                "network_bytes_recv": network.bytes_recv,
                "timestamp": datetime.utcnow().isoformat()
            }
            
            # Calculate derived metrics
            metrics["system_health_score"] = self._calculate_health_score(metrics)
            
            return metrics
            
        except Exception as e:
            self.logger.error(f"Error collecting system metrics: {e}")
            return {"error": str(e), "timestamp": datetime.utcnow().isoformat()}
    
    async def get_application_metrics(self) -> Dict[str, Any]:
        """Get real-time application metrics."""
        try:
            # Simulate application metrics - in real implementation, 
            # these would come from your application monitoring
            import random
            
            metrics = {
                "http_requests_total": random.randint(100, 1000),
                "http_request_duration_seconds": random.uniform(0.1, 2.0),
                "http_errors_total": random.randint(0, 50),
                "active_connections": random.randint(10, 100),
                "cache_hit_rate": random.uniform(0.7, 0.99),
                "database_connections": random.randint(5, 50),
                "model_predictions_total": random.randint(50, 500),
                "model_accuracy": random.uniform(0.85, 0.98),
                "anomaly_detection_rate": random.uniform(0.01, 0.1),
                "model_inference_duration_seconds": random.uniform(0.01, 0.5),
                "timestamp": datetime.utcnow().isoformat()
            }
            
            return metrics
            
        except Exception as e:
            self.logger.error(f"Error collecting application metrics: {e}")
            return {"error": str(e), "timestamp": datetime.utcnow().isoformat()}
    
    async def get_business_metrics(self) -> Dict[str, Any]:
        """Get real-time business metrics."""
        try:
            import random
            from datetime import datetime, timedelta
            
            # Simulate business metrics
            base_date = datetime.utcnow() - timedelta(hours=24)
            
            metrics = {
                "anomalies_detected_total": random.randint(10, 100),
                "data_samples_processed": random.randint(1000, 10000),
                "system_uptime": (datetime.utcnow() - base_date).total_seconds(),
                "cost_per_detection": random.uniform(0.001, 0.01),
                "user_satisfaction_score": random.uniform(0.8, 1.0),
                "sla_compliance_rate": random.uniform(0.95, 1.0),
                "revenue_impact": random.uniform(1000, 10000),
                "timestamp": datetime.utcnow().isoformat()
            }
            
            return metrics
            
        except Exception as e:
            self.logger.error(f"Error collecting business metrics: {e}")
            return {"error": str(e), "timestamp": datetime.utcnow().isoformat()}
    
    def _calculate_health_score(self, metrics: Dict[str, Any]) -> float:
        """Calculate overall system health score."""
        try:
            cpu_score = max(0, 100 - metrics.get("cpu_usage_percent", 0))
            memory_score = max(0, 100 - metrics.get("memory_usage_percent", 0))
            disk_score = max(0, 100 - metrics.get("disk_usage_percent", 0))
            
            # Weighted average
            health_score = (cpu_score * 0.4 + memory_score * 0.4 + disk_score * 0.2)
            return round(health_score, 1)
            
        except Exception:
            return 50.0  # Default neutral score


class EnhancedRealtimeDashboardService(DashboardService):
    """Enhanced dashboard service with real-time features and customization."""
    
    def __init__(self, metrics_service, websocket_manager=None):
        super().__init__(metrics_service)
        self.websocket_manager = websocket_manager
        self.metrics_collector = RealTimeMetricsCollector()
        self.user_preferences: Dict[UUID, UserPreferences] = {}
        self.active_subscriptions: Dict[str, set] = {}
        self.custom_widgets: Dict[UUID, WidgetConfiguration] = {}
        
        # Background tasks
        self._background_tasks: set = set()
        self._metrics_broadcast_task = None
        
        self.logger.info("Enhanced real-time dashboard service initialized")
    
    async def start_realtime_services(self):
        """Start real-time background services."""
        if self.websocket_manager and not self._metrics_broadcast_task:
            self._metrics_broadcast_task = asyncio.create_task(
                self._metrics_broadcast_loop()
            )
            self._background_tasks.add(self._metrics_broadcast_task)
            self.logger.info("Real-time metrics broadcasting started")
    
    async def stop_realtime_services(self):
        """Stop real-time background services."""
        if self._metrics_broadcast_task:
            self._metrics_broadcast_task.cancel()
            
        # Cancel all background tasks
        for task in self._background_tasks:
            if not task.cancelled():
                task.cancel()
        
        # Wait for tasks to complete
        if self._background_tasks:
            await asyncio.gather(*self._background_tasks, return_exceptions=True)
        
        self._background_tasks.clear()
        self.logger.info("Real-time services stopped")
    
    async def _metrics_broadcast_loop(self):
        """Background loop for broadcasting metrics to WebSocket clients."""
        while True:
            try:
                # Collect all metrics
                system_metrics = await self.metrics_collector.get_system_metrics()
                app_metrics = await self.metrics_collector.get_application_metrics()
                business_metrics = await self.metrics_collector.get_business_metrics()
                
                # Broadcast to subscribed clients
                if self.websocket_manager:
                    await self.websocket_manager.broadcast_to_topic(
                        "system_metrics", system_metrics
                    )
                    await self.websocket_manager.broadcast_to_topic(
                        "application_metrics", app_metrics
                    )
                    await self.websocket_manager.broadcast_to_topic(
                        "business_metrics", business_metrics
                    )
                
                # Wait before next broadcast
                await asyncio.sleep(5)  # 5-second intervals
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                self.logger.error(f"Error in metrics broadcast loop: {e}")
                await asyncio.sleep(10)  # Wait longer on error
    
    async def create_custom_dashboard(
        self,
        name: str,
        description: str,
        widgets: List[WidgetConfiguration],
        owner_id: UUID,
        customization: DashboardCustomization,
        is_public: bool = False
    ) -> Dashboard:
        """Create a custom dashboard with user-defined widgets."""
        
        dashboard = await self.create_dashboard(
            name=name,
            description=description,
            dashboard_type=DashboardType.CUSTOM,
            owner_id=owner_id,
            is_public=is_public
        )
        
        # Apply customization
        dashboard.theme = customization.theme
        dashboard.auto_refresh = customization.auto_refresh
        dashboard.refresh_interval = customization.refresh_interval
        
        # Clear default widgets and add custom ones
        dashboard.widgets.clear()
        
        for widget_config in widgets:
            widget = DashboardWidget(
                widget_id=widget_config.widget_id,
                title=widget_config.title,
                widget_type=widget_config.widget_type,
                metrics=widget_config.metrics,
                time_range=widget_config.time_range,
                position=widget_config.position,
                chart_type=widget_config.chart_type,
                thresholds=widget_config.thresholds
            )
            
            dashboard.add_widget(widget)
            self.custom_widgets[widget_config.widget_id] = widget_config
        
        self.logger.info(f"Created custom dashboard: {name} with {len(widgets)} widgets")
        return dashboard
    
    async def get_realtime_dashboard_data(
        self,
        dashboard_id: UUID,
        user_id: UUID,
        include_realtime: bool = True
    ) -> Dict[str, Any]:
        """Get dashboard data with real-time metrics."""
        
        # Get base dashboard data
        dashboard_data = await self.get_dashboard_data(dashboard_id, user_id)
        
        if not dashboard_data:
            return None
        
        # Add real-time metrics if requested
        if include_realtime:
            dashboard_data["realtime_metrics"] = {
                "system": await self.metrics_collector.get_system_metrics(),
                "application": await self.metrics_collector.get_application_metrics(),
                "business": await self.metrics_collector.get_business_metrics()
            }
        
        # Add user preferences
        if user_id in self.user_preferences:
            preferences = self.user_preferences[user_id]
            dashboard_data["user_preferences"] = {
                "theme": preferences.preferred_theme,
                "auto_refresh": preferences.auto_refresh_enabled,
                "default_time_range": preferences.default_time_range
            }
        
        return dashboard_data
    
    async def subscribe_to_realtime_updates(
        self,
        dashboard_id: UUID,
        user_id: UUID,
        connection_id: str
    ) -> bool:
        """Subscribe a user to real-time dashboard updates."""
        
        if dashboard_id not in self.dashboards:
            return False
        
        dashboard = self.dashboards[dashboard_id]
        if not dashboard.can_view(user_id):
            return False
        
        # Add to subscription tracking
        subscription_key = f"{dashboard_id}:{user_id}"
        if subscription_key not in self.active_subscriptions:
            self.active_subscriptions[subscription_key] = set()
        
        self.active_subscriptions[subscription_key].add(connection_id)
        
        self.logger.info(f"User {user_id} subscribed to dashboard {dashboard_id} updates")
        return True
    
    async def unsubscribe_from_realtime_updates(
        self,
        dashboard_id: UUID,
        user_id: UUID,
        connection_id: str
    ) -> bool:
        """Unsubscribe a user from real-time dashboard updates."""
        
        subscription_key = f"{dashboard_id}:{user_id}"
        if subscription_key in self.active_subscriptions:
            self.active_subscriptions[subscription_key].discard(connection_id)
            
            # Remove empty subscriptions
            if not self.active_subscriptions[subscription_key]:
                del self.active_subscriptions[subscription_key]
            
            self.logger.info(f"User {user_id} unsubscribed from dashboard {dashboard_id} updates")
            return True
        
        return False
    
    async def update_user_preferences(
        self,
        user_id: UUID,
        preferences: UserPreferences
    ) -> bool:
        """Update user dashboard preferences."""
        
        self.user_preferences[user_id] = preferences
        self.logger.info(f"Updated preferences for user {user_id}")
        return True
    
    async def get_user_preferences(self, user_id: UUID) -> Optional[UserPreferences]:
        """Get user dashboard preferences."""
        return self.user_preferences.get(user_id)
    
    async def get_dashboard_performance_metrics(
        self,
        dashboard_id: UUID,
        time_range: str = "24h"
    ) -> Dict[str, Any]:
        """Get performance metrics for a specific dashboard."""
        
        # Simulate performance metrics - in real implementation,
        # these would come from monitoring the dashboard usage
        import random
        
        metrics = {
            "dashboard_id": str(dashboard_id),
            "time_range": time_range,
            "page_views": random.randint(100, 1000),
            "unique_users": random.randint(10, 100),
            "average_session_duration": random.uniform(300, 1800),  # seconds
            "bounce_rate": random.uniform(0.1, 0.4),
            "load_time_avg": random.uniform(0.5, 3.0),  # seconds
            "load_time_p95": random.uniform(2.0, 5.0),  # seconds
            "error_rate": random.uniform(0.0, 0.05),
            "user_satisfaction": random.uniform(0.7, 1.0),
            "most_viewed_widgets": [
                {"widget_id": str(uuid4()), "title": "System Health", "views": random.randint(50, 200)},
                {"widget_id": str(uuid4()), "title": "CPU Usage", "views": random.randint(30, 150)},
                {"widget_id": str(uuid4()), "title": "Response Time", "views": random.randint(40, 180)}
            ],
            "timestamp": datetime.utcnow().isoformat()
        }
        
        return metrics
    
    async def export_dashboard_configuration(
        self,
        dashboard_id: UUID,
        format: str = "json"
    ) -> Dict[str, Any]:
        """Export dashboard configuration for backup or sharing."""
        
        if dashboard_id not in self.dashboards:
            return None
        
        dashboard = self.dashboards[dashboard_id]
        
        config = {
            "dashboard": {
                "name": dashboard.name,
                "description": dashboard.description,
                "type": dashboard.dashboard_type.value,
                "theme": dashboard.theme,
                "auto_refresh": dashboard.auto_refresh,
                "refresh_interval": dashboard.refresh_interval
            },
            "widgets": [],
            "export_metadata": {
                "exported_at": datetime.utcnow().isoformat(),
                "version": "1.0",
                "format": format
            }
        }
        
        for widget in dashboard.widgets:
            widget_config = {
                "title": widget.title,
                "type": widget.widget_type,
                "chart_type": widget.chart_type,
                "metrics": widget.metrics,
                "time_range": widget.time_range,
                "position": widget.position,
                "thresholds": widget.thresholds
            }
            
            # Add custom configuration if available
            if widget.widget_id in self.custom_widgets:
                custom_config = self.custom_widgets[widget.widget_id]
                widget_config.update({
                    "data_source": custom_config.data_source,
                    "filters": custom_config.filters,
                    "aggregation": custom_config.aggregation,
                    "display_options": custom_config.display_options
                })
            
            config["widgets"].append(widget_config)
        
        return config
    
    async def import_dashboard_configuration(
        self,
        config: Dict[str, Any],
        owner_id: UUID,
        override_name: Optional[str] = None
    ) -> Optional[Dashboard]:
        """Import dashboard configuration from exported data."""
        
        try:
            dashboard_config = config.get("dashboard", {})
            widgets_config = config.get("widgets", [])
            
            # Create dashboard
            name = override_name or dashboard_config.get("name", "Imported Dashboard")
            description = dashboard_config.get("description", "Imported from configuration")
            
            dashboard = await self.create_dashboard(
                name=name,
                description=description,
                dashboard_type=DashboardType.CUSTOM,
                owner_id=owner_id
            )
            
            # Apply dashboard settings
            dashboard.theme = dashboard_config.get("theme", "light")
            dashboard.auto_refresh = dashboard_config.get("auto_refresh", True)
            dashboard.refresh_interval = dashboard_config.get("refresh_interval", 30)
            
            # Clear default widgets
            dashboard.widgets.clear()
            
            # Add imported widgets
            for widget_config in widgets_config:
                widget = DashboardWidget(
                    widget_id=uuid4(),
                    title=widget_config.get("title", "Untitled Widget"),
                    widget_type=widget_config.get("type", "chart"),
                    metrics=widget_config.get("metrics", []),
                    time_range=widget_config.get("time_range", "1h"),
                    position=widget_config.get("position", {}),
                    chart_type=widget_config.get("chart_type", "line"),
                    thresholds=widget_config.get("thresholds", [])
                )
                
                dashboard.add_widget(widget)
            
            self.logger.info(f"Imported dashboard: {name} with {len(widgets_config)} widgets")
            return dashboard
            
        except Exception as e:
            self.logger.error(f"Error importing dashboard configuration: {e}")
            return None
    
    async def get_dashboard_insights(
        self,
        dashboard_id: UUID,
        time_range: str = "24h"
    ) -> Dict[str, Any]:
        """Get AI-powered insights and recommendations for dashboard."""
        
        # Get dashboard data
        dashboard_data = await self.get_realtime_dashboard_data(
            dashboard_id, uuid4(), include_realtime=True
        )
        
        if not dashboard_data:
            return {"error": "Dashboard not found"}
        
        # Analyze metrics and generate insights
        insights = {
            "dashboard_id": str(dashboard_id),
            "analysis_time": datetime.utcnow().isoformat(),
            "insights": [],
            "recommendations": [],
            "anomalies_detected": [],
            "performance_summary": {}
        }
        
        # Get real-time metrics for analysis
        realtime_metrics = dashboard_data.get("realtime_metrics", {})
        system_metrics = realtime_metrics.get("system", {})
        app_metrics = realtime_metrics.get("application", {})
        
        # Generate insights based on metrics
        if system_metrics:
            cpu_usage = system_metrics.get("cpu_usage_percent", 0)
            memory_usage = system_metrics.get("memory_usage_percent", 0)
            health_score = system_metrics.get("system_health_score", 100)
            
            if cpu_usage > 80:
                insights["insights"].append({
                    "type": "warning",
                    "title": "High CPU Usage Detected",
                    "description": f"CPU usage is at {cpu_usage:.1f}%, which may impact performance",
                    "severity": "medium",
                    "timestamp": datetime.utcnow().isoformat()
                })
                
                insights["recommendations"].append({
                    "type": "optimization",
                    "title": "Consider CPU Optimization",
                    "description": "Review resource-intensive processes and consider scaling",
                    "priority": "medium"
                })
            
            if memory_usage > 85:
                insights["insights"].append({
                    "type": "critical",
                    "title": "High Memory Usage",
                    "description": f"Memory usage is at {memory_usage:.1f}%, approaching critical levels",
                    "severity": "high",
                    "timestamp": datetime.utcnow().isoformat()
                })
            
            if health_score < 70:
                insights["anomalies_detected"].append({
                    "type": "system_health",
                    "title": "System Health Below Normal",
                    "description": f"Overall system health score is {health_score}, below recommended threshold",
                    "impact": "high",
                    "timestamp": datetime.utcnow().isoformat()
                })
        
        # Application metrics insights
        if app_metrics:
            error_rate = app_metrics.get("http_errors_total", 0) / max(app_metrics.get("http_requests_total", 1), 1)
            response_time = app_metrics.get("http_request_duration_seconds", 0)
            
            if error_rate > 0.05:  # 5% error rate
                insights["insights"].append({
                    "type": "error",
                    "title": "Elevated Error Rate",
                    "description": f"HTTP error rate is {error_rate:.2%}, above normal threshold",
                    "severity": "high",
                    "timestamp": datetime.utcnow().isoformat()
                })
            
            if response_time > 1.5:  # 1.5 second response time
                insights["recommendations"].append({
                    "type": "performance",
                    "title": "Optimize Response Times",
                    "description": f"Average response time is {response_time:.2f}s, consider optimization",
                    "priority": "medium"
                })
        
        # Performance summary
        insights["performance_summary"] = {
            "overall_status": "healthy" if health_score > 80 else "degraded" if health_score > 60 else "critical",
            "key_metrics": {
                "system_health_score": system_metrics.get("system_health_score", 0),
                "cpu_usage": system_metrics.get("cpu_usage_percent", 0),
                "memory_usage": system_metrics.get("memory_usage_percent", 0),
                "response_time": app_metrics.get("http_request_duration_seconds", 0)
            },
            "trend_analysis": "stable",  # In real implementation, this would analyze historical data
            "forecast": "normal"  # In real implementation, this would use ML for forecasting
        }
        
        return insights


# Convenience function for creating enhanced dashboard service
def create_enhanced_dashboard_service(
    metrics_service,
    websocket_manager=None
) -> EnhancedRealtimeDashboardService:
    """Create and configure enhanced dashboard service."""
    service = EnhancedRealtimeDashboardService(metrics_service, websocket_manager)
    return service
