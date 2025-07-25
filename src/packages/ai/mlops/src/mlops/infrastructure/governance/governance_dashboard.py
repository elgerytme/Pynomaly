"""
Governance Dashboard and Visualization

Comprehensive dashboard for ML governance, compliance monitoring,
and risk visualization with real-time insights and executive reporting.
"""

import asyncio
import json
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Union
from dataclasses import dataclass, field
from enum import Enum
import uuid

import pandas as pd
import numpy as np
from pydantic import BaseModel, Field
import structlog
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots

from .ml_governance_framework import (
    MLGovernanceFramework, ComplianceFramework, GovernanceRisk,
    AuditEventType
)
from .compliance_automation import ComplianceAutomationEngine
from .regulatory_compliance import RegulatoryComplianceManager


class DashboardMetric(Enum):
    """Dashboard metric types."""
    COMPLIANCE_SCORE = "compliance_score"
    RISK_LEVEL = "risk_level"
    POLICY_VIOLATIONS = "policy_violations"
    AUDIT_EVENTS = "audit_events"
    MODEL_HEALTH = "model_health"
    DATA_QUALITY = "data_quality"
    APPROVAL_PENDING = "approval_pending"
    CERTIFICATION_STATUS = "certification_status"


class VisualizationType(Enum):
    """Types of visualizations."""
    LINE_CHART = "line_chart"
    BAR_CHART = "bar_chart"
    PIE_CHART = "pie_chart"
    HEATMAP = "heatmap"
    GAUGE = "gauge"
    SCATTER_PLOT = "scatter_plot"
    TREEMAP = "treemap"
    SANKEY = "sankey"


@dataclass
class DashboardWidget:
    """Dashboard widget configuration."""
    widget_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    title: str = ""
    description: str = ""
    
    # Widget configuration
    metric_type: DashboardMetric = DashboardMetric.COMPLIANCE_SCORE
    visualization_type: VisualizationType = VisualizationType.LINE_CHART
    
    # Data configuration
    data_source: str = ""
    filters: Dict[str, Any] = field(default_factory=dict)
    time_range: str = "7d"  # 1h, 1d, 7d, 30d, 90d
    
    # Display configuration
    position: Dict[str, int] = field(default_factory=dict)  # x, y, width, height
    refresh_interval: int = 300  # seconds
    alert_thresholds: Dict[str, float] = field(default_factory=dict)
    
    # Access control
    required_roles: List[str] = field(default_factory=list)
    visibility: str = "public"  # public, private, role-based


@dataclass
class DashboardAlert:
    """Dashboard alert configuration."""
    alert_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    widget_id: str = ""
    
    # Alert conditions
    metric: str = ""
    threshold_type: str = "greater_than"  # greater_than, less_than, equals, not_equals
    threshold_value: float = 0.0
    
    # Alert configuration
    severity: GovernanceRisk = GovernanceRisk.MEDIUM
    enabled: bool = True
    cooldown_minutes: int = 30
    
    # Notification
    notification_channels: List[str] = field(default_factory=list)
    escalation_rules: Dict[str, Any] = field(default_factory=dict)
    
    # Status
    last_triggered: Optional[datetime] = None
    trigger_count: int = 0


class GovernanceDashboard:
    """Comprehensive governance dashboard system."""
    
    def __init__(self,
                 governance_framework: MLGovernanceFramework,
                 compliance_automation: ComplianceAutomationEngine,
                 regulatory_manager: RegulatoryComplianceManager,
                 config: Dict[str, Any] = None):
        
        self.governance = governance_framework
        self.compliance_automation = compliance_automation
        self.regulatory_manager = regulatory_manager
        self.config = config or {}
        self.logger = structlog.get_logger(__name__)
        
        # Dashboard components
        self.widgets: Dict[str, DashboardWidget] = {}
        self.alerts: Dict[str, DashboardAlert] = {}
        self.data_aggregator = DashboardDataAggregator(governance_framework)
        self.visualization_engine = VisualizationEngine()
        
        # Real-time data cache
        self.data_cache: Dict[str, Any] = {}
        self.cache_timestamps: Dict[str, datetime] = {}
        
        # Background tasks
        self.background_tasks: List[asyncio.Task] = []
        self.is_running = False
        
        # Initialize default dashboard
        self._initialize_default_dashboard()
    
    def _initialize_default_dashboard(self) -> None:
        """Initialize default dashboard widgets."""
        
        default_widgets = [
            {
                "title": "Overall Compliance Score",
                "metric_type": DashboardMetric.COMPLIANCE_SCORE,
                "visualization_type": VisualizationType.GAUGE,
                "position": {"x": 0, "y": 0, "width": 4, "height": 3}
            },
            {
                "title": "Risk Distribution",
                "metric_type": DashboardMetric.RISK_LEVEL,
                "visualization_type": VisualizationType.PIE_CHART,
                "position": {"x": 4, "y": 0, "width": 4, "height": 3}
            },
            {
                "title": "Policy Violations Trend",
                "metric_type": DashboardMetric.POLICY_VIOLATIONS,
                "visualization_type": VisualizationType.LINE_CHART,
                "position": {"x": 8, "y": 0, "width": 4, "height": 3}
            },
            {
                "title": "Audit Events Timeline",
                "metric_type": DashboardMetric.AUDIT_EVENTS,
                "visualization_type": VisualizationType.BAR_CHART,
                "position": {"x": 0, "y": 3, "width": 8, "height": 4}
            },
            {
                "title": "Certification Status",
                "metric_type": DashboardMetric.CERTIFICATION_STATUS,
                "visualization_type": VisualizationType.TREEMAP,
                "position": {"x": 8, "y": 3, "width": 4, "height": 4}
            }
        ]
        
        for widget_config in default_widgets:
            self.add_widget(**widget_config)
    
    async def add_widget(self,
                        title: str,
                        metric_type: DashboardMetric,
                        visualization_type: VisualizationType,
                        **kwargs) -> str:
        """Add a new dashboard widget."""
        
        widget = DashboardWidget(
            title=title,
            metric_type=metric_type,
            visualization_type=visualization_type,
            **kwargs
        )
        
        self.widgets[widget.widget_id] = widget
        
        self.logger.info(
            "Dashboard widget added",
            widget_id=widget.widget_id,
            title=title,
            metric_type=metric_type.value,
            visualization_type=visualization_type.value
        )
        
        return widget.widget_id
    
    async def create_alert(self,
                          widget_id: str,
                          metric: str,
                          threshold_type: str,
                          threshold_value: float,
                          **kwargs) -> str:
        """Create a dashboard alert."""
        
        if widget_id not in self.widgets:
            raise ValueError(f"Widget {widget_id} not found")
        
        alert = DashboardAlert(
            widget_id=widget_id,
            metric=metric,
            threshold_type=threshold_type,
            threshold_value=threshold_value,
            **kwargs
        )
        
        self.alerts[alert.alert_id] = alert
        
        self.logger.info(
            "Dashboard alert created",
            alert_id=alert.alert_id,
            widget_id=widget_id,
            metric=metric,
            threshold=f"{threshold_type} {threshold_value}"
        )
        
        return alert.alert_id
    
    async def get_dashboard_data(self,
                               dashboard_id: str = "default",
                               user_roles: List[str] = None) -> Dict[str, Any]:
        """Get complete dashboard data for rendering."""
        
        user_roles = user_roles or []
        dashboard_data = {
            "dashboard_id": dashboard_id,
            "generated_at": datetime.utcnow().isoformat(),
            "widgets": {},
            "alerts": [],
            "summary": {}
        }
        
        # Get widget data
        for widget_id, widget in self.widgets.items():
            # Check access permissions
            if widget.required_roles and not any(role in user_roles for role in widget.required_roles):
                continue
            
            widget_data = await self._get_widget_data(widget)
            dashboard_data["widgets"][widget_id] = widget_data
        
        # Get active alerts
        active_alerts = await self._get_active_alerts()
        dashboard_data["alerts"] = active_alerts
        
        # Get executive summary
        summary = await self._generate_executive_summary()
        dashboard_data["summary"] = summary
        
        return dashboard_data
    
    async def _get_widget_data(self, widget: DashboardWidget) -> Dict[str, Any]:
        """Get data for a specific widget."""
        
        # Check cache first
        cache_key = f"{widget.widget_id}_{widget.time_range}"
        if (cache_key in self.data_cache and 
            cache_key in self.cache_timestamps and
            (datetime.utcnow() - self.cache_timestamps[cache_key]).total_seconds() < widget.refresh_interval):
            
            cached_data = self.data_cache[cache_key]
            cached_data["from_cache"] = True
            return cached_data
        
        # Fetch fresh data
        raw_data = await self.data_aggregator.get_metric_data(
            widget.metric_type,
            widget.time_range,
            widget.filters
        )
        
        # Generate visualization
        visualization = await self.visualization_engine.create_visualization(
            raw_data,
            widget.visualization_type,
            {
                "title": widget.title,
                "description": widget.description
            }
        )
        
        widget_data = {
            "widget_id": widget.widget_id,
            "title": widget.title,
            "description": widget.description,
            "metric_type": widget.metric_type.value,
            "visualization_type": widget.visualization_type.value,
            "position": widget.position,
            "data": raw_data,
            "visualization": visualization,
            "last_updated": datetime.utcnow().isoformat(),
            "from_cache": False
        }
        
        # Cache the data
        self.data_cache[cache_key] = widget_data
        self.cache_timestamps[cache_key] = datetime.utcnow()
        
        return widget_data
    
    async def _get_active_alerts(self) -> List[Dict[str, Any]]:
        """Get currently active alerts."""
        
        active_alerts = []
        
        for alert_id, alert in self.alerts.items():
            if not alert.enabled:
                continue
            
            # Check if alert should trigger
            widget = self.widgets.get(alert.widget_id)
            if not widget:
                continue
            
            # Get current metric value
            current_data = await self.data_aggregator.get_metric_data(
                widget.metric_type,
                "1h",  # Recent data for alerts
                widget.filters
            )
            
            current_value = self._extract_current_value(current_data, alert.metric)
            
            # Check threshold
            should_trigger = await self._check_alert_threshold(
                current_value, alert.threshold_type, alert.threshold_value
            )
            
            if should_trigger:
                # Check cooldown
                if (alert.last_triggered and 
                    (datetime.utcnow() - alert.last_triggered).total_seconds() < alert.cooldown_minutes * 60):
                    continue
                
                active_alerts.append({
                    "alert_id": alert_id,
                    "widget_id": alert.widget_id,
                    "widget_title": widget.title,
                    "metric": alert.metric,
                    "current_value": current_value,
                    "threshold_value": alert.threshold_value,
                    "threshold_type": alert.threshold_type,
                    "severity": alert.severity.value,
                    "triggered_at": datetime.utcnow().isoformat()
                })
                
                # Update alert status
                alert.last_triggered = datetime.utcnow()
                alert.trigger_count += 1
        
        return active_alerts
    
    async def _generate_executive_summary(self) -> Dict[str, Any]:
        """Generate executive summary for the dashboard."""
        
        # Get compliance scores across frameworks
        compliance_scores = {}
        for framework in ComplianceFramework:
            score_data = await self.data_aggregator.get_metric_data(
                DashboardMetric.COMPLIANCE_SCORE,
                "30d",
                {"framework": framework.value}
            )
            if score_data and score_data.get("values"):
                compliance_scores[framework.value] = score_data["values"][-1] if score_data["values"] else 0
        
        # Get recent violations
        violation_data = await self.data_aggregator.get_metric_data(
            DashboardMetric.POLICY_VIOLATIONS,
            "7d",
            {}
        )
        
        recent_violations = sum(violation_data.get("values", [])) if violation_data else 0
        
        # Get pending approvals
        approval_data = await self.data_aggregator.get_metric_data(
            DashboardMetric.APPROVAL_PENDING,
            "1d",
            {}
        )
        
        pending_approvals = approval_data.get("current_value", 0) if approval_data else 0
        
        # Calculate overall health score
        avg_compliance = sum(compliance_scores.values()) / len(compliance_scores) if compliance_scores else 0
        
        # Determine health status
        if avg_compliance >= 0.9:
            health_status = "excellent"
        elif avg_compliance >= 0.8:
            health_status = "good"
        elif avg_compliance >= 0.7:
            health_status = "fair"
        else:
            health_status = "poor"
        
        summary = {
            "overall_health_score": avg_compliance,
            "health_status": health_status,
            "compliance_scores": compliance_scores,
            "recent_violations": recent_violations,
            "pending_approvals": pending_approvals,
            "key_insights": await self._generate_key_insights(compliance_scores, recent_violations),
            "recommendations": await self._generate_recommendations(avg_compliance, recent_violations)
        }
        
        return summary
    
    async def _generate_key_insights(self,
                                   compliance_scores: Dict[str, float],
                                   recent_violations: int) -> List[str]:
        """Generate key insights for executive summary."""
        
        insights = []
        
        # Compliance insights
        if compliance_scores:
            best_framework = max(compliance_scores.items(), key=lambda x: x[1])
            worst_framework = min(compliance_scores.items(), key=lambda x: x[1])
            
            insights.append(f"Best compliance: {best_framework[0]} ({best_framework[1]:.1%})")
            
            if worst_framework[1] < 0.8:
                insights.append(f"Attention needed: {worst_framework[0]} ({worst_framework[1]:.1%})")
        
        # Violation insights
        if recent_violations > 0:
            insights.append(f"{recent_violations} policy violations in the last 7 days")
        else:
            insights.append("No policy violations in the last 7 days")
        
        # Trend insights
        # In production, would analyze trends over time
        insights.append("Compliance scores trending stable")
        
        return insights
    
    async def _generate_recommendations(self,
                                      avg_compliance: float,
                                      recent_violations: int) -> List[str]:
        """Generate recommendations based on current metrics."""
        
        recommendations = []
        
        if avg_compliance < 0.8:
            recommendations.append("Review and strengthen compliance policies")
            recommendations.append("Increase compliance training frequency")
        
        if recent_violations > 5:
            recommendations.append("Investigate root causes of policy violations")
            recommendations.append("Consider automated enforcement mechanisms")
        
        if avg_compliance >= 0.9 and recent_violations == 0:
            recommendations.append("Maintain current compliance practices")
            recommendations.append("Consider expanding governance scope")
        
        return recommendations
    
    async def start_monitoring(self) -> None:
        """Start dashboard monitoring and data collection."""
        
        if self.is_running:
            return
        
        self.is_running = True
        
        # Start background tasks
        self.background_tasks = [
            asyncio.create_task(self._data_collection_loop()),
            asyncio.create_task(self._alert_monitoring_loop()),
            asyncio.create_task(self._cache_cleanup_loop())
        ]
        
        self.logger.info("Dashboard monitoring started")
    
    async def stop_monitoring(self) -> None:
        """Stop dashboard monitoring."""
        
        if not self.is_running:
            return
        
        self.is_running = False
        
        # Cancel background tasks
        for task in self.background_tasks:
            task.cancel()
        
        await asyncio.gather(*self.background_tasks, return_exceptions=True)
        
        self.logger.info("Dashboard monitoring stopped")
    
    async def _data_collection_loop(self) -> None:
        """Background data collection for all widgets."""
        
        while self.is_running:
            try:
                for widget_id, widget in self.widgets.items():
                    # Refresh data based on widget refresh interval
                    cache_key = f"{widget_id}_{widget.time_range}"
                    
                    if (cache_key not in self.cache_timestamps or
                        (datetime.utcnow() - self.cache_timestamps[cache_key]).total_seconds() >= widget.refresh_interval):
                        
                        # Refresh widget data
                        await self._get_widget_data(widget)
                
                await asyncio.sleep(60)  # Check every minute
                
            except Exception as e:
                self.logger.error("Error in data collection loop", error=str(e))
                await asyncio.sleep(60)
    
    async def _alert_monitoring_loop(self) -> None:
        """Background alert monitoring."""
        
        while self.is_running:
            try:
                # Check all alerts
                await self._get_active_alerts()
                await asyncio.sleep(30)  # Check every 30 seconds
                
            except Exception as e:
                self.logger.error("Error in alert monitoring loop", error=str(e))
                await asyncio.sleep(30)
    
    async def _cache_cleanup_loop(self) -> None:
        """Background cache cleanup."""
        
        while self.is_running:
            try:
                # Clean up old cache entries
                current_time = datetime.utcnow()
                expired_keys = []
                
                for cache_key, timestamp in self.cache_timestamps.items():
                    if (current_time - timestamp).total_seconds() > 3600:  # 1 hour
                        expired_keys.append(cache_key)
                
                for key in expired_keys:
                    del self.data_cache[key]
                    del self.cache_timestamps[key]
                
                if expired_keys:
                    self.logger.debug(f"Cleaned up {len(expired_keys)} expired cache entries")
                
                await asyncio.sleep(1800)  # Clean every 30 minutes
                
            except Exception as e:
                self.logger.error("Error in cache cleanup loop", error=str(e))
                await asyncio.sleep(1800)
    
    def _extract_current_value(self, data: Dict[str, Any], metric: str) -> float:
        """Extract current value from metric data."""
        
        if not data:
            return 0.0
        
        if metric == "current_value" and "current_value" in data:
            return data["current_value"]
        
        if "values" in data and data["values"]:
            return data["values"][-1]  # Latest value
        
        return 0.0
    
    async def _check_alert_threshold(self,
                                   current_value: float,
                                   threshold_type: str,
                                   threshold_value: float) -> bool:
        """Check if alert threshold is met."""
        
        if threshold_type == "greater_than":
            return current_value > threshold_value
        elif threshold_type == "less_than":
            return current_value < threshold_value
        elif threshold_type == "equals":
            return abs(current_value - threshold_value) < 0.001
        elif threshold_type == "not_equals":
            return abs(current_value - threshold_value) >= 0.001
        
        return False


class DashboardDataAggregator:
    """Aggregates data for dashboard metrics."""
    
    def __init__(self, governance_framework: MLGovernanceFramework):
        self.governance = governance_framework
        self.logger = structlog.get_logger(__name__)
    
    async def get_metric_data(self,
                            metric_type: DashboardMetric,
                            time_range: str,
                            filters: Dict[str, Any]) -> Dict[str, Any]:
        """Get aggregated data for a specific metric."""
        
        end_time = datetime.utcnow()
        start_time = self._parse_time_range(time_range, end_time)
        
        if metric_type == DashboardMetric.COMPLIANCE_SCORE:
            return await self._get_compliance_score_data(start_time, end_time, filters)
        elif metric_type == DashboardMetric.RISK_LEVEL:
            return await self._get_risk_level_data(start_time, end_time, filters)
        elif metric_type == DashboardMetric.POLICY_VIOLATIONS:
            return await self._get_policy_violations_data(start_time, end_time, filters)
        elif metric_type == DashboardMetric.AUDIT_EVENTS:
            return await self._get_audit_events_data(start_time, end_time, filters)
        elif metric_type == DashboardMetric.APPROVAL_PENDING:
            return await self._get_approval_pending_data(filters)
        elif metric_type == DashboardMetric.CERTIFICATION_STATUS:
            return await self._get_certification_status_data(filters)
        else:
            return {"error": f"Unknown metric type: {metric_type}"}
    
    def _parse_time_range(self, time_range: str, end_time: datetime) -> datetime:
        """Parse time range string to start datetime."""
        
        if time_range == "1h":
            return end_time - timedelta(hours=1)
        elif time_range == "1d":
            return end_time - timedelta(days=1)
        elif time_range == "7d":
            return end_time - timedelta(days=7)
        elif time_range == "30d":
            return end_time - timedelta(days=30)
        elif time_range == "90d":
            return end_time - timedelta(days=90)
        else:
            return end_time - timedelta(days=7)  # Default
    
    async def _get_compliance_score_data(self,
                                       start_time: datetime,
                                       end_time: datetime,
                                       filters: Dict[str, Any]) -> Dict[str, Any]:
        """Get compliance score data over time."""
        
        # In production, would query actual compliance scores
        # For now, generate sample data
        time_points = pd.date_range(start=start_time, end=end_time, freq='H')
        
        # Simulate compliance scores with some variation
        base_score = 0.85
        scores = []
        
        for i, _ in enumerate(time_points):
            # Add some realistic variation
            variation = np.sin(i * 0.1) * 0.05 + np.random.normal(0, 0.02)
            score = max(0.0, min(1.0, base_score + variation))
            scores.append(score)
        
        return {
            "metric_type": "compliance_score",
            "time_range": {
                "start": start_time.isoformat(),
                "end": end_time.isoformat()
            },
            "timestamps": [t.isoformat() for t in time_points],
            "values": scores,
            "current_value": scores[-1] if scores else 0,
            "average": np.mean(scores) if scores else 0,
            "trend": "stable"  # In production, would calculate actual trend
        }
    
    async def _get_risk_level_data(self,
                                 start_time: datetime,
                                 end_time: datetime,
                                 filters: Dict[str, Any]) -> Dict[str, Any]:
        """Get risk level distribution data."""
        
        # Get audit events to analyze risk levels
        audit_events = await self.governance.get_audit_trail(
            start_date=start_time,
            end_date=end_time,
            limit=1000
        )
        
        # Count by risk level
        risk_counts = {
            "low": 0,
            "medium": 0,
            "high": 0,
            "critical": 0
        }
        
        for event in audit_events:
            risk_level = event.get("risk_level", "medium")
            if risk_level in risk_counts:
                risk_counts[risk_level] += 1
        
        total = sum(risk_counts.values()) or 1
        
        return {
            "metric_type": "risk_level",
            "distribution": {
                "low": risk_counts["low"] / total,
                "medium": risk_counts["medium"] / total,
                "high": risk_counts["high"] / total,
                "critical": risk_counts["critical"] / total
            },
            "counts": risk_counts,
            "total_events": total
        }
    
    async def _get_policy_violations_data(self,
                                        start_time: datetime,
                                        end_time: datetime,
                                        filters: Dict[str, Any]) -> Dict[str, Any]:
        """Get policy violations data over time."""
        
        # Get audit events with policy violations
        audit_events = await self.governance.get_audit_trail(
            start_date=start_time,
            end_date=end_time,
            limit=1000
        )
        
        # Filter for violation events
        violations = [
            event for event in audit_events
            if event.get("policy_violations") or 
            event.get("event_type") == "policy_violation"
        ]
        
        # Group by day
        violations_by_day = {}
        for event in violations:
            date_key = event["timestamp"][:10]  # Extract date part
            violations_by_day[date_key] = violations_by_day.get(date_key, 0) + 1
        
        # Generate time series
        time_points = pd.date_range(start=start_time.date(), end=end_time.date(), freq='D')
        values = []
        
        for date in time_points:
            date_key = date.strftime('%Y-%m-%d')
            values.append(violations_by_day.get(date_key, 0))
        
        return {
            "metric_type": "policy_violations",
            "timestamps": [t.isoformat() for t in time_points],
            "values": values,
            "total_violations": sum(values),
            "average_per_day": np.mean(values) if values else 0
        }


class VisualizationEngine:
    """Generates visualizations for dashboard widgets."""
    
    async def create_visualization(self,
                                 data: Dict[str, Any],
                                 visualization_type: VisualizationType,
                                 config: Dict[str, Any]) -> Dict[str, Any]:
        """Create visualization from data."""
        
        if visualization_type == VisualizationType.LINE_CHART:
            return self._create_line_chart(data, config)
        elif visualization_type == VisualizationType.BAR_CHART:
            return self._create_bar_chart(data, config)
        elif visualization_type == VisualizationType.PIE_CHART:
            return self._create_pie_chart(data, config)
        elif visualization_type == VisualizationType.GAUGE:
            return self._create_gauge_chart(data, config)
        elif visualization_type == VisualizationType.HEATMAP:
            return self._create_heatmap(data, config)
        elif visualization_type == VisualizationType.TREEMAP:
            return self._create_treemap(data, config)
        else:
            return {"error": f"Unsupported visualization type: {visualization_type}"}
    
    def _create_line_chart(self, data: Dict[str, Any], config: Dict[str, Any]) -> Dict[str, Any]:
        """Create line chart visualization."""
        
        if "timestamps" not in data or "values" not in data:
            return {"error": "Missing timestamps or values for line chart"}
        
        return {
            "type": "line_chart",
            "config": {
                "title": config.get("title", "Line Chart"),
                "x_axis": "Time",
                "y_axis": data.get("metric_type", "Value"),
                "data": {
                    "x": data["timestamps"],
                    "y": data["values"]
                }
            }
        }
    
    def _create_bar_chart(self, data: Dict[str, Any], config: Dict[str, Any]) -> Dict[str, Any]:
        """Create bar chart visualization."""
        
        if "timestamps" not in data or "values" not in data:
            return {"error": "Missing timestamps or values for bar chart"}
        
        return {
            "type": "bar_chart",
            "config": {
                "title": config.get("title", "Bar Chart"),
                "x_axis": "Time",
                "y_axis": data.get("metric_type", "Value"),
                "data": {
                    "x": data["timestamps"],
                    "y": data["values"]
                }
            }
        }
    
    def _create_pie_chart(self, data: Dict[str, Any], config: Dict[str, Any]) -> Dict[str, Any]:
        """Create pie chart visualization."""
        
        if "distribution" not in data and "counts" not in data:
            return {"error": "Missing distribution or counts for pie chart"}
        
        chart_data = data.get("distribution", data.get("counts", {}))
        
        return {
            "type": "pie_chart",
            "config": {
                "title": config.get("title", "Pie Chart"),
                "data": {
                    "labels": list(chart_data.keys()),
                    "values": list(chart_data.values())
                }
            }
        }
    
    def _create_gauge_chart(self, data: Dict[str, Any], config: Dict[str, Any]) -> Dict[str, Any]:
        """Create gauge chart visualization."""
        
        current_value = data.get("current_value", 0)
        
        return {
            "type": "gauge",
            "config": {
                "title": config.get("title", "Gauge"),
                "value": current_value,
                "min": 0,
                "max": 1,
                "thresholds": [
                    {"value": 0.7, "color": "red"},
                    {"value": 0.8, "color": "yellow"},
                    {"value": 1.0, "color": "green"}
                ]
            }
        }
    
    def _create_treemap(self, data: Dict[str, Any], config: Dict[str, Any]) -> Dict[str, Any]:
        """Create treemap visualization."""
        
        if "counts" not in data:
            return {"error": "Missing counts for treemap"}
        
        counts = data["counts"]
        
        return {
            "type": "treemap",
            "config": {
                "title": config.get("title", "Treemap"),
                "data": {
                    "labels": list(counts.keys()),
                    "values": list(counts.values()),
                    "parents": [""] * len(counts)  # All top-level
                }
            }
        }