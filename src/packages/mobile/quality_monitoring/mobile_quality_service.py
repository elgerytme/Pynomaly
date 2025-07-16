"""Mobile Data Quality Monitoring Service.

Enhanced mobile-optimized service for real-time data quality monitoring,
alerts, and incident management on mobile devices.
"""

import asyncio
import json
import logging
from typing import Dict, Any, List, Optional, Callable
from datetime import datetime, timedelta
from dataclasses import dataclass, field
from enum import Enum
import uuid
from collections import defaultdict, deque

from ...interfaces.data_quality_interface import (
    DataQualityInterface, QualityReport, QualityIssue, QualityLevel
)

# Type aliases for backward compatibility
QualityMetric = Dict[str, float]
QualityThreshold = Dict[str, float]
QualityPrediction = Dict[str, Any]
QualityState = str
QualityTrend = str
QualityAnomaly = Dict[str, Any]
QualityLineage = Dict[str, Any]

logger = logging.getLogger(__name__)


class AlertSeverity(Enum):
    """Alert severity levels for mobile notifications."""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"
    EMERGENCY = "emergency"


class AlertType(Enum):
    """Types of mobile quality alerts."""
    QUALITY_DEGRADATION = "quality_degradation"
    THRESHOLD_BREACH = "threshold_breach"
    ANOMALY_DETECTED = "anomaly_detected"
    SYSTEM_FAILURE = "system_failure"
    PREDICTION_WARNING = "prediction_warning"
    INCIDENT_ESCALATION = "incident_escalation"


@dataclass
class MobileAlert:
    """Mobile-optimized quality alert."""
    alert_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    alert_type: AlertType = AlertType.QUALITY_DEGRADATION
    severity: AlertSeverity = AlertSeverity.MEDIUM
    title: str = ""
    message: str = ""
    dataset_id: str = ""
    metric_name: str = ""
    
    # Mobile-specific fields
    push_notification_sent: bool = False
    requires_immediate_attention: bool = False
    can_auto_resolve: bool = False
    mobile_actions: List[str] = field(default_factory=list)
    
    # Incident management
    incident_id: Optional[str] = None
    assigned_to: Optional[str] = None
    resolution_time_estimate: Optional[timedelta] = None
    
    # Metadata
    created_at: datetime = field(default_factory=datetime.utcnow)
    updated_at: datetime = field(default_factory=datetime.utcnow)
    resolved_at: Optional[datetime] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def to_mobile_dict(self) -> Dict[str, Any]:
        """Convert to mobile-friendly dictionary."""
        return {
            'alert_id': self.alert_id,
            'type': self.alert_type.value,
            'severity': self.severity.value,
            'title': self.title,
            'message': self.message,
            'dataset_id': self.dataset_id,
            'metric_name': self.metric_name,
            'requires_attention': self.requires_immediate_attention,
            'can_auto_resolve': self.can_auto_resolve,
            'mobile_actions': self.mobile_actions,
            'incident_id': self.incident_id,
            'assigned_to': self.assigned_to,
            'created_at': self.created_at.isoformat(),
            'updated_at': self.updated_at.isoformat(),
            'resolved_at': self.resolved_at.isoformat() if self.resolved_at else None,
            'is_resolved': self.resolved_at is not None,
            'age_minutes': int((datetime.utcnow() - self.created_at).total_seconds() / 60),
            'metadata': self.metadata
        }


@dataclass
class MobileIncident:
    """Mobile-optimized incident management."""
    incident_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    title: str = ""
    description: str = ""
    severity: AlertSeverity = AlertSeverity.MEDIUM
    status: str = "open"  # open, in_progress, resolved, closed
    
    # Quality context
    affected_datasets: List[str] = field(default_factory=list)
    affected_metrics: List[str] = field(default_factory=list)
    quality_impact_score: float = 0.0
    
    # Mobile workflow
    mobile_workflow_steps: List[str] = field(default_factory=list)
    completed_steps: List[str] = field(default_factory=list)
    next_action: Optional[str] = None
    estimated_resolution_time: Optional[timedelta] = None
    
    # Assignment and tracking
    assigned_to: Optional[str] = None
    escalated_to: Optional[str] = None
    resolution_notes: str = ""
    
    # Timestamps
    created_at: datetime = field(default_factory=datetime.utcnow)
    updated_at: datetime = field(default_factory=datetime.utcnow)
    resolved_at: Optional[datetime] = None
    
    def to_mobile_dict(self) -> Dict[str, Any]:
        """Convert to mobile-friendly dictionary."""
        return {
            'incident_id': self.incident_id,
            'title': self.title,
            'description': self.description,
            'severity': self.severity.value,
            'status': self.status,
            'affected_datasets': self.affected_datasets,
            'affected_metrics': self.affected_metrics,
            'quality_impact_score': self.quality_impact_score,
            'mobile_workflow_steps': self.mobile_workflow_steps,
            'completed_steps': self.completed_steps,
            'next_action': self.next_action,
            'progress_percentage': len(self.completed_steps) / len(self.mobile_workflow_steps) * 100 if self.mobile_workflow_steps else 0,
            'assigned_to': self.assigned_to,
            'escalated_to': self.escalated_to,
            'resolution_notes': self.resolution_notes,
            'created_at': self.created_at.isoformat(),
            'updated_at': self.updated_at.isoformat(),
            'resolved_at': self.resolved_at.isoformat() if self.resolved_at else None,
            'is_resolved': self.resolved_at is not None,
            'age_hours': int((datetime.utcnow() - self.created_at).total_seconds() / 3600)
        }


@dataclass
class MobileQualityDashboard:
    """Mobile-optimized quality dashboard data."""
    dashboard_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    user_id: str = ""
    
    # Quality overview
    overall_quality_score: float = 0.0
    total_datasets: int = 0
    healthy_datasets: int = 0
    degraded_datasets: int = 0
    critical_datasets: int = 0
    
    # Alerts and incidents
    active_alerts: int = 0
    critical_alerts: int = 0
    open_incidents: int = 0
    
    # Quality metrics
    quality_metrics: List[Dict[str, Any]] = field(default_factory=list)
    trending_metrics: List[Dict[str, Any]] = field(default_factory=list)
    
    # Mobile-specific features
    offline_data_available: bool = False
    last_sync_time: datetime = field(default_factory=datetime.utcnow)
    refresh_interval_seconds: int = 30
    
    # Personalization
    favorite_datasets: List[str] = field(default_factory=list)
    custom_alert_rules: List[Dict[str, Any]] = field(default_factory=list)
    
    def to_mobile_dict(self) -> Dict[str, Any]:
        """Convert to mobile-friendly dictionary."""
        return {
            'dashboard_id': self.dashboard_id,
            'user_id': self.user_id,
            'overall_quality_score': self.overall_quality_score,
            'dataset_summary': {
                'total': self.total_datasets,
                'healthy': self.healthy_datasets,
                'degraded': self.degraded_datasets,
                'critical': self.critical_datasets,
                'health_percentage': (self.healthy_datasets / self.total_datasets * 100) if self.total_datasets > 0 else 0
            },
            'alert_summary': {
                'active': self.active_alerts,
                'critical': self.critical_alerts,
                'open_incidents': self.open_incidents
            },
            'quality_metrics': self.quality_metrics,
            'trending_metrics': self.trending_metrics,
            'offline_data_available': self.offline_data_available,
            'last_sync_time': self.last_sync_time.isoformat(),
            'refresh_interval_seconds': self.refresh_interval_seconds,
            'favorite_datasets': self.favorite_datasets,
            'custom_alert_rules': self.custom_alert_rules
        }


class MobileQualityMonitoringService:
    """Mobile-optimized data quality monitoring service."""
    
    def __init__(self, 
                 base_quality_service=None,
                 push_notification_service=None,
                 offline_storage_service=None):
        """Initialize mobile quality monitoring service.
        
        Args:
            base_quality_service: Base quality monitoring service
            push_notification_service: Push notification service
            offline_storage_service: Offline storage service
        """
        self.base_quality_service = base_quality_service
        self.push_notification_service = push_notification_service
        self.offline_storage_service = offline_storage_service
        
        # Mobile-specific storage
        self.mobile_alerts: Dict[str, MobileAlert] = {}
        self.mobile_incidents: Dict[str, MobileIncident] = {}
        self.mobile_dashboards: Dict[str, MobileQualityDashboard] = {}
        
        # Configuration
        self.config = {
            'push_notifications_enabled': True,
            'offline_sync_enabled': True,
            'real_time_updates_enabled': True,
            'max_alerts_per_user': 1000,
            'max_incidents_per_user': 100,
            'dashboard_refresh_interval': 30,
            'alert_throttling_minutes': 5,
            'critical_alert_immediate_push': True
        }
        
        # Alert throttling
        self.alert_throttle: Dict[str, datetime] = {}
        
        logger.info("Mobile Quality Monitoring Service initialized")
    
    async def create_mobile_dashboard(self, user_id: str, 
                                    favorite_datasets: List[str] = None) -> MobileQualityDashboard:
        """Create mobile-optimized quality dashboard.
        
        Args:
            user_id: User identifier
            favorite_datasets: List of favorite dataset IDs
            
        Returns:
            Mobile quality dashboard
        """
        dashboard = MobileQualityDashboard(
            user_id=user_id,
            favorite_datasets=favorite_datasets or []
        )
        
        # Populate dashboard with current quality data
        await self._populate_dashboard_data(dashboard)
        
        # Store dashboard
        self.mobile_dashboards[dashboard.dashboard_id] = dashboard
        
        logger.info(f"Created mobile dashboard for user {user_id}")
        return dashboard
    
    async def get_mobile_dashboard(self, dashboard_id: str) -> Optional[MobileQualityDashboard]:
        """Get mobile dashboard by ID.
        
        Args:
            dashboard_id: Dashboard identifier
            
        Returns:
            Mobile quality dashboard or None
        """
        dashboard = self.mobile_dashboards.get(dashboard_id)
        if dashboard:
            await self._populate_dashboard_data(dashboard)
        return dashboard
    
    async def refresh_mobile_dashboard(self, dashboard_id: str) -> Optional[MobileQualityDashboard]:
        """Refresh mobile dashboard data.
        
        Args:
            dashboard_id: Dashboard identifier
            
        Returns:
            Refreshed mobile quality dashboard
        """
        dashboard = self.mobile_dashboards.get(dashboard_id)
        if not dashboard:
            return None
        
        await self._populate_dashboard_data(dashboard)
        dashboard.last_sync_time = datetime.utcnow()
        
        # Save to offline storage if enabled
        if self.config['offline_sync_enabled'] and self.offline_storage_service:
            await self.offline_storage_service.save_dashboard(dashboard.to_mobile_dict())
        
        logger.debug(f"Refreshed mobile dashboard {dashboard_id}")
        return dashboard
    
    async def create_mobile_alert(self, 
                                dataset_id: str,
                                metric_name: str,
                                alert_type: AlertType,
                                severity: AlertSeverity,
                                title: str,
                                message: str,
                                metadata: Dict[str, Any] = None) -> MobileAlert:
        """Create mobile-optimized quality alert.
        
        Args:
            dataset_id: Dataset identifier
            metric_name: Metric name
            alert_type: Type of alert
            severity: Alert severity
            title: Alert title
            message: Alert message
            metadata: Additional metadata
            
        Returns:
            Mobile alert
        """
        alert = MobileAlert(
            alert_type=alert_type,
            severity=severity,
            title=title,
            message=message,
            dataset_id=dataset_id,
            metric_name=metric_name,
            metadata=metadata or {}
        )
        
        # Configure mobile-specific properties
        alert.requires_immediate_attention = severity in [AlertSeverity.CRITICAL, AlertSeverity.EMERGENCY]
        alert.can_auto_resolve = alert_type in [AlertType.THRESHOLD_BREACH, AlertType.ANOMALY_DETECTED]
        
        # Generate mobile actions
        alert.mobile_actions = self._generate_mobile_actions(alert)
        
        # Store alert
        self.mobile_alerts[alert.alert_id] = alert
        
        # Send push notification if enabled
        if self._should_send_push_notification(alert):
            await self._send_push_notification(alert)
        
        logger.info(f"Created mobile alert {alert.alert_id} for dataset {dataset_id}")
        return alert
    
    async def create_mobile_incident(self, 
                                   title: str,
                                   description: str,
                                   severity: AlertSeverity,
                                   affected_datasets: List[str],
                                   affected_metrics: List[str] = None) -> MobileIncident:
        """Create mobile-optimized quality incident.
        
        Args:
            title: Incident title
            description: Incident description
            severity: Incident severity
            affected_datasets: List of affected datasets
            affected_metrics: List of affected metrics
            
        Returns:
            Mobile incident
        """
        incident = MobileIncident(
            title=title,
            description=description,
            severity=severity,
            affected_datasets=affected_datasets,
            affected_metrics=affected_metrics or []
        )
        
        # Generate mobile workflow
        incident.mobile_workflow_steps = self._generate_mobile_workflow(incident)
        incident.next_action = incident.mobile_workflow_steps[0] if incident.mobile_workflow_steps else None
        
        # Calculate quality impact
        incident.quality_impact_score = await self._calculate_quality_impact(incident)
        
        # Store incident
        self.mobile_incidents[incident.incident_id] = incident
        
        # Create associated alert
        alert = await self.create_mobile_alert(
            dataset_id=affected_datasets[0] if affected_datasets else "",
            metric_name=affected_metrics[0] if affected_metrics else "overall",
            alert_type=AlertType.INCIDENT_ESCALATION,
            severity=severity,
            title=f"Incident Created: {title}",
            message=f"New incident requires attention: {description}",
            metadata={'incident_id': incident.incident_id}
        )
        
        incident.incident_id = alert.alert_id
        
        logger.info(f"Created mobile incident {incident.incident_id}")
        return incident
    
    async def get_mobile_alerts(self, user_id: str, 
                              limit: int = 50,
                              severity_filter: Optional[AlertSeverity] = None) -> List[Dict[str, Any]]:
        """Get mobile alerts for user.
        
        Args:
            user_id: User identifier
            limit: Maximum number of alerts
            severity_filter: Filter by severity
            
        Returns:
            List of mobile alerts
        """
        alerts = list(self.mobile_alerts.values())
        
        # Filter by severity if specified
        if severity_filter:
            alerts = [alert for alert in alerts if alert.severity == severity_filter]
        
        # Sort by created time (newest first)
        alerts.sort(key=lambda x: x.created_at, reverse=True)
        
        # Limit results
        alerts = alerts[:limit]
        
        # Convert to mobile format
        mobile_alerts = [alert.to_mobile_dict() for alert in alerts]
        
        logger.debug(f"Retrieved {len(mobile_alerts)} mobile alerts for user {user_id}")
        return mobile_alerts
    
    async def get_mobile_incidents(self, user_id: str, 
                                 limit: int = 20,
                                 status_filter: Optional[str] = None) -> List[Dict[str, Any]]:
        """Get mobile incidents for user.
        
        Args:
            user_id: User identifier
            limit: Maximum number of incidents
            status_filter: Filter by status
            
        Returns:
            List of mobile incidents
        """
        incidents = list(self.mobile_incidents.values())
        
        # Filter by status if specified
        if status_filter:
            incidents = [incident for incident in incidents if incident.status == status_filter]
        
        # Sort by created time (newest first)
        incidents.sort(key=lambda x: x.created_at, reverse=True)
        
        # Limit results
        incidents = incidents[:limit]
        
        # Convert to mobile format
        mobile_incidents = [incident.to_mobile_dict() for incident in incidents]
        
        logger.debug(f"Retrieved {len(mobile_incidents)} mobile incidents for user {user_id}")
        return mobile_incidents
    
    async def resolve_mobile_alert(self, alert_id: str, 
                                 resolution_notes: str = "") -> bool:
        """Resolve mobile alert.
        
        Args:
            alert_id: Alert identifier
            resolution_notes: Resolution notes
            
        Returns:
            True if successfully resolved
        """
        alert = self.mobile_alerts.get(alert_id)
        if not alert:
            return False
        
        alert.resolved_at = datetime.utcnow()
        alert.updated_at = datetime.utcnow()
        alert.metadata['resolution_notes'] = resolution_notes
        
        logger.info(f"Resolved mobile alert {alert_id}")
        return True
    
    async def update_incident_status(self, incident_id: str, 
                                   status: str,
                                   completed_step: str = None,
                                   notes: str = "") -> bool:
        """Update mobile incident status.
        
        Args:
            incident_id: Incident identifier
            status: New status
            completed_step: Completed workflow step
            notes: Update notes
            
        Returns:
            True if successfully updated
        """
        incident = self.mobile_incidents.get(incident_id)
        if not incident:
            return False
        
        incident.status = status
        incident.updated_at = datetime.utcnow()
        
        if completed_step and completed_step not in incident.completed_steps:
            incident.completed_steps.append(completed_step)
            
            # Update next action
            remaining_steps = [step for step in incident.mobile_workflow_steps 
                             if step not in incident.completed_steps]
            incident.next_action = remaining_steps[0] if remaining_steps else None
        
        if status == "resolved":
            incident.resolved_at = datetime.utcnow()
            incident.resolution_notes = notes
        
        logger.info(f"Updated mobile incident {incident_id} status to {status}")
        return True
    
    async def get_quality_metrics_for_mobile(self, dataset_id: str) -> List[Dict[str, Any]]:
        """Get quality metrics optimized for mobile display.
        
        Args:
            dataset_id: Dataset identifier
            
        Returns:
            List of mobile-optimized quality metrics
        """
        # This would integrate with the base quality service
        # For now, return mock data
        metrics = [
            {
                'metric_name': 'completeness',
                'current_value': 0.95,
                'threshold': 0.90,
                'trend': 'stable',
                'status': 'healthy',
                'last_updated': datetime.utcnow().isoformat(),
                'mobile_chart_data': [0.94, 0.95, 0.96, 0.95, 0.95]
            },
            {
                'metric_name': 'accuracy',
                'current_value': 0.87,
                'threshold': 0.85,
                'trend': 'improving',
                'status': 'healthy',
                'last_updated': datetime.utcnow().isoformat(),
                'mobile_chart_data': [0.83, 0.84, 0.85, 0.86, 0.87]
            },
            {
                'metric_name': 'consistency',
                'current_value': 0.78,
                'threshold': 0.80,
                'trend': 'degrading',
                'status': 'warning',
                'last_updated': datetime.utcnow().isoformat(),
                'mobile_chart_data': [0.82, 0.81, 0.80, 0.79, 0.78]
            }
        ]
        
        return metrics
    
    async def _populate_dashboard_data(self, dashboard: MobileQualityDashboard):
        """Populate dashboard with current quality data."""
        # Mock data for now - would integrate with real quality service
        dashboard.overall_quality_score = 0.85
        dashboard.total_datasets = 25
        dashboard.healthy_datasets = 18
        dashboard.degraded_datasets = 6
        dashboard.critical_datasets = 1
        
        dashboard.active_alerts = len([a for a in self.mobile_alerts.values() if not a.resolved_at])
        dashboard.critical_alerts = len([a for a in self.mobile_alerts.values() 
                                       if not a.resolved_at and a.severity == AlertSeverity.CRITICAL])
        dashboard.open_incidents = len([i for i in self.mobile_incidents.values() 
                                      if i.status in ['open', 'in_progress']])
        
        # Quality metrics for mobile display
        dashboard.quality_metrics = [
            {
                'name': 'Overall Quality',
                'value': dashboard.overall_quality_score,
                'trend': 'stable',
                'status': 'healthy' if dashboard.overall_quality_score >= 0.8 else 'warning'
            },
            {
                'name': 'Completeness',
                'value': 0.92,
                'trend': 'improving',
                'status': 'healthy'
            },
            {
                'name': 'Accuracy',
                'value': 0.88,
                'trend': 'stable',
                'status': 'healthy'
            },
            {
                'name': 'Consistency',
                'value': 0.75,
                'trend': 'degrading',
                'status': 'warning'
            }
        ]
        
        # Trending metrics
        dashboard.trending_metrics = [
            {
                'name': 'Data Freshness',
                'trend': 'improving',
                'change_percentage': 5.2
            },
            {
                'name': 'Schema Compliance',
                'trend': 'stable',
                'change_percentage': 0.1
            },
            {
                'name': 'Data Volume',
                'trend': 'increasing',
                'change_percentage': 12.3
            }
        ]
    
    def _generate_mobile_actions(self, alert: MobileAlert) -> List[str]:
        """Generate mobile actions for an alert."""
        actions = []
        
        if alert.alert_type == AlertType.THRESHOLD_BREACH:
            actions.extend([
                "View metric details",
                "Adjust threshold",
                "Investigate root cause",
                "Assign to team member"
            ])
        elif alert.alert_type == AlertType.ANOMALY_DETECTED:
            actions.extend([
                "View anomaly details",
                "Mark as false positive",
                "Investigate data source",
                "Create incident"
            ])
        elif alert.alert_type == AlertType.SYSTEM_FAILURE:
            actions.extend([
                "Check system status",
                "Restart services",
                "Escalate to ops team",
                "Create incident"
            ])
        
        actions.extend([
            "Resolve alert",
            "Snooze for 1 hour",
            "Share with team"
        ])
        
        return actions
    
    def _generate_mobile_workflow(self, incident: MobileIncident) -> List[str]:
        """Generate mobile workflow steps for an incident."""
        workflow = []
        
        if incident.severity in [AlertSeverity.CRITICAL, AlertSeverity.EMERGENCY]:
            workflow.extend([
                "Acknowledge incident",
                "Assess impact scope",
                "Notify stakeholders",
                "Implement immediate fixes",
                "Monitor resolution",
                "Document lessons learned"
            ])
        else:
            workflow.extend([
                "Acknowledge incident",
                "Investigate root cause",
                "Implement solution",
                "Test resolution",
                "Close incident"
            ])
        
        return workflow
    
    async def _calculate_quality_impact(self, incident: MobileIncident) -> float:
        """Calculate quality impact score for an incident."""
        # Mock calculation - would integrate with real quality service
        base_score = 0.5
        
        # Increase based on affected datasets
        dataset_impact = min(len(incident.affected_datasets) * 0.1, 0.3)
        
        # Increase based on severity
        severity_impact = {
            AlertSeverity.LOW: 0.0,
            AlertSeverity.MEDIUM: 0.1,
            AlertSeverity.HIGH: 0.2,
            AlertSeverity.CRITICAL: 0.3,
            AlertSeverity.EMERGENCY: 0.4
        }.get(incident.severity, 0.0)
        
        return min(base_score + dataset_impact + severity_impact, 1.0)
    
    def _should_send_push_notification(self, alert: MobileAlert) -> bool:
        """Determine if push notification should be sent."""
        if not self.config['push_notifications_enabled']:
            return False
        
        # Always send for critical alerts
        if alert.severity in [AlertSeverity.CRITICAL, AlertSeverity.EMERGENCY]:
            return True
        
        # Check throttling
        throttle_key = f"{alert.dataset_id}:{alert.metric_name}"
        if throttle_key in self.alert_throttle:
            time_since_last = datetime.utcnow() - self.alert_throttle[throttle_key]
            if time_since_last.total_seconds() < (self.config['alert_throttling_minutes'] * 60):
                return False
        
        self.alert_throttle[throttle_key] = datetime.utcnow()
        return True
    
    async def _send_push_notification(self, alert: MobileAlert):
        """Send push notification for alert."""
        if not self.push_notification_service:
            logger.warning("Push notification service not available")
            return
        
        notification_data = {
            'title': alert.title,
            'body': alert.message,
            'data': {
                'alert_id': alert.alert_id,
                'alert_type': alert.alert_type.value,
                'severity': alert.severity.value,
                'dataset_id': alert.dataset_id,
                'requires_attention': alert.requires_immediate_attention
            }
        }
        
        try:
            await self.push_notification_service.send_notification(notification_data)
            alert.push_notification_sent = True
            logger.info(f"Push notification sent for alert {alert.alert_id}")
        except Exception as e:
            logger.error(f"Failed to send push notification for alert {alert.alert_id}: {e}")
    
    async def sync_offline_data(self, user_id: str) -> Dict[str, Any]:
        """Sync data for offline mobile access.
        
        Args:
            user_id: User identifier
            
        Returns:
            Offline data package
        """
        if not self.config['offline_sync_enabled']:
            return {}
        
        # Get user's dashboard
        user_dashboard = None
        for dashboard in self.mobile_dashboards.values():
            if dashboard.user_id == user_id:
                user_dashboard = dashboard
                break
        
        if not user_dashboard:
            return {}
        
        # Prepare offline data
        offline_data = {
            'sync_timestamp': datetime.utcnow().isoformat(),
            'dashboard': user_dashboard.to_mobile_dict(),
            'recent_alerts': await self.get_mobile_alerts(user_id, limit=20),
            'active_incidents': await self.get_mobile_incidents(user_id, limit=10, status_filter='open'),
            'quality_metrics': {},
            'offline_actions': [
                'View dashboard',
                'Review alerts',
                'Update incident status',
                'Add notes',
                'Mark alerts as read'
            ]
        }
        
        # Get quality metrics for favorite datasets
        for dataset_id in user_dashboard.favorite_datasets:
            offline_data['quality_metrics'][dataset_id] = await self.get_quality_metrics_for_mobile(dataset_id)
        
        # Save to offline storage
        if self.offline_storage_service:
            await self.offline_storage_service.save_offline_data(user_id, offline_data)
        
        logger.info(f"Synchronized offline data for user {user_id}")
        return offline_data
    
    async def get_mobile_summary(self, user_id: str) -> Dict[str, Any]:
        """Get mobile summary for user.
        
        Args:
            user_id: User identifier
            
        Returns:
            Mobile summary data
        """
        summary = {
            'user_id': user_id,
            'summary_timestamp': datetime.utcnow().isoformat(),
            'overall_status': 'healthy',
            'quality_score': 0.85,
            'alerts': {
                'total': len(self.mobile_alerts),
                'critical': len([a for a in self.mobile_alerts.values() if a.severity == AlertSeverity.CRITICAL]),
                'unresolved': len([a for a in self.mobile_alerts.values() if not a.resolved_at])
            },
            'incidents': {
                'total': len(self.mobile_incidents),
                'open': len([i for i in self.mobile_incidents.values() if i.status == 'open']),
                'in_progress': len([i for i in self.mobile_incidents.values() if i.status == 'in_progress'])
            },
            'trending_up': ['Data Freshness', 'Completeness'],
            'trending_down': ['Consistency'],
            'requires_attention': [],
            'recent_activity': []
        }
        
        # Add items requiring attention
        for alert in self.mobile_alerts.values():
            if not alert.resolved_at and alert.requires_immediate_attention:
                summary['requires_attention'].append({
                    'type': 'alert',
                    'id': alert.alert_id,
                    'title': alert.title,
                    'severity': alert.severity.value
                })
        
        # Add recent activity
        all_items = []
        for alert in self.mobile_alerts.values():
            all_items.append({
                'type': 'alert',
                'title': alert.title,
                'timestamp': alert.created_at
            })
        
        for incident in self.mobile_incidents.values():
            all_items.append({
                'type': 'incident',
                'title': incident.title,
                'timestamp': incident.created_at
            })
        
        # Sort by timestamp and take recent items
        all_items.sort(key=lambda x: x['timestamp'], reverse=True)
        summary['recent_activity'] = [
            {
                'type': item['type'],
                'title': item['title'],
                'timestamp': item['timestamp'].isoformat()
            }
            for item in all_items[:10]
        ]
        
        return summary