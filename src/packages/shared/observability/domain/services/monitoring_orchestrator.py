"""Enhanced monitoring orchestrator for coordinating all monitoring services."""

import asyncio
import logging
import threading
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Callable, Set
from dataclasses import dataclass, field
from enum import Enum
from collections import defaultdict
import json

from .intelligent_alerting_service import IntelligentAlertingService, AlertSeverity
from .business_metrics_service import BusinessMetricsService
from .performance_profiler import PerformanceProfiler
from ..monitoring.metrics_collector import MetricsCollector, get_metrics_collector
from ..monitoring.health_checker import HealthChecker
from ..monitoring.performance_monitor import PerformanceMonitor


class MonitoringServiceType(Enum):
    """Types of monitoring services."""
    METRICS_COLLECTOR = "metrics_collector"
    HEALTH_CHECKER = "health_checker"
    PERFORMANCE_MONITOR = "performance_monitor"
    ALERTING_SERVICE = "alerting_service"
    BUSINESS_METRICS = "business_metrics"
    PERFORMANCE_PROFILER = "performance_profiler"


class ServiceHealth(Enum):
    """Health status of monitoring services."""
    HEALTHY = "healthy"
    DEGRADED = "degraded"
    UNHEALTHY = "unhealthy"
    UNKNOWN = "unknown"


@dataclass
class ServiceDependency:
    """Represents a dependency between monitoring services."""
    service_type: MonitoringServiceType
    dependent_on: MonitoringServiceType
    criticality: str = "high"  # high, medium, low
    recovery_action: Optional[Callable] = None


@dataclass
class MonitoringAlert:
    """Enhanced monitoring alert with correlation information."""
    alert_id: str
    service_type: MonitoringServiceType
    severity: AlertSeverity
    message: str
    timestamp: datetime
    context: Dict[str, Any]
    correlation_id: Optional[str] = None
    root_cause_probability: float = 0.0
    related_alerts: List[str] = field(default_factory=list)
    suppressed: bool = False


@dataclass
class MonitoringDashboard:
    """Real-time monitoring dashboard data."""
    overall_health: ServiceHealth
    service_statuses: Dict[MonitoringServiceType, ServiceHealth]
    active_alerts: List[MonitoringAlert]
    key_metrics: Dict[str, float]
    performance_summary: Dict[str, Any]
    business_metrics_summary: Dict[str, Any]
    recent_events: List[Dict[str, Any]]
    last_updated: datetime


class MonitoringOrchestrator:
    """Enhanced monitoring orchestrator that coordinates all monitoring services."""
    
    def __init__(
        self,
        alerting_service: Optional[IntelligentAlertingService] = None,
        business_metrics_service: Optional[BusinessMetricsService] = None,
        enable_auto_recovery: bool = True,
        dashboard_update_interval: int = 30
    ):
        """Initialize monitoring orchestrator.
        
        Args:
            alerting_service: Intelligent alerting service
            business_metrics_service: Business metrics service
            enable_auto_recovery: Whether to enable automatic service recovery
            dashboard_update_interval: Dashboard update interval in seconds
        """
        self.logger = logging.getLogger(__name__)
        
        # Core services
        self.alerting_service = alerting_service
        self.business_metrics_service = business_metrics_service
        self.metrics_collector = get_metrics_collector()
        self.health_checker = HealthChecker()
        self.performance_monitor = PerformanceMonitor()
        self.performance_profiler = PerformanceProfiler()
        
        # Configuration
        self.enable_auto_recovery = enable_auto_recovery
        self.dashboard_update_interval = dashboard_update_interval
        
        # Service registry and health tracking
        self.service_registry: Dict[MonitoringServiceType, Any] = {
            MonitoringServiceType.METRICS_COLLECTOR: self.metrics_collector,
            MonitoringServiceType.HEALTH_CHECKER: self.health_checker,
            MonitoringServiceType.PERFORMANCE_MONITOR: self.performance_monitor,
            MonitoringServiceType.PERFORMANCE_PROFILER: self.performance_profiler,
        }
        
        if self.alerting_service:
            self.service_registry[MonitoringServiceType.ALERTING_SERVICE] = self.alerting_service
            
        if self.business_metrics_service:
            self.service_registry[MonitoringServiceType.BUSINESS_METRICS] = self.business_metrics_service
        
        self.service_health: Dict[MonitoringServiceType, ServiceHealth] = {}
        self.service_dependencies = self._initialize_service_dependencies()
        
        # Alert correlation
        self.alert_correlation_window = timedelta(minutes=5)
        self.active_alerts: Dict[str, MonitoringAlert] = {}
        self.alert_correlation_rules = self._initialize_correlation_rules()
        
        # Dashboard and state management
        self.dashboard_data: Optional[MonitoringDashboard] = None
        self.event_history: List[Dict[str, Any]] = []
        
        # Background tasks
        self._running = False
        self._orchestration_thread: Optional[threading.Thread] = None
        self._dashboard_thread: Optional[threading.Thread] = None
        
        self.logger.info("Monitoring orchestrator initialized")
    
    def _initialize_service_dependencies(self) -> List[ServiceDependency]:
        """Initialize service dependency graph."""
        dependencies = [
            # Alerting depends on metrics
            ServiceDependency(
                MonitoringServiceType.ALERTING_SERVICE,
                MonitoringServiceType.METRICS_COLLECTOR,
                "high"
            ),
            # Business metrics depend on metrics collector
            ServiceDependency(
                MonitoringServiceType.BUSINESS_METRICS,
                MonitoringServiceType.METRICS_COLLECTOR,
                "high"
            ),
            # Performance profiler depends on performance monitor
            ServiceDependency(
                MonitoringServiceType.PERFORMANCE_PROFILER,
                MonitoringServiceType.PERFORMANCE_MONITOR,
                "medium"
            ),
        ]
        return dependencies
    
    def _initialize_correlation_rules(self) -> List[Dict[str, Any]]:
        """Initialize alert correlation rules."""
        return [
            {
                "name": "high_memory_usage_cascade",
                "description": "High memory usage can cause performance degradation",
                "primary_pattern": {"metric": "memory_usage", "threshold": "> 0.8"},
                "secondary_patterns": [
                    {"metric": "response_time", "threshold": "> 1000"},
                    {"metric": "error_rate", "threshold": "> 0.05"}
                ],
                "correlation_probability": 0.8
            },
            {
                "name": "model_performance_degradation",
                "description": "Model accuracy drop can indicate data drift",
                "primary_pattern": {"metric": "model_accuracy", "threshold": "< 0.85"},
                "secondary_patterns": [
                    {"metric": "data_drift_score", "threshold": "> 0.3"},
                    {"metric": "prediction_confidence", "threshold": "< 0.7"}
                ],
                "correlation_probability": 0.9
            },
            {
                "name": "system_overload",
                "description": "System overload cascade failure",
                "primary_pattern": {"metric": "cpu_usage", "threshold": "> 0.9"},
                "secondary_patterns": [
                    {"metric": "queue_size", "threshold": "> 1000"},
                    {"metric": "processing_latency", "threshold": "> 5000"}
                ],
                "correlation_probability": 0.85
            }
        ]
    
    def start(self):
        """Start the monitoring orchestrator."""
        if self._running:
            return
        
        self._running = True
        
        # Start orchestration thread
        self._orchestration_thread = threading.Thread(
            target=self._orchestration_loop, 
            daemon=True
        )
        self._orchestration_thread.start()
        
        # Start dashboard update thread
        self._dashboard_thread = threading.Thread(
            target=self._dashboard_update_loop,
            daemon=True
        )
        self._dashboard_thread.start()
        
        # Start dependent services
        if self.business_metrics_service:
            self.business_metrics_service.start()
        
        self.logger.info("Monitoring orchestrator started")
    
    def stop(self):
        """Stop the monitoring orchestrator."""
        self._running = False
        
        # Stop dependent services
        if self.business_metrics_service:
            self.business_metrics_service.stop()
        
        # Wait for threads to finish
        if self._orchestration_thread:
            self._orchestration_thread.join(timeout=5)
        
        if self._dashboard_thread:
            self._dashboard_thread.join(timeout=5)
        
        self.logger.info("Monitoring orchestrator stopped")
    
    def _orchestration_loop(self):
        """Main orchestration loop."""
        while self._running:
            try:
                # Check service health
                self._check_service_health()
                
                # Process alerts
                self._process_alerts()
                
                # Handle service failures
                if self.enable_auto_recovery:
                    self._handle_service_failures()
                
                # Correlate alerts
                self._correlate_alerts()
                
                # Record orchestration metrics
                self._record_orchestration_metrics()
                
                # Sleep before next iteration
                threading.Event().wait(10)  # 10 second intervals
                
            except Exception as e:
                self.logger.error(f"Error in orchestration loop: {e}")
                threading.Event().wait(10)
    
    def _dashboard_update_loop(self):
        """Dashboard update loop."""
        while self._running:
            try:
                self.dashboard_data = self._generate_dashboard_data()
                threading.Event().wait(self.dashboard_update_interval)
            except Exception as e:
                self.logger.error(f"Error updating dashboard: {e}")
                threading.Event().wait(self.dashboard_update_interval)
    
    def _check_service_health(self):
        """Check health of all registered monitoring services."""
        for service_type, service in self.service_registry.items():
            try:
                if hasattr(service, 'health_check'):
                    health_result = service.health_check()
                    
                    if isinstance(health_result, dict):
                        if health_result.get("status") == "healthy":
                            self.service_health[service_type] = ServiceHealth.HEALTHY
                        elif health_result.get("status") in ["degraded", "warning"]:
                            self.service_health[service_type] = ServiceHealth.DEGRADED
                        else:
                            self.service_health[service_type] = ServiceHealth.UNHEALTHY
                    else:
                        # Assume healthy if health_check returns non-dict
                        self.service_health[service_type] = ServiceHealth.HEALTHY
                else:
                    # Service doesn't have health check - assume healthy if accessible
                    self.service_health[service_type] = ServiceHealth.HEALTHY
                    
            except Exception as e:
                self.logger.error(f"Health check failed for {service_type.value}: {e}")
                self.service_health[service_type] = ServiceHealth.UNHEALTHY
    
    def _process_alerts(self):
        """Process and correlate alerts from all services."""
        # Get new alerts from alerting service
        if self.alerting_service:
            try:
                # Check for new alerts (this is a simplified approach)
                recent_alerts = self._get_recent_alerts()
                
                for alert_data in recent_alerts:
                    alert = MonitoringAlert(
                        alert_id=alert_data.get("alert_id", str(datetime.now().timestamp())),
                        service_type=MonitoringServiceType.ALERTING_SERVICE,
                        severity=AlertSeverity(alert_data.get("severity", "medium")),
                        message=alert_data.get("message", "Unknown alert"),
                        timestamp=datetime.now(),
                        context=alert_data.get("context", {})
                    )
                    
                    self.active_alerts[alert.alert_id] = alert
                    
            except Exception as e:
                self.logger.error(f"Error processing alerts: {e}")
    
    def _get_recent_alerts(self) -> List[Dict[str, Any]]:
        """Get recent alerts from monitoring services."""
        # This is a placeholder - in practice, this would query
        # the alerting service for recent alerts
        alerts = []
        
        # Check service health and generate alerts
        for service_type, health in self.service_health.items():
            if health == ServiceHealth.UNHEALTHY:
                alerts.append({
                    "alert_id": f"service_health_{service_type.value}_{int(datetime.now().timestamp())}",
                    "severity": "high",
                    "message": f"Service {service_type.value} is unhealthy",
                    "context": {"service_type": service_type.value, "health": health.value}
                })
            elif health == ServiceHealth.DEGRADED:
                alerts.append({
                    "alert_id": f"service_degraded_{service_type.value}_{int(datetime.now().timestamp())}",
                    "severity": "medium", 
                    "message": f"Service {service_type.value} is degraded",
                    "context": {"service_type": service_type.value, "health": health.value}
                })
        
        return alerts
    
    def _correlate_alerts(self):
        """Correlate related alerts to reduce noise."""
        current_time = datetime.now()
        correlation_window_start = current_time - self.alert_correlation_window
        
        # Get alerts within correlation window
        recent_alerts = [
            alert for alert in self.active_alerts.values()
            if alert.timestamp >= correlation_window_start and not alert.suppressed
        ]
        
        # Apply correlation rules
        for rule in self.alert_correlation_rules:
            self._apply_correlation_rule(rule, recent_alerts)
    
    def _apply_correlation_rule(self, rule: Dict[str, Any], alerts: List[MonitoringAlert]):
        """Apply a specific correlation rule to alerts."""
        rule_name = rule["name"]
        correlation_probability = rule["correlation_probability"]
        
        # Find primary alerts matching the pattern
        primary_alerts = []
        for alert in alerts:
            if self._alert_matches_pattern(alert, rule["primary_pattern"]):
                primary_alerts.append(alert)
        
        # For each primary alert, find correlated secondary alerts
        for primary_alert in primary_alerts:
            correlated_alerts = []
            
            for alert in alerts:
                if alert.alert_id != primary_alert.alert_id:
                    for secondary_pattern in rule["secondary_patterns"]:
                        if self._alert_matches_pattern(alert, secondary_pattern):
                            correlated_alerts.append(alert)
                            break
            
            # If we found correlated alerts, update relationships
            if correlated_alerts:
                correlation_id = f"corr_{rule_name}_{primary_alert.alert_id}"
                
                # Update primary alert
                primary_alert.correlation_id = correlation_id
                primary_alert.root_cause_probability = correlation_probability
                primary_alert.related_alerts = [a.alert_id for a in correlated_alerts]
                
                # Update secondary alerts (suppress them)
                for secondary_alert in correlated_alerts:
                    secondary_alert.correlation_id = correlation_id
                    secondary_alert.suppressed = True
                    secondary_alert.root_cause_probability = 1.0 - correlation_probability
                
                self.logger.info(
                    f"Correlated {len(correlated_alerts)} alerts under rule '{rule_name}'"
                )
    
    def _alert_matches_pattern(self, alert: MonitoringAlert, pattern: Dict[str, Any]) -> bool:
        """Check if an alert matches a correlation pattern."""
        # This is a simplified pattern matching - in practice, this would be more sophisticated
        metric_name = pattern.get("metric")
        threshold = pattern.get("threshold")
        
        # Check if alert context contains the metric
        context = alert.context
        if metric_name in context:
            metric_value = context[metric_name]
            
            # Parse threshold (e.g., "> 0.8", "< 1000")
            if threshold.startswith(">"):
                threshold_value = float(threshold[1:].strip())
                return metric_value > threshold_value
            elif threshold.startswith("<"):
                threshold_value = float(threshold[1:].strip())
                return metric_value < threshold_value
        
        return False
    
    def _handle_service_failures(self):
        """Handle service failures with automatic recovery."""
        for service_type, health in self.service_health.items():
            if health == ServiceHealth.UNHEALTHY:
                self._attempt_service_recovery(service_type)
    
    def _attempt_service_recovery(self, service_type: MonitoringServiceType):
        """Attempt to recover a failed service."""
        service = self.service_registry.get(service_type)
        if not service:
            return
        
        try:
            # Attempt to restart/reinitialize service
            if hasattr(service, 'restart'):
                service.restart()
                self.logger.info(f"Attempted restart of {service_type.value}")
            elif hasattr(service, 'initialize'):
                service.initialize()
                self.logger.info(f"Attempted reinitialization of {service_type.value}")
            
            # Record recovery attempt
            self._record_event({
                "type": "service_recovery_attempt",
                "service_type": service_type.value,
                "timestamp": datetime.now().isoformat(),
                "success": None  # Will be determined in next health check
            })
            
        except Exception as e:
            self.logger.error(f"Service recovery failed for {service_type.value}: {e}")
    
    def _record_orchestration_metrics(self):
        """Record metrics about the orchestration process."""
        if not self.metrics_collector:
            return
        
        try:
            # Record service health metrics
            for service_type, health in self.service_health.items():
                health_score = {
                    ServiceHealth.HEALTHY: 1.0,
                    ServiceHealth.DEGRADED: 0.5,
                    ServiceHealth.UNHEALTHY: 0.0,
                    ServiceHealth.UNKNOWN: 0.25
                }.get(health, 0.0)
                
                self.metrics_collector.record_metric(
                    f"monitoring.service_health.{service_type.value}",
                    health_score,
                    {"service_type": service_type.value}
                )
            
            # Record alert metrics
            total_alerts = len(self.active_alerts)
            suppressed_alerts = sum(1 for alert in self.active_alerts.values() if alert.suppressed)
            
            self.metrics_collector.record_metric(
                "monitoring.alerts.total",
                total_alerts,
                {"status": "active"}
            )
            
            self.metrics_collector.record_metric(
                "monitoring.alerts.suppressed",
                suppressed_alerts,
                {"status": "suppressed"} 
            )
            
            # Record overall health
            overall_health = self._calculate_overall_health()
            self.metrics_collector.record_metric(
                "monitoring.overall_health",
                overall_health,
                {"orchestrator": "enabled"}
            )
            
        except Exception as e:
            self.logger.error(f"Error recording orchestration metrics: {e}")
    
    def _calculate_overall_health(self) -> float:
        """Calculate overall system health score."""
        if not self.service_health:
            return 0.0
        
        health_scores = []
        for health in self.service_health.values():
            score = {
                ServiceHealth.HEALTHY: 1.0,
                ServiceHealth.DEGRADED: 0.5,
                ServiceHealth.UNHEALTHY: 0.0,
                ServiceHealth.UNKNOWN: 0.25
            }.get(health, 0.0)
            health_scores.append(score)
        
        return sum(health_scores) / len(health_scores)
    
    def _generate_dashboard_data(self) -> MonitoringDashboard:
        """Generate real-time dashboard data."""
        current_time = datetime.now()
        
        # Calculate overall health
        overall_health_score = self._calculate_overall_health()
        overall_health = ServiceHealth.HEALTHY
        if overall_health_score < 0.3:
            overall_health = ServiceHealth.UNHEALTHY
        elif overall_health_score < 0.7:
            overall_health = ServiceHealth.DEGRADED
        
        # Get active non-suppressed alerts
        active_alerts = [
            alert for alert in self.active_alerts.values()
            if not alert.suppressed and 
            (current_time - alert.timestamp) < timedelta(hours=1)
        ]
        
        # Get key metrics
        key_metrics = {}
        if self.metrics_collector:
            try:
                recent_metrics = self.metrics_collector.get_recent_metrics(hours=1)
                key_metrics = {
                    "total_metrics_collected": len(recent_metrics),
                    "metrics_per_minute": len(recent_metrics) / 60 if recent_metrics else 0,
                    "overall_health_score": overall_health_score
                }
            except Exception as e:
                self.logger.warning(f"Error getting key metrics: {e}")
        
        # Get performance summary
        performance_summary = {}
        if self.performance_profiler:
            try:
                performance_summary = {
                    "active_profiles": len(getattr(self.performance_profiler, '_active_profiles', [])),
                    "total_operations_profiled": getattr(self.performance_profiler, '_total_operations', 0)
                }
            except Exception:
                pass
        
        # Get business metrics summary
        business_metrics_summary = {}
        if self.business_metrics_service:
            try:
                dashboard_data = self.business_metrics_service.generate_business_dashboard()
                business_metrics_summary = {
                    "metrics_tracked": dashboard_data.get("system_health", {}).get("metrics_tracked", 0),
                    "sla_overview": dashboard_data.get("sla_overview", {})
                }
            except Exception as e:
                self.logger.warning(f"Error getting business metrics: {e}")
        
        return MonitoringDashboard(
            overall_health=overall_health,
            service_statuses=dict(self.service_health),
            active_alerts=active_alerts,
            key_metrics=key_metrics,
            performance_summary=performance_summary,  
            business_metrics_summary=business_metrics_summary,
            recent_events=self.event_history[-10:],  # Last 10 events
            last_updated=current_time
        )
    
    def _record_event(self, event: Dict[str, Any]):
        """Record a monitoring event."""
        self.event_history.append(event)
        
        # Keep only last 1000 events
        if len(self.event_history) > 1000:
            self.event_history = self.event_history[-1000:]
    
    def get_dashboard_data(self) -> Optional[MonitoringDashboard]:
        """Get current dashboard data."""
        return self.dashboard_data
    
    def get_service_health(self, service_type: MonitoringServiceType) -> ServiceHealth:
        """Get health status of a specific service."""
        return self.service_health.get(service_type, ServiceHealth.UNKNOWN)
    
    def get_active_alerts(self, include_suppressed: bool = False) -> List[MonitoringAlert]:
        """Get currently active alerts."""
        if include_suppressed:
            return list(self.active_alerts.values())
        else:
            return [alert for alert in self.active_alerts.values() if not alert.suppressed]
    
    def acknowledge_alert(self, alert_id: str, acknowledged_by: str) -> bool:
        """Acknowledge an alert."""
        if alert_id in self.active_alerts:
            alert = self.active_alerts[alert_id]
            alert.context["acknowledged"] = True
            alert.context["acknowledged_by"] = acknowledged_by
            alert.context["acknowledged_at"] = datetime.now().isoformat()
            
            self._record_event({
                "type": "alert_acknowledged",
                "alert_id": alert_id,
                "acknowledged_by": acknowledged_by,
                "timestamp": datetime.now().isoformat()
            })
            
            return True
        return False
    
    def resolve_alert(self, alert_id: str, resolved_by: str, resolution_notes: str = "") -> bool:
        """Resolve an alert."""
        if alert_id in self.active_alerts:
            alert = self.active_alerts[alert_id]
            alert.context["resolved"] = True
            alert.context["resolved_by"] = resolved_by
            alert.context["resolved_at"] = datetime.now().isoformat()
            alert.context["resolution_notes"] = resolution_notes
            
            # Remove from active alerts
            del self.active_alerts[alert_id]
            
            self._record_event({
                "type": "alert_resolved",
                "alert_id": alert_id,
                "resolved_by": resolved_by,
                "resolution_notes": resolution_notes,
                "timestamp": datetime.now().isoformat()
            })
            
            return True
        return False
    
    def force_service_check(self, service_type: MonitoringServiceType) -> Dict[str, Any]:
        """Force a health check on a specific service."""
        service = self.service_registry.get(service_type)
        if not service:
            return {"error": f"Service {service_type.value} not found"}
        
        try:
            if hasattr(service, 'health_check'):
                result = service.health_check()
                
                # Update health status immediately
                self._check_service_health()
                
                return {
                    "service_type": service_type.value,
                    "health_check_result": result,
                    "forced_at": datetime.now().isoformat()
                }
            else:
                return {"error": f"Service {service_type.value} does not support health checks"}
                
        except Exception as e:
            return {"error": f"Health check failed: {str(e)}"}
    
    def get_orchestrator_stats(self) -> Dict[str, Any]:
        """Get statistics about the orchestrator."""
        return {
            "orchestrator_running": self._running,
            "services_registered": len(self.service_registry),
            "services_healthy": sum(
                1 for health in self.service_health.values() 
                if health == ServiceHealth.HEALTHY
            ),
            "services_unhealthy": sum(
                1 for health in self.service_health.values()
                if health == ServiceHealth.UNHEALTHY
            ),
            "active_alerts": len(self.get_active_alerts()),
            "suppressed_alerts": len(self.get_active_alerts(include_suppressed=True)) - len(self.get_active_alerts()),
            "correlation_rules": len(self.alert_correlation_rules),
            "events_recorded": len(self.event_history),
            "dashboard_last_updated": self.dashboard_data.last_updated.isoformat() if self.dashboard_data else None,
            "auto_recovery_enabled": self.enable_auto_recovery
        }


# Global orchestrator instance
_monitoring_orchestrator: Optional[MonitoringOrchestrator] = None


def initialize_monitoring_orchestrator(
    alerting_service: Optional[IntelligentAlertingService] = None,
    business_metrics_service: Optional[BusinessMetricsService] = None,
    enable_auto_recovery: bool = True
) -> MonitoringOrchestrator:
    """Initialize global monitoring orchestrator.
    
    Args:
        alerting_service: Intelligent alerting service
        business_metrics_service: Business metrics service
        enable_auto_recovery: Whether to enable automatic service recovery
        
    Returns:
        Initialized monitoring orchestrator
    """
    global _monitoring_orchestrator
    _monitoring_orchestrator = MonitoringOrchestrator(
        alerting_service=alerting_service,
        business_metrics_service=business_metrics_service,
        enable_auto_recovery=enable_auto_recovery
    )
    return _monitoring_orchestrator


def get_monitoring_orchestrator() -> Optional[MonitoringOrchestrator]:
    """Get global monitoring orchestrator instance.
    
    Returns:
        Monitoring orchestrator instance or None
    """
    return _monitoring_orchestrator