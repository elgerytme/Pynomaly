"""Health monitoring service for system and model health checks."""

import asyncio
import json
import psutil
import time
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Union
from dataclasses import dataclass, asdict
from enum import Enum
import structlog

from ...infrastructure.repositories.model_repository import ModelRepository

logger = structlog.get_logger()


class HealthStatus(Enum):
    """Health status enumeration."""
    HEALTHY = "healthy"
    WARNING = "warning"
    CRITICAL = "critical"
    UNKNOWN = "unknown"


class AlertSeverity(Enum):
    """Alert severity levels."""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


@dataclass
class HealthMetric:
    """Individual health metric."""
    name: str
    value: Union[float, int, str, bool]
    unit: Optional[str] = None
    status: HealthStatus = HealthStatus.HEALTHY
    threshold: Optional[float] = None
    message: Optional[str] = None
    timestamp: datetime = None
    
    def __post_init__(self):
        if self.timestamp is None:
            self.timestamp = datetime.now()


@dataclass
class SystemHealth:
    """System health information."""
    cpu_usage: float
    memory_usage: float
    disk_usage: float
    network_io: Dict[str, int]
    uptime: float
    load_average: List[float]
    status: HealthStatus
    timestamp: datetime
    details: Optional[Dict[str, Any]] = None


@dataclass
class ModelHealth:
    """Model health information."""
    model_id: str
    status: HealthStatus
    last_prediction: Optional[datetime]
    prediction_count: int
    error_count: int
    average_response_time: float
    memory_usage: Optional[float]
    accuracy_score: Optional[float]
    last_training: Optional[datetime]
    version: Optional[str]
    issues: List[str]
    timestamp: datetime


@dataclass
class ServiceHealth:
    """Service health information."""
    service_name: str
    status: HealthStatus
    uptime: float
    request_count: int
    error_rate: float
    response_time: float
    last_error: Optional[str]
    timestamp: datetime


@dataclass
class HealthAlert:
    """Health alert information."""
    alert_id: str
    alert_type: str
    severity: AlertSeverity
    component: str
    message: str
    details: Dict[str, Any]
    timestamp: datetime
    resolved: bool = False
    acknowledged: bool = False
    resolution_time: Optional[datetime] = None


class HealthMonitoringService:
    """Service for monitoring system and model health."""
    
    def __init__(self):
        self.model_repository = ModelRepository()
        self.alerts: Dict[str, HealthAlert] = {}
        self.health_history: List[Dict[str, Any]] = []
        self.monitoring_active = False
        self.monitoring_task: Optional[asyncio.Task] = None
        self.thresholds = {
            "cpu_usage": 80.0,
            "memory_usage": 85.0,
            "disk_usage": 90.0,
            "response_time": 1000.0,  # ms
            "error_rate": 5.0,  # %
        }
        
    async def get_system_health(self) -> SystemHealth:
        """Get current system health metrics."""
        try:
            cpu_percent = psutil.cpu_percent(interval=1)
            memory = psutil.virtual_memory()
            disk = psutil.disk_usage('/')
            network = psutil.net_io_counters()
            boot_time = psutil.boot_time()
            uptime = time.time() - boot_time
            
            # Load average (Unix systems)
            try:
                load_avg = list(psutil.getloadavg())
            except (AttributeError, OSError):
                load_avg = [0.0, 0.0, 0.0]
            
            # Determine overall status
            status = HealthStatus.HEALTHY
            if (cpu_percent > self.thresholds["cpu_usage"] or 
                memory.percent > self.thresholds["memory_usage"] or
                (disk.total - disk.free) / disk.total * 100 > self.thresholds["disk_usage"]):
                status = HealthStatus.WARNING
            
            if (cpu_percent > 95 or memory.percent > 95 or 
                (disk.total - disk.free) / disk.total * 100 > 95):
                status = HealthStatus.CRITICAL
            
            network_io = {
                "bytes_sent": network.bytes_sent,
                "bytes_recv": network.bytes_recv,
                "packets_sent": network.packets_sent,
                "packets_recv": network.packets_recv
            }
            
            return SystemHealth(
                cpu_usage=cpu_percent,
                memory_usage=memory.percent,
                disk_usage=(disk.total - disk.free) / disk.total * 100,
                network_io=network_io,
                uptime=uptime,
                load_average=load_avg,
                status=status,
                timestamp=datetime.now(),
                details={
                    "memory_total": memory.total,
                    "memory_available": memory.available,
                    "disk_total": disk.total,
                    "disk_free": disk.free,
                    "cpu_count": psutil.cpu_count()
                }
            )
            
        except Exception as e:
            logger.error("Failed to get system health", error=str(e))
            return SystemHealth(
                cpu_usage=0.0,
                memory_usage=0.0,
                disk_usage=0.0,
                network_io={},
                uptime=0.0,
                load_average=[0.0, 0.0, 0.0],
                status=HealthStatus.UNKNOWN,
                timestamp=datetime.now()
            )
    
    async def get_model_health(self, model_id: str) -> ModelHealth:
        """Get health metrics for a specific model."""
        try:
            model = self.model_repository.load(model_id)
            if not model:
                return ModelHealth(
                    model_id=model_id,
                    status=HealthStatus.CRITICAL,
                    last_prediction=None,
                    prediction_count=0,
                    error_count=0,
                    average_response_time=0.0,
                    memory_usage=None,
                    accuracy_score=None,
                    last_training=None,
                    version=None,
                    issues=["Model not found"],
                    timestamp=datetime.now()
                )
            
            # Mock health data - in production, this would come from monitoring
            issues = []
            status = HealthStatus.HEALTHY
            
            # Check model age
            if hasattr(model, 'created_at'):
                model_age = (datetime.now() - model.created_at).days
                if model_age > 30:
                    issues.append(f"Model is {model_age} days old - consider retraining")
                    status = HealthStatus.WARNING
            
            # Check if model has performance metrics
            accuracy = getattr(model, 'accuracy', None)
            if accuracy and accuracy < 0.8:
                issues.append(f"Model accuracy is low: {accuracy:.2f}")
                status = HealthStatus.WARNING
            
            return ModelHealth(
                model_id=model_id,
                status=status,
                last_prediction=datetime.now() - timedelta(minutes=5),  # Mock
                prediction_count=1000,  # Mock
                error_count=2,  # Mock
                average_response_time=45.0,  # Mock
                memory_usage=128.0,  # Mock MB
                accuracy_score=accuracy,
                last_training=getattr(model, 'created_at', None),
                version=getattr(model, 'version', '1.0.0'),
                issues=issues,
                timestamp=datetime.now()
            )
            
        except Exception as e:
            logger.error("Failed to get model health", model_id=model_id, error=str(e))
            return ModelHealth(
                model_id=model_id,
                status=HealthStatus.UNKNOWN,
                last_prediction=None,
                prediction_count=0,
                error_count=0,
                average_response_time=0.0,
                memory_usage=None,
                accuracy_score=None,
                last_training=None,
                version=None,
                issues=[f"Health check failed: {str(e)}"],
                timestamp=datetime.now()
            )
    
    async def get_service_health(self, service_name: str) -> ServiceHealth:
        """Get health metrics for a service component."""
        # Mock service health - in production, this would integrate with actual service metrics
        try:
            # Simulate service health based on service name
            if service_name == "detection_api":
                return ServiceHealth(
                    service_name=service_name,
                    status=HealthStatus.HEALTHY,
                    uptime=86400.0,  # 1 day
                    request_count=5000,
                    error_rate=1.2,
                    response_time=125.0,
                    last_error=None,
                    timestamp=datetime.now()
                )
            elif service_name == "model_training":
                return ServiceHealth(
                    service_name=service_name,
                    status=HealthStatus.WARNING,
                    uptime=3600.0,  # 1 hour
                    request_count=50,
                    error_rate=8.0,
                    response_time=5000.0,
                    last_error="Training timeout on large dataset",
                    timestamp=datetime.now()
                )
            else:
                return ServiceHealth(
                    service_name=service_name,
                    status=HealthStatus.UNKNOWN,
                    uptime=0.0,
                    request_count=0,
                    error_rate=0.0,
                    response_time=0.0,
                    last_error="Service not monitored",
                    timestamp=datetime.now()
                )
                
        except Exception as e:
            logger.error("Failed to get service health", service_name=service_name, error=str(e))
            return ServiceHealth(
                service_name=service_name,
                status=HealthStatus.UNKNOWN,
                uptime=0.0,
                request_count=0,
                error_rate=0.0,
                response_time=0.0,
                last_error=str(e),
                timestamp=datetime.now()
            )
    
    async def get_comprehensive_health(self) -> Dict[str, Any]:
        """Get comprehensive health report for all components."""
        try:
            # Get system health
            system_health = await self.get_system_health()
            
            # Get all models health
            model_ids = self.model_repository.list_models()
            models_health = {}
            for model_id in model_ids[:10]:  # Limit to first 10 models
                models_health[model_id] = await self.get_model_health(model_id)
            
            # Get services health
            services = ["detection_api", "model_training", "streaming", "monitoring"]
            services_health = {}
            for service in services:
                services_health[service] = await self.get_service_health(service)
            
            # Calculate overall status
            overall_status = HealthStatus.HEALTHY
            critical_count = 0
            warning_count = 0
            
            # Check system status
            if system_health.status == HealthStatus.CRITICAL:
                critical_count += 1
            elif system_health.status == HealthStatus.WARNING:
                warning_count += 1
            
            # Check models status
            for model_health in models_health.values():
                if model_health.status == HealthStatus.CRITICAL:
                    critical_count += 1
                elif model_health.status == HealthStatus.WARNING:
                    warning_count += 1
            
            # Check services status
            for service_health in services_health.values():
                if service_health.status == HealthStatus.CRITICAL:
                    critical_count += 1
                elif service_health.status == HealthStatus.WARNING:
                    warning_count += 1
            
            if critical_count > 0:
                overall_status = HealthStatus.CRITICAL
            elif warning_count > 0:
                overall_status = HealthStatus.WARNING
            
            # Get recent alerts
            recent_alerts = [
                alert for alert in self.alerts.values()
                if not alert.resolved and 
                (datetime.now() - alert.timestamp).total_seconds() < 3600
            ]
            
            return {
                "overall_status": overall_status.value,
                "timestamp": datetime.now().isoformat(),
                "system": asdict(system_health),
                "models": {k: asdict(v) for k, v in models_health.items()},
                "services": {k: asdict(v) for k, v in services_health.items()},
                "alerts": [asdict(alert) for alert in recent_alerts],
                "summary": {
                    "total_components": 1 + len(models_health) + len(services_health),
                    "healthy_components": sum(1 for _ in [system_health] + list(models_health.values()) + list(services_health.values()) 
                                            if _.status == HealthStatus.HEALTHY),
                    "warning_components": warning_count,
                    "critical_components": critical_count,
                    "active_alerts": len(recent_alerts)
                }
            }
            
        except Exception as e:
            logger.error("Failed to get comprehensive health", error=str(e))
            return {
                "overall_status": HealthStatus.UNKNOWN.value,
                "timestamp": datetime.now().isoformat(),
                "error": str(e)
            }
    
    async def create_alert(self, alert_type: str, severity: AlertSeverity, 
                          component: str, message: str, details: Dict[str, Any] = None) -> str:
        """Create a new health alert."""
        import uuid
        
        alert_id = str(uuid.uuid4())
        alert = HealthAlert(
            alert_id=alert_id,
            alert_type=alert_type,
            severity=severity,
            component=component,
            message=message,
            details=details or {},
            timestamp=datetime.now()
        )
        
        self.alerts[alert_id] = alert
        
        logger.warning("Health alert created", 
                      alert_id=alert_id,
                      alert_type=alert_type,
                      severity=severity.value,
                      component=component,
                      message=message)
        
        return alert_id
    
    async def resolve_alert(self, alert_id: str, resolution_message: str = None) -> bool:
        """Resolve a health alert."""
        if alert_id not in self.alerts:
            return False
        
        alert = self.alerts[alert_id]
        alert.resolved = True
        alert.resolution_time = datetime.now()
        
        if resolution_message:
            alert.details["resolution_message"] = resolution_message
        
        logger.info("Health alert resolved", 
                   alert_id=alert_id,
                   resolution_message=resolution_message)
        
        return True
    
    async def acknowledge_alert(self, alert_id: str) -> bool:
        """Acknowledge a health alert."""
        if alert_id not in self.alerts:
            return False
        
        self.alerts[alert_id].acknowledged = True
        
        logger.info("Health alert acknowledged", alert_id=alert_id)
        
        return True
    
    async def get_alerts(self, resolved: Optional[bool] = None, 
                        severity: Optional[AlertSeverity] = None) -> List[HealthAlert]:
        """Get health alerts with optional filtering."""
        alerts = list(self.alerts.values())
        
        if resolved is not None:
            alerts = [alert for alert in alerts if alert.resolved == resolved]
        
        if severity is not None:
            alerts = [alert for alert in alerts if alert.severity == severity]
        
        # Sort by timestamp (newest first)
        alerts.sort(key=lambda x: x.timestamp, reverse=True)
        
        return alerts
    
    async def start_monitoring(self, interval_seconds: int = 60):
        """Start continuous health monitoring."""
        if self.monitoring_active:
            return
        
        self.monitoring_active = True
        self.monitoring_task = asyncio.create_task(self._monitoring_loop(interval_seconds))
        
        logger.info("Health monitoring started", interval_seconds=interval_seconds)
    
    async def stop_monitoring(self):
        """Stop continuous health monitoring."""
        self.monitoring_active = False
        
        if self.monitoring_task:
            self.monitoring_task.cancel()
            try:
                await self.monitoring_task
            except asyncio.CancelledError:
                pass
        
        logger.info("Health monitoring stopped")
    
    async def _monitoring_loop(self, interval_seconds: int):
        """Background monitoring loop."""
        while self.monitoring_active:
            try:
                # Get comprehensive health
                health_report = await self.get_comprehensive_health()
                
                # Store in history
                self.health_history.append(health_report)
                
                # Keep only last 100 entries
                if len(self.health_history) > 100:
                    self.health_history.pop(0)
                
                # Check for alerts
                await self._check_for_alerts(health_report)
                
                # Wait for next check
                await asyncio.sleep(interval_seconds)
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error("Monitoring loop error", error=str(e))
                await asyncio.sleep(interval_seconds)
    
    async def _check_for_alerts(self, health_report: Dict[str, Any]):
        """Check health report and create alerts if needed."""
        try:
            # Check system health
            system = health_report.get("system", {})
            if system.get("cpu_usage", 0) > self.thresholds["cpu_usage"]:
                await self.create_alert(
                    "high_cpu_usage",
                    AlertSeverity.HIGH,
                    "system",
                    f"CPU usage is {system['cpu_usage']:.1f}%",
                    {"cpu_usage": system["cpu_usage"], "threshold": self.thresholds["cpu_usage"]}
                )
            
            if system.get("memory_usage", 0) > self.thresholds["memory_usage"]:
                await self.create_alert(
                    "high_memory_usage",
                    AlertSeverity.HIGH,
                    "system",
                    f"Memory usage is {system['memory_usage']:.1f}%",
                    {"memory_usage": system["memory_usage"], "threshold": self.thresholds["memory_usage"]}
                )
            
            # Check service health
            services = health_report.get("services", {})
            for service_name, service_health in services.items():
                if service_health.get("error_rate", 0) > self.thresholds["error_rate"]:
                    await self.create_alert(
                        "high_error_rate",
                        AlertSeverity.MEDIUM,
                        f"service_{service_name}",
                        f"Service {service_name} error rate is {service_health['error_rate']:.1f}%",
                        {"service": service_name, "error_rate": service_health["error_rate"]}
                    )
            
        except Exception as e:
            logger.error("Alert checking failed", error=str(e))
    
    def get_health_history(self, limit: int = 50) -> List[Dict[str, Any]]:
        """Get recent health history."""
        return self.health_history[-limit:]
    
    def update_thresholds(self, new_thresholds: Dict[str, float]):
        """Update health monitoring thresholds."""
        self.thresholds.update(new_thresholds)
        logger.info("Health thresholds updated", thresholds=self.thresholds)