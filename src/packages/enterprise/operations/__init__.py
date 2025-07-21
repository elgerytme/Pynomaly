"""Enterprise operations - monitoring, alerting, SRE concerns."""

from typing import Dict, Any, List, Optional, Callable
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
import asyncio
import structlog

logger = structlog.get_logger(__name__)


class HealthStatus(str, Enum):
    """Health check status."""
    HEALTHY = "healthy"
    DEGRADED = "degraded"
    UNHEALTHY = "unhealthy"
    UNKNOWN = "unknown"


class AlertSeverity(str, Enum):
    """Alert severity levels."""
    INFO = "info"
    WARNING = "warning"
    ERROR = "error"
    CRITICAL = "critical"


@dataclass
class HealthCheck:
    """Health check result."""
    name: str
    status: HealthStatus
    message: str = ""
    duration_ms: float = 0.0
    timestamp: datetime = field(default_factory=datetime.utcnow)
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class Metric:
    """System metric."""
    name: str
    value: float
    unit: str = ""
    tags: Dict[str, str] = field(default_factory=dict)
    timestamp: datetime = field(default_factory=datetime.utcnow)


@dataclass
class Alert:
    """System alert."""
    id: str
    title: str
    description: str
    severity: AlertSeverity
    service: str
    tags: Dict[str, str] = field(default_factory=dict)
    timestamp: datetime = field(default_factory=datetime.utcnow)
    resolved: bool = False
    resolved_at: Optional[datetime] = None


class HealthProvider(ABC):
    """Abstract health check provider."""
    
    @abstractmethod
    async def check_health(self) -> HealthCheck:
        """Perform health check."""
        pass


class MetricsCollector(ABC):
    """Abstract metrics collector."""
    
    @abstractmethod
    async def collect_metrics(self) -> List[Metric]:
        """Collect system metrics."""
        pass
    
    @abstractmethod
    async def record_metric(self, metric: Metric) -> None:
        """Record a metric."""
        pass


class AlertManager(ABC):
    """Abstract alert manager."""
    
    @abstractmethod
    async def send_alert(self, alert: Alert) -> None:
        """Send alert."""
        pass
    
    @abstractmethod
    async def resolve_alert(self, alert_id: str) -> None:
        """Resolve alert."""
        pass


class EnterpriseOperationsService:
    """Enterprise operations service for monitoring, alerting, SRE."""
    
    def __init__(
        self,
        health_providers: List[HealthProvider] = None,
        metrics_collector: Optional[MetricsCollector] = None,
        alert_manager: Optional[AlertManager] = None,
        health_check_interval: int = 30,
        metrics_collection_interval: int = 60
    ):
        self.health_providers = health_providers or []
        self.metrics_collector = metrics_collector
        self.alert_manager = alert_manager
        self.health_check_interval = health_check_interval
        self.metrics_collection_interval = metrics_collection_interval
        
        self.logger = logger.bind(service="enterprise_operations")
        self._running = False
        self._tasks = []
    
    async def start(self) -> None:
        """Start background monitoring tasks."""
        if self._running:
            return
        
        self._running = True
        self.logger.info("Starting enterprise operations monitoring")
        
        # Start health check task
        if self.health_providers:
            task = asyncio.create_task(self._health_check_loop())
            self._tasks.append(task)
        
        # Start metrics collection task
        if self.metrics_collector:
            task = asyncio.create_task(self._metrics_collection_loop())
            self._tasks.append(task)
    
    async def stop(self) -> None:
        """Stop background monitoring tasks."""
        if not self._running:
            return
        
        self._running = False
        self.logger.info("Stopping enterprise operations monitoring")
        
        # Cancel all tasks
        for task in self._tasks:
            task.cancel()
        
        # Wait for tasks to complete
        await asyncio.gather(*self._tasks, return_exceptions=True)
        self._tasks.clear()
    
    async def get_system_health(self) -> Dict[str, Any]:
        """Get comprehensive system health."""
        health_checks = []
        overall_status = HealthStatus.HEALTHY
        
        for provider in self.health_providers:
            try:
                start_time = datetime.utcnow()
                health_check = await provider.check_health()
                duration = (datetime.utcnow() - start_time).total_seconds() * 1000
                health_check.duration_ms = duration
                
                health_checks.append(health_check)
                
                # Update overall status
                if health_check.status == HealthStatus.UNHEALTHY:
                    overall_status = HealthStatus.UNHEALTHY
                elif health_check.status == HealthStatus.DEGRADED and overall_status == HealthStatus.HEALTHY:
                    overall_status = HealthStatus.DEGRADED
                    
            except Exception as e:
                error_check = HealthCheck(
                    name=f"{provider.__class__.__name__}",
                    status=HealthStatus.UNHEALTHY,
                    message=f"Health check failed: {str(e)}"
                )
                health_checks.append(error_check)
                overall_status = HealthStatus.UNHEALTHY
                self.logger.error("Health check failed", provider=provider.__class__.__name__, error=str(e))
        
        return {
            "status": overall_status.value,
            "timestamp": datetime.utcnow().isoformat(),
            "checks": [
                {
                    "name": check.name,
                    "status": check.status.value,
                    "message": check.message,
                    "duration_ms": check.duration_ms,
                    "metadata": check.metadata
                }
                for check in health_checks
            ]
        }
    
    async def record_metric(
        self,
        name: str,
        value: float,
        unit: str = "",
        tags: Optional[Dict[str, str]] = None
    ) -> None:
        """Record a system metric."""
        if not self.metrics_collector:
            return
        
        metric = Metric(
            name=name,
            value=value,
            unit=unit,
            tags=tags or {}
        )
        
        try:
            await self.metrics_collector.record_metric(metric)
            self.logger.debug("Metric recorded", name=name, value=value)
        except Exception as e:
            self.logger.error("Failed to record metric", name=name, error=str(e))
    
    async def send_alert(
        self,
        title: str,
        description: str,
        severity: AlertSeverity,
        service: str,
        tags: Optional[Dict[str, str]] = None
    ) -> str:
        """Send system alert."""
        if not self.alert_manager:
            self.logger.warning("No alert manager configured", title=title)
            return ""
        
        alert = Alert(
            id=f"alert_{datetime.utcnow().timestamp()}",
            title=title,
            description=description,
            severity=severity,
            service=service,
            tags=tags or {}
        )
        
        try:
            await self.alert_manager.send_alert(alert)
            self.logger.info("Alert sent", alert_id=alert.id, title=title, severity=severity.value)
            return alert.id
        except Exception as e:
            self.logger.error("Failed to send alert", title=title, error=str(e))
            return ""
    
    async def _health_check_loop(self) -> None:
        """Background health check loop."""
        while self._running:
            try:
                health_status = await self.get_system_health()
                
                # Send alert if system is unhealthy
                if health_status["status"] == HealthStatus.UNHEALTHY.value:
                    unhealthy_checks = [
                        check for check in health_status["checks"] 
                        if check["status"] == HealthStatus.UNHEALTHY.value
                    ]
                    
                    await self.send_alert(
                        title="System Health Alert",
                        description=f"Unhealthy components: {[c['name'] for c in unhealthy_checks]}",
                        severity=AlertSeverity.ERROR,
                        service="system_health"
                    )
                
                await asyncio.sleep(self.health_check_interval)
                
            except Exception as e:
                self.logger.error("Health check loop error", error=str(e))
                await asyncio.sleep(self.health_check_interval)
    
    async def _metrics_collection_loop(self) -> None:
        """Background metrics collection loop."""
        while self._running:
            try:
                if self.metrics_collector:
                    metrics = await self.metrics_collector.collect_metrics()
                    self.logger.debug("Collected metrics", count=len(metrics))
                
                await asyncio.sleep(self.metrics_collection_interval)
                
            except Exception as e:
                self.logger.error("Metrics collection loop error", error=str(e))
                await asyncio.sleep(self.metrics_collection_interval)


class DatabaseHealthProvider(HealthProvider):
    """Database health check provider."""
    
    def __init__(self, database_connection):
        self.db = database_connection
    
    async def check_health(self) -> HealthCheck:
        """Check database health."""
        try:
            # Example database ping
            # await self.db.execute("SELECT 1")
            
            return HealthCheck(
                name="database",
                status=HealthStatus.HEALTHY,
                message="Database connection healthy"
            )
        except Exception as e:
            return HealthCheck(
                name="database",
                status=HealthStatus.UNHEALTHY,
                message=f"Database connection failed: {str(e)}"
            )


class SystemMetricsCollector(MetricsCollector):
    """System metrics collector."""
    
    def __init__(self):
        self.metrics_store = []
    
    async def collect_metrics(self) -> List[Metric]:
        """Collect system metrics."""
        import psutil
        
        metrics = [
            Metric("cpu_usage", psutil.cpu_percent(), "percent"),
            Metric("memory_usage", psutil.virtual_memory().percent, "percent"),
            Metric("disk_usage", psutil.disk_usage('/').percent, "percent"),
        ]
        
        return metrics
    
    async def record_metric(self, metric: Metric) -> None:
        """Record metric."""
        self.metrics_store.append(metric)


__all__ = [
    "EnterpriseOperationsService",
    "HealthProvider",
    "MetricsCollector", 
    "AlertManager",
    "HealthStatus",
    "AlertSeverity",
    "HealthCheck",
    "Metric",
    "Alert",
    "DatabaseHealthProvider",
    "SystemMetricsCollector"
]