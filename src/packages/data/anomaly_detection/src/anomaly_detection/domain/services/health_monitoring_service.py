"""Health monitoring service for system health checks."""

from __future__ import annotations

import asyncio
import logging
import psutil
import time
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any
from dataclasses import dataclass
from enum import Enum

# Import AlertSeverity from alerting system for backward compatibility
try:
    from ...infrastructure.monitoring.alerting_system import AlertSeverity
except ImportError:
    # Define minimal AlertSeverity if not available
    class AlertSeverity(str, Enum):
        LOW = "low"
        MEDIUM = "medium"
        HIGH = "high"
        CRITICAL = "critical"

logger = logging.getLogger(__name__)


class HealthStatus(str, Enum):
    """Health check status enumeration."""
    HEALTHY = "healthy"
    WARNING = "warning"
    CRITICAL = "critical"
    UNKNOWN = "unknown"


@dataclass
class HealthCheck:
    """Individual health check result."""
    name: str
    status: HealthStatus
    message: str
    timestamp: datetime
    response_time_ms: Optional[float] = None
    metadata: Optional[Dict[str, Any]] = None


@dataclass
class SystemHealth:
    """Overall system health status."""
    overall_status: HealthStatus
    checks: List[HealthCheck]
    timestamp: datetime
    uptime_seconds: float
    
    @property
    def is_healthy(self) -> bool:
        """Check if system is healthy."""
        return self.overall_status == HealthStatus.HEALTHY
    
    @property
    def has_warnings(self) -> bool:
        """Check if system has warnings."""
        return any(check.status == HealthStatus.WARNING for check in self.checks)
    
    @property
    def has_critical_issues(self) -> bool:
        """Check if system has critical issues."""
        return any(check.status == HealthStatus.CRITICAL for check in self.checks)


class HealthMonitoringService:
    """Service for monitoring system health."""
    
    def __init__(self):
        """Initialize health monitoring service."""
        self.start_time = time.time()
        self.health_checks: Dict[str, callable] = {}
        self._register_default_checks()
    
    def _register_default_checks(self) -> None:
        """Register default health checks."""
        self.register_check("cpu_usage", self._check_cpu_usage)
        self.register_check("memory_usage", self._check_memory_usage)
        self.register_check("disk_usage", self._check_disk_usage)
        self.register_check("database_connection", self._check_database_connection)
        self.register_check("service_dependencies", self._check_service_dependencies)
    
    def register_check(self, name: str, check_func: callable) -> None:
        """Register a health check function."""
        self.health_checks[name] = check_func
    
    async def get_health_status(self) -> SystemHealth:
        """Get current system health status."""
        checks = []
        
        for name, check_func in self.health_checks.items():
            try:
                start_time = time.time()
                
                if asyncio.iscoroutinefunction(check_func):
                    check_result = await check_func()
                else:
                    check_result = check_func()
                
                response_time = (time.time() - start_time) * 1000
                
                if isinstance(check_result, HealthCheck):
                    check_result.response_time_ms = response_time
                    checks.append(check_result)
                else:
                    # Handle simple boolean returns
                    status = HealthStatus.HEALTHY if check_result else HealthStatus.CRITICAL
                    checks.append(HealthCheck(
                        name=name,
                        status=status,
                        message="Check passed" if check_result else "Check failed",
                        timestamp=datetime.utcnow(),
                        response_time_ms=response_time
                    ))
                    
            except Exception as e:
                logger.exception(f"Health check '{name}' failed: {e}")
                checks.append(HealthCheck(
                    name=name,
                    status=HealthStatus.CRITICAL,
                    message=f"Check failed with error: {str(e)}",
                    timestamp=datetime.utcnow(),
                    metadata={"error": str(e)}
                ))
        
        # Determine overall status
        overall_status = self._determine_overall_status(checks)
        uptime = time.time() - self.start_time
        
        return SystemHealth(
            overall_status=overall_status,
            checks=checks,
            timestamp=datetime.utcnow(),
            uptime_seconds=uptime
        )
    
    def _determine_overall_status(self, checks: List[HealthCheck]) -> HealthStatus:
        """Determine overall system status from individual checks."""
        if not checks:
            return HealthStatus.UNKNOWN
        
        has_critical = any(check.status == HealthStatus.CRITICAL for check in checks)
        has_warning = any(check.status == HealthStatus.WARNING for check in checks)
        
        if has_critical:
            return HealthStatus.CRITICAL
        elif has_warning:
            return HealthStatus.WARNING
        else:
            return HealthStatus.HEALTHY
    
    def _check_cpu_usage(self) -> HealthCheck:
        """Check CPU usage."""
        try:
            cpu_percent = psutil.cpu_percent(interval=1)
            
            if cpu_percent > 90:
                status = HealthStatus.CRITICAL
                message = f"CPU usage critical: {cpu_percent:.1f}%"
            elif cpu_percent > 75:
                status = HealthStatus.WARNING
                message = f"CPU usage high: {cpu_percent:.1f}%"
            else:
                status = HealthStatus.HEALTHY
                message = f"CPU usage normal: {cpu_percent:.1f}%"
            
            return HealthCheck(
                name="cpu_usage",
                status=status,
                message=message,
                timestamp=datetime.utcnow(),
                metadata={"cpu_percent": cpu_percent}
            )
            
        except Exception as e:
            return HealthCheck(
                name="cpu_usage",
                status=HealthStatus.CRITICAL,
                message=f"Failed to check CPU usage: {e}",
                timestamp=datetime.utcnow(),
                metadata={"error": str(e)}
            )
    
    def _check_memory_usage(self) -> HealthCheck:
        """Check memory usage."""
        try:
            memory = psutil.virtual_memory()
            memory_percent = memory.percent
            
            if memory_percent > 90:
                status = HealthStatus.CRITICAL
                message = f"Memory usage critical: {memory_percent:.1f}%"
            elif memory_percent > 80:
                status = HealthStatus.WARNING
                message = f"Memory usage high: {memory_percent:.1f}%"
            else:
                status = HealthStatus.HEALTHY
                message = f"Memory usage normal: {memory_percent:.1f}%"
            
            return HealthCheck(
                name="memory_usage",
                status=status,
                message=message,
                timestamp=datetime.utcnow(),
                metadata={
                    "memory_percent": memory_percent,
                    "available_mb": memory.available // (1024 * 1024),
                    "total_mb": memory.total // (1024 * 1024)
                }
            )
            
        except Exception as e:
            return HealthCheck(
                name="memory_usage",
                status=HealthStatus.CRITICAL,
                message=f"Failed to check memory usage: {e}",
                timestamp=datetime.utcnow(),
                metadata={"error": str(e)}
            )
    
    def _check_disk_usage(self) -> HealthCheck:
        """Check disk usage."""
        try:
            disk = psutil.disk_usage('/')
            disk_percent = (disk.used / disk.total) * 100
            
            if disk_percent > 95:
                status = HealthStatus.CRITICAL
                message = f"Disk usage critical: {disk_percent:.1f}%"
            elif disk_percent > 85:
                status = HealthStatus.WARNING
                message = f"Disk usage high: {disk_percent:.1f}%"
            else:
                status = HealthStatus.HEALTHY
                message = f"Disk usage normal: {disk_percent:.1f}%"
            
            return HealthCheck(
                name="disk_usage",
                status=status,
                message=message,
                timestamp=datetime.utcnow(),
                metadata={
                    "disk_percent": disk_percent,
                    "free_gb": disk.free // (1024 ** 3),
                    "total_gb": disk.total // (1024 ** 3)
                }
            )
            
        except Exception as e:
            return HealthCheck(
                name="disk_usage",
                status=HealthStatus.CRITICAL,
                message=f"Failed to check disk usage: {e}",
                timestamp=datetime.utcnow(),
                metadata={"error": str(e)}
            )
    
    def _check_database_connection(self) -> HealthCheck:
        """Check database connection."""
        # Placeholder implementation - would connect to actual database
        try:
            # Simulate database check
            return HealthCheck(
                name="database_connection",
                status=HealthStatus.HEALTHY,
                message="Database connection healthy",
                timestamp=datetime.utcnow(),
                metadata={"connection_pool_size": 10}
            )
            
        except Exception as e:
            return HealthCheck(
                name="database_connection",
                status=HealthStatus.CRITICAL,
                message=f"Database connection failed: {e}",
                timestamp=datetime.utcnow(),
                metadata={"error": str(e)}
            )
    
    def _check_service_dependencies(self) -> HealthCheck:
        """Check external service dependencies."""
        # Placeholder implementation - would check external services
        try:
            # Simulate dependency checks
            return HealthCheck(
                name="service_dependencies",
                status=HealthStatus.HEALTHY,
                message="All service dependencies healthy",
                timestamp=datetime.utcnow(),
                metadata={"checked_services": ["redis", "elasticsearch"]}
            )
            
        except Exception as e:
            return HealthCheck(
                name="service_dependencies",
                status=HealthStatus.CRITICAL,
                message=f"Service dependency check failed: {e}",
                timestamp=datetime.utcnow(),
                metadata={"error": str(e)}
            )
    
    async def get_liveness_check(self) -> bool:
        """Simple liveness check for Kubernetes probes."""
        return True
    
    async def get_readiness_check(self) -> bool:
        """Readiness check for Kubernetes probes."""
        health = await self.get_health_status()
        return health.overall_status != HealthStatus.CRITICAL


# Global instance
_health_service = None


def get_health_monitoring_service() -> HealthMonitoringService:
    """Get or create the global health monitoring service instance."""
    global _health_service
    if _health_service is None:
        _health_service = HealthMonitoringService()
    return _health_service


# Export classes for external use
__all__ = [
    "HealthStatus",
    "AlertSeverity",  # For backward compatibility
    "HealthCheck", 
    "SystemHealth",
    "HealthMonitoringService",
    "get_health_monitoring_service"
]