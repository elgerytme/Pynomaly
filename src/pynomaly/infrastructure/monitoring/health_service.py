"""Advanced health monitoring service."""

from __future__ import annotations

import asyncio
import psutil
import time
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional, Union
from dataclasses import dataclass, field
from enum import Enum

import redis
from sqlalchemy import text
from sqlalchemy.engine import Engine

from pynomaly.domain.exceptions import InfrastructureError


class HealthStatus(Enum):
    """Health check status."""
    HEALTHY = "healthy"
    DEGRADED = "degraded"
    UNHEALTHY = "unhealthy"
    UNKNOWN = "unknown"


@dataclass
class HealthCheck:
    """Individual health check result."""
    name: str
    status: HealthStatus
    message: str
    duration_ms: float
    timestamp: datetime = field(default_factory=datetime.utcnow)
    details: Dict[str, Any] = field(default_factory=dict)


@dataclass
class SystemMetrics:
    """System resource metrics."""
    cpu_percent: float
    memory_percent: float
    disk_percent: float
    memory_available_mb: float
    disk_available_gb: float
    load_average: List[float]
    network_io: Dict[str, int]
    process_count: int
    uptime_seconds: float


class HealthService:
    """Comprehensive health monitoring service."""
    
    def __init__(self, max_history: int = 100):
        """Initialize health service.
        
        Args:
            max_history: Maximum number of health check results to keep
        """
        self.max_history = max_history
        self._check_history: List[Dict[str, HealthCheck]] = []
        self._start_time = time.time()
        
    async def perform_comprehensive_health_check(
        self,
        database_engine: Optional[Engine] = None,
        redis_client: Optional[redis.Redis] = None,
        custom_checks: Optional[Dict[str, Any]] = None
    ) -> Dict[str, HealthCheck]:
        """Perform comprehensive health check across all system components.
        
        Args:
            database_engine: SQLAlchemy engine for database checks
            redis_client: Redis client for cache checks
            custom_checks: Additional custom health checks
            
        Returns:
            Dictionary of health check results
        """
        checks = {}
        
        # System resource checks
        checks.update(await self._check_system_resources())
        
        # Database checks
        if database_engine:
            checks.update(await self._check_database(database_engine))
        
        # Redis/Cache checks
        if redis_client:
            checks.update(await self._check_redis(redis_client))
        
        # Application-specific checks
        checks.update(await self._check_application_health())
        
        # Custom checks
        if custom_checks:
            checks.update(await self._run_custom_checks(custom_checks))
        
        # Store in history
        self._store_check_results(checks)
        
        return checks
    
    async def _check_system_resources(self) -> Dict[str, HealthCheck]:
        """Check system resource health."""
        checks = {}
        start_time = time.time()
        
        try:
            # CPU usage
            cpu_percent = psutil.cpu_percent(interval=0.1)
            cpu_status = (
                HealthStatus.HEALTHY if cpu_percent < 70 else
                HealthStatus.DEGRADED if cpu_percent < 90 else
                HealthStatus.UNHEALTHY
            )
            
            checks["cpu"] = HealthCheck(
                name="CPU Usage",
                status=cpu_status,
                message=f"CPU usage: {cpu_percent:.1f}%",
                duration_ms=(time.time() - start_time) * 1000,
                details={"cpu_percent": cpu_percent}
            )
            
            # Memory usage
            memory = psutil.virtual_memory()
            memory_status = (
                HealthStatus.HEALTHY if memory.percent < 80 else
                HealthStatus.DEGRADED if memory.percent < 95 else
                HealthStatus.UNHEALTHY
            )
            
            checks["memory"] = HealthCheck(
                name="Memory Usage",
                status=memory_status,
                message=f"Memory usage: {memory.percent:.1f}%",
                duration_ms=(time.time() - start_time) * 1000,
                details={
                    "memory_percent": memory.percent,
                    "memory_available_mb": memory.available / 1024 / 1024,
                    "memory_total_mb": memory.total / 1024 / 1024
                }
            )
            
            # Disk usage
            disk = psutil.disk_usage('/')
            disk_percent = (disk.used / disk.total) * 100
            disk_status = (
                HealthStatus.HEALTHY if disk_percent < 85 else
                HealthStatus.DEGRADED if disk_percent < 95 else
                HealthStatus.UNHEALTHY
            )
            
            checks["disk"] = HealthCheck(
                name="Disk Usage",
                status=disk_status,
                message=f"Disk usage: {disk_percent:.1f}%",
                duration_ms=(time.time() - start_time) * 1000,
                details={
                    "disk_percent": disk_percent,
                    "disk_available_gb": disk.free / 1024 / 1024 / 1024,
                    "disk_total_gb": disk.total / 1024 / 1024 / 1024
                }
            )
            
        except Exception as e:
            checks["system"] = HealthCheck(
                name="System Resources",
                status=HealthStatus.UNHEALTHY,
                message=f"Failed to check system resources: {e}",
                duration_ms=(time.time() - start_time) * 1000,
                details={"error": str(e)}
            )
        
        return checks
    
    async def _check_database(self, engine: Engine) -> Dict[str, HealthCheck]:
        """Check database connectivity and performance."""
        checks = {}
        start_time = time.time()
        
        try:
            # Connection test
            with engine.connect() as conn:
                # Simple query test
                result = conn.execute(text("SELECT 1")).fetchone()
                
                if result and result[0] == 1:
                    duration = (time.time() - start_time) * 1000
                    status = (
                        HealthStatus.HEALTHY if duration < 100 else
                        HealthStatus.DEGRADED if duration < 500 else
                        HealthStatus.UNHEALTHY
                    )
                    
                    checks["database"] = HealthCheck(
                        name="Database Connectivity",
                        status=status,
                        message=f"Database responsive in {duration:.1f}ms",
                        duration_ms=duration,
                        details={"response_time_ms": duration}
                    )
                else:
                    checks["database"] = HealthCheck(
                        name="Database Connectivity",
                        status=HealthStatus.UNHEALTHY,
                        message="Database query returned unexpected result",
                        duration_ms=(time.time() - start_time) * 1000
                    )
                    
        except Exception as e:
            checks["database"] = HealthCheck(
                name="Database Connectivity",
                status=HealthStatus.UNHEALTHY,
                message=f"Database connection failed: {e}",
                duration_ms=(time.time() - start_time) * 1000,
                details={"error": str(e)}
            )
        
        return checks
    
    async def _check_redis(self, redis_client: redis.Redis) -> Dict[str, HealthCheck]:
        """Check Redis connectivity and performance."""
        checks = {}
        start_time = time.time()
        
        try:
            # Ping test
            response = redis_client.ping()
            
            if response:
                duration = (time.time() - start_time) * 1000
                status = (
                    HealthStatus.HEALTHY if duration < 10 else
                    HealthStatus.DEGRADED if duration < 50 else
                    HealthStatus.UNHEALTHY
                )
                
                # Get Redis info
                info = redis_client.info()
                
                checks["redis"] = HealthCheck(
                    name="Redis Cache",
                    status=status,
                    message=f"Redis responsive in {duration:.1f}ms",
                    duration_ms=duration,
                    details={
                        "response_time_ms": duration,
                        "connected_clients": info.get("connected_clients", 0),
                        "used_memory_mb": info.get("used_memory", 0) / 1024 / 1024,
                        "keyspace_hits": info.get("keyspace_hits", 0),
                        "keyspace_misses": info.get("keyspace_misses", 0)
                    }
                )
            else:
                checks["redis"] = HealthCheck(
                    name="Redis Cache",
                    status=HealthStatus.UNHEALTHY,
                    message="Redis ping failed",
                    duration_ms=(time.time() - start_time) * 1000
                )
                
        except Exception as e:
            checks["redis"] = HealthCheck(
                name="Redis Cache",
                status=HealthStatus.UNHEALTHY,
                message=f"Redis connection failed: {e}",
                duration_ms=(time.time() - start_time) * 1000,
                details={"error": str(e)}
            )
        
        return checks
    
    async def _check_application_health(self) -> Dict[str, HealthCheck]:
        """Check application-specific health indicators."""
        checks = {}
        start_time = time.time()
        
        try:
            # Application uptime
            uptime_seconds = time.time() - self._start_time
            
            checks["uptime"] = HealthCheck(
                name="Application Uptime",
                status=HealthStatus.HEALTHY,
                message=f"Running for {uptime_seconds:.0f} seconds",
                duration_ms=(time.time() - start_time) * 1000,
                details={
                    "uptime_seconds": uptime_seconds,
                    "uptime_hours": uptime_seconds / 3600,
                    "start_time": datetime.fromtimestamp(self._start_time).isoformat()
                }
            )
            
            # Memory leaks detection (basic)
            current_process = psutil.Process()
            memory_info = current_process.memory_info()
            
            # Simple heuristic: if RSS > 1GB, flag as potentially problematic
            memory_mb = memory_info.rss / 1024 / 1024
            memory_status = (
                HealthStatus.HEALTHY if memory_mb < 512 else
                HealthStatus.DEGRADED if memory_mb < 1024 else
                HealthStatus.UNHEALTHY
            )
            
            checks["application_memory"] = HealthCheck(
                name="Application Memory",
                status=memory_status,
                message=f"Process using {memory_mb:.1f}MB",
                duration_ms=(time.time() - start_time) * 1000,
                details={
                    "rss_mb": memory_mb,
                    "vms_mb": memory_info.vms / 1024 / 1024,
                    "num_threads": current_process.num_threads()
                }
            )
            
        except Exception as e:
            checks["application"] = HealthCheck(
                name="Application Health",
                status=HealthStatus.UNHEALTHY,
                message=f"Failed to check application health: {e}",
                duration_ms=(time.time() - start_time) * 1000,
                details={"error": str(e)}
            )
        
        return checks
    
    async def _run_custom_checks(self, custom_checks: Dict[str, Any]) -> Dict[str, HealthCheck]:
        """Run custom health checks."""
        checks = {}
        
        for name, check_func in custom_checks.items():
            start_time = time.time()
            
            try:
                if asyncio.iscoroutinefunction(check_func):
                    result = await check_func()
                else:
                    result = check_func()
                
                # Expect result to be a dict with status and message
                if isinstance(result, dict) and "status" in result:
                    checks[name] = HealthCheck(
                        name=name,
                        status=HealthStatus(result["status"]),
                        message=result.get("message", "Custom check completed"),
                        duration_ms=(time.time() - start_time) * 1000,
                        details=result.get("details", {})
                    )
                else:
                    checks[name] = HealthCheck(
                        name=name,
                        status=HealthStatus.HEALTHY if result else HealthStatus.UNHEALTHY,
                        message=f"Custom check returned: {result}",
                        duration_ms=(time.time() - start_time) * 1000
                    )
                    
            except Exception as e:
                checks[name] = HealthCheck(
                    name=name,
                    status=HealthStatus.UNHEALTHY,
                    message=f"Custom check failed: {e}",
                    duration_ms=(time.time() - start_time) * 1000,
                    details={"error": str(e)}
                )
        
        return checks
    
    def _store_check_results(self, checks: Dict[str, HealthCheck]) -> None:
        """Store health check results in history."""
        self._check_history.append(checks)
        
        # Keep only the most recent results
        if len(self._check_history) > self.max_history:
            self._check_history = self._check_history[-self.max_history:]
    
    def get_overall_status(self, checks: Dict[str, HealthCheck]) -> HealthStatus:
        """Get overall system health status."""
        if not checks:
            return HealthStatus.UNKNOWN
        
        statuses = [check.status for check in checks.values()]
        
        # If any check is unhealthy, overall is unhealthy
        if HealthStatus.UNHEALTHY in statuses:
            return HealthStatus.UNHEALTHY
        
        # If any check is degraded, overall is degraded
        if HealthStatus.DEGRADED in statuses:
            return HealthStatus.DEGRADED
        
        # All checks are healthy
        return HealthStatus.HEALTHY
    
    def get_system_metrics(self) -> SystemMetrics:
        """Get current system metrics."""
        try:
            # CPU and memory
            cpu_percent = psutil.cpu_percent(interval=0.1)
            memory = psutil.virtual_memory()
            disk = psutil.disk_usage('/')
            
            # Load average
            try:
                load_avg = list(psutil.getloadavg())
            except AttributeError:
                # Windows doesn't have load average
                load_avg = [0.0, 0.0, 0.0]
            
            # Network I/O
            try:
                net_io = psutil.net_io_counters()
                network_io = {
                    "bytes_sent": net_io.bytes_sent,
                    "bytes_recv": net_io.bytes_recv,
                    "packets_sent": net_io.packets_sent,
                    "packets_recv": net_io.packets_recv
                }
            except:
                network_io = {"bytes_sent": 0, "bytes_recv": 0}
            
            return SystemMetrics(
                cpu_percent=cpu_percent,
                memory_percent=memory.percent,
                disk_percent=(disk.used / disk.total) * 100,
                memory_available_mb=memory.available / 1024 / 1024,
                disk_available_gb=disk.free / 1024 / 1024 / 1024,
                load_average=load_avg,
                network_io=network_io,
                process_count=len(psutil.pids()),
                uptime_seconds=time.time() - self._start_time
            )
            
        except Exception as e:
            raise InfrastructureError(f"Failed to get system metrics: {e}")
    
    def get_health_history(self, hours: int = 24) -> List[Dict[str, HealthCheck]]:
        """Get health check history for the specified number of hours."""
        cutoff_time = datetime.utcnow() - timedelta(hours=hours)
        
        return [
            checks for checks in self._check_history
            if any(
                check.timestamp >= cutoff_time
                for check in checks.values()
            )
        ]
    
    def get_health_summary(self) -> Dict[str, Any]:
        """Get a summary of current health status."""
        if not self._check_history:
            return {"status": "unknown", "message": "No health checks performed"}
        
        latest_checks = self._check_history[-1]
        overall_status = self.get_overall_status(latest_checks)
        metrics = self.get_system_metrics()
        
        return {
            "overall_status": overall_status.value,
            "total_checks": len(latest_checks),
            "healthy_checks": sum(1 for c in latest_checks.values() if c.status == HealthStatus.HEALTHY),
            "degraded_checks": sum(1 for c in latest_checks.values() if c.status == HealthStatus.DEGRADED),
            "unhealthy_checks": sum(1 for c in latest_checks.values() if c.status == HealthStatus.UNHEALTHY),
            "last_check_time": max(c.timestamp for c in latest_checks.values()).isoformat(),
            "uptime_hours": metrics.uptime_seconds / 3600,
            "cpu_percent": metrics.cpu_percent,
            "memory_percent": metrics.memory_percent,
            "disk_percent": metrics.disk_percent
        }