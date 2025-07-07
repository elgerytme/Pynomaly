"""Comprehensive health monitoring for Pynomaly."""

import asyncio
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, Dict, List, Optional, Union

import psutil
from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncSession


class HealthStatus(str, Enum):
    """Health check status enumeration."""
    HEALTHY = "healthy"
    DEGRADED = "degraded"
    UNHEALTHY = "unhealthy"
    UNKNOWN = "unknown"


@dataclass
class HealthCheckResult:
    """Result of a health check."""
    name: str
    status: HealthStatus
    message: str = ""
    details: Dict[str, Any] = field(default_factory=dict)
    timestamp: float = field(default_factory=time.time)
    duration_ms: float = 0.0


@dataclass
class ServiceHealthCheck:
    """Definition of a service health check."""
    name: str
    check_function: Callable[[], Union[HealthCheckResult, bool, Dict[str, Any]]]
    timeout_seconds: float = 5.0
    critical: bool = True
    interval_seconds: float = 30.0


class HealthMonitor:
    """Comprehensive health monitoring system."""
    
    def __init__(self):
        """Initialize health monitor."""
        self.checks: Dict[str, ServiceHealthCheck] = {}
        self.last_results: Dict[str, HealthCheckResult] = {}
        self._monitoring_task: Optional[asyncio.Task] = None
        self._is_monitoring = False
    
    def register_check(self, check: ServiceHealthCheck) -> None:
        """Register a health check.
        
        Args:
            check: Health check definition
        """
        self.checks[check.name] = check
    
    def unregister_check(self, name: str) -> None:
        """Unregister a health check.
        
        Args:
            name: Name of the health check to remove
        """
        self.checks.pop(name, None)
        self.last_results.pop(name, None)
    
    async def run_check(self, name: str) -> HealthCheckResult:
        """Run a specific health check.
        
        Args:
            name: Name of the health check to run
            
        Returns:
            Health check result
        """
        if name not in self.checks:
            return HealthCheckResult(
                name=name,
                status=HealthStatus.UNKNOWN,
                message=f"Health check '{name}' not found"
            )
        
        check = self.checks[name]
        start_time = time.time()
        
        try:
            # Run the check with timeout
            result = await asyncio.wait_for(
                asyncio.create_task(self._execute_check(check)),
                timeout=check.timeout_seconds
            )
            
            duration_ms = (time.time() - start_time) * 1000
            
            # Process result based on type
            if isinstance(result, HealthCheckResult):
                result.duration_ms = duration_ms
                return result
            elif isinstance(result, bool):
                return HealthCheckResult(
                    name=name,
                    status=HealthStatus.HEALTHY if result else HealthStatus.UNHEALTHY,
                    message="Check passed" if result else "Check failed",
                    duration_ms=duration_ms
                )
            elif isinstance(result, dict):
                status = HealthStatus.HEALTHY
                if "status" in result:
                    status = HealthStatus(result["status"])
                elif "healthy" in result:
                    status = HealthStatus.HEALTHY if result["healthy"] else HealthStatus.UNHEALTHY
                
                return HealthCheckResult(
                    name=name,
                    status=status,
                    message=result.get("message", ""),
                    details=result.get("details", {}),
                    duration_ms=duration_ms
                )
            else:
                return HealthCheckResult(
                    name=name,
                    status=HealthStatus.HEALTHY,
                    message=str(result),
                    duration_ms=duration_ms
                )
                
        except asyncio.TimeoutError:
            return HealthCheckResult(
                name=name,
                status=HealthStatus.UNHEALTHY,
                message=f"Health check timed out after {check.timeout_seconds}s",
                duration_ms=(time.time() - start_time) * 1000
            )
        except Exception as e:
            return HealthCheckResult(
                name=name,
                status=HealthStatus.UNHEALTHY,
                message=f"Health check failed: {str(e)}",
                duration_ms=(time.time() - start_time) * 1000
            )
    
    async def _execute_check(self, check: ServiceHealthCheck) -> Any:
        """Execute a health check function.
        
        Args:
            check: Health check to execute
            
        Returns:
            Check result
        """
        if asyncio.iscoroutinefunction(check.check_function):
            return await check.check_function()
        else:
            return check.check_function()
    
    async def run_all_checks(self) -> Dict[str, HealthCheckResult]:
        """Run all registered health checks.
        
        Returns:
            Dictionary of health check results
        """
        tasks = [
            self.run_check(name) for name in self.checks.keys()
        ]
        
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        check_results = {}
        for i, name in enumerate(self.checks.keys()):
            result = results[i]
            if isinstance(result, Exception):
                check_results[name] = HealthCheckResult(
                    name=name,
                    status=HealthStatus.UNHEALTHY,
                    message=f"Check execution failed: {str(result)}"
                )
            else:
                check_results[name] = result
                self.last_results[name] = result
        
        return check_results
    
    def get_overall_health(self, results: Optional[Dict[str, HealthCheckResult]] = None) -> HealthCheckResult:
        """Get overall system health status.
        
        Args:
            results: Optional health check results (uses last results if not provided)
            
        Returns:
            Overall health status
        """
        if results is None:
            results = self.last_results
        
        if not results:
            return HealthCheckResult(
                name="overall",
                status=HealthStatus.UNKNOWN,
                message="No health checks have been run"
            )
        
        critical_checks = [
            name for name, check in self.checks.items() 
            if check.critical and name in results
        ]
        
        # Check critical services first
        unhealthy_critical = [
            name for name in critical_checks
            if results[name].status == HealthStatus.UNHEALTHY
        ]
        
        if unhealthy_critical:
            return HealthCheckResult(
                name="overall",
                status=HealthStatus.UNHEALTHY,
                message=f"Critical services unhealthy: {', '.join(unhealthy_critical)}",
                details={
                    "unhealthy_critical": unhealthy_critical,
                    "total_checks": len(results),
                    "critical_checks": len(critical_checks)
                }
            )
        
        # Check for degraded services
        degraded_checks = [
            name for name, result in results.items()
            if result.status == HealthStatus.DEGRADED
        ]
        
        unhealthy_non_critical = [
            name for name, result in results.items()
            if result.status == HealthStatus.UNHEALTHY and name not in critical_checks
        ]
        
        if degraded_checks or unhealthy_non_critical:
            return HealthCheckResult(
                name="overall",
                status=HealthStatus.DEGRADED,
                message="Some services are degraded or non-critical services are unhealthy",
                details={
                    "degraded": degraded_checks,
                    "unhealthy_non_critical": unhealthy_non_critical,
                    "total_checks": len(results)
                }
            )
        
        # All checks healthy
        return HealthCheckResult(
            name="overall",
            status=HealthStatus.HEALTHY,
            message="All services healthy",
            details={
                "total_checks": len(results),
                "critical_checks": len(critical_checks)
            }
        )
    
    async def start_monitoring(self) -> None:
        """Start continuous health monitoring."""
        if self._is_monitoring:
            return
        
        self._is_monitoring = True
        self._monitoring_task = asyncio.create_task(self._monitoring_loop())
    
    async def stop_monitoring(self) -> None:
        """Stop continuous health monitoring."""
        self._is_monitoring = False
        if self._monitoring_task:
            self._monitoring_task.cancel()
            try:
                await self._monitoring_task
            except asyncio.CancelledError:
                pass
            self._monitoring_task = None
    
    async def _monitoring_loop(self) -> None:
        """Continuous monitoring loop."""
        while self._is_monitoring:
            try:
                await self.run_all_checks()
                await asyncio.sleep(min(check.interval_seconds for check in self.checks.values()))
            except asyncio.CancelledError:
                break
            except Exception as e:
                # Log error but continue monitoring
                import logging
                logging.getLogger(__name__).error(f"Error in monitoring loop: {e}")
                await asyncio.sleep(5)


# Built-in health checks
def create_system_health_check() -> ServiceHealthCheck:
    """Create system resource health check."""
    
    def check_system_resources() -> Dict[str, Any]:
        """Check system resources (CPU, memory, disk)."""
        try:
            # CPU usage
            cpu_percent = psutil.cpu_percent(interval=1)
            
            # Memory usage
            memory = psutil.virtual_memory()
            memory_percent = memory.percent
            
            # Disk usage
            disk = psutil.disk_usage('/')
            disk_percent = disk.percent
            
            # Determine status
            status = HealthStatus.HEALTHY
            issues = []
            
            if cpu_percent > 90:
                status = HealthStatus.UNHEALTHY
                issues.append(f"High CPU usage: {cpu_percent:.1f}%")
            elif cpu_percent > 80:
                status = HealthStatus.DEGRADED
                issues.append(f"Elevated CPU usage: {cpu_percent:.1f}%")
            
            if memory_percent > 95:
                status = HealthStatus.UNHEALTHY
                issues.append(f"High memory usage: {memory_percent:.1f}%")
            elif memory_percent > 85:
                if status == HealthStatus.HEALTHY:
                    status = HealthStatus.DEGRADED
                issues.append(f"Elevated memory usage: {memory_percent:.1f}%")
            
            if disk_percent > 95:
                status = HealthStatus.UNHEALTHY
                issues.append(f"High disk usage: {disk_percent:.1f}%")
            elif disk_percent > 85:
                if status == HealthStatus.HEALTHY:
                    status = HealthStatus.DEGRADED
                issues.append(f"Elevated disk usage: {disk_percent:.1f}%")
            
            return {
                "status": status.value,
                "message": "; ".join(issues) if issues else "System resources normal",
                "details": {
                    "cpu_percent": cpu_percent,
                    "memory_percent": memory_percent,
                    "disk_percent": disk_percent,
                    "memory_available_gb": memory.available / (1024**3),
                    "disk_free_gb": disk.free / (1024**3)
                }
            }
            
        except Exception as e:
            return {
                "status": HealthStatus.UNHEALTHY.value,
                "message": f"Failed to check system resources: {str(e)}"
            }
    
    return ServiceHealthCheck(
        name="system_resources",
        check_function=check_system_resources,
        timeout_seconds=5.0,
        critical=True,
        interval_seconds=30.0
    )


def create_database_health_check(session_factory: Callable) -> ServiceHealthCheck:
    """Create database connectivity health check.
    
    Args:
        session_factory: Function to create database session
        
    Returns:
        Database health check
    """
    
    async def check_database() -> Dict[str, Any]:
        """Check database connectivity and performance."""
        try:
            start_time = time.time()
            
            async with session_factory() as session:
                # Simple connectivity test
                result = await session.execute(text("SELECT 1"))
                await result.fetchone()
                
                # Check connection pool if available
                pool_info = {}
                if hasattr(session.bind, 'pool'):
                    pool = session.bind.pool
                    pool_info = {
                        "pool_size": pool.size(),
                        "checked_in": pool.checkedin(),
                        "checked_out": pool.checkedout(),
                        "invalidated": pool.invalidated()
                    }
            
            response_time = (time.time() - start_time) * 1000
            
            # Determine status based on response time
            status = HealthStatus.HEALTHY
            if response_time > 1000:  # > 1 second
                status = HealthStatus.UNHEALTHY
            elif response_time > 500:  # > 500ms
                status = HealthStatus.DEGRADED
            
            return {
                "status": status.value,
                "message": f"Database responsive ({response_time:.1f}ms)",
                "details": {
                    "response_time_ms": response_time,
                    **pool_info
                }
            }
            
        except Exception as e:
            return {
                "status": HealthStatus.UNHEALTHY.value,
                "message": f"Database connection failed: {str(e)}"
            }
    
    return ServiceHealthCheck(
        name="database",
        check_function=check_database,
        timeout_seconds=5.0,
        critical=True,
        interval_seconds=30.0
    )


def create_redis_health_check(redis_client) -> ServiceHealthCheck:
    """Create Redis connectivity health check.
    
    Args:
        redis_client: Redis client instance
        
    Returns:
        Redis health check
    """
    
    async def check_redis() -> Dict[str, Any]:
        """Check Redis connectivity and performance."""
        try:
            start_time = time.time()
            
            # Ping Redis
            if hasattr(redis_client, 'ping'):
                if asyncio.iscoroutinefunction(redis_client.ping):
                    await redis_client.ping()
                else:
                    redis_client.ping()
            
            response_time = (time.time() - start_time) * 1000
            
            # Get Redis info
            info = {}
            if hasattr(redis_client, 'info'):
                try:
                    if asyncio.iscoroutinefunction(redis_client.info):
                        redis_info = await redis_client.info()
                    else:
                        redis_info = redis_client.info()
                    
                    info = {
                        "connected_clients": redis_info.get("connected_clients", 0),
                        "used_memory_human": redis_info.get("used_memory_human", "unknown"),
                        "uptime_in_seconds": redis_info.get("uptime_in_seconds", 0)
                    }
                except Exception:
                    pass
            
            status = HealthStatus.HEALTHY
            if response_time > 100:  # > 100ms
                status = HealthStatus.DEGRADED
            
            return {
                "status": status.value,
                "message": f"Redis responsive ({response_time:.1f}ms)",
                "details": {
                    "response_time_ms": response_time,
                    **info
                }
            }
            
        except Exception as e:
            return {
                "status": HealthStatus.UNHEALTHY.value,
                "message": f"Redis connection failed: {str(e)}"
            }
    
    return ServiceHealthCheck(
        name="redis",
        check_function=check_redis,
        timeout_seconds=3.0,
        critical=False,
        interval_seconds=30.0
    )


# Global health monitor instance
_global_health_monitor: Optional[HealthMonitor] = None


def get_health_monitor() -> HealthMonitor:
    """Get global health monitor instance.
    
    Returns:
        Global health monitor
    """
    global _global_health_monitor
    if _global_health_monitor is None:
        _global_health_monitor = HealthMonitor()
    return _global_health_monitor


def setup_default_health_checks(
    health_monitor: Optional[HealthMonitor] = None,
    session_factory: Optional[Callable] = None,
    redis_client: Optional[Any] = None,
) -> HealthMonitor:
    """Set up default health checks.
    
    Args:
        health_monitor: Health monitor instance (creates new if None)
        session_factory: Database session factory
        redis_client: Redis client instance
        
    Returns:
        Configured health monitor
    """
    if health_monitor is None:
        health_monitor = get_health_monitor()
    
    # System resources check
    health_monitor.register_check(create_system_health_check())
    
    # Database check
    if session_factory:
        health_monitor.register_check(create_database_health_check(session_factory))
    
    # Redis check
    if redis_client:
        health_monitor.register_check(create_redis_health_check(redis_client))
    
    return health_monitor