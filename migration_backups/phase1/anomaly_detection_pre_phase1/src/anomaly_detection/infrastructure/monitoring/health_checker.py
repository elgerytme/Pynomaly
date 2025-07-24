"""Health checking and service status monitoring."""

from __future__ import annotations

import asyncio
import time
from typing import Dict, Any, List, Optional, Callable, Awaitable
from dataclasses import dataclass
from datetime import datetime, timedelta
from enum import Enum
import threading

from ..logging import get_logger
from ..config.settings import get_settings

logger = get_logger(__name__)
settings = get_settings()


class HealthStatus(Enum):
    """Health check status levels."""
    HEALTHY = "healthy"
    DEGRADED = "degraded"
    UNHEALTHY = "unhealthy"
    UNKNOWN = "unknown"


@dataclass
class HealthCheckResult:
    """Result of a health check."""
    
    name: str
    status: HealthStatus
    message: str
    duration_ms: float
    timestamp: datetime
    details: Dict[str, Any]


class HealthCheck:
    """Base class for health checks."""
    
    def __init__(
        self,
        name: str,
        timeout_seconds: float = 30.0,
        interval_seconds: float = 60.0,
        critical: bool = False
    ):
        """Initialize health check.
        
        Args:
            name: Unique name for this health check
            timeout_seconds: Maximum time to wait for check completion
            interval_seconds: How often to run this check
            critical: Whether this check failure should mark service as unhealthy
        """
        self.name = name
        self.timeout_seconds = timeout_seconds
        self.interval_seconds = interval_seconds
        self.critical = critical
        self.last_result: Optional[HealthCheckResult] = None
        self.last_run: Optional[datetime] = None
    
    async def check(self) -> HealthCheckResult:
        """Perform the health check. Override in subclasses."""
        raise NotImplementedError("Subclasses must implement check method")
    
    def is_due(self) -> bool:
        """Check if this health check is due to run."""
        if self.last_run is None:
            return True
        
        time_since_last = datetime.utcnow() - self.last_run
        return time_since_last.total_seconds() >= self.interval_seconds


class DatabaseHealthCheck(HealthCheck):
    """Health check for database connectivity."""
    
    def __init__(self, **kwargs):
        super().__init__(name="database", critical=True, **kwargs)
    
    async def check(self) -> HealthCheckResult:
        """Check database connectivity."""
        start_time = time.perf_counter()
        
        try:
            # Simulate database check (replace with actual DB connectivity test)
            await asyncio.sleep(0.1)  # Simulate DB query
            
            duration_ms = (time.perf_counter() - start_time) * 1000
            
            return HealthCheckResult(
                name=self.name,
                status=HealthStatus.HEALTHY,
                message="Database connection successful",
                duration_ms=duration_ms,
                timestamp=datetime.utcnow(),
                details={
                    "connection_pool_size": 10,  # Example metric
                    "active_connections": 2
                }
            )
        
        except Exception as e:
            duration_ms = (time.perf_counter() - start_time) * 1000
            
            return HealthCheckResult(
                name=self.name,
                status=HealthStatus.UNHEALTHY,
                message=f"Database connection failed: {str(e)}",
                duration_ms=duration_ms,
                timestamp=datetime.utcnow(),
                details={"error": str(e)}
            )


class ModelRepositoryHealthCheck(HealthCheck):
    """Health check for model repository."""
    
    def __init__(self, **kwargs):
        super().__init__(name="model_repository", critical=False, **kwargs)
    
    async def check(self) -> HealthCheckResult:
        """Check model repository accessibility."""
        start_time = time.perf_counter()
        
        try:
            from ...infrastructure.repositories.model_repository import ModelRepository
            
            # Check if model directory is accessible
            repo = ModelRepository()
            stats = repo.get_repository_stats()
            
            duration_ms = (time.perf_counter() - start_time) * 1000
            
            return HealthCheckResult(
                name=self.name,
                status=HealthStatus.HEALTHY,
                message="Model repository accessible",
                duration_ms=duration_ms,
                timestamp=datetime.utcnow(),
                details={
                    "total_models": stats["total_models"],
                    "storage_size_mb": stats["storage_size_mb"],
                    "storage_path": stats["storage_path"]
                }
            )
        
        except Exception as e:
            duration_ms = (time.perf_counter() - start_time) * 1000
            
            return HealthCheckResult(
                name=self.name,
                status=HealthStatus.UNHEALTHY,
                message=f"Model repository check failed: {str(e)}",
                duration_ms=duration_ms,
                timestamp=datetime.utcnow(),
                details={"error": str(e)}
            )


class AlgorithmHealthCheck(HealthCheck):
    """Health check for algorithm availability."""
    
    def __init__(self, **kwargs):
        super().__init__(name="algorithms", critical=True, **kwargs)
    
    async def check(self) -> HealthCheckResult:
        """Check that required ML algorithms are available."""
        start_time = time.perf_counter()
        
        try:
            from ...domain.services.detection_service import DetectionService
            import numpy as np
            
            # Test basic algorithm functionality
            service = DetectionService()
            test_data = np.random.randn(10, 2).astype(np.float64)
            
            # Test isolation forest
            result = service.detect_anomalies(test_data, algorithm="iforest", contamination=0.1)
            
            duration_ms = (time.perf_counter() - start_time) * 1000
            
            available_algorithms = service.list_available_algorithms()
            
            return HealthCheckResult(
                name=self.name,
                status=HealthStatus.HEALTHY,
                message="All algorithms available and functional",
                duration_ms=duration_ms,
                timestamp=datetime.utcnow(),
                details={
                    "available_algorithms": available_algorithms,
                    "test_result": {
                        "success": result.success,
                        "anomalies_detected": result.anomaly_count
                    }
                }
            )
        
        except Exception as e:
            duration_ms = (time.perf_counter() - start_time) * 1000
            
            return HealthCheckResult(
                name=self.name,
                status=HealthStatus.UNHEALTHY,
                message=f"Algorithm check failed: {str(e)}",
                duration_ms=duration_ms,
                timestamp=datetime.utcnow(),
                details={"error": str(e)}
            )


class MemoryHealthCheck(HealthCheck):
    """Health check for memory usage."""
    
    def __init__(self, memory_threshold_percent: float = 85.0, **kwargs):
        super().__init__(name="memory", critical=False, **kwargs)
        self.memory_threshold_percent = memory_threshold_percent
    
    async def check(self) -> HealthCheckResult:
        """Check memory usage."""
        start_time = time.perf_counter()
        
        try:
            import psutil
            
            memory_info = psutil.virtual_memory()
            memory_percent = memory_info.percent
            
            duration_ms = (time.perf_counter() - start_time) * 1000
            
            if memory_percent > self.memory_threshold_percent:
                status = HealthStatus.DEGRADED
                message = f"Memory usage high: {memory_percent:.1f}%"
            else:
                status = HealthStatus.HEALTHY
                message = f"Memory usage normal: {memory_percent:.1f}%"
            
            return HealthCheckResult(
                name=self.name,
                status=status,
                message=message,
                duration_ms=duration_ms,
                timestamp=datetime.utcnow(),
                details={
                    "memory_percent": memory_percent,
                    "available_bytes": memory_info.available,
                    "total_bytes": memory_info.total,
                    "used_bytes": memory_info.used
                }
            )
        
        except ImportError:
            # psutil not available
            duration_ms = (time.perf_counter() - start_time) * 1000
            
            return HealthCheckResult(
                name=self.name,
                status=HealthStatus.UNKNOWN,
                message="Memory monitoring not available (psutil not installed)",
                duration_ms=duration_ms,
                timestamp=datetime.utcnow(),
                details={}
            )
        except Exception as e:
            duration_ms = (time.perf_counter() - start_time) * 1000
            
            return HealthCheckResult(
                name=self.name,
                status=HealthStatus.UNHEALTHY,
                message=f"Memory check failed: {str(e)}",
                duration_ms=duration_ms,
                timestamp=datetime.utcnow(),
                details={"error": str(e)}
            )


class DiskHealthCheck(HealthCheck):
    """Health check for disk space."""
    
    def __init__(self, disk_threshold_percent: float = 85.0, **kwargs):
        super().__init__(name="disk", critical=False, **kwargs)
        self.disk_threshold_percent = disk_threshold_percent
    
    async def check(self) -> HealthCheckResult:
        """Check disk space usage."""
        start_time = time.perf_counter()
        
        try:
            import psutil
            import os
            
            # Check disk usage for current working directory
            disk_usage = psutil.disk_usage(os.getcwd())
            disk_percent = (disk_usage.used / disk_usage.total) * 100
            
            duration_ms = (time.perf_counter() - start_time) * 1000
            
            if disk_percent > self.disk_threshold_percent:
                status = HealthStatus.DEGRADED
                message = f"Disk usage high: {disk_percent:.1f}%"
            else:
                status = HealthStatus.HEALTHY
                message = f"Disk usage normal: {disk_percent:.1f}%"
            
            return HealthCheckResult(
                name=self.name,
                status=status,
                message=message,
                duration_ms=duration_ms,
                timestamp=datetime.utcnow(),
                details={
                    "disk_percent": disk_percent,
                    "free_bytes": disk_usage.free,
                    "total_bytes": disk_usage.total,
                    "used_bytes": disk_usage.used,
                    "path": os.getcwd()
                }
            )
        
        except ImportError:
            # psutil not available
            duration_ms = (time.perf_counter() - start_time) * 1000
            
            return HealthCheckResult(
                name=self.name,
                status=HealthStatus.UNKNOWN,
                message="Disk monitoring not available (psutil not installed)",
                duration_ms=duration_ms,
                timestamp=datetime.utcnow(),
                details={}
            )
        except Exception as e:
            duration_ms = (time.perf_counter() - start_time) * 1000
            
            return HealthCheckResult(
                name=self.name,
                status=HealthStatus.UNHEALTHY,
                message=f"Disk check failed: {str(e)}",
                duration_ms=duration_ms,
                timestamp=datetime.utcnow(),
                details={"error": str(e)}
            )


class HealthChecker:
    """Centralized health checking service."""
    
    def __init__(self):
        """Initialize health checker with default checks."""
        self.checks: Dict[str, HealthCheck] = {}
        self.results: Dict[str, HealthCheckResult] = {}
        self.overall_status = HealthStatus.UNKNOWN
        self.last_check_time: Optional[datetime] = None
        
        # Thread safety
        self._lock = threading.RLock()
        
        # Register default health checks
        self._register_default_checks()
        
        logger.info("Health checker initialized", check_count=len(self.checks))
    
    def _register_default_checks(self) -> None:
        """Register default health checks."""
        self.register_check(AlgorithmHealthCheck())
        self.register_check(ModelRepositoryHealthCheck())
        self.register_check(MemoryHealthCheck())
        self.register_check(DiskHealthCheck())
        # Note: DatabaseHealthCheck disabled by default since we don't have a real DB
    
    def register_check(self, health_check: HealthCheck) -> None:
        """Register a health check.
        
        Args:
            health_check: Health check to register
        """
        with self._lock:
            self.checks[health_check.name] = health_check
        
        logger.info("Health check registered", 
                   check_name=health_check.name,
                   critical=health_check.critical)
    
    def unregister_check(self, name: str) -> bool:
        """Unregister a health check.
        
        Args:
            name: Name of health check to remove
            
        Returns:
            True if check was removed, False if not found
        """
        with self._lock:
            if name in self.checks:
                del self.checks[name]
                if name in self.results:
                    del self.results[name]
                logger.info("Health check unregistered", check_name=name)
                return True
        
        return False
    
    async def run_check(self, name: str) -> Optional[HealthCheckResult]:
        """Run a specific health check.
        
        Args:
            name: Name of the health check to run
            
        Returns:
            Health check result or None if check not found
        """
        with self._lock:
            check = self.checks.get(name)
        
        if not check:
            logger.warning("Health check not found", check_name=name)
            return None
        
        logger.debug("Running health check", check_name=name)
        
        try:
            # Run the check with timeout
            result = await asyncio.wait_for(
                check.check(),
                timeout=check.timeout_seconds
            )
            
            # Update check metadata
            check.last_result = result
            check.last_run = datetime.utcnow()
            
            # Store result
            with self._lock:
                self.results[name] = result
            
            logger.debug("Health check completed",
                        check_name=name,
                        status=result.status.value,
                        duration_ms=result.duration_ms)
            
            return result
            
        except asyncio.TimeoutError:
            logger.warning("Health check timed out", 
                          check_name=name,
                          timeout_seconds=check.timeout_seconds)
            
            timeout_result = HealthCheckResult(
                name=name,
                status=HealthStatus.UNHEALTHY,
                message=f"Health check timed out after {check.timeout_seconds}s",
                duration_ms=check.timeout_seconds * 1000,
                timestamp=datetime.utcnow(),
                details={"timeout": True}
            )
            
            check.last_result = timeout_result
            check.last_run = datetime.utcnow()
            
            with self._lock:
                self.results[name] = timeout_result
            
            return timeout_result
        
        except Exception as e:
            logger.error("Health check failed with exception",
                        check_name=name,
                        error=str(e))
            
            error_result = HealthCheckResult(
                name=name,
                status=HealthStatus.UNHEALTHY,
                message=f"Health check failed: {str(e)}",
                duration_ms=0,
                timestamp=datetime.utcnow(),
                details={"error": str(e), "exception_type": type(e).__name__}
            )
            
            check.last_result = error_result
            check.last_run = datetime.utcnow()
            
            with self._lock:
                self.results[name] = error_result
            
            return error_result
    
    async def run_all_checks(self, force: bool = False) -> Dict[str, HealthCheckResult]:
        """Run all registered health checks.
        
        Args:
            force: If True, run all checks regardless of schedule
            
        Returns:
            Dictionary of check results
        """
        start_time = time.perf_counter()
        
        with self._lock:
            checks_to_run = []
            for check in self.checks.values():
                if force or check.is_due():
                    checks_to_run.append(check.name)
        
        if not checks_to_run:
            logger.debug("No health checks due to run")
            return self.results.copy()
        
        logger.info("Running health checks", 
                   checks_to_run=checks_to_run,
                   force=force)
        
        # Run checks concurrently
        tasks = [self.run_check(name) for name in checks_to_run]
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # Update overall status
        self._update_overall_status()
        
        duration_ms = (time.perf_counter() - start_time) * 1000
        
        logger.info("Health checks completed",
                   checks_run=len(checks_to_run),
                   overall_status=self.overall_status.value,
                   duration_ms=duration_ms)
        
        return self.results.copy()
    
    def _update_overall_status(self) -> None:
        """Update overall health status based on individual checks."""
        with self._lock:
            if not self.results:
                self.overall_status = HealthStatus.UNKNOWN
                return
            
            # Check critical failures first
            critical_checks = [
                check for check in self.checks.values() 
                if check.critical and check.name in self.results
            ]
            
            for check in critical_checks:
                result = self.results[check.name]
                if result.status == HealthStatus.UNHEALTHY:
                    self.overall_status = HealthStatus.UNHEALTHY
                    self.last_check_time = datetime.utcnow()
                    return
            
            # Check for any degraded status
            has_degraded = any(
                result.status == HealthStatus.DEGRADED 
                for result in self.results.values()
            )
            
            # Check for any unknown status
            has_unknown = any(
                result.status == HealthStatus.UNKNOWN
                for result in self.results.values()
            )
            
            if has_degraded:
                self.overall_status = HealthStatus.DEGRADED
            elif has_unknown:
                self.overall_status = HealthStatus.DEGRADED  # Treat unknown as degraded
            else:
                self.overall_status = HealthStatus.HEALTHY
            
            self.last_check_time = datetime.utcnow()
    
    def get_health_summary(self) -> Dict[str, Any]:
        """Get overall health summary.
        
        Returns:
            Dictionary with health summary information
        """
        with self._lock:
            summary = {
                "overall_status": self.overall_status.value,
                "last_check_time": self.last_check_time.isoformat() if self.last_check_time else None,
                "total_checks": len(self.checks),
                "checks": {}
            }
            
            status_counts = {status.value: 0 for status in HealthStatus}
            
            for name, result in self.results.items():
                check = self.checks.get(name)
                
                summary["checks"][name] = {
                    "status": result.status.value,
                    "message": result.message,
                    "duration_ms": result.duration_ms,
                    "timestamp": result.timestamp.isoformat(),
                    "critical": check.critical if check else False,
                    "details": result.details
                }
                
                status_counts[result.status.value] += 1
            
            summary["status_counts"] = status_counts
        
        return summary
    
    def is_healthy(self) -> bool:
        """Check if service is healthy overall.
        
        Returns:
            True if overall status is healthy, False otherwise
        """
        return self.overall_status == HealthStatus.HEALTHY
    
    def get_unhealthy_checks(self) -> List[str]:
        """Get list of unhealthy check names.
        
        Returns:
            List of check names that are unhealthy
        """
        with self._lock:
            return [
                name for name, result in self.results.items()
                if result.status == HealthStatus.UNHEALTHY
            ]


# Global health checker instance
_global_health_checker: Optional[HealthChecker] = None


def get_health_checker() -> HealthChecker:
    """Get the global health checker instance."""
    global _global_health_checker
    
    if _global_health_checker is None:
        _global_health_checker = HealthChecker()
    
    return _global_health_checker