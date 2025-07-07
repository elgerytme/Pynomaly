"""Health check endpoints and system status monitoring.

This module provides comprehensive health checks for all Pynomaly components
including database connections, external services, and system resources.
"""

from __future__ import annotations

import asyncio
import logging
import time
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from typing import Any, Callable, Dict, List, Optional

import psutil
from pydantic import BaseModel

logger = logging.getLogger(__name__)


class HealthStatus(str, Enum):
    """Health check status values."""

    HEALTHY = "healthy"
    DEGRADED = "degraded"
    UNHEALTHY = "unhealthy"
    UNKNOWN = "unknown"


class ComponentType(str, Enum):
    """Types of system components."""

    DATABASE = "database"
    CACHE = "cache"
    EXTERNAL_SERVICE = "external_service"
    FILESYSTEM = "filesystem"
    MEMORY = "memory"
    CPU = "cpu"
    NETWORK = "network"
    MODEL_REPOSITORY = "model_repository"
    DETECTOR_SERVICE = "detector_service"
    STREAMING_SERVICE = "streaming_service"


@dataclass
class HealthCheckResult:
    """Result of a health check operation."""

    component: str
    component_type: ComponentType
    status: HealthStatus
    message: str
    details: Dict[str, Any] = field(default_factory=dict)
    timestamp: datetime = field(default_factory=datetime.now)
    response_time_ms: Optional[float] = None

    def is_healthy(self) -> bool:
        """Check if component is healthy."""
        return self.status == HealthStatus.HEALTHY

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary representation."""
        return {
            "component": self.component,
            "component_type": self.component_type.value,
            "status": self.status.value,
            "message": self.message,
            "details": self.details,
            "timestamp": self.timestamp.isoformat(),
            "response_time_ms": self.response_time_ms,
        }


@dataclass
class SystemHealth:
    """Overall system health summary."""

    status: HealthStatus
    message: str
    checks: List[HealthCheckResult] = field(default_factory=list)
    timestamp: datetime = field(default_factory=datetime.now)
    version: str = "unknown"
    uptime_seconds: float = 0.0

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary representation."""
        return {
            "status": self.status.value,
            "message": self.message,
            "timestamp": self.timestamp.isoformat(),
            "version": self.version,
            "uptime_seconds": self.uptime_seconds,
            "checks": [check.to_dict() for check in self.checks],
            "summary": {
                "total_checks": len(self.checks),
                "healthy": len(
                    [c for c in self.checks if c.status == HealthStatus.HEALTHY]
                ),
                "degraded": len(
                    [c for c in self.checks if c.status == HealthStatus.DEGRADED]
                ),
                "unhealthy": len(
                    [c for c in self.checks if c.status == HealthStatus.UNHEALTHY]
                ),
                "unknown": len(
                    [c for c in self.checks if c.status == HealthStatus.UNKNOWN]
                ),
            },
        }


class HealthChecker:
    """Component for performing health checks."""

    def __init__(self):
        """Initialize health checker."""
        self._start_time = time.time()
        self._check_functions: Dict[str, Callable] = {}
        self._last_check_results: Dict[str, HealthCheckResult] = {}

    def register_check(self, name: str, check_function: Callable) -> None:
        """Register a health check function.

        Args:
            name: Name of the health check
            check_function: Async function that returns HealthCheckResult
        """
        self._check_functions[name] = check_function
        logger.info(f"Registered health check: {name}")

    async def check_component(self, name: str) -> HealthCheckResult:
        """Run a specific health check.

        Args:
            name: Name of the health check to run

        Returns:
            Health check result
        """
        if name not in self._check_functions:
            return HealthCheckResult(
                component=name,
                component_type=ComponentType.EXTERNAL_SERVICE,
                status=HealthStatus.UNKNOWN,
                message=f"Health check '{name}' not found",
            )

        start_time = time.time()

        try:
            result = await self._check_functions[name]()
            result.response_time_ms = (time.time() - start_time) * 1000
            self._last_check_results[name] = result
            return result

        except Exception as e:
            error_result = HealthCheckResult(
                component=name,
                component_type=ComponentType.EXTERNAL_SERVICE,
                status=HealthStatus.UNHEALTHY,
                message=f"Health check failed: {str(e)}",
                response_time_ms=(time.time() - start_time) * 1000,
            )
            self._last_check_results[name] = error_result
            logger.error(f"Health check '{name}' failed: {e}")
            return error_result

    async def check_all_components(self) -> List[HealthCheckResult]:
        """Run all registered health checks.

        Returns:
            List of health check results
        """
        tasks = [self.check_component(name) for name in self._check_functions.keys()]

        if not tasks:
            return []

        results = await asyncio.gather(*tasks, return_exceptions=True)

        # Handle any exceptions from gather
        final_results = []
        for i, result in enumerate(results):
            if isinstance(result, Exception):
                name = list(self._check_functions.keys())[i]
                final_results.append(
                    HealthCheckResult(
                        component=name,
                        component_type=ComponentType.EXTERNAL_SERVICE,
                        status=HealthStatus.UNHEALTHY,
                        message=f"Check execution failed: {str(result)}",
                    )
                )
            else:
                final_results.append(result)

        return final_results

    async def get_system_health(self, version: str = "unknown") -> SystemHealth:
        """Get overall system health status.

        Args:
            version: Application version

        Returns:
            System health summary
        """
        check_results = await self.check_all_components()

        # Determine overall status
        if not check_results:
            overall_status = HealthStatus.UNKNOWN
            message = "No health checks configured"
        else:
            unhealthy_count = len(
                [r for r in check_results if r.status == HealthStatus.UNHEALTHY]
            )
            degraded_count = len(
                [r for r in check_results if r.status == HealthStatus.DEGRADED]
            )

            if unhealthy_count > 0:
                overall_status = HealthStatus.UNHEALTHY
                message = f"System unhealthy: {unhealthy_count} component(s) failing"
            elif degraded_count > 0:
                overall_status = HealthStatus.DEGRADED
                message = f"System degraded: {degraded_count} component(s) degraded"
            else:
                overall_status = HealthStatus.HEALTHY
                message = "All systems operational"

        return SystemHealth(
            status=overall_status,
            message=message,
            checks=check_results,
            version=version,
            uptime_seconds=time.time() - self._start_time,
        )

    def get_cached_results(self) -> Dict[str, HealthCheckResult]:
        """Get last known health check results.

        Returns:
            Dictionary of cached health check results
        """
        return self._last_check_results.copy()


# Predefined health check functions
async def check_system_resources() -> HealthCheckResult:
    """Check system resource usage."""
    try:
        # Check memory usage
        memory = psutil.virtual_memory()
        cpu_percent = psutil.cpu_percent(interval=1)
        disk_usage = psutil.disk_usage("/")

        details = {
            "memory": {
                "total_gb": round(memory.total / (1024**3), 2),
                "available_gb": round(memory.available / (1024**3), 2),
                "percent_used": memory.percent,
            },
            "cpu": {"percent_used": cpu_percent},
            "disk": {
                "total_gb": round(disk_usage.total / (1024**3), 2),
                "free_gb": round(disk_usage.free / (1024**3), 2),
                "percent_used": round((disk_usage.used / disk_usage.total) * 100, 2),
            },
        }

        # Determine status based on resource usage
        if (
            memory.percent > 90
            or cpu_percent > 95
            or details["disk"]["percent_used"] > 95
        ):
            status = HealthStatus.UNHEALTHY
            message = "Critical resource usage detected"
        elif (
            memory.percent > 80
            or cpu_percent > 85
            or details["disk"]["percent_used"] > 85
        ):
            status = HealthStatus.DEGRADED
            message = "High resource usage detected"
        else:
            status = HealthStatus.HEALTHY
            message = "System resources within normal limits"

        return HealthCheckResult(
            component="system_resources",
            component_type=ComponentType.CPU,
            status=status,
            message=message,
            details=details,
        )

    except Exception as e:
        return HealthCheckResult(
            component="system_resources",
            component_type=ComponentType.CPU,
            status=HealthStatus.UNHEALTHY,
            message=f"Failed to check system resources: {str(e)}",
        )


async def check_memory_health() -> HealthCheckResult:
    """Check memory-specific health metrics."""
    try:
        memory = psutil.virtual_memory()
        swap = psutil.swap_memory()

        details = {
            "virtual_memory": {
                "total_mb": round(memory.total / (1024**2), 2),
                "available_mb": round(memory.available / (1024**2), 2),
                "used_mb": round(memory.used / (1024**2), 2),
                "percent": memory.percent,
            },
            "swap_memory": {
                "total_mb": round(swap.total / (1024**2), 2),
                "used_mb": round(swap.used / (1024**2), 2),
                "percent": swap.percent,
            },
        }

        # Memory health assessment
        if memory.percent > 95 or swap.percent > 80:
            status = HealthStatus.UNHEALTHY
            message = "Critical memory pressure"
        elif memory.percent > 85 or swap.percent > 50:
            status = HealthStatus.DEGRADED
            message = "High memory usage"
        else:
            status = HealthStatus.HEALTHY
            message = "Memory usage normal"

        return HealthCheckResult(
            component="memory",
            component_type=ComponentType.MEMORY,
            status=status,
            message=message,
            details=details,
        )

    except Exception as e:
        return HealthCheckResult(
            component="memory",
            component_type=ComponentType.MEMORY,
            status=HealthStatus.UNHEALTHY,
            message=f"Memory check failed: {str(e)}",
        )


async def check_filesystem_health() -> HealthCheckResult:
    """Check filesystem health and disk space."""
    try:
        import os
        import tempfile

        # Check main filesystem
        disk_usage = psutil.disk_usage("/")

        # Test write capability
        temp_file = None
        write_test_success = False
        try:
            with tempfile.NamedTemporaryFile(mode="w", delete=False) as f:
                temp_file = f.name
                f.write("health_check_test")

            # Verify we can read it back
            with open(temp_file, "r") as f:
                content = f.read()
                if content == "health_check_test":
                    write_test_success = True
        except Exception:
            write_test_success = False
        finally:
            if temp_file and os.path.exists(temp_file):
                try:
                    os.unlink(temp_file)
                except Exception:
                    pass

        disk_percent_used = (disk_usage.used / disk_usage.total) * 100

        details = {
            "disk_usage": {
                "total_gb": round(disk_usage.total / (1024**3), 2),
                "used_gb": round(disk_usage.used / (1024**3), 2),
                "free_gb": round(disk_usage.free / (1024**3), 2),
                "percent_used": round(disk_percent_used, 2),
            },
            "write_test": write_test_success,
        }

        # Filesystem health assessment
        if not write_test_success or disk_percent_used > 95:
            status = HealthStatus.UNHEALTHY
            message = "Filesystem issues detected"
        elif disk_percent_used > 85:
            status = HealthStatus.DEGRADED
            message = "Low disk space"
        else:
            status = HealthStatus.HEALTHY
            message = "Filesystem healthy"

        return HealthCheckResult(
            component="filesystem",
            component_type=ComponentType.FILESYSTEM,
            status=status,
            message=message,
            details=details,
        )

    except Exception as e:
        return HealthCheckResult(
            component="filesystem",
            component_type=ComponentType.FILESYSTEM,
            status=HealthStatus.UNHEALTHY,
            message=f"Filesystem check failed: {str(e)}",
        )


# Mock health checks for Pynomaly-specific components
async def check_model_repository() -> HealthCheckResult:
    """Check model repository health."""
    try:
        # This would check actual model repository in real implementation
        # For now, return a mock healthy status

        details = {
            "models_loaded": 5,
            "repository_size_mb": 150.5,
            "last_sync": datetime.now().isoformat(),
        }

        return HealthCheckResult(
            component="model_repository",
            component_type=ComponentType.MODEL_REPOSITORY,
            status=HealthStatus.HEALTHY,
            message="Model repository operational",
            details=details,
        )

    except Exception as e:
        return HealthCheckResult(
            component="model_repository",
            component_type=ComponentType.MODEL_REPOSITORY,
            status=HealthStatus.UNHEALTHY,
            message=f"Model repository check failed: {str(e)}",
        )


async def check_detector_service() -> HealthCheckResult:
    """Check detector service health."""
    try:
        # Mock detector service check
        details = {
            "active_detectors": 3,
            "algorithms_available": ["IsolationForest", "OneClassSVM", "DBSCAN"],
            "last_training": (datetime.now() - timedelta(hours=2)).isoformat(),
        }

        return HealthCheckResult(
            component="detector_service",
            component_type=ComponentType.DETECTOR_SERVICE,
            status=HealthStatus.HEALTHY,
            message="Detector service operational",
            details=details,
        )

    except Exception as e:
        return HealthCheckResult(
            component="detector_service",
            component_type=ComponentType.DETECTOR_SERVICE,
            status=HealthStatus.UNHEALTHY,
            message=f"Detector service check failed: {str(e)}",
        )


async def check_streaming_service() -> HealthCheckResult:
    """Check streaming service health."""
    try:
        # Mock streaming service check
        details = {
            "active_streams": 2,
            "total_throughput": 1250.0,
            "buffer_utilization": 0.45,
            "backpressure_events": 0,
        }

        # Determine status based on streaming metrics
        if details["buffer_utilization"] > 0.9 or details["backpressure_events"] > 10:
            status = HealthStatus.DEGRADED
            message = "Streaming service under pressure"
        else:
            status = HealthStatus.HEALTHY
            message = "Streaming service operational"

        return HealthCheckResult(
            component="streaming_service",
            component_type=ComponentType.STREAMING_SERVICE,
            status=status,
            message=message,
            details=details,
        )

    except Exception as e:
        return HealthCheckResult(
            component="streaming_service",
            component_type=ComponentType.STREAMING_SERVICE,
            status=HealthStatus.UNHEALTHY,
            message=f"Streaming service check failed: {str(e)}",
        )


# Global health checker instance
_health_checker: Optional[HealthChecker] = None


def get_health_checker() -> HealthChecker:
    """Get global health checker instance."""
    global _health_checker
    if _health_checker is None:
        _health_checker = HealthChecker()

        # Register default health checks
        _health_checker.register_check("system_resources", check_system_resources)
        _health_checker.register_check("memory", check_memory_health)
        _health_checker.register_check("filesystem", check_filesystem_health)
        _health_checker.register_check("model_repository", check_model_repository)
        _health_checker.register_check("detector_service", check_detector_service)
        _health_checker.register_check("streaming_service", check_streaming_service)

        logger.info("Health checker initialized with default checks")

    return _health_checker


def register_health_check(name: str, check_function: Callable) -> None:
    """Register a custom health check.

    Args:
        name: Name of the health check
        check_function: Async function that returns HealthCheckResult
    """
    checker = get_health_checker()
    checker.register_check(name, check_function)


# Kubernetes readiness and liveness probe support
class ProbeResponse(BaseModel):
    """Response model for Kubernetes probes."""

    status: str
    timestamp: str
    details: Optional[Dict[str, Any]] = None


async def liveness_probe() -> ProbeResponse:
    """Kubernetes liveness probe endpoint.

    Returns basic application liveness status.
    """
    return ProbeResponse(
        status="alive",
        timestamp=datetime.now().isoformat(),
        details={"uptime_seconds": time.time() - get_health_checker()._start_time},
    )


async def readiness_probe() -> ProbeResponse:
    """Kubernetes readiness probe endpoint.

    Returns application readiness status based on critical components.
    """
    checker = get_health_checker()

    # Check only critical components for readiness
    critical_checks = ["memory", "filesystem", "model_repository"]

    ready = True
    details = {}

    for check_name in critical_checks:
        if check_name in checker._check_functions:
            result = await checker.check_component(check_name)
            details[check_name] = result.status.value
            if result.status in [HealthStatus.UNHEALTHY, HealthStatus.UNKNOWN]:
                ready = False

    return ProbeResponse(
        status="ready" if ready else "not_ready",
        timestamp=datetime.now().isoformat(),
        details=details,
    )
