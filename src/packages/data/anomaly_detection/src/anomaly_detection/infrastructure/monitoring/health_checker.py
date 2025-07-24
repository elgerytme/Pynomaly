"""Health checker module for system health monitoring."""

from typing import Dict, Any, Optional, List, Callable
from datetime import datetime
from enum import Enum
from dataclasses import dataclass
import structlog

logger = structlog.get_logger(__name__)


class HealthStatus(Enum):
    """Health status enumeration."""
    HEALTHY = "healthy"
    WARNING = "warning"
    CRITICAL = "critical"
    UNKNOWN = "unknown"


@dataclass
class HealthCheck:
    """Health check result."""
    name: str
    status: HealthStatus
    message: str
    timestamp: datetime
    details: Optional[Dict[str, Any]] = None


class HealthChecker:
    """System health checker."""
    
    def __init__(self):
        self.checks: Dict[str, Callable] = {}
        logger.info("HealthChecker initialized")
    
    def register_check(self, name: str, check_func: Callable) -> None:
        """Register a health check function."""
        self.checks[name] = check_func
    
    def run_check(self, name: str) -> HealthCheck:
        """Run a specific health check."""
        if name not in self.checks:
            return HealthCheck(
                name=name,
                status=HealthStatus.UNKNOWN,
                message=f"Check '{name}' not found",
                timestamp=datetime.utcnow()
            )
        
        try:
            result = self.checks[name]()
            return result
        except Exception as e:
            return HealthCheck(
                name=name,
                status=HealthStatus.CRITICAL,
                message=f"Check failed: {str(e)}",
                timestamp=datetime.utcnow()
            )
    
    def run_all_checks(self) -> List[HealthCheck]:
        """Run all registered health checks."""
        results = []
        for name in self.checks:
            results.append(self.run_check(name))
        return results


# Global instance
_health_checker: Optional[HealthChecker] = None


def get_health_checker() -> HealthChecker:
    """Get the global health checker instance."""
    global _health_checker
    
    if _health_checker is None:
        _health_checker = HealthChecker()
    
    return _health_checker