"""Domain-aware health monitoring for anomaly detection service."""

import asyncio
import time
from datetime import datetime, timedelta
from dataclasses import dataclass
from enum import Enum
from typing import Dict, List, Optional, Any, Callable
import logging

from .alerting_system import AlertingSystem, AlertSeverity

logger = logging.getLogger(__name__)


class HealthStatus(str, Enum):
    """Health status levels."""
    HEALTHY = "healthy"
    DEGRADED = "degraded"
    UNHEALTHY = "unhealthy"
    UNKNOWN = "unknown"


class DomainType(str, Enum):
    """Domain types for health monitoring."""
    AI_ML = "ai_ml"
    AI_MLOPS = "ai_mlops"
    DATA_PROCESSING = "data_processing"
    SHARED_INFRASTRUCTURE = "shared_infrastructure"
    SHARED_OBSERVABILITY = "shared_observability"
    APPLICATION = "application"


@dataclass
class HealthCheck:
    """Health check configuration."""
    name: str
    domain: DomainType
    check_function: Callable[[], Dict[str, Any]]
    interval_seconds: int = 60
    timeout_seconds: int = 30
    retries: int = 3
    enabled: bool = True


@dataclass
class HealthCheckResult:
    """Health check result."""
    name: str
    domain: DomainType
    status: HealthStatus
    message: str
    timestamp: datetime
    execution_time_ms: float
    metadata: Dict[str, Any]
    error: Optional[str] = None


class DomainHealthMonitor:
    """Monitor health across domain boundaries."""
    
    def __init__(self, alerting_system: Optional[AlertingSystem] = None):
        """Initialize domain health monitor."""
        self.health_checks: Dict[str, HealthCheck] = {}
        self.latest_results: Dict[str, HealthCheckResult] = {}
        self.alerting_system = alerting_system or AlertingSystem()
        self._monitoring = False
        self._monitoring_task: Optional[asyncio.Task] = None
        
    def register_health_check(self, health_check: HealthCheck) -> None:
        """Register a new health check."""
        self.health_checks[health_check.name] = health_check
        logger.info(f"Registered health check: {health_check.name} ({health_check.domain})")
    
    async def execute_health_check(self, health_check: HealthCheck) -> HealthCheckResult:
        """Execute a single health check."""
        start_time = time.time()
        
        try:
            # Execute with timeout
            result_data = await asyncio.wait_for(
                asyncio.get_event_loop().run_in_executor(
                    None, health_check.check_function
                ),
                timeout=health_check.timeout_seconds
            )
            
            execution_time = (time.time() - start_time) * 1000
            
            # Determine status from result
            status = HealthStatus(result_data.get("status", HealthStatus.UNKNOWN))
            message = result_data.get("message", "Health check completed")
            metadata = result_data.get("metadata", {})
            
            return HealthCheckResult(
                name=health_check.name,
                domain=health_check.domain,
                status=status,
                message=message,
                timestamp=datetime.utcnow(),
                execution_time_ms=execution_time,
                metadata=metadata
            )
            
        except asyncio.TimeoutError:
            execution_time = (time.time() - start_time) * 1000
            return HealthCheckResult(
                name=health_check.name,
                domain=health_check.domain,
                status=HealthStatus.UNHEALTHY,
                message=f"Health check timed out after {health_check.timeout_seconds}s",
                timestamp=datetime.utcnow(),
                execution_time_ms=execution_time,
                metadata={},
                error="timeout"
            )
            
        except Exception as e:
            execution_time = (time.time() - start_time) * 1000
            return HealthCheckResult(
                name=health_check.name,
                domain=health_check.domain,
                status=HealthStatus.UNHEALTHY,
                message=f"Health check failed: {str(e)}",
                timestamp=datetime.utcnow(),
                execution_time_ms=execution_time,
                metadata={},
                error=str(e)
            )
    
    async def run_all_health_checks(self) -> Dict[str, HealthCheckResult]:
        """Run all enabled health checks."""
        tasks = []
        
        for health_check in self.health_checks.values():
            if health_check.enabled:
                tasks.append(self.execute_health_check(health_check))
        
        if not tasks:
            return {}
        
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # Process results
        health_results = {}
        for result in results:
            if isinstance(result, Exception):
                logger.error(f"Health check failed with exception: {result}")
                continue
                
            health_results[result.name] = result
            self.latest_results[result.name] = result
            
            # Create alerts for unhealthy services
            if result.status == HealthStatus.UNHEALTHY:
                self.alerting_system.create_alert(
                    title=f"Domain Health Alert: {result.name}",
                    message=result.message,
                    severity=AlertSeverity.HIGH,
                    source=f"domain_monitor_{result.domain}",
                    metadata={
                        "domain": result.domain,
                        "health_check": result.name,
                        "execution_time_ms": result.execution_time_ms,
                        "error": result.error
                    }
                )
            
            elif result.status == HealthStatus.DEGRADED:
                self.alerting_system.create_alert(
                    title=f"Domain Performance Alert: {result.name}",
                    message=result.message,
                    severity=AlertSeverity.MEDIUM,
                    source=f"domain_monitor_{result.domain}",
                    metadata={
                        "domain": result.domain,
                        "health_check": result.name,
                        "execution_time_ms": result.execution_time_ms
                    }
                )
        
        return health_results
    
    def get_domain_health(self, domain: DomainType) -> Dict[str, Any]:
        """Get health status for a specific domain."""
        domain_results = [
            result for result in self.latest_results.values()
            if result.domain == domain
        ]
        
        if not domain_results:
            return {
                "domain": domain,
                "status": HealthStatus.UNKNOWN,
                "message": "No health checks configured",
                "checks": [],
                "summary": {}
            }
        
        # Determine overall domain status
        statuses = [result.status for result in domain_results]
        if HealthStatus.UNHEALTHY in statuses:
            overall_status = HealthStatus.UNHEALTHY
        elif HealthStatus.DEGRADED in statuses:
            overall_status = HealthStatus.DEGRADED
        elif HealthStatus.HEALTHY in statuses:
            overall_status = HealthStatus.HEALTHY
        else:
            overall_status = HealthStatus.UNKNOWN
        
        return {
            "domain": domain,
            "status": overall_status,
            "message": f"Domain health based on {len(domain_results)} checks",
            "checks": [
                {
                    "name": result.name,
                    "status": result.status,
                    "message": result.message,
                    "execution_time_ms": result.execution_time_ms,
                    "timestamp": result.timestamp.isoformat(),
                    "error": result.error
                }
                for result in domain_results
            ],
            "summary": {
                "total_checks": len(domain_results),
                "healthy": len([r for r in domain_results if r.status == HealthStatus.HEALTHY]),
                "degraded": len([r for r in domain_results if r.status == HealthStatus.DEGRADED]),
                "unhealthy": len([r for r in domain_results if r.status == HealthStatus.UNHEALTHY]),
                "avg_execution_time_ms": sum(r.execution_time_ms for r in domain_results) / len(domain_results)
            }
        }
    
    def get_overall_health(self) -> Dict[str, Any]:
        """Get overall system health across all domains."""
        if not self.latest_results:
            return {
                "status": HealthStatus.UNKNOWN,
                "message": "No health checks have been run",
                "domains": {},
                "summary": {
                    "total_checks": 0,
                    "healthy": 0,
                    "degraded": 0,
                    "unhealthy": 0,
                    "last_check": None
                }
            }
        
        # Get health for each domain
        domains = {}
        for domain in DomainType:
            domains[domain.value] = self.get_domain_health(domain)
        
        # Calculate overall status
        all_results = list(self.latest_results.values())
        statuses = [result.status for result in all_results]
        
        if HealthStatus.UNHEALTHY in statuses:
            overall_status = HealthStatus.UNHEALTHY
        elif HealthStatus.DEGRADED in statuses:
            overall_status = HealthStatus.DEGRADED
        elif HealthStatus.HEALTHY in statuses:
            overall_status = HealthStatus.HEALTHY
        else:
            overall_status = HealthStatus.UNKNOWN
        
        return {
            "status": overall_status,
            "message": f"Overall health based on {len(all_results)} checks across {len(domains)} domains",
            "domains": domains,
            "summary": {
                "total_checks": len(all_results),
                "healthy": len([r for r in all_results if r.status == HealthStatus.HEALTHY]),
                "degraded": len([r for r in all_results if r.status == HealthStatus.DEGRADED]),
                "unhealthy": len([r for r in all_results if r.status == HealthStatus.UNHEALTHY]),
                "last_check": max(r.timestamp for r in all_results).isoformat() if all_results else None
            }
        }
    
    async def start_monitoring(self) -> None:
        """Start continuous health monitoring."""
        if self._monitoring:
            logger.warning("Health monitoring is already running")
            return
        
        self._monitoring = True
        self._monitoring_task = asyncio.create_task(self._monitoring_loop())
        logger.info("Started domain health monitoring")
    
    async def stop_monitoring(self) -> None:
        """Stop continuous health monitoring."""
        if not self._monitoring:
            return
        
        self._monitoring = False
        if self._monitoring_task:
            self._monitoring_task.cancel()
            try:
                await self._monitoring_task
            except asyncio.CancelledError:
                pass
        
        logger.info("Stopped domain health monitoring")
    
    async def _monitoring_loop(self) -> None:
        """Main monitoring loop."""
        while self._monitoring:
            try:
                await self.run_all_health_checks()
                
                # Wait for the shortest interval among all health checks
                min_interval = min(
                    (check.interval_seconds for check in self.health_checks.values() if check.enabled),
                    default=60
                )
                
                await asyncio.sleep(min_interval)
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Error in health monitoring loop: {e}")
                await asyncio.sleep(60)  # Wait before retrying


# Default health checks for each domain
def create_default_health_checks() -> List[HealthCheck]:
    """Create default health checks for all domains."""
    
    def ai_ml_health_check() -> Dict[str, Any]:
        """Check AI/ML domain health."""
        try:
            # Test scikit-learn import and basic functionality
            from sklearn.ensemble import IsolationForest
            import numpy as np
            
            # Quick test
            data = np.random.rand(10, 2)
            detector = IsolationForest(random_state=42)
            detector.fit(data)
            
            return {
                "status": HealthStatus.HEALTHY,
                "message": "AI/ML algorithms available and functional",
                "metadata": {
                    "sklearn_available": True,
                    "test_data_shape": data.shape
                }
            }
        except Exception as e:
            return {
                "status": HealthStatus.UNHEALTHY,
                "message": f"AI/ML domain check failed: {str(e)}",
                "metadata": {"error": str(e)}
            }
    
    def data_processing_health_check() -> Dict[str, Any]:
        """Check data processing domain health."""
        try:
            import pandas as pd
            import numpy as np
            
            # Test basic data operations
            df = pd.DataFrame(np.random.rand(5, 3), columns=['a', 'b', 'c'])
            result = df.describe()
            
            return {
                "status": HealthStatus.HEALTHY,
                "message": "Data processing capabilities available",
                "metadata": {
                    "pandas_available": True,
                    "numpy_available": True,
                    "test_dataframe_shape": df.shape
                }
            }
        except Exception as e:
            return {
                "status": HealthStatus.UNHEALTHY,
                "message": f"Data processing domain check failed: {str(e)}",
                "metadata": {"error": str(e)}
            }
    
    def infrastructure_health_check() -> Dict[str, Any]:
        """Check shared infrastructure health."""
        try:
            import os
            import tempfile
            
            # Test file system access
            with tempfile.NamedTemporaryFile() as tmp:
                tmp.write(b"health check")
                tmp.flush()
                size = os.path.getsize(tmp.name)
            
            return {
                "status": HealthStatus.HEALTHY,
                "message": "Infrastructure components accessible",
                "metadata": {
                    "filesystem_writable": True,
                    "temp_file_size": size
                }
            }
        except Exception as e:
            return {
                "status": HealthStatus.UNHEALTHY,
                "message": f"Infrastructure check failed: {str(e)}",
                "metadata": {"error": str(e)}
            }
    
    def application_health_check() -> Dict[str, Any]:
        """Check application domain health."""  
        try:
            # Test core application imports
            from anomaly_detection.domain.services.detection_service import DetectionService
            service = DetectionService()
            
            return {
                "status": HealthStatus.HEALTHY,
                "message": "Application layer services available",
                "metadata": {
                    "detection_service_available": True,
                    "service_initialized": True
                }
            }
        except Exception as e:
            return {
                "status": HealthStatus.UNHEALTHY,
                "message": f"Application domain check failed: {str(e)}",
                "metadata": {"error": str(e)}
            }
    
    return [
        HealthCheck(
            name="ai_ml_domain",
            domain=DomainType.AI_ML,
            check_function=ai_ml_health_check,
            interval_seconds=120
        ),
        HealthCheck(
            name="data_processing_domain", 
            domain=DomainType.DATA_PROCESSING,
            check_function=data_processing_health_check,
            interval_seconds=120
        ),
        HealthCheck(
            name="infrastructure_domain",
            domain=DomainType.SHARED_INFRASTRUCTURE,
            check_function=infrastructure_health_check,
            interval_seconds=300
        ),
        HealthCheck(
            name="application_domain",
            domain=DomainType.APPLICATION,
            check_function=application_health_check,
            interval_seconds=60
        )
    ]


# Global monitor instance
_domain_health_monitor = None

def get_domain_health_monitor() -> DomainHealthMonitor:
    """Get or create the global domain health monitor."""
    global _domain_health_monitor
    if _domain_health_monitor is None:
        _domain_health_monitor = DomainHealthMonitor()
        
        # Register default health checks
        for health_check in create_default_health_checks():
            _domain_health_monitor.register_health_check(health_check)
    
    return _domain_health_monitor