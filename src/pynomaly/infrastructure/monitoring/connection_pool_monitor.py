"""Connection pool monitoring and alerting system."""

from __future__ import annotations

import asyncio
import time
from dataclasses import dataclass
from typing import Any, Callable

import structlog

from pynomaly.infrastructure.persistence.connection_pool_integration import (
    get_database_integration,
)
from pynomaly.infrastructure.performance.connection_pooling import (
    ConnectionPoolManager,
    PoolStats,
    get_connection_pool_manager,
)

logger = structlog.get_logger(__name__)


@dataclass
class PoolAlert:
    """Connection pool alert."""
    
    pool_name: str
    alert_type: str
    severity: str  # low, medium, high, critical
    message: str
    timestamp: float
    metrics: dict[str, Any]


@dataclass
class PoolThresholds:
    """Connection pool alert thresholds."""
    
    # Connection utilization thresholds
    high_utilization_threshold: float = 0.8  # 80% of max connections
    critical_utilization_threshold: float = 0.95  # 95% of max connections
    
    # Error rate thresholds
    high_error_rate_threshold: float = 0.1  # 10% error rate
    critical_error_rate_threshold: float = 0.25  # 25% error rate
    
    # Response time thresholds (seconds)
    slow_response_threshold: float = 1.0  # 1 second
    critical_response_threshold: float = 5.0  # 5 seconds
    
    # Connection health thresholds
    min_healthy_connections: int = 2
    max_failed_health_checks: int = 5


class ConnectionPoolMonitor:
    """Monitor connection pools and generate alerts."""

    def __init__(
        self,
        pool_manager: ConnectionPoolManager | None = None,
        check_interval: float = 30.0,
        alert_callback: Callable[[PoolAlert], None] | None = None,
        enable_auto_recovery: bool = True,
    ):
        """Initialize connection pool monitor.
        
        Args:
            pool_manager: Connection pool manager to monitor
            check_interval: Monitoring interval in seconds
            alert_callback: Callback function for alerts
            enable_auto_recovery: Whether to enable automatic recovery
        """
        self.pool_manager = pool_manager or get_connection_pool_manager()
        self.check_interval = check_interval
        self.alert_callback = alert_callback
        self.enable_auto_recovery = enable_auto_recovery
        
        # Monitoring state
        self.thresholds = PoolThresholds()
        self.monitoring_task: asyncio.Task | None = None
        self.alerts_history: list[PoolAlert] = []
        self.max_alerts_history = 1000
        
        # Pool health tracking
        self.pool_health_history: dict[str, list[dict[str, Any]]] = {}
        self.max_health_history = 100
        
        # Recovery tracking
        self.recovery_attempts: dict[str, int] = {}
        self.last_recovery_time: dict[str, float] = {}
        self.recovery_cooldown = 300  # 5 minutes

    def start_monitoring(self) -> None:
        """Start connection pool monitoring."""
        if self.monitoring_task is None:
            self.monitoring_task = asyncio.create_task(self._monitoring_loop())
            logger.info(
                "Connection pool monitoring started",
                interval=self.check_interval,
                auto_recovery=self.enable_auto_recovery,
            )

    async def stop_monitoring(self) -> None:
        """Stop connection pool monitoring."""
        if self.monitoring_task:
            self.monitoring_task.cancel()
            try:
                await self.monitoring_task
            except asyncio.CancelledError:
                pass
            self.monitoring_task = None
            logger.info("Connection pool monitoring stopped")

    async def _monitoring_loop(self) -> None:
        """Main monitoring loop."""
        while True:
            try:
                await asyncio.sleep(self.check_interval)
                await self._check_all_pools()
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error("Pool monitoring error", error=str(e))

    async def _check_all_pools(self) -> None:
        """Check all connection pools."""
        pool_names = self.pool_manager.list_pools()
        
        for pool_name in pool_names:
            try:
                await self._check_pool(pool_name)
            except Exception as e:
                logger.error(
                    "Error checking pool",
                    pool_name=pool_name,
                    error=str(e),
                )

    async def _check_pool(self, pool_name: str) -> None:
        """Check individual connection pool."""
        try:
            pool_info = self.pool_manager.get_pool_info(pool_name)
            stats = pool_info["stats"]
            pool_details = pool_info["pool_info"]
            
            # Track health history
            health_data = {
                "timestamp": time.time(),
                "stats": stats,
                "pool_info": pool_details,
            }
            
            if pool_name not in self.pool_health_history:
                self.pool_health_history[pool_name] = []
                
            self.pool_health_history[pool_name].append(health_data)
            if len(self.pool_health_history[pool_name]) > self.max_health_history:
                self.pool_health_history[pool_name].pop(0)
            
            # Check various metrics
            await self._check_connection_utilization(pool_name, stats, pool_details)
            await self._check_error_rates(pool_name, stats)
            await self._check_response_times(pool_name, stats)
            await self._check_connection_health(pool_name, stats, pool_details)
            
        except Exception as e:
            logger.error(
                "Failed to check pool",
                pool_name=pool_name,
                error=str(e),
            )

    async def _check_connection_utilization(
        self,
        pool_name: str,
        stats: PoolStats,
        pool_details: dict[str, Any],
    ) -> None:
        """Check connection utilization."""
        try:
            # Calculate utilization based on pool type
            if "size" in pool_details:
                # SQLAlchemy pool
                total_connections = pool_details.get("size", 0)
                active_connections = pool_details.get("checked_out", 0)
            else:
                # Other pool types
                total_connections = stats.total_connections
                active_connections = stats.active_connections
                
            if total_connections > 0:
                utilization = active_connections / total_connections
                
                if utilization >= self.thresholds.critical_utilization_threshold:
                    await self._create_alert(
                        pool_name=pool_name,
                        alert_type="high_utilization",
                        severity="critical",
                        message=f"Critical connection utilization: {utilization:.1%}",
                        metrics={
                            "utilization": utilization,
                            "active_connections": active_connections,
                            "total_connections": total_connections,
                        },
                    )
                    
                    # Trigger auto-recovery if enabled
                    if self.enable_auto_recovery:
                        await self._attempt_recovery(pool_name, "high_utilization")
                        
                elif utilization >= self.thresholds.high_utilization_threshold:
                    await self._create_alert(
                        pool_name=pool_name,
                        alert_type="high_utilization",
                        severity="high",
                        message=f"High connection utilization: {utilization:.1%}",
                        metrics={
                            "utilization": utilization,
                            "active_connections": active_connections,
                            "total_connections": total_connections,
                        },
                    )
                    
        except Exception as e:
            logger.error(
                "Failed to check connection utilization",
                pool_name=pool_name,
                error=str(e),
            )

    async def _check_error_rates(self, pool_name: str, stats: PoolStats) -> None:
        """Check error rates."""
        try:
            if stats.total_requests > 0:
                error_rate = stats.failed_requests / stats.total_requests
                
                if error_rate >= self.thresholds.critical_error_rate_threshold:
                    await self._create_alert(
                        pool_name=pool_name,
                        alert_type="high_error_rate",
                        severity="critical",
                        message=f"Critical error rate: {error_rate:.1%}",
                        metrics={
                            "error_rate": error_rate,
                            "failed_requests": stats.failed_requests,
                            "total_requests": stats.total_requests,
                        },
                    )
                    
                    # Trigger auto-recovery if enabled
                    if self.enable_auto_recovery:
                        await self._attempt_recovery(pool_name, "high_error_rate")
                        
                elif error_rate >= self.thresholds.high_error_rate_threshold:
                    await self._create_alert(
                        pool_name=pool_name,
                        alert_type="high_error_rate",
                        severity="high",
                        message=f"High error rate: {error_rate:.1%}",
                        metrics={
                            "error_rate": error_rate,
                            "failed_requests": stats.failed_requests,
                            "total_requests": stats.total_requests,
                        },
                    )
                    
        except Exception as e:
            logger.error(
                "Failed to check error rates",
                pool_name=pool_name,
                error=str(e),
            )

    async def _check_response_times(self, pool_name: str, stats: PoolStats) -> None:
        """Check response times."""
        try:
            avg_response_time = stats.avg_response_time
            
            if avg_response_time >= self.thresholds.critical_response_threshold:
                await self._create_alert(
                    pool_name=pool_name,
                    alert_type="slow_response",
                    severity="critical",
                    message=f"Critical response time: {avg_response_time:.2f}s",
                    metrics={
                        "avg_response_time": avg_response_time,
                        "successful_requests": stats.successful_requests,
                    },
                )
                
            elif avg_response_time >= self.thresholds.slow_response_threshold:
                await self._create_alert(
                    pool_name=pool_name,
                    alert_type="slow_response",
                    severity="medium",
                    message=f"Slow response time: {avg_response_time:.2f}s",
                    metrics={
                        "avg_response_time": avg_response_time,
                        "successful_requests": stats.successful_requests,
                    },
                )
                
        except Exception as e:
            logger.error(
                "Failed to check response times",
                pool_name=pool_name,
                error=str(e),
            )

    async def _check_connection_health(
        self,
        pool_name: str,
        stats: PoolStats,
        pool_details: dict[str, Any],
    ) -> None:
        """Check connection health."""
        try:
            # Check for connection errors
            if stats.connection_errors >= self.thresholds.max_failed_health_checks:
                await self._create_alert(
                    pool_name=pool_name,
                    alert_type="connection_health",
                    severity="high",
                    message=f"Multiple connection errors: {stats.connection_errors}",
                    metrics={
                        "connection_errors": stats.connection_errors,
                        "total_connections": stats.total_connections,
                    },
                )
                
            # Check for insufficient healthy connections
            healthy_connections = stats.total_connections - stats.connection_errors
            if healthy_connections < self.thresholds.min_healthy_connections:
                await self._create_alert(
                    pool_name=pool_name,
                    alert_type="connection_health",
                    severity="high",
                    message=f"Insufficient healthy connections: {healthy_connections}",
                    metrics={
                        "healthy_connections": healthy_connections,
                        "min_required": self.thresholds.min_healthy_connections,
                    },
                )
                
        except Exception as e:
            logger.error(
                "Failed to check connection health",
                pool_name=pool_name,
                error=str(e),
            )

    async def _create_alert(
        self,
        pool_name: str,
        alert_type: str,
        severity: str,
        message: str,
        metrics: dict[str, Any],
    ) -> None:
        """Create and handle alert."""
        alert = PoolAlert(
            pool_name=pool_name,
            alert_type=alert_type,
            severity=severity,
            message=message,
            timestamp=time.time(),
            metrics=metrics,
        )
        
        # Add to history
        self.alerts_history.append(alert)
        if len(self.alerts_history) > self.max_alerts_history:
            self.alerts_history.pop(0)
        
        # Log alert
        logger.warning(
            "Connection pool alert",
            pool_name=pool_name,
            alert_type=alert_type,
            severity=severity,
            message=message,
            metrics=metrics,
        )
        
        # Call alert callback if provided
        if self.alert_callback:
            try:
                self.alert_callback(alert)
            except Exception as e:
                logger.error("Alert callback failed", error=str(e))

    async def _attempt_recovery(self, pool_name: str, reason: str) -> None:
        """Attempt automatic recovery."""
        current_time = time.time()
        
        # Check recovery cooldown
        if (
            pool_name in self.last_recovery_time
            and current_time - self.last_recovery_time[pool_name] < self.recovery_cooldown
        ):
            logger.info(
                "Recovery attempt skipped - cooldown active",
                pool_name=pool_name,
                reason=reason,
            )
            return
            
        # Track recovery attempts
        self.recovery_attempts[pool_name] = self.recovery_attempts.get(pool_name, 0) + 1
        self.last_recovery_time[pool_name] = current_time
        
        try:
            logger.info(
                "Attempting pool recovery",
                pool_name=pool_name,
                reason=reason,
                attempt=self.recovery_attempts[pool_name],
            )
            
            # Get database integration and perform recovery
            db_integration = get_database_integration()
            recovery_result = await db_integration.emergency_recovery()
            
            if recovery_result.get("status") == "success":
                logger.info(
                    "Pool recovery successful",
                    pool_name=pool_name,
                    reason=reason,
                    result=recovery_result,
                )
                
                # Reset recovery counter on success
                self.recovery_attempts[pool_name] = 0
                
            else:
                logger.error(
                    "Pool recovery failed",
                    pool_name=pool_name,
                    reason=reason,
                    result=recovery_result,
                )
                
        except Exception as e:
            logger.error(
                "Pool recovery error",
                pool_name=pool_name,
                reason=reason,
                error=str(e),
            )

    def get_monitoring_dashboard(self) -> dict[str, Any]:
        """Get comprehensive monitoring dashboard data."""
        current_time = time.time()
        
        # Get current pool stats
        pool_stats = self.pool_manager.get_all_stats()
        
        # Recent alerts (last hour)
        recent_alerts = [
            alert for alert in self.alerts_history
            if current_time - alert.timestamp < 3600
        ]
        
        # Alert summary by severity
        alert_summary = {
            "critical": len([a for a in recent_alerts if a.severity == "critical"]),
            "high": len([a for a in recent_alerts if a.severity == "high"]),
            "medium": len([a for a in recent_alerts if a.severity == "medium"]),
            "low": len([a for a in recent_alerts if a.severity == "low"]),
        }
        
        # Pool health summary
        pool_health = {}
        for pool_name, stats in pool_stats.items():
            if stats.total_requests > 0:
                error_rate = stats.failed_requests / stats.total_requests
            else:
                error_rate = 0.0
                
            pool_health[pool_name] = {
                "status": "healthy" if error_rate < 0.1 else "unhealthy",
                "error_rate": error_rate,
                "avg_response_time": stats.avg_response_time,
                "active_connections": stats.active_connections,
                "total_connections": stats.total_connections,
                "last_check": current_time,
            }
        
        return {
            "monitoring_status": {
                "running": self.monitoring_task is not None,
                "check_interval": self.check_interval,
                "auto_recovery": self.enable_auto_recovery,
                "last_check": current_time,
            },
            "pool_health": pool_health,
            "alert_summary": alert_summary,
            "recent_alerts": recent_alerts[-10:],  # Last 10 alerts
            "recovery_attempts": self.recovery_attempts,
            "thresholds": {
                "high_utilization": self.thresholds.high_utilization_threshold,
                "critical_utilization": self.thresholds.critical_utilization_threshold,
                "high_error_rate": self.thresholds.high_error_rate_threshold,
                "critical_error_rate": self.thresholds.critical_error_rate_threshold,
                "slow_response": self.thresholds.slow_response_threshold,
                "critical_response": self.thresholds.critical_response_threshold,
            },
        }

    def update_thresholds(self, new_thresholds: dict[str, Any]) -> None:
        """Update monitoring thresholds."""
        for key, value in new_thresholds.items():
            if hasattr(self.thresholds, key):
                setattr(self.thresholds, key, value)
                logger.info(
                    "Updated monitoring threshold",
                    threshold=key,
                    value=value,
                )

    def get_alerts_history(self, limit: int | None = None) -> list[PoolAlert]:
        """Get alerts history."""
        if limit is None:
            return self.alerts_history.copy()
        return self.alerts_history[-limit:].copy()

    def clear_alerts_history(self) -> None:
        """Clear alerts history."""
        self.alerts_history.clear()
        logger.info("Alerts history cleared")


# Global connection pool monitor
_pool_monitor: ConnectionPoolMonitor | None = None


def get_connection_pool_monitor(
    check_interval: float = 30.0,
    alert_callback: Callable[[PoolAlert], None] | None = None,
    enable_auto_recovery: bool = True,
) -> ConnectionPoolMonitor:
    """Get or create global connection pool monitor.
    
    Args:
        check_interval: Monitoring interval in seconds
        alert_callback: Callback function for alerts
        enable_auto_recovery: Whether to enable automatic recovery
        
    Returns:
        Connection pool monitor instance
    """
    global _pool_monitor
    
    if _pool_monitor is None:
        _pool_monitor = ConnectionPoolMonitor(
            check_interval=check_interval,
            alert_callback=alert_callback,
            enable_auto_recovery=enable_auto_recovery,
        )
    
    return _pool_monitor


async def start_connection_pool_monitoring() -> None:
    """Start global connection pool monitoring."""
    monitor = get_connection_pool_monitor()
    monitor.start_monitoring()


async def stop_connection_pool_monitoring() -> None:
    """Stop global connection pool monitoring."""
    global _pool_monitor
    
    if _pool_monitor:
        await _pool_monitor.stop_monitoring()
        _pool_monitor = None