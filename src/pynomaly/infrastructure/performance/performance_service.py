"""Performance management service for comprehensive performance optimization."""

from __future__ import annotations

import asyncio
import time
from typing import Any, Dict, List, Optional

import structlog

from .connection_pooling import ConnectionPoolManager, PoolConfiguration
from .query_optimization import QueryOptimizer

logger = structlog.get_logger(__name__)


class PerformanceService:
    """Comprehensive performance management service."""
    
    def __init__(
        self,
        pool_manager: ConnectionPoolManager,
        query_optimizer: Optional[QueryOptimizer] = None,
        monitoring_interval: float = 300.0  # 5 minutes
    ):
        """Initialize performance service.
        
        Args:
            pool_manager: Connection pool manager
            query_optimizer: Query optimizer (optional)
            monitoring_interval: Performance monitoring interval in seconds
        """
        self.pool_manager = pool_manager
        self.query_optimizer = query_optimizer
        self.monitoring_interval = monitoring_interval
        
        # Performance monitoring
        self._monitoring_task: Optional[asyncio.Task] = None
        self._performance_history: List[Dict[str, Any]] = []
        self._max_history = 288  # 24 hours at 5-minute intervals
        
        # Performance alerts
        self._alert_thresholds = {
            "pool_error_rate": 0.1,  # 10% error rate
            "avg_response_time": 2.0,  # 2 seconds
            "cache_hit_rate": 0.5,  # 50% hit rate
            "slow_query_count": 10,  # 10 slow queries per interval
        }
        
        self._alerts: List[Dict[str, Any]] = []
        self._max_alerts = 100
    
    def start_monitoring(self) -> None:
        """Start performance monitoring."""
        if self._monitoring_task is None:
            self._monitoring_task = asyncio.create_task(self._monitoring_loop())
            logger.info("Performance monitoring started", interval=self.monitoring_interval)
    
    async def stop_monitoring(self) -> None:
        """Stop performance monitoring."""
        if self._monitoring_task:
            self._monitoring_task.cancel()
            try:
                await self._monitoring_task
            except asyncio.CancelledError:
                pass
            self._monitoring_task = None
            logger.info("Performance monitoring stopped")
    
    async def _monitoring_loop(self) -> None:
        """Performance monitoring loop."""
        while True:
            try:
                await asyncio.sleep(self.monitoring_interval)
                await self._collect_performance_metrics()
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error("Performance monitoring error", error=str(e))
    
    async def _collect_performance_metrics(self) -> None:
        """Collect performance metrics."""
        try:
            timestamp = time.time()
            metrics = {
                "timestamp": timestamp,
                "connection_pools": {},
                "query_performance": {},
                "cache_performance": {},
                "alerts": []
            }
            
            # Collect pool metrics
            pool_stats = self.pool_manager.get_all_stats()
            for pool_name, stats in pool_stats.items():
                pool_metrics = {
                    "active_connections": stats.active_connections,
                    "total_requests": stats.total_requests,
                    "successful_requests": stats.successful_requests,
                    "failed_requests": stats.failed_requests,
                    "error_rate": stats.failed_requests / max(1, stats.total_requests),
                    "avg_response_time": stats.avg_response_time,
                    "connection_errors": stats.connection_errors
                }
                
                metrics["connection_pools"][pool_name] = pool_metrics
                
                # Check for alerts
                error_rate = pool_metrics["error_rate"]
                if error_rate > self._alert_thresholds["pool_error_rate"]:
                    await self._create_alert(
                        "high_pool_error_rate",
                        f"Pool {pool_name} has high error rate: {error_rate:.2%}",
                        {"pool_name": pool_name, "error_rate": error_rate}
                    )
                
                if pool_metrics["avg_response_time"] > self._alert_thresholds["avg_response_time"]:
                    await self._create_alert(
                        "slow_pool_response",
                        f"Pool {pool_name} has slow response time: {pool_metrics['avg_response_time']:.2f}s",
                        {"pool_name": pool_name, "response_time": pool_metrics["avg_response_time"]}
                    )
            
            # Collect query metrics
            if self.query_optimizer:
                query_summary = self.query_optimizer.performance_tracker.get_performance_summary()
                metrics["query_performance"] = query_summary
                
                # Check for slow query alerts
                if query_summary["slow_queries"] > self._alert_thresholds["slow_query_count"]:
                    await self._create_alert(
                        "high_slow_query_count",
                        f"High number of slow queries: {query_summary['slow_queries']}",
                        {"slow_query_count": query_summary["slow_queries"]}
                    )
                
                # Collect cache metrics
                cache_stats = self.query_optimizer.cache.get_stats()
                metrics["cache_performance"] = cache_stats
                
                # Check cache hit rate
                if cache_stats["hit_rate"] < self._alert_thresholds["cache_hit_rate"]:
                    await self._create_alert(
                        "low_cache_hit_rate",
                        f"Low cache hit rate: {cache_stats['hit_rate']:.2%}",
                        {"hit_rate": cache_stats["hit_rate"]}
                    )
            
            # Store metrics
            self._performance_history.append(metrics)
            if len(self._performance_history) > self._max_history:
                self._performance_history.pop(0)
            
            logger.debug("Performance metrics collected", pools=len(metrics["connection_pools"]))
            
        except Exception as e:
            logger.error("Failed to collect performance metrics", error=str(e))
    
    async def _create_alert(
        self,
        alert_type: str,
        message: str,
        data: Dict[str, Any]
    ) -> None:
        """Create performance alert.
        
        Args:
            alert_type: Type of alert
            message: Alert message
            data: Alert data
        """
        alert = {
            "timestamp": time.time(),
            "type": alert_type,
            "message": message,
            "data": data,
            "resolved": False
        }
        
        self._alerts.append(alert)
        if len(self._alerts) > self._max_alerts:
            self._alerts.pop(0)
        
        logger.warning("Performance alert created", alert=alert)
    
    async def optimize_all(self) -> Dict[str, Any]:
        """Perform comprehensive performance optimization.
        
        Returns:
            Optimization results
        """
        results = {
            "timestamp": time.time(),
            "database_optimization": {},
            "pool_optimization": {},
            "cache_optimization": {},
            "recommendations": []
        }
        
        try:
            # Database optimization
            if self.query_optimizer:
                db_results = await self.query_optimizer.optimize_database()
                results["database_optimization"] = db_results
                
                # Add recommendations based on metrics
                query_summary = self.query_optimizer.performance_tracker.get_performance_summary()
                
                if query_summary["slow_queries"] > 0:
                    results["recommendations"].append({
                        "type": "query_optimization",
                        "message": f"Found {query_summary['slow_queries']} slow queries that could benefit from optimization",
                        "action": "review_slow_queries"
                    })
                
                cache_stats = self.query_optimizer.cache.get_stats()
                if cache_stats["hit_rate"] < 0.7:
                    results["recommendations"].append({
                        "type": "cache_optimization",
                        "message": f"Cache hit rate is {cache_stats['hit_rate']:.1%}, consider increasing cache size or TTL",
                        "action": "tune_cache_settings"
                    })
            
            # Pool optimization recommendations
            pool_stats = self.pool_manager.get_all_stats()
            for pool_name, stats in pool_stats.items():
                error_rate = stats.failed_requests / max(1, stats.total_requests)
                
                if error_rate > 0.05:  # More than 5% error rate
                    results["recommendations"].append({
                        "type": "pool_reliability",
                        "message": f"Pool {pool_name} has {error_rate:.1%} error rate, consider health check tuning",
                        "action": "tune_pool_health_checks"
                    })
                
                if stats.avg_response_time > 1.0:  # More than 1 second
                    results["recommendations"].append({
                        "type": "pool_performance",
                        "message": f"Pool {pool_name} has slow response time ({stats.avg_response_time:.2f}s), consider increasing pool size",
                        "action": "increase_pool_size"
                    })
            
            logger.info("Performance optimization completed", recommendations_count=len(results["recommendations"]))
            
        except Exception as e:
            results["error"] = str(e)
            logger.error("Performance optimization failed", error=str(e))
        
        return results
    
    def get_performance_summary(self) -> Dict[str, Any]:
        """Get current performance summary.
        
        Returns:
            Performance summary
        """
        if not self._performance_history:
            return {"status": "no_data"}
        
        latest = self._performance_history[-1]
        
        # Calculate trends if we have enough data
        trends = {}
        if len(self._performance_history) >= 2:
            previous = self._performance_history[-2]
            
            # Pool trends
            for pool_name in latest["connection_pools"]:
                if pool_name in previous["connection_pools"]:
                    current_response_time = latest["connection_pools"][pool_name]["avg_response_time"]
                    previous_response_time = previous["connection_pools"][pool_name]["avg_response_time"]
                    
                    if previous_response_time > 0:
                        trend = (current_response_time - previous_response_time) / previous_response_time
                        trends[f"pool_{pool_name}_response_time"] = trend
            
            # Query performance trends
            if "query_performance" in latest and "query_performance" in previous:
                current_avg = latest["query_performance"].get("avg_time", 0)
                previous_avg = previous["query_performance"].get("avg_time", 0)
                
                if previous_avg > 0:
                    trends["query_avg_time"] = (current_avg - previous_avg) / previous_avg
        
        return {
            "timestamp": latest["timestamp"],
            "connection_pools": latest["connection_pools"],
            "query_performance": latest.get("query_performance", {}),
            "cache_performance": latest.get("cache_performance", {}),
            "trends": trends,
            "active_alerts": len([a for a in self._alerts if not a["resolved"]]),
            "monitoring_duration": len(self._performance_history) * self.monitoring_interval / 3600,  # hours
        }
    
    def get_performance_history(
        self,
        hours: Optional[float] = None,
        limit: Optional[int] = None
    ) -> List[Dict[str, Any]]:
        """Get performance history.
        
        Args:
            hours: Number of hours of history to return
            limit: Maximum number of entries to return
            
        Returns:
            Performance history
        """
        if hours is not None:
            cutoff_time = time.time() - (hours * 3600)
            history = [
                entry for entry in self._performance_history
                if entry["timestamp"] >= cutoff_time
            ]
        else:
            history = self._performance_history.copy()
        
        if limit is not None:
            history = history[-limit:]
        
        return history
    
    def get_alerts(
        self,
        unresolved_only: bool = True,
        limit: Optional[int] = None
    ) -> List[Dict[str, Any]]:
        """Get performance alerts.
        
        Args:
            unresolved_only: Whether to return only unresolved alerts
            limit: Maximum number of alerts to return
            
        Returns:
            List of alerts
        """
        if unresolved_only:
            alerts = [alert for alert in self._alerts if not alert["resolved"]]
        else:
            alerts = self._alerts.copy()
        
        if limit is not None:
            alerts = alerts[-limit:]
        
        return alerts
    
    def resolve_alert(self, alert_index: int) -> bool:
        """Resolve a performance alert.
        
        Args:
            alert_index: Index of the alert to resolve
            
        Returns:
            True if alert was resolved, False if not found
        """
        if 0 <= alert_index < len(self._alerts):
            self._alerts[alert_index]["resolved"] = True
            self._alerts[alert_index]["resolved_at"] = time.time()
            logger.info("Performance alert resolved", index=alert_index)
            return True
        return False
    
    def update_alert_thresholds(self, thresholds: Dict[str, float]) -> None:
        """Update alert thresholds.
        
        Args:
            thresholds: New threshold values
        """
        for key, value in thresholds.items():
            if key in self._alert_thresholds:
                old_value = self._alert_thresholds[key]
                self._alert_thresholds[key] = value
                logger.info(
                    "Alert threshold updated",
                    threshold=key,
                    old_value=old_value,
                    new_value=value
                )
    
    async def cleanup(self) -> None:
        """Cleanup performance service resources."""
        await self.stop_monitoring()
        logger.info("Performance service cleanup completed")