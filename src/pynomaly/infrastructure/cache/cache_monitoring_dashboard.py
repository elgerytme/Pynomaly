"""Redis cache monitoring dashboard for Issue #99.

This module provides a comprehensive monitoring dashboard for the enhanced Redis
caching system with real-time metrics, visualizations, and alerting capabilities.
"""

from __future__ import annotations

import asyncio
import json
import logging
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional

from pynomaly.infrastructure.logging.structured_logger import StructuredLogger

from .redis_enhanced import EnhancedRedisCache, get_enhanced_redis_cache
from .cache_integration import get_cache_integration_manager

logger = StructuredLogger(__name__)


@dataclass
class AlertRule:
    """Configuration for cache monitoring alerts."""
    
    name: str
    metric: str
    operator: str  # "gt", "lt", "eq", "gte", "lte"
    threshold: float
    duration_seconds: int = 60
    severity: str = "warning"  # "info", "warning", "error", "critical"
    enabled: bool = True
    description: str = ""
    
    def evaluate(self, value: float) -> bool:
        """Evaluate if alert condition is met."""
        if self.operator == "gt":
            return value > self.threshold
        elif self.operator == "gte":
            return value >= self.threshold
        elif self.operator == "lt":
            return value < self.threshold
        elif self.operator == "lte":
            return value <= self.threshold
        elif self.operator == "eq":
            return value == self.threshold
        else:
            return False


@dataclass
class Alert:
    """Cache monitoring alert."""
    
    rule: AlertRule
    triggered_at: datetime
    current_value: float
    resolved_at: Optional[datetime] = None
    acknowledged_at: Optional[datetime] = None
    acknowledged_by: Optional[str] = None
    
    @property
    def is_active(self) -> bool:
        """Check if alert is still active."""
        return self.resolved_at is None
    
    @property
    def duration(self) -> timedelta:
        """Get alert duration."""
        end_time = self.resolved_at or datetime.utcnow()
        return end_time - self.triggered_at


@dataclass
class MetricSnapshot:
    """Snapshot of cache metrics at a point in time."""
    
    timestamp: datetime
    hit_rate: float
    operations_per_second: float
    avg_response_time_ms: float
    memory_usage_mb: float
    connection_count: int
    cache_size: int
    evictions: int
    compression_ratio: float
    error_rate: float
    
    @classmethod
    def from_cache_stats(cls, stats: Dict[str, Any]) -> MetricSnapshot:
        """Create metric snapshot from cache statistics."""
        enhanced_cache = stats.get("enhanced_cache", {})
        metrics = enhanced_cache.get("metrics", {})
        performance = enhanced_cache.get("performance", {})
        memory = enhanced_cache.get("memory", {})
        compression = enhanced_cache.get("compression", {})
        
        return cls(
            timestamp=datetime.utcnow(),
            hit_rate=metrics.get("hit_rate", 0.0),
            operations_per_second=metrics.get("operations_per_second", 0.0),
            avg_response_time_ms=performance.get("avg_response_time_ms", 0.0),
            memory_usage_mb=memory.get("usage_bytes", 0) / 1024 / 1024,
            connection_count=metrics.get("connection_count", 0),
            cache_size=metrics.get("cache_size", 0),
            evictions=metrics.get("evictions", 0),
            compression_ratio=compression.get("ratio", 1.0),
            error_rate=0.0  # Would need error tracking implementation
        )


class CacheMonitoringDashboard:
    """Comprehensive cache monitoring dashboard."""
    
    def __init__(
        self,
        enhanced_cache: Optional[EnhancedRedisCache] = None,
        max_snapshots: int = 1000,
        snapshot_interval: int = 30,
        enable_alerting: bool = True,
    ):
        """Initialize cache monitoring dashboard.
        
        Args:
            enhanced_cache: Enhanced Redis cache instance
            max_snapshots: Maximum number of metric snapshots to retain
            snapshot_interval: Interval between metric snapshots in seconds
            enable_alerting: Enable alerting system
        """
        self.enhanced_cache = enhanced_cache or get_enhanced_redis_cache()
        self.max_snapshots = max_snapshots
        self.snapshot_interval = snapshot_interval
        self.enable_alerting = enable_alerting
        
        # Metric snapshots for time series data
        self.metric_snapshots: List[MetricSnapshot] = []
        
        # Alert management
        self.alert_rules: List[AlertRule] = []
        self.active_alerts: List[Alert] = []
        self.alert_history: List[Alert] = []
        
        # Monitoring state
        self.monitoring_task: Optional[asyncio.Task] = None
        self.is_monitoring = False
        
        # Initialize default alert rules
        self._setup_default_alert_rules()
        
        logger.info("Cache monitoring dashboard initialized")
    
    def _setup_default_alert_rules(self) -> None:
        """Setup default alert rules for cache monitoring."""
        default_rules = [
            AlertRule(
                name="Low Hit Rate",
                metric="hit_rate",
                operator="lt",
                threshold=0.5,
                severity="warning",
                description="Cache hit rate below 50%"
            ),
            AlertRule(
                name="Critical Hit Rate",
                metric="hit_rate",
                operator="lt",
                threshold=0.3,
                severity="critical",
                description="Cache hit rate below 30%"
            ),
            AlertRule(
                name="High Response Time",
                metric="avg_response_time_ms",
                operator="gt",
                threshold=100.0,
                severity="warning",
                description="Average response time above 100ms"
            ),
            AlertRule(
                name="Critical Response Time",
                metric="avg_response_time_ms",
                operator="gt",
                threshold=500.0,
                severity="critical",
                description="Average response time above 500ms"
            ),
            AlertRule(
                name="High Memory Usage",
                metric="memory_usage_mb",
                operator="gt",
                threshold=1000.0,
                severity="warning",
                description="Memory usage above 1GB"
            ),
            AlertRule(
                name="Critical Memory Usage",
                metric="memory_usage_mb",
                operator="gt",
                threshold=2000.0,
                severity="critical",
                description="Memory usage above 2GB"
            ),
            AlertRule(
                name="Low Compression Ratio",
                metric="compression_ratio",
                operator="lt",
                threshold=1.5,
                severity="info",
                description="Compression ratio below 1.5x"
            ),
            AlertRule(
                name="High Eviction Rate",
                metric="evictions",
                operator="gt",
                threshold=100.0,
                severity="warning",
                description="High number of cache evictions"
            ),
        ]
        
        self.alert_rules.extend(default_rules)
        logger.info(f"Configured {len(default_rules)} default alert rules")
    
    async def start_monitoring(self) -> None:
        """Start cache monitoring."""
        if self.is_monitoring:
            logger.warning("Cache monitoring already running")
            return
        
        self.is_monitoring = True
        self.monitoring_task = asyncio.create_task(self._monitoring_loop())
        logger.info("Cache monitoring started")
    
    async def stop_monitoring(self) -> None:
        """Stop cache monitoring."""
        if not self.is_monitoring:
            return
        
        self.is_monitoring = False
        
        if self.monitoring_task:
            self.monitoring_task.cancel()
            try:
                await self.monitoring_task
            except asyncio.CancelledError:
                pass
        
        logger.info("Cache monitoring stopped")
    
    async def _monitoring_loop(self) -> None:
        """Main monitoring loop."""
        while self.is_monitoring:
            try:
                # Collect metrics snapshot
                await self._collect_metrics_snapshot()
                
                # Check alerts if enabled
                if self.enable_alerting:
                    await self._check_alerts()
                
                # Wait for next iteration
                await asyncio.sleep(self.snapshot_interval)
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Monitoring loop error: {e}")
                await asyncio.sleep(self.snapshot_interval)
    
    async def _collect_metrics_snapshot(self) -> None:
        """Collect and store metrics snapshot."""
        try:
            stats = await self.enhanced_cache.get_comprehensive_stats()
            snapshot = MetricSnapshot.from_cache_stats(stats)
            
            # Add to snapshots list
            self.metric_snapshots.append(snapshot)
            
            # Maintain max snapshots limit
            if len(self.metric_snapshots) > self.max_snapshots:
                self.metric_snapshots.pop(0)
            
            logger.debug(
                "Collected metrics snapshot",
                hit_rate=snapshot.hit_rate,
                ops_per_sec=snapshot.operations_per_second,
                response_time_ms=snapshot.avg_response_time_ms
            )
            
        except Exception as e:
            logger.error(f"Failed to collect metrics snapshot: {e}")
    
    async def _check_alerts(self) -> None:
        """Check alert rules and trigger alerts."""
        if not self.metric_snapshots:
            return
        
        latest_snapshot = self.metric_snapshots[-1]
        
        for rule in self.alert_rules:
            if not rule.enabled:
                continue
            
            try:
                # Get metric value from snapshot
                metric_value = getattr(latest_snapshot, rule.metric, None)
                if metric_value is None:
                    continue
                
                # Check if alert condition is met
                should_trigger = rule.evaluate(metric_value)
                
                # Check if we already have an active alert for this rule
                existing_alert = next(
                    (alert for alert in self.active_alerts if alert.rule.name == rule.name),
                    None
                )
                
                if should_trigger and not existing_alert:
                    # Trigger new alert
                    alert = Alert(
                        rule=rule,
                        triggered_at=datetime.utcnow(),
                        current_value=metric_value
                    )
                    self.active_alerts.append(alert)
                    self.alert_history.append(alert)
                    
                    logger.warning(
                        f"Alert triggered: {rule.name}",
                        metric=rule.metric,
                        current_value=metric_value,
                        threshold=rule.threshold,
                        severity=rule.severity
                    )
                    
                elif not should_trigger and existing_alert:
                    # Resolve existing alert
                    existing_alert.resolved_at = datetime.utcnow()
                    self.active_alerts.remove(existing_alert)
                    
                    logger.info(
                        f"Alert resolved: {rule.name}",
                        duration=str(existing_alert.duration)
                    )
                    
            except Exception as e:
                logger.error(f"Error checking alert rule {rule.name}: {e}")
    
    def add_alert_rule(self, rule: AlertRule) -> None:
        """Add custom alert rule."""
        self.alert_rules.append(rule)
        logger.info(f"Added alert rule: {rule.name}")
    
    def remove_alert_rule(self, rule_name: str) -> bool:
        """Remove alert rule by name."""
        for i, rule in enumerate(self.alert_rules):
            if rule.name == rule_name:
                del self.alert_rules[i]
                logger.info(f"Removed alert rule: {rule_name}")
                return True
        return False
    
    def acknowledge_alert(self, alert_id: str, acknowledged_by: str) -> bool:
        """Acknowledge an active alert."""
        for alert in self.active_alerts:
            if str(id(alert)) == alert_id:
                alert.acknowledged_at = datetime.utcnow()
                alert.acknowledged_by = acknowledged_by
                logger.info(f"Alert acknowledged: {alert.rule.name} by {acknowledged_by}")
                return True
        return False
    
    def get_dashboard_data(self, time_range_minutes: int = 60) -> Dict[str, Any]:
        """Get comprehensive dashboard data."""
        try:
            cutoff_time = datetime.utcnow() - timedelta(minutes=time_range_minutes)
            
            # Filter snapshots by time range
            recent_snapshots = [
                snapshot for snapshot in self.metric_snapshots
                if snapshot.timestamp >= cutoff_time
            ]
            
            # Calculate summary statistics
            if recent_snapshots:
                hit_rates = [s.hit_rate for s in recent_snapshots]
                response_times = [s.avg_response_time_ms for s in recent_snapshots]
                ops_per_sec = [s.operations_per_second for s in recent_snapshots]
                
                summary_stats = {
                    "hit_rate": {
                        "current": hit_rates[-1] if hit_rates else 0,
                        "average": sum(hit_rates) / len(hit_rates) if hit_rates else 0,
                        "min": min(hit_rates) if hit_rates else 0,
                        "max": max(hit_rates) if hit_rates else 0,
                    },
                    "response_time": {
                        "current": response_times[-1] if response_times else 0,
                        "average": sum(response_times) / len(response_times) if response_times else 0,
                        "min": min(response_times) if response_times else 0,
                        "max": max(response_times) if response_times else 0,
                    },
                    "throughput": {
                        "current": ops_per_sec[-1] if ops_per_sec else 0,
                        "average": sum(ops_per_sec) / len(ops_per_sec) if ops_per_sec else 0,
                        "min": min(ops_per_sec) if ops_per_sec else 0,
                        "max": max(ops_per_sec) if ops_per_sec else 0,
                    }
                }
            else:
                summary_stats = {}
            
            # Time series data for charts
            time_series_data = []
            for snapshot in recent_snapshots:
                time_series_data.append({
                    "timestamp": snapshot.timestamp.isoformat(),
                    "hit_rate": snapshot.hit_rate,
                    "response_time_ms": snapshot.avg_response_time_ms,
                    "ops_per_second": snapshot.operations_per_second,
                    "memory_usage_mb": snapshot.memory_usage_mb,
                    "connection_count": snapshot.connection_count,
                    "compression_ratio": snapshot.compression_ratio,
                })
            
            # Active alerts
            active_alerts_data = []
            for alert in self.active_alerts:
                active_alerts_data.append({
                    "id": str(id(alert)),
                    "rule_name": alert.rule.name,
                    "metric": alert.rule.metric,
                    "severity": alert.rule.severity,
                    "current_value": alert.current_value,
                    "threshold": alert.rule.threshold,
                    "triggered_at": alert.triggered_at.isoformat(),
                    "duration": str(alert.duration),
                    "acknowledged": alert.acknowledged_at is not None,
                    "acknowledged_by": alert.acknowledged_by,
                })
            
            # Alert history (last 24 hours)
            alert_history_cutoff = datetime.utcnow() - timedelta(hours=24)
            recent_alert_history = [
                {
                    "rule_name": alert.rule.name,
                    "severity": alert.rule.severity,
                    "triggered_at": alert.triggered_at.isoformat(),
                    "resolved_at": alert.resolved_at.isoformat() if alert.resolved_at else None,
                    "duration": str(alert.duration),
                }
                for alert in self.alert_history
                if alert.triggered_at >= alert_history_cutoff
            ]
            
            dashboard_data = {
                "metadata": {
                    "generated_at": datetime.utcnow().isoformat(),
                    "time_range_minutes": time_range_minutes,
                    "snapshot_count": len(recent_snapshots),
                    "monitoring_active": self.is_monitoring,
                },
                "summary": summary_stats,
                "time_series": time_series_data,
                "alerts": {
                    "active": active_alerts_data,
                    "history": recent_alert_history,
                    "total_rules": len(self.alert_rules),
                    "enabled_rules": len([r for r in self.alert_rules if r.enabled]),
                },
                "health_indicators": self._get_health_indicators(),
            }
            
            return dashboard_data
            
        except Exception as e:
            logger.error(f"Failed to generate dashboard data: {e}")
            return {
                "error": str(e),
                "generated_at": datetime.utcnow().isoformat(),
            }
    
    def _get_health_indicators(self) -> Dict[str, Any]:
        """Get health indicators for dashboard."""
        if not self.metric_snapshots:
            return {"status": "no_data"}
        
        latest = self.metric_snapshots[-1]
        
        # Determine health status based on latest metrics
        health_score = 0
        max_score = 5
        
        # Hit rate indicator
        if latest.hit_rate >= 0.8:
            health_score += 2
        elif latest.hit_rate >= 0.6:
            health_score += 1
        
        # Response time indicator
        if latest.avg_response_time_ms <= 50:
            health_score += 2
        elif latest.avg_response_time_ms <= 100:
            health_score += 1
        
        # Active alerts indicator
        if len(self.active_alerts) == 0:
            health_score += 1
        
        # Determine overall health
        if health_score >= 4:
            overall_health = "excellent"
        elif health_score >= 3:
            overall_health = "good"
        elif health_score >= 2:
            overall_health = "fair"
        else:
            overall_health = "poor"
        
        return {
            "status": overall_health,
            "score": health_score,
            "max_score": max_score,
            "indicators": {
                "hit_rate": {
                    "value": latest.hit_rate,
                    "status": "good" if latest.hit_rate >= 0.8 else "warning" if latest.hit_rate >= 0.6 else "poor"
                },
                "response_time": {
                    "value": latest.avg_response_time_ms,
                    "status": "good" if latest.avg_response_time_ms <= 50 else "warning" if latest.avg_response_time_ms <= 100 else "poor"
                },
                "active_alerts": {
                    "value": len(self.active_alerts),
                    "status": "good" if len(self.active_alerts) == 0 else "warning"
                },
                "memory_usage": {
                    "value": latest.memory_usage_mb,
                    "status": "good" if latest.memory_usage_mb <= 500 else "warning" if latest.memory_usage_mb <= 1000 else "poor"
                }
            }
        }
    
    async def export_metrics(
        self, 
        format: str = "json", 
        time_range_hours: int = 24
    ) -> str:
        """Export metrics data in specified format."""
        cutoff_time = datetime.utcnow() - timedelta(hours=time_range_hours)
        
        export_snapshots = [
            snapshot for snapshot in self.metric_snapshots
            if snapshot.timestamp >= cutoff_time
        ]
        
        if format.lower() == "json":
            export_data = {
                "export_metadata": {
                    "generated_at": datetime.utcnow().isoformat(),
                    "time_range_hours": time_range_hours,
                    "total_snapshots": len(export_snapshots),
                },
                "metrics": [
                    {
                        "timestamp": s.timestamp.isoformat(),
                        "hit_rate": s.hit_rate,
                        "operations_per_second": s.operations_per_second,
                        "avg_response_time_ms": s.avg_response_time_ms,
                        "memory_usage_mb": s.memory_usage_mb,
                        "connection_count": s.connection_count,
                        "compression_ratio": s.compression_ratio,
                        "evictions": s.evictions,
                    }
                    for s in export_snapshots
                ]
            }
            return json.dumps(export_data, indent=2)
        
        elif format.lower() == "csv":
            lines = ["timestamp,hit_rate,ops_per_second,response_time_ms,memory_mb,connections,compression_ratio,evictions"]
            
            for s in export_snapshots:
                lines.append(
                    f"{s.timestamp.isoformat()},{s.hit_rate},{s.operations_per_second},"
                    f"{s.avg_response_time_ms},{s.memory_usage_mb},{s.connection_count},"
                    f"{s.compression_ratio},{s.evictions}"
                )
            
            return "\n".join(lines)
        
        else:
            raise ValueError(f"Unsupported export format: {format}")
    
    async def generate_performance_report(self) -> Dict[str, Any]:
        """Generate comprehensive performance report."""
        if not self.metric_snapshots:
            return {"error": "No metrics data available"}
        
        # Calculate time-based statistics
        now = datetime.utcnow()
        last_hour = [s for s in self.metric_snapshots if s.timestamp >= now - timedelta(hours=1)]
        last_day = [s for s in self.metric_snapshots if s.timestamp >= now - timedelta(days=1)]
        last_week = [s for s in self.metric_snapshots if s.timestamp >= now - timedelta(days=7)]
        
        def calculate_stats(snapshots):
            if not snapshots:
                return {}
            
            hit_rates = [s.hit_rate for s in snapshots]
            response_times = [s.avg_response_time_ms for s in snapshots]
            ops_per_sec = [s.operations_per_second for s in snapshots]
            
            return {
                "hit_rate_avg": sum(hit_rates) / len(hit_rates),
                "hit_rate_min": min(hit_rates),
                "hit_rate_max": max(hit_rates),
                "response_time_avg": sum(response_times) / len(response_times),
                "response_time_min": min(response_times),
                "response_time_max": max(response_times),
                "throughput_avg": sum(ops_per_sec) / len(ops_per_sec),
                "throughput_min": min(ops_per_sec),
                "throughput_max": max(ops_per_sec),
            }
        
        report = {
            "report_metadata": {
                "generated_at": now.isoformat(),
                "total_snapshots": len(self.metric_snapshots),
                "time_range": {
                    "oldest": self.metric_snapshots[0].timestamp.isoformat() if self.metric_snapshots else None,
                    "newest": self.metric_snapshots[-1].timestamp.isoformat() if self.metric_snapshots else None,
                }
            },
            "performance_summary": {
                "last_hour": calculate_stats(last_hour),
                "last_day": calculate_stats(last_day),
                "last_week": calculate_stats(last_week),
            },
            "alert_summary": {
                "total_alerts_24h": len([a for a in self.alert_history if a.triggered_at >= now - timedelta(days=1)]),
                "active_alerts": len(self.active_alerts),
                "critical_alerts_24h": len([
                    a for a in self.alert_history 
                    if a.triggered_at >= now - timedelta(days=1) and a.rule.severity == "critical"
                ]),
            },
            "recommendations": self._generate_recommendations(),
        }
        
        return report
    
    def _generate_recommendations(self) -> List[str]:
        """Generate performance recommendations based on current metrics."""
        recommendations = []
        
        if not self.metric_snapshots:
            return ["No metrics data available for recommendations"]
        
        latest = self.metric_snapshots[-1]
        
        # Hit rate recommendations
        if latest.hit_rate < 0.5:
            recommendations.append("Consider implementing cache warming strategies to improve hit rate")
        elif latest.hit_rate < 0.7:
            recommendations.append("Review cache TTL settings and key patterns to optimize hit rate")
        
        # Response time recommendations
        if latest.avg_response_time_ms > 100:
            recommendations.append("High response times detected - consider Redis cluster or connection optimization")
        elif latest.avg_response_time_ms > 50:
            recommendations.append("Response times could be improved - review network latency and Redis configuration")
        
        # Memory recommendations
        if latest.memory_usage_mb > 1000:
            recommendations.append("High memory usage - consider implementing more aggressive eviction policies")
        elif latest.memory_usage_mb > 500:
            recommendations.append("Monitor memory usage trends and consider cache size optimization")
        
        # Compression recommendations
        if latest.compression_ratio < 1.5:
            recommendations.append("Low compression ratio - review compression settings and data types being cached")
        
        # Eviction recommendations
        if latest.evictions > 50:
            recommendations.append("High eviction rate - consider increasing cache memory or optimizing TTL values")
        
        if not recommendations:
            recommendations.append("Cache performance looks good - no immediate optimizations needed")
        
        return recommendations


# Global dashboard instance
_cache_monitoring_dashboard: Optional[CacheMonitoringDashboard] = None


def get_cache_monitoring_dashboard(**kwargs) -> CacheMonitoringDashboard:
    """Get or create global cache monitoring dashboard."""
    global _cache_monitoring_dashboard
    
    if _cache_monitoring_dashboard is None:
        _cache_monitoring_dashboard = CacheMonitoringDashboard(**kwargs)
    
    return _cache_monitoring_dashboard


async def close_cache_monitoring_dashboard() -> None:
    """Close global cache monitoring dashboard."""
    global _cache_monitoring_dashboard
    
    if _cache_monitoring_dashboard:
        await _cache_monitoring_dashboard.stop_monitoring()
        _cache_monitoring_dashboard = None