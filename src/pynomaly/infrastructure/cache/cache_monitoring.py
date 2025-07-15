"""Comprehensive cache monitoring and alerting system for Issue #99."""

from __future__ import annotations

import asyncio
import json
import logging
import time
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from typing import Any, Callable, Dict, List, Optional, Set

try:
    import aiohttp
except ImportError:
    aiohttp = None
from pydantic import BaseModel

logger = logging.getLogger(__name__)


class AlertSeverity(str, Enum):
    """Alert severity levels."""
    
    INFO = "info"
    WARNING = "warning"
    CRITICAL = "critical"
    EMERGENCY = "emergency"


class AlertType(str, Enum):
    """Types of cache alerts."""
    
    HIGH_MISS_RATE = "high_miss_rate"
    SLOW_RESPONSE = "slow_response"
    CONNECTION_FAILURE = "connection_failure"
    MEMORY_PRESSURE = "memory_pressure"
    HIGH_ERROR_RATE = "high_error_rate"
    CIRCUIT_BREAKER_OPEN = "circuit_breaker_open"
    CACHE_UNAVAILABLE = "cache_unavailable"
    PERFORMANCE_DEGRADATION = "performance_degradation"
    ANOMALOUS_PATTERN = "anomalous_pattern"


@dataclass
class Alert:
    """Cache alert data structure."""
    
    alert_type: AlertType
    severity: AlertSeverity
    message: str
    timestamp: datetime = field(default_factory=datetime.utcnow)
    metadata: Dict[str, Any] = field(default_factory=dict)
    resolved: bool = False
    resolved_at: Optional[datetime] = None
    acknowledgments: List[str] = field(default_factory=list)


@dataclass
class MonitoringThresholds:
    """Configurable monitoring thresholds."""
    
    # Performance thresholds
    max_response_time_ms: float = 100.0
    min_hit_rate: float = 0.8
    max_error_rate: float = 0.05
    
    # Memory thresholds
    max_memory_usage_mb: int = 1000
    max_memory_usage_percent: float = 80.0
    
    # Connection thresholds
    max_connection_failures: int = 5
    max_timeout_rate: float = 0.1
    
    # Pattern detection
    anomaly_detection_window_minutes: int = 15
    anomaly_threshold_multiplier: float = 2.0


@dataclass
class CacheHealthScore:
    """Overall cache health assessment."""
    
    score: float  # 0-100
    status: str  # "excellent", "good", "warning", "critical"
    factors: Dict[str, float] = field(default_factory=dict)
    recommendations: List[str] = field(default_factory=list)


class MetricsAggregator:
    """Aggregate and analyze cache metrics over time."""
    
    def __init__(self, window_size_minutes: int = 60):
        self.window_size = timedelta(minutes=window_size_minutes)
        self.metrics_history: List[Dict[str, Any]] = []
        self._lock = asyncio.Lock()
    
    async def add_metrics(self, metrics: Dict[str, Any]):
        """Add metrics sample to history."""
        async with self._lock:
            metrics['timestamp'] = datetime.utcnow()
            self.metrics_history.append(metrics)
            
            # Remove old metrics outside the window
            cutoff_time = datetime.utcnow() - self.window_size
            self.metrics_history = [
                m for m in self.metrics_history 
                if m['timestamp'] > cutoff_time
            ]
    
    def get_average_metric(self, metric_name: str) -> float:
        """Get average value of a metric over the window."""
        if not self.metrics_history:
            return 0.0
        
        values = [
            m.get(metric_name, 0) 
            for m in self.metrics_history 
            if metric_name in m
        ]
        
        return sum(values) / len(values) if values else 0.0
    
    def get_trend(self, metric_name: str) -> str:
        """Get trend direction for a metric."""
        if len(self.metrics_history) < 2:
            return "stable"
        
        # Compare recent vs older values
        recent_count = len(self.metrics_history) // 4  # Last 25%
        recent_values = [
            m.get(metric_name, 0) 
            for m in self.metrics_history[-recent_count:]
        ]
        older_values = [
            m.get(metric_name, 0) 
            for m in self.metrics_history[:-recent_count]
        ]
        
        if not recent_values or not older_values:
            return "stable"
        
        recent_avg = sum(recent_values) / len(recent_values)
        older_avg = sum(older_values) / len(older_values)
        
        change_percent = ((recent_avg - older_avg) / older_avg) * 100 if older_avg > 0 else 0
        
        if change_percent > 10:
            return "increasing"
        elif change_percent < -10:
            return "decreasing"
        else:
            return "stable"
    
    def detect_anomalies(self, metric_name: str, threshold_multiplier: float = 2.0) -> bool:
        """Detect anomalies in metric values."""
        if len(self.metrics_history) < 10:  # Need sufficient data
            return False
        
        values = [
            m.get(metric_name, 0) 
            for m in self.metrics_history 
            if metric_name in m
        ]
        
        if not values:
            return False
        
        # Calculate mean and standard deviation
        mean = sum(values) / len(values)
        variance = sum((x - mean) ** 2 for x in values) / len(values)
        std_dev = variance ** 0.5
        
        # Check if recent values are anomalous
        recent_values = values[-3:]  # Last 3 values
        for value in recent_values:
            if abs(value - mean) > threshold_multiplier * std_dev:
                return True
        
        return False


class AlertManager:
    """Manage cache alerts and notifications."""
    
    def __init__(self):
        self.active_alerts: Dict[str, Alert] = {}
        self.alert_history: List[Alert] = []
        self.notification_handlers: List[Callable] = []
        self._lock = asyncio.Lock()
    
    def add_notification_handler(self, handler: Callable[[Alert], None]):
        """Add notification handler for alerts."""
        self.notification_handlers.append(handler)
    
    async def create_alert(
        self, 
        alert_type: AlertType, 
        severity: AlertSeverity, 
        message: str, 
        metadata: Optional[Dict[str, Any]] = None
    ) -> Alert:
        """Create and process new alert."""
        alert = Alert(
            alert_type=alert_type,
            severity=severity,
            message=message,
            metadata=metadata or {}
        )
        
        async with self._lock:
            alert_key = f"{alert_type}_{alert.timestamp.strftime('%Y%m%d_%H%M')}"
            
            # Check if similar alert already exists
            existing_alert = self.active_alerts.get(alert_key)
            if existing_alert and not existing_alert.resolved:
                # Update existing alert
                existing_alert.metadata.update(alert.metadata)
                return existing_alert
            
            # Add new alert
            self.active_alerts[alert_key] = alert
            self.alert_history.append(alert)
            
            # Send notifications
            await self._send_notifications(alert)
            
            logger.warning(f"Cache alert created: {alert.alert_type} - {alert.message}")
            
            return alert
    
    async def resolve_alert(self, alert_key: str, resolver: str = "system"):
        """Resolve an active alert."""
        async with self._lock:
            if alert_key in self.active_alerts:
                alert = self.active_alerts[alert_key]
                alert.resolved = True
                alert.resolved_at = datetime.utcnow()
                alert.acknowledgments.append(f"Resolved by {resolver}")
                
                logger.info(f"Cache alert resolved: {alert.alert_type}")
    
    async def acknowledge_alert(self, alert_key: str, acknowledger: str):
        """Acknowledge an alert."""
        async with self._lock:
            if alert_key in self.active_alerts:
                alert = self.active_alerts[alert_key]
                alert.acknowledgments.append(f"Acknowledged by {acknowledger}")
                
                logger.info(f"Cache alert acknowledged: {alert.alert_type}")
    
    async def _send_notifications(self, alert: Alert):
        """Send notifications for an alert."""
        for handler in self.notification_handlers:
            try:
                if asyncio.iscoroutinefunction(handler):
                    await handler(alert)
                else:
                    handler(alert)
            except Exception as e:
                logger.error(f"Failed to send notification: {e}")
    
    def get_active_alerts(self) -> List[Alert]:
        """Get all active (unresolved) alerts."""
        return [alert for alert in self.active_alerts.values() if not alert.resolved]
    
    def get_alert_summary(self) -> Dict[str, Any]:
        """Get summary of alert status."""
        active_alerts = self.get_active_alerts()
        
        severity_counts = {}
        type_counts = {}
        
        for alert in active_alerts:
            severity_counts[alert.severity] = severity_counts.get(alert.severity, 0) + 1
            type_counts[alert.alert_type] = type_counts.get(alert.alert_type, 0) + 1
        
        return {
            "total_active": len(active_alerts),
            "severity_breakdown": severity_counts,
            "type_breakdown": type_counts,
            "total_historical": len(self.alert_history),
            "last_24h": len([
                a for a in self.alert_history 
                if a.timestamp > datetime.utcnow() - timedelta(hours=24)
            ])
        }


class CacheHealthAnalyzer:
    """Analyze cache health and provide recommendations."""
    
    def __init__(self, thresholds: MonitoringThresholds):
        self.thresholds = thresholds
    
    def analyze_health(self, metrics: Dict[str, Any], trends: Dict[str, str]) -> CacheHealthScore:
        """Analyze overall cache health."""
        factors = {}
        recommendations = []
        
        # Performance factor (40% weight)
        performance_score = self._analyze_performance(metrics, recommendations)
        factors['performance'] = performance_score
        
        # Reliability factor (30% weight)
        reliability_score = self._analyze_reliability(metrics, recommendations)
        factors['reliability'] = reliability_score
        
        # Resource utilization factor (20% weight)
        resource_score = self._analyze_resources(metrics, recommendations)
        factors['resources'] = resource_score
        
        # Trends factor (10% weight)
        trends_score = self._analyze_trends(trends, recommendations)
        factors['trends'] = trends_score
        
        # Calculate weighted overall score
        overall_score = (
            performance_score * 0.4 +
            reliability_score * 0.3 +
            resource_score * 0.2 +
            trends_score * 0.1
        )
        
        # Determine status
        if overall_score >= 90:
            status = "excellent"
        elif overall_score >= 75:
            status = "good"
        elif overall_score >= 60:
            status = "warning"
        else:
            status = "critical"
        
        return CacheHealthScore(
            score=overall_score,
            status=status,
            factors=factors,
            recommendations=recommendations
        )
    
    def _analyze_performance(self, metrics: Dict[str, Any], recommendations: List[str]) -> float:
        """Analyze performance metrics."""
        score = 100.0
        
        # Hit rate analysis
        hit_rate = metrics.get('hit_rate', 0)
        if hit_rate < self.thresholds.min_hit_rate:
            score -= (self.thresholds.min_hit_rate - hit_rate) * 100
            recommendations.append(f"Improve cache hit rate (current: {hit_rate:.1%})")
        
        # Response time analysis
        avg_response_time = metrics.get('average_response_time', 0)
        if avg_response_time > self.thresholds.max_response_time_ms:
            score -= min(50, (avg_response_time - self.thresholds.max_response_time_ms) / 10)
            recommendations.append(f"Optimize response time (current: {avg_response_time:.1f}ms)")
        
        return max(0, score)
    
    def _analyze_reliability(self, metrics: Dict[str, Any], recommendations: List[str]) -> float:
        """Analyze reliability metrics."""
        score = 100.0
        
        # Error rate analysis
        error_rate = metrics.get('error_rate', 0)
        if error_rate > self.thresholds.max_error_rate:
            score -= (error_rate - self.thresholds.max_error_rate) * 1000
            recommendations.append(f"Reduce error rate (current: {error_rate:.1%})")
        
        # Connection stability
        connection_errors = metrics.get('connection_errors', 0)
        if connection_errors > self.thresholds.max_connection_failures:
            score -= min(30, connection_errors * 5)
            recommendations.append("Improve connection stability")
        
        return max(0, score)
    
    def _analyze_resources(self, metrics: Dict[str, Any], recommendations: List[str]) -> float:
        """Analyze resource utilization."""
        score = 100.0
        
        # Memory usage analysis
        memory_usage = metrics.get('memory_usage_mb', 0)
        if memory_usage > self.thresholds.max_memory_usage_mb:
            score -= min(40, (memory_usage - self.thresholds.max_memory_usage_mb) / 100)
            recommendations.append(f"Optimize memory usage (current: {memory_usage}MB)")
        
        return max(0, score)
    
    def _analyze_trends(self, trends: Dict[str, str], recommendations: List[str]) -> float:
        """Analyze trend patterns."""
        score = 100.0
        
        # Check for negative trends
        negative_trends = [k for k, v in trends.items() if v == "decreasing" and "rate" in k.lower()]
        if negative_trends:
            score -= len(negative_trends) * 10
            recommendations.append("Monitor declining performance trends")
        
        return max(0, score)


class CacheMonitor:
    """Main cache monitoring system."""
    
    def __init__(
        self, 
        cache_instance,
        thresholds: Optional[MonitoringThresholds] = None,
        check_interval_seconds: int = 30
    ):
        self.cache = cache_instance
        self.thresholds = thresholds or MonitoringThresholds()
        self.check_interval = check_interval_seconds
        
        self.metrics_aggregator = MetricsAggregator()
        self.alert_manager = AlertManager()
        self.health_analyzer = CacheHealthAnalyzer(self.thresholds)
        
        self.monitoring_task: Optional[asyncio.Task] = None
        self.is_monitoring = False
        
        # Setup default notification handlers
        self._setup_default_handlers()
    
    def _setup_default_handlers(self):
        """Setup default alert notification handlers."""
        
        async def log_alert_handler(alert: Alert):
            """Log alert to standard logging."""
            log_level = {
                AlertSeverity.INFO: logging.INFO,
                AlertSeverity.WARNING: logging.WARNING,
                AlertSeverity.CRITICAL: logging.CRITICAL,
                AlertSeverity.EMERGENCY: logging.CRITICAL
            }.get(alert.severity, logging.WARNING)
            
            logger.log(log_level, f"CACHE ALERT [{alert.severity}] {alert.alert_type}: {alert.message}")
        
        self.alert_manager.add_notification_handler(log_alert_handler)
    
    def add_webhook_handler(self, webhook_url: str, headers: Optional[Dict[str, str]] = None):
        """Add webhook notification handler."""
        
        async def webhook_handler(alert: Alert):
            """Send alert to webhook."""
            if aiohttp is None:
                logger.warning("aiohttp not available, skipping webhook notification")
                return
                
            payload = {
                "alert_type": alert.alert_type,
                "severity": alert.severity,
                "message": alert.message,
                "timestamp": alert.timestamp.isoformat(),
                "metadata": alert.metadata
            }
            
            try:
                async with aiohttp.ClientSession() as session:
                    async with session.post(
                        webhook_url,
                        json=payload,
                        headers=headers or {}
                    ) as response:
                        if response.status >= 400:
                            logger.error(f"Webhook notification failed: {response.status}")
            except Exception as e:
                logger.error(f"Failed to send webhook notification: {e}")
        
        self.alert_manager.add_notification_handler(webhook_handler)
    
    def add_email_handler(self, smtp_config: Dict[str, Any], recipients: List[str]):
        """Add email notification handler."""
        
        async def email_handler(alert: Alert):
            """Send alert via email."""
            # Email implementation would go here
            # For now, just log the intention
            logger.info(f"Would send email alert to {recipients}: {alert.message}")
        
        self.alert_manager.add_notification_handler(email_handler)
    
    async def start_monitoring(self):
        """Start the monitoring system."""
        if self.is_monitoring:
            logger.warning("Monitoring is already running")
            return
        
        self.is_monitoring = True
        self.monitoring_task = asyncio.create_task(self._monitoring_loop())
        logger.info("Cache monitoring started")
    
    async def stop_monitoring(self):
        """Stop the monitoring system."""
        self.is_monitoring = False
        
        if self.monitoring_task:
            self.monitoring_task.cancel()
            try:
                await self.monitoring_task
            except asyncio.CancelledError:
                pass
        
        logger.info("Cache monitoring stopped")
    
    async def _monitoring_loop(self):
        """Main monitoring loop."""
        while self.is_monitoring:
            try:
                await self._perform_health_check()
                await asyncio.sleep(self.check_interval)
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Error in monitoring loop: {e}")
                await asyncio.sleep(self.check_interval)
    
    async def _perform_health_check(self):
        """Perform comprehensive health check."""
        try:
            # Get cache information
            cache_info = await self.cache.get_info()
            health_check = await self.cache.health_check()
            
            # Extract metrics
            metrics = cache_info.get('metrics', {})
            redis_info = cache_info.get('redis_info', {})
            circuit_breaker = cache_info.get('circuit_breaker', {})
            
            # Add health check results
            metrics.update({
                'health_status': health_check['status'],
                'health_response_time': health_check.get('response_time_ms', 0),
                'connection_errors': redis_info.get('rejected_connections', 0),
                'memory_usage_mb': self._parse_memory_usage(redis_info.get('memory_usage', '0')),
            })
            
            # Add to metrics history
            await self.metrics_aggregator.add_metrics(metrics)
            
            # Check thresholds and create alerts
            await self._check_thresholds(metrics, circuit_breaker)
            
            # Analyze trends and detect anomalies
            trends = self._analyze_trends()
            await self._check_anomalies(metrics)
            
            # Update health score
            health_score = self.health_analyzer.analyze_health(metrics, trends)
            
            logger.debug(f"Health check completed - Score: {health_score.score:.1f} ({health_score.status})")
            
        except Exception as e:
            logger.error(f"Health check failed: {e}")
            await self.alert_manager.create_alert(
                AlertType.CACHE_UNAVAILABLE,
                AlertSeverity.CRITICAL,
                f"Cache health check failed: {str(e)}"
            )
    
    def _parse_memory_usage(self, memory_str: str) -> float:
        """Parse Redis memory usage string to MB."""
        try:
            if 'K' in memory_str:
                return float(memory_str.replace('K', '')) / 1024
            elif 'M' in memory_str:
                return float(memory_str.replace('M', ''))
            elif 'G' in memory_str:
                return float(memory_str.replace('G', '')) * 1024
            else:
                return float(memory_str) / (1024 * 1024)  # Assume bytes
        except:
            return 0.0
    
    async def _check_thresholds(self, metrics: Dict[str, Any], circuit_breaker: Dict[str, Any]):
        """Check metrics against thresholds and create alerts."""
        
        # Hit rate check
        hit_rate = metrics.get('hit_rate', 0)
        if hit_rate < self.thresholds.min_hit_rate:
            await self.alert_manager.create_alert(
                AlertType.HIGH_MISS_RATE,
                AlertSeverity.WARNING,
                f"Cache hit rate is low: {hit_rate:.1%} (threshold: {self.thresholds.min_hit_rate:.1%})",
                {"hit_rate": hit_rate, "threshold": self.thresholds.min_hit_rate}
            )
        
        # Response time check
        response_time = metrics.get('average_response_time', 0)
        if response_time > self.thresholds.max_response_time_ms:
            await self.alert_manager.create_alert(
                AlertType.SLOW_RESPONSE,
                AlertSeverity.WARNING,
                f"Cache response time is high: {response_time:.1f}ms (threshold: {self.thresholds.max_response_time_ms}ms)",
                {"response_time": response_time, "threshold": self.thresholds.max_response_time_ms}
            )
        
        # Error rate check
        error_rate = metrics.get('error_rate', 0)
        if error_rate > self.thresholds.max_error_rate:
            await self.alert_manager.create_alert(
                AlertType.HIGH_ERROR_RATE,
                AlertSeverity.CRITICAL,
                f"Cache error rate is high: {error_rate:.1%} (threshold: {self.thresholds.max_error_rate:.1%})",
                {"error_rate": error_rate, "threshold": self.thresholds.max_error_rate}
            )
        
        # Circuit breaker check
        if circuit_breaker.get('state') == 'OPEN':
            await self.alert_manager.create_alert(
                AlertType.CIRCUIT_BREAKER_OPEN,
                AlertSeverity.CRITICAL,
                "Cache circuit breaker is OPEN - service degraded",
                {"failure_count": circuit_breaker.get('failure_count', 0)}
            )
        
        # Memory usage check
        memory_usage = metrics.get('memory_usage_mb', 0)
        if memory_usage > self.thresholds.max_memory_usage_mb:
            await self.alert_manager.create_alert(
                AlertType.MEMORY_PRESSURE,
                AlertSeverity.WARNING,
                f"Cache memory usage is high: {memory_usage:.1f}MB (threshold: {self.thresholds.max_memory_usage_mb}MB)",
                {"memory_usage": memory_usage, "threshold": self.thresholds.max_memory_usage_mb}
            )
    
    def _analyze_trends(self) -> Dict[str, str]:
        """Analyze metric trends."""
        return {
            'hit_rate': self.metrics_aggregator.get_trend('hit_rate'),
            'response_time': self.metrics_aggregator.get_trend('average_response_time'),
            'error_rate': self.metrics_aggregator.get_trend('error_rate'),
            'memory_usage': self.metrics_aggregator.get_trend('memory_usage_mb'),
        }
    
    async def _check_anomalies(self, metrics: Dict[str, Any]):
        """Check for anomalous patterns."""
        anomaly_metrics = ['hit_rate', 'average_response_time', 'error_rate']
        
        for metric in anomaly_metrics:
            if self.metrics_aggregator.detect_anomalies(metric, self.thresholds.anomaly_threshold_multiplier):
                await self.alert_manager.create_alert(
                    AlertType.ANOMALOUS_PATTERN,
                    AlertSeverity.WARNING,
                    f"Anomalous pattern detected in {metric}",
                    {"metric": metric, "current_value": metrics.get(metric, 0)}
                )
    
    async def get_dashboard_data(self) -> Dict[str, Any]:
        """Get comprehensive dashboard data."""
        cache_info = await self.cache.get_info()
        health_check = await self.cache.health_check()
        
        # Get metrics and trends
        metrics = cache_info.get('metrics', {})
        trends = self._analyze_trends()
        
        # Analyze health
        health_score = self.health_analyzer.analyze_health(metrics, trends)
        
        # Get alert summary
        alert_summary = self.alert_manager.get_alert_summary()
        
        return {
            "timestamp": datetime.utcnow().isoformat(),
            "health": {
                "score": health_score.score,
                "status": health_score.status,
                "factors": health_score.factors,
                "recommendations": health_score.recommendations,
            },
            "metrics": {
                **metrics,
                "health_response_time": health_check.get('response_time_ms', 0),
                "circuit_breaker_state": cache_info.get('circuit_breaker', {}).get('state', 'UNKNOWN'),
            },
            "trends": trends,
            "alerts": {
                **alert_summary,
                "active_alerts": [
                    {
                        "type": alert.alert_type,
                        "severity": alert.severity,
                        "message": alert.message,
                        "timestamp": alert.timestamp.isoformat(),
                    }
                    for alert in self.alert_manager.get_active_alerts()
                ]
            },
            "redis_info": cache_info.get('redis_info', {}),
            "configuration": cache_info.get('configuration', {}),
        }


# Global monitoring instance
_monitor_instance: Optional[CacheMonitor] = None


def get_cache_monitor(cache_instance, thresholds: Optional[MonitoringThresholds] = None) -> CacheMonitor:
    """Get or create global cache monitor instance."""
    global _monitor_instance
    
    if _monitor_instance is None:
        _monitor_instance = CacheMonitor(cache_instance, thresholds)
    
    return _monitor_instance